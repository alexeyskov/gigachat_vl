"""Immutable, provenance-bearing manifests for bounded research runs.

Source parquet shards remain unmodified; image content is materialized only for
selected records. Raw HF train splits are never silently replaced with tests.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import io
import json
import os
import tempfile
import shutil
import time

from src.dataset.vision_cache import image_content_hash, canonical_json, ensure_disk_space


def _atomic_write(path, text):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.manifest-')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def partition_for_group(group_id: str, seed: int = 42, validation_fraction: float = 0.02):
    h = hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()
    return "validation" if int(h[:16], 16)/2**64 < validation_fraction else "train"


def iter_parquet_rows(paths):
    import pyarrow.parquet as pq
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=16):
            for index, row in enumerate(batch.to_pylist()):
                yield str(path), row


def materialize_image(value, image_dir: Path, reserve_bytes: int):
    from PIL import Image, ImageOps
    image_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        raw = value.get("bytes")
        image = Image.open(io.BytesIO(raw)) if raw is not None else Image.open(value['path'])
    elif isinstance(value, (str, Path)):
        raw = None
        image = Image.open(value)
    elif hasattr(value, 'convert'):
        raw, image = None, value
    else:
        raise ValueError(f"Unsupported image value {type(value).__name__}")
    image.load()
    content_hash = image_content_hash(image)
    path = image_dir / content_hash[:2] / (content_hash + (".image" if raw else ".png"))
    if not path.exists():
        if raw is None:
            out = io.BytesIO()
            ImageOps.exif_transpose(image).convert('RGB').save(out, format='PNG')
            raw = out.getvalue()
        ensure_disk_space(image_dir, len(raw), reserve_bytes)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.image-')
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(raw)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    image.close()
    return str(path.resolve()), content_hash


def extract_text_turns(row):
    messages = row.get('messages', [])
    prefix = []
    for message in messages:
        role, content = message.get('role'), message.get('content')
        if not isinstance(content, str) or not content.strip():
            continue
        if role == 'assistant' and prefix and prefix[-1]['role'] == 'user':
            yield list(prefix), content.strip()
        prefix.append({'role': role, 'content': content.strip()})


def prepare_from_specs(specs, output_dir, *, reserve_bytes=120*1024**3,
                       seed=42, validation_fraction=0.02):
    """Build manifests atomically; specs have kind/repo/revision/config/files/limit.

    Each output record includes image_paths/image_hashes (possibly empty),
    question/answer, messages (when available), source, task_group and partition.
    Returns audit. Existing manifest directories are immutable, not overwritten.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_json = canonical_json({'specs': specs, 'seed': seed, 'validation_fraction': validation_fraction})
    manifest_id = hashlib.sha256(spec_json.encode()).hexdigest()
    audit_path = output_dir / 'audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit['manifest_id'] != manifest_id:
            raise FileExistsError('Existing manifests have a different spec; use a new output directory')
        for stage in ['alignment','sft','text']:
            for partition in ['train','validation']:
                manifest = output_dir / f'{stage}_{partition}.jsonl'
                actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
                if actual != audit[f'{stage}_{partition}_sha256']:
                    raise ValueError(f'Manifest checksum changed: {manifest}')
        return audit
    image_dir = output_dir / 'images'
    by_stage_partition = {(s,p): [] for s in ['alignment','sft','text'] for p in ['train','validation']}
    seen_record = set()
    audit = {'manifest_id': manifest_id, 'seed': seed, 'validation_fraction': validation_fraction,
             'source_counts': {}, 'filter_counts': {}, 'official_eval_audit': 'pending source-ID/near-duplicate external evaluation audit; internal validation only'}
    filters = Counter()
    for spec in specs:
        count, rows = 0, 0
        started = time.monotonic()
        print(json.dumps({'event':'prepare_source_start','repo':spec['repo'],'config':spec.get('config'),'requested':spec['limit']}),flush=True)
        stage = {'caption':'alignment', 'finevision':'sft', 'text':'text'}[spec['kind']]
        for shard, row in iter_parquet_rows(spec['files']):
            rows += 1
            if rows % 250 == 0:
                print(json.dumps({'event':'prepare_progress','repo':spec['repo'],'config':spec.get('config'),'rows':rows,'records':count,'elapsed_s':round(time.monotonic()-started,1),'disk_free_gib':round(shutil.disk_usage(output_dir).free/1024**3,2)}),flush=True)
            source = {'repo': spec['repo'], 'revision': spec['revision'],
                      'config': spec.get('config'), 'split': 'train',
                      'row_id': str(row.get('id', f"{Path(shard).name}:{rows-1}")), 'shard': Path(shard).name}
            if spec['kind'] == 'caption':
                image_values = [row.get('image')]
                turns = [('Describe this image in detail.', row.get('caption'))]
            elif spec['kind'] == 'finevision':
                image_values = row.get('images') or []
                if len(image_values) != 1:
                    filters['non_single_image'] += 1
                    continue
                turns = [(turn.get('user'), turn.get('assistant')) for turn in row.get('texts', [])]
            else:
                image_values = []
                turns = list(extract_text_turns(row))[:1]
            if any(v is None for v in image_values):
                filters['missing_image'] += 1
                continue
            try:
                paths_hashes = [materialize_image(v, image_dir, reserve_bytes) for v in image_values]
            except (OSError, ValueError) as error:
                if isinstance(error, OSError) and 'reserve' in str(error): raise
                filters['invalid_image'] += 1
                continue
            paths = [p for p,h in paths_hashes]
            hashes = [h for p,h in paths_hashes]
            # Single-image scope: all turns of this image receive the same split across sources.
            group = hashes[0] if hashes else source['repo']+':'+source['row_id']
            partition = partition_for_group(group, seed, validation_fraction)
            for turn_index, (question, answer) in enumerate(turns):
                if not question or not answer:
                    filters['empty_turn'] += 1
                    continue
                messages = question if isinstance(question, list) else [{'role':'user','content':str(question)}]
                question_string = messages[-1]['content']
                answer = str(answer).strip()
                record_key = hashlib.sha256(canonical_json([hashes,messages,answer]).encode()).hexdigest()
                if record_key in seen_record:
                    filters['duplicate_record'] += 1
                    continue
                seen_record.add(record_key)
                record = {'record_id':record_key, 'image_paths':paths, 'image_hashes':hashes,
                          'image': paths[0] if paths else None,
                          'question':question_string, 'answer':answer, 'messages':messages,
                          'source':source, 'dialogue_id':source['repo']+':'+source['config']+':'+source['row_id'],
                          'turn_index':turn_index, 'task_group':spec.get('task_group',stage),
                          'partition':partition, 'language':'en'}
                by_stage_partition[stage,partition].append(record)
                count += 1
                if count >= spec['limit']: break
            if count >= spec['limit']: break
        audit['source_counts'][spec['repo']+'/'+str(spec.get('config'))] = {'records':count,'scanned_rows':rows,'requested':spec['limit']}
        print(json.dumps({'event':'prepare_source_done','repo':spec['repo'],'config':spec.get('config'),'records':count,'requested':spec['limit'],'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    for (stage, partition), records in by_stage_partition.items():
        path = output_dir / f'{stage}_{partition}.jsonl'
        data = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
        ensure_disk_space(output_dir, len(data.encode()), reserve_bytes)
        _atomic_write(path, data)
        audit[f'{stage}_{partition}_records'] = len(records)
        audit[f'{stage}_{partition}_sha256'] = hashlib.sha256(data.encode()).hexdigest()
    audit['filter_counts'] = dict(filters)
    all_records = [r for records in by_stage_partition.values() for r in records]
    audit['unique_images'] = len({h for r in all_records for h in r['image_hashes']})
    _atomic_write(output_dir/'specs.json', spec_json+'\n')
    _atomic_write(audit_path, json.dumps(audit,indent=2)+'\n')
    return audit


class ResearchManifestDataset:
    def __init__(self, path):
        self.path = Path(path)
        self.records = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
    def __len__(self): return len(self.records)
    def __getitem__(self, index): return dict(self.records[index])



def image_perceptual_hash(path):
    """64-bit dHash for a reproducible near-duplicate audit (not semantic identity)."""
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        image=ImageOps.exif_transpose(image).convert('L').resize((9,8))
        pixels=list(image.getdata())
    value=0
    for y in range(8):
        for x in range(8):
            value=(value<<1) | (pixels[y*9+x] > pixels[y*9+x+1])
    return f'{value:016x}'


def audit_eval_overlap(train_manifest_paths, eval_manifest_paths, *, near_hamming=4):
    """Audit immutable image/source identity and optional dHash near duplicates.

    Missing evaluation input is rejected instead of producing a false clean
    audit. dHash candidates are reported for review; they are not silently
    removed, because perceptually similar diagrams may be distinct examples.
    """
    if not eval_manifest_paths:
        raise ValueError('Provide held-out evaluation manifests for an overlap audit')
    if not 0 <= near_hamming <= 4:
        raise ValueError('near_hamming must be between 0 and 4')
    eval_hashes=set(); eval_source_ids=set(); eval_records=0
    eval_phashes={}; image_phash_cache={}; missing_phashes=0
    def phashes(record):
        nonlocal missing_phashes
        known=record.get('image_phashes')
        if known is not None: return known
        values=[]
        for path in record.get('image_paths',[]):
            if path not in image_phash_cache:
                image_phash_cache[path]=image_perceptual_hash(path)
            values.append(image_phash_cache[path])
        if len(values) != len(record.get('image_hashes',[])):
            missing_phashes+=1
        return values
    for manifest in eval_manifest_paths:
        for line in Path(manifest).read_text().splitlines():
            if not line.strip(): continue
            record=json.loads(line); eval_records+=1
            eval_hashes.update(record.get('image_hashes',[]))
            for h,p in zip(record.get('image_hashes',[]),phashes(record)):
                eval_phashes.setdefault(int(p,16),set()).add(h)
            source=record.get('source',{})
            if source.get('original_record_id') is not None:
                eval_source_ids.add((source.get('original_repo',source.get('repo')),source['original_record_id']))
    if not eval_records: raise ValueError('Evaluation manifests contain zero records')
    # Five disjoint chunks: <=4 changed bits imply at least one unchanged chunk.
    chunks=[(0,13),(13,13),(26,13),(39,13),(52,12)]
    buckets={}
    for value in eval_phashes:
        for i,(shift,bits) in enumerate(chunks):
            buckets.setdefault((i,(value>>shift)&((1<<bits)-1)),set()).add(value)
    overlap=[]; near=[]; train_count=0
    for manifest in train_manifest_paths:
        for line in Path(manifest).read_text().splitlines():
            if not line.strip(): continue
            record=json.loads(line); train_count+=1
            source=record.get('source',{})
            source_key=(source.get('original_repo',source.get('repo')),source.get('original_record_id'))
            shared=sorted(set(record.get('image_hashes',[])) & eval_hashes)
            if shared or source_key in eval_source_ids:
                overlap.append({'manifest':str(manifest),'record_id':record['record_id'],'image_hashes':shared,'source_match':source_key in eval_source_ids})
            for h,p in zip(record.get('image_hashes',[]),phashes(record)):
                if h in eval_hashes: continue
                value=int(p,16); candidates=set()
                for i,(shift,bits) in enumerate(chunks):
                    candidates.update(buckets.get((i,(value>>shift)&((1<<bits)-1)),set()))
                for candidate in candidates:
                    distance=(value^candidate).bit_count()
                    if distance<=near_hamming:
                        near.append({'record_id':record['record_id'],'train_image_hash':h,
                                     'eval_image_hashes':sorted(eval_phashes[candidate]),'hamming':distance})
    return {'train_records':train_count,'eval_records':eval_records,'eval_unique_images':len(eval_hashes),
            'overlap_records':len(overlap),'overlaps':overlap,
            'near_duplicate_candidates':near,'near_duplicate_candidate_count':len(near),
            'near_duplicate_method':f'dHash64 hamming <= {near_hamming}; manual review required',
            'records_missing_perceptual_hash':missing_phashes,
            'near_duplicate_audit':'complete' if not missing_phashes else 'incomplete; missing image paths/phashes'}
