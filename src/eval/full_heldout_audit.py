"""Streaming image-only exclusion audit for the complete selected benchmark splits.

Never writes a copy of every benchmark image. Repeated image byte payloads reuse
canonical RGB hashes/dHashes. All official split references and source limitations
remain explicit; no existing training profile/model is modified by this audit.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import io
import json
import time
import threading

from PIL import Image,ImageOps
from src.dataset.vision_cache import canonical_json,ensure_disk_space
from src.eval.research_eval_data import SOURCE_SPECS,convert_row,iter_rows,training_image_index,overlap_candidates,read_jsonl

SPLIT_FILES={
 'vqav2':{'validation':['data/validation-']},
 'gqa':{'testdev':['testdev_balanced_images/','testdev_balanced_instructions/']},
 'textvqa':{'validation':['data/validation-']},
 'docvqa':{'validation':['DocVQA/validation-']},
 'chartqa':{'test':['data/test-']},
 'ai2d':{'test':['data/test-']},
 'scienceqa':{'validation':['data/validation-'],'test':['data/test-']},
 'aokvqa':{'validation':['data/validation-']},
}
EXPECTED_ROWS={'vqav2':{'validation':214354},'gqa':{'testdev':12578},'textvqa':{'validation':5000},
 'docvqa':{'validation':5349},'chartqa':{'test':2500},'ai2d':{'test':3088},
 'scienceqa':{'validation':4241,'test':4241},'aokvqa':{'validation':1145}}


_EMIT_LOCK=threading.Lock()

def emit(event,**fields):
    with _EMIT_LOCK:
        print(json.dumps({'time':time.time(),'event':event,**fields}),flush=True)


def selected_source_files(root):
    root=Path(root)
    result={}
    for benchmark,splits in SPLIT_FILES.items():
        spec=SOURCE_SPECS[benchmark]
        meta=json.loads((root/'eval-metadata'/(spec['repo'].replace('/','--')+'.json')).read_text())
        result[benchmark]={'repo':spec['repo'],'revision':meta['revision'],'splits':{}}
        for split,prefixes in splits.items():
            files=[file for file in meta['files'] if any(file['path'].startswith(prefix) for prefix in prefixes)]
            if not files: raise ValueError(f'No files for {benchmark}/{split}')
            result[benchmark]['splits'][split]=files
    return result


class RawImageSignatureCache:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.values={};self.hits=0;self.decodes=0;repair_tail=False
        if self.path.exists():
            lines=self.path.read_text().splitlines()
            for index,line in enumerate(lines):
                try: row=json.loads(line)
                except json.JSONDecodeError:
                    if index==len(lines)-1:
                        repair_tail=True;break
                    raise
                self.values[row['raw_sha256']]=row['signature']
        if repair_tail:
            self.path.write_text(''.join(json.dumps({'raw_sha256':key,'signature':value})+'\n' for key,value in self.values.items()))
        self.handle=self.path.open('a')

    def _remember(self,raw_sha,signature):
        self.values[raw_sha]=signature
        self.handle.write(json.dumps({'raw_sha256':raw_sha,'signature':signature})+'\n')
        if len(self.values)%1000==0:self.handle.flush()
        return signature

    def seed_existing_manifest(self,path):
        if not Path(path).exists():return
        for record in read_jsonl(path):
            for image_path,image_hash,phash in zip(record.get('image_paths',[]),record.get('image_hashes',[]),record.get('image_phashes',[])):
                raw=Path(image_path).read_bytes();raw_sha=hashlib.sha256(raw).hexdigest()
                if raw_sha not in self.values:self._remember(raw_sha,{'content_hash':image_hash,'phash':phash,'size':None})

    def signature(self,value):
        if not isinstance(value,dict) or value.get('bytes') is None:
            raise ValueError('Expected inline image bytes in the pinned held-out parquet mirror')
        raw=value['bytes'];raw_sha=hashlib.sha256(raw).hexdigest()
        if raw_sha in self.values:
            self.hits+=1
            return self.values[raw_sha]
        with Image.open(io.BytesIO(raw)) as image:
            rgb=ImageOps.exif_transpose(image).convert('RGB')
            h=hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode()+rgb.tobytes()).hexdigest()
            small=rgb.convert('L').resize((9,8));pixels=list(small.getdata());phash=0
            for y in range(8):
                for x in range(8):phash=(phash<<1)|(pixels[y*9+x]>pixels[y*9+x+1])
            signature={'content_hash':h,'phash':f'{phash:016x}','size':[rgb.width,rgb.height]}
        self.decodes+=1
        return self._remember(raw_sha,signature)

    def close(self):self.handle.flush();self.handle.close()


def full_heldout_audit(root,output,train_manifests,*,workers=8,reserve_bytes=120*1024**3):
    import pyarrow.parquet as pq
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();sources=selected_source_files(root)
    index,training_hashes=training_image_index(train_manifests,root/'eval-metadata',workers)
    config={'schema_version':1,'sources':sources,'training_manifest_hashes':training_hashes,'near_hamming':4,
            'scope':'complete split allowlist, not every private/hidden test split of each benchmark',
            'image_decode_policy':'PIL_EXIF_transpose_RGB','image_copy_policy':'no_full_heldout_image_extraction'}
    fingerprint=hashlib.sha256(canonical_json(config).encode()).hexdigest()
    summary={'schema_version':1,'status':'running','coverage_complete':False,'manifest_fingerprint':fingerprint,
             'scope':config['scope'],'source_coverage':{},'training_was_modified':False,'trained_models_decontaminated':False}
    if (output/'audit.json').exists():
        previous=json.loads((output/'audit.json').read_text())
        if previous.get('status')=='complete' and previous.get('manifest_fingerprint')==fingerprint:return previous
        if previous.get('manifest_fingerprint')!=fingerprint:raise FileExistsError('Full audit inputs changed; use a new output directory')
    (output/'resolved_config.json').write_text(json.dumps(config,indent=2))
    cache=RawImageSignatureCache(root/'eval-metadata/raw-image-signatures-exif-rgb-v1.jsonl')
    cache.seed_existing_manifest(root/'eval-heldout-v3/evaluation.jsonl')
    cache.seed_existing_manifest(root/'eval-scienceqa-full-overlap-v1/scienceqa_full_heldout_images.jsonl')
    images={};matches={};source_image_ids=defaultdict(set)
    questions_path=output/'heldout_question_image_ids.jsonl'
    try:
        with questions_path.open('w') as questions:
            for benchmark,source_info in sources.items():
                spec=SOURCE_SPECS[benchmark]
                counts=Counter();split_rows={};unique=set();noimage=0;missing_qid=0;missing_image_id=0;sample_failures=[]
                for split,files in source_info['splits'].items():
                    data_files=files
                    gqa_images=None
                    if benchmark=='gqa':
                        image_file=next(file for file in files if 'images/' in file['path'])
                        image_path=root/'eval-raw'/source_info['repo'].replace('/','--')/image_file['path']
                        gqa_images={row['id']:row['image'] for _,row in iter_rows(image_path)}
                        data_files=[file for file in files if 'instructions/' in file['path']]
                    rows_scanned=0
                    for file in sorted(data_files,key=lambda item:item['path']):
                        path=root/'eval-raw'/source_info['repo'].replace('/','--')/file['path']
                        if not path.exists() or path.stat().st_size!=file['size']: raise ValueError(f'Missing/incomplete pinned source file {path}')
                        metadata=pq.ParquetFile(path)
                        emit('full_audit_file_start',benchmark=benchmark,split=split,file=file['path'],rows=metadata.metadata.num_rows)
                        for file_row,row in iter_rows(path):
                            global_row=rows_scanned;rows_scanned+=1;counts['rows_scanned']+=1
                            source={'repo':source_info['repo'],'revision':source_info['revision'],'config':spec['config'],'split':split,
                                    'shard':file['path'],'source_is_hf_mirror':True}
                            try:
                                record=convert_row(benchmark,row,global_row if benchmark in ['scienceqa','chartqa'] else file_row,source,root/'eval-metadata',gqa_images=gqa_images)
                                if record is None:
                                    if benchmark!='scienceqa': raise ValueError('Unexpected record without a usable image/reference')
                                    noimage+=1;continue
                                signature=cache.signature(record.pop('image_value'))
                            except Exception as error:
                                sample_failures.append({'shard':file['path'],'row':file_row,'error_type':type(error).__name__,'error':str(error)})
                                summary['source_coverage'][benchmark]={'status':'partial','sample_failures':sample_failures,'counts':dict(counts)}
                                raise
                            image_hash=signature['content_hash'];unique.add(image_hash);counts['image_bearing_rows']+=1
                            if record['source'].get('original_question_id') is None:missing_qid+=1
                            original_image_id=record['source'].get('original_image_id')
                            if original_image_id is None:missing_image_id+=1
                            else:source_image_ids[benchmark,str(original_image_id)].add(image_hash)
                            entry=images.setdefault(image_hash,{'image_hash':image_hash,'phash':signature['phash'],'size':signature['size'],
                                                                 'benchmarks':set(),'splits':set(),'question_count':0})
                            entry['benchmarks'].add(benchmark);entry['splits'].add(benchmark+'/'+split);entry['question_count']+=1
                            questions.write(json.dumps({'benchmark':benchmark,'split':split,'image_hash':image_hash,
                                'original_question_id':record['source'].get('original_question_id'),
                                'original_image_id':original_image_id,'source_original_id':record['source'].get('original_id'),
                                'shard':file['path'],'file_row_index':file_row})+'\n')
                            if image_hash not in matches:
                                found=overlap_candidates({'image_hashes':[image_hash],'image_phashes':[signature['phash']]},index)
                                matches[image_hash]=found
                                if found:
                                    exact_caption={ref['record_id'] for match in found if match['kind']=='exact' for ref in match['references'] if Path(ref['manifest']).name.startswith('alignment_')}
                                    if exact_caption:emit('exact_caption_overlap_detected',benchmark=benchmark,image_hash=image_hash,caption_record_ids=sorted(exact_caption))
                            if counts['rows_scanned']%2000==0:
                                emit('full_audit_progress',benchmark=benchmark,rows=counts['rows_scanned'],unique_images=len(unique),
                                     decoded_unique_images=cache.decodes,raw_byte_cache_hits=cache.hits)
                    split_rows[split]=rows_scanned
                    if rows_scanned!=EXPECTED_ROWS[benchmark][split]:
                        raise ValueError(f'Official split row coverage mismatch for {benchmark}/{split}: {rows_scanned} vs {EXPECTED_ROWS[benchmark][split]}')
                summary['source_coverage'][benchmark]={'status':'complete','revision':source_info['revision'],'split_rows':split_rows,
                    'counts':dict(counts),'known_non_image_rows_excluded_by_protocol':noimage,'unique_images':len(unique),
                    'original_question_ids_unavailable':missing_qid,'original_image_ids_unavailable':missing_image_id,
                    'original_question_id_limitation':('AI2D mirror omits original question IDs; original diagram filenames and exact pinned shard row indices retained' if benchmark=='ai2d' else 'Original release uses annotation file/index rather than a standalone question ID; original_id and chart filenames retained') if missing_qid else None,
                    'sample_failure_count':0,'sample_failures':sample_failures}
                emit('full_audit_source_done',benchmark=benchmark,**summary['source_coverage'][benchmark])
        exact_records={};near_only_records={};all_records={};overlap_images=[]
        for image_hash,found in matches.items():
            if not found:continue
            entry=images[image_hash]
            overlap_images.append({**entry,'benchmarks':sorted(entry['benchmarks']),'splits':sorted(entry['splits']),'overlaps':found})
            for match in found:
                for ref in match['references']:
                    key=ref['manifest'],ref['record_id']
                    all_records[key]=ref
                    if match['kind']=='exact':exact_records[key]=ref
                    else:near_only_records[key]=ref
        near_only_records={key:value for key,value in near_only_records.items() if key not in exact_records}
        outputs={'full_heldout_image_index.jsonl':[{**value,'benchmarks':sorted(value['benchmarks']),'splits':sorted(value['splits'])} for value in images.values()],
                 'overlap_images.jsonl':overlap_images,'exact_training_exclusion_candidates.jsonl':list(exact_records.values()),
                 'near_only_training_review_candidates.jsonl':list(near_only_records.values()),
                 'conservative_training_exclusion_candidates.jsonl':list(all_records.values())}
        file_hashes={'heldout_question_image_ids.jsonl':hashlib.sha256(questions_path.read_bytes()).hexdigest()}
        for name,rows in outputs.items():
            text=''.join(json.dumps(row)+'\n' for row in rows)
            ensure_disk_space(output,len(text.encode()),reserve_bytes);(output/name).write_text(text)
            file_hashes[name]=hashlib.sha256(text.encode()).hexdigest()
        summary.update(status='complete',coverage_complete=True,declared_split_coverage_complete=True,
            all_official_validation_test_coverage_complete=False,primary_all_official_heldout_decontamination_complete=False,
            coverage_scope_wording='All DECLARED heldout splits only; not all official validation/test splits universally',
            image_identity='SHA256 of decoded EXIF-normalized RGB pixels; encoding/resizing variants may differ and only dHash review covers near matches',
            benchmark_count=len(sources),unique_heldout_images=len(images),
            exact_training_records=len(exact_records),near_only_training_review_records=len(near_only_records),
            conservative_training_exclusion_records=len(all_records),
            exact_record_counts_by_manifest=dict(Counter(Path(ref['manifest']).name for ref in exact_records.values())),
            near_only_record_counts_by_manifest=dict(Counter(Path(ref['manifest']).name for ref in near_only_records.values())),
            exact_caption_records=sum(Path(ref['manifest']).name.startswith('alignment_') for ref in exact_records.values()),
            official_image_id_encoding_variants=sum(len(value)>1 for value in source_image_ids.values()),
            raw_signature_cache_hits=cache.hits,new_unique_decodes=cache.decodes,image_copies_written=0,
            elapsed_seconds=time.monotonic()-started,file_sha256=file_hashes,
            excluded_other_source_splits='Only the complete explicit split allowlist was audited; private/hidden-label tests and unselected GQA splits are not implied.')
        (output/'audit.json').write_text(json.dumps(summary,indent=2))
        report=Path('research_results/full_eval_image_exclusion_audit.json');report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(summary,indent=2))
        emit('full_audit_complete',output=str(output),exact_caption_records=summary['exact_caption_records'],elapsed_seconds=summary['elapsed_seconds'])
        return summary
    except BaseException as error:
        summary.update(status='partial',coverage_complete=False,error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        (output/'audit.json').write_text(json.dumps(summary,indent=2))
        Path('research_results/full_eval_image_exclusion_audit.json').write_text(json.dumps(summary,indent=2))
        emit('full_audit_partial',error_type=type(error).__name__,error=str(error))
        raise
    finally:cache.close()
