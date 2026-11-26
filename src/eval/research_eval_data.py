"""Bounded held-out benchmark manifests with explicit training-overlap exclusion.

These are pinned HF repackagings of original benchmark held-out splits, not
claims that a partial shard is the full official benchmark. A clean evaluation
subset does not retroactively decontaminate a trained model or training corpus.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import random

from src.dataset.research_data import materialize_image,image_perceptual_hash
from src.dataset.vision_cache import canonical_json,ensure_disk_space

SOURCE_SPECS={
 'vqav2':{'scoring_protocol':'vqav2_official_legacy','repo':'lmms-lab/VQAv2','config':'default','split':'validation','task_group':'general_vqa','scorer':'vqa_accuracy','files':['data/validation-00000-of-00068.parquet'],'original_home':'https://visualqa.org/'},
 'gqa':{'repo':'lmms-lab/GQA','config':'testdev_balanced_instructions','split':'testdev','task_group':'general_vqa','scorer':'exact_match','files':['testdev_balanced_images/testdev-00000-of-00001.parquet','testdev_balanced_instructions/testdev-00000-of-00001.parquet'],'original_home':'https://cs.stanford.edu/people/dorarad/gqa/'},
 'textvqa':{'scoring_protocol':'textvqa_evalai','repo':'lmms-lab/TextVQA','config':'default','split':'validation','task_group':'ocr_documents','scorer':'vqa_accuracy','files':['data/validation-00000-of-00003.parquet'],'original_home':'https://textvqa.org/'},
 'docvqa':{'repo':'lmms-lab/DocVQA','config':'DocVQA','split':'validation','task_group':'ocr_documents','scorer':'anls','files':['DocVQA/validation-00000-of-00006.parquet'],'original_home':'https://www.docvqa.org/'},
 'chartqa':{'repo':'HuggingFaceM4/ChartQA','config':'default','split':'test','task_group':'charts_diagrams','scorer':'chart_relaxed','files':['data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet'],'original_home':'https://github.com/vis-nlp/ChartQA'},
 'ai2d':{'repo':'lmms-lab/AI2D','config':'default','split':'test','task_group':'charts_diagrams','scorer':'mcq_accuracy','files':['data/test-00000-of-00002.parquet'],'original_home':'https://allenai.org/data/diagrams'},
 'scienceqa':{'repo':'derek-thomas/ScienceQA','config':'default','split':'test','task_group':'knowledge_reasoning','scorer':'mcq_accuracy','files':['data/test-00000-of-00001-f0e719df791966ff.parquet'],'original_home':'https://github.com/lupantech/ScienceQA'},
 'aokvqa':{'repo':'HuggingFaceM4/A-OKVQA','config':'default','split':'validation','task_group':'knowledge_reasoning','scorer':'mcq_accuracy','files':['data/validation-00000-of-00001-b2bd0de231b6326a.parquet'],'original_home':'https://allenai.org/project/a-okvqa'},
}
SHORT_ANSWER='Answer with a short answer only.'
MCQ_ANSWER='Answer with the letter of the correct option only.'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def iter_rows(path):
    import pyarrow.parquet as pq
    index=0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            yield index,row
            index+=1


def convert_row(benchmark,row,row_index,source,metadata_dir,*,gqa_images=None):
    """Normalize source labels/prompts without adding training-only hints or answers."""
    spec=SOURCE_SPECS[benchmark]
    source=dict(source,benchmark=benchmark,original_home=spec['original_home'],mirror_row_index=row_index)
    image=row.get('image');choices=None;answer_index=None
    if benchmark=='gqa':
        image=gqa_images[row['imageId']]
        question=row['question'];answers=[str(row['answer'])]
        source.update(original_id=str(row['id']),original_question_id=str(row['id']),original_image_id=str(row['imageId']))
    elif benchmark=='vqav2':
        question=row['question'];answers=[a['answer'] for a in row['answers']]
        if len(answers)!=10: raise ValueError('VQAv2 requires all ten annotator answers')
        source.update(original_id=str(row['question_id']),original_question_id=str(row['question_id']),original_image_id=str(row['image_id']))
    elif benchmark=='textvqa':
        if row.get('set_name') not in ['val','validation']: raise ValueError('TextVQA source is not validation')
        question=row['question'];answers=list(row['answers'])
        if len(answers)!=10: raise ValueError('TextVQA requires all ten annotator answers')
        source.update(original_id=str(row['question_id']),original_question_id=str(row['question_id']),original_image_id=str(row['image_id']))
    elif benchmark=='docvqa':
        if row.get('data_split') not in ['val','validation']: raise ValueError('DocVQA source is not validation')
        question=row['question'];answers=list(row['answers'])
        source.update(original_id=str(row['questionId']),original_question_id=str(row['questionId']),original_document_id=str(row['docId']),
                      original_image_id=str(row['ucsf_document_id'])+'/'+str(row['ucsf_document_page_no']))
    elif benchmark=='chartqa':
        question=row['query'];answers=list(row['label'])
        annotation='test_human.json' if row['human_or_machine']==0 else 'test_augmented.json'
        original=_metadata_json(Path(metadata_dir)/('ChartQA-original-'+annotation))
        local_index=row_index if row['human_or_machine']==0 else row_index-1250
        expected=original[local_index]
        if expected['query']!=question or str(expected['label'])!=str(answers[0]):
            raise ValueError('ChartQA mirror order does not match original test annotation')
        source.update(original_id=annotation+':'+str(local_index),original_question_id=None,original_image_id=expected['imgname'],
                      original_annotation_file=annotation,original_annotation_index=local_index)
    elif benchmark=='ai2d':
        question=row['question'];choices=list(row['options']);answer_index=int(row['answer'])
        image_name=image.get('path') if isinstance(image,dict) else None
        if not image_name: raise ValueError('AI2D mirror lost original diagram filename')
        source.update(original_id=None,original_question_id=None,original_image_id=image_name,
                      original_id_limitation='Original question ID is absent in mirror; pinned shard row and original diagram filename retained')
    elif benchmark=='scienceqa':
        if image is None: return None
        root=Path(metadata_dir)
        # Cache JSON metadata after first use; no hidden answer/solution enters prompt.
        problems=_metadata_json(root/'ScienceQA-original-problems.json')
        splits=_metadata_json(root/'ScienceQA-original-pid_splits.json')
        split_key={'validation':'val','val':'val','test':'test'}[source['split']]
        pid=splits[split_key][row_index];expected=problems[pid]
        for key in ['question','choices','answer','hint']:
            if expected[key]!=row[key]: raise ValueError(f'ScienceQA mirror/test PID mismatch for {pid}: {key}')
        question=row['question'];choices=list(row['choices']);answer_index=int(row['answer'])
        if row.get('hint'): question+='\nContext: '+row['hint']
        source.update(original_id=pid,original_question_id=pid,original_image_id=f'{split_key}/{pid}/image.png')
    elif benchmark=='aokvqa':
        question=row['question'];choices=list(row['choices']);answer_index=int(row['correct_choice_idx'])
        source.update(original_id=str(row['question_id']),original_question_id=str(row['question_id']))
    else: raise ValueError(benchmark)
    if image is None or not question or (choices is None and not answers): return None
    record={'benchmark':benchmark,'task_group':spec['task_group'],'scorer':spec['scorer'],
            'base_question':question,'question':question,'source':source,'language':'en','image_value':image}
    if spec.get('scoring_protocol'): record['scoring_protocol']=spec['scoring_protocol']
    if choices is not None:
        if not 0<=answer_index<len(choices): raise ValueError('MCQ answer index is out of range')
        letter=chr(ord('A')+answer_index)
        record.update(choices=choices,answer_index=answer_index,correct_choice=letter,answers=[letter])
        record['question']+='\n'+'\n'.join(f'{chr(65+i)}. {choice}' for i,choice in enumerate(choices))+'\n'+MCQ_ANSWER
    else:
        record['answers']=[str(answer) for answer in answers]
        record['question']+='\n'+SHORT_ANSWER
    return record


_METADATA={}
def _metadata_json(path):
    key=str(path)
    if key not in _METADATA: _METADATA[key]=json.loads(Path(path).read_text())
    return _METADATA[key]


def _phash_job(item):
    content_hash,path=item
    return content_hash,image_perceptual_hash(path)


def training_image_index(manifests,cache_dir,workers=8):
    files={str(Path(path).resolve()):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in manifests}
    fingerprint=hashlib.sha256(canonical_json(files).encode()).hexdigest()
    root=Path(cache_dir);root.mkdir(parents=True,exist_ok=True)
    cache=root/('training-image-index-'+fingerprint+'.json')
    if cache.exists(): return json.loads(cache.read_text()),files
    paths={};references=defaultdict(list)
    for filename in manifests:
        for record in read_jsonl(filename):
            for content_hash,path in zip(record.get('image_hashes',[]),record.get('image_paths',[])):
                paths[content_hash]=path
                references[content_hash].append({'manifest':str(Path(filename).resolve()),'record_id':record['record_id'],
                                                  'partition':record.get('partition','train')})
    hashes={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(content_hash,phash) in enumerate(pool.map(_phash_job,sorted(paths.items()),chunksize=32),start=1):
            hashes[content_hash]=phash
            if i%1000==0: print(json.dumps({'event':'training_phash_progress','completed':i,'total':len(paths)}),flush=True)
    result={'manifest_fingerprint':fingerprint,'images':{h:{'phash':p,'references':references[h]} for h,p in hashes.items()}}
    cache.write_text(json.dumps(result))
    return result,files


def overlap_candidates(record,training_index,near_hamming=4):
    """Exact and indexed dHash candidates; near matches are conservatively excluded."""
    index=training_index['images'];result=[]
    buckets=training_index.get('_buckets')
    if buckets is None:
        buckets=defaultdict(set)
        for image_hash,value in index.items():
            n=int(value['phash'],16)
            for i,(shift,bits) in enumerate([(0,13),(13,13),(26,13),(39,13),(52,12)]):
                buckets[i,(n>>shift)&((1<<bits)-1)].add(image_hash)
        training_index['_buckets']=buckets
    for image_hash,phash in zip(record['image_hashes'],record['image_phashes']):
        if image_hash in index:
            result.append({'kind':'exact','eval_image_hash':image_hash,'train_image_hash':image_hash,'distance':0,
                           'references':index[image_hash]['references']})
        value=int(phash,16);candidates=set()
        for i,(shift,bits) in enumerate([(0,13),(13,13),(26,13),(39,13),(52,12)]):
            candidates.update(buckets.get((i,(value>>shift)&((1<<bits)-1)),set()))
        for candidate in sorted(candidates):
            if candidate==image_hash: continue
            distance=(value^int(index[candidate]['phash'],16)).bit_count()
            if distance<=near_hamming:
                result.append({'kind':'near_duplicate_candidate','eval_image_hash':image_hash,'train_image_hash':candidate,'distance':distance,
                               'references':index[candidate]['references']})
    return result


def prepare_evaluation(root,output,train_manifests,*,per_benchmark=64,candidate_multiplier=2,seed=42,workers=8,reserve_bytes=120*1024**3):
    root=Path(root).resolve();output=Path(output).resolve()
    if not train_manifests: raise ValueError('Training manifests are required for overlap audit')
    if per_benchmark<=0 or candidate_multiplier<1: raise ValueError('Invalid bounded evaluation size')
    output.mkdir(parents=True,exist_ok=True)
    config={'schema_version':1,'per_benchmark':per_benchmark,'candidate_multiplier':candidate_multiplier,'seed':seed,
            'near_hamming':4,'scienceqa_image_candidate_factor':4,'sources':{},'training_manifest_hashes':{str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in train_manifests}}
    for name,spec in SOURCE_SPECS.items():
        meta=_metadata_json(root/'eval-metadata'/(spec['repo'].replace('/','--')+'.json'))
        config['sources'][name]={**spec,'revision':meta['revision']}
    for path in sorted((root/'eval-metadata').glob('*-original-*.json')):
        config.setdefault('original_annotation_hashes',{})[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint=hashlib.sha256(canonical_json(config).encode()).hexdigest()
    audit_path=output/'audit.json'
    if audit_path.exists():
        saved=json.loads(audit_path.read_text())
        if saved['manifest_fingerprint']!=fingerprint: raise FileExistsError('Different evaluation config requires a new output directory')
        for name,checksum in saved['file_sha256'].items():
            if hashlib.sha256((output/name).read_bytes()).hexdigest()!=checksum: raise ValueError('Evaluation manifest changed')
        return saved
    index,manifest_hashes=training_image_index(train_manifests,root/'eval-metadata',workers)
    selected=[];excluded=[];all_candidates=[];source_counts={};seen=set()
    for benchmark,spec in SOURCE_SPECS.items():
        meta=config['sources'][benchmark]
        local=[root/'eval-raw'/spec['repo'].replace('/','--')/file for file in spec['files']]
        gqa_images=None
        if benchmark=='gqa':
            gqa_images={row['id']:row['image'] for _,row in iter_rows(local[0])}
        path=local[-1]
        import pyarrow.parquet as pq
        total_rows=pq.ParquetFile(path).metadata.num_rows
        order=list(range(total_rows));random.Random(seed).shuffle(order)
        # Selected records are a deterministic bounded sample of the available
        # shard, never a claim of a random sample from undownloaded whole split.
        multiplier=candidate_multiplier*(4 if benchmark=='scienceqa' else 1)
        targets=set(order[:min(total_rows,per_benchmark*multiplier)])
        drafts=[]
        for row_index,row in iter_rows(path):
            if row_index not in targets: continue
            source={'repo':spec['repo'],'revision':meta['revision'],'config':spec['config'],'split':spec['split'],
                    'shard':spec['files'][-1],'source_is_hf_mirror':True}
            draft=convert_row(benchmark,row,row_index,source,root/'eval-metadata',gqa_images=gqa_images)
            if draft is not None: drafts.append((order.index(row_index),draft))
        drafts.sort(key=lambda pair:pair[0])
        clean=0;rejected=0
        for _,record in drafts:
            image=record.pop('image_value')
            image_path,image_hash=materialize_image(image,output/'images',reserve_bytes)
            record.update(image_paths=[image_path],image_hashes=[image_hash],image_phashes=[image_perceptual_hash(image_path)])
            record['record_id']=hashlib.sha256(canonical_json([benchmark,record['source'],image_hash,record['base_question']]).encode()).hexdigest()
            if record['record_id'] in seen: continue
            seen.add(record['record_id'])
            overlaps=overlap_candidates(record,index)
            record['overlap_audit']={'checked_training_manifest_fingerprint':index['manifest_fingerprint'],'exact_or_near_matches':len(overlaps)}
            all_candidates.append(record)
            if overlaps:
                excluded.append({'record':record,'overlaps':overlaps,'action':'excluded_from_evaluation_subset_not_removed_from_training'})
                rejected+=1
            elif clean<per_benchmark:
                selected.append(record);clean+=1
        source_counts[benchmark]={'source_candidates':len(drafts),'clean_selected':clean,'excluded_overlap_candidates':rejected,'requested':per_benchmark}
        print(json.dumps({'event':'eval_source_prepared','benchmark':benchmark,**source_counts[benchmark]}),flush=True)
        if clean==0: raise ValueError(f'No clean evaluation records for {benchmark}; enlarge bounded source pool or retrain after exclusion')
    outputs={'evaluation.jsonl':selected,'evaluation_candidates.jsonl':all_candidates,'excluded_evaluation_candidates.jsonl':excluded}
    train_exclusions={}
    for exclusion in excluded:
        for match in exclusion['overlaps']:
            for ref in match['references']:
                key=ref['manifest'],ref['record_id']
                entry=train_exclusions.setdefault(key,{**ref,'reason':'overlap_with_bounded_heldout_candidate','candidate_eval_ids':[],'not_applied':True})
                entry['candidate_eval_ids'].append(exclusion['record']['record_id'])
    outputs['training_exclusion_candidates.jsonl']=list(train_exclusions.values())
    checksums={}
    for name,records in outputs.items():
        text=''.join(json.dumps(record,ensure_ascii=False)+'\n' for record in records)
        ensure_disk_space(output,len(text.encode()),reserve_bytes)
        (output/name).write_text(text)
        checksums[name]=hashlib.sha256(text.encode()).hexdigest()
    audit={'schema_version':1,'status':'prepared_bounded_subset','manifest_fingerprint':fingerprint,
           'selected_records':len(selected),'task_group_counts':dict(Counter(row['task_group'] for row in selected)),
           'benchmark_counts':source_counts,'candidate_overlap_records':len(excluded),
           'candidate_exact_overlap_records':sum(any(m['kind']=='exact' for m in row['overlaps']) for row in excluded),
           'candidate_near_only_overlap_records':sum(not any(m['kind']=='exact' for m in row['overlaps']) for row in excluded),
           'training_exclusion_candidate_records':len(train_exclusions),
           'selected_exact_or_near_training_overlap':0,'training_was_modified':False,'trained_models_decontaminated':False,
           'training_manifest_hashes':manifest_hashes,'training_image_count':len(index['images']),
           'selection_scope':'bounded first available heldout shard per source; deterministic seed sample; not full official benchmark',
           'near_duplicate_method':'dHash64 Hamming<=4, conservatively excluded, may include false positives',
           'original_question_id_unavailable':['ai2d'],'file_sha256':checksums}
    (output/'resolved_config.json').write_text(json.dumps(config,indent=2))
    (output/'README.md').write_text(render_readme(config,audit))
    audit_path.write_text(json.dumps(audit,indent=2))
    return audit


def render_readme(config,audit):
    lines=['# Bounded held-out multimodal evaluation','',
           'This is a deterministic subset of downloaded official held-out split mirrors, not the full benchmark.',
           'Selection excludes exact and dHash-near overlap with the listed pilot training/internal-validation manifests.',
           'This does not modify training data and does not decontaminate existing models. Full official benchmarks require a full overlap audit and retraining where necessary.','',
           '| Benchmark | HF mirror | Split | Revision | Scorer | Selected |','| --- | --- | --- | --- | --- | --- |']
    for name,source in config['sources'].items():
        lines.append(f"| {name} | [{source['repo']}](https://huggingface.co/datasets/{source['repo']}) | {source['split']} | `{source['revision']}` | {source['scorer']} | {audit['benchmark_counts'][name]['clean_selected']} |")
    lines+=['','VQAv2/TextVQA retain all ten annotator answers. GQA uses its single short answer. DocVQA uses reference aliases with ANLS. ChartQA uses 5% relaxed numeric accuracy. AI2D, ScienceQA and A-OKVQA use multiple-choice accuracy.',
            'ScienceQA includes supplied hint context, excludes lecture and solution, and selects only image-bearing questions. A-OKVQA uses multiple-choice labels, not direct-answer scoring. OCR tokens and rationales are never appended to prompts.',
            'Each open-answer prompt ends with: '+SHORT_ANSWER,'Each MCQ prompt lists A/B/etc options and ends with: '+MCQ_ANSWER,
            'ScienceQA PIDs and ChartQA image/annotation IDs are checked against original release JSON. AI2D retains the original diagram filename; the HF mirror omits official question IDs, so those are explicitly null and pinned mirror row IDs are retained.',
            'See excluded_evaluation_candidates.jsonl and training_exclusion_candidates.jsonl for conservative overlap findings; exclusions were not applied to training.','']
    return '\n'.join(lines)


def derive_scoring_protocols(parent,output):
    """Create a protocol-only immutable derivative; preserve image paths and selection."""
    parent=Path(parent).resolve();output=Path(output).resolve()
    parent_audit=json.loads((parent/'audit.json').read_text())
    protocols={'vqav2':'vqav2_official_legacy','textvqa':'textvqa_evalai'}
    config=json.loads((parent/'resolved_config.json').read_text())
    config.update(derived_from_eval_manifest_fingerprint=parent_audit['manifest_fingerprint'],scoring_protocols=protocols)
    fingerprint=hashlib.sha256(canonical_json(config).encode()).hexdigest()
    if (output/'audit.json').exists():
        saved=json.loads((output/'audit.json').read_text())
        if saved['manifest_fingerprint']!=fingerprint: raise FileExistsError('Existing protocol derivative has different provenance')
        for name,expected in saved['file_sha256'].items():
            if hashlib.sha256((output/name).read_bytes()).hexdigest()!=expected: raise ValueError('Protocol derivative changed')
        return saved
    output.mkdir(parents=True,exist_ok=True)
    audit=dict(parent_audit);checksums={}
    for name,expected in parent_audit['file_sha256'].items():
        data=(parent/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected: raise ValueError('Parent evaluation manifest changed')
        rows=[json.loads(line) for line in data.decode().splitlines()]
        for row in rows:
            record=row.get('record',row)
            if record.get('benchmark') in protocols: record['scoring_protocol']=protocols[record['benchmark']]
        text=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
        (output/name).write_text(text);checksums[name]=hashlib.sha256(text.encode()).hexdigest()
    audit.update(derived_from_eval_manifest_fingerprint=parent_audit['manifest_fingerprint'],manifest_fingerprint=fingerprint,
                 file_sha256=checksums,scoring_protocols=protocols)
    (output/'resolved_config.json').write_text(json.dumps(config,indent=2))
    (output/'README.md').write_text((parent/'README.md').read_text()+'\nProtocol-only derivative: VQAv2 uses vqav2_official_legacy; TextVQA uses textvqa_evalai. Images and selected records unchanged.\n')
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    return audit


def full_scienceqa_overlap(root,output,train_manifests,*,workers=8,reserve_bytes=120*1024**3):
    """Audit all image-bearing official ScienceQA validation and test examples.

    This is a preparation/exclusion audit, not evaluation metrics and not an
    automatic training-corpus change. Other official benchmark splits remain
    outside this full-split audit until separately downloaded and checked.
    """
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    spec=SOURCE_SPECS['scienceqa'];meta=_metadata_json(root/'eval-metadata'/(spec['repo'].replace('/','--')+'.json'))
    index,manifest_hashes=training_image_index(train_manifests,root/'eval-metadata',workers)
    config={'scope':'all_image_bearing_ScienceQA_validation_and_test','revision':meta['revision'],
            'training_manifest_hashes':manifest_hashes,'near_hamming':4}
    fingerprint=hashlib.sha256(canonical_json(config).encode()).hexdigest()
    if (output/'audit.json').exists():
        audit=json.loads((output/'audit.json').read_text())
        if audit['manifest_fingerprint']!=fingerprint: raise FileExistsError('Full ScienceQA audit inputs changed')
        return audit
    records=[];overlaps=[];counts=Counter();split_counts=Counter();official_rows={}
    for split,needle in [('validation','data/validation-'),('test','data/test-')]:
        file=next(f['path'] for f in meta['files'] if f['path'].startswith(needle))
        path=root/'eval-raw'/spec['repo'].replace('/','--')/file
        import pyarrow.parquet as pq
        actual_rows=pq.ParquetFile(path).metadata.num_rows
        split_key='val' if split=='validation' else 'test'
        expected_rows=len(_metadata_json(root/'eval-metadata/ScienceQA-original-pid_splits.json')[split_key])
        if actual_rows!=expected_rows: raise ValueError('ScienceQA mirror does not contain every official split row')
        official_rows[split]=actual_rows
        for row_index,row in iter_rows(path):
            source={'repo':spec['repo'],'revision':meta['revision'],'config':'default','split':split,
                    'shard':file,'source_is_hf_mirror':True}
            record=convert_row('scienceqa',row,row_index,source,root/'eval-metadata')
            if record is None: continue
            image_path,image_hash=materialize_image(record.pop('image_value'),output/'images',reserve_bytes)
            record.update(image_paths=[image_path],image_hashes=[image_hash],image_phashes=[image_perceptual_hash(image_path)])
            record['record_id']=hashlib.sha256(canonical_json(['scienceqa',split,record['source']['original_id']]).encode()).hexdigest()
            matches=overlap_candidates(record,index)
            records.append(record);split_counts[split]+=1
            if matches:
                overlaps.append({'record':record,'overlaps':matches})
                counts['exact_eval_records' if any(m['kind']=='exact' for m in matches) else 'near_only_eval_records']+=1
            if len(records)%500==0: print(json.dumps({'event':'full_scienceqa_overlap_progress','records':len(records),'split':split,'overlap_counts':dict(counts)}),flush=True)
    train_exclusions={};exact_train={};exact_images=set()
    for item in overlaps:
        for match in item['overlaps']:
            for ref in match['references']:
                key=ref['manifest'],ref['record_id']
                entry=train_exclusions.setdefault(key,{**ref,'not_applied':True,'eval_original_ids':[],'match_kinds':[]})
                entry['eval_original_ids'].append(item['record']['source']['original_id']);entry['match_kinds'].append(match['kind'])
                if match['kind']=='exact':
                    exact_train[key]=ref;exact_images.add(match['eval_image_hash'])
    files={'scienceqa_full_heldout_images.jsonl':records,'overlapping_scienceqa_examples.jsonl':overlaps,
           'training_exclusion_candidates.jsonl':list(train_exclusions.values()),'exact_training_exclusion_candidates.jsonl':list(exact_train.values())}
    checksums={}
    for name,rows in files.items():
        text=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
        ensure_disk_space(output,len(text.encode()),reserve_bytes);(output/name).write_text(text)
        checksums[name]=hashlib.sha256(text.encode()).hexdigest()
    audit={'schema_version':1,'status':'full_scienceqa_overlap_audited','manifest_fingerprint':fingerprint,
           'scope':config['scope'],'image_bearing_records':len(records),'split_record_counts':dict(split_counts),
           'official_total_rows_by_split':official_rows,'official_release_split_row_counts_match':True,
           **dict(counts),'exact_unique_images':len(exact_images),'unique_exact_training_records':len(exact_train),
           'exact_records_by_pilot_manifest':dict(Counter(Path(ref['manifest']).name for ref in exact_train.values())),
           'conservative_exclusion_records_by_pilot_manifest':dict(Counter(Path(ref['manifest']).name for ref in train_exclusions.values())),
           'training_was_modified':False,'trained_models_decontaminated':False,'training_manifest_hashes':manifest_hashes,
           'file_sha256':checksums,'other_benchmarks_full_audit':'not performed'}
    (output/'resolved_config.json').write_text(json.dumps(config,indent=2));(output/'audit.json').write_text(json.dumps(audit,indent=2))
    return audit


def derive_scienceqa_disjoint_training(parent,scienceqa_audit_dir,output):
    """Drop exact official ScienceQA val/test image groups from future SFT only.

    Caption manifests are byte-preserved for alignment checkpoint reuse. Existing
    profiles/models remain untouched. Near-only matches remain review candidates.
    This implements only the documented ScienceQA scope, not all-benchmark primary
    decontamination; primary full-suite exclusion still has to be completed.
    """
    parent=Path(parent).resolve();official=Path(scienceqa_audit_dir).resolve();output=Path(output).resolve()
    parent_audit=json.loads((parent/'audit.json').read_text())
    science_audit=json.loads((official/'audit.json').read_text())
    if not science_audit.get('official_release_split_row_counts_match'):
        raise ValueError('ScienceQA full original validation/test row coverage was not verified')
    reference=official/'scienceqa_full_heldout_images.jsonl'
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=science_audit['file_sha256'][reference.name]:
        raise ValueError('Official ScienceQA image manifest changed')
    heldout=defaultdict(list)
    for record in read_jsonl(reference):
        for image_hash in record['image_hashes']:
            heldout[image_hash].append({'original_id':record['source']['original_id'],'split':record['source']['split']})
    lineage={'parent_manifest_id':parent_audit['manifest_id'],'scienceqa_audit_fingerprint':science_audit['manifest_fingerprint'],
             'official_image_manifest_sha256':science_audit['file_sha256'][reference.name],
             'scope':'SFT exact ScienceQA validation/test image-group exclusion only','caption_manifests_unchanged':True,
             'near_only_exclusions_applied':False,'all_benchmark_exclusion_complete':False}
    if (output/'audit.json').exists():
        saved=json.loads((output/'audit.json').read_text())
        if saved.get('scienceqa_exclusion_lineage')!=lineage: raise FileExistsError('Existing training derivative has different provenance')
        return saved
    counts=Counter();excluded=[];contents={};stage_counts=Counter();excluded_by_source=Counter();retained_images=set()
    for stage in ['alignment','sft','text']:
        for partition in ['train','validation']:
            filename=f'{stage}_{partition}.jsonl';raw=(parent/filename).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=parent_audit[f'{stage}_{partition}_sha256']:
                raise ValueError('Parent training manifest changed')
            kept=[]
            for line in raw.decode().splitlines():
                record=json.loads(line);key=record['source']['repo'],record['source']['config']
                matching=sorted(set(record.get('image_hashes',[]))&set(heldout))
                if stage=='alignment' and matching:
                    raise ValueError('Caption alignment overlaps official ScienceQA; unchanged alignment reuse requires a separate retraining decision')
                if stage=='sft' and matching:
                    excluded.append({'record_id':record['record_id'],'source':record['source'],'parent_manifest':str(parent/filename),
                                     'partition':partition,'matched_image_hashes':matching,
                                     'official_scienceqa_references':[item for image_hash in matching for item in heldout[image_hash]],
                                     'reason':'exact_official_ScienceQA_heldout_image_group','applied_to_future_profile_only':True})
                    stage_counts[filename]+=1;excluded_by_source['/'.join(key)]+=1
                else:
                    kept.append(record);counts[key]+=1;retained_images.update(record.get('image_hashes',[]))
            # Preserve all alignment/text bytes exactly, including original order.
            contents[filename]=raw if stage!='sft' else ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept).encode()
    specs=json.loads((parent/'specs.json').read_text())
    specs['scienceqa_exclusion_lineage']=lineage
    for spec in specs['specs']:
        spec['limit']=counts[spec['repo'],spec['config']]
    audit=dict(parent_audit)
    audit.update(manifest_id=hashlib.sha256(canonical_json(specs).encode()).hexdigest(),
                 derived_from_manifest_id=parent_audit['manifest_id'],derived_from_path=str(parent),
                 scienceqa_exclusion_lineage=lineage,exact_excluded_records=len(excluded),
                 exact_excluded_records_by_manifest=dict(stage_counts),exact_excluded_records_by_source=dict(excluded_by_source),
                 existing_models_retrained=False,primary_full_eval_decontamination_complete=False,
                 unique_images=len(retained_images),sft_exact_scienceqa_overlap_remaining=0)
    output.mkdir(parents=True,exist_ok=True)
    for name,data in contents.items():
        (output/name).write_bytes(data)
        field=name.removesuffix('.jsonl')
        audit[field+'_records']=len(data.splitlines());audit[field+'_sha256']=hashlib.sha256(data).hexdigest()
    for name,source_counts in audit['source_counts'].items():
        repo,config=name.rsplit('/',1)
        source_counts['parent_profile_records']=source_counts['records'];source_counts['records']=counts[repo,config]
        source_counts['requested']=counts[repo,config];source_counts['actual_after_scienceqa_exact_exclusion']=counts[repo,config]
    (output/'specs.json').write_text(canonical_json(specs)+'\n')
    (output/'scienceqa_exact_exclusion_decisions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in excluded))
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    from scripts.audit_research_manifest import audit_prepared
    (output/'source_schema_language_audit.json').write_text(json.dumps(audit_prepared(output),indent=2))
    (output/'README.md').write_text('# Future SFT profile with exact ScienceQA image-group exclusion\n\n'+
        'Original caption and text manifests are byte-identical to parent; image paths are reused. Only SFT train/internal-validation records with exact image hashes occurring in the entire official ScienceQA validation/test image-bearing splits were removed. Near-only matches were not removed. Existing pilot runs are unchanged and were not retroactively retrained. Full all-benchmark primary decontamination remains pending.\n')
    return audit
