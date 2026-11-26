#!/usr/bin/env python3
"""Audit prepared research manifests without modifying their immutable contents.

English-source provenance and script screening are reported separately: a Latin
script screen is not a general-purpose language classifier or an eval leak audit.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import hashlib
import json
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dataset.research_data import partition_for_group
from src.dataset.vision_cache import canonical_json
from src.utils.research_storage import _atomic_json

ENGLISH_SOURCES={('dnth/pixmo-cap-images','data'),('HuggingFaceM4/FineVision','vqav2'),
 ('HuggingFaceM4/FineVision','scienceqa'),('HuggingFaceM4/FineVision','textvqa'),
 ('HuggingFaceM4/FineVision','chartqa'),('HuggingFaceTB/smoltalk','all')}
REQUIRED={'record_id','image_paths','image_hashes','question','answer','source','task_group','partition','language'}


def language_script_flag(text):
    letters=[char for char in text if char.isalpha()]
    latin=sum('LATIN' in unicodedata.name(char,'') for char in letters)
    other=len(letters)-latin
    ratio=latin/max(1,len(letters))
    return len(letters)>=8 and other>=8 and ratio<(.8 if len(letters)>=30 else .5),{'alphabetic':len(letters),'latin_fraction':ratio}


def audit_prepared(output):
    import pyarrow.parquet as pq
    root=Path(output).resolve()
    audit=json.loads((root/'audit.json').read_text())
    specs_payload=json.loads((root/'specs.json').read_text())
    actual_manifest_id=hashlib.sha256(canonical_json(specs_payload).encode()).hexdigest()
    if actual_manifest_id!=audit['manifest_id']: raise ValueError('Spec fingerprint does not match audit manifest ID')
    spec_by_source={(spec['repo'],spec['config']):spec for spec in specs_payload['specs']}
    raw_schemas=[]
    for source,spec in spec_by_source.items():
        if spec['kind'] not in ['caption','finevision','text']: raise ValueError('Unknown source kind')
        required_columns={'caption':{'image','caption'},'finevision':{'images','texts','source'},'text':{'messages','source'}}[spec['kind']]
        for filename in spec['files']:
            path=Path(filename)
            if 'train-' not in path.name: raise ValueError(f'Non-train shard selected: {path}')
            parquet=pq.ParquetFile(path)
            if not required_columns.issubset(parquet.schema_arrow.names): raise ValueError(f'Source schema mismatch: {path}')
            # hf_hub_download(local_dir=...) records the exact pinned commit.
            candidates=[parent/'.cache/huggingface/download'/path.relative_to(parent).with_suffix(path.suffix+'.metadata') for parent in path.parents if parent.is_relative_to(ROOT/'datasets')]
            metadata=next((p for p in candidates if p.exists()),None)
            if metadata is None: raise ValueError(f'No downloaded shard provenance metadata for {path}')
            commit=metadata.read_text().splitlines()[0]
            if commit!=spec['revision']: raise ValueError(f'Source shard commit mismatch: {path}')
            raw_schemas.append({'source':list(source),'shard':path.name,'revision':commit,'columns':parquet.schema_arrow.names,'rows':parquet.metadata.num_rows})
    ids=set();partitions={};counts=Counter();stagecounts=Counter();unknown_sources=set();flags=[]
    for stage in ['alignment','sft','text']:
        for partition in ['train','validation']:
            path=root/f'{stage}_{partition}.jsonl'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=audit[f'{stage}_{partition}_sha256']:
                raise ValueError(f'Manifest changed: {path}')
            local_count=0
            for line in path.read_text().splitlines():
                if not line.strip(): continue
                row=json.loads(line);local_count+=1;stagecounts[stage,partition]+=1
                if not REQUIRED.issubset(row): raise ValueError(f'Manifest missing columns: {row.get("record_id")}')
                if row['record_id'] in ids: raise ValueError('Duplicate record ID across manifests')
                ids.add(row['record_id'])
                if row['partition']!=partition or row['language']!='en': raise ValueError('Partition/language label mismatch')
                source=row['source'];key=(source['repo'],source['config'])
                if key not in spec_by_source: raise ValueError('Manifest source not in source specs')
                spec=spec_by_source[key]
                if source['revision']!=spec['revision'] or source['split']!='train': raise ValueError('Source revision/split mismatch')
                if source['shard'] not in {Path(p).name for p in spec['files']}: raise ValueError('Source shard not selected by args')
                if row['task_group']!=spec['task_group']: raise ValueError('Task-group label differs from actual source args')
                if key not in ENGLISH_SOURCES: unknown_sources.add(key)
                if len(row['image_paths'])!=len(row['image_hashes']): raise ValueError('Image order/hash count mismatch')
                for image_path,image_hash in zip(row['image_paths'],row['image_hashes']):
                    if not Path(image_path).is_file() or image_hash not in Path(image_path).name:
                        raise ValueError('Materialized image missing or filename differs from content ID')
                    if image_hash in partitions and partitions[image_hash]!=partition: raise ValueError('Image occurs in train and validation')
                    partitions[image_hash]=partition
                group=row['image_hashes'][0] if row['image_hashes'] else source['repo']+':'+source['row_id']
                expected=partition_for_group(group,specs_payload['seed'],specs_payload['validation_fraction'])
                if expected!=partition: raise ValueError('Image-group split changed')
                counts[key]+=1
                for field in ['question','answer']:
                    flagged,stats=language_script_flag(row[field])
                    if flagged: flags.append({'record_id':row['record_id'],'field':field,'source':list(key),**stats})
            if local_count!=audit[f'{stage}_{partition}_records']: raise ValueError('Manifest count differs from audit')
    quotas=[]
    for key,spec in spec_by_source.items():
        count=counts[key]
        if count!=audit['source_counts']['/'.join(key)]['records']: raise ValueError('Source record count differs from audit')
        quotas.append({'source':list(key),'requested':spec['limit'],'actual':count,'quota_satisfied':count==spec['limit']})
    return {'schema_version':1,'status':'passed' if not unknown_sources and not flags and all(q['quota_satisfied'] for q in quotas) else 'review_required',
            'manifest_id':audit['manifest_id'],'manifest_columns_valid':True,'source_arguments_match':True,
            'raw_shard_revision_and_schema_valid':True,'train_validation_image_overlap':0,
            'record_count':len(ids),'unique_images':len(partitions),'source_quotas':quotas,'raw_shards':raw_schemas,
            'english_source_allowlist_matched':not unknown_sources,'unknown_english_source_pairs':[list(x) for x in sorted(unknown_sources)],
            'language_screen_method':'Unicode Latin script heuristic, not a general language identifier',
            'language_script_flagged_fields':len(flags),'language_script_flags':flags,
            'official_evaluation_overlap_audit':'pending; this audit is internal source/schema/split/language screening only'}


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',default='datasets/research/pilot-v1')
    args=parser.parse_args()
    result=audit_prepared(args.output)
    target=Path(args.output)/'source_schema_language_audit.json'
    _atomic_json(target,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['raw_shards','language_script_flags']},indent=2))

if __name__=='__main__':main()
