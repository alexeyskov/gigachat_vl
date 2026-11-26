#!/usr/bin/env python3
"""Create an immutable English-screened derivative while reusing parent images.

Screening is a documented Unicode-script/instruction heuristic, not perfect
language identification. Translation into English is allowed unless its actual
question/answer fails script screening; requested foreign-language output is
excluded even when the target language also uses the Latin alphabet.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_research_manifest import language_script_flag,audit_prepared
from src.dataset.research_data import extract_text_turns,iter_parquet_rows,partition_for_group
from src.dataset.vision_cache import canonical_json
from src.utils.research_storage import _atomic_json

LANGUAGES='English|French|Spanish|German|Russian|Urdu|Chinese|Arabic|Hindi|Korean|Japanese|Portuguese|Italian|Dutch|Polish|Ukrainian|Turkish|Vietnamese|Thai|Indonesian|Persian|Hebrew|Greek|Romanian|Czech|Swedish|Danish|Norwegian|Finnish'
TRANSLATE_TARGET=re.compile(r'\btranslat(?:e|ion|ing)\b[\s\S]{0,250}?\b(?:to|into)\s+(?:the\s+)?('+LANGUAGES+r')\b',re.I)
OUTPUT_TARGET=re.compile(r'^\s*(?:please\s+)?(?:respond|answer|reply|write)\b[\s\S]{0,100}?\bin\s+('+LANGUAGES+r')\b',re.I)


def filter_reasons(record):
    reasons=[]
    for field in ['question','answer']:
        flagged,stats=language_script_flag(record[field])
        if flagged: reasons.append({'reason':'non_latin_script_screen','field':field,**stats})
    target=TRANSLATE_TARGET.search(record['question']) or OUTPUT_TARGET.search(record['question'])
    if target and target.group(1).lower()!='english':
        reasons.append({'reason':'foreign_output_instruction','requested_language':target.group(1).lower()})
    return reasons


def derive(parent,output):
    parent=Path(parent).resolve();output=Path(output).resolve()
    parent_audit=json.loads((parent/'audit.json').read_text())
    if output.exists() and (output/'audit.json').exists():
        previous=json.loads((output/'audit.json').read_text())
        if previous.get('derived_from_manifest_id')!=parent_audit['manifest_id']:
            raise FileExistsError('Existing derivative belongs to a different parent manifest')
        audit_prepared(output)
        return previous
    payload=json.loads((parent/'specs.json').read_text())
    records={};seen=set();decisions=[];source_counts=Counter()
    for stage in ['alignment','sft','text']:
        for partition in ['train','validation']:
            path=parent/f'{stage}_{partition}.jsonl'
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=parent_audit[f'{stage}_{partition}_sha256']:
                raise ValueError('Parent manifest changed')
            values=[]
            for line in raw.decode().splitlines():
                record=json.loads(line);seen.add(record['record_id'])
                reasons=filter_reasons(record)
                if reasons:
                    if stage!='text': raise ValueError('A non-text record needs manual source replenishment before derivative creation')
                    decisions.append({'record_id':record['record_id'],'source':record['source'],'reasons':reasons,'action':'excluded'})
                else:
                    values.append(record);source_counts[record['source']['repo'],record['source']['config']]+=1
            records[stage,partition]=values
    replenished=[]
    for spec in payload['specs']:
        key=spec['repo'],spec['config']
        if source_counts[key]>=spec['limit']: continue
        if spec['kind']!='text': raise ValueError('Only text sources can be replenished without image extraction')
        rows=0
        for shard,row in iter_parquet_rows(spec['files']):
            rows+=1
            source={'repo':spec['repo'],'revision':spec['revision'],'config':spec['config'],'split':'train',
                    'row_id':str(row.get('id',f'{Path(shard).name}:{rows-1}')),'shard':Path(shard).name,
                    'original_source':row.get('source')}
            for turn_index,(messages,answer) in enumerate(list(extract_text_turns(row))[:1]):
                record_id=hashlib.sha256(canonical_json([[],messages,answer]).encode()).hexdigest()
                if record_id in seen: continue
                seen.add(record_id)
                partition=partition_for_group(source['repo']+':'+source['row_id'],payload['seed'],payload['validation_fraction'])
                record={'record_id':record_id,'image_paths':[],'image_hashes':[],'image':None,
                        'question':messages[-1]['content'],'answer':answer,'messages':messages,'source':source,
                        'dialogue_id':source['repo']+':'+source['config']+':'+source['row_id'],'turn_index':turn_index,
                        'task_group':spec['task_group'],'partition':partition,'language':'en'}
                reasons=filter_reasons(record)
                if reasons:
                    decisions.append({'record_id':record_id,'source':source,'reasons':reasons,'action':'excluded_during_replenishment'})
                    continue
                records['text',partition].append(record);source_counts[key]+=1;replenished.append(record_id)
            if source_counts[key]>=spec['limit']: break
        if source_counts[key]!=spec['limit']: raise ValueError('Not enough screened English text records to replenish quota')
    payload.update(derived_from_manifest_id=parent_audit['manifest_id'],
                   filter_policy={'version':1,'method':'Unicode Latin script screen + explicit foreign-output instruction targets',
                                  'perfect_language_identification':False,'english_translation_target_allowed':True,
                                  'reuse_parent_image_paths':True},
                   excluded_record_ids=sorted(d['record_id'] for d in decisions),replenished_record_ids=replenished)
    audit=dict(parent_audit)
    audit.update(manifest_id=hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
                 derived_from_manifest_id=parent_audit['manifest_id'],derived_from_path=str(parent),
                 english_filter_policy=payload['filter_policy'],excluded_record_count=len(decisions),
                 replenished_record_count=len(replenished),
                 filter_reason_counts=dict(Counter(reason['reason'] for d in decisions for reason in d['reasons'])))
    output.mkdir(parents=True,exist_ok=True)
    for (stage,partition),values in records.items():
        text=''.join(json.dumps(record,ensure_ascii=False)+'\n' for record in values)
        path=output/f'{stage}_{partition}.jsonl';path.write_text(text)
        audit[f'{stage}_{partition}_records']=len(values)
        audit[f'{stage}_{partition}_sha256']=hashlib.sha256(text.encode()).hexdigest()
    # Record counts remain exact; source scanned_rows now describes the parent
    # extraction, while derivative replenishment is recorded separately.
    audit['source_counts']={name:{**value,'actual_after_english_filter':source_counts[(name.rsplit('/',1)[0],name.rsplit('/',1)[1])]} for name,value in parent_audit['source_counts'].items()}
    (output/'specs.json').write_text(canonical_json(payload)+'\n')
    (output/'english_filter_decisions.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in decisions))
    _atomic_json(output/'audit.json',audit)
    screening=audit_prepared(output)
    _atomic_json(output/'source_schema_language_audit.json',screening)
    return audit


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--parent',default='datasets/research/pilot-v1')
    parser.add_argument('--output',default='datasets/research/pilot-en-v1')
    args=parser.parse_args()
    result=derive(args.parent,args.output)
    print(json.dumps({key:value for key,value in result.items() if key not in ['source_counts']},indent=2))

if __name__=='__main__': main()
