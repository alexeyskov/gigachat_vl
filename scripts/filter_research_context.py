#!/usr/bin/env python3
"""Immutable, exact-tokenizer context screening without loading language weights.

Only text rows with zero available supervised positions are removed. Alignment
and multimodal rows must all fit and are byte-preserved. Answers are never given
an extra target cap; retained answer truncation is the trainer's context limit.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dataset.vision_cache import canonical_json,ensure_disk_space

MANIFESTS=tuple(f'{stage}_{split}.jsonl' for stage in ('alignment','sft','text') for split in ('train','validation'))
DEFAULT_MODEL='Tiiny/SmallThinker-4BA0.6B-Instruct'

def digest(raw):return hashlib.sha256(raw).hexdigest()

def measure_record(record,tokenize,*,max_length=4096,vision_tokens=729):
    pre,post,answer=tokenize(record)
    if not answer:raise ValueError('Tokenizer returned an empty answer')
    expanded=len(pre)+len(post)+len(record.get('image_paths',[]))*vision_tokens
    available=max_length-expanded
    return {'expanded_prompt_tokens':expanded,'available_target_tokens':available,
            'original_target_tokens':len(answer),'supervised_target_tokens':min(len(answer),max(0,available)),
            'answer_truncated':len(answer)>max(0,available),'expanded_prompt_overflow':available<1}

def preflight_rows(rows,tokenize,*,max_length=4096,vision_tokens=729):
    report={'records':0,'text_records':0,'image_records':0,'text_target_tokens':0,'multimodal_target_tokens':0,
            'answer_truncated_records':0,'expanded_prompt_overflow':[],'max_expanded_prompt_tokens':0}
    for row in rows:
        stats=measure_record(row,tokenize,max_length=max_length,vision_tokens=vision_tokens)
        image=bool(row.get('image_paths'))
        report['records']+=1;report['image_records' if image else 'text_records']+=1
        report['multimodal_target_tokens' if image else 'text_target_tokens']+=stats['supervised_target_tokens']
        report['answer_truncated_records']+=int(stats['answer_truncated'])
        report['max_expanded_prompt_tokens']=max(report['max_expanded_prompt_tokens'],stats['expanded_prompt_tokens'])
        if stats['expanded_prompt_overflow']:report['expanded_prompt_overflow'].append(row['record_id'])
    total=report['text_target_tokens']+report['multimodal_target_tokens']
    report['text_supervised_token_fraction']=report['text_target_tokens']/total if total else 0.0
    return report

def make_native_tokenizer(model=DEFAULT_MODEL):
    # Only local, pinned config/tokenizer assets; no weights, vision encoder or GPU.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from scripts.run_research_stage import ResearchModel,resolve_model,training_prompt_messages
    from transformers import AutoTokenizer,AutoConfig
    from src.model.smallthinker import register_smallthinker
    register_smallthinker()
    path,ref=resolve_model(model)
    tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=False,local_files_only=True)
    config=AutoConfig.from_pretrained(path,trust_remote_code=False,local_files_only=True)
    dummy=SimpleNamespace(tokenizer=tokenizer,base=SimpleNamespace(config=config),args=SimpleNamespace(prompt_schema_version=2))
    tokenize=lambda row:ResearchModel.tokenize(dummy,row)
    code=inspect.getsource(ResearchModel.tokenize)+'\n'+inspect.getsource(training_prompt_messages)
    assets={name:digest((Path(path)/name).read_bytes()) for name in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja') if (Path(path)/name).is_file()}
    binding={'model_ref':ref,'prompt_schema_version':2,'tokenizer_asset_sha256':assets,
             'tokenize_source_sha256':digest(code.encode()),'tokenize_entrypoint':'scripts.run_research_stage.ResearchModel.tokenize',
             'language_weights_loaded':False,'vision_encoder_loaded':False,'gpu_used':False,'hf_offline':True}
    return tokenize,binding

def verify_derivative(output,lineage):
    audit=json.loads((output/'audit.json').read_text())
    if audit.get('context_filter_lineage')!=lineage:raise FileExistsError('Existing context derivative has different provenance/policy')
    for name in MANIFESTS:
        if digest((output/name).read_bytes())!=audit[name.removesuffix('.jsonl')+'_sha256']:
            raise ValueError('Immutable context derivative manifest changed')
    if digest((output/'specs.json').read_bytes())!=audit['specs_file_sha256']:raise ValueError('Immutable derivative specs changed')
    if digest((output/'context_exclusion_decisions.jsonl').read_bytes())!=audit['context_exclusion_decisions_sha256']:
        raise ValueError('Immutable context decisions changed')
    return audit

def derive(parent,output,tokenize,binding,*,max_length=4096,vision_tokens=729,reserve_bytes=120*2**30,run_source_audit=True):
    parent=Path(parent).resolve();output=Path(output).resolve()
    if parent==output:raise ValueError('Derivative output must differ from parent')
    parent_audit=json.loads((parent/'audit.json').read_text());parent_specs=(parent/'specs.json').read_bytes()
    raws={name:(parent/name).read_bytes() for name in MANIFESTS}
    for name,raw in raws.items():
        if digest(raw)!=parent_audit[name.removesuffix('.jsonl')+'_sha256']:raise ValueError('Parent manifest changed')
    policy={'version':2,'max_length':max_length,'fixed_visual_tokens_per_image':vision_tokens,
            'drop_condition':'expanded_prompt_tokens >= max_length (no supervised target position)',
            'only_text_rows_may_be_excluded':True,'caption_and_multimodal_overflow_is_fatal':True,
            'caption_and_multimodal_manifests_byte_preserved':True,'reuse_parent_image_paths':True,
            'arbitrary_target_cap':None,'answer_truncation':'only existing trainer max_length residual context',
            'replenishment_applied':False}
    lineage={'parent_manifest_id':parent_audit['manifest_id'],'parent_path':str(parent),
             'parent_audit_sha256':digest((parent/'audit.json').read_bytes()),'parent_specs_sha256':digest(parent_specs),
             'parent_manifest_sha256':{name:digest(raw) for name,raw in raws.items()},'policy':policy,'tokenizer':binding}
    if (output/'audit.json').is_file():return verify_derivative(output,lineage)
    if output.exists():raise FileExistsError('Refusing incomplete or unrelated output directory')
    contents={};decisions=[];counts=Counter();by_manifest=Counter();reports={}
    for name,raw in raws.items():
        kept=[];rows=[]
        for line in raw.splitlines(keepends=True):
            if not line.strip():kept.append(line);continue
            row=json.loads(line);stats=measure_record(row,tokenize,max_length=max_length,vision_tokens=vision_tokens)
            if stats['expanded_prompt_overflow']:
                if not name.startswith('text_'):raise ValueError(f'Caption/multimodal context overflow is fatal: {name} {row["record_id"]}')
                decisions.append({'record_id':row['record_id'],'source':row['source'],'parent_manifest':str(parent/name),
                    'partition':row['partition'],'reason':'expanded_prompt_leaves_no_supervised_target_position',**stats})
                by_manifest[name]+=1
            else:
                kept.append(line);rows.append(row);counts[row['source']['repo'],row['source']['config']]+=1
        contents[name]=b''.join(kept)
        reports[name]=preflight_rows(rows,tokenize,max_length=max_length,vision_tokens=vision_tokens)
        if not name.startswith('text_') and contents[name]!=raw:raise AssertionError('Non-text manifest bytes changed')
    specs=json.loads(parent_specs);specs['context_filter_lineage']=lineage
    specs['context_excluded_record_ids']=sorted(row['record_id'] for row in decisions)
    for spec in specs['specs']:spec['limit']=counts[spec['repo'],spec['config']]
    specs_raw=(canonical_json(specs)+'\n').encode()
    decision_raw=''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in decisions).encode()
    audit=deepcopy(parent_audit)
    audit.update(manifest_id=digest(canonical_json(specs).encode()),derived_from_manifest_id=parent_audit['manifest_id'],
        derived_from_path=str(parent),context_filter_lineage=lineage,context_excluded_records=len(decisions),
        context_excluded_records_by_manifest=dict(by_manifest),context_excluded_record_ids=specs['context_excluded_record_ids'],
        context_preflight=reports,context_exclusion_decisions_sha256=digest(decision_raw),specs_file_sha256=digest(specs_raw),
        source_code_run_metadata={'filter_script_sha256':digest(Path(__file__).read_bytes()),
             'python_version':sys.version,'utc_started':datetime.now(timezone.utc).isoformat()},existing_models_retrained=False)
    for name,raw in contents.items():
        key=name.removesuffix('.jsonl');audit[key+'_records']=reports[name]['records'];audit[key+'_sha256']=digest(raw)
    for name,source in audit['source_counts'].items():
        repo,config=name.rsplit('/',1);source['parent_context_profile_records']=source['records']
        source['records']=counts[repo,config];source['requested']=counts[repo,config];source['actual_after_context_filter']=counts[repo,config]
    output.parent.mkdir(parents=True,exist_ok=True)
    ensure_disk_space(output.parent,sum(len(x) for x in contents.values())+len(specs_raw)+len(parent_specs)+2**20,reserve_bytes)
    staging=output.parent/(output.name+'.staging-'+uuid.uuid4().hex)
    try:
        staging.mkdir()
        for name,raw in contents.items():(staging/name).write_bytes(raw)
        (staging/'specs.json').write_bytes(specs_raw);(staging/'parent_specs.json').write_bytes(parent_specs)
        (staging/'context_exclusion_decisions.jsonl').write_bytes(decision_raw)
        (staging/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        if run_source_audit:
            from scripts.audit_research_manifest import audit_prepared
            result=audit_prepared(staging)
            if result['status']!='passed':raise ValueError('Derivative source/schema/language audit did not pass')
            (staging/'source_schema_language_audit.json').write_text(json.dumps(result,indent=2)+'\n')
        staging.rename(output)
    finally:
        if staging.exists():shutil.rmtree(staging)
    return audit

def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--parent',default='datasets/research/pilot-en-scienceqa-disjoint-v1')
    p.add_argument('--output',default='datasets/research/pilot-en-scienceqa-context-v2')
    p.add_argument('--model',default=DEFAULT_MODEL);p.add_argument('--max-length',type=int,default=4096)
    p.add_argument('--vision-tokens',type=int,default=729);p.add_argument('--reserve-gb',type=float,default=120)
    p.add_argument('--preflight-manifest-dir');p.add_argument('--preflight-report')
    args=p.parse_args();tokenize,binding=make_native_tokenizer(args.model)
    if args.preflight_manifest_dir:
        reports={name:dict(preflight_rows(read_jsonl(Path(args.preflight_manifest_dir)/name),tokenize,max_length=args.max_length,vision_tokens=args.vision_tokens),
                           manifest_sha256=digest((Path(args.preflight_manifest_dir)/name).read_bytes())) for name in ('sft_train.jsonl','sft_validation.jsonl')}
        result={'schema_version':1,'status':'passed' if all(not r['expanded_prompt_overflow'] for r in reports.values()) else 'failed',
                'scope':'exact tokenizer target-context preflight, no model weights/GPU; not official generation evaluation',
                **binding,'max_length':args.max_length,'fixed_visual_tokens_per_image':args.vision_tokens,'reports':reports}
        profile=Path(args.output)
        if (profile/'audit.json').is_file():
            audit=json.loads((profile/'audit.json').read_text())
            lineage=audit['context_filter_lineage']
            parent=Path(lineage['parent_path'])
            preserved={name:((profile/name).read_bytes()==(parent/name).read_bytes()) for name in MANIFESTS if not name.startswith('text_')}
            result['profile_context_audit']={'manifest_id':audit['manifest_id'],'source_profile':str(profile.resolve()),
                'excluded_text_records':audit['context_excluded_records'],'excluded_record_ids':audit['context_excluded_record_ids'],
                'caption_multimodal_bytes_preserved':preserved,'source_manifests':audit['context_preflight'],
                'source_schema_language_audit_status':json.loads((profile/'source_schema_language_audit.json').read_text())['status']}
            if not all(preserved.values()) or any(r['expanded_prompt_overflow'] for r in audit['context_preflight'].values()):result['status']='failed'
        if args.preflight_report:Path(args.preflight_report).write_text(json.dumps(result,indent=2)+'\n')
    else:
        result=derive(args.parent,args.output,tokenize,binding,max_length=args.max_length,vision_tokens=args.vision_tokens,reserve_bytes=int(args.reserve_gb*2**30))
        result={key:result[key] for key in ('manifest_id','context_excluded_records','context_excluded_records_by_manifest','text_train_records','text_validation_records','context_excluded_record_ids')}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
