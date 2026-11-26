#!/usr/bin/env python3
"""Opt-in fixed B4 held-out evaluation, isolated from frozen B1 artifacts.

Every checkpoint must pass native B1/B4 token parity on32 fixed records across
all four image modes. Four-row journal shards commit atomically: resume never
changes original batch membership or reruns a committed prediction batch.
"""
from __future__ import annotations
from collections import Counter
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.research_inference import load_research_checkpoint,generate_prediction
from src.eval.research_batch_inference import generate_batch_predictions,eos_token_ids,require_idle_gpu
from src.eval.research_ablation import transform_image_ablation,write_transformed_manifest
from src.eval.research_metrics import score_prediction,aggregate_scores
from src.dataset.vision_cache import image_content_hash,canonical_json
from src.utils.research_checkpoint_compat import checkpoint_identity,same_checkpoint_identity
from src.utils.research_storage import DiskBudget,_atomic_json

MODES=('none','shuffle','blank','text_only')
BENCHMARK_GROUPS={'vqav2':'general_vqa','gqa':'general_vqa','textvqa':'ocr_documents','docvqa':'ocr_documents',
                  'chartqa':'charts_diagrams','ai2d':'charts_diagrams','scienceqa':'knowledge_reasoning','aokvqa':'knowledge_reasoning'}
ROW_FIELDS=('record_id','question','image_hashes','answers','scorer','task_group','source','benchmark','choices','scoring_protocol','image_ablation')
FROZEN_SOURCES=('src/eval/research_inference.py','scripts/eval_research_checkpoint.py','src/eval/research_metrics.py',
 'scripts/run_research_stage.py','src/model/research_moe.py','src/model/smallthinker.py','src/model/research_quantization.py',
 'src/model/research_vision.py','src/model/gigachat_vl.py','src/dataset/vision_cache.py','scripts/extract_vision_features.py',
 'src/eval/research_ablation.py','src/utils/research_checkpoint_compat.py','src/utils/research_checkpoint_bundle.py','src/utils/research_checkpoint_io.py')
PROFILE={'schema_version':1,'name':'batch4_v1','batch_size':4,'manifest_records':512,'max_new_tokens':64,'seed':42,
 'modes':list(MODES),'generation':'native greedy KV cache, num_beams1','padding_side':'left',
 'position_ids':'real-token attention-mask cumulative positions; left padding position0',
 'eos':'native generation-config IDs; retain first EOS, discard only later batch padding',
 'membership':'original manifest order, fixed non-overlapping four-row batches; no bucketing/repacking/fallback',
 'guard_selection':'one original four-row batch per benchmark, minimum SHA256(seed42,benchmark,start-index)',
 'guard_modes':list(MODES),'guard_exact_generated_token_ids_required':True,
 'journal':'atomic complete four-row shard; original global batch index and four IDs',
 'score':'unchanged frozen strict scorers, equal four-task-group macro'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(canonical_json(value).encode()).hexdigest()
def protocol_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def read_rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def source_identity():return {p:sha(ROOT/p) for p in (*FROZEN_SOURCES,'src/eval/research_batch_inference.py','scripts/eval_research_checkpoint_batch4.py')}
def package_identity():
    import torch
    result={}
    for name in ('torch','transformers','peft','bitsandbytes','tokenizers','huggingface-hub'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return {'python':sys.version,'packages':result,'cuda_runtime':torch.version.cuda,
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}

def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():raise ValueError('Artifact must not be a symlink')
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.replace(temp,path)
        directory=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(temp):os.unlink(temp)

def public_record(row):
    # References/options/source annotations never reach either generator. The
    # declared question already contains all supplied MCQ options/context.
    return {key:row[key] for key in ('record_id','question','image_paths','image_hashes') if key in row}


def validate_manifest(rows):
    if len(rows)!=512 or len({r['record_id'] for r in rows})!=512:raise ValueError('Exactly512 unique original records are required')
    if Counter(r.get('benchmark') for r in rows)!=Counter({b:64 for b in BENCHMARK_GROUPS}):raise ValueError('Exactly64 records from each declared benchmark are required')
    for row in rows:
        if row['task_group']!=BENCHMARK_GROUPS[row['benchmark']]:raise ValueError('Benchmark/task-group mapping differs')
        if len(row.get('image_paths',[]))!=1 or len(row.get('image_hashes',[]))!=1:raise ValueError('Original records must contain one ordered image/hash')
        if not isinstance(row.get('question'),str) or not row['question'].strip():raise ValueError('Question is empty')
        if not isinstance(row.get('answers'),list) or not row['answers']:raise ValueError('Reference list missing')
    groups={r['task_group'] for r in rows}
    if groups!=set(BENCHMARK_GROUPS.values()):raise ValueError('All four task groups are mandatory')


def guard_indices(rows):
    selected=[]
    for benchmark in BENCHMARK_GROUPS:
        candidates=[i for i in range(0,len(rows),4) if {r['benchmark'] for r in rows[i:i+4]}=={benchmark}]
        if not candidates:raise ValueError('Manifest lacks an intact four-row production batch for a benchmark')
        start=min(candidates,key=lambda i:digest([42,benchmark,i]))
        selected.extend(range(start,start+4))
    result=sorted(selected)
    if len(result)!=32 or len(set(result))!=32:raise ValueError('Guard must select32 distinct original rows')
    return result


def transformed_inputs(rows,manifest_sha,base,reserve_bytes):
    transformed={};descriptors={}
    for mode in MODES:
        output=base/mode
        changed,descriptor=transform_image_ablation(rows,variant=mode,seed=42,output_dir=output,
            source_manifest_sha256=manifest_sha,reserve_bytes=reserve_bytes)
        if mode!='none':write_transformed_manifest(changed,output/'transformed_manifest.jsonl',reserve_bytes=reserve_bytes)
        transformed[mode]=changed;descriptors[mode]=descriptor
    return transformed,descriptors


def rgb_census(rows):
    images={}
    for row in rows:
        for path,image_hash in zip(row['image_paths'],row['image_hashes']):
            if image_hash not in images:
                if image_content_hash(path)!=image_hash:raise ValueError('Original canonical RGB hash differs from manifest')
                images[image_hash]=str(Path(path).resolve())
    return {'unique_images':len(images),'canonical_rgb_hashes_sha256':digest(sorted(images)),
            'record_ordered_rgb_hashes_sha256':digest([[r['record_id'],r['image_hashes']] for r in rows])}


def guard_binding(identity,manifest_sha,rows,descriptors,rgb,sources,environment):
    indices=guard_indices(rows)
    return {'schema_version':1,'profile':PROFILE,'profile_sha256':digest(PROFILE),'checkpoint_identity':identity,
      'manifest_sha256':manifest_sha,'record_ids':[rows[i]['record_id'] for i in indices],
      'original_batch_start_indices':[indices[i] for i in range(0,32,4)],
      'transformed_manifests':{mode:descriptors[mode] for mode in MODES},'rgb_census':rgb,
      'source_sha256':sources,'environment':environment}


def validate_guard(report,binding):
    if report.get('binding')!=binding or report.get('binding_sha256')!=digest(binding):raise ValueError('Guard checkpoint/source/profile/manifest binding differs')
    if report.get('status')!='passed' or set(report.get('modes',{}))!=set(MODES):raise ValueError('All four native B1/B4 parity modes must pass')
    for mode,proof in report['modes'].items():
        if (proof.get('status')!='passed' or proof.get('record_ids')!=binding['record_ids'] or
            len(proof.get('native_b1_token_ids',[]))!=32 or proof.get('native_b1_token_ids')!=proof.get('batch4_token_ids') or
            proof.get('exact_token_parity') is not True):raise ValueError('Guard does not prove exact32-row B1/B4 token parity')
    if not isinstance(report.get('eos_token_ids'),list):raise ValueError('Guard native EOS policy missing')
    return report


def run_guard(model,transformed,binding,path,progress):
    path=Path(path)
    if path.exists():
        report=json.loads(path.read_text())
        if report.get('binding')!=binding:raise ValueError('Existing guard belongs to a different profile/checkpoint/source/manifest')
        if report.get('status')=='passed':return validate_guard(report,binding)
        if report.get('status')=='failed':raise ValueError('A failed parity guard blocks this namespace; no implicit fallback or overwrite')
    else:report={'schema_version':1,'status':'running','binding':binding,'binding_sha256':digest(binding),'modes':{}}
    ids=binding['record_ids'];eos=sorted(eos_token_ids(model))
    if 'eos_token_ids' in report and report['eos_token_ids']!=eos:raise ValueError('Native EOS config changed during guard resume')
    report['eos_token_ids']=eos;_atomic_json(path,report)
    try:
        for mode in MODES:
            if report['modes'].get(mode,{}).get('status')=='passed':continue
            records_by_id={r['record_id']:r for r in transformed[mode]};selected=[records_by_id[k] for k in ids]
            native=[];batch=[]
            for i,row in enumerate(selected):
                native.append(generate_prediction(model,public_record(row),max_new_tokens=64))
                progress('guard_native_b1',variant=mode,completed_records=i+1,total_records=32)
            for offset in range(0,32,4):
                batch.extend(generate_batch_predictions(model,[public_record(r) for r in selected[offset:offset+4]],max_new_tokens=64))
                progress('guard_batch4',variant=mode,completed_records=offset+4,total_records=32)
            a=[r['generated_token_ids'] for r in native];b=[r['generated_token_ids'] for r in batch]
            if any(native[i].get('prompt_sha256')!=batch[i].get('prompt_sha256') for i in range(32)):
                raise ValueError('Native B1/B4 prompt identity differs')
            proof={'status':'passed' if a==b else 'failed','record_ids':ids,'native_b1_token_ids':a,'batch4_token_ids':b,
                   'exact_token_parity':a==b,'mismatched_record_ids':[ids[i] for i in range(32) if a[i]!=b[i]]}
            report['modes'][mode]=proof;_atomic_json(path,report)
            if a!=b:raise ValueError(f'Native B1/B4 token parity failed for{mode}; no batch fallback allowed')
        report['status']='passed';_atomic_json(path,report);return validate_guard(report,binding)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'failed',
                      error_type=type(error).__name__,error=str(error));_atomic_json(path,report);raise


def make_protocol(identity,manifest_sha,records,descriptor,rgb,guard,sources,environment):
    return {'schema_version':1,'profile':PROFILE,'profile_sha256':digest(PROFILE),'checkpoint_sha256':identity['sha256'],
      'checkpoint_artifact_identity':identity,'manifest_sha256':manifest_sha,'manifest_records':512,
      'selected_record_ids_sha256':protocol_digest([r['record_id'] for r in records]),'image_ablation':descriptor,'rgb_census':rgb,
      'generation':{'max_new_tokens':64,'do_sample':False,'num_beams':1,'use_cache':True,'batch_size':4,
        'padding_side':'left','position_ids':'real-token cumulative','eos_token_ids':guard['eos_token_ids'],
        'autocast':'cuda_bf16_preserving_master_dtypes','prompt':'native_user_only_image_marker_no_gold',
        'logits_to_keep':'1_when_supported','no_batch_size_fallback':True},
      'fixed_batch_membership_sha256':digest([[r['record_id'] for r in records[i:i+4]] for i in range(0,512,4)]),
      'parity_guard_binding_sha256':guard['binding_sha256'],'parity_guard_proof_sha256':digest(guard),
      'source_sha256':sources,'environment':environment}


def validate_prediction(row,expected,protocol_sha,index,members,eos):
    sentinel=object()
    if any(row.get(key,sentinel)!=expected.get(key,sentinel) for key in ROW_FIELDS):raise ValueError('Committed prediction semantic/image/source metadata differs')
    if row.get('protocol_sha256')!=protocol_sha or row.get('batch_index')!=index or row.get('batch_record_ids')!=members:
        raise ValueError('Committed prediction protocol or fixed batch membership differs')
    ids=row.get('generated_token_ids')
    if (not isinstance(ids,list) or not 1<=len(ids)<=64 or any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in ids)
        or row.get('generated_tokens')!=len(ids) or not isinstance(row.get('prediction'),str)):
        raise ValueError('Committed generated token metadata differs')
    if any(token in eos for token in ids[:-1]):raise ValueError('Committed prediction contains post-EOS padding tokens')
    if row.get('batch_size')!=4 or row.get('padding_side')!='left' or row.get('batch_generation_schema_version')!=1:
        raise ValueError('Committed row is not fixed B4 native generation')
    if not isinstance(row.get('prompt_sha256'),str) or len(row['prompt_sha256'])!=64:raise ValueError('Committed prompt identity missing')
    actual=score_prediction(expected,row['prediction'])
    if (not isinstance(row.get('score'),(int,float)) or isinstance(row['score'],bool) or
        not math.isfinite(row['score']) or abs(row['score']-actual)>1e-12):raise ValueError('Committed score differs from frozen strict scorer')


def read_committed_shards(output,records,protocol):
    directory=Path(output)/'prediction_batches';protocol_sha=protocol_digest(protocol);rows=[];committed={}
    if directory.is_symlink():raise ValueError('Batch journal directory may not be a symlink')
    if directory.exists():
        expected_names={f'batch-{i:04d}.jsonl' for i in range(128)}
        if any(p.name not in expected_names for p in directory.glob('*.jsonl')):raise ValueError('Unexpected batch journal shard')
    for index,start in enumerate(range(0,512,4)):
        path=directory/f'batch-{index:04d}.jsonl'
        if not path.exists():continue
        if path.is_symlink():raise ValueError('Committed batch shard may not be a symlink')
        chunk=records[start:start+4];values=read_rows(path);members=[r['record_id'] for r in chunk]
        if len(values)!=4 or [r['record_id'] for r in values]!=members:raise ValueError('Atomic batch shard is incomplete or reordered')
        checksum=digest([{k:v for k,v in row.items() if k!='batch_payload_sha256'} for row in values])
        if any(row.get('batch_payload_sha256')!=checksum for row in values):raise ValueError('Committed four-row payload checksum differs')
        for row,expected in zip(values,chunk):validate_prediction(row,expected,protocol_sha,index,members,protocol['generation']['eos_token_ids'])
        committed[index]={'path':str(path),'sha256':sha(path),'record_ids':members};rows.extend(values)
    return rows,committed


def commit_batch(output,index,values):
    path=Path(output)/'prediction_batches'/f'batch-{index:04d}.jsonl'
    if path.exists() or path.is_symlink():raise FileExistsError('Never overwrite a committed prediction batch')
    checksum=digest(values)
    data=''.join(json.dumps(dict(r,batch_payload_sha256=checksum),ensure_ascii=False,sort_keys=True)+'\n' for r in values).encode()
    atomic_bytes(path,data)


def export_view(output,records,protocol):
    rows,committed=read_committed_shards(output,records,protocol)
    atomic_bytes(Path(output)/'predictions.jsonl',''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows).encode())
    return rows,committed


def compare_native_cache(directory,records,protocol,rows):
    directory=Path(directory);parent=json.loads((directory/'protocol.json').read_text());summary=json.loads((directory/'summary.json').read_text())
    raw=read_rows(directory/'predictions.jsonl');by_id={r['record_id']:r for r in raw};parent_sha=hashlib.sha256(json.dumps(parent,sort_keys=True).encode()).hexdigest()
    if (parent.get('checkpoint_sha256')!=protocol['checkpoint_sha256'] or parent.get('manifest_sha256')!=protocol['manifest_sha256'] or
        parent.get('image_ablation',{}).get('variant')!='none' or len(raw)!=512 or len(by_id)!=512 or
        summary.get('complete_manifest') is not True or summary.get('protocol_sha256')!=parent_sha or
        parent.get('generation',{}).get('max_new_tokens')!=64 or parent.get('generation',{}).get('do_sample') is not False or
        parent.get('generation',{}).get('use_cache') is not True or parent.get('generation',{}).get('num_beams')!=1 or
        parent.get('generation',{}).get('batch_size',1)!=1 or parent.get('profile') is not None):
        raise ValueError('Existing native B1 cache does not match exact checkpoint/manifest/ordinary64-token protocol')
    if not all(name in parent.get('source_sha256',{}) for name in FROZEN_SOURCES):raise ValueError('Native B1 source provenance is incomplete')
    for name,digest_value in parent.get('source_sha256',{}).items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=digest_value:raise ValueError('Native B1 cache source provenance differs')
    current={r['record_id']:r for r in rows};mismatches=[]
    for expected in records:
        old=by_id.get(expected['record_id'])
        if old is None or old.get('protocol_sha256')!=parent_sha:raise ValueError('Native B1 prediction identity differs')
        if any(old.get(k)!=expected.get(k) for k in ROW_FIELDS):raise ValueError('Native B1 metadata/reference/image differs')
        if not math.isfinite(float(old['score'])) or abs(float(old['score'])-score_prediction(expected,old['prediction']))>1e-12:raise ValueError('Native B1 frozen score differs')
        if old['generated_token_ids']!=current[expected['record_id']]['generated_token_ids']:mismatches.append(expected['record_id'])
    computed=aggregate_scores(raw,require_four_groups=True)
    if any(summary.get(k)!=v for k,v in computed.items()):raise ValueError('Native B1 summary differs from verified raw scores')
    return {'status':'passed' if not mismatches else 'failed','compared_records':512,'exact_generated_token_ids':not mismatches,
            'mismatched_record_ids':mismatches,'native_raw_predictions_sha256':sha(directory/'predictions.jsonl'),
            'native_protocol_file_sha256':sha(directory/'protocol.json'),'original_cache_modified':False}


def parse_args(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output');p.add_argument('--checkpoint',choices=['best'],default='best')
    p.add_argument('--image-ablation',choices=MODES,default='none');p.add_argument('--max-new-tokens',type=int,default=64)
    p.add_argument('--ablation-seed',type=int,default=42);p.add_argument('--reserve-gb',type=float,default=120)
    p.add_argument('--cache-dir',default='cache/vision');p.add_argument('--resume',action='store_true')
    p.add_argument('--compare-native-cache');return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    if args.max_new_tokens!=64 or args.ablation_seed!=42:raise ValueError('Common batch4_v1 requires64 tokens and seed42')
    if args.compare_native_cache and args.image_ablation!='none':raise ValueError('Native full-cache comparison is ordinary-none only')
    run=Path(args.run).resolve();base=run/'evaluation/batch4_v1';output=base/args.image_ablation
    if any(path.is_symlink() for path in (run/'evaluation',base,output)):raise ValueError('Batch4 namespace directories may not be symlinks')
    if args.output and Path(args.output).resolve()!=output:raise ValueError('Batch4 output must be the separate run/evaluation/batch4_v1/<variant> namespace')
    manifest=Path(args.manifest).resolve();raw=manifest.read_bytes();manifest_sha=hashlib.sha256(raw).hexdigest();original=read_rows(manifest);validate_manifest(original)
    budget=DiskBudget(base,args.reserve_gb);budget.ensure_space(64*2**20);base.mkdir(parents=True,exist_ok=True)
    lock=(base/'.batch4.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        source=source_identity();environment=package_identity();checkpoint=(run/'best/state.pt').resolve(strict=True)
        identity=checkpoint_identity(checkpoint);model=None;output.mkdir(parents=True,exist_ok=True);last=[time.monotonic()]
    except BaseException:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close();raise
    def progress(phase,**values):
        budget.ensure_space(0)
        row={'event':phase,'time':time.time(),'profile':'batch4_v1','image_ablation':args.image_ablation,**values}
        _atomic_json(output/'progress.json',row)
        if time.monotonic()-last[0]>=30 or phase=='completed':
            print(json.dumps(row),flush=True);last[0]=time.monotonic()
    def load_model():
        nonlocal model
        if model is None:
            require_idle_gpu();progress('loading_model');model,provenance=load_research_checkpoint(checkpoint,cache_dir=args.cache_dir,reserve_gb=args.reserve_gb)
            _atomic_json(output/'model_provenance.json',provenance)
        return model
    try:
        rgb=rgb_census(original);transformed,descriptors=transformed_inputs(original,manifest_sha,base,int(args.reserve_gb*2**30))
        binding=guard_binding(identity,manifest_sha,original,descriptors,rgb,source,environment);guard_path=base/'parity_guard.json'
        if guard_path.is_file():
            existing_guard=json.loads(guard_path.read_text())
            if existing_guard.get('binding')!=binding:raise ValueError('Existing guard checkpoint/source/profile/manifest binding differs')
            if existing_guard.get('status')=='failed':raise ValueError('Failed parity guard blocks this namespace; no implicit fallback or overwrite')
        else:existing_guard={}
        if existing_guard.get('status')=='passed':guard=validate_guard(existing_guard,binding)
        else:guard=run_guard(load_model(),transformed,binding,guard_path,progress)
        records=transformed[args.image_ablation];protocol=make_protocol(identity,manifest_sha,records,descriptors[args.image_ablation],rgb,guard,source,environment)
        protocol_path=output/'protocol.json'
        if protocol_path.exists():
            if not args.resume:raise ValueError('Existing batch4 evaluation requires--resume')
            if json.loads(protocol_path.read_text())!=protocol:raise ValueError('Resume profile/checkpoint/guard/manifest/source/environment differs')
        else:
            if (output/'predictions.jsonl').exists() or (output/'prediction_batches').exists():raise ValueError('Orphan journal without its protocol is refused')
            _atomic_json(protocol_path,protocol)
        cached_rows,committed=read_committed_shards(output,records,protocol)
        if (output/'summary.json').exists():
            previous=json.loads((output/'summary.json').read_text());actual=aggregate_scores(cached_rows,require_four_groups=True)
            if (len(cached_rows)!=512 or previous.get('protocol_sha256')!=protocol_digest(protocol) or
                previous.get('journal_shards')!={str(k):v for k,v in committed.items()} or
                previous.get('raw_predictions_sha256')!=sha(output/'predictions.jsonl') or
                read_rows(output/'predictions.jsonl')!=cached_rows or
                any(previous.get(k)!=v for k,v in actual.items())):
                raise ValueError('Completed batch4 summary/raw-view/journal integrity differs')
            if args.compare_native_cache:
                comparison=compare_native_cache(args.compare_native_cache,records,protocol,cached_rows)
                _atomic_json(output/'native_b1_comparison.json',comparison)
                if comparison['status']!='passed':raise ValueError('Full512 ordinary B1/B4 token comparison failed')
            progress('completed',completed_records=512,group_macro_score=previous['group_macro_score'])
            return previous
        for index,start in enumerate(range(0,512,4)):
            if index in committed:continue
            budget.ensure_space(1024**2);active=load_model();chunk=records[start:start+4];members=[r['record_id'] for r in chunk]
            begin=time.monotonic();generated=generate_batch_predictions(active,[public_record(r) for r in chunk],max_new_tokens=64)
            if len(generated)!=4:raise ValueError('B4 generator did not return exactly four rows')
            elapsed=time.monotonic()-begin;values=[]
            for expected,result in zip(chunk,generated):
                row={key:expected[key] for key in ROW_FIELDS if key in expected}
                row.update(result,score=score_prediction(expected,result['prediction']),protocol_sha256=protocol_digest(protocol),
                    batch_index=index,batch_record_ids=members,generation_seconds=elapsed/4,batch_wall_seconds=elapsed)
                validate_prediction(row,expected,protocol_digest(protocol),index,members,protocol['generation']['eos_token_ids']);values.append(row)
            commit_batch(output,index,values);committed[index]=True
            progress('generated_batch',completed_records=len(committed)*4,manifest_records=512,batch_index=index)
        rows,shards=export_view(output,records,protocol)
        if len(rows)!=512:raise ValueError('Only complete512-record output can receive a summary')
        if args.compare_native_cache:
            comparison=compare_native_cache(args.compare_native_cache,records,protocol,rows);_atomic_json(output/'native_b1_comparison.json',comparison)
            if comparison['status']!='passed':raise ValueError('Full512 ordinary B1/B4 token comparison failed; no deployment claim')
        if (sha(manifest)!=manifest_sha or source_identity()!=source or
            not same_checkpoint_identity(identity,checkpoint_identity(checkpoint))):raise ValueError('Input/checkpoint/source changed during batch4 evaluation')
        metrics=aggregate_scores(rows,require_four_groups=True)
        metrics.update(complete_manifest=True,manifest_records=512,protocol_sha256=protocol_digest(protocol),
            checkpoint_sha256=identity['sha256'],checkpoint_artifact_identity=identity,manifest_sha256=manifest_sha,
            image_ablation=descriptors[args.image_ablation],profile=PROFILE,profile_sha256=digest(PROFILE),
            parity_guard_proof_sha256=digest(guard),raw_predictions_sha256=sha(output/'predictions.jsonl'),
            journal_shards=shards,all_checkpoint_or_recipe_parity_proven=False)
        _atomic_json(output/'summary.json',metrics);progress('completed',completed_records=512,group_macro_score=metrics['group_macro_score'])
        return metrics
    except BaseException as error:
        _atomic_json(output/'failure.json',{'status':'interrupted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'failed',
            'profile_sha256':digest(PROFILE),'checkpoint_identity':identity,'manifest_sha256':manifest_sha,
            'error_type':type(error).__name__,'error':str(error),'fixed_batch_size':4,'automatic_fallback':False})
        raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    def stopped(signum,_frame):raise InterruptedError('Batch4 evaluator interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped);main()
