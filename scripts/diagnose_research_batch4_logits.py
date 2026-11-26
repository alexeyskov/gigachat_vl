#!/usr/bin/env python3
"""Isolated first-step native B1/B4 logit diagnosis; never changes a parity gate.

GPU execution requires --allow-gpu. Exclusive operation is the default. Optional
concurrency accepts only one explicitly pinned live native-evaluation PID and
checks its birth/argv/progress and VRAM before every generation call.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.eval.research_inference import build_generation_inputs,load_research_checkpoint,_autocast
from src.eval.research_batch_inference import build_batch_generation_inputs,eos_token_ids,trim_generated_ids
from src.model.research_moe import set_modality_mask
from src.utils.research_checkpoint_compat import checkpoint_identity,load_research_payload,same_checkpoint_identity
from src.utils.research_storage import DiskBudget,_atomic_json

FAILED_RECORD='8417a34586bc9833bea97d350f498f52ae412101dbefc185ff485227d8a14fbf'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def read_rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def public(row):return {k:row[k] for k in ('record_id','question','image_paths','image_hashes') if k in row}


def failed_batch(guard,records,record_id=FAILED_RECORD):
    if guard.get('status')!='failed' or guard.get('modes',{}).get('none',{}).get('status')!='failed':raise ValueError('A preserved failed ordinary-none B4 guard is required')
    proof=guard['modes']['none'];binding=guard['binding']
    if record_id not in proof.get('mismatched_record_ids',[]):raise ValueError('Requested record is not a failed guard mismatch')
    ids=[r['record_id'] for r in records]
    if len(records)!=512 or len(set(ids))!=512:raise ValueError('The original unique512-record manifest is required')
    index=ids.index(record_id);start=index-index%4;chunk=records[start:start+4]
    if start not in binding['original_batch_start_indices']:raise ValueError('Mismatch does not lie in a selected production batch')
    selected=binding['record_ids'];offset=selected.index(chunk[0]['record_id'])
    if selected[offset:offset+4]!=[r['record_id'] for r in chunk]:raise ValueError('Guard and original fixed four-row membership differ')
    native=proof['native_b1_token_ids'][offset:offset+4];batch=proof['batch4_token_ids'][offset:offset+4]
    if len(native)!=4 or len(batch)!=4:raise ValueError('Failed guard token evidence is incomplete')
    return chunk,{'original_batch_start':start,'failed_row_offset':index-start,'guard_row_offsets':list(range(offset,offset+4)),
                  'expected_native_token_ids':native,'expected_batch4_token_ids':batch}


def process_info(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'start_time':int(fields[19]),'state':fields[0],
                'argv':[os.fsdecode(x) for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x],
                'cwd':str(Path(f'/proc/{pid}/cwd').resolve())}
    except (OSError,IndexError,ValueError):return None


def gpu_snapshot():
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    lines=[x.strip() for x in gpu.stdout.splitlines() if x.strip()]
    if len(lines)!=1:raise RuntimeError('Exactly one physical GPU is required for this diagnostic')
    free,total=[int(x.strip())*2**20 for x in lines[0].split(',')]
    rows=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    processes={}
    for line in rows.stdout.splitlines():
        if not line.strip():continue
        pid,memory=[x.strip() for x in line.split(',')]
        if not pid.isdigit() or not memory.isdigit():raise RuntimeError('Unknown GPU process memory; refusing concurrency')
        processes[int(pid)]=int(memory)*2**20
    return {'free_bytes':free,'total_bytes':total,'processes':processes}


def argument(argv,flag):
    hits=[argv[i+1] for i,x in enumerate(argv[:-1]) if x==flag]
    if len(hits)!=1:raise ValueError('Pinned native evaluator has missing/duplicate '+flag)
    return hits[0]


class RuntimeGuard:
    def __init__(self,allow_known_eval=None,*,memory_cap_gib=12,deadline_seconds=120,proc=process_info,gpu=gpu_snapshot):
        if not 0<memory_cap_gib<=12 or not 0<deadline_seconds<=120:raise ValueError('Memory cap must be<=12GiB and deadline<=120s')
        self.started=time.monotonic();self.deadline=deadline_seconds;self.cap=int(memory_cap_gib*2**30)
        self.proc=proc;self.gpu=gpu;self.path=Path(allow_known_eval).resolve() if allow_known_eval else None
        self.peer=json.loads(self.path.read_text()) if self.path else None
        self.binding_sha=sha(self.path) if self.path else None;self.checks=[]
        if self.peer:
            if (not isinstance(self.peer.get('pid'),int) or isinstance(self.peer['pid'],bool) or not isinstance(self.peer.get('start_time'),int)
                or not isinstance(self.peer.get('argv'),list) or not all(isinstance(x,str) for x in self.peer['argv'])):
                raise ValueError('Known evaluator binding needs exact pid, kernel start_time and full argv list')
    def check(self):
        if time.monotonic()-self.started>=self.deadline:raise InterruptedError('Own diagnostic exceeded120s deadline')
        gpu=self.gpu();allowed={os.getpid()};completed=None
        if self.peer:
            if sha(self.path)!=self.binding_sha:raise InterruptedError('Root-selected evaluator binding file changed')
            actual=self.proc(self.peer['pid'])
            if actual is None or actual['state'] in ('Z','T','t') or actual['start_time']!=self.peer['start_time'] or actual['argv']!=self.peer['argv']:
                raise InterruptedError('Pinned native evaluation is no longer the same live process')
            cwd=Path(actual['cwd']);argv=actual['argv']
            scripts=[(cwd/x).resolve() for x in argv if x.endswith('eval_research_checkpoint.py')]
            if scripts!=[(ROOT/'scripts/eval_research_checkpoint.py').resolve()]:raise ValueError('Only the frozen native evaluation CLI may be allowed')
            output=Path(argument(argv,'--output'));output=(cwd/output).resolve() if not output.is_absolute() else output.resolve()
            progress_path=output/'progress.json'
            if self.peer.get('progress_path') and Path(self.peer['progress_path']).resolve()!=progress_path:raise ValueError('Pinned progress path differs from exact peer output')
            progress=json.loads(progress_path.read_text());completed=progress.get('completed_records')
            if (progress.get('event')!='generated' or progress.get('selected_records')!=512 or progress.get('manifest_records')!=512 or
                not isinstance(completed,int) or isinstance(completed,bool) or not 0<=completed<256):
                raise InterruptedError('Known evaluation is not safely below256/512; aborting only own work')
            allowed.add(self.peer['pid'])
            if self.peer['pid'] not in gpu['processes']:raise InterruptedError('Pinned evaluator is no longer the observed GPU task')
        if set(gpu['processes'])-allowed:raise InterruptedError('Unexpected GPU task; existing processes were left untouched')
        if gpu['free_bytes']<24*2**30:raise InterruptedError('Global free VRAM is below24GiB')
        if gpu['processes'].get(os.getpid(),0)>self.cap:raise InterruptedError('Own NVML memory exceeds declared<=12GiB cap')
        if torch.cuda.is_initialized() and max(torch.cuda.memory_reserved(),torch.cuda.max_memory_allocated())>self.cap:
            raise InterruptedError('Own tensor memory exceeds declared cap')
        self.checks.append({'time':time.time(),'peer_completed_records':completed,'free_bytes':gpu['free_bytes'],
                            'own_nvml_bytes':gpu['processes'].get(os.getpid(),0)})
        return gpu
    def arm_allocator_cap(self):
        snapshot=self.check()
        # Keep additional allowance for CUDA context/bitsandbytes allocations
        # outside PyTorch's allocator; NVML is rechecked at every boundary.
        torch.cuda.set_per_process_memory_fraction((self.cap*2/3)/snapshot['total_bytes'])


@torch.no_grad()
def capture_generation(model,records,*,batched,max_new_tokens=64):
    if batched:
        prepared=build_batch_generation_inputs(model,[public(r) for r in records],max_new_tokens=max_new_tokens)
        kwargs=dict(inputs_embeds=prepared['inputs_embeds'],attention_mask=prepared['attention_mask'],position_ids=prepared['position_ids'])
        metadata=prepared['rows'];width=prepared['padded_prefill_tokens']
    else:
        if len(records)!=1:raise ValueError('Native B1 capture requires exactly one row')
        one=build_generation_inputs(model,public(records[0]),max_new_tokens=max_new_tokens);prepared=one;metadata=[one];width=one['prefill_tokens']
        kwargs=dict(inputs_embeds=one['inputs_embeds'],attention_mask=one['attention_mask'])
    set_modality_mask(model.base,prepared['modality_mask'],prepared['attention_mask'])
    kwargs.update(do_sample=False,num_beams=1,max_new_tokens=max_new_tokens,use_cache=True,return_dict_in_generate=True,
                  output_scores=True,output_logits=True,pad_token_id=model.tokenizer.pad_token_id)
    if 'logits_to_keep' in inspect.signature(model.base.forward).parameters:kwargs['logits_to_keep']=1
    with _autocast(kwargs['inputs_embeds'].device):generated=model.base.generate(**kwargs)
    steps=len(generated.scores)
    if not steps or not getattr(generated,'logits',None):raise ValueError('Native generation did not expose first-step raw logits/scores')
    raw=generated.logits[0].detach().cpu().clone();scores=generated.scores[0].detach().cpu().clone()
    if raw.shape!=scores.shape or raw.ndim!=2 or raw.shape[0]!=len(records):raise ValueError('Unexpected first-step capture shape')
    ids=generated.sequences[:,-steps:].detach().cpu().tolist();eos=eos_token_ids(model)
    if batched:ids=[trim_generated_ids(x,eos) for x in ids]
    result={'token_ids':ids,'predictions':[model.tokenizer.decode(x,skip_special_tokens=True).strip() for x in ids],
            'raw_logits':raw,'output_scores':scores,'raw_logits_dtype':str(raw.dtype),'output_scores_dtype':str(scores.dtype),'rows':[]}
    for row,info in zip(records,metadata):
        offset=width-info['prefill_tokens'] if batched else 0
        result['rows'].append({'record_id':row['record_id'],'prompt_sha256':info['prompt_sha256'],
            'prefill_tokens':info['prefill_tokens'],'visual_tokens':info['visual_tokens'],'left_padding_offset':offset,
            'first_real_position_id':int(prepared['position_ids'][len(result['rows']),offset]) if batched else 0})
    return result


def compare_logits(native,batch,tokenizer):
    if native.shape!=batch.shape or native.ndim!=2:raise ValueError('Aligned first-step matrices must share[B,V] shape')
    result=[]
    for a,b in zip(native.double(),batch.double()):
        finite=torch.isfinite(a)&torch.isfinite(b)
        if not finite.any() or torch.isnan(a).any() or torch.isnan(b).any():raise ValueError('Non-finite diagnostic logits')
        difference=(a[finite]-b[finite]).abs();k=min(10,a.numel())
        top_a,idx_a=torch.topk(a,k);top_b,idx_b=torch.topk(b,k)
        def top(values,indices):
            return [{'token_id':int(i),'token':tokenizer.decode([int(i)],skip_special_tokens=False),
                     'logit':float(v) if math.isfinite(float(v)) else str(float(v))} for v,i in zip(values,indices)]
        qa=torch.quantile(difference,torch.tensor([.5,.9,.99,.999,1.],dtype=torch.float64)).tolist()
        result.append({'max_abs_diff':float(difference.max()),'rms_diff':float(torch.sqrt(torch.mean(difference.square()))),
            'abs_diff_quantiles':dict(zip(['p50','p90','p99','p99_9','max'],qa)),
            'finite_pair_count':int(finite.sum()),'nonfinite_pattern_changes':int((torch.isfinite(a)!=torch.isfinite(b)).sum()),
            'native_top10':top(top_a,idx_a),'batch4_top10':top(top_b,idx_b),
            'native_top1_top2_margin':float(top_a[0]-top_a[1]),'batch4_top1_top2_margin':float(top_b[0]-top_b[1]),
            'native_first_argmax':int(a.argmax()),'batch4_first_argmax':int(b.argmax()),'first_argmax_changed':int(a.argmax())!=int(b.argmax())})
    return result


def require_reproduction(native,batch,evidence):
    if native!=evidence['expected_native_token_ids'] or batch!=evidence['expected_batch4_token_ids']:
        raise ValueError('Captured full token sequences did not reproduce preserved guard; no logit interpretation permitted')


def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--guard',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--record-id',default=FAILED_RECORD);p.add_argument('--allow-gpu',action='store_true')
    p.add_argument('--allow-known-eval',help='Root-created JSON with pid,start_time,argv and optional exact progress_path')
    p.add_argument('--memory-cap-gib',type=float,default=12);p.add_argument('--deadline-seconds',type=float,default=120)
    p.add_argument('--reserve-gib',type=float,default=120);p.add_argument('--cache-dir',default='cache/vision');args=p.parse_args(argv)
    if not args.allow_gpu:raise RuntimeError('Diagnostic disabled unless root explicitly supplies--allow-gpu')
    output=Path(args.output).resolve();archive=output.with_suffix('.first_step_logits.pt')
    if output.exists() or archive.exists():raise FileExistsError('Use new diagnostic artifacts; failed gates and old reports are preserved')
    guard_path=Path(args.guard).resolve();guard=json.loads(guard_path.read_text());manifest=Path(args.manifest).resolve()
    if sha(manifest)!=guard['binding']['manifest_sha256']:raise ValueError('Original manifest differs from failed guard')
    records,evidence=failed_batch(guard,read_rows(manifest),args.record_id)
    sources=guard['binding']['source_sha256']
    for name,expected in sources.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=expected:raise ValueError('Frozen guard source differs')
    checkpoint=Path(guard['binding']['checkpoint_identity']['primary_path']);identity=checkpoint_identity(checkpoint)
    if not same_checkpoint_identity(identity,guard['binding']['checkpoint_identity']):raise ValueError('Checkpoint differs from failed guard')
    payload=load_research_payload(checkpoint,map_location='cpu',mmap=True)
    if payload.get('config',{}).get('recipe')!='4q' or payload['config'].get('stage')!='SFT' or payload.get('training_completed') is not True:
        raise ValueError('Diagnostic is restricted to the completed compacted4q checkpoint')
    del payload
    safety=RuntimeGuard(args.allow_known_eval,memory_cap_gib=args.memory_cap_gib,deadline_seconds=args.deadline_seconds)
    safety.arm_allocator_cap();DiskBudget(output.parent,args.reserve_gib).ensure_space(64*2**20)
    from scripts.eval_research_checkpoint_batch4 import package_identity
    environment=package_identity()
    if environment!=guard['binding']['environment']:raise ValueError('Precision/package environment differs from failed guard')
    report={'schema_version':1,'status':'running','scope':'numerical diagnosis only; no relaxed gate, deployment, score or speed claim',
      'checkpoint_identity':identity,'guard_path':str(guard_path),'guard_sha256':sha(guard_path),'manifest_sha256':sha(manifest),
      'source_sha256':dict(sources,**{'scripts/diagnose_research_batch4_logits.py':sha(__file__)}),
      'environment':environment,'record_ids':[r['record_id'] for r in records],'batch_evidence':evidence,
      'capture':'native generate output_scores=True + output_logits=True; first step only archived',
      'shared_known_eval_binding':safety.peer,'shared_known_eval_binding_sha256':safety.binding_sha,
      'runtime_limits':{'minimum_global_free_vram_bytes':24*2**30,'own_memory_cap_bytes':safety.cap,
                        'pytorch_allocator_cap_bytes':int(safety.cap*2/3),'gpu_work_deadline_seconds':args.deadline_seconds,
                        'peer_completed_records_must_be_below':256,
                        'cap_enforcement':'lower PyTorch allocator cap plus NVML/tensor checks before and after every generate call'},
      'timing_or_speed_claims':False,'other_processes_signalled':False,'frozen_inputs_modified':False}
    _atomic_json(output,report)
    try:
        safety.check();model,provenance=load_research_checkpoint(checkpoint,cache_dir=args.cache_dir,reserve_gb=args.reserve_gib)
        report['model_provenance']=provenance;native=[]
        for row in records:
            safety.check();native.append(capture_generation(model,[row],batched=False));safety.check()
        safety.check();batch=capture_generation(model,records,batched=True);safety.check()
        native_ids=[x['token_ids'][0] for x in native]
        report['captured_native_token_ids']=native_ids;report['captured_batch4_token_ids']=batch['token_ids']
        report['full_guard_token_sequences_reproduced']=False
        report['native_predictions']=[x['predictions'][0] for x in native];report['batch4_predictions']=batch['predictions']
        require_reproduction(native_ids,batch['token_ids'],evidence)
        first_raw=torch.cat([x['raw_logits'] for x in native]);first_scores=torch.cat([x['output_scores'] for x in native])
        report['full_guard_token_sequences_reproduced']=True
        report['captured_dtypes']={'native_raw':[x['raw_logits_dtype'] for x in native],'batch4_raw':batch['raw_logits_dtype'],
                                   'native_scores':[x['output_scores_dtype'] for x in native],'batch4_scores':batch['output_scores_dtype']}
        report['native_prompt_rows']=[x['rows'][0] for x in native];report['batch4_prompt_rows']=batch['rows']
        report['raw_logits_comparison']=compare_logits(first_raw,batch['raw_logits'],model.tokenizer)
        report['processed_output_scores_comparison']=compare_logits(first_scores,batch['output_scores'],model.tokenizer)
        for field in ('raw_logits_comparison','processed_output_scores_comparison'):
            for row,record in zip(report[field],records):row['record_id']=record['record_id']
        report['changed_processed_first_argmax_record_ids']=[r['record_id'] for r in report['processed_output_scores_comparison'] if r['first_argmax_changed']]
        archive.parent.mkdir(parents=True,exist_ok=True);fd,temp=tempfile.mkstemp(prefix='.logits-',dir=archive.parent)
        try:
            with os.fdopen(fd,'wb') as handle:
                torch.save({'record_ids':report['record_ids'],'native_raw_logits':first_raw,'batch4_raw_logits':batch['raw_logits'],
                            'native_output_scores':first_scores,'batch4_output_scores':batch['output_scores'],
                            'checkpoint_sha256':identity['sha256'],'guard_sha256':report['guard_sha256']},handle)
                handle.flush();os.fsync(handle.fileno())
            os.replace(temp,archive)
        finally:
            if os.path.exists(temp):os.unlink(temp)
        if (sha(guard_path)!=report['guard_sha256'] or sha(manifest)!=report['manifest_sha256'] or
            any(sha(ROOT/name)!=expected for name,expected in report['source_sha256'].items()) or
            not same_checkpoint_identity(identity,checkpoint_identity(checkpoint))):
            raise ValueError('Frozen diagnostic inputs changed')
        report.update(status='passed_diagnostic_reproduction',raw_first_step_archive=str(archive),raw_first_step_archive_sha256=sha(archive),
                      safety_checks=safety.checks,peak_own_allocated_bytes=torch.cuda.max_memory_allocated(),peak_own_reserved_bytes=torch.cuda.max_memory_reserved())
        _atomic_json(output,report);return report
    except BaseException as error:
        report.update(status='aborted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'failed',error_type=type(error).__name__,error=str(error),safety_checks=safety.checks)
        _atomic_json(output,report);raise

if __name__=='__main__':
    def interrupted(signum,_frame):raise InterruptedError('Own diagnostic interrupted by signal '+str(signum))
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted);main()
