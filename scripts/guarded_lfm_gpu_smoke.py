#!/usr/bin/env python3
"""Run one tiny LFM CUDA alignment update only alongside quantized SmallThinker.

The existing SmallThinker process is never stopped or changed. This wrapper only
terminates its own smoke child if the allowed co-residency condition changes.
"""
from pathlib import Path
import json
import math
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.research_storage import _atomic_json

OUTPUT=ROOT/'runs/research/lfm25__siglip2__A0_bf16__smoke_s42'
REPORT=ROOT/'research_results/lfm_gpu_checkpoint_validation.json'


def emit(event,**fields):print(json.dumps({'time':time.time(),'event':event,**fields}),flush=True)


def flag_value(argv,flag,default=None):
    return argv[argv.index(flag)+1] if flag in argv else default


def gpu_state():
    row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    total,used,free=[float(x.strip()) for x in row.split(',')]
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
    processes=[]
    for row in rows:
        if not row.strip():continue
        pid,memory=[x.strip() for x in row.split(',')]
        try:argv=Path('/proc',pid,'cmdline').read_bytes().decode().split('\0')
        except FileNotFoundError:continue
        processes.append({'pid':int(pid),'memory_mib':float(memory),'argv':argv})
    return {'total_mib':total,'used_mib':used,'free_mib':free,'processes':processes}


def residency_allowed(state,own_pid=None):
    others=[p for p in state['processes'] if p['pid']!=own_pid]
    if len(others)!=1:return False,'requires exactly one existing quantized SmallThinker CUDA process'
    argv=others[0]['argv']
    if not any('run_research_stage.py' in x for x in argv):return False,'existing CUDA process is not the authorized SmallThinker trainer'
    if flag_value(argv,'--model','Tiiny/SmallThinker-4BA0.6B-Instruct')!='Tiiny/SmallThinker-4BA0.6B-Instruct':return False,'existing CUDA process is not SmallThinker'
    if flag_value(argv,'--stage')!='A0' or flag_value(argv,'--recipe')!='4q':return False,'existing SmallThinker stage is not quantized A0'
    if own_pid is None:
        estimate=24*1024
        if state['used_mib']+estimate>85*1024 or state['free_mib']<estimate+2048:
            return False,'combined conservative estimate would exceed 85GiB or leave insufficient margin'
    elif state['used_mib']>85*1024 or state['free_mib']<2048:
        return False,'live aggregate memory exceeded guard limit'
    return True,'allowed'


def main():
    started=time.monotonic();state=gpu_state();allowed,reason=residency_allowed(state)
    report={'schema_version':1,'status':'running' if allowed else 'deferred','preflight':state,'reason':reason,
            'scope':'one real CUDA frozen-LFM alignment update; not full VLM training or task benchmark',
            'estimated_additional_lfm_gib':24,'aggregate_limit_gib':85,'existing_smallthinker_process_modified':False}
    _atomic_json(ROOT/'research_results/lfm_gpu_smoke_preflight.json',report)
    if not allowed:
        _atomic_json(REPORT,report);emit('lfm_gpu_smoke_deferred',reason=reason);return
    if (OUTPUT/'completed.json').exists():raise FileExistsError('Completed LFM smoke already exists; refusing silent duplicate')
    argv=[sys.executable,'-u',str(ROOT/'scripts/run_research_stage.py'),'--model','LiquidAI/LFM2.5-8B-A1B',
          '--stage','A0','--recipe','4f','--manifest','datasets/research/smoke-v1/alignment_train.jsonl',
          '--validation','datasets/research/smoke-v1/alignment_validation.jsonl','--output',str(OUTPUT),
          '--max-records','4','--max-steps','1','--batch-size','2','--effective-batch-size','4',
          '--eval-records','2','--eval-every','1']
    emit('lfm_gpu_smoke_start',argv=argv,preflight_used_gib=state['used_mib']/1024)
    child=subprocess.Popen(argv,cwd=ROOT)
    report['smoke_child_pid']=child.pid
    try:
        while child.poll() is None:
            time.sleep(2)
            current=gpu_state();okay,why=residency_allowed(current,child.pid)
            if not okay:
                report['guard_abort_state']=current
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise RuntimeError('Own LFM smoke terminated to preserve SmallThinker: '+why)
        if child.returncode!=0:raise RuntimeError(f'LFM smoke child failed with exit code {child.returncode}')
        completed=json.loads((OUTPUT/'completed.json').read_text())
        metrics=[json.loads(line) for line in (OUTPUT/'metrics.jsonl').read_text().splitlines()]
        train=next(row for row in reversed(metrics) if row['event']=='train')
        val=next(row for row in reversed(metrics) if row['event']=='validation')
        import torch
        payload=torch.load(OUTPUT/'last/state.pt',map_location='cpu',weights_only=False,mmap=True)
        if not payload['model'] or any(not name.startswith('projector.') for name in payload['model']):raise ValueError('Frozen LFM smoke unexpectedly saved original language weights')
        if not all(torch.isfinite(tensor).all() for tensor in payload['model'].values()):raise ValueError('Projector checkpoint contains non-finite weights')
        moments=[state['exp_avg'] for state in payload['optimizer']['state'].values() if 'exp_avg' in state]
        if not moments or not all(torch.isfinite(moment).all() for moment in moments) or not any(moment.abs().max()>0 for moment in moments):
            raise ValueError('No finite nonzero projector gradient evidence in first-step optimizer moments')
        reference=ROOT/'runs/research/smallthinker4__siglip2__A0_bf16__pilot_s42/last/state.pt'
        small=torch.load(reference,map_location='cpu',weights_only=False,mmap=True)
        same_cache=payload['cache_fingerprint']==small['cache_fingerprint']
        if not same_cache:raise ValueError('LFM/SmallThinker vision cache fingerprints unexpectedly differ')
        if train['cache']['misses']!=0 or train['cache']['encoded_images']!=0 or train['cache']['hits']!=4:
            raise ValueError('Actual cross-backbone four-image cache reuse did not meet expected4hits/0misses')
        if completed['steps']!=1 or completed['records']!=4 or not math.isfinite(train['loss']) or not math.isfinite(val['loss']):raise ValueError('Tiny LFM smoke did not finish exactly one finite four-record update')
        report.update(status='passed',completed=completed,train_loss=train['loss'],validation_loss=val['loss'],
            lfm_peak_allocated_gib=train['peak_vram_gb'],train_cache=train['cache'],cross_backbone_cache_fingerprint_matches=True,
            cache_fingerprint=payload['cache_fingerprint'],saved_tensor_keys=sorted(payload['model']),
            projector_optimizer_nonzero_moments=True,elapsed_seconds=time.monotonic()-started,
            limits='One tiny frozen-backbone CUDA alignment update only; no full LLM training/QLoRA/VLM quality claim')
        _atomic_json(REPORT,report);emit('lfm_gpu_smoke_passed',report=str(REPORT),**{k:report[k] for k in ['train_loss','validation_loss','lfm_peak_allocated_gib','train_cache']})
    except BaseException as error:
        if child.poll() is None:child.terminate();child.wait(timeout=20)
        report.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        _atomic_json(REPORT,report);raise

if __name__=='__main__':main()
