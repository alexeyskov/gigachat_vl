#!/usr/bin/env python3
"""Use remaining GPU memory for the first exact planned SmallThinker SFT."""
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from src.utils.research_stock_determinism import activate
from src.utils.research_additive_runner_v1 import main,sha
from src.utils.research_lossless_delta import memory_guard
from src.utils.research_storage import _atomic_json,DiskBudget

def run():
    import torch
    activate();memory_guard(12*2**30);DiskBudget(ROOT,32).ensure_space(2*2**30)
    plan_path=ROOT/'configs/research/additive_visual_plan_v1.json'
    plan=json.loads(plan_path.read_text());job=plan['jobs'][1]
    if (job['backend'],job['vision'],job['stage'],job['mode'],job['learned_masks'])!=('smallthinker4','siglip2','SFT','fixed',False):
        raise ValueError('Only the first exact planned SFT may use this window')
    a1=Path(job['output'].removesuffix('__SFT')+'__A1')
    done=json.loads((a1/'completed.json').read_text())
    if done['steps']!=4 or done['records']!=256:raise ValueError('Completed same-architecture A1 required')
    free,total=torch.cuda.mem_get_info()
    if free<18*2**30:raise MemoryError('Parallel first SFT requires18GiB actually free')
    torch.cuda.set_per_process_memory_fraction(16*2**30/total)
    out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    _atomic_json(out/'parallel_capacity_receipt.json',dict(plan_sha256=sha(plan_path),
        launcher_sha256=sha(__file__),process_memory_cap_GiB=16,actual_free_before_GiB=free/2**30,
        legacy_trainer_signals_sent=0,capacity_policy='native whole-group OOM rollback, same effective64 batch',time=time.time()))
    main(job['argv'])

if __name__=='__main__':run()
