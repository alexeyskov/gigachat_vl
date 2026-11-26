#!/usr/bin/env python3
"""Idempotent Supervisor entry for the first repaired LFM SFT."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
requested=os.environ.get('CUDA_VISIBLE_DEVICES','0');os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts import research_additive_queue_v1 as native

if __name__=='__main__':
    plan=json.loads((ROOT/'configs/research/additive_fixed_plan_v2.json').read_text())
    job=next(j for j in plan['jobs'] if j['id']=='lfm25__siglip2__fixed__R1__SFT')
    if native.complete(job):
        print(json.dumps(dict(status='completed_repaired_SFT_preserved',run=job['id'])),flush=True)
    else:
        os.environ['CUDA_VISIBLE_DEVICES']=requested
        os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'scripts/run_research_additive_init_dtype_v1.py'),*job['argv']])
