#!/usr/bin/env python3
"""Fresh CPU-native roundtrip validation of the seven explicitly named diagnostics.

This script never prunes. Deletion is a separate, explicitly authorized action
using prune_technical_smoke.py after every retained best/last object validates.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.research_storage import _atomic_json

RUNS=[
 'smallthinker4__siglip2__4f__SFT_smoke_s42',
 'smallthinker4__siglip2__2a__SFT_smoke_s42',
 'smallthinker4__siglip2__2b__SFT_smoke_s42',
 'smallthinker4__siglip2__2c__SFT_smoke_s42',
 'smallthinker4__siglip2__3a__SFT_smoke_s42',
 'smallthinker4__siglip2__3b__SFT_smoke_s42',
 'smallthinker4__siglip2__3c__SFT_smoke_s42',
]


def emit(event,**fields):print(json.dumps({'time':time.time(),'event':event,**fields}),flush=True)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ['', '-1']:raise RuntimeError('Validation batch must hide GPUs')
    target=ROOT/'research_results/technical_smoke_validation_batch.json'
    report={'schema_version':1,'status':'running','explicit_run_basenames':RUNS,'runs':[],
            'pruning_performed':False,'gpu_used':False,'cpu_threads_per_validator':8}
    started=time.monotonic()
    try:
        for basename in RUNS:
            run=ROOT/'runs/research'/basename
            emit('technical_roundtrip_start',run=str(run))
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/validate_research_checkpoint.py'),
                            '--run',str(run),'--device','cpu','--native-full-only'],cwd=ROOT,check=True)
            receipt=json.loads((run/'checkpoint_roundtrip.json').read_text())
            if receipt.get('status')!='passed' or not receipt.get('checkpoints'):raise ValueError('Fresh full-native checkpoint validation did not pass')
            for checkpoint in receipt['checkpoints']:
                if checkpoint.get('validation_mode')!='cpu_native_full_reconstruction' or checkpoint.get('native_logits_roundtrip_max_abs_diff',1)>1e-5:
                    raise ValueError('Checkpoint receipt lacks actual finite native reconstruction proof')
            report['runs'].append({'run':str(run),'status':'passed','retained_checkpoint_objects':len(receipt['checkpoints']),
                'validated_tensor_count':receipt['validated_tensor_count'],
                'retained_payload_bytes':sum(row['checkpoint_size_bytes'] for row in receipt['checkpoints'])})
            _atomic_json(target,report);emit('technical_roundtrip_passed',**report['runs'][-1])
        report.update(status='passed',ready_for_pruning=True,elapsed_seconds=time.monotonic()-started,
                      retained_payload_bytes=sum(row['retained_payload_bytes'] for row in report['runs']))
        _atomic_json(target,report);emit('all_explicit_technical_roundtrips_passed',report=str(target),elapsed_seconds=report['elapsed_seconds'])
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        _atomic_json(target,report);raise

if __name__=='__main__':main()
