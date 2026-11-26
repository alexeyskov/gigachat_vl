#!/usr/bin/env python3
"""Training-first V5 with explicit corrected SmallThinker/Qwen router numerics.

V4 admission, lifecycle, parent-death handshake and full replay remain native.
The resource receipt binds both frozen controllers and the isolated correction.
"""
from __future__ import annotations
import argparse
import hashlib
import os
import signal
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REQUESTED_CVD=os.environ.get('CUDA_VISIBLE_DEVICES')
if '--worker-request' not in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES']=''
V5='scripts/research_budget18_queue_v5.py'
V5_SHA='9209068c051465bdaa38a4da023de5853f3d7088bd6874e5973917684379383d'
if hashlib.sha256((ROOT/V5).read_bytes()).hexdigest()!=V5_SHA:
    raise ValueError('Frozen training-first V5 changed')
from scripts import research_budget18_queue_v5 as v5
v4=v5.v4
old=v5.old
OWNER='research_budget18_training_first_v6'
BACKEND='src.utils.research_small_router_stability_v1'
_native_completed=v4.validate_completed


def corrected_completed(job):
    if job['backend']=='smallthinker4' and job['encoder']=='qwen25':
        from src.utils.research_small_router_stability_v1 import validate_execution
        validate_execution(old.read(old.directory(job)/'resolved_config.json'))
    return _native_completed(job)


def install():
    # Function globals are private controller copies, not the canonical V1.
    # __file__ binds request verification and worker entry to this V6 source.
    v4.MODULES['smallthinker4']=BACKEND
    v4.SCOPE.update(__file__=str(Path(__file__).resolve()),validate_completed=corrected_completed)
    v4.OLD_SCOPE['__file__']=str(Path(__file__).resolve())
    v4.validate_completed=corrected_completed


class Queue(v5.Queue):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from src.utils.research_small_router_stability_v1 import definition
        receipt={'owner':OWNER,'source_sha256':old.sha(__file__),
                 'training_first_controller_source_sha256':V5_SHA,
                 'worker_lifecycle_source_sha256':v5.V4_SHA,
                 'corrected_numerical_execution':definition(),
                 'scientific_cells':18,'training_stages':42,
                 'existing_SigLIP_runs_unchanged':True}
        if receipt not in self.state['controller_revisions']:
            self.state['controller_revisions'].append(receipt)
        self.persist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,default=old.PLAN)
    p.add_argument('--state',type=Path,default=ROOT/'runs/research/budget18_queue_state_v1.json')
    p.add_argument('--resource-ready',type=Path)
    p.add_argument('--resource-ready-sha')
    p.add_argument('--run-queue',action='store_true')
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--worker-request',type=Path)
    p.add_argument('--request-sha256')
    args=p.parse_args()
    install()
    if args.worker_request:
        if REQUESTED_CVD!='0':
            raise ValueError('Owned GPU worker must be launched with GPU0 only')
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        return v4.worker(args.worker_request,args)
    if args.resource_ready is None or not args.resource_ready_sha:
        p.error('Explicit source-bound resource readiness required')
    binding={'path':str(args.resource_ready.resolve(strict=True)),'sha256':args.resource_ready_sha}
    v4._RESOURCE=binding
    v4.verify_plan(old.read(args.spec))
    if not args.run_queue:
        print('Validated18-cell/42-stage training-first V6 metadata; no GPU launched')
        return 0
    queue=Queue(args.spec,args.state,args.retry_failed,binding)
    signal.signal(signal.SIGTERM,queue.signal)
    signal.signal(signal.SIGINT,queue.signal)
    return queue.run()


if __name__=='__main__':raise SystemExit(main())
