#!/usr/bin/env python3
"""V7 scheduling plus source-bound native Giga external-evaluation stop token."""
import argparse,hashlib,os,signal,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
REQUESTED_CVD=os.environ.get('CUDA_VISIBLE_DEVICES')
if '--worker-request' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts import research_budget18_queue_v7 as v7
v4=v7.v4;old=v7.old;OWNER='research_budget18_training_first_v8'


def install():
    v7.install();v4.MODULES['giga31']='src.utils.research_giga_eval_stop_v1'
    v4.SCOPE['__file__']=str(Path(__file__).resolve());v4.OLD_SCOPE['__file__']=str(Path(__file__).resolve())


class Queue(v7.Queue):
    def __init__(self,*a,**k):
        super().__init__(*a,**k)
        from src.eval.research_giga_eot_v1 import definition
        receipt=dict(owner=OWNER,source_sha256=old.sha(__file__),giga_external_generation_stop=definition(),
            training_and_original_checkpoint_selection_unchanged=True)
        if receipt not in self.state['controller_revisions']:self.state['controller_revisions'].append(receipt)
        self.persist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,default=old.PLAN);p.add_argument('--state',type=Path,default=ROOT/'runs/research/budget18_queue_state_v1.json')
    p.add_argument('--resource-ready',type=Path);p.add_argument('--resource-ready-sha');p.add_argument('--run-queue',action='store_true')
    p.add_argument('--retry-failed',action='store_true');p.add_argument('--worker-request',type=Path);p.add_argument('--request-sha256')
    args=p.parse_args();install()
    if args.worker_request:
        if REQUESTED_CVD!='0':raise ValueError('Owned GPU worker requires GPU0')
        os.environ['CUDA_VISIBLE_DEVICES']='0';return v4.worker(args.worker_request,args)
    if not args.resource_ready or not args.resource_ready_sha:p.error('Explicit source-bound readiness required')
    binding=dict(path=str(args.resource_ready.resolve(strict=True)),sha256=args.resource_ready_sha)
    v4._RESOURCE=binding;v4.verify_plan(old.read(args.spec))
    if not args.run_queue:print('Validated V8 metadata: same18/42; Giga external native-turn stop');return 0
    queue=Queue(args.spec,args.state,args.retry_failed,binding)
    signal.signal(signal.SIGTERM,queue.signal);signal.signal(signal.SIGINT,queue.signal);return queue.run()


if __name__=='__main__':raise SystemExit(main())
