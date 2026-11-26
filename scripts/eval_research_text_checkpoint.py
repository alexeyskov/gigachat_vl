#!/usr/bin/env python3
"""Opt-in nativeB1 primary text producer; metadata preflight by default."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal

ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
spec=importlib.util.spec_from_file_location('research_text_generation_cli',ROOT/'src/eval/research_text_generation.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-public-only',action='store_true')
    p.add_argument('--parent-bundle',type=Path,default=ROOT/'datasets/research/primary-text-en-v1')
    p.add_argument('--public-bundle',type=Path,default=ROOT/'datasets/research/primary-text-public-en-v1')
    p.add_argument('--model-id',choices=list(owner.metrics.TOKEN_IDS),default='smallthinker4')
    choice=p.add_mutually_exclusive_group()
    choice.add_argument('--baseline-precision',choices=('bf16','nf4',owner.FULL_INIT_CONTROL))
    choice.add_argument('--run',type=Path)
    p.add_argument('--checkpoint',choices=('best','last'),default='best')
    p.add_argument('--tasks',nargs='+',choices=owner.TASKS,default=list(owner.TASKS))
    p.add_argument('--output',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-gpu',action='store_true');p.add_argument('--cache-dir',default='cache/vision')
    p.add_argument('--reserve-gb',type=float,default=120.)
    args=p.parse_args(argv)
    if args.prepare_public_only:
        if args.allow_gpu or args.run or args.baseline_precision:p.error('Public derivative preparation is a separate CPU-only command')
        result=owner.prepare_public_bundle(args.parent_bundle,args.public_bundle,reserve_gb=args.reserve_gb)
        print(json.dumps({'status':'public_inputs_prepared_no_model','output':str(args.public_bundle),
            'files_sha256':result['files_sha256'],'model_loaded':False,'GPU_used':False}));return 0
    if not args.run and not args.baseline_precision:p.error('Select --run or explicit --baseline-precision bf16/nf4')
    context=owner.public_context(args.public_bundle,args.model_id,tasks=args.tasks)
    plan=owner.learned_plan(args.run,args.checkpoint,context) if args.run else owner.baseline_plan(context,args.baseline_precision)
    if not args.allow_gpu:
        print(json.dumps({'status':'blocked_backend_admission' if plan.get('admission_blockers') else
            'blocked_missing_local_weights' if plan.get('missing_weights') else 'metadata_preflight_only_no_generation',
            'model':args.model_id,'stage':plan.get('stage','baseline_before_vision'),'records':len(context['records']),
            'tasks':args.tasks,'context':owner.CONTEXT,'fixed_date':owner.DATE,'plan':plan,
            'original_gold_files_opened':False,'weight_downloads_allowed':False,'GPU_used':False},indent=2));return 0
    if not args.output:p.error('Actual generation requires an explicit NEW --output namespace')
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)};received=[]
    def interrupted(signum,_):
        if not received:received.append(signum);raise InterruptedError('Text producer interrupted; committed prefix remains resumable')
    try:
        for signum in handlers:signal.signal(signum,interrupted)
        result=owner.evaluate(context,plan,args.output,resume=args.resume,allow_gpu=True,
            cache_dir=args.cache_dir,reserve_gb=args.reserve_gb,progress=lambda row:print(json.dumps(row),flush=True))
    finally:
        for signum,handler in handlers.items():signal.signal(signum,handler)
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
