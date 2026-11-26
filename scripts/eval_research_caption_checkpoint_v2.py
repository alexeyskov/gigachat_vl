#!/usr/bin/env python3
"""Explicit v2 caption owner; CPU preflight default, ROOT schedules GPU later."""
import argparse
import json
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval import research_caption_generation_v2 as caption


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path);parser.add_argument('--checkpoint',choices=('best','last'),default='best')
    parser.add_argument('--bundle',type=Path,default=caption.DEFAULT_BUNDLE)
    parser.add_argument('--output',type=Path);parser.add_argument('--cache-dir',default='cache/vision')
    parser.add_argument('--allow-gpu',action='store_true');parser.add_argument('--score-only',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--reserve-gb',type=float,default=120.)
    args=parser.parse_args(argv)
    if caption.profile_status()['status'].startswith('closed'):
        print(json.dumps(caption.profile_status(),indent=2));return 2
    received=[];handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def interrupt(sig,_):
        if not received:received.append(sig);raise InterruptedError('Caption v2 owner interrupted')
    try:
        for sig in handlers:signal.signal(sig,interrupt)
        context=caption.prepare_context(args.run,args.bundle,checkpoint=args.checkpoint,cache_dir=args.cache_dir)
        if not args.allow_gpu and not args.score_only:
            print(json.dumps({'status':'cpu_preflight_only_no_generation','owner':caption.OWNER,'records':len(context['records']),
                'data_protocol_sha256':caption.DATA_PROTOCOL_SHA256,'static_protocol_sha256':caption.protocol_digest(context['static']),
                'checkpoint_sha256':context['static']['checkpoint']['artifact']['sha256'],'gold_references_opened':False,'gpu_launched':False},indent=2))
            return 0
        output=args.output or args.run/'evaluation'/caption.DEFAULT_OUTPUT
        result=caption.evaluate(context,output,resume=args.resume,allow_gpu=args.allow_gpu,score_only=args.score_only,
            reserve_gb=args.reserve_gb,progress=lambda event:print(json.dumps(event),flush=True))
        print(json.dumps(result,indent=2),flush=True);return 0
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=='__main__':raise SystemExit(main())
