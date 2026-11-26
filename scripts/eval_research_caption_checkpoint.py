#!/usr/bin/env python3
"""Pinned372 caption owner. CPU preflight by default; ROOT alone schedules GPU."""
import argparse
import json
import signal
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.research_caption_generation import prepare_context,evaluate,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path);parser.add_argument('--checkpoint',choices=('best','last'),default='best')
    parser.add_argument('--bundle',type=Path,default=ROOT/'datasets/research/caption-eval-pilot-en-v1')
    parser.add_argument('--output',type=Path);parser.add_argument('--cache-dir',default='cache/vision')
    parser.add_argument('--allow-gpu',action='store_true');parser.add_argument('--score-only',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--reserve-gb',type=float,default=120.)
    args=parser.parse_args();context=prepare_context(args.run,args.bundle,checkpoint=args.checkpoint,cache_dir=args.cache_dir)
    if not args.allow_gpu and not args.score_only:
        print(json.dumps({'status':'cpu_preflight_only_no_generation','records':len(context['records']),
            'static_protocol_sha256':digest(context['static']),'checkpoint_sha256':context['static']['checkpoint']['artifact']['sha256'],
            'gold_references_opened':False,'gpu_launched':False},indent=2));return
    output=args.output or args.run/'evaluation/caption_pilot_en_v1'
    received=[];handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupt(sig,_):
        if not received:received.append(sig);raise InterruptedError('Caption owner interrupted')
    try:
        for sig in handlers:signal.signal(sig,interrupt)
        result=evaluate(context,output,resume=args.resume,allow_gpu=args.allow_gpu,score_only=args.score_only,reserve_gb=args.reserve_gb,
            progress=lambda event:print(json.dumps(event),flush=True))
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
