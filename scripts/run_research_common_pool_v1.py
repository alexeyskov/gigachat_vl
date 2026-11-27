#!/usr/bin/env python3
"""A fresh GPU process for one R4/R5 phase or its independent native replay."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True,type=Path)
    parser.add_argument('--mode',choices=('train','evaluate','validate','probe'),default='train')
    parser.add_argument('--allow-gpu',action='store_true')
    args=parser.parse_args()
    if not args.allow_gpu: raise ValueError('GPU execution requires --allow-gpu')
    from src.utils import research_common_pool_training_v1 as training
    job=training.read(args.job)
    result=(training.validate(job) if args.mode=='validate' else
            training.train(job,evaluate_existing=args.mode=='evaluate',probe=args.mode=='probe'))
    print(json.dumps(dict(event='worker_completed',mode=args.mode,result=result),default=str),flush=True)


if __name__=='__main__': main()
