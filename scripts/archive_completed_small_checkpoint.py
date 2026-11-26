#!/usr/bin/env python3
"""Prospective Small archival: closed measured CPU/GPU gate; preflight default."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils import research_checkpoint_archival as archival


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--activation',type=Path)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--reserve-gb',type=float,default=120)
    parser.add_argument('--max-staging-gib',type=float,default=24)
    args=parser.parse_args()
    try:
        result=(archival.publish_completed_checkpoint(args.run,activation=args.activation,
            reserve_gb=args.reserve_gb,max_staging_bytes=int(args.max_staging_gib*2**30),
            progress=lambda event,**fields:print(json.dumps({'event':event,**fields}),flush=True)) if args.publish
            else archival.preflight(args.run,activation=args.activation))
    except (OSError,ValueError,RuntimeError,MemoryError) as error:
        print(json.dumps({'status':'refused','error_type':type(error).__name__,'error':str(error),
            'scientific_publication_completed':False}),flush=True);raise SystemExit(2)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
