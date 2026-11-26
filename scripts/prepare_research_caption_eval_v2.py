#!/usr/bin/env python3
"""Explicit v2 signal-safe pilot caption preparation; never replaces v1."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.research_caption_data_v2 import prepare_caption_bundle, V2_PROFILE


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, default=ROOT/'datasets/research/pilot-en-v1/alignment_validation.jsonl')
    parser.add_argument('--exclude-train', type=Path, action='append', help='Repeat to replace the two default train exclusions')
    parser.add_argument('--output',type=Path,required=True,help='New explicit v2 bundle path; original v1 stays immutable')
    parser.add_argument('--preparation-profile',choices=(V2_PROFILE,),default=V2_PROFILE,
        help='Signal-safe v2 protocol (legacy CLI/default live in the unchanged v1 script)')
    parser.add_argument('--seed', type=int, default=42); parser.add_argument('--cap', type=int, default=1024)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--reserve-gb', type=float, default=120.)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument('--memory-reserve-gb', type=float, default=2.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.reserve_gb) or args.reserve_gb < 0: parser.error('--reserve-gb must be finite and nonnegative')
    if not math.isfinite(args.memory_reserve_gb) or args.memory_reserve_gb < 0: parser.error('--memory-reserve-gb must be finite and nonnegative')
    exclusions = args.exclude_train if args.exclude_train is not None else [
        ROOT/'datasets/research/pilot-en-v1/alignment_train.jsonl',
        ROOT/'datasets/research/pilot-en-scienceqa-context-v2-mixture/sft_train.jsonl']
    interrupted = []; shutdowns=[]; handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    def progress(event):
        if event.get('event')=='pixel_audit_workers_closed':shutdowns.append(event)
        print(json.dumps(event),flush=True)
    def stop(signum, _frame):
        if not interrupted:
            interrupted.append(signum)
            raise InterruptedError('Caption pixel audit interrupted')
    try:
        for signum in handlers: signal.signal(signum, stop)
        result = prepare_caption_bundle(args.pool, exclusions, args.output, seed=args.seed, cap=args.cap,
            resume=args.resume, reserve_bytes=int(args.reserve_gb*2**30), project_root=ROOT, workers=args.workers,
            memory_reserve_bytes=int(args.memory_reserve_gb*2**30), preparation_source_paths=[Path(__file__)],
            preparation_profile=args.preparation_profile,
            progress=progress)
    except InterruptedError:
        if not interrupted: raise
        joined=all(row['alive_worker_threads']==0 and row['all_futures_done'] for row in shutdowns)
        print(json.dumps({'event': 'caption_audit_interrupted', 'signal': interrupted[0], 'workers_joined':joined,
            'worker_shutdowns':shutdowns}),flush=True)
        if not joined:raise RuntimeError('Interrupted caption workers did not complete shutdown')
        return 128+interrupted[0]
    finally:
        for signum, handler in handlers.items(): signal.signal(signum, handler)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
