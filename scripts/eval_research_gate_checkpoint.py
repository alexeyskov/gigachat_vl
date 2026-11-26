#!/usr/bin/env python3
"""Isolated P07 serial owner; CPU preflight by default, ROOT schedules GPU."""
import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.eval.research_gate_evaluation import prepare_context, evaluate, digest
from src.eval.research_gate_controls import CONDITIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--native-cache', type=Path, help='Mandatory existing same-checkpoint full512 native original; never generated here')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cache-dir', default='cache/vision')
    parser.add_argument('--reserve-gb', type=float, default=120.)
    parser.add_argument('--allow-gpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-completed', action='store_true', help='CPU-only complete journal revalidation/scoring; pending rows refuse')
    args = parser.parse_args()
    context = prepare_context(args.run, checkpoint=args.checkpoint, condition=args.condition,
        manifest=args.manifest, native_cache=args.native_cache, cache_dir=args.cache_dir)
    if not args.allow_gpu and not args.verify_completed:
        print(json.dumps({'status': 'cpu_preflight_only_no_generation', 'records': 512, 'condition': args.condition,
            'admission_sha256': digest(context['static']), 'native_reference_records': 512,
            'checkpoint_sha256': context['static']['base']['checkpoint']['artifact']['sha256'],
            'GPU_launched': False, 'learned64_guard_executed': False, 'P07_complete': False}, indent=2)); return
    if args.verify_completed and args.allow_gpu: raise ValueError('CPU verify-completed cannot enable GPU')
    seen = []; handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def stop(sig, _):
        if not seen: seen.append(sig); raise InterruptedError('P07 owner interrupted')
    try:
        for sig in handlers: signal.signal(sig, stop)
        result = evaluate(context, output=args.output, resume=args.resume, allow_gpu=args.allow_gpu,
            cache_dir=args.cache_dir, reserve_gb=args.reserve_gb,
            progress=lambda value: print(json.dumps(value), flush=True))
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
