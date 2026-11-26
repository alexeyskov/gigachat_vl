#!/usr/bin/env python3
"""Stock Small P07 serial facade; bounded metadata by default, ROOT-only GPU."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.eval import research_gate_stock_evaluation as facade


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--allow-gpu' in argv and 'torch' in sys.modules: raise RuntimeError('Actual stock P07 requires a fresh process before Torch import')
    facade.stock.bootstrap_environment()
    os.environ.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    owner = facade.producer(); parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    parser.add_argument('--condition', choices=owner.controls.CONDITIONS, default='learned')
    parser.add_argument('--manifest', type=Path); parser.add_argument('--native-cache', type=Path)
    parser.add_argument('--output', type=Path); parser.add_argument('--cache-dir', default='cache/vision')
    parser.add_argument('--reserve-gb', type=float, default=120.)
    parser.add_argument('--numerical-profile', choices=(facade.stock.PROFILE,))
    parser.add_argument('--allow-gpu', action='store_true'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-completed', action='store_true')
    args = parser.parse_args(argv)
    if not args.allow_gpu and not args.verify_completed:
        print(json.dumps(facade.metadata_plan(args.run, checkpoint=args.checkpoint, condition=args.condition,
            manifest=args.manifest, native_cache=args.native_cache), indent=2)); return 0
    if args.verify_completed and args.allow_gpu: parser.error('CPU verification cannot enable GPU')
    if args.run is None or args.numerical_profile != facade.stock.PROFILE:
        parser.error('Full admission requires --run and explicit --numerical-profile stock_deterministic_small_v1')
    facade.verify_release()
    context = owner.prepare_context(args.run, checkpoint=args.checkpoint, condition=args.condition,
        manifest=args.manifest, native_cache=args.native_cache, cache_dir=args.cache_dir)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}; received = []
    def stop(sig, _):
        if not received: received.append(sig); raise InterruptedError('Stock P07 owner interrupted')
    try:
        for sig in handlers: signal.signal(sig, stop)
        result = owner.evaluate(context, output=args.output, resume=args.resume, allow_gpu=args.allow_gpu,
            cache_dir=args.cache_dir, reserve_gb=args.reserve_gb, progress=lambda value: print(json.dumps(value), flush=True))
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)
    print(json.dumps(result, indent=2)); return 0


if __name__ == '__main__': raise SystemExit(main())
