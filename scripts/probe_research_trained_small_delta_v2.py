#!/usr/bin/env python3
"""HF-cache metadata v2 adapter. Defaults to preflight; never publishes roles."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.utils import research_trained_delta_probe_v2 as adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'runs/research/smallthinker4__siglip2__4f__SFT_pilot_scienceqa_context_v2_balanced_s42')
    parser.add_argument('--checkpoint', choices=('best','last'), default='best')
    parser.add_argument('--output', type=Path, default=ROOT/'research_results/trained_small_delta_probe_v2')
    parser.add_argument('--anchor-census', action='store_true')
    parser.add_argument('--run-probe', action='store_true'); parser.add_argument('--cleanup-verified', action='store_true')
    parser.add_argument('--advise-source-cache', action='store_true'); parser.add_argument('--max-staging-gib', type=float, default=24)
    parser.add_argument('--reserve-gb', type=float, default=120); parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.max_staging_gib <= 24 or args.reserve_gb < 120 or not 1 <= args.threads <= 8:
        parser.error('Require staging<=24GiB, reserve>=120GiB, and 1..8 CPU threads')
    if args.anchor_census and args.run_probe: parser.error('Choose metadata census or actual derived-copy probe')
    if args.anchor_census:
        result = adapter.anchor_census(); result.pop('shapes')
        result['source_sha256'] = adapter.source_binding()
        result['torch_imported'] = 'torch' in sys.modules
    elif args.run_probe:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        torch.set_num_threads(args.threads)
        result = adapter.run_probe(args.run, args.output, role=args.checkpoint,
            max_staging_bytes=int(args.max_staging_gib*2**30), reserve_gb=args.reserve_gb,
            advise_source=args.advise_source_cache, cleanup_verified=args.cleanup_verified,
            progress=lambda event, **fields: print(json.dumps({'event':event,**fields}),flush=True))
    else:
        result = adapter.preflight(args.run, role=args.checkpoint, max_staging_bytes=int(args.max_staging_gib*2**30))
    print(json.dumps(result, indent=2),flush=True)


if __name__ == '__main__': main()
