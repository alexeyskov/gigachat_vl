#!/usr/bin/env python3
"""P07 Small c gate-control proposal. CPU preflight; no implicit GPU launch."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.research_gate_controls import CONDITIONS, digest, prepare_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    context = prepare_context(args.run, checkpoint=args.checkpoint, condition=args.condition, manifest=args.manifest)
    print(json.dumps({'status': 'cpu_preflight_only_no_generation', 'owner': context['static']['owner'],
        'condition': args.condition, 'records': len(context['records']),
        'static_protocol_sha256': digest(context['static']),
        'checkpoint_sha256': context['static']['checkpoint']['artifact']['sha256'],
        'actual_gate_registry_checked': False, 'GPU_launched': False, 'predictions_created': False,
        'scope': 'P07 context/registry proposal; production journal owner not enabled'}, indent=2))


if __name__ == '__main__':
    main()
