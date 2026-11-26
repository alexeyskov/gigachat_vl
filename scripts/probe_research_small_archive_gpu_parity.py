#!/usr/bin/env python3
"""Actual trained4f archive parity proposal; metadata preflight is the default."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from src.utils import research_archive_gpu_parity as parity


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-report',type=Path,default=ROOT/'research_results/trained_small_delta_probe_v2/verified.json')
    p.add_argument('--output',type=Path,default=ROOT/'research_results/trained_small_archive_gpu_parity_actual')
    p.add_argument('--allow-gpu',action='store_true')
    p.add_argument('--nonce',help='Explicit ROOT scheduling64hex nonce; no implicit launch')
    p.add_argument('--timeout-seconds',type=int,default=1200)
    p.add_argument('--parent-pid',type=int)
    p.add_argument('--parent-start',type=int)
    p.add_argument('--parent-command-sha256')
    p.add_argument('--deadline-monotonic',type=float)
    p.add_argument('--cache-dir',default='cache/vision')
    p.add_argument('--reserve-gb',type=float,default=120.)
    args=p.parse_args(argv)
    if not args.allow_gpu:
        result=parity.preflight(args.cpu_report,reserve_gb=args.reserve_gb)
        print(json.dumps(result,indent=2));return 0 if result['status']!='gate_closed' else 2
    if args.nonce is None or len(args.nonce)!=64 or any(c not in '0123456789abcdef' for c in args.nonce):
        p.error('Actual GPU work requires an explicit ROOT64hex scheduling nonce')
    if (args.parent_pid is None or args.parent_start is None or args.parent_command_sha256 is None or
            args.deadline_monotonic is None):p.error('Actual GPU work requires exact coordinator parent identity and lease deadline')
    # Admission and all source gates run before ResourceGuard can query/init a GPU.
    plan=parity.prepare_plan(args.cpu_report,reserve_gb=args.reserve_gb)
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)};seen=[]
    def interrupted(signum,_):
        if not seen:seen.append(signum);raise InterruptedError('Own archive parity interrupted; source roles remain authoritative')
    try:
        for s in handlers:signal.signal(s,interrupted)
        result=parity.execute(plan,args.output,nonce=args.nonce,timeout=args.timeout_seconds,
            cache_dir=args.cache_dir,expected_parent={'pid':args.parent_pid,'start_time':args.parent_start,
                'command_sha256':args.parent_command_sha256},coordinator_deadline=args.deadline_monotonic)
    finally:
        for s,h in handlers.items():signal.signal(s,h)
    print(json.dumps({'status':result['status'],'report':str(args.output/'report.json'),
        'source_authoritative':result['source_authoritative'],'publication_performed':False},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
