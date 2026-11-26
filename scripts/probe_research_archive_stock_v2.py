#!/usr/bin/env python3
"""Metadata-only stock archival preflight; explicit ROOT scheduling for GPU proof."""
import argparse,json,os
from pathlib import Path
import signal,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils import research_archive_stock_v2 as stock
DEFAULT_CPU=ROOT/'research_results/trained_small_delta_actual4f_best_v2/verified.json'
DEFAULT_OUTPUT=ROOT/'research_results/trained_small_archive_stock_gpu_v2'

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-report',type=Path,default=DEFAULT_CPU)
 p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--allow-gpu',action='store_true');p.add_argument('--nonce')
 p.add_argument('--parent-pid',type=int);p.add_argument('--parent-start',type=int);p.add_argument('--parent-command-sha256')
 p.add_argument('--deadline-monotonic',type=float);p.add_argument('--timeout-seconds',type=int,default=1170)
 p.add_argument('--reserve-gb',type=float,default=120);p.add_argument('--cache-dir',default='cache/vision');a=p.parse_args(argv)
 if a.allow_gpu and 'torch' in sys.modules:raise RuntimeError('Fresh process required before deterministic workspace setup')
 os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 helper=stock.gpu_helper_v2()
 try:plan=helper.prepare_plan(a.cpu_report,reserve_gb=a.reserve_gb)
 except (ValueError,OSError,KeyError) as error:
  if a.allow_gpu:raise
  print(json.dumps({'status':'gate_closed','reason':str(error),'GPU_launched':False}));return 2
 if not a.allow_gpu:
  print(json.dumps({'status':'metadata_preflight_only','plan':plan,'GPU_launched':False},indent=2));return 0
 if any(v is None for v in (a.nonce,a.parent_pid,a.parent_start,a.parent_command_sha256,a.deadline_monotonic)):p.error('Explicit ROOT nonce/parent identity/shared deadline required')
 handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
 def interrupted(signum,_):raise InterruptedError('Own stock archive GPU proof interrupted '+str(signum))
 try:
  for s in handlers:signal.signal(s,interrupted)
  result=helper.execute(plan,a.output,nonce=a.nonce,timeout=a.timeout_seconds,cache_dir=a.cache_dir,
   expected_parent={'pid':a.parent_pid,'start_time':a.parent_start,'command_sha256':a.parent_command_sha256},coordinator_deadline=a.deadline_monotonic)
 finally:
  for s,h in handlers.items():signal.signal(s,h)
 print(json.dumps({'status':result['status'],'report':str(a.output/'report.json'),'publication_performed':False,'legacy_validation_passed':False}));return 0

if __name__=='__main__':raise SystemExit(main())
