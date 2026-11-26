#!/usr/bin/env python3
"""Continue the next fixed-block pair immediately while queue metadata migrates."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_additive_queue_v1 as original
from src.utils.research_storage import _atomic_json

def main():
    import fcntl
    plan=original.verify_plan(original.PLAN,'e3e531f9d3b2de09fdece91786402a65b567afd4109ed9d31523aca486129339')
    jobs=[j for j in plan['jobs'] if j['id'].startswith('smallthinker4__qwen25__fixed__R3__')]
    if [j['stage'] for j in jobs]!=['A1','SFT'] or any(j['mode']!='fixed' for j in jobs):raise ValueError('Exact fixed-mask pair required')
    out=ROOT/'research_results/fixed_only_v2';out.mkdir(exist_ok=True,parents=True)
    child=None
    def stop(_n,_f):
        if child and child.poll() is None:child.terminate()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (ROOT/'runs/research/budget18_queue_state_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for job in jobs:
            if job['stage']=='A1':
                proof=Path(jobs[1]['output'])/'native_reload_proof.json'
                if proof.exists() and json.loads(proof.read_text()).get('status')=='passed' and original.complete(jobs[1]):
                    # The main queue may already have retired this consumed A1.
                    # A later service restart must not recreate or retrain it.
                    continue
            if original.complete(job):continue
            if job['stage']=='A1':original.backend.extract_projector(job['backend'],job['vision'],original.BASE/'projectors'/f"{job['backend']}__{job['vision']}.pt")
            path=ROOT/f"logs/research/fixed_only_bootstrap_{job['stage']}.log"
            with path.open('ab') as stream:
                child=subprocess.Popen([sys.executable,'-u','scripts/run_research_additive_v1.py',*job['argv']],
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=stream,stderr=subprocess.STDOUT)
                _atomic_json(out/'bootstrap_state.json',dict(status='running',job=job['id'],pid=child.pid,log=str(path),started_unix=time.time()))
                print(json.dumps(dict(event='fixed_worker_started',job=job['id'],pid=child.pid)),flush=True)
                while child.poll() is None:
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:print(json.dumps(dict(event='fixed_worker_running',job=job['id'],pid=child.pid)),flush=True)
                if child.returncode:raise RuntimeError(f'Fixed bootstrap worker failed: {path}')
                if not original.complete(job):raise ValueError('Fixed worker did not publish complete artifact')
        _atomic_json(out/'bootstrap_state.json',dict(status='completed',completed_unix=time.time()))

if __name__=='__main__':main()
