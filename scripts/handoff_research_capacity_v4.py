#!/usr/bin/env python3
"""Change only CPU ownership after the active2-block GPU worker exits normally."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_budget18_queue_v1 as q
from scripts import research_additive_queue_v3 as old
from scripts import research_capacity_queue_v4 as new
OUT=new.OUT/'handoff'


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan-sha',required=True);args=p.parse_args()
    new.verify_plan(new.PLAN,args.plan_sha);old.verify_plan(old.PLAN,new.OLD_SHA)
    OUT.mkdir(parents=True,exist_ok=True)
    info=q.supervisor_info('research-additive-v1');parent=q.proc(info['pid']);state=q.read(old.STATE)
    job=state['active_job'];row=state['jobs'][job];worker=q.proc(row['pid'])
    if (info['statename']!='RUNNING' or not parent or parent['state']=='T'
        or 'scripts/research_additive_queue_v3.py' not in parent['argv']
        or state['plan_sha256']!=new.OLD_SHA or not q.same(worker,q.proc(row['pid']))
        or worker['ppid']!=parent['pid'] or worker['argv'][2:]!=row['argv']):
        raise ValueError('Exact active V3-owned worker required; no signals sent')
    config=Path('/etc/supervisor/conf.d/research-additive-v1.conf');original_config=config.read_text()
    lease=dict(active=True,controller=parent,coordinator=q.proc(os.getpid()),worker=worker,
        previous_job=job,GPU_worker_signals_sent=0,plan_sha256=args.plan_sha,created_unix=time.time())
    path=OUT/'lease.json';q.atomic(path,lease);guard=None;paused=False
    def stop(_n,_f):raise InterruptedError('Capacity handoff interrupted; restore CPU owner')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        guard=subprocess.Popen([sys.executable,'scripts/run_qwen25_native_baseline_parallel_v1.py','--guardian',str(path)],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
        os.kill(parent['pid'],signal.SIGSTOP);paused=True;start=time.monotonic();last=0
        while q.same(worker,q.proc(worker['pid'])):
            if time.monotonic()-start>7200:raise TimeoutError('Natural-boundary handoff exceeded2hours')
            if time.monotonic()-last>30:
                print(json.dumps(dict(event='active_training_continues_four_block_queue_waits',job=job,GPU_worker_signals_sent=0)),flush=True);last=time.monotonic()
            time.sleep(2)
        os.kill(parent['pid'],signal.SIGTERM);os.kill(parent['pid'],signal.SIGCONT);paused=False
        deadline=time.monotonic()+180
        while q.same(parent,q.proc(parent['pid'])):
            if time.monotonic()>deadline:raise TimeoutError('Old CPU queue did not release its locks')
            time.sleep(1)
        # Publish source identity before activation; the new queue may finalize
        # the old worker's checkpoint without retraining it.
        q.atomic(OUT/'completed.json',dict(status='source_worker_finished_naturally',previous_job=job,
            worker=worker,GPU_worker_signals_sent=0,plan_sha256=args.plan_sha))
        lines=original_config.splitlines()
        lines=[('command=/venv/main/bin/python -u scripts/research_capacity_queue_v4.py --run --plan-sha '+args.plan_sha
                if x.startswith('command=') else 'stdout_logfile=/workspace/gigachat_vl/logs/research/capacity_queue_v4.log'
                if x.startswith('stdout_logfile=') else x) for x in lines]
        config.write_text('\n'.join(lines)+'\n')
        subprocess.run(['supervisorctl','reread'],check=True)
        subprocess.run(['supervisorctl','update','research-additive-v1'],check=True)
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            active=q.supervisor_info('research-additive-v1')
            if active['statename']=='RUNNING':break
            time.sleep(1)
        if active['statename']!='RUNNING':raise RuntimeError('Capacity queue failed to activate')
        q.atomic(OUT/'completed.json',dict(status='completed_after_natural_GPU_exit',previous_job=job,
            worker=worker,new_owner=active,GPU_worker_signals_sent=0,plan_sha256=args.plan_sha))
    finally:
        if paused and q.same(parent,q.proc(parent['pid'])):os.kill(parent['pid'],signal.SIGCONT)
        q.atomic(path,dict(lease,active=False,released_unix=time.time()))
        if guard:
            try:guard.wait(timeout=5)
            except subprocess.TimeoutExpired:guard.terminate();guard.wait(timeout=5)


if __name__=='__main__':main()
