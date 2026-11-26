#!/usr/bin/env python3
"""Admit the new queue after the active legacy GPU trainer exits naturally."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts import research_budget18_queue_v1 as q
OUT=ROOT/'research_results/additive_visual_v1/handoff'
OLD='research-budget18-v7';NEW='research-additive-v1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan-sha',required=True);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'scripts/research_additive_queue_v1.py','--plan-sha',args.plan_sha],cwd=ROOT,check=True)
    probe=q.read(ROOT/'research_results/additive_visual_v1/cuda_probe.json')
    if probe['status']!='passed' or len(probe['cases'])!=6:raise ValueError('Six actual CUDA native/NF4 branch checks required')
    service=q.supervisor_info(OLD);parent=q.proc(service['pid'])
    state=q.read(ROOT/'runs/research/budget18_queue_state_v1.json');job=state['active_job'];item=state['jobs'][job]['train_process']
    worker=q.proc(item['pid']);request=q.read(item['request'])
    if (service['statename']!='RUNNING' or not q.same(parent,q.proc(parent['pid'])) or parent['state']=='T'
        or not q.same(worker,q.proc(worker['pid'])) or worker['ppid']!=parent['pid']
        or request['kind']!='train' or request['job_id']!=job
        or request['controller_source_sha256']!=q.sha(ROOT/'scripts/research_budget18_queue_v7.py')
        or q.sha(item['request'])!=q.value(worker['argv'],'--request-sha256')):
        raise ValueError('Exact V7-owned active trainer required; no signals sent')
    lease=dict(active=True,controller=parent,coordinator=q.proc(os.getpid()),training_worker=worker,
               current_job=job,training_worker_signals_sent=0,created_unix=time.time(),plan_sha256=args.plan_sha)
    lease_path=OUT/'lease.json';q.atomic(lease_path,lease);guard=None;paused=False;old_exited=False;success=False
    old_config=Path('/etc/supervisor/conf.d/'+OLD+'.conf');original_config=old_config.read_text()
    def interrupted(_n,_f):raise InterruptedError('Handoff interrupted; restore previous admission owner')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        guard=subprocess.Popen([sys.executable,'scripts/run_qwen25_native_baseline_parallel_v1.py','--guardian',str(lease_path)],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
        os.kill(parent['pid'],signal.SIGSTOP);paused=True;start=time.monotonic();last=0
        while q.same(worker,q.proc(worker['pid'])):
            if time.monotonic()-start>7200:raise TimeoutError('Bounded2hour handoff expired')
            if time.monotonic()-last>30:
                print(json.dumps(dict(event='old_trainer_continues_new_queue_waits',job=job,pid=worker['pid'],
                    training_worker_signals_sent=0)),flush=True);last=time.monotonic()
            time.sleep(2)
        if not q.same(parent,q.proc(parent['pid'])):raise ValueError('Previous controller identity changed')
        os.kill(parent['pid'],signal.SIGTERM);os.kill(parent['pid'],signal.SIGCONT);paused=False
        deadline=time.monotonic()+180
        while q.same(parent,q.proc(parent['pid'])):
            if time.monotonic()>deadline:raise TimeoutError('Old CPU controller did not release its lease')
            time.sleep(1)
        old_exited=True
        if original_config.count('autostart=true')!=1:raise ValueError('Old autostart setting differs')
        old_config.write_text(original_config.replace('autostart=true','autostart=false'))
        subprocess.run(['supervisorctl','reread'],check=True)
        subprocess.run(['supervisorctl','update',OLD],check=True)
        subprocess.run(['supervisorctl','update',NEW],check=True)
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            info=q.supervisor_info(NEW)
            if info['statename']=='RUNNING':break
            time.sleep(1)
        if info['statename']!='RUNNING':raise RuntimeError('New queue failed to activate')
        success=True
        q.atomic(OUT/'completed.json',dict(status='completed_after_natural_trainer_exit',previous_job=job,
            new_controller=info,training_worker_signals_sent=0,elapsed_seconds=time.monotonic()-start,
            legacy_R1_R3_scope='old artifacts retained; unrun common-pool additions superseded, never labelled complete'))
    finally:
        if paused and q.same(parent,q.proc(parent['pid'])):os.kill(parent['pid'],signal.SIGCONT)
        if old_exited and not success:
            old_config.write_text(original_config)
            subprocess.run(['supervisorctl','reread'],check=False)
            subprocess.run(['supervisorctl','update',OLD],check=False)
            subprocess.run(['supervisorctl','start',OLD],check=False)
        q.atomic(lease_path,dict(lease,active=False,released_unix=time.time(),success=success))
        if guard:
            try:guard.wait(timeout=5)
            except subprocess.TimeoutExpired:guard.terminate();guard.wait(timeout=5)

if __name__=='__main__':main()
