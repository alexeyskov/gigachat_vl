#!/usr/bin/env python3
"""Switch CPU queue owners only after their current GPU trainer naturally exits."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts import research_budget18_queue_v1 as q
OUT=ROOT/'research_results/budget18_giga_native_eot_audit_v1/handoff_v8'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    service=q.supervisor_info('research-budget18-v7');parent=q.proc(service['pid'])
    state=q.read(ROOT/'runs/research/budget18_queue_state_v1.json');job=state['active_job'];item=state['jobs'][job]['train_process']
    worker=q.proc(item['pid']);request=q.read(item['request'])
    if (service['statename']!='RUNNING' or not q.same(parent,q.proc(parent['pid'])) or parent['state']=='T'
        or not q.same(worker,q.proc(worker['pid'])) or worker['ppid']!=parent['pid']
        or request['kind']!='train' or request['job_id']!=job
        or request['controller_source_sha256']!=q.sha(ROOT/'scripts/research_budget18_queue_v7.py')
        or q.sha(item['request'])!=q.value(worker['argv'],'--request-sha256')):
        raise ValueError('Exact live V7-owned trainer required; no signals sent')
    subprocess.run([sys.executable,'scripts/research_budget18_queue_v8.py',
        '--resource-ready','configs/research/budget18_resource_v8_ready.json','--resource-ready-sha',
        q.sha(ROOT/'configs/research/budget18_resource_v8_ready.json')],cwd=ROOT,check=True)
    lease=dict(active=True,controller=parent,coordinator=q.proc(os.getpid()),training_worker=worker,
               current_job=job,training_worker_signals_sent=0,created_unix=time.time())
    lease_path=OUT/'lease.json';q.atomic(lease_path,lease);guard=None;paused=False
    def interrupted(_n,_f):raise InterruptedError('Handoff interrupted; resume original CPU owner')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        guard=subprocess.Popen([sys.executable,'scripts/run_qwen25_native_baseline_parallel_v1.py','--guardian',str(lease_path)],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
        os.kill(parent['pid'],signal.SIGSTOP);paused=True;start=time.monotonic();last=0
        while q.same(worker,q.proc(worker['pid'])):
            if time.monotonic()-start>7200:raise TimeoutError('Bounded2hour handoff expired; original queue restored')
            if time.monotonic()-last>55:
                print(json.dumps(dict(event='waiting_for_current_trainer_natural_exit',job=job,pid=worker['pid'],
                    training_worker_signals_sent=0)),flush=True);last=time.monotonic()
            time.sleep(2)
        # The GPU worker is already gone. Only the CPU controller is signalled.
        if not q.same(parent,q.proc(parent['pid'])):raise ValueError('Original CPU owner identity changed')
        os.kill(parent['pid'],signal.SIGTERM);os.kill(parent['pid'],signal.SIGCONT);paused=False
        deadline=time.monotonic()+180
        while q.same(parent,q.proc(parent['pid'])):
            if time.monotonic()>deadline:raise TimeoutError('CPU owner did not release its lock')
            time.sleep(1)
        old_config=Path('/etc/supervisor/conf.d/research-budget18-v7.conf')
        raw=old_config.read_text()
        if raw.count('autostart=true')!=1:raise ValueError('Old service autostart setting differs')
        old_config.write_text(raw.replace('autostart=true','autostart=false'))
        subprocess.run(['supervisorctl','reread'],check=True)
        subprocess.run(['supervisorctl','update','research-budget18-v7'],check=True)
        subprocess.run(['supervisorctl','update','research-budget18-v8'],check=True)
        time.sleep(2);new=q.supervisor_info('research-budget18-v8')
        if new['statename']!='RUNNING':raise RuntimeError('New queue did not start')
        q.atomic(OUT/'completed.json',dict(status='completed_after_natural_trainer_exit',previous_job=job,
            new_controller=new,training_worker_signals_sent=0,elapsed_seconds=time.monotonic()-start))
        print(json.dumps(dict(event='handoff_completed',new_controller=new)),flush=True)
    finally:
        if paused and q.same(parent,q.proc(parent['pid'])):os.kill(parent['pid'],signal.SIGCONT)
        q.atomic(lease_path,dict(lease,active=False,released_unix=time.time()))
        if guard:
            try:guard.wait(timeout=5)
            except subprocess.TimeoutExpired:guard.terminate();guard.wait(timeout=5)


if __name__=='__main__':main()
