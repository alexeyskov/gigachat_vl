#!/usr/bin/env python3
"""Admit a bounded Qwen baseline alongside an owned, memory-compatible trainer.

Never signals a training worker. A short CPU-controller lease prevents its next
GPU admission racing the baseline. An independent guardian releases the lease
even if this coordinator is killed. The current trainer continues throughout.
"""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts import research_budget18_queue_v1 as q
OUT=ROOT/'research_results/qwen25_native_int4_internal116_v1'
SERVICE='research-budget18-v6'
CONTROLLER=ROOT/'scripts/research_budget18_queue_v6.py'


def stop_owned(identity,timeout=30):
    if not q.same(identity,q.proc(identity['pid'])):return
    os.killpg(identity['pid'],signal.SIGTERM)
    deadline=time.monotonic()+timeout
    while q.same(identity,q.proc(identity['pid'])) and time.monotonic()<deadline:time.sleep(.5)
    if q.same(identity,q.proc(identity['pid'])):os.killpg(identity['pid'],signal.SIGKILL)


def guardian(lease_path):
    """Own process group, no CUDA imports, can survive coordinator SIGKILL."""
    while True:
        lease=q.read(lease_path)
        if not lease.get('active'):return
        if not q.same(lease['coordinator'],q.proc(lease['coordinator']['pid'])):
            child=lease.get('baseline_worker')
            if child:stop_owned(child,timeout=10)
            time.sleep(3)
            parent=lease['controller']
            if q.same(parent,q.proc(parent['pid'])):os.kill(parent['pid'],signal.SIGCONT)
            q.atomic(lease_path,dict(lease,active=False,recovered_by_guardian=True,recovered_unix=time.time()))
            return
        time.sleep(1)


def latest_step(job):
    path=ROOT/'runs/research'/job/'metrics.jsonl'
    if not path.exists():return None
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-100000));lines=f.read().splitlines()
    for line in reversed(lines):
        try:row=json.loads(line)
        except ValueError:continue
        if row.get('event')=='train':return row
    return None


def target():
    service=q.supervisor_info(SERVICE)
    if service['statename']!='RUNNING':return None,'controller_not_running'
    parent=q.proc(service['pid']);state=q.read(ROOT/'runs/research/budget18_queue_state_v1.json')
    job=state.get('active_job');entry=state.get('jobs',{}).get(job,{})
    item=entry.get('train_process',{});trainer=q.proc(item.get('pid',-1))
    if (not trainer or not parent or trainer['state'] in ('Z','X') or not trainer['argv']
        or item.get('status')!='running' or trainer['ppid']!=parent['pid']):
        return None,'waiting_for_owned_trainer'
    if parent['state']=='T':return None,'waiting_for_existing_controller_lease'
    request=q.read(item['request']);argv=request['argv']
    if (request['kind']!='train' or request['job_id']!=job
        or request['controller_source_sha256']!=q.sha(CONTROLLER)
        or any(request['parent'].get(k)!=parent[k] for k in ('pid','birth','argv'))
        or q.sha(item['request'])!=q.value(trainer['argv'],'--request-sha256')):
        raise ValueError('Live training request ownership/source differs')
    stage=q.value(argv,'--stage');recipe=q.value(argv,'--recipe')
    if stage!='A0' and not(stage=='SFT' and recipe in ('1b','1c')):return None,'memory_incompatible_stage:'+str(job)
    step=latest_step(job)
    if not step or step['step']<2 or step['total_steps']-step['step']<12:return None,'waiting_for_long_training_window'
    if step.get('peak_vram_gb',100)>60:return None,'training_peak_above_parallel_budget'
    result=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    free_mib=float(result.stdout.strip())
    apps=subprocess.run(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    if set(apps.stdout.split())!={str(trainer['pid'])}:return None,'foreign_GPU_process_present'
    if free_mib<24576:return None,'waiting_for_24GiB_free_GPU'
    from src.utils.research_lossless_delta import memory_guard
    from src.utils.research_storage import DiskBudget
    try:memory_guard(32*2**30);DiskBudget(ROOT,32).ensure_space(32*2**20)
    except MemoryError:return None,'waiting_for_host_memory_headroom'
    return dict(controller=parent,trainer=trainer,job=job,training_step=step,free_GPU_MiB=free_mib),'ready'


def run_window(info):
    window=OUT/('parallel_'+str(time.time_ns()));window.mkdir(parents=True)
    lease_path=window/'lease.json';child=None;guard=None;leased=False
    lease=dict(info,active=True,coordinator=q.proc(os.getpid()),training_worker_signals_sent=0,
               coordinator_source_sha256=q.sha(Path(__file__)),
               evaluator_source_sha256=q.sha(ROOT/'scripts/eval_qwen25_native_internal116_v1.py'),
               created_unix=time.time(),controller_lease='only CPU queue owner; active trainer continues')
    q.atomic(lease_path,lease)
    try:
        guard=subprocess.Popen([sys.executable,'-u',__file__,'--guardian',str(lease_path)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
        os.kill(info['controller']['pid'],signal.SIGSTOP);leased=True
        if not q.same(info['trainer'],q.proc(info['trainer']['pid'])):return False
        command=[sys.executable,'-u',str(ROOT/'scripts/eval_qwen25_native_internal116_v1.py'),'--parent',str(os.getpid())]
        with (window/'worker.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            lease['baseline_worker']=q.proc(child.pid);q.atomic(lease_path,lease)
            started=time.monotonic()
            while child.poll() is None:
                if time.monotonic()-started>800:
                    stop_owned(lease['baseline_worker']);break
                time.sleep(2)
            child.wait(timeout=30)
        q.atomic(window/'completed.json',dict(returncode=child.returncode,
            training_step_before=info['training_step'],training_step_after=latest_step(info['job']),
            training_worker_signals_sent=0,elapsed_seconds=time.monotonic()-started))
        if child.returncode:raise RuntimeError('Baseline worker failed; inspect '+str(window/'worker.log'))
        return (OUT/'summary.json').exists()
    finally:
        if child and child.poll() is None:
            identity=lease.get('baseline_worker')
            if identity:stop_owned(identity)
            child.wait(timeout=30)
        if leased and q.same(info['controller'],q.proc(info['controller']['pid'])):
            os.kill(info['controller']['pid'],signal.SIGCONT)
        q.atomic(lease_path,dict(lease,active=False,released_unix=time.time()))
        if guard:
            try:guard.wait(timeout=5)
            except subprocess.TimeoutExpired:guard.terminate();guard.wait(timeout=5)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--guardian',type=Path);args=p.parse_args()
    if args.guardian:return guardian(args.guardian)
    def interrupted(_n,_f):raise InterruptedError('Baseline coordinator interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    last=0
    while not (OUT/'summary.json').exists():
        info,reason=target() if (OUT/'source_ready.json').exists() else (None,'waiting_for_source')
        if info:
            print(json.dumps({'event':'admitted_parallel_baseline','job':info['job'],'step':info['training_step']['step']}),flush=True)
            if run_window(info):break
        if time.monotonic()-last>55:
            print(json.dumps({'event':'baseline_waiting','reason':reason,'time':time.time()}),flush=True);last=time.monotonic()
        time.sleep(10)
    print(json.dumps({'event':'baseline_completed','summary':str(OUT/'summary.json')}),flush=True)
    from scripts.retire_qwen25_native_baseline_source_v1 import retire
    retire()


if __name__=='__main__':main()
