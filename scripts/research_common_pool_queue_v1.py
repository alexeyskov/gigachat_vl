#!/usr/bin/env python3
"""Six separate R4/R5 arms, sequential GPU stages, model-only best/last."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils import research_common_pool_training_v1 as training
from src.utils import research_additive_runner_v1 as core
from src.utils.research_storage import DiskBudget,_atomic_json

BASE=training.BASE
OUT=ROOT/'research_results/common_pool_v1'
PLAN=ROOT/'configs/research/common_pool_plan_v1.json'
STATE=BASE/'queue_state.json'


def build_plan():
    jobs=[]
    for key,repo in core.MODELS.items():
        for method in ('R4','R5'):
            arm=f'{key}__qwen25__{method}__common8_v1'
            source=ROOT/f'runs/research/additive_visual_v1/projectors/{key}__qwen25.pt'
            for phase in training.PHASES:
                output=BASE/arm/phase
                jobs.append(dict(id=arm+'__'+phase,arm=arm,phase=phase,model_key=key,model=repo,
                    vision=core.VISIONS['qwen25'],method=method,output=str(output),source_checkpoint=str(source)))
                source=output/'last/state.pt'
    return dict(name='common_pool_R4_R5_v1',configurations=6,phase_jobs=24,execution=training.definition(),
                jobs=jobs,initialization='fresh pinned NF4 LLM and experts, retained aligned P only; no old SFT weights imported',
                old18_weights='not modified or retired; paper_v1 export remains available')


def verify_plan(digest):
    if training.sha(PLAN)!=digest or training.read(PLAN)!=build_plan():
        raise ValueError('Common-pool source/data/architecture plan changed')
    return training.read(PLAN)


def complete(job):
    run=Path(job['output'])
    if not (run/'completed.json').is_file() or not (run/'native_reload_proof.json').is_file():return False
    proof=training.read(run/'native_reload_proof.json');done=training.read(run/'completed.json')
    if proof['status']!='passed' or proof['mismatches'] or proof['checkpoint_sha256']!=done['checkpoint_sha256'] or proof['execution']!=training.definition():return False
    if (run/'checkpoint_pruned.json').is_file():
        return job['phase']!='DET' and training.read(run/'checkpoint_pruned.json')['status']=='pruned'
    if not (run/'last/state.pt').is_file() or (run/'best').resolve()!=(run/'last').resolve():
        raise ValueError('One shared owned best/last object required')
    return True


def retire_predecessor(job):
    if job['phase']=='A1':return
    run=Path(job['output']);source=Path(job['source_checkpoint'])
    source_run=source.parent.parent
    if not source_run.resolve().is_relative_to(BASE.resolve()) or source_run.name not in training.PHASES[:-1]:
        raise ValueError('Retirement is limited to this new arm, never old18/download weights')
    if (source_run/'checkpoint_pruned.json').exists():return
    if not complete(job):raise ValueError('Successor independent proof required before retirement')
    info=training.read(run/'initialization.json')
    if training.sha(source)!=info['source_sha256']:raise ValueError('Consumed source SHA differs')
    from scripts.retire_superseded_research_v1 import inspect_run,retire_one
    from scripts.research_archival_stock_queue_v5 import scan_anchor_users
    try:
        result=retire_one(inspect_run(source_run),run/'predecessor_retired.json',scan_anchor_users)
        _atomic_json(run/'predecessor_retired.json',dict(status='consumed_new_phase_retired',**result))
    except RuntimeError as error:
        _atomic_json(run/'predecessor_retired.json',dict(status='deferred_open_or_mapped',error=str(error)))


def gpu_idle():
    return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()


def maintain_memory():
    from scripts.guard_research_control_tail_memory_v1 import memory,reclaim
    if memory()['available_GiB']<72:
        reclaim(training.read(ROOT/'configs/research/fixed_capacity_eval_plan_v5.json'))


def status(state,plan):
    active=state.get('active_job')
    job=next((item for item in plan['jobs'] if item['id']==active),None)
    value=dict(status=state['status'],active_job=active,phase_jobs_completed=sum(item.get('status')=='completed' for item in state['jobs'].values()),
               total_phase_jobs=24,final_configurations_completed=sum(item.get('status')=='completed' and key.endswith('__DET') for key,item in state['jobs'].items()),
               total_configurations=6,free_disk_GiB=DiskBudget(BASE,32).report()['free_gib'],updated_UTC=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))
    if job:
        run=Path(job['output'])
        for name in ('metrics.jsonl','evaluation_progress.json','test_progress.json'):
            path=run/name
            if path.exists():
                lines=path.read_text().splitlines()
                if lines:
                    try:value[name]=json.loads(lines[-1])
                    except json.JSONDecodeError:pass
    OUT.mkdir(parents=True,exist_ok=True);_atomic_json(OUT/'status.json',value)
    text=f"# R4/R5: общий пул экспертов\n\n{value['updated_UTC']}\n\nСтатус: {value['status']}. Завершено {value['final_configurations_completed']}/6 конфигураций, {value['phase_jobs_completed']}/24 этапов.\n\nАктивный этап: `{active}`. Свободно {value['free_disk_GiB']:.1f} ГиБ.\n\n"
    if 'metrics.jsonl' in value:
        row=value['metrics.jsonl'];text+=f"Последняя запись: {json.dumps(row,ensure_ascii=False)}\n"
    temporary=ROOT/'RESEARCH_COMMON_POOL_STATUS.md.tmp';temporary.write_text(text);temporary.replace(ROOT/'RESEARCH_COMMON_POOL_STATUS.md')


def main():
    p=argparse.ArgumentParser();p.add_argument('--write-plan',action='store_true');p.add_argument('--plan-sha');p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.write_plan:
        value=build_plan()
        if PLAN.exists() and training.read(PLAN)!=value:raise ValueError('Existing common-pool plan differs')
        _atomic_json(PLAN,value);print(training.sha(PLAN));return
    plan=verify_plan(args.plan_sha)
    if not args.run:print(json.dumps(dict(status='CPU_plan_preflight_passed',jobs=len(plan['jobs']))));return
    BASE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    admission=(ROOT/'runs/research/budget18_queue_state_v1.lock').open('a');fcntl.flock(admission,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(BASE/'.queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=training.read(STATE) if STATE.exists() else dict(status='running',plan_sha256=args.plan_sha,jobs={})
    if state['plan_sha256']!=args.plan_sha:raise ValueError('Existing queue has a different plan')
    stop=False;child=None
    def persist():
        state['updated_unix']=time.time();_atomic_json(STATE,state);status(state,plan)
    def interrupt(_n,_f):
        nonlocal stop
        stop=True
        if child and child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def execute(job,mode):
        nonlocal child
        while not gpu_idle():
            if stop:return
            print(json.dumps(dict(event='waiting_for_GPU',job=job['id'])),flush=True);time.sleep(15)
        maintain_memory();DiskBudget(BASE,32).ensure_space(16*2**30 if mode=='train' else 256*2**20)
        run=Path(job['output']);run.mkdir(parents=True,exist_ok=True);_atomic_json(run/'job.json',job)
        logdir=ROOT/'logs/research/common_pool_v1';logdir.mkdir(parents=True,exist_ok=True)
        log=logdir/(job['id']+'__'+mode+'.log')
        with log.open('ab') as handle:
            child=subprocess.Popen([sys.executable,'-u','scripts/run_research_common_pool_v1.py','--job',str(run/'job.json'),'--mode',mode,'--allow-gpu'],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'),
                stdout=handle,stderr=subprocess.STDOUT)
            state.update(status='running',active_job=job['id'])
            state['jobs'].setdefault(job['id'],{}).update(status=mode,pid=child.pid,log=str(log),started_unix=time.time())
            persist();print(json.dumps(dict(event='worker_started',job=job['id'],mode=mode,pid=child.pid,log=str(log))),flush=True)
            while child.poll() is None:
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:maintain_memory()
                    except Exception as error:print(json.dumps(dict(event='memory_warning',error=str(error))),flush=True)
                    persist();print(json.dumps(dict(event='worker_running',job=job['id'],mode=mode)),flush=True)
            code=child.returncode;child=None
            if code:raise RuntimeError(f'{mode} exited {code}: {log}')
    failed_arms=set()
    try:
        persist()
        for job in plan['jobs']:
            if stop:break
            verify_plan(args.plan_sha)
            item=state['jobs'].setdefault(job['id'],{})
            if job['arm'] in failed_arms:
                item.update(status='blocked_by_predecessor');persist();continue
            if complete(job):
                retire_predecessor(job);item.update(status='completed');persist();continue
            for attempt in range(1,4):
                try:
                    run=Path(job['output'])
                    if not (run/'completed.json').exists():
                        mode='evaluate' if (run/'training_completed.json').exists() and (run/'last/state.pt').is_file() else 'train'
                        execute(job,mode)
                    if stop:break
                    execute(job,'validate')
                    if not complete(job):raise ValueError('Independent proof missing')
                    retire_predecessor(job)
                    item.update(status='completed',completed_unix=time.time());persist();break
                except Exception as error:
                    if stop:break
                    item.setdefault('failures',[]).append(dict(attempt=attempt,time=time.time(),error=str(error)))
                    item.update(status='retry_pending' if attempt<3 else 'requires_repair');persist()
                    print(json.dumps(dict(event='worker_failure',job=job['id'],attempt=attempt,error=str(error))),flush=True)
                    if attempt==3:failed_arms.add(job['arm'])
        state['status']='stopped' if stop else 'completed_with_errors' if failed_arms else 'completed'
        persist()
    finally:
        if child and child.poll() is None:child.terminate();child.wait(timeout=180)
        admission.close();lock.close()


if __name__=='__main__':main()
