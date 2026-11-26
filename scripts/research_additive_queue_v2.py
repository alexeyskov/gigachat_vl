#!/usr/bin/env python3
"""Fixed blocks only:12 QLoRA configurations plus6 full-training controls.

Reuse the exact fixed branch math, data, completed checkpoints and workers.
Scope has its own plan binding; never rewrite the original30-cell plan.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_additive_queue_v1 as native
from src.utils.research_storage import _atomic_json
PLAN=ROOT/'configs/research/additive_fixed_plan_v2.json'
STATE=native.STATE
PREVIOUS_SHA='e3e531f9d3b2de09fdece91786402a65b567afd4109ed9d31523aca486129339'


def build_plan():
    previous=ROOT/'configs/research/additive_visual_plan_v1.json'
    if native.backend.sha(previous)!=PREVIOUS_SHA:raise ValueError('Original plan changed')
    full=native.build_plan()
    if json.loads(previous.read_text())!=full:raise ValueError('Original sources/branch math changed')
    result=copy.deepcopy(full)
    result.update(name='additive_visual_fixed_only_v2',new_scientific_configurations=12,
        total_primary_configurations=18,fixed_only=True,
        supersedes=dict(plan=str(previous),sha256=PREVIOUS_SHA,user_instruction='fixed blocks only; routed experiments excluded'))
    result['jobs']=[j for j in result['jobs'] if j['mode']=='fixed']
    result['contracts']=[c for c in result['contracts'] if c['branch']['mode']=='fixed']
    result['source_sha256']['scripts/research_additive_queue_v2.py']=native.backend.sha(__file__)
    return result


def verify_plan(path,expected_sha):
    if Path(path).resolve()!=PLAN or native.backend.sha(path)!=expected_sha:raise ValueError('Fixed-only plan path/SHA changed')
    value=json.loads(Path(path).read_text())
    if value!=build_plan():raise ValueError('Fixed-only scope/source contract changed')
    return value


def migrated_state(plan,old,new_sha):
    ids={j['id'] for j in plan['jobs']}
    if old['plan_sha256'] not in (PREVIOUS_SHA,new_sha):raise ValueError('Unexpected source state')
    state=copy.deepcopy(old)
    state['jobs']={k:v for k,v in state['jobs'].items() if k in ids}
    for row in state['jobs'].values():
        row.pop('retry_after_unix',None)
        if row.get('status')!='completed':
            row['status']='pending';row.pop('error',None)
    if state.get('active_job') not in ids:state['active_job']=None
    state.update(plan_sha256=new_sha,current_revision='additive_visual_fixed_only_v2',
        active_plan_path=str(PLAN),status='waiting_fixed_bootstrap',routed_experiments_excluded=True)
    state.setdefault('scope_revisions',[]).append(dict(name='fixed_only_v2',previous_plan_sha256=old['plan_sha256'],
        new_plan_sha256=new_sha,reused_completed_fixed_stages=sum(r.get('status')=='completed' for r in state['jobs'].values()),time=time.time()))
    return state


class Queue(native.Queue):
    def persist(self):
        self.state.update(current_revision='additive_visual_fixed_only_v2',active_plan_path=str(PLAN),routed_experiments_excluded=True)
        return super().persist()


def wait_bootstrap():
    from scripts.research_budget18_queue_v1 import supervisor_info
    path=ROOT/'research_results/fixed_only_v2/bootstrap_state.json'
    while True:
        service=supervisor_info('research-fixed-bootstrap-v2')
        state=json.loads(path.read_text()) if path.exists() else {}
        if service['statename']=='EXITED' and service['exitstatus']==0 and state.get('status')=='completed':return
        if service['statename'] in ('FATAL','STOPPED') or service['statename']=='EXITED' and service['exitstatus']:
            raise RuntimeError('Fixed bootstrap did not complete successfully')
        print(json.dumps(dict(event='fixed_queue_waits_for_owned_fixed_training',bootstrap=state)),flush=True)
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-plan',action='store_true');p.add_argument('--migrate-state',action='store_true')
    p.add_argument('--plan-sha');p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.write_plan:
        value=build_plan()
        if PLAN.exists() and json.loads(PLAN.read_text())!=value:raise ValueError('Fixed-only plan already differs')
        _atomic_json(PLAN,value);print(native.backend.sha(PLAN));return
    plan=verify_plan(PLAN,args.plan_sha)
    if args.migrate_state:
        import fcntl
        from scripts.research_budget18_queue_v1 import supervisor_info
        if supervisor_info('research-additive-v1')['statename']!='STOPPED':raise ValueError('Previous CPU owner must be stopped before migration')
        with STATE.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            old=json.loads(STATE.read_text());new=migrated_state(plan,old,args.plan_sha)
            out=ROOT/'research_results/fixed_only_v2';out.mkdir(exist_ok=True,parents=True)
            if not (out/'state_before_migration.json').exists():(out/'state_before_migration.json').write_bytes(STATE.read_bytes())
            _atomic_json(STATE,new)
            print(json.dumps(dict(status='fixed_only_state_migrated',jobs=len(plan['jobs']),reused_stages=sum(r.get('status')=='completed' for r in new['jobs'].values()))))
        return
    if not args.run:print(json.dumps(dict(status='CPU_preflight_passed',new_configs=12,controls=6,total=18,stages=24)));return
    wait_bootstrap()
    native.PLAN=PLAN;native.verify_plan=verify_plan
    queue=Queue(plan,STATE,args.plan_sha)
    signal.signal(signal.SIGTERM,queue.signal);signal.signal(signal.SIGINT,queue.signal)
    return queue.run()


if __name__=='__main__':raise SystemExit(main())
