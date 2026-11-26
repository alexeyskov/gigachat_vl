#!/usr/bin/env python3
"""Fixed-only queue with exact packed-LoRA A1/SFT initialization compatibility."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_additive_queue_v2 as previous
native=previous.native
from src.utils.research_storage import _atomic_json
from src.utils.research_additive_init_dtype_v1 import definition as bridge_definition
PLAN=ROOT/'configs/research/additive_fixed_plan_v3.json'
STATE=previous.STATE
OLD_SHA='896d2fd940b27b5f15cebb4672966c915b3d23016ae943cc8b86f17935688453'
ENTRY='scripts/run_research_additive_init_dtype_v1.py'


def build_plan():
    old=previous.verify_plan(previous.PLAN,OLD_SHA)
    value=copy.deepcopy(old)
    value.update(name='additive_visual_fixed_only_init_bridge_v3',initialization_read_bridge=bridge_definition(),
        supersedes=dict(plan=str(previous.PLAN),sha256=OLD_SHA,reason='allow exact frozen BF16 packed LoRA to FP32 trainable initialization; no checkpoint mutation'))
    for job in value['jobs']:
        if job['stage']=='SFT' and job['backend'] in ('lfm25','giga31'):job['train_entry']=ENTRY
    value['source_sha256'].update({p:native.backend.sha(ROOT/p) for p in
        ('scripts/research_additive_queue_v3.py',ENTRY,'src/utils/research_additive_init_dtype_v1.py')})
    return value


def verify_plan(path,digest):
    if Path(path).resolve()!=PLAN or native.backend.sha(path)!=digest:raise ValueError('V3 plan path/SHA changed')
    value=json.loads(Path(path).read_text())
    if value!=build_plan():raise ValueError('V3 source/initialization compatibility binding changed')
    return value


def actual_command(job,argv,kind):
    result=list(argv)
    if kind=='train' and job.get('train_entry'):
        if result[0]!='scripts/run_research_additive_v1.py':raise ValueError('Unexpected native training entry')
        result[0]=job['train_entry']
    return result


class Queue(previous.Queue):
    def persist(self):
        self.state.update(current_revision='additive_visual_fixed_init_bridge_v3',active_plan_path=str(PLAN),routed_experiments_excluded=True)
        return native.Queue.persist(self)
    def execute(self,job,argv,kind):
        return native.Queue.execute(self,job,actual_command(job,argv,kind),kind)


def wait_repair():
    from scripts.research_budget18_queue_v1 import supervisor_info
    run=ROOT/'runs/research/additive_visual_v1/lfm25__siglip2__fixed__R1__SFT'
    while True:
        info=supervisor_info('research-fixed-init-repair-v1')
        if info['statename']=='EXITED' and info['exitstatus']==0 and (run/'completed.json').exists():return
        if info['statename'] in ('STOPPED','FATAL') or info['statename']=='EXITED' and info['exitstatus']:
            raise RuntimeError('First repaired SFT did not complete; preserve failure evidence')
        print(json.dumps(dict(event='waiting_for_actual_repaired_SFT',run=run.name)),flush=True)
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-plan',action='store_true');p.add_argument('--migrate-state',action='store_true')
    p.add_argument('--plan-sha');p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.write_plan:
        value=build_plan()
        if PLAN.exists() and json.loads(PLAN.read_text())!=value:raise ValueError('Existing V3 plan differs')
        _atomic_json(PLAN,value);print(native.backend.sha(PLAN));return
    plan=verify_plan(PLAN,args.plan_sha)
    if args.migrate_state:
        from scripts.research_budget18_queue_v1 import supervisor_info
        if supervisor_info('research-additive-v1')['statename']!='STOPPED':raise ValueError('Previous CPU owner must be stopped')
        with STATE.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            before=STATE.read_bytes();state=json.loads(before)
            if state['plan_sha256']!=OLD_SHA:raise ValueError('Exact fixed-only V2 state required')
            out=ROOT/'research_results/fixed_init_bridge_v3';out.mkdir(exist_ok=True,parents=True)
            (out/'state_before.json').write_bytes(before)
            state.update(plan_sha256=args.plan_sha,active_plan_path=str(PLAN),current_revision='additive_visual_fixed_init_bridge_v3',
                status='waiting_actual_repaired_SFT',active_job='lfm25__siglip2__fixed__R1__SFT',last_progress=None)
            for row in state['jobs'].values():
                if row.get('status')=='retry_pending':
                    row.update(status='pending');row.pop('retry_after_unix',None);row.pop('error',None)
            state.setdefault('scope_revisions',[]).append(dict(name='exact_packed_init_read_bridge_v3',time=time.time(),previous_plan_sha256=OLD_SHA,new_plan_sha256=args.plan_sha))
            _atomic_json(STATE,state);print('Migrated V3; completed A1 and SmallThinker SFT preserved')
        return
    if not args.run:print(json.dumps(dict(status='CPU_preflight_passed',scope='same12fixed+6controls',packed_SFT_bridge_jobs=8)));return
    wait_repair();native.PLAN=PLAN;native.verify_plan=verify_plan
    queue=Queue(plan,STATE,args.plan_sha)
    signal.signal(signal.SIGTERM,queue.signal);signal.signal(signal.SIGINT,queue.signal)
    return queue.run()


if __name__=='__main__':raise SystemExit(main())
