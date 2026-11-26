#!/usr/bin/env python3
"""User-authorized retirement of superseded checkpoint objects, never live runs.

Keep the new additive matrix, its six full-training controls, the six legacy
projector sources still opened by the runner, and full-training A0 sources.
Retain lightweight metrics/configs and receipts; record every removed object.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager,ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.research_storage import _atomic_json,BestLastCheckpointManager,DiskBudget
PLAN=ROOT/'configs/research/additive_visual_plan_v1.json'
OUT=ROOT/'research_results/storage_retirement_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def stat(path):
    s=Path(path).lstat()
    return {n:getattr(s,n) for n in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_nlink','st_blocks')}


def protected_names(plan):
    names={Path(c['run']).name:'current full-training control' for c in plan['controls']}
    for model in ('smallthinker4','lfm25','giga31'):
        for vision in ('siglip2','qwen25'):
            names[f'{model}__{vision}__1b__SFT__budget18_s42']='projector source still required by native additive runner'
            names[f'{model}__{vision}__A0_bf16__budget18_s42']='full-training alignment source'
    names['additive_visual_v1']='current scientific series and projector artifacts'
    return names


def files(directory):
    result={}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('Symlinks inside checkpoint objects are refused')
        if path.is_file():result[str(path.relative_to(directory))]=stat(path)
        elif not path.is_dir():raise ValueError('Only ordinary checkpoint files/directories may be retired')
    return result


def inspect_run(run):
    if run.is_symlink() or run.resolve().parent!=run.parent.resolve():raise ValueError('Run must be a local ordinary directory')
    for name in ('resolved_config.json','.checkpoint_manager.json'):
        if not (run/name).is_file() or (run/name).is_symlink():raise ValueError('Owned local run metadata required')
    config=read(run/'resolved_config.json');owner=read(run/'.checkpoint_manager.json')['owner']
    configured=Path(config['output'])
    configured=configured if configured.is_absolute() else ROOT/configured
    if configured.resolve()!=run.resolve():raise ValueError('Run output/config binding differs')
    store=run/'.checkpoint_objects'
    if store.is_symlink() or not store.is_dir():raise ValueError('Local checkpoint object store required')
    objects=[]
    for path in sorted(store.iterdir()):
        marker=path/'.research_checkpoint.json'
        if path.is_symlink() or not path.is_dir() or not marker.is_file() or marker.is_symlink():raise ValueError('Unmanaged object in store')
        if read(marker).get('owner')!=owner:raise ValueError('Foreign checkpoint owner')
        objects.append(dict(path=str(path),files=files(path),marker_sha256=sha(marker)))
    for role in ('best','last'):
        p=run/role
        if os.path.lexists(p):
            if not p.is_symlink() or p.resolve(strict=True) not in [Path(o['path']) for o in objects]:
                raise ValueError('Role pointer escapes its owned object store')
    return dict(run=str(run),owner=owner,objects=objects,
        config_sha256=sha(run/'resolved_config.json'),manager_sha256=sha(run/'.checkpoint_manager.json'),
        logical_bytes=sum(s['st_size'] for o in objects for s in o['files'].values()),
        allocated_bytes=sum(s['st_blocks']*512 for o in objects for s in o['files'].values()))


def discover(root,plan):
    runs=root/'runs/research';protected=protected_names(plan);ids=set()
    for name in protected:
        store=runs/name/'.checkpoint_objects'
        if store.is_dir():
            for p in store.rglob('*'):
                if p.is_file() and not p.is_symlink():s=p.stat();ids.add((s.st_dev,s.st_ino))
    selected=[];skipped=[]
    for run in sorted(runs.iterdir()):
        if not run.is_dir() or run.is_symlink() or run.name in protected:continue
        store=run/'.checkpoint_objects'
        if not store.is_dir() or not any(store.iterdir()):continue
        row=inspect_run(run)
        if any((s['st_dev'],s['st_ino']) in ids for o in row['objects'] for s in o['files'].values()):
            skipped.append(dict(run=str(run),reason='inode shared with a protected checkpoint'));continue
        selected.append(row)
    return dict(schema_version=1,scope='explicit user retirement of superseded experiments',
        latest_plan_sha256=sha(root/'configs/research/additive_visual_plan_v1.json'),
        protected_runs=protected,runs=selected,skipped=skipped,
        logical_bytes=sum(r['logical_bytes'] for r in selected),
        allocated_bytes=sum(r['allocated_bytes'] for r in selected))


def revalidate(row):
    if inspect_run(Path(row['run']))!=row:raise ValueError('Candidate checkpoint changed since preview')


def retire_one(row,receipt_path,scanner):
    run=Path(row['run']);manager=BestLastCheckpointManager(run,DiskBudget(run,0),mode=read(run/'.checkpoint_manager.json')['mode'])
    with manager._lock():
        revalidate(row)
        identities={(s['st_dev'],s['st_ino']) for o in row['objects'] for s in o['files'].values()}
        scan=scanner(identities)
        receipt=dict(schema_version=1,owner=row['owner'],status='pruning',source_run=str(run),
            scope='user_authorized_superseded_experiment_weights',consumer_runs=[],
            logical_bytes_removed=row['logical_bytes'],objects=row['objects'],scan=scan,
            maintenance_receipt=str(receipt_path),started_unix=time.time())
        _atomic_json(run/'checkpoint_pruned.json',receipt)
        for role in ('best','last'):(run/role).unlink(missing_ok=True)
        for obj in row['objects']:shutil.rmtree(obj['path'])
        receipt.update(status='pruned',finished_unix=time.time());_atomic_json(run/'checkpoint_pruned.json',receipt)
        return dict(run=str(run),logical_bytes_removed=row['logical_bytes'],receipt_sha256=sha(run/'checkpoint_pruned.json'))


@contextmanager
def admission_maintenance(directory):
    """Briefly lease only the CPU queue owner; its GPU worker keeps running."""
    from scripts import research_budget18_queue_v1 as q
    info=q.supervisor_info('research-additive-v1');parent=q.proc(info['pid'])
    if (info['statename']!='RUNNING' or not parent or parent['state']=='T'
        or not any(p in parent['argv'] for p in ('scripts/research_additive_queue_v1.py','scripts/research_additive_queue_v2.py','scripts/research_additive_queue_v3.py','scripts/research_capacity_queue_v4.py','scripts/research_capacity_queue_v5.py'))):
        raise ValueError('Exact running additive CPU owner required')
    state=read(ROOT/'runs/research/additive_visual_v1/queue_state.json')
    if state['status']!='running':raise ValueError('Maintenance is restricted to the additive training phase')
    lease=dict(active=True,controller=parent,coordinator=q.proc(os.getpid()),GPU_worker_signals_sent=0,time=time.time())
    path=directory/'admission_lease.json';_atomic_json(path,lease)
    guard=subprocess.Popen([sys.executable,'scripts/run_qwen25_native_baseline_parallel_v1.py','--guardian',str(path)],
        cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
    paused=False
    try:
        os.kill(parent['pid'],signal.SIGSTOP);paused=True
        if not q.same(parent,q.proc(parent['pid'])):raise ValueError('CPU owner identity changed')
        yield
    finally:
        if paused and q.same(parent,q.proc(parent['pid'])):os.kill(parent['pid'],signal.SIGCONT)
        _atomic_json(path,dict(lease,active=False,released_unix=time.time()))
        try:guard.wait(timeout=5)
        except subprocess.TimeoutExpired:guard.terminate();guard.wait(timeout=5)


def apply(plan,path,*,plan_path=None,lease_factory=admission_maintenance):
    from scripts.research_archival_stock_queue_v5 import scan_anchor_users
    if sha(PLAN)!=plan['latest_plan_sha256']:raise ValueError('Latest scientific plan changed')
    path.parent.mkdir(parents=True,exist_ok=True)
    status=dict(status='prepared',plan_sha256=sha(plan_path or OUT/'plan.json'),before=DiskBudget(ROOT,32).report(),retired=[])
    _atomic_json(path,status)
    for row in plan['runs']:revalidate(row)
    with lease_factory(path.parent):
        legacy=ROOT/'runs/research/budget18_queue_state_v1.json'
        original=legacy.read_bytes();(path.parent/'legacy_queue_state_before.json').write_bytes(original)
        state=json.loads(original);expected=hashlib.sha256(original).hexdigest()
        for row in plan['runs']:
            status['retired'].append(retire_one(row,path,scan_anchor_users))
            name=Path(row['run']).name
            if name in state['jobs']:
                if sha(legacy)!=expected:raise ValueError('Legacy metadata writer changed during leased maintenance')
                state['jobs'][name].update(checkpoint_pruned=True,
                    superseded_weights_retired=True,retirement_receipt=str(Path(row['run'])/'checkpoint_pruned.json'),
                    status='superseded_weights_retired')
                _atomic_json(legacy,state);expected=sha(legacy)
            status.update(status='retiring',last_retired=name);_atomic_json(path,status)
        status.update(status='completed',after=DiskBudget(ROOT,32).report(),GPU_worker_signals_sent=0,
            logical_bytes_removed=sum(r['logical_bytes_removed'] for r in status['retired']),
            legacy_queue_state_after_sha256=sha(legacy),finished_unix=time.time())
        _atomic_json(path,status)
    return status


def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');p.add_argument('--plan-sha');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.apply:
        value=discover(ROOT,read(PLAN));_atomic_json(OUT/'plan.json',value)
        print(json.dumps(dict(runs=len(value['runs']),GiB=value['logical_bytes']/2**30,plan_sha256=sha(OUT/'plan.json'))));return
    if not args.plan_sha or sha(OUT/'plan.json')!=args.plan_sha:raise ValueError('Explicit preview SHA required')
    result=apply(read(OUT/'plan.json'),OUT/'completed.json')
    print(json.dumps(dict(status=result['status'],runs=len(result['retired']),removed_GiB=result['logical_bytes_removed']/2**30,
        free_GiB=result['after']['free_gib'],GPU_worker_signals_sent=0)),flush=True)


if __name__=='__main__':main()
