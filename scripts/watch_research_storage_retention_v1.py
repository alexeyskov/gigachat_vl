#!/usr/bin/env python3
"""Retire the six legacy projector-source checkpoints once all new A1s finish."""
import json
import os
from pathlib import Path
import sys
import time
import fcntl
from contextlib import contextmanager
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import retire_superseded_research_v1 as cleanup
from src.utils.research_storage import _atomic_json
OUT=ROOT/'research_results/storage_retirement_v1/projector_sources_after_consumption'


def all_alignment_consumed(plan,state):
    jobs=[j for j in plan['jobs'] if j['stage']=='A1']
    expected=plan.get('new_scientific_configurations',24)
    return expected in (12,24) and len(jobs)==expected and all(state.get('jobs',{}).get(j['id'],{}).get('status')=='completed' for j in jobs)


@contextmanager
def stopped_owner_maintenance(_directory):
    from scripts.research_budget18_queue_v1 import supervisor_info
    if supervisor_info('research-additive-v1')['statename'] not in ('EXITED','STOPPED'):
        raise BlockingIOError('Training/control queue is still active')
    with (ROOT/'runs/research/budget18_queue_state_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completed.json').exists() and cleanup.read(OUT/'completed.json').get('status')=='completed':return
    from scripts.research_budget18_queue_v1 import supervisor_info
    while True:
        path=ROOT/'runs/research/additive_visual_v1/queue_state.json'
        state=cleanup.read(path)
        cleanup.PLAN=Path(state.get('active_plan_path',ROOT/'configs/research/additive_visual_plan_v1.json'))
        plan=cleanup.read(cleanup.PLAN)
        done=sum(j['stage']=='A1' and state.get('jobs',{}).get(j['id'],{}).get('status')=='completed' for j in plan['jobs'])
        info=supervisor_info('research-additive-v1')
        eligible=all_alignment_consumed(plan,state)
        active=info['statename']=='RUNNING' and state.get('status')=='running'
        idle=info['statename'] in ('EXITED','STOPPED')
        if eligible and (active or idle):break
        print(json.dumps(dict(event='projector_source_retention',new_A1_completed=done,total=plan['new_scientific_configurations'],
            old_sources_retained_until_consumed=True)),flush=True)
        time.sleep(30)
    from src.utils.research_additive_runner_v1 import MODELS,VISIONS,extract_projector,sha
    snapshot=path.read_bytes();(OUT/'new_queue_all_A1_completed.json').write_bytes(snapshot)
    if not all_alignment_consumed(plan,json.loads(snapshot)):raise ValueError('A1 completion changed')
    sources=[];artifacts=[]
    for model in MODELS:
        for vision in VISIONS:
            artifact=ROOT/f'runs/research/additive_visual_v1/projectors/{model}__{vision}.pt'
            source=ROOT/f'runs/research/{model}__{vision}__1b__SFT__budget18_s42'
            if (source/'last/state.pt').exists():
                # Before deleting the large source, recheck exact cached P and
                # all original frozen-projector provenance through the runner.
                extract_projector(model,vision,artifact)
                sources.append(cleanup.inspect_run(source))
            if not artifact.is_file():raise ValueError('Mandatory retained projector artifact missing')
            artifacts.append(dict(path=str(artifact),sha256=sha(artifact),bytes=artifact.stat().st_size))
    value=dict(schema_version=1,scope='legacy projector-source weights consumed by all active new alignment stages',
        latest_plan_sha256=sha(cleanup.PLAN),runs=sources,
        completion_snapshot_sha256=sha(OUT/'new_queue_all_A1_completed.json'),retained_projector_artifacts=artifacts,
        logical_bytes=sum(s['logical_bytes'] for s in sources))
    _atomic_json(OUT/'plan.json',value)
    lease=cleanup.admission_maintenance if active else stopped_owner_maintenance
    result=cleanup.apply(value,OUT/'completed.json',plan_path=OUT/'plan.json',lease_factory=lease)
    print(json.dumps(dict(event='consumed_legacy_sources_retired',GiB=result['logical_bytes_removed']/2**30,
        retained_projectors=6,GPU_worker_signals_sent=0)),flush=True)


if __name__=='__main__':main()
