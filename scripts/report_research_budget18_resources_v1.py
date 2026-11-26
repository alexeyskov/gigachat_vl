#!/usr/bin/env python3
"""Read-only18-cell census of native allocator peaks and sampled resources."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'configs/research/budget18_plan_v1.json'
PLAN_SHA='ec5333b8db9ec234d37e283d2299df0de3daaa4d0a1f4da0ad2a6601b9adb9cc'
MONITOR=ROOT/'logs/research/budget18_resources.jsonl'


def prefix_rows(path):
    """Bind a complete-line prefix; the append-only writer may keep running."""
    with Path(path).open('rb') as handle:
        before=os.fstat(handle.fileno())
        if before.st_size>256*2**20:
            raise ValueError('Resource census expects bounded local logs')
        raw=handle.read(before.st_size)
        after=os.fstat(handle.fileno())
        if (len(raw)!=before.st_size or after.st_size<before.st_size
                or after.st_dev!=before.st_dev or after.st_ino!=before.st_ino):
            raise ValueError('Log truncated or replaced during snapshot')
    complete=raw[:raw.rfind(b'\n')+1]
    binding={'path':str(Path(path).resolve()),'prefix_bytes':len(complete),
             'prefix_sha256':hashlib.sha256(complete).hexdigest(),
             'incomplete_trailing_bytes_excluded':len(raw)-len(complete)}
    return [json.loads(line) for line in complete.splitlines()],binding


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def collect():
    raw=PLAN.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PLAN_SHA:
        raise ValueError('Exact18-cell intent required')
    jobs=json.loads(raw)['jobs']
    if len(jobs)!=42 or sum(j['stage']=='SFT' for j in jobs)!=18:
        raise ValueError('Complete18-cell matrix required')
    samples,source=prefix_rows(MONITOR)
    result={j['id']:{'run_id':j['id'],'stage':j['stage'],'backend':j['backend'],
                    'encoder':j['encoder'],'recipe':j['recipe'],
                    'observed_phases':{},'training_worker_samples':0} for j in jobs}
    for sample in samples:
        if sample.get('owner')!='budget18_process_monitor_v1' or not finite(sample.get('time')):
            raise ValueError('Unknown or malformed resource observation')
        seen=set()
        for activity in sample.get('activities',[]):
            name=activity.get('run_id')
            if name not in result or activity.get('activity_verified') is not True or name in seen:
                continue
            seen.add(name)
            row=result[name];phase=activity.get('phase','unknown')
            row['observed_phases'][phase]=row['observed_phases'].get(phase,0)+1
            if phase!='train':
                continue
            row['training_worker_samples']+=1
            row.setdefault('first_training_worker_sample_unix',sample['time'])
            row['last_training_worker_sample_unix']=sample['time']
            measures={
                'sampled_cgroup_max_GiB':sample.get('ram',{}).get('cgroup',{}).get('current_bytes'),
                'sampled_min_free_disk_GiB':sample.get('disk',{}).get('free_bytes')}
            for key,value in measures.items():
                if finite(value):
                    value/=2**30
                    row[key]=(min if 'min_' in key else max)(row.get(key,value),value)
            for gpu in sample.get('gpus',[]):
                if gpu.get('index')!=0:
                    continue
                memory=gpu.get('memory.used')
                if finite(memory):
                    row['sampled_NVML_max_GiB']=max(row.get('sampled_NVML_max_GiB',0.),memory/1024)
                load=gpu.get('utilization.gpu')
                if finite(load):
                    row.setdefault('_GPU_utilization',[]).append(load)
    for name,row in result.items():
        utilization=sorted(row.pop('_GPU_utilization',[]))
        if utilization:
            n=len(utilization)
            row['sampled_GPU_utilization_median_percent']=(utilization[(n-1)//2]+utilization[n//2])/2
            row['sampled_GPU_utilization_min_percent']=utilization[0]
            row['sampled_GPU_utilization_max_percent']=utilization[-1]
        run=ROOT/'runs/research'/name
        if (run/'metrics.jsonl').exists():
            events,binding=prefix_rows(run/'metrics.jsonl')
            train=[e for e in events if e.get('event')=='train']
            for event in events:
                if event.get('event')=='validation' and event.get('records')==116 and finite(event.get('group_macro_score')):
                    row['internal116_group_macro_score']=event['group_macro_score']
                if event.get('event')=='text_validation' and event.get('text_only_records')==34 and finite(event.get('text_only_heldout_ce')):
                    row['text34_heldout_ce']=event['text_only_heldout_ce']
            if train:
                row['native_metrics']=binding
                row['native_torch_peak_allocated_GiB']=max(e['peak_vram_gb'] for e in train)
                row['observed_successful_updates']=max(e['step'] for e in train)
                row['train_updates_elapsed_seconds']=train[-1]['elapsed_seconds']
                row['final_observed_cache']=train[-1]['cache']
                row['final_observed_microbatch']=train[-1]['microbatch_size']
                if row.get('training_worker_samples'):
                    row['first_training_worker_sample_after_first_update_seconds']=row['first_training_worker_sample_unix']-train[0]['time']
                    row['early_training_sample_available']=row['first_training_worker_sample_after_first_update_seconds']<=60
            if (run/'completed.json').exists():
                completed_bytes=(run/'completed.json').read_bytes()
                done=json.loads(completed_bytes)
                row.update(stage_completed=True,completed_steps=done['steps'],completed_records=done['records'],stage_elapsed_seconds=done['elapsed_seconds'])
                row['completion_sha256']=hashlib.sha256(completed_bytes).hexdigest()
        row['sampled_resource_measurement_available']=bool(row['training_worker_samples'])
    return {'status':'read_only_resource_census','created_unix':time.time(),
            'rows':[row for row in result.values() if row['stage']=='SFT'],
            'stage_rows':list(result.values()),
            'source_monitor_snapshot':source,'plan_sha256':PLAN_SHA,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':{'Torch_peak':'native max_memory_allocated reported by the training worker',
                     'NVML_and_cgroup':'maximum among30second snapshots during verified train-worker activity; includes other instance processes and file cache',
                     'train_phase':'worker lifecycle, including loading/saving/endCE; separately labelled internal_selection excluded',
                     'coverage':'partial historical coverage and short unobserved stages remain explicit; sampled maxima are not continuous peaks',
                     'ongoing_runs':'values describe the bound complete-line prefix only'}}


def main():
    import sys
    sys.path.insert(0,str(ROOT))
    from src.utils.research_storage import _atomic_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'research_results/budget18_resource_census_v1/report.json')
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    report=collect();_atomic_json(args.output,report)
    print(json.dumps({'cells':len(report['rows']),'sampled':sum(r['sampled_resource_measurement_available'] for r in report['rows'])}))


if __name__=='__main__':main()
