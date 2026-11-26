#!/usr/bin/env python3
"""Replace only our CPU controller after its owned GPU worker naturally exits."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_budget18_queue_v1 as base
OUT=ROOT/'research_results/budget18_controller_v2_handoff'
RESOURCE=ROOT/'configs/research/budget18_resource_v3_ready.json'
RESOURCE_SHA='95f9df2858b14eca039a9c19b65a6ca3d38f150645dafe606e954869f6f8f444'


def worker_children(parent):
    children=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        p=base.proc(int(entry.name))
        if not p or p['ppid']!=parent['pid']:continue
        argv=p['argv']
        if not any(Path(a).name=='research_budget18_queue_v1.py' for a in argv):continue
        request=base.value(argv,'--worker-request')
        if not request:continue
        path=Path(request).resolve(strict=True)
        if not path.is_relative_to(ROOT/'logs/research/budget18_workers'):raise ValueError('Foreign worker request')
        req=base.read(path)
        if (base.sha(path)!=base.value(argv,'--request-sha256')
                or any(req['parent'].get(k)!=parent[k] for k in ('pid','birth','argv'))
                or req.get('controller_source_sha256')!=base.sha(ROOT/'scripts/research_budget18_queue_v1.py')):
            raise ValueError('Worker is not owned by the exact old parent')
        children.append((p,req))
    if len(children)>1:raise ValueError('Multiple owned GPU workers')
    return children


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-pid',type=int,required=True)
    parser.add_argument('--parent-birth',type=int,required=True)
    parser.add_argument('--new-source-sha',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if base.sha(RESOURCE)!=RESOURCE_SHA or base.sha(ROOT/'scripts/research_budget18_queue_v2.py')!=args.new_source_sha:
        raise ValueError('Approved new resource/controller bytes changed')
    parent=base.proc(args.parent_pid)
    if not parent or parent['birth']!=args.parent_birth or not any(Path(a).name=='research_budget18_queue_v1.py' for a in parent['argv']):
        raise ValueError('Old controller process identity differs')
    service=base.supervisor_info('research-budget18')
    target=base.supervisor_info('research-budget18-v2')
    if service['statename']!='RUNNING' or service['pid']!=parent['pid'] or target['statename']!='STOPPED':
        raise ValueError('Only the exact running old and stopped new services may hand off')
    children=worker_children(parent)
    if not args.execute:
        print(json.dumps({'status':'preflight_no_signals','parent':parent,'children':children}));return
    if OUT.exists():raise ValueError('Existing handoff receipt needs explicit recovery')
    OUT.mkdir();base.atomic(OUT/'intent.json',{'parent':parent,'new_controller_sha256':args.new_source_sha,
                'resource_ready_sha256':RESOURCE_SHA,'new_service':'research-budget18-v2',
                'GPU_workers_will_not_be_signaled':True,'source_sha256':base.sha(__file__)})
    paused=False;resume_on_error=True
    def interrupted(_number,_frame):
        raise InterruptedError('Handoff observer interrupted; restore its stopped CPU owner')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        # If an observer races a process exit, resnapshot AFTER stopping the CPU
        # owner. Its GPU child stays running; the owner cannot spawn another one.
        while True:
            if not base.same(parent,base.proc(parent['pid'])):raise RuntimeError('Old parent changed')
            if worker_children(parent):
                os.kill(parent['pid'],signal.SIGSTOP);paused=True
                children=worker_children(parent)
                if children:break
                os.kill(parent['pid'],signal.SIGCONT);paused=False
            time.sleep(.2)
        child,request=children[0]
        base.atomic(OUT/'waiting_worker.json',{'worker':child,'request':request,'parent_stopped_only':True})
        last=0
        while base.same(child,base.proc(child['pid'])):
            if not base.same(parent,base.proc(parent['pid'])):raise RuntimeError('Stopped parent disappeared')
            if time.monotonic()-last>=30:
                print(json.dumps({'event':'waiting_for_natural_GPU_worker_exit','pid':child['pid'],'kind':request['kind']}),flush=True);last=time.monotonic()
            time.sleep(1)
        base.atomic(OUT/'worker_terminal.json',{'worker':child,'observed':base.proc(child['pid']),'worker_signals_sent':0})
        # Queue TERM while the parent is stopped; pending TERM is delivered on
        # CONT before it can launch another worker. The finished child is reaped.
        os.kill(parent['pid'],signal.SIGTERM)
        os.kill(parent['pid'],signal.SIGCONT);paused=False;resume_on_error=False
        deadline=time.monotonic()+60
        while base.same(parent,base.proc(parent['pid'])):
            if time.monotonic()>deadline:raise RuntimeError('Old CPU owner did not exit')
            time.sleep(.2)
        state=base.read(ROOT/'runs/research/budget18_queue_state_v1.json')
        matches=[r for row in state['jobs'].values() for k,r in row.items()
                 if k.endswith('_process') and isinstance(r,dict) and r.get('pid')==child['pid']]
        if len(matches)!=1 or matches[0].get('returncode')!=0:
            raise RuntimeError('Natural worker did not complete successfully; manual diagnosis required')
        if base.sha(ROOT/'scripts/research_budget18_queue_v2.py')!=args.new_source_sha or base.sha(RESOURCE)!=RESOURCE_SHA:
            raise ValueError('New source changed during boundary wait')
        subprocess.run(['supervisorctl','start','research-budget18-v2'],check=True)
        base.atomic(OUT/'completed.json',{'status':'completed','old_parent':parent,'naturally_finished_worker':child,
                    'worker_returncode':0,'GPU_worker_signals_sent':0,'new_service':base.supervisor_info('research-budget18-v2'),
                    'new_controller_sha256':args.new_source_sha,'resource_ready_sha256':RESOURCE_SHA})
    finally:
        if paused and resume_on_error and base.same(parent,base.proc(parent['pid'])):
            os.kill(parent['pid'],signal.SIGCONT)


if __name__=='__main__':main()
