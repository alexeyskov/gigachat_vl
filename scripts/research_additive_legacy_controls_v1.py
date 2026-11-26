#!/usr/bin/env python3
"""Complete only retained native full-finetune controls after additive training.

The sealed legacy plan/state are not relabelled. Unrun common-pool R3 jobs stay
historical pending/superseded, explicitly outside the new primary matrix.
"""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
REQUESTED_CVD='0' if '--worker-request' in sys.argv else ''
from scripts import research_budget18_queue_v7 as v7
v4=v7.v4;old=v7.old

def install():
    v7.install()
    v4.SCOPE['__file__']=str(Path(__file__).resolve());v4.OLD_SCOPE['__file__']=str(Path(__file__).resolve())

class Controls(v7.Queue):
    def run(self):
        targets=[j for j in self.spec['jobs'] if j['recipe']=='4f']
        self.state.update(status='running',scheduling_phase='additive_revision_native_full_controls_only');self.persist()
        try:
            while not self.stopped:
                for job in targets:
                    row=self.state['jobs'].setdefault(job['id'],{})
                    if row.get('training_completed') and (job['stage']!='SFT' or v7.v5.verified_replay(job,row)):continue
                    if any(not self.state['jobs'].get(p,{}).get('training_completed') for p in job['requires']):continue
                    if row.get('retry_after_unix',0)>time.time():continue
                    v4.verify_plan(self.spec);v4.resources()
                    self.state['active_job']=job['id'];self.persist()
                    try:
                        if not row.get('training_completed'):self.adopt_or_train(job,row)
                        if job['stage']=='SFT':self.finalize_training(job,row)
                        else:row['status']='completed'
                        self.prune();self.persist()
                    except Exception as error:
                        if self.stopped:break
                        self.failed(job,row,error,'full_control_training_or_native_replay')
                if all(self.state['jobs'].get(j['id'],{}).get('training_completed') and
                       (j['stage']!='SFT' or v7.v5.verified_replay(j,self.state['jobs'][j['id']])) for j in targets):
                    self.state.update(status='legacy_controls_complete_other_legacy_work_superseded',active_job=None);self.persist();return 0
                if not self.stopped:time.sleep(30)
            return 0
        finally:
            if self.child and self.child.poll() is None:self.child.terminate();self.child.wait(timeout=180)
            self.handle.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--worker-request',type=Path);p.add_argument('--request-sha256')
    args=p.parse_args();install()
    ready=ROOT/'configs/research/budget18_resource_v7_ready.json'
    binding=dict(path=str(ready),sha256=old.sha(ready));v4._RESOURCE=binding
    if args.worker_request:
        os.environ['CUDA_VISIBLE_DEVICES']='0';return v4.worker(args.worker_request,args)
    v4.verify_plan(old.read(old.PLAN))
    if not args.run:print('Native full control tail preflight passed');return
    queue=Controls(old.PLAN,ROOT/'runs/research/budget18_queue_state_v1.json',True,binding)
    signal.signal(signal.SIGTERM,queue.signal);signal.signal(signal.SIGINT,queue.signal)
    return queue.run()

if __name__=='__main__':raise SystemExit(main())
