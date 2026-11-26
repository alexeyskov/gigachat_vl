#!/usr/bin/env python3
"""Source-bound R2 header compatibility, failure isolation and automatic retry.

Training keeps priority. A failed validation cannot strand independent jobs.
Completed checkpoints are adopted only through the original full replay gate.
"""
from __future__ import annotations
import argparse,copy,hashlib,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
REQUESTED_CVD=os.environ.get('CUDA_VISIBLE_DEVICES')
if '--worker-request' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
V6_SHA='4f233fd6ebfd54207c67cb54b057b7794defa75d0224f1fe42879cf744d75ac6'
if hashlib.sha256((ROOT/'scripts/research_budget18_queue_v6.py').read_bytes()).hexdigest()!=V6_SHA:raise ValueError('Frozen V6 source changed')
from scripts import research_budget18_queue_v6 as v6
v4=v6.v4;old=v6.old;v5=v6.v5
OWNER='research_budget18_training_first_v7'


def install():
    v6.install()
    v4.MODULES.update(lfm25='src.utils.research_lfm_payload_bridge_v1',giga31='src.utils.research_giga_payload_bridge_v1')
    v4.SCOPE['__file__']=str(Path(__file__).resolve());v4.OLD_SCOPE['__file__']=str(Path(__file__).resolve())


def retry_delay(count):return min(600,60*2**min(max(0,count-1),4))


class Queue(v6.Queue):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from src.utils.research_qwen_payload_bridge_v1 import definition
        receipt=dict(owner=OWNER,source_sha256=old.sha(__file__),parent_source_sha256=V6_SHA,
            derived_R2_read_bridge=definition(),failure_policy='isolate failed job; retry; continue independent training; verified external work fallback',
            native_checkpoint_gate_unchanged=True,scientific_cells=18,training_stages=42)
        if receipt not in self.state['controller_revisions']:self.state['controller_revisions'].append(receipt)
        self.persist()

    def execute(self,job,kind,argv,row):
        prior=row.get(kind+'_process')
        if prior and prior.get('status') in ('failed','completed','interrupted'):
            row.setdefault('process_execution_history',[]).append(copy.deepcopy(prior))
        return super().execute(job,kind,argv,row)

    def finalize_training(self,job,row):
        if not v5.verified_replay(job,row):
            report=old.directory(job)/'checkpoint_roundtrip.json'
            if report.exists() and old.read(report).get('status')=='passed':
                identity=self.oracle._checkpoint_validation_identity(job)
                row['checkpoint_validation']=self.oracle._validate_checkpoint_report(identity)
                row['replay_adopted_from_existing_native_report']=True;self.persist()
        return super().finalize_training(job,row)

    def failed(self,job,row,error,phase):
        attempts=row.setdefault('automatic_retry_history',[])
        item=dict(time=time.time(),phase=phase,error_type=type(error).__name__,error=str(error))
        attempts.append(item)
        row.update(status='failed_retry_pending',error_type=item['error_type'],error=item['error'],
                   retry_after_unix=time.time()+retry_delay(len(attempts)))
        self.state['last_failure']=dict(job=job['id'],**item)
        self.persist();old.emit('job_failed_independent_jobs_continue',job=job['id'],failure=item)

    def run(self):
        self.state.update(status='running',scheduling_phase='training_first');self.persist()
        try:
            while not self.stopped:
                progress=False
                for job in self.spec['jobs']:
                    if self.stopped:break
                    row=self.state['jobs'].setdefault(job['id'],{})
                    if row.get('checkpoint_pruned'):self.check_pruned(job,row);continue
                    if any(not self.state['jobs'].get(p,{}).get('training_completed') for p in job['requires']):continue
                    if row.get('training_completed') and (job['stage']!='SFT' or v5.verified_replay(job,row)):continue
                    if row.get('retry_after_unix',0)>time.time():continue
                    # Global provenance/resource guards are never bypassed by
                    # failure isolation; verify them before the per-job block.
                    v4.verify_plan(self.spec);v4.resources()
                    self.state.update(active_job=job['id'],scheduling_phase='training_first',status='running');self.persist()
                    try:
                        if row.get('training_completed'):row['training_proof']=v4.validate_completed(job)
                        else:self.adopt_or_train(job,row)
                        if job['stage']=='SFT':self.finalize_training(job,row)
                        else:row['status']='completed';self.persist()
                        row.pop('error',None);row.pop('error_type',None);row.pop('retry_after_unix',None)
                        self.prune();self.persist();progress=True
                    except InterruptedError:
                        if self.stopped:break
                        raise
                    except Exception as error:self.failed(job,row,error,'training_or_checkpoint_validation')
                if self.stopped:break
                if progress:continue
                pending=[j for j in self.spec['jobs'] if not self.state['jobs'].get(j['id'],{}).get('training_completed')
                    or (j['stage']=='SFT' and not v5.verified_replay(j,self.state['jobs'][j['id']]))]
                # Useful fallback when a failed branch is cooling down: never
                # skip its gate, but allow already verified models to be tested.
                evaluated=False
                for job in self.spec['jobs']:
                    row=self.state['jobs'][job['id']]
                    if job['stage']!='SFT' or row.get('status')=='completed' or not row.get('training_completed') or not v5.verified_replay(job,row):continue
                    if row.get('retry_after_unix',0)>time.time():continue
                    self.state.update(active_job=job['id'],scheduling_phase='verified_evaluation_fallback' if pending else 'full_external_evaluation');self.persist()
                    try:self.postprocess(job,row);row['status']='completed';row.pop('retry_after_unix',None);self.persist();evaluated=True
                    except InterruptedError:
                        if self.stopped:break
                        raise
                    except Exception as error:self.failed(job,row,error,'external_evaluation')
                    break  # Recheck training admission after each full model.
                if self.stopped:break
                if evaluated:continue
                unfinished=[j for j in self.spec['jobs'] if j['stage']=='SFT' and self.state['jobs'][j['id']].get('status')!='completed']
                if not pending and not unfinished:
                    self.state['status']='completed';self.persist();return 0
                self.state.update(status='waiting_for_retry',pending_training_jobs=[j['id'] for j in pending]);self.persist()
                old.emit('queue_retry_wait',pending_training=len(pending),pending_external=len(unfinished),seconds=30)
                for _ in range(30):
                    if self.stopped:break
                    time.sleep(1)
            self.state['status']='interrupted';self.persist();return 0
        except BaseException as error:
            self.state.update(status='failed',controller_error_type=type(error).__name__,controller_error=str(error));self.persist();raise
        finally:
            if self.child and self.child.poll() is None and old.same(self.child_info,old.proc(self.child.pid)):
                self.child.terminate()
                try:self.child.wait(timeout=30)
                except old.subprocess.TimeoutExpired:self.child.kill();self.child.wait()
            self.handle.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,default=old.PLAN);p.add_argument('--state',type=Path,default=ROOT/'runs/research/budget18_queue_state_v1.json')
    p.add_argument('--resource-ready',type=Path);p.add_argument('--resource-ready-sha');p.add_argument('--run-queue',action='store_true')
    p.add_argument('--retry-failed',action='store_true');p.add_argument('--worker-request',type=Path);p.add_argument('--request-sha256')
    args=p.parse_args();install()
    if args.worker_request:
        if REQUESTED_CVD!='0':raise ValueError('Owned GPU worker requires GPU0')
        os.environ['CUDA_VISIBLE_DEVICES']='0';return v4.worker(args.worker_request,args)
    if not args.resource_ready or not args.resource_ready_sha:p.error('Explicit source-bound readiness required')
    binding=dict(path=str(args.resource_ready.resolve(strict=True)),sha256=args.resource_ready_sha)
    v4._RESOURCE=binding;v4.verify_plan(old.read(args.spec))
    if not args.run_queue:print('Validated V7 metadata:18cells/42stages; no GPU launch');return 0
    queue=Queue(args.spec,args.state,args.retry_failed,binding)
    signal.signal(signal.SIGTERM,queue.signal);signal.signal(signal.SIGINT,queue.signal)
    return queue.run()


if __name__=='__main__':raise SystemExit(main())
