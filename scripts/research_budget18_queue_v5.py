#!/usr/bin/env python3
"""Train all42 stages first; preserve all18 full evaluations as a second pass.

Reuses the exact V4 GPU worker, resource admission and numerical backends.
Completed SFTs remain evaluation-pending until all four512 views pass.
"""
from __future__ import annotations
import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
V4_SHA='c75e29bbd4b8e54dc2dba90408c074dc60a4fd1c3408d92357e8a76b225109f5'
if hashlib.sha256((ROOT/'scripts/research_budget18_queue_v4.py').read_bytes()).hexdigest()!=V4_SHA:raise ValueError('Frozen V4 changed')
from scripts import research_budget18_queue_v4 as v4
old=v4.old
OWNER='research_budget18_training_first_v5'


def verified_replay(job,row):
    proof=row.get('checkpoint_validation',{})
    if proof.get('status')!='completed':return False
    path=Path(proof['report']);report=old.read(path)
    if (old.sha(path)!=proof['report_sha256'] or report.get('status')!='passed'
            or report.get('mode')!='full_gpu_replay' or report.get('run')!=str(old.directory(job))):
        raise ValueError('Retained full GPU replay proof changed')
    if not report.get('checkpoints'):raise ValueError('Full replay has no physical checkpoints')
    paths={Path(e['checkpoint_path']).resolve(strict=True) for e in report['checkpoints']}
    if any((old.directory(job)/role/'state.pt').resolve(strict=True) not in paths for role in ('best','last')):
        raise ValueError('Current roles do not match replayed objects')
    for entry in report['checkpoints']:
        p=Path(entry['checkpoint_path']);s=p.stat()
        if s.st_size!=entry['checkpoint_size_bytes'] or s.st_mtime_ns!=entry['checkpoint_mtime_ns']:
            raise ValueError('Replayed physical checkpoint changed')
    return True


class Queue(v4.Queue):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        receipt={'owner':OWNER,'source_sha256':old.sha(__file__),'worker_controller_source_sha256':V4_SHA,
                 'scheduling':'all42training_stages_then_all18four_view_evaluations',
                 'SFT_GPU_replay_before_intermediate_prune':True,'scientific_scope_or_math_changed':False}
        if receipt not in self.state['controller_revisions']:self.state['controller_revisions'].append(receipt)
        self.persist()

    def finalize_training(self,job,row):
        if job['stage']!='SFT':return
        from scripts.compact_research_checkpoints import compact_completed_checkpoint
        if not row.get('compaction'):
            v4.resources(idle=True);row['status']='compacting';self.persist()
            reserve=120 if job['id'] in v4.MANUAL else 32
            row['compaction']=compact_completed_checkpoint(old.directory(job),reserve_gb=reserve);self.persist()
        row['training_proof']=v4.validate_completed(job);self.persist()
        if not verified_replay(job,row):
            row['status']='validating_checkpoint';self.persist()
            self.execute(job,'validate',['--run',str(old.directory(job)),'--device','cuda'],row)
            identity=self.oracle._checkpoint_validation_identity(job)
            row['checkpoint_validation']=self.oracle._validate_checkpoint_report(identity);self.persist()
        if row.get('status')!='completed':
            row['status']='trained_pending_evaluation'
            row['external_evaluation_deferred_until_training_pass_complete']=True
        self.persist()

    def prune(self):
        from scripts.research_queue import prune_checkpoint_run
        for job in self.spec['jobs']:
            row=self.state['jobs'].get(job['id'],{})
            if not job['discard_checkpoint_after_dependents'] or not row.get('training_completed') or row.get('checkpoint_pruned'):continue
            if job['stage']=='A1' and any(not self.state['jobs'].get(p,{}).get('checkpoint_pruned') for p in job['requires']):continue
            consumers=[j for j in self.spec['jobs'] if job['id'] in j['requires']]
            eligible=bool(consumers)
            for consumer in consumers:
                record=self.state['jobs'].get(consumer['id'],{})
                if not record.get('training_completed'):eligible=False;break
                if consumer['stage']=='SFT' and not verified_replay(consumer,record):eligible=False;break
            if eligible:
                row['prune_receipt']=prune_checkpoint_run(old.directory(job),[old.directory(j) for j in consumers])
                row['checkpoint_pruned']=True;self.persist()

    def check_pruned(self,job,row):
        receipt=old.read(old.directory(job)/'checkpoint_pruned.json')
        if (job['stage']=='SFT' or receipt.get('status')!='pruned'
                or receipt.get('owner')!=row['training_proof']['manager_owner']
                or receipt.get('source_run')!=str(old.directory(job))
                or old.sha(old.directory(job)/'resolved_config.json')!=row['training_proof']['config_sha256']):
            raise ValueError('Owned intermediate pruning receipt changed')

    def run(self):
        self.state.update(status='running',scheduling_phase='training_first');self.persist()
        try:
            for job in self.spec['jobs']:
                if self.stopped:break
                row=self.state['jobs'].setdefault(job['id'],{})
                if row.get('checkpoint_pruned'):self.check_pruned(job,row);continue
                if any(not self.state['jobs'].get(p,{}).get('training_completed') for p in job['requires']):raise RuntimeError('Training prerequisite incomplete')
                self.state['active_job']=job['id'];self.persist()
                try:
                    v4.verify_plan(self.spec);v4.resources()
                    if row.get('training_completed'):row['training_proof']=v4.validate_completed(job)
                    else:self.adopt_or_train(job,row)
                    if job['stage']=='SFT':self.finalize_training(job,row)
                    else:row['status']='completed';self.persist()
                    self.prune()
                except BaseException as error:
                    row.update(status='interrupted' if isinstance(error,InterruptedError) else 'failed',error_type=type(error).__name__,error=str(error));self.persist();raise
            if not self.stopped:
                self.state['scheduling_phase']='full_external_evaluation';self.persist()
                for job in self.spec['jobs']:
                    if self.stopped:break
                    if job['stage']!='SFT':continue
                    row=self.state['jobs'][job['id']];self.state['active_job']=job['id'];self.persist()
                    try:
                        self.postprocess(job,row);row['status']='completed';self.persist()
                    except BaseException as error:
                        row.update(status='interrupted' if isinstance(error,InterruptedError) else 'failed',error_type=type(error).__name__,error=str(error));self.persist();raise
            self.state['status']='interrupted' if self.stopped else 'completed';self.persist();return 0
        finally:
            if self.child and self.child.poll() is None and old.same(self.child_info,old.proc(self.child.pid)):
                self.child.terminate()
                try:self.child.wait(timeout=30)
                except old.subprocess.TimeoutExpired:self.child.kill();self.child.wait()
            self.handle.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,default=old.PLAN);p.add_argument('--state',type=Path,default=ROOT/'runs/research/budget18_queue_state_v1.json');p.add_argument('--resource-ready',type=Path,required=True);p.add_argument('--resource-ready-sha',required=True);p.add_argument('--run-queue',action='store_true');p.add_argument('--retry-failed',action='store_true');args=p.parse_args()
    binding={'path':str(args.resource_ready.resolve(strict=True)),'sha256':args.resource_ready_sha};v4._RESOURCE=binding;v4.verify_plan(old.read(args.spec))
    if not args.run_queue:print(json.dumps({'status':'CPU_metadata_preflight','jobs':42,'cells':18,'worker_source':V4_SHA,'schedule':OWNER}));return
    queue=Queue(args.spec,args.state,args.retry_failed,binding);signal.signal(signal.SIGTERM,queue.signal);signal.signal(signal.SIGINT,queue.signal);return queue.run()


if __name__=='__main__':main()
