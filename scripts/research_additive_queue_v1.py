#!/usr/bin/env python3
"""Restartable Supervisor queue for the additive architecture revision."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
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
from src.utils import research_additive_runner_v1 as backend
from src.utils.research_storage import _atomic_json,DiskBudget

BASE=ROOT/'runs/research/additive_visual_v1'
PLAN=ROOT/'configs/research/additive_visual_plan_v1.json'
STATE=BASE/'queue_state.json'


def emit(event,**values):
    print(json.dumps(dict(event=event,time=time.time(),**values)),flush=True)


def build_plan():
    from dataclasses import asdict
    from src.model.research_additive_visual_v1 import VisualBranchConfig
    jobs=[];contracts=[]
    for model,repo in backend.MODELS.items():
        for vision,encoder in backend.VISIONS.items():
            for masks in (False,True):
                for mode in ('fixed','routed'):
                    config=VisualBranchConfig(mode=mode,blocks=2 if mode=='fixed' else 8,learned_masks=masks)
                    contract=backend.definition(asdict(config))
                    if contract not in contracts:contracts.append(contract)
                    prefix=f'{model}__{vision}__{mode}__R{3 if masks else 1}'
                    previous=None
                    for stage in ('A1','SFT'):
                        out=BASE/(prefix+'__'+stage)
                        source=BASE/'projectors'/f'{model}__{vision}.pt' if stage=='A1' else BASE/(prefix+'__A1')/'last/state.pt'
                        argv=['--visual-branch',mode, *(['--learned-masks'] if masks else []),
                              '--allow-gpu','--model',repo,'--vision',encoder,
                              '--stage',stage,'--recipe','1c' if masks else '1b',
                              '--manifest',str(backend.DATA/('alignment_a1_train.jsonl' if stage=='A1' else 'sft_multimodal_train.jsonl')),
                              '--validation',str(backend.DATA/('alignment_validation.jsonl' if stage=='A1' else 'sft_validation.jsonl')),
                              '--initial-checkpoint',str(source),'--output',str(out),
                              '--seed','42','--epochs','1','--batch-size','64','--effective-batch-size','64',
                              '--max-length','4096','--eval-every','10000','--reserve-gb','32']
                        if stage=='SFT':argv+=['--text-manifest',str(backend.DATA/'text_train.jsonl'),
                            '--text-fraction','.15','--text-loss-weight','.15','--selection-metric','group_macro']
                        if vision=='qwen25':argv+=['--vision-preprocessing-profile','qwen_native_bounded_729']
                        jobs.append(dict(id=out.name,backend=model,vision=vision,mode=mode,learned_masks=masks,
                            branch=asdict(config),stage=stage,output=str(out),requires=[] if previous is None else [previous],
                            steps=4 if stage=='A1' else 56,records=256 if stage=='A1' else 3529,argv=argv))
                        previous=out.name
    controls=[dict(id=f'{m}__{v}__4f__SFT__budget18_s42',
                   run=str(ROOT/f'runs/research/{m}__{v}__4f__SFT__budget18_s42'))
              for m in backend.MODELS for v in backend.VISIONS]
    return dict(schema_version=1,name='additive_visual_transformer_v1',seed=42,
        new_scientific_configurations=24,reused_full_finetune_controls=6,total_primary_configurations=30,
        supersedes='legacy common-pool R1/R3 primary interpretation; old artifacts retained as historical results',
        data_release_sha256=backend.RELEASE_SHA,contracts=contracts,jobs=jobs,controls=controls,
        source_sha256={p:backend.sha(ROOT/p) for p in ('scripts/research_additive_queue_v1.py',
            'scripts/research_additive_legacy_controls_v1.py','scripts/validate_research_additive_v1.py')},
        storage='completed delta-only best/last shared object; A1 retired after independently reloaded SFT proof',
        legacy_control_tail=True)


def verify_plan(path,expected_sha):
    if backend.sha(path)!=expected_sha:raise ValueError('Additive plan bytes changed')
    plan=json.loads(path.read_text())
    if plan!=build_plan():raise ValueError('Additive plan source/architecture/data contract changed')
    return plan


def gpu_idle():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return not out.strip()


def complete(job):
    out=Path(job['output']);done=out/'completed.json'
    if not done.exists():return False
    actual=json.loads(done.read_text());config=json.loads((out/'resolved_config.json').read_text())
    if actual['steps']!=job['steps'] or actual['records']!=job['records']:
        raise ValueError('Incomplete/mismatched training receipt')
    if config['additive_execution']!=backend.definition(job['branch']):raise ValueError('Completed architecture contract differs')
    for role in ('best','last'):
        if not (out/role).is_symlink() or not (out/role/'state.pt').is_file():raise ValueError('Owned completed best/last required')
    if (out/'best').resolve()!=(out/'last').resolve():raise ValueError('Expected one final physical checkpoint')
    return True


def retire_a1(job):
    """Retire only this revision's owned A1 after the SFT reload proof passes."""
    import shutil
    from src.utils.research_storage import BestLastCheckpointManager
    out=Path(job['output'])
    if job['stage']!='SFT' or not (out/'native_reload_proof.json').exists():return
    proof=json.loads((out/'native_reload_proof.json').read_text())
    if proof.get('status')!='passed':return
    predecessor=Path(job['output'].removesuffix('__SFT')+'__A1')
    if (predecessor/'weights_retired.json').exists():return
    manager=BestLastCheckpointManager(predecessor,DiskBudget(predecessor,32))
    with manager._lock():
        targets={manager._resolve_owned_pointer(role) for role in ('best','last')}
        if None in targets:raise ValueError('Missing owned A1 role')
        if any(not manager._owns(target) for target in targets):raise ValueError('Refusing foreign A1 retirement')
        receipt=dict(reason='completed SFT reloaded independently; A1 training metadata retained',
                     sft_proof_sha256=backend.sha(out/'native_reload_proof.json'),
                     objects=[dict(path=str(p),state_sha256=backend.sha(p/'state.pt')) for p in targets])
        _atomic_json(predecessor/'weights_retired.json',receipt)
        for role in ('best','last'):(predecessor/role).unlink()
        for target in targets:
            if not manager._owns(target):raise ValueError('Refusing foreign checkpoint retirement')
            shutil.rmtree(target)


class Queue:
    def __init__(self,plan,path,plan_sha):
        self.plan=plan;self.path=path;self.plan_sha=plan_sha;self.stop=False;self.child=None
        BASE.mkdir(parents=True,exist_ok=True)
        # Share the admission lease with legacy18, in addition to this series' lock.
        self.lock=(ROOT/'runs/research/budget18_queue_state_v1.lock').open('a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.ownlock=path.with_suffix('.lock').open('a');fcntl.flock(self.ownlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.state=json.loads(path.read_text()) if path.exists() else dict(plan_sha256=plan_sha,jobs={})
        if self.state['plan_sha256']!=plan_sha:raise ValueError('Refusing mixed architecture queue state')
        self.persist()

    def persist(self):
        self.state['updated_unix']=time.time();self.state['free_disk_GiB']=DiskBudget(BASE,32).report()['free_gib']
        _atomic_json(self.path,self.state)

    def signal(self,_n,_f):
        self.stop=True
        if self.child and self.child.poll() is None:self.child.terminate()

    def execute(self,job,argv,kind):
        if not gpu_idle():raise RuntimeError('GPU occupied; waiting without touching foreign process')
        DiskBudget(BASE,32).ensure_space(2*2**30)
        logdir=ROOT/'logs/research/additive_visual_v1';logdir.mkdir(exist_ok=True,parents=True)
        logfile=logdir/(job['id']+'__'+kind+'.log')
        row=self.state['jobs'].setdefault(job['id'],{})
        with logfile.open('ab') as stream:
            self.child=subprocess.Popen([sys.executable,'-u',*argv],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                         CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdout=stream,stderr=subprocess.STDOUT)
            row.update(status=kind,pid=self.child.pid,log=str(logfile),argv=argv,started_unix=time.time())
            self.state['active_job']=job['id'];self.persist();emit('worker_started',job=job['id'],kind=kind,pid=self.child.pid,log=str(logfile))
            while self.child.poll() is None:
                try:self.child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    metrics=Path(job['output'])/'metrics.jsonl';last=None
                    if metrics.exists():
                        with metrics.open('rb') as f:
                            f.seek(max(0,metrics.stat().st_size-16384));lines=f.read().splitlines()
                        for line in reversed(lines):
                            try:last=json.loads(line);break
                            except ValueError:pass
                    self.state['last_progress']=last;self.persist()
                    emit('worker_progress',job=job['id'],kind=kind,last=last,free_disk_GiB=self.state['free_disk_GiB'])
            code=self.child.returncode;self.child=None
            if code:raise RuntimeError(f'{kind} worker exited {code}: {logfile}')

    def run(self):
        self.state.update(status='running',current_revision='additive_visual_transformer_v1');self.persist()
        try:
            while not self.stop:
                progress=False
                for job in self.plan['jobs']:
                    if self.stop:break
                    row=self.state['jobs'].setdefault(job['id'],{})
                    if row.get('status')=='completed':continue
                    if any(self.state['jobs'].get(p,{}).get('status')!='completed' for p in job['requires']):continue
                    if row.get('retry_after_unix',0)>time.time():continue
                    verify_plan(PLAN,self.plan_sha)
                    try:
                        if not complete(job):
                            if job['stage']=='A1':
                                backend.extract_projector(job['backend'],job['vision'],BASE/'projectors'/f"{job['backend']}__{job['vision']}.pt")
                            self.execute(job,['scripts/run_research_additive_v1.py',*job['argv']],'train')
                            if not complete(job):raise ValueError('Trainer returned without complete artifact')
                        if job['stage']=='SFT':
                            proof=Path(job['output'])/'native_reload_proof.json'
                            if not proof.exists() or json.loads(proof.read_text()).get('status')!='passed':
                                self.execute(job,['scripts/validate_research_additive_v1.py','--run',job['output']],'reload')
                            retire_a1(job)
                        row.update(status='completed',completed_unix=time.time());row.pop('error',None)
                        self.persist();emit('job_completed',job=job['id']);progress=True
                    except Exception as error:
                        if self.stop:break
                        attempts=row.setdefault('failures',[]);attempts.append(dict(time=time.time(),error=str(error)))
                        row.update(status='retry_pending',error=str(error),retry_after_unix=time.time()+min(600,30*2**min(len(attempts),4)))
                        self.persist();emit('job_retry_independent_work_continues',job=job['id'],error=str(error))
                if self.stop:break
                if all(self.state['jobs'].get(j['id'],{}).get('status')=='completed' for j in self.plan['jobs']):
                    self.state['status']='additive_training_complete_controls_pending';self.persist()
                    # Release the shared legacy lease before the native control tail.
                    fcntl.flock(self.lock,fcntl.LOCK_UN)
                    code=subprocess.call([sys.executable,'scripts/research_additive_legacy_controls_v1.py','--run'],cwd=ROOT,env=os.environ)
                    if code:raise RuntimeError('Full-finetune control tail failed; retry from native state')
                    self.state['status']='completed';self.persist();return 0
                if not progress:time.sleep(30)
            self.state['status']='stopped';self.persist();return 0
        finally:
            if self.child and self.child.poll() is None:
                self.child.terminate();self.child.wait(timeout=180)
            self.lock.close();self.ownlock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write-plan',action='store_true')
    p.add_argument('--plan-sha');p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.write_plan:
        value=build_plan()
        if PLAN.exists() and json.loads(PLAN.read_text())!=value:raise ValueError('Existing additive plan differs; use a new revision')
        _atomic_json(PLAN,value);print(backend.sha(PLAN));return
    plan=verify_plan(PLAN,args.plan_sha)
    if not args.run:print(json.dumps(dict(status='CPU_preflight_passed',jobs=len(plan['jobs']),configurations=30)));return
    queue=Queue(plan,STATE,args.plan_sha)
    signal.signal(signal.SIGTERM,queue.signal);signal.signal(signal.SIGINT,queue.signal)
    return queue.run()


if __name__=='__main__':raise SystemExit(main())
