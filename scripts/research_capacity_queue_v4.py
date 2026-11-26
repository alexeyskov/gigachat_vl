#!/usr/bin/env python3
"""Four fixed blocks, at most two SFT passes; preserve two-block comparisons."""
from __future__ import annotations
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
from scripts import research_additive_queue_v3 as prior
native=prior.native
from src.utils import research_capacity_v4 as capacity
from src.utils.research_storage import _atomic_json,BestLastCheckpointManager,DiskBudget
BASE=ROOT/'runs/research/fixed_capacity_v4'
PLAN=ROOT/'configs/research/fixed_capacity_plan_v4.json'
STATE=prior.STATE
OLD_SHA='5d76b8395b26b39da8871f63692ccb46db8d375c5637d225d3b8571e29b86bc6'
OUT=ROOT/'research_results/capacity_v4'
ENTRY='scripts/run_research_capacity_v4.py'


def build_plan():
    old=prior.verify_plan(prior.PLAN,OLD_SHA)
    arms=[];jobs=[]
    pairs=[('smallthinker4','qwen25',False)]+[(m,v,g) for m in capacity.core.MODELS for v in capacity.core.VISIONS for g in (False,True)
        if (m,v,g)!=('smallthinker4','qwen25',False)]
    for model,vision,masks in pairs:
        arm=f'{model}__{vision}__fixed4__R{3 if masks else 1}'
        original=next(j for j in old['jobs'] if j['backend']==model and j['vision']==vision and j['learned_masks']==masks and j['stage']=='SFT')
        common=list(original['argv']);i=common.index('--visual-branch');del common[i:i+2]
        previous=None;stages=[]
        for stage,pass_index in [('A1',1),('SFT',1),('SFT',2)]:
            name=arm+'__A1' if stage=='A1' else arm+f'__P{pass_index}__SFT'
            out=BASE/name
            initial=(ROOT/f'runs/research/additive_visual_v1/projectors/{model}__{vision}.pt' if stage=='A1' else
                     BASE/(arm+'__A1')/'last/state.pt' if pass_index==1 else BASE/(arm+'__P1__SFT')/'last/state.pt')
            argv=list(common)
            def replace(flag,value):
                if flag in argv:argv[argv.index(flag)+1]=str(value)
                else:argv.extend((flag,str(value)))
            replace('--output',out);replace('--stage',stage);replace('--initial-checkpoint',initial)
            if stage=='A1':
                replace('--manifest',capacity.core.DATA/'alignment_a1_train.jsonl');replace('--validation',capacity.core.DATA/'alignment_validation.jsonl')
                replace('--selection-metric','ce')
                for flag in ('--text-manifest','--text-fraction','--text-loss-weight'):
                    if flag in argv:
                        idx=argv.index(flag);del argv[idx:idx+2]
            argv+=['--visual-blocks','4','--training-pass',str(pass_index)]
            branch=dict(original['branch'],blocks=4)
            job=dict(id=name,arm=arm,backend=model,vision=vision,mode='fixed',learned_masks=masks,branch=branch,
                stage=stage,training_pass=pass_index,output=str(out),requires=[] if previous is None else [previous],
                steps=4 if stage=='A1' else 56,records=256 if stage=='A1' else 3529,argv=argv,
                a1_run=str(BASE/(arm+'__A1')))
            jobs.append(job);stages.append(name);previous=name
        arms.append(dict(id=arm,backend=model,vision=vision,learned_masks=masks,stages=stages))
    # One matched extra pass tests whether longer training alone helps the
    # already verified best two-block SmallThinker/Qwen configuration.
    base=next(j for j in old['jobs'] if j['id']=='smallthinker4__qwen25__fixed__R1__SFT')
    control=copy.deepcopy(jobs[2]);control.update(id='smallthinker4__qwen25__fixed2__R1__P2__SFT',
        arm='two_block_continuation_control',branch=base['branch'],requires=[],
        output=str(BASE/'smallthinker4__qwen25__fixed2__R1__P2__SFT'),a1_run=None)
    argv=list(base['argv']);idx=argv.index('--visual-branch');del argv[idx:idx+2]
    argv[argv.index('--output')+1]=control['output']
    argv[argv.index('--initial-checkpoint')+1]=str(Path(base['output'])/'last/state.pt')
    argv+=['--visual-blocks','2','--training-pass','2'];control['argv']=argv
    jobs.insert(2,control)  # Start the requested4-block model before the longer-training control.
    return dict(schema_version=4,name='fixed_capacity_v4',fixed_only=True,blocks=4,seed=42,
        new_scientific_configurations=12,reused_full_finetune_controls=6,total_primary_configurations=18,
        SFT_stages_total=25,maximum_SFT_passes=2,two_block_continuation_control=True,
        predecessor=dict(path=str(prior.PLAN),sha256=OLD_SHA),jobs=jobs,arms=arms,controls=old['controls'],
        source_sha256={p:capacity.core.sha(ROOT/p) for p in (*capacity.SOURCES,'scripts/research_capacity_queue_v4.py')},
        data_release_sha256=capacity.core.RELEASE_SHA,
        continuation='one additional pass with fresh AdamW, half LR, deterministic epoch1 data order; not optimizer resume')


def verify_plan(path,digest):
    if Path(path).resolve()!=PLAN or capacity.core.sha(path)!=digest:raise ValueError('Capacity plan path/SHA changed')
    value=json.loads(Path(path).read_text())
    if value!=build_plan():raise ValueError('Capacity source/budget/architecture binding changed')
    return value


original_complete=native.complete


def complete(job):
    if not original_complete(job):return False
    cfg=json.loads((Path(job['output'])/'resolved_config.json').read_text())
    if (cfg.get('capacity_execution')!=capacity.execution(job['branch'],job['training_pass'])
        or cfg.get('training_pass')!=job['training_pass'] or cfg['model']!=capacity.core.MODELS[job['backend']]
        or cfg['vision']!=capacity.core.VISIONS[job['vision']]):raise ValueError('Capacity stage configuration differs')
    return True


def retire_a1(job):
    if job['stage']!='SFT' or job['training_pass']!=1 or not job.get('a1_run'):return
    run=Path(job['a1_run']);proof=Path(job['output'])/'native_reload_proof.json'
    if json.loads(proof.read_text()).get('status')!='passed':raise ValueError('A1 retirement requires passed fresh GPU replay')
    if (run/'weights_retired.json').exists():return
    manager=BestLastCheckpointManager(run,DiskBudget(run,32))
    with manager._lock():
        targets={manager._resolve_owned_pointer(r) for r in ('best','last')}
        if None in targets or any(not manager._owns(p) for p in targets):raise ValueError('Owned capacity A1 required')
        _atomic_json(run/'weights_retired.json',dict(reason='P1 SFT independently reloaded',proof_sha256=capacity.core.sha(proof),
            objects=[dict(path=str(p),sha256=capacity.core.sha(p/'state.pt')) for p in targets]))
        for role in ('best','last'):(run/role).unlink()
        import shutil
        for p in targets:shutil.rmtree(p)


class Queue(native.Queue):
    def __init__(self,*a,**k):
        BASE.mkdir(parents=True,exist_ok=True)
        super().__init__(*a,**k)
    def persist(self):
        self.state.update(current_revision='fixed_capacity_v4',active_plan_path=str(PLAN),active_runs_root=str(BASE),routed_experiments_excluded=True)
        result=super().persist()
        summaries=[]
        for arm in self.plan.get('arms',[]):
            candidates=[]
            for pass_index in (1,2):
                run=BASE/(arm['id']+f'__P{pass_index}__SFT');p=run/'native_reload_proof.json'
                if p.exists():
                    proof=json.loads(p.read_text())
                    if proof.get('status')=='passed':candidates.append(dict(training_pass=pass_index,score=proof['score'],checkpoint=str(run/'last/state.pt')))
            if candidates:
                best=max(candidates,key=lambda c:(c['score'],-c['training_pass']))
                summaries.append(dict(arm=arm['id'],passes=candidates,best=best,last=candidates[-1]))
        if OUT.exists():_atomic_json(OUT/'best_last_by_arm.json',dict(arms=summaries))
        return result
    def execute(self,job,argv,kind):
        argv=list(argv)
        if kind=='train':argv[0]=ENTRY
        return super().execute(job,argv,kind)
    def finalize_baseline(self):
        receipt=OUT/'handoff/completed.json'
        if not receipt.exists():raise ValueError('Natural GPU-boundary handoff receipt required')
        info=json.loads(receipt.read_text());old=prior.verify_plan(prior.PLAN,OLD_SHA)
        job=next(j for j in old['jobs'] if j['id']==info['previous_job'])
        status='source_stage_not_completed'
        if job['stage']=='SFT' and (Path(job['output'])/'completed.json').exists():
            proof=Path(job['output'])/'native_reload_proof.json'
            if not proof.exists() or json.loads(proof.read_text()).get('status')!='passed':
                self.execute(job,['scripts/validate_research_additive_v1.py','--run',job['output']],'baseline_reload')
            # The original helper's path convention matches this old2 run.
            original_retire_a1(job)
            status='completed_and_independently_replayed'
        self.state['previous_baseline_finalized']=dict(job=job['id'],status=status,time=time.time());self.persist()
    def run(self):
        if not self.state.get('previous_baseline_finalized'):self.finalize_baseline()
        return super().run()


original_retire_a1=native.retire_a1


def main():
    p=argparse.ArgumentParser();p.add_argument('--write-plan',action='store_true');p.add_argument('--plan-sha');p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.write_plan:
        value=build_plan()
        if PLAN.exists() and json.loads(PLAN.read_text())!=value:raise ValueError('Existing capacity plan differs')
        _atomic_json(PLAN,value);print(capacity.core.sha(PLAN));return
    plan=verify_plan(PLAN,args.plan_sha)
    if not args.run:print(json.dumps(dict(status='CPU_preflight_passed',arms=12,SFT_passes=25,stages=len(plan['jobs']))));return
    OUT.mkdir(parents=True,exist_ok=True)
    with STATE.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=json.loads(STATE.read_text())
        if old['plan_sha256']!=args.plan_sha:
            if old['plan_sha256']!=OLD_SHA:raise ValueError('Unexpected predecessor queue state')
            (OUT/'state_before.json').write_bytes(STATE.read_bytes())
            _atomic_json(STATE,dict(plan_sha256=args.plan_sha,jobs={},status='baseline_finalization',
                active_plan_path=str(PLAN),current_revision='fixed_capacity_v4',previous_plan_sha256=OLD_SHA))
    def retained_projector(_model,_vision,path):return capacity.validate_cached_projector(path)['origin']
    native.PLAN=PLAN;native.verify_plan=verify_plan;native.complete=complete;native.retire_a1=retire_a1
    native.backend.extract_projector=retained_projector
    queue=Queue(plan,STATE,args.plan_sha)
    signal.signal(signal.SIGTERM,queue.signal);signal.signal(signal.SIGINT,queue.signal)
    return queue.run()


if __name__=='__main__':raise SystemExit(main())
