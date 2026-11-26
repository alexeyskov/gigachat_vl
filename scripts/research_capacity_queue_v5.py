#!/usr/bin/env python3
"""Capacity matrix with correct private evaluation and final-weight recovery."""
from __future__ import annotations
import argparse,copy,fcntl,json,os,signal,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_capacity_queue_v4 as prior
from src.utils import research_capacity_eval_v5 as repair
from src.utils.research_storage import _atomic_json,BestLastCheckpointManager,DiskBudget
native=prior.native;core=repair.core
BASE=ROOT/'runs/research/fixed_capacity_eval_v5';OUT=ROOT/'research_results/capacity_eval_v5'
PLAN=ROOT/'configs/research/fixed_capacity_eval_plan_v5.json';STATE=prior.STATE
OLD_SHA='1d325c558902e6cb7c4d838fba590ff88609acb65567831af983316ee68e60fd'
ENTRY='scripts/run_research_capacity_eval_v5.py'


def build_plan():
 old=prior.verify_plan(prior.PLAN,OLD_SHA);value=copy.deepcopy(old)
 for job in value['jobs']:
  if job['stage']!='SFT':continue
  job['output']=str(BASE/job['id']);argv=job['argv'];argv[argv.index('--output')+1]=job['output']
  source=Path(argv[argv.index('--initial-checkpoint')+1])
  if source.is_relative_to(prior.BASE) and '__P1__SFT' in str(source):
   argv[argv.index('--initial-checkpoint')+1]=str(BASE/source.relative_to(prior.BASE))
  job['train_entry']=ENTRY
 priority='lfm25__siglip2__fixed4__R3__P1__SFT'
 value['jobs'].sort(key=lambda j:0 if j['stage']=='A1' else 1 if j['id']==priority else 2)
 value.update(schema_version=5,name='fixed_capacity_private_eval_v5',
  predecessor=dict(path=str(prior.PLAN),sha256=OLD_SHA),evaluation_repair=repair.definition(),
  recovery='model-only final-update weights before evaluation; failed eval retries without training',
  source_sha256={**old['source_sha256'],**{p:core.sha(ROOT/p) for p in (*repair.SOURCES,'scripts/research_capacity_queue_v5.py')}})
 return value


def verify_plan(path,digest):
 if Path(path).resolve()!=PLAN or core.sha(path)!=digest:raise ValueError('V5 plan SHA/path changed')
 value=json.loads(PLAN.read_text())
 if value!=build_plan():raise ValueError('V5 source/data/architecture budget binding changed')
 return value


def complete(job):
 if not prior.original_complete(job):return False
 config=json.loads((Path(job['output'])/'resolved_config.json').read_text())
 if config.get('capacity_execution')!=prior.capacity.execution(job['branch'],job['training_pass']):raise ValueError('Training profile changed')
 if job['stage']=='SFT' and config.get('capacity_evaluation_v5')!=repair.definition():raise ValueError('Private evaluation binding missing')
 return True


def retire(job):
 prior.retire_a1(job)
 if job['stage']!='SFT':return
 run=Path(job['output']);recovery=run/'recovery';proof=run/'native_reload_proof.json'
 if not recovery.exists():return
 if (run.parent!=BASE or recovery.is_symlink() or json.loads(proof.read_text()).get('status')!='passed'):
  raise ValueError('Only owned independently verified recovery weights may be retired')
 receipt=json.loads((recovery/'receipt.json').read_text())
 if receipt['source_binding']!=repair.definition() or receipt['checkpoint_sha256']!=core.sha(recovery/'model.pt'):raise ValueError('Recovery snapshot changed')
 _atomic_json(run/'recovery_retired.json',dict(reason='Final model passed independent GPU reload/replay',receipt=receipt,time=time.time()))
 (recovery/'model.pt').unlink();(recovery/'receipt.json').unlink();recovery.rmdir()


class Queue(native.Queue):
 def __init__(self,*a,**k):
  BASE.mkdir(parents=True,exist_ok=True);super().__init__(*a,**k)
 def persist(self):
  self.state.update(current_revision='fixed_capacity_private_eval_v5',active_plan_path=str(PLAN),active_runs_root=str(BASE))
  result=super().persist();summaries=[]
  for arm in self.plan.get('arms',[]):
   candidates=[]
   for index in (1,2):
    run=BASE/(arm['id']+f'__P{index}__SFT');p=run/'native_reload_proof.json'
    if p.exists():
     proof=json.loads(p.read_text())
     if proof.get('status')=='passed':candidates.append(dict(training_pass=index,score=proof['score'],checkpoint=str(run/'last/state.pt')))
   if candidates:summaries.append(dict(arm=arm['id'],passes=candidates,best=max(candidates,key=lambda c:(c['score'],-c['training_pass'])),last=candidates[-1]))
  OUT.mkdir(parents=True,exist_ok=True);_atomic_json(OUT/'best_last_by_arm.json',dict(arms=summaries))
  return result
 def execute(self,job,argv,kind):
  argv=list(argv)
  if kind=='train':
   if job['stage']!='SFT':raise ValueError('All V4 A1 must be reused; no extra alignment training')
   run=Path(job['output']);recovery=run/'recovery/model.pt'
   if recovery.exists():argv=['scripts/recover_research_capacity_eval_v5.py','--run',str(run)]
   else:
    attempts=self.state['jobs'].get(job['id'],{}).get('failures',[])
    if len(attempts)>=2:
     _atomic_json(OUT/'attention_required.json',dict(job=job['id'],reason='Repeated training failure before recovery snapshot; refuse costly endless retraining',failures=attempts))
     raise ValueError('Repeated failure requires repair; automatic full retraining blocked')
    argv[0]=ENTRY
  return super().execute(job,argv,kind)


def main():
 p=argparse.ArgumentParser();p.add_argument('--write-plan',action='store_true');p.add_argument('--plan-sha');p.add_argument('--run',action='store_true');args=p.parse_args()
 if args.write_plan:
  value=build_plan()
  if PLAN.exists() and json.loads(PLAN.read_text())!=value:raise ValueError('Existing V5 plan differs')
  _atomic_json(PLAN,value);print(core.sha(PLAN));return
 plan=verify_plan(PLAN,args.plan_sha)
 if not args.run:print(json.dumps(dict(status='CPU_preflight_passed',stages=len(plan['jobs']),reused_A1=12)));return
 OUT.mkdir(parents=True,exist_ok=True)
 if not (OUT/'handoff/completed.json').exists():raise ValueError('Natural-worker-boundary handoff required')
 with STATE.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);old=json.loads(STATE.read_text())
  if old['plan_sha256']!=args.plan_sha:
   if old['plan_sha256']!=OLD_SHA:raise ValueError('Exact predecessor V4 state required')
   (OUT/'state_before.json').write_bytes(STATE.read_bytes());jobs={}
   for job in plan['jobs']:
    if job['stage']=='A1':
     if old['jobs'].get(job['id'],{}).get('status')!='completed' or not complete(job):raise ValueError('All twelve actual V4 A1 must be complete')
     jobs[job['id']]=copy.deepcopy(old['jobs'][job['id']])
   _atomic_json(STATE,dict(plan_sha256=args.plan_sha,jobs=jobs,status='starting_repaired_private_evaluation',previous_plan_sha256=OLD_SHA))
 native.PLAN=PLAN;native.verify_plan=verify_plan;native.complete=complete;native.retire_a1=retire
 queue=Queue(plan,STATE,args.plan_sha);signal.signal(signal.SIGTERM,queue.signal);signal.signal(signal.SIGINT,queue.signal)
 return queue.run()

if __name__=='__main__':raise SystemExit(main())
