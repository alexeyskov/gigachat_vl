#!/usr/bin/env python3
"""One extra mixed SFT for all18 variants, after the current matrix finishes."""
from __future__ import annotations
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils import research_detection_training_v1 as training
from src.utils import research_additive_runner_v1 as old
from src.utils.research_storage import _atomic_json,DiskBudget
BASE=ROOT/'runs/research/detection_mix_v1';OUT=ROOT/'research_results/detection_mix_v1'
PLAN=ROOT/'configs/research/detection_mix_plan_v1.json';STATE=BASE/'queue_state.json'
SOURCES=('scripts/research_detection_mix_queue_v1.py','scripts/preflight_research_detection_mix_v1.py','scripts/predict_research_detections_v1.py')


def build_plan():
 previous=json.loads((ROOT/'configs/research/fixed_capacity_eval_plan_v5.json').read_text())
 best=json.loads((ROOT/'research_results/capacity_eval_v5/best_last_by_arm.json').read_text())['arms']
 jobs=[]
 for model,repo in old.MODELS.items():
  for vision,encoder in old.VISIONS.items():
   for method in ('R3','R1','R2'):
    name=f'{model}__{vision}__{method}__detection_mix_v1'
    if method=='R2':
     source=ROOT/f'runs/research/{model}__{vision}__4f__SFT__budget18_s42';branch=None;recipe='4f'
    else:
     arm=f'{model}__{vision}__fixed4__{method}';match=next(a for a in best if a['arm']==arm)
     source=Path(match['best']['checkpoint']).parents[1]
     if source.name=='last':raise ValueError('Unexpected predecessor path')
     original=next(j for j in previous['jobs'] if j['id']==source.name)
     branch=original['branch'];recipe='1c' if method=='R3' else '1b'
    jobs.append(dict(id=name,model=repo,model_key=model,vision=encoder,vision_key=vision,method=method,recipe=recipe,branch=branch,
     source_run=str(source),source_checkpoint=str(source/'last/state.pt'),output=str(BASE/name),
     source_selection='best original internal116 checkpoint' if method!='R2' else 'completed full-finetune control',
     retire_consumed_full_source=method=='R2'))
 return dict(name='final_SFT_small_detection_mix_v1',seed=42,epochs=1,total_configurations=18,
  detection_examples=450,multimodal_replay_examples=2550,text_examples=529,steps_per_job=56,
  detection_execution=training.definition(),jobs=jobs,
  source_sha256={p:training.sha(ROOT/p) for p in SOURCES},
  start_condition='previous18-cell matrix completed and its Supervisor owner EXITED0; old owner autostart disabled',
  scope='same R1/R3 QLoRA+four visual blocks and R2 full training, frozen vision+MLP; fresh optimizer half LR; one extra common mixed-data epoch',
  storage='last BEFORE evaluation, scored best/last hardlink the same physical weights; consumed old full checkpoints retired only after new GPU proof')


def verify_plan(digest):
 if training.sha(PLAN)!=digest:raise ValueError('Detection plan SHA changed')
 value=json.loads(PLAN.read_text())
 if value!=build_plan():raise ValueError('Detection source/data/budget/initialization plan changed')
 return value


def complete(job):
 run=Path(job['output']);proof=run/'native_reload_proof.json';done=run/'completed.json'
 if not proof.exists() or not done.exists():return False
 p=json.loads(proof.read_text());d=json.loads(done.read_text())
 if (p['status']!='passed' or p['mismatches'] or p['source_execution']!=training.definition()
   or d['steps']!=56 or d['records']!=3529 or p['checkpoint_sha256']!=d['checkpoint_sha256']):return False
 if not (run/'last/state.pt').is_file() or (run/'last').resolve()!=(run/'best').resolve():raise ValueError('One shared best/last owned final object required')
 return True


def retire_source(job):
 if not job['retire_consumed_full_source']:return
 run=Path(job['source_run']);target=Path(job['output']);receipt=target/'source_retired.json'
 if receipt.exists() and json.loads(receipt.read_text()).get('status') in ('already_retired','retired_consumed_full_source'):return
 if not complete(job):raise ValueError('Native OD+VQA replay/test proof required before consuming old full source')
 from scripts import retire_superseded_research_v1 as cleanup
 from scripts.research_archival_stock_queue_v5 import scan_anchor_users
 if (run/'checkpoint_pruned.json').exists():
  _atomic_json(receipt,dict(status='already_retired',source=str(run)));return
 if run.parent!=ROOT/'runs/research' or '__4f__SFT__budget18_s42' not in run.name:raise ValueError('Only known consumed baseline full weights eligible')
 inspected=cleanup.inspect_run(run)
 initialization=json.loads((target/'initialization.json').read_text())
 checkpoint=(run/'last/state.pt').resolve(strict=True)
 if training.sha(checkpoint)!=initialization['source_sha256']:raise ValueError('Full predecessor changed after it was consumed')
 try:result=cleanup.retire_one(inspected,receipt,scan_anchor_users)
 except RuntimeError as error:
  _atomic_json(receipt,dict(status='deferred_source_still_open_or_mapped',source=str(run),error=str(error)))
  return
 _atomic_json(receipt,dict(status='retired_consumed_full_source',proof=str(target/'native_reload_proof.json'),**result))


def gpu_idle():
 return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()


def maintain_memory():
 from scripts.guard_research_control_tail_memory_v1 import memory,reclaim
 if memory()['available_GiB']<72:
  previous=json.loads((ROOT/'configs/research/fixed_capacity_eval_plan_v5.json').read_text())
  reclaim(previous)


def wait_for_previous():
 from scripts.research_budget18_queue_v1 import supervisor_info
 while True:
  state=json.loads((ROOT/'runs/research/additive_visual_v1/queue_state.json').read_text());service=supervisor_info('research-additive-v1')
  if state['status']=='completed' and service['statename']=='EXITED' and service['exitstatus']==0 and gpu_idle():break
  _atomic_json(OUT/'admission_status.json',dict(status='waiting_for_previous_matrix',previous_phase=state['status'],previous_service=service['statename'],updated_unix=time.time()))
  print(json.dumps(dict(event='detection_queue_waits_for_current_training_to_finish',phase=state['status'])),flush=True);time.sleep(30)
 # Prevent the retired previous-stage full weights from being reopened by an
 # obsolete autostart queue on a later server restart. Current GPU work is over.
 conf=Path('/etc/supervisor/conf.d/research-additive-v1.conf');text=conf.read_text()
 if 'autostart=true' in text:
  conf.write_text(text.replace('autostart=true','autostart=false'))
  subprocess.run(['supervisorctl','reread'],check=True);subprocess.run(['supervisorctl','update','research-additive-v1'],check=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--write-plan',action='store_true');p.add_argument('--plan-sha');p.add_argument('--run',action='store_true');args=p.parse_args()
 if args.write_plan:
  value=build_plan()
  if PLAN.exists() and json.loads(PLAN.read_text())!=value:raise ValueError('Existing detection plan differs; new revision required')
  _atomic_json(PLAN,value);print(training.sha(PLAN));return
 plan=verify_plan(args.plan_sha)
 if not args.run:print(json.dumps(dict(status='CPU_plan_preflight_passed',jobs=len(plan['jobs']))));return
 BASE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 # An existing detection run may have already retired old full sources. The old
 # matrix remains archived; its disabled owner need not be EXITED again.
 if not STATE.exists():wait_for_previous()
 handle=(ROOT/'runs/research/budget18_queue_state_v1.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 own=STATE.with_suffix('.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads(STATE.read_text()) if STATE.exists() else dict(plan_sha256=args.plan_sha,status='running',jobs={})
 if state['plan_sha256']!=args.plan_sha:raise ValueError('Refuse mixed detection source profiles')
 if state.get('status')=='requires_repair':
  print('Detection queue needs repair; refusing to repeat expensive training automatically',flush=True)
  handle.close();own.close();return
 stop=False;child=None
 def persist():state['updated_unix']=time.time();state['free_disk_GiB']=DiskBudget(BASE,32).report()['free_gib'];_atomic_json(STATE,state)
 def interrupt(_n,_f):
  nonlocal stop
  stop=True
  if child and child.poll() is None:child.terminate()
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 def execute(job,script,arguments,kind):
  nonlocal child
  if not gpu_idle():raise RuntimeError('GPU busy: no foreign worker stopped')
  maintain_memory();DiskBudget(BASE,32).ensure_space(2*2**30)
  if kind=='train':
   # Reserve the actual next model checkpoint BEFORE spending GPU time.
   # Validation and recovery publish hardlinks and do not need a second model.
   expected=int(Path(job['source_checkpoint']).stat().st_size*1.10)+8*2**30
   DiskBudget(BASE,32).ensure_space(expected)
  run=Path(job['output']);run.mkdir(parents=True,exist_ok=True);job_path=run/'job.json';_atomic_json(job_path,job)
  logdir=ROOT/'logs/research/detection_mix_v1';logdir.mkdir(parents=True,exist_ok=True);log=logdir/(job['id']+'__'+kind+'.log')
  row=state['jobs'].setdefault(job['id'],{})
  with log.open('ab') as f:
   child=subprocess.Popen([sys.executable,'-u',script,'--job',str(job_path),*arguments],cwd=ROOT,
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdout=f,stderr=subprocess.STDOUT)
   row.update(status=kind,pid=child.pid,log=str(log),started_unix=time.time());state.update(active_job=job['id'],status='running');persist()
   print(json.dumps(dict(event='detection_worker_started',job=job['id'],kind=kind,pid=child.pid)),flush=True)
   while child.poll() is None:
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:
     # Memory-cache pressure causes cache advice and a warning, never the
     # legacy controller's destructive termination of an active full trainer.
     try:maintain_memory()
     except Exception as error:print(json.dumps(dict(event='memory_maintenance_warning',error=str(error))),flush=True)
     persist();print(json.dumps(dict(event='detection_worker_running',job=job['id'],kind=kind,pid=child.pid)),flush=True)
   code=child.returncode;child=None
   if code:raise RuntimeError(f'{kind} exited{code}: {log}')
 try:
  persist()
  for job in plan['jobs']:
   if stop:break
   verify_plan(args.plan_sha);row=state['jobs'].setdefault(job['id'],{})
   if complete(job):
    retire_source(job);row.update(status='completed');persist();continue
   attempts=0
   while not stop:
    try:
     run=Path(job['output'])
     if not (run/'completed.json').exists():
      trained=(run/'training_completed.json').exists() and (run/'last/state.pt').is_file()
      execute(job,'scripts/run_research_detection_mix_v1.py',['--allow-gpu']+(['--evaluate-existing'] if trained else []),'evaluation_recovery' if trained else 'train')
     execute(job,'scripts/validate_research_detection_mix_v1.py',[],'native_validation_and_test')
     if not complete(job):raise ValueError('Complete independent detection proof missing')
     retire_source(job);row.update(status='completed',completed_unix=time.time());persist();break
    except Exception as error:
     row.setdefault('failures',[]).append(dict(time=time.time(),error=str(error)));attempts+=1
     trained=(Path(job['output'])/'training_completed.json').exists()
     if trained and attempts<3:
      row.update(status='retry_evaluation_only');persist();continue
     row.update(status='requires_repair');state['status']='requires_repair';persist()
     _atomic_json(OUT/'attention_required.json',dict(job=job['id'],error=str(error),trained_weights_available=trained,
      policy='no automatic whole-stage retraining after an unhandled error'))
     return
  state['status']='stopped' if stop else 'completed';persist()
 finally:
  if child and child.poll() is None:child.terminate();child.wait(timeout=180)
  handle.close();own.close()

if __name__=='__main__':main()
