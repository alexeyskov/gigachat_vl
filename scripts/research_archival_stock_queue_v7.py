#!/usr/bin/env python3
"""Same V6 CPU controller; archived full roles replay in fresh independent children."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OWNER='fresh_role_replay_archival_stock_queue_v7';SOURCE='scripts/research_archival_stock_queue_v7.py';TEST='tests/test_research_archival_stock_queue_v7.py'
V6='scripts/research_archival_stock_queue_v6.py';V6_SHA='e46379faab87b59d44dd157cb05b2baa1f99050d46deb121f9df9d8a78a3cc5c'
V6_READY=ROOT/'configs/research/archival_stock_queue_v6_ready.json';V6_READY_SHA='e1dd0bde1954a0bbe7cee7eee71da990d2b8ec4448b0f54d450fa58eb666ce31'
READY=ROOT/'configs/research/archival_stock_queue_v7_ready.json';VALIDATOR=ROOT/'scripts/validate_research_archived_roles_v1.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fresh():
 from scripts import validate_research_archived_roles_v1 as f
 return f
def source_binding():
 if sha(ROOT/V6)!=V6_SHA or sha(V6_READY)!=V6_READY_SHA:raise ValueError('Frozen V6 controller changed')
 value=json.loads(V6_READY.read_bytes())['source_sha256']
 if any(sha(ROOT/n)!=h for n,h in value.items()):raise ValueError('Frozen V6 dependency changed')
 return {**value,**fresh().source_binding(),str(V6_READY.relative_to(ROOT)):V6_READY_SHA,str(fresh().READY.relative_to(ROOT)):sha(fresh().READY),SOURCE:sha(ROOT/SOURCE),TEST:sha(ROOT/TEST)}
def readiness():
 value=json.loads(READY.read_bytes());fresh().readiness()
 if value.get('owner')!=OWNER or value.get('status')!='CPU_route_tests_passed_fresh_role_GPU_pending' or value.get('source_sha256')!=source_binding():raise ValueError('Exact fresh-role V7 controller release required')
 return {'path':str(READY),'sha256':sha(READY)}
def arg(command,name):return command[command.index(name)+1]
class _Spawn:
 def __init__(self,original,live):self.original=original;self.live=live
 def __getattr__(self,name):return getattr(self.original,name)
 def Popen(self,command,*a,**k):
  script=Path(command[2]).resolve() if len(command)>2 and command[1]=='-u' else None
  if script!=VALIDATOR.resolve():return self.original.Popen(command,*a,**k)
  queue=self.live['queue'];queue._guard()
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or Path(command[0]).resolve()!=Path(sys.executable).resolve() or arg(command,'--numerical-profile')!='stock_deterministic_small_v1' or '--allow-gpu' not in command or arg(command,'--device')!='cuda':raise ValueError('Exact CPU-parent fresh-role replay command required')
  run=Path(arg(command,'--run')).resolve();jobs=[j for j in queue.spec['jobs'] if Path(arg(j['argv'],'--output')).resolve()==run]
  if len(jobs)!=1 or not queue._routed(jobs[0]):raise ValueError('Fresh replay only for current registered completed archives')
  environment=dict(os.environ if k.get('env') is None else k['env']);environment['CUDA_VISIBLE_DEVICES']='';k['env']=environment
  # This CPU orchestration child owns explicit fresh physical admission and
  # retained/anchor advice before EACH GPU worker. Original GPU policy unchanged.
  return subprocess.Popen(command,*a,**k)
def load_controller(proof_sha):
 source_binding();v6=types.ModuleType('_fresh_replay_private_v6');v6.__file__=str(ROOT/V6);exec(compile((ROOT/V6).read_bytes(),v6.__file__,'exec'),v6.__dict__)
 module=v6.load_controller(proof_sha);original_class=module.queue_class;old_command=module.stock.command_profile;live={}
 def command(argv):
  result=old_command(argv)
  if len(result)>2 and Path(result[2]).name=='validate_research_stock_deterministic.py' and 'queue' in live:
   run=Path(arg(result,'--run')).resolve();jobs=[j for j in live['queue'].spec['jobs'] if Path(arg(j['argv'],'--output')).resolve()==run]
   if len(jobs)==1 and live['queue']._routed(jobs[0]):result[2]=str(VALIDATOR)
  return result
 module.stock.command_profile=command
 def queue_class():
  parent=original_class();seen=set()
  for cls in parent.__mro__:
   for value in cls.__dict__.values():
    if callable(value) and hasattr(value,'__globals__') and 'Popen' in value.__code__.co_names and id(value.__globals__) not in seen:
     g=value.__globals__;seen.add(id(g));g['subprocess']=_Spawn(g['subprocess'],live)
  class FreshReplayQueue(parent):
   def __init__(self,*a,**k):
    super().__init__(*a,**k)
    try:
     descriptor={'owner':OWNER,'controller_source':SOURCE,'controller_sha256':sha(ROOT/SOURCE),'release':readiness(),'validator_source':str(VALIDATOR),'validator_release':fresh().readiness(),'validator_orchestrator_CUDA_VISIBLE_DEVICES':'','each_fresh_role_worker_CUDA_VISIBLE_DEVICES':'0','same_SPEC_state_training_stock_math_and_immutable_publication':True,'pre_each_worker_advice':'unchanged owned retained+exactinactiveSmallanchors; actualhead/no cachecredit'}
     extra=self.state.setdefault('extra',{});existing=extra.get(OWNER)
     if existing is not None and existing!=descriptor:raise ValueError('Fresh replay controller profile changed')
     previous=dict(self.state['numerical_execution']);history=self.state.setdefault('controller_execution_history',[])
     if previous not in history:history.append(previous)
     extra[OWNER]=descriptor;self.state['numerical_execution'].update(controller_source=SOURCE,controller_sha256=descriptor['controller_sha256'],fresh_role_replay_owner=OWNER);live['queue']=self
    except BaseException:self.lock_handle.close();raise
   def _guard(self):readiness();return super()._guard()
   def _validate_checkpoint_report(self,identity):
    result=super()._validate_checkpoint_report(identity)
    jobs=[j for j in self.spec['jobs'] if Path(arg(j['argv'],'--output')).resolve()==identity['run']]
    if len(jobs)==1 and self._routed(jobs[0]):result['fresh_role_replay']=fresh().validate_completed_report(identity['run'])
    return result
  return FreshReplayQueue
 module.queue_class=queue_class;return module
def main(argv=None):
 if 'torch' in sys.modules:raise RuntimeError('Fresh V7 parent requires fresh process before Torch import')
 os.environ['CUDA_VISIBLE_DEVICES']='';p=argparse.ArgumentParser(add_help=False);p.add_argument('--gpu-proof-sha');a,remaining=p.parse_known_args(argv)
 return load_controller(a.gpu_proof_sha).main(remaining)
if __name__=='__main__':raise SystemExit(main())
