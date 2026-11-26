#!/usr/bin/env python3
"""CPU-isolated parent around the unchanged v2 archival stock controller."""
from __future__ import annotations
import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
from types import ModuleType
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OWNER='cpu_isolated_archival_stock_queue_v3'
SOURCE='scripts/research_archival_stock_queue_v3.py'
TEST='tests/test_research_archival_stock_queue_v3.py'
READY=ROOT/'configs/research/archival_stock_queue_v3_ready.json'
V2='scripts/research_archival_stock_queue_v2.py'
V2_SHA='0e61020e84f79e4575aa007f03d9ef7da2fe67ecbd174711242e871a321bea89'
V2_READY=ROOT/'configs/research/archival_stock_queue_v2_ready.json'
V2_READY_SHA='904b8babacb35e1fd4f8408cf33519cd9c887fb74066ccb1a425138c3d868c19'
STOCK='scripts/research_stock_pilot_queue.py'
STOCK_SHA='46ae4aa3fead95c31513d8d0176ceb2fd44a391aed282e603ae9280796fe639c'
GPU_SCRIPTS=('run_research_stock_deterministic.py','eval_research_stock_deterministic.py','validate_research_stock_deterministic.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bootstrap():
 if 'torch' in sys.modules:raise RuntimeError('CPU archival controller requires a fresh process before Torch import')
 os.environ['CUDA_VISIBLE_DEVICES']=''
def assert_cpu():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Archival controller parent must keep CUDA hidden')
 torch=sys.modules.get('torch')
 if torch is not None and torch.cuda.is_initialized():raise RuntimeError('Archival controller initialized CUDA')
def child_environment(command,environment=None):
 assert_cpu();env=dict(os.environ if environment is None else environment)
 if not isinstance(command,(tuple,list)) or not command or any(not isinstance(x,str) for x in command):raise ValueError('Explicit string argv required')
 python=Path(command[0]).resolve()==Path(sys.executable).resolve() or Path(command[0]).name.startswith('python')
 script=command[2] if len(command)>2 and command[1]=='-u' else None
 approved=python and script is not None and Path(script).is_absolute() and Path(script).resolve() in {(ROOT/'scripts'/n).resolve() for n in GPU_SCRIPTS}
 if approved:
  if Path(command[0]).resolve()!=Path(sys.executable).resolve() or '--allow-gpu' not in command or '--numerical-profile' not in command:
   raise ValueError('Known GPU facade requires exact interpreter and explicit stock flags')
  i=command.index('--numerical-profile')
  if i+1==len(command) or command[i+1]!='stock_deterministic_small_v1':raise ValueError('GPU child numerical profile differs')
  if '--cpu-structural-only' in command or any(command[i]=='--device' and command[i+1]=='cpu' for i in range(len(command)-1)):env['CUDA_VISIBLE_DEVICES']=''
  else:env['CUDA_VISIBLE_DEVICES']='0'
 else:
  if python and any(x in ('--allow-gpu','cuda') or x.startswith(('cuda:','--device=cuda')) for x in command):raise ValueError('Unknown Python CUDA command refused')
  env['CUDA_VISIBLE_DEVICES']=''
 return env
class _Subprocess:
 def __init__(self,module):self.module=module
 def __getattr__(self,name):return getattr(self.module,name)
 def Popen(self,command,*a,**k):
  k['env']=child_environment(command,k.get('env'));return self.module.Popen(command,*a,**k)
def source_binding():
 if sha(ROOT/V2)!=V2_SHA or sha(ROOT/STOCK)!=STOCK_SHA or sha(V2_READY)!=V2_READY_SHA:raise ValueError('Frozen v2 controller release changed')
 value=json.loads(V2_READY.read_bytes())['source_sha256']
 if any(sha(ROOT/n)!=h for n,h in value.items()):raise ValueError('Frozen v2 controller dependency changed')
 return {**value,str(V2_READY.relative_to(ROOT)):V2_READY_SHA,SOURCE:sha(ROOT/SOURCE),TEST:sha(ROOT/TEST)}
def readiness():
 value=json.loads(READY.read_bytes())
 if value.get('owner')!=OWNER or value.get('status')!='cpu_isolation_tests_passed' or value.get('source_sha256')!=source_binding():raise ValueError('Exact CPU isolated controller release required')
 return {'path':str(READY),'sha256':sha(READY)}
def load_controller():
 assert_cpu();source_binding();stock=ModuleType('_cpu_stock_private');stock.__file__=str(ROOT/STOCK)
 exec(compile((ROOT/STOCK).read_bytes(),stock.__file__,'exec'),stock.__dict__)
 original=stock.private_queues
 def queues():
  modules=original()
  for m in modules.values():m.subprocess=_Subprocess(subprocess)
  return modules
 stock.private_queues=queues
 m=ModuleType('_cpu_archival_v2_private');m.__file__=str(ROOT/V2);m.__dict__['_cpu_stock']=stock
 tree=ast.parse((ROOT/V2).read_bytes());changed=0
 for i,n in enumerate(tree.body):
  if isinstance(n,ast.ImportFrom) and n.module=='scripts' and any(a.name=='research_stock_pilot_queue' for a in n.names):
   tree.body[i]=ast.Assign([ast.Name('stock',ast.Store())],ast.Name('_cpu_stock',ast.Load()));changed+=1
 if changed!=1:raise ValueError('Pinned v2 stock-controller import changed')
 exec(compile(ast.fix_missing_locations(tree),m.__file__,'exec'),m.__dict__)
 original_class=m.queue_class
 def queue_class():
  parent=original_class()
  class CPUQueue(parent):
   def __init__(self,*a,**k):
    assert_cpu();super().__init__(*a,**k)
    try:
     descriptor={'owner':OWNER,'controller_source':SOURCE,'controller_sha256':sha(ROOT/SOURCE),'parent_CUDA_VISIBLE_DEVICES':'',
      'GPU_child_CUDA_VISIBLE_DEVICES':'0','GPU_child_scripts':list(GPU_SCRIPTS),'v2_source_sha256':V2_SHA,'v2_ready_sha256':V2_READY_SHA,
      'same_SPEC_state_argv_activation_and_training_numerical_profile':True,'release':readiness()}
     extra=self.state.setdefault('extra',{});old=extra.get(OWNER)
     if old is not None and old!=descriptor:raise ValueError('CPU controller isolation profile changed')
     previous=dict(self.state['numerical_execution']);history=self.state.setdefault('controller_execution_history',[])
     if previous not in history:history.append(previous)
     extra[OWNER]=descriptor;self.state['numerical_execution'].update(controller_source=SOURCE,controller_sha256=descriptor['controller_sha256'],process_isolation_owner=OWNER)
    except BaseException:self.lock_handle.close();raise
   def _guard(self):assert_cpu();readiness();return super()._guard()
  return CPUQueue
 m.queue_class=queue_class;return m
def main(argv=None):
 bootstrap();return load_controller().main(argv)
if __name__=='__main__':raise SystemExit(main())
