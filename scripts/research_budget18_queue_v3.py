#!/usr/bin/env python3
"""Resource32 controller V3: native replay CLI receives no unsupported reserve flag.

The immutable42-job intent/state and three historical manual stages remain v1.
Future actual argv/profile/worker source are separately recorded. No model math
or frozen source is patched; the reused controller methods have private globals.
"""
from __future__ import annotations
import argparse,ast,contextlib,fcntl,gc,hashlib,importlib,inspect,json,os,signal,sys,textwrap,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE='scripts/research_budget18_queue_v1.py';BASE_SHA='9bb6f7d22939d8fbb280766dd953debc31c581a3b20a7cff9a58e93e4cae745d'
if hashlib.sha256((ROOT/BASE).read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Frozen budget18queuev1 changed')
from scripts import research_budget18_queue_v1 as old
OWNER='research_budget18_queue_resource32_v3';MODULES={k:'src.utils.research_'+label+'_budget_v3' for k,label in [('smallthinker4','small'),('lfm25','lfm'),('giga31','giga')]}
MANUAL=set(old.MANUAL_SERVICES);_RESOURCE=None
INVENTORY='research_results/model_inventory.json';INVENTORY_SHA='452698c4a570fa4da1e81fe3f5b79ed23a22ed77a990ad99424460c22ebe39a0'
SCAN_SOURCE='scripts/research_archival_stock_queue_v5.py';SCAN_SHA='d112d49616139f2b517de7ab3dc57cf1b693dc8e333d87c87476500d865a99b0'
HF_METADATA={
 'smallthinker4':('a24932bb880c85c04c794283202a9194b1b42709',{'config.json':'40adb346175e86f3840b0b16224acce3f01d0002b2bba76ffd2818f3fce1e876','model.safetensors.index.json':'52e7f8942e0237e0b35d793c375b5c15a57a455a51fafe1b0110e91d52529f5d'}),
 'lfm25':('dd83f5cb02aa2938b66bba9a9998d38d0517c1f3',{'config.json':'9c0255c2d5c744c99b760a12edca2572935348dae340e79e2e6625af975d2d68'}),
 'giga31':('189fff27a1dee68473960c3d5bca53e0e07a3191',{'config.json':'6a6b8260f08791c4968f70934903e9aa892a53ee3f2e2e05b61a68ea1ff33503','model.safetensors.index.json':'1f08c4954c2e8319ad6abe35f9874cd28c4e74bb8cc589984e31bf9f0267c1fa'})}
HF_BLOBS={
 'smallthinker4':('f3b04799c8eb96f944c2a016ece300aeb6f77379fc21ea3c39d8de34a55b9416','70bc6c00e97e2f3c90ab5e4d80b15f6256f544f99fc507aa5d75ed0b82553c57'),
 'lfm25':('8f73bfeecc3db3e37fe983738805e9421d5ed73a0099179026f284b012a43e9e',),
 'giga31':('b99f460ca4318e5608d84c132f4e4100e0bc8ddef72f242e596e48e326429fec','daaf8a540f4eb407bd66fa841f1f4f5907234f0ce71e37a195f94de4cf83dba4','f4ced83a707a3c932d32079fbc59a803fd6eb138735309ecab8dd05763f912c7','b929fd3067bc0527a723f9e9812de3e03ef032b5005c20dd92861e5a1dee4b89','15c2095ae455e39ec6688a74466fb6d6565a0ed6f128e97ad025d665b83c6a05','7dfc56894c87686bf8fd9713d5306d9af1c9477aeff74d576a04ca269e0df68b')}

def physical(path):
 s=Path(path).stat();return {k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}

def gpu_idle():
 result=old.subprocess.run(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
 if result.stdout.strip():raise RuntimeError('GPU occupied; inactive-file advice refused')

@contextlib.contextmanager
def advice_candidates(queue):
 """Only retained completed owned budget18 objects and exact pinned HF weights."""
 if old.sha(ROOT/INVENTORY)!=INVENTORY_SHA:raise ValueError('Pinned original-model inventory changed')
 paths={};bindings={}
 with contextlib.ExitStack() as stack:
  for job in queue.spec['jobs']:
   row=queue.state['jobs'].get(job['id'],{});out=old.directory(job)
   if not row.get('training_completed') or row.get('checkpoint_pruned'):continue
   if old.live_external(job):raise ValueError('Completed checkpoint has live external trainer')
   for name in ('resolved_config.json','completed.json','.checkpoint_manager.json'):
    p=out/name
    if p.is_symlink() or not p.is_file():raise ValueError('Completed owned source metadata missing')
    bindings[p]=old.sha(p)
   config=old.read(out/'resolved_config.json');completion=old.read(out/'completed.json');owner=old.read(out/'.checkpoint_manager.json')['owner']
   if config.get('model')!=old.MODELS[job['backend']] or config.get('stage')!=job['stage'] or config.get('recipe')!=job['recipe'] or completion.get('steps')!=job['expected_steps'] or completion.get('records')!=job['expected_records']:raise ValueError('Completed advice source differs from exact budget18 job')
   lock=stack.enter_context((out/'.checkpoint.lock').open('rb'));fcntl.flock(lock,fcntl.LOCK_SH)
   for role in ('best','last'):
    pointer=out/role
    if not pointer.is_symlink():raise ValueError('Advice requires owned role pointer')
    target=pointer.resolve(strict=True);marker=target/'.research_checkpoint.json'
    if target.parent!=out.resolve()/'.checkpoint_objects' or marker.is_symlink() or old.read(marker).get('owner')!=owner:raise ValueError('Advice refuses foreign checkpoint')
    bindings[marker]=old.sha(marker);p=target/'state.pt'
    if p.is_symlink() or not p.is_file():raise ValueError('Advice source must be ordinary owned state')
    paths[p]=physical(p)
  inventory={r['repo']:r for r in old.read(ROOT/INVENTORY)}
  for family,(revision,metadata) in HF_METADATA.items():
   repo=old.MODELS[family];entry=inventory[repo]
   if entry['sha']!=revision:raise ValueError('Advice refuses unpinned model revision')
   snapshot=Path('/workspace/.hf_home/hub')/('models--'+repo.replace('/','--'))/'snapshots'/revision
   for name,digest in metadata.items():
    p=snapshot/name
    if old.sha(p)!=digest:raise ValueError('Pinned model metadata changed')
    bindings[p]=digest
   weights={r['path']:r['bytes'] for r in entry['files'] if r['path'].endswith('.safetensors')}
   if 'model.safetensors.index.json' in metadata and set(old.read(snapshot/'model.safetensors.index.json')['weight_map'].values())!=set(weights):raise ValueError('Pinned weight index differs')
   for (name,size),blob in zip(sorted(weights.items()),HF_BLOBS[family],strict=True):
    p=(snapshot/name).resolve(strict=True)
    if p!=Path('/workspace/.hf_home/hub/blobs')/blob[:2]/blob or p.is_symlink() or not p.is_file() or p.stat().st_size!=size:raise ValueError('Pinned weight physical path/size differs')
    paths[p]=physical(p)
  yield paths,bindings

def advise_inactive(queue,*,candidates=None,scan=None,advisor=None,guard=None):
 """Read-only inactive cache advice; actual no-credit host admission follows."""
 from src.utils.research_lossless_delta import memory_guard
 from src.utils.research_archive_gpu_parity import host_snapshot
 if old.sha(ROOT/SCAN_SOURCE)!=SCAN_SHA:raise ValueError('Pinned all-UID fd/maps scanner changed')
 from scripts.research_archival_stock_queue_v5 import scan_anchor_users
 guard=guard or memory_guard;scan=scan or scan_anchor_users
 if queue.child is not None:raise ValueError('Advice requires no owned worker handle')
 gpu_idle();gc.collect();before=host_snapshot()
 with (candidates or advice_candidates)(queue) as (paths,bindings):
  ids={(s['st_dev'],s['st_ino']) for s in paths.values()};first=scan(ids)
  for p,s in paths.items():
   if physical(p)!=s:raise ValueError('Inactive source changed before advice')
   fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
   try:
    if physical(p)!=s or {k:getattr(os.fstat(fd),k) for k in s}!=s:raise ValueError('Inactive source descriptor identity differs')
    (advisor or os.posix_fadvise)(fd,0,0,os.POSIX_FADV_DONTNEED)
   finally:os.close(fd)
   if physical(p)!=s:raise ValueError('Inactive source changed during advice')
  second=scan(ids)
  if any(old.sha(p)!=digest for p,digest in bindings.items()):raise ValueError('Inactive source role/config metadata changed')
  gpu_idle();after=guard(0)
  return {'status':'readonly_inactive_file_advice_completed','files':[{'path':str(p),'stat':s} for p,s in paths.items()],'source_metadata_sha256':{str(p):h for p,h in bindings.items()},'scan_before':first,'scan_after':second,'host_before':before,'host_after':after,'assumed_cache_credit_bytes':0,'files_or_pointers_or_weights_modified':False}

def resources(*,idle=False):
 from src.utils.research_storage import DiskBudget
 from src.utils.research_lossless_delta import memory_guard
 disk=DiskBudget(ROOT,32).ensure_space(0);host=memory_guard(0)
 if idle:
  result=old.subprocess.run(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
  if result.stdout.strip():raise RuntimeError('GPU occupied; no concurrent resource32worker launched')
 return {'disk':disk,'host':host}

def verify_resource(binding):
 if not isinstance(binding,dict) or old.sha(binding['path'])!=binding['sha256']:raise ValueError('Explicit immutable resourcev3 READY SHA required')
 ready=old.read(binding['path'])
 if ready.get('status')!='passed':raise ValueError('Resourcev3 CPU tests must be passed')
 sources=ready.get('source_sha256',{})
 if not all(str(Path(m.replace('.','/')+'.py')) in sources for m in MODULES.values()):raise ValueError('Resource readiness omits backendsource')
 for path,digest in sources.items():
  if old.sha(ROOT/path)!=digest:raise ValueError('Resource/backend source changed: '+path)
 prerequisites=ready.get('prerequisites',{})
 items=list(prerequisites.values()) if isinstance(prerequisites,dict) else prerequisites
 bypath={str((ROOT/item['path']).resolve()):item for item in items}
 required={str((ROOT/'research_results/budget18_cache_actual_retirement_v1/completed.json').resolve()):{'status':'completed','files_retired':17981},str((ROOT/'configs/research/completed_r2_finalization_v1_ready.json').resolve()):{'status':'passed'}}
 for path,expected in required.items():
  item=bypath.get(path)
  if not item or old.sha(path)!=item.get('sha256'):raise ValueError('Resource readiness prerequisite bytes missing/changed')
  proof=old.read(path)
  if any(proof.get(k)!=v for k,v in expected.items()):raise ValueError('Cache retirement or completedR2 prerequisite not complete')
 return ready

def actual_argv(job,kind,argv):
 result=list(argv)
 if job['id'] in MANUAL:return result
 def replace(flag,item):
  if flag in result:result[result.index(flag)+1]=item
  else:result.extend((flag,item))
 if kind=='validate':
  if result.count('--reserve-gb')>1:raise ValueError('Duplicate unsupported validator reserve')
  if '--reserve-gb' in result:
   index=result.index('--reserve-gb');del result[index:index+2]
 else:replace('--reserve-gb','32')
 module=importlib.import_module(MODULES[job['backend']]);profile=old.PROFILES['smallthinker4'] if job['backend']=='smallthinker4' else module.definition()['name'];replace('--numerical-profile',profile)
 return result

def expected_profile(family,recipe=None):
 if family=='smallthinker4':return old.expected_profile(family,recipe)
 return importlib.import_module(MODULES[family]).definition()

def backend_module(family,recipe):return MODULES[family]

def _function(original,scope):
 return types.FunctionType(original.__code__,scope,original.__name__,original.__defaults__,original.__closure__)

def _compile_function(original,scope,transformer):
 tree=ast.parse(textwrap.dedent(inspect.getsource(original)));tree=transformer.visit(tree);ast.fix_missing_locations(tree);exec(compile(tree,str(ROOT/BASE),'exec'),scope);return scope[original.__name__]

class SmallImports(ast.NodeTransformer):
 def visit_ImportFrom(self,node):
  if node.module in ('src.utils.research_small_budget_v1','src.utils.research_small_budget_v2'):node.module='src.utils.research_small_budget_v3'
  if node.module=='src.utils':
   for item in node.names:
    if item.name=='research_small_budget_v1':item.name='research_small_budget_v3'
  return node
class Reserve(ast.NodeTransformer):
 def visit_Constant(self,node):return ast.copy_location(ast.Constant(32 if type(node.value)is int else '32'),node) if node.value==120 or node.value=='120' else node
class Requests(ast.NodeTransformer):
 def visit_Assign(self,node):
  if any(isinstance(t,ast.Name) and t.id=='req' for t in node.targets) and isinstance(node.value,ast.Dict):
   node.value.keys.extend([ast.Constant('resource_ready'),ast.Constant('job_id'),ast.Constant('source_intent_argv')]);node.value.values.extend([ast.Attribute(ast.Name('self',ast.Load()),'resource_binding',ast.Load()),ast.Subscript(ast.Name('job',ast.Load()),ast.Constant('id'),ast.Load()),ast.Subscript(ast.Name('job',ast.Load()),ast.Constant('argv'),ast.Load())])
  return self.generic_visit(node)
class Worker(ast.NodeTransformer):
 def visit_If(self,node):
  if ast.unparse(node.test)=="family == 'smallthinker4'":
   node.body=ast.parse("module=importlib.import_module(backend)\nreturn (module.eval_main if kind=='eval' else module.validate_main)(['--allow-gpu',*argv])").body;node.orelse=[];return node
  return self.generic_visit(node)

SCOPE=dict(old.__dict__);SCOPE.update(__file__=str(Path(__file__).resolve()),resources=resources,expected_profile=expected_profile,backend_module=backend_module)
_future_completed=_compile_function(old.validate_completed,SCOPE,SmallImports())
def validate_completed(job):
 if job['id'] in MANUAL:return old.validate_completed(job)
 config=old.read(old.directory(job)/'resolved_config.json')
 if config.get('reserve_gb')!=32:raise ValueError('Future resource32 run must record actual32GiB reserve')
 if job['stage']=='SFT' and job['recipe']=='4f':
  from src.utils.research_completed_r2_v1 import definition
  if config.get('completed_r2_execution_policy')!=definition():raise ValueError('FutureR2 lacks exact completed-model-only execution policy')
 return _future_completed(job)
SCOPE['validate_completed']=validate_completed
_original_verify=old.verify_plan
def verify_plan(spec):_original_verify(spec);verify_resource(_RESOURCE);return spec
SCOPE['verify_plan']=verify_plan
_execute=_compile_function(old.Queue.execute,SCOPE,Requests())
_postprocess=_compile_function(old.Queue.postprocess,SCOPE,Reserve())
_manual_postprocess=_function(old.Queue.postprocess,SCOPE)
_run=_function(old.Queue.run,SCOPE);_adopt=_function(old.Queue.adopt_or_train,SCOPE)
OLD_SCOPE=dict(SCOPE,backend_module=old.backend_module,small_validator=old.small_validator)
_old_worker=_function(old.worker,OLD_SCOPE)
_new_worker=_compile_function(old.worker,SCOPE,Worker())

def worker(path,args):
 req=old.read(path);binding=req.get('resource_ready');verify_resource(binding)
 if req.get('job_id') not in {j['id'] for j in old.jobs()}:raise ValueError('Worker job not in immutable42-job intent')
 SCOPE['ARGS']=args;OLD_SCOPE['ARGS']=args
 return (_old_worker if req['job_id'] in MANUAL else _new_worker)(path)

class Queue(old.Queue):
 def __init__(self,spec,state,retry=False,resource_binding=None):
  global _RESOURCE
  _RESOURCE=resource_binding;verify_resource(_RESOURCE);super().__init__(spec,state,retry);self.resource_binding=dict(_RESOURCE)
  self.oracle.spec=dict(self.spec,reserve_gb=32)
  from src.utils.research_storage import DiskBudget
  self.oracle.budget=DiskBudget(ROOT,32)
  receipt={'owner':OWNER,'source_sha256':old.sha(__file__),'base_source_sha256':BASE_SHA,'resource_ready':self.resource_binding,'future_disk_reserve_gib':32,'host_headroom_gib':32,'manual_stages_preserve_source_reserve120':sorted(MANUAL),'prune_order':'A1 waits until its A0 parent checkpoint is pruned'}
  existing=self.state.setdefault('controller_revisions',[])
  if receipt not in existing:existing.append(receipt)
  self.persist()
 def execute(self,job,kind,argv,row):
  if job['id'] not in MANUAL:
   row.setdefault('inactive_file_advice',[]).append(advise_inactive(self));row['inactive_file_advice']=row['inactive_file_advice'][-16:];self.persist()
  actual=actual_argv(job,kind,argv);row.setdefault('actual_execution_history',[]).append({'kind':kind,'source_intent_argv':list(job['argv']),'actual_argv':actual,'resource_ready':self.resource_binding,'controller_source_sha256':old.sha(__file__)})
  self.persist();return _execute(self,job,kind,actual,row)
 def adopt_or_train(self,job,row):
  if job['stage']=='SFT' and job['recipe']=='4f':
   from src.utils.research_completed_r2_v1 import recover_if_published
   if recover_if_published(old.directory(job)):
    row.update(training_completed=True,model_only_publication_recovered_without_optimizer_updates=True);self.persist()
  return _adopt(self,job,row)
 def postprocess(self,job,row):return (_manual_postprocess if job['id'] in MANUAL else _postprocess)(self,job,row)
 run=_run
 def prune(self):
  from scripts.research_queue import prune_checkpoint_run
  for job in self.spec['jobs']:
   row=self.state['jobs'].get(job['id'],{})
   if not job['discard_checkpoint_after_dependents'] or row.get('status')!='completed' or row.get('checkpoint_pruned'):continue
   if job['stage']=='A1' and any(not self.state['jobs'].get(p,{}).get('checkpoint_pruned') for p in job['requires']):continue
   consumers=[j for j in self.spec['jobs'] if job['id'] in j['requires']]
   if consumers and all(self.state['jobs'].get(j['id'],{}).get('status')=='completed' for j in consumers):
    row['prune_receipt']=prune_checkpoint_run(old.directory(job),[old.directory(j) for j in consumers]);row['checkpoint_pruned']=True;self.persist()

def main():
 global _RESOURCE
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,default=old.PLAN);p.add_argument('--state',type=Path,default=ROOT/'runs/research/budget18_queue_state_v1.json');p.add_argument('--resource-ready',type=Path);p.add_argument('--resource-ready-sha');p.add_argument('--run-queue',action='store_true');p.add_argument('--retry-failed',action='store_true');p.add_argument('--worker-request',type=Path);p.add_argument('--request-sha256');args=p.parse_args()
 if args.worker_request:worker(args.worker_request,args);return 0
 if args.resource_ready is None or not args.resource_ready_sha:p.error('Explicit --resource-ready and --resource-ready-sha required')
 _RESOURCE={'path':str(args.resource_ready.resolve(strict=True)),'sha256':args.resource_ready_sha};verify_plan(old.read(args.spec))
 if not args.run_queue:print(json.dumps({'status':'metadata_only_resource32_V3','jobs':42,'source':old.sha(__file__),'resource_ready':_RESOURCE}));return 0
 queue=Queue(args.spec,args.state,args.retry_failed,_RESOURCE);signal.signal(signal.SIGTERM,queue.signal);signal.signal(signal.SIGINT,queue.signal);return queue.run()
if __name__=='__main__':raise SystemExit(main())
