#!/usr/bin/env python3
"""Future4-branch lossless archival overlay for the unchanged stock10-job queue."""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,os
from pathlib import Path
import signal,sys
from types import ModuleType
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_stock_pilot_queue as stock
from scripts.research_queue import argv_value,output_dir,artifacts_for,timestamp,emit
from src.utils import research_archive_stock_v2 as contract
OWNER='prospective_small_archival_stock_queue_v2'
SCRIPT='scripts/research_archival_stock_queue_v2.py'
TEST='tests/test_research_archival_stock_queue_v2.py'
READY=ROOT/'configs/research/archival_stock_queue_v2_ready.json'
SPEC=ROOT/'configs/research/smallthinker_stock_deterministic_pilot_v1.json'
STATE=SPEC.with_suffix('.state.json')
SPEC_SHA='2690cc406441b0c0a8b81fd52f572ca591f504077433c5ab6def4cb896ede995'
ELIGIBLE=('3c','2a','2b','2c')
PINS={'scripts/research_stock_pilot_queue.py':'46ae4aa3fead95c31513d8d0176ceb2fd44a391aed282e603ae9280796fe639c',
 'src/utils/research_checkpoint_archival_queue.py':'f16d1620975a23bc30315811f98daf5d1bf060a4591e9dd41ba3fd8ffc563ef1',
 'scripts/research_archival_parallel_queue_v1.py':'875cbd0c08cdfb08a171364b727121e69e916131e4ea7c7c5fea72f699eae2dc',
 'src/utils/research_archive_stock_v2.py':'0e9baa74cdb950088debb96c40e198550ecf9e814db8cc2477e0c0adfebe31a3'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify_sources():
 if any(sha(ROOT/n)!=h for n,h in {**PINS,**stock.PINS}.items()):raise ValueError('Frozen stock/archive queue dependency changed')
def source_binding():
 verify_sources();return {**PINS,**stock.PINS,SCRIPT:sha(ROOT/SCRIPT),TEST:sha(ROOT/TEST),**stock.stock.definition()['source_sha256']}
def readiness():
 value=read(READY)
 if value.get('owner')!=OWNER or value.get('status')!='cpu_tests_passed_actual_activation_gate_closed' or value.get('source_sha256')!=source_binding():raise ValueError('Exact archival stock controller CPU release required')
 return {'path':str(READY),'sha256':sha(READY),'source_sha256':value['source_sha256']}
def spec_snapshot(path):
 path=Path(path).resolve(strict=True)
 if path!=SPEC.resolve() or sha(path)!=SPEC_SHA:raise ValueError('Only same immutable stock10-job SPEC accepted')
 value=read(path);stock.verify_inputs(value)
 if 'trained_small_archival_activation' in value or any('archive_completed_small' in job for job in value['jobs']):raise ValueError('Archival controls must remain outside SPEC/argv')
 return value
def activation(expected_sha,publisher):
 if not isinstance(expected_sha,str) or len(expected_sha)!=64 or any(x not in '0123456789abcdef' for x in expected_sha):raise ValueError('Explicit actual v2 activation SHA required')
 value=publisher.activation_evidence()
 if value['path']!=str(contract.ACTIVATION.resolve()) or value['sha256']!=expected_sha:raise ValueError('Fixed new activation path/SHA differs')
 return value

def _owned_retained(job):
 """Bounded metadata only. No nested manager lock, hashes or tensor reads."""
 from src.utils import research_trained_delta_probe as probe
 run=output_dir(job['argv'])
 if run.is_symlink():raise ValueError('Advice run must be its actual owned directory')
 config=contract.read(run/'resolved_config.json');done=contract.read(run/'completed.json');manager=contract.read(run/'.checkpoint_manager.json')
 expected=Path(config.get('output',''));expected=(expected if expected.is_absolute() else ROOT/expected).resolve()
 if (expected!=run.resolve() or config.get('model')!=stock.stock.SMALL or config.get('stage')!=argv_value(job['argv'],'--stage') or
  config.get('recipe')!=argv_value(job['argv'],'--recipe') or type(done.get('steps')) is not int or done['steps']<1 or
  done.get('status') in ('failed','incomplete','interrupted') or manager.get('mode') not in ('min','max') or not isinstance(manager.get('owner'),str) or not manager['owner']):
  raise ValueError('Completed source metadata differs from exact stock job')
 objects=run/'.checkpoint_objects'
 if objects.is_symlink() or not objects.is_dir():raise ValueError('Completed advice objects must be manager owned')
 roles={}
 for role in ('best','last'):
  pointer=run/role
  if not pointer.is_symlink():raise ValueError('Advice needs both retained manager pointers')
  target=pointer.resolve(strict=True)
  if target.parent!=objects.resolve() or not target.name.startswith('object-'):raise ValueError('Advice role escaped manager objects')
  marker=contract.read(target/'.research_checkpoint.json');path=target/'state.pt'
  if (marker.get('owner')!=manager['owner'] or type(marker.get('step')) is not int or not 0<marker['step']<=done['steps'] or
    (role=='last' and marker['step']!=done['steps']) or path.is_symlink() or not path.is_file()):raise ValueError('Advice state ownership/step differs')
  roles[role]={'path':str(path),'stat':probe.stat(path),'marker':marker,'marker_sha256':sha(target/'.research_checkpoint.json'),'pointer':os.readlink(pointer)}
 return {'run':str(run),'roles':roles,'metadata_sha256':{n:sha(run/n) for n in ('resolved_config.json','completed.json','.checkpoint_manager.json')}}

def advise_completed(queue,current_run,*,selected_source=None,lookup=None,advisor=None,gpu=None,host=None,process=None):
 """Read-only eviction, after reaped trainer + idle GPU; measured head, zero credit."""
 from src.utils import research_trained_delta_probe as probe,research_archive_gpu_parity as parity
 from scripts import probe_research_parallel_b1 as proc
 lookup=lookup or _owned_retained;advisor=advisor or probe.advise_source_cache;gpu=gpu or proc.gpu_snapshot;host=host or parity.host_snapshot;process=process or proc.process_info
 jobs=queue.spec['jobs'];matches=[i for i,j in enumerate(jobs) if output_dir(j['argv'])==Path(current_run).resolve()]
 if len(matches)!=1:raise ValueError('Advice current run must belong to exact stock SPEC')
 record=queue.state['jobs'].get(jobs[matches[0]]['id'],{})
 if record.get('training_completed') is not True or record.get('returncode')!=0 or (record.get('pid') and process(record['pid']) is not None):
  raise ValueError('Current trainer must be completed, successful and reaped before advice')
 if gpu()['processes']:raise ValueError('Cache advice requires idle GPU before publication')
 candidates=[]
 for job in jobs[:matches[0]+1]:
  run=output_dir(job['argv']);row=queue.state['jobs'].get(job['id'],{})
  if row.get('checkpoint_pruned'):
   artifacts_for(job,allow_pruned=True);continue
  if not (run/'completed.json').exists():continue
  if row.get('training_completed') is not True and row.get('status')!='completed':raise ValueError('Completed cache candidate lacks durable queue completion')
  candidates.append((job,lookup(job)))
 before={j['id']:value for j,value in candidates};unique={}
 for job,value in candidates:
  for role,item in value['roles'].items():
   path=Path(item['path']);probe._no_source_mappings(path)
   if path in unique and unique[path]['stat']!=item['stat']:raise ValueError('Ambiguous shared advice file')
   unique.setdefault(path,{'stat':item['stat'],'roles':[]})['roles'].append({'job':job['id'],'role':role})
 if selected_source is not None and Path(selected_source).resolve() not in unique:raise ValueError('Post-journal source is not a retained owned role')
 initial=host();events=[]
 for path,item in unique.items():events.append({'path':str(path),'roles':item['roles'],'advice':advisor(path,item['stat'])})
 after={j['id']:lookup(j) for j,_ in candidates}
 if before!=after or gpu()['processes']:raise ValueError('Cache advice metadata/roles/stat or idle GPU changed')
 final=host();available=parity.conservative_host_headroom(final)
 if available<32*2**30:raise MemoryError('Actual host headroom remains below32GiB after readonly advice')
 return {'scope':'completed_current_and_prior_stock_SPEC_owned_roles','selected_source':str(selected_source) if selected_source else None,
  'events':events,'unique_owned_files':len(unique),'metadata_roles_stat_unchanged':True,'own_mmaps_absent':True,'GPU_idle':True,
  'host_before':initial,'host_after':final,'actual_available_after_bytes':available,'reclaimable_cache_credit_assumed_bytes':0,
  'original_encoding_memory_guard_still_required':True,'checkpoint_bytes_or_pointers_changed':False}

def _behavior(publisher,base):
 """Reuse frozen60-line mixin and strict NoBenefit/retry bodies privately."""
 verify_sources();name='src/utils/research_checkpoint_archival_queue.py'
 module=ModuleType('_archival_stock_private_mixin');module.__file__=str(ROOT/name)
 tree=ast.parse((ROOT/name).read_bytes());replaced=0
 for i,node in enumerate(tree.body):
  if isinstance(node,ast.ImportFrom) and node.module=='src.utils' and any(x.name=='research_checkpoint_archival' for x in node.names):
   tree.body[i]=ast.Assign([ast.Name('archival',ast.Store())],ast.Name('_publisher',ast.Load()));replaced+=1
 if replaced!=1:raise ValueError('Pinned archival mixin import changed')
 module.__dict__['_publisher']=publisher;exec(compile(ast.fix_missing_locations(tree),str(ROOT/name),'exec'),module.__dict__)
 old=ast.parse((ROOT/'scripts/research_archival_parallel_queue_v1.py').read_bytes())
 cls=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=='ArchivalParallelResearchQueue')
 methods=[copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_archival_job','_validate_skip','compact_completed_job','evaluate_completed_job')]
 if len(methods)!=4:raise ValueError('Pinned NoBenefit behavior changed')
 for n in methods:
  if n.name in ('compact_completed_job','evaluate_completed_job'):n.body.insert(0,ast.parse('self._registered(job)').body[0])
  if n.name=='_validate_skip':
   for i,statement in enumerate(n.body):
    if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='journal' for t in statement.targets):
     n.body.insert(i+1,ast.parse("if journal.get('protocol') != archival.OWNER: raise ValueError('NoBenefit journal owner differs')").body[0]);break
 behavior=ast.ClassDef(name='_ArchivalBehavior',bases=[ast.Name('_mixin',ast.Load()),ast.Name('_base',ast.Load())],keywords=[],body=methods,decorator_list=[])
 rawroles=copy.deepcopy(next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='_native_raw_roles'))
 namespace=dict(globals(),archival=publisher,_mixin=module.ProspectiveSmallArchivalMixin,_base=base)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[rawroles,behavior],type_ignores=[])),'<stock_v2_archival_behavior>','exec'),namespace)
 return namespace['_ArchivalBehavior']

def queue_class():
 publisher=contract.publisher_v2();live={};original_publish=publisher.publish_completed_checkpoint;original_derive=publisher._derive_unlocked
 def cache(run,source=None):
  queue=live['queue'];receipt=advise_completed(queue,run,selected_source=source)
  history=queue.state.setdefault('archival_cache_advice',[]);history.append(receipt);del history[:-16]
  queue.persist();emit('archival_scoped_cache_advice',run=str(run),unique_owned_files=receipt.get('unique_owned_files'),actual_available_after_bytes=receipt.get('actual_available_after_bytes'),reclaimable_cache_credit_assumed_bytes=0);return receipt
 def publish(run,**kwargs):
  cache(run)
  kwargs.setdefault('progress',lambda phase,**details:emit('archival_progress',run=str(run),phase=phase,**details))
  return original_publish(run,**kwargs)
 def derive(source,stage,plan,*args,**kwargs):
  cache(plan['run'],source);return original_derive(source,stage,plan,*args,**kwargs)
 publisher.publish_completed_checkpoint=publish;publisher._derive_unlocked=derive
 base=stock.queue_class();behavior=_behavior(publisher,base)
 class ArchivalStockQueue(behavior):
  def __init__(self,spec_path,state_path=None,retry_failed=False,*,activation_sha):
   if state_path is not None and Path(state_path).resolve()!=STATE.resolve():raise ValueError('Same existing stock state path required')
   bound=activation(activation_sha,publisher);ready=readiness();original=spec_snapshot(spec_path)
   self.activation_sha=activation_sha;self.activation_bound=bound;self.archival_ready=ready
   super().__init__(spec_path,state_path,retry_failed)
   try:
    if self.spec!=original:raise ValueError('Parent altered stock SPEC')
    disk=read(self.state_path) if self.state_path.exists() else {}
    previous=copy.deepcopy(disk.get('numerical_execution',{}));history=copy.deepcopy(disk.get('controller_execution_history',[]))
    if not isinstance(history,list):raise ValueError('Controller execution history must be a list')
    descriptor={'owner':OWNER,'controller_source':SCRIPT,'controller_sha256':sha(ROOT/SCRIPT),'spec_file_sha256':SPEC_SHA,
     'spec_sha256':self.spec_hash,'activation':bound,'execution_ready':ready,'eligible_full_recipes':list(ELIGIBLE),
     'raw_authoritative_recipes':['3a'],'transient_job_copy_only':True,'numerical_profile':stock.stock.definition()}
    extra=self.state.setdefault('extra',{});existing=extra.get(OWNER)
    if existing is not None and existing!=descriptor:raise ValueError('Archival controller profile changed')
    if previous and previous.get('controller_sha256')!=descriptor['controller_sha256'] and previous not in history:history.append(previous)
    extra[OWNER]=descriptor;self.state['controller_execution_history']=history
    self.state['numerical_execution'].update(controller_source=SCRIPT,controller_sha256=descriptor['controller_sha256'],archival_execution_owner=OWNER)
    live['queue']=self
   except BaseException:self.lock_handle.close();raise
  def _guard(self):
   if activation(self.activation_sha,publisher)!=self.activation_bound or readiness()!=self.archival_ready:raise ValueError('Actual archival activation/release changed')
   if spec_snapshot(self.spec_path)!=self.spec or digest(self.spec)!=self.spec_hash:raise ValueError('Stock SPEC/argv changed')
  def _registered(self,job):
   if job not in self.spec['jobs']:raise ValueError('Unregistered/modified stock job')
  def _stock_boundary(self,job):
   self._guard();self._registered(job);return super()._stock_boundary(job)
 return ArchivalStockQueue

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,default=SPEC);p.add_argument('--state',type=Path)
 p.add_argument('--activation-sha');p.add_argument('--run-queue',action='store_true');p.add_argument('--retry-failed',action='store_true');p.add_argument('--stop-before-job')
 a=p.parse_args(argv)
 if not a.run_queue:
  try:publisher=contract.publisher_v2();evidence=activation(a.activation_sha,publisher);ready=readiness();spec_snapshot(a.spec)
  except (OSError,ValueError,KeyError) as error:
   print(json.dumps({'status':'closed_preflight','reason':str(error),'jobs_started':False,'signals_sent':False}));return 2
  print(json.dumps({'status':'eligible_metadata_only','activation':evidence,'ready':ready,'jobs_started':False}));return 0
 queue=queue_class()(a.spec,a.state,a.retry_failed,activation_sha=a.activation_sha)
 if a.stop_before_job and a.stop_before_job not in {j['id'] for j in queue.spec['jobs']}:
  queue.lock_handle.close();raise ValueError('Maintenance boundary must name an unchanged stock job')
 queue.stop_before_job=a.stop_before_job
 signal.signal(signal.SIGINT,queue.signal);signal.signal(signal.SIGTERM,queue.signal)
 try:return queue.run()
 except stock.MaintenanceBoundary as boundary:
  print(json.dumps({'status':'awaiting_scheduled_maintenance','next_job':str(boundary),'GPU_child_running':False}));return 0
if __name__=='__main__':raise SystemExit(main())
