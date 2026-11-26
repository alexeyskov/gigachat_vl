#!/usr/bin/env python3
"""Fresh native GPU replay process per unique owned archived BEST/LAST object."""
from __future__ import annotations
import argparse,contextlib,fcntl,hashlib,json,math,os,secrets,signal,subprocess,sys,time,zipfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OWNER='fresh_process_archived_role_replay_v1';SOURCE='scripts/validate_research_archived_roles_v1.py';TEST='tests/test_research_archived_roles_v1.py'
READY=ROOT/'configs/research/archived_roles_replay_v1_ready.json';NAMESPACE='validation/archived_roles_replay_v1'
HELPER_READY_SHA='3a16684fd6a0dfd902e9645807b84355c5750d619345a26a5618f16fd00dd7a9'
V5_READY_SHA='47a7cd96ad5c0d3323adb75b555b036ed60331939e6c39aef9873644feda0af1'

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p)
 if p.is_symlink() or not p.is_file() or p.stat().st_size>32*2**20:raise ValueError('Bounded local replay evidence required')
 return json.loads(p.read_bytes())
def atomic(p,v):
 from scripts.probe_research_parallel_b1 import atomic_json
 atomic_json(Path(p),v)
@contextlib.contextmanager
def namespace_lock(run):
 run=Path(run);base=run/NAMESPACE
 for p in (run/'validation',base):
  if p.is_symlink():raise ValueError('Fresh replay namespace may not be a symlink')
 base.mkdir(parents=True,exist_ok=True);marker=base/'owner.json';expected={'owner':OWNER,'run':str(run)}
 if marker.exists():
  if read(marker)!=expected:raise ValueError('Foreign replay namespace preserved')
 elif any(base.iterdir()):raise ValueError('Existing unowned replay namespace preserved')
 else:atomic(marker,expected)
 lock=base/'.lock'
 if lock.is_symlink():raise ValueError('Foreign replay namespace lock')
 with lock.open('a+b') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:yield base
  finally:fcntl.flock(handle,fcntl.LOCK_UN)
def shared():
 from src.utils import research_archived_eval_v2 as h
 if sha(h.READY)!=HELPER_READY_SHA:raise ValueError('Pinned phased archive helper changed')
 h.verify_release();return h
def source_binding():
 from scripts import research_archival_stock_queue_v5 as v5
 if sha(v5.READY)!=V5_READY_SHA:raise ValueError('Pinned advice release changed')
 return {**v5.source_binding(),str(v5.READY.relative_to(ROOT)):V5_READY_SHA,**shared().source_binding(),'scripts/probe_research_parallel_b1.py':sha(ROOT/'scripts/probe_research_parallel_b1.py'),SOURCE:sha(ROOT/SOURCE),TEST:sha(ROOT/TEST)}
def readiness():
 v=read(READY)
 if v.get('owner')!=OWNER or v.get('status')!='CPU_tests_passed_fresh_role_GPU_replay_pending' or v.get('source_sha256')!=source_binding():raise ValueError('Exact fresh-role replay release required')
 return {'path':str(READY),'sha256':sha(READY)}
def metadata(run):
 h=shared();context=h.inspect(run,'best');config=context['config']
 if config.get('stage')!='SFT' or config.get('recipe') not in ('3c','2a','2b','2c'):raise ValueError('Only completed archived future full Small SFT roles accepted')
 return context
def unique_roles(context):
 roles={}
 for role in ('best','last'):
  item=context['roles'][role]
  key=item['object'];entry=roles.setdefault(key,{'object':key,'roles':[],'marker':item['marker']});entry['roles'].append(role)
 return list(roles.values())
def assert_idle():
 from scripts.probe_research_parallel_b1 import gpu_snapshot
 if gpu_snapshot()['processes']:raise RuntimeError('Fresh role replay requires idle GPU')
def advice(run):
 from scripts import research_archival_stock_queue_v5 as v5,research_archival_stock_queue_v2 as v2
 spec=v2.spec_snapshot(v2.SPEC);state=read(v2.STATE)
 queue=SimpleNamespace(spec=spec,state=state,_routed=lambda j:v5.routed(j,state['jobs'].get(j['id'],{})))
 return {'retained':v2.advise_completed(queue,Path(run)),'anchors':v5.advise_anchors(queue,Path(run))}
def validate_receipt(report,entry,context):
 if report.get('schema_version')!=1 or report.get('run')!=context['run'] or report.get('status')!='passed' or report.get('mode')!='full_gpu_replay' or len(report.get('checkpoints',[]))!=1:raise ValueError('One complete native full GPU role receipt required')
 from src.utils import research_stock_determinism as stock
 if report.get('numerical_profile')!=stock.definition():raise ValueError('Native replay numerical profile differs')
 receipt=report['checkpoints'][0];role=entry['roles'][0];item=context['archive_admission']['publisher_plan']['roles'][role];st=item['stat'];marker=item['marker']
 expected={'schema_version':1,'status':'passed','checkpoint_owner':marker['owner'],'checkpoint_path':item['state'],'checkpoint_labels':entry['roles'],'checkpoint_step':marker['step'],'checkpoint_size_bytes':st['st_size'],'checkpoint_mtime_ns':st['st_mtime_ns'],'selection_metric_replayed':True,'official_benchmark_score':False,'restored_tensor_max_abs_diff':0.,'projector_frozen':True,'projector_unchanged_during_validation':True,'all_tensors_finite':True,'expected_keys_match':True}
 if any(receipt.get(k)!=v for k,v in expected.items()) or type(receipt.get('validated_tensor_count')) is not int or receipt['validated_tensor_count']<1:raise ValueError('Native role owner/step/restore/selection receipt differs')
 from scripts.research_grounding_queue import checkpoint_selection_binding
 from src.eval.research_selection import PROTOCOL_NAME
 selection=checkpoint_selection_binding(context['config']);saved=marker['metric'];actual=receipt.get('group_macro_score_replayed');difference=receipt.get('group_macro_score_abs_diff');tolerance=2e-3+2e-4*abs(saved)
 if (receipt.get('selection_protocol')!=PROTOCOL_NAME or receipt.get('heldout_ce_replayed') is not False or receipt.get('validation_records')!=selection['records'] or receipt.get('selection_manifest_sha256')!=selection['manifest_sha256'] or receipt.get('group_macro_score_saved')!=saved or any(type(x) not in (int,float) or not math.isfinite(x) for x in (actual,difference)) or not 0<=actual<=100 or not math.isclose(difference,abs(actual-saved),abs_tol=1e-12) or difference>tolerance or not math.isclose(receipt.get('group_macro_score_tolerance',-1),tolerance,abs_tol=1e-12)):raise ValueError('Exact native internal selection provenance/value differs')
 return receipt

def native_worker(request):
 """Only private checkpoint-list and report-path hooks; native math unchanged."""
 from src.utils import research_stock_determinism as stock
 stock.activate();stock.assert_active();h=shared();context=request['context'];h.assert_context(context)
 runner=stock.runner();native_model=runner.ResearchModel
 def guarded_model(*a,**k):
  from src.utils import research_lossless_delta as codec
  codec.memory_guard(h.implementation().CONSTRUCTOR_ALLOWANCE+h.implementation().METADATA_LIMIT)
  value=native_model(*a,**k)
  codec.memory_guard(h.implementation().METADATA_LIMIT);return value
 runner.ResearchModel=guarded_model
 module=stock._clone('scripts/validate_research_checkpoint.py',{'scripts.run_research_stage':runner});original=module.managed_checkpoints;entry=request['entry'];run=Path(request['run']);destination=Path(request['worker_report'])
 def selected(root):
  rows=original(root);matches=[r for r in rows if str(r['path'].parent)==entry['object']]
  if len(matches)!=1 or matches[0]['labels']!=entry['roles'] or matches[0]['marker']!=entry['marker']:raise ValueError('Single exact owned object selection changed')
  return matches
 native_atomic=module.atomic_json
 def output(path,value):
  if Path(path)!=run/'checkpoint_roundtrip.json':raise ValueError('Unexpected native replay write path')
  value['numerical_profile']=stock.definition();value['fresh_role_execution']={k:request[k] for k in ('owner','nonce','request_sha256','source_sha256','entry')};native_atomic(destination,value)
 module.managed_checkpoints=selected;module.atomic_json=output
 with h.implementation()._shared(run):
  h.assert_context(context);result=module.validate_run(run,device='cuda');h.assert_context(context)
 return result

def worker(path,expected_sha):
 request=read(path)
 if sha(path)!=expected_sha or request.get('owner')!=OWNER or request.get('source_sha256')!=source_binding():raise ValueError('Bound fresh worker request/source differs')
 from scripts import probe_research_parallel_b1 as proc
 actual=proc.process_info(os.getppid());parent=request['parent']
 argv=parent['argv'];parent_script=argv[2] if len(argv)>2 and argv[1]=='-u' else None
 if not proc.same_process(parent,actual) or parent_script is None or Path(parent_script).resolve()!=ROOT/SOURCE or '--worker-request' in argv or request.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():raise ValueError('Owned fresh replay parent changed')
 proc.arm_parent_death(parent['pid'])
 if not proc.same_process(parent,proc.process_info(os.getppid())):raise ValueError('Replay parent died during PDeath handshake')
 if time.monotonic()>=request['deadline_monotonic']:raise TimeoutError('Fresh replay shared deadline expired')
 readiness();request['request_sha256']=expected_sha;identity=proc.process_info(os.getpid())
 atomic(Path(path).parent/'worker_startup.json',{'owner':OWNER,'nonce':request['nonce'],'request_sha256':expected_sha,'identity':identity,'parent':parent})
 return native_worker(request)

def wait_startup(child,command,directory,nonce,request_sha,deadline):
 from scripts import probe_research_parallel_b1 as proc
 raw=None;end=min(deadline,time.monotonic()+15)
 while time.monotonic()<end:
  live=proc.process_info(child.pid)
  if live is not None:
   if live['ppid']!=os.getpid() or (raw is not None and live['start_time']!=raw['start_time']):raise RuntimeError('Fresh replay worker PID reused/foreign parent')
   raw=raw or live
   if live['argv'] and live['argv']!=command:raise RuntimeError('Fresh replay worker foreign nonempty argv')
   p=directory/'worker_startup.json'
   if p.exists():
    receipt=read(p)
    if receipt.get('owner')!=OWNER or receipt.get('nonce')!=nonce or receipt.get('request_sha256')!=request_sha or not proc.same_process(receipt.get('identity'),live) or live['argv']!=command:raise RuntimeError('Fresh worker startup registration differs')
    return receipt['identity']
  if child.poll() is not None:raise RuntimeError('Fresh replay worker exited before startup')
  time.sleep(.05)
 raise TimeoutError('Fresh replay worker15s startup deadline')
def stop(child):
 # A Popen child cannot have its PID reused before this parent reaps it.
 if child is None or child.poll() is not None:return
 from scripts import probe_research_parallel_b1 as proc
 live=proc.process_info(child.pid)
 if live is not None and live['ppid']!=os.getpid():raise RuntimeError('Refuse foreign replay child cleanup')
 child.terminate()
 try:child.wait(timeout=10)
 except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)

def wait_gpu_exit(pid,identity,deadline):
 from scripts import probe_research_parallel_b1 as proc
 end=min(deadline,time.monotonic()+30)
 while True:
  processes=proc.gpu_snapshot()['processes']
  if not processes:return
  if any(p['pid']!=pid for p in processes):raise RuntimeError('Foreign GPU process after owned replay exit')
  live=proc.process_info(pid)
  if live is not None and (live['start_time']!=identity['start_time'] or live['ppid']!=identity['ppid'] or (live['argv'] and live['argv']!=identity['argv'])):raise RuntimeError('Exited replay PID reused before NVML cleanup')
  if time.monotonic()>=end:raise TimeoutError('Exited replay GPU entry did not clear')
  time.sleep(.1)

def run_worker(request,directory,deadline):
 from scripts import probe_research_parallel_b1 as proc
 request['parent']=proc.process_info(os.getpid());request['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip();path=directory/'request.json';atomic(path,request);h=sha(path);command=[sys.executable,'-u',str(ROOT/SOURCE),'--worker-request',str(path),'--request-sha256',h];env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false');child=None
 try:
  with (directory/'worker.log').open('xb') as log:
   child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   identity=wait_startup(child,command,directory,request['nonce'],h,deadline);atomic(directory/'child.json',{'identity':identity,'command':command,'request_sha256':h})
   while child.poll() is None:
    if time.monotonic()>=deadline:raise TimeoutError('Bounded fresh-role replay deadline')
    current=proc.process_info(child.pid)
    if current is not None and current['argv'] and not proc.same_process(identity,current):raise RuntimeError('Fresh replay child identity changed')
    gpu=proc.gpu_snapshot()
    if any(p['pid']!=child.pid for p in gpu['processes']):raise RuntimeError('Foreign GPU process during exclusive fresh replay')
    time.sleep(2)
   if child.returncode!=0:raise RuntimeError(f'Fresh role worker returned{child.returncode}; inspect{directory}/worker.log')
  wait_gpu_exit(child.pid,identity,deadline)
  report=read(request['worker_report']);expected={k:request[k] for k in ('owner','nonce','source_sha256','entry')};expected['request_sha256']=h
  if report.get('fresh_role_execution')!=expected:raise ValueError('Fresh native receipt request/nonce/source binding differs')
  return report
 finally:stop(child)

def merge_reports(context,entries,reports):
 if len(entries)!=len(reports) or not entries:raise ValueError('Both unique retained roles must finish freshly')
 receipts=[validate_receipt(report,entry,context) for entry,report in zip(entries,reports)]
 if {x for receipt in receipts for x in receipt['checkpoint_labels']}!={'best','last'} or len({r['checkpoint_path'] for r in receipts})!=len(entries):raise ValueError('Full unique best/last coverage required')
 return {'schema_version':1,'status':'passed','run':context['run'],'mode':'full_gpu_replay','checkpoints':receipts,'validated_tensor_count':sum(r['validated_tensor_count'] for r in receipts),'numerical_profile':reports[0]['numerical_profile']}

def validate_completed_report(run,context=None):
 """Audit owned raw role receipts; a historical native passed flag is insufficient."""
 h=shared();context=context or h.checkpoint_context(run,'best');h.assert_context(context);root=Path(context['run']);result=read(root/'checkpoint_roundtrip.json');execution=result.get('fresh_role_replay',{});directory=Path(execution.get('attempt',''))
 if directory.is_symlink() or directory.resolve()!=directory or directory.parent!=root/NAMESPACE or not directory.name.startswith('attempt-') or execution.get('owner')!=OWNER or execution.get('source_sha256')!=source_binding() or execution.get('release')!=readiness() or execution.get('publication_sha256')!=context['files']['checkpoint_archival.json']['sha256']:raise ValueError('Owned fresh-role execution/source/publication proof required')
 attempt=read(directory/'attempt.json');entries=unique_roles(context);reports=[]
 if attempt.get('owner')!=OWNER or attempt.get('status')!='passed' or attempt.get('source_sha256')!=source_binding() or len(attempt.get('workers',[]))!=len(entries) or attempt.get('canonical_report_sha256')!=sha(root/'checkpoint_roundtrip.json'):raise ValueError('Complete fresh role attempt required')
 if (directory/'source.zip').is_symlink():raise ValueError('Local owned replay source archive required')
 with zipfile.ZipFile(directory/'source.zip') as z:
  if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(source_binding()):raise ValueError('Exact replay source archive members required')
  for name,expected in source_binding().items():
   if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise ValueError('Replay archived source bytes differ')
 for index,(entry,row) in enumerate(zip(entries,attempt['workers'])):
  childdir=directory/f'object-{index:02}'
  if childdir.is_symlink() or not childdir.is_dir():raise ValueError('Local owned role namespace required')
  if row.get('entry')!=entry or row.get('status')!='passed' or row.get('directory')!=str(childdir) or any(sha(childdir/n)!=expected for n,expected in row.get('files',{}).items()) or set(row.get('files',{}))!={'request.json','worker_startup.json','child.json','report.json'}:raise ValueError('Fresh role file/entry attribution differs')
  request=read(childdir/'request.json');startup=read(childdir/'worker_startup.json');child=read(childdir/'child.json');report=read(childdir/'report.json');request_sha=sha(childdir/'request.json');h.assert_context(request['context'])
  command=[sys.executable,'-u',str(ROOT/SOURCE),'--worker-request',str(childdir/'request.json'),'--request-sha256',request_sha];identity=startup.get('identity',{});selected=request['context']
  if (child.get('command')!=command or identity.get('argv')!=command or identity.get('ppid')!=request['parent'].get('pid') or identity.get('command_sha256')!=digest(command) or type(identity.get('pid')) is not int or type(identity.get('start_time')) is not int or selected.get('run')!=str(root) or selected.get('role')!=entry['roles'][0] or selected.get('checkpoint_path')!=str(Path(entry['object'])/'state.pt') or selected.get('roles')!=context['roles']):raise ValueError('Exact stored worker command/parent/selected-role binding differs')
  expected={k:request[k] for k in ('owner','nonce','source_sha256','entry')};expected['request_sha256']=request_sha
  if request.get('entry')!=entry or request.get('run')!=str(root) or request.get('source_sha256')!=source_binding() or report.get('fresh_role_execution')!=expected or startup.get('request_sha256')!=request_sha or startup.get('nonce')!=request['nonce'] or startup.get('parent')!=request['parent'] or child.get('request_sha256')!=request_sha or child.get('identity')!=startup.get('identity'):raise ValueError('Fresh role request/startup/native receipt differs')
  reports.append(report)
 merged=merge_reports(context,entries,reports)
 if any(result.get(k)!=v for k,v in merged.items()):raise ValueError('Canonical merged replay differs from raw fresh reports')
 return {'owner':OWNER,'attempt':str(directory),'canonical_report_sha256':sha(root/'checkpoint_roundtrip.json'),'unique_objects':len(entries),'validated_tensor_count':merged['validated_tensor_count'],'source_sha256':source_binding()}

def _execute_owned(run,timeout=1200):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Replay orchestrator parent must hide CUDA')
 readiness();sources=source_binding();h=shared();context=metadata(run);run=Path(context['run']);assert_idle()
 from src.utils.research_storage import DiskBudget
 DiskBudget(run,120).ensure_space(64*2**20)
 directory=run/NAMESPACE/('attempt-'+secrets.token_hex(16));directory.mkdir(parents=True);attempt={'owner':OWNER,'status':'running','run':str(run),'source_sha256':sources,'release':readiness(),'started_unix':time.time(),'workers':[]};atomic(directory/'attempt.json',attempt)
 old=run/'checkpoint_roundtrip.json'
 if old.exists():
  raw=old.read_bytes()
  with (directory/'previous_checkpoint_roundtrip.json').open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
  attempt['previous_report']={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 with zipfile.ZipFile(directory/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for name,expected in sorted(sources.items()):
   raw=(ROOT/name).read_bytes()
   if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Fresh replay source changed before archival')
   z.writestr(name,raw)
 with (directory/'source.zip').open('rb') as handle:os.fsync(handle.fileno())
 attempt['source_zip_sha256']=sha(directory/'source.zip');atomic(directory/'attempt.json',attempt)
 deadline=time.monotonic()+timeout;reports=[];entries=unique_roles(context)
 try:
  with h.implementation()._shared(run):
   for index,entry in enumerate(entries):
    assert_idle();full=h.checkpoint_context(run,entry['roles'][0]);h.assert_context(context);resource=advice(run);assert_idle();h.assert_context(full)
    from src.utils import research_lossless_delta as codec
    resource['actual_pre_worker_guard']=codec.memory_guard(full['archive_admission']['transactions'][entry['roles'][0]]['load_extra_bytes'])
    childdir=directory/f'object-{index:02}';childdir.mkdir();request={'owner':OWNER,'run':str(run),'entry':entry,'context':full,'nonce':secrets.token_hex(32),'source_sha256':sources,'deadline_monotonic':deadline,'worker_report':str(childdir/'report.json')}
    attempt['workers'].append({'entry':entry,'admission':full,'cache_advice':resource,'directory':str(childdir)});atomic(directory/'attempt.json',attempt)
    report=run_worker(request,childdir,deadline);validate_receipt(report,entry,full);reports.append(report);attempt['workers'][-1].update(status='passed',files={n:sha(childdir/n) for n in ('request.json','worker_startup.json','child.json','report.json')});atomic(directory/'attempt.json',attempt)
   assert_idle();final=h.checkpoint_context(run,'best');h.assert_context(context)
   if source_binding()!=sources:raise ValueError('Fresh replay sources changed')
   result=merge_reports(final,entries,reports);result.update(started_unix=attempt['started_unix'],finished_unix=time.time(),fresh_role_replay={'owner':OWNER,'source_sha256':sources,'release':readiness(),'attempt':str(directory),'attempt_nonce_independent_workers':True,'physical_archives_verified_before_and_after':True,'publication_sha256':final['files']['checkpoint_archival.json']['sha256']})
   atomic(directory/'completed_report.json',result);atomic(old,result);attempt.update(status='passed',finished_unix=time.time(),canonical_report_sha256=sha(old));atomic(directory/'attempt.json',attempt);return result
 except BaseException as error:
  attempt.update(status='failed',error_type=type(error).__name__,error=str(error),finished_unix=time.time());atomic(directory/'attempt.json',attempt);raise

def execute(run,timeout=1200):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Replay orchestrator parent must hide CUDA')
 run=Path(metadata(run)['run'])
 with namespace_lock(run):return _execute_owned(run,timeout)

def main(argv=None):
 parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run');parser.add_argument('--device',choices=['cuda'],default='cuda');parser.add_argument('--numerical-profile',choices=['stock_deterministic_small_v1']);parser.add_argument('--allow-gpu',action='store_true');parser.add_argument('--timeout-seconds',type=int,default=1200);parser.add_argument('--worker-request');parser.add_argument('--request-sha256');a=parser.parse_args(argv)
 if 'torch' in sys.modules:raise RuntimeError('Fresh role replay requires fresh process before Torch import')
 if a.worker_request:worker(a.worker_request,a.request_sha256);return 0
 os.environ['CUDA_VISIBLE_DEVICES']=''
 if not a.allow_gpu:
  print(json.dumps({'owner':OWNER,'status':'metadata_only','run':a.run,'GPU_used':False,'source_sha256':source_binding()}));return 0
 if not a.run or a.numerical_profile!='stock_deterministic_small_v1' or not 120<=a.timeout_seconds<=1800:raise ValueError('Explicit stock run/profile and bounded timeout required')
 def interrupt(signum,frame):raise KeyboardInterrupt(f'Signal{signum} during role replay')
 previous={s:signal.signal(s,interrupt) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)};timer=signal.setitimer(signal.ITIMER_REAL,a.timeout_seconds)
 try:print(json.dumps(execute(a.run,a.timeout_seconds),indent=2));return 0
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for s,h in previous.items():signal.signal(s,h)
  if timer[0]:signal.setitimer(signal.ITIMER_REAL,*timer)
if __name__=='__main__':raise SystemExit(main())
