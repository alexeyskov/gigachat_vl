#!/usr/bin/env python3
"""Same CPU queue with proof-gated one-decode archived image evaluation."""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,math,os,pwd,random,subprocess,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OWNER='single_decode_archival_stock_queue_v5'
SOURCE='scripts/research_archival_stock_queue_v5.py';TEST='tests/test_research_archival_stock_queue_v5.py'
V4='scripts/research_archival_stock_queue_v4.py';V4_SHA='da9a42b4dee2a46cbd47ad6253f23c8ba647475e5572cd17d07228050c65ee63'
V4_READY=ROOT/'configs/research/archival_stock_queue_v4_ready.json';V4_READY_SHA='9dd13d253d1522e522fa381cfd626636b2119bd2164acc2d6be6530b2b96453d'
READY=ROOT/'configs/research/archival_stock_queue_v5_ready.json'
PROOF=ROOT/'research_results/single_decode_3c_actual_GPU_v1/report.json'
PROOF_SCRIPT=ROOT/'research_results/probe_single_decode_3c_gpu_v1.py'
PROOF_SCRIPT_SHA='fdefbe824706b4ce0406cc12de1cbad6d21eb2e4208782dbe59ceabc76433520'
SINGLE_READY_SHA='260fc0c446cde231ca26beb7c99e938830667d5c09b5b18bbd0abdd86221e37c'
NEW_SCRIPT=ROOT/'scripts/eval_research_archived_native_v1.py'
RECIPES=('3c','2a','2b','2c')
ANCHOR_BYTES={'model-00001-of-00002.safetensors':4999721256,'model-00002-of-00002.safetensors':3537628448}
BENCHMARKS=('ai2d','aokvqa','chartqa','docvqa','gqa','scienceqa','textvqa','vqav2')
NATIVE_SOURCES=('src/eval/research_inference.py','scripts/eval_research_checkpoint.py','src/eval/research_metrics.py','scripts/run_research_stage.py',
 'src/model/research_moe.py','src/model/smallthinker.py','src/model/research_quantization.py','src/model/research_vision.py',
 'src/model/gigachat_vl.py','src/dataset/vision_cache.py','scripts/extract_vision_features.py','src/eval/research_ablation.py',
 'src/utils/research_checkpoint_compat.py','src/utils/research_checkpoint_bundle.py','src/utils/research_checkpoint_io.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p)
 if p.is_symlink() or not p.is_file() or p.stat().st_size>32*2**20:raise ValueError('Bounded ordinary controller evidence required')
 return json.loads(p.read_bytes())
def single():
 from src.eval import research_archived_native_evaluation_v1 as module
 if sha(module.READY)!=SINGLE_READY_SHA:raise ValueError('Pinned one-decode evaluator release changed')
 module.verify_release();return module
def source_binding():
 if sha(ROOT/V4)!=V4_SHA or sha(V4_READY)!=V4_READY_SHA:raise ValueError('Frozen v4 controller changed')
 value=read(V4_READY)['source_sha256']
 if any(sha(ROOT/n)!=h for n,h in value.items()):raise ValueError('Frozen v4 dependency changed')
 return {**value,str(V4_READY.relative_to(ROOT)):V4_READY_SHA,**single().source_binding(),str(PROOF_SCRIPT.relative_to(ROOT)):PROOF_SCRIPT_SHA,SOURCE:sha(ROOT/SOURCE),TEST:sha(ROOT/TEST)}
def readiness():
 value=read(READY)
 if value.get('owner')!=OWNER or value.get('status')!='cpu_route_tests_passed_GPU_proof_required' or value.get('source_sha256')!=source_binding():raise ValueError('Exact v5 CPU route release required')
 return {'path':str(READY),'sha256':sha(READY)}
def proof_evidence(expected_sha):
 if not isinstance(expected_sha,str) or len(expected_sha)!=64 or sha(PROOF)!=expected_sha:raise ValueError('Explicit actual ROOT GPU proof SHA required')
 report=read(PROOF);base=PROOF.parent
 expected={'owner':'ROOT','status':'passed_64_native_generations_and_full_runtime_census','records':64,
  'single_decode_release_sha256':SINGLE_READY_SHA,'runtime_census_exact':True,'all_raw_generation_fields_exact':True,
  'actual_candidate_GPU_validated':True,'scientific_training_updates':0,'scientific_benchmark_claim':False,'fresh_process_per_model':True,
  'script_sha256':PROOF_SCRIPT_SHA,'benchmark_counts':dict.fromkeys(BENCHMARKS,8)}
 if any(report.get(k)!=v for k,v in expected.items()) or sha(PROOF_SCRIPT)!=PROOF_SCRIPT_SHA or sha(base/'records.jsonl')!=report.get('records_sha256'):
  raise ValueError('Actual complete 64/full-state ROOT proof required; running/failed cannot activate')
 with (base/'records.jsonl').open() as f:records=[json.loads(line) for line in f if line.strip()]
 ids=[r['record_id'] for r in records]
 if len(ids)!=64 or len(set(ids))!=64:raise ValueError('GPU proof records incomplete')
 counts={name:sum((r.get('benchmark') or r['source']['benchmark'])==name for r in records) for name in BENCHMARKS}
 if counts!=expected['benchmark_counts']:raise ValueError('GPU proof benchmark scope differs')
 manifest=ROOT/'datasets/research/eval-heldout-v3/evaluation.jsonl'
 with manifest.open() as f:full=[json.loads(line) for line in f if line.strip()]
 groups={name:[r for r in full if (r.get('benchmark') or r['source']['benchmark'])==name] for name in BENCHMARKS}
 rng=random.Random(42);chosen=[r for name in BENCHMARKS for r in rng.sample(groups[name],8)];chosen.sort(key=lambda r:r['record_id'])
 if len(full)!=512 or chosen!=records:raise ValueError('GPU proof64 differs from exact declared source512/seed42 selection')
 left,right=read(base/'reference.json'),read(base/'candidate.json')
 for mode,worker in (('reference',left),('candidate',right)):
  if worker.get('owner')!='ROOT' or worker.get('mode')!=mode or worker.get('status')!='passed_worker' or worker.get('stock_profile')!=single().stock.definition():raise ValueError('GPU worker identity/profile differs')
  if [r.get('record_id') for r in worker.get('predictions',[])]!=ids or worker.get('cache',{}).get('misses')!=0 or worker['cache'].get('encode_calls')!=0:raise ValueError('GPU worker raw64/cache evidence differs')
 census=left.get('census',{})
 if not census or len(census)!=report.get('tensor_census_entries') or census!=right.get('census') or left['predictions']!=right['predictions']:
  raise ValueError('Independent raw census/token comparison failed')
 if not any(n.startswith('parameter:') for n in census) or not any(n.startswith('buffer:') for n in census):raise ValueError('GPU census omitted parameters/buffers')
 proof_run=ROOT/'runs/research/smallthinker4__siglip2__3c__SFT_pilot_scienceqa_context_v2_balanced_s42__stockdet_v1'
 context=single().shared.inspect(proof_run,'best');loaded=right['provenance']
 if loaded.get('archived_evaluation')!=single().shared.execution_profile() or loaded.get('archived_physical_sha256')!=context['artifact']['sha256'] or loaded.get('archived_publication_sha256')!=context['files']['checkpoint_archival.json']['sha256']:
  raise ValueError('GPU proof selected archive/phase profile/publication differs')
 return {'path':str(PROOF),'sha256':expected_sha,'script_sha256':PROOF_SCRIPT_SHA,'single_decode_release_sha256':SINGLE_READY_SHA,
  'records_sha256':report['records_sha256'],'source512_sha256':sha(manifest),'checkpoint_sha256':context['artifact']['sha256'],
  'reference_sha256':sha(base/'reference.json'),'candidate_sha256':sha(base/'candidate.json'),'records':64,'tensor_census_entries':len(census)}
def arg(command,name,default=None):
 try:return command[command.index(name)+1]
 except (ValueError,IndexError):return default
def eligible(job):return arg(job['argv'],'--stage')=='SFT' and arg(job['argv'],'--recipe') in RECIPES
def routed(job,record):return eligible(job) and not record.get('checkpoint_archival_skip') and record.get('checkpoint_archival',{}).get('status')=='completed'
def run_dir(job):return Path(arg(job['argv'],'--output')).resolve()
def preserve_attempts(state,jobs):
 for job in jobs:
  record=state['jobs'].get(job['id'],{})
  if not routed(job,record):continue
  old=record.get('evaluation',{})
  if old and old.get('output')!=str(single().output_path(run_dir(job),'none','best')):
   if old.get('status')=='completed':raise ValueError('Completed old archived evaluation needs explicit separate adoption; no relabel')
   path=Path(old.get('output',''))/'predictions.jsonl'
   if path.is_file() and path.stat().st_size:raise ValueError('Partial old archived predictions preserved; no implicit migration')
   history=record.setdefault('evaluation_execution_history',[])
   if old not in history:history.append(copy.deepcopy(old))
def _live_output(output):
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:argv=[os.fsdecode(v) for v in (p/'cmdline').read_bytes().split(b'\0') if v]
  except OSError:continue
  if any(Path(v).name==NEW_SCRIPT.name for v in argv) and str(output) in argv:raise RuntimeError('Owned one-decode output has a live evaluator; no duplicate launch')
def scan_anchor_users(identities):
 """Read fd/maps as each live process owner; no service changes or signals."""
 groups={}
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  try:uid=path.stat().st_uid
  except FileNotFoundError:continue
  groups.setdefault(uid,[]).append(int(path.name))
 code="""
import json,os,sys
from pathlib import Path
d=json.loads(sys.argv[1]);ids={tuple(x) for x in d['ids']};checked=[];opened=[];errors=[]
for pid in d['pids']:
 p=Path('/proc')/str(pid)
 try:
  for f in (p/'fd').iterdir():
   try:s=f.stat()
   except FileNotFoundError:continue
   if (s.st_dev,s.st_ino) in ids:opened.append({'pid':pid,'fd':str(f)})
  for line in (p/'maps').read_text().splitlines():
   q=line.split(None,5)
   if len(q)<5:continue
   a,b=q[3].split(':')
   if (os.makedev(int(a,16),int(b,16)),int(q[4])) in ids:opened.append({'pid':pid,'mapping':line})
  checked.append(pid)
 except FileNotFoundError:continue
 except PermissionError as e:
  if p.exists():errors.append({'pid':pid,'error':str(e)})
print(json.dumps({'uid':os.geteuid(),'checked_pids':checked,'opened':opened,'errors':errors}))
"""
 results=[]
 for uid,pids in groups.items():
  command=[sys.executable,'-c',code,json.dumps({'ids':list(identities),'pids':pids})]
  if uid!=os.geteuid():command=['runuser','-u',pwd.getpwuid(uid).pw_name,'--',*command]
  row=json.loads(subprocess.run(command,capture_output=True,text=True,check=True,timeout=20).stdout)
  if row['opened'] or row['errors']:raise RuntimeError('Inactive pinned anchor scan found open/map/unreadable process')
  results.append(row)
 return results
def advise_anchors(queue,run,*,inspect=None,assert_context=None,scan=None,advisor=None,gpu=None,process=None,host=None):
 from src.utils import research_trained_delta_probe as probe,research_archive_gpu_parity as parity
 from scripts import probe_research_parallel_b1 as proc
 inspect=inspect or single().shared.inspect;assert_context=assert_context or single().shared.assert_context;scan=scan or scan_anchor_users
 gpu=gpu or proc.gpu_snapshot;process=process or proc.process_info;host=host or parity.host_snapshot
 run=Path(run).resolve();jobs=[j for j in queue.spec['jobs'] if run_dir(j)==run]
 if len(jobs)!=1 or not queue._routed(jobs[0]):raise ValueError('Anchor advice requires this completed archived stock job')
 record=queue.state['jobs'][jobs[0]['id']]
 if record.get('training_completed') is not True or record.get('returncode')!=0 or (record.get('pid') and process(record['pid']) is not None) or gpu()['processes']:
  raise ValueError('Anchor advice requires successful reaped trainer and idle GPU')
 context=inspect(run,'best');header=context['archive_admission']['transactions']['best']['reader_header'];items=header['anchor_files']
 sizes=ANCHOR_BYTES
 if context['config'].get('model')!='Tiiny/SmallThinker-4BA0.6B-Instruct' or len(items)!=2 or {x['shard'] for x in items}!=set(sizes):raise ValueError('Only exact pinned Small two-shard anchors may be advised')
 paths={}
 for item in items:
  snapshot=Path(item['snapshot'])
  if snapshot.name!='a24932bb880c85c04c794283202a9194b1b42709' or item['bytes']!=sizes[item['shard']]:raise ValueError('Anchor revision/declared size differs')
  path=(snapshot/item['shard']).resolve(strict=True)
  if not path.is_file() or path.is_symlink() or path.stat().st_size!=item['bytes']:raise ValueError('Anchor physical stat/size differs')
  paths[path]=probe.stat(path)
 identities={(value['st_dev'],value['st_ino']) for value in paths.values()};before_scan=scan(identities);initial=host();events=[]
 for path,before in paths.items():
  if probe.stat(path)!=before:raise ValueError('Anchor changed before readonly advice')
  if advisor is None:
   fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
   try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
   finally:os.close(fd)
  else:advisor(path,before)
  if probe.stat(path)!=before:raise ValueError('Anchor changed during readonly advice')
  events.append({'path':str(path),'stat':before})
 after_scan=scan(identities);assert_context(context)
 if gpu()['processes']:raise ValueError('GPU occupancy changed during anchor advice')
 final=host();available=parity.conservative_host_headroom(final)
 if available<32*2**30:raise MemoryError('Actual remaining host/cgroup headroom below unchanged32GiB floor')
 return {'scope':'only exact inactive pinned Small BF16 anchor shards from admitted reader_header','events':events,
  'process_owner_scan_before':before_scan,'process_owner_scan_after':after_scan,'host_before':initial,'host_after':final,
  'actual_available_after_bytes':available,'reclaimable_cache_credit_assumed_bytes':0,'bytes_pointers_or_weights_changed':False}
def _edited_summary(cls):
 """Change only three paired-summary paths in the pinned private method."""
 method=cls.evaluate_completed_job;tree=ast.parse((ROOT/'scripts/research_grounding_queue.py').read_bytes())
 definition=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GroundingResearchQueue')
 function=copy.deepcopy(next(n for n in definition.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_completed_job'))
 replacements={'baseline':'read_rows(self._ordinary_predictions(job))','variants':"{mode:read_rows(self._variant_predictions(job,mode)) for mode in MODES}",'destination':'self._paired_summary_destination(job)'}
 changed=[]
 for node in ast.walk(function):
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements:
   name=node.targets[0].id;node.value=ast.parse(replacements[name],mode='eval').body;changed.append(name)
 if sorted(changed)!=sorted(replacements):raise ValueError('Pinned paired-summary path structure changed')
 temporary=ast.ClassDef(name='_SummaryOnly',bases=[],keywords=[],body=[function],decorator_list=[]);namespace=dict(method.__globals__)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[temporary],type_ignores=[])),'<v5-summary-paths>','exec'),namespace)
 code=namespace['_SummaryOnly'].evaluate_completed_job.__code__
 if code.co_freevars!=method.__code__.co_freevars:raise ValueError('Private summary super closure differs')
 cls.evaluate_completed_job=types.FunctionType(code,method.__globals__,method.__name__,method.__defaults__,method.__closure__)
class _Spawn:
 def __init__(self,original,live,advice):self.original=original;self.live=live;self.advice=advice
 def __getattr__(self,name):return getattr(self.original,name)
 def Popen(self,command,*a,**k):
  queue=self.live['queue'];queue._guard();command=list(command);script=Path(command[2]).resolve() if len(command)>2 and command[1]=='-u' else None
  new=script==NEW_SCRIPT.resolve()
  if new:
   scope=self.original.Popen.__func__.__globals__;scope['assert_cpu']()
   if Path(command[0]).resolve()!=Path(sys.executable).resolve() or '--allow-gpu' not in command or arg(command,'--numerical-profile')!='stock_deterministic_small_v1':raise ValueError('Exact one-decode GPU command required')
   expected=single().output_path(arg(command,'--run'),arg(command,'--image-ablation','none'),arg(command,'--checkpoint','best'))
   if str(expected)!=arg(command,'--output'):raise ValueError('One-decode command output differs')
   jobs=[j for j in queue.spec['jobs'] if run_dir(j)==Path(arg(command,'--run')).resolve()]
   if len(jobs)!=1 or not queue._routed(jobs[0]):raise ValueError('New GPU evaluator requires exact registered archived job')
   _live_output(expected);environment=dict(os.environ if k.get('env') is None else k['env']);environment['CUDA_VISIBLE_DEVICES']='0';k['env']=environment
  if new or (script is not None and script.name in ('eval_research_stock_deterministic.py','validate_research_stock_deterministic.py')):
   receipt=self.advice(queue,Path(arg(command,'--run')).resolve());history=queue.state.setdefault('pre_evaluation_cache_advice',[])
   run=Path(arg(command,'--run')).resolve();jobs=[j for j in queue.spec['jobs'] if run_dir(j)==run]
   anchor=advise_anchors(queue,run) if len(jobs)==1 and queue._routed(jobs[0]) else None
   history.append({'command_script':str(script),'receipt':receipt,'pinned_anchor_advice':anchor});del history[:-16];queue.persist();queue._guard()
  if new:return self.original.module.Popen(command,*a,**k)
  return self.original.Popen(command,*a,**k)
def _strict_protocol(identity):
 directory=identity['output'];protocol=read(directory/'protocol.json')
 expected={**single().source_binding(),**{n:sha(ROOT/n) for n in NATIVE_SOURCES}}
 if protocol.get('single_decode_execution')!=single().execution_profile() or protocol.get('source_sha256')!=expected or protocol.get('numerical_profile')!=single().stock.definition():raise ValueError('Exact full one-decode evaluator source/profile required')
 environment={'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','TOKENIZERS_PARALLELISM':'false'}
 if protocol.get('single_decode_environment')!=environment:raise ValueError('One-decode child environment differs')
 loaded=read(directory/'model_provenance.json')
 if loaded.get('archived_evaluation')!=single().shared.execution_profile() or loaded.get('archived_physical_sha256')!=identity['checkpoint_sha256'] or loaded.get('archived_publication_sha256')!=sha(identity['checkpoint'].parent.parent.parent/'checkpoint_archival.json'):
  raise ValueError('Actual one-decode model/artifact/publication provenance differs')
 config=read(identity['checkpoint'].parent.parent.parent/'resolved_config.json');marker=read(identity['checkpoint'].parent/'.research_checkpoint.json')
 if loaded.get('numerical_profile')!=single().stock.definition() or loaded.get('stage')!='SFT' or loaded.get('recipe')!=config['recipe'] or loaded.get('training_step')!=marker['step']:
  raise ValueError('One-decode model stage/recipe/step differs')
 rows=[]
 with (directory/'predictions.jsonl').open() as f:rows=[json.loads(line) for line in f if line.strip()]
 records=identity.get('transformed_records',identity['records']);sentinel=object()
 if protocol.get('selected_record_ids_sha256')!=hashlib.sha256(json.dumps([r['record_id'] for r in records]).encode()).hexdigest():raise ValueError('One-decode selected record order differs')
 fields=('record_id','question','image_hashes','answers','scorer','task_group','source','benchmark','choices','scoring_protocol','image_ablation')
 if [r['record_id'] for r in rows]!=[r['record_id'] for r in records]:raise ValueError('One-decode raw ordered corpus differs')
 for row,record in zip(rows,records):
  if any(row.get(k,sentinel)!=record.get(k,sentinel) for k in fields):raise ValueError('One-decode raw record metadata/reference differs')
  tokens=row.get('generated_token_ids')
  if not isinstance(tokens,list) or len(tokens)>64 or any(type(v) is not int or v<0 for v in tokens) or row.get('generated_tokens')!=len(tokens):raise ValueError('One-decode raw generated token contract differs')
def load_controller(gpu_proof_sha):
 source_binding();proof=proof_evidence(gpu_proof_sha);v4=types.ModuleType('_single_route_private_v4');v4.__file__=str(ROOT/V4)
 exec(compile((ROOT/V4).read_bytes(),v4.__file__,'exec'),v4.__dict__);module=v4.load_controller();original_class=module.queue_class;command_profile=module.stock.command_profile;live={}
 def command(command):
  result=command_profile(command)
  if len(result)>2 and Path(result[2]).name=='eval_research_stock_deterministic.py' and single().NAMESPACE in Path(arg(result,'--output','')).parts:result[2]=str(NEW_SCRIPT)
  return result
 module.stock.command_profile=command
 def queue_class():
  parent=original_class();grounding=next(c for c in parent.__mro__ if c.__name__=='GroundingResearchQueue');_edited_summary(grounding)
  namespace=parent.compact_completed_job.__globals__;seen=set()
  for cls in parent.__mro__:
   for value in cls.__dict__.values():
    if callable(value) and hasattr(value,'__globals__') and 'Popen' in value.__code__.co_names and id(value.__globals__) not in seen:
     g=value.__globals__;seen.add(id(g));g['subprocess']=_Spawn(g['subprocess'],live,namespace['advise_completed'])
  class SingleQueue(parent):
   def __init__(self,*a,**k):
    super().__init__(*a,**k)
    try:
     descriptor={'owner':OWNER,'controller_source':SOURCE,'controller_sha256':sha(ROOT/SOURCE),'release':readiness(),'GPU_proof':proof,
      'new_GPU_child_script':str(NEW_SCRIPT),'new_GPU_child_CUDA_VISIBLE_DEVICES':'0','original_v3_child_policy_unchanged':True,'SPEC_and_trainer_validator_unchanged':True,
      'pre_child_advice':'same owned retained policy plus exact inactive Small reader_header anchor shards; allUIDfdmaps; measured head/no credit'}
     extra=self.state.setdefault('extra',{});existing=extra.get(OWNER)
     if existing is not None and existing!=descriptor:raise ValueError('One-decode controller profile changed')
     preserve_attempts(self.state,self.spec['jobs'])
     previous=dict(self.state['numerical_execution']);history=self.state.setdefault('controller_execution_history',[])
     if previous not in history:history.append(previous)
     extra[OWNER]=descriptor;self.state['numerical_execution'].update(controller_source=SOURCE,controller_sha256=descriptor['controller_sha256'],single_decode_execution_owner=OWNER)
     self.gpu_proof_sha=gpu_proof_sha;self.single_proof=proof;live['queue']=self
    except BaseException:self.lock_handle.close();raise
   def _guard(self):
    if proof_evidence(self.gpu_proof_sha)!=self.single_proof:raise ValueError('Root GPU execution proof changed')
    readiness();return super()._guard()
   def _evaluation_identity(self,job):
    result=super()._evaluation_identity(job)
    if self._routed(job):result['output']=single().output_path(run_dir(job),'none','best')
    return result
   def _routed(self,job):
    return routed(job,self.state['jobs'].get(job['id'],{}))
   def _grounding_identity(self,job,variant):
    if not self._routed(job):return super()._grounding_identity(job,variant)
    result=self._evaluation_identity(job);result['output']=single().output_path(run_dir(job),variant,'best')
    g=grounding._grounding_identity.__globals__;transformed,descriptor=g['transform_image_ablation'](result['records'],variant=variant,seed=42,output_dir=result['output'],source_manifest_sha256=result['manifest_sha256'],reserve_bytes=int(self.spec.get('reserve_gb',120)*2**30))
    g['write_transformed_manifest'](transformed,result['output']/'transformed_manifest.jsonl',reserve_bytes=int(self.spec.get('reserve_gb',120)*2**30));result.update(transformed_records=transformed,image_ablation=descriptor);return result
   def _validate_evaluation(self,job,identity):
    result=super()._validate_evaluation(job,identity)
    if self._routed(job):_strict_protocol(identity);result['execution_owner']=single().OWNER
    return result
   def _validate_grounding(self,job,identity,variant):
    result=super()._validate_grounding(job,identity,variant)
    if self._routed(job):_strict_protocol(identity);result['execution_owner']=single().OWNER
    return result
   def _ordinary_predictions(self,job):return (single().output_path(run_dir(job),'none','best') if self._routed(job) else (ROOT/Path(job['evaluation']['output'])).resolve())/'predictions.jsonl'
   def _variant_predictions(self,job,mode):return (single().output_path(run_dir(job),mode,'best') if self._routed(job) else run_dir(job)/'evaluation/grounding_v1'/mode)/'predictions.jsonl'
   def _paired_summary_destination(self,job):return (run_dir(job)/'evaluation'/single().NAMESPACE if self._routed(job) else run_dir(job)/'evaluation/grounding_v1')/'paired_summary.json'
  return SingleQueue
 module.queue_class=queue_class;return module
def main(argv=None):
 if 'torch' in sys.modules:raise RuntimeError('CPU v5 controller requires a fresh process before Torch import')
 os.environ['CUDA_VISIBLE_DEVICES']='';p=argparse.ArgumentParser(add_help=False);p.add_argument('--gpu-proof-sha');options,remaining=p.parse_known_args(argv)
 try:module=load_controller(options.gpu_proof_sha)
 except (OSError,ValueError,KeyError) as error:
  if '--run-queue' in remaining:raise
  print(json.dumps({'owner':OWNER,'status':'closed_GPU_proof_required','reason':str(error),'jobs_started':False}));return 2
 return module.main(remaining)
if __name__=='__main__':raise SystemExit(main())
