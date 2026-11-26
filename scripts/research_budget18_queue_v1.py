#!/usr/bin/env python3
"""Finite budget18 queue: CPU owner, one owned GPU worker, native strict metrics."""
from __future__ import annotations
import argparse,ctypes,fcntl,hashlib,json,math,os,signal,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if '--worker-request' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
OWNER='research_budget18_queue_v1';DATA=ROOT/'datasets/research/budget18-en-v1';RELEASE_SHA='910448c7fd680ec949d6521e32716aee2832a99fc534d185f0a06b226c0ec1c0'
PLAN=ROOT/'configs/research/budget18_plan_v1.json'
MODELS={'smallthinker4':'Tiiny/SmallThinker-4BA0.6B-Instruct','lfm25':'LiquidAI/LFM2.5-8B-A1B','giga31':'ai-sage/GigaChat3.1-10B-A1.8B-bf16'}
VISIONS={'siglip2':'google/siglip2-so400m-patch14-384','qwen25':'Qwen/Qwen2.5-VL-7B-Instruct'}
READY={'smallthinker4':'configs/research/small_budget_v1_ready.json','lfm25':'configs/research/lfm_budget_execution_v1_ready.json','giga31':'configs/research/giga_budget_execution_v1_ready.json'}
PROFILES={'smallthinker4':'stock_deterministic_small_v1','lfm25':'budget_lfm_native_stock_deterministic_v1','giga31':'budget_giga_native_stock_deterministic_v1'}
READY_V2={'smallthinker4':'configs/research/small_budget_v2_ready.json','lfm25':'configs/research/lfm_budget_execution_v2_ready.json','giga31':'configs/research/giga_budget_execution_v2_ready.json'}
PROFILES_V2={'smallthinker4':'stock_deterministic_small_v1','lfm25':'budget_lfm_qlora_learned_gates_v2','giga31':'budget_giga_qlora_learned_gates_v2'}
BACKEND_MODULES_V2={'smallthinker4':'src.utils.research_small_budget_v2','lfm25':'src.utils.research_lfm_budget_v2','giga31':'src.utils.research_giga_budget_v2'}
BACKEND_MODULES={'smallthinker4':'src.utils.research_small_budget_v1','lfm25':'src.utils.research_lfm_budget_v1','giga31':'src.utils.research_giga_budget_v1'}
PINNED_ORACLES={'scripts/research_queue.py':'446d928181aba5f77d7e02692279741eb10d8fc3d3409af985e07d9239263bda','scripts/research_grounding_queue.py':'b99dfd413edd4176288734262197a72b3df4720e01d562e4b76f625b70d0a1d9'}

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
 with tmp.open('x') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def emit(event,**v):print(json.dumps(dict(event=event,time=time.time(),**v)),flush=True)
def value(argv,flag):
 found=[argv[i+1] for i,x in enumerate(argv[:-1]) if x==flag]
 if len(found)>1:raise ValueError('Duplicate argument '+flag)
 return found[0] if found else None
def directory(job):return ROOT/value(job['argv'],'--output')
def proc(pid):
 try:
  p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(') ',1)[1].split();argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  return {'pid':pid,'birth':int(fields[19]),'ppid':int(fields[1]),'state':fields[0],'argv':argv}
 except (OSError,ValueError):return None
def same(a,b):return bool(a and b and all(a[k]==b[k] for k in ('pid','birth','argv')) and b['state'] not in ('Z','X'))
def resources(*,idle=False):
 from src.utils.research_storage import DiskBudget
 from src.utils.research_lossless_delta import memory_guard
 disk=DiskBudget(ROOT,120).ensure_space(0);host=memory_guard(0)
 if idle:
  result=subprocess.run(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
  if result.stdout.strip():raise RuntimeError('GPU occupied; no concurrent budget worker launched: '+result.stdout.strip())
 return {'disk':disk,'host':host}

def jobs():
 result=[]
 for family,model in MODELS.items():
  for encoder,vision in VISIONS.items():
   runroot=f'runs/research/{family}__{encoder}__'
   def add(stage,recipe,suffix,requires,initial=None,precision=None):
    out=runroot+suffix+'__budget18_s42';name=Path(out).name
    micro=16 if family=='smallthinker4' and encoder=='siglip2' and (stage=='A0' and precision=='nf4' or stage=='A1' and recipe=='1b') else 64 if stage=='A0' else 32 if family=='smallthinker4' else 16
    full=stage=='SFT' and recipe=='4f'
    argv=['--numerical-profile',(PROFILES_V2 if recipe=='1c' else PROFILES)[family],'--data-proof',str(DATA/'release.json'),'--model',model,'--vision',vision,'--stage',stage,'--recipe',recipe,'--manifest',str(DATA/('sft_multimodal_train.jsonl' if stage=='SFT' else 'alignment_a1_train.jsonl' if stage=='A1' else 'alignment_train.jsonl')),'--validation',str(DATA/('sft_validation.jsonl' if stage=='SFT' else 'alignment_validation.jsonl')),'--output',out,'--epochs','1','--seed','42','--batch-size',str(micro),'--effective-batch-size','64','--max-length','4096','--eval-every','10000','--reserve-gb','120','--optimizer-backend','cpu_grad_bnb_paged_adamw8' if full else 'torch_adamw32']
    if encoder=='qwen25':argv+=['--vision-preprocessing-profile','qwen_native_bounded_729']
    if initial:argv+=['--initial-checkpoint',str(directory(initial)/'best/state.pt')]
    job={'id':name,'backend':family,'encoder':encoder,'argv':argv,'requires':[x['id'] for x in requires],'stage':stage,'recipe':recipe,'expected_records':{'A0':4096,'A1':256,'SFT':3529}[stage],'expected_steps':{'A0':64,'A1':4,'SFT':56}[stage],'discard_checkpoint_after_dependents':stage!='SFT'}
    if stage=='SFT':
     argv+=['--text-manifest',str(DATA/'text_train.jsonl'),'--text-loss-weight','.15','--selection-metric','group_macro']
     job.update(compact_completed=True,evaluation={'manifest':str(DATA/'heldout_evaluation.jsonl'),'output':str(ROOT/out/'evaluation/budget18_native_v1/none'),'checkpoint':'best','max_new_tokens':64})
    result.append(job);return job
   nf=add('A0','1b','A0_nf4',[],precision='nf4');a1=add('A1','1b','1b__A1',[nf],nf);add('SFT','1b','1b__SFT',[a1],a1)
   bf=add('A0','4f','A0_bf16',[],precision='bf16');add('SFT','4f','4f__SFT',[bf],bf);a3=add('A1','1c','1c__A1',[nf],nf);add('SFT','1c','1c__SFT',[a3],a3)
 return result

MANUAL_SERVICES={
 'smallthinker4__siglip2__A0_nf4__budget18_s42':'research-budget18-small-siglip-a0-nf4',
 'smallthinker4__siglip2__1b__A1__budget18_s42':'research-budget18-small-siglip-r1-a1',
 'smallthinker4__siglip2__1b__SFT__budget18_s42':'research-budget18-small-siglip-r1-sft'}
def supervisor_info(name):
 code="from supervisor.options import ClientOptions;from supervisor.xmlrpc import SupervisorTransport;from xmlrpc.client import ServerProxy;import json,sys;x=ClientOptions();x.realize([],doc='status');s=ServerProxy('http://127.0.0.1',transport=SupervisorTransport(x.username,x.password,x.serverurl));p=s.supervisor.getProcessInfo(sys.argv[1]);print(json.dumps({k:p[k] for k in ['name','pid','statename','exitstatus','start','stop']}))"
 result=subprocess.run(['/usr/bin/python3','-c',code,name],capture_output=True,text=True,timeout=10,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 return json.loads(result.stdout)

def make_plan():
 return {'schema_version':1,'owner':OWNER,'queue_id':'budget18-en-v1-seed42','reserve_gb':120,'host_headroom_gib':32,'controller_source_sha256':sha(__file__),'data_release':{'path':str(DATA/'release.json'),'sha256':RELEASE_SHA},'backend_readiness':{k:{'path':p,'sha256':sha(ROOT/p)} for k,p in READY.items()},'backend_readiness_v2':{k:{'path':p,'sha256':sha(ROOT/p)} for k,p in READY_V2.items()},'source_sha256':dict(PINNED_ORACLES),'jobs':jobs(),'scientific_cells':18,'training_jobs':42,'manual_supervisor_adoptions':MANUAL_SERVICES,'fixed_date':'2026-11-22','R2_schedule':'A0_to_SFT_no_A1','evaluation':'nativeB1_512_none_shuffle_blank_text_only_plus_native_checkpoint_replay_and1000paired_image_bootstrap','legacy_outputs_or_metrics_retagged':False}
def verify_plan(spec):
 if spec.get('owner')!=OWNER or spec.get('jobs')!=jobs() or len(spec['jobs'])!=42 or spec.get('controller_source_sha256')!=sha(__file__):raise ValueError('Exact immutable42-job budget plan/source required')
 if sha(DATA/'release.json')!=RELEASE_SHA or spec['data_release']['sha256']!=RELEASE_SHA:raise ValueError('Data release changed')
 for family,item in [(k,v) for section in ('backend_readiness','backend_readiness_v2') for k,v in spec[section].items()]:
  if item['path'] not in (READY[family],READY_V2[family]) or sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Backend readiness changed')
  ready=read(ROOT/item['path'])
  if ready.get('status')!='passed':raise ValueError('Backend CPU readiness is not passed')
  for path,h in ready.get('source_sha256',{}).items():
   if sha(ROOT/path)!=h:raise ValueError('Backend code changed: '+path)
 for p,h in PINNED_ORACLES.items():
  if sha(ROOT/p)!=h:raise ValueError('Frozen lifecycle/scoring oracle changed')
 return spec

def oracle(spec):
 from scripts.research_grounding_queue import GroundingResearchQueue
 from src.utils.research_storage import DiskBudget
 o=object.__new__(GroundingResearchQueue);o.spec=spec;o.budget=DiskBudget(ROOT,120);o.interrupted_by=None
 return o

def backend_module(family,recipe):return (BACKEND_MODULES_V2 if recipe=='1c' else BACKEND_MODULES)[family]

def expected_profile(family,recipe=None):
 if family=='smallthinker4':
  from src.utils.research_stock_determinism import definition
 else:
  import importlib
  definition=importlib.import_module(backend_module(family,recipe)).definition
 return definition()

def validate_completed(job):
 """Metadata-only managed artifact adoption; tensor body/quality gates follow."""
 from src.utils.research_storage import BestLastCheckpointManager,DiskBudget
 out=directory(job);config=read(out/'resolved_config.json');completion=read(out/'completed.json');owner=read(out/'.checkpoint_manager.json')
 if completion.get('steps')!=job['expected_steps'] or completion.get('records')!=job['expected_records']:raise ValueError('Completed stage exact step/record count differs')
 for flag in ('--model','--vision','--stage','--recipe','--manifest','--validation','--batch-size','--seed','--epochs','--effective-batch-size','--max-length','--eval-every','--optimizer-backend'):
  expected=value(job['argv'],flag);actual=config.get(flag[2:].replace('-','_'))
  if isinstance(actual,(int,float)):expected=float(expected)
  elif flag in ('--manifest','--validation'):actual=str(Path(actual).resolve())
  if actual!=expected:raise ValueError('Completed config differs: '+flag)
 expected_initial=value(job['argv'],'--initial-checkpoint')
 if expected_initial and Path(config.get('initial_checkpoint','')).resolve()!=Path(expected_initial).resolve():raise ValueError('Completed predecessor checkpoint differs')
 if config.get('vision_preprocessing_profile')!=value(job['argv'],'--vision-preprocessing-profile'):raise ValueError('Completed vision preprocessing differs')
 if config.get('numerical_profile')!=expected_profile(job['backend'],job['recipe']):raise ValueError('Completed numerical profile differs')
 if config.get('resume') not in (False,None) or config.get('max_records') is not None or config.get('max_steps') is not None:raise ValueError('Technical/truncated run cannot be adopted')
 if config.get('manifest_sha256')!=sha(value(job['argv'],'--manifest')) or config.get('validation_sha256')!=sha(value(job['argv'],'--validation')):raise ValueError('Completed stage manifest source changed')
 if job['stage']=='SFT' and (config.get('text_manifest_sha256')!=sha(DATA/'text_train.jsonl') or config.get('text_loss_weight')!=.15 or config.get('selection_metric')!='group_macro'):raise ValueError('Completed SFT objective/source differs')
 if job['backend']=='smallthinker4':
  from src.utils import research_small_budget_v1 as small
  if job['recipe']=='1c':
   from src.utils.research_small_budget_v2 import implementation
   small=implementation()
  if out!=small.FRESH_A0:small.validate_execution(config)
  elif job['stage']!='A0' or job['recipe']!='1b':raise ValueError('Fresh A0 exception cannot be reused')
 else:
  import importlib
  module=importlib.import_module(backend_module(job['backend'],job['recipe']));impl=module.implementation() if hasattr(module,'implementation') else module;impl.validate_config(config);impl.validate_context(config)
 manager=BestLastCheckpointManager(out,DiskBudget(out,0),mode=owner['mode'])
 with manager._lock():
  roles={name:manager._resolve_owned_pointer(name) for name in ('best','last')}
  for name,path in roles.items():
   if path is None:raise ValueError('Completed retained role missing')
   marker=read(path/'.research_checkpoint.json')
   if marker.get('step')!=job['expected_steps']:raise ValueError('Final-only retained checkpoint step differs')
 return {'status':'completed_training_metadata_verified_no_quality_claim','completed':completion,'config_sha256':sha(out/'resolved_config.json'),'manager_owner':owner['owner'],'roles':{k:str(v/'state.pt') for k,v in roles.items()},'source':'completed_owned_terminal_artifacts','verified_unix':time.time()}

def live_external(job):
 allowed={'run_research_small_budget_v1.py','run_research_stock_deterministic.py','run_research_lfm_budget_v1.py','run_research_giga_budget_v1.py'};matches=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  info=proc(int(p.name))
  if not info or info['state'] in ('Z','X'):continue
  argv=info['argv']
  if not any(Path(a).name in allowed for a in argv):continue
  destination=value(argv,'--output')
  if destination and (ROOT/destination).resolve()==directory(job):matches.append(info)
 if len(matches)>1:raise RuntimeError('Multiple external trainers name this exact run')
 return matches[0] if matches else None

def small_validator(run):
 from src.utils import research_stock_determinism as stock,research_small_budget_v1 as budget
 config=read(Path(run)/'resolved_config.json');stock.activate()
 if config['recipe']=='1c':
  from src.utils.research_small_budget_v2 import implementation
  budget=implementation()
 with budget.patched_stock_runner(DATA/'release.json'):
  private=stock.runner()
  if config['vision']==VISIONS['qwen25']:
   from src.utils import research_derived_stock_v1 as composed,research_derived_vision_v1 as derived
   binding=derived.resolve_derived_vision(config['vision'],preprocessing_profile=config['vision_preprocessing_profile']);private=composed.runner(__import__('types').SimpleNamespace(**config),binding)
  module=stock._clone('scripts/validate_research_checkpoint.py',{'scripts.run_research_stage':private});native=module.atomic_json
  module.atomic_json=lambda path,payload:native(path,dict(payload,numerical_profile=stock.definition(),budget18_execution=budget.definition()))
  return module.validate_run(Path(run).resolve(),device='cuda')

def worker(request):
 if sha(request)!=ARGS.request_sha256:raise ValueError('Worker request bytes changed')
 req=read(request)
 if req.get('kind') not in ('train','eval','validate') or req.get('backend') not in MODELS:raise ValueError('Unknown fixed budget worker dispatch')
 if req['owner']!=OWNER or req['controller_source_sha256']!=sha(__file__) or req['data_release_sha256']!=RELEASE_SHA:raise ValueError('Worker source/data identity differs')
 parent=req['parent'];actual=proc(parent['pid'])
 if not same(parent,actual) or os.getppid()!=parent['pid']:raise RuntimeError('Worker parent birth/argv changed')
 libc=ctypes.CDLL(None,use_errno=True)
 if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError('Cannot arm parent-death signal')
 if not same(parent,proc(parent['pid'])) or os.getppid()!=parent['pid']:raise RuntimeError('Parent died during worker handshake')
 atomic(Path(request).with_name('worker_ready.json'),{'nonce':req['nonce'],'worker':proc(os.getpid()),'request_sha256':sha(request)})
 resources(idle=True)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Owned GPU worker must expose only GPU0')
 kind=req['kind'];family=req['backend'];argv=req['argv'];import importlib
 recipe=value(argv,'--recipe') if kind=='train' else read(Path(value(argv,'--run'))/'resolved_config.json')['recipe'];backend=backend_module(family,recipe)
 if kind=='train':return importlib.import_module(backend).train_main(['--allow-gpu',*argv]) if family!='smallthinker4' else importlib.import_module(backend).main(['--allow-gpu',*argv])
 if family=='smallthinker4':
  from src.utils import research_stock_determinism as stock
  if kind=='validate':return small_validator(value(argv,'--run'))
  config=read(Path(value(argv,'--run'))/'resolved_config.json')
  if config['vision']==VISIONS['qwen25']:
   from src.utils.research_derived_stock_v1 import evaluation_main
   return evaluation_main(['--allow-gpu','--numerical-profile',stock.PROFILE,*argv])
  return stock.eval_main(['--allow-gpu','--numerical-profile',stock.PROFILE,*argv])
 module=importlib.import_module(backend);return (module.eval_main if kind=='eval' else module.validate_main)(['--allow-gpu',*argv])

class Queue:
 def __init__(self,spec,state,retry=False):
  self.spec=verify_plan(read(spec));self.path=Path(state);self.path.parent.mkdir(parents=True,exist_ok=True);self.handle=self.path.with_suffix('.lock').open('a');fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.spec_sha=digest(self.spec);self.state=read(self.path) if self.path.exists() else {'owner':OWNER,'spec_sha256':self.spec_sha,'jobs':{},'created':time.time()}
  if self.state.get('owner')!=OWNER or self.state.get('spec_sha256')!=self.spec_sha:raise ValueError('Queue state/spec differs')
  self.retry=retry;self.stopped=False;self.child=None;self.child_info=None;self.oracle=oracle(self.spec)
 def persist(self):self.state['updated']=time.time();atomic(self.path,self.state)
 def signal(self,signum,_frame):
  self.stopped=True;self.state['interrupted_by']=signum;self.persist()
  if self.child and same(self.child_info,proc(self.child.pid)):
   try:os.killpg(self.child.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 def execute(self,job,kind,argv,row):
  resources(idle=True);verify_plan(self.spec)
  if self.stopped:raise InterruptedError('No new worker after queue interrupt')
  folder=ROOT/'logs/research/budget18_workers'/job['id']/(kind+'_'+uuid.uuid4().hex);folder.mkdir(parents=True);request=folder/'request.json';parent=proc(os.getpid())
  req={'owner':OWNER,'kind':kind,'backend':job['backend'],'argv':argv,'parent':parent,'nonce':uuid.uuid4().hex,'controller_source_sha256':sha(__file__),'data_release_sha256':RELEASE_SHA};atomic(request,req)
  command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-request',str(request),'--request-sha256',sha(request)];receipt={'status':'running','command':command,'request':str(request),'log':str(folder/'worker.log'),'started':time.time()};row[kind+'_process']=receipt;self.persist()
  with (folder/'worker.log').open('a') as log:
   self.child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic();self.child_info=proc(self.child.pid);receipt['pid']=self.child.pid;self.persist();last=start
   deadline=time.monotonic()+15
   while not (folder/'worker_ready.json').is_file() and self.child.poll() is None:
    current=proc(self.child.pid)
    if current and current['argv'] and current['argv']!=command:raise RuntimeError('Owned worker nonempty argv differs')
    if self.child_info and current and (current['birth']!=self.child_info['birth'] or current['ppid']!=os.getpid()):raise RuntimeError('Owned worker birth/PPID changed')
    if current:self.child_info=current
    if time.monotonic()>deadline:raise TimeoutError('Owned worker startup handshake exceeded15s')
    time.sleep(.05)
   if (folder/'worker_ready.json').is_file():
    ready=read(folder/'worker_ready.json')
    if ready.get('nonce')!=req['nonce'] or ready.get('worker',{}).get('argv')!=command or ready['worker'].get('ppid')!=os.getpid():raise ValueError('Worker startup nonce/final argv/PPID differs')
    self.child_info=ready['worker']
   if self.stopped and self.child.poll() is None and same(self.child_info,proc(self.child.pid)):os.killpg(self.child.pid,signal.SIGTERM)
   interrupted_at=None
   while self.child.poll() is None:
    if self.stopped:
     interrupted_at=interrupted_at or time.monotonic()
     if time.monotonic()-interrupted_at>30:
      if same(self.child_info,proc(self.child.pid)):os.killpg(self.child.pid,signal.SIGKILL)
    if time.monotonic()-last>=30:resources();self.persist();emit('worker_running',job=job['id'],kind=kind,pid=self.child.pid);last=time.monotonic()
    time.sleep(1)
   code=self.child.returncode;self.child=None;self.child_info=None
  receipt.update(status='completed' if code==0 else 'failed',returncode=code,finished=time.time(),wall_seconds=time.monotonic()-start);self.persist()
  if self.stopped:raise InterruptedError('Queue interrupted; only its owned worker was signaled')
  if code:raise RuntimeError(f'Owned {kind} worker exit{code}: {receipt["log"]}')
  ready=read(folder/'worker_ready.json')
  if ready.get('nonce')!=req['nonce'] or ready.get('request_sha256')!=sha(request) or ready.get('worker',{}).get('pid')!=receipt['pid']:raise ValueError('Owned worker startup receipt differs')
 def adopt_or_train(self,job,row):
  out=directory(job);service=MANUAL_SERVICES.get(job['id']);supervisor=supervisor_info(service) if service else None;live=live_external(job)
  if supervisor:
   row['external_supervisor_before']=supervisor
   if live and (supervisor['statename']!='RUNNING' or supervisor['pid']!=live['pid']):raise RuntimeError('Manual trainer Supervisor PID differs')
   self.persist()
  if live:
   row.update(status='waiting_external_owned_trainer',external_trainer=live);self.persist();emit('waiting_external_trainer',job=job['id'],pid=live['pid'])
   while same(live,proc(live['pid'])):
    if self.stopped:raise InterruptedError('Stopped while waiting; external trainer not signaled')
    resources();time.sleep(2)
   row['external_trainer_naturally_exited']=True;self.persist()
  if (out/'completed.json').exists():
   if service:
    final=supervisor_info(service);deadline=time.monotonic()+15
    while final['statename']=='RUNNING' and time.monotonic()<deadline and final['start']==supervisor['start']:
     if self.stopped:raise InterruptedError('Stopped waiting for external Supervisor natural reap')
     time.sleep(.1);final=supervisor_info(service)
    if final['statename']!='EXITED' or final['exitstatus']!=0 or (supervisor and final['start']!=supervisor['start']):raise ValueError('Manual Supervisor did not naturally exit0 under unchanged startup identity')
    row['external_supervisor_final']=final
   elif not row.get('training_completed') and row.get('train_process',{}).get('returncode')!=0:raise ValueError('Unregistered completed output cannot be silently adopted')
   row.update(training_completed=True,training_proof=validate_completed(job),training_adopted_without_new_updates=True);self.persist();return
  if live:raise RuntimeError('External trainer exited without complete terminal artifacts; no automatic retraining')
  if row.get('training_completed'):raise RuntimeError('Previously completed source disappeared; no retraining')
  argv=list(job['argv'])
  if (out/'last/state.pt').is_file():argv.append('--resume')
  self.execute(job,'train',argv,row);row.update(training_completed=True,training_proof=validate_completed(job));self.persist()
 def postprocess(self,job,row):
  if job['stage']!='SFT':return
  from scripts.compact_research_checkpoints import compact_completed_checkpoint
  from scripts.research_grounding_queue import read_rows,paired_image_bootstrap,MODES
  if not row.get('compaction'):
   resources(idle=True);row['status']='compacting';self.persist();row['compaction']=compact_completed_checkpoint(directory(job),reserve_gb=120);self.persist()
  identity=self.oracle._evaluation_identity(job);original_checkpoint=identity['checkpoint_sha256']
  for variant in ('none',*MODES):
   key='evaluation' if variant=='none' else 'grounding_'+variant
   current=identity if variant=='none' else self.oracle._grounding_identity(job,variant)
   if row.get(key,{}).get('status')!='completed':
    args=['--run',str(directory(job)),'--manifest',str(DATA/'heldout_evaluation.jsonl'),'--output',str(current['output']),'--checkpoint','best','--max-new-tokens','64','--reserve-gb','120']
    if variant!='none':args+=['--image-ablation',variant,'--ablation-seed','42']
    if (current['output']/'protocol.json').exists():args.append('--resume')
    row['status']='evaluating_'+variant;self.persist();self.execute(job,'eval',args,row)
   fresh=self.oracle._evaluation_identity(job) if variant=='none' else self.oracle._grounding_identity(job,variant)
   if fresh['checkpoint_sha256']!=original_checkpoint:raise ValueError('Checkpoint changed during evaluation sequence')
   protocol=read(fresh['output']/'protocol.json')
   if protocol.get('numerical_profile')!=expected_profile(job['backend'],job['recipe']):raise ValueError('Evaluation numerical profile differs')
   row[key]=self.oracle._validate_evaluation(job,fresh) if variant=='none' else self.oracle._validate_grounding(job,fresh,variant);self.persist()
   if variant=='none':
    checkpoint=self.oracle._checkpoint_validation_identity(job)
    if row.get('checkpoint_validation',{}).get('status')!='completed':self.execute(job,'validate',['--run',str(directory(job)),'--device','cuda'],row)
    row['checkpoint_validation']=self.oracle._validate_checkpoint_report(checkpoint);self.persist()
  records=read_rows(DATA/'heldout_evaluation.jsonl');base=read_rows(Path(row['evaluation']['output'])/'predictions.jsonl');variants={m:read_rows(Path(row['grounding_'+m]['output'])/'predictions.jsonl') for m in MODES}
  summary={'owner':OWNER,'status':'completed','ordinary_evaluation':row['evaluation'],'checkpoint_validation':row['checkpoint_validation'],'paired_image_group_bootstrap':paired_image_bootstrap(base,variants,records),'source_sha256':sha(__file__),'checkpoint_sha256':original_checkpoint}
  p=directory(job)/'evaluation/budget18_native_v1/paired_summary.json';atomic(p,summary);row['paired_summary']={'path':str(p),'sha256':sha(p)};self.persist()
 def prune(self):
  from scripts.research_queue import prune_checkpoint_run
  for job in self.spec['jobs']:
   row=self.state['jobs'].get(job['id'],{})
   if not job['discard_checkpoint_after_dependents'] or row.get('status')!='completed' or row.get('checkpoint_pruned'):continue
   consumers=[j for j in self.spec['jobs'] if job['id'] in j['requires']]
   if consumers and all(self.state['jobs'].get(j['id'],{}).get('status')=='completed' for j in consumers):
    row['prune_receipt']=prune_checkpoint_run(directory(job),[directory(j) for j in consumers]);row['checkpoint_pruned']=True;self.persist()
 def run(self):
  self.state['status']='running';self.persist()
  try:
   for job in self.spec['jobs']:
    if self.stopped:break
    row=self.state['jobs'].setdefault(job['id'],{})
    if row.get('status')=='completed':
     if row.get('checkpoint_pruned'):
      receipt=read(directory(job)/'checkpoint_pruned.json')
      if job['stage']=='SFT' or receipt.get('status')!='pruned' or receipt.get('owner')!=row['training_proof']['manager_owner'] or receipt.get('source_run')!=str(directory(job)):raise ValueError('Completed intermediate pruning receipt differs')
      if sha(directory(job)/'resolved_config.json')!=row['training_proof']['config_sha256']:raise ValueError('Pruned completed source config changed')
     else:
      validate_completed(job)
      if job['stage']=='SFT':self.postprocess(job,row)
     continue
    if row.get('status')=='failed' and not self.retry:raise RuntimeError('Explicit --retry-failed required: '+job['id'])
    if any(self.state['jobs'].get(d,{}).get('status')!='completed' for d in job['requires']):raise RuntimeError('Dependency incomplete')
    try:
     verify_plan(self.spec);resources();self.state['active_job']=job['id'];self.persist();self.adopt_or_train(job,row);self.postprocess(job,row);row['status']='completed';self.persist();self.prune()
    except BaseException as e:
     row.update(status='interrupted' if isinstance(e,InterruptedError) else 'failed',error_type=type(e).__name__,error=str(e));self.persist();raise
   self.state['status']='interrupted' if self.stopped else 'completed';self.persist();return 0
  finally:
   if self.child and self.child.poll() is None and same(self.child_info,proc(self.child.pid)):
    try:os.killpg(self.child.pid,signal.SIGTERM);self.child.wait(timeout=10)
    except subprocess.TimeoutExpired:
     if same(self.child_info,proc(self.child.pid)):os.killpg(self.child.pid,signal.SIGKILL);self.child.wait(timeout=10)
    except ProcessLookupError:pass
   self.handle.close()

ARGS=None
def main():
 global ARGS
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,default=PLAN);p.add_argument('--state',type=Path,default=ROOT/'runs/research/budget18_queue_state_v1.json');p.add_argument('--write-plan',action='store_true');p.add_argument('--run-queue',action='store_true');p.add_argument('--retry-failed',action='store_true');p.add_argument('--worker-request',type=Path);p.add_argument('--request-sha256');ARGS=p.parse_args()
 if ARGS.worker_request:
  worker(ARGS.worker_request);return 0
 if ARGS.write_plan:
  if ARGS.spec.exists():raise FileExistsError('Immutable plan already exists')
  atomic(ARGS.spec,make_plan());print(sha(ARGS.spec));return 0
 spec=verify_plan(read(ARGS.spec))
 if not ARGS.run_queue:print(json.dumps({'status':'metadata_preflight_only','training_jobs':len(spec['jobs']),'scientific_cells':18,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']}));return 0
 queue=Queue(ARGS.spec,ARGS.state,ARGS.retry_failed);signal.signal(signal.SIGTERM,queue.signal);signal.signal(signal.SIGINT,queue.signal);return queue.run()
if __name__=='__main__':raise SystemExit(main())
