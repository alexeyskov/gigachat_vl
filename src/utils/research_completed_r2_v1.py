"""Prospective final-only completed4f model checkpoint policy, opt-in only.

Frozen numerical functions are unchanged. End-validation rows/textCE are durable
before the sole native ZIP publication. Only the completed optimizer is omitted.
"""
from __future__ import annotations
import ast,contextlib,hashlib,io,json,math,os,time,zipfile,fcntl,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OWNER='budget18_completed_R2_model_only_v1';DIRECTORY='completed_r2_finalization_v1'
NATIVE='scripts/run_research_stage.py';NATIVE_SHA='f7fd22d1c94fb337af2be60d3324415b2b2c8cd3d62e9799f2da4c05ace87cfc'
STREAM='src/utils/research_budget_stream_save_v1.py';_IMPORT_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def sha(p,*,check=None):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while True:
   if check is not None:check()
   b=f.read(8*2**20)
   if not b:break
   h.update(b)
 return h.hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
def read(p):
 p=Path(p)
 if p.is_symlink() or not p.is_file():raise ValueError('Owned finalization JSON must be a local regular file')
 return json.loads(p.read_bytes())
def atomic(p,v):
 from src.utils.research_storage import _atomic_json
 _atomic_json(Path(p),v)
def definition():
 if sha(__file__)!=_IMPORT_SHA or sha(ROOT/NATIVE)!=NATIVE_SHA:raise ValueError('CompletedR2/native source changed')
 return {'owner':OWNER,'version':1,'only_recipe':'4f','only_stage':'SFT','step':56,'seen':3529,'selection_records':116,'text_validation_records':34,'optimizer_resume_after_completed_stage':False,'model_scheduler_RNG_dtype_aliases_preserved':True,'disk_reserve_setting_and32GiB_host_floor_unchanged':True,'source_sha256':{'src/utils/research_completed_r2_v1.py':_IMPORT_SHA,NATIVE:NATIVE_SHA,STREAM:sha(ROOT/STREAM)}}
def eligible(config):
 if (config.get('recipe')!='4f' or config.get('stage')!='SFT' or config.get('seed')!=42 or config.get('epochs')!=1 or config.get('effective_batch_size')!=64 or config.get('eval_every')!=10000 or config.get('max_steps') is not None or config.get('max_records') is not None or config.get('selection_metric')!='group_macro' or config.get('selection_manifest') or config.get('text_validation') or config.get('text_loss_weight')!=.15):raise ValueError('Only exact completed budget18 final4fSFT eligible')
 steps=config.get('budget18_expected_steps',config.get('budget_context',{}).get('total_optimizer_steps'))
 if steps!=56:raise ValueError('Whole56step budget proof required')
 return True
def _directory(run):
 run=Path(run).absolute()
 if run.resolve()!=run or run.is_symlink() or (run/DIRECTORY).is_symlink():raise ValueError('Owned finalization namespace cannot be symlink')
 directory=run/DIRECTORY;directory.mkdir(parents=True,exist_ok=True);return directory
def attach(args):
 eligible(vars(args));args.completed_r2_execution_policy=definition();directory=_directory(args.output);receipt={'owner':OWNER,'policy':definition()}
 if (directory/'source.json').exists():
  if read(directory/'source.json')!=receipt:raise ValueError('CompletedR2 source policy changed')
 else:
  from src.utils.research_storage import DiskBudget
  DiskBudget(args.output,args.reserve_gb).ensure_space(2**20)
  atomic(directory/'source.json',receipt)
 if (directory/'source.py').exists():
  if (directory/'source.py').is_symlink() or sha(directory/'source.py')!=_IMPORT_SHA:raise ValueError('CompletedR2 copied source changed')
 else:
  with (directory/'source.py').open('xb') as f:f.write(Path(__file__).read_bytes());f.flush();os.fsync(f.fileno())
 return args
def _last_train(run):
 last=None
 with (Path(run)/'metrics.jsonl').open() as f:
  for line in f:
   row=json.loads(line)
   if row.get('event')=='train':last=row
 if last is None or last.get('step')!=56 or last.get('completed_records')!=3529:raise ValueError('Durable final56/3529 train progress required before validation')
 return last
def _files(run):
 return {name:{'sha256':sha(Path(run)/'selection'/name),'bytes':(Path(run)/'selection'/name).stat().st_size} for role in ('best','last') for name in (role+'.predictions.jsonl',role+'.protocol.json',role+'.metrics.json')}
def _end_validation(run,payload=None):
 directory=_directory(run);end=read(directory/'end_validation.json')
 if end.get('owner')!=OWNER or end.get('status')!='passed' or end['files']!=_files(run) or end.get('text',{}).get('text_only_records')!=34 or not math.isfinite(end['text']['text_only_heldout_ce']):raise ValueError('Complete immutable116+34 end-validation receipt required')
 rows=[]
 with (Path(run)/'selection/last.predictions.jsonl').open() as f:
  for line in f:rows.append(json.loads(line))
 if len(rows)!=116 or len({r['record_id'] for r in rows})!=116 or any(r.get('validation_step')!=56 for r in rows):raise ValueError('Full unique116 final native predictions required')
 if payload is not None:
  if payload.get('selection_protocol')!=end['selection_protocol'] or payload.get('selection_statistics')!=end['selection_metrics'] or payload.get('selection_value')!=end['selection_metrics']['group_macro_score']:raise ValueError('Final native payload and durable selection differ')
 return end
def _schema(model):return {n:{'shape':list(t.shape),'dtype':str(t.dtype),'stride':list(t.stride()),'storage_offset':t.storage_offset()} for n,t in model.items()}
def install_module(module):
 if module.__name__=='scripts.run_research_stage':raise ValueError('Canonical runner modification forbidden')
 if getattr(module,'_completed_R2_policy_installed',False):return module
 import torch
 from src.eval.research_selection import evaluate_internal_group_macro
 native_text=module.evaluate_text_only_ce;native_main=module.main;NativeManager=module.BestLastCheckpointManager
 def selection(model,selected,**kwargs):
  eligible(vars(model.args));_last_train(model.args.output)
  result=evaluate_internal_group_macro(model,selected,**kwargs)
  if result['metrics'].get('records')!=116 or len(result['predictions'])!=116 or len(result['metrics'].get('groups',{}))!=4:raise ValueError('Native final selection must contain all116/fourgroups')
  module.write_selection_artifacts(Path(model.args.output),result,56,True)
  atomic(_directory(model.args.output)/'selection.json',{'owner':OWNER,'selection_protocol':result['protocol'],'selection_metrics':result['metrics'],'files':_files(model.args.output)})
  return result
 def text(model,records,batch_size):
  result=native_text(model,records,batch_size)
  if len(records)!=34 or result is None or result.get('text_only_records')!=34 or result.get('text_only_supervised_tokens',0)<=0 or not math.isfinite(result['text_only_heldout_ce']):raise ValueError('Exact34heldout textCE required before final model-only publication')
  directory=_directory(model.args.output);selected=read(directory/'selection.json')
  if selected['files']!=_files(model.args.output):raise ValueError('Selection files changed before text validation')
  atomic(directory/'end_validation.json',dict(selected,status='passed',text=result,step=56,seen=3529));return result
 module.evaluate_text_only_ce=text;module.__dict__['_completed_R2_group_macro']=selection
 tree=ast.parse((ROOT/NATIVE).read_bytes());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');changes=0
 class ImportHook(ast.NodeTransformer):
  def visit_ImportFrom(self,node):
   nonlocal changes
   if node.module=='src.eval.research_selection' and len(node.names)==1 and node.names[0].name=='evaluate_internal_group_macro':
    changes+=1;return ast.Assign([ast.Name('evaluate_internal_group_macro',ast.Store())],ast.Name('_completed_R2_group_macro',ast.Load()))
   return node
 main=ImportHook().visit(main)
 if changes!=1:raise ValueError('Expected single frozen selection import hook changed')
 exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),str(ROOT/NATIVE),'exec'),module.__dict__);compiled_main=module.main
 class FinalManager(NativeManager):
  def save_torch(self,payload,**kwargs):
   config=payload['config'];eligible(config)
   if config.get('completed_r2_execution_policy')!=definition() or payload.get('step')!=56 or payload.get('seen')!=3529 or 'optimizer' not in payload or kwargs.get('checkpoint_format','legacy')!='legacy':raise ValueError('Only complete final nativeR2 payload with explicit policy can finalize')
   if kwargs.get('step')!=56 or kwargs.get('metric')!=payload.get('selection_value'):raise ValueError('Manager/native final step or metric differs')
   end=_end_validation(self.root,payload);config_file=Path(self.root)/'resolved_config.json'
   if read(config_file)!=config:raise ValueError('Native final configuration differs from durable config')
   original_keys=set(payload);filtered=dict(payload);del filtered['optimizer'];filtered.update(training_completed=True,resumable_optimizer=False,checkpoint_finalization={'owner':OWNER,'optimizer_removed_after_all_steps_and_end_validation':True})
   intent={'owner':OWNER,'policy':definition(),'config_sha256':sha(config_file),'end_validation_sha256':sha(_directory(self.root)/'end_validation.json'),'original_payload_keys':sorted(original_keys),'retained_payload_keys':sorted(filtered),'model_schema':_schema(filtered['model']),'steps':56,'records':3529}
   directory=_directory(self.root)
   if (directory/'intent.json').exists() and read(directory/'intent.json')!=intent:raise ValueError('Finalization intent differs; refuse overwrite')
   atomic(directory/'intent.json',intent);self._completed_R2_pending=filtered;self._completed_R2_intent=intent
   kwargs['metadata']=dict(kwargs.get('metadata',{}),completed_r2_owner=OWNER,training_completed=True,resumable_optimizer=False,intent_sha256=sha(directory/'intent.json'))
   try:
    result=super().save_torch(filtered,**kwargs)
    if result['best']['step']!=56 or (Path(self.root)/'best').resolve()!=(Path(self.root)/'last').resolve():raise ValueError('Final best/last must share the one final object')
    atomic(directory/'published.json',{'owner':OWNER,'status':'published_completed_model_only','intent_sha256':sha(directory/'intent.json'),'object':str((Path(self.root)/'last').resolve()),'resumable_optimizer':False});return result
   finally:self._completed_R2_pending=None
  def save(self,writer,**kwargs):
   payload=getattr(self,'_completed_R2_pending',None)
   if payload is None:raise ValueError('CompletedR2 manager cannot publish an unguarded checkpoint')
   def stream(directory):
    from src.utils.research_budget_stream_save_v1 import _Writer,CHUNK
    from src.utils.research_lossless_delta import memory_guard
    largest=max((t.untyped_storage().nbytes() for t in payload['model'].values()),default=0);memory_guard(2*largest+CHUNK)
    digest=hashlib.sha256()
    class HashWriter(_Writer):
     def write(inner,data):
      view=memoryview(data)
      for start in range(0,len(view),CHUNK):digest.update(view[start:start+CHUNK]);super(HashWriter,inner).write(view[start:start+CHUNK])
      return len(view)
    with (Path(directory)/'state.pt').open('xb') as f:
     sink=HashWriter(f,guard=memory_guard,budget=self.budget);torch.save(payload,sink);sink.commit()
    self._budget_stream_write={'only_new_staging_output_advised':True,'native_torch_serialization_unchanged':True,'assumed_cache_credit_bytes':0,'written_bytes':sink.total,'max_unflushed_output_bytes':CHUNK,'minimum_measured_available_bytes':sink.minimum}
    atomic(Path(directory)/'.completed_r2_final.json',{'owner':OWNER,'intent_sha256':sha(_directory(self.root)/'intent.json'),'state_sha256':digest.hexdigest(),'state_bytes':sink.total,'policy':definition()})
   return super().save(stream,**kwargs)
 module.BestLastCheckpointManager=FinalManager
 def policy_main(*a,**kw):
  args=attach(module.parse_args());old_parse=module.parse_args;module.parse_args=lambda:args
  try:
   if recover_if_published(args.output,expected_args=vars(args)):return
   return compiled_main(*a,**kw)
  finally:module.parse_args=old_parse
 module.main=policy_main;module._completed_R2_policy_installed=True;return module

def recover_if_published(run,*,expected_args=None):
 """No training/model constructor/optimizer replay; only a bound final object."""
 run=Path(run).absolute();directory=run/DIRECTORY
 if not (directory/'intent.json').exists():return False
 lock=run/'.checkpoint.lock'
 if not lock.is_file() or lock.is_symlink():raise ValueError('Owned manager lock required for finalization recovery')
 with lock.open('r+') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX)
  try:return _recover_locked(run,expected_args=expected_args)
  finally:fcntl.flock(handle,fcntl.LOCK_UN)
def _recover_locked(run,*,expected_args=None):
 directory=run/DIRECTORY
 intent=read(directory/'intent.json')
 if read(directory/'source.json')!={'owner':OWNER,'policy':definition()} or (directory/'source.py').is_symlink() or sha(directory/'source.py')!=_IMPORT_SHA:raise ValueError('Recovery copied source proof differs')
 if intent.get('owner')!=OWNER or intent.get('policy')!=definition() or sha(run/'resolved_config.json')!=intent['config_sha256'] or sha(directory/'end_validation.json')!=intent['end_validation_sha256']:raise ValueError('Finalization recovery source/config/end-validation differs')
 _end_validation(run);config=read(run/'resolved_config.json');eligible(config)
 if expected_args is not None and any(config.get(k)!=v for k,v in expected_args.items() if k!='resume'):raise ValueError('Recovered completed run differs from current invocation')
 targets=[];missing=[];manager=read(run/'.checkpoint_manager.json')
 for role in ('best','last'):
  p=run/role
  if not os.path.lexists(p):missing.append(role);continue
  if not p.is_symlink():raise ValueError('Recovery role is not owned symlink')
  targets.append(p.resolve(strict=True))
 if not targets:
  objects=run/'.checkpoint_objects'
  if objects.is_symlink() or not objects.is_dir():raise ValueError('Owned immutable object directory required')
  matches=[]
  for candidate in objects.iterdir():
   if candidate.is_symlink() or not candidate.is_dir() or not candidate.name.startswith('object-'):continue
   marker_path=candidate/'.research_checkpoint.json';receipt_path=candidate/'.completed_r2_final.json'
   if not marker_path.is_file() or not receipt_path.is_file():continue
   mark=read(marker_path);record=read(receipt_path)
   if mark.get('owner')==manager.get('owner') and mark.get('step')==56 and record.get('owner')==OWNER and record.get('intent_sha256')==sha(directory/'intent.json'):matches.append(candidate.resolve(strict=True))
  if not matches:return False
  if len(matches)!=1:raise ValueError('Ambiguous final immutable objects; recovery cannot choose scientific weights')
  targets=matches
 obj=targets[0]
 if any(p!=obj for p in targets) or obj.parent!=(run/'.checkpoint_objects').resolve() or obj.is_symlink():raise ValueError('Recovery requires shared owned final best/last')
 marker=read(obj/'.research_checkpoint.json');receipt=read(obj/'.completed_r2_final.json')
 if marker.get('owner')!=manager.get('owner') or marker.get('step')!=56 or receipt.get('owner')!=OWNER or receipt.get('policy')!=definition() or receipt['intent_sha256']!=sha(directory/'intent.json'):raise ValueError('Recovery owned object/source/step differs')
 state=obj/'state.pt'
 from src.utils.research_lossless_delta import memory_guard
 if state.is_symlink() or state.stat().st_size!=receipt['state_bytes'] or sha(state,check=lambda:memory_guard(8*2**20))!=receipt['state_sha256']:raise ValueError('Final model-only checkpoint physical bytes changed')
 memory_guard(128*2**20)
 import torch
 payload=torch.load(state,map_location='cpu',mmap=True,weights_only=False)
 if set(payload)!=set(intent['retained_payload_keys']) or payload.get('optimizer') is not None or payload.get('training_completed') is not True or payload.get('resumable_optimizer') is not False or payload['step']!=56 or payload['seen']!=3529 or payload['config']!=config or _schema(payload['model'])!=intent['model_schema']:raise ValueError('Recovered native model-only payload differs')
 _end_validation(run,payload)
 for role in missing:
  temporary=run/('.completed-R2-'+role+'-'+uuid.uuid4().hex)
  try:os.symlink(os.path.relpath(obj,run),temporary);os.replace(temporary,run/role)
  finally:temporary.unlink(missing_ok=True)
 from src.utils.research_storage import _fsync_directory
 _fsync_directory(run)
 completion={'steps':56,'records':3529,'elapsed_seconds':payload.get('elapsed_seconds',0),'completed_r2_recovery':True,'optimizer_resume_performed':False}
 if (run/'completed.json').exists():
  current=read(run/'completed.json')
  if current.get('steps')!=56 or current.get('records')!=3529:raise ValueError('Existing completion differs')
 else:atomic(run/'completed.json',completion)
 atomic(directory/'recovered.json',{'owner':OWNER,'status':'passed','completion':completion,'state_sha256':receipt['state_sha256'],'repaired_missing_roles':missing});return True

@contextlib.contextmanager
def patched_backend(backend):
 """Use only for explicit4f/SFT worker invocation; no file/global patch."""
 if hasattr(backend,'patch_module'):
  original=backend.patch_module
  def patch(*a,**k):return install_module(original(*a,**k))
  backend.patch_module=patch
  try:yield
  finally:backend.patch_module=original
 else:
  api=backend.implementation() if hasattr(backend,'implementation') else backend;original=api.runner
  def runner(*a,**k):return install_module(original(*a,**k))
  api.runner=runner
  try:yield
  finally:api.runner=original
