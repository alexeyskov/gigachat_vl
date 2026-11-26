"""Explicit reduced-budget native LFM execution; canonical research sources unchanged.

FP32 full-training masters, strict stock Torch determinism, CPU-gradient Adam8
for full SFT, complete projector restores, full-target context preflight and one
final native best/last object. Full-size CUDA fit remains a measured-run gate.
"""
from __future__ import annotations
import argparse,hashlib,io,json,math,os,sys,types,uuid,zipfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
MODEL='LiquidAI/LFM2.5-8B-A1B';VISION='google/siglip2-so400m-patch14-384'
REVISION='dd83f5cb02aa2938b66bba9a9998d38d0517c1f3'
QWEN='Qwen/Qwen2.5-VL-7B-Instruct';QWEN_PROFILE='qwen_native_bounded_729'
PROFILE='budget_lfm_native_stock_deterministic_v1';GIB=2**30
STORAGE='src/utils/research_budget_storage_v1.py'
STORAGE_SHA='f5e33c4fdea84f9ec8e180dda786e4301cfd02f122fc985a24d91d82858acbfe'
OWN=('src/utils/research_lfm_budget_v1.py','scripts/run_research_lfm_budget_v1.py','scripts/eval_research_lfm_budget_v1.py','scripts/validate_research_lfm_budget_v1.py')
CLOCK_SOURCE='src/eval/research_text_generation.py'
CLOCK_SHA='b311a58969a28888169ed965c5d919b9dee2ddf4743787655e54af9d33182e0e'
STRICT_P='src/utils/research_lfm_checkpoint_requirements_v1.py'
STRICT_P_SHA='aa3aacb07b148a8b03f7ade44b27ff9bb48c46f674f4fbe172093985c0db648c'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for raw in iter(lambda:f.read(8*2**20),b''):h.update(raw)
 return h.hexdigest()
def bootstrap():
 from src.utils.research_stock_determinism import bootstrap_environment
 bootstrap_environment()
def definition():
 from src.utils import research_stock_determinism as stock
 base=stock.definition()
 from src.utils import research_derived_vision_v1 as derived
 if sha(ROOT/STRICT_P)!=STRICT_P_SHA or sha(ROOT/STORAGE)!=STORAGE_SHA or sha(ROOT/CLOCK_SOURCE)!=CLOCK_SHA:raise ValueError('Frozen complete-projector/storage helper changed')
 return {'name':PROFILE,'schema_version':1,'model':MODEL,'model_revision':REVISION,'visions':[VISION,QWEN],'recipes':['1b','4f','3c'],
  'deterministic_algorithms':True,'warn_only':False,'cublas_workspace_config':stock.WORKSPACE,
  'matmul_allow_tf32':True,'torch_threads':8,'runtime_versions':base['runtime_versions'],'package_versions':base['package_versions'],
  'fixed_native_template_date_UTC':'2026-11-22','full_master_dtype':'float32','compute_dtype':'bfloat16','full_SFT_optimizer':'cpu_grad_bnb_paged_adamw8',
  'full_optimizer_moment_bits':8,'full_optimizer_is_not_FP32_AdamW':True,'vision_tokens':729,
  'minimum_disk_reserve_GiB':120,'checkpoint_policy':'one_final_native_object_shared_best_last_complete_optimizer_scheduler_RNG',
  'source_sha256':dict(base['source_sha256'],**{p:sha(ROOT/p) for p in (*OWN,STRICT_P,STORAGE,CLOCK_SOURCE)}),
  'derived_vision_code_sha256':derived.code_binding(),'actual_full_size_LFM_GPU_validated':False}
def fixed_clock(tokenizer):
 from src.eval.research_text_generation import FixedClockTokenizer,DATE
 if DATE!='2026-11-22' or sha(ROOT/CLOCK_SOURCE)!=CLOCK_SHA:raise ValueError('Pinned native template clock changed')
 return tokenizer if isinstance(tokenizer,FixedClockTokenizer) else FixedClockTokenizer(tokenizer)

def activate():
 from src.utils import research_stock_determinism as stock
 stock.activate()
def assert_active():
 from src.utils import research_stock_determinism as stock
 stock.assert_active()
def validate_config(config):
 if config.get('model')!=MODEL or config.get('numerical_profile')!=definition():raise ValueError('Explicit exact LFM budget numerical profile required; legacy/mixed resume refused')

def host_guard(extra):
 from src.utils.research_lossless_delta import memory_guard
 return memory_guard(int(extra),headroom_bytes=32*GIB)

def validate_budget_args(args,*,technical=False):
 if not math.isfinite(args.reserve_gb) or args.reserve_gb<120:raise ValueError('Preserve finite minimum120GiB disk reserve')
 if args.model!=MODEL or args.vision not in (VISION,QWEN):raise ValueError('This version supports pinned native LFM + SigLIP2/Qwen729 only')
 if args.max_length!=4096:raise ValueError('Reduced series uses fixed4096 native context')
 if args.recipe not in ('1b','4f','3c') or args.seed!=42 or args.epochs!=1 or args.effective_batch_size!=64:raise ValueError('Budget series requires1b/4f/3c,seed42,one epoch,effective64')
 if args.eval_every!=10000 or args.checkpoint_format!='legacy':raise ValueError('Budget series requires final-only native checkpoint, eval_every10000')
 if args.alignment_checkpoint:raise ValueError('Use fully validated --initial-checkpoint, not projector-only shortcut')
 if not technical and (args.max_steps is not None or args.max_records is not None):raise ValueError('Scientific budget manifest must run one complete epoch; truncation only explicit technical smoke')
 expected='cpu_grad_bnb_paged_adamw8' if args.recipe=='4f' and args.stage!='A0' or args.recipe=='3c' and args.stage=='SFT' else 'torch_adamw32'
 if args.optimizer_backend!=expected:raise ValueError('Explicit optimizer backend required for this stage: '+expected)
 if args.stage=='SFT' and args.text_loss_weight!=.15:raise ValueError('SFT requires fixed .85/.15 stream-normalized loss')
 if args.vision==VISION and args.vision_preprocessing_profile is not None:raise ValueError('SigLIP2 uses unchanged native729 preprocessing')
 if args.vision==QWEN and args.vision_preprocessing_profile!=QWEN_PROFILE:raise ValueError('Qwen requires explicit qwen_native_bounded_729 derived profile')

def context_preflight(module,args,*,tokenizer=None,model_ref=None):
 """No model weights/vision/RGB body read. Reject ANY full-target truncation."""
 if tokenizer is None:
  from transformers import AutoTokenizer
  model_path,model_ref=module.resolve_model(args.model)
  tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
 tokenizer=fixed_clock(tokenizer)
 fake=SimpleNamespace(tokenizer=tokenizer,args=args,base=SimpleNamespace(config=SimpleNamespace(model_type='lfm2_moe')))
 fields=('manifest','validation','text_manifest','text_validation','selection_manifest');files={};counts={};maxima={};identity={}
 for field in fields:
  value=getattr(args,field,None)
  if not value:continue
  path=Path(value).resolve(strict=True);st=path.stat();before=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns);digest=sha(path);count=0;maximum=0;image=0;tokens=0
  with path.open() as f:
   for line in f:
    if not line.strip():continue
    record=json.loads(line);n=len(record.get('image_paths',[]))
    if n>1 or len(record.get('image_hashes',[]))!=n:raise ValueError('Single-image records need canonical RGB hashes')
    if any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in record.get('image_hashes',[])):raise ValueError('Canonical RGB SHA256 required')
    pre,post,answer=module.ResearchModel.tokenize(fake,record)
    expanded=len(pre)+len(post)+len(answer)+729*n
    if expanded>args.max_length:raise ValueError(f'FULL input+target context overflow {field}:{record.get("record_id",count)}={expanded}>{args.max_length}; no target truncation admitted')
    count+=1;image+=bool(n);tokens+=len(answer);maximum=max(maximum,expanded)
  now=path.stat()
  if before!=(now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns):raise ValueError('Manifest changed during context census')
  files[field]={'path':str(path),'sha256':digest,'bytes':st.st_size};counts[field]={'records':count,'image_records':image,'text_records':count-image,'full_supervised_tokens':tokens};maxima[field]=maximum
 if not counts.get('manifest',{}).get('records') or not counts.get('validation',{}).get('records'):raise ValueError('Nonempty train/validation required')
 mm=counts['manifest']['records'];wanted=round(mm*args.text_fraction/(1-args.text_fraction)) if args.stage=='SFT' and args.text_manifest else 0
 if wanted>counts.get('text_manifest',{}).get('records',0):raise ValueError('Insufficient declared text replay')
 total=mm+wanted;selected_total=min(total,args.max_records or total)
 if math.ceil(total/64)>10000:raise ValueError('Budget context would trigger intermediate checkpoints')
 return {'schema_version':1,'scope':'all_declared_inputs_native_schema2_full_target_context_no_pixels_or_weights',
  'model_ref':model_ref,'prompt_schema_version':2,'vision_tokens_per_image_upper_bound':729,'vision_tokens_exact':args.vision==VISION,'max_length':args.max_length,
  'files':files,'counts':counts,'max_expanded_tokens':maxima,'declared_epoch_records':total,'actual_epoch_records':selected_total,'actual_text_replay_records':wanted,
  'total_optimizer_steps':min(args.max_steps or math.ceil(selected_total/64),math.ceil(selected_total/64)),
  'source_rgb_verified_by_this_tokenizer_audit':False,'all_full_targets_fit':True,
  'native_tokenize_source_sha256':sha(ROOT/'scripts/run_research_stage.py')}

def bind_data_proof(path,context):
 path=Path(path).resolve(strict=True)
 if path.stat().st_size>8*2**20:raise ValueError('Bounded JSON source proof required')
 before=path.stat();raw=path.read_bytes();value=json.loads(raw)
 if not isinstance(value,dict) or value.get('status')!='passed' or value.get('profile')!='budget18-en-v1':raise ValueError('Passed exact budget18 data source proof required')
 strings=set()
 def walk(v):
  if isinstance(v,str):strings.add(v)
  elif isinstance(v,dict):
   for child in v.values():walk(child)
  elif isinstance(v,list):
   for child in v:walk(child)
 walk(value)
 for item in context['files'].values():
  declared=value.get('files',{}).get(Path(item['path']).name,{})
  if item['sha256'] not in strings or declared.get('sha256')!=item['sha256'] or declared.get('path')!=item['path'] or declared.get('bytes')!=item['bytes']:raise ValueError('Data source proof does not bind every actual input manifest SHA/path/size')
 after=path.stat()
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('Data source proof changed while binding')
 return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'all_context_manifest_SHA_present':True,
  'manifest_hashes':{k:v['sha256'] for k,v in context['files'].items()},'source_audit_semantics':'ROOT released data producer receipt; exact bytes and manifest hashes bound, no image audit rerun here'}

def validate_context(config):
 receipt=config.get('budget_context')
 if not isinstance(receipt,dict) or not receipt.get('all_full_targets_fit'):raise ValueError('Whole-input context receipt missing')
 for item in receipt['files'].values():
  if sha(item['path'])!=item['sha256']:raise ValueError('Bound context manifest changed')
 proof=config.get('budget_data_proof')
 if proof and (sha(proof['path'])!=proof['sha256'] or bind_data_proof(proof['path'],receipt)!=proof):raise ValueError('Budget data source proof changed')
 return receipt

def memory_admission(model):
 import torch
 parameters=[p for p in model.parameters() if p.requires_grad];n=sum(p.numel() for p in parameters);largest=max((p.numel() for p in parameters),default=0)
 if any(p.dtype!=torch.float32 for p in parameters):raise ValueError('Every trainable tensor must have FP32 master storage')
 inference_only=getattr(model.args,'budget_runtime_inference',False)
 offload=model.args.optimizer_backend=='cpu_grad_bnb_paged_adamw8' and not inference_only
 # Includes small-state FP32 fallback conservatively: <=4096 elements per tensor.
 small=sum(p.numel() for p in parameters if p.numel()<4096);host=6*n+6*small+2*GIB if offload else 2*GIB
 host_snapshot=host_guard(host)
 shadow=sum(p.numel()*2 for p in model.llm.parameters() if p.is_floating_point() and p.dtype==torch.float32)
 gpu_extra=shadow+(0 if inference_only else 4*largest+2*GIB if offload else 12*n)+10*GIB
 free,total=torch.cuda.mem_get_info()
 if free<gpu_extra:raise MemoryError(f'LFM estimated future CUDA allocation refused: free={free}, extra={gpu_extra}; full-size fit not established')
 return {'trainable_parameters':n,'largest_parameter':largest,'cpu_grad_offload':offload,'future_host_bytes_estimate':host,
  'host':host_snapshot,'future_gpu_bytes_estimate':gpu_extra,'gpu_free_bytes':free,'gpu_total_bytes':total,
  'estimate_only':True,'actual_full_size_fit_proven':False,'CPU_page_cache_credit':0}

def payload_storage_bytes(payload):
 from src.utils.research_budget_storage_v1 import payload_storage_budget
 return payload_storage_budget(payload)['anticipated_bytes']

def validate_complete_state(model,state,required=None):
 import torch
 from src.utils.research_lfm_checkpoint_requirements_v1 import validate_lfm_initialization_projector
 validate_lfm_initialization_projector(model,state)
 expected=model.state_dict();required=set(required) if required is not None else {n for n,p in model.named_parameters() if p.requires_grad}
 if required-set(state):raise ValueError('Checkpoint omits learned LFM tensors: '+str(sorted(required-set(state))[:8]))
 if set(state)-set(expected):raise ValueError('Unknown learned checkpoint tensors')
 for name,value in state.items():
  if not torch.is_tensor(value) or value.shape!=expected[name].shape or value.dtype!=expected[name].dtype:raise ValueError('LFM saved tensor shape/dtype differs: '+name)
  if value.is_floating_point():
   for block in value.reshape(-1).split(4*2**20):
    if not torch.isfinite(block).all():raise ValueError('Nonfinite learned LFM tensor')
 return True

def required_initial_names(model,previous):
 names={n for n,p in model.named_parameters()}
 if previous['stage']=='A0':return {n for n in names if n.startswith('projector.')}
 if previous['stage']=='A1' and previous['recipe'] in ('1b','3c'):
  return {n for n in names if n.startswith('projector.') or any(x in n for x in ('visual_experts','router_adapter','router_new','router_mass','modality_gate_logits'))}
 return {n for n,p in model.named_parameters() if p.requires_grad}|{n for n in names if n.startswith('projector.')}

def vision_binding(args):
 if args.vision!=QWEN:return None
 from src.utils import research_derived_vision_v1 as derived
 binding=derived.resolve_derived_vision(args.vision,preprocessing_profile=args.vision_preprocessing_profile)
 derived.configure_args(args,binding)
 return binding

def runner(binding=None,*,inference_only=False):
 from src.utils import research_stock_determinism as stock
 module=stock._clone('scripts/run_research_stage.py');Native=module.ResearchModel;native_initial=module.validate_initial_checkpoint;native_resume=module.validate_resume_config
 from src.utils.research_lfm_checkpoint_requirements_v1 import validate_lfm_initialization_projector,validate_lfm_learned_state
 class BudgetModel(Native):
  def __init__(self,args):
   validate_config(vars(args));validate_context(vars(args))
   if not math.isfinite(args.reserve_gb) or args.reserve_gb<120:raise ValueError('Preserve finite minimum120GiB disk reserve')
   if not getattr(args,'budget_data_proof',None):raise ValueError('Actual constructor requires released data source proof')
   assert_active();host_guard(20*GIB)
   if inference_only:args.budget_runtime_inference=True
   if binding is not None:
    from src.utils import research_derived_vision_v1 as derived
    derived.assert_binding(binding,full=True,check=derived.host_guard)
   super().__init__(args)
   self.tokenizer=fixed_clock(self.tokenizer)
   if self.base.config.model_type!='lfm2_moe' or self.model_ref!={'repo':MODEL,'revision':REVISION}:raise ValueError('Exact pinned native LFM2-MoE required')
   self.budget_memory_admission=memory_admission(self)
   print(json.dumps({'event':'budget_memory_admission',**self.budget_memory_admission}),flush=True)
  def load_state_dict(self,state,*a,**kw):
   required=getattr(self,'_budget_initial_required',None)
   if hasattr(self,'_budget_initial_required'):del self._budget_initial_required
   validate_complete_state(self,state,required)
   return super().load_state_dict(state,*a,**kw)
 module.ResearchModel=BudgetModel
 def initial(payload,model,args):
  validate_config(payload.get('config',{}));validate_config(vars(args));validate_context(payload['config'])
  if not payload['config'].get('budget_data_proof'):raise ValueError('Initialization lacks data source proof')
  if payload['config'].get('budget_technical_smoke') and not getattr(args,'budget_technical_smoke',False):raise ValueError('Technical smoke checkpoint cannot initialize scientific training')
  native_initial(payload,model,args)
  required=required_initial_names(model,payload['config']);validate_complete_state(model,payload['model'],required)
  model._budget_initial_required=required
  return True
 module.validate_initial_checkpoint=initial
 def resume(saved,current):
  validate_config(saved);validate_config(current);validate_context(saved);validate_context(current)
  return native_resume(saved,current)
 module.validate_resume_config=resume
 native_binding=module.validate_model_binding
 def model_binding(payload,model):
  if model.args.vision==QWEN:
   from src.utils import research_derived_vision_v1 as derived
   derived.validate_payload_binding(payload,vision_binding(model.args))
  return native_binding(payload,model)
 module.validate_model_binding=model_binding
 def validate_final(payload):
  validate_config(payload['config']);validate_context(payload['config'])
  if payload['step']!=payload['config']['budget_context']['total_optimizer_steps']:raise ValueError('Only complete final-step checkpoints admitted')
 native_state=module.trainable_state_dict
 def state_for_checkpoint(model):
  host_guard(sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)+4*GIB)
  return native_state(model)
 module.trainable_state_dict=state_for_checkpoint
 from src.utils.research_budget_storage_v1 import install_manager
 install_manager(module,validate_final=validate_final,before_save=lambda payload,report:host_guard(2*GIB))
 if binding is not None:
  from src.utils import research_derived_vision_v1 as derived
  module=derived.install_private_runner_hooks(module,binding,model_repo=MODEL)
 else:
  def dispatch(args):
   if args.vision==QWEN:return runner(vision_binding(args),inference_only=inference_only).ResearchModel(args)
   return BudgetModel(args)
  dispatch.tokenize=Native.tokenize
  module.ResearchModel=dispatch
 return module

def source_archive(output,*,reserve_gb,create=True):
 from src.utils.research_storage import _atomic_json,DiskBudget
 output=Path(output);output.mkdir(parents=True,exist_ok=True);profile=definition();buffer=io.BytesIO()
 with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
  for name in sorted((*OWN,STRICT_P,STORAGE,CLOCK_SOURCE)):
   info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.external_attr=0o100644<<16;archive.writestr(info,(ROOT/name).read_bytes())
 data=buffer.getvalue();directory=output/'budget_lfm_execution_v1';receipt={'numerical_profile':profile,'source_zip_sha256':hashlib.sha256(data).hexdigest()}
 if directory.exists():
  if json.loads((directory/'receipt.json').read_text())!=receipt or (directory/'source.zip').read_bytes()!=data:raise ValueError('Execution sidecar differs')
  return receipt
 if not create:raise FileNotFoundError('Mandatory LFM source sidecar missing')
 DiskBudget(output,reserve_gb).ensure_space(len(data)+2**20)
 temporary=output/('.budget_lfm_execution_v1.'+uuid.uuid4().hex+'.tmp');temporary.mkdir()
 with (temporary/'source.zip').open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 _atomic_json(temporary/'receipt.json',receipt);fd=os.open(temporary,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 os.rename(temporary,directory);fd=os.open(output,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return receipt

def _options(argv):
 parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--metadata',action='store_true');parser.add_argument('--preflight',action='store_true');parser.add_argument('--allow-gpu',action='store_true');parser.add_argument('--technical-smoke',action='store_true');parser.add_argument('--data-proof');parser.add_argument('--numerical-profile',choices=[PROFILE],default=PROFILE)
 return parser.parse_known_args(argv)
def train_main(argv=None):
 opts,remaining=_options(argv)
 if opts.metadata:
  print(json.dumps({'status':'metadata_only_no_model_construction','numerical_profile':definition()},sort_keys=True));return
 bootstrap();module=runner();args=module.parse_args(remaining);module.validate_args(args);validate_budget_args(args,technical=opts.technical_smoke)
 args.numerical_profile=definition();args.budget_technical_smoke=opts.technical_smoke
 args.budget_data_proof=None
 binding=vision_binding(args)
 if binding is not None:module=runner(binding)
 args.budget_context=context_preflight(module,args)
 if opts.data_proof:args.budget_data_proof=bind_data_proof(opts.data_proof,args.budget_context)
 if not opts.technical_smoke:
  n=args.budget_context['counts']['manifest'];extra=args.budget_context['actual_text_replay_records']
  expected={'A0':(4096,4096,0),'A1':(256,256,0),'SFT':(3529,3000,529)}[args.stage]
  actual=(n['records']+extra,n['image_records'],n['text_records']+extra)
  if actual!=expected:raise ValueError('Declared reduced budget data count differs: '+str((actual,expected)))
 if opts.preflight:
  print(json.dumps({'status':'CPU_context_preflight_only','config':vars(args)},sort_keys=True));return vars(args)
 if not opts.allow_gpu:raise ValueError('Actual LFM execution needs explicit --allow-gpu')
 if args.budget_data_proof is None:raise ValueError('Actual execution requires --data-proof exact source receipt')
 activate();source_archive(args.output,reserve_gb=args.reserve_gb);module.parse_args=lambda:args;return module.main()

def inference():
 from src.utils import research_stock_determinism as stock
 private=runner(inference_only=True);module=stock._clone('src/eval/research_inference.py',{'scripts.run_research_stage':private});native=module._validate_learned_state
 from src.utils.research_lfm_checkpoint_requirements_v1 import validate_lfm_initialization_projector
 def learned(model,state):validate_complete_state(model,state);return native(model,state)
 module._validate_learned_state=learned;load=module.load_research_checkpoint
 def checkpoint(path,**kwargs):
  from src.utils.research_checkpoint_compat import load_research_payload
  saved=load_research_payload(path,map_location='cpu',mmap=True);validate_config(saved['config']);validate_context(saved['config'])
  if saved.get('model_ref')!={'repo':MODEL,'revision':REVISION}:raise ValueError('Checkpoint pinned LFM revision differs')
  runs=[p for p in Path(path).absolute().parents if (p/'resolved_config.json').is_file()]
  if not runs:raise ValueError('Owned run configuration/source sidecar missing')
  source_archive(runs[0],reserve_gb=0,create=False)
  if saved['config']['vision']==QWEN:
   from src.utils import research_derived_vision_v1 as derived
   derived.validate_payload_binding(saved,vision_binding(SimpleNamespace(**saved['config'])))
  del saved
  activate();model,provenance=load(path,**kwargs)
  return model,dict(provenance,numerical_profile=definition())
 module.load_research_checkpoint=checkpoint;return module

def eval_main(argv=None):
 opts,remaining=_options(argv)
 if opts.metadata:print(json.dumps({'status':'metadata_only_no_model_construction','numerical_profile':definition()}));return
 if not opts.allow_gpu:raise ValueError('Actual evaluation requires --allow-gpu')
 activate()
 from src.utils import research_stock_determinism as stock
 module=stock._clone('scripts/eval_research_checkpoint.py',{'src.eval.research_inference':inference()},bind_eval_protocol=True);module._bind_stock_protocol=lambda value:dict(value,numerical_profile=definition());return module.main(remaining)

def validate_main(argv=None):
 opts,remaining=_options(argv)
 if opts.metadata:print(json.dumps({'status':'metadata_only_no_model_construction','numerical_profile':definition()}));return
 if not opts.allow_gpu and '--cpu-structural-only' not in remaining:raise ValueError('Actual validation requires --allow-gpu or CPU structural mode')
 activate()
 from src.utils import research_stock_determinism as stock
 module=stock._clone('scripts/validate_research_checkpoint.py',{'scripts.run_research_stage':runner(inference_only=True)})
 native_validate=module.validate_run
 def validate(run,**kwargs):
  source_archive(run,reserve_gb=0,create=False);validate_config(json.loads((Path(run)/'resolved_config.json').read_text()));return native_validate(run,**kwargs)
 module.validate_run=validate;old=sys.argv
 try:sys.argv=[old[0],*remaining];return module.main()
 finally:sys.argv=old
