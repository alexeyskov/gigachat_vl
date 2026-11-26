"""Isolated SmallThinker budget guards/storage; stock numerical math unchanged."""
from __future__ import annotations
import argparse,ast,hashlib,io,json,math,os,sys,zipfile
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OWNER='small_stock_budget18_execution_v1';FIELD='budget18_execution'
DATA=ROOT/'datasets/research/budget18-en-v1';RELEASE=DATA/'release.json'
RELEASE_SHA='910448c7fd680ec949d6521e32716aee2832a99fc534d185f0a06b226c0ec1c0'
FRESH_A0=ROOT/'runs/research/smallthinker4__siglip2__A0_nf4__budget18_s42'
SIG='google/siglip2-so400m-patch14-384';QWEN='Qwen/Qwen2.5-VL-7B-Instruct'
OWN=('src/utils/research_small_budget_v1.py','scripts/run_research_small_budget_v1.py')
STORAGE='src/utils/research_budget_storage_v1.py';STORAGE_SHA='f5e33c4fdea84f9ec8e180dda786e4301cfd02f122fc985a24d91d82858acbfe'
PKEYS=('projector.0.weight','projector.0.bias','projector.2.weight','projector.2.bias')
CLOCK='src/eval/research_text_generation.py';CLOCK_SHA='b311a58969a28888169ed965c5d919b9dee2ddf4743787655e54af9d33182e0e'
def fixed_clock(tokenizer):
 if sha(ROOT/CLOCK)!=CLOCK_SHA:raise ValueError('Frozen native fixed-clock source changed')
 tree=ast.parse((ROOT/CLOCK).read_bytes());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FixedClockTokenizer']
 if len(nodes)!=1:raise ValueError('Fixed clock class changed')
 namespace={'DATE':'2026-11-22'};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(ROOT/CLOCK),'exec'),namespace)
 return namespace['FixedClockTokenizer'](tokenizer)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def read(path):
 p=Path(path)
 if p.is_symlink() or not p.is_file() or p.stat().st_size>16*2**20:raise ValueError('Bounded local metadata required')
 return json.loads(p.read_bytes())
def binding(path):
 p=Path(path).resolve(strict=True);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def definition():
 if sha(ROOT/STORAGE)!=STORAGE_SHA:raise ValueError('Frozen exact native storage helper changed')
 return {'owner':OWNER,'schema_version':1,'profile':'budget18-en-v1','source_sha256':{name:sha(ROOT/name) for name in (*OWN,STORAGE,CLOCK)},'data_release':{'path':str(RELEASE),'sha256':RELEASE_SHA},'stock_numerical_profile_unchanged':True,'fixed_clock':{'date':'2026-11-22','source_sha256':CLOCK_SHA,'pinned_Small_template_has_no_date_dependency_existing_stock_evaluation_is_prompt_equivalent':True},'seed':42,'epochs':1,'effective_batch_size':64,'eval_every':10000,'max_length':4096,'final_only_native_checkpoint':True,'R2_4f_A1_allowed':False}
def rows(path):
 with Path(path).open('rb') as f:
  for line in f:
   if not line.endswith(b'\n'):raise ValueError('Complete literalLF rows required')
   yield json.loads(line)
def data_proof(path):
 p=Path(path).resolve(strict=True)
 if p!=RELEASE or sha(p)!=RELEASE_SHA:raise ValueError('Exact frozen budget18 release910448 required')
 value=read(p)
 if value.get('status')!='passed' or value.get('profile')!='budget18-en-v1':raise ValueError('Passed budget18 data required')
 return value
def input_binding(path,proof):
 p=Path(path).resolve(strict=True);declared=proof['files'].get(p.name)
 actual=binding(p)
 if declared!=actual or p.parent!=DATA:raise ValueError('Input path/bytes/SHA differs from frozen budget18 data')
 return actual
def validate_execution(config):
 from src.utils import research_stock_determinism as stock
 stock.validate_config(config)
 if config.get(FIELD)!=definition():raise ValueError('Exact additional budget execution required; no legacy resume/relabel')
 proof=data_proof(config['budget18_data_proof']['path'])
 if config['budget18_data_proof']!=binding(RELEASE):raise ValueError('Budget source proof changed')
 for item in config['budget18_inputs'].values():
  if input_binding(item['path'],proof)!=item:raise ValueError('Budget input changed')
 return True
def require_complete_P(state,projector=None):
 import torch
 actual={name for name in state if isinstance(name,str) and name.startswith('projector.')}
 if actual!=set(PKEYS):raise ValueError('All four exact frozen FP32 projector tensors are mandatory')
 if projector is not None:expected={'projector.'+n:tuple(p.shape) for n,p in projector.named_parameters()}
 else:
  w=state[PKEYS[0]]
  if not torch.is_tensor(w) or w.ndim!=2:raise ValueError('MLP projector shape required')
  hidden,feature=w.shape;expected=dict(zip(PKEYS,((hidden,feature),(hidden,),(hidden,hidden),(hidden,))))
 for name in PKEYS:
  value=state[name]
  if not torch.is_tensor(value) or value.dtype!=torch.float32 or value.layout!=torch.strided or tuple(value.shape)!=expected[name]:raise ValueError('Projector shape/dtype/layout differs: '+name)
  for block in value.detach().reshape(-1).split(4*2**20):
   if not torch.isfinite(block).all():raise ValueError('Nonfinite projector')
 return True
def require_learned_state(model,state,required=None):
 import torch
 require_complete_P(state,model.projector);expected=model.state_dict();needed=set(required) if required is not None else {n for n,p in model.named_parameters() if p.requires_grad}
 if needed-set(state) or set(state)-set(expected):raise ValueError('Missing learned or unknown saved tensor keys')
 for name,value in state.items():
  if not torch.is_tensor(value) or tuple(value.shape)!=tuple(expected[name].shape) or value.dtype!=expected[name].dtype:raise ValueError('Saved learned tensor shape/dtype differs')
  if value.is_floating_point():
   for block in value.detach().reshape(-1).split(4*2**20):
    if not torch.isfinite(block).all():raise ValueError('Nonfinite learned tensor')
 return True
def stage_manifest(stage):return {'A0':'alignment_train.jsonl','A1':'alignment_a1_train.jsonl','SFT':'sft_train.jsonl'}[stage]
def initial_origin(args):
 if args.stage=='A0':return None
 if args.resume:
  saved=read(Path(args.output)/'resolved_config.json');validate_execution(saved)
  return saved.get('budget18_initialization_origin')
 if not args.initial_checkpoint:raise ValueError('Budget stages require full --initial-checkpoint')
 checkpoint=Path(args.initial_checkpoint).resolve(strict=True);runs=[p for p in checkpoint.parents if (p/'resolved_config.json').is_file()]
 if not runs:raise ValueError('Owned predecessor run required')
 run=runs[0];old=read(run/'resolved_config.json');from src.utils import research_stock_determinism as stock
 stock.validate_config(old)
 expected='A0' if args.stage=='A1' or args.recipe=='4f' else 'A1'
 if old.get('stage')!=expected or (expected=='A1' and old.get('recipe')!=args.recipe):raise ValueError('Budget uses A0→A1→SFT for1b/3c, A0→SFT for4f; no SFT reinitialization')
 labels=[role for role in ('best','last') if (run/role/'state.pt').exists() and (run/role/'state.pt').resolve()==checkpoint]
 completed=read(run/'completed.json')
 if not labels or completed.get('steps')!={'A0':64,'A1':4}[expected] or completed.get('records')!={'A0':4096,'A1':256}[expected]:raise ValueError('Exact completed retained predecessor required')
 if old.get('vision')!=args.vision or old.get('vision_preprocessing_profile')!=args.vision_preprocessing_profile:raise ValueError('Predecessor vision/profile differs')
 if FIELD not in old:
  if run!=FRESH_A0 or expected!='A0' or old.get('recipe')!='1b' or args.recipe!='1b' or old.get('vision')!=SIG:raise ValueError('Only the already-running fresh budget18 NF4 stock A0 may lack the new field')
  for key,value in {'seed':42,'epochs':1,'effective_batch_size':64,'max_length':4096,'eval_every':10000,'max_records':None,'max_steps':None}.items():
   if old.get(key)!=value:raise ValueError('Fresh source A0 exact budget differs')
  if Path(old['output']).resolve()!=FRESH_A0 or Path(old['manifest']).resolve()!=DATA/'alignment_train.jsonl' or old.get('manifest_sha256')!=sha(DATA/'alignment_train.jsonl') or old.get('validation_sha256')!=sha(DATA/'alignment_validation.jsonl'):raise ValueError('Fresh A0 source/manifest/config does not match budget18')
  kind='explicit_fresh_existing_stock_A0_origin_no_config_or_payload_rewrite'
 else:validate_execution(old);kind='budget18_predecessor'
 return {'kind':kind,'run':str(run),'checkpoint_path':str(checkpoint),'roles':labels,'stage':expected,'recipe':old['recipe'],'config':binding(run/'resolved_config.json'),'completed':binding(run/'completed.json'),'manifest_sha256':old['manifest_sha256'],'data_release_sha256':RELEASE_SHA}
def configure(args,proof_path):
 from src.utils import research_stock_determinism as stock
 if args.model!=stock.SMALL or args.recipe not in ('1b','4f','3c') or args.stage=='A1' and args.recipe=='4f':raise ValueError('Small budget1b/4f/3c only;4f A1 disallowed')
 if args.vision not in (SIG,QWEN) or (args.vision==SIG and args.vision_preprocessing_profile is not None) or (args.vision==QWEN and args.vision_preprocessing_profile!='qwen_native_bounded_729'):raise ValueError('Exact Sig native729 or explicit Qwen bounded729 required')
 for key,value in {'seed':42,'epochs':1,'effective_batch_size':64,'max_length':4096,'eval_every':10000,'max_records':None,'max_steps':None,'checkpoint_format':'legacy'}.items():
  if getattr(args,key,None)!=value:raise ValueError('Exact budget policy differs: '+key)
 if not math.isfinite(args.reserve_gb) or args.reserve_gb<120 or args.alignment_checkpoint or args.selection_manifest or args.text_validation:raise ValueError('Preserve120GiB/full initial-checkpoint/default frozen selection and validation')
 if args.stage=='SFT' and (args.text_loss_weight!=.15 or args.selection_metric!='group_macro' or args.selection_per_group!=32):raise ValueError('SFT fixed.85/.15 normalized objective and native116 group macro required')
 proof=data_proof(proof_path);inputs={};wanted=stage_manifest(args.stage)
 if args.stage=='SFT' and args.text_manifest:
  wanted='sft_multimodal_train.jsonl'
  if Path(args.text_manifest).resolve()!=DATA/'text_train.jsonl' or args.text_fraction!=.15:raise ValueError('Exact529 text replay required')
  inputs['text_manifest']=input_binding(args.text_manifest,proof)
 if Path(args.manifest).resolve()!=DATA/wanted or Path(args.validation).resolve()!=DATA/('sft_validation.jsonl' if args.stage=='SFT' else 'alignment_validation.jsonl'):raise ValueError('Stage-specific frozen budget manifests required')
 inputs.update(manifest=input_binding(args.manifest,proof),validation=input_binding(args.validation,proof))
 if args.stage!='SFT' and args.text_manifest:raise ValueError('Caption stages cannot add text replay')
 expected={'A0':(4096,4096,0),'A1':(256,256,0),'SFT':(3529,3000,529)}[args.stage];train=list(rows(args.manifest));image=sum(bool(r['image_paths']) for r in train);text=len(train)-image
 if args.text_manifest:text+=len(list(rows(args.text_manifest)))
 if (image+text,image,text)!=expected:raise ValueError('Whole epoch exact record counts required')
 args.numerical_profile=stock.definition();setattr(args,FIELD,definition());args.budget18_inputs=inputs;args.budget18_data_proof=binding(RELEASE);args.budget18_expected_steps={'A0':64,'A1':4,'SFT':56}[args.stage]
 args.allow_legacy_a0_initialization=False;args.budget18_initialization_origin=initial_origin(args)
 existing=Path(args.output)/'resolved_config.json'
 if existing.exists():validate_execution(read(existing))
 return args
def sidecar(output,reserve_gb=120):
 from src.utils.research_storage import _atomic_json,DiskBudget
 root=Path(output);directory=root/'budget18_execution_v1';buffer=io.BytesIO()
 with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
  for name in sorted((*OWN,STORAGE,CLOCK)):
   info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.external_attr=0o100644<<16;z.writestr(info,(ROOT/name).read_bytes())
 raw=buffer.getvalue();receipt={'execution':definition(),'source_zip_sha256':hashlib.sha256(raw).hexdigest()}
 if directory.exists():
  if read(directory/'receipt.json')!=receipt or (directory/'source.zip').read_bytes()!=raw:raise ValueError('Budget sidecar changed')
 else:
  DiskBudget(root,reserve_gb).ensure_space(len(raw)+2**20);directory.mkdir(parents=True)
  with (directory/'source.zip').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  _atomic_json(directory/'receipt.json',receipt)
 return receipt
def patch_module(module,proof):
 if module.__name__=='scripts.run_research_stage':raise ValueError('Only private modules may be patched')
 native_parse=module.parse_args;native_initial=module.validate_initial_checkpoint;native_resume=module.validate_resume_config;native_binding=module.validate_model_binding;NativeModel=module.ResearchModel
 class BudgetModel(NativeModel):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);self.tokenizer=fixed_clock(self.tokenizer)
  def load_state_dict(self,state,*args,**kwargs):
   required=getattr(self,'_budget18_initial_required',None)
   if hasattr(self,'_budget18_initial_required'):del self._budget18_initial_required
   require_learned_state(self,state,required);return super().load_state_dict(state,*args,**kwargs)
 module.ResearchModel=BudgetModel
 def parse(argv=None):
  args=configure(native_parse(argv),proof);sidecar(args.output,args.reserve_gb);return args
 def initial(payload,model,args):
  require_complete_P(payload['model'],model.projector);origin=initial_origin(args)
  previous=payload.get('config',{})
  if payload.get('step')!={'A0':64,'A1':4}[origin['stage']] or previous!=read(Path(origin['run'])/'resolved_config.json'):raise ValueError('Initial payload differs from completed source configuration/step')
  required=set(PKEYS)
  if origin['stage']=='A0':
   if set(payload['model'])!=required:raise ValueError('A0 initialization must be projector-only, never hidden old SFT weights')
  else:required|={n for n,p in model.named_parameters() if any(part in n for part in ('visual_experts','router_adapter','router_new','router_mass','modality_gate_logits'))}
  require_learned_state(model,payload['model'],required);result=native_initial(payload,model,args);model._budget18_initial_required=required;return result
 def resume(saved,current):validate_execution(saved);validate_execution(current);return native_resume(saved,current)
 def model_binding(payload,model):require_complete_P(payload['model'],model.projector);return native_binding(payload,model)
 def final(payload):
  config=payload['config'];validate_execution(config);require_complete_P(payload['model'])
  if payload['step']!=config['budget18_expected_steps']:raise ValueError('Budget checkpoint must be the single final epoch object')
 module.parse_args=parse;module.validate_initial_checkpoint=initial;module.validate_resume_config=resume;module.validate_model_binding=model_binding
 from src.utils.research_budget_storage_v1 import install_manager
 install_manager(module,validate_final=final);return module
@contextmanager
def patched_stock_runner(proof):
 from src.utils import research_stock_determinism as stock
 original=stock.runner
 def factory():return patch_module(original(),proof)
 stock.runner=factory
 try:yield
 finally:stock.runner=original
def main(argv=None):
 parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False);parser.add_argument('--allow-gpu',action='store_true');parser.add_argument('--metadata',action='store_true');parser.add_argument('--preflight',action='store_true');parser.add_argument('--data-proof',type=Path);parser.add_argument('--export-root',type=Path);opts,rest=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
 from src.utils import research_stock_determinism as stock
 if opts.metadata or not(opts.allow_gpu or opts.preflight):
  print(json.dumps({'status':'metadata_only_no_model_GPU','budget_execution':definition(),'numerical_profile':stock.definition()},indent=2));return 0
 if opts.data_proof is None:raise ValueError('Actual/preflight budget invocation requires explicit --data-proof')
 if '--allow-legacy-a0-initialization' in rest:raise ValueError('No generic legacy initialization; the exact freshstockA0 exception is explicit in this wrapper')
 data_proof(opts.data_proof)
 if opts.preflight:
  _,forwarded=stock._options(rest,legacy_initialization=True);module=stock.runner();args=configure(module.parse_args(forwarded),opts.data_proof);print(json.dumps({'status':'passed_CPU_data_budget_preflight_no_model','config':vars(args)},indent=2));return vars(args)
 public=argparse.ArgumentParser(add_help=False);public.add_argument('--vision',default=SIG);vision,_=public.parse_known_args(rest)
 with patched_stock_runner(opts.data_proof):
  if vision.vision==QWEN:
   from src.utils.research_derived_stock_v1 import training_main
   extra=['--export-root',str(opts.export_root)] if opts.export_root is not None else []
   return training_main(['--allow-gpu',*extra,*rest])
  return stock.train_main(['--allow-gpu',*rest])
