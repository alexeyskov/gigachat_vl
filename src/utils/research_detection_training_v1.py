"""Additional final SFT with15% prompted detection, preserving the old matrix."""
from __future__ import annotations
import gc,hashlib,json,math,os,random,time,types
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'datasets/research/detection-v1';OLD=ROOT/'datasets/research/budget18-en-v1'
SOURCES=('src/utils/research_detection_training_v1.py','src/dataset/research_detection_v1.py',
 'src/eval/research_detection_v1.py','scripts/run_research_detection_mix_v1.py','scripts/validate_research_detection_mix_v1.py')
WEIGHTS={'detection':.15,'multimodal_replay':.70,'text':.15}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def definition():
 from src.utils import research_stock_determinism as stock
 backbone_sources=('src/model/research_giga_loading_v1.py','src/model/research_giga_loading_v2.py','src/model/research_giga_peft.py',
  'src/model/research_quantization.py','src/utils/research_optim.py','src/utils/research_resource_vision_v3.py',
  'src/utils/research_small_router_stability_v1.py','src/utils/research_small_budget_v3.py','src/eval/research_text_generation.py',
  'src/utils/research_budget_stream_save_v1.py','src/utils/research_lossless_delta.py',
  'src/dataset/vision_cache.py','research_results/model_inventory.json')
 return dict(name='mixed_final_SFT_detection_v1',seed=42,epochs=1,records=3529,steps=56,effective_batch=64,
  source_sha256={p:sha(ROOT/p) for p in SOURCES},release_sha256=sha(DATA/'release.json'),
  native_source_contract=stock.definition(),backbone_source_sha256={p:sha(ROOT/p) for p in backbone_sources},
  objective=dict(weights=WEIGHTS,normalization='separate whole-effective-batch supervised-token denominators; absent stream contributes zero without renormalization'),
  coordinate_system='xyxy_normalized_1000',vision_and_MLP_projector='frozen',
  initialization='best old SFT by original internal116; fresh optimizer, half original stage LR',
  checkpoints='one completed model-only last saved BEFORE evaluation; best/last share the same physical weights after scoring')

def load_rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def validate_release():
 release=json.loads((DATA/'release.json').read_text())
 if release.get('status')!='passed' or release['counts']!={'detection':450,'MM_replay':2550,'MM_total':3000,'text':529,'total':3529,'optimizer_updates':56,'validation_detection':64,'test_detection':64}:raise ValueError('Exact small mixed detection release required')
 for name,item in release['files'].items():
  if sha(DATA/name)!=item['sha256']:raise ValueError('Frozen detection manifest changed')
 if sha(OLD/'release.json')!=release['legacy_release_sha256']:raise ValueError('Previous dataset release changed')
 if sha(DATA/'images.json')!=release['images_sha256'] or sha(DATA/'audit.json')!=release['audit_sha256']:raise ValueError('Detection image/audit bindings changed')
 for path,digest in release['source_sha256'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Detection data producer source changed')
 return release


def backend(args):
 from src.utils import research_additive_runner_v1 as additive
 from src.utils import research_stock_determinism as stock
 stock.bootstrap_environment()
 if getattr(args,'additive_execution',None):
  branch=args.additive_execution['branch']
  if args.additive_execution!=additive.definition(branch):raise ValueError('Frozen additive architecture changed')
  module,inference,selection=additive.runner(branch,model_repo=args.model,vision_repo=args.vision)
 else:
  proxy=types.ModuleType('_detection_full_runner_proxy')
  inference=stock._clone('src/eval/research_inference.py',{'scripts.run_research_stage':proxy})
  selection=stock._clone('src/eval/research_selection.py',{'src.eval.research_inference':inference})
  dependencies={'src.eval.research_selection':selection}
  if args.model==additive.MODELS['giga31']:
   from src.model.research_giga_loading_v2 import ScopedPEFT,install_loader
   dependencies['peft']=ScopedPEFT()
  module=stock._clone('scripts/run_research_stage.py',dependencies)
  if args.model==additive.MODELS['giga31']:install_loader(module)
  if args.model==additive.MODELS['smallthinker4']:
   from src.utils.research_small_router_stability_v1 import private_smallthinker
   module.register_smallthinker=private_smallthinker().register_smallthinker
  if args.vision==additive.VISIONS['qwen25']:
   from src.utils.research_resource_vision_v3 import derived
   binding=derived.resolve_derived_vision(args.vision,preprocessing_profile='qwen_native_bounded_729')
   module=derived.install_private_runner_hooks(module,binding,model_repo=args.model)
  proxy.__dict__.update(module.__dict__)
 return module,inference,selection


def configure_args(job):
 from src.utils import research_stock_determinism as stock
 from src.utils import research_additive_runner_v1 as additive
 raw=stock._clone('scripts/run_research_stage.py')
 flags=['--model',job['model'],'--vision',job['vision'],'--stage','SFT','--recipe',job['recipe'],
  '--manifest',str(DATA/'sft_multimodal_train.jsonl'),'--validation',str(OLD/'sft_validation.jsonl'),
  '--text-manifest',str(DATA/'text_train.jsonl'),'--output',job['output'],'--initial-checkpoint',job['source_checkpoint'],
  '--epochs','1','--seed','42','--batch-size','64' if job['recipe']!='4f' else '32','--effective-batch-size','64',
  '--max-length','4096','--eval-every','10000','--reserve-gb','32','--text-fraction','.15','--text-loss-weight','.15',
  '--selection-metric','group_macro','--optimizer-backend','cpu_grad_bnb_paged_adamw8' if job['recipe']=='4f' else 'torch_adamw32']
 if job['vision']==additive.VISIONS['qwen25']:flags+=['--vision-preprocessing-profile','qwen_native_bounded_729']
 args=raw.parse_args(flags);raw.validate_args(args)
 if job.get('branch'):args.additive_execution=additive.definition(job['branch'])
 if job['vision']==additive.VISIONS['qwen25']:
  from src.utils.research_resource_vision_v3 import derived
  derived.configure_args(args,derived.resolve_derived_vision(args.vision,preprocessing_profile='qwen_native_bounded_729'))
 args.detection_execution=definition();args.detection_job_id=job['id']
 return args


def validate_initial(model,saved,*,detection_weights=False):
 import torch
 expected=model.state_dict();required={n for n,p in model.named_parameters() if p.requires_grad}|{'projector.'+k for k in model.projector.state_dict()}
 if required-set(saved['model']) or set(saved['model'])-set(expected):raise ValueError('Initialization learned tensor set differs')
 if saved.get('step')!=56 or saved.get('seen')!=3529:raise ValueError('Completed56/3529 predecessor required')
 for key in ('model','vision','recipe','seed','stage'):
  if saved.get('config',{}).get(key)!=getattr(model.args,key):raise ValueError('Source experiment differs: '+key)
 if not detection_weights and saved.get('config',{}).get('detection_execution'):raise ValueError('Do not silently repeat the detection continuation')
 if detection_weights and saved.get('config',{}).get('detection_execution')!=definition():raise ValueError('Detection recovery/source binding differs')
 for key,actual in [('model_ref',model.model_ref),('vision_ref',model.vision_ref),('cache_fingerprint',model.cache.fingerprint)]:
  if saved.get(key)!=actual:raise ValueError('Exact initialization reference/cache differs: '+key)
 for name,value in saved['model'].items():
  target=expected[name]
  if value.shape!=target.shape or value.dtype!=target.dtype:raise ValueError('Exact initialization shape/dtype differs: '+name)
  for block in value.reshape(-1).split(2**20):
   if not torch.isfinite(block).all():raise ValueError('Nonfinite initialization tensor')
 return True


def source_cache_advice(path):
 from scripts.research_archival_stock_queue_v5 import scan_anchor_users
 path=Path(path).resolve(strict=True);s=path.stat()
 try:scan_anchor_users({(s.st_dev,s.st_ino)})
 except RuntimeError:return dict(status='still_mapped_or_open_no_advice',path=str(path))
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
 finally:os.close(fd)
 return dict(status='clean_source_cache_advised_after_load',path=str(path),bytes=s.st_size,weights_changed=False)


def construct(args,checkpoint,*,detection_weights=False):
 import torch
 from src.utils import research_stock_determinism as stock
 from src.utils.research_small_budget_v3 import fixed_clock
 stock.activate();module,inference,selection=backend(args);module.set_seed(args.seed)
 model=module.ResearchModel(args);model.tokenizer=fixed_clock(model.tokenizer)
 if args.model=='ai-sage/GigaChat3.1-10B-A1.8B-bf16':
  from src.eval.research_giga_eot_v1 import install_generation_stop
  install_generation_stop(model)
 saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
 validate_initial(model,saved,detection_weights=detection_weights)
 result=model.load_state_dict(saved['model'],strict=False)
 if result.unexpected_keys:raise ValueError('Unexpected initialization keys')
 actual=model.state_dict();exact=0
 for name,value in saved['model'].items():
  target=actual[name]
  for a,b in zip(target.reshape(-1).split(2**20),value.reshape(-1).split(2**20),strict=True):
   if not torch.equal(a.detach().cpu(),b):raise ValueError('Initialization values not exactly restored: '+name)
  exact+=1
 info=dict(source_sha256=sha(checkpoint),learned_tensors_exactly_loaded=exact,
  model_ref=saved['model_ref'],vision_ref=saved['vision_ref'],cache_fingerprint=saved['cache_fingerprint'])
 del saved,actual,value,target,a,b;gc.collect();info['source_cache_advice']=source_cache_advice(checkpoint)
 info['model_source_cache_advice']=[source_cache_advice(p) for p in Path(model.model_path).glob('*.safetensors')]
 return model,module,inference,selection,info


def train_group(model,optimizer,prepared,kinds,microbatch):
 """Three-stream weighted CE, with whole-group OOM/RNG rollback."""
 import numpy as np
 import torch
 denominators={key:sum(len(row[2]) for row,kind in zip(prepared,kinds,strict=True) if kind==key) for key in WEIGHTS}
 python_rng=random.getstate();numpy_rng=np.random.get_state();cpu_rng=torch.get_rng_state()
 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
 while True:
  sums={key:0. for key in WEIGHTS}
  optimizer.zero_grad(set_to_none=True)
  loss=None
  try:
   for start in range(0,len(prepared),microbatch):
    batch=prepared[start:start+microbatch];flags=kinds[start:start+microbatch]
    for key,weight in WEIGHTS.items():
     selected=[row for row,kind in zip(batch,flags,strict=True) if kind==key]
     if not selected:continue
     loss,count,_=model.forward_loss(selected)
     if count!=sum(len(row[2]) for row in selected):raise ValueError('Loss denominator differs from full prepared targets')
     sums[key]+=float(loss.detach());(loss*(weight/denominators[key])).backward()
     del loss
   break
  except torch.cuda.OutOfMemoryError:
   loss=None;selected=None
   optimizer.zero_grad(set_to_none=True)
   if microbatch<=1:raise
   # Drop all live traceback frames/graphs before retrying the whole group.
   random.setstate(python_rng);np.random.set_state(numpy_rng);torch.set_rng_state(cpu_rng)
   if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
   microbatch=max(1,microbatch//2)
  torch.cuda.empty_cache()
 ces={key:(sums[key]/denominators[key] if denominators[key] else 0.) for key in WEIGHTS}
 return dict(loss=sum(WEIGHTS[k]*ces[k] for k in WEIGHTS),stream_ce=ces,supervised_tokens=denominators,microbatch=microbatch)


def evaluate_detection(model,inference,selection,records,*,max_tokens=256,progress=None):
 from src.eval.research_detection_v1 import score_record,aggregate
 scored=[]
 with selection.selection_generation_mode(model):
  for index,record in enumerate(records):
   public={k:record[k] for k in ('record_id','question','image_paths','image_hashes')}
   generated=inference.generate_prediction(model,public,max_new_tokens=max_tokens)
   value=score_record(record,generated['prediction']);value['generation']=generated;scored.append(value)
   if progress:progress(dict(event='detection_eval',completed=index+1,total=len(records)))
 return dict(metrics=aggregate(scored),predictions=scored)


def store_evaluation(run,name,result):
 from src.utils.research_storage import _atomic_json
 directory=Path(run)/'evaluation';directory.mkdir(exist_ok=True)
 path=directory/(name+'.predictions.jsonl');tmp=path.with_suffix('.jsonl.tmp')
 tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result['predictions']));tmp.replace(path)
 _atomic_json(directory/(name+'.metrics.json'),result['metrics'])


def save_final_weights(model,module,args,initial_info):
 import torch
 from src.utils.research_storage import BestLastCheckpointManager,DiskBudget,trainable_state_dict,_atomic_json
 from src.utils.research_lossless_delta import memory_guard
 from src.utils.research_budget_stream_save_v1 import _Writer
 run=Path(args.output);budget=DiskBudget(run,32)
 required_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)+sum(p.numel()*p.element_size() for p in model.projector.parameters())
 memory_guard(required_bytes+2*2**30)
 state=trainable_state_dict(model);state.update({'projector.'+k:v.detach().cpu().clone() for k,v in model.projector.state_dict().items()})
 payload=dict(model=state,config=vars(args),step=56,seen=3529,optimizer=None,resumable_optimizer=False,
  training_completed=True,checkpoint_kind='final_mixed_detection_weights_v1',evaluation_embedded=False,
  manifest_sha256=sha(DATA/'sft_multimodal_train.jsonl'),model_ref=model.model_ref,vision_ref=model.vision_ref,
  cache_fingerprint=model.cache.fingerprint,initialization=initial_info)
 manager=BestLastCheckpointManager(run,budget,mode='max')
 def writer(directory):
  with (directory/'state.pt').open('xb') as f:
   sink=_Writer(f,guard=lambda n:memory_guard(n),budget=budget);torch.save(payload,sink);sink.commit()
 manager.save(writer,step=56,metric=None,anticipated_bytes=required_bytes+64*2**20,
  metadata=dict(evaluation='pending',training_completed=True,optimizer_saved=False))
 del state,payload;gc.collect()
 _atomic_json(run/'training_completed.json',dict(steps=56,records=3529,OD_examples=450,objective=WEIGHTS,checkpoint_sha256=sha(run/'last/state.pt')))
 source_cache_advice(run/'last/state.pt')
 return manager


def finalize_evaluation(model,module,inference,selection,args):
 from src.utils.research_storage import BestLastCheckpointManager,DiskBudget,_atomic_json
 run=Path(args.output)
 def progress(row):_atomic_json(run/'evaluation_progress.json',row)
 detection=evaluate_detection(model,inference,selection,load_rows(DATA/'detection_validation.jsonl'),progress=progress)
 store_evaluation(run,'after_detection_validation',detection)
 original=module.prepare_group_macro_selection(args)
 legacy=selection.evaluate_internal_group_macro(model,original,max_new_tokens=64,progress=lambda r:progress({k:v for k,v in r.items() if k not in ('prediction','score')}))
 store_evaluation(run,'after_internal116',legacy)
 manager=BestLastCheckpointManager(run,DiskBudget(run,32),mode='max');source=(run/'last/state.pt').resolve(strict=True)
 report=manager.save(lambda directory:os.link(source,directory/'state.pt'),step=56,metric=detection['metrics']['primary_score'],anticipated_bytes=0,
  metadata=dict(evaluation='passed',primary_metric='positive_macro_prompt_F1_IoU50',
   detection_validation_sha256=sha(run/'evaluation/after_detection_validation.predictions.jsonl'),internal116_score=legacy['metrics']['group_macro_score']))
 _atomic_json(run/'completed.json',dict(steps=56,records=3529,detection_primary_score=detection['metrics']['primary_score'],
  internal116_score=legacy['metrics']['group_macro_score'],checkpoint_sha256=sha(run/'last/state.pt')))
 source_cache_advice(run/'last/state.pt')
 return detection,legacy


def train(job,*,evaluate_existing=False):
 import torch
 from src.utils.research_storage import DiskBudget,_atomic_json
 validate_release();args=configure_args(job);run=Path(args.output);run.mkdir(parents=True,exist_ok=True)
 if (run/'resolved_config.json').exists() and json.loads((run/'resolved_config.json').read_text())!=vars(args):raise ValueError('Detection run config changed')
 _atomic_json(run/'resolved_config.json',vars(args))
 def log(row):
  module.log(run/'metrics.jsonl',row)
 if evaluate_existing:
  model,module,inference,selection,info=construct(args,run/'last/state.pt',detection_weights=True)
  return finalize_evaluation(model,module,inference,selection,args)
 if (run/'last/state.pt').exists():raise ValueError('Trained last exists: evaluation-only recovery required')
 model,module,inference,selection,info=construct(args,args.initial_checkpoint)
 _atomic_json(run/'initialization.json',info)
 def progress(row):_atomic_json(run/'evaluation_progress.json',row)
 before=evaluate_detection(model,inference,selection,load_rows(DATA/'detection_validation.jsonl'),progress=progress)
 store_evaluation(run,'before_detection_validation',before)
 legacy_records=module.prepare_group_macro_selection(args)
 if len(legacy_records['records'])!=116:raise ValueError('Exact previous internal116 required')
 source_run=next(p for p in Path(args.initial_checkpoint).resolve().parents if (p/'resolved_config.json').is_file())
 before_rows=load_rows(source_run/'selection/last.predictions.jsonl')
 from src.eval.research_metrics import score_prediction,aggregate_scores
 from src.eval.research_giga_eot_v1 import terminated_prefix
 normalized=[]
 for record,row in zip(legacy_records['records'],before_rows,strict=True):
  if record['record_id']!=row['record_id']:raise ValueError('Source internal116 population differs')
  prediction=row['prediction']
  if args.model=='ai-sage/GigaChat3.1-10B-A1.8B-bf16':prediction=terminated_prefix(row['generation']['generated_token_ids'],model.tokenizer)['prediction']
  normalized.append(dict(row,prediction=prediction,score=score_prediction(record,prediction)))
 store_evaluation(run,'before_internal116',dict(metrics=dict(aggregate_scores(normalized),source='verified predecessor saved predictions; native Giga EOT prefix if applicable'),predictions=normalized))
 log(dict(event='before_detection_validation',**before['metrics']))
 parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
 groups={}
 for name,p in parameters:
  lr=5e-6 if args.recipe=='4f' else 5e-4 if 'gate_logits' in name else 5e-5
  decay=0. if p.ndim<2 or 'gate_logits' in name or name.endswith('.bias') or 'norm' in name else .01
  groups.setdefault((lr,decay),[]).append(p)
 from src.utils.research_optim import make_full_training_optimizer
 from src.utils.research_lossless_delta import memory_guard
 memory_guard((6*sum(p.numel() for _,p in parameters) if args.recipe=='4f' else 0)+2*2**30)
 optimizer=make_full_training_optimizer([dict(params=p,lr=lr,weight_decay=wd) for (lr,wd),p in groups.items()],backend=args.optimizer_backend)
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/max(1,int(56*.03)),1)*.5*(1+math.cos(math.pi*min(s,56)/56)))
 records=load_rows(DATA/'sft_multimodal_train.jsonl')+load_rows(DATA/'text_train.jsonl')
 random.Random(42).shuffle(records);seen=0;od_seen=0;micro=args.batch_size;start=time.monotonic()
 try:
  for step,offset in enumerate(range(0,len(records),64),1):
   DiskBudget(run,32).ensure_space(64*2**20)
   batch=records[offset:offset+64];kinds=['detection' if r.get('task_group')=='object_detection' else 'multimodal_replay' if r.get('image_paths') else 'text' for r in batch]
   model.train();model.extractor.vision_tower.eval();prepared=model.prepare(batch)
   t0=time.monotonic();result=train_group(model,optimizer,prepared,kinds,micro);micro=result['microbatch']
   if not math.isfinite(result['loss']):raise FloatingPointError('Nonfinite mixed detection objective')
   if hasattr(optimizer,'clip_grad_norm_'):optimizer.clip_grad_norm_(1.)
   else:torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.,error_if_nonfinite=True)
   optimizer.step();scheduler.step();torch.cuda.synchronize()
   seen+=len(batch);od_seen+=kinds.count('detection')
   log(dict(event='train',step=step,total_steps=56,records_seen=seen,detection_seen=od_seen,elapsed_seconds=time.monotonic()-start,
    step_seconds=time.monotonic()-t0,peak_vram_GiB=torch.cuda.max_memory_allocated()/2**30,**result))
  if step!=56 or seen!=3529 or od_seen!=450:raise ValueError('Complete fixed mixed final-stage budget required')
 finally:
  if hasattr(optimizer,'close'):optimizer.close()
  if hasattr(optimizer,'_release_paged_buffers'):optimizer._release_paged_buffers()
  optimizer.state.clear()
 model.zero_grad(set_to_none=True)
 del optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
 save_final_weights(model,module,args,info)
 log(dict(event='trained_weights_saved_before_evaluation',step=56,records=3529))
 return finalize_evaluation(model,module,inference,selection,args)
