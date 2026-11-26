#!/usr/bin/env python3
"""Isolated fixed3b BEST170 selection diagnostic; default is metadata only."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OWNER='smallthinker3b_selection_replay_diagnostic_v1'
RUN=ROOT/'runs/research/smallthinker4__siglip2__3b__SFT_pilot_scienceqa_context_v2_balanced_s42'
OUTPUT=ROOT/'research_results/selection_replay_3b_best170_diagnostic_v1'
READY=ROOT/'research_results/selection_replay_3b_ready.json'
CHECKPOINT_SHA256='496b6da2169896269bee4f2d129f7644baa28d8b0b60ac7db4fb2dccdcd0d1eb'
SELECTION_SHA256='325a06fa729330994e816e5692611b7c5f03c9463bcc1d8304a229a23e4a6b8a'
GIB=2**30
SOURCES=('scripts/probe_research_selection_replay.py','tests/test_research_selection_replay.py',
 'scripts/validate_research_checkpoint.py','scripts/run_research_stage.py',
 'src/eval/research_selection.py','src/eval/research_inference.py','src/eval/research_metrics.py',
 'src/model/research_moe.py','src/model/smallthinker.py','src/model/research_vision.py',
 'src/dataset/vision_cache.py','src/utils/research_checkpoint_compat.py',
 'src/utils/research_archive_gpu_parity.py','scripts/probe_research_parallel_b1.py',
 'src/utils/research_checkpoint_bundle.py','src/utils/research_checkpoint_io.py',
 'src/model/research_quantization.py','src/model/visual_expert_gates.py','src/model/text_moe_backbones.py',
 'src/model/gigachat_vl.py','scripts/extract_vision_features.py','third_party/vqa_normalization/tables.json','tests/test_research_inference.py')
FIELDS=('record_id','task_group','scorer','answers','source','choices','benchmark')
GENERATION_FIELDS=('prediction','generated_token_ids','generated_tokens','prompt_sha256','prefill_tokens','visual_tokens','logits_to_keep')

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def stat(p):
 s=Path(p).stat();return {k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
def sha(p,check=None):
 p=Path(p);before=stat(p);h=hashlib.sha256()
 with p.open('rb') as f:
  while chunk:=f.read(8*2**20):
   if check:check()
   h.update(chunk)
 if stat(p)!=before:raise ValueError('Diagnostic source changed while hashing')
 return h.hexdigest()
def read(p):
 p=Path(p)
 if p.is_symlink() or not p.is_file() or p.stat().st_size>16*2**20:raise ValueError('Bounded independent diagnostic metadata required')
 return json.loads(p.read_bytes())
def rows(p):
 p=Path(p)
 if p.is_symlink() or p.stat().st_size>16*2**20:raise ValueError('Bounded independent raw selection required')
 with p.open() as handle:return [json.loads(line) for line in handle if line.strip()]
def write(p,v):
 p=Path(p);raw=canonical(v)+b'\n'
 if len(raw)>16*2**20 or p.is_symlink():raise ValueError('Bounded own diagnostic output required')
 t=p.with_name('.'+p.name+'.tmp')
 try:
  with t.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.replace(t,p)
  descriptor=os.open(p.parent,os.O_DIRECTORY)
  try:os.fsync(descriptor)
  finally:os.close(descriptor)
 finally:t.unlink(missing_ok=True)
def source_binding():return {p:sha(ROOT/p) for p in SOURCES}
def verify_release():
 r=read(READY)
 if r.get('owner')!=OWNER or r.get('status')!='cpu_tests_passed_actual_gpu_not_run' or r.get('source_sha256')!=source_binding():
  raise ValueError('Reviewed exact diagnostic release required')
 return {'path':str(READY),'sha256':sha(READY)}
def file_binding(p):
 p=Path(p).resolve(strict=True);return {'path':str(p),'stat':stat(p),'sha256':sha(p)}

def prepare_plan(run=RUN):
 run=Path(run).resolve(strict=True)
 if run!=RUN.resolve():raise ValueError('Only explicit fixed3b SFT source is permitted')
 from scripts.validate_research_checkpoint import managed_checkpoints,check_manifest_hashes
 config=read(run/'resolved_config.json')
 if (config.get('recipe')!='3b' or config.get('stage')!='SFT' or config.get('seed')!=42 or
  config.get('selection_metric')!='group_macro' or config.get('selection_max_new_tokens')!=64):
  raise ValueError('Fixed3b selection profile changed')
 check_manifest_hashes(config)
 entries=managed_checkpoints(run);entry=next(e for e in entries if 'best' in e['labels'])
 if entry['marker'].get('step')!=170:raise ValueError('Diagnostic requires unchanged BEST170')
 checkpoint=entry['path']
 if checkpoint.is_symlink() or not checkpoint.is_file():raise ValueError('Physical original checkpoint required')
 files={}
 for name in ['resolved_config.json','trainable_parameters.json','completed.json','.checkpoint_manager.json',
  'checkpoint_roundtrip.json','checkpoint_compaction.json','provenance.json','provenance_latest.json','selection/best.metrics.json','selection/best.protocol.json','selection/best.predictions.jsonl']:
  files[name]=file_binding(run/name)
 files['checkpoint_marker']=file_binding(checkpoint.parent/'.research_checkpoint.json')
 files['active_controller_ready']=file_binding(ROOT/'configs/research/parallel_grounding_queue_ready.json')
 files['native_ordinary_protocol']=file_binding(run/'evaluation/heldout_v3/protocol.json')
 files['native_ordinary_summary']=file_binding(run/'evaluation/heldout_v3/summary.json')
 prefix=ROOT/'logs/research/queue'/run.name
 for key,suffix in [('training_log','.log'),('failed_validator_log','.checkpoint-validation.log')]:
  files[key]=file_binding(Path(str(prefix)+suffix))
 saved=rows(files['selection/best.predictions.jsonl']['path']);metrics=read(files['selection/best.metrics.json']['path'])
 protocol=read(files['selection/best.protocol.json']['path'])
 if (len(saved)!=116 or len({x['record_id'] for x in saved})!=116 or
  metrics.get('validation_step')!=170 or protocol.get('validation_step')!=170 or protocol.get('selection_manifest_sha256')!=SELECTION_SHA256 or
  metrics.get('group_macro_score')!=entry['marker']['metric']):raise ValueError('Saved BEST170 raw116/selection artifacts differ')
 if any(row.get('validation_step')!=170 for row in saved):raise ValueError('Saved raw rows are not BEST170')
 failed=read(files['checkpoint_roundtrip.json']['path'])
 if failed.get('status')!='failed' or 'group macro replay mismatch' not in failed.get('error',''):
  raise ValueError('Expected original failed replay evidence required')
 return {'schema_version':1,'owner':OWNER,'run':str(run),'config':config,'checkpoint':str(checkpoint),
  'checkpoint_stat':stat(checkpoint),'checkpoint_sha256':CHECKPOINT_SHA256,'marker':entry['marker'],
  'roles':{label:str((run/label/'state.pt').resolve(strict=True)) for label in ('best','last')},
  'files':files,'source_sha256':source_binding(),'saved_metric':metrics['group_macro_score'],
  'records':116,'expected_internal_profile':{'matmul_allow_tf32':True,'torch_threads':8,'batch_size':1,'max_new_tokens':64},
  'historical_runtime_environment_directly_recorded':False,'official_benchmark_score':False,'scientific_report_replacement':False}

def assert_plan(plan,check=None,full=False):
 if plan['source_sha256']!=source_binding():raise ValueError('Diagnostic/native source changed')
 if stat(plan['checkpoint'])!=plan['checkpoint_stat']:raise ValueError('Physical checkpoint stat changed')
 if {k:str((Path(plan['run'])/k/'state.pt').resolve(strict=True)) for k in ('best','last')}!=plan['roles']:
  raise ValueError('Managed roles changed')
 for item in plan['files'].values():
  if stat(item['path'])!=item['stat'] or sha(item['path'],check)!=item['sha256']:raise ValueError('Raw diagnostic evidence changed')
 if full and sha(plan['checkpoint'],check)!=plan['checkpoint_sha256']:raise ValueError('Physical checkpoint SHA differs from preserved native512')

def compare_rows(selection,saved,current):
 from src.eval.research_metrics import score_prediction,aggregate_scores
 records=selection['records']
 if len(saved)!=len(records) or len(current)!=len(records):raise ValueError('Incomplete diagnostic raw selection')
 comparisons=[]
 for record,old,new in zip(records,saved,current):
  for row in (old,new):
   if any(row.get(k)!=record.get(k) for k in FIELDS):raise ValueError('Raw selection identity/reference/source mismatch')
   g=row.get('generation',{})
   if (g.get('prediction')!=row.get('prediction') or not isinstance(g.get('generated_token_ids'),list) or
    any(type(v) is not int or v<0 for v in g['generated_token_ids']) or g.get('generated_tokens')!=len(g['generated_token_ids']) or
    not isinstance(g.get('prompt_sha256'),str) or len(g['prompt_sha256'])!=64 or not 0<=len(g['generated_token_ids'])<=64):
    raise ValueError('Invalid raw generation evidence')
   score=score_prediction(record,row['prediction'])
   if not math.isfinite(row.get('score',float('nan'))) or score!=row['score']:raise ValueError('Raw score differs from frozen scorer')
  changed=[k for k in GENERATION_FIELDS if old['generation'].get(k)!=new['generation'].get(k)]
  comparisons.append({'record_id':record['record_id'],'task_group':record['task_group'],'question':record['question'],
   'image_paths':copy.deepcopy(record.get('image_paths',[])),'image_hashes':copy.deepcopy(record.get('image_hashes',[])),
   'reference':copy.deepcopy(record['answers']),'changed_generation_fields':changed,'tokens_equal':old['generation']['generated_token_ids']==new['generation']['generated_token_ids'],
   'prompt_equal':old['generation']['prompt_sha256']==new['generation']['prompt_sha256'],
   'saved_score':old['score'],'replayed_score':new['score'],'score_delta':new['score']-old['score'],
   'saved_prediction':old['prediction'],'replayed_prediction':new['prediction']})
 return {'records':len(records),'mismatches':[r for r in comparisons if r['changed_generation_fields'] or r['score_delta']!=0],
  'all_rows':comparisons,'saved_metrics_recomputed':aggregate_scores(saved,require_four_groups=True),
  'replayed_metrics_recomputed':aggregate_scores(current,require_four_groups=True),
  'token_mismatch_records':sum(not r['tokens_equal'] for r in comparisons),'score_mismatch_records':sum(r['score_delta']!=0 for r in comparisons),
  'prompt_mismatch_records':sum(not r['prompt_equal'] for r in comparisons)}

def first_divergent_logit(saved_ids,generated):
 ids=generated.sequences[0].tolist();count=len(generated.scores);ids=ids[-count:] if count else []
 first=next((i for i,(a,b) in enumerate(zip(saved_ids,ids)) if a!=b),min(len(saved_ids),len(ids)) if len(saved_ids)!=len(ids) else None)
 if first is None:return None
 result={'first_divergent_generated_index':first,'saved_token_id':saved_ids[first] if first<len(saved_ids) else None,
  'current_token_id':ids[first] if first<len(ids) else None,'historical_logits_available':False}
 if first>=count:return dict(result,current_score_row_available=False,reason='current_generation_ended_before_divergent_step')
 # One current score row only; historical logits were never persisted.
 values=generated.scores[first][0].detach().float().cpu().tolist()
 def value(i):
  if i is None or not 0<=i<len(values):return None
  v=float(values[i]);return v if math.isfinite(v) else str(v)
 top=sorted(range(len(values)),key=lambda i:(-values[i],i))[:10]
 a=value(result['current_token_id']);b=value(result['saved_token_id'])
 return dict(result,current_score_row_available=True,current_vocab_size=len(values),
  current_top10=[{'token_id':i,'logit':value(i)} for i in top],current_chosen_logit=a,saved_chosen_current_logit=b,
  current_minus_saved_chosen_margin=a-b if isinstance(a,float) and isinstance(b,float) else None)

def tensor_digest(value):
 import torch
 flat=value.detach().reshape(-1).contiguous().view(torch.uint8);h=hashlib.sha256()
 for chunk in flat.split(8*2**20):h.update(chunk.cpu().numpy().tobytes())
 return h.hexdigest()

def embedding_alias(model):
 a=model.base.get_input_embeddings().weight;b=model.base.get_output_embeddings().weight
 same_storage=(a.device==b.device and a.untyped_storage().data_ptr()==b.untyped_storage().data_ptr() and a.storage_offset()==b.storage_offset() and a.stride()==b.stride())
 result={'same_parameter_object':a is b,'same_storage_layout':same_storage,'input_dtype':str(a.dtype),'output_dtype':str(b.dtype),
  'input_shape':list(a.shape),'output_shape':list(b.shape)}
 if not result['same_parameter_object'] or not same_storage:raise ValueError('Pinned SmallThinker tied input/output embedding alias not restored')
 return result

def model_runtime_census(model):
 result={'modules':{},'parameters':{},'buffers':{}}
 roots=[('model',model)];extractor=getattr(model,'extractor',None)
 if extractor is not None:
  roots.extend(('extractor.'+k,v) for k in ('vision_tower','vision_projector') if (v:=getattr(extractor,k,None)) is not None)
 for prefix,root in roots:
  for name,m in root.named_modules():
   cache=getattr(m,'_frozen_grouped_cache',None)
   result['modules'][prefix+'.'+name]={'class':type(m).__module__+'.'+type(m).__name__,'training':m.training,
    **{k:getattr(m,k) for k in ('gradient_checkpointing','warmup_force_new','use_grouped_mm') if isinstance(getattr(m,k,None),(bool,int,float,str))},
    'frozen_grouped_cache_present':cache is not None,'frozen_grouped_cache_type':type(cache).__name__,
    'frozen_grouped_cache_keys':sorted(str(k) for k in cache) if isinstance(cache,dict) else None,
    'frozen_grouped_cache_tensor_metadata':[{'slot':i,'dtype':str(v.dtype),'shape':list(v.shape),'device':str(v.device),'bytes':v.numel()*v.element_size()} for i,v in enumerate(cache or ()) if hasattr(v,'detach')] if isinstance(cache,(tuple,list)) else [],
    'rope_attributes':{k:{'dtype':str(v.dtype),'shape':list(v.shape),'sha256':tensor_digest(v)} for k in ('inv_freq','original_inv_freq') if (v:=getattr(m,k,None)) is not None and hasattr(v,'detach')}}
  for kind,iterator in (('parameters',root.named_parameters(remove_duplicate=False)),('buffers',root.named_buffers(remove_duplicate=False))):
   for name,v in iterator:result[kind][prefix+'.'+name]={'dtype':str(v.dtype),'device':str(v.device),'shape':list(v.shape),'requires_grad':v.requires_grad,**({'sha256':tensor_digest(v)} if kind=='buffers' else {})}
 return result

class NativeRuntime:
 def load(self,plan,check):
  import torch
  from transformers import set_seed
  from scripts.run_research_stage import ResearchModel,validate_model_binding
  from scripts.validate_research_checkpoint import structural_checks,_tensor_equal_after_restore
  from src.utils.research_checkpoint_compat import load_research_payload
  set_seed(plan['config']['seed']);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
  payload=load_research_payload(Path(plan['checkpoint']),map_location='cpu',mmap=True)
  structural=structural_checks(payload,plan['config'],read(plan['files']['trainable_parameters.json']['path']))
  if payload['step']!=170 or payload.get('selection_protocol',{}).get('selection_manifest_sha256') is None:raise ValueError('Checkpoint step/protocol differs')
  args=argparse.Namespace(**dict(plan['config'],prompt_schema_version=plan['config'].get('prompt_schema_version',1)))
  check();model=ResearchModel(args);validate_model_binding(payload,model)
  restored=model.load_state_dict(payload['model'],strict=False)
  if restored.unexpected_keys:raise ValueError('Unexpected restored tensor keys')
  available=model.state_dict()
  for name,value in payload['model'].items():
   check();_tensor_equal_after_restore(available[name],value,name)
   if available[name].dtype!=value.dtype:raise ValueError('Saved/restored tensor dtype differs: '+name)
  if any(p.requires_grad for p in model.projector.parameters()):raise ValueError('Aligned projector unexpectedly trainable')
  self.projector={name:value.detach().cpu().clone() for name,value in model.projector.state_dict().items()}
  result={'structural':structural,'restored_tensor_values_exact_frozen_validator':True,'tied_embedding_alias':embedding_alias(model),'model_ref':model.model_ref,'vision_ref':model.vision_ref,
   'cache_fingerprint':model.cache.fingerprint,'saved_selection_protocol':payload['selection_protocol']}
  self.saved={r['record_id']:r for r in rows(plan['files']['selection/best.predictions.jsonl']['path'])};self.logit_evidence=[];self.prompts=[];self.selection_census={}
  result['runtime_census_before_selection']=model_runtime_census(model)
  del payload;gc.collect();return model,result
 def selection(self,plan):
  from scripts.validate_research_checkpoint import select_validation_records
  from scripts.run_research_stage import load_records
  return select_validation_records(plan['config'],load_records)
 def evaluate(self,model,selection,check,progress):
  from src.eval.research_selection import evaluate_internal_group_macro
  from src.eval.research_inference import generate_prediction
  def predict(m,r,**kwargs):
   check();v=self.observed_generate(m,r,generate_prediction,phase='full116',**kwargs);check();return v
  return evaluate_internal_group_macro(model,selection,max_new_tokens=64,generator=predict,progress=progress)
 def repeat(self,model,selection,mismatches,check):
  from src.eval.research_selection import selection_generation_mode
  from src.eval.research_inference import generate_prediction
  from src.eval.research_metrics import score_prediction
  ids={r['record_id'] for r in mismatches};result=[]
  with selection_generation_mode(model):
   for r in selection['records']:
    if r['record_id'] not in ids:continue
    public={k:r[k] for k in ('record_id','question','image_paths','image_hashes') if k in r}
    check();g=self.observed_generate(model,public,generate_prediction,phase='mismatch_repeat_'+str(getattr(self,'repeat_number',1)),max_new_tokens=64);check()
    result.append({'record_id':r['record_id'],'generation':g,'score':score_prediction(r,g['prediction'])})
  return result
 def observed_generate(self,model,record,predict,*,phase,**kwargs):
  base=model.base;original=base.generate;absent=object();old=base.__dict__.get('generate',absent)
  tokenizer=getattr(model,'tokenizer',None);template=None;encodes=[];restores=[]
  if tokenizer is not None:
   for name in ('apply_chat_template','encode'):
    native=getattr(tokenizer,name);previous=tokenizer.__dict__.get(name,absent)
    def hook(*a,_name=name,_native=native,**k):
     nonlocal template
     result=_native(*a,**k)
     if _name=='apply_chat_template':template=result
     else:encodes.append({'text':a[0] if a else k.get('text'),'token_ids':list(result)})
     return result
    setattr(tokenizer,name,hook);restores.append((name,previous))
  def observed(*a,**k):
   output=original(*a,**k)
   evidence=first_divergent_logit(self.saved[record['record_id']]['generation']['generated_token_ids'],output)
   if evidence is not None:self.logit_evidence.append(dict(evidence,record_id=record['record_id'],phase=phase))
   return output
  base.generate=observed
  try:
   result=predict(model,record,**kwargs)
   if tokenizer is not None:
    if not isinstance(template,str) or hashlib.sha256(template.encode()).hexdigest()!=result['prompt_sha256']:raise ValueError('Observed actual prompt differs from generation fingerprint')
    prompt={'record_id':record['record_id'],'phase':phase,'prompt':template,'prompt_sha256':result['prompt_sha256'],'native_encode_calls':encodes}
    self.prompts.append(prompt)
    if hasattr(self,'on_generated'):self.on_generated(record,result,prompt)
   if phase not in self.selection_census:self.selection_census[phase]=model_runtime_census(model)
   return result
  finally:
   for name,previous in restores:
    if previous is absent:tokenizer.__dict__.pop(name,None)
    else:setattr(tokenizer,name,previous)
   if old is absent:base.__dict__.pop('generate',None)
   else:base.generate=old
 def finish(self,model):
  from scripts.validate_research_checkpoint import _tensor_equal_after_restore
  for name,tensor in model.projector.state_dict().items():_tensor_equal_after_restore(tensor,self.projector[name],name)
  embedding_alias(model)

def runtime_environment():
 import torch
 from src.utils.research_archive_gpu_parity import math_context
 packages={}
 for name in ('torch','transformers','tokenizers','peft','bitsandbytes','huggingface-hub','numpy'):
  try:packages[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:packages[name]=None
 return {'python':sys.version,'packages':packages,'torch_cuda':torch.version.cuda,'context':math_context(),
  'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,
  'cudnn_deterministic':torch.backends.cudnn.deterministic,
  'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
  'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
  'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
  'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','TOKENIZERS_PARALLELISM','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')},
  'generation':'unchanged frozen native B1 greedy cached max_new_tokens64 under selection_generation_mode'}

def execute(plan,output,*,nonce,timeout=900,expected_parent=None,deadline=None,repeats=1,runtime=None,guard=None):
 injected=runtime is not None or guard is not None
 if type(timeout) is not int or not 120<=timeout<=900 or type(repeats) is not int or not 0<=repeats<=2:raise ValueError('Bounded diagnostic timeout/repeats required')
 output=Path(output).resolve()
 if output.exists() or output.is_symlink() or output.is_relative_to(Path(plan['run'])) or Path(plan['run']).is_relative_to(output):raise ValueError('New separate diagnostic namespace required')
 if not injected:release=verify_release()
 else:release={'CPU_fixture':True}
 from src.utils.research_archive_gpu_parity import ResourceGuard
 resource=guard or ResourceGuard(dict(plan,reserve_gb=120),nonce,timeout,expected_parent=expected_parent,coordinator_deadline=deadline)
 engine=runtime or NativeRuntime();output.mkdir(parents=True,exist_ok=False)
 report={'schema_version':1,'owner':OWNER,'status':'running','producer':'CPU_fixture' if injected else 'ROOT',
  'plan':plan,'release':release,'nonce':nonce,'started_monotonic':time.monotonic(),'source_authoritative':True,
  'scientific_report_replacement':False,'training_updates':0,'metric_tolerance_changed':False,'repeats_requested':repeats}
 write(output/'report.json',report)
 watchdog=None
 try:
  if not injected:watchdog=start_watchdog(output,resource.deadline)
  with resource:
   resource.phase='source_evidence';resource.check();assert_plan(plan,resource.fast_check,full=True)
   with zipfile.ZipFile(output/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for name,h in plan['source_sha256'].items():
     raw=(ROOT/name).read_bytes()
     if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Source archive changed')
     z.writestr(name,raw)
   report['source_archive']={'path':str(output/'source.zip'),'sha256':sha(output/'source.zip'),'members':list(plan['source_sha256'])}
   evidence=output/'original_evidence';evidence.mkdir();report['copied_evidence']={}
   if not injected:
    released=Path(release['path']).read_bytes()
    if hashlib.sha256(released).hexdigest()!=release['sha256']:raise ValueError('Diagnostic release changed')
    with (evidence/'diagnostic_ready.json').open('xb') as f:f.write(released);f.flush();os.fsync(f.fileno())
   for name,item in plan['files'].items():
    raw=Path(item['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Original raw evidence changed')
    target=evidence/(name.replace('/','__'))
    with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    report['copied_evidence'][name]={'path':str(target),'sha256':sha(target),'bytes':len(raw),'original':item}
   if not injected:
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('Fresh diagnostic CUDA context required')
    torch.cuda.set_per_process_memory_fraction(40*GIB/resource.initial['total_bytes'],device=0)
   resource.phase='restoring_original_best170';model,restored=engine.load(plan,resource.check);report['restoration']=restored
   if not injected:report['runtime_environment']=runtime_environment()
   selection=engine.selection(plan)
   if len(selection['records'])!=116:raise ValueError('Exact116 reconstructed rows required')
   write(output/'selection_records.json',selection)
   saved=rows(plan['files']['selection/best.predictions.jsonl']['path'])
   if selection['protocol']['selection_manifest_sha256']!=restored['saved_selection_protocol']['selection_manifest_sha256']:
    raise ValueError('Reconstructed payload selection digest differs')
   def progress(event):
    resource.check();write(output/'progress.json',dict(event,phase=resource.phase,elapsed_seconds=time.monotonic()-report['started_monotonic']))
   raw_dir=output/'generated_rows';raw_dir.mkdir();index_by_id={r['record_id']:i for i,r in enumerate(selection['records'])}
   def on_generated(record,g,prompt):
    write(raw_dir/(prompt['phase']+'_'+str(index_by_id[record['record_id']]).zfill(3)+'.json'),{'record_id':record['record_id'],'generation':g,'prompt':prompt})
   if not injected:engine.on_generated=on_generated
   resource.phase='full116_replay';current=engine.evaluate(model,selection,resource.check,progress)
   write(output/'replayed_selection.json',current)
   comparison=compare_rows(selection,saved,current['predictions'])
   if comparison['saved_metrics_recomputed']['group_macro_score']!=plan['saved_metric']:raise ValueError('Saved raw rows do not reproduce marker metric')
   if comparison['replayed_metrics_recomputed']['group_macro_score']!=current['metrics']['group_macro_score']:raise ValueError('Current raw rows do not reproduce current metric')
   write(output/'comparison.json',comparison)
   report['comparison']=comparison;report['repeated_mismatches']=[]
   for index in range(repeats):
    resource.phase='mismatch_repeat_'+str(index+1)
    engine.repeat_number=index+1
    repeated=engine.repeat(model,selection,comparison['mismatches'],resource.check)
    write(output/('mismatch_repeat_'+str(index+1)+'.json'),repeated);report['repeated_mismatches'].append(repeated)
   engine.finish(model)
   if not injected:
    report['current_logit_diagnostics']=engine.logit_evidence;write(output/'current_logit_diagnostics.json',engine.logit_evidence)
    write(output/'actual_prompts.json',engine.prompts);write(output/'inside_selection_runtime_census.json',engine.selection_census)
    report['runtime_census_after_selection']=model_runtime_census(model)
   del model;gc.collect()
   if not injected:
    import torch
    torch.cuda.empty_cache();torch.cuda.synchronize()
    report['peak_pytorch_allocated_bytes']=torch.cuda.max_memory_allocated()
   assert_plan(plan,resource.fast_check,full=True);resource.check()
   report.update(status='completed_diagnostic_only',actual116_completed=True,source_inputs_unchanged=True,
    resource_samples=resource.samples,finished_monotonic=time.monotonic(),scientific_validation_status='unresolved_diagnostic_not_validation')
   write(output/'report.json',report);return report
 except BaseException as e:
  report.update(status='failed_diagnostic',error_type=type(e).__name__,error=str(e),finished_monotonic=time.monotonic())
  write(output/'report.json',report);raise
 finally:
  if watchdog is not None:
   write(output/'deadline_lease.json',dict(read(output/'deadline_lease.json'),status='released'))
   try:watchdog.wait(timeout=3)
   except subprocess.TimeoutExpired:pass

def start_watchdog(output,deadline):
 from scripts import probe_research_parallel_b1 as proc
 own=proc.process_info(os.getpid())
 if own is None or time.monotonic()>=deadline-10:raise ValueError('Own live bounded deadline required')
 lease={'owner':OWNER,'status':'active','process':own,'deadline':deadline,'nonce':read(output/'report.json')['nonce']}
 write(output/'deadline_lease.json',lease)
 watcher=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--deadline-watchdog',str(output/'deadline_lease.json')],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 try:
  until=min(time.monotonic()+5,deadline-5)
  while time.monotonic()<until:
   if watcher.poll() is not None:raise RuntimeError('Own deadline watcher exited before readiness')
   ready=output/'deadline_watchdog_ready.json'
   if ready.is_file():
    r=read(ready);live=proc.process_info(watcher.pid)
    if r.get('owner')!=OWNER or r.get('nonce')!=lease['nonce'] or r.get('deadline')!=deadline or r.get('watched')!=own or not proc.same_process(r.get('watcher'),live) or live.get('ppid')!=own['pid']:raise ValueError('Own deadline watcher readiness identity differs')
    return watcher
   time.sleep(.02)
  raise TimeoutError('Own bounded deadline watcher did not become ready')
 except BaseException:
  write(output/'deadline_lease.json',dict(lease,status='released'))
  try:watcher.wait(timeout=3)
  except subprocess.TimeoutExpired:pass
  raise

def deadline_watchdog(path):
 from scripts import probe_research_parallel_b1 as proc
 lease=read(path);expected=lease['process'];nonce=lease['nonce'];deadline=lease['deadline']
 if lease.get('owner')!=OWNER or not isinstance(nonce,str) or len(nonce)!=64 or not time.monotonic()<deadline<=time.monotonic()+900:return 2
 if '--nonce' not in expected.get('argv',[]) or nonce not in expected['argv']:return 2
 if not proc.same_process(expected,proc.process_info(expected['pid'])):return 0
 own=proc.process_info(os.getpid())
 if own is None or own.get('ppid')!=expected['pid']:return 2
 write(Path(path).parent/'deadline_watchdog_ready.json',{'owner':OWNER,'nonce':nonce,'deadline':deadline,'watched':expected,'watcher':own})
 while time.monotonic()<deadline:
  current=read(path)
  if current.get('status')!='active' or current.get('nonce')!=nonce:return 0
  if not proc.same_process(expected,proc.process_info(expected['pid'])):return 0
  if time.monotonic()>=deadline-5:
   os.kill(expected['pid'],signal.SIGTERM);break
  time.sleep(.25)
 while time.monotonic()<deadline:
  current=read(path)
  if current.get('status')!='active' or current.get('nonce')!=nonce:return 0
  if not proc.same_process(expected,proc.process_info(expected['pid'])):return 0
  time.sleep(.1)
 current=read(path)
 if current.get('status')=='active' and current.get('nonce')==nonce and proc.same_process(expected,proc.process_info(expected['pid'])):os.kill(expected['pid'],signal.SIGKILL)
 return 0

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--allow-gpu',action='store_true');p.add_argument('--output',type=Path,default=OUTPUT)
 p.add_argument('--nonce');p.add_argument('--timeout-seconds',type=int,default=900);p.add_argument('--mismatch-repeats',type=int,default=1)
 p.add_argument('--parent-pid',type=int);p.add_argument('--parent-start',type=int);p.add_argument('--parent-command-sha256');p.add_argument('--deadline-monotonic',type=float)
 p.add_argument('--deadline-watchdog',type=Path);args=p.parse_args(argv)
 if args.deadline_watchdog:return deadline_watchdog(args.deadline_watchdog)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 plan=prepare_plan()
 if not args.allow_gpu:
  print(json.dumps({'status':'metadata_preflight_only','plan':plan,'GPU_launched':False},indent=2));return 0
 if any(v is None for v in (args.nonce,args.parent_pid,args.parent_start,args.parent_command_sha256,args.deadline_monotonic)):p.error('Actual diagnostic requires explicit ROOT nonce and parent/deadline identity')
 handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
 def interrupted(signum,_):raise InterruptedError('Own diagnostic interrupted '+str(signum))
 try:
  for s in handlers:signal.signal(s,interrupted)
  result=execute(plan,args.output,nonce=args.nonce,timeout=args.timeout_seconds,repeats=args.mismatch_repeats,
   expected_parent={'pid':args.parent_pid,'start_time':args.parent_start,'command_sha256':args.parent_command_sha256},deadline=args.deadline_monotonic)
 finally:
  for s,h in handlers.items():signal.signal(s,h)
 print(json.dumps({'status':result['status'],'report':str(args.output/'report.json'),'mismatches':len(result['comparison']['mismatches'])}));return 0

if __name__=='__main__':raise SystemExit(main())
