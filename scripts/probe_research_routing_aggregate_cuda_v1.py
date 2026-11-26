#!/usr/bin/env python3
"""ROOT-only tiny CUDA parity probe; default mode opens no weights or GPU."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils import research_routing_training_v1 as release
from src.utils import research_stock_determinism as stock
stock.bootstrap_environment()  # Before any Torch import in the actual fresh CLI.
OWNER='tiny_stock_small_routing_cuda_probe_v1'
SOURCE='scripts/probe_research_routing_aggregate_cuda_v1.py'
TEST='tests/test_research_routing_aggregate_cuda_v1.py'
READY=ROOT/'configs/research/routing_aggregate_cuda_probe_v1_ready.json'
COLLECTOR_READY_SHA='150fa3abc9f4a2968e887e37dc8e6e4d54fa917e57d837b471327545f89aba10'
VARIANTS=('4q','4f')+tuple(str(f)+p for f in (1,2,3) for p in 'abc')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_binding():
 if sha(release.READY)!=COLLECTOR_READY_SHA:raise ValueError('Frozen collector CPU release changed')
 release.verify_release()
 return {**release.source_binding(),str(release.READY.relative_to(ROOT)):COLLECTOR_READY_SHA,SOURCE:sha(ROOT/SOURCE),TEST:sha(ROOT/TEST)}
def metadata():
 return {'owner':OWNER,'schema_version':1,'status':'metadata_only_GPU_not_used','source_sha256':source_binding(),
  'stock_numerical_profile':stock.definition(),'variants':list(VARIANTS),'dispatch_modes':['eager','grouped_requested'],
  'shape':{'batch':2,'sequence':17,'hidden':64,'intermediate':128,'old_experts':8,'added_experts':2,'top_k':4},
  'NF4_scope':'Actual bnb NF4 old FFNs plus PEFT Linear4bit r16/alpha32/dropout.05 for4q/1abc; dense routers kept frozen as native quantized runner.',
  'quantized_grouped_scope':'Native eager fallback is mandatory for PEFT/NF4; never labelled as grouped execution.',
  'checks':['native/observer outputs, input/all-parameter gradients, CPU/CUDA RNG and post-SGD tensor bits exact',
   'BF16 autocast; F32 full masters and adapter masters; actual int64/float64 diagnostic reductions',
   'checkpoint recomputation/internal evaluation excluded; synthetic failed-attempt rollback; owned counter save/restore'],
  'limits':{'Torch_allocator_bytes':2**30,'own_NVML_bytes':2*2**30,'default_seconds':60,'maximum_seconds':170},
  'actual_GPU_validated':False,'full_model_training_overhead_measured':False,'current_queue_adopted':False,'scientific_rows':0}
def verify_release():
 value=json.loads(READY.read_bytes())
 if value.get('owner')!=OWNER or value.get('status')!='CPU_tests_passed_GPU_not_run' or value.get('source_sha256')!=source_binding():raise ValueError('Exact tiny CUDA probe CPU release required')
 return {'path':str(READY),'sha256':sha(READY)}
def atomic(path,value):
 path=Path(path);temp=path.with_name('.'+path.name+'.tmp')
 with temp.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
def nvml_processes():
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
 found={}
 for line in raw.splitlines():
  if line.strip():pid,mib=line.split(',');found[int(pid.strip())]=int(mib.strip())*2**20
 return found

def tiny(recipe,*,device='cuda',grouped=False,recompute=False,frozen=False):
 """Independent native modules from seeded random tensors, never HF weights."""
 import torch
 from torch import nn
 from src.model.smallthinker import SmallThinkerConfig,SmallThinkerMoeBlock
 from src.model.research_moe import ResearchSmallThinkerMoE,configure_research_trainability
 cfg=SmallThinkerConfig(vocab_size=128,hidden_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
  moe_ffn_hidden_size=128,moe_num_primary_experts=8,moe_num_active_primary_experts=4,pad_token_id=0,bos_token_id=1,eos_token_id=2)
 quantized=recipe=='4q' or recipe.startswith('1')
 if quantized and device!='cuda':raise ValueError('Real NF4 probe needs CUDA; no dense substitute')
 class Tiny(nn.Module):
  def __init__(self):
   super().__init__();self.config=cfg;self.moe=SmallThinkerMoeBlock(cfg);self.dropout=nn.Dropout(.125)
  def forward(self,x):
   value=self.dropout(self.projector(x) if hasattr(self,'projector') else x)
   if recompute:
    from torch.utils.checkpoint import checkpoint
    return checkpoint(self.moe,value,value,use_reentrant=False)[0]
   return self.moe(value,value)[0]
 model=Tiny().to(device=device,dtype=torch.bfloat16 if quantized else torch.float32)
 if quantized:
  import bitsandbytes as bnb
  from peft import LoraConfig,get_peft_model
  for expert in model.moe.experts:
   for name in ('up','gate','down'):
    original=getattr(expert,name);linear=bnb.nn.Linear4bit(original.in_features,original.out_features,bias=False,
     compute_dtype=torch.bfloat16,compress_statistics=True,quant_type='nf4')
    linear.weight=bnb.nn.Params4bit(original.weight.detach().cpu(),requires_grad=False,compress_statistics=True,quant_type='nf4')
    setattr(expert,name,linear.to(device))
  model.requires_grad_(False);model.is_loaded_in_4bit=True
  model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',target_modules=['up','gate','down'])).get_base_model()
  from peft.tuners.lora.bnb import Linear4bit
  adapters=[module for module in model.modules() if isinstance(module,Linear4bit)]
  if len(adapters)!=24:raise ValueError('Actual PEFT bnb Linear4bit dispatch omitted a tiny expert projection')
  for module in adapters:
   if (module.base_layer.compute_dtype!=torch.bfloat16 or module.base_layer.weight.dtype!=torch.uint8 or
       module.base_layer.weight.quant_state is None or module.base_layer.weight.quant_type!='nf4'):
    raise ValueError('Actual NF4 quantized storage/compute contract differs')
   if module.r['default']!=16 or module.lora_alpha['default']!=32 or module.lora_dropout['default'].p!=.05:
    raise ValueError('Native PEFT adapter rank/alpha/dropout differs')
 if recipe[0] in '123':
  model.moe=ResearchSmallThinkerMoE(model.moe,int(recipe[0]),recipe[1],added_experts=2,router_width=32)
  configure_research_trainability(model,'SFT')
 for parameter in model.parameters():
  if parameter.requires_grad and parameter.is_floating_point():parameter.data=parameter.data.float()
 if frozen:
  model.requires_grad_(False);model.projector=nn.Linear(64,64,device=device,dtype=torch.float32)
 if any(p.is_floating_point() and p.requires_grad and p.dtype!=torch.float32 for p in model.parameters()):
  raise ValueError('Trainable master dtype must remain FP32')
 model.moe.use_grouped_mm=grouped
 return model

def batch(device):
 import torch
 R=release.routing.TaggedRow
 rows=[R(([1,2],[3,4,5],list(range(8)),[torch.zeros(4,3)]),{'record_id':'synthetic_mm','task_group':'vqa'}),
       R(([],[1,2,3],[4,5],[]),{'record_id':'synthetic_text','task_group':'raw_text'})]
 visual=torch.zeros((2,17),dtype=torch.bool,device=device);visual[0,2:6]=True
 valid=torch.zeros_like(visual);valid[0,:]=True;valid[1,:5]=True
 return rows,visual,valid

def rng():
 import torch
 return {'CPU':torch.get_rng_state().clone(),'CUDA':torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None}
def set_rng(value):
 import torch
 torch.set_rng_state(value['CPU'])
 if value['CUDA'] is not None:torch.cuda.set_rng_state(value['CUDA'])
def tensor_hash(tensor):
 import torch
 return hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def census(model):return {name:tensor_hash(p) for name,p in model.state_dict().items()}

def pair(recipe,*,device='cuda',grouped=False,recompute=False,frozen=False,synthetic_OOM=False):
 """Callable on CPU only for unquantized unit fixtures; actual CLI is CUDA only."""
 import torch
 from contextlib import nullcontext
 seed=3817+VARIANTS.index(recipe);torch.manual_seed(seed)
 original=tiny(recipe,device=device,grouped=grouped,recompute=recompute,frozen=frozen)
 torch.manual_seed(seed);observed=tiny(recipe,device=device,grouped=grouped,recompute=recompute,frozen=frozen)
 if census(original)!=census(observed):raise ValueError('Independent tiny models initial tensor bits differ')
 collector=release.routing.RoutingAggregate(observed,['vqa','raw_text'])
 rows,visual,valid=batch(device)
 for model in (original,observed):
  if hasattr(model.moe,'set_modality_mask'):model.moe.set_modality_mask(visual,valid)
 generator=torch.Generator().manual_seed(seed+9);values=torch.randn((2,17,64),generator=generator).to(device)
 loss_weight=torch.randn((2,17,64),generator=generator).to(device)
 before=rng();calls={'original':0,'observed':0};active='original';native_grouped=getattr(torch,'_grouped_mm',None)
 def grouped_mm(*a,**k):calls[active]+=1;return native_grouped(*a,**k)
 if device=='cuda':torch._grouped_mm=grouped_mm
 def forward(model,scope=None):
  x=values.detach().clone().requires_grad_();model.train()
  with (scope if scope is not None else nullcontext()),torch.autocast(device_type=device,dtype=torch.bfloat16):output=model(x)
  (output*loss_weight).sum().backward()
  tensors={'output':output,'input_gradient':x.grad};tensors.update({'gradient:'+n:p.grad for n,p in model.named_parameters() if p.grad is not None})
  if any(not torch.isfinite(v).all().item() for v in tensors.values()):raise ValueError('Tiny outputs/gradients nonfinite')
  return {key:tensor_hash(value) for key,value in tensors.items()},rng(),{n:p.grad is None for n,p in model.named_parameters()}
 started=time.monotonic()
 try:
  set_rng(before);expected,expected_rng,expected_none=forward(original)
  active='observed';set_rng(before);collector.begin_window([rows]);collector.begin_attempt()
  if synthetic_OOM:
   try:
    with collector.forward_scope(rows),torch.autocast(device_type=device,dtype=torch.bfloat16):observed(values)
    raise torch.cuda.OutOfMemoryError('SYNTHETIC after routing, not an allocator capacity experiment')
   except torch.cuda.OutOfMemoryError:
    collector.abort_attempt();observed.zero_grad(set_to_none=True);set_rng(before);collector.begin_attempt()
  actual,actual_rng,actual_none=forward(observed,collector.forward_scope(rows));collector.finish_attempt()
  if expected!=actual or expected_none!=actual_none:raise ValueError('Native/observer output, input or parameter-gradient bits differ')
  if any((left is None)!=(actual_rng[key] is None) or (left is not None and not torch.equal(left,actual_rng[key])) for key,left in expected_rng.items()):raise ValueError('Native/observer RNG differs')
  staged={k:v.clone() for k,v in collector.layers['moe']['stage'].items()}
  observed.eval()
  with torch.no_grad(),torch.autocast(device_type=device,dtype=torch.bfloat16):observed(values)
  if any(not torch.equal(v,collector.layers['moe']['stage'][k]) for k,v in staged.items()):raise ValueError('Internal evaluation contaminated pending aggregate')
  for model in (original,observed):
   parameters=[p for p in model.parameters() if p.requires_grad]
   if parameters:
    optimizer=torch.optim.SGD(parameters,lr=.001,foreach=False);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    optimizer.step();scheduler.step()
  if device=='cuda':torch.cuda.synchronize()
  collector.commit(1,2);state=collector.export_state(step=1,seen=2)
  counts=state['layers']['moe']['counters']
  if counts['tokens'].tolist()!=[[13,4],[5,0]] or not torch.equal(counts['dispatch_counts'],counts['norm_dispatch_counts']):raise ValueError('Actual GPU masks/norm coverage differs')
  expected_slots=80 if recipe.endswith('a') else 88
  if int(counts['dispatch_counts'].sum())!=expected_slots or state['discarded_attempts']!=int(synthetic_OOM):raise ValueError('Successful-window dispatch or rollback count differs')
  if census(original)!=census(observed):raise ValueError('Post-SGD native/observer state bits differ')
  collector.close();restored=release.routing.RoutingAggregate(observed,['vqa','raw_text'])
  try:
   restored.restore_state(state,step=1,seen=2)
   if restored.export_state(step=1,seen=2)['state_sha256']!=state['state_sha256']:raise ValueError('Counter export/restore differs')
  finally:restored.close()
  quantized=recipe=='4q' or recipe.startswith('1')
  if device=='cuda' and grouped and not quantized and min(calls.values())==0:raise ValueError('Requested dense grouped path did not exercise actual _grouped_mm')
  if device=='cuda' and grouped and quantized and any(calls.values()):raise ValueError('Quantized case unexpectedly bypassed native eager fallback')
  return {'variant':recipe,'status':'passed','requested_dispatch':'grouped' if grouped else 'eager','actual_grouped_mm_calls':calls,
   'actual_quantized_old_experts':quantized,'actual_NF4_PEFT_projection_count':24 if quantized else 0,'recomputation':recompute,'frozen_base_with_trainable_projector':frozen,'synthetic_OOM_rollback':synthetic_OOM,
   'output_all_gradient_hashes':actual,'all_named_parameter_grad_none':actual_none,'CPU_CUDA_RNG_exact':True,
   'post_optimizer_parameter_bits_exact':True,'valid_tokens':22,'actual_dispatch_slots':expected_slots,
   'aggregate_state_sha256':state['state_sha256'],'elapsed_seconds_tiny_pair':time.monotonic()-started}
 finally:
  collector.close()
  if device=='cuda':torch._grouped_mm=native_grouped

def run(args):
 if 'torch' in sys.modules:raise RuntimeError('Actual GPU probe requires a fresh process before Torch import')
 if args.output is None or args.output.exists():raise ValueError('Explicit new output directory required; no overwrite/resume')
 proof=verify_release()
 if nvml_processes():raise RuntimeError('Exclusive idle GPU required; no concurrent CUDA process')
 import torch
 stock.activate()
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('One visible CUDA GPU required')
 torch.cuda.init();torch.cuda.set_per_process_memory_fraction(2**30/torch.cuda.get_device_properties(0).total_memory)
 args.output.mkdir(parents=True);started=time.monotonic();result=metadata();result.update(status='running',GPU_used=True,release=proof,pid=os.getpid(),started_unix=time.time(),cases=[])
 def guard():
  active=nvml_processes()
  if set(active)-{os.getpid()}:raise RuntimeError('Foreign CUDA process appeared')
  if active.get(os.getpid(),0)>2*2**30 or torch.cuda.max_memory_reserved()>2**30:raise RuntimeError('Tiny process exceeded2GiB NVML/1GiB allocator budget')
  if time.monotonic()-started>args.max_seconds:raise TimeoutError('Bounded tiny probe elapsed budget reached')
  stock.assert_active()
  return {'NVML_bytes':active.get(os.getpid(),0),'Torch_peak_allocated':torch.cuda.max_memory_allocated(),'Torch_peak_reserved':torch.cuda.max_memory_reserved()}
 atomic(args.output/'report.json',result)
 try:
  for recipe in VARIANTS:
   for grouped in (False,True):
    guard();item=pair(recipe,grouped=grouped,recompute=grouped,synthetic_OOM=recipe=='3c' and grouped)
    item['memory']=guard();result['cases'].append(item);atomic(args.output/'report.json',result)
    print(json.dumps({k:item[k] for k in ('variant','requested_dispatch','status','elapsed_seconds_tiny_pair','actual_grouped_mm_calls')}),flush=True)
   torch.cuda.empty_cache()
  guard();item=pair('4f',grouped=True,recompute=True,frozen=True);item['memory']=guard();result['cases'].append(item)
  result.update(status='passed_tiny_GPU_parity_only',actual_GPU_validated=True,finished_unix=time.time(),elapsed_seconds=time.monotonic()-started,
   actual_cases=len(result['cases']),full_model_training_overhead_measured=False,current_queue_adopted=False,scientific_rows=0)
 except BaseException as error:
  result.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started);raise
 finally:atomic(args.output/'report.json',result)
 return 0

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--allow-gpu',action='store_true');parser.add_argument('--output',type=Path)
 parser.add_argument('--max-seconds',type=int,default=60);args=parser.parse_args(argv)
 if not 10<=args.max_seconds<=170:parser.error('--max-seconds must be10..170; ROOT external timeout180 required')
 if not args.allow_gpu:print(json.dumps(metadata(),indent=2));return 0
 return run(args)
if __name__=='__main__':raise SystemExit(main())
