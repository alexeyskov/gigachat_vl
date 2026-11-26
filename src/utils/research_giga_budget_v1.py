"""Giga specialization of the new common reduced-budget workflow, no frozen edits."""
from __future__ import annotations
import ast,hashlib,json,os,sys,types,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
MODEL='ai-sage/GigaChat3.1-10B-A1.8B-bf16';REVISION='189fff27a1dee68473960c3d5bca53e0e07a3191'
PROFILE='budget_giga_native_stock_deterministic_v1';BASE='src/utils/research_lfm_budget_v1.py'
OWN=('src/utils/research_giga_budget_v1.py','src/model/research_giga_loading_v1.py','src/model/research_giga_peft.py','src/utils/research_budget_stream_save_v1.py','scripts/run_research_giga_budget_v1.py','scripts/eval_research_giga_budget_v1.py','scripts/validate_research_giga_budget_v1.py',BASE)
WEIGHT_BYTES={'model-00000-of-00005.safetensors':4986065392,'model-00001-of-00005.safetensors':4085021224,'model-00002-of-00005.safetensors':4085022040,'model-00003-of-00005.safetensors':4085022240,'model-00004-of-00005.safetensors':4085022240,'model-00005-of-00005.safetensors':1634008560}
_BASE_SHA='7f0958baf3ee4e36906e7c6500684797b3acb6a5c2c83d3674185e54ac331773';_MODULE=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _projector(model,state):
 from src.utils.research_lfm_checkpoint_requirements_v1 import validate_projector_tensors
 if getattr(model.base.config,'model_type',None)!='deepseek_v3':raise ValueError('Native Giga model required for full projector check')
 return validate_projector_tensors(state,feature_dim=model.projector[0].in_features,hidden_size=model.base.config.hidden_size)
def _learned(model,state):
 _projector(model,state)
 from src.eval.research_inference import _validate_learned_state
 return _validate_learned_state(model,state)

def advise_giga_sources(model):
 """Root-approved read-only HF source advice after GPU load, never state edits."""
 from scripts.research_archival_stock_queue_v5 import scan_anchor_users
 from scripts.probe_research_parallel_b1 import gpu_snapshot
 from src.utils.research_archive_gpu_parity import host_snapshot,conservative_host_headroom
 from src.utils.research_trained_delta_probe import stat
 if model.model_ref!={'repo':MODEL,'revision':REVISION}:raise ValueError('Advice only for exact pinned Giga source')
 snapshot=Path(model.model_path)
 if snapshot.resolve().name!=REVISION:raise ValueError('Pinned Giga snapshot required')
 gc.collect();paths={}
 for name,size in WEIGHT_BYTES.items():
  p=(snapshot/name).resolve(strict=True)
  if not p.is_file() or p.is_symlink() or p.stat().st_size!=size:raise ValueError('Giga source physical size changed')
  paths[p]=stat(p)
 if any(row['pid']!=os.getpid() for row in gpu_snapshot()['processes']):raise ValueError('Foreign GPU process during post-load source advice')
 ids={(s['st_dev'],s['st_ino']) for s in paths.values()};before=host_snapshot();scan_before=scan_anchor_users(ids)
 for p,s in paths.items():
  if stat(p)!=s:raise ValueError('Giga source changed before advice')
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
  finally:os.close(fd)
  if stat(p)!=s:raise ValueError('Giga source changed during advice')
 scan_after=scan_anchor_users(ids);after=host_snapshot();available=conservative_host_headroom(after)
 if available<32*2**30:raise MemoryError('Actual host headroom below32GiB after source advice')
 return {'scope':'only exact pinned original Giga safetensor physical files after load; allUIDfdmaps0','files':[{'path':str(p),'stat':s} for p,s in paths.items()],'scan_before':scan_before,'scan_after':scan_after,'host_before':before,'host_after':after,'actual_available_after_bytes':available,'assumed_reclaim_credit_bytes':0,'weights_or_config_or_pointers_changed':False}

class _Stock:
 def __getattr__(self,name):
  from src.utils import research_stock_determinism as stock
  return getattr(stock,name)
 def _clone(self,relative,dependencies=None,bind_eval_protocol=False):
  from src.utils import research_stock_determinism as stock
  if relative!='scripts/run_research_stage.py':return stock._clone(relative,dependencies,bind_eval_protocol)
  from src.model.research_giga_loading_v1 import ScopedPEFT,install_loader
  module=stock._clone(relative,{'peft':ScopedPEFT()});install_loader(module);Native=module.ResearchModel
  from src.utils.research_budget_stream_save_v1 import install_manager
  install_manager(module,guard=lambda extra:implementation().host_guard(extra))
  class GigaNative(Native):
   def __init__(self,args):
    from src.utils.research_lossless_delta import memory_guard
    memory_guard(2*sum(WEIGHT_BYTES.values())+2*2**30)
    super().__init__(args)
    self.budget_source_cache_advice=advise_giga_sources(self)
    original=getattr(self.llm,'_giga_budget_loading',None)
    self.model_loading_report={'profile':PROFILE,'native_loading':original,'ordinary_peft':getattr(self.llm,'_research_giga_scope',None),'packed_nf4':getattr(self,'packed_quantization_report',None),'source_cache_advice':self.budget_source_cache_advice,'actual_full_size_training_fit_proven':False}
  module.ResearchModel=GigaNative;return module

def implementation():
 global _MODULE
 if sha(ROOT/BASE)!=_BASE_SHA:raise ValueError('Common budget workflow changed after Giga specialization import')
 if _MODULE is not None:return _MODULE
 tree=ast.parse((ROOT/BASE).read_bytes());changed=0
 class Specialize(ast.NodeTransformer):
  def visit_ImportFrom(self,node):
   nonlocal changed
   if node.module=='src.utils' and len(node.names)==1 and node.names[0].name=='research_stock_determinism':
    changed+=1;return ast.Assign([ast.Name(node.names[0].asname or 'research_stock_determinism',ast.Store())],ast.Name('_giga_stock',ast.Load()))
   if node.module=='src.utils.research_lfm_checkpoint_requirements_v1':
    replacements={'validate_lfm_initialization_projector':'_giga_projector','validate_lfm_learned_state':'_giga_learned'}
    if all(n.name in replacements for n in node.names):return [ast.Assign([ast.Name(n.asname or n.name,ast.Store())],ast.Name(replacements[n.name],ast.Load())) for n in node.names]
   if node.module=='transformers' and len(node.names)==1 and node.names[0].name=='AutoTokenizer':
    return ast.Assign([ast.Name(node.names[0].asname or 'AutoTokenizer',ast.Store())],ast.Name('_giga_tokenizer',ast.Load()))
   return node
  def visit_Constant(self,node):
   replacements={'lfm2_moe':'deepseek_v3','budget_lfm_execution_v1':'budget_giga_execution_v1','actual_full_size_LFM_GPU_validated':'actual_full_size_Giga_GPU_validated'}
   return ast.copy_location(ast.Constant(replacements[node.value]),node) if isinstance(node.value,str) and node.value in replacements else node
 tree=Specialize().visit(tree)
 if changed<3:raise ValueError('Expected common budget stock import hooks changed')
 from src.model.research_giga_loading_v1 import tokenizer_proxy
 module=types.ModuleType('_giga_private_common_budget_v1');module.__file__=str(ROOT/BASE);module.__dict__.update(_giga_stock=_Stock(),_giga_projector=_projector,_giga_learned=_learned,_giga_tokenizer=tokenizer_proxy())
 exec(compile(ast.fix_missing_locations(tree),module.__file__,'exec'),module.__dict__)
 module.MODEL=MODEL;module.REVISION=REVISION;module.PROFILE=PROFILE;module.OWN=OWN
 original=module.definition
 def definition():
  if sha(ROOT/BASE)!=_BASE_SHA:raise ValueError('Common budget code changed')
  value=original();value.update(giga_loading_route='nativeDSV3 config int1→float1.0 in-memory; ordinary scopedPEFT+native packedNF4; unused auxiliary MTP layer26 only',post_load_source_cache_advice='exact pinned sourcefiles only; allUIDfdmaps0; actualhead/no credit',checkpoint_writer='unchanged native Torch ZIP; owned NEW staging-only64MiB fsync+readonly FADV; actual32GiB floor/no assumed credit',common_budget_source_sha256=_BASE_SHA)
  value['source_sha256'].update({'scripts/research_archival_stock_queue_v5.py':sha(ROOT/'scripts/research_archival_stock_queue_v5.py'),'scripts/probe_research_parallel_b1.py':sha(ROOT/'scripts/probe_research_parallel_b1.py')})
  return value
 module.definition=definition;_MODULE=module;return module

def bootstrap():return implementation().bootstrap()
def definition():return implementation().definition()
def train_main(argv=None):return implementation().train_main(argv)
def eval_main(argv=None):return implementation().eval_main(argv)
def validate_main(argv=None):return implementation().validate_main(argv)
def runner(*a,**k):return implementation().runner(*a,**k)
