"""Opt-in stock numeric facade over the sealed gold-free text producer."""
from __future__ import annotations
import ast,copy,hashlib,json,os
from pathlib import Path
from types import ModuleType
ROOT=Path(__file__).resolve().parents[2]
from src.utils import research_stock_determinism as stock
OWNER='research_primary_text_stock_native_b1_v1'
PARENT_READY=ROOT/'configs/research/primary_text_producer_ready.json'
PARENT_READY_SHA='db81a8d15bbb17b9d9113b8c1746cd29ce466babe01db0ac48d99d0d37b337c1'
SOURCE='src/eval/research_text_generation.py'
SOURCE_SHA='b311a58969a28888169ed965c5d919b9dee2ddf4743787655e54af9d33182e0e'
NEW=('src/eval/research_text_stock_generation.py','scripts/eval_research_text_stock_checkpoint.py','tests/test_research_text_stock_generation.py')
READY=ROOT/'configs/research/primary_text_stock_producer_ready.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_parent():
 if sha(PARENT_READY)!=PARENT_READY_SHA or sha(ROOT/SOURCE)!=SOURCE_SHA:raise ValueError('Sealed text producer release changed')
 ready=json.loads(PARENT_READY.read_bytes())
 if ready.get('status')!='passed' or any(sha(ROOT/n)!=h for n,h in ready['source_sha256'].items()):raise ValueError('Sealed text producer/dependencies differ')
 return ready

def stock_scope(context):
 if context['model']['model_id']!='smallthinker4' or context['model']['repo']!=stock.SMALL:raise ValueError('Stock text execution is verified for pinned SmallThinker only')

def execution_profile():
 return {'owner':OWNER,'stock_numerical_definition':stock.definition(),'legacy_or_mixed_resume_allowed':False,
  'dtype_control':'explicit existing baseline precision or actual preserved learned-checkpoint dtypes',
  'custom_kernels':False,'fallback':False}

class _LazyInference:
 """Metadata paths never import the native Torch runner."""
 def __init__(self):self.value=None
 def get(self):
  if self.value is None:self.value=stock.inference()
  return self.value
 def __getattr__(self,name):return getattr(self.get(),name)


def producer():
 verify_parent();path=ROOT/SOURCE;tree=ast.parse(path.read_bytes());native=_LazyInference()
 class Redirect(ast.NodeTransformer):
  changes=0
  def visit_ImportFrom(self,n):
   if n.module=='src.eval.research_inference':
    self.changes+=1
    return [ast.Assign([ast.Name(a.asname or a.name,ast.Store())],ast.Attribute(ast.Name('_stock_native',ast.Load()),a.name,ast.Load())) for a in n.names]
   if n.module=='src.eval' and len(n.names)==1 and n.names[0].name=='research_inference':
    self.changes+=1;return ast.Assign([ast.Name(n.names[0].asname or 'research_inference',ast.Store())],ast.Call(ast.Attribute(ast.Name('_stock_native',ast.Load()),'get',ast.Load()),[],[]))
   return n
 redirect=Redirect();tree=redirect.visit(tree)
 if redirect.changes!=4:raise ValueError('Pinned text native-import hook structure changed')
 module=ModuleType('_private_stock_text_producer_v1');module.__file__=str(path);module.__package__='src.eval';module.__dict__['_stock_native']=native
 exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__);module.OWNER=OWNER
 old_sources=module.source_files
 def sources():return {**old_sources(),**stock.definition()['source_sha256'],**{n:sha(ROOT/n) for n in NEW}}
 module.source_files=sources
 def bind(plan):
  value=copy.deepcopy(plan);value['numerical_profile']['stock_execution']=execution_profile();return value
 old_baseline=module.baseline_plan
 def baseline(context,precision):stock_scope(context);return bind(old_baseline(context,precision))
 module.baseline_plan=baseline
 old_learned=module.learned_plan
 def learned(run,role,context):
  stock_scope(context);config=module.read(Path(run)/'resolved_config.json');stock.validate_config(config)
  return bind(old_learned(run,role,context))
 module.learned_plan=learned
 old_runtime=module.NativeTextRuntime
 class StockRuntime(old_runtime):
  def load_baseline(self,plan):
   if plan.get('numerical_profile',{}).get('stock_execution')!=execution_profile():raise ValueError('Explicit stock baseline plan required')
   stock.activate();model,loaded=super().load_baseline(plan);stock.assert_active()
   return model,dict(loaded,stock_execution=execution_profile())
  def load_learned(self,checkpoint,cache_dir,reserve_gb):
   stock.activate();model,loaded=super().load_learned(checkpoint,cache_dir,reserve_gb);stock.assert_active()
   if loaded.get('numerical_profile')!=stock.definition():raise ValueError('Loaded checkpoint lacks exact stock numerical profile')
   return model,dict(loaded,stock_execution=execution_profile())
  def generate(self,*a,**k):
   stock.assert_active();result=super().generate(*a,**k);stock.assert_active();return result
  def candidate_likelihoods(self,*a,**k):
   stock.assert_active();result=super().candidate_likelihoods(*a,**k);stock.assert_active();return result
 module.NativeTextRuntime=StockRuntime
 old_describe=module.describe_model
 def describe(*a,**k):
  stock.assert_active();value=old_describe(*a,**k)
  import torch
  value.update(stock_execution=execution_profile(),CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
   torch_threads=torch.get_num_threads(),deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
  return value
 module.describe_model=describe
 old_static=module.static_context
 def static(context,plan):
  stock_scope(context)
  if plan.get('numerical_profile',{}).get('stock_execution')!=execution_profile():raise ValueError('Missing/mixed stock text plan')
  return old_static(context,plan)
 module.static_context=static
 return module


def verify_release():
 module=producer();r=json.loads(READY.read_bytes())
 if r.get('owner')!=OWNER or r.get('status')!='cpu_tests_passed_actual_gpu_not_run' or r.get('source_sha256')!=module.source_files() or r.get('execution_profile')!=execution_profile():raise ValueError('Exact stock text facade CPU release required')
 return {'path':str(READY),'sha256':sha(READY)}
