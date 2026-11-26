"""Isolated QLoRA learned-gate recipe1c; v1 1b/4f runs remain authoritative."""
from __future__ import annotations
import ast,hashlib,sys,types,uuid
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];BASE='src/utils/research_lfm_budget_v1.py';BASE_SHA='7f0958baf3ee4e36906e7c6500684797b3acb6a5c2c83d3674185e54ac331773'
PROFILE='budget_lfm_qlora_learned_gates_v2'
OWN=('src/utils/research_lfm_budget_v2.py','scripts/run_research_lfm_budget_v2.py','scripts/eval_research_lfm_budget_v2.py','scripts/validate_research_lfm_budget_v2.py',BASE)
_MODULE=None

def build(*,previous_api=None,stock_proxy=None,backend_type='lfm2_moe',model=None,revision=None,profile=PROFILE,own=OWN,sidecar='budget_lfm_execution_v2',projector_validator=None,learned_validator=None,tokenizer_class=None):
 """Giga may supply its sealed v1 API/loader guards; no global module patch."""
 from src.utils import research_lfm_budget_v1 as previous
 previous=previous_api or previous
 if previous.sha(ROOT/BASE)!=BASE_SHA:raise ValueError('Frozen budget v1 source changed')
 tree=ast.parse((ROOT/BASE).read_bytes())
 class Version(ast.NodeTransformer):
  def visit_Constant(self,node):
   mapping={'3c':'1c','lfm2_moe':backend_type,'budget_lfm_execution_v1':sidecar}
   if isinstance(node.value,str) and node.value in mapping:return ast.copy_location(ast.Constant(mapping[node.value]),node)
   return node
  def visit_ImportFrom(self,node):
   if tokenizer_class is not None and node.module=='transformers' and len(node.names)==1 and node.names[0].name=='AutoTokenizer':return ast.Assign([ast.Name('AutoTokenizer',ast.Store())],ast.Name('_tokenizer_class',ast.Load()))
   if stock_proxy is not None and node.module=='src.utils' and len(node.names)==1 and node.names[0].name=='research_stock_determinism':return ast.Assign([ast.Name(node.names[0].asname or 'research_stock_determinism',ast.Store())],ast.Name('_stock_proxy',ast.Load()))
   if projector_validator is not None and node.module=='src.utils.research_lfm_checkpoint_requirements_v1':
    names={'validate_lfm_initialization_projector':'_projector_validator','validate_lfm_learned_state':'_learned_validator'}
    if all(n.name in names for n in node.names):return [ast.Assign([ast.Name(n.asname or n.name,ast.Store())],ast.Name(names[n.name],ast.Load())) for n in node.names]
   return node
 module=types.ModuleType('_budget_qlora_c_v2_'+uuid.uuid4().hex);module.__file__=str(ROOT/BASE);module.__dict__.update(_stock_proxy=stock_proxy,_projector_validator=projector_validator,_learned_validator=learned_validator,_tokenizer_class=tokenizer_class)
 exec(compile(ast.fix_missing_locations(Version().visit(tree)),module.__file__,'exec'),module.__dict__)
 module.MODEL=model or previous.MODEL;module.REVISION=revision or previous.REVISION;module.PROFILE=profile;module.OWN=own
 original_definition=module.definition
 def definition():
  result=original_definition();result.update(name=profile,recipes=['1c'],full_SFT_optimizer=None,optimizer_backend='torch_adamw32',full_optimizer_moment_bits=32,full_optimizer_is_not_FP32_AdamW=False,
   language_base='frozen NF4 with rank16 LoRA; new visual experts/router residual and static learned text/visual gates',
   router_family='family1 frozen original router + MLP residual; no conditional family3 router',
   common_v1_source_sha256=BASE_SHA,previous_v1_profile=previous.definition(),allowed_cross_profile_initialization='only completed scientific v1 recipe1b NF4 A0 P-only -> v2 recipe1c A1, never resume or legacy full3c')
  return result
 module.definition=definition
 native_budget=module.validate_budget_args
 def validate_budget(args,*,technical=False):
  if args.recipe!='1c' or args.stage not in ('A1','SFT'):raise ValueError('v2 is QLoRA1c A1/SFT only; shared NF4 A0 uses v1 recipe1b')
  surrogate=SimpleNamespace(**vars(args));surrogate.recipe='1b';native_budget(surrogate,technical=technical)
 module.validate_budget_args=validate_budget
 native_runner=module.runner
 def runner(*a,**kw):
  private=native_runner(*a,**kw);same_profile_initial=private.validate_initial_checkpoint
  def initial(payload,model,args):
   previous_config=payload.get('config',{})
   if previous_config.get('numerical_profile')==definition():return same_profile_initial(payload,model,args)
   previous.validate_config(previous_config);previous.validate_context(previous_config)
   validate_shared_A0(payload,args,previous)
   from src.utils import research_stock_determinism as stock
   raw=stock._clone('scripts/run_research_stage.py');private.validate_model_binding(payload,model);raw.validate_initial_checkpoint(payload,model,args)
   module.validate_complete_state(model,payload['model'],set(payload['model']));model._budget_initial_required=set(payload['model']);return True
  private.validate_initial_checkpoint=initial;return private
 module.runner=runner;return module

def validate_shared_A0(payload,args,previous):
 config=payload['config'];keys={'projector.0.weight','projector.0.bias','projector.2.weight','projector.2.bias'}
 if (args.stage!='A1' or args.recipe!='1c' or config.get('stage')!='A0' or config.get('recipe')!='1b' or
     config.get('budget_technical_smoke') or payload.get('step')!=64 or payload.get('seen')!=4096 or set(payload.get('model',{}))!=keys):raise ValueError('Only full scientific v1 NF4 A0 P-only64/4096 may initialize1cA1')
 if config['budget_context'].get('total_optimizer_steps')!=64 or config['budget_context'].get('actual_epoch_records')!=4096:raise ValueError('Shared A0 was truncated')
 return True

def implementation():
 global _MODULE
 if _MODULE is None:_MODULE=build()
 return _MODULE

def bootstrap():return implementation().bootstrap()
def definition():return implementation().definition()
def runner(*a,**kw):return implementation().runner(*a,**kw)
def train_main(argv=None):return implementation().train_main(argv)
def eval_main(argv=None):return implementation().eval_main(argv)
def validate_main(argv=None):return implementation().validate_main(argv)
