"""Preserve private generation dependencies when offsetting the SFT order.

Training/loss/architecture v4 stays frozen. An atomic model-only recovery file
is written after the final update, before any heldout generation.
"""
from __future__ import annotations
import ast
from contextlib import contextmanager
import inspect,json,os,time
from pathlib import Path
from src.utils import research_capacity_v4 as prior
core=prior.core;ROOT=core.ROOT
SOURCES=('src/utils/research_capacity_eval_v5.py','scripts/run_research_capacity_eval_v5.py',
         'scripts/recover_research_capacity_eval_v5.py')


def definition():
 return dict(name='capacity_private_eval_binding_v5',
  source_sha256={p:core.sha(ROOT/p) for p in SOURCES},
  training='unchanged v4; epoch offset preserved; original private imports restored',
  recovery='atomic model-only final-update weights before generation; evaluation retry never retrains')


def fixed_offset(module):
 """Recompile the pinned main with BOTH private-import and seed transforms."""
 from src.utils import research_stock_determinism as stock
 stock.definition()
 tree=ast.parse((ROOT/'scripts/run_research_stage.py').read_text())
 mains=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main']
 if len(mains)!=1:raise ValueError('Exactly one pinned native main required')
 keys=[k for k,v in module.__dict__.items() if k.startswith('_stock_dependency_') and hasattr(v,'evaluate_internal_group_macro')]
 if len(keys)!=1:raise ValueError('Exact private selection dependency required')
 selected=keys[0]
 class Offset(ast.NodeTransformer):
  count=0
  def visit_Call(self,node):
   if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
       and node.func.value.id=='random' and node.func.attr=='Random' and len(node.args)==1
       and ast.unparse(node.args[0])=='args.seed + epoch'):
    self.count+=1
    node.args[0]=ast.BinOp(node.args[0],ast.Add(),ast.Attribute(ast.Name('args',ast.Load()),'data_epoch_offset',ast.Load()))
   return self.generic_visit(node)
 change=Offset();main=change.visit(mains[0])
 if change.count!=1:raise ValueError('Exactly one epoch-order seed hook required')
 tree=ast.Module(body=[main],type_ignores=[])
 tree=stock._Imports({'src.eval.research_selection':selected}).visit(tree)
 if any(isinstance(n,ast.ImportFrom) and n.module=='src.eval.research_selection' for n in ast.walk(tree)):
  raise ValueError('Private selection import escaped binding')
 exec(compile(ast.fix_missing_locations(tree),str(ROOT/'scripts/run_research_stage.py'),'exec'),module.__dict__)
 return selected


def recovery_checkpoint(module,model,args):
 import torch
 from src.utils.research_storage import DiskBudget,_atomic_json
 frame=inspect.currentframe().f_back
 while frame is not None and frame.f_code is not module.main.__code__:frame=frame.f_back
 if frame is None or frame.f_locals.get('step')!=56 or frame.f_locals.get('seen')!=3529:
  raise ValueError('Recovery snapshot requires actual final native optimizer update56/3529')
 local=frame.f_locals
 run=Path(args.output);directory=run/'recovery';directory.mkdir(exist_ok=True)
 target=directory/'model.pt'
 if target.exists():raise ValueError('Use evaluation-only recovery for an existing trained model')
 state=module.trainable_state_dict(model)
 state.update({'projector.'+k:v.detach().cpu().clone() for k,v in model.projector.state_dict().items()})
 config=json.loads((run/'resolved_config.json').read_text())
 payload=dict(model=state,config=config,step=56,seen=3529,optimizer=None,resumable_optimizer=False,
  training_completed=True,manifest_sha256=core.sha(args.manifest),model_ref=model.model_ref,
  vision_ref=model.vision_ref,cache_fingerprint=model.cache.fingerprint,
  loss_objective=local['objective_profile'],active_microbatch_size=local['microbatch_size'],
  elapsed_seconds=local['history_elapsed']+time.monotonic()-local['start'],
  recovery_kind='trained_pending_evaluation_v5',selection_value=None)
 size=sum(t.numel()*t.element_size() for t in state.values())
 DiskBudget(run,32).ensure_space(size+64*2**20)
 temporary=directory/'model.pt.tmp'
 with temporary.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
 temporary.replace(target)
 _atomic_json(directory/'receipt.json',dict(status='trained_pending_evaluation',step=56,seen=3529,
  checkpoint_sha256=core.sha(target),source_binding=definition(),created_unix=time.time()))
 module.log(run/'metrics.jsonl',dict(event='trained_weights_safe_before_evaluation',step=56,checkpoint=str(target)))
 return target


def factory(args,branch):
 args.capacity_evaluation_v5=definition()
 native_offset=prior._offset_training_order
 prior._offset_training_order=fixed_offset
 try:module=ORIGINAL_FACTORY(args,branch)
 finally:prior._offset_training_order=native_offset
 selection=next(v for k,v in module.__dict__.items() if k.startswith('_stock_dependency_') and hasattr(v,'evaluate_internal_group_macro'))
 NativeModel=module.ResearchModel
 class Model(NativeModel):
  def __init__(self,current):
   super().__init__(current)
   if current.stage!='SFT':return
   records=module.prepare_group_macro_selection(current)
   record={k:v for k,v in records['records'][0].items() if k in ('record_id','question','image_paths','image_hashes')}
   # A tiny native prefill+decode and heldout CE probe before spending a pass.
   # The selection context restores RNG, requires_grad and training flags.
   with selection.selection_generation_mode(self):
    generated=selection.generate_prediction(self,record,max_new_tokens=2)
   ce=prior.multimodal_ce(self,records['records'][:1],selection,batch_size=1)
   from src.utils.research_storage import _atomic_json
   _atomic_json(Path(current.output)/'pretraining_evaluation_smoke.json',dict(status='passed',
    generated_tokens=len(generated['generated_token_ids']),heldout_ce=ce,
    weights='constructor only; smoke is not a trained quality score',binding=definition()))
 module.ResearchModel=Model
 native=selection.evaluate_internal_group_macro
 def evaluate(model,records_or_selection,**kwargs):
  recovery_checkpoint(module,model,args)
  return native(model,records_or_selection,**kwargs)
 selection.evaluate_internal_group_macro=evaluate
 return module

ORIGINAL_FACTORY=prior.runner


@contextmanager
def repaired_factory():
 native=prior.runner;prior.runner=factory
 try:yield
 finally:prior.runner=native


def main(argv=None):
 with repaired_factory():return prior.main(argv)
