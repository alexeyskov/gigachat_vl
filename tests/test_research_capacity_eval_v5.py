import ast,json,types
from dataclasses import asdict
from types import SimpleNamespace
import pytest
import torch
from src.model.research_additive_visual_v1 import VisualBranchConfig
from src.utils import research_capacity_eval_v5 as repair


def code_names(code):
 names=set(code.co_names)
 for child in code.co_consts:
  if isinstance(child,types.CodeType):names|=code_names(child)
 return names


def test_fixed_main_keeps_private_selection_and_native_line_numbers():
 args=SimpleNamespace(model=repair.core.MODELS['smallthinker4'],vision=repair.core.VISIONS['siglip2'],
  training_pass=1,data_epoch_offset=0,lr_scale=1.)
 module=repair.factory(args,asdict(VisualBranchConfig(blocks=4)))
 keys={k for k,v in module.__dict__.items() if k.startswith('_stock_dependency_') and hasattr(v,'evaluate_internal_group_macro')}
 assert len(keys)==1
 assert keys<=code_names(module.main.__code__)
 assert 'data_epoch_offset' in code_names(module.main.__code__)
 native=next(n for n in ast.parse((repair.ROOT/'scripts/run_research_stage.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
 assert module.main.__code__.co_firstlineno==native.lineno
 assert not torch.cuda.is_initialized()


def test_four_block_prefill_after_training_batch_uses_private_modality_setter():
 args=SimpleNamespace(model=repair.core.MODELS['smallthinker4'],vision=repair.core.VISIONS['siglip2'],
  training_pass=1,data_epoch_offset=0,lr_scale=1.)
 module=repair.factory(args,asdict(VisualBranchConfig(blocks=4)))
 private=next(v for k,v in module.__dict__.items() if k.startswith('_stock_dependency_') and hasattr(v,'evaluate_internal_group_macro'))
 inference=private.generate_prediction.__wrapped__.__globals__
 from src.model import research_additive_visual_v1 as additive
 assert inference['set_modality_mask'] is additive.set_modality_mask
 # Also prove the nested main evaluate resolves to this same private module.
 key=next(k for k,v in module.__dict__.items() if v is private)
 assert key in code_names(module.main.__code__)


def test_snapshot_refuses_non_native_or_early_training_call(tmp_path):
 args=SimpleNamespace(output=str(tmp_path))
 with pytest.raises(ValueError,match='actual final'):
  repair.recovery_checkpoint(SimpleNamespace(main=lambda:None),None,args)
 assert not list(tmp_path.iterdir())


def test_plan_reuses_all_A1_and_changes_only_SFT_outputs():
 from scripts import research_capacity_queue_v5 as queue
 plan=queue.build_plan();old=queue.prior.build_plan()
 assert len(plan['jobs'])==37 and len(plan['arms'])==12 and plan['SFT_stages_total']==25
 for job in plan['jobs']:
  source=next(j for j in old['jobs'] if j['id']==job['id'])
  if job['stage']=='A1':assert job==source
  else:
   assert job['output'].startswith(str(queue.BASE))
   assert job['branch']==source['branch'] and job['records']==source['records'] and job['steps']==source['steps']
 assert next(j for j in plan['jobs'] if j['stage']=='SFT')['id']=='lfm25__siglip2__fixed4__R3__P1__SFT'


def test_final_weights_are_saved_before_evaluation_can_fail(tmp_path):
 import time
 from torch import nn
 manifest=tmp_path/'train.jsonl';manifest.write_text('{}\n')
 (tmp_path/'resolved_config.json').write_text(json.dumps(dict(stage='SFT',capacity_evaluation_v5=repair.definition())))
 model=nn.Module();model.projector=nn.Sequential(nn.Linear(2,2),nn.GELU(),nn.Linear(2,2))
 model.model_ref={'repo':'base'};model.vision_ref={'repo':'vision'};model.cache=SimpleNamespace(fingerprint='cache')
 module=SimpleNamespace(trainable_state_dict=lambda m:m.state_dict(),log=lambda *a:None)
 args=SimpleNamespace(output=str(tmp_path),manifest=str(manifest))
 def main():
  step=56;seen=3529;objective_profile={'name':'unchanged'};microbatch_size=64
  history_elapsed=0;start=time.monotonic()
  checkpoint=repair.recovery_checkpoint(module,model,args)
  raise RuntimeError('simulate later evaluation failure')
 module.main=main
 with pytest.raises(RuntimeError,match='later evaluation'):module.main()
 saved=torch.load(tmp_path/'recovery/model.pt',weights_only=False)
 assert saved['step']==56 and saved['seen']==3529 and saved['optimizer'] is None
 for name,tensor in model.state_dict().items():assert torch.equal(saved['model'][name],tensor)
 assert json.loads((tmp_path/'recovery/receipt.json').read_text())['status']=='trained_pending_evaluation'


def test_queue_retries_saved_final_weights_without_training(tmp_path,monkeypatch):
 from scripts import research_capacity_queue_v5 as queue
 monkeypatch.setattr(queue,'OUT',tmp_path)
 run=tmp_path/'run';(run/'recovery').mkdir(parents=True);(run/'recovery/model.pt').write_bytes(b'snapshot')
 owner=object.__new__(queue.Queue);owner.state={'jobs':{'arm':{'failures':[{},{}]}}}
 seen=[]
 monkeypatch.setattr(queue.native.Queue,'execute',lambda self,j,a,k:seen.append(a))
 job=dict(id='arm',output=str(run),stage='SFT')
 owner.execute(job,['old_training_script.py'],'train')
 assert seen==[['scripts/recover_research_capacity_eval_v5.py','--run',str(run)]]
 (run/'recovery/model.pt').unlink()
 with pytest.raises(ValueError,match='automatic full retraining blocked'):
  owner.execute(job,['old_training_script.py'],'train')
