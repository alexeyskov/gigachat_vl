"""Versioned1c QLoRA guards and v1 A0 bridge, CPU only."""
import copy,json,os,subprocess,sys
from types import SimpleNamespace
import pytest
from src.utils import research_lfm_budget_v1 as old,research_lfm_budget_v2 as v2


def args(stage='SFT',**kw):
 x=dict(model=old.MODEL,vision=old.VISION,vision_preprocessing_profile=None,recipe='1c',stage=stage,seed=42,epochs=1,effective_batch_size=64,eval_every=10000,checkpoint_format='legacy',alignment_checkpoint=None,max_steps=None,max_records=None,optimizer_backend='torch_adamw32',text_loss_weight=.15,max_length=4096,reserve_gb=120)
 x.update(kw);return SimpleNamespace(**x)


def test_new_profile_quantized_family1_gates_not_full3c_or_moment8():
 impl=v2.implementation();profile=impl.definition()
 assert profile['recipes']==['1c'] and profile['name']!=old.PROFILE and profile['optimizer_backend']=='torch_adamw32' and profile['full_optimizer_moment_bits']==32 and not profile['full_optimizer_is_not_FP32_AdamW']
 assert 'no conditional family3' in profile['router_family'];impl.validate_budget_args(args())
 with pytest.raises(ValueError):impl.validate_budget_args(args(optimizer_backend='cpu_grad_bnb_paged_adamw8'))
 for recipe in ('3c','1b','4f'):
  with pytest.raises(ValueError):impl.validate_budget_args(args(recipe=recipe))


def test_shared_completed_scientific_NF4_A0_only_not_full_or_technical():
 keys=('projector.0.weight','projector.0.bias','projector.2.weight','projector.2.bias');payload={'config':{'stage':'A0','recipe':'1b','budget_technical_smoke':False,'budget_context':{'total_optimizer_steps':64,'actual_epoch_records':4096}},'step':64,'seen':4096,'model':dict.fromkeys(keys)}
 assert v2.validate_shared_A0(payload,args('A1'),old)
 for change in ('full','technical','partial','step','seen','stage'):
  bad=copy.deepcopy(payload)
  if change=='full':bad['config']['recipe']='4f'
  elif change=='technical':bad['config']['budget_technical_smoke']=True
  elif change=='partial':bad['model'].pop(keys[0])
  elif change=='step':bad['step']=1
  elif change=='seen':bad['seen']=64
  else:bad['config']['stage']='SFT'
  with pytest.raises(ValueError):v2.validate_shared_A0(bad,args('A1'),old)
 with pytest.raises(ValueError):v2.validate_shared_A0(payload,args('SFT'),old)


def test_1c_initial_required_set_covers_experts_residual_router_gates():
 import torch
 from torch import nn
 class M(nn.Module):
  def __init__(self):
   super().__init__();self.visual_experts=nn.Linear(2,2);self.router_adapter=nn.Linear(2,2);self.modality_gate_logits=nn.Parameter(torch.ones(4,2));self.projector=nn.Linear(2,2);self.old=nn.Linear(2,2)
 m=M();names=v2.implementation().required_initial_names(m,{'stage':'A1','recipe':'1c'})
 assert any('modality_gate_logits' in n for n in names) and any('visual_experts' in n for n in names) and not any(n.startswith('old.') for n in names)

@pytest.mark.parametrize('mode',['run','eval','validate'])
def test_actual_metadata_CLI_no_Torch_or_model(mode):
 code="import runpy,sys;sys.argv=['x','--metadata'];runpy.run_path('scripts/"+mode+"_research_lfm_budget_v2.py',run_name='__main__');assert 'torch' not in sys.modules"
 p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
 assert json.loads(p.stdout)['numerical_profile']['recipes']==['1c']


def test_legacy_v1_resume_rejected_not_renamed():
 impl=v2.implementation();oldcfg={'model':old.MODEL,'numerical_profile':old.definition()}
 with pytest.raises(ValueError):impl.validate_config(oldcfg)
 assert old.definition()['recipes']==['1b','4f','3c']
