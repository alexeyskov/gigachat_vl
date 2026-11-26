"""The1c addendum preserves native routing and explicit priorA0 lineage."""
import json,os,subprocess,sys
from types import SimpleNamespace
import pytest
import torch
from src.utils import research_giga_budget_v1 as previous
from src.utils import research_giga_budget_v2 as budget

def args(**overrides):
 values=dict(reserve_gb=120.,model=previous.MODEL,vision='google/siglip2-so400m-patch14-384',vision_preprocessing_profile=None,max_length=4096,recipe='1c',stage='A1',seed=42,epochs=1,effective_batch_size=64,eval_every=10000,checkpoint_format='legacy',alignment_checkpoint=None,max_steps=None,max_records=None,optimizer_backend='torch_adamw32',text_loss_weight=None)
 values.update(overrides);return SimpleNamespace(**values)
def test_only_QLoRA_1c_A1_SFT_torchAdam32_admitted():
 module=budget.implementation();module.validate_budget_args(args());module.validate_budget_args(args(stage='SFT',text_loss_weight=.15))
 for changes in ({'stage':'A0'},{'recipe':'3c'},{'recipe':'4f'},{'optimizer_backend':'cpu_grad_bnb_paged_adamw8'},{'reserve_gb':119}):
  with pytest.raises(ValueError):module.validate_budget_args(args(**changes))
 profile=budget.definition();assert profile['recipes']==['1c'] and profile['optimizer_backend']=='torch_adamw32' and profile['router_family'].startswith('family1') and not profile['full_optimizer_is_not_FP32_AdamW']
 assert profile['previous_v1_profile']==previous.definition() and profile['model']==previous.MODEL

def test_shared_A0_guard_rejects_full_old_otherrecipe_and_truncated_sources():
 from src.utils.research_lfm_budget_v2 import validate_shared_A0
 payload={'config':{'stage':'A0','recipe':'1b','budget_context':{'total_optimizer_steps':64,'actual_epoch_records':4096}},'step':64,'seen':4096,'model':{n:torch.ones(1) for n in ('projector.0.weight','projector.0.bias','projector.2.weight','projector.2.bias')}}
 assert validate_shared_A0(payload,args(),previous)
 for field,value in [('stage','A1'),('recipe','3c'),('budget_technical_smoke',True)]:
  bad=dict(payload,config=dict(payload['config'],**{field:value}))
  with pytest.raises(ValueError):validate_shared_A0(bad,args(),previous)
 for changes in ({'step':1},{'seen':64},{'model':{'projector.0.weight':torch.ones(1)}}):
  with pytest.raises(ValueError):validate_shared_A0(dict(payload,**changes),args(),previous)

def test_actual_Giga_context_tokenizer_proxy_and_fixed_clock_are_inherited():
 import inspect
 module=budget.implementation();assert module.context_preflight.__globals__['_tokenizer_class']
 tokenizer=module.context_preflight.__globals__['_tokenizer_class'].from_pretrained('/workspace/.hf_home/hub/models--ai-sage--GigaChat3.1-10B-A1.8B-bf16/snapshots/'+previous.REVISION)
 wrapped=module.fixed_clock(tokenizer);prompt=wrapped.apply_chat_template([{'role':'user','content':'Hello.'}],tokenize=False,add_generation_prompt=True)
 assert prompt and '2026' in module.definition()['fixed_native_template_date_UTC']

def test_tiny_native_Giga_family1c_gate_and_router_gradients_frozen_original():
 from transformers import AutoModelForCausalLM,DeepseekV3Config
 from src.model.research_moe import install_research_moe,configure_research_trainability,set_modality_mask
 cfg=DeepseekV3Config(vocab_size=32,hidden_size=8,intermediate_size=12,moe_intermediate_size=6,num_hidden_layers=2,first_k_dense_replace=1,num_attention_heads=2,num_key_value_heads=2,n_routed_experts=4,n_shared_experts=1,num_experts_per_tok=2,n_group=1,topk_group=1,routed_scaling_factor=1.,q_lora_rank=None,kv_lora_rank=4,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4,pad_token_id=0)
 model=AutoModelForCausalLM.from_config(cfg,attn_implementation='eager',experts_implementation='eager').float()
 with torch.no_grad():
  for n,b in model.named_buffers():
   if 'e_score_correction_bias' in n:b.zero_()
 original=list(model.parameters());model.requires_grad_(False);wrappers=install_research_moe(model,1,'c',added_experts=2);configure_research_trainability(model,stage='A1');tokens=torch.tensor([[1,4,7,2]]);mask=torch.tensor([[False,True,False,False]]);set_modality_mask(model,mask,torch.ones_like(mask));loss=model(tokens,labels=tokens,use_cache=False).loss;loss.backward()
 wrapper=wrappers[0];assert torch.isfinite(loss) and wrapper.family==1 and wrapper.policy=='c'
 assert all(p.grad is None and not p.requires_grad for p in original)
 assert any(p.grad is not None for n,p in wrapper.named_parameters() if 'modality_gate_logits' in n)
 assert any(p.grad is not None for n,p in wrapper.named_parameters() if 'router_adapter' in n)

def test_three_metadata_entrypoints_are_weight_free_noTorch_and_distinct_profile():
 for action in ('run','eval','validate'):
  code=f"import runpy,sys;sys.argv=['x','--metadata'];runpy.run_path('scripts/{action}_research_giga_budget_v2.py',run_name='__main__');assert 'torch' not in sys.modules"
  proc=subprocess.run([sys.executable,'-c',code],cwd=budget.ROOT,capture_output=True,text=True);assert proc.returncode==0,proc.stderr
  profile=json.loads(proc.stdout)['numerical_profile'];assert profile['name']==budget.PROFILE and profile['recipes']==['1c']
