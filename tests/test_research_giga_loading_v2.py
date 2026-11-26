import json
from pathlib import Path
from types import ModuleType,SimpleNamespace
import pytest
import torch
from src.model import research_giga_loading_v2 as loading

def test_actual_native_MTP_weight_conversion_and_set_loadinginfo_json_roundtrip(tmp_path):
 from transformers import AutoModelForCausalLM,AutoConfig,DeepseekV3Config
 from safetensors.torch import save_file
 cfg=DeepseekV3Config(vocab_size=32,hidden_size=8,intermediate_size=12,moe_intermediate_size=6,num_hidden_layers=2,first_k_dense_replace=1,num_attention_heads=2,num_key_value_heads=2,n_routed_experts=4,n_shared_experts=1,num_experts_per_tok=2,n_group=1,topk_group=1,routed_scaling_factor=1.,q_lora_rank=None,kv_lora_rank=4,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4,pad_token_id=0)
 native=AutoModelForCausalLM.from_config(cfg,attn_implementation='eager',experts_implementation='eager');state=native.state_dict()
 for i in range(4):
  for part,shape in [('gate_proj',(6,8)),('up_proj',(6,8)),('down_proj',(8,6))]:state[f'model.layers.26.mlp.experts.{i}.{part}.weight']=torch.zeros(shape)
 cfg.save_pretrained(tmp_path);save_file(state,tmp_path/'model.safetensors');seen={}
 def tiny_native(path,**kw):
  model,info=AutoModelForCausalLM.from_pretrained(tmp_path,local_files_only=True,experts_implementation='eager',output_loading_info=True);seen.update(info);return model,info
 private=ModuleType('_actual_tiny_Giga_adapter');private.AutoConfig=AutoConfig;private.AutoModelForCausalLM=SimpleNamespace(from_pretrained=tiny_native);loading.install_loader(private)
 pinned=Path('/workspace/.hf_home/hub/models--ai-sage--GigaChat3.1-10B-A1.8B-bf16/snapshots/189fff27a1dee68473960c3d5bca53e0e07a3191')
 restored=private.AutoModelForCausalLM.from_pretrained(pinned)
 assert isinstance(seen['unexpected_keys'],set) and seen['unexpected_keys']==loading.CONVERTED_MTP
 report=restored._giga_budget_loading;assert set(report['native_loading_info']['unexpected_keys'])==loading.CONVERTED_MTP and report['unused_MTP_index_keys']==210
 assert json.loads(json.dumps(report))==report and not report['LM_core_missing_or_mismatched_keys']
 for n,t in native.state_dict().items():assert torch.equal(t,restored.state_dict()[n])

@pytest.mark.parametrize('info',[{'missing_keys':{'model.layers.0.self_attn.q_proj.weight'}},{'mismatched_keys':{('model.norm.weight',(3,),(8,))}},{'unexpected_keys':{'model.layers.26.not_in_source.weight'}},{'unexpected_keys':{'model.layers.0.unknown'}}])
def test_unrelated_missing_mismatched_or_unexpected_keys_still_fail(info):
 from transformers import AutoConfig
 private=ModuleType('_Giga_adapter_bad');private.AutoConfig=AutoConfig
 private.AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda path,**kw:(SimpleNamespace(config=kw['config']),info));loading.install_loader(private)
 path='/workspace/.hf_home/hub/models--ai-sage--GigaChat3.1-10B-A1.8B-bf16/snapshots/189fff27a1dee68473960c3d5bca53e0e07a3191'
 with pytest.raises(ValueError,match='coverage'):private.AutoModelForCausalLM.from_pretrained(path)
