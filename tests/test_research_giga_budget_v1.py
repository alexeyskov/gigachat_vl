"""Giga native config/PEFT/route fixes on CPU; no original weight/GPU reads."""
import copy,json,os,subprocess,sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from torch import nn
from src.model import research_giga_loading_v1 as loading
from src.utils import research_giga_budget_v1 as budget


def test_int_scaling_normalized_inmemory_without_source_mutation():
 from transformers import AutoConfig
 raw={'model_type':'deepseek_v3','routed_scaling_factor':1,'rope_scaling':{'factor':64}}
 before=copy.deepcopy(raw);fixed=loading.normalize_config(raw)
 assert raw==before and type(fixed['routed_scaling_factor']) is float and fixed['routed_scaling_factor']==1.
 cfg=AutoConfig.for_model('deepseek_v3',routed_scaling_factor=fixed['routed_scaling_factor']);assert cfg.routed_scaling_factor==1.
 for invalid in (True,0,float('nan'),'1'):
  with pytest.raises(ValueError):loading.normalize_config(dict(raw,routed_scaling_factor=invalid))


def test_actual_pinned_metadata_loader_injects_correct_config_preserves_quant_request():
 from transformers import AutoConfig
 path=Path('/workspace/.hf_home/hub/models--ai-sage--GigaChat3.1-10B-A1.8B-bf16/snapshots')/loading.REVISION;raw=(path/'config.json').read_bytes();capture={}
 def load(p,**kwargs):capture.update(kwargs);return SimpleNamespace(config=kwargs['config']),{'missing_keys':[],'unexpected_keys':['model.layers.26.shared_head.norm.weight'],'mismatched_keys':[],'error_msgs':[]}
 module=ModuleType('_test_private_giga');module.AutoConfig=AutoConfig;module.AutoModelForCausalLM=SimpleNamespace(from_pretrained=load);loading.install_loader(module)
 marker=object();model=module.AutoModelForCausalLM.from_pretrained(str(path),dtype=torch.bfloat16,quantization_config=marker)
 assert type(capture['config'].routed_scaling_factor) is float and capture['config'].routed_scaling_factor==1.
 assert capture['quantization_config'] is marker and capture['local_files_only'] and not capture['trust_remote_code'] and capture['experts_implementation']=='eager'
 assert (path/'config.json').read_bytes()==raw and model._giga_budget_loading['source_file_rewritten'] is False
 assert type(module.AutoConfig.from_pretrained(path).routed_scaling_factor) is float
 assert model._giga_budget_loading['LM_core_missing_or_mismatched_keys'] is False and capture['output_loading_info'] is True
 module2=ModuleType('_test_private_giga_bad');module2.AutoConfig=AutoConfig;module2.AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda p,**kw:(SimpleNamespace(config=kw['config']),{'missing_keys':['model.layers.0.self_attn.q_proj.weight']}));loading.install_loader(module2)
 with pytest.raises(ValueError,match='coverage'):module2.AutoModelForCausalLM.from_pretrained(path)

def test_actual_cached_Giga_tokenizer_uses_normalized_config_and_native_prefix():
 from src.eval.research_text_generation import FixedClockTokenizer
 path=Path('/workspace/.hf_home/hub/models--ai-sage--GigaChat3.1-10B-A1.8B-bf16/snapshots')/loading.REVISION
 tokenizer=FixedClockTokenizer(loading.tokenizer_proxy().from_pretrained(path))
 messages=[{'role':'user','content':'Describe the image.'}]
 prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 full=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':'A cat.'}],tokenize=False,add_generation_prompt=False)
 assert full.startswith(prompt) and tokenizer.encode(full[len(prompt):],add_special_tokens=False)


def tiny():
 from transformers import AutoModelForCausalLM,DeepseekV3Config
 cfg=DeepseekV3Config(vocab_size=32,hidden_size=8,intermediate_size=12,moe_intermediate_size=6,num_hidden_layers=2,first_k_dense_replace=1,num_attention_heads=2,num_key_value_heads=2,n_routed_experts=4,n_shared_experts=1,num_experts_per_tok=2,n_group=1,topk_group=1,routed_scaling_factor=1.,q_lora_rank=None,kv_lora_rank=4,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4,pad_token_id=0)
 model=AutoModelForCausalLM.from_config(cfg,attn_implementation='eager',experts_implementation='eager')
 with torch.no_grad():
  for name,b in model.named_buffers():
   if 'e_score_correction_bias' in name:b.zero_()
 return model


def test_real_native_tiny_ordinary_scoped_PEFT_no_packed_remap_and_backward():
 from peft import LoraConfig
 from src.model.research_giga_peft import resolve_giga_ordinary_targets
 native=tiny();native.eval();tokens=torch.tensor([[1,4,7,2]])
 with torch.no_grad():before=native(tokens).logits.clone()
 original=native.model.layers[1].mlp.experts;ids={n:id(p) for n,p in original.named_parameters()};targets=resolve_giga_ordinary_targets(native)
 config=LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,lora_dropout=.05,bias='none',target_modules=targets)
 wrapped=loading.ScopedPEFT().get_peft_model(native,config)
 assert wrapped.get_base_model() is native and not wrapped.peft_config['default'].target_parameters
 assert native.model.layers[1].mlp.experts is original and ids=={n:id(p) for n,p in original.named_parameters()}
 with torch.no_grad():torch.testing.assert_close(wrapped(tokens).logits,before,atol=0,rtol=0)
 wrapped.train();out=wrapped(tokens,labels=tokens,use_cache=False);out.loss.backward()
 assert torch.isfinite(out.loss) and any(p.grad is not None for n,p in wrapped.named_parameters() if 'lora_' in n)
 assert all(p.grad is None for p in original.parameters())


def test_tiny_full_3c_native_router_visual_experts_and_saved_model_bits():
 from src.model.research_moe import install_research_moe,configure_research_trainability,set_modality_mask
 model=tiny().float();wrappers=install_research_moe(model,3,'c',added_experts=2);configure_research_trainability(model,stage='SFT');tokens=torch.tensor([[1,4,7,2]]);mask=torch.tensor([[False,True,False,False]]);set_modality_mask(model,mask,torch.ones_like(mask));loss=model(tokens,labels=tokens,use_cache=False).loss;loss.backward()
 assert torch.isfinite(loss) and wrappers and any(p.grad is not None for p in wrappers[0].router_mass.parameters())
 assert all(p.dtype==torch.float32 for p in model.parameters() if p.requires_grad)
 state={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};restored=tiny().float();install_research_moe(restored,3,'c',added_experts=2);result=restored.load_state_dict(state,strict=False);assert not result.unexpected_keys
 for n,p in restored.named_parameters():
  if n in state:assert torch.equal(p,state[n])


def test_complete_Giga_projector_guard_not_LFM_specific():
 model=SimpleNamespace(base=SimpleNamespace(config=SimpleNamespace(model_type='deepseek_v3',hidden_size=8)),projector=nn.Sequential(nn.Linear(6,8),nn.GELU(),nn.Linear(8,8)))
 state={'projector.'+k:v for k,v in model.projector.state_dict().items()};assert budget._projector(model,state)['hidden_size']==8
 bad=dict(state);bad.pop('projector.0.weight')
 with pytest.raises(ValueError,match='Complete'):budget._projector(model,bad)


def test_readonly_postload_advice_scope_stats_and_foreign_GPU_refusal(tmp_path,monkeypatch):
 from scripts import research_archival_stock_queue_v5 as scan
 from scripts import probe_research_parallel_b1 as proc
 from src.utils import research_archive_gpu_parity as host
 path=tmp_path/loading.REVISION;path.mkdir();sizes={'one.safetensors':7,'two.safetensors':9}
 for n,size in sizes.items():(path/n).write_bytes(b'x'*size)
 monkeypatch.setattr(budget,'WEIGHT_BYTES',sizes);monkeypatch.setattr(scan,'scan_anchor_users',lambda ids:[]);monkeypatch.setattr(proc,'gpu_snapshot',lambda:{'processes':[{'pid':os.getpid()}]});monkeypatch.setattr(host,'host_snapshot',lambda:{'unused':1});monkeypatch.setattr(host,'conservative_host_headroom',lambda value:100*2**30)
 model=SimpleNamespace(model_ref={'repo':budget.MODEL,'revision':budget.REVISION},model_path=str(path));before={p:p.stat() for p in path.iterdir()};result=budget.advise_giga_sources(model)
 assert result['assumed_reclaim_credit_bytes']==0 and not result['weights_or_config_or_pointers_changed']
 assert before=={p:p.stat() for p in path.iterdir()}
 monkeypatch.setattr(proc,'gpu_snapshot',lambda:{'processes':[{'pid':999999}]})
 with pytest.raises(ValueError,match='Foreign'):budget.advise_giga_sources(model)


def test_all_three_Giga_metadata_entrypoints_are_weightfree_and_NoTorch():
 for action in ('run','eval','validate'):
  code=f"import runpy,sys;sys.argv=['x','--metadata'];runpy.run_path('scripts/{action}_research_giga_budget_v1.py',run_name='__main__');assert 'torch' not in sys.modules"
  process=subprocess.run([sys.executable,'-c',code],cwd=budget.ROOT,capture_output=True,text=True);assert process.returncode==0,process.stderr
  profile=json.loads(process.stdout)['numerical_profile'];assert profile['model']==budget.MODEL and profile['model_revision']==budget.REVISION and profile['visions']==['google/siglip2-so400m-patch14-384','Qwen/Qwen2.5-VL-7B-Instruct']
