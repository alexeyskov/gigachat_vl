"""Tiny real native DeepSeek CPU/meta tests; no quantized CUDA claims."""
import copy
import io

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM,DeepseekV3Config

from src.model.research_giga_peft import (
    get_giga_ordinary_peft_model,load_ordinary_adapter_state_dict,
    ordinary_adapter_state_dict,resolve_giga_ordinary_targets,
)


def tiny_model(device='cpu',dtype=torch.float32):
    config=DeepseekV3Config(vocab_size=32,hidden_size=8,intermediate_size=12,moe_intermediate_size=6,
        num_hidden_layers=2,first_k_dense_replace=1,num_attention_heads=2,num_key_value_heads=2,
        n_routed_experts=4,n_shared_experts=2,num_experts_per_tok=2,n_group=1,topk_group=1,
        routed_scaling_factor=1.0,q_lora_rank=None,kv_lora_rank=4,
        qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4,pad_token_id=0)
    with torch.device(device):
        model=AutoModelForCausalLM.from_config(config,attn_implementation='eager',experts_implementation='eager')
    model=model.to(dtype=dtype)
    with torch.no_grad():
        for name,buffer in model.named_buffers():
            if 'e_score_correction_bias' in name:buffer.zero_()
    return model


def nonzero_adapters(model):
    with torch.no_grad():
        for name,parameter in model.named_parameters():
            if 'lora_' in name:parameter.normal_(std=.03)


def test_original_public_api_error_is_reproduced_and_exact_scoped_api_keeps_contract():
    from peft import LoraConfig,PeftModelForCausalLM,get_peft_model
    original=tiny_model();targets=resolve_giga_ordinary_targets(original)
    with pytest.raises(ValueError,match='ParamWrapper does not work with lora_dropout'):
        get_peft_model(copy.deepcopy(original),LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,
            lora_dropout=.05,bias='none',target_modules=targets))
    source_config=copy.deepcopy(original.config.to_dict())
    packed=original.model.layers[1].mlp.experts
    router=original.model.layers[1].mlp.gate
    head=original.lm_head
    linears={name:original.get_submodule(name) for name in targets}
    wrapped,receipt=get_giga_ordinary_peft_model(original,target_modules=list(reversed(targets)))
    assert type(wrapped) is PeftModelForCausalLM
    assert wrapped.get_base_model() is original and wrapped.config is original.config
    assert original.config.to_dict()==source_config
    assert original.model.layers[1].mlp.experts is packed
    assert original.model.layers[1].mlp.gate is router and original.lm_head is head
    assert receipt['targets']==targets and not wrapped.peft_config['default'].target_parameters
    expected=sum(16*(module.in_features+module.out_features) for module in linears.values())
    assert receipt['ordinary_lora_parameters']==expected
    for name,module in linears.items():
        adapted=original.get_submodule(name)
        assert adapted.get_base_layer() is module
        assert adapted.r['default']==16 and adapted.lora_alpha['default']==32
        assert adapted.lora_dropout['default'].p==.05
    assert all(parameter.requires_grad==('lora_' in name) for name,parameter in wrapped.named_parameters())
    assert not receipt['packed_parameters_targeted_by_peft']


def test_real_cpu_forward_backward_optimizer_step_updates_only_adapters():
    model=tiny_model();wrapped,_=get_giga_ordinary_peft_model(model)
    wrapped.train();tokens=torch.tensor([[1,4,6,7,2],[1,5,9,3,2]])
    before={name:parameter.detach().clone() for name,parameter in wrapped.named_parameters()}
    optimizer=torch.optim.AdamW([parameter for parameter in wrapped.parameters() if parameter.requires_grad],lr=.01)
    loss=wrapped(input_ids=tokens,labels=tokens,use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is None for name,parameter in wrapped.named_parameters() if 'lora_' not in name)
    assert any(parameter.grad is not None and parameter.grad.abs().sum()>0 for name,parameter in wrapped.named_parameters() if 'lora_B' in name)
    optimizer.step()
    assert any(not torch.equal(parameter,before[name]) for name,parameter in wrapped.named_parameters() if 'lora_B' in name)
    for name,parameter in wrapped.named_parameters():
        if 'lora_' not in name:torch.testing.assert_close(parameter,before[name],atol=0,rtol=0)


def test_native_peft_linear_full_model_parity_with_seeded_dropout_gradients():
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear as PeftLinear
    native=tiny_model();reference=copy.deepcopy(native)
    wrapped,receipt=get_giga_ordinary_peft_model(native)
    reference.requires_grad_(False)
    config=LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,lora_dropout=.05,
                      bias='none',target_modules=receipt['targets'])
    for name in receipt['targets']:
        parent,_,leaf=name.rpartition('.')
        owner=reference.get_submodule(parent)
        setattr(owner,leaf,PeftLinear(getattr(owner,leaf),'default',config=config,r=16,lora_alpha=32))
    nonzero_adapters(wrapped)
    reference.load_state_dict(wrapped.get_base_model().state_dict(),strict=True)
    wrapped.train();reference.train()
    tokens=torch.tensor([[1,4,6,7,2],[1,5,9,3,2]])
    torch.manual_seed(99);rng=torch.get_rng_state()
    actual=wrapped(input_ids=tokens,labels=tokens,use_cache=False)
    actual.loss.backward();after=torch.get_rng_state()
    torch.set_rng_state(rng)
    expected=reference(input_ids=tokens,labels=tokens,use_cache=False)
    expected.loss.backward()
    torch.testing.assert_close(actual.logits,expected.logits,atol=0,rtol=0)
    torch.testing.assert_close(actual.loss,expected.loss,atol=0,rtol=0)
    assert torch.equal(torch.get_rng_state(),after)
    actual_parameters=dict(wrapped.get_base_model().named_parameters())
    for name,parameter in reference.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(actual_parameters[name].grad,parameter.grad,atol=0,rtol=0)


def test_canonical_full_state_and_adapter_only_roundtrip(tmp_path):
    original=tiny_model();same_base=copy.deepcopy(original)
    wrapped,receipt=get_giga_ordinary_peft_model(original)
    nonzero_adapters(wrapped);wrapped.eval()
    tokens=torch.tensor([[1,4,6,7,2]])
    with torch.no_grad():expected=wrapped(input_ids=tokens,use_cache=False).logits
    adapter=ordinary_adapter_state_dict(wrapped)
    assert len(adapter)==2*len(receipt['targets'])
    assert all(name.startswith('base_model.model.') and '.lora_' in name and '.experts.' not in name for name in adapter)
    path=tmp_path/'ordinary.pt';torch.save(adapter,path)
    restored,_=get_giga_ordinary_peft_model(same_base)
    report=load_ordinary_adapter_state_dict(restored,torch.load(path,weights_only=True))
    assert report['loaded_parameters']==receipt['ordinary_lora_parameters']
    restored.eval()
    with torch.no_grad():torch.testing.assert_close(restored(input_ids=tokens,use_cache=False).logits,expected,atol=0,rtol=0)
    buffer=io.BytesIO();torch.save(wrapped.state_dict(),buffer);buffer.seek(0)
    independent,_=get_giga_ordinary_peft_model(tiny_model())
    independent.load_state_dict(torch.load(buffer,weights_only=True),strict=True);independent.eval()
    with torch.no_grad():torch.testing.assert_close(independent(input_ids=tokens,use_cache=False).logits,expected,atol=0,rtol=0)


@pytest.mark.parametrize('kind',['missing','extra','shape','nonfinite','dtype'])
def test_adapter_restore_rejects_bad_state_before_any_mutation(kind):
    wrapped,_=get_giga_ordinary_peft_model(tiny_model());state=ordinary_adapter_state_dict(wrapped)
    first=next(iter(state));before={name:value.clone() for name,value in state.items()}
    if kind=='missing':state.pop(first)
    elif kind=='extra':state['base_model.model.model.layers.1.mlp.experts.gate_up_proj']=torch.zeros(1)
    elif kind=='shape':state[first]=torch.zeros(1)
    elif kind=='nonfinite':state[first].fill_(float('nan'))
    else:state[first]=state[first].to(torch.int64)
    with pytest.raises(ValueError):load_ordinary_adapter_state_dict(wrapped,state)
    after=ordinary_adapter_state_dict(wrapped)
    for name,value in before.items():torch.testing.assert_close(after[name],value,atol=0,rtol=0)


@pytest.mark.parametrize('autocast',[True,False])
@pytest.mark.parametrize('device',['cpu','meta'])
def test_native_adapter_dtype_and_meta_contract(autocast,device):
    native=tiny_model(device,torch.bfloat16)
    wrapped,receipt=get_giga_ordinary_peft_model(native,autocast_adapter_dtype=autocast)
    assert all(parameter.device.type==device for parameter in wrapped.parameters())
    # Native PEFT skips moving/casting a meta base layer, so its new factors
    # retain the default FP32 construction dtype even when autocast=False.
    adapter_dtype=torch.float32 if autocast or device=='meta' else torch.bfloat16
    for name,parameter in wrapped.named_parameters():
        assert parameter.dtype==(adapter_dtype if 'lora_' in name else torch.bfloat16)
    assert receipt['autocast_adapter_dtype']==autocast


def test_preflight_rejects_omitted_ambiguous_duplicate_and_repeated_scope_without_mutation():
    model=tiny_model();targets=resolve_giga_ordinary_targets(model)
    for bad in [targets[:-1],targets+['lm_head'],targets+[targets[0]],'up_proj']:
        with pytest.raises(ValueError):get_giga_ordinary_peft_model(model,target_modules=bad)
        assert all(parameter.requires_grad for parameter in model.parameters())
    get_giga_ordinary_peft_model(model)
    with pytest.raises(ValueError,match='already installed'):get_giga_ordinary_peft_model(model)


def test_failures_restore_rng_flags_modes_and_native_module_identities(monkeypatch):
    import peft
    native=tiny_model();native.model.layers[0].self_attn.q_proj.eval()
    targets={name:native.get_submodule(name) for name in resolve_giga_ordinary_targets(native)}
    flags=[(parameter,parameter.requires_grad) for parameter in native.parameters()]
    modes=[(module,module.training) for module in native.modules()]
    real=peft.get_peft_model
    def injected_failure(*args,**kwargs):
        real(*args,**kwargs);torch.rand(7)
        raise RuntimeError('injected post-injection failure')
    monkeypatch.setattr(peft,'get_peft_model',injected_failure)
    torch.manual_seed(55);before=torch.get_rng_state()
    with pytest.raises(RuntimeError,match='injected'):get_giga_ordinary_peft_model(native)
    assert torch.equal(torch.get_rng_state(),before)
    assert not hasattr(native,'peft_config')
    assert all(native.get_submodule(name) is module for name,module in targets.items())
    assert all(parameter.requires_grad==requires_grad for parameter,requires_grad in flags)
    assert all(module.training==training for module,training in modes)


def test_existing_packed_lora_parameter_flags_are_preserved_without_injection():
    model=tiny_model();bank=model.model.layers[1].mlp.experts
    # Structural sentinel only: native CPU experts do not execute these fields.
    # Actual NF4 expert CUDA conversion is explicitly outside this test.
    bank.register_parameter('lora_A',nn.Parameter(torch.ones(2,3),requires_grad=True))
    bank.register_parameter('lora_B',nn.Parameter(torch.zeros(3,2),requires_grad=False))
    original_a=bank.lora_A;original_b=bank.lora_B;gate_up=bank.gate_up_proj
    wrapped,receipt=get_giga_ordinary_peft_model(model)
    assert model.model.layers[1].mlp.experts is bank
    assert bank.lora_A is original_a and bank.lora_A.requires_grad
    assert bank.lora_B is original_b and not bank.lora_B.requires_grad
    assert bank.gate_up_proj is gate_up and not gate_up.requires_grad
    assert receipt['existing_packed_lora_flags_preserved']==2
    assert all('.experts.' not in name for name in ordinary_adapter_state_dict(wrapped))


def test_unknown_library_version_fails_before_native_mutation(monkeypatch):
    import src.model.research_giga_peft as helper
    native=tiny_model();monkeypatch.setattr(helper,'version',lambda _: 'unknown')
    with pytest.raises(NotImplementedError,match='validated only'):get_giga_ordinary_peft_model(native)
    assert all(parameter.requires_grad for parameter in native.parameters())


def test_native_generation_binding_and_eval_modes_are_preserved():
    model=tiny_model();model.eval()
    wrapped,receipt=get_giga_ordinary_peft_model(model)
    assert not wrapped.training and not wrapped.base_model.training
    assert all(not model.get_submodule(name).training for name in receipt['targets'])
    nonzero_adapters(wrapped)
    tokens=torch.tensor([[1,4,6]])
    with torch.no_grad():
        actual=wrapped.generate(input_ids=tokens,attention_mask=torch.ones_like(tokens),max_new_tokens=2,do_sample=False)
        expected=model.generate(input_ids=tokens,attention_mask=torch.ones_like(tokens),max_new_tokens=2,do_sample=False)
    assert torch.equal(actual,expected)
    assert wrapped.base_model_prepare_inputs_for_generation.__self__ is model


def test_custom_linear_and_wrong_checkpointing_order_fail_before_mutation():
    class UnvalidatedLinear(nn.Linear):pass
    model=tiny_model();old=model.model.layers[0].self_attn.q_proj
    model.model.layers[0].self_attn.q_proj=UnvalidatedLinear(old.in_features,old.out_features,bias=False)
    with pytest.raises(NotImplementedError,match='CUDA validation'):get_giga_ordinary_peft_model(model)
    assert all(parameter.requires_grad for parameter in model.parameters())
    model=tiny_model();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    with pytest.raises(ValueError,match='before enabling gradient checkpointing'):get_giga_ordinary_peft_model(model)
    assert all(parameter.requires_grad for parameter in model.parameters())
