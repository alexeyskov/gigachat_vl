"""Actual tiny NF4 CUDA proof; shared-GPU timings are not throughput claims."""
import copy
import gc
import io
import subprocess

import pytest
import torch
from torch import nn

from src.model.research_giga_peft import (
    get_giga_ordinary_peft_model,ordinary_adapter_state_dict,
    load_ordinary_adapter_state_dict,resolve_giga_ordinary_targets,
)


@pytest.fixture(scope='module',autouse=True)
def gpu_budget():
    try:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    except (OSError,subprocess.CalledProcessError,ValueError):pytest.skip('No nvidia-smi CUDA device')
    if free<=4096:pytest.skip('Tiny isolated CUDA proof requires over 4 GiB free before allocating')
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.cuda.set_device(0);gc.collect();torch.cuda.empty_cache()
    baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    yield
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()-baseline
    assert peak<256*2**20,f'Tiny CUDA tensor budget exceeded: {peak}'
    print({'giga_tiny_cuda_tensor_peak_bytes':peak,'shared_gpu_throughput_claim':False})
    gc.collect();torch.cuda.empty_cache()


@pytest.fixture(scope='module')
def source():
    from transformers import AutoModelForCausalLM,DeepseekV3Config
    torch.manual_seed(135)
    config=DeepseekV3Config(vocab_size=64,hidden_size=64,intermediate_size=128,moe_intermediate_size=64,
        num_hidden_layers=2,first_k_dense_replace=1,num_attention_heads=2,num_key_value_heads=2,
        n_routed_experts=4,n_shared_experts=1,num_experts_per_tok=2,n_group=1,topk_group=1,
        routed_scaling_factor=1.0,q_lora_rank=None,kv_lora_rank=64,
        qk_nope_head_dim=32,qk_rope_head_dim=32,v_head_dim=32,pad_token_id=0)
    model=AutoModelForCausalLM.from_config(config,attn_implementation='eager',experts_implementation='eager').bfloat16()
    with torch.no_grad():
        for name,buffer in model.named_buffers():
            if 'e_score_correction_bias' in name:buffer.zero_()
    state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    assert sum(value.numel()*value.element_size() for value in state.values())<2**20
    return config,state


def quantized_native(source,packed=False):
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from src.model.research_quantization import quantize_packed_experts
    config,state=source
    model=AutoModelForCausalLM.from_config(copy.deepcopy(config),attn_implementation='eager',experts_implementation='eager')
    model.load_state_dict(state,strict=True);model.to('cuda',dtype=torch.bfloat16).requires_grad_(False)
    targets=resolve_giga_ordinary_targets(model)
    for name in targets:
        parent_name,_,leaf=name.rpartition('.');parent=model.get_submodule(parent_name)
        original=getattr(parent,leaf)
        replacement=bnb.nn.Linear4bit(original.in_features,original.out_features,bias=original.bias is not None,
            compute_dtype=torch.bfloat16,compress_statistics=True,quant_type='nf4',device='meta')
        replacement.weight=bnb.nn.Params4bit(original.weight.detach().contiguous(),requires_grad=False,
            compress_statistics=True,quant_type='nf4',module=replacement).to('cuda')
        if original.bias is not None:
            replacement.bias=nn.Parameter(original.bias.detach().clone(),requires_grad=False)
        setattr(parent,leaf,replacement)
    model.is_loaded_in_4bit=True;model.is_loaded_in_8bit=False;model.is_quantized=True
    model.quantization_method='bitsandbytes';model.hf_device_map={'':0}
    model.config.quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    if packed:
        quantize_packed_experts(model,rank=16,alpha=32,dropout=.05)
        # Match the runner's independent FP32-master policy for custom factors.
        for parameter in model.parameters():
            if parameter.requires_grad:parameter.data=parameter.data.float()
    return model


def native_reference(model):
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit as PeftLinear4bit
    targets=resolve_giga_ordinary_targets(model)
    config=LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,lora_dropout=.05,bias='none',target_modules=targets)
    for name in targets:
        parent_name,_,leaf=name.rpartition('.');parent=model.get_submodule(parent_name)
        setattr(parent,leaf,PeftLinear4bit(getattr(parent,leaf),'default',config=config,r=16,lora_alpha=32))
    # This is the same native post-construction policy invoked by PeftModel.
    from peft.tuners.tuners_utils import cast_adapter_dtype
    cast_adapter_dtype(model,adapter_name='default',autocast_adapter_dtype=True)
    return model


def trainable(model):return {name:parameter for name,parameter in model.named_parameters() if parameter.requires_grad}


@pytest.mark.parametrize('packed',[False,True])
def test_actual_bnb_dispatch_native_parity_all_gradients_rng_and_frozen_bases(source,packed):
    from peft.tuners.lora.bnb import Linear4bit as PeftLinear4bit
    native=quantized_native(source,packed)
    packed_bank=native.model.layers[1].mlp.experts;router=native.model.layers[1].mlp.gate;head=native.lm_head
    packed_ids={name:id(parameter) for name,parameter in packed_bank.named_parameters()}
    packed_flags={name:parameter.requires_grad for name,parameter in packed_bank.named_parameters()}
    configuration=native.config.quantization_config
    wrapped,receipt=get_giga_ordinary_peft_model(native)
    assert native.config.quantization_config is configuration
    assert native.model.layers[1].mlp.experts is packed_bank
    assert native.model.layers[1].mlp.gate is router and native.lm_head is head
    assert {name:id(parameter) for name,parameter in packed_bank.named_parameters()}==packed_ids
    assert {name:parameter.requires_grad for name,parameter in packed_bank.named_parameters()}==packed_flags
    assert receipt['dispatch_metadata']['is_loaded_in_4bit']
    assert receipt['dispatch_metadata']['config_quantization_config_preserved']
    assert receipt['dispatch_metadata']['quantization_method']=='bitsandbytes'
    assert receipt['quantized_ordinary_targets']==receipt['targets']
    assert not receipt['packed_parameters_targeted_by_peft']
    for name in receipt['targets']:
        module=native.get_submodule(name)
        assert type(module) is PeftLinear4bit
        assert module.get_base_layer().compute_dtype==torch.bfloat16
        assert not module.get_base_layer().weight.requires_grad
        assert module.r['default']==16 and module.lora_alpha['default']==32
        assert module.lora_dropout['default'].p==.05
        assert module.lora_A['default'].weight.dtype==module.lora_B['default'].weight.dtype==torch.float32
    reference=native_reference(quantized_native(source,packed))
    with torch.no_grad():
        for parameter in trainable(native).values():parameter.normal_(std=.02)
        actual_parameters=trainable(native);expected_parameters=trainable(reference)
        assert actual_parameters.keys()==expected_parameters.keys()
        for name,value in actual_parameters.items():expected_parameters[name].copy_(value)
    frozen={name:parameter.detach().clone() for name,parameter in native.named_parameters() if not parameter.requires_grad}
    tokens=torch.tensor([[1,4,6,7,9,2],[1,5,8,3,11,2]],device='cuda')
    inputs=native.get_input_embeddings()(tokens).detach().requires_grad_(True)
    reference_inputs=inputs.detach().clone().requires_grad_(True)
    wrapped.train();reference.train()
    torch.manual_seed(19);cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        actual=wrapped(inputs_embeds=inputs,labels=tokens,use_cache=False)
    actual.loss.backward();after_cpu=torch.get_rng_state();after_cuda=torch.cuda.get_rng_state()
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        expected=reference(inputs_embeds=reference_inputs,labels=tokens,use_cache=False)
    expected.loss.backward()
    torch.testing.assert_close(actual.logits,expected.logits,atol=0,rtol=0)
    torch.testing.assert_close(actual.loss,expected.loss,atol=0,rtol=0)
    torch.testing.assert_close(inputs.grad,reference_inputs.grad,atol=0,rtol=0)
    assert torch.equal(torch.get_rng_state(),after_cpu)
    assert torch.equal(torch.cuda.get_rng_state(),after_cuda)
    for name,value in actual_parameters.items():
        assert value.dtype==expected_parameters[name].dtype==torch.float32
        if value.grad is None:assert expected_parameters[name].grad is None
        else:
            assert torch.isfinite(value.grad).all()
            torch.testing.assert_close(value.grad,expected_parameters[name].grad,atol=0,rtol=0)
    assert all(parameter.grad is None for parameter in native.parameters() if not parameter.requires_grad)
    old={name:value.detach().clone() for name,value in actual_parameters.items()}
    optimizer=torch.optim.AdamW(actual_parameters.values(),lr=.003);optimizer.step()
    assert any(not torch.equal(value,old[name]) for name,value in actual_parameters.items())
    for name,value in frozen.items():torch.testing.assert_close(dict(native.named_parameters())[name],value,atol=0,rtol=0)


@pytest.mark.parametrize('packed',[False,True])
def test_actual_nf4_optimizer_adapter_checkpoint_roundtrip_and_cached_generation(source,packed):
    native=quantized_native(source,packed);wrapped,receipt=get_giga_ordinary_peft_model(native)
    tokens=torch.tensor([[1,4,6,7,2]],device='cuda');wrapped.train()
    optimizer=torch.optim.AdamW(trainable(wrapped).values(),lr=.003)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=wrapped(input_ids=tokens,labels=tokens,use_cache=False).loss
    loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    learned={name:value.detach().cpu().clone() for name,value in trainable(wrapped).items()}
    ordinary=ordinary_adapter_state_dict(wrapped)
    buffer=io.BytesIO();torch.save({'model':learned,'ordinary':ordinary,'optimizer':optimizer.state_dict()},buffer);buffer.seek(0)
    payload=torch.load(buffer,map_location='cpu',weights_only=True)
    restored,_=get_giga_ordinary_peft_model(quantized_native(source,packed))
    result=restored.load_state_dict(payload['model'],strict=False)
    assert not result.unexpected_keys
    load_ordinary_adapter_state_dict(restored,payload['ordinary'])
    restored_optimizer=torch.optim.AdamW(trainable(restored).values(),lr=.003)
    restored_optimizer.load_state_dict(payload['optimizer'])
    assert len(restored_optimizer.state)==len(optimizer.state)
    original_state=optimizer.state_dict()['state'];restored_state=restored_optimizer.state_dict()['state']
    assert original_state.keys()==restored_state.keys()
    for identifier,values in original_state.items():
        for name,value in values.items():
            torch.testing.assert_close(value.detach().cpu(),restored_state[identifier][name].detach().cpu(),atol=0,rtol=0)
    wrapped.eval();restored.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected=wrapped(input_ids=tokens,use_cache=False).logits
        actual=restored(input_ids=tokens,use_cache=False).logits
        generated=wrapped.generate(input_ids=tokens,attention_mask=torch.ones_like(tokens),min_new_tokens=3,max_new_tokens=3,do_sample=False,use_cache=True)
        resumed=restored.generate(input_ids=tokens,attention_mask=torch.ones_like(tokens),min_new_tokens=3,max_new_tokens=3,do_sample=False,use_cache=True)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert torch.equal(generated,resumed)
    assert generated.shape[1]==tokens.shape[1]+3
    assert wrapped.base_model_prepare_inputs_for_generation.__self__ is native
    assert all(value.dtype==torch.float32 for value in trainable(restored).values())
    # A checkpoint must preserve the next optimizer update, not just inference.
    wrapped.train();restored.train()
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    with torch.autocast('cuda',dtype=torch.bfloat16):next_loss=wrapped(input_ids=tokens,labels=tokens,use_cache=False).loss
    next_loss.backward();optimizer.step()
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
    with torch.autocast('cuda',dtype=torch.bfloat16):resumed_loss=restored(input_ids=tokens,labels=tokens,use_cache=False).loss
    resumed_loss.backward();restored_optimizer.step()
    torch.testing.assert_close(next_loss,resumed_loss,atol=0,rtol=0)
    for name,value in trainable(wrapped).items():
        torch.testing.assert_close(value,trainable(restored)[name],atol=0,rtol=0)


def test_materialized_nf4_requires_consistent_flags_config_and_bf16_compute(source):
    model=quantized_native(source)
    targets=resolve_giga_ordinary_targets(model)
    model.is_loaded_in_4bit=False
    with pytest.raises(ValueError,match='is_loaded_in_4bit'):get_giga_ordinary_peft_model(model)
    model.is_loaded_in_4bit=True;config=model.config.quantization_config;delattr(model.config,'quantization_config')
    with pytest.raises(ValueError,match='quantization_config'):get_giga_ordinary_peft_model(model)
    model.config.quantization_config=config
    first=model.get_submodule(targets[0]);first.compute_dtype=torch.float32
    with pytest.raises(ValueError,match='BF16 compute'):get_giga_ordinary_peft_model(model)
    assert not hasattr(model,'peft_config')
    assert all('.lora_' not in name for name,_ in model.named_parameters())
