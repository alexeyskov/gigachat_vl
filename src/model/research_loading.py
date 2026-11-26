"""Explicit GPT-OSS final-phase loading prototype; not an active runner change.

Official source route is MXFP4 -> differentiable bf16 -> optional native NF4 and
per-expert/attention LoRA. A bitsandbytes quantization config must never override
the source MXFP4 metadata directly. Only pinned, already local checkpoints are
accepted; this helper never initiates model downloads or schedules GPT training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import torch
from torch import nn
from transformers import AutoModelForCausalLM, Mxfp4Config

from src.model.research_quantization import (
    NF4LoRALinear, quantize_packed_experts, set_packed_lora_trainable,
)

RECIPES={'1a','1b','1c','2a','2b','2c','3a','3b','3c','4q','4f'}
ATTENTION_PROJECTIONS={'q_proj','k_proj','v_proj','o_proj'}


def _memory(device):
    if not torch.cuda.is_available():return {'available':False}
    return {'available':True,'allocated_bytes':torch.cuda.memory_allocated(device),
            'reserved_bytes':torch.cuda.memory_reserved(device),
            'allocator_global_peak_bytes':torch.cuda.max_memory_allocated(device)}


def _native_banks(model):
    banks=[(name,module) for name,module in model.named_modules() if module.__class__.__name__=='GptOssExperts']
    if not banks:raise ValueError('Expected materialized native GptOssExperts after MXFP4 dequantization')
    for name,bank in banks:
        for parameter in ('gate_up_proj','down_proj'):
            value=getattr(bank,parameter,None)
            if not isinstance(value,torch.Tensor) or value.ndim!=3 or value.dtype!=torch.bfloat16 or value.device.type!='cuda':
                raise ValueError(f'{name}.{parameter} must be dequantized bf16 CUDA native expert weights')
    return banks


def quantize_gpt_oss_attention(model,rank=16,alpha=32,dropout=.05):
    """Convert original ordinary attention Linear layers, preserving frozen bias."""
    candidates=[]
    for parent_name,parent in model.named_modules():
        if not (parent_name.endswith('self_attn') or parent_name.endswith('attention')):continue
        for name,child in parent.named_children():
            if name in ATTENTION_PROJECTIONS and type(child) is nn.Linear:
                candidates.append((parent_name,parent,name,child))
    if not candidates:raise ValueError('No ordinary GPT-OSS attention projection Linear modules found')
    for parent_name,_,name,module in candidates:
        if module.weight.device.type!='cuda' or module.weight.dtype!=torch.bfloat16:
            raise ValueError(f'{parent_name}.{name} needs materialized bf16 CUDA weights before NF4 conversion')
    report={'modules':[],'original_weight_parameters':0,'original_bias_parameters':0,
            'original_weight_bytes':0,'lora_parameters':0,'rank':rank}
    for parent_name,parent,name,module in candidates:
        replacement=NF4LoRALinear(module.weight,rank=rank,alpha=alpha,dropout=dropout,
                                  compute_dtype=torch.bfloat16,double_quant=True,bias=module.bias)
        setattr(parent,name,replacement)
        report['modules'].append(f'{parent_name}.{name}')
        report['original_weight_parameters']+=module.weight.numel()
        report['original_weight_bytes']+=module.weight.numel()*module.weight.element_size()
        report['original_bias_parameters']+=0 if module.bias is None else module.bias.numel()
        report['lora_parameters']+=sum(p.numel() for n,p in replacement.named_parameters() if 'lora_' in n)
    report['excluded_modules']='lm_head, embedding, norms, sinks, router and expert-bank modules'
    return report


def _nf4_state_bytes(model):
    seen=set();total=0
    def visit(state):
        nonlocal total
        if state is None:return
        if isinstance(state,torch.Tensor):
            key=(str(state.device),state.data_ptr(),state.numel()*state.element_size())
            if key not in seen:seen.add(key);total+=key[2]
        else:
            for attr in ('absmax','code','offset','state2'):visit(getattr(state,attr,None))
    for module in model.modules():
        if isinstance(module,NF4LoRALinear):visit(module.base.weight.quant_state)
    return total


def load_gpt_oss_for_research(checkpoint_path,*,revision=None,stage='A0',recipe='4f',
                              device='cuda:0',lora_rank=16,lora_alpha=32,lora_dropout=.05):
    """Return ``(model, conversion_report)`` from a pinned local GPT checkpoint.

    The returned language base is frozen. QLoRA adapters train on SFT and on A1
    only for baseline4q; family1 A1 leaves original LLM adapters frozen. A0 builds
    rank0 NF4 bases. Full profiles return dequantized bf16, with fp32 masters and
    full-training policy left to the caller. Install E+K after this helper.
    """
    if stage not in {'A0','A1','SFT'} or recipe not in RECIPES:
        raise ValueError('Unsupported research stage or recipe')
    if lora_rank<1 or lora_alpha<=0 or not 0<=lora_dropout<1:
        raise ValueError('Invalid LoRA configuration')
    target=torch.device(device)
    if target.type!='cuda':raise ValueError('GPT-OSS research loader requires an explicit CUDA device')
    if target.index is None:target=torch.device('cuda:0')
    checkpoint=Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_dir() or not (checkpoint/'config.json').is_file():
        raise ValueError('Checkpoint must be an existing local directory with config.json; downloads are not performed')
    pinned=revision or (checkpoint.name if re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}',checkpoint.name) else None)
    if not isinstance(pinned,str) or not re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}',pinned):
        raise ValueError('A pinned model revision SHA is required')
    config=json.loads((checkpoint/'config.json').read_text())
    if config.get('model_type')!='gpt_oss' or (config.get('quantization_config') or {}).get('quant_method')!='mxfp4':
        raise ValueError('Expected original GPT-OSS MXFP4 source metadata')
    quantized=recipe.startswith('1') or recipe=='4q'
    rank=0 if stage=='A0' else lora_rank
    before=_memory(target)
    mxfp4=Mxfp4Config(dequantize=True)
    # dequantize is a loading attribute and intentionally absent from to_dict()
    # on this Transformers version: inspect the attribute, never infer from JSON.
    if mxfp4.dequantize is not True:raise RuntimeError('MXFP4 dequantization loading attribute was not retained')
    model=AutoModelForCausalLM.from_pretrained(
        str(checkpoint),local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,
        attn_implementation='eager',device_map={'':str(target)},quantization_config=mxfp4,
    )
    if getattr(model.config,'model_type',None)!='gpt_oss':raise ValueError('Loaded checkpoint is not GPT-OSS')
    model.requires_grad_(False)
    banks=_native_banks(model)
    original_parameters=sum(p.numel() for p in model.parameters())
    original_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
    packed_parameters=sum(getattr(bank,name).numel() for _,bank in banks for name in ('gate_up_proj','down_proj'))
    report={'backend':'gpt_oss','stage':stage,'recipe':recipe,
            'model_ref':{'path':str(checkpoint),'revision':pinned,'config_sha256':hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest()},
            'loading_route':['original_mxfp4','dequantized_bf16']+(['native_nf4_frozen_base']+(['native_lora'] if rank else []) if quantized else []),
            'source_quantization_config_class':'Mxfp4Config','source_dequantize_loading_attribute':True,
            'direct_bitsandbytes_source_override':False,'device':str(target),'compute_dtype':'torch.bfloat16',
            'attention_implementation':'eager','attention_note':'Native GPT-OSS attention sinks do not support generic SDPA on the pinned Transformers version.',
            'original_logical_parameter_count':original_parameters,'original_registered_parameter_bytes':original_bytes,
            'packed_projection_parameter_count':packed_parameters,'native_expert_banks':len(banks),
            'qlora':quantized,'rank':rank if quantized else None,'packed_conversion':None,'attention_conversion':None,
            'memory_before':before,'base_frozen_for_caller_policy':True}
    # Do not retain references to replaced bf16 expert banks when measuring the
    # final NF4 footprint or constructing the ordinary attention adapters.
    del banks
    if quantized:
        report['packed_conversion']=quantize_packed_experts(model,rank=rank,alpha=lora_alpha,
            dropout=lora_dropout,compute_dtype=torch.bfloat16,double_quant=True)
        report['attention_conversion']=quantize_gpt_oss_attention(model,rank=rank,alpha=lora_alpha,dropout=lora_dropout)
        adapters_enabled=stage=='SFT' or (stage=='A1' and recipe=='4q')
        # This function visits every NF4LoRALinear, including ordinary attention.
        set_packed_lora_trainable(model,enabled=adapters_enabled)
        unexpected=[name for name,p in model.named_parameters() if p.requires_grad and not ('lora_A' in name or 'lora_B' in name)]
        if unexpected:raise RuntimeError(f'Original parameters unexpectedly trainable after GPT NF4 conversion: {unexpected[:10]}')
        report['lora_enabled']=adapters_enabled and rank>0
        model.config.research_gpt_quantization={'route':'mxfp4_bf16_native_nf4_lora','rank':rank,
            'alpha':lora_alpha,'dropout':lora_dropout,'dense_scope':'attention_qkvo','packed_scope':'all_expert_gate_up_down'}
    else:report['lora_enabled']=False
    report['nf4_linear_modules']=sum(isinstance(module,NF4LoRALinear) for module in model.modules())
    report['lora_parameter_count']=sum(p.numel() for n,p in model.named_parameters() if 'lora_A' in n or 'lora_B' in n)
    report['trainable_parameter_count']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    report['registered_parameter_bytes_after']=sum(p.numel()*p.element_size() for p in model.parameters())
    report['nf4_side_state_tensor_bytes']=_nf4_state_bytes(model) if quantized else 0
    report['frozen_router_parameter_count']=sum(p.numel() for n,p in model.named_parameters() if '.router.' in n)
    report['memory_after']=_memory(target)
    report['memory_caveats']=['Allocator peak is process-global, not a reset operation-scoped measurement.',
        'Registered parameter bytes plus NF4 side tensors exclude activations, CUDA workspace and external allocations.',
        'The original GPT-OSS-20B checkpoint is not validated merely by a tiny randomly initialized smoke.']
    return model,report
