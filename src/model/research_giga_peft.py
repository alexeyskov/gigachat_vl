"""Isolated, exact ordinary-linear PEFT scope for future GigaChat training.

PEFT 0.21.2 incorrectly converts explicitly resolved dense/shared FFN targets
into packed expert factors when given a Transformers 5 DeepSeek model. The
public get_peft_model API works correctly on an ordinary nn.Module tree. We
inject into a tree containing only the exact existing 2D linears, then attach
its actual PEFT layers and native PeftModelForCausalLM to the original model.

This changes neither the native configuration nor any global PEFT registry.
The final native wrapper rebind is version-guarded and round-trip tested. This
module is deliberately not wired into the current training/inference runner.
Ordinary nn.Linear and materialized CUDA bitsandbytes NF4 Linear4bit are
supported. The quantized path preserves native dispatch metadata so PEFT uses
its dedicated bnb Linear4bit adapter, never its generic dense adapter.
"""
from __future__ import annotations

from contextlib import nullcontext
from importlib.metadata import version
import random
import weakref

import numpy as np
import torch
from torch import nn

TESTED_VERSIONS={'peft':'0.21.2','transformers':'5.15.0'}
TESTED_BNB_VERSION='0.50.2'
PROTOCOL='giga_exact_ordinary_peft_scope_v1'


def _check_versions():
    actual={name:version(name) for name in TESTED_VERSIONS}
    if actual!=TESTED_VERSIONS:
        raise NotImplementedError(f'Giga scoped PEFT wrapper is validated only for {TESTED_VERSIONS}; installed {actual}')
    return actual


def resolve_giga_ordinary_targets(model):
    """Use the runner's explicit ordinary attention/dense/shared linear scope."""
    if getattr(getattr(model,'config',None),'model_type',None)!='deepseek_v3':
        raise ValueError('Expected native DeepseekV3 GigaChat causal LM')
    return [name for name,module in model.named_modules()
            if isinstance(module,nn.Linear) and name!='lm_head'
            and not name.endswith(('.gate','.router')) and '.experts.' not in name]


class _ScopeConfig(dict):
    """Expose quantization attributes without a Transformer model_type."""
    def __getattr__(self,name):
        try:return self[name]
        except KeyError as error:raise AttributeError(name) from error

    def to_dict(self):return dict(self)


class _OrdinaryLinearScope(nn.Module):
    """Registration tree of existing linears; it never owns a packed bank."""
    def __init__(self,model,targets):
        super().__init__()
        # A generic nn.Module config has no Transformer model_type to remap.
        self.config=_ScopeConfig()
        if hasattr(model.config,'quantization_config'):
            self.config['quantization_config']=model.config.quantization_config
        # PEFT's native bnb dispatcher explicitly tests is_loaded_in_4bit.
        # Copy quantizer metadata, not the Transformer config/model_type.
        for attribute in ('is_loaded_in_4bit','is_loaded_in_8bit','is_quantized',
                          'quantization_method','hf_device_map','hf_quantizer'):
            if hasattr(model,attribute):setattr(self,attribute,getattr(model,attribute))
        self.name_or_path=getattr(model,'name_or_path','')
        self.is_gradient_checkpointing=False
        self._native_ref=weakref.ref(model)
        for name,module in targets.items():
            parent=self
            parts=name.split('.')
            for part in parts[:-1]:
                if part not in parent._modules:parent.add_module(part,nn.Module())
                parent=parent._modules[part]
            parent.add_module(parts[-1],module)

    def prepare_inputs_for_generation(self,*args,**kwargs):
        return self._native_ref().prepare_inputs_for_generation(*args,**kwargs)


def _rng_snapshot():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _rng_restore(state):
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
    if state[3] is not None:torch.cuda.set_rng_state_all(state[3])


def get_giga_ordinary_peft_model(model,target_modules=None,*,adapter_name='default',autocast_adapter_dtype=True):
    """Return a native PeftModelForCausalLM and an exact-scope receipt.

    Fixed research contract: rank=16, alpha=32, dropout=.05, bias='none'.
    Every ordinary eligible linear must be explicitly present; no suffix, regex
    or packed target expansion is accepted. Reapplication and unknown versions
    fail before mutation. Existing packed LoRA A/B flags are preserved; all
    original base parameters are frozen, as in QLoRA. Install before enabling
    gradient checkpointing, matching the research runner's construction order.
    Packed modules are neither registered in the injection tree nor replaced.
    """
    versions=_check_versions()
    from peft import LoraConfig,PeftModelForCausalLM,get_peft_model
    from peft.tuners.tuners_utils import BaseTunerLayer
    if any(isinstance(module,BaseTunerLayer) for module in model.modules()) or hasattr(model,'peft_config'):
        raise ValueError('Ordinary PEFT adapters are already installed')
    if getattr(model.config,'research_moe',None) is not None:
        raise ValueError('Install ordinary PEFT before E+K research-router expansion')
    if getattr(model,'is_gradient_checkpointing',False):
        raise ValueError('Install ordinary PEFT before enabling gradient checkpointing')
    expected=resolve_giga_ordinary_targets(model)
    if not expected:raise ValueError('No eligible ordinary GigaChat linear modules')
    if isinstance(target_modules,str):raise ValueError('Targets must be explicit module paths, not a regex or suffix')
    supplied=expected if target_modules is None else list(target_modules)
    if any(not isinstance(name,str) or not name for name in supplied) or len(supplied)!=len(set(supplied)):
        raise ValueError('Targets must be unique, exact module paths')
    if set(supplied)!=set(expected):
        raise ValueError(f'Ordinary scope mismatch: missing={sorted(set(expected)-set(supplied))}, extra={sorted(set(supplied)-set(expected))}')
    if not isinstance(adapter_name,str) or not adapter_name or '.' in adapter_name or 'lora_' in adapter_name:
        raise ValueError('Invalid native PEFT adapter name')
    targets={name:model.get_submodule(name) for name in expected}
    quantized={};unsupported={}
    for name,module in targets.items():
        if type(module) is nn.Linear:continue
        if type(module).__module__.startswith('bitsandbytes.'):
            if version('bitsandbytes')!=TESTED_BNB_VERSION:
                raise NotImplementedError(f'Giga bnb dispatcher validated only for bitsandbytes {TESTED_BNB_VERSION}')
            import bitsandbytes as bnb
            if type(module) is bnb.nn.Linear4bit:
                quantized[name]=module
                continue
        unsupported[name]=type(module).__name__
    if unsupported:
        raise NotImplementedError(f'Quantized/custom ordinary linears require separate actual CUDA validation: {unsupported}')
    if quantized:
        if not getattr(model,'is_loaded_in_4bit',False):
            raise ValueError('Materialized bnb ordinary linears require native is_loaded_in_4bit metadata')
        qconfig=getattr(model.config,'quantization_config',None)
        values=qconfig.to_dict() if hasattr(qconfig,'to_dict') else qconfig
        if not isinstance(values,dict) or not values.get('load_in_4bit'):
            raise ValueError('Materialized bnb ordinary linears require matching config.quantization_config')
        if not autocast_adapter_dtype:
            raise ValueError('The research NF4 profile requires FP32 adapter masters (autocast_adapter_dtype=True)')
        for name,module in quantized.items():
            if (module.weight.device.type!='cuda' or module.weight.quant_state is None
                or not getattr(module.weight,'bnb_quantized',False)):
                raise ValueError(f'{name}: NF4 weights must be materialized and quantized on CUDA')
            if (module.compute_dtype!=torch.bfloat16 or module.weight.quant_type!='nf4'
                or not module.weight.compress_statistics):
                raise ValueError(f'{name}: research NF4 requires BF16 compute and double quantization')
    flags={id(parameter):(parameter,parameter.requires_grad) for parameter in model.parameters()}
    modes=[(module,module.training) for module in model.modules()]
    packed_flags={id(parameter):(parameter,parameter.requires_grad) for name,parameter in model.named_parameters()
                  if '.experts.' in name and name.rsplit('.',1)[-1] in ('lora_A','lora_B')}
    root_training=model.training
    rng=_rng_snapshot()
    try:
        model.requires_grad_(False)
        scope=_OrdinaryLinearScope(model,targets)
        config=LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,lora_dropout=.05,
                          bias='none',target_modules=expected)
        context=torch.device('meta') if all(module.weight.device.type=='meta' for module in targets.values()) else nullcontext()
        with context:
            wrapped=get_peft_model(scope,config,adapter_name=adapter_name,
                                   autocast_adapter_dtype=autocast_adapter_dtype)
        if type(wrapped) is not PeftModelForCausalLM or wrapped.base_model.model is not scope:
            raise RuntimeError('Native PEFT wrapper contract changed')
        if set(wrapped.base_model.targeted_module_names)!=set(expected) or wrapped.base_model.targeted_parameter_names:
            raise RuntimeError('PEFT scope changed during ordinary injection')
        if config.target_parameters or config.r!=16 or config.lora_alpha!=32 or config.lora_dropout!=.05:
            raise RuntimeError('PEFT changed the fixed ordinary-adapter contract')
        adapted={name:scope.get_submodule(name) for name in expected}
        for name,module in adapted.items():
            if not isinstance(module,BaseTunerLayer) or module.get_base_layer() is not targets[name]:
                raise RuntimeError('PEFT did not retain the original ordinary base linear')
            if (module.r[adapter_name]!=16 or module.lora_alpha[adapter_name]!=32
                or module.lora_dropout[adapter_name].p!=.05 or module.scaling[adapter_name]!=2.):
                raise RuntimeError('PEFT changed a resolved ordinary layer rank/alpha/dropout/scaling')
            if name in quantized:
                from peft.tuners.lora.bnb import Linear4bit as PeftLinear4bit
                if type(module) is not PeftLinear4bit:
                    raise RuntimeError('Materialized NF4 base was dispatched through a generic dense PEFT adapter')
                if any(parameter.dtype!=torch.float32 for parameter in
                       (module.lora_A[adapter_name].weight,module.lora_B[adapter_name].weight)):
                    raise RuntimeError('NF4 ordinary adapter masters must remain FP32')
        # The real model has never been a target of PEFT's v4/v5 conversion.
        for name,module in adapted.items():
            parent,_,leaf=name.rpartition('.')
            setattr(model.get_submodule(parent),leaf,module)
            module.train(targets[name].training)
        wrapped.base_model.model=model
        wrapped.base_model_prepare_inputs_for_generation=model.prepare_inputs_for_generation
        model.peft_config=wrapped.peft_config
        wrapped.training=root_training;wrapped.base_model.training=root_training
        for module,training in modes:module.training=training
        for parameter,requires_grad in packed_flags.values():parameter.requires_grad_(requires_grad)
        shapes=[]
        for name,module in adapted.items():
            a=module.lora_A[adapter_name].weight;b=module.lora_B[adapter_name].weight
            shapes.append({'name':name,'in_features':module.in_features,'out_features':module.out_features,
                           'lora_A_shape':list(a.shape),'lora_B_shape':list(b.shape),
                           'adapter_class':f'{type(module).__module__}.{type(module).__name__}',
                           'adapter_dtype':str(a.dtype),'base_dtype':str(module.get_base_layer().weight.dtype),
                           'base_compute_dtype':str(getattr(module.get_base_layer(),'compute_dtype',None)),
                           'adapter_parameters':a.numel()+b.numel()})
        receipt={'protocol':PROTOCOL,'versions':versions,'rank':16,'alpha':32,'dropout':.05,'bias':'none',
                 'adapter_name':adapter_name,'autocast_adapter_dtype':autocast_adapter_dtype,
                 'targets':expected,'target_shapes':shapes,
                 'ordinary_lora_parameters':sum(row['adapter_parameters'] for row in shapes),
                 'packed_modules_replaced':False,'packed_parameters_targeted_by_peft':[],
                 'existing_packed_lora_flags_preserved':len(packed_flags),
                 'state_key_prefix':'base_model.model.',
                 'configuration_model_type_changed':False,'global_monkeypatches':False,
                 'dispatch_metadata':{
                     'is_loaded_in_4bit':bool(getattr(scope,'is_loaded_in_4bit',False)),
                     'is_loaded_in_8bit':bool(getattr(scope,'is_loaded_in_8bit',False)),
                     'quantization_method':str(getattr(scope,'quantization_method',None)),
                     'config_quantization_config_preserved':getattr(scope.config,'quantization_config',None) is getattr(model.config,'quantization_config',None)},
                 'quantized_ordinary_targets':list(quantized),
                 'native_bnb_4bit_dispatch_validated':bool(quantized),
                 'bitsandbytes_version':version('bitsandbytes') if quantized else None,
                 'full_model_memory_capacity_validated':False,
                 'restore_protocol':'rebuild using this helper, then native load_state_dict with canonical full keys; generic PEFT.from_pretrained conversion is not validated'}
        wrapped._research_giga_scope=receipt
        return wrapped,receipt
    except Exception:
        for name,module in targets.items():
            parent,_,leaf=name.rpartition('.')
            setattr(model.get_submodule(parent),leaf,module)
        if hasattr(model,'peft_config'):delattr(model,'peft_config')
        for parameter,requires_grad in flags.values():parameter.requires_grad_(requires_grad)
        for module,training in modes:module.training=training
        _rng_restore(rng)
        raise


def ordinary_adapter_state_dict(wrapped):
    """Canonical native full-state keys, excluding frozen/packed tensors."""
    receipt=getattr(wrapped,'_research_giga_scope',None)
    if not receipt or receipt.get('protocol')!=PROTOCOL:raise ValueError('No validated Giga ordinary scope receipt')
    adapter=receipt['adapter_name']
    names={f'base_model.model.{target}.lora_{factor}.{adapter}.weight'
           for target in receipt['targets'] for factor in ('A','B')}
    parameters=dict(wrapped.named_parameters())
    if not names<=parameters.keys():raise ValueError('Ordinary adapter keys changed')
    return {name:parameters[name].detach().cpu().clone() for name in sorted(names)}


def load_ordinary_adapter_state_dict(wrapped,state):
    """Exact adapter restore without invoking Transformers-v4/v5 conversion."""
    expected=ordinary_adapter_state_dict(wrapped)
    if set(state)!=set(expected):raise ValueError('Ordinary adapter checkpoint keys differ from resolved scope')
    for name,value in state.items():
        if (not isinstance(value,torch.Tensor) or value.shape!=expected[name].shape
            or value.dtype!=expected[name].dtype or not torch.isfinite(value).all()):
            raise ValueError(f'Invalid ordinary adapter tensor: {name}')
    result=wrapped.load_state_dict(state,strict=False)
    if result.unexpected_keys:raise ValueError(result.unexpected_keys)
    return {'loaded_tensors':len(state),'loaded_parameters':sum(value.numel() for value in state.values())}
