"""Lossless read bridge for frozen packed LoRA BF16 A1 -> trainable FP32 SFT.

The native constructor stores frozen A1 factors in BF16, and promotes trainable
SFT factors to FP32. Only validate a widened view; loading still receives the
original checkpoint tensors. No checkpoint/config/optimizer/math is rewritten.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import re

PINS={'LiquidAI/LFM2.5-8B-A1B':'dd83f5cb02aa2938b66bba9a9998d38d0517c1f3',
      'ai-sage/GigaChat3.1-10B-A1.8B-bf16':'189fff27a1dee68473960c3d5bca53e0e07a3191'}
PATTERN=re.compile(r'(?:^|\.)original\.experts\.expert_modules\.\d+\.(?:gate_up_proj|down_proj)\.lora_[AB]$')


def definition():
    return dict(name='frozen_packed_A1_bf16_to_SFT_fp32_validation_v1',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint_bytes_changed=False,actual_load_uses_original_payload=True,
        operation='exact BF16 to FP32 widening of packed rank16 LoRA factors only')


def validation_view(payload,model,args):
    import torch
    from src.model.research_quantization import NF4LoRALinear
    config=payload.get('config',{})
    eligible=(args.stage=='SFT' and config.get('stage')=='A1'
        and config.get('model')==getattr(args,'model',None) in PINS
        and config.get('recipe')==getattr(args,'recipe',None) in ('1b','1c')
        and payload.get('step')==4 and payload.get('seen')==256
        and config.get('seed')==getattr(args,'seed',None)==42
        and config.get('additive_execution')==getattr(args,'additive_execution',None)
        and config.get('additive_execution',{}).get('branch',{}).get('mode')=='fixed'
        and payload.get('model_ref')==getattr(model,'model_ref',None)=={'repo':args.model,'revision':PINS.get(args.model)})
    if not eligible:return payload,[]
    expected=model.state_dict();converted={};names=[]
    for name,tensor in payload['model'].items():
        target=expected.get(name)
        if target is None or tensor.dtype==target.dtype:continue
        if (not PATTERN.search(name) or tensor.dtype!=torch.bfloat16 or target.dtype!=torch.float32
            or tensor.shape!=target.shape or tensor.ndim!=2):
            raise ValueError('Unsupported initialization dtype/shape mismatch: '+name)
        module=model.get_submodule(name.rsplit('.',1)[0])
        if not isinstance(module,NF4LoRALinear) or module.rank!=16 or not model.get_parameter(name).requires_grad:
            raise ValueError('Widening requires an actual trainable rank16 packed NF4 LoRA factor')
        if not torch.isfinite(tensor).all():raise ValueError('Nonfinite frozen LoRA factor')
        converted[name]=tensor.float();names.append(name)
    if not names:return payload,[]
    view=dict(payload);view['model']=dict(payload['model'],**converted)
    return view,names


@contextmanager
def patched_runner():
    from src.utils import research_additive_runner_v1 as backend
    factory=backend.runner
    def runner(*a,**k):
        module,inference,selection=factory(*a,**k);native=module.validate_initial_checkpoint
        def validate(payload,model,args):
            view,names=validation_view(payload,model,args)
            result=native(view,model,args)
            if names:
                from src.utils.research_storage import _atomic_json
                receipt=dict(bridge=definition(),converted_validation_tensors=names,count=len(names),
                    source_model_ref=payload['model_ref'],source_step=payload['step'],source_seen=payload['seen'],
                    native_initial_validator_passed=True,actual_load_unchanged=True)
                _atomic_json(Path(args.output)/'initialization_dtype_bridge_v1.json',receipt)
                print(__import__('json').dumps(dict(event='packed_LoRA_A1_SFT_dtype_bridge',count=len(names),model=args.model)),flush=True)
            return result
        module.validate_initial_checkpoint=validate
        return module,inference,selection
    backend.runner=runner
    try:yield
    finally:backend.runner=factory
