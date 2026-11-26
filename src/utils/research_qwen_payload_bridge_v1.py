"""Strict compatibility for completed R2 payloads missing duplicate vision headers.

The exact six source-bound values remain in saved config. Reconstruct a shallow
validation view only; never rewrite checkpoints, tensors, labels or numerics.
The resource vision module is already a private clone of the frozen original.
"""
from __future__ import annotations
import contextlib,copy,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OWNER='budget18_completed_R2_derived_header_bridge_v1'
QWEN='Qwen/Qwen2.5-VL-7B-Instruct'
MODELS={'LiquidAI/LFM2.5-8B-A1B':'dd83f5cb02aa2938b66bba9a9998d38d0517c1f3',
        'ai-sage/GigaChat3.1-10B-A1.8B-bf16':'189fff27a1dee68473960c3d5bca53e0e07a3191'}


def definition():
    return dict(owner=OWNER,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='completed 4f SFT only; duplicate header reconstructed from exact saved config',
        checkpoint_bytes_changed=False,numerical_profile_changed=False)


def validation_view(payload,binding,derived):
    contract=derived.config_contract(binding)
    present=set(contract)&set(payload)
    if present:
        # Existing or partially present metadata must pass the original strict
        # validator. Never fill a partially corrupted header or override values.
        return payload,False
    config=payload.get('config',{})
    if config.get('vision')!=QWEN:return payload,False
    from src.utils.research_completed_r2_v1 import definition as completed_definition
    expected=completed_definition()
    conditions=(config.get('model') in MODELS and config.get('recipe')=='4f' and config.get('stage')=='SFT'
        and config.get('seed')==42 and config.get('epochs')==1 and config.get('effective_batch_size')==64
        and payload.get('step')==56 and payload.get('seen')==3529
        and payload.get('training_completed') is True and payload.get('resumable_optimizer') is False
        and config.get('completed_r2_execution_policy')==expected
        and payload.get('checkpoint_finalization',{}).get('owner')==expected['owner']
        and payload.get('model_ref')=={'repo':config.get('model'),'revision':MODELS.get(config.get('model'))})
    if not conditions:raise ValueError('Missing derived header is not an exact completed R2 checkpoint')
    if any(config.get(k)!=v for k,v in contract.items()):raise ValueError('Saved config does not prove the exact derived vision contract')
    view=dict(payload)
    for key,value in contract.items():view[key]=copy.deepcopy(value)
    return view,True


@contextlib.contextmanager
def patched_private_validation():
    from src.utils.research_resource_vision_v3 import derived
    native=derived.validate_payload_binding
    def validate(payload,binding):
        view,bridged=validation_view(payload,binding,derived)
        result=native(view,binding)
        if bridged:
            print(json.dumps(dict(event='validated_completed_R2_derived_header_bridge',
                model=payload['config']['model'],step=payload['step'],bridge=definition())),flush=True)
        return result
    derived.validate_payload_binding=validate
    try:yield
    finally:derived.validate_payload_binding=native


def invoke(api,kind,argv=None):
    with patched_private_validation():
        return getattr(api,kind)(argv)
