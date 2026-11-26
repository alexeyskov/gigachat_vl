"""Narrow future LFM checkpoint guards; frozen loaders are not modified.

A projector remains mandatory when A1/SFT freezes its parameters. Native
_validate_learned_state only requires currently trainable parameters and at
least one projector key, which otherwise allows a partially missing frozen P.
Call these guards before loading or freezing a newly reconstructed model.
"""
from __future__ import annotations

OWNER='lfm_complete_alignment_projector_v1'
LFM='LiquidAI/LFM2.5-8B-A1B'
PROJECTOR_KEYS=('projector.0.weight','projector.0.bias','projector.2.weight','projector.2.bias')


def validate_projector_tensors(state,*,feature_dim,hidden_size):
    """Check the complete existing two-layer FP32 MLP without copying weights."""
    import torch
    if type(feature_dim) is not int or feature_dim<1 or type(hidden_size) is not int or hidden_size<1:
        raise ValueError('Positive exact projector dimensions required')
    expected=dict(zip(PROJECTOR_KEYS,((hidden_size,feature_dim),(hidden_size,),(hidden_size,hidden_size),(hidden_size,))))
    actual={key for key in state if isinstance(key,str) and key.startswith('projector.')}
    if actual!=set(expected):raise ValueError(f'Complete alignment projector required: missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}')
    for key,shape in expected.items():
        value=state[key]
        if not torch.is_tensor(value) or tuple(value.shape)!=shape or value.dtype!=torch.float32 or value.layout!=torch.strided:
            raise ValueError('Projector tensor shape/dtype/layout differs: '+key)
        # Keep mmap/source allocations and dtype unchanged, with bounded scratch.
        for block in value.detach().reshape(-1).split(4*2**20):
            if not torch.isfinite(block).all():raise ValueError('Nonfinite alignment projector: '+key)
    return {'owner':OWNER,'projector_keys':list(PROJECTOR_KEYS),'feature_dim':feature_dim,'hidden_size':hidden_size,
            'projector_parameters':sum(v.numel() for k,v in state.items() if k in expected),'projector_dtype':'torch.float32',
            'weights_changed':False,'all_LLM_required_keys_checked':False}


def _dimensions(model):
    import torch
    from torch import nn
    config=getattr(getattr(model,'base',None),'config',None)
    if getattr(config,'model_type',None)!='lfm2_moe':raise ValueError('Expected an already reconstructed native LFM2-MoE model')
    projector=getattr(model,'projector',None);hidden=int(config.hidden_size)
    if (type(projector) is not nn.Sequential or len(projector)!=3 or type(projector[0]) is not nn.Linear or
            type(projector[1]) is not nn.GELU or type(projector[2]) is not nn.Linear or
            projector[0].out_features!=hidden or projector[2].in_features!=hidden or projector[2].out_features!=hidden or
            projector[0].bias is None or projector[2].bias is None or
            any(p.dtype!=torch.float32 for p in projector.parameters())):
        raise ValueError('Exact native FP32 Linear-GELU-Linear alignment projector required')
    return projector[0].in_features,hidden


def validate_lfm_initialization_projector(model,state):
    """Complement existing stage/source/precision guards; A0 may contain P only.

    This does not redefine which LLM keys are expected from a previous stage.
    The caller must retain native validate_initial_checkpoint/model_ref/cache
    and resume-config guards; those are not bypassed by this P-only check.
    """
    feature,hidden=_dimensions(model)
    return validate_projector_tensors(state,feature_dim=feature,hidden_size=hidden)


def validate_lfm_learned_state(model,state):
    """Complete P plus unchanged native all-learned-key/type/finite validation."""
    stage=getattr(getattr(model,'args',None),'stage',None)
    if stage not in ('A0','A1','SFT'):raise ValueError('Declared reconstructed training stage required before generation freeze')
    if stage!='A0' and not any(p.requires_grad for p in model.llm.parameters()):
        raise ValueError('Validate learned LFM tensors before freezing reconstructed training parameters')
    receipt=validate_lfm_initialization_projector(model,state)
    from src.eval.research_inference import _validate_learned_state
    _validate_learned_state(model,state)
    return dict(receipt,all_LLM_required_keys_checked=True,stage=stage)


def load_lfm_learned_state(model,state):
    """Drop-in future restore operation; no constructors or global patching."""
    receipt=validate_lfm_learned_state(model,state)
    result=model.load_state_dict(state,strict=False)
    if result.unexpected_keys:raise ValueError('Unexpected LFM checkpoint keys')
    return receipt
