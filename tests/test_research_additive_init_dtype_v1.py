from types import SimpleNamespace
import copy
import pytest
import torch
from torch import nn
from src.model.research_quantization import NF4LoRALinear
from src.utils.research_additive_init_dtype_v1 import validation_view,PINS


def fixture():
    model=nn.Module();model.original=nn.Module();model.original.experts=nn.Module()
    expert=nn.Module();factor=NF4LoRALinear.__new__(NF4LoRALinear);nn.Module.__init__(factor)
    factor.rank=16;factor.lora_A=nn.Parameter(torch.randn(16,8));factor.lora_B=nn.Parameter(torch.zeros(12,16))
    expert.gate_up_proj=factor;model.original.experts.expert_modules=nn.ModuleList([expert])
    model.projector=nn.Linear(8,8)
    repo=next(iter(PINS));model.model_ref={'repo':repo,'revision':PINS[repo]}
    config=dict(stage='A1',model=repo,recipe='1b',seed=42,additive_execution={'branch':{'mode':'fixed'}})
    args=SimpleNamespace(**dict(config,stage='SFT'))
    state={n:t.clone().bfloat16() if 'lora_' in n else t.clone() for n,t in model.state_dict().items()}
    payload=dict(model=state,config=config,model_ref=model.model_ref,step=4,seen=256)
    return model,args,payload


def test_validation_widens_exact_values_but_native_load_gets_original_tensors():
    model,args,payload=fixture();old=copy.deepcopy(payload)
    view,names=validation_view(payload,model,args)
    assert len(names)==2 and view['config'] is payload['config']
    for name in names:
        assert payload['model'][name].dtype==torch.bfloat16
        assert view['model'][name].dtype==torch.float32
        assert torch.equal(view['model'][name].bfloat16(),payload['model'][name])
        assert torch.equal(payload['model'][name],old['model'][name])
    model.load_state_dict(payload['model'])
    for name in names:assert torch.equal(model.state_dict()[name],payload['model'][name].float())


@pytest.mark.parametrize('change',['shape','fp16','projector','nan','rank'])
def test_bridge_rejects_unrelated_or_lossy_mismatches(change):
    model,args,payload=fixture();name=next(n for n in payload['model'] if n.endswith('lora_A'))
    if change=='shape':payload['model'][name]=torch.zeros(4,8,dtype=torch.bfloat16)
    if change=='fp16':payload['model'][name]=payload['model'][name].half()
    if change=='projector':payload['model']['projector.weight']=payload['model']['projector.weight'].bfloat16()
    if change=='nan':payload['model'][name][0,0]=float('nan')
    if change=='rank':model.original.experts.expert_modules[0].gate_up_proj.rank=8
    with pytest.raises(ValueError):validation_view(payload,model,args)


def test_bridge_never_applies_to_other_stages_or_references():
    for key in ('stage','revision'):
        model,args,payload=fixture()
        if key=='stage':payload['config']['stage']='SFT'
        else:payload['model_ref']={'repo':args.model,'revision':'wrong'}
        view,names=validation_view(payload,model,args)
        assert view is payload and names==[]
