"""Actual tiny native LFM/PEFT/restore proof; no CUDA or NF4 substitution."""
import copy,io
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM,Lfm2MoeConfig
from peft import LoraConfig,get_peft_model
from peft.tuners.lora.layer import Linear as PeftLinear
from src.model.research_moe import install_research_moe,configure_research_trainability,set_modality_mask
from src.utils.research_lfm_checkpoint_requirements_v1 import (
    validate_projector_tensors,validate_lfm_initialization_projector,validate_lfm_learned_state,load_lfm_learned_state,
)
from src.eval.research_inference import _validate_learned_state,freeze_for_generation


def native():
    cfg=Lfm2MoeConfig(vocab_size=32,hidden_size=16,intermediate_size=24,moe_intermediate_size=12,num_hidden_layers=3,
        num_dense_layers=1,num_experts=4,num_experts_per_tok=2,num_attention_heads=2,num_key_value_heads=1,
        layer_types=['conv','full_attention','conv'],max_position_embeddings=128)
    return AutoModelForCausalLM.from_config(cfg,attn_implementation='eager',experts_implementation='eager')


def ordinary_targets(model):
    # Exact current production runner predicate; explicit paths, no suffix fix.
    return [n for n,m in model.named_modules() if isinstance(m,nn.Linear) and n!='lm_head'
            and not n.endswith(('.gate','.router')) and '.experts.' not in n]


def peft(model):
    targets=ordinary_targets(model)
    return get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,lora_dropout=.05,bias='none',target_modules=targets))


class Research(nn.Module):
    def __init__(self,stage='SFT',recipe='4f'):
        super().__init__();self.args=SimpleNamespace(stage=stage,recipe=recipe,model='LiquidAI/LFM2.5-8B-A1B')
        self.llm=native();self.llm.requires_grad_(stage!='A0');self.base=self.llm
        self.projector=nn.Sequential(nn.Linear(5,16),nn.GELU(),nn.Linear(16,16));self.projector.requires_grad_(stage=='A0')
        if stage!='A0' and recipe[0] in '123':
            if stage=='A1':self.llm.requires_grad_(False)
            install_research_moe(self.base,int(recipe[0]),recipe[1],added_experts=2)
            configure_research_trainability(self.base,stage=stage)
    def state_delta(self):
        return {n:p.detach().clone() for n,p in self.named_parameters() if p.requires_grad or n.startswith('projector.')}
    def inputs(self):
        text=self.base.get_input_embeddings()(torch.tensor([[1,4,5,6,2]]));image=self.projector(torch.arange(10).reshape(1,2,5).float()/7)
        values=torch.cat((text[:,:1],image,text[:,1:]),dim=1);mask=torch.tensor([[False,True,True,False,False,False,False]])
        set_modality_mask(self.base,mask,torch.ones_like(mask));return values


def test_actual_public_PEFT_exact_LFM_scope_no_Giga_workaround_or_packed_double_wrap():
    model=native();targets=ordinary_targets(model);packed={n:m for n,m in model.named_modules() if m.__class__.__name__=='Lfm2MoeExperts'}
    routers={n:m for n,m in model.named_modules() if m.__class__.__name__=='Lfm2MoeTopKRouter'};head=model.lm_head
    wrapped=peft(model)
    assert wrapped.peft_config['default'].target_parameters in (None,[])
    assert set(wrapped.base_model.targeted_module_names)==set(targets) and len(targets)==11
    assert set(n for n,m in model.named_modules() if isinstance(m,PeftLinear))==set(targets)
    assert all(model.get_submodule(n) is m for n,m in {**packed,**routers}.items()) and model.lm_head is head
    assert all('lora_' in n for n,p in wrapped.named_parameters() if p.requires_grad)
    for name in targets:
        layer=model.get_submodule(name);assert layer.r['default']==16 and layer.lora_alpha['default']==32 and layer.lora_dropout['default'].p==.05
    ids=torch.tensor([[1,4,5,6,2],[1,5,6,7,2]]);before={n:p.detach().clone() for n,p in wrapped.named_parameters()}
    optimizer=torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=.01)
    loss=wrapped(input_ids=ids,labels=ids,use_cache=False).loss;loss.backward();optimizer.step()
    assert torch.isfinite(loss) and any(not torch.equal(p,before[n]) for n,p in wrapped.named_parameters() if 'lora_B' in n)
    assert all(torch.equal(p,before[n]) and p.grad is None for n,p in wrapped.named_parameters() if 'lora_' not in n)


@pytest.mark.parametrize('stage',['A1','SFT'])
def test_real_frozen_projector_partial_state_bug_reproduced_then_guard_refuses(stage):
    model=Research(stage);state=model.state_delta();missing='projector.0.bias';state.pop(missing)
    # This is the actual frozen loader's existing gap, not a mocked predicate.
    _validate_learned_state(model,state)
    with pytest.raises(ValueError,match='Complete alignment projector'):load_lfm_learned_state(model,state)


@pytest.mark.parametrize('key',['projector.0.weight','projector.0.bias','projector.2.weight','projector.2.bias'])
def test_every_frozen_projector_tensor_is_mandatory_even_with_complete_native_LLM(key):
    model=Research();state=model.state_delta();state.pop(key)
    with pytest.raises(ValueError,match='Complete alignment projector'):validate_lfm_learned_state(model,state)


@pytest.mark.parametrize('change',['shape','dtype','nonfinite','extra'])
def test_wrong_projector_shape_precision_finiteness_and_unknown_P_refuse(change):
    model=Research();state=model.state_delta()
    if change=='shape':state['projector.0.weight']=torch.zeros(17,5)
    elif change=='dtype':state['projector.0.weight']=state['projector.0.weight'].bfloat16()
    elif change=='nonfinite':state['projector.0.weight'][0,0]=float('nan')
    else:state['projector.unexpected']=torch.ones(1)
    with pytest.raises(ValueError):validate_lfm_learned_state(model,state)


def test_A0_P_only_initialization_is_valid_but_not_mislabelled_complete_SFT():
    A0=Research('A0');state=A0.state_delta();assert set(state)=={'projector.0.weight','projector.0.bias','projector.2.weight','projector.2.bias'}
    assert validate_lfm_learned_state(A0,state)['all_LLM_required_keys_checked']
    SFT=Research('SFT');assert not validate_lfm_initialization_projector(SFT,state)['all_LLM_required_keys_checked']
    with pytest.raises(ValueError,match='omits learned'):validate_lfm_learned_state(SFT,state)
    partial=dict(state);partial.pop('projector.2.bias')
    with pytest.raises(ValueError,match='Complete alignment projector'):validate_lfm_initialization_projector(SFT,partial)


@pytest.mark.parametrize('recipe',['4f','2a','2b','2c','3a','3b','3c'])
def test_real_native_full_F32_EplusK_backward_exact_delta_restore_and_generation(recipe):
    torch.manual_seed(71);model=Research('SFT',recipe);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    inputs=model.inputs();hidden=model.base.model(inputs_embeds=inputs,use_cache=False,return_dict=True).last_hidden_state
    loss=nn.functional.cross_entropy(model.base.lm_head(hidden[:,3:-1]).reshape(-1,32),torch.tensor([5,6,2]));loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in model.llm.parameters())
    assert all(p.grad is None for p in model.projector.parameters());optimizer.step()
    state=model.state_delta();out=io.BytesIO();torch.save(state,out);state=torch.load(io.BytesIO(out.getvalue()),weights_only=False,map_location='cpu')
    again=Research('SFT',recipe);receipt=load_lfm_learned_state(again,state);assert receipt['all_LLM_required_keys_checked']
    assert all(torch.equal(p,dict(model.named_parameters())[n]) for n,p in again.named_parameters())
    model.eval();again.eval();freeze_for_generation(model);freeze_for_generation(again)
    for item in (model,again):item.base.generation_config.eos_token_id=None
    with torch.no_grad():
        a=model.base.generate(inputs_embeds=model.inputs(),max_new_tokens=3,do_sample=False,use_cache=True)
        b=again.base.generate(inputs_embeds=again.inputs(),max_new_tokens=3,do_sample=False,use_cache=True)
    assert torch.equal(a,b)


def test_missing_LLM_state_and_prefrozen_SFT_factory_refuse():
    model=Research();state=model.state_delta();state.pop(next(n for n in state if n.startswith('llm.')))
    with pytest.raises(ValueError,match='omits learned'):validate_lfm_learned_state(model,state)
    model.requires_grad_(False)
    with pytest.raises(ValueError,match='before freezing'):validate_lfm_learned_state(model,model.state_delta())


def test_frozen_stock_Small_profile_refuses_LFM_and_legacy_checkpoint_relabelling():
    from src.utils import research_stock_determinism as stock
    with pytest.raises(ValueError,match='SmallThinker only'):stock.validate_config({'model':'LiquidAI/LFM2.5-8B-A1B','numerical_profile':stock.definition()})
    with pytest.raises(ValueError,match='Missing, legacy'):stock.validate_config({'model':stock.SMALL})
    assert not torch.cuda.is_initialized()
