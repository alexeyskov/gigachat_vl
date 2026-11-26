"""Native auxiliary-loss parity and explicit eligibility/provenance contracts."""

from types import SimpleNamespace

import pytest
import torch

from src.model.research_auxiliary import (
    RouterLayerState,eligible_switch_balance_loss,native_auxiliary_contract,
    native_router_auxiliary_loss,router_state_from_logits,
)


def test_official_gpt_coefficient_is_inactive_without_router_output_flag():
    config=SimpleNamespace(model_type='gpt_oss',router_aux_loss_coef=.9,output_router_logits=False)
    contract=native_auxiliary_contract(config)
    assert contract['native_coefficient']==.9 and not contract['enabled']
    assert native_router_auxiliary_loss(config,[])['loss']==0
    assert native_auxiliary_contract(config,output_router_logits=True)['enabled']
    with pytest.raises(RuntimeError,match='no differentiable'):
        native_router_auxiliary_loss(config,[],output_router_logits=True)


@pytest.mark.parametrize('kind',['smallthinker','lfm2_moe','deepseek_v3'])
def test_no_invented_auxiliary_or_z_penalty_for_backbones_without_one(kind):
    contract=native_auxiliary_contract(SimpleNamespace(model_type=kind))
    assert contract['native_coefficient']==0 and not contract['enabled']
    assert contract['native_z_loss_coefficient'] is None


def test_unsupported_declared_penalty_is_not_silently_discarded():
    config=SimpleNamespace(model_type='gpt_oss',router_z_loss_coef=.01)
    with pytest.raises(NotImplementedError,match='Unimplemented declared'):
        native_router_auxiliary_loss(config,[])


def test_free_routing_loss_and_logit_gradients_match_native_gpt_with_padding():
    from transformers.models.gpt_oss.modeling_gpt_oss import load_balancing_loss_func
    torch.manual_seed(126)
    logits=tuple(torch.randn(10,4,requires_grad=True) for _ in range(3))
    mask=torch.tensor([[1,1,1,1,1],[1,1,0,0,0]])
    states=[router_state_from_logits(layer,2,token_mask=mask) for layer in logits]
    actual=eligible_switch_balance_loss(states)
    expected=load_balancing_loss_func(logits,num_experts=4,top_k=2,attention_mask=mask)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    grads=torch.autograd.grad(actual,logits,retain_graph=True)
    native_grads=torch.autograd.grad(expected,logits)
    for value,reference in zip(grads,native_grads):
        torch.testing.assert_close(value,reference,atol=1e-7,rtol=1e-5)
        assert value[7:].abs().sum()==0


def test_hard_separation_does_not_balance_towards_forbidden_experts():
    logits=torch.tensor([[0.,0.,0.,0.,1000.,1000.],
                         [1000.,1000.,1000.,1000.,0.,0.]],requires_grad=True)
    eligible=torch.tensor([[1,1,1,1,0,0],[0,0,0,0,1,1]],dtype=torch.bool)
    state=router_state_from_logits(logits,4,eligibility=eligible)
    loss=eligible_switch_balance_loss([state])
    # Uniform text routing has native k=4 baseline, uniform visual routing k=2.
    torch.testing.assert_close(loss,torch.tensor(3.))
    loss.backward()
    assert logits.grad[~eligible].abs().sum()==0
    assert torch.isfinite(logits.grad).all()


def test_router_loss_explicit_coefficient_and_missing_eligibility_failures():
    logits=torch.randn(5,4,requires_grad=True)
    state=router_state_from_logits(logits,2)
    config=SimpleNamespace(model_type='gpt_oss',router_aux_loss_coef=.9,output_router_logits=True)
    result=native_router_auxiliary_loss(config,[state])
    torch.testing.assert_close(result['loss'],.9*result['balance_loss'])
    forbidden=torch.zeros_like(logits,dtype=torch.bool)
    with pytest.raises(ValueError,match='no eligible expert'):
        router_state_from_logits(logits,2,eligibility=forbidden)


def test_empty_padding_is_zero_and_does_not_create_nans():
    logits=torch.randn(5,4,requires_grad=True)
    state=router_state_from_logits(logits,2,token_mask=torch.zeros(5,dtype=torch.bool),
                                   eligibility=torch.zeros_like(logits,dtype=torch.bool))
    loss=eligible_switch_balance_loss([state])
    loss.backward()
    assert loss==0 and torch.isfinite(logits.grad).all()
