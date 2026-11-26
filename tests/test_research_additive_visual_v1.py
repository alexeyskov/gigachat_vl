import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.model.research_additive_visual_v1 import (
    AdditiveVisualBank, AdditiveVisualMoE, VisualBranchConfig, generation)


def bank(mode="routed", masks=True):
    torch.manual_seed(42)
    value = AdditiveVisualBank(12, VisualBranchConfig(mode=mode, width=8,
        heads=2, blocks=4 if mode == "routed" else 2, top_k=2, learned_masks=masks))
    # Exercise the actual branch: zero-init up must not conceal a broken mask.
    nn.init.normal_(value.up.weight, std=.1)
    return value


@pytest.mark.parametrize("mode", ["fixed", "routed"])
def test_no_future_answer_leakage_and_exact_text_only_bypass(mode):
    value = bank(mode).eval()
    x = torch.randn(2, 7, 12)
    visual = torch.tensor([[0,0,1,1,0,0,0], [0,0,0,0,0,0,0]], dtype=torch.bool)
    valid = torch.ones_like(visual)
    first = value(x, visual, valid)
    changed = x.clone(); changed[:, 5:] = torch.randn_like(changed[:, 5:])*100
    second = value(changed, visual, valid)
    torch.testing.assert_close(first[:, :5], second[:, :5], rtol=0, atol=0)
    assert torch.equal(first[1], torch.zeros_like(first[1]))
    assert torch.equal(first[0, :2], torch.zeros_like(first[0, :2]))
    assert first[0, 2:].abs().sum() > 0
    # Adding an image in the future cannot activate a previous text token.
    no_image = torch.zeros_like(visual)
    torch.testing.assert_close(value(x, no_image, valid)[:, :2], first[:, :2], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["fixed", "routed"])
def test_cached_decode_matches_full_causal_prefix_and_clears(mode):
    value = bank(mode).eval()
    x = torch.randn(2, 7, 12)
    visual = torch.tensor([[0,1,1,0,0,0,0], [0,0,0,0,0,0,0]], dtype=torch.bool)
    valid = torch.ones_like(visual); valid[1, 2] = False
    with torch.no_grad():
        expected = value(x, visual, valid)
        value.cache_enabled = True
        outputs = [value(x[:, :4], visual[:, :4], valid[:, :4])]
        for i in range(4, 7):
            outputs.append(value(x[:, i:i+1], visual[:, i:i+1], valid[:, i:i+1]))
        actual = torch.cat(outputs, 1)
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=2e-7)
    assert actual[0, -1].abs().sum() > 0
    assert not actual[1].any()
    value.clear_cache(); value.cache_enabled = False
    assert value._context is None


@pytest.mark.parametrize("mode", ["fixed", "routed"])
def test_visual_dispatch_guaranteed_and_masks_and_router_learn(mode):
    value = bank(mode)
    x = torch.randn(1, 6, 12, requires_grad=True)
    visual = torch.tensor([[0,1,1,0,0,0]], dtype=torch.bool)
    y = value(x, visual, torch.ones_like(visual))
    y.square().sum().backward()
    count = value.last_dispatch["counts"]
    assert count[1].sum() == 4  # exactly two extra blocks for each image token
    assert count[0].sum() == 6  # text after image also has an extra branch
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert value.modality_gate_logits.grad.abs().sum() > 0
    assert value.blocks[0].q.weight.grad is not None or mode == "routed"
    if mode == "routed":
        assert value.router[-1].weight.grad.abs().sum() > 0


class NativeMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(12, 5, bias=False)
        self.experts = nn.ModuleList([nn.Linear(12, 12, bias=False) for _ in range(5)])

    def forward(self, router_input, hidden_states):
        logits = self.router(router_input)
        self.chosen = logits.topk(2, -1).indices
        self.weights = logits.gather(-1, self.chosen).softmax(-1)
        outputs = torch.stack([e(hidden_states) for e in self.experts], -2)
        y = outputs.gather(-2, self.chosen[..., None].expand(-1,-1,-1,12))
        return (y*self.weights[..., None]).sum(-2), logits


@pytest.mark.parametrize("mode", ["fixed", "routed"])
def test_native_slots_weights_and_router_identity_remain_unchanged(mode):
    torch.manual_seed(0)
    native = NativeMoE(); control = copy.deepcopy(native)
    wrapper = AdditiveVisualMoE(native, bank(mode).config, 12, smallthinker=True)
    x = torch.randn(1, 6, 12)
    visual = torch.tensor([[0,1,1,0,0,0]], dtype=torch.bool)
    wrapper.set_modality_mask(visual, torch.ones_like(visual))
    expected, old_logits = control(x, x)
    actual, logits = wrapper(x, x)
    assert wrapper.original is native and wrapper.original.router is native.router
    assert torch.equal(control.chosen, native.chosen)
    assert torch.equal(control.weights, native.weights)
    assert torch.equal(logits, old_logits)
    assert torch.equal(actual, expected)  # zero-init exact identity
    assert wrapper.visual_experts.last_dispatch["counts"][1].sum() == 4
    # After learning, the extra delta changes; original routing still does not.
    nn.init.normal_(wrapper.visual_experts.up.weight, std=.1)
    actual, _ = wrapper(x, x)
    assert not torch.equal(actual, expected)
    assert torch.equal(control.chosen, native.chosen)
    assert torch.equal(control.weights, native.weights)
    model = nn.Sequential(wrapper).eval()
    with torch.no_grad(), generation(model):
        wrapper(x, x)
        assert wrapper.visual_experts._context is not None
    assert not wrapper.visual_experts.cache_enabled and wrapper.visual_experts._context is None


def test_request_cache_cleanup_after_exception():
    wrapper = AdditiveVisualMoE(NativeMoE(), bank().config, 12, smallthinker=True)
    model = nn.Sequential(wrapper).eval()
    with pytest.raises(RuntimeError, match="sentinel"):
        with torch.no_grad(), generation(model):
            wrapper(torch.randn(1,3,12), torch.randn(1,3,12))
            raise RuntimeError("sentinel")
    assert wrapper.visual_experts._context is None
    assert not wrapper.visual_experts.cache_enabled


@pytest.mark.parametrize("kind", ["smallthinker","lfm2_moe","deepseek_v3"])
@pytest.mark.parametrize("mode", ["fixed","routed"])
def test_real_native_moe_modules_preserve_output_and_shared_experts(kind,mode):
    if kind == 'smallthinker':
        from src.model.smallthinker import SmallThinkerConfig,SmallThinkerMoeBlock
        config=SmallThinkerConfig(hidden_size=8,num_attention_heads=2,num_key_value_heads=1,moe_ffn_hidden_size=12,
            moe_num_primary_experts=4,moe_num_active_primary_experts=2)
        native=SmallThinkerMoeBlock(config)
    elif kind == 'lfm2_moe':
        from transformers import Lfm2MoeConfig
        from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeSparseMoeBlock
        config=Lfm2MoeConfig(hidden_size=8,intermediate_size=12,moe_intermediate_size=6,
            num_attention_heads=2,num_key_value_heads=1,num_experts=4,num_experts_per_tok=2)
        native=Lfm2MoeSparseMoeBlock(config)
    else:
        from transformers import DeepseekV3Config
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
        config=DeepseekV3Config(hidden_size=8,intermediate_size=12,moe_intermediate_size=6,
            n_routed_experts=4,n_shared_experts=1,num_experts_per_tok=2,n_group=1,topk_group=1,
            routed_scaling_factor=1.0)
        native=DeepseekV3MoE(config)
    for p in native.parameters():nn.init.normal_(p,std=.05)
    x=torch.randn(1,6,8)
    inputs=(x,x) if kind=='smallthinker' else (x,)
    expected=native(*inputs)
    wrapper=AdditiveVisualMoE(native,VisualBranchConfig(mode=mode,width=8,heads=2,
        blocks=2,top_k=2,learned_masks=True),8,smallthinker=kind=='smallthinker')
    wrapper.set_modality_mask(torch.tensor([[0,1,1,0,0,0]],dtype=torch.bool))
    actual=wrapper(*inputs)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    output=actual[0] if isinstance(actual,tuple) else actual
    output.square().sum().backward()
    assert wrapper.visual_experts.up.weight.grad.abs().sum()>0
    if kind=='deepseek_v3':assert wrapper.original.shared_experts is native.shared_experts
