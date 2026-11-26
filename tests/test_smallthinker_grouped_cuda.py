"""Actual grouped-GEMM SmallThinker parity, caching and weight freshness."""

import copy

import pytest
import torch
from torch import nn

from src.model.smallthinker import SmallThinkerConfig, SmallThinkerMoeBlock
from src.model.research_moe import ResearchSmallThinkerMoE


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Grouped CUDA kernels required")


def block():
    cfg = SmallThinkerConfig(hidden_size=32, moe_ffn_hidden_size=64,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        num_hidden_layers=1, moe_num_primary_experts=4,
        moe_num_active_primary_experts=2)
    return SmallThinkerMoeBlock(cfg).to("cuda")


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_smallthinker_grouped_cuda_gradient_parity(frozen, wrapped):
    torch.manual_seed(24)
    fast = block()
    if wrapped:
        fast = ResearchSmallThinkerMoE(fast, 3, "c")
        fast.set_modality_mask(torch.tensor([[True, False]*7], device="cuda"))
        fast.warmup_force_new = True
    fast.requires_grad_(not frozen)
    slow = copy.deepcopy(fast)
    slow.use_grouped_mm = False
    x = torch.randn(1, 14, 32, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual, _ = fast(x, x)
        expected, _ = slow(xr, xr)
    torch.testing.assert_close(actual, expected, atol=.003, rtol=.03)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, atol=.003, rtol=.04)
    for (name, p), (_, q) in zip(fast.named_parameters(), slow.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=.005, rtol=.05)
    if frozen:
        assert fast._frozen_grouped_cache is not None
    else:
        assert fast._frozen_grouped_cache is None


def test_frozen_grouped_cache_reuse_and_parameter_version_invalidation():
    fast = block().requires_grad_(False)
    x = torch.randn(1, 8, 32, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fast(x, x)
        original = fast._frozen_grouped_cache
        fast(x, x)
        assert fast._frozen_grouped_cache is original
        with torch.no_grad():
            fast.experts[0].gate.weight.add_(.1)
        fast(x, x)
        assert fast._frozen_grouped_cache is not original
        # Cache tensors remain ordinary tensors even if first built at eval.
        fast._frozen_grouped_cache = None
        with torch.inference_mode():
            fast(x, x)
        assert not fast._frozen_grouped_cache[1].is_inference()
        result, _ = fast(x, x)
    result.square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_cache_not_saved_and_trainable_stacks_refresh_after_update():
    fast = block()
    slow = copy.deepcopy(fast)
    slow.use_grouped_mm = False
    x = torch.randn(1, 8, 32, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        first = fast(x, x)[0]
    with torch.no_grad():
        for owner in (fast, slow):
            owner.experts[0].down.weight.add_(.2)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        second, _ = fast(x, x)
        reference, _ = slow(x, x)
    torch.testing.assert_close(second, reference, atol=.003, rtol=.03)
    assert not torch.equal(first, second)
    assert not any("cache" in key for key in fast.state_dict())
