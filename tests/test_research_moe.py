"""Research routing contracts on tiny, real native SmallThinker blocks."""

import copy

import pytest
import torch
from torch import nn

from src.model.research_moe import (
    ResearchSmallThinkerMoE,
    ResearchPackedMoE,
    configure_research_trainability,
    install_research_moe,
    set_modality_mask,
)
from src.model.smallthinker import (
    SmallThinkerConfig,
    SmallThinkerForCausalLM,
    SmallThinkerMoeBlock,
)


def config(softmax=False):
    return SmallThinkerConfig(
        vocab_size=32, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        moe_ffn_hidden_size=12, moe_num_primary_experts=4,
        moe_num_active_primary_experts=2,
        moe_primary_router_apply_softmax=softmax,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )


@pytest.mark.parametrize("softmax", [False, True])
@pytest.mark.parametrize("family", [1, 3])
def test_zero_added_native_equivalence(softmax, family):
    torch.manual_seed(4)
    original = SmallThinkerMoeBlock(config(softmax))
    extended = ResearchSmallThinkerMoE(copy.deepcopy(original), family, "b", added_experts=0)
    router_input, states = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    expected, expected_scores = original(router_input, states)
    actual, actual_scores = extended(router_input, states)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_scores, expected_scores)


@pytest.mark.parametrize("family", [1, 2, 3])
@pytest.mark.parametrize("softmax", [False, True])
def test_hard_separation_and_padding(family, softmax):
    module = ResearchSmallThinkerMoE(SmallThinkerMoeBlock(config(softmax)), family, "a", added_experts=1)
    mask = torch.tensor([[True, False, True, False]])
    attention = torch.tensor([[True, True, True, False]])
    module.set_modality_mask(mask, attention)
    states = torch.randn(1, 4, 8)
    actual, _ = module(states, states)
    indices, weights = module.last_indices, module.last_weights
    selected = weights > 0
    assert torch.all(indices[0][selected[0]] == 4)
    assert torch.all(indices[2][selected[2]] == 4)
    assert torch.all(indices[1][selected[1]] < 4)
    assert weights[3].sum() == 0
    assert actual[0, 3].abs().sum() == 0
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(weights[:3].sum(-1), torch.ones(3))
    actual.sum().backward()
    assert module.visual_experts[0].down.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


def test_hierarchical_distribution_keeps_old_ratios():
    module = ResearchSmallThinkerMoE(SmallThinkerMoeBlock(config()), 3, "b")
    x = torch.randn(5, 8)
    native = module.primary_router(x).float().sigmoid()
    p = native / native.sum(-1, keepdim=True)
    scores, _, alpha = module.routing_scores(x)
    torch.testing.assert_close(alpha, torch.full((5, 1), 0.05))
    torch.testing.assert_close(scores[:, :4], 0.95*p)
    torch.testing.assert_close(scores.sum(-1), torch.ones(5))


@pytest.mark.parametrize("family", [1, 2, 3])
def test_warmup_new_experts_have_gradients(family):
    module = ResearchSmallThinkerMoE(SmallThinkerMoeBlock(config()), family, "c")
    module.warmup_force_new = True
    module.set_modality_mask(torch.tensor([[True, False, True, False, True, False]]))
    x = torch.randn(1, 6, 8, requires_grad=True)
    output, _ = module(x, x)
    output.square().sum().backward()
    assert (module.last_indices >= 4).any()
    assert module.modality_gate_logits.grad is not None
    assert module.modality_gate_logits.grad.abs().sum() > 0
    assert sum(e.down.weight.grad.abs().sum() for e in module.visual_experts if e.down.weight.grad is not None) > 0
    if family == 1:
        assert module.primary_router.weight.grad is None
        assert module.router_adapter.down.weight.grad.abs().sum() > 0
    elif family == 3:
        assert module.router_mass.down.bias.grad.abs().sum() > 0


def test_gates_block_bias_outputs():
    module = ResearchSmallThinkerMoE(SmallThinkerMoeBlock(config()), 2, "c")
    class Biased(nn.Module):
        def forward(self, x):
            return torch.ones_like(x)
    module.experts = nn.ModuleList([Biased() for _ in module.experts])
    module.visual_experts = nn.ModuleList([Biased() for _ in module.visual_experts])
    with torch.no_grad():
        module.modality_gate_logits[:, 1] = -torch.inf
    module.set_modality_mask(torch.tensor([[True, False]]))
    x = torch.randn(1, 2, 8)
    actual, _ = module(x, x)
    assert actual[0, 0].abs().sum() == 0
    assert actual[0, 1].abs().sum() > 0


def test_masks_checkpoint_state_and_cached_decoding():
    module = ResearchSmallThinkerMoE(SmallThinkerMoeBlock(config()), 3, "a")
    module.set_modality_mask(torch.tensor([[True, False, True]]))
    assert not any("mask" in name for name in module.state_dict())
    module.eval()
    # Beam expansion and appended text tokens.
    module(torch.randn(2, 5, 8), torch.randn(2, 5, 8))
    assert (module.last_indices[0] >= 4).all()
    assert (module.last_indices[3] < 4).all()
    # Cached generation: only a new text token is fed.
    module(torch.randn(2, 1, 8), torch.randn(2, 1, 8))
    assert (module.last_indices < 4).all()
    module.train()
    with pytest.raises(ValueError, match="Stale"):
        module(torch.randn(1, 1, 8), torch.randn(1, 1, 8))


def test_install_trainability_and_state_roundtrip():
    original = SmallThinkerForCausalLM(config())
    module = install_research_moe(original, 3, "c")[0]
    original.requires_grad_(False)
    configure_research_trainability(original, "A1")
    assert not module.primary_router.weight.requires_grad
    assert module.router_new.down.weight.requires_grad
    assert module.warmup_force_new
    configure_research_trainability(original, "SFT")
    assert module.primary_router.weight.requires_grad
    assert not module.warmup_force_new
    clone = SmallThinkerForCausalLM(config())
    clone_module = install_research_moe(clone, 3, "c")[0]
    clone.load_state_dict(original.state_dict(), strict=True)
    mask = torch.tensor([[False, True, False, True]])
    set_modality_mask(original, mask)
    set_modality_mask(clone, mask)
    x = torch.randn(1, 4, 8)
    torch.testing.assert_close(module(x, x)[0], clone_module(x, x)[0])
    with pytest.raises(ValueError, match="already installed"):
        install_research_moe(original, 2, "a")


def test_unsupported_architecture_rejected():
    model = nn.Module()
    model.config = type("Config", (), {"model_type": "unknown_future_moe"})()
    with pytest.raises(NotImplementedError):
        install_research_moe(model, 1, "b")


@pytest.mark.parametrize("softmax", [False, True])
def test_sorted_native_dispatch_matches_original_formula_and_gradients(softmax):
    torch.manual_seed(8)
    module = SmallThinkerMoeBlock(config(softmax))
    reference = copy.deepcopy(module)
    router_input, states = torch.randn(2, 7, 8), torch.randn(2, 7, 8)
    actual, selected = module(router_input, states)
    flat = states.reshape(-1, 8)
    logits = reference.primary_router(router_input.reshape(-1, 8))
    native_logits, indices = logits.topk(2, dim=-1)
    if softmax:
        weights = native_logits.softmax(-1, dtype=torch.float32)
    else:
        weights = native_logits.sigmoid()
        weights = weights / weights.sum(-1, keepdim=True)
    expected = torch.zeros_like(flat)
    for idx, expert in enumerate(reference.experts):
        token, slot = torch.where(indices == idx)
        expected.index_add_(0, token, expert(flat[token]) * weights[token, slot, None])
    torch.testing.assert_close(actual, expected.reshape_as(actual))
    torch.testing.assert_close(selected, native_logits)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name, p), (_, q) in zip(module.named_parameters(), reference.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad)


def test_end_to_end_checkpointed_hidden_states_backward():
    torch.manual_seed(9)
    llm = SmallThinkerForCausalLM(config())
    wrapped = install_research_moe(llm, 3, "c")[0]
    llm.gradient_checkpointing_enable()
    llm.train()
    set_modality_mask(llm, torch.tensor([[False, True, True, False, False]]))
    inputs = torch.randn(1, 5, 8, requires_grad=True)
    hidden = llm.model(inputs_embeds=inputs, attention_mask=torch.ones(1, 5, dtype=torch.long), use_cache=False).last_hidden_state
    llm.lm_head(hidden[:, -2:]).square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0
    assert wrapped.router_mass.down.bias.grad is not None


def packed_block(kind, norm=True, groups=1):
    if kind == "lfm2_moe":
        from transformers import Lfm2MoeConfig
        from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeSparseMoeBlock
        cfg = Lfm2MoeConfig(hidden_size=8, intermediate_size=12, moe_intermediate_size=6,
            num_attention_heads=2, num_key_value_heads=1, num_experts=4,
            num_experts_per_tok=2, norm_topk_prob=norm, use_expert_bias=True,
            routed_scaling_factor=1.5)
        module = Lfm2MoeSparseMoeBlock(cfg)
        module.expert_bias.copy_(torch.tensor([0.1, -0.2, 0.05, 0.0]))
    else:
        from transformers import DeepseekV3Config
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
        cfg = DeepseekV3Config(hidden_size=8, intermediate_size=12, moe_intermediate_size=6,
            num_attention_heads=2, num_key_value_heads=1, n_routed_experts=4,
            n_shared_experts=2, num_experts_per_tok=2, n_group=groups,
            topk_group=1, norm_topk_prob=norm, routed_scaling_factor=1.5,
            q_lora_rank=None, kv_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4)
        module = DeepseekV3MoE(cfg)
        module.gate.e_score_correction_bias.copy_(torch.tensor([0.1, -0.2, 0.05, 0.0]))
    for p in module.parameters():
        nn.init.normal_(p, std=0.1)
    return cfg, module


@pytest.mark.parametrize("kind", ["lfm2_moe", "deepseek_v3"])
@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("family", [1, 3])
def test_packed_zero_added_native_output_and_gradients(kind, norm, family):
    torch.manual_seed(11)
    cfg, reference = packed_block(kind, norm)
    module = ResearchPackedMoE(copy.deepcopy(reference), cfg, family, "b", added_experts=0)
    states = torch.randn(2, 5, 8)
    actual = module(states)
    expected = reference(states)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name, p) in module.experts.named_parameters():
        torch.testing.assert_close(p.grad, dict(reference.experts.named_parameters())[name].grad)
    torch.testing.assert_close(module.primary_router.weight.grad, reference.gate.weight.grad)


@pytest.mark.parametrize("kind", ["lfm2_moe", "deepseek_v3"])
@pytest.mark.parametrize("family", [2, 3])
@pytest.mark.parametrize("policy", ["a", "b", "c"])
def test_packed_research_expansion_backward(kind, family, policy):
    cfg, base = packed_block(kind)
    module = ResearchPackedMoE(base, cfg, family, policy)
    module.warmup_force_new = True
    mask = torch.tensor([[True, False, True, False, True]])
    module.set_modality_mask(mask)
    states = torch.randn(1, 5, 8, requires_grad=True)
    actual = module(states)
    assert torch.isfinite(actual).all()
    actual.square().sum().backward()
    assert torch.isfinite(states.grad).all()
    assert module.experts.gate_up_proj.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.visual_experts.parameters())
    if policy == "a":
        assert (module.last_indices[0] >= 4).all()
        assert (module.last_indices[1] < 4).all()
    if policy == "c":
        assert module.modality_gate_logits.grad.abs().sum() > 0
        if kind == "deepseek_v3":
            assert module.modality_gate_logits.shape == (8, 2)
            assert module.modality_gate_logits.grad[-2:].abs().sum() > 0


def test_packed_shared_experts_are_closed_to_visual_positions_in_a():
    cfg, base = packed_block("deepseek_v3")
    module = ResearchPackedMoE(base, cfg, 3, "a")
    states = torch.randn(4, 8)
    visual = torch.tensor([True, False, True, False])
    output = module._shared_output(states, visual, torch.ones(4, dtype=torch.bool))
    assert output[visual].abs().sum() == 0
    assert output[~visual].abs().sum() > 0


def test_shared_per_expert_gate_decomposition_matches_native_when_open():
    cfg, base = packed_block("deepseek_v3")
    module = ResearchPackedMoE(base, cfg, 3, "c")
    with torch.no_grad():
        module.modality_gate_logits.fill_(torch.inf)
    states = torch.randn(4, 8)
    actual = module._shared_output(states, torch.tensor([True, False, True, False]), torch.ones(4, dtype=torch.bool))
    torch.testing.assert_close(actual, base.shared_experts(states))


def test_grouped_expansion_rejected_but_grouped_zero_added_native_parity():
    cfg, base = packed_block("deepseek_v3", groups=2)
    with pytest.raises(NotImplementedError, match="group allocation"):
        ResearchPackedMoE(base, cfg, 3, "b", added_experts=2)
    module = ResearchPackedMoE(copy.deepcopy(base), cfg, 3, "b", added_experts=0)
    states = torch.randn(1, 7, 8)
    torch.testing.assert_close(module(states), base(states))


@pytest.mark.parametrize("kind", ["lfm2_moe", "deepseek_v3"])
def test_install_packed_and_roundtrip(kind):
    cfg, block = packed_block(kind)
    model = nn.Module()
    model.config = cfg
    model.block = block
    wrapped = install_research_moe(model, 3, "c")[0]
    assert wrapped.spec()["backend"] == kind
    clone = nn.Module()
    clone.config = cfg
    clone.block = packed_block(kind)[1]
    restored = install_research_moe(clone, 3, "c")[0]
    clone.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(1, 5, 8)
    torch.testing.assert_close(restored(x), wrapped(x))


def gpt_block():
    from transformers import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
    cfg = GptOssConfig(hidden_size=8, intermediate_size=12, num_local_experts=4,
        num_experts_per_tok=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, num_hidden_layers=1, layer_types=["full_attention"],
        rope_parameters={"rope_type":"default", "rope_theta":10000.0})
    module = GptOssMLP(cfg)
    for p in module.parameters():
        nn.init.normal_(p, std=.1)
    return cfg, module


@pytest.mark.parametrize("family", [1, 3])
def test_gpt_oss_zero_added_native_output_scores_and_gradient_parity(family):
    cfg, base = gpt_block()
    wrapped = ResearchPackedMoE(copy.deepcopy(base), cfg, family, "b", added_experts=0)
    x = torch.randn(2, 6, 8)
    actual, scores = wrapped(x)
    expected, native_scores = base(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(scores, native_scores)
    assert scores.requires_grad
    actual.square().sum().backward()
    expected.square().sum().backward()
    for name, p in wrapped.experts.named_parameters():
        torch.testing.assert_close(p.grad, dict(base.experts.named_parameters())[name].grad)
    torch.testing.assert_close(wrapped.primary_router.bias.grad, base.router.bias.grad)


@pytest.mark.parametrize("family", [1, 2, 3])
@pytest.mark.parametrize("policy", ["a", "b", "c"])
def test_gpt_oss_research_expansion(family, policy):
    cfg, base = gpt_block()
    wrapped = ResearchPackedMoE(base, cfg, family, policy)
    wrapped.warmup_force_new = True
    wrapped.set_modality_mask(torch.tensor([[False, True]*4]))
    x = torch.randn(1, 8, 8, requires_grad=True)
    output, scores = wrapped(x)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(scores.sum(-1), torch.ones(8))
    output.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in wrapped.visual_experts.parameters())
    assert torch.isfinite(x.grad).all()
