import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.model import research_common_pool_v1 as common
from src.model.smallthinker import SmallThinkerConfig, SmallThinkerMoeBlock


def native(backend):
    if backend == 'smallthinker':
        config = SmallThinkerConfig(vocab_size=32, hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4, moe_ffn_hidden_size=12,
            moe_num_primary_experts=4, moe_num_active_primary_experts=2)
        return config, SmallThinkerMoeBlock(config)
    if backend == 'lfm2_moe':
        from transformers import Lfm2MoeConfig
        from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeSparseMoeBlock
        config = Lfm2MoeConfig(hidden_size=8, intermediate_size=16, moe_intermediate_size=12,
            num_experts=4, num_experts_per_tok=2, num_hidden_layers=1, num_dense_layers=0)
        return config, Lfm2MoeSparseMoeBlock(config)
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
    config = DeepseekV3Config(hidden_size=8, intermediate_size=16, moe_intermediate_size=12,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, num_hidden_layers=1,
        first_k_dense_replace=0, n_group=1, topk_group=1)
    return config, DeepseekV3MoE(config)


@pytest.mark.parametrize('backend', ['smallthinker', 'lfm2_moe', 'deepseek_v3'])
@pytest.mark.parametrize('method', ['R4', 'R5'])
def test_common_topk_no_masks_gradients_and_roundtrip(backend, method):
    torch.manual_seed(42)
    config, block = native(backend)
    for p in block.parameters():
        nn.init.normal_(p, std=.05)
    model = nn.Module(); model.config=config; model.block=block
    old_router = block.primary_router if backend == 'smallthinker' else block.gate
    old_weights = old_router.weight.detach().clone()
    model.requires_grad_(False)
    wrapper = common.install(model, method)[0]
    common.configure(model, 'SFT')
    assert wrapper.old_count == 4 and wrapper.added_count == 8 and wrapper.top_k == 2
    assert len(wrapper.visual_experts) == 8
    assert wrapper.primary_router.weight.requires_grad == (method == 'R4')
    if method == 'R5':
        torch.testing.assert_close(wrapper.primary_router.weight, old_weights, rtol=0, atol=0)
        inputs = torch.randn(5,8)
        assert wrapper.router_adapter(inputs).count_nonzero() == 0
        scores, logits, _ = wrapper.routing_scores(inputs)
        torch.testing.assert_close(logits[:,:4], wrapper._router_logits(inputs), rtol=0, atol=0)
    # Make a new expert win for both text and visual tokens. No eligibility mask.
    with torch.no_grad():
        if method == 'R4':
            wrapper.primary_router.weight.zero_()
            wrapper.primary_router.weight[4].fill_(1)
        else:
            wrapper.router_adapter.lora_A.fill_(.1)
            wrapper.router_adapter.lora_B.zero_()
            wrapper.router_adapter.lora_B[4].fill_(.2)
    visual = torch.tensor([[False, True, False, True]])
    wrapper.set_modality_mask(visual)
    states = torch.ones(1,4,8,requires_grad=True)
    value = wrapper(states,states)[0] if backend == 'smallthinker' else wrapper(states)
    assert (wrapper.last_indices >= 4).any(-1).all()
    assert wrapper.last_indices.shape == (4,2)
    value.square().sum().backward()
    assert all(p.grad is None for p in wrapper.experts.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in wrapper.visual_experts.parameters())
    assert torch.isfinite(states.grad).all()
    if method == 'R4':
        assert wrapper.primary_router.weight.grad.abs().sum() > 0
    else:
        assert wrapper.primary_router.weight.grad is None
        assert wrapper.router_adapter.lora_B.grad.abs().sum() > 0
    clone = copy.deepcopy(model)
    clone.load_state_dict(model.state_dict(), strict=True)
    expected = clone.block(states,states)[0] if backend == 'smallthinker' else clone.block(states)
    torch.testing.assert_close(value,expected,rtol=0,atol=0)
    assert wrapper.diagnostics()['method'] == method
    assert not common.configure(model, 'INFERENCE')


@pytest.mark.parametrize('backend', ['smallthinker', 'lfm2_moe', 'deepseek_v3'])
def test_caption_warmup_covers_all_eight_without_sft_forcing(backend):
    config, block = native(backend)
    for parameter in block.parameters(): nn.init.normal_(parameter,std=.05)
    model=nn.Module();model.config=config;model.block=block
    model.requires_grad_(False)
    wrapper=common.install(model,'R5')[0]
    common.configure(model,'A1')
    states=torch.ones(1,32,8)
    wrapper.set_modality_mask(torch.zeros(1,32,dtype=torch.bool))
    if backend=='smallthinker':wrapper(states,states)
    else:wrapper(states)
    assert set(range(4,12)).issubset(set(wrapper.last_indices.flatten().tolist()))
    assert wrapper.last_indices.shape[-1]==2
    common.configure(model,'SFT')
    assert not wrapper.warmup_force_new
