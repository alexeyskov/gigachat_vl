"""CUDA integration tests for actual NF4 expert input/adapter gradients."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.model.research_quantization import (
    NF4LoRALinear, quantize_packed_experts, set_packed_lora_trainable,
)
from src.model.research_moe import install_research_moe, set_modality_mask

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="NF4 expert tests require CUDA")


def test_actual_nf4_forward_and_input_adapter_gradients():
    import bitsandbytes.functional as bnf
    torch.manual_seed(42)
    weight = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16) * .1
    module = NF4LoRALinear(weight, rank=4, alpha=8, dropout=0)
    with torch.no_grad():
        module.lora_B.normal_(std=.01)
    x = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    dequantized = bnf.dequantize_4bit(module.base.weight, module.base.weight.quant_state)
    actual = module(x)
    expected = F.linear(xr, dequantized) + 2*F.linear(F.linear(xr, module.lora_A), module.lora_B)
    torch.testing.assert_close(actual, expected, atol=.01, rtol=.025)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, atol=.02, rtol=.04)
    assert module.lora_A.grad.abs().sum() > 0
    assert module.lora_B.grad.abs().sum() > 0
    assert module.base.weight.grad is None
    assert module.base.weight.dtype == torch.uint8
    assert module.base.weight.numel() <= weight.numel()//2 + 1


def model_with_bank(kind):
    if kind == "lfm2_moe":
        from transformers import Lfm2MoeConfig
        from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeSparseMoeBlock
        cfg = Lfm2MoeConfig(hidden_size=32, moe_intermediate_size=64,
            num_attention_heads=2, num_key_value_heads=1,
            num_experts=4, num_experts_per_tok=2)
        block = Lfm2MoeSparseMoeBlock(cfg)
    elif kind == "deepseek_v3":
        from transformers import DeepseekV3Config
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
        cfg = DeepseekV3Config(hidden_size=32, moe_intermediate_size=64,
            num_attention_heads=2, num_key_value_heads=1,
            n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
            n_group=1, topk_group=1)
        block = DeepseekV3MoE(cfg)
    else:
        from transformers import GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
        cfg = GptOssConfig(hidden_size=32, intermediate_size=64,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            num_local_experts=4, num_experts_per_tok=2, num_hidden_layers=1,
            layer_types=["full_attention"],
            rope_parameters={"rope_type":"default", "rope_theta":10000.0})
        block = GptOssMLP(cfg)
    for p in block.parameters():
        nn.init.normal_(p, std=.05)
    model = nn.Module()
    model.config = cfg
    model.block = block
    return model.to(device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("kind", ["lfm2_moe", "deepseek_v3", "gpt_oss"])
@pytest.mark.parametrize("rank", [0, 4])
def test_quantized_packed_native_and_extended_forward(kind, rank):
    model = model_with_bank(kind)
    old_bytes = sum(p.numel()*p.element_size() for p in model.block.experts.parameters())
    report = quantize_packed_experts(model, rank=rank, alpha=8, dropout=0)
    assert report["original_weight_bytes"] == old_bytes
    assert report["quantized_parameter_bytes"] < old_bytes//2
    assert not hasattr(model.block.experts, "gate_up_proj")
    model.requires_grad_(False)
    set_packed_lora_trainable(model, rank > 0)
    x = torch.randn(1, 8, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual = model.block(x)
    if isinstance(actual, tuple): actual = actual[0]
    actual.float().square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    if rank:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in model.named_parameters() if "lora_B" in name)
    wrapped = install_research_moe(model, 1, "c")[0]
    wrapped.warmup_force_new = True
    set_modality_mask(model, torch.tensor([[True, False]*4], device="cuda"))
    x2 = x.detach().clone().requires_grad_(True)
    result = model.block(x2)
    if isinstance(result, tuple): result = result[0]
    result.float().square().sum().backward()
    assert torch.isfinite(x2.grad).all()
    assert wrapped.modality_gate_logits.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in wrapped.visual_experts.parameters())
    for module in model.modules():
        if isinstance(module, NF4LoRALinear):
            assert not module.base.weight.requires_grad


def test_nf4_trainable_adapter_checkpoint_roundtrip(tmp_path):
    original = model_with_bank("lfm2_moe")
    fresh = copy.deepcopy(original)
    quantize_packed_experts(original, rank=4, alpha=8, dropout=0)
    quantize_packed_experts(fresh, rank=4, alpha=8, dropout=0)
    with torch.no_grad():
        for module in original.modules():
            if isinstance(module, NF4LoRALinear):
                module.lora_B.normal_(std=.01)
    adapter_state = {n:p.detach().cpu() for n,p in original.named_parameters() if "lora_" in n}
    torch.save(adapter_state, tmp_path/"adapters.pt")
    restored = torch.load(tmp_path/"adapters.pt", weights_only=True)
    unexpected = fresh.load_state_dict(restored, strict=False).unexpected_keys
    assert not unexpected
    x = torch.randn(1, 8, 32, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(original.block(x), fresh.block(x))


@pytest.mark.parametrize("kind", ["lfm2_moe", "deepseek_v3", "gpt_oss"])
@pytest.mark.parametrize("fp32_autocast", [False, True])
def test_packed_grouped_mm_cuda_output_and_gradient_parity(kind, fp32_autocast):
    original = model_with_bank(kind)
    fast = install_research_moe(original, 3, "c")[0]
    if fp32_autocast:
        fast.float()
    slow = copy.deepcopy(fast)
    slow.use_grouped_mm = False
    fast.warmup_force_new = slow.warmup_force_new = True
    mask = torch.tensor([[True, False]*7], device="cuda")
    fast.set_modality_mask(mask)
    slow.set_modality_mask(mask)
    x = torch.randn(1, 14, 32, device="cuda", dtype=torch.float32 if fp32_autocast else torch.bfloat16, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=fp32_autocast):
        actual, expected = fast(x), slow(xr)
        if isinstance(actual, tuple): actual, expected = actual[0], expected[0]
    torch.testing.assert_close(actual, expected, atol=.001, rtol=.03)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, atol=.002, rtol=.05)
    for (name, p), (_, q) in zip(fast.named_parameters(), slow.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=.002, rtol=.05)
