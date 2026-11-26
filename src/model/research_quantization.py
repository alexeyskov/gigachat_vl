"""Per-expert NF4+LoRA for native LFM2/DeepSeek-V3/dequantized GPT-OSS FFNs.

Ordinary attention/shared linear layers can be quantized by Transformers first.
This module covers the original 3D expert tensors which that conversion skips.
Each selected expert uses bitsandbytes' native Linear4bit autograd: frozen NF4
base, differentiable inputs, trainable low-rank update. No original bf16 tensor
is retained after replacement, and the full expert bank is never dequantized.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class NF4LoRALinear(nn.Module):
    def __init__(self, weight: torch.Tensor, rank=16, alpha=32, dropout=0.05,
                 compute_dtype=torch.bfloat16, double_quant=True, bias=None):
        super().__init__()
        import bitsandbytes as bnb
        if weight.ndim != 2 or weight.device.type != "cuda":
            raise ValueError("NF4 expert conversion requires a CUDA 2D weight.")
        if rank < 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid LoRA configuration.")
        self.out_features, self.in_features = weight.shape
        self.rank, self.alpha = int(rank), float(alpha)
        self.scaling = float(alpha / rank) if rank else 0.0
        self.dropout = nn.Dropout(dropout)
        # Meta construction avoids a redundant full-size nn.Linear allocation.
        self.base = bnb.nn.Linear4bit(
            self.in_features, self.out_features, bias=bias is not None,
            compute_dtype=compute_dtype, compress_statistics=double_quant,
            quant_type="nf4", device="meta",
        )
        self.base.weight = bnb.nn.Params4bit(
            weight.detach().contiguous(), requires_grad=False,
            compress_statistics=double_quant, quant_type="nf4", module=self.base,
        ).to(weight.device)
        if bias is not None:
            self.base.bias = nn.Parameter(bias.detach().to(device=weight.device, dtype=compute_dtype).clone(), requires_grad=False)
        if rank:
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=weight.device, dtype=compute_dtype))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=weight.device, dtype=compute_dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x):
        result = self.base(x)
        if self.rank:
            adapted = F.linear(F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
            result = result + adapted.to(result.dtype) * self.scaling
        return result


class NF4LoRAExpert(nn.Module):
    def __init__(self, gate_up, down, act_fn, gate_up_bias=None, down_bias=None,
                 interleaved=False, **kwargs):
        super().__init__()
        self.gate_up_proj = NF4LoRALinear(gate_up, bias=gate_up_bias, **kwargs)
        self.down_proj = NF4LoRALinear(down, bias=down_bias, **kwargs)
        self.act_fn = act_fn
        self.interleaved = interleaved

    def forward(self, x):
        if self.interleaved:
            return self.down_proj(self.act_fn(self.gate_up_proj(x)))
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class QuantizedPackedExperts(nn.Module):
    """The same call contract as native Lfm2MoeExperts/DeepseekV3Experts."""
    def __init__(self, original, **kwargs):
        super().__init__()
        self.num_experts = original.num_experts
        gpt_style = original.__class__.__name__ == "GptOssExperts"
        self.hidden_dim = original.hidden_size if gpt_style else original.hidden_dim
        self.intermediate_dim = original.intermediate_size if gpt_style else original.intermediate_dim
        if gpt_style:
            from src.model.research_moe import GptOssActivation
            self.alpha, self.limit = original.alpha, original.limit
            self.act_fn = GptOssActivation(self.alpha, self.limit)
            self.expert_modules = nn.ModuleList([
                NF4LoRAExpert(original.gate_up_proj[e].T, original.down_proj[e].T,
                    self.act_fn, gate_up_bias=original.gate_up_proj_bias[e],
                    down_bias=original.down_proj_bias[e], interleaved=True, **kwargs)
                for e in range(self.num_experts)
            ])
        else:
            self.act_fn = original.act_fn
            self.expert_modules = nn.ModuleList([
                NF4LoRAExpert(original.gate_up_proj[e], original.down_proj[e], original.act_fn, **kwargs)
                for e in range(self.num_experts)
            ])
        self.quantization_spec = dict(
            method="per_expert_nf4_lora", rank=kwargs.get("rank", 16),
            alpha=kwargs.get("alpha", 32), dropout=kwargs.get("dropout", 0.05),
            double_quant=kwargs.get("double_quant", True),
            compute_dtype=str(kwargs.get("compute_dtype", torch.bfloat16)),
            expert_projection_scope=["packed_gate_up_proj", "down_proj"],
        )

    def forward_expert(self, expert_index, states):
        return self.expert_modules[expert_index](states)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        flat_experts = top_k_index.reshape(-1)
        order = flat_experts.argsort(stable=True)
        counts = torch.bincount(flat_experts, minlength=self.num_experts).tolist()
        top_k = top_k_index.shape[-1]
        offset = 0
        for e, count in enumerate(counts):
            if count:
                selected = order[offset:offset+count]
                token, slot = selected // top_k, selected % top_k
                delta = self.forward_expert(e, hidden_states[token]) * top_k_weights[token, slot, None]
                output.index_add_(0, token, delta.to(output.dtype))
            offset += count
        return output


def quantize_packed_experts(llm, rank=16, alpha=32, dropout=0.05,
                            compute_dtype=torch.bfloat16, double_quant=True):
    """Replace native packed experts in-place, before research-router installation.

    Reconstruct these frozen bases from the pinned original model on resume;
    training checkpoints only need LoRA parameters, not another NF4 base copy.
    """
    backend = getattr(llm.config, "model_type", None)
    if backend not in ("lfm2_moe", "deepseek_v3", "gpt_oss"):
        raise NotImplementedError("Packed NF4 conversion currently supports LFM2-MoE, DeepSeek-V3 and dequantized GPT-OSS.")
    matches = [(parent, name, module) for parent in llm.modules()
               for name, module in parent.named_children()
               if module.__class__.__name__ in ("Lfm2MoeExperts", "DeepseekV3Experts", "GptOssExperts")]
    if not matches:
        raise ValueError("No native packed expert banks found, or conversion already applied.")
    # Validate the complete bank before mutation; no silent mixed backend.
    for _, _, module in matches:
        if module.gate_up_proj.device.type != "cuda" or module.gate_up_proj.ndim != 3:
            raise ValueError("All native packed expert banks must be materialized on CUDA.")
    report = {"backend": backend, "banks": 0, "experts": 0,
              "original_weight_bytes": 0, "quantized_parameter_bytes": 0,
              "lora_parameters": 0, "rank": rank, "alpha": alpha,
              "double_quant": double_quant, "dropout": dropout}
    kwargs = dict(rank=rank, alpha=alpha, dropout=dropout,
                  compute_dtype=compute_dtype, double_quant=double_quant)
    for parent, name, module in matches:
        original_bytes = sum(p.numel()*p.element_size() for p in module.parameters())
        replacement = QuantizedPackedExperts(module, **kwargs)
        setattr(parent, name, replacement)
        report["banks"] += 1
        report["experts"] += replacement.num_experts
        report["original_weight_bytes"] += original_bytes
        for n, p in replacement.named_parameters():
            if "lora_" in n:
                report["lora_parameters"] += p.numel()
            else:
                report["quantized_parameter_bytes"] += p.numel()*p.element_size()
    llm.config.research_packed_quantization = dict(
        method="per_expert_nf4_lora", rank=rank, alpha=alpha, dropout=dropout,
        double_quant=double_quant, compute_dtype=str(compute_dtype),
    )
    return report


def set_packed_lora_trainable(llm, enabled=True):
    for module in llm.modules():
        if isinstance(module, NF4LoRALinear):
            module.base.requires_grad_(False)
            if module.lora_A is not None:
                module.lora_A.requires_grad_(enabled)
                module.lora_B.requires_grad_(enabled)
