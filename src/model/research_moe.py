"""Explicit E+K MoE routing for the research matrix.

SmallThinker, LFM2-MoE, DeepSeek-V3 and GPT-OSS have explicit expert/routing
contracts. Other architectures are rejected rather than quietly running the older
residual-only visual FFN. Install after loading the language checkpoint, before
constructing the optimizer. Keep modality masks set through backward: gradient
checkpoint recomputation reads the same masks.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F


def _linear_float(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Use fp32 scoring without converting the stored trainable parameters."""
    if isinstance(module, nn.Linear):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(
                x.float(), module.weight.float(),
                None if module.bias is None else module.bias.float(),
            )
    return module(x).float()


class RouterMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, width: int):
        super().__init__()
        self.up = nn.Linear(input_size, width)
        self.down = nn.Linear(width, output_size)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, x):
        return _linear_float(self.down, F.gelu(_linear_float(self.up, x)))


class ResearchSmallThinkerMoE(nn.Module):
    """Retains pre-attention router input and the native sigmoid/top-k rule.

    ``family=1`` adds residual router logits to a frozen original router;
    ``family=2`` replaces it; ``family=3`` allocates mass using the old full
    distribution. ``policy`` is a/b/c as defined in RESEARCH_PIPELINE.md.
    """

    def __init__(self, base: nn.Module, family: int, policy: str,
                 added_experts: int = 2, router_width: int = 128,
                 gate_initial: float = 0.9, initial_new_mass: float = 0.05):
        super().__init__()
        if family not in (1, 2, 3):
            raise ValueError("Research MoE family must be 1, 2 or 3.")
        if policy not in ("a", "b", "c"):
            raise ValueError("Modality policy must be a, b or c.")
        if added_experts < 0:
            raise ValueError("added_experts must be nonnegative.")
        if policy == "a" and added_experts == 0:
            raise ValueError("Hard separation needs at least one new expert.")
        if not (0 < gate_initial < 1 and 0 < initial_new_mass < 1):
            raise ValueError("Gate and group-mass initialization must be inside (0,1).")
        if not hasattr(base, "primary_router") or not isinstance(base.experts, nn.ModuleList):
            raise TypeError("Expected the native SmallThinker MoE block.")
        self.config = base.config
        self.family, self.policy = family, policy
        self.old_count = len(base.experts)
        self.added_count = added_experts
        self.top_k = int(base.config.moe_num_active_primary_experts)
        self.apply_softmax = bool(base.config.moe_primary_router_apply_softmax)
        self.hidden_size = int(base.config.hidden_size)
        self.router_width = router_width
        self.use_grouped_mm = True
        self.experts = base.experts
        reference = next(base.primary_router.parameters())
        kwargs = {"device": reference.device, "dtype": reference.dtype}
        # New experts are fresh ReGLU FFNs, never deepcopy quantized old weights.
        from src.model.smallthinker import SmallThinkerExpert
        self.visual_experts = nn.ModuleList([
            SmallThinkerExpert(base.config).to(**kwargs) for _ in range(added_experts)
        ])
        for expert in self.visual_experts:
            for p in expert.parameters():
                nn.init.normal_(p, mean=0.0, std=float(getattr(base.config, "initializer_range", 0.02)))
        total = self.old_count + self.added_count
        if family == 2:
            self.primary_router = nn.Linear(self.hidden_size, total, bias=False).to(**kwargs)
            nn.init.normal_(self.primary_router.weight, std=float(getattr(base.config, "initializer_range", 0.02)))
        else:
            self.primary_router = base.primary_router
        if family == 1 and added_experts:
            self.router_adapter = RouterMLP(self.hidden_size, total, router_width).to(**kwargs)
            self.primary_router.requires_grad_(False)
        elif family == 3 and added_experts:
            self.router_new = RouterMLP(self.hidden_size + self.old_count, added_experts, router_width).to(**kwargs)
            self.router_mass = RouterMLP(self.hidden_size + self.old_count, 1, router_width).to(**kwargs)
            nn.init.constant_(self.router_mass.down.bias, math.log(initial_new_mass / (1 - initial_new_mass)))
        if policy == "c":
            init = math.log(gate_initial / (1 - gate_initial))
            self.modality_gate_logits = nn.Parameter(torch.full((total, 2), init, device=reference.device, dtype=torch.float32))
        else:
            self.register_parameter("modality_gate_logits", None)
        # These are per-batch runtime data, intentionally excluded from state_dict.
        self._modality_mask = None
        self._attention_mask = None
        self.warmup_force_new = False
        self._diagnostic_counts = None
        self._diagnostic_mass = None
        self._diagnostic_alpha = None
        self.last_indices = None
        self.last_weights = None

    def set_modality_mask(self, mask, attention_mask=None):
        if mask is not None and mask.ndim not in (1, 2):
            raise ValueError("Modality mask must have shape [tokens] or [batch, sequence].")
        if attention_mask is not None and mask is not None and attention_mask.shape != mask.shape:
            raise ValueError("Attention and modality masks must have identical shapes.")
        self._modality_mask = None if mask is None else mask.detach().bool()
        self._attention_mask = None if attention_mask is None else attention_mask.detach().bool()

    def _resolve_one_mask(self, value, hidden_states, *, default):
        shape = hidden_states.shape[:-1]
        if value is None:
            return torch.full(shape, default, device=hidden_states.device, dtype=torch.bool).flatten()
        value = value.to(device=hidden_states.device)
        if tuple(value.shape) == tuple(shape) or value.numel() == hidden_states.numel() // hidden_states.shape[-1]:
            return value.reshape(-1)
        if len(shape) == 2 and value.ndim == 2 and shape[0] % value.shape[0] == 0:
            value = value.repeat_interleave(shape[0] // value.shape[0], dim=0)
            if shape[1] >= value.shape[1]:
                return F.pad(value, (0, shape[1] - value.shape[1]), value=default).reshape(-1)
            if not self.training:
                # Cached generation receives just newly generated text positions.
                return torch.full(shape, default, device=hidden_states.device, dtype=torch.bool).flatten()
        raise ValueError(f"Stale modality/attention mask {tuple(value.shape)} for hidden states {tuple(shape)}.")

    def _native_scores(self, logits):
        return logits.softmax(-1) if self.apply_softmax else logits.sigmoid()

    def routing_scores(self, router_input):
        old_or_new_logits = self._router_logits(router_input)
        alpha = None
        if self.family == 1 and self.added_count:
            correction = self.router_adapter(router_input)
            logits = torch.cat((old_or_new_logits + correction[:, :self.old_count],
                                correction[:, self.old_count:]), dim=-1)
            scores = self._native_scores(logits)
        elif self.family == 3 and self.added_count:
            scores_old = self._native_scores(old_or_new_logits)
            p_old = scores_old / scores_old.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
            extended_input = torch.cat((router_input.float(), p_old), dim=-1)
            q_new = self.router_new(extended_input).softmax(-1)
            alpha = self.router_mass(extended_input).sigmoid()
            scores = torch.cat(((1-alpha) * p_old, alpha * q_new), dim=-1)
            logits = scores.clamp_min(torch.finfo(torch.float32).tiny).log()
        else:
            logits = old_or_new_logits
            scores = self._native_scores(logits)
        return scores, logits, alpha

    def _router_logits(self, inputs):
        return _linear_float(self.primary_router, inputs)

    def _selection_scores(self, scores, logits, alpha, inputs):
        return logits

    def _routing_weights(self, scores, logits, indices, slot_valid):
        if self.family != 3 or not self.added_count:
            if self.apply_softmax:
                safe_logits = logits.masked_fill(~slot_valid, -torch.inf)
                safe_logits = torch.where(slot_valid.any(-1, keepdim=True), safe_logits, 0.0)
                weights = safe_logits.softmax(-1)
                return torch.where(slot_valid, weights, 0.0)
            weights = logits.sigmoid() * slot_valid
        else:
            weights = scores.gather(-1, indices) * slot_valid
        return weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)

    def _run_expert(self, expert_index, x):
        if expert_index < self.old_count:
            return self.experts[expert_index](x)
        return self.visual_experts[expert_index-self.old_count](x)

    def _shared_output(self, states, visual, valid):
        return None

    def _grouped_original_dispatch(self, states, visual, weights, order, counts, k):
        from src.model.smallthinker import smallthinker_grouped_dispatch
        return smallthinker_grouped_dispatch(self, self.experts, states, weights,
                                             order, counts, visual, self.modality_gate_logits)

    def forward(self, router_input, hidden_states):
        shape = hidden_states.shape
        states = hidden_states.reshape(-1, shape[-1])
        inputs = router_input.reshape(-1, router_input.shape[-1])
        visual = self._resolve_one_mask(self._modality_mask, hidden_states, default=False)
        valid = self._resolve_one_mask(self._attention_mask, hidden_states, default=True)
        scores, logits, alpha = self.routing_scores(inputs)
        eligible = torch.ones_like(scores, dtype=torch.bool)
        if self.policy == "a":
            eligible[:, :self.old_count] = ~visual[:, None]
            eligible[:, self.old_count:] = visual[:, None]
        eligible &= valid[:, None]
        k = min(self.top_k, scores.shape[-1])
        # Select by logits where possible to avoid sigmoid saturation ties.
        chosen = self._selection_scores(scores, logits, alpha, inputs).masked_fill(~eligible, -torch.inf)
        _, indices = chosen.topk(k, dim=-1)
        selected_logits = logits.gather(-1, indices)
        slot_valid = eligible.gather(-1, indices)
        if self.training and self.warmup_force_new and self.added_count and self.policy != "a":
            # A1-only exploration avoids permanently dead new experts under top-k.
            # Replace at most the final slot if top-k selected only old experts.
            need_new = valid & ~((indices >= self.old_count) & slot_valid).any(-1)
            best_new = chosen[:, self.old_count:].argmax(-1) + self.old_count
            # Rotate the new expert choice per token to break identical-init ties.
            if self.added_count > 1:
                best_new = (best_new-self.old_count + torch.arange(len(states), device=states.device) % self.added_count) % self.added_count + self.old_count
            indices = indices.clone()
            indices[:, -1] = torch.where(need_new, best_new, indices[:, -1])
            selected_logits = logits.gather(-1, indices)
            slot_valid = eligible.gather(-1, indices)
        weights = self._routing_weights(scores, selected_logits, indices, slot_valid)
        weights = weights.to(states.dtype)
        output = torch.zeros_like(states)
        total_experts = self.old_count + self.added_count
        gates = self.modality_gate_logits.sigmoid() if self.modality_gate_logits is not None else None
        # One host synchronization per layer (bucket lengths), rather than a
        # CUDA nonzero()/where synchronization for each of ~34 experts.
        buckets = torch.where(slot_valid, indices, total_experts).reshape(-1)
        order = buckets.argsort(stable=True)
        counts_per_expert = torch.bincount(buckets, minlength=total_experts+1).tolist()
        grouped_output = self._grouped_original_dispatch(states, visual, weights, order, counts_per_expert, k)
        if grouped_output is not None:
            output = grouped_output
        offset = 0
        for expert_idx in range(total_experts):
            count = counts_per_expert[expert_idx]
            if count and not (grouped_output is not None and expert_idx < self.old_count):
                selected_slots = order[offset:offset+count]
                token_idx = selected_slots // k
                slot = selected_slots % k
                x = states[token_idx]
                if gates is None:
                    delta = self._run_expert(expert_idx, x)
                else:
                    g = gates[expert_idx, visual[token_idx].long()].to(states.dtype)[:, None]
                    delta = g * self._run_expert(expert_idx, g * x)
                output.index_add_(0, token_idx, (delta * weights[token_idx, slot, None]).to(output.dtype))
            offset += count
        shared = self._shared_output(states, visual, valid)
        if shared is not None:
            output = output + shared
        # Diagnostic tensors stay on GPU; synchronize only when diagnostics() is called.
        with torch.no_grad():
            self.last_indices = indices.detach()
            self.last_weights = weights.detach()
            group = visual.long()[:, None].expand_as(indices)
            counts = torch.zeros((2, total_experts), device=states.device, dtype=torch.int64)
            counts.view(-1).scatter_add_(0, (group * total_experts + indices).reshape(-1), slot_valid.long().reshape(-1))
            self._diagnostic_counts = counts
            p_full = scores / scores.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
            mass = torch.stack([(p_full * (valid & (visual == bool(v)))[:, None]).sum(0) for v in range(2)])
            self._diagnostic_mass = mass
            self._diagnostic_alpha = None if alpha is None else torch.stack([
                (alpha * (valid & (visual == bool(v)))[:, None]).sum() for v in range(2)
            ])
        return output.reshape(shape), weights if getattr(self, "gpt_style", False) else selected_logits

    def diagnostics(self):
        return {
            "family": self.family, "policy": self.policy,
            "old_experts": self.old_count, "new_experts": self.added_count,
            "dispatch_counts_text_visual": None if self._diagnostic_counts is None else self._diagnostic_counts.cpu().tolist(),
            "probability_mass_text_visual": None if self._diagnostic_mass is None else self._diagnostic_mass.cpu().tolist(),
            "new_group_alpha_sum_text_visual": None if self._diagnostic_alpha is None else self._diagnostic_alpha.cpu().tolist(),
            "modality_gates_text_visual": None if self.modality_gate_logits is None else self.modality_gate_logits.detach().sigmoid().cpu().tolist(),
        }

    def spec(self):
        return {"backend": "smallthinker", "family": self.family, "policy": self.policy,
                "added_experts": self.added_count, "router_width": self.router_width}


class PackedVisualExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, act_fn):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class GptOssActivation(nn.Module):
    """Native GPT-OSS interleaved, clipped SwiGLU (including the up + 1)."""
    def __init__(self, alpha=1.702, limit=7.0):
        super().__init__()
        self.alpha, self.limit = alpha, limit

    def forward(self, gate_up):
        gate = gate_up[..., ::2].clamp(max=self.limit)
        up = gate_up[..., 1::2].clamp(min=-self.limit, max=self.limit)
        return (up+1) * gate * torch.sigmoid(gate*self.alpha)


class GptOssVisualExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, alpha=1.702, limit=7.0):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size*2, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act_fn = GptOssActivation(alpha, limit)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class ResearchPackedMoE(ResearchSmallThinkerMoE):
    """LFM2-MoE, DeepSeek-V3 and GPT-OSS dispatch retaining packed parameters.

    Reads expert weight *views*, so full-training gradients flow into the original
    three-dimensional tensors. No unpacked copies or quantization claims are made.
    Requested GigaChat3.1 uses one routing group. Multi-group E+K expansion is
    rejected because an arbitrary K would change expert-group assignments.
    """

    def __init__(self, base, config, family, policy, added_experts=2,
                 router_width=128, gate_initial=0.9, initial_new_mass=0.05):
        nn.Module.__init__(self)
        if family not in (1, 2, 3) or policy not in ("a", "b", "c"):
            raise ValueError("Expected research family 1/2/3 and policy a/b/c.")
        if added_experts < 0 or (policy == "a" and added_experts == 0):
            raise ValueError("Invalid number of added experts for this policy.")
        if not (0 < gate_initial < 1 and 0 < initial_new_mass < 1):
            raise ValueError("Gate initialization must be in (0,1).")
        self.config = config
        self.backend = config.model_type
        self.gpt_style = self.backend == "gpt_oss"
        native_router = base.router if self.gpt_style else base.gate
        self.family, self.policy = family, policy
        self.router_width = router_width
        self.experts = base.experts
        self.old_count = int(base.experts.num_experts)
        self.added_count = added_experts
        self.hidden_size = int(config.hidden_size)
        self.top_k = int(native_router.top_k)
        self.apply_softmax = self.gpt_style
        self.norm_topk_prob = bool(getattr(native_router, "norm_topk_prob", True))
        self.routed_scaling_factor = float(getattr(native_router, "routed_scaling_factor", 1.0))
        self.normalization_epsilon = 1e-6 if self.backend == "lfm2_moe" else 1e-20
        self.num_group = int(getattr(native_router, "num_group", 1))
        self.topk_group = int(getattr(native_router, "topk_group", 1))
        if self.num_group != 1 and added_experts:
            raise NotImplementedError("E+K expansion of grouped DeepSeek routing requires a separately defined group allocation; requested GigaChat3.1 uses n_group=1.")
        reference = native_router.weight
        device_dtype = {"device": reference.device, "dtype": reference.dtype}
        if self.gpt_style:
            self.gpt_activation = GptOssActivation(base.experts.alpha, base.experts.limit)
            self.visual_experts = nn.ModuleList([
                GptOssVisualExpert(self.hidden_size, config.intermediate_size, base.experts.alpha, base.experts.limit).to(**device_dtype)
                for _ in range(added_experts)
            ])
        else:
            self.visual_experts = nn.ModuleList([
                PackedVisualExpert(self.hidden_size, config.moe_intermediate_size, base.experts.act_fn).to(**device_dtype)
                for _ in range(added_experts)
            ])
        for expert in self.visual_experts:
            for p in expert.parameters():
                nn.init.normal_(p, std=float(config.initializer_range))
        total = self.old_count + added_experts
        if family == 2:
            self.primary_router = nn.Linear(self.hidden_size, total, bias=self.gpt_style).to(**device_dtype)
            nn.init.normal_(self.primary_router.weight, std=float(config.initializer_range))
            if self.primary_router.bias is not None:
                nn.init.zeros_(self.primary_router.bias)
        else:
            self.primary_router = native_router
        if self.gpt_style:
            bias = torch.zeros(self.old_count, device=reference.device)
        elif self.backend == "lfm2_moe":
            bias = getattr(base, "expert_bias", torch.zeros(self.old_count, device=reference.device))
        else:
            bias = native_router.e_score_correction_bias
        old_bias = torch.zeros_like(bias) if family == 2 else bias.detach().clone()
        self.register_buffer("routing_bias", torch.cat((old_bias.float(), torch.zeros(added_experts, device=reference.device))))
        if family == 1 and added_experts:
            self.router_adapter = RouterMLP(self.hidden_size, total, router_width).to(**device_dtype)
            self.primary_router.requires_grad_(False)
        elif family == 3 and added_experts:
            self.router_new = RouterMLP(self.hidden_size+self.old_count, added_experts, router_width).to(**device_dtype)
            self.router_mass = RouterMLP(self.hidden_size+self.old_count, 1, router_width).to(**device_dtype)
            nn.init.constant_(self.router_mass.down.bias, math.log(initial_new_mass/(1-initial_new_mass)))
        self.shared_experts = getattr(base, "shared_experts", None)
        self.shared_count = int(getattr(config, "n_shared_experts", 0) or 0) if self.shared_experts is not None else 0
        if (policy == "c" and self.shared_count > 1 and
                self.shared_experts.gate_proj.weight.dtype == torch.uint8):
            raise NotImplementedError("Separate gates for multiple packed quantized shared experts require separate projections; requested GigaChat3.1 has one shared expert.")
        if policy == "c":
            self.modality_gate_logits = nn.Parameter(torch.full(
                (total+self.shared_count, 2), math.log(gate_initial/(1-gate_initial)),
                device=reference.device, dtype=torch.float32,
            ))
        else:
            self.register_parameter("modality_gate_logits", None)
        self._modality_mask = self._attention_mask = None
        self.warmup_force_new = False
        self._diagnostic_counts = self._diagnostic_mass = self._diagnostic_alpha = None
        self.last_indices = self.last_weights = None
        self.use_grouped_mm = True

    def _router_logits(self, inputs):
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            bias = getattr(self.primary_router, "bias", None)
            return F.linear(inputs.float(), self.primary_router.weight.float(), None if bias is None else bias.float())

    def _selection_scores(self, scores, logits, alpha, inputs):
        if self.gpt_style:
            return logits
        bias = self.routing_bias
        if self.family == 3 and self.added_count:
            # Express the old correction in the same probability units as its
            # scaled old-group scores. New experts start with zero correction.
            old_sum = self._router_logits(inputs).sigmoid().sum(-1, keepdim=True)
            corrected_old = scores[:, :self.old_count] + (1-alpha)*bias[:self.old_count] / old_sum.clamp_min(1e-20)
            result = torch.cat((corrected_old, scores[:, self.old_count:]), dim=-1)
        else:
            result = scores + bias
        if self.num_group != 1:
            grouped = result.reshape(-1, self.num_group, self.old_count//self.num_group)
            group_score = grouped.topk(2, dim=-1).values.sum(-1)
            group_ids = group_score.topk(self.topk_group, dim=-1).indices
            groups_allowed = torch.zeros_like(group_score, dtype=torch.bool).scatter_(1, group_ids, True)
            allowed = groups_allowed[:, :, None].expand_as(grouped).reshape_as(result)
            result = result.masked_fill(~allowed, -torch.inf)
        return result

    def _routing_weights(self, scores, logits, indices, slot_valid):
        if self.gpt_style:
            return ResearchSmallThinkerMoE._routing_weights(self, scores, logits, indices, slot_valid)
        weights = scores.gather(-1, indices) * slot_valid
        if self.norm_topk_prob or self.policy == "a":
            weights = weights / (weights.sum(-1, keepdim=True) + self.normalization_epsilon)
        return weights * self.routed_scaling_factor

    def _run_expert(self, expert_index, x):
        if expert_index >= self.old_count:
            return self.visual_experts[expert_index-self.old_count](x)
        if hasattr(self.experts, "forward_expert"):
            return self.experts.forward_expert(expert_index, x)
        if self.gpt_style:
            gate_up = x@self.experts.gate_up_proj[expert_index] + self.experts.gate_up_proj_bias[expert_index]
            return self.gpt_activation(gate_up)@self.experts.down_proj[expert_index] + self.experts.down_proj_bias[expert_index]
        gate, up = F.linear(x, self.experts.gate_up_proj[expert_index]).chunk(2, dim=-1)
        return F.linear(self.experts.act_fn(gate)*up, self.experts.down_proj[expert_index])

    def _grouped_original_dispatch(self, states, visual, weights, order, counts, k):
        # Native packed bf16/fp16 weights are already in grouped-MM format.
        # NF4 banks intentionally use native bitsandbytes per-expert kernels.
        if states.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
            compute_dtype = torch.get_autocast_dtype("cuda")
        else:
            compute_dtype = states.dtype
        if (not self.use_grouped_mm or states.device.type != "cuda" or
                compute_dtype not in (torch.bfloat16, torch.float16) or
                not hasattr(self.experts, "gate_up_proj") or
                not hasattr(torch, "_grouped_mm")):
            return None
        count = sum(counts[:self.old_count])
        output = torch.zeros_like(states)
        if not count:
            return output
        selected_slots = order[:count]
        token, slot = selected_slots // k, selected_slots % k
        x = states[token]
        offsets = torch.tensor(counts[:self.old_count], device=states.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        if self.modality_gate_logits is not None or self.gpt_style:
            expert_ids = torch.repeat_interleave(
                torch.arange(self.old_count, device=states.device),
                torch.tensor(counts[:self.old_count], device=states.device),
                output_size=count,
            )
        g = None
        if self.modality_gate_logits is not None:
            g = self.modality_gate_logits.sigmoid()[expert_ids, visual[token].long()].to(states.dtype)[:, None]
            x = g*x
        # Cast differentiably under autocast, just as F.linear does. Never cache
        # these casts across optimizer steps: doing so would use stale weights.
        gate_up_weights = self.experts.gate_up_proj.to(compute_dtype)
        if not self.gpt_style:
            gate_up_weights = gate_up_weights.transpose(-1, -2)
        gate_up = torch._grouped_mm(x.to(compute_dtype), gate_up_weights, offs=offsets)
        if self.gpt_style:
            gate_up = gate_up + self.experts.gate_up_proj_bias[expert_ids]
            activated = self.gpt_activation(gate_up).to(compute_dtype).contiguous()
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            activated = (self.experts.act_fn(gate)*up).contiguous()
        down_weights = self.experts.down_proj.to(compute_dtype)
        if not self.gpt_style:
            down_weights = down_weights.transpose(-1, -2)
        delta = torch._grouped_mm(activated, down_weights, offs=offsets)
        if self.gpt_style:
            delta = delta + self.experts.down_proj_bias[expert_ids]
        if g is not None:
            delta = delta*g
        output.index_add_(0, token, (delta*weights[token, slot, None]).to(output.dtype))
        return output

    def _shared_output(self, states, visual, valid):
        if self.shared_experts is None:
            return None
        if self.policy != "c":
            allowed = valid & (~visual if self.policy == "a" else True)
            return self.shared_experts(states) * allowed[:, None]
        if self.shared_count == 1:
            # Use the original module call so native Linear4bit/PEFT layers retain
            # their own quantization/autograd contracts in QLoRA experiments.
            g = self.modality_gate_logits[self.old_count+self.added_count].sigmoid()[visual.long()].to(states.dtype)[:, None]
            g = g * valid[:, None]
            return g*self.shared_experts(g*states)
        output = torch.zeros_like(states)
        width = int(self.config.moe_intermediate_size)
        for shared_idx in range(self.shared_count):
            expert_idx = self.old_count+self.added_count+shared_idx
            g = self.modality_gate_logits[expert_idx].sigmoid()[visual.long()].to(states.dtype)[:, None]
            g = g * valid[:, None]
            start, end = shared_idx*width, (shared_idx+1)*width
            gate = F.linear(g*states, self.shared_experts.gate_proj.weight[start:end])
            up = F.linear(g*states, self.shared_experts.up_proj.weight[start:end])
            output = output + g*F.linear(self.shared_experts.act_fn(gate)*up, self.shared_experts.down_proj.weight[:, start:end])
        return output

    def forward(self, hidden_states):
        output, router_scores = super().forward(hidden_states, hidden_states)
        if self.gpt_style:
            return output, router_scores
        return output

    def spec(self):
        result = super().spec()
        result.update(backend=self.backend, shared_experts=self.shared_count,
                      norm_topk_prob=self.norm_topk_prob,
                      routed_scaling_factor=self.routed_scaling_factor,
                      grouped_mm_enabled=self.use_grouped_mm)
        return result


def iter_research_moe(llm) -> Iterator[ResearchSmallThinkerMoE]:
    for module in llm.modules():
        if isinstance(module, ResearchSmallThinkerMoE):
            yield module


def install_research_moe(llm, family, policy, added_experts=2, **kwargs):
    """Install supported E+K blocks atomically; return the installed wrappers."""
    if any(iter_research_moe(llm)):
        raise ValueError("Research MoE is already installed.")
    backend = getattr(llm.config, "model_type", None)
    if backend not in ("smallthinker", "lfm2_moe", "deepseek_v3", "gpt_oss"):
        raise NotImplementedError("Research E+K dispatch does not support this architecture; it must not silently fall back to residual FFNs.")
    block_type = {"smallthinker": "SmallThinkerMoeBlock", "lfm2_moe": "Lfm2MoeSparseMoeBlock", "deepseek_v3": "DeepseekV3MoE", "gpt_oss": "GptOssMLP"}[backend]
    matches = [(parent, name, child) for parent in llm.modules()
               for name, child in parent.named_children()
               if child.__class__.__name__ == block_type]
    if not matches:
        raise ValueError(f"No native {backend} MoE blocks found; install before residual visual wrappers.")
    def wrap(child):
        if backend == "smallthinker":
            return ResearchSmallThinkerMoE(child, int(family), policy, added_experts, **kwargs)
        return ResearchPackedMoE(child, llm.config, int(family), policy, added_experts, **kwargs)
    prepared = [(parent, name, wrap(child))
                for parent, name, child in matches]
    for parent, name, wrapper in prepared:
        setattr(parent, name, wrapper)
    llm.config.research_moe = {"family": int(family), "policy": policy,
                               "added_experts": added_experts, **kwargs}
    return [wrapper for _, _, wrapper in prepared]


def set_modality_mask(llm, mask, attention_mask=None):
    for wrapper in iter_research_moe(llm):
        wrapper.set_modality_mask(mask, attention_mask)


def configure_research_trainability(llm, stage="SFT"):
    """Set research block trainability after the caller's global base/LoRA policy.

    This deliberately does not enable old experts: the caller owns the full-vs-
    LoRA setup, and A1 must keep them frozen. Frozen quantized weights must never
    accidentally be enabled here.
    """
    if stage not in ("A0", "A1", "SFT", "INFERENCE"):
        raise ValueError("Unknown stage.")
    enabled = stage in ("A1", "SFT")
    for wrapper in iter_research_moe(llm):
        wrapper.visual_experts.requires_grad_(enabled)
        wrapper.primary_router.requires_grad_(enabled and wrapper.family != 1 and (wrapper.family == 2 or stage == "SFT"))
        for name in ("router_adapter", "router_new", "router_mass"):
            module = getattr(wrapper, name, None)
            if module is not None:
                module.requires_grad_(enabled)
        if wrapper.modality_gate_logits is not None:
            wrapper.modality_gate_logits.requires_grad_(enabled)
        wrapper.warmup_force_new = stage == "A1"


def set_router_warmup(llm, enabled=True):
    for wrapper in iter_research_moe(llm):
        wrapper.warmup_force_new = bool(enabled)
