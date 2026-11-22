"""Input gates for the native Transformers 5.15 DeepSeek-V2 FFNs.

The pretrained FFN remains owned by its caller. This module owns only gate
parameters, so wrapping a model does not duplicate or rename its base weights.
Routing uses the unmodified native router; gates act on image-origin positions
after dispatch. They do not remove visual information already mixed into text
positions by attention.

The implementation supports unquantized packed routed experts and ordinary,
bias-free shared/dense projections. LoRA, quantized projections, expert parallel
dispatch, and custom FFN implementations require separate integrations.
"""

import math
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


class NativeDeepseekVisualGates(nn.Module):
    """Independent scalar hard-concrete input gates for DeepSeek-V2 experts.

    ``init_open`` is the initial deterministic gate value, not the probability
    used by the L0 penalty. Training samples one gate per expert per forward;
    evaluation uses the stretched, clipped sigmoid. Both can be exactly zero.
    """

    temperature = 2.0 / 3.0
    stretch_low = -0.1
    stretch_high = 1.1

    def __init__(self, base_mlp: nn.Module, config: Any, init_open: float = 0.9):
        super().__init__()
        if not math.isfinite(init_open) or not 0.0 < init_open <= 1.0:
            raise ValueError("init_open must be finite and in (0, 1].")

        self.hidden_size = int(config.hidden_size)
        self.is_moe = hasattr(base_mlp, "experts")
        self.num_routed_experts = 0
        self.num_shared_experts = 0
        self.expert_intermediate_size = 0
        self.num_experts_per_token = 0
        if self.is_moe:
            self.num_routed_experts = int(
                getattr(config, "num_experts", None)
                or getattr(config, "n_routed_experts", 0)
            )
            self.num_shared_experts = int(getattr(config, "n_shared_experts", 0) or 0)
            self.expert_intermediate_size = int(config.moe_intermediate_size)
            self.num_experts_per_token = int(config.num_experts_per_tok)
            if self.num_routed_experts < 1 or self.num_shared_experts < 0:
                raise ValueError("Invalid routed/shared expert counts in DeepSeek-V2 config.")
            if not 1 <= self.num_experts_per_token <= self.num_routed_experts:
                raise ValueError("num_experts_per_tok must be between 1 and the routed expert count.")

        self._validate_base(base_mlp)
        parameter = next(base_mlp.parameters())
        unscaled = (float(init_open) - self.stretch_low) / (
            self.stretch_high - self.stretch_low
        )
        initial_logit = math.log(unscaled / (1.0 - unscaled))
        for name, count in (
            ("routed_logits", self.num_routed_experts),
            ("shared_logits", self.num_shared_experts),
            ("dense_logits", 0 if self.is_moe else 1),
        ):
            self.register_parameter(
                name,
                nn.Parameter(
                    torch.full((count,), initial_logit, device=parameter.device, dtype=torch.float32)
                ) if count else None,
            )

    @staticmethod
    def _validate_ffn(ffn: nn.Module, hidden_size: int, intermediate_size: int, name: str) -> None:
        shapes = {
            "gate_proj": (intermediate_size, hidden_size),
            "up_proj": (intermediate_size, hidden_size),
            "down_proj": (hidden_size, intermediate_size),
        }
        for projection_name, expected_shape in shapes.items():
            projection = getattr(ffn, projection_name, None)
            if type(projection) is not nn.Linear:
                raise TypeError(
                    f"{name}.{projection_name} must be an ordinary nn.Linear; "
                    "LoRA, quantized, and custom projections are unsupported by visual input gates."
                )
            if projection.bias is not None:
                raise ValueError(f"{name}.{projection_name} must be bias-free.")
            if type(projection.weight) is not nn.Parameter or not projection.weight.is_floating_point():
                raise TypeError(f"{name}.{projection_name} must have an unquantized floating Parameter.")
            if tuple(projection.weight.shape) != expected_shape:
                raise ValueError(
                    f"{name}.{projection_name} has shape {tuple(projection.weight.shape)}; "
                    f"expected {expected_shape}."
                )
        if not callable(getattr(ffn, "act_fn", None)):
            raise TypeError(f"{name} must expose a callable act_fn.")

    def _validate_base(self, base_mlp: nn.Module) -> None:
        if not self.is_moe:
            intermediate_size = getattr(base_mlp, "intermediate_size", None)
            if intermediate_size is None:
                raise TypeError("Expected a native DeepSeek-V2 dense FFN with intermediate_size.")
            self._validate_ffn(base_mlp, self.hidden_size, int(intermediate_size), "dense FFN")
            return

        if int(getattr(base_mlp, "ep_size", 1)) != 1:
            raise ValueError("Visual input gates do not support expert-parallel dispatch.")
        experts = getattr(base_mlp, "experts", None)
        expected_shapes = {
            "gate_up_proj": (
                self.num_routed_experts, 2 * self.expert_intermediate_size, self.hidden_size
            ),
            "down_proj": (
                self.num_routed_experts, self.hidden_size, self.expert_intermediate_size
            ),
        }
        for name, expected_shape in expected_shapes.items():
            parameter = getattr(experts, name, None)
            if type(parameter) is not nn.Parameter or not parameter.is_floating_point():
                raise TypeError(
                    "Visual input gates require native Transformers 5.15 DeepseekV2 packed "
                    f"experts with an unquantized {name} Parameter."
                )
            if tuple(parameter.shape) != expected_shape:
                raise ValueError(
                    f"Packed expert {name} has shape {tuple(parameter.shape)}; "
                    f"expected {expected_shape}."
                )
        if not callable(getattr(experts, "act_fn", None)):
            raise TypeError("Packed experts must expose a callable act_fn.")
        if not callable(getattr(base_mlp, "gate", None)):
            raise TypeError("Native DeepSeek-V2 MoE must expose its gate router.")
        if self.num_shared_experts:
            self._validate_ffn(
                getattr(base_mlp, "shared_experts", None),
                self.hidden_size,
                self.num_shared_experts * self.expert_intermediate_size,
                "shared FFN",
            )
        elif getattr(base_mlp, "shared_experts", None) is not None:
            raise ValueError("The config declares no shared experts but the FFN has a shared branch.")

    def _logit_groups(self):
        for name in ("routed", "shared", "dense"):
            logits = getattr(self, f"{name}_logits")
            if logits is not None:
                yield name, logits

    def _gate(self, logits: torch.Tensor, sample: bool) -> torch.Tensor:
        logits = logits.float()
        if sample:
            eps = torch.finfo(logits.dtype).eps
            uniform = torch.rand_like(logits).clamp(min=eps, max=1.0 - eps)
            logits = (logits + uniform.log() - torch.log1p(-uniform)) / self.temperature
        return (
            logits.sigmoid() * (self.stretch_high - self.stretch_low) + self.stretch_low
        ).clamp(0.0, 1.0)

    def regularization_loss(self) -> torch.Tensor:
        """Differentiable mean probability that a gate is nonzero (expected L0)."""
        offset = self.temperature * math.log(-self.stretch_low / self.stretch_high)
        probabilities = [torch.sigmoid(logits.float() - offset) for _, logits in self._logit_groups()]
        return torch.cat(probabilities).mean()

    def gate_values(self) -> Dict[str, torch.Tensor]:
        """Deterministic gate values, detached for diagnostics regardless of mode."""
        return {name: self._gate(logits, sample=False).detach() for name, logits in self._logit_groups()}

    @staticmethod
    def _mask_input(hidden: torch.Tensor, visual_mask: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return torch.where(
            visual_mask.unsqueeze(-1),
            hidden * gate.to(device=hidden.device, dtype=hidden.dtype),
            hidden,
        )

    def forward(
        self,
        base_mlp: nn.Module,
        hidden_states: torch.Tensor,
        visual_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Check again because PEFT/quantization may replace projections after installation.
        self._validate_base(base_mlp)
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden states ending in hidden_size={self.hidden_size}.")
        if visual_token_mask is None:
            return base_mlp(hidden_states)
        if visual_token_mask.dtype != torch.bool:
            raise TypeError("visual_token_mask must be a boolean origin mask.")
        if tuple(visual_token_mask.shape) != tuple(hidden_states.shape[:-1]):
            raise ValueError("visual_token_mask must match the hidden-state token dimensions.")
        mask = visual_token_mask.to(device=hidden_states.device)
        if not torch.any(mask):
            return base_mlp(hidden_states)

        if not self.is_moe:
            gated_input = self._mask_input(hidden_states, mask, self._gate(self.dense_logits, self.training)[0])
            return base_mlp(gated_input)

        # Router selection and weights are unchanged by the visual input gates.
        routing = base_mlp.gate(hidden_states)
        if not isinstance(routing, tuple) or len(routing) != 3:
            raise TypeError("Expected native DeepSeek-V2 router output (logits, weights, indices).")
        router_logits, topk_weights, topk_indices = routing
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        flat_mask = mask.reshape(-1)
        num_tokens = flat_hidden.shape[0]
        if not all(torch.is_tensor(value) for value in routing):
            raise TypeError("Native DeepSeek-V2 router outputs must all be tensors.")
        if (
            tuple(router_logits.shape) != (num_tokens, self.num_routed_experts)
            or topk_indices.ndim != 2
            or topk_indices.shape[0] != num_tokens
            or topk_indices.shape != topk_weights.shape
            or topk_indices.shape[1] != self.num_experts_per_token
            or topk_indices.dtype not in (torch.int32, torch.int64)
            or not topk_weights.is_floating_point()
        ):
            raise ValueError("Unsupported native DeepSeek-V2 router output shapes or dtypes.")
        if torch.any(topk_indices < 0) or torch.any(topk_indices >= self.num_routed_experts):
            raise ValueError("DeepSeek-V2 router selected an out-of-range expert.")

        routed_gates = self._gate(self.routed_logits, self.training)
        experts = base_mlp.experts
        output = torch.zeros_like(flat_hidden)
        expert_mask = F.one_hot(topk_indices.long(), num_classes=self.num_routed_experts).permute(2, 1, 0)
        active_experts = expert_mask.sum(dim=(-1, -2)).nonzero(as_tuple=False).flatten()
        for expert_idx in active_experts:
            topk_pos, token_idx = torch.where(expert_mask[expert_idx])
            selected = self._mask_input(flat_hidden[token_idx], flat_mask[token_idx], routed_gates[expert_idx])
            gate, up = F.linear(selected, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            expert_output = F.linear(experts.act_fn(gate) * up, experts.down_proj[expert_idx])
            weighted = expert_output * topk_weights[token_idx, topk_pos, None]
            output.index_add_(0, token_idx, weighted.to(output.dtype))

        if self.num_shared_experts:
            shared = base_mlp.shared_experts
            shared_gates = self._gate(self.shared_logits, self.training)
            for expert_idx in range(self.num_shared_experts):
                start = expert_idx * self.expert_intermediate_size
                end = start + self.expert_intermediate_size
                selected = self._mask_input(flat_hidden, flat_mask, shared_gates[expert_idx])
                gate = F.linear(selected, shared.gate_proj.weight[start:end])
                up = F.linear(selected, shared.up_proj.weight[start:end])
                output = output + F.linear(shared.act_fn(gate) * up, shared.down_proj.weight[:, start:end])

        return output.reshape_as(hidden_states)
