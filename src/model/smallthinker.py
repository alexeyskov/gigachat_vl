"""Checkpoint-compatible SmallThinker for the project's native Transformers stack.

Architecture reference (Apache-2.0):
https://huggingface.co/Tiiny/SmallThinker-4BA0.6B-Instruct

The published implementation targets Transformers 4.53 and does not compute a
training loss. Reuse native Llama attention, caching and causal loss, preserving
SmallThinker's checkpoint names, ReGLU experts and pre-attention router input.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


class SmallThinkerConfig(LlamaConfig):
    model_type = "smallthinker"
    # Llama's tensor-parallel plans address an MLP that SmallThinker does not have.
    base_model_tp_plan = {}
    base_model_pp_plan = {}

    def __init__(
        self,
        moe_ffn_hidden_size=768,
        moe_num_primary_experts=32,
        moe_num_active_primary_experts=4,
        moe_primary_router_apply_softmax=False,
        rope_layout=None,
        sliding_window_layout=None,
        sliding_window_size=4096,
        **kwargs,
    ):
        defaults = dict(
            vocab_size=151936, hidden_size=1536, num_hidden_layers=32,
            num_attention_heads=12, num_key_value_heads=2, head_dim=128,
            max_position_embeddings=32768, tie_word_embeddings=True,
            bos_token_id=151643, eos_token_id=[151643, 151645],
        )
        defaults.update(kwargs)
        defaults["intermediate_size"] = moe_ffn_hidden_size
        defaults["hidden_act"] = "relu"
        defaults.pop("model_type", None)
        defaults.pop("auto_map", None)
        if "rope_parameters" not in defaults:
            defaults.setdefault("rope_theta", 1_500_000.0)
        super().__init__(**defaults)
        self.moe_ffn_hidden_size = moe_ffn_hidden_size
        self.moe_num_primary_experts = moe_num_primary_experts
        self.moe_num_active_primary_experts = moe_num_active_primary_experts
        self.moe_primary_router_apply_softmax = moe_primary_router_apply_softmax
        self.rope_layout = [1] * self.num_hidden_layers if rope_layout is None else rope_layout
        self.sliding_window_layout = (
            [0] * self.num_hidden_layers if sliding_window_layout is None else sliding_window_layout
        )
        self.sliding_window_size = sliding_window_size
        if len(self.rope_layout) != self.num_hidden_layers or not all(self.rope_layout):
            raise ValueError("Native SmallThinker requires RoPE in every layer, as in the 4B checkpoint.")
        if len(self.sliding_window_layout) != self.num_hidden_layers or any(self.sliding_window_layout):
            raise ValueError("Native SmallThinker supports the 4B checkpoint's full-attention layout.")
        if not 1 <= moe_num_active_primary_experts <= moe_num_primary_experts:
            raise ValueError("Invalid SmallThinker expert counts.")


class SmallThinkerExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.moe_ffn_hidden_size, bias=False)
        self.gate = nn.Linear(config.hidden_size, config.moe_ffn_hidden_size, bias=False)
        self.down = nn.Linear(config.moe_ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down(self.up(hidden_states) * F.relu(self.gate(hidden_states)))


class SmallThinkerMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.primary_router = nn.Linear(config.hidden_size, config.moe_num_primary_experts, bias=False)
        self.experts = nn.ModuleList(
            SmallThinkerExpert(config) for _ in range(config.moe_num_primary_experts)
        )

    def forward(self, router_input, hidden_states):
        shape = hidden_states.shape
        states = hidden_states.reshape(-1, shape[-1])
        logits = self.primary_router(router_input.reshape(-1, shape[-1]))
        selected_logits, indices = logits.topk(self.config.moe_num_active_primary_experts, dim=-1)
        if self.config.moe_primary_router_apply_softmax:
            weights = selected_logits.softmax(dim=-1, dtype=torch.float32)
        else:
            weights = selected_logits.sigmoid()
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(states.dtype)
        output = torch.zeros_like(states)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, slot = torch.where(indices == expert_idx)
            if token_idx.numel():
                delta = expert(states[token_idx]) * weights[token_idx, slot, None]
                output.index_add_(0, token_idx, delta.to(output.dtype))
        return output.reshape(shape), selected_logits


class SmallThinkerDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.block_sparse_moe = SmallThinkerMoeBlock(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, **kwargs):
        router_input = hidden_states
        attention, _ = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        hidden_states = hidden_states + attention
        delta, _ = self.block_sparse_moe(router_input, self.post_attention_layernorm(hidden_states))
        return hidden_states + delta


class SmallThinkerModel(LlamaModel):
    config_class = SmallThinkerConfig
    _no_split_modules = ["SmallThinkerDecoderLayer"]
    _can_record_outputs = {"hidden_states": SmallThinkerDecoderLayer, "attentions": LlamaAttention}

    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            SmallThinkerDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class SmallThinkerForCausalLM(LlamaForCausalLM):
    config_class = SmallThinkerConfig
    _no_split_modules = ["SmallThinkerDecoderLayer"]
    _can_record_outputs = SmallThinkerModel._can_record_outputs

    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = SmallThinkerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


def register_smallthinker():
    AutoConfig.register("smallthinker", SmallThinkerConfig, exist_ok=True)
    AutoModelForCausalLM.register(SmallThinkerConfig, SmallThinkerForCausalLM, exist_ok=True)
