"""Opt-in additive visual Transformers. The pretrained MoE is a separate branch.

No enlarged native top-k, replacement slots, or corrections to native logits.
Fixed and routed modes use the same causal Transformer block. Runtime caches
are request-local, excluded from checkpoints, and enabled only by generation().
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VisualBranchConfig:
    mode: str = "fixed"
    width: int = 256
    heads: int = 4
    blocks: int = 2
    top_k: int = 2
    expansion: int = 4
    learned_masks: bool = False
    gate_initial: float = .9

    def __post_init__(self):
        if self.mode not in ("fixed", "routed"):
            raise ValueError("mode must be fixed or routed")
        if any(type(v) is not int or v <= 0 for v in
               (self.width, self.heads, self.blocks, self.top_k, self.expansion)):
            raise ValueError("Positive integer branch dimensions required")
        if self.width % self.heads or self.top_k > self.blocks:
            raise ValueError("Invalid heads or extra-branch top-k")
        if not 0 < self.gate_initial < 1:
            raise ValueError("Gate initialization must be inside (0,1)")


class CausalVisualTransformer(nn.Module):
    """Sparse queries attend to the complete valid causal source, with positions.

    Position signals are sinusoidal at absolute token positions. This branch
    cannot see future answer tokens during teacher forcing. Gates weight source
    attention mass and query output, rather than touching the native experts.
    """
    def __init__(self, config):
        super().__init__()
        w = config.width
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(w)
        self.q = nn.Linear(w, w, bias=False)
        self.k = nn.Linear(w, w, bias=False)
        self.v = nn.Linear(w, w, bias=False)
        self.o = nn.Linear(w, w, bias=False)
        self.norm2 = nn.LayerNorm(w)
        self.ffn = nn.Sequential(nn.Linear(w, w*config.expansion), nn.GELU(),
                                 nn.Linear(w*config.expansion, w))

    def forward(self, context, valid, query_flat, query_positions, source_gates):
        batch, length, width = context.shape
        # A selected query is identified by its batch and absolute position.
        rows = query_flat
        pos = query_positions
        query = context[rows, pos]
        source = self.norm1(context)
        q = self.q(source[rows, pos]).reshape(-1, self.heads, width//self.heads)
        k = self.k(source).reshape(batch, length, self.heads, -1).transpose(1, 2)
        v = self.v(source).reshape(batch, length, self.heads, -1).transpose(1, 2)
        # One batched SDPA per expert, rather than a GPU synchronization for
        # every sample/expert pair. No [queries, full_sequence, width] copies.
        counts = torch.bincount(rows, minlength=batch)
        maximum = int(counts.max())
        offsets = counts.cumsum(0)-counts
        slot = torch.arange(len(rows), device=context.device)-offsets[rows]
        packed_q = q.new_zeros((batch, maximum, self.heads, width//self.heads))
        packed_q[rows, slot] = q
        positions = pos.new_full((batch, maximum), -1)
        positions[rows, slot] = pos
        eligible = ((torch.arange(length, device=context.device)[None, None, :] <= positions[:, :, None])
                    & valid[:, None, :])
        # The modality mask scales information from source tokens; output gates
        # separately scale the receiving query. Both remain differentiable.
        v = v * source_gates[:, None, :, None].to(v.dtype)
        attended = F.scaled_dot_product_attention(packed_q.transpose(1, 2),
            k, v, attn_mask=eligible[:, None], dropout_p=0.)
        output = attended.transpose(1, 2)[rows, slot].reshape(len(rows), width)
        hidden = query + self.o(output)
        return hidden + self.ffn(self.norm2(hidden))


class AdditiveVisualBank(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.config = config
        self.down = nn.Linear(hidden_size, config.width, bias=False)
        self.blocks = nn.ModuleList([CausalVisualTransformer(config) for _ in range(config.blocks)])
        self.up = nn.Linear(config.width, hidden_size, bias=False)
        # Exact original-model function at initialization, while the new branch
        # is still dispatched. The first update learns up; subsequent ones learn
        # the block/router/masks. No permanently zero gate suppresses the branch.
        nn.init.zeros_(self.up.weight)
        self.router = (nn.Sequential(nn.Linear(config.width+2, 128), nn.GELU(),
                                    nn.Linear(128, config.blocks)) if config.mode == "routed" else None)
        if self.router is not None:
            nn.init.normal_(self.router[-1].weight, std=.02)
            nn.init.zeros_(self.router[-1].bias)
        if config.learned_masks:
            self.modality_gate_logits = nn.Parameter(torch.full((config.blocks, 2),
                math.log(config.gate_initial/(1-config.gate_initial)), dtype=torch.float32))
        else:
            self.register_parameter("modality_gate_logits", None)
        self.cache_enabled = False
        self.clear_cache()
        self.last_dispatch = None
        self.last_training_dispatch = None
        self.warmup_force_new = False

    def clear_cache(self):
        self._context = self._valid = self._visual = None

    def forward(self, states, visual, valid):
        config = self.config
        batch, length, hidden = states.shape
        offset = 0 if self._context is None else self._context.shape[1]
        if not self.cache_enabled:
            offset = 0
        # An absolute positional signal makes attention spatially/order aware.
        position = torch.arange(offset, offset+length, device=states.device).float()[:, None]
        frequency = torch.exp(torch.arange(0, config.width, 2, device=states.device).float()
                              * (-math.log(10000.)/config.width))
        positional = torch.zeros(length, config.width, device=states.device)
        positional[:, 0::2] = torch.sin(position*frequency)
        positional[:, 1::2] = torch.cos(position*frequency[:config.width//2])
        latent = self.down(states) + positional.to(states.dtype)
        if self.cache_enabled and self._context is not None:
            if self._context.shape[0] != batch:
                raise ValueError("Generation batch changed without resetting the visual cache")
            context = torch.cat((self._context, latent), 1)
            context_valid = torch.cat((self._valid, valid), 1)
            context_visual = torch.cat((self._visual, visual), 1)
        else:
            context, context_valid, context_visual = latent, valid, visual
        if self.cache_enabled:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("Visual generation cache requires eval and no_grad")
            self._context, self._valid, self._visual = context.detach(), context_valid, context_visual
        # Text-only requests and text BEFORE the first image are exact bypasses.
        active = context_visual.logical_and(context_valid).cumsum(1)[:, offset:].gt(0) & valid
        where = active.nonzero(as_tuple=False)
        output = torch.zeros_like(latent)
        counts = torch.zeros((2, config.blocks), device=states.device, dtype=torch.long)
        if len(where):
            rows, positions = where.unbind(1)
            q_visual = visual[rows, positions]
            if config.mode == "fixed":
                indices = torch.arange(config.blocks, device=states.device).expand(len(where), -1)
                weights = torch.full(indices.shape, 1/config.blocks, device=states.device, dtype=torch.float32)
            else:
                router_input = torch.cat((latent[rows, positions],
                    F.one_hot(q_visual.long(), 2).to(latent.dtype)), -1)
                logits = self.router(router_input).float()
                indices = logits.topk(config.top_k, -1).indices
                if self.training and self.warmup_force_new:
                    # Exploration changes ONLY extra slots; original MoE is untouched.
                    indices = (torch.arange(len(where), device=states.device)[:, None]
                               + torch.arange(config.top_k, device=states.device)) % config.blocks
                weights = logits.gather(-1, indices).softmax(-1)
            gates = None if self.modality_gate_logits is None else self.modality_gate_logits.sigmoid()
            for expert, block in enumerate(self.blocks):
                selected, slots = (indices == expert).nonzero(as_tuple=True)
                if not len(selected):
                    continue
                source_gates = (torch.ones_like(context_valid, dtype=torch.float32) if gates is None else
                                gates[expert, context_visual.long()])
                delta = block(context, context_valid, rows[selected], positions[selected]+offset, source_gates)
                if gates is not None:
                    delta = delta * gates[expert, q_visual[selected].long(), None].to(delta.dtype)
                delta = delta * weights[selected, slots, None].to(delta.dtype)
                flat_indices = rows[selected]*length+positions[selected]
                output.reshape(-1, config.width).index_add_(0, flat_indices, delta.to(output.dtype))
                counts[:, expert] = torch.bincount(q_visual[selected].long(), minlength=2)
        self.last_dispatch = dict(counts=counts.detach(), visual_tokens=(visual & valid).sum().detach(),
                                  active_text_tokens=(active & ~visual).sum().detach())
        if self.training:
            self.last_training_dispatch = self.last_dispatch
        return self.up(output)


class AdditiveVisualMoE(nn.Module):
    def __init__(self, original, config, hidden_size, *, smallthinker):
        super().__init__()
        self.original = original
        self.smallthinker = smallthinker
        reference = next(original.parameters())
        self.visual_experts = AdditiveVisualBank(hidden_size, config).to(device=reference.device, dtype=torch.float32)
        self._modality_mask = self._attention_mask = None

    def set_modality_mask(self, mask, attention_mask=None):
        if mask is not None and (mask.ndim != 2 or attention_mask is not None and mask.shape != attention_mask.shape):
            raise ValueError("Identical [batch, sequence] modality/attention masks required")
        self._modality_mask = None if mask is None else mask.detach().bool()
        self._attention_mask = None if attention_mask is None else attention_mask.detach().bool()

    def forward(self, *args, **kwargs):
        # Call the ACTUAL original module, with its original router and top-k.
        native = self.original(*args, **kwargs)
        states = (args[1] if self.smallthinker else args[0]) if args else kwargs["hidden_states"]
        if states.ndim != 3:
            raise ValueError("Extra Transformers require [batch, sequence, hidden]")
        shape = states.shape[:2]
        def resolve(mask, default):
            if mask is None:
                return torch.full(shape, default, dtype=torch.bool, device=states.device)
            if self.visual_experts.cache_enabled and self.visual_experts._context is not None:
                if mask.shape[0] != shape[0] or shape[1] != 1:
                    raise ValueError("Only batch-stable incremental generation is supported")
                return torch.full(shape, default, dtype=torch.bool, device=states.device)
            if tuple(mask.shape) == shape:
                return mask.to(states.device)
            raise ValueError("Stale additive-branch modality mask")
        visual, valid = resolve(self._modality_mask, False), resolve(self._attention_mask, True)
        with torch.autocast(states.device.type, dtype=torch.bfloat16, enabled=states.device.type == "cuda"):
            delta = self.visual_experts(states.to(self.visual_experts.down.weight.dtype), visual, valid).to(states.dtype)
        if isinstance(native, tuple):
            return (native[0]+delta, *native[1:])
        return native+delta

    def diagnostics(self):
        bank = self.visual_experts
        result = {"architecture":"additive_visual_transformer_v1", **asdict(bank.config),
                  "native_routing":"original_module_unmodified_separate_topk", "new_branch_competes_with_native":False}
        snapshot = bank.last_training_dispatch or bank.last_dispatch
        if snapshot is not None:
            result.update({k:v.cpu().tolist() for k,v in snapshot.items()})
            result['dispatch_scope'] = 'last training forward/recompute, not an epoch census' if bank.last_training_dispatch else 'last forward'
            result['expected_visual_slots'] = result['visual_tokens']*(bank.config.blocks if bank.config.mode=='fixed' else bank.config.top_k)
            result['visual_dispatch_complete'] = sum(result['counts'][1]) == result['expected_visual_slots']
        if bank.modality_gate_logits is not None:
            result["extra_only_text_visual_gates"] = bank.modality_gate_logits.detach().sigmoid().cpu().tolist()
        return result


def iter_additive(llm):
    return (m for m in llm.modules() if isinstance(m, AdditiveVisualMoE))


def install_additive_visual(llm, config):
    if any(iter_additive(llm)):
        raise ValueError("Additive branch already installed")
    backend = llm.config.model_type
    types = {"smallthinker":"SmallThinkerMoeBlock", "lfm2_moe":"Lfm2MoeSparseMoeBlock", "deepseek_v3":"DeepseekV3MoE"}
    if backend not in types:
        raise ValueError("Unsupported additive backbone")
    matches = [(p,n,c) for p in llm.modules() for n,c in p.named_children() if c.__class__.__name__ == types[backend]]
    if not matches:
        raise ValueError("No native MoE modules found")
    prepared = [(p,n,AdditiveVisualMoE(c,config,llm.config.hidden_size,smallthinker=backend == "smallthinker")) for p,n,c in matches]
    for parent,name,wrapper in prepared:
        setattr(parent,name,wrapper)
    llm.config.additive_visual_v1 = asdict(config)
    return [w for _,_,w in prepared]


def set_modality_mask(llm, mask, attention_mask=None):
    for wrapper in iter_additive(llm):
        wrapper.set_modality_mask(mask, attention_mask)


@contextmanager
def generation(llm):
    banks = [w.visual_experts for w in iter_additive(llm)]
    if any(b.cache_enabled for b in banks):
        raise RuntimeError("Nested visual generation scopes are unsupported")
    try:
        for bank in banks:
            bank.clear_cache(); bank.cache_enabled = True
        yield
    finally:
        for bank in banks:
            bank.clear_cache(); bank.cache_enabled = False
