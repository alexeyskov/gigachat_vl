"""R4/R5: eight native-shape FFN experts compete in the original-size top-k.

This is separate from the frozen additive branch study. There is no modality
eligibility mask, separate visual top-k, or unconditional visual residual.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from src.model.research_moe import ResearchSmallThinkerMoE, ResearchPackedMoE, iter_research_moe
from src.utils.research_small_router_stability_v1 import normalize_sigmoid


class RouterLoRA(nn.Module):
    """Train rank-r residual logits for E+8 outputs, keeping original W frozen."""
    def __init__(self, hidden, outputs, rank=16, alpha=32):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, hidden, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(outputs, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha/rank

    def forward(self, inputs):
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return F.linear(F.linear(inputs.float(), self.lora_A), self.lora_B)*self.scale


def caption_exploration(wrapper, selection_scores):
    if wrapper.training and wrapper.warmup_force_new:
        # Rotate one guaranteed new candidate per token during caption warmup.
        # Routing weights still use the unmodified finite native scores/logits.
        # Without this, ties between eight zero-initialized R5 heads can leave
        # some experts entirely unused during a four-update warmup.
        chosen = selection_scores.clone()
        desired = wrapper.old_count + torch.arange(len(chosen),device=chosen.device)%wrapper.added_count
        bonus = chosen.max(-1).values.detach()+1
        return chosen.scatter(1,desired[:,None],bonus[:,None])
    return selection_scores


class CommonSmall(ResearchSmallThinkerMoE):
    def __init__(self, base, method):
        if method not in ('R4', 'R5'):
            raise ValueError('Expected R4 or R5')
        super().__init__(base, 2 if method == 'R4' else 1, 'b', added_experts=8)
        self.method = method
        if method == 'R5':
            self.router_adapter = RouterLoRA(self.hidden_size, self.old_count+8).to(self.primary_router.weight.device)

    def _routing_weights(self, scores, logits, indices, slot_valid):
        if self.apply_softmax:
            return super()._routing_weights(scores, logits, indices, slot_valid)
        return normalize_sigmoid(logits, slot_valid=slot_valid, clamp_tiny=True)

    def _selection_scores(self, scores, logits, alpha, inputs):
        return caption_exploration(self,super()._selection_scores(scores,logits,alpha,inputs))

    def diagnostics(self):
        return dict(ResearchSmallThinkerMoE.diagnostics(self), architecture='common_pool_v1', method=self.method,
                    top_k=self.top_k, modality_restrictions=False,
                    force_new_during_caption_warmup=self.warmup_force_new)


class CommonPacked(ResearchPackedMoE):
    def __init__(self, base, config, method):
        if method not in ('R4', 'R5'):
            raise ValueError('Expected R4 or R5')
        super().__init__(base, config, 2 if method == 'R4' else 1, 'b', added_experts=8)
        self.method = method
        if method == 'R5':
            self.router_adapter = RouterLoRA(self.hidden_size, self.old_count+8).to(self.primary_router.weight.device)

    def _selection_scores(self, scores, logits, alpha, inputs):
        return caption_exploration(self,super()._selection_scores(scores,logits,alpha,inputs))

    diagnostics = CommonSmall.diagnostics


def install(llm, method):
    if any(iter_research_moe(llm)):
        raise ValueError('MoE already expanded')
    backend = llm.config.model_type
    kinds = {'smallthinker':'SmallThinkerMoeBlock', 'lfm2_moe':'Lfm2MoeSparseMoeBlock',
             'deepseek_v3':'DeepseekV3MoE'}
    if backend not in kinds:
        raise ValueError('Unsupported R4/R5 backbone')
    matches = [(parent, name, child) for parent in llm.modules()
               for name, child in parent.named_children() if child.__class__.__name__ == kinds[backend]]
    if not matches:
        raise ValueError('No native MoE layers found')
    prepared = [(p, n, CommonSmall(c, method) if backend == 'smallthinker'
                 else CommonPacked(c, llm.config, method)) for p, n, c in matches]
    for parent, name, wrapper in prepared:
        setattr(parent, name, wrapper)
    llm.config.common_pool_v1 = dict(method=method, added_experts=8, top_k='native unchanged', policy='unrestricted')
    return [wrapper for _, _, wrapper in prepared]


def configure(llm, stage):
    for wrapper in iter_research_moe(llm):
        enabled = stage in ('A1', 'SFT')
        wrapper.visual_experts.requires_grad_(enabled)
        wrapper.primary_router.requires_grad_(enabled and wrapper.method == 'R4')
        if wrapper.method == 'R5':
            wrapper.router_adapter.requires_grad_(enabled)
        # Exploration is confined to caption warmup and documented explicitly.
        # Every instruction/detection update and all evaluation use pure top-k.
        wrapper.warmup_force_new = stage == 'A1'
        assert wrapper.modality_gate_logits is None and wrapper.policy == 'b'
