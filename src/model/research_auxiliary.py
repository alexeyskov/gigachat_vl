"""Auditable native auxiliary-loss contracts and eligibility-aware routing loss.

No loss is silently enabled. In particular, the published GPT-OSS-20B config has
coefficient 0.9 but output_router_logits=False: its native default applies no
balancing penalty. Enabling native router output is a distinct training profile.
These helpers do not install collectors or change the active training runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


def native_auxiliary_contract(config, *, output_router_logits=None):
    kind=getattr(config,'model_type',None)
    flag=bool(getattr(config,'output_router_logits',False)) if output_router_logits is None else bool(output_router_logits)
    coefficient=float(getattr(config,'router_aux_loss_coef',0.) or 0.) if kind=='gpt_oss' else 0.
    if not math.isfinite(coefficient) or coefficient<0:raise ValueError('Invalid native router auxiliary coefficient')
    unsupported={}
    for name in ('router_z_loss_coef','router_z_loss_coefficient','z_loss_coef','aux_loss_alpha','moe_aux_loss_coeff'):
        value=getattr(config,name,None)
        if value is not None and float(value)!=0:unsupported[name]=value
    if kind!='gpt_oss' and getattr(config,'router_aux_loss_coef',0):
        unsupported['router_aux_loss_coef']=getattr(config,'router_aux_loss_coef')
    mechanism={'gpt_oss':'Switch-style loss only when router output is enabled',
               'deepseek_v3':'fixed pretrained noaux_tc correction bias; native HF forward has no bias update or auxiliary scalar loss',
               'lfm2_moe':'optional fixed expert_bias; native HF forward has no bias update or auxiliary scalar loss',
               'smallthinker':'native checkpoint-compatible forward has no auxiliary scalar loss'}.get(kind,'unsupported native architecture')
    return {'model_type':kind,'native_coefficient':coefficient,
            'configured_output_router_logits':bool(getattr(config,'output_router_logits',False)),
            'requested_output_router_logits':flag,'enabled':kind=='gpt_oss' and flag and coefficient!=0,
            'native_z_loss_coefficient':None,'unsupported_nonzero_coefficients':unsupported,
            'mechanism':mechanism,'collector_required_when_enabled':True,
            'native_output_recorder_invalid_after_router_replacement':kind=='gpt_oss',
            'statistic_scope':'concatenate equal-width layers, condition on allowed expert sets',
            'gradient_accumulation_contract':'exact logical-batch equivalence requires global detached dispatch statistics; averaging microbatch products of means is different'}


@dataclass
class RouterLayerState:
    probabilities:torch.Tensor
    selected_experts:torch.Tensor
    token_mask:torch.Tensor|None=None
    eligibility:torch.Tensor|None=None
    slot_mask:torch.Tensor|None=None


def router_state_from_logits(logits,top_k,*,token_mask=None,eligibility=None):
    if logits.ndim!=2 or not 1<=top_k<=logits.shape[-1]:raise ValueError('Invalid router logits/top-k shape')
    valid=torch.ones(logits.shape[0],device=logits.device,dtype=torch.bool) if token_mask is None else token_mask.reshape(-1).bool().to(logits.device)
    eligible=torch.ones_like(logits,dtype=torch.bool) if eligibility is None else eligibility.bool().to(logits.device)
    if eligible.shape!=logits.shape or valid.numel()!=logits.shape[0]:raise ValueError('Router mask shapes differ')
    if torch.any(valid & ~eligible.any(-1)):raise ValueError('A valid token has no eligible expert')
    masked=logits.float().masked_fill(~eligible,-torch.inf)
    safe=torch.where(valid[:,None],masked,torch.zeros_like(masked))
    probabilities=safe.softmax(-1)
    indices=masked.topk(top_k,dim=-1).indices
    slots=eligible.gather(-1,indices) & valid[:,None]
    return RouterLayerState(probabilities,indices,valid,eligible,slots)


def eligible_switch_balance_loss(states):
    """Native Switch loss for free routing, conditional loss for hard masks.

    For each allowed-expert set G, use E_G * sum_e(f_G,e * mean_p_G,e), where f
    counts dispatched slots per valid token, matching native top-k scaling.
    Combine groups by their valid-token counts. Forbidden experts and padding
    contribute neither probabilities nor desired utilization. With a common E,
    all-eligible routing exactly matches native GPT-OSS load-balancing loss.
    """
    states=list(states)
    if not states:raise ValueError('No router statistics collected')
    pools={}
    zero=states[0].probabilities.sum()*0
    for state in states:
        p=state.probabilities.float();indices=state.selected_experts
        if p.ndim!=2 or indices.ndim!=2 or indices.shape[0]!=p.shape[0]:raise ValueError('Invalid router statistics dimensions')
        if indices.device!=p.device:raise ValueError('Router statistics must share a device')
        if torch.any(indices<0) or torch.any(indices>=p.shape[-1]):raise ValueError('Selected expert outside bank')
        mask=torch.ones(p.shape[0],device=p.device,dtype=torch.bool) if state.token_mask is None else state.token_mask.reshape(-1).bool().to(p.device)
        eligible=torch.ones_like(p,dtype=torch.bool) if state.eligibility is None else state.eligibility.bool().to(p.device)
        slots=torch.ones_like(indices,dtype=torch.bool) if state.slot_mask is None else state.slot_mask.bool().to(p.device)
        if mask.numel()!=p.shape[0] or eligible.shape!=p.shape or slots.shape!=indices.shape:raise ValueError('Router statistic masks differ in shape')
        if torch.any(mask & ~eligible.any(-1)):raise ValueError('A valid token has no eligible expert')
        if torch.any((slots & mask[:,None]) & ~eligible.gather(-1,indices)):
            raise ValueError('Dispatch selected a forbidden expert')
        if not torch.isfinite(p).all() or torch.any(p<0):raise ValueError('Invalid routing probabilities')
        if not mask.any():continue
        key=(p.shape[-1],indices.shape[-1],p.device)
        pools.setdefault(key,[]).append((p[mask],indices[mask],eligible[mask],slots[mask]))
    numerator=zero;denominator=0
    for (width,_top_k,_device),parts in pools.items():
        probability=torch.cat([part[0] for part in parts])
        indices=torch.cat([part[1] for part in parts])
        eligibility=torch.cat([part[2] for part in parts])
        slots=torch.cat([part[3] for part in parts])
        patterns,inverse=torch.unique(eligibility,dim=0,return_inverse=True)
        for group_index,pattern in enumerate(patterns):
            tokens=inverse==group_index
            count=int(tokens.sum())
            group_p=probability[tokens][:,pattern]
            mass=group_p.sum(-1,keepdim=True)
            if torch.any(mass<=0):raise ValueError('Eligible expert probabilities have zero total mass')
            group_p=group_p/mass
            dispatch=F.one_hot(indices[tokens],num_classes=width).float()*slots[tokens][...,None]
            fraction=dispatch.sum((0,1))[pattern]/count
            loss=int(pattern.sum())*(fraction*group_p.mean(0)).sum()
            numerator=numerator+count*loss
            denominator+=count
    return numerator/denominator if denominator else zero


def native_router_auxiliary_loss(config,states,*,output_router_logits=None):
    contract=native_auxiliary_contract(config,output_router_logits=output_router_logits)
    if contract['unsupported_nonzero_coefficients']:
        raise NotImplementedError(f"Unimplemented declared native auxiliary coefficients: {contract['unsupported_nonzero_coefficients']}")
    if not contract['enabled']:
        return {'loss':0.,'balance_loss':None,'contract':contract}
    states=list(states)
    if not states:raise RuntimeError('Native router loss is enabled but no differentiable E+K router statistics were collected')
    balance=eligible_switch_balance_loss(states)
    return {'loss':contract['native_coefficient']*balance,'balance_loss':balance,'contract':contract}
