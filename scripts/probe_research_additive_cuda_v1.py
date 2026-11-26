#!/usr/bin/env python3
"""Small real-native/NF4 GPU backward checks; never constructs full models."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from src.utils.research_stock_determinism import activate
activate()
import torch
from src.model.research_additive_visual_v1 import AdditiveVisualMoE,VisualBranchConfig
from src.model.research_quantization import QuantizedPackedExperts
from src.utils.research_storage import _atomic_json

def main():
    (ROOT/'research_results/additive_visual_v1').mkdir(parents=True,exist_ok=True)
    free,_=torch.cuda.mem_get_info()
    if free<8*2**30:raise MemoryError('Tiny parallel CUDA check requires8GiB free')
    rows=[]
    for kind in ('smallthinker','lfm2_moe','deepseek_v3'):
        for mode in ('fixed','routed'):
            torch.manual_seed(42)
            if kind=='smallthinker':
                from src.model.smallthinker import SmallThinkerConfig,SmallThinkerMoeBlock
                cfg=SmallThinkerConfig(hidden_size=64,num_attention_heads=4,num_key_value_heads=2,
                    moe_ffn_hidden_size=128,moe_num_primary_experts=4,moe_num_active_primary_experts=2)
                native=SmallThinkerMoeBlock(cfg)
            elif kind=='lfm2_moe':
                from transformers import Lfm2MoeConfig
                from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeSparseMoeBlock
                cfg=Lfm2MoeConfig(hidden_size=64,moe_intermediate_size=128,intermediate_size=128,
                    num_attention_heads=4,num_key_value_heads=2,num_experts=4,num_experts_per_tok=2)
                native=Lfm2MoeSparseMoeBlock(cfg)
            else:
                from transformers import DeepseekV3Config
                from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
                cfg=DeepseekV3Config(hidden_size=64,moe_intermediate_size=128,intermediate_size=128,
                    n_routed_experts=4,n_shared_experts=1,num_experts_per_tok=2,n_group=1,topk_group=1,
                    routed_scaling_factor=1.)
                native=DeepseekV3MoE(cfg)
            for p in native.parameters():torch.nn.init.normal_(p,std=.03)
            native.to(device='cuda',dtype=torch.bfloat16)
            if kind!='smallthinker':native.experts=QuantizedPackedExperts(native.experts,rank=4)
            native.requires_grad_(False)
            wrapper=AdditiveVisualMoE(native,VisualBranchConfig(mode=mode,width=32,heads=4,
                blocks=2 if mode=='fixed' else 8,top_k=2,learned_masks=True),64,smallthinker=kind=='smallthinker')
            x=torch.randn(2,32,64,device='cuda',dtype=torch.bfloat16)
            visual=torch.zeros((2,32),device='cuda',dtype=torch.bool);visual[:,2:18]=True
            wrapper.set_modality_mask(visual,torch.ones_like(visual))
            inputs=(x,x) if kind=='smallthinker' else (x,)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                before=native(*inputs);actual=wrapper(*inputs)
            old=before[0] if isinstance(before,tuple) else before
            y=actual[0] if isinstance(actual,tuple) else actual
            if not torch.equal(old,y):raise ValueError('Zero-init changed original native output')
            opt=torch.optim.AdamW(wrapper.visual_experts.parameters(),lr=.001)
            for step in range(3):
                opt.zero_grad()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    actual=wrapper(*inputs);y=actual[0] if isinstance(actual,tuple) else actual
                    loss=(y.float()-.5).square().mean()
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None):raise ValueError('Nonfinite CUDA backward')
                opt.step()
            evidence=wrapper.diagnostics()
            if not evidence['visual_dispatch_complete']:raise ValueError('Visual token missed extra branch')
            if wrapper.visual_experts.modality_gate_logits.grad.abs().sum()==0:raise ValueError('Masks failed to learn')
            rows.append(dict(backend=kind,packed_NF4=kind!='smallthinker',zero_init_native_exact=True,
                backward_updates=3,final_loss=float(loss.detach()),**evidence))
    _atomic_json(ROOT/'research_results/additive_visual_v1/cuda_probe.json',dict(status='passed',cases=rows,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,full_model_training_claimed=False))
    print(json.dumps(dict(status='passed',cases=len(rows),peak_GiB=torch.cuda.max_memory_allocated()/2**30)),flush=True)

if __name__=='__main__':main()
