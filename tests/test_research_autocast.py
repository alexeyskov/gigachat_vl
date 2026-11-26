"""Small real-CUDA probe of autocast shadows; not a full-model fit claim."""

import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Autocast memory measurement requires CUDA')


class PackedViewBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up=nn.Parameter(torch.randn(4,1024,512,device='cuda')*.02)
        self.down=nn.Parameter(torch.randn(4,512,512,device='cuda')*.02)
    def forward(self,x):
        outputs=[]
        for index,part in enumerate(x.chunk(4,0)):
            gate,up=F.linear(part,self.gate_up[index]).chunk(2,-1)
            outputs.append(F.linear(F.silu(gate)*up,self.down[index]))
        return torch.cat(outputs)


def autocast_probe(kind,cache_enabled,offload=False):
    torch.manual_seed(184)
    if kind=='leaf_linears':
        layers=nn.ModuleList([nn.Sequential(nn.Linear(1024,1024,bias=False,device='cuda'),nn.GELU()) for _ in range(8)])
        x=torch.randn(64,1024,device='cuda',requires_grad=True)
    else:
        layers=nn.ModuleList([PackedViewBlock() for _ in range(3)])
        x=torch.randn(64,512,device='cuda',requires_grad=True)
    optimizer=None
    if offload:
        from src.utils.research_optim import CPUGradientOffloadOptimizer
        optimizer=CPUGradientOffloadOptimizer(layers.parameters(),optim_bits=8)
    baseline=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=cache_enabled):
        output=x
        for layer in layers:
            output=checkpoint(layer,output,use_reentrant=False)
        loss=output.float().square().sum()
        torch.cuda.synchronize()
        forward_resident=torch.cuda.memory_allocated()-baseline
        forward_peak=torch.cuda.max_memory_allocated()-baseline
    after_context=torch.cuda.memory_allocated()-baseline
    loss.backward();torch.cuda.synchronize()
    maximum=torch.cuda.max_memory_allocated()
    gradients=[(optimizer._cpu_grads[p].clone() if optimizer is not None else p.grad.detach().cpu().clone())
               for p in layers.parameters()]
    result=output.detach().cpu().clone()
    metrics={'kind':kind,'cache_enabled':cache_enabled,'gradient_offload':offload,
             'parameter_bytes':sum(p.numel()*p.element_size() for p in layers.parameters()),
             'forward_resident_extra_bytes':forward_resident,'forward_peak_extra_bytes':forward_peak,
             'after_context_extra_bytes':after_context,'process_peak_allocated_bytes':maximum,
             'claim':'tiny_checkpointed_probe_only_not_full_model_capacity'}
    if optimizer is not None:optimizer.close()
    del output,loss,x,layers
    torch.cuda.empty_cache()
    return result,gradients,metrics


@pytest.mark.parametrize('kind',['leaf_linears','packed_weight_views'])
@pytest.mark.parametrize('offload',[False,True])
def test_no_autocast_weight_cache_preserves_outputs_and_gradients(kind,offload):
    enabled,enabled_grads,true_metrics=autocast_probe(kind,True,offload)
    disabled,disabled_grads,false_metrics=autocast_probe(kind,False,offload)
    torch.testing.assert_close(enabled,disabled,atol=0,rtol=0)
    for left,right in zip(enabled_grads,disabled_grads):
        torch.testing.assert_close(left,right,atol=0,rtol=0)
    assert max(true_metrics['process_peak_allocated_bytes'],false_metrics['process_peak_allocated_bytes'])<2**30
    if kind=='leaf_linears':
        assert true_metrics['forward_resident_extra_bytes']-false_metrics['forward_resident_extra_bytes']>8*2**20
    print(json.dumps({'autocast_probe':[true_metrics,false_metrics]}))
