#!/usr/bin/env python3
"""Read-only localization of the reproducible seventh-batch forward failure."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from src.utils.research_stock_determinism import bootstrap_environment
bootstrap_environment()
import torch
from src.utils.research_storage import _atomic_json
from src.utils.research_small_budget_v3 import main
from scripts.research_budget18_queue_v4 import resources


def run():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    resources(idle=True)
    argv = json.loads(Path(sys.argv[1]).read_bytes())['argv']
    out = ROOT/'runs/research/diagnostic_small_qwen_A0_gradient_v3'
    argv[argv.index('--output')+1] = str(out)
    native_call = torch.nn.Module._call_impl
    native_clip = torch.nn.utils.clip_grad_norm_
    counter = 0
    layer = None

    def inspect(module, *args, **kwargs):
        nonlocal layer
        kind = type(module).__name__
        if kind == 'SmallThinkerDecoderLayer':
            layer = module.self_attn.layer_idx
        result = native_call(module, *args, **kwargs)
        if counter < 6 or kind not in ('SmallThinkerMoeBlock', 'SmallThinkerExpert',
                'LlamaAttention', 'LlamaRMSNorm', 'SmallThinkerDecoderLayer'):
            return result
        value = result[0] if isinstance(result, tuple) else result
        if not torch.is_tensor(value) or torch.isfinite(value).all():
            return result
        report = dict(status='first_nonfinite_forward', module_type=kind, layer_idx=layer,
                      shape=list(value.shape), nan=int(torch.isnan(value).sum()),
                      inf=int(torch.isinf(value).sum()), diagnostic_not_primary=True)
        if kind == 'SmallThinkerMoeBlock':
            bad = ~torch.isfinite(value.reshape(-1, value.shape[-1])).all(-1)
            router = args[0].reshape(-1, args[0].shape[-1])[bad]
            with torch.no_grad():
                logits = module.primary_router(router)
                top = logits.topk(module.config.moe_num_active_primary_experts, dim=-1).values
                report.update(bad_flat_positions=bad.nonzero().flatten().cpu().tolist(),
                              router_input_max_abs=float(router.abs().max()),
                              router_logits_dtype=str(logits.dtype),
                              selected_logits=top.float().cpu().tolist(),
                              sigmoid_denominator=top.sigmoid().sum(-1).float().cpu().tolist(),
                              fp32_sigmoid_denominator=top.float().sigmoid().sum(-1).cpu().tolist())
        _atomic_json(out/'first_nonfinite_forward.json', report)
        raise FloatingPointError('Localized first nonfinite '+kind)

    def clipping(parameters, *args, **kwargs):
        nonlocal counter
        counter += 1
        result = native_clip(parameters, *args, **kwargs)
        if counter >= 7:
            raise RuntimeError('Diagnostic stops before seventh optimizer update')
        return result

    torch.nn.Module._call_impl = inspect
    torch.nn.utils.clip_grad_norm_ = clipping
    try:
        return main(['--allow-gpu', *argv])
    finally:
        torch.nn.Module._call_impl = native_call
        torch.nn.utils.clip_grad_norm_ = native_clip


if __name__ == '__main__':
    run()
