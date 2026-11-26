#!/usr/bin/env python3
"""Reproduce failed A0 with read-only gradient diagnostics, outside primary runs."""
from __future__ import annotations
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ['CUDA_VISIBLE_DEVICES']='0'
from src.utils.research_stock_determinism import bootstrap_environment
bootstrap_environment()
import torch
from src.utils.research_storage import _atomic_json
from src.utils.research_small_budget_v3 import main
from scripts.research_budget18_queue_v4 import resources


def run():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    resources(idle=True)
    request=Path(sys.argv[1]);argv=json.loads(request.read_bytes())['argv']
    out=ROOT/'runs/research/diagnostic_small_qwen_A0_gradient_v2'
    argv[argv.index('--output')+1]=str(out)
    original=torch.nn.utils.clip_grad_norm_;counter=0
    native_call=torch.nn.Module._call_impl
    def inspect_module(module,*args,**kwargs):
        result=native_call(module,*args,**kwargs)
        if counter>=6 and type(module).__name__ in ('SmallThinkerDecoderLayer','LlamaAttention','LlamaRMSNorm','Sequential'):
            value=result if torch.is_tensor(result) else result[0] if isinstance(result,tuple) and result and torch.is_tensor(result[0]) else None
            if value is not None and not torch.isfinite(value).all():
                report={'status':'first_nonfinite_forward','module_type':type(module).__name__,'shape':list(value.shape),'dtype':str(value.dtype),'nan':int(torch.isnan(value).sum()),'inf':int(torch.isinf(value).sum()),'input_finite':bool(torch.isfinite(args[0]).all()) if args and torch.is_tensor(args[0]) else None}
                if value.ndim==3:report['bad_query_positions']=torch.nonzero(~torch.isfinite(value).all(-1))[:30].cpu().tolist()
                _atomic_json(out/'first_nonfinite_forward.json',report)
                raise FloatingPointError('First nonfinite forward in '+type(module).__name__)
        return result
    torch.nn.Module._call_impl=inspect_module
    def clipping(parameters,*args,**kwargs):
        nonlocal counter
        counter+=1;parameters=list(parameters);values=[]
        for i,p in enumerate(parameters):
            if p.grad is None:continue
            g=p.grad.detach();finite=torch.isfinite(g);f=g[finite]
            values.append({'parameter_ordinal':i,'shape':list(p.shape),'grad_dtype':str(g.dtype),
                'nan':int(torch.isnan(g).sum()),'inf':int(torch.isinf(g).sum()),
                'max_abs_finite':float(f.abs().max()) if f.numel() else None,
                'finite_L2_float64':float(f.double().norm())})
        report={'clip_call':counter,'created_unix':time.time(),'gradient_parameters':values,'diagnostic_not_primary':True}
        _atomic_json(out/'gradient_diagnostics.json',report)
        try:result=original(parameters,*args,**kwargs)
        except RuntimeError as e:
            report.update(status='native_clip_failed',error=str(e));_atomic_json(out/'gradient_diagnostics.json',report);raise
        if counter==6:torch.autograd.set_detect_anomaly(True)
        if counter>=7:raise RuntimeError('Technical probe stopped after seventh finite clip; no seventh optimizer update')
        return result
    torch.nn.utils.clip_grad_norm_=clipping
    try:return main(['--allow-gpu',*argv])
    finally:
        torch.nn.utils.clip_grad_norm_=original;torch.nn.Module._call_impl=native_call


if __name__=='__main__':run()
