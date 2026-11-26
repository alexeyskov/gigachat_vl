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
    out=ROOT/'runs/research/diagnostic_small_qwen_A0_gradient_v1'
    argv[argv.index('--output')+1]=str(out)
    original=torch.nn.utils.clip_grad_norm_;counter=0
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
        if counter>=7:raise RuntimeError('Technical probe stopped after seventh finite clip; no seventh optimizer update')
        return result
    torch.nn.utils.clip_grad_norm_=clipping
    try:return main(['--allow-gpu',*argv])
    finally:torch.nn.utils.clip_grad_norm_=original


if __name__=='__main__':run()
