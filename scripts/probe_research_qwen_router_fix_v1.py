#!/usr/bin/env python3
"""Replay six updates exactly and verify the formerly failing seventh backward."""
import json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='0'
from src.utils.research_stock_determinism import bootstrap_environment
bootstrap_environment()
import torch
from src.utils.research_small_router_stability_v1 import main, definition
from src.utils.research_storage import _atomic_json
from scripts.research_budget18_queue_v4 import resources


class ProbeComplete(Exception):
    pass


def run():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    resources(idle=True)
    argv=json.loads(Path(sys.argv[1]).read_bytes())['argv']
    out=ROOT/'runs/research/diagnostic_small_qwen_router_fix_v1'
    argv[argv.index('--output')+1]=str(out)
    native=torch.nn.utils.clip_grad_norm_
    steps=[]
    def clipping(parameters,*args,**kwargs):
        parameters=list(parameters)
        finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
        norm=native(parameters,*args,**kwargs)
        steps.append({'batch':len(steps)+1,'all_gradients_finite':finite,'native_preclip_norm':float(norm)})
        if len(steps)==7:
            events=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
            losses=[e['loss'] for e in events if e.get('event')=='train']
            expected=[4.255384040085124,2.7895582350763273,2.7761746495327104,2.7856007842154265,2.7181065694319546,2.6930517275544026]
            if losses!=expected or not finite:
                raise AssertionError('First six native losses or corrected backward differs')
            _atomic_json(out/'proof.json',{'status':'passed_seventh_backward_and_exact_first_six_losses',
                'policy':definition(),'gradients':steps,'first_six_losses':losses,
                'seventh_optimizer_step_performed':False,'primary_experiment':False,'created_unix':time.time()})
            raise ProbeComplete('Verified finite seventh backward; technical probe ends before optimizer update')
        return norm
    torch.nn.utils.clip_grad_norm_=clipping
    try:
        try:main(['--allow-gpu',*argv])
        except ProbeComplete:return 0
    finally:
        torch.nn.utils.clip_grad_norm_=native


if __name__=='__main__':raise SystemExit(run())
