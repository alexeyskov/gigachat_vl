#!/usr/bin/env python3
"""Conditional BF16 rounding census of the existing GPU-exported FP32 gates."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.environ['CUDA_VISIBLE_DEVICES']=''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(source,run):
    import torch
    torch.set_num_threads(1)
    source=Path(source)
    with source.open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows or {r['stage'] for r in rows}!={'A1','SFT'}:
        raise ValueError('Both actual A1 and SFT gate censuses required')
    groups={}
    for stage in ('A1','SFT'):
        for origin in sorted({r['origin'] for r in rows}):
            selected=[r for r in rows if r['stage']==stage and r['origin']==origin]
            values=[[float(r['text_gate']),float(r['visual_gate'])] for r in selected]
            if not values or any(not math.isfinite(x) or not 0<=x<=1 for pair in values for x in pair):
                raise ValueError('Finite text/visual gate pairs required')
            probabilities=torch.tensor(values,dtype=torch.float32)
            rounded=probabilities.to(torch.bfloat16).float();delta=rounded[:,1]-rounded[:,0]
            result={'expert_block_pairs':len(selected),
                'FP32_visual_greater':int((probabilities[:,1]>probabilities[:,0]).sum()),
                'BF16_visual_greater':int((delta>0).sum()),'BF16_text_greater':int((delta<0).sum()),
                'BF16_equal':int((delta==0).sum()),
                'unique_BF16_text':sorted(set(rounded[:,0].tolist())),
                'unique_BF16_visual':sorted(set(rounded[:,1].tolist())),
                'BF16_text_mean':float(rounded[:,0].mean()),'BF16_visual_mean':float(rounded[:,1].mean())}
            if sum(result[k] for k in ('BF16_visual_greater','BF16_text_greater','BF16_equal'))!=len(selected):
                raise ValueError('Exhaustive three-way rounding counts differ')
            groups[stage+'_'+origin]=result
    return {'status':'CPU_conditional_BF16_gate_rounding_census','run':run,'created_unix':time.time(),
        'source':{'path':str(source),'sha256':sha(source)},'source_sha256':sha(Path(__file__)),
        'groups':groups,'GPU_dtype_probe_performed':False,
        'scope':'BF16 cast of the exported actual FP32 sigmoid probabilities; conditional analysis, not a new GPU forward or measurement of runtime routing.',
        'limitations':['Apply these counts when the gate is cast to BF16 before expert input/output multiplication.',
            'FP32 activation/gate paths can preserve smaller differences; a declared BF16 matmul profile alone does not prove every intermediate dtype.',
            'Parameter preferences and rounding counts do not prove causal expert specialization.']}


def main():
    from src.utils.research_storage import _atomic_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--run',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=collect(args.source,args.run);args.output.parent.mkdir(parents=True,exist_ok=True)
    _atomic_json(args.output,report)
    print(json.dumps({'run':args.run,'groups':report['groups']}))


if __name__=='__main__':main()
