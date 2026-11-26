#!/usr/bin/env python3
"""Actual native GPU reload and exact 116-record generation replay."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from src.utils.research_stock_determinism import activate
from src.utils.research_additive_runner_v1 import load_checkpoint,sha
from src.utils.research_storage import _atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    activate();start=time.monotonic()
    model,module,inference,selection,saved=load_checkpoint(args.run/'last/state.pt')
    expected=[json.loads(line) for line in (args.run/'selection/last.predictions.jsonl').read_text().splitlines()]
    records=module.prepare_group_macro_selection(saved['config'])
    if len(records['records'])!=116 or len(expected)!=116:raise ValueError('Expected exact internal116')
    def progress(row):
        _atomic_json(args.run/'reload_progress.json',{k:v for k,v in row.items() if k not in ('prediction','score')})
    actual=selection.evaluate_internal_group_macro(model,records,max_new_tokens=64,progress=progress)
    mismatches=[]
    for index,(before,after) in enumerate(zip(expected,actual['predictions'],strict=True)):
        for field in ('prediction','score'):
            if before.get(field)!=after.get(field):mismatches.append(dict(index=index,field=field))
        for field in ('generated_token_ids','prompt_sha256'):
            if field not in before.get('generation',{}) or before['generation'][field]!=after.get('generation',{}).get(field):
                mismatches.append(dict(index=index,field=field))
    report=dict(status='passed' if not mismatches else 'failed',records=116,
        learned_tensors_exactly_restored=len(saved['model']),mismatches=mismatches,
        score=actual['metrics']['group_macro_score'],
        checkpoint_sha256=sha(args.run/'last/state.pt'),elapsed_seconds=time.monotonic()-start,
        architecture=saved['config']['additive_execution'],scope='actual native GPU generation; independent fresh model reconstruction')
    _atomic_json(args.run/'native_reload_proof.json',report)
    print(json.dumps(report),flush=True)
    if mismatches:raise ValueError('Independent additive checkpoint replay differs')

if __name__=='__main__':main()
