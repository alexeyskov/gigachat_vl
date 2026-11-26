#!/usr/bin/env python3
"""Finish evaluation from final-update weights, without optimizer/training."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from src.utils.research_capacity_eval_v5 import core,prior,definition
from src.utils.research_storage import _atomic_json,BestLastCheckpointManager,DiskBudget

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
 from src.utils.research_stock_determinism import activate
 activate();start=time.monotonic();path=run/'recovery/model.pt';receipt=json.loads((run/'recovery/receipt.json').read_text())
 if receipt['checkpoint_sha256']!=core.sha(path) or receipt['source_binding']!=definition():raise ValueError('Recovery receipt/source changed')
 model,module,inference,selection,saved=core.load_checkpoint(path)
 if (saved.get('recovery_kind')!='trained_pending_evaluation_v5' or saved['step']!=56 or saved['seen']!=3529
     or saved['config'].get('capacity_evaluation_v5')!=definition()):raise ValueError('Bound final-update snapshot required')
 module.log(run/'metrics.jsonl',dict(event='evaluation_only_recovery',optimizer_updates=0,source=str(path)))
 records=module.prepare_group_macro_selection(saved['config'])
 def progress(row):
  public={k:v for k,v in row.items() if k not in ('prediction','score')};public['step']=56
  _atomic_json(run/'selection_progress.json',public)
 result=selection.evaluate_internal_group_macro(model,records,max_new_tokens=64,progress=progress)
 ce=prior.multimodal_ce(model,records['records'],selection)
 _atomic_json(run/'multimodal_validation_ce.json',dict(**ce,training_pass=saved['config']['training_pass']))
 text=module.evaluate_text_only_ce(model,records['text_records'],4)
 metric=result['metrics']['group_macro_score']
 payload=dict(saved,selection_metric=result['protocol']['name'],selection_value=metric,
  selection_protocol=result['protocol'],selection_statistics=result['metrics'],
  elapsed_seconds=saved['elapsed_seconds']+time.monotonic()-start)
 from src.utils.research_budget_storage_v1 import payload_storage_budget
 manager=BestLastCheckpointManager(run,DiskBudget(run,32),mode='max')
 report=manager.save_torch(payload,step=56,metric=metric,
  anticipated_bytes=payload_storage_budget(payload)['anticipated_bytes'],
  metadata=dict(selection_metric=payload['selection_metric'],selection_direction='max',official_benchmark_score=False))
 module.write_selection_artifacts(run,result,56,is_best=report['best']['step']==56)
 module.log(run/'metrics.jsonl',dict(event='validation',step=56,group_macro_score=metric,records=116,evaluation_recovered=True))
 module.log(run/'metrics.jsonl',dict(event='text_validation',step=56,**(text or {})))
 _atomic_json(run/'completed.json',dict(steps=56,records=3529,elapsed_seconds=payload['elapsed_seconds'],evaluation_recovered=True))

if __name__=='__main__':main()
