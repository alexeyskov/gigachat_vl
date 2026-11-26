#!/usr/bin/env python3
"""CPU-only native turn-prefix audit after both Giga full controls finish."""
import hashlib,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
OUT=ROOT/'research_results/capacity_eval_v5/full_controls_native_eot'
NAMES=('giga31__siglip2__4f__SFT__budget18_s42','giga31__qwen25__4f__SFT__budget18_s42')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 runs=[ROOT/'runs/research'/name for name in NAMES]
 if not all((r/'completed.json').exists() and (r/'selection/best.predictions.jsonl').exists() for r in runs):return
 previous=OUT/'report.json'
 if previous.exists():
  data=json.loads(previous.read_text())
  if all(any(row['id']==r.name and row['source_predictions_sha256']==sha(r/'selection/best.predictions.jsonl') for row in data['rows']) for r in runs):return
 from huggingface_hub import snapshot_download
 from src.model.research_giga_loading_v2 import tokenizer_proxy
 from src.eval.research_giga_eot_v1 import REVISION,terminated_prefix
 from src.eval.research_metrics import aggregate_scores,score_prediction
 snapshot=snapshot_download('ai-sage/GigaChat3.1-10B-A1.8B-bf16',revision=REVISION,local_files_only=True)
 tokenizer=tokenizer_proxy().from_pretrained(snapshot,local_files_only=True)
 selected=[json.loads(l) for l in (ROOT/'datasets/research/budget18-en-v1/internal_selection_116.jsonl').read_text().splitlines()]
 OUT.mkdir(parents=True,exist_ok=True);report=dict(status='completed_CPU_native_turn_end_prefix_audit',rows=[],
  scope='saved generated tokens truncated at native assistant EOT128001; original predictions/scores preserved; no GPU generation')
 for run in runs:
  path=run/'selection/best.predictions.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()]
  if len(rows)!=116 or len(selected)!=116:raise ValueError('Exact116 population required')
  scored=[];stops=0
  for record,row in zip(selected,rows,strict=True):
   if record['record_id']!=row['record_id']:raise ValueError('Selection order differs')
   ids=row['generation']['generated_token_ids']
   if tokenizer.decode(ids,skip_special_tokens=True).strip()!=row['prediction']:raise ValueError('Saved token IDs differ from original prediction')
   prefix=terminated_prefix(ids,tokenizer);stops+=int(prefix['native_turn_end_found'])
   scored.append(dict(row,prediction=prefix['prediction'],score=score_prediction(record,prefix['prediction'])))
  temp=OUT/(run.name+'.predictions.jsonl.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored));temp.replace(OUT/(run.name+'.predictions.jsonl'))
  report['rows'].append(dict(id=run.name,source_predictions_sha256=sha(path),original=aggregate_scores(rows),native_turn_end=aggregate_scores(scored),records_with_native_turn_end=stops,total_records=116))
 temp=OUT/'report.json.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temp.replace(previous)
 print(json.dumps(dict(status=report['status'],rows=len(report['rows']))))

if __name__=='__main__':main()
