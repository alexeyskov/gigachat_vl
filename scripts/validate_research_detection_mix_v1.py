#!/usr/bin/env python3
"""Fresh native reload; exact weights,8OD+8VQA replay,64 heldout detection tests."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from src.utils import research_detection_training_v1 as training
from src.utils.research_storage import _atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);args=p.parse_args();job=json.loads(args.job.read_text());run=Path(job['output'])
 configured=training.configure_args(job)
 if json.loads((run/'resolved_config.json').read_text())!=vars(configured):raise ValueError('Reload detection config differs')
 start=time.monotonic();model,module,inference,selection,info=training.construct(configured,run/'last/state.pt',detection_weights=True)
 expected={r['record_id']:r for r in training.load_rows(run/'evaluation/after_detection_validation.predictions.jsonl')}
 records=training.load_rows(training.DATA/'detection_validation.jsonl');subset=[]
 for kind in ('class_all','spatial','all80','negative'):subset += [r for r in records if r['detection']['query_kind']==kind][:2]
 actual=training.evaluate_detection(model,inference,selection,subset)
 mismatches=[]
 for row in actual['predictions']:
  old=expected[row['record_id']]
  for field in ('prediction','format_valid','iou50','iou75'):
   if row[field]!=old[field]:mismatches.append(dict(id=row['record_id'],field=field))
  for field in ('generated_token_ids','prompt_sha256'):
   if row['generation'][field]!=old['generation'][field]:mismatches.append(dict(id=row['record_id'],field=field))
 legacy=module.prepare_group_macro_selection(configured)
 expected_legacy={r['record_id']:r for r in training.load_rows(run/'evaluation/after_internal116.predictions.jsonl')}
 with selection.selection_generation_mode(model):
  for record in legacy['records'][::14][:8]:
   generated=inference.generate_prediction(model,{k:record[k] for k in ('record_id','question','image_paths','image_hashes')},max_new_tokens=64)
   old=expected_legacy[record['record_id']]
   for field in ('prediction','generated_token_ids','prompt_sha256'):
    expected_value=old['prediction'] if field=='prediction' else old['generation'][field]
    if generated[field]!=expected_value:mismatches.append(dict(id=record['record_id'],field=field))
 def progress(row):_atomic_json(run/'test_progress.json',row)
 test=training.evaluate_detection(model,inference,selection,training.load_rows(training.DATA/'detection_test.jsonl'),progress=progress)
 training.store_evaluation(run,'heldout_detection_test',test)
 report=dict(status='passed' if not mismatches else 'failed',checkpoint_sha256=info['source_sha256'],
  learned_tensors_exactly_restored=info['learned_tensors_exactly_loaded'],detection_validation_examples_replayed=len(subset),
  internal116_examples_replayed=8,heldout_test_records=64,mismatches=mismatches,test_metrics=test['metrics'],
  source_execution=training.definition(),elapsed_seconds=time.monotonic()-start,
  scope='exact learned tensor restoration;8detection+8oldVQA generation replay;64heldout detection test; not a116-example full replay')
 _atomic_json(run/'native_reload_proof.json',report);print(json.dumps(report),flush=True)
 if mismatches:raise ValueError('Independent native detection/VQA replay mismatch')

if __name__=='__main__':main()
