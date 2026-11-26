#!/usr/bin/env python3
"""All input/target context and image-byte validation, no GPU construction."""
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.research_detection_mix_queue_v1 import build_plan
from src.utils import research_detection_training_v1 as training
from src.utils.research_small_budget_v3 import fixed_clock

def main():
 import torch
 training.validate_release();plan=build_plan();reports=[]
 for key in ('smallthinker4','lfm25','giga31'):
  job=next(j for j in plan['jobs'] if j['model_key']==key);args=training.configure_args(job);module,_,_=training.backend(args)
  snapshot,_=module.resolve_model(args.model)
  if key=='giga31':
   from src.model.research_giga_loading_v2 import tokenizer_proxy
   tokenizer=tokenizer_proxy().from_pretrained(snapshot,local_files_only=True)
  else:tokenizer=module.AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
  tokenizer=fixed_clock(tokenizer)
  fake=SimpleNamespace(tokenizer=tokenizer,args=args,base=SimpleNamespace(config=SimpleNamespace(model_type={'smallthinker4':'smallthinker','lfm25':'lfm2_moe','giga31':'deepseek_v3'}[key])))
  maxima={};counts={};maximum_OD_target=0
  for name in ('sft_multimodal_train.jsonl','text_train.jsonl','detection_validation.jsonl','detection_test.jsonl'):
   values=training.load_rows(training.DATA/name);maximum=0
   for record in values:
    pre,post,answer=module.ResearchModel.tokenize(fake,record);expanded=len(pre)+len(post)+len(answer)+729*len(record.get('image_paths',[]))
    if expanded>4096:raise ValueError('No silent target truncation: '+record['record_id'])
    if record.get('task_group')=='object_detection':maximum_OD_target=max(maximum_OD_target,len(answer))
    maximum=max(maximum,expanded)
   maxima[name]=maximum;counts[name]=len(values)
  if maximum_OD_target>256:raise ValueError('Detection generation cap shorter than a gold target')
  reports.append(dict(model=key,counts=counts,max_expanded_tokens=maxima,max_detection_answer_tokens=maximum_OD_target))
  print(json.dumps(reports[-1]),flush=True)
 images=json.loads((training.DATA/'images.json').read_text())['images']
 for im in images:
  if training.sha(im['path'])!=im['file_sha256']:raise ValueError('COCO image bytes changed')
 if torch.cuda.is_initialized():raise ValueError('CPU preflight unexpectedly initialized CUDA')
 out=ROOT/'research_results/detection_mix_v1';out.mkdir(exist_ok=True,parents=True)
 (out/'data_context_preflight.json').write_text(json.dumps(dict(status='passed',jobs=18,tokenizer_reports=reports,image_assets_verified=len(images),CUDA_initialized=False,
  release_sha256=training.sha(training.DATA/'release.json')),indent=2)+'\n')
 print('All18 detection job profiles passed full-target context and image preflight')

if __name__=='__main__':main()
