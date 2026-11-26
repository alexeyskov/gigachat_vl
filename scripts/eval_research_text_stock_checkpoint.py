#!/usr/bin/env python3
"""Stock text producer facade; metadata-only by default, no gold/weight downloads."""
import argparse,json,os
from pathlib import Path
import signal,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval import research_text_stock_generation as facade

def main(argv=None):
 argv=list(sys.argv[1:] if argv is None else argv)
 if '--allow-gpu' in argv and 'torch' in sys.modules:raise RuntimeError('Stock GPU text work requires a fresh process before Torch import')
 facade.stock.bootstrap_environment()
 os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 owner=facade.producer();p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--public-bundle',type=Path,default=ROOT/'datasets/research/primary-text-public-en-v1')
 p.add_argument('--model-id',choices=('smallthinker4',),default='smallthinker4')
 choice=p.add_mutually_exclusive_group(required=True);choice.add_argument('--baseline-precision',choices=('bf16','nf4',owner.FULL_INIT_CONTROL));choice.add_argument('--run',type=Path)
 p.add_argument('--checkpoint',choices=('best','last'),default='best');p.add_argument('--tasks',nargs='+',choices=owner.TASKS,default=list(owner.TASKS))
 p.add_argument('--numerical-profile',choices=(facade.stock.PROFILE,));p.add_argument('--output',type=Path);p.add_argument('--resume',action='store_true')
 p.add_argument('--allow-gpu',action='store_true');p.add_argument('--cache-dir',default='cache/vision');p.add_argument('--reserve-gb',type=float,default=120.)
 a=p.parse_args(argv);context=owner.public_context(a.public_bundle,a.model_id,tasks=a.tasks)
 plan=owner.learned_plan(a.run,a.checkpoint,context) if a.run else owner.baseline_plan(context,a.baseline_precision)
 if not a.allow_gpu:
  print(json.dumps({'status':'blocked_missing_local_weights' if plan.get('missing_weights') else 'metadata_preflight_only_no_generation',
   'records':len(context['records']),'tasks':a.tasks,'plan':plan,'gold_reference_files_opened':False,'Torch_imported':'torch' in sys.modules,'GPU_used':False,'weight_downloads_allowed':False},indent=2));return 0
 if a.numerical_profile!=facade.stock.PROFILE or not a.output:p.error('Actual work requires explicit --numerical-profile stock_deterministic_small_v1 and NEW --output')
 facade.verify_release();handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)};received=[]
 def interrupted(signum,_):
  if not received:received.append(signum);raise InterruptedError('Stock text producer interrupted; committed rows retained')
 try:
  for s in handlers:signal.signal(s,interrupted)
  result=owner.evaluate(context,plan,a.output,resume=a.resume,allow_gpu=True,cache_dir=a.cache_dir,reserve_gb=a.reserve_gb,progress=lambda row:print(json.dumps(row),flush=True))
 finally:
  for s,h in handlers.items():signal.signal(s,h)
 print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
