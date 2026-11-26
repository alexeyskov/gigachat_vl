#!/usr/bin/env python3
"""Archived Small text producer, gold-free metadata by default."""
import argparse,json,os,signal,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval import research_text_stock_archived_v2 as facade
def main(argv=None):
 argv=list(sys.argv[1:] if argv is None else argv)
 if '--allow-gpu' in argv and 'torch' in sys.modules:raise RuntimeError('Archived text GPU work requires a fresh process before Torch import')
 facade.stock.bootstrap_environment();os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 owner=facade.producer();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
 p.add_argument('--checkpoint',choices=('best','last'),default='best');p.add_argument('--public-bundle',type=Path,default=ROOT/'datasets/research/primary-text-public-en-v1')
 p.add_argument('--tasks',nargs='+',choices=owner.TASKS,default=list(owner.TASKS));p.add_argument('--numerical-profile',choices=(facade.stock.PROFILE,))
 p.add_argument('--output',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--allow-gpu',action='store_true');p.add_argument('--cache-dir',default='cache/vision');p.add_argument('--reserve-gb',type=float,default=120.)
 a=p.parse_args(argv);context=owner.public_context(a.public_bundle,'smallthinker4',tasks=a.tasks);plan=owner.learned_plan(a.run,a.checkpoint,context)
 if not a.allow_gpu:
  print(json.dumps({'status':'metadata_only_expected_SHA_full_admission_pending','records':len(context['records']),'plan':plan,'Torch_imported':'torch' in sys.modules,'weight_bodies_opened':False,'gold_reference_files_opened':False,'GPU_used':False},indent=2));return 0
 if a.numerical_profile!=facade.stock.PROFILE:p.error('Actual work requires explicit --numerical-profile stock_deterministic_small_v1')
 facade.verify_release();received=[];handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
 def interrupted(sig,_):
  if not received:received.append(sig);raise InterruptedError('Archived text interrupted; committed rows retained')
 try:
  for s in handlers:signal.signal(s,interrupted)
  result=owner.evaluate(context,plan,a.output,resume=a.resume,allow_gpu=True,cache_dir=a.cache_dir,reserve_gb=a.reserve_gb,progress=lambda row:print(json.dumps(row),flush=True))
 finally:
  for s,h in handlers.items():signal.signal(s,h)
 print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
