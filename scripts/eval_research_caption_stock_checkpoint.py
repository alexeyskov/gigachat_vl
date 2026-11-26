#!/usr/bin/env python3
"""Stock caption metadata by default; exclusive native GPU work is opt-in."""
import argparse,json,os,signal,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval import research_caption_stock_generation as caption
def main(argv=None):
 argv=list(sys.argv[1:] if argv is None else argv)
 if '--allow-gpu' in argv and 'torch' in sys.modules:raise RuntimeError('Stock caption GPU work requires a fresh process before Torch import')
 caption.stock.bootstrap_environment();os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',choices=('best','last'),default='best')
 p.add_argument('--bundle',type=Path,default=caption.DEFAULT_BUNDLE);p.add_argument('--output',type=Path);p.add_argument('--cache-dir',default='cache/vision')
 p.add_argument('--numerical-profile',choices=(caption.stock.PROFILE,));p.add_argument('--allow-gpu',action='store_true');p.add_argument('--score-only',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--reserve-gb',type=float,default=120.)
 a=p.parse_args(argv)
 if not a.allow_gpu and not a.score_only:
  print(json.dumps({'status':'metadata_only_full_preflight_pending','plan':caption.metadata_plan(a.run,a.bundle,checkpoint=a.checkpoint),
   'Torch_imported':'torch' in sys.modules,'weight_bodies_opened':False,'gold_references_opened':False,'GPU_used':False},indent=2));return 0
 if a.numerical_profile!=caption.stock.PROFILE:p.error('Actual/score-only work requires explicit --numerical-profile stock_deterministic_small_v1')
 caption.verify_release();received=[];handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
 def interrupted(sig,_):
  if not received:received.append(sig);raise InterruptedError('Stock caption interrupted; committed rows preserved')
 try:
  for s in handlers:signal.signal(s,interrupted)
  context=caption.prepare_context(a.run,a.bundle,checkpoint=a.checkpoint,cache_dir=a.cache_dir)
  result=caption.evaluate(context,a.output,allow_gpu=a.allow_gpu,score_only=a.score_only,resume=a.resume,reserve_gb=a.reserve_gb,progress=lambda row:print(json.dumps(row),flush=True))
 finally:
  for s,h in handlers.items():signal.signal(s,h)
 print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
