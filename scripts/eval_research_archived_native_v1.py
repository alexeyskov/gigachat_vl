#!/usr/bin/env python3
"""Single guarded archive decode, frozen native image scoring/ablations."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval import research_archived_native_evaluation_v1 as facade
def main(argv=None):
 argv=list(sys.argv[1:] if argv is None else argv)
 if '--allow-gpu' in argv and 'torch' in sys.modules:raise RuntimeError('Single-decode GPU evaluation requires a fresh process before Torch import')
 facade.stock.bootstrap_environment();os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
 p.add_argument('--checkpoint',choices=('best','last'),default='best');p.add_argument('--output',type=Path);p.add_argument('--max-new-tokens',type=int,default=64);p.add_argument('--max-records',type=int)
 p.add_argument('--resume',action='store_true');p.add_argument('--gpu',type=int,default=0);p.add_argument('--cache-dir',default='cache/vision');p.add_argument('--reserve-gb',type=float,default=120.)
 p.add_argument('--image-ablation',choices=('none','shuffle','blank','text_only'),default='none');p.add_argument('--ablation-seed',type=int,default=42)
 p.add_argument('--numerical-profile',choices=(facade.stock.PROFILE,));p.add_argument('--allow-gpu',action='store_true');args=p.parse_args(argv)
 if not args.allow_gpu:
  value=facade.metadata(args.run,args.checkpoint);value['output']=str(facade.output_path(args.run,args.image_ablation,args.checkpoint,args.output));print(json.dumps(value,indent=2));return 0
 if args.numerical_profile!=facade.stock.PROFILE:p.error('Actual work requires explicit --numerical-profile stock_deterministic_small_v1')
 result=facade.evaluate(args);print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
