#!/usr/bin/env python3
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if '--allow-gpu' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
from src.utils import research_detection_training_v1 as training

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--allow-gpu',action='store_true');p.add_argument('--evaluate-existing',action='store_true');p.add_argument('--preflight',action='store_true');args=p.parse_args()
 job=json.loads(args.job.read_text());training.validate_release();configured=training.configure_args(job)
 if args.preflight:
  print(json.dumps(dict(status='CPU_arguments_preflight_passed',id=job['id'],config=vars(configured)),sort_keys=True));return
 if not args.allow_gpu:raise ValueError('--allow-gpu required')
 return training.train(job,evaluate_existing=args.evaluate_existing)

if __name__=='__main__':main()
