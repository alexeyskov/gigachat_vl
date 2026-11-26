#!/usr/bin/env python3
"""Run native full checkpoint replay with the explicit duplicate-header bridge."""
import argparse,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--family',choices=['lfm25','giga31'],required=True)
args,rest=p.parse_known_args()
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
if args.family=='lfm25':from src.utils.research_lfm_payload_bridge_v1 import validate_main
else:from src.utils.research_giga_payload_bridge_v1 import validate_main
validate_main(['--allow-gpu',*rest])
