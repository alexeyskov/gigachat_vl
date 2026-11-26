#!/usr/bin/env python3
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if '--allow-gpu' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
from src.utils.research_capacity_v4 import main
if __name__=='__main__':main()
