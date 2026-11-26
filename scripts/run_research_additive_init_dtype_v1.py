#!/usr/bin/env python3
"""Original fixed-branch training with explicit exact packed-LoRA read bridge."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if '--allow-gpu' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
from src.utils.research_additive_init_dtype_v1 import patched_runner
from src.utils.research_additive_runner_v1 import main
if __name__=='__main__':
    with patched_runner():main()
