#!/usr/bin/env python3
"""Explicit new additive architecture; frozen legacy runners remain unchanged."""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if '--allow-gpu' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
from src.utils.research_additive_runner_v1 import main
if __name__ == '__main__':main()
