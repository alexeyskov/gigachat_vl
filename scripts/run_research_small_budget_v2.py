#!/usr/bin/env python3
"""Explicit nativeQLoRA learned-modality-gate Smallbudget1c execution."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.research_small_budget_v2 import main
if __name__=='__main__':main()
