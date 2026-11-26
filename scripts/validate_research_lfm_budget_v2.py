#!/usr/bin/env python3
"""Explicit QLoRA1c learned-gate budget entry; v1 remains unchanged."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.research_lfm_budget_v2 import bootstrap,validate_main
bootstrap()
if __name__=="__main__":validate_main()
