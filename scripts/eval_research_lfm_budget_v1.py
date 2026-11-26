#!/usr/bin/env python3
"""Native greedy evaluation under the exact LFM budget profile."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.research_lfm_budget_v1 import bootstrap,eval_main
bootstrap()
if __name__=='__main__':eval_main()
