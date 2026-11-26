#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.research_stock_determinism import bootstrap_environment
bootstrap_environment()
from src.utils.research_small_budget_v3 import eval_main
if __name__=='__main__':eval_main()
