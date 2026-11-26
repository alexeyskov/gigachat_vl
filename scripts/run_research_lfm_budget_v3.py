#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.research_lfm_budget_v3 import bootstrap,train_main
bootstrap()
if __name__=='__main__':train_main()
