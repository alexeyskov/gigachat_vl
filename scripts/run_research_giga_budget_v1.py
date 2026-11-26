#!/usr/bin/env python3
"""Explicit Giga reduced-budget facade; no GPU without --allow-gpu."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.research_giga_budget_v1 import bootstrap,train_main
bootstrap()
if __name__=='__main__':train_main()
