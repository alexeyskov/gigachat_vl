#!/usr/bin/env python3
"""SmallThinker budget18 guards; stock numerical profile and native math."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.research_stock_determinism import bootstrap_environment
bootstrap_environment()
from src.utils.research_small_budget_v1 import main
if __name__=='__main__':main()
