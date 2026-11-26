#!/usr/bin/env python3
"""Generation only for checkpoints explicitly saved under the new stock profile."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.research_stock_determinism import bootstrap_environment, eval_main
bootstrap_environment()
if __name__ == "__main__":
    eval_main()
