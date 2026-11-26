#!/usr/bin/env python3
"""Native evaluation of explicitly composed stock/derived checkpoints."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.research_derived_stock_v1 import evaluation_main

if __name__ == '__main__':
    evaluation_main()
