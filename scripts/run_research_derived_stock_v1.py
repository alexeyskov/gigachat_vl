#!/usr/bin/env python3
"""Local-only Qwen/Gemma vision exports with the explicit Small stock profile."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.research_derived_stock_v1 import training_main

if __name__ == '__main__':
    training_main()
