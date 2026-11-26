#!/usr/bin/env python3
"""Strict native replay only for explicitly stock-deterministic future runs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.research_stock_determinism import bootstrap_environment, validate_main
bootstrap_environment()
if __name__ == "__main__":
    validate_main()
