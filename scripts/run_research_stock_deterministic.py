#!/usr/bin/env python3
"""Future Small jobs only; explicit stock numerical profile, frozen native math."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.research_stock_determinism import bootstrap_environment, train_main
bootstrap_environment()
if __name__ == "__main__":
    train_main()
