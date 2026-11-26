#!/usr/bin/env python3
"""Prospective instrumented Small training; metadata only until explicit opt-in."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.research_routing_training_v1 import training_main

if __name__ == '__main__': raise SystemExit(training_main())
