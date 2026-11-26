#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.research_giga_budget_v2 import bootstrap,eval_main
if __name__=='__main__':
 bootstrap();eval_main(sys.argv[1:])
