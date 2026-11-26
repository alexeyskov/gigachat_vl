"""Native Giga numerical profile with an explicit completed-R2 read bridge."""
from src.utils import research_giga_budget_v3 as base
from src.utils.research_qwen_payload_bridge_v1 import invoke
def definition():return base.definition()
def implementation():return base.implementation()
def train_main(argv=None):return invoke(base,'train_main',argv)
def eval_main(argv=None):return invoke(base,'eval_main',argv)
def validate_main(argv=None):return invoke(base,'validate_main',argv)
