"""Explicit Giga QLoRA1c addendum; released1b/4f v1 remains unchanged."""
from __future__ import annotations
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREVIOUS='src/utils/research_giga_budget_v1.py';PREVIOUS_SHA='c1a4a58729e7cea3c9b241a97ec5e06cf15fcfd2dba3c00dfb1bcf510aa89e88'
COMMON='src/utils/research_lfm_budget_v2.py';COMMON_SHA='e6bf3517aeba4f1287530ec1b2dc94f0d164815ebe80d5cfa86c62f394764a3c'
PROFILE='budget_giga_qlora_learned_gates_v2'
OWN=('src/utils/research_giga_budget_v2.py','scripts/run_research_giga_budget_v2.py','scripts/eval_research_giga_budget_v2.py','scripts/validate_research_giga_budget_v2.py',PREVIOUS,COMMON,'src/model/research_giga_loading_v1.py','src/model/research_giga_peft.py','src/utils/research_budget_stream_save_v1.py')
_MODULE=None
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def implementation():
 global _MODULE
 if sha(ROOT/PREVIOUS)!=PREVIOUS_SHA or sha(ROOT/COMMON)!=COMMON_SHA:raise ValueError('Pinned Giga/common1c source changed')
 if _MODULE is not None:return _MODULE
 from src.utils import research_giga_budget_v1 as previous
 from src.utils.research_lfm_budget_v2 import build
 from src.model.research_giga_loading_v1 import tokenizer_proxy
 module=build(previous_api=previous.implementation(),stock_proxy=previous._Stock(),backend_type='deepseek_v3',model=previous.MODEL,revision=previous.REVISION,profile=PROFILE,own=OWN,sidecar='budget_giga_execution_v2',projector_validator=previous._projector,learned_validator=previous._learned,tokenizer_class=tokenizer_proxy())
 original=module.definition
 def definition():
  if sha(ROOT/PREVIOUS)!=PREVIOUS_SHA or sha(ROOT/COMMON)!=COMMON_SHA:raise ValueError('Pinned Giga/common1c source changed')
  result=original();result.pop('actual_full_size_LFM_GPU_validated',None)
  result.update(actual_full_size_Giga_GPU_validated=False,common_v2_source_sha256=COMMON_SHA,giga_v1_source_sha256=PREVIOUS_SHA,giga_loading_route='same pinned native config/tokenizer+ordinary scopedPEFT+native packedNF4 as releasedGigaV1',checkpoint_writer='same nativeTorch/new-owned-staging-only bounded fsync+FADV as GigaV1; complete optimizer/RNG retained')
  return result
 module.definition=definition;_MODULE=module;return module
def bootstrap():return implementation().bootstrap()
def definition():return implementation().definition()
def runner(*a,**k):return implementation().runner(*a,**k)
def train_main(argv=None):return implementation().train_main(argv)
def eval_main(argv=None):return implementation().eval_main(argv)
def validate_main(argv=None):return implementation().validate_main(argv)
