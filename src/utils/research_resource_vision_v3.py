"""Resource-only clones of the frozen vision/Small composition contracts.

Feature computation, original export proofs and feature keys are unchanged.
The independent outer execution binding records this32GiB disk policy.
"""
from __future__ import annotations
import hashlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DERIVED = 'src/utils/research_derived_vision_v1.py'
COMPOSED = 'src/utils/research_derived_stock_v1.py'
_COMPOSED = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definition():
    return {'owner': 'resource32_vision_execution_v3', 'disk_floor_GiB': 32,
            'host_floor_GiB': 32, 'feature_math_or_cache_contract_changed': False,
            'source_sha256': {p: sha(ROOT / p) for p in
                              (DERIVED, COMPOSED, 'src/utils/research_resource_vision_v3.py')}}


def clone(relative, *, composed=False):
    source = (ROOT / relative).read_text()
    if composed:
        old = 'from src.utils import research_derived_vision_v1 as derived'
        assert source.count(old) == 1
        source = source.replace(old, 'from src.utils.research_resource_vision_v3 import derived')
    # Only admission/default arguments change; frozen feature functions remain
    # literal copies and retain their independent original export/key contract.
    replacements = {'args.reserve_gb<120': 'args.reserve_gb<32',
                    'reserve_gb<120': 'reserve_gb<32',
                    'args.reserve_gb < 120': 'args.reserve_gb < 32',
                    'reserve_gb < 120': 'reserve_gb < 32',
                    'reserve_gb=120': 'reserve_gb=32',
                    'Preserve120GiB': 'Preserve32GiB'}
    changed = 0
    for old, new in replacements.items():
        changed += source.count(old)
        source = source.replace(old, new)
    if changed < 3:
        raise ValueError('Frozen resource guard shape changed')
    name = '_resource32_' + ('composed' if composed else 'derived') + '_v3'
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / relative)
    sys.modules[name] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


derived = clone(DERIVED)


def composed():
    global _COMPOSED
    if _COMPOSED is None:
        module = clone(COMPOSED, composed=True)
        original = module.definition
        module.definition = lambda: dict(original(), resource_execution=definition())
        _COMPOSED = module
    return _COMPOSED
