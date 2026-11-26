"""Opt-in Small1c budget execution, preserving the released v1/runtime math.

The1b/4f jobs continue to usev1. This private clone changes only the additional
execution/source contract and admits the preregistered1b NF4 A0→1c A1 edge.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[2]
BASE = 'src/utils/research_small_budget_v1.py'
BASE_SHA = '4bdfe3ea66eabd2249a7ad7c253f2d9bb938ba06214882cce3371b39193517d6'
OWN = ('src/utils/research_small_budget_v2.py',
       'scripts/run_research_small_budget_v2.py', BASE)
OWNER = 'small_stock_budget18_execution_1c_v2'
_MODULE = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation():
    global _MODULE
    if _sha(ROOT / BASE) != BASE_SHA:
        raise ValueError('Released Smallbudgetv1 source changed')
    if _MODULE is not None:
        return _MODULE
    tree = ast.parse((ROOT / BASE).read_bytes())
    changes = {'recipe': 0, 'sidecar': 0}

    class Extend(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == '3c':
                changes['recipe'] += 1
                return ast.copy_location(ast.Constant('1c'), node)
            if node.value == 'budget18_execution_v1':
                changes['sidecar'] += 1
                return ast.copy_location(ast.Constant('budget18_execution_small_1c_v2'), node)
            return node

    tree = Extend().visit(tree)
    if changes != {'recipe': 1, 'sidecar': 1}:
        raise ValueError('Expected narrow Smallv1 extension points changed')
    private = types.ModuleType('_research_small_budget_1c_v2')
    private.__file__ = str(ROOT / BASE)
    exec(compile(ast.fix_missing_locations(tree), private.__file__, 'exec'), private.__dict__)
    private.OWN, private.OWNER = OWN, OWNER
    native_definition, native_configure = private.definition, private.configure
    native_initial_origin = private.initial_origin

    def definition():
        if _sha(ROOT / BASE) != BASE_SHA:
            raise ValueError('Released Smallbudgetv1 source changed')
        result = native_definition()
        result.update(owner=OWNER, execution_version=2, supported_recipe='1c',
                      frozen_v1_source_sha256=BASE_SHA,
                      routing='native family1 frozen old router plus trainable RouterMLP residual',
                      modality='native per-expert learned modality gates c',
                      language_backbone='native NF4 frozen base plus native rank16 LoRA',
                      numerical_profile_unchanged=True,
                      source_A0_recipe='1b', source_A0_projector_only=True,
                      legacy_A1_or_SFT_relabel_allowed=False)
        return result

    def initial_origin(args):
        # Only A0 is shared. A1 and SFT belong to this exact1c v2 execution.
        if args.stage == 'A1' and not args.resume:
            from src.utils import research_small_budget_v1 as original
            equivalent = copy.copy(args)
            equivalent.recipe = '1b'
            origin = original.initial_origin(equivalent)
            if origin['stage'] != 'A0' or origin['recipe'] != '1b':
                raise ValueError('1c requires the exact completed NF4 A0 recipe1b')
            return dict(origin, transition='shared_exact_NF4_1b_A0_to_native_1c_A1',
                        target_recipe='1c', v1_source_sha256=BASE_SHA)
        return native_initial_origin(args)

    def configure(args, proof_path):
        if args.recipe != '1c' or args.stage not in ('A1', 'SFT'):
            raise ValueError('Smallv2 only runs native1c A1/SFT; shared A0 and1b/4f usev1')
        return native_configure(args, proof_path)

    private.definition, private.initial_origin, private.configure = definition, initial_origin, configure
    _MODULE = private
    return private


def definition():
    return implementation().definition()


def main(argv=None):
    return implementation().main(argv)

