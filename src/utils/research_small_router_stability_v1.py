"""Explicit SmallThinker/Qwen-only fix for normalized-sigmoid underflow.

The frozen native sources and completed SigLIP runs remain unchanged. Private
SmallThinker classes retain the native calculation on nonzero-denominator rows;
only zero-denominator rows use the equivalent log-sigmoid/softmax formula.
"""
from __future__ import annotations
import ast
import hashlib
import inspect
import json
import sys
import textwrap
import types
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELD = 'small_qwen_router_numerics_v1'
QWEN = 'Qwen/Qwen2.5-VL-7B-Instruct'
OWN = 'src/utils/research_small_router_stability_v1.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definition():
    return {'name': 'small_qwen_normalized_sigmoid_zero_denominator_v1',
            'source_sha256': {OWN: sha(ROOT/OWN)},
            'scope': 'SmallThinker with derived Qwen bounded729, all stages and replay',
            'normal_rows': 'native arithmetic unchanged',
            'zero_denominator_rows': 'softmax(logsigmoid(float32(selected_logits)))',
            'topk_and_eligibility_unchanged': True,
            'masked_slots': 'exact zero; all-masked rows exact zero',
            'nonfinite_gradient_guard_preserved': True,
            'native_sources_modified': False,
            'existing_SigLIP_checkpoints_or_metrics_rewritten': False}


def normalize_sigmoid(logits, *, slot_valid=None, clamp_tiny=False):
    import torch
    from torch.nn import functional as F
    scores = logits.sigmoid()
    if slot_valid is not None:
        scores = scores*slot_valid
    denominator = scores.sum(-1, keepdim=True)
    zero = denominator == 0
    # Avoid 0/0 even in the inactive autograd branch of torch.where.
    safe_denominator = denominator.masked_fill(zero, 1)
    if clamp_tiny:
        safe_denominator = safe_denominator.clamp_min(torch.finfo(torch.float32).tiny)
    ordinary = scores/safe_denominator
    stable_logits = F.logsigmoid(logits.float())
    if slot_valid is not None:
        stable_logits = stable_logits.masked_fill(~slot_valid, -torch.inf)
        any_slot = slot_valid.any(-1, keepdim=True)
        stable_logits = torch.where(any_slot, stable_logits, 0.)
    fallback = stable_logits.softmax(-1).to(ordinary.dtype)
    if slot_valid is not None:
        fallback = fallback.masked_fill(~slot_valid, 0.)
    return torch.where(zero, fallback, ordinary)


class _Normalize(ast.NodeTransformer):
    count = 0

    def visit_If(self, node):
        if ast.unparse(node.test) == 'self.config.moe_primary_router_apply_softmax':
            self.count += 1
            node.orelse = ast.parse('weights = _safe_normalize_sigmoid(selected_logits)').body
        return self.generic_visit(node)


def private_smallthinker():
    from src.utils import research_stock_determinism as stock
    module = stock._clone('src/model/smallthinker.py')
    original = module.SmallThinkerMoeBlock.forward
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    transform = _Normalize()
    tree = transform.visit(tree)
    if transform.count != 1:
        raise ValueError('Exact single native normalized-sigmoid branch required')
    namespace = dict(original.__globals__, _safe_normalize_sigmoid=normalize_sigmoid)
    exec(compile(ast.fix_missing_locations(tree), str(ROOT/OWN), 'exec'), namespace)
    module.SmallThinkerMoeBlock.forward = namespace['forward']
    return module


def validate_execution(config):
    from src.utils import research_small_budget_v3 as base
    base.validate_execution(config)
    if config.get('vision') == QWEN:
        if config.get(FIELD) != definition():
            raise ValueError('Explicit source-bound Qwen router stability policy required')
    elif FIELD in config:
        raise ValueError('No implicit router-policy change for existing SigLIP runs')
    return True


@contextmanager
def patched_runner():
    from src.utils import research_stock_determinism as stock
    original = stock.runner
    private = private_smallthinker()

    def factory():
        module = original()
        module.register_smallthinker = private.register_smallthinker
        parse_native = module.parse_args
        resume_native = module.validate_resume_config
        Model = module.ResearchModel

        def parse(argv=None):
            args = parse_native(argv)
            if args.vision != QWEN:
                raise ValueError('Isolated router correction requires the Qwen donor')
            setattr(args, FIELD, definition())
            saved = Path(args.output)/'resolved_config.json'
            if saved.exists():
                validate_execution(json.loads(saved.read_bytes()))
            if args.initial_checkpoint:
                parents = [p for p in Path(args.initial_checkpoint).resolve(strict=True).parents
                           if (p/'resolved_config.json').is_file()]
                if not parents:
                    raise ValueError('Source-bound corrected predecessor required')
                validate_execution(json.loads((parents[0]/'resolved_config.json').read_bytes()))
            return args

        class CorrectedModel(Model):
            def __init__(self, args):
                if args.vision != QWEN or getattr(args, FIELD, None) != definition():
                    raise ValueError('Model construction lacks explicit router correction')
                super().__init__(args)
                for wrapper in self.wrappers:
                    if wrapper.family != 1 or wrapper.apply_softmax:
                        raise ValueError('Only current normalized-sigmoid R1/R3 are supported')
                    def weights(owner, scores, logits, indices, slot_valid):
                        return normalize_sigmoid(logits, slot_valid=slot_valid, clamp_tiny=True)
                    wrapper._routing_weights = types.MethodType(weights, wrapper)

        def resume(saved, current):
            validate_execution(saved)
            validate_execution(current)
            return resume_native(saved, current)

        module.parse_args = parse
        module.ResearchModel = CorrectedModel
        module.validate_resume_config = resume
        return module

    stock.runner = factory
    try:
        yield
    finally:
        stock.runner = original


def _vision(argv, *, run=False):
    args = list(sys.argv[1:] if argv is None else argv)
    if run:
        return json.loads((Path(args[args.index('--run')+1])/'resolved_config.json').read_bytes())['vision']
    return args[args.index('--vision')+1] if '--vision' in args else None


def main(argv=None):
    from src.utils import research_small_budget_v3 as base
    if _vision(argv) != QWEN:
        return base.main(argv)
    with patched_runner():
        return base.main(argv)


def eval_main(argv=None):
    from src.utils import research_small_budget_v3 as base
    if _vision(argv, run=True) != QWEN:
        return base.eval_main(argv)
    with patched_runner():
        return base.eval_main(argv)


def validate_main(argv=None):
    from src.utils import research_small_budget_v3 as base
    if _vision(argv, run=True) != QWEN:
        return base.validate_main(argv)
    with patched_runner():
        return base.validate_main(argv)


def implementation():
    return sys.modules[__name__]
