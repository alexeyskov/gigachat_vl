"""Explicit SmallThinker stock-profile execution with local vision exports.

The stock numerical contract and derived source/cache contract are composed on
private runners. Original donor weight files are never resolved by this module.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from src.utils import research_derived_vision_v1 as derived
from src.utils import research_stock_determinism as stock

stock.bootstrap_environment()
ROOT = derived.ROOT
OWNER = 'research_derived_stock_small_v1'
FIELD = 'derived_stock_execution'
OWN = ('src/utils/research_derived_stock_v1.py',
       'scripts/run_research_derived_stock_v1.py',
       'scripts/eval_research_derived_stock_v1.py',
       'tests/test_research_derived_stock_v1.py')
GPU_PROOF = 'research_results/derived_vision_cuda_actual_v2/report.json'
GPU_SHA = '153c2cff9cba870105fe81cef44baebb5c88601dd856eb04e649bec6d7bc4b9d'


def definition():
    """Only code and bounded metadata; no Torch, gold, original weight reads."""
    proof = derived.read(ROOT / GPU_PROOF)
    if (derived.sha(ROOT / GPU_PROOF) != GPU_SHA or
            proof.get('status') != 'passed_technical_four_pairs_no_pruning_authority' or
            proof.get('completed_pairs') != 4):
        raise ValueError('Exact completed ROOT donor CUDA prerequisite required')
    return {'owner': OWNER, 'schema_version': 1,
            'stock_profile': stock.definition(),
            'derived_code_sha256': derived.code_binding(),
            'source_sha256': {name: derived.sha(ROOT / name) for name in OWN},
            'technical_donor_GPU_proof_sha256': GPU_SHA,
            'technical_proof_scope': 'two synthetic RGB fixtures per donor; not corpus/quality validation',
            'original_donor_weights_required': False, 'automatic_fallback': False}


def validate_config(config, binding):
    stock.validate_config(config)
    if config.get(FIELD) != definition():
        raise ValueError('Exact composed stock/derived config required; no legacy relabel')
    if any(config.get(k) != v for k, v in derived.config_contract(binding).items()):
        raise ValueError('Saved derived source/profile/cache contract differs')


def configure(args, binding):
    if args.model != stock.SMALL:
        raise ValueError('Composed profile supports pinned SmallThinker only')
    if not math.isfinite(args.reserve_gb) or args.reserve_gb < 120:
        raise ValueError('Preserve120GiB disk reserve')
    derived.configure_args(args, binding)
    expected = definition()
    if hasattr(args, FIELD) and getattr(args, FIELD) != expected:
        raise ValueError('Composed execution config changed')
    setattr(args, FIELD, expected)
    if hasattr(args, 'numerical_profile') and args.numerical_profile != stock.definition():
        raise ValueError('Existing numerical profile differs')
    args.numerical_profile = stock.definition()
    args.allow_legacy_a0_initialization = False
    return args


def validate_payload(payload, binding):
    validate_config(payload['config'], binding)
    derived.validate_payload_binding(payload, binding)
    if payload.get(FIELD) != definition():
        raise ValueError('Checkpoint lacks exact composed execution binding')


def runner(args, binding, *, model_ref=None):
    configure(args, binding)
    derived.assert_binding(binding, full=True, check=derived.host_guard)
    module = derived.install_private_runner_hooks(stock.runner(), binding,
        model_repo=args.model, model_ref=model_ref)
    manager = module.BestLastCheckpointManager
    native_binding = module.validate_model_binding

    class ComposedManager(manager):
        def save_torch(self, payload, *a, **k):
            result = dict(payload)
            if FIELD in result and result[FIELD] != definition():
                raise ValueError('Existing composed payload binding differs')
            result[FIELD] = copy.deepcopy(definition())
            validate_config(result['config'], binding)
            return super().save_torch(result, *a, **k)

    def check(payload, model):
        validate_payload(payload, binding)
        return native_binding(payload, model)

    module.BestLastCheckpointManager = ComposedManager
    module.validate_model_binding = check
    return module


def load_checkpoint(checkpoint, *, export_root=derived.DEFAULT_EXPORT_ROOT,
                    cache_dir='cache/vision', reserve_gb=120):
    if not math.isfinite(reserve_gb) or reserve_gb < 120:
        raise ValueError('Preserve120GiB disk reserve')
    if derived.host_guard() < 72 * 2**30:
        raise MemoryError('Composed checkpoint loading requires72GiB actual host headroom')
    with Path(checkpoint).open('rb') as handle:
        if handle.read(4) != b'PK\x03\x04':
            raise ValueError('Composed v1 supports native ZIP mmap checkpoints only; archive profile required')
    stock.activate()
    from src.utils.research_checkpoint_compat import load_research_payload
    from src.eval.research_inference import _validate_learned_state, freeze_for_generation
    saved = load_research_payload(Path(checkpoint), map_location='cpu', mmap=True)
    config = copy.deepcopy(saved['config'])
    ref = saved['vision_ref']
    binding = derived.resolve_derived_vision(ref['repo'], revision=ref['revision'],
        export_root=export_root, preprocessing_profile=config.get('vision_preprocessing_profile'))
    validate_payload(saved, binding)
    args = SimpleNamespace(**config)
    args.cache_dir = str(cache_dir)
    args.reserve_gb = reserve_gb
    module = runner(args, binding, model_ref=saved['model_ref'])
    model = module.ResearchModel(args)
    module.validate_model_binding(saved, model)
    _validate_learned_state(model, saved['model'])
    result = model.load_state_dict(saved['model'], strict=False)
    if result.unexpected_keys:
        raise ValueError('Unknown composed checkpoint tensor keys')
    provenance = {key: saved[key] for key in ('model_ref', 'vision_ref', 'cache_fingerprint')}
    provenance.update(stage=config['stage'], recipe=config['recipe'], training_step=saved['step'],
        numerical_profile=stock.definition(), **{FIELD: definition()},
        vision_export_ref=binding['export_ref'], vision_profile_ref=binding['profile_ref'])
    del saved
    stock.assert_active()
    return freeze_for_generation(model), provenance


def training_main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--allow-gpu', action='store_true')
    parser.add_argument('--export-root', type=Path, default=derived.DEFAULT_EXPORT_ROOT)
    parser.add_argument('--numerical-profile', choices=[stock.PROFILE], required=True)
    opts, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    # Parse only public repo/profile values in the default no-Torch metadata path.
    public = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    public.add_argument('--model', default=stock.SMALL)
    public.add_argument('--vision', required=True)
    public.add_argument('--vision-preprocessing-profile')
    values, _ = public.parse_known_args(remaining)
    if values.model != stock.SMALL:
        raise ValueError('Composed profile supports pinned SmallThinker only')
    binding = derived.resolve_derived_vision(values.vision, export_root=opts.export_root,
        preprocessing_profile=values.vision_preprocessing_profile)
    if not opts.allow_gpu:
        print(json.dumps({'status': 'metadata_only_no_training', FIELD: definition(),
            'binding': binding, 'GPU_launched': False, 'original_donor_files_opened': False}, indent=2))
        return 0
    stock.activate()
    args = stock.runner().parse_args(remaining)
    configure(args, binding)
    config_path = Path(args.output) / 'resolved_config.json'
    if config_path.exists():
        validate_config(derived.read(config_path), binding)
    stock.execution_sources(args.output, create=True, reserve_gb=args.reserve_gb)
    module = runner(args, binding)
    module.parse_args = lambda: args
    return module.main()


def evaluator(*, export_root=derived.DEFAULT_EXPORT_ROOT):
    from functools import partial
    from types import ModuleType
    from src.eval import research_inference as original
    native = ModuleType('_derived_stock_native_loader')
    native.load_research_checkpoint = (load_checkpoint if export_root == derived.DEFAULT_EXPORT_ROOT
        else partial(load_checkpoint, export_root=export_root))
    for name in ('GENERATION_SCHEMA_VERSION', 'generate_prediction', 'sha256_file'):
        setattr(native, name, getattr(original, name))
    module = stock._clone('scripts/eval_research_checkpoint.py',
        {'src.eval.research_inference': native}, bind_eval_protocol=True)
    bind = module._bind_stock_protocol
    module._bind_stock_protocol = lambda value: dict(bind(value), **{FIELD: definition()})
    return module


def evaluation_main(argv=None):
    public = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    public.add_argument('--export-root', type=Path, default=derived.DEFAULT_EXPORT_ROOT)
    values, forwarded = public.parse_known_args(sys.argv[1:] if argv is None else argv)
    opts, remaining = stock._options(forwarded)
    if opts.metadata:
        print(json.dumps(definition(), indent=2))
        return 0
    if not opts.allow_gpu:
        raise ValueError('Evaluation requires explicit --allow-gpu')
    stock.activate()
    module = evaluator(export_root=values.export_root)
    parsed = module.parse_args(remaining)
    stock.validate_run_context(parsed.run)
    config = derived.read(Path(parsed.run) / 'resolved_config.json')
    binding = derived.resolve_derived_vision(config['vision'],
        export_root=values.export_root, preprocessing_profile=config.get('vision_preprocessing_profile'))
    validate_config(config, binding)
    return module.main(remaining)
