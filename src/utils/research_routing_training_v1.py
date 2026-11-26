"""Separate prospective stock training facade for transactional P06 aggregates."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import uuid
import zipfile

from src.utils import research_routing_aggregate_v1 as routing
from src.utils import research_stock_determinism as stock

ROOT = routing.ROOT
OWNER = 'stock_small_routing_instrumentation_v1'
FIELD = 'routing_aggregate_profile'
STATE = 'routing_aggregate_state'
OWN = ('src/utils/research_routing_training_v1.py', 'scripts/run_research_routing_aggregate_v1.py',
       'tests/test_research_routing_training_v1.py')
READY = ROOT / 'configs/research/routing_aggregate_training_v1_ready.json'


def _file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(8 * 2**20): value.update(block)
    return value.hexdigest()


def source_binding():
    return {**routing.source_binding(), **stock.definition()['source_sha256'],
        **{name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (*OWN, 'RESEARCH_PIPELINE.md')}}


def verify_release():
    value = json.loads(READY.read_bytes())
    if (value.get('owner') != OWNER or value.get('status') != 'cpu_tests_passed_actual_GPU_parity_pending' or
            value.get('source_sha256') != source_binding() or value.get('stock_numerical_profile') != stock.definition()):
        raise ValueError('Exact prospective routing collector CPU release required')
    return {'path': str(READY), 'sha256': hashlib.sha256(READY.read_bytes()).hexdigest()}


def definition(groups, *, gate_stride=10, inputs=None):
    return {'owner': OWNER, 'schema_version': 1, 'source_sha256': source_binding(),
        'stock_numerical_profile': stock.definition(), 'collector': routing.profile(groups, gate_stride=gate_stride),
        'training_inputs': inputs or {}, 'automatic_current_queue_adoption': False,
        'commit': 'after native optimizer.step, scheduler.step, CUDA synchronize, step and seen increment',
        'private_hooks': 'routing observations and transaction/checkpoint provenance only; native loss/RNG/math unchanged',
        'legacy_routing_jsonl_scope': 'unchanged last-forward snapshot; routing_aggregate_v1 is the new committed-window corpus aggregate',
        'actual_GPU_parity_and_overhead_validated': False, 'P06_complete': False}


def inspect_inputs(args):
    groups = set(); files = {}
    for field in ('manifest', 'text_manifest'):
        value = getattr(args, field, None)
        if not value: continue
        path = Path(value).resolve(strict=True); before = path.stat(); digest = hashlib.sha256()
        with path.open('rb') as handle:
            for line in handle:
                digest.update(line)
                if not line.strip(): continue
                record = json.loads(line); group = record.get('task_group'); identity = record.get('record_id')
                if not isinstance(group, str) or not group or not isinstance(identity, str) or not identity:
                    raise ValueError('Every training row needs explicit raw task_group and record_id for P06')
                groups.add(group)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Routing training manifest changed during metadata inspection')
        files[field] = {'path': str(path), 'sha256': digest.hexdigest()}
    if not groups: raise ValueError('Routing instrumentation needs a nonempty training manifest')
    return sorted(groups), files


def _clone_function(module, name, rewrite):
    source = (ROOT / 'scripts/run_research_stage.py').read_bytes()
    node = next(value for value in ast.parse(source).body if isinstance(value, ast.FunctionDef) and value.name == name)
    node = rewrite.visit(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(ROOT / 'scripts/run_research_stage.py'), 'exec'), module.__dict__)


def instrument_runner(module, config):
    """Private module only. No canonical model class or optimizer is patched."""
    if module.__name__ == 'scripts.run_research_stage': raise ValueError('Canonical runner instrumentation is forbidden')
    live = {}; original_model = module.ResearchModel
    class ObservedModel(original_model):
        def __init__(self, args):
            if getattr(args, FIELD, None) != config: raise ValueError('Exact prospective routing profile required')
            for field, expected in config['training_inputs'].items():
                actual = Path(getattr(args, field)).resolve(strict=True)
                if str(actual) != expected['path'] or _file_sha(actual) != expected['sha256']:
                    raise ValueError('Routing training input changed before model construction')
            super().__init__(args)
            collector = routing.RoutingAggregate(self.llm, config['collector']['task_groups'], gate_stride=config['collector']['gate_snapshot_stride'])
            self._routing_collector = collector; live['model'] = self
        def prepare(self, records):
            values = super().prepare(records)
            if len(values) != len(records): raise ValueError('Routing prepared-row provenance count differs')
            return [routing.TaggedRow(value, record) for value, record in zip(values, records)]
        def _forward_loss(self, rows, components=False):
            with self._routing_collector.forward_scope(rows): return super()._forward_loss(rows, components=components)
    module.ResearchModel = ObservedModel
    module._routing_begin_attempt = lambda model: model._routing_collector.begin_attempt()
    module._routing_abort_attempt = lambda model: model._routing_collector.abort_attempt()
    module._routing_commit = lambda model, step, seen: model._routing_collector.commit(step, seen)
    class Attempts(ast.NodeTransformer):
        begins = aborts = 0
        def visit_While(self, node):
            node = self.generic_visit(node)
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                node.body.insert(0, ast.parse('_routing_begin_attempt(model)').body[0]); self.begins += 1
            return node
        def visit_ExceptHandler(self, node):
            if ast.unparse(node.type) == 'torch.cuda.OutOfMemoryError':
                node.body.insert(0, ast.parse('_routing_abort_attempt(model)').body[0]); self.aborts += 1
            return self.generic_visit(node)
    attempts = Attempts(); _clone_function(module, 'train_accumulation_group', attempts)
    if (attempts.begins, attempts.aborts) != (1, 1): raise ValueError('Pinned native OOM retry structure changed')
    train = module.train_accumulation_group
    def accumulate(model, optimizer, prepared, *args, **kwargs):
        collector = model._routing_collector; collector.begin_window(prepared)
        try:
            result = train(model, optimizer, prepared, *args, **kwargs); collector.finish_attempt(); return result
        except BaseException:
            collector.abort_window(); raise
    module.train_accumulation_group = accumulate
    class Commit(ast.NodeTransformer):
        commits = 0
        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == 'seen' and ast.unparse(node.value) == 'count':
                self.commits += 1; return [node, ast.parse('_routing_commit(model, step, seen)').body[0]]
            return self.generic_visit(node)
    commits = Commit(); _clone_function(module, 'main', commits)
    if commits.commits != 1: raise ValueError('Pinned native successful optimizer-step boundary changed')
    original_binding = module.validate_model_binding
    def binding(payload, model):
        result = original_binding(payload, model)
        if getattr(model.args, 'resume', False):
            if payload.get('config', {}).get(FIELD) != config or STATE not in payload:
                raise ValueError('Legacy/mixed routing resume is refused; stage initialization is separate')
            model._routing_collector.restore_state(payload[STATE], step=payload['step'], seen=payload['seen'])
            # The model checkpoint is authoritative if the process died after
            # its atomic publication but before the derived JSON report. Repair
            # only from that exact owned restored state, with no forward/update.
            _write_report(Path(model.args.output), payload[STATE], config, _last_binding(model.args.output, payload[STATE]))
        return result
    module.validate_model_binding = binding
    manager = module.BestLastCheckpointManager
    class ObservedManager(manager):
        def save_torch(self, payload, *args, **kwargs):
            if payload.get('config', {}).get(FIELD) != config: raise ValueError('Routing checkpoint profile differs')
            collector = live['model']._routing_collector
            state = collector.export_state(step=payload['step'], seen=payload['seen'])
            result = super().save_torch(dict(payload, **{STATE: state}), *args, **kwargs)
            checkpoint = _last_binding(live['model'].args.output, state)
            _write_report(Path(live['model'].args.output), state, config, checkpoint)
            return result
    module.BestLastCheckpointManager = ObservedManager
    module._routing_live = live
    return module


def _last_binding(output, state):
    output = Path(output).resolve(strict=True); pointer = output / 'last'
    if not pointer.is_symlink(): raise ValueError('Routing report requires an owned LAST role')
    target = pointer.resolve(strict=True)
    if target.parent != output / '.checkpoint_objects': raise ValueError('Routing LAST escaped the manager object store')
    owner = json.loads((output / '.checkpoint_manager.json').read_bytes()).get('owner')
    marker = target / '.research_checkpoint.json'
    def identity():
        value = marker.stat(); return tuple(getattr(value, name) for name in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
    before = identity(); raw = marker.read_bytes(); parsed = json.loads(raw)
    if not owner or parsed.get('owner') != owner or parsed.get('step') != state['step'] or pointer.resolve() != target or identity() != before:
        raise ValueError('Routing report owner/LAST/aggregate step changed')
    return {'object': target.name, 'marker_sha256': hashlib.sha256(raw).hexdigest(), 'marker': parsed,
        'routing_state_sha256': state['state_sha256'], 'physical_model_hash_computed': False}


def _write_report(output, state, config, checkpoint):
    """Checkpoint-linked metrics; no source/model tensor bytes or gold targets."""
    import torch
    from src.utils.research_storage import DiskBudget, _atomic_json
    def plain(value):
        if torch.is_tensor(value): return value.tolist()
        if isinstance(value, dict): return {key: plain(child) for key, child in value.items()}
        if isinstance(value, list): return [plain(child) for child in value]
        return value
    target = output / 'routing_aggregate_v1'; target.mkdir(exist_ok=True)
    result = plain(state); result.update(metrics_scope='all committed training windows through this checkpoint', actual_training_collection=True)
    result.update(training_instrumentation_profile=config, checkpoint_binding=checkpoint)
    for name, value in state['layers'].items():
        counts = value['counters']; totals = counts['tokens']; dispatch = counts['dispatch_counts']
        def means(numerator, denominator):
            return [[None if float(denominator[g, m]) == 0 else float(numerator[g, m] / denominator[g, m]) for m in range(2)] for g in range(len(totals))]
        old = value['metadata']['old']; groups = []
        for g in range(len(totals)):
            modalities = []
            for m in range(2):
                n = int(totals[g, m]); selected = int(dispatch[g, m].sum())
                modalities.append({'valid_tokens': n, 'actual_dispatch_slots': selected,
                    'dead_experts_in_observed_stratum': None if not n else torch.where(dispatch[g, m] == 0)[0].tolist(),
                    'dispatch_fraction_per_expert': None if not selected else (dispatch[g, m].double() / selected).tolist(),
                    'expert_mean_effective_output_l2': [None if not int(k) else float(v / k) for v, k in zip(counts['output_l2_sum'][g, m], counts['norm_dispatch_counts'][g, m])],
                    'expert_mean_actual_weighted_contribution_l2': [None if not int(k) else float(v / k) for v, k in zip(counts['weighted_output_l2_sum'][g, m], counts['norm_dispatch_counts'][g, m])],
                    'pre_mask_old_new_mass_mean': None if not int(counts['full_entropy_tokens'][g, m]) else [float(counts['pre_mask_mass_sum'][g, m, :old].sum() / counts['full_entropy_tokens'][g, m]), float(counts['pre_mask_mass_sum'][g, m, old:].sum() / counts['full_entropy_tokens'][g, m])],
                    'eligible_old_new_mass_mean': None if not int(counts['eligible_entropy_tokens'][g, m]) else [float(counts['eligible_mass_sum'][g, m, :old].sum() / counts['eligible_entropy_tokens'][g, m]), float(counts['eligible_mass_sum'][g, m, old:].sum() / counts['eligible_entropy_tokens'][g, m])]})
            groups.append(modalities)
        result['layers'][name]['derived'] = {'task_group_text_visual': groups,
            'full_entropy_mean_nats': means(counts['full_entropy_sum'], counts['full_entropy_tokens']),
            'eligible_entropy_mean_nats': means(counts['eligible_entropy_sum'], counts['eligible_entropy_tokens']),
            'R3_alpha_mean': means(counts['alpha_sum'], totals) if value['metadata']['family'] == 3 else None,
            'shared_contribution': {'mechanism_present': False, 'structural_norm': 0},
            'overflow_dropped': {'capacity_mechanism_present': False, 'structural_count': 0}}
    raw = routing.canonical(result)
    DiskBudget(output, reserve_gb=120).ensure_space(len(raw) + 2**20)
    path = target / f'step-{state["step"]:08d}.json'
    if path.exists() and json.loads(path.read_bytes()) != result:
        raise ValueError('An existing routing checkpoint report cannot be silently relabeled')
    _atomic_json(path, result)


def _source_archive(output, config):
    from src.utils.research_storage import DiskBudget, _atomic_json
    output = Path(output); directory = output / 'routing_aggregate_source_v1'; output.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sorted(config['source_sha256'].items()):
            raw = (ROOT / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected: raise ValueError('Routing source changed before archive')
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)); archive.writestr(info, raw)
    data = stream.getvalue(); receipt = {'profile': config, 'source_zip_sha256': hashlib.sha256(data).hexdigest()}
    path = directory / 'source.zip'; marker = directory / 'source.json'
    if directory.exists():
        if not path.is_file() or not marker.is_file() or path.read_bytes() != data or json.loads(marker.read_bytes()) != receipt:
            raise ValueError('Prospective routing source archive changed')
    else:
        DiskBudget(output, reserve_gb=120).ensure_space(len(data) + 2**20)
        stage = output / ('.routing_aggregate_source_v1.' + uuid.uuid4().hex + '.tmp'); stage.mkdir()
        with (stage / 'source.zip').open('xb') as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
        _atomic_json(stage / 'source.json', receipt)
        descriptor = os.open(stage, os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        os.rename(stage, directory)
        descriptor = os.open(output, os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)


def training_main(argv=None):
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument('--allow-gpu', action='store_true')
    parser.add_argument('--metadata', action='store_true'); parser.add_argument('--numerical-profile', choices=(stock.PROFILE,))
    parser.add_argument('--allow-legacy-a0-initialization', action='store_true')
    parser.add_argument('--gate-snapshot-stride', type=int, default=10)
    options, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if options.allow_gpu and not options.metadata and 'torch' in sys.modules:
        raise RuntimeError('Prospective routing GPU training requires a fresh process before Torch import')
    stock.bootstrap_environment()
    if options.metadata or not options.allow_gpu:
        print(json.dumps({'status': 'prospective_metadata_only', 'source_sha256': source_binding(), 'GPU_used': False,
            'actual_collected_training_rows': 0, 'P06_complete': False}, indent=2)); return 0
    if options.numerical_profile != stock.PROFILE: raise ValueError('Explicit unchanged stock numerical profile required')
    verify_release()
    stock.activate(); module = stock.runner(); args = module.parse_args(remaining)
    if args.model != stock.SMALL: raise ValueError('Routing instrumentation supports pinned Small only')
    args.numerical_profile = stock.definition()
    args.allow_legacy_a0_initialization = options.allow_legacy_a0_initialization
    groups, inputs = inspect_inputs(args); config = definition(groups, gate_stride=options.gate_snapshot_stride, inputs=inputs)
    setattr(args, FIELD, config)
    existing = Path(args.output) / 'resolved_config.json'
    if existing.exists() and json.loads(existing.read_bytes()).get(FIELD) != config:
        raise ValueError('Existing run lacks the identical routing instrumentation profile; no historical output adoption')
    _source_archive(args.output, config); stock.execution_sources(args.output, create=True, reserve_gb=args.reserve_gb)
    module = instrument_runner(module, config); module.parse_args = lambda: args
    try: return module.main()
    finally:
        if 'model' in module._routing_live: module._routing_live['model']._routing_collector.close()
