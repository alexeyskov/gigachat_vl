"""Small-only stock numerical facade over the sealed serial P07 owner.

The private copies change inference imports, namespace and provenance fields.
Gate endpoints, native generation, scorers and lifetime closures stay inherited.
Metadata inspection never opens checkpoint tensor bodies or constructs a model.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

from src.utils import research_stock_determinism as stock

ROOT = Path(__file__).resolve().parents[2]
OWNER = 'research_small_gate_control_stock_serial_p07_v1'
NAMESPACE = 'gate_controls_stock_p07_v1'
PARENT_READY = 'configs/research/gate_serial_evaluator_ready.json'
PARENT_READY_SHA = '71a5752032342765483ea20fe70b7a5bc973341ecec9077dde5d35a1359794e5'
PINS = {
    'src/eval/research_gate_evaluation.py': 'ed520c80593a9749761536abf3bac290e7d42ea55a439d7d5777acced7d02e0c',
    'src/eval/research_gate_controls.py': '1171e75bf9792a4ce2487898d40d9666ed64384409a724b521ed4bd92484c7e0',
}
NEW = ('src/eval/research_gate_stock_evaluation.py', 'scripts/eval_research_gate_stock_checkpoint.py',
       'tests/test_research_gate_stock_evaluation.py')
READY = ROOT / 'configs/research/gate_stock_serial_evaluator_ready.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_parent():
    if sha(ROOT / PARENT_READY) != PARENT_READY_SHA:
        raise ValueError('Sealed P07 owner release changed')
    ready = json.loads((ROOT / PARENT_READY).read_bytes())
    if ready.get('status') != 'cpu_proposal_tests_passed':
        raise ValueError('Sealed P07 release status differs')
    for name, expected in {**ready['source_sha256'], **PINS}.items():
        if sha(ROOT / name) != expected: raise ValueError('Sealed P07 dependency changed: ' + name)
    return ready


def execution_profile():
    return {'owner': OWNER, 'numerical_definition': stock.definition(),
        'model_scope': 'completed Small 1c/2c/3c SFT only; no baseline substitution',
        'checkpoint_format': 'native_Torch_ZIP_only', 'archived_checkpoints_supported': False,
        'parameter_dtype_policy': 'preserved native checkpoint dtypes; full 2c/3c language and projector FP32',
        'legacy_or_mixed_resume_allowed': False, 'custom_kernels': False}


def source_files():
    return {**verify_parent()['source_sha256'], PARENT_READY: PARENT_READY_SHA,
        **stock.definition()['source_sha256'], **{name: sha(ROOT / name) for name in NEW}}


class _LazyInference:
    def __init__(self): self.value = None
    def get(self):
        if self.value is None: self.value = stock.inference()
        return self.value
    def __getattr__(self, name): return getattr(self.get(), name)


def _clone(relative, native, controls=None):
    raw = (ROOT / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINS[relative]: raise ValueError('P07 source changed before private load')
    class Redirect(ast.NodeTransformer):
        changes = 0
        def visit_ImportFrom(self, node):
            if node.module == 'src.eval.research_inference':
                self.changes += 1
                return [ast.Assign([ast.Name(a.asname or a.name, ast.Store())],
                    ast.Attribute(ast.Name('_stock_native', ast.Load()), a.name, ast.Load())) for a in node.names]
            if node.module == 'src.eval' and len(node.names) == 1:
                alias = node.names[0]
                if alias.name == 'research_inference':
                    self.changes += 1
                    return ast.Assign([ast.Name(alias.asname or alias.name, ast.Store())],
                        ast.Call(ast.Attribute(ast.Name('_stock_native', ast.Load()), 'get', ast.Load()), [], []))
                if alias.name == 'research_gate_controls' and controls is not None:
                    self.changes += 1
                    return ast.Assign([ast.Name(alias.asname or alias.name, ast.Store())], ast.Name('_private_controls', ast.Load()))
            return node
    transform = Redirect(); tree = transform.visit(ast.parse(raw))
    expected = 2 if controls is None else 1
    if transform.changes != expected: raise ValueError('Pinned P07 import hook structure changed')
    module = ModuleType('_private_stock_' + Path(relative).stem)
    module.__file__ = str(ROOT / relative); module.__package__ = 'src.eval'
    module.__dict__.update(_stock_native=native, _private_controls=controls)
    exec(compile(ast.fix_missing_locations(tree), module.__file__, 'exec'), module.__dict__)
    return module


def _validate_loaded(model, loaded):
    stock.validate_config(vars(model.args))
    if model.args.stage != 'SFT' or model.args.recipe not in ('1c', '2c', '3c'):
        raise ValueError('P07 stock runtime requires Small 1c/2c/3c SFT')
    if loaded.get('numerical_profile') != stock.definition():
        raise ValueError('P07 loaded checkpoint lacks exact stock profile')
    if model.args.recipe in ('2c', '3c'):
        import torch
        for name, value in (*model.llm.named_parameters(), *model.projector.named_parameters()):
            if value.is_floating_point() and value.dtype != torch.float32:
                raise ValueError('Full P07 stock parameters must preserve FP32 masters: ' + name)


def producer():
    verify_parent(); native = _LazyInference()
    controls = _clone('src/eval/research_gate_controls.py', native)
    module = _clone('src/eval/research_gate_evaluation.py', native, controls)
    module.OWNER = OWNER; module.NAMESPACE = NAMESPACE
    original_reference = module.validate_reference
    def reference(context, directory, **kwargs):
        # Check the numeric contract before expensive original-row/pixel checks.
        path = Path(directory)
        for filename in ('protocol.json', 'model_provenance.json'):
            if module.read_json(path / filename).get('numerical_profile') != stock.definition():
                raise ValueError('P07 original512 must use the same exact stock profile')
        return original_reference(context, directory, **kwargs)
    module.validate_reference = reference
    original_prepare = module.prepare_context
    def prepare(run, **kwargs):
        config = stock.validate_run_context(run)
        if config.get('stage') != 'SFT' or config.get('recipe') not in controls.RECIPES:
            raise ValueError('P07 stock supports completed Small 1c/2c/3c SFT only')
        for role in ('best', 'last'):
            target = (Path(run) / role).resolve(strict=True)
            if module.read_json(target / '.research_checkpoint.json').get('checkpoint_archival'):
                raise ValueError('Archived P07 checkpoints require a separate reviewed lineage/decoder-memory adapter')
            with (target / 'state.pt').open('rb') as handle:
                if handle.read(4) != b'PK\x03\x04': raise ValueError('P07 stock v1 requires native Torch ZIP checkpoints')
        value = original_prepare(run, **kwargs)
        value['static']['stock_execution'] = execution_profile()
        value['static']['stock_source_files'] = {n: module.descriptor(ROOT / n) for n in source_files()}
        return value
    module.prepare_context = prepare
    original_assert = module.assert_context
    def assert_context(context, **kwargs):
        if context['static'].get('stock_execution') != execution_profile(): raise ValueError('P07 stock execution profile changed')
        expected = source_files(); files = context['static'].get('stock_source_files', {})
        if set(files) != set(expected): raise ValueError('P07 stock source census changed')
        for name, item in files.items():
            if item.get('path') != str((ROOT / name).resolve()) or item.get('sha256') != expected[name]:
                raise ValueError('P07 stock source binding changed')
            if module.stat(item['path']) != item['stat']: raise ValueError('P07 stock source stat changed')
        stock.validate_run_context(context['static']['base']['checkpoint']['run'])
        return original_assert(context, **kwargs)
    module.assert_context = assert_context
    original_runtime = module.NativeRuntime
    class StockRuntime(original_runtime):
        def load(self, *args, **kwargs):
            stock.activate(); model, loaded = super().load(*args, **kwargs); stock.assert_active()
            _validate_loaded(model, loaded); return model, loaded
        def describe(self, model, provenance):
            stock.assert_active(); result = super().describe(model, provenance)
            result['stock_execution'] = execution_profile(); return result
        def generate(self, *args, **kwargs):
            stock.assert_active(); result = super().generate(*args, **kwargs); stock.assert_active(); return result
    module.NativeRuntime = StockRuntime
    original_protocol = module.make_protocol
    def make_protocol(context, receipt, runtime, **kwargs):
        if runtime.get('stock_execution') != execution_profile(): raise ValueError('P07 runtime stock profile missing/changed')
        result = original_protocol(context, receipt, runtime, **kwargs)
        result['stock_execution'] = execution_profile(); return result
    module.make_protocol = make_protocol
    original_row = module.make_row
    def make_row(*args, **kwargs):
        result = original_row(*args, **kwargs)
        result['stock_execution_sha256'] = module.digest(execution_profile())
        result['stock_source_sha256'] = module.digest(source_files())
        if len(module.canonical(result)) > module.MAX_ROW_BYTES: raise ValueError('P07 stock row bound exceeded')
        return result
    module.make_row = make_row
    return module


def metadata_plan(run=None, *, checkpoint='best', condition='learned', manifest=None, native_cache=None):
    """Small files/stat only. Full admission/pixels/checkpoint SHA are deferred."""
    module = producer()
    if checkpoint not in ('best', 'last') or condition not in module.controls.CONDITIONS: raise ValueError('Unknown P07 role/condition')
    manifest = Path(manifest) if manifest is not None else ROOT / module.controls.MANIFEST
    manifest_file = module.descriptor(manifest)
    if manifest_file['sha256'] != module.controls.MANIFEST_SHA256: raise ValueError('P07 fixed manifest changed')
    result = {'owner': OWNER, 'status': 'metadata_profile_only_no_generation', 'execution_profile': execution_profile(),
        'source_sha256': source_files(), 'records': 512, 'generation_max_new_tokens': 64, 'condition': condition,
        'manifest': manifest_file, 'Torch_imported': 'torch' in sys.modules, 'GPU_used': False, 'checkpoint_tensor_bodies_opened': False,
        'full_admission_executed': False, 'native_reference_full512_verified': False,
        'learned64_guard_executed': False, 'P06_complete': False, 'P07_complete': False}
    if run is None: return result
    run = Path(run).resolve(strict=True); config = stock.validate_run_context(run)
    if config.get('stage') != 'SFT' or config.get('recipe') not in module.controls.RECIPES:
        raise ValueError('P07 stock requires completed Small 1c/2c/3c SFT')
    files = {n: module.descriptor(run / n) for n in ('.checkpoint_manager.json', 'completed.json', 'checkpoint_compaction.json', 'resolved_config.json')}
    completed = module.read_json(run / 'completed.json'); manager = module.read_json(run / '.checkpoint_manager.json')
    compaction = module.read_json(run / 'checkpoint_compaction.json')
    declared = Path(config.get('output', '')); declared = (declared if declared.is_absolute() else ROOT / declared).resolve()
    if (declared != run or not isinstance(manager.get('owner'), str) or not manager['owner'] or
            compaction.get('status') != 'completed' or compaction.get('training_completed') is not True or
            compaction.get('resumable_optimizer') is not False or Path(compaction.get('run_dir', '')).resolve() != run):
        raise ValueError('P07 completed owned optimizer compaction metadata required')
    if type(completed.get('steps')) is not int or completed['steps'] < 1 or completed.get('status') in ('failed', 'incomplete', 'interrupted'):
        raise ValueError('P07 completed SFT receipt required')
    roles = {}
    for name in ('best', 'last'):
        pointer = run / name
        if not pointer.is_symlink(): raise ValueError('P07 owned best/last roles required')
        target = pointer.resolve(strict=True); marker = module.read_json(target / '.research_checkpoint.json')
        if marker.get('checkpoint_archival'):
            raise ValueError('Archived P07 checkpoints require a separate reviewed lineage/decoder-memory adapter')
        if (target.parent != run / '.checkpoint_objects' or marker.get('owner') != manager.get('owner') or
                marker.get('training_completed') is not True or marker.get('optimizer_state_removed') is not True or
                marker.get('resumable_optimizer') is not False or type(marker.get('step')) is not int or
                not 0 < marker['step'] <= completed['steps'] or (name == 'last' and marker['step'] != completed['steps'])):
            raise ValueError('P07 completed compacted role metadata differs')
        if not any(item.get('new_object') == target.name for item in compaction.get('checkpoints', [])):
            raise ValueError('P07 role missing from completed compaction metadata')
        payload = target / 'state.pt'
        if payload.is_symlink() or not payload.is_file(): raise ValueError('P07 local checkpoint required')
        roles[name] = {'pointer': str(pointer.readlink()), 'marker': module.descriptor(target / '.research_checkpoint.json'), 'checkpoint_stat': module.stat(payload)}
    directory = Path(native_cache) if native_cache else run / 'evaluation/heldout_v3'
    for name in ('protocol.json', 'model_provenance.json'):
        if module.read_json(directory / name).get('numerical_profile') != stock.definition():
            raise ValueError('P07 original512 profile missing/changed')
    result.update(status='metadata_only_full_admission_pending', run=str(run), role=checkpoint, files=files, roles=roles,
        original512_directory=str(directory.resolve()), namespace=str(run / 'evaluation' / NAMESPACE / condition / checkpoint))
    return result


def verify_release():
    value = json.loads(READY.read_bytes())
    if (value.get('owner') != OWNER or value.get('status') != 'cpu_tests_passed_actual_gpu_not_run' or
            value.get('source_sha256') != source_files() or value.get('execution_profile') != execution_profile()):
        raise ValueError('Exact P07 stock facade CPU release required')
    return {'path': str(READY), 'sha256': sha(READY)}
