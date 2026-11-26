"""Separate P07 consumer of the shared owned-archive admission/CPU loader.

Only private eligibility/import/provenance hooks differ from the sealed raw
stock gate owner. Native gate endpoints, prompts, scoring and closures remain.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import ModuleType

from src.eval import research_gate_stock_evaluation as raw
from src.utils import research_archived_eval_v1 as shared

ROOT = raw.ROOT
OWNER = 'research_small_gate_control_archived_stock_p07_v1'
NAMESPACE = 'gate_controls_archived_stock_p07_v1'
RAW_READY_SHA = 'eef513e49669abf3ebc326b10015c2a1f5baf8da0864341ec05c9be35abcb0d1'
RAW_SOURCE = 'src/eval/research_gate_stock_evaluation.py'
RAW_SOURCE_SHA = '1a659c1e6e7bd6f636cdd183551d94a24b3114a8c5f4926e7d737257b02b384d'
SHARED_READY_SHA = '5258d5b454d77c87c349ad2bb3819143d65abaff701669b2f271e534434c2738'
OWN = ('src/eval/research_gate_archived_evaluation_v1.py', 'scripts/eval_research_gate_archived_checkpoint_v1.py',
       'tests/test_research_gate_archived_evaluation_v1.py')
READY = ROOT / 'configs/research/gate_archived_serial_evaluator_v1_ready.json'
ARCHIVE_FIELDS = ('archived_evaluation', 'archived_publication_sha256', 'archived_physical_sha256')


def verify_parents():
    if raw.sha(raw.READY) != RAW_READY_SHA or raw.sha(ROOT / RAW_SOURCE) != RAW_SOURCE_SHA:
        raise ValueError('Sealed raw stock P07 release changed')
    raw.verify_release()
    if shared.sha(shared.READY) != SHARED_READY_SHA: raise ValueError('Shared archived decoder release changed')
    shared.verify_release()


def source_files():
    verify_parents()
    return {**raw.source_files(), **shared.source_binding(),
        str(raw.READY.relative_to(ROOT)): RAW_READY_SHA,
        str(shared.READY.relative_to(ROOT)): SHARED_READY_SHA, **{name: raw.sha(ROOT / name) for name in OWN}}


def execution_profile():
    return {**raw.execution_profile(), 'owner': OWNER, 'checkpoint_format': 'owned_guarded_native_legacy',
        'archived_checkpoints_supported': True, 'raw_checkpoints_supported': False,
        'archived_loader': shared.execution_profile(), 'source_sha256': source_files(),
        'loaded_reference_comparison': 'canonical native provenance remains exact; archive loader receipt is separate'}


def _private_raw():
    verify_parents(); source = (ROOT / RAW_SOURCE).read_bytes()
    private = ModuleType('_private_archived_stock_gate_facade_v1'); private.__file__ = str(ROOT / RAW_SOURCE); private.__package__ = 'src.eval'
    exec(compile(source, private.__file__, 'exec'), private.__dict__)
    private.OWNER = OWNER; private.NAMESPACE = NAMESPACE
    private.execution_profile = execution_profile; private.source_files = source_files
    # Replace the one explicit raw-only rejection loop with the shared lineage
    # context below. No scoring, gate, generation or lifetime branch is removed.
    tree = ast.parse(source); factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'producer')
    class Eligibility(ast.NodeTransformer):
        removed = 0
        def visit_FunctionDef(self, node):
            if node.name == 'prepare':
                kept = []
                for item in node.body:
                    errors = [n.value for n in ast.walk(item) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
                    if isinstance(item, ast.For) and 'Archived P07 checkpoints require a separate reviewed lineage/decoder-memory adapter' in errors:
                        self.removed += 1
                    else: kept.append(item)
                node.body = kept
            return self.generic_visit(node)
    edit = Eligibility(); factory = edit.visit(factory)
    if edit.removed != 1: raise ValueError('Exactly one pinned raw-only eligibility loop must be replaced')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])), private.__file__, 'exec'), private.__dict__)
    base_clone = private._clone
    def clone(relative, native, controls=None):
        module = base_clone(relative, native, controls)
        if relative == 'src/eval/research_gate_controls.py':
            original = (ROOT / relative).read_bytes()
            node = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_context')
            class Context(ast.NodeTransformer):
                changed = 0
                def visit_ImportFrom(self, value):
                    if value.module == 'src.eval.research_caption_generation' and any(n.name == 'checkpoint_context' for n in value.names):
                        self.changed += 1
                        rest = [n for n in value.names if n.name != 'checkpoint_context']
                        return [ast.ImportFrom(value.module, rest, value.level),
                            ast.Assign([ast.Name('checkpoint_context', ast.Store())], ast.Name('_archive_checkpoint_context', ast.Load()))]
                    return value
            rewrite = Context(); node = rewrite.visit(node)
            if rewrite.changed != 1: raise ValueError('Pinned gate checkpoint-context import changed')
            module.__dict__['_archive_checkpoint_context'] = shared.checkpoint_context
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), module.__file__, 'exec'), module.__dict__)
        return module
    private._clone = clone
    class LazyArchived(private._LazyInference):
        def get(self):
            if self.value is None: self.value = shared.inference()
            return self.value
    private._LazyInference = LazyArchived
    return private


def producer():
    module = _private_raw().producer(); runtime = module.NativeRuntime
    class ArchivedRuntime(runtime):
        def load(self, *args, **kwargs):
            model, loaded = super().load(*args, **kwargs)
            receipt = {key: loaded.get(key) for key in ARCHIVE_FIELDS}
            if receipt['archived_evaluation'] != shared.execution_profile():
                raise ValueError('P07 shared archive loader provenance missing/changed')
            if any(not isinstance(receipt[key], str) or not module.HEX.fullmatch(receipt[key]) for key in ARCHIVE_FIELDS[1:]):
                raise ValueError('P07 archive physical/publication SHA missing')
            self.archive_receipt = receipt
            return model, {key: value for key, value in loaded.items() if key not in ARCHIVE_FIELDS}
        def describe(self, model, provenance):
            value = super().describe(model, provenance); value['archived_checkpoint_decode'] = self.archive_receipt
            return value
    module.NativeRuntime = ArchivedRuntime
    original_protocol = module.make_protocol
    def protocol(context, receipt, runtime, **kwargs):
        loaded = runtime.get('archived_checkpoint_decode', {})
        checkpoint = context['static']['base']['checkpoint']
        report = checkpoint['files'][shared._publisher().REPORT]
        if (loaded.get('archived_evaluation') != shared.execution_profile() or
                loaded.get('archived_physical_sha256') != checkpoint['artifact']['sha256'] or
                loaded.get('archived_publication_sha256') != report['sha256']):
            raise ValueError('P07 archive decoder/reference context differs')
        return original_protocol(context, receipt, runtime, **kwargs)
    module.make_protocol = protocol
    return module


def metadata_plan(run=None, *, checkpoint='best', condition='learned', manifest=None, native_cache=None):
    verify_parents()
    if condition not in ('learned', 'all_open', 'visual_closed', 'text_closed', 'all_closed'): raise ValueError('Unknown P07 condition')
    # The sealed raw helper supplies the exact fixed-manifest/source-only gate;
    # passing no run deliberately avoids its raw-only checkpoint eligibility.
    result = raw.metadata_plan(None, checkpoint=checkpoint, condition=condition, manifest=manifest)
    result.update(owner=OWNER, execution_profile=execution_profile(), source_sha256=source_files())
    if run is None: return result
    context = shared.inspect(run, checkpoint)
    if context['config'].get('recipe') not in ('2c', '3c'):
        raise ValueError('Archived P07 supports full Small 2c/3c only')
    directory = Path(native_cache) if native_cache else Path(run) / 'evaluation/heldout_v3'
    for name in ('protocol.json', 'model_provenance.json'):
        if raw.producer().read_json(directory / name).get('numerical_profile') != raw.stock.definition():
            raise ValueError('P07 original512 stock profile differs')
    result.update(status='archived_metadata_only_full_admission_pending', checkpoint=context,
        original512_directory=str(directory.resolve()), namespace=str(Path(context['run']) / 'evaluation' / NAMESPACE / condition / checkpoint))
    return result


def verify_release():
    value = shared.read(READY)
    if (value.get('owner') != OWNER or value.get('status') != 'cpu_tests_passed_actual_gpu_not_run' or
            value.get('source_sha256') != source_files() or value.get('execution_profile') != execution_profile()):
        raise ValueError('Exact archived P07 consumer CPU release required')
    return {'path': str(READY), 'sha256': shared.sha(READY)}
