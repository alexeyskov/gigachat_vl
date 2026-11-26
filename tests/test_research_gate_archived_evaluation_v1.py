"""Archive-consumer CPU wiring; shared loader owns real compressed fixtures."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.eval import research_gate_archived_evaluation_v1 as facade


def test_actual_default_metadata_cli_no_torch_weight_body_or_generation():
    command = [sys.executable, '-c', "import sys,runpy;sys.argv=['gate-archive'];"
        "runpy.run_path('scripts/eval_research_gate_archived_checkpoint_v1.py',run_name='__main__');assert 'torch' not in sys.modules"]
    result = subprocess.run(command, cwd=facade.ROOT, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['owner'] == facade.OWNER and plan['Torch_imported'] is False and plan['GPU_used'] is False
    assert plan['execution_profile']['archived_loader']['CPU_payload_decodes_per_load'] == 1
    assert plan['native_reference_full512_verified'] is False and plan['P07_complete'] is False


def test_private_factory_leaves_raw_controls_and_source_release_unchanged():
    original = facade.raw.producer(); before = facade.raw.OWNER, facade.raw.NAMESPACE, facade.raw.sha(facade.raw.READY)
    private = facade.producer()
    assert private.OWNER == facade.OWNER and private.NAMESPACE == facade.NAMESPACE
    assert private.controls is not original.controls and private._stock_native.value is None
    assert before == (facade.raw.OWNER, facade.raw.NAMESPACE, facade.raw.sha(facade.raw.READY))
    assert facade.raw.execution_profile()['archived_checkpoints_supported'] is False
    assert facade.execution_profile()['archived_checkpoints_supported'] is True


def test_private_controls_replace_only_checkpoint_eligibility_with_shared_context(monkeypatch):
    calls = []; checkpoint = {'config': {'model': facade.raw.stock.SMALL, 'recipe': '3c'}, 'synthetic_shared_context': True}
    def context(run, role): calls.append((run, role)); return checkpoint
    monkeypatch.setattr(facade.shared, 'checkpoint_context', context)
    private = facade.producer()
    value = private.controls.prepare_context('synthetic', condition='all_open', checkpoint='last')
    assert calls == [('synthetic', 'last')] and value['static']['checkpoint'] is checkpoint
    assert set(value['records'][0]) == private.controls.PUBLIC_FIELDS
    assert len(value['records']) == 512


def test_private_producer_no_longer_enters_removed_raw_only_marker_loop(monkeypatch):
    private = facade.producer()
    monkeypatch.setattr(facade.raw.stock, 'validate_run_context', lambda _: {'stage': 'SFT', 'recipe': '3c'})
    base = {'static': {'checkpoint': {'run': '/synthetic/nonexistent'}, 'condition': 'all_open'}, 'records': []}
    monkeypatch.setattr(private.controls, 'prepare_context', lambda *a, **k: base)
    monkeypatch.setattr(private, 'validate_reference', lambda *a, **k: {'binding': {}, 'guard_indices': [], 'records': [], 'rows': []})
    value = private.prepare_context('/synthetic/nonexistent', condition='all_open')
    assert value['static']['stock_execution'] == facade.execution_profile()
    assert str(facade.shared.READY.relative_to(facade.ROOT)) in value['static']['stock_source_files']


def test_loader_receipt_is_separate_and_canonical_native_provenance_remains_exact(monkeypatch):
    from test_research_gate_stock_evaluation import tiny_model
    private = facade.producer(); model = tiny_model(); canonical = {'numerical_profile': facade.raw.stock.definition(), 'native': 'same source'}
    receipt = {'archived_evaluation': facade.shared.execution_profile(), 'archived_publication_sha256': 'a' * 64, 'archived_physical_sha256': 'b' * 64}
    private._stock_native.value = SimpleNamespace(load_research_checkpoint=lambda *a, **k: (model, {**canonical, **receipt}))
    monkeypatch.setattr(facade.raw.stock, 'activate', lambda: None); monkeypatch.setattr(facade.raw.stock, 'assert_active', lambda: None)
    engine = private.NativeRuntime(); restored, provenance = engine.load('/synthetic', '/cache', 120)
    assert restored is model and provenance == canonical
    assert engine.archive_receipt == receipt and not any(k in provenance for k in facade.ARCHIVE_FIELDS)
    parent = private.NativeRuntime.__mro__[1]
    monkeypatch.setattr(parent, 'describe', lambda self, model, provenance: {'loaded': provenance})
    assert engine.describe(model, provenance) == {'loaded': canonical, 'archived_checkpoint_decode': receipt}


def test_missing_archive_loader_receipt_refuses_despite_valid_stock_profile(monkeypatch):
    from test_research_gate_stock_evaluation import tiny_model
    private = facade.producer()
    private._stock_native.value = SimpleNamespace(load_research_checkpoint=lambda *a, **k: (tiny_model(), {'numerical_profile': facade.raw.stock.definition()}))
    monkeypatch.setattr(facade.raw.stock, 'activate', lambda: None); monkeypatch.setattr(facade.raw.stock, 'assert_active', lambda: None)
    with pytest.raises(ValueError, match='loader provenance'): private.NativeRuntime().load('/synthetic', '/cache', 120)


def protocol_fixture(private, monkeypatch):
    canonical = {'same': 'canonical native provenance'}
    context = {'static': {'base': {'condition': 'all_open', 'checkpoint': {'role': 'best', 'artifact': {'sha256': 'b' * 64},
        'files': {'checkpoint_archival.json': {'sha256': 'a' * 64}}}},
        'original_native512': {'native_model_provenance': canonical, 'eos_token_ids': [2]}}, 'base': {}}
    runtime = {'loaded': canonical, 'context_limit': 128, 'eos_token_ids': [2], 'stock_execution': facade.execution_profile(),
        'archived_checkpoint_decode': {'archived_evaluation': facade.shared.execution_profile(),
            'archived_publication_sha256': 'a' * 64, 'archived_physical_sha256': 'b' * 64}}
    monkeypatch.setattr(private.controls, 'control_protocol', lambda base, receipt, loaded: {'observed_gate_control': receipt})
    return context, runtime


def test_protocol_binds_actual_archive_and_keeps_exact_original_identity(monkeypatch):
    private = facade.producer(); context, runtime = protocol_fixture(private, monkeypatch)
    protocol = private.make_protocol(context, {}, runtime, synthetic=True)
    assert private.validate_protocol(context, protocol) == protocol and facade.NAMESPACE in protocol['namespace']
    wrong = copy.deepcopy(runtime); wrong['loaded']['same'] = 'wrong source'
    with pytest.raises(ValueError, match='differs from all512'): private.make_protocol(context, {}, wrong, synthetic=True)
    for field in ('archived_publication_sha256', 'archived_physical_sha256'):
        wrong = copy.deepcopy(runtime); wrong['archived_checkpoint_decode'][field] = 'c' * 64
        with pytest.raises(ValueError, match='decoder/reference'): private.make_protocol(context, {}, wrong, synthetic=True)


def test_metadata_dispatch_uses_shared_inspect_only_and_explicit_full_c_scope(tmp_path, monkeypatch):
    context = {'run': str(tmp_path), 'config': {'recipe': '3c'}, 'metadata_only': True}
    calls = []
    monkeypatch.setattr(facade.shared, 'inspect', lambda run, role: (calls.append((run, role)) or context))
    directory = tmp_path / 'evaluation/heldout_v3'; directory.mkdir(parents=True)
    for name in ('protocol.json', 'model_provenance.json'):
        (directory / name).write_text(json.dumps({'numerical_profile': facade.raw.stock.definition()}))
    value = facade.metadata_plan(tmp_path, checkpoint='last', condition='visual_closed')
    assert calls == [(tmp_path, 'last')] and value['checkpoint']['metadata_only'] is True
    assert facade.NAMESPACE in value['namespace'] and value['full_admission_executed'] is False
    context['config']['recipe'] = '4f'
    with pytest.raises(ValueError, match='2c/3c'): facade.metadata_plan(tmp_path)
    context['config']['recipe'] = '3c'; (directory / 'model_provenance.json').write_text('{}')
    with pytest.raises(ValueError, match='stock profile'): facade.metadata_plan(tmp_path)


def test_tiny_gate_endpoint_and_bits_still_match_sealed_raw_controls():
    import torch
    from test_research_gate_evaluation import tiny_block
    private = facade.producer(); original = facade.raw.producer(); block = tiny_block()
    mask = torch.tensor([[True, False]]); states = torch.arange(8).float().reshape(1, 2, 4)
    block.set_modality_mask(mask); before = block.modality_gate_logits.clone()
    with torch.no_grad(), original.controls.gate_control(block, 'visual_closed', recipe='2c') as a:
        expected = block(states, states)[0]
    with torch.no_grad(), private.controls.gate_control(block, 'visual_closed', recipe='2c') as b:
        actual = block(states, states)[0]
    assert a == b and torch.equal(actual, expected) and torch.equal(before, block.modality_gate_logits)
    assert 'sigmoid' not in block.modality_gate_logits.__dict__


def test_sealed_shared_release_change_refuses_before_any_model(monkeypatch):
    actual = facade.shared.sha
    monkeypatch.setattr(facade.shared, 'sha', lambda path: '0' * 64 if Path(path) == facade.shared.READY else actual(path))
    with pytest.raises(ValueError, match='decoder release changed'): facade.producer()
