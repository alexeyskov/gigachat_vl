"""CPU wiring/metadata fixtures; no scientific gate predictions or GPU work."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.eval import research_gate_stock_evaluation as facade


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def run(tmp_path):
    run = tmp_path / 'synthetic_metadata'; run.mkdir()
    config = {'model': facade.stock.SMALL, 'vision': 'synthetic/vision', 'recipe': '3c', 'stage': 'SFT',
        'output': str(run), 'numerical_profile': facade.stock.definition()}
    write(run / 'resolved_config.json', config)
    write(run / '.checkpoint_manager.json', {'owner': 'synthetic-owner'})
    write(run / 'completed.json', {'steps': 2})
    write(run / 'checkpoint_compaction.json', {'status': 'completed', 'training_completed': True,
        'resumable_optimizer': False, 'run_dir': str(run), 'checkpoints': [{'new_object': 'object-tiny'}]})
    target = run / '.checkpoint_objects/object-tiny'; target.mkdir(parents=True)
    (target / 'state.pt').write_bytes(b'SYNTHETIC_METADATA_ONLY_NO_TENSOR_BODY_READ')
    write(target / '.research_checkpoint.json', {'owner': 'synthetic-owner', 'step': 2,
        'training_completed': True, 'optimizer_state_removed': True, 'resumable_optimizer': False})
    for role in ('best', 'last'): (run / role).symlink_to(Path('.checkpoint_objects/object-tiny'))
    for name in ('protocol.json', 'model_provenance.json'):
        write(run / 'evaluation/heldout_v3' / name, {'numerical_profile': facade.stock.definition()})
    facade.stock.execution_sources(run, create=True, reserve_gb=0)
    return run


def test_default_actual_cli_metadata_imports_no_torch_or_model_weights():
    source = ("import runpy,sys;sys.argv=['gate-stock'];"
        "runpy.run_path('scripts/eval_research_gate_stock_checkpoint.py',run_name='__main__');"
        "assert 'torch' not in sys.modules")
    value = subprocess.run([sys.executable, '-c', source], cwd=facade.ROOT, capture_output=True, text=True, check=True)
    data = json.loads(value.stdout)
    assert data['Torch_imported'] is False and data['GPU_used'] is False
    assert data['checkpoint_tensor_bodies_opened'] is False and data['full_admission_executed'] is False
    assert data['P07_complete'] is False and data['records'] == 512


def test_private_clones_keep_canonical_owner_controls_and_native_unloaded():
    from src.eval import research_gate_controls as controls
    from src.eval import research_gate_evaluation as owner
    old = (owner.NativeRuntime, controls.NativeGateRuntime, owner.OWNER, controls.OWNER)
    private = facade.producer()
    assert private is not owner and private.controls is not controls
    assert private._stock_native is private.controls._stock_native
    assert private._stock_native.value is None
    assert (owner.NativeRuntime, controls.NativeGateRuntime, owner.OWNER, controls.OWNER) == old
    assert private.NAMESPACE == facade.NAMESPACE and private.OWNER == facade.OWNER


def test_metadata_reads_no_payload_body_and_does_not_claim_full_admission(run, monkeypatch):
    original = Path.open
    def checked(path, *args, **kwargs):
        if path.name == 'state.pt': raise AssertionError('Metadata opened checkpoint tensor bytes')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', checked)
    plan = facade.metadata_plan(run, condition='all_open')
    assert plan['status'] == 'metadata_only_full_admission_pending'
    assert plan['roles']['best']['checkpoint_stat']['st_size'] == 43
    assert plan['full_admission_executed'] is False
    assert facade.NAMESPACE in plan['namespace']


@pytest.mark.parametrize('change', ('legacy', 'changed_profile', 'wrong_model', 'wrong_stage', 'wrong_recipe'))
def test_metadata_rejects_legacy_mixed_or_unsupported_stage(run, change):
    path = run / 'resolved_config.json'; config = json.loads(path.read_text())
    if change == 'legacy': config.pop('numerical_profile')
    elif change == 'changed_profile': config['numerical_profile']['warn_only'] = True
    elif change == 'wrong_model': config['model'] = 'unsupported/model'
    elif change == 'wrong_stage': config['stage'] = 'A1'
    else: config['recipe'] = '3a'
    write(path, config)
    with pytest.raises(ValueError): facade.metadata_plan(run)


@pytest.mark.parametrize('change', ('reference_legacy', 'reference_mixed', 'wrong_owner', 'nonterminal_last', 'bad_manifest'))
def test_metadata_rejects_wrong_role_reference_and_manifest(run, tmp_path, change):
    kwargs = {}
    if change.startswith('reference'):
        path = run / 'evaluation/heldout_v3/protocol.json'; data = json.loads(path.read_text())
        if change == 'reference_legacy': data.pop('numerical_profile')
        else: data['numerical_profile']['warn_only'] = True
        write(path, data)
    elif change == 'bad_manifest':
        manifest = tmp_path / 'changed.jsonl'; manifest.write_text('{}\n'); kwargs['manifest'] = manifest
    else:
        path = run / '.checkpoint_objects/object-tiny/.research_checkpoint.json'; data = json.loads(path.read_text())
        data['owner' if change == 'wrong_owner' else 'step'] = 'foreign' if change == 'wrong_owner' else 1
        write(path, data)
    with pytest.raises(ValueError): facade.metadata_plan(run, **kwargs)


def tiny_model(recipe='3c', dtype=None):
    import torch
    from torch import nn
    model = nn.Module(); model.llm = nn.Linear(2, 2, bias=False, dtype=dtype or torch.float32)
    model.projector = nn.Linear(2, 2, dtype=torch.float32)
    model.args = SimpleNamespace(model=facade.stock.SMALL, recipe=recipe, stage='SFT', numerical_profile=facade.stock.definition())
    return model.eval().requires_grad_(False)


def test_runtime_activates_before_private_native_constructor_and_requires_payload_profile(monkeypatch):
    private = facade.producer(); events = []; model = tiny_model()
    monkeypatch.setattr(facade.stock, 'activate', lambda: events.append('activate'))
    monkeypatch.setattr(facade.stock, 'assert_active', lambda: events.append('assert'))
    def load(*args, **kwargs):
        assert events == ['activate']; events.append('construct')
        return model, {'numerical_profile': facade.stock.definition()}
    private._stock_native.value = SimpleNamespace(load_research_checkpoint=load)
    restored, loaded = private.NativeRuntime().load('synthetic', 'unused', 120)
    assert restored is model and events == ['activate', 'construct', 'assert']
    private._stock_native.value.load_research_checkpoint = lambda *a, **k: (model, {})
    events.clear()
    with pytest.raises(ValueError, match='lacks exact stock'): private.NativeRuntime().load('synthetic', 'unused', 120)


def test_full_master_dtype_guard_preserves_parameters_and_allows_native_quantized_recipe():
    import torch
    model = tiny_model(); before = {n: id(p) for n, p in model.named_parameters()}
    facade._validate_loaded(model, {'numerical_profile': facade.stock.definition()})
    assert before == {n: id(p) for n, p in model.named_parameters()}
    with pytest.raises(ValueError, match='FP32 masters'):
        facade._validate_loaded(tiny_model(dtype=torch.bfloat16), {'numerical_profile': facade.stock.definition()})
    facade._validate_loaded(tiny_model(recipe='1c', dtype=torch.bfloat16), {'numerical_profile': facade.stock.definition()})
    with pytest.raises(ValueError, match='1c/2c/3c SFT'):
        facade._validate_loaded(tiny_model(recipe='3a'), {'numerical_profile': facade.stock.definition()})


def test_metadata_refuses_compaction_receipt_drift_without_payload_reads(run):
    path = run / 'checkpoint_compaction.json'; value = json.loads(path.read_text()); value['status'] = 'failed'
    write(path, value)
    with pytest.raises(ValueError, match='compaction metadata'): facade.metadata_plan(run)


def test_archived_checkpoint_metadata_explicitly_refuses_separate_decoder_profile(run):
    path = run / '.checkpoint_objects/object-tiny/.research_checkpoint.json'
    marker = json.loads(path.read_text()); marker['checkpoint_archival'] = {'synthetic': 'lineage'}; write(path, marker)
    with pytest.raises(ValueError, match='separate reviewed lineage/decoder-memory adapter'):
        facade.metadata_plan(run)


def test_private_generation_observer_projects_gold_and_restores_native_builder(monkeypatch):
    private = facade.producer(); monkeypatch.setattr(facade.stock, 'assert_active', lambda: None)
    public = {'record_id': 'synthetic', 'question': 'Describe the diagram.', 'image_paths': ['/synthetic.png'], 'image_hashes': ['a' * 64]}
    native = SimpleNamespace(); model = object(); calls = []
    def build(actual, record, **kwargs):
        assert actual is model and record == public and set(record) == private.controls.PUBLIC_FIELDS
        calls.append(record)
        return {'prefix_token_ids': [1], 'suffix_token_ids': [3], 'prompt_sha256': 'b' * 64, 'visual_tokens': 2, 'prefill_tokens': 4}
    native.build_generation_inputs = build
    def generate(actual, record, **kwargs):
        prepared = native.build_generation_inputs(actual, record, **kwargs)
        return {'prediction': 'yes', 'generated_token_ids': [4], 'generated_tokens': 1, 'logits_to_keep': 1,
            **{k: prepared[k] for k in ('prompt_sha256', 'visual_tokens', 'prefill_tokens')}}
    native.generate_prediction = generate; private._stock_native.value = native
    row = dict(public, answers=['SECRET'], scorer='exact_match')
    result = private.NativeRuntime().generate(model, row)
    assert len(calls) == 1 and native.build_generation_inputs is build
    assert result['prefix_token_ids'] == [1] and result['generated_token_ids'] == [4]
    assert 'answers' not in result and 'scorer' not in result
    assert private._stock_native.value is native


def test_runtime_profile_and_new_namespace_bind_protocol_reconstruction_and_resume(monkeypatch):
    private = facade.producer(); loaded = {'native': 'synthetic'}
    context = {'static': {'base': {'condition': 'all_open', 'checkpoint': {'role': 'best'}},
        'original_native512': {'native_model_provenance': loaded, 'eos_token_ids': [2]}, 'stock_execution': facade.execution_profile()}}
    context['base'] = {'synthetic': 'unchanged context'}
    monkeypatch.setattr(private.controls, 'control_protocol', lambda base, receipt, loaded: {'observed_gate_control': receipt})
    runtime = {'loaded': loaded, 'eos_token_ids': [2], 'context_limit': 128, 'stock_execution': facade.execution_profile()}
    protocol = private.make_protocol(context, {}, runtime, synthetic=True)
    assert facade.NAMESPACE in protocol['namespace'] and protocol['stock_execution'] == facade.execution_profile()
    assert private.validate_protocol(context, protocol) == protocol
    for key in ('numerical_definition', 'owner'):
        changed = copy.deepcopy(protocol); changed['runtime']['stock_execution'].pop(key)
        with pytest.raises(ValueError, match='stock profile'): private.validate_protocol(context, changed)
    assert protocol['legacy_cache_compatible'] is False


def test_tiny_real_gate_controls_and_frozen_scores_are_identical():
    import torch
    from src.eval import research_gate_controls as original
    from src.eval.research_metrics import score_prediction
    from src.model.smallthinker import SmallThinkerConfig, SmallThinkerMoeBlock
    from src.model.research_moe import ResearchSmallThinkerMoE
    private = facade.producer(); torch.manual_seed(42)
    config = SmallThinkerConfig(vocab_size=8, hidden_size=4, num_hidden_layers=1, num_attention_heads=1,
        num_key_value_heads=1, head_dim=4, moe_ffn_hidden_size=4, moe_num_primary_experts=2,
        moe_num_active_primary_experts=1, max_position_embeddings=128)
    block = ResearchSmallThinkerMoE(SmallThinkerMoeBlock(config), 3, 'c', added_experts=2).eval().requires_grad_(False)
    block.set_modality_mask(torch.tensor([[True, False]])); states = torch.randn(1, 2, 4)
    bits = block.modality_gate_logits.detach().clone()
    with torch.no_grad(), original.gate_control(block, 'all_open', recipe='3c') as first: expected = block(states, states)[0]
    with torch.no_grad(), private.controls.gate_control(block, 'all_open', recipe='3c') as second: actual = block(states, states)[0]
    assert torch.equal(actual, expected) and first == second
    assert torch.equal(bits, block.modality_gate_logits) and 'sigmoid' not in block.modality_gate_logits.__dict__
    rows = [({'scorer': 'exact_match', 'answers': ['yes']}, 'yes'),
        ({'scorer': 'chart_relaxed', 'answers': ['100']}, '104'), ({'scorer': 'anls', 'answers': ['hello']}, 'hellp')]
    assert [score_prediction(r, p) for r, p in rows] == [1., 1., .8]


def test_parent_source_and_ready_mismatch_refuse_before_private_load(monkeypatch):
    actual = facade.sha
    monkeypatch.setattr(facade, 'sha', lambda path: '0' * 64 if str(path).endswith(facade.PARENT_READY) else actual(path))
    with pytest.raises(ValueError, match='release changed'): facade.producer()
