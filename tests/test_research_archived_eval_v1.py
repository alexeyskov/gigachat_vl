"""Real tiny compressed CPU archives; synthetic owners/activation only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest
import torch

from src.utils import research_archived_eval_v1 as helper
from src.utils import research_checkpoint_archival as publisher
from src.utils import research_trained_delta_probe as probe
from src.utils.research_checkpoint_compat import load_research_payload
from test_research_trained_delta_probe import fixture
from test_research_checkpoint_archival import prepared


def write(path, value):
    Path(path).write_bytes(probe.canonical(value) + b'\n')


@pytest.fixture
def archived(prepared, monkeypatch):
    value = prepared; run = value['run']; config = value['payload']['config']
    config.update(numerical_profile=helper.stock.definition(), vision='fixture/frozen_vision')
    # A finite scientific-like tiny model; metadata still includes negative
    # zero, aliased/viewed normal buffers and all original RNG/scheduler fields.
    value['payload']['model'][value['key']].nan_to_num_(nan=0.)
    torch.save(value['payload'], value['source'])
    with zipfile.ZipFile(value['source'], 'a') as archive:
        prefix = archive.namelist()[0].split('/')[0]
        archive.writestr(prefix + '/synthetic_test_padding', b'\0' * (2 * 2**20))
    write(run / 'resolved_config.json', config)
    helper.stock.execution_sources(run, create=True, reserve_gb=0)
    evidence = publisher.activation_evidence()
    cpu, gpu, activation = run / 'fixture_cpu.json', run / 'fixture_gpu.json', run / 'synthetic_activation.json'
    write(cpu, {'synthetic_CPU_only': True}); write(gpu, {'synthetic_CPU_only': True})
    write(activation, {'trained_probe_report': str(cpu), 'root_gpu_parity_report': str(gpu)})
    monkeypatch.setattr(helper.archive, 'ACTIVATION', activation)
    monkeypatch.setattr(helper, '_publisher', lambda: publisher)
    monkeypatch.setattr(helper, '_activation', lambda p: evidence)
    # Only the proposed constructor scratch allowance is scaled for tiny CPU
    # tests. Frozen stream/v1 admission keeps its actual32GiB headroom floor.
    monkeypatch.setattr(helper, 'CONSTRUCTOR_ALLOWANCE', 64 * 2**10)
    expected = probe.payload_signature(value['payload'])
    report = publisher.publish_completed_checkpoint(run, snapshot=value['snapshot'], max_staging_bytes=192 * 2**20)
    return dict(value, report=report, expected_signature=expected)


def test_default_import_and_profile_are_no_torch_no_checkpoint_reads():
    command = [sys.executable, '-c', 'import sys,json; from src.utils import research_archived_eval_v1 as a; '
        "p=a.execution_profile();assert 'torch' not in sys.modules;print(json.dumps(p))"]
    result = subprocess.run(command, cwd=helper.ROOT, check=True, capture_output=True, text=True)
    profile = json.loads(result.stdout)
    assert profile['CPU_payload_decodes_per_load'] == 1 and profile['headroom_bytes'] == 32 * 2**30
    assert profile['fallback'] is False


def test_metadata_and_full_context_do_not_decode_then_real_single_payload_roundtrip(archived, monkeypatch):
    source = archived['run'] / 'best/state.pt'; original = Path.open
    def no_body(path, *args, **kwargs):
        if path.name == 'state.pt': raise AssertionError('Metadata opened archive tensor body')
        return original(path, *args, **kwargs)
    with monkeypatch.context() as local:
        local.setattr(Path, 'open', no_body)
        metadata = helper.inspect(archived['run'])
    assert metadata['metadata_only'] is True and metadata['artifact']['components_fully_hashed'] is False
    assert metadata['archive_admission']['physical_archives_verified'] is False
    context = helper.checkpoint_context(archived['run'])
    assert context['metadata_only'] is False and context['artifact']['components_fully_hashed'] is True
    assert context['artifact']['sha256'] == probe.file_sha(source)
    helper.assert_context(context)
    with helper._shared(archived['run']):
        restored = helper.load_payload(source, context['archive_admission'])
    assert probe.payload_signature(restored) == archived['expected_signature']
    assert restored['normal_same_object'] is restored['normal_backing']
    assert restored['normal_view'].storage_offset() == 3
    assert restored['config']['numerical_profile'] == helper.stock.definition()
    assert set(restored) == set(archived['payload'])


@pytest.mark.parametrize('change', ('budget', 'header', 'transaction', 'signature', 'context'))
def test_tampered_admission_projection_cannot_bypass_declared_guard(archived, change):
    context = helper.checkpoint_context(archived['run']); saved = copy.deepcopy(context)
    plan = saved['archive_admission']; selected = plan['transactions']['best']
    if change == 'budget': selected['load_extra_bytes'] = 0
    elif change == 'header': selected['reader_header']['decoded_model_bytes'] += 4
    elif change == 'transaction': selected['transaction']['source_state_sha256'] = '0' * 64
    elif change == 'signature': selected['transaction']['derived']['payload_sha256'] = '0' * 64
    else: saved['artifact']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='projection'):
        helper.assert_context(saved)


@pytest.mark.parametrize('change', ('physical_bytes', 'source_marker', 'stock_profile', 'role', 'anchor'))
def test_wrong_source_anchor_profile_marker_or_role_refuses(archived, change):
    context = helper.checkpoint_context(archived['run']); run = archived['run']
    if change == 'physical_bytes':
        path = run / 'best/state.pt'
        with path.open('r+b') as handle: handle.seek(-1, 2); byte = handle.read(1); handle.seek(-1, 2); handle.write(bytes([byte[0] ^ 1]))
    elif change == 'source_marker':
        path = run / 'best/.research_checkpoint.json'; marker = json.loads(path.read_text()); marker['metric'] += 1; write(path, marker)
    elif change == 'stock_profile':
        path = run / 'resolved_config.json'; config = json.loads(path.read_text()); config['numerical_profile']['warn_only'] = True; write(path, config)
    elif change == 'role':
        (run / 'best').unlink(); (run / 'best').symlink_to('.checkpoint_objects/foreign')
    else:
        path = archived['snapshot'] / 'model.safetensors'
        with path.open('r+b') as handle: handle.seek(-1, 2); byte = handle.read(1); handle.seek(-1, 2); handle.write(bytes([byte[0] ^ 1]))
    if change == 'anchor':
        # Metadata size may remain unchanged; canonical decoder SHA catches it.
        with pytest.raises(ValueError, match='SHA|digest|changed|checksum'):
            with helper._shared(run): helper.load_payload(run / 'best/state.pt', context['archive_admission'])
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            helper.assert_context(context)


def test_frozen_compatibility_loader_called_once_before_native_constructor(archived, monkeypatch):
    from torch import nn
    import huggingface_hub
    import src.utils.research_checkpoint_compat as compatibility
    context = helper.checkpoint_context(archived['run']); events = []; original_load = compatibility.load_research_payload
    def counted(*a, **k): events.append('decode'); return original_load(*a, **k)
    monkeypatch.setattr(compatibility, 'load_research_payload', counted)
    monkeypatch.setattr(helper.stock, 'activate', lambda: events.append('activate'))
    monkeypatch.setattr(helper.stock, 'assert_active', lambda: None)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda *a, **k: str(archived['snapshot']))
    runner = ModuleType('_synthetic_native_runner')
    runner.resolve_model = lambda repo: (_ for _ in ()).throw(AssertionError('Native pinned resolver must replace this'))
    class Tiny(nn.Module):
        def __init__(self, args):
            assert events == ['activate', 'decode']; events.append('construct')
            super().__init__(); self.args = args; self.llm = nn.Module(); self.llm.config = SimpleNamespace(use_cache=False)
            self.llm.model = nn.Module(); self.llm.model.layers = nn.ModuleList([nn.Module()])
            self.llm.model.layers[0].self_attn = nn.Module(); self.llm.model.layers[0].self_attn.q_proj = nn.Linear(6, 4, bias=False)
            self.projector = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 4))
            self.cache = SimpleNamespace(fingerprint=archived['payload']['cache_fingerprint'])
            assert runner.resolve_model(args.model)[1] == archived['payload']['model_ref']
        @property
        def base(self): return self.llm
    runner.ResearchModel = Tiny; monkeypatch.setattr(helper.stock, 'runner', lambda: runner)
    native = helper.inference(context)
    model, provenance = native.load_research_checkpoint(context['checkpoint_path'], reserve_gb=120)
    assert events == ['activate', 'decode', 'construct']
    assert all(torch.equal(model.state_dict()[name], value) for name, value in archived['payload']['model'].items())
    assert all(not value.requires_grad for value in model.parameters()) and not model.training
    assert provenance['numerical_profile'] == helper.stock.definition()
    assert provenance['archived_evaluation']['CPU_payload_decodes_per_load'] == 1
    assert provenance['archived_physical_sha256'] == context['artifact']['sha256']


def test_bad_decoded_complete_payload_or_profile_refuses_before_constructor(archived, monkeypatch):
    import src.utils.research_checkpoint_compat as compatibility
    context = helper.checkpoint_context(archived['run']); original = compatibility.load_research_payload
    def changed(*a, **k):
        value = original(*a, **k); value['scheduler']['_step_count'] += 1; return value
    monkeypatch.setattr(compatibility, 'load_research_payload', changed)
    with helper._shared(archived['run']), pytest.raises(ValueError, match='complete payload'):
        helper.load_payload(context['checkpoint_path'], context['archive_admission'])


def test_metadata_boolean_cannot_substitute_for_fresh_physical_hash(archived, monkeypatch):
    import src.utils.research_checkpoint_compat as compatibility
    run = archived['run']; path = run / 'best/state.pt'
    with path.open('r+b') as handle:
        handle.seek(-1, 2); value = handle.read(1); handle.seek(-1, 2); handle.write(bytes([value[0] ^ 1]))
    context = helper.inspect(run); plan = context['archive_admission']; plan['physical_archives_verified'] = True
    monkeypatch.setattr(compatibility, 'load_research_payload', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Decode must not begin')))
    with pytest.raises(ValueError, match='physical bytes'):
        helper.load_payload(context['checkpoint_path'], plan)


def test_actual_activation_pin_mutation_refuses_before_evidence_validation(monkeypatch, tmp_path):
    value = tmp_path / 'activation.json'; value.write_text('{}')
    monkeypatch.setattr(helper.archive, 'ACTIVATION', value)
    with pytest.raises(ValueError, match='ROOT-approved'):
        helper._activation(SimpleNamespace(activation_evidence=lambda: (_ for _ in ()).throw(AssertionError('Shouldnotrun'))))
