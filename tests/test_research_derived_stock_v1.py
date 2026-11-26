"""Composition/resume contracts and actual tiny native model restore."""
import copy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType

import pytest

from src.utils import research_derived_stock_v1 as composed
from src.utils import research_derived_vision_v1 as derived
from src.utils import research_stock_determinism as stock
from tests.test_research_derived_vision_v1 import export, resolve, args, fake_runner


def test_actual_metadata_has_no_torch_network_original_weights():
    code = """
import sys,socket
from pathlib import Path
old=Path.open
def opened(p,*a,**k):
    if 'models--Qwen' in str(p) or 'models--google--gemma' in str(p):
        raise AssertionError('Original donor must never be opened')
    return old(p,*a,**k)
Path.open=opened
socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(AssertionError('network'))
from src.utils.research_derived_stock_v1 import training_main
training_main(['--numerical-profile','stock_deterministic_small_v1','--vision','google/gemma-4-E4B-it'])
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-c', code], cwd=composed.ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value['status'] == 'metadata_only_no_training'
    assert value['original_donor_files_opened'] is False


@pytest.mark.parametrize('change', ['stock', 'composition', 'export', 'payload'])
def test_no_legacy_or_changed_config_payload_relabelling(export, tmp_path, change):
    binding = resolve(export)
    config = composed.configure(args(binding, tmp_path), binding)
    payload = derived.bind_checkpoint_payload({'config': copy.deepcopy(vars(config)),
        'model_ref': {'repo': stock.SMALL, 'revision': 'd'*40},
        'vision_ref': binding['upstream_ref'], 'cache_fingerprint': binding['cache_fingerprint'],
        composed.FIELD: composed.definition()}, binding)
    composed.validate_payload(payload, binding)
    if change == 'stock':
        payload['config'].pop('numerical_profile')
    elif change == 'composition':
        payload['config'][composed.FIELD]['owner'] = 'legacy'
    elif change == 'export':
        payload['config']['vision_export_ref'] = {}
    else:
        payload.pop(composed.FIELD)
    with pytest.raises(ValueError):
        composed.validate_payload(payload, binding)


def test_private_stock_and_derived_manager_restore_guards(export, tmp_path, monkeypatch):
    binding = resolve(export)
    config = args(binding, tmp_path)
    private = fake_runner(binding)
    monkeypatch.setattr(stock, 'runner', lambda: private)
    monkeypatch.setattr(derived, 'host_guard', lambda: 100*2**30)
    model_runner = composed.runner(config, binding)
    model = model_runner.ResearchModel(config)
    payload = model_runner.BestLastCheckpointManager().save_torch({
        'config': copy.deepcopy(vars(config)), 'model_ref': {'repo': stock.SMALL, 'revision': 'd'*40},
        'vision_ref': model.vision_ref, 'cache_fingerprint': model.cache.fingerprint})
    composed.validate_payload(payload, binding)
    assert model.cache.stats['requests'] == 0
    model_runner.validate_model_binding(payload, model)
    with pytest.raises(ValueError, match='Existing composed payload'):
        model_runner.BestLastCheckpointManager().save_torch(dict(payload, **{composed.FIELD: {}}))
    payload[composed.FIELD] = {}
    with pytest.raises(ValueError):
        model_runner.validate_model_binding(payload, model)


def test_actual_tiny_native_checkpoint_restores_model_cache_and_generation(export, tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from tests.test_research_inference import TinyResearchModel, record
    from src.eval.research_inference import freeze_for_generation, generate_prediction
    from src.utils.research_storage import BestLastCheckpointManager

    binding = resolve(export)
    config = args(binding, tmp_path)
    config.stage = 'SFT'; config.recipe = '3c'; config.max_length = 64
    composed.configure(config, binding)
    ref = {'repo': stock.SMALL, 'revision': 'd'*40}
    monkeypatch.setattr(derived, '_local_language_resolver', lambda *a, **k: (str(tmp_path), ref))
    monkeypatch.setattr(derived, 'host_guard', lambda: 100*2**30)
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('network'))
    def tiny_runner():
        module = ModuleType('tiny_composed_native_runner')
        class Model(TinyResearchModel):
            def __init__(self, value):
                stock.assert_active()
                super().__init__(wrapped=True); self.args = value
                self.model_ref = module.resolve_model(value.model)[1]
                _, self.vision_ref = module.resolve_model(value.vision)
                self.extractor = SimpleNamespace(backend='qwen2_5_vl', feature_dim=3)
                self.cache = module.VisionFeatureCache(value.cache_dir, 'never_legacy')
                self.encoder_calls = 0
            def encode(self, paths):
                self.encoder_calls += 1
                return [torch.tensor([[1.,2.,3.],[4.,5.,6.]]) for _ in paths]
        module.ResearchModel = Model
        module.BestLastCheckpointManager = BestLastCheckpointManager
        module.validate_model_binding = lambda *a: None
        return module
    monkeypatch.setattr(stock, 'runner', tiny_runner)
    stock.activate(); torch.manual_seed(42)
    private = composed.runner(config, binding, model_ref=ref)
    source = private.ResearchModel(config)
    saved = {'config': copy.deepcopy(vars(config)), 'model_ref': ref,
        'vision_ref': source.vision_ref, 'cache_fingerprint': source.cache.fingerprint,
        composed.FIELD: composed.definition(), 'step': 1, 'model': source.state_dict()}
    path = tmp_path/'state.pt'
    torch.save(derived.bind_checkpoint_payload(saved, binding), path)
    first = generate_prediction(freeze_for_generation(source), record(), max_new_tokens=4)
    restored, provenance = composed.load_checkpoint(path, export_root=export[1], cache_dir=config.cache_dir)
    second = generate_prediction(restored, record(), max_new_tokens=4)
    assert first == second and source.encoder_calls == 1 and restored.encoder_calls == 0
    assert provenance[composed.FIELD] == composed.definition()
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name])
    stock.assert_active()
    assert not torch.cuda.is_initialized()


def test_loader_rejects_archive_before_payload_or_torch_construction(tmp_path, monkeypatch):
    monkeypatch.setattr(derived, 'host_guard', lambda: 100*2**30)
    path = tmp_path/'archive.pt'; path.write_bytes(b'not-a-native-ZIP')
    monkeypatch.setattr(stock, 'activate', lambda: pytest.fail('Archive must refuse before Torch activation'))
    with pytest.raises(ValueError, match='native ZIP'):
        composed.load_checkpoint(path)


def test_cli_requires_explicit_profile_and_small_only():
    with pytest.raises(SystemExit):
        composed.training_main(['--vision','google/gemma-4-E4B-it'])
    with pytest.raises(ValueError, match='SmallThinker'):
        composed.training_main(['--numerical-profile',stock.PROFILE,'--model','other/model',
                                '--vision','google/gemma-4-E4B-it'])


def test_actual_private_evaluator_imports_complete_native_interface_and_binds_protocol():
    from src.eval import research_inference as native
    original = native.load_research_checkpoint
    module = composed.evaluator()
    assert module.GENERATION_SCHEMA_VERSION == native.GENERATION_SCHEMA_VERSION
    assert module.sha256_file is native.sha256_file
    assert module.generate_prediction is native.generate_prediction
    assert module.load_research_checkpoint is composed.load_checkpoint
    assert native.load_research_checkpoint is original
    protocol = module._bind_stock_protocol({'fixture': 'unchanged'})
    assert protocol['fixture'] == 'unchanged'
    assert protocol['numerical_profile'] == stock.definition()
    assert protocol[composed.FIELD] == composed.definition()


def test_custom_export_root_reaches_native_evaluation_loader(tmp_path, monkeypatch):
    seen = []
    def loader(checkpoint, **kwargs):
        seen.append((checkpoint, kwargs))
        return 'tiny-loaded', {}
    monkeypatch.setattr(composed, 'load_checkpoint', loader)
    module = composed.evaluator(export_root=tmp_path)
    assert module.load_research_checkpoint('checkpoint.pt', reserve_gb=120) == ('tiny-loaded', {})
    assert seen == [('checkpoint.pt', {'reserve_gb': 120, 'export_root': tmp_path})]
