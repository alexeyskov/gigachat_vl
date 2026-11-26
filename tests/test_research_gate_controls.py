"""Actual tiny native Small dispatch; no mock successes for gate mathematics."""
import json
from types import SimpleNamespace

import pytest
import torch

from src.eval import research_gate_controls as controls
from src.model.research_moe import ResearchSmallThinkerMoE, install_research_moe, set_modality_mask
from src.model.smallthinker import SmallThinkerConfig, SmallThinkerForCausalLM, SmallThinkerMoeBlock


def config(layers=1):
    return SmallThinkerConfig(vocab_size=32, hidden_size=8, num_hidden_layers=layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, moe_ffn_hidden_size=12,
        moe_num_primary_experts=4, moe_num_active_primary_experts=2,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, max_position_embeddings=128)


def block(family=2):
    torch.manual_seed(5)
    result = ResearchSmallThinkerMoE(SmallThinkerMoeBlock(config()), family, 'c', added_experts=2)
    result.modality_gate_logits.data.copy_(torch.linspace(-2, 2, 12).reshape(6, 2))
    return result.requires_grad_(False).eval()


def model(family=2):
    torch.manual_seed(5)
    result = SmallThinkerForCausalLM(config(2))
    install_research_moe(result, family, 'c')
    return result.requires_grad_(False).eval()


@pytest.mark.parametrize('family', (1, 2, 3))
@pytest.mark.parametrize('condition', controls.CONDITIONS)
def test_native_dispatch_exact_gate_conditions_and_raw_parameter_identity(family, condition):
    module = block(family)
    visual = torch.tensor([[True, False, True, False]])
    module.set_modality_mask(visual)
    x = torch.randn(1, 4, 8)
    parameter = module.modality_gate_logits
    original = controls.tensor_identity(parameter)
    pointer, version = parameter.data_ptr(), parameter._version
    original_state = {name: controls.tensor_identity(value) for name, value in module.state_dict().items()}
    with torch.no_grad():
        baseline = module(x, x)[0]
        expected_gates = controls.applied_gates(parameter, condition)
        # Independent branch formula with actual routed indices and frozen FFNs.
        indices, weights = module.last_indices.clone(), module.last_weights.clone()
        expected = torch.zeros_like(x).reshape(-1, 8)
        for token in range(4):
            for slot in range(indices.shape[1]):
                expert = int(indices[token, slot]); g = expected_gates[expert, int(visual[0, token])]
                expected[token] += weights[token, slot] * g * module._run_expert(expert, g * x[0, token])
        with controls.gate_control(module, condition, recipe=f'{family}c') as receipt:
            assert module.modality_gate_logits is parameter and parameter.data_ptr() == pointer
            assert torch.equal(parameter.sigmoid(), expected_gates)
            actual = module(x, x)[0]
            torch.testing.assert_close(actual, expected.reshape_as(x), rtol=2e-5, atol=1e-7)
            if condition == 'learned':
                assert 'sigmoid' not in parameter.__dict__
                assert torch.equal(actual, baseline)
            if condition in ('visual_closed', 'all_closed'):
                assert torch.count_nonzero(actual[visual]) == 0
            if condition in ('text_closed', 'all_closed'):
                assert torch.count_nonzero(actual[~visual]) == 0
            assert receipt['parameter_values_written'] is False
            assert receipt['gates'][0]['gate_logit_tensor'] == original
            assert receipt['gates'][0]['applied_values'] == expected_gates.tolist()
        assert 'sigmoid' not in parameter.__dict__
        assert parameter._version == version and parameter.data_ptr() == pointer
        assert controls.tensor_identity(parameter) == original
        assert torch.equal(module(x, x)[0], baseline)
    assert original_state == {name: controls.tensor_identity(value) for name, value in module.state_dict().items()}


def test_real_two_layer_logits_and_cached_generation_learned_identity():
    native = model(3)
    inputs = torch.tensor([[1, 8, 7, 6]])
    set_modality_mask(native, torch.tensor([[False, True, True, False]]))
    with torch.no_grad():
        baseline = native(inputs).logits
        expected = native.generate(inputs, max_new_tokens=3, do_sample=False, use_cache=True)
        set_modality_mask(native, torch.tensor([[False, True, True, False]]))
        with controls.gate_control(native, 'learned', recipe='3c'):
            assert torch.equal(native(inputs).logits, baseline)
            actual = native.generate(inputs, max_new_tokens=3, do_sample=False, use_cache=True)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize('condition', controls.CONDITIONS)
def test_inference_mode_endpoints_and_restoration(condition):
    module = block()
    original = controls.tensor_identity(module.modality_gate_logits)
    with torch.inference_mode():
        with controls.gate_control(module, condition):
            gates = module.modality_gate_logits.sigmoid()
            assert torch.equal(gates, controls.applied_gates(module.modality_gate_logits, condition))
    assert controls.tensor_identity(module.modality_gate_logits) == original


def test_actual_native_unwrapped_and_packed_subclass_refused():
    from src.model.research_moe import ResearchPackedMoE
    native = SmallThinkerForCausalLM(config()).eval().requires_grad_(False)
    with torch.no_grad(), pytest.raises(ValueError, match='unwrapped'):
        with controls.gate_control(native, 'learned'):
            pass
    subclass = block()
    subclass.__class__ = ResearchPackedMoE
    with torch.no_grad(), pytest.raises(ValueError, match='exact Small'):
        with controls.gate_control(subclass, 'all_open'):
            pass


def test_ideal_biasless_reglu_cubic_law_includes_outer_gate():
    module = block()
    expert = module.experts[0]
    x = torch.randn(5, 8)
    with torch.no_grad():
        for g in (0., 0.25, 0.8, 1.):
            torch.testing.assert_close(g * expert(g * x), (g ** 3) * expert(x), rtol=1e-5, atol=1e-8)


def test_scopes_do_not_nest_or_leak_after_exception():
    module = block()
    original = controls.tensor_identity(module.modality_gate_logits)
    with torch.no_grad():
        with pytest.raises(RuntimeError, match='intentional'):
            with controls.gate_control(module, 'all_open'):
                with pytest.raises(ValueError):
                    with controls.gate_control(module, 'visual_closed'):
                        pass
                with pytest.raises(ValueError):
                    with controls.gate_control(block(), 'learned'):
                        pass
                raise RuntimeError('intentional')
        with controls.gate_control(module, 'text_closed'):
            assert module.modality_gate_logits.sigmoid()[:, 0].eq(0).all()
    assert controls.tensor_identity(module.modality_gate_logits) == original
    assert 'sigmoid' not in module.modality_gate_logits.__dict__


@pytest.mark.parametrize('failure', ('train', 'requires_grad', 'nan', 'family', 'dispatch', 'foreign_sigmoid', 'warmup', 'unknown'))
def test_unsupported_or_corrupt_contract_refused(failure):
    module = block()
    condition, recipe = 'all_open', '2c'
    if failure == 'train': module.train()
    elif failure == 'requires_grad': module.modality_gate_logits.requires_grad_(True)
    elif failure == 'nan': module.modality_gate_logits.data[0, 0] = float('nan')
    elif failure == 'family': recipe = '1c'
    elif failure == 'dispatch': module._shared_output = lambda *_: None
    elif failure == 'foreign_sigmoid': module.modality_gate_logits.sigmoid = lambda: torch.ones(6, 2)
    elif failure == 'warmup': module.warmup_force_new = True
    elif failure == 'unknown': condition = 'random'
    with torch.no_grad(), pytest.raises(ValueError):
        with controls.gate_control(module, condition, recipe=recipe):
            pytest.fail('invalid scope entered')


def test_parameter_mutation_detected_without_leaking_override():
    module = block()
    with torch.no_grad(), pytest.raises(ValueError, match='changed inside scope'):
        with controls.gate_control(module, 'all_open'):
            module.modality_gate_logits.add_(1)
    assert 'sigmoid' not in module.modality_gate_logits.__dict__


def test_endpoint_return_is_private_clone_and_method_tampering_refused():
    module = block()
    with torch.no_grad():
        with controls.gate_control(module, 'all_open'):
            module.modality_gate_logits.sigmoid().zero_()
            assert module.modality_gate_logits.sigmoid().eq(1).all()
        with pytest.raises(ValueError, match='changed inside scope'):
            with controls.gate_control(module, 'all_open'):
                module.modality_gate_logits.sigmoid = lambda: torch.zeros(6, 2)
    assert 'sigmoid' not in module.modality_gate_logits.__dict__


def test_grad_enabled_refused_and_non_small_rejected():
    with pytest.raises(ValueError, match='no_grad'):
        with controls.gate_control(block(), 'learned'):
            pass
    with torch.no_grad(), pytest.raises(ValueError, match='no verified Small'):
        with controls.gate_control(torch.nn.Linear(3, 4).eval(), 'all_open'):
            pass


def test_source_drift_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(controls, 'ROOT', tmp_path)
    monkeypatch.setattr(controls, 'FROZEN_SOURCES', {'native.py': '0' * 64})
    (tmp_path / 'native.py').write_text('drift')
    with pytest.raises(ValueError, match='source differs'):
        controls.verify_frozen_sources()


def record():
    return {'record_id': 'heldout', 'question': 'What colour? Answer briefly.',
        'image_paths': ['/absolute/a.png'], 'image_hashes': ['a' * 64]}


def test_public_projection_never_reads_gold():
    class Trap(dict):
        def __getitem__(self, key):
            if key not in controls.PUBLIC_FIELDS: raise AssertionError('gold read')
            return super().__getitem__(key)
        def items(self): raise AssertionError('all items leaked')
    actual = record()
    assert controls.public_record(Trap(**actual, answers=['secret'], choices=['secret'])) == actual
    assert controls.public_record({**actual, 'answers': ['other']}) == actual


def test_native_runtime_observes_real_builder_without_gold(monkeypatch):
    from src.eval import research_inference as native
    called = []
    model = object()
    prepared = {'prefix_token_ids': [1, 2], 'suffix_token_ids': [3], 'prompt_sha256': 'b' * 64,
        'visual_tokens': 2, 'prefill_tokens': 5}
    def builder(actual_model, actual_record, **kwargs):
        assert actual_model is model and set(actual_record) == controls.PUBLIC_FIELDS
        assert kwargs == {'max_new_tokens': 64}
        called.append(actual_record)
        return prepared
    def generate(actual_model, actual_record, **kwargs):
        native.build_generation_inputs(actual_model, actual_record, **kwargs)
        return {'prediction': 'red', 'prompt_sha256': 'b' * 64, 'generated_token_ids': [7], 'generated_tokens': 1}
    monkeypatch.setattr(native, 'build_generation_inputs', builder)
    monkeypatch.setattr(native, 'generate_prediction', generate)
    result = controls.NativeGateRuntime().generate(model, {**record(), 'answers': ['secret']})
    assert called == [record()] and native.build_generation_inputs is builder
    assert result['prefix_token_ids'] == [1, 2]
    assert result['prompt_token_ids_sha256'] == controls.digest(prepared)


def test_actual_native_visual_generation_has_same_prompt_and_identity_control():
    class Tokenizer:
        pad_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            return 'user ' + messages[0]['content'] + ' assistant'
        def encode(self, text, **kwargs):
            return [3 + ord(char) % 29 for char in text]
        def decode(self, ids, **kwargs):
            return ' '.join(str(token) for token in ids)
    class Cache:
        def get_many(self, paths, encode, **kwargs):
            return [torch.tensor([[1., 2., 3.], [4., 5., 6.]]) for _ in paths]
    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = model(1)
            self.base.generation_config.eos_token_id = 2
            self.projector = torch.nn.Linear(3, 8, bias=False)
            self.tokenizer = Tokenizer()
            self.cache = Cache()
            self.args = SimpleNamespace(max_length=128)
        def encode(self, _):
            raise AssertionError('tiny test cache should hit')
    native = Visual().eval().requires_grad_(False)
    first = {**record(), 'question': 'Color?', 'answers': ['SECRET_RED']}
    second = {**first, 'answers': ['SECRET_BLUE']}
    runtime = controls.NativeGateRuntime()
    with torch.no_grad():
        baseline = runtime.generate(native, first)
        with controls.gate_control(native, 'learned', recipe='1c'):
            actual = runtime.generate(native, second)
    assert actual == baseline
    assert actual['visual_tokens'] == 2
    assert actual['prefill_tokens'] == len(actual['prefix_token_ids']) + 2 + len(actual['suffix_token_ids'])
    assert all('sigmoid' not in block.modality_gate_logits.__dict__ for _, block, _ in controls.gate_registry(native))


def test_preflight_import_and_help_never_import_torch():
    import os
    import subprocess
    import sys
    child = subprocess.run([sys.executable, '-c',
        "import sys; import src.eval.research_gate_controls; assert 'torch' not in sys.modules"],
        cwd=controls.ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    help_result = subprocess.run([sys.executable, 'scripts/eval_research_gate_control.py', '--help'],
        cwd=controls.ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
    assert help_result.returncode == 0
    assert '--condition' in help_result.stdout


def protocol_fixture():
    context = {'static': {'condition': 'all_open', 'checkpoint': {'role': 'best',
        'config': {'recipe': '2c', 'model': 'Small', 'vision': 'Vision'},
        'roles': {'best': {'marker': {'step': 100}}}}}}
    receipt = {'owner': controls.OWNER, 'condition': 'all_open', 'recipe': '2c',
        'parameter_values_written': False, 'gates': [{'exact': [1., 1.]}]}
    receipt['gates_sha256'] = controls.digest(receipt['gates'])
    provenance = {'recipe': '2c', 'stage': 'SFT', 'training_step': 100,
        'model_ref': {'repo': 'Small', 'revision': 'a' * 40},
        'vision_ref': {'repo': 'Vision', 'revision': 'b' * 40}, 'cache_fingerprint': 'c' * 64}
    return context, receipt, provenance


def test_new_secondary_protocol_binds_gates_and_refuses_legacy_or_modified_resume():
    context, receipt, provenance = protocol_fixture()
    protocol = controls.control_protocol(context, receipt, provenance)
    assert controls.validate_resume(json.loads(json.dumps(protocol)), protocol)
    assert protocol['legacy_cache_compatible'] is False
    assert protocol['namespace'].endswith('/all_open')
    for field in ('observed_gate_control', 'loaded_model', 'static', 'owner'):
        altered = json.loads(json.dumps(protocol))
        altered[field] = 'drift'
        with pytest.raises(ValueError, match='resume'):
            controls.validate_resume(altered, protocol)
    with pytest.raises(ValueError, match='resume'):
        controls.validate_resume({'schema_version': 1, 'checkpoint_sha256': 'd' * 64}, protocol)


@pytest.mark.parametrize('mutation', ('gates', 'condition', 'family', 'step', 'model', 'vision', 'revision', 'cache'))
def test_runtime_protocol_corruption_refused(mutation):
    context, receipt, provenance = protocol_fixture()
    if mutation == 'gates': receipt['gates'][0]['exact'][0] = 0.
    elif mutation == 'condition': receipt['condition'] = 'learned'
    elif mutation == 'family': receipt['recipe'] = '3c'
    elif mutation == 'step': provenance['training_step'] = 170
    elif mutation == 'model': provenance['model_ref']['repo'] = 'Other'
    elif mutation == 'vision': provenance['vision_ref']['repo'] = 'Other'
    elif mutation == 'revision': provenance['vision_ref']['revision'] = 'main'
    elif mutation == 'cache': provenance['cache_fingerprint'] = None
    with pytest.raises(ValueError):
        controls.control_protocol(context, receipt, provenance)
