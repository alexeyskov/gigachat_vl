"""Synthetic CPU lifecycle fixtures; never scientific model-quality evidence."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
import torch

from src.eval import research_gate_controls as controls
from src.eval import research_gate_evaluation as owner
from src.eval import research_parallel_protocol as reference
from src.eval.research_metrics import aggregate_scores, score_prediction
from src.model.research_moe import ResearchSmallThinkerMoE
from src.model.smallthinker import SmallThinkerConfig, SmallThinkerMoeBlock


def json_file(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(owner.canonical(value) + b'\n')


def json_rows(path, values):
    Path(path).write_bytes(b''.join(owner.canonical(value) + b'\n' for value in values))


def tiny_block():
    torch.manual_seed(37)
    cfg = SmallThinkerConfig(vocab_size=16, hidden_size=4, num_hidden_layers=1,
        num_attention_heads=1, num_key_value_heads=1, head_dim=4, moe_ffn_hidden_size=4,
        moe_num_primary_experts=2, moe_num_active_primary_experts=1,
        max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    block = ResearchSmallThinkerMoE(SmallThinkerMoeBlock(cfg), 2, 'c', added_experts=2)
    block.modality_gate_logits.data.copy_(torch.linspace(-1, 1, 8).reshape(4, 2))
    return block.eval().requires_grad_(False)


def synthetic_generated(record):
    """Deterministic fake text for lifecycle testing, never a quality result."""
    benchmark = record['question'].split()[0]
    prediction = 'A' if benchmark in ('ai2d', 'scienceqa', 'aokvqa') else '2' if benchmark == 'chartqa' else 'yes'
    result = {'prediction': prediction, 'generated_token_ids': [3], 'generated_tokens': 1,
        'prompt_sha256': hashlib.sha256(record['question'].encode()).hexdigest(), 'prefill_tokens': 4,
        'visual_tokens': 2, 'logits_to_keep': 1, 'prefix_token_ids': [1], 'suffix_token_ids': [4]}
    captured = {key: result[key] for key in ('prefix_token_ids', 'suffix_token_ids', 'prompt_sha256', 'visual_tokens', 'prefill_tokens')}
    result['prompt_token_ids_sha256'] = owner.digest(captured)
    return result


class SyntheticRuntime:
    """Real tiny gate/FFN forward, synthetic fixed emitted tokens; explicit tag."""
    def __init__(self, context, *, interrupt_after=None, drift_guard=False):
        self.context = context; self.calls = []; self.models = []; self.interrupt_after = interrupt_after
        self.drift_guard = drift_guard

    def load(self, checkpoint, cache_dir, reserve_gb):
        actual = tiny_block()
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        actual.load_state_dict(payload['model'], strict=True)
        self.models.append(actual)
        return actual, self.context['static']['original_native512']['native_model_provenance']

    def describe(self, model, loaded):
        return {'loaded': loaded, 'context_limit': 128, 'eos_token_ids': [2],
            'model_type': 'smallthinker', 'master_parameter_dtypes': ['torch.float32'],
            'pinned_snapshots': self.context['_fixture_snapshots'], 'runtime': {'device': 'CPU synthetic lifecycle fixture'}}

    def generate(self, model, record):
        assert set(record) == controls.PUBLIC_FIELDS, 'Gold/scorer metadata reached model'
        if self.interrupt_after is not None and len(self.calls) >= self.interrupt_after:
            raise InterruptedError('synthetic interruption after durable prefix')
        self.calls.append(owner.clone(record))
        with torch.no_grad():
            model.set_modality_mask(torch.tensor([[True, False]]))
            states = torch.ones(1, 2, 4)
            output, _ = model(states, states)
            assert torch.isfinite(output).all()
        generated = synthetic_generated(record)
        if self.drift_guard:
            generated['generated_token_ids'] = [5]
        return generated


@pytest.fixture
def context(tmp_path, monkeypatch):
    from PIL import Image
    from src.eval.research_caption_generation import _image
    run = tmp_path / 'synthetic-only' / 'run'; run.mkdir(parents=True)
    image = tmp_path / 'synthetic-only' / 'rgb.png'
    Image.new('RGB', (2, 2), (11, 29, 43)).save(image)
    image_hash = hashlib.sha256(b'RGB:2:2:' + bytes((11, 29, 43)) * 4).hexdigest()
    records = []
    for benchmark in reference.BENCHMARKS:
        for index in range(64):
            scorer = reference.SCORERS[benchmark]
            answer = 'A' if scorer == 'mcq_accuracy' else '2' if benchmark == 'chartqa' else 'yes'
            row = {'record_id': f'{benchmark}-{index}', 'question': f'{benchmark} synthetic question {index}',
                'image_paths': [str(image)], 'image_hashes': [image_hash], 'answers': [answer] * (10 if scorer == 'vqa_accuracy' else 1),
                'benchmark': benchmark, 'task_group': reference.BENCHMARKS[benchmark], 'scorer': scorer,
                'source': {'repo': 'synthetic/fixture', 'revision': 'a' * 40, 'split': 'validation',
                    'shard': 'synthetic.parquet', 'benchmark': benchmark}}
            if scorer == 'vqa_accuracy': row['scoring_protocol'] = 'vqav2_official_legacy' if benchmark == 'vqav2' else 'textvqa_evalai'
            if scorer == 'mcq_accuracy': row['choices'] = ['yes', 'no']
            records.append(row)
    manifest = tmp_path / 'synthetic-only' / 'evaluation.jsonl'; json_rows(manifest, records)
    monkeypatch.setattr(controls, 'MANIFEST_SHA256', owner.file_sha(manifest))
    config = {'model': 'Tiiny/SmallThinker-4BA0.6B-Instruct', 'vision': 'google/siglip2-so400m-patch14-384',
        'recipe': '2c', 'stage': 'SFT', 'output': str(run), 'cache_dir': str(tmp_path / 'cache')}
    json_file(run / 'resolved_config.json', config)
    json_file(run / '.checkpoint_manager.json', {'owner': 'synthetic-owner'})
    (run / '.checkpoint.lock').touch()
    target = run / '.checkpoint_objects' / 'object-synthetic'; target.mkdir(parents=True)
    state = tiny_block().state_dict(); torch.save({'model': state, 'config': config, 'step': 2}, target / 'state.pt')
    json_file(target / '.research_checkpoint.json', {'owner': 'synthetic-owner', 'step': 2,
        'training_completed': True, 'resumable_optimizer': False, 'optimizer_state_removed': True, 'created_unix': 1})
    for role in ('best', 'last'): (run / role).symlink_to(Path('.checkpoint_objects') / target.name)
    json_file(run / 'completed.json', {'steps': 2})
    json_file(run / 'checkpoint_compaction.json', {'status': 'completed', 'training_completed': True,
        'resumable_optimizer': False, 'run_dir': str(run), 'checkpoints': [{'new_object': target.name}]})
    provenance = {'model_ref': {'repo': config['model'], 'revision': 'a' * 40},
        'vision_ref': {'repo': config['vision'], 'revision': 'b' * 40},
        'recipe': '2c', 'stage': 'SFT', 'training_step': 2, 'cache_fingerprint': 'c' * 64}
    model_assets, vision_assets = tmp_path / 'model-assets', tmp_path / 'vision-assets'
    json_file(model_assets / 'config.json', {'model_type': 'smallthinker', 'num_hidden_layers': 1})
    json_file(model_assets / 'generation_config.json', {'eos_token_id': 2})
    json_file(vision_assets / 'config.json', {'model_type': 'synthetic-vision'})
    monkeypatch.setattr(owner, 'generation_asset', lambda _: (owner.descriptor(model_assets / 'generation_config.json'), [2]))
    native = run / 'evaluation' / 'heldout_v3'; native.mkdir(parents=True)
    from src.utils.research_checkpoint_compat import checkpoint_identity
    identity = checkpoint_identity(target / 'state.pt')
    native_sources = {name: owner.file_sha(owner.ROOT / name) for name in (
        'src/eval/research_inference.py', 'src/eval/research_metrics.py', 'scripts/eval_research_checkpoint.py')}
    protocol = {'schema_version': 1, 'checkpoint_sha256': identity['sha256'], 'manifest_sha256': owner.file_sha(manifest),
        'manifest_records': 512, 'selected_record_ids_sha256': hashlib.sha256(json.dumps([r['record_id'] for r in records]).encode()).hexdigest(),
        'image_ablation': {'schema_version': 1, 'variant': 'none', 'seed': 42, 'records': 512,
            'source_manifest_sha256': owner.file_sha(manifest), 'transformed_manifest_sha256': owner.file_sha(manifest)},
        'generation': reference.NATIVE_GENERATION, 'source_sha256': native_sources, 'environment': owner.native_environment()}
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    raw = []
    for record in records:
        generated = synthetic_generated(record)
        row = {key: record[key] for key in reference.ROW_FIELDS if key in record}
        row.update({key: generated[key] for key in reference.GEN_FIELDS},
            protocol_sha256=protocol_sha, score=score_prediction(record, generated['prediction']))
        raw.append(row)
    summary = {**aggregate_scores(raw), 'complete_manifest': True, 'manifest_records': 512,
        'protocol_sha256': protocol_sha, 'checkpoint_sha256': identity['sha256'],
        'manifest_sha256': owner.file_sha(manifest), 'image_ablation': protocol['image_ablation']}
    json_file(native / 'protocol.json', protocol); json_rows(native / 'predictions.jsonl', raw)
    json_file(native / 'summary.json', summary); json_file(native / 'model_provenance.json', provenance)
    result = owner.prepare_context(run, condition='visual_closed', manifest=manifest, native_cache=native)
    result['_fixture_snapshots'] = {name: {'snapshot': str(path), 'files': {
        str(file.relative_to(path)): owner.descriptor(file) for file in path.iterdir()}} for name, path in (
            ('model', model_assets), ('vision', vision_assets))}
    return result


def test_full512_synthetic_cpu_lifecycle_has_real_gate_scopes_and_exact_independent_scores(context):
    engine = SyntheticRuntime(context)
    result = owner.evaluate(context, allow_gpu=True, runtime=engine)
    assert result['records'] == 512 and result['complete_manifest'] is True
    assert result['evidence_scope'] == 'synthetic_CPU_lifecycle_fixture'
    assert result['primary_completion_claim'] is False and result['P06_complete'] is False
    assert result['group_macro_score'] == 100.
    assert result['paired_original_minus_control']['original_minus_variant_group_macro_points'] == 0.
    assert result['paired_original_minus_control']['paired_95_percentile_ci'] == [0., 0.]
    assert len(engine.calls) == 64 + 512
    output = owner.output_path(context)
    rows = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == 512 and rows[0]['original_gate_pre_sha256'] == rows[0]['original_gate_post_sha256']
    assert len(list((output / 'records').glob('*.json'))) == 512
    proof = owner.read_json(output / 'generation_complete.json')
    assert proof['all_named_parameter_raw_bits_preserved'] is True
    assert all('answers' not in row['generated'] and 'choices' not in row['generated'] for row in rows)
    assert all('sigmoid' not in model.modality_gate_logits.__dict__ for model in engine.models)
    # Complete CPU replay must not instantiate a model or silently change output.
    repeated = owner.evaluate(context, resume=True)
    assert repeated == result


def test_interrupted_prefix_resume_repeats_fresh64_only_then_pending_and_restores(context):
    engine = SyntheticRuntime(context, interrupt_after=64 + 3)
    with pytest.raises(InterruptedError): owner.evaluate(context, allow_gpu=True, runtime=engine)
    output = owner.output_path(context)
    assert len(list((output / 'records').glob('*.json'))) == 3
    assert not (output / 'generation_complete.json').exists() and not (output / 'summary.json').exists()
    assert all('sigmoid' not in model.modality_gate_logits.__dict__ for model in engine.models)
    engine2 = SyntheticRuntime(context)
    result = owner.evaluate(context, resume=True, allow_gpu=True, runtime=engine2)
    assert len(engine2.calls) == 64 + 509 and result['records'] == 512
    assert len(list((output / 'guards').glob('*.json'))) == 2


def test_guard_mismatch_commits_no_controls_and_blocks_namespace(context):
    engine = SyntheticRuntime(context, drift_guard=True)
    with pytest.raises(ValueError, match='learned identity'):
        owner.evaluate(context, allow_gpu=True, runtime=engine)
    output = owner.output_path(context)
    assert not (output / 'protocol.json').exists()
    assert not (output / 'records').exists() and not (output / 'summary.json').exists()
    with pytest.raises(ValueError, match='Prior learned64 parity failure'):
        owner.evaluate(context, resume=True, allow_gpu=True, runtime=SyntheticRuntime(context))


@pytest.mark.parametrize('mutation', ('checkpoint', 'manifest', 'reference', 'owner_source', 'environment', 'image'))
def test_actual_file_and_environment_mutations_fail_closed(context, mutation, monkeypatch):
    static = context['static']
    if mutation == 'checkpoint': path = Path(static['base']['checkpoint']['checkpoint_path'])
    elif mutation == 'manifest': path = Path(static['base']['manifest']['path'])
    elif mutation == 'reference': path = Path(static['original_native512']['files']['predictions.jsonl']['path'])
    elif mutation == 'owner_source':
        static['sources']['src/eval/research_gate_evaluation.py']['stat']['st_size'] += 1
        path = None
    elif mutation == 'environment':
        monkeypatch.setenv('OMP_NUM_THREADS', 'p07-drift')
        path = None
    else: path = Path(context['records'][0]['image_paths'][0])
    if path is not None: path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed'):
        owner.evaluate(context, allow_gpu=False)


@pytest.mark.parametrize('mutation', ('missing', 'partial', 'wrong_checkpoint', 'wrong_score', 'wrong_tokens', 'wrong_source', 'wrong_summary'))
def test_original_native512_reference_is_required_and_fully_validated(context, mutation):
    binding = context['static']['original_native512']; directory = Path(binding['directory'])
    if mutation == 'missing': (directory / 'predictions.jsonl').unlink()
    elif mutation == 'partial':
        path = directory / 'predictions.jsonl'; path.write_bytes(b'\n'.join(path.read_bytes().splitlines()[:-1]) + b'\n')
    elif mutation == 'wrong_checkpoint':
        value = owner.read_json(directory / 'protocol.json'); value['checkpoint_sha256'] = 'd' * 64
        json_file(directory / 'protocol.json', value)
    elif mutation in ('wrong_score', 'wrong_tokens'):
        path = directory / 'predictions.jsonl'; rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == 'wrong_score': rows[0]['score'] = .1
        else: rows[0]['generated_token_ids'] = [2, 3]; rows[0]['generated_tokens'] = 2
        json_rows(path, rows)
    elif mutation == 'wrong_source':
        value = owner.read_json(directory / 'protocol.json'); value['source_sha256']['src/eval/research_metrics.py'] = '0' * 64
        json_file(directory / 'protocol.json', value)
    else:
        value = owner.read_json(directory / 'summary.json'); value['group_macro_score'] -= 1
        json_file(directory / 'summary.json', value)
    with pytest.raises((ValueError, FileNotFoundError)):
        owner.validate_reference(context['base'], directory)


def interrupted(context, count=2):
    with pytest.raises(InterruptedError):
        owner.evaluate(context, allow_gpu=True, runtime=SyntheticRuntime(context, interrupt_after=64 + count))
    return owner.output_path(context)


@pytest.mark.parametrize('mutation', ('gap', 'token', 'gate', 'prompt', 'guard', 'foreign', 'legacy'))
def test_journal_corruption_and_legacy_cache_refused(context, mutation):
    output = interrupted(context)
    if mutation == 'gap': (output / 'records/000000.json').unlink()
    elif mutation in ('token', 'gate', 'prompt'):
        path = output / 'records/000000.json'; row = owner.read_json(path)
        if mutation == 'token': row['generated']['generated_token_ids'] = [9]
        elif mutation == 'gate': row['original_gate_post_sha256'] = '0' * 64
        else: row['generated']['prefix_token_ids'] = [8]
        json_file(path, row)
    elif mutation == 'guard':
        guard = next((output / 'guards').glob('*.json')); value = owner.read_json(guard); value['status'] = 'running'; json_file(guard, value)
    elif mutation == 'foreign': (output / 'records' / 'unowned.json').write_text('{}')
    else:
        json_file(output / 'protocol.json', {'schema_version': 1, 'checkpoint_sha256': 'd' * 64})
    with pytest.raises(ValueError): owner.evaluate(context, resume=True, allow_gpu=False)


def test_killed_atomic_temp_preserved_and_truncated_raw_prefix_rebuilt(context):
    output = interrupted(context)
    stage = output / 'records' / ('.000002.json.' + uuid.uuid4().hex + '.tmp')
    stage.write_bytes(b'{"incomplete":')
    raw = output / 'predictions.jsonl'; fragment = raw.read_bytes()[:-13]; raw.write_bytes(fragment)
    result = owner.evaluate(context, resume=True, allow_gpu=True, runtime=SyntheticRuntime(context))
    assert result['records'] == 512 and not stage.exists()
    assert any(path.read_bytes() == b'{"incomplete":' for path in (output / 'recovery').glob('*.uncommitted'))
    assert any(path.read_bytes() == fragment for path in (output / 'recovery').glob('*.fragment'))


def test_divergent_raw_projection_is_not_truncated_or_repaired(context):
    output = interrupted(context)
    path = output / 'predictions.jsonl'; path.write_bytes(b'foreign raw bytes\n')
    with pytest.raises(ValueError, match='diverges'):
        owner.evaluate(context, resume=True, allow_gpu=False)
    assert path.read_bytes() == b'foreign raw bytes\n'


def test_no_gpu_default_refuses_pending_without_model_load_or_fake_summary(context):
    with pytest.raises(RuntimeError, match='--allow-gpu'): owner.evaluate(context)
    output = owner.output_path(context)
    assert not (output / 'summary.json').exists() and not (output / 'records').exists()


def test_cross_condition_resume_and_output_role_masquerade_refused(context):
    output = interrupted(context)
    changed = copy.deepcopy(context); changed['static']['base']['condition'] = 'all_open'
    with pytest.raises(ValueError, match='namespace'):
        owner.evaluate(changed, output=output, resume=True)
    with pytest.raises(ValueError, match='namespace'):
        owner.output_path(context, Path(context['static']['base']['checkpoint']['run']) / 'evaluation/heldout_v3')


def test_applied_endpoints_and_learned_bits_reconstructed_without_torch(context):
    module = tiny_block()
    with torch.no_grad():
        with controls.gate_control(module, 'learned', recipe='2c') as learned: learned = owner.clone(learned)
        for condition in controls.CONDITIONS:
            with controls.gate_control(module, condition, recipe='2c') as changed:
                assert owner.validate_control_against_guard(changed, learned)
                bad = owner.clone(changed); bad['gates'][0]['applied_values'][0][0] = .123
                bad['gates_sha256'] = owner.digest(bad['gates'])
                with pytest.raises(ValueError): owner.validate_control_against_guard(bad, learned)


def test_frozen_partial_release_bytes_are_not_changed():
    assert owner.verify_partial_release()['sha256'] == owner.PARTIAL_READY_SHA256


def test_raw_parameter_census_detects_data_mutation_even_without_version_change():
    model = tiny_block(); before = owner.parameter_bits(model); versions = owner.parameter_census(model)
    model.experts[0].up.weight.data.add_(.01)
    assert owner.parameter_census(model) == versions
    assert owner.parameter_bits(model) != before
    assert owner.parameter_bits(torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1.))]))['bytes'] == 4


def test_unbound_cache_directory_and_mutated_inmemory_gold_are_rejected(context, tmp_path):
    with pytest.raises(ValueError, match='cache directory'):
        owner.evaluate(context, cache_dir=tmp_path / 'other')
    context['original']['records'][0]['answers'][0] = 'tampered'
    with pytest.raises(ValueError, match='scorer/gold records changed'):
        owner.assert_context(context)


def test_data_write_mutation_blocks_full512_publication_and_all_retry(context):
    class MutatingRuntime(SyntheticRuntime):
        changed = False
        def generate(self, model, record):
            if not self.changed and 'sigmoid' in model.modality_gate_logits.__dict__:
                # Deliberately bypass the version counter. The final full raw
                # census must reject these synthetic rows before publication.
                model.experts[0].up.weight.data.add_(.01)
                self.changed = True
            return super().generate(model, record)
    engine = MutatingRuntime(context)
    with pytest.raises(ValueError, match='raw bits changed'):
        owner.evaluate(context, allow_gpu=True, runtime=engine)
    output = owner.output_path(context)
    assert len(list((output / 'records').glob('*.json'))) == 512
    assert (output / 'failed_integrity.json').exists()
    assert not (output / 'generation_complete.json').exists() and not (output / 'summary.json').exists()
    assert owner.read_json(output / 'progress.json')['status'] == 'failed_integrity'
    assert 'sigmoid' not in engine.models[0].modality_gate_logits.__dict__
    with pytest.raises(ValueError, match='prior full-parameter integrity failure'):
        owner.evaluate(context, resume=True, allow_gpu=True, runtime=SyntheticRuntime(context))


def test_data_write_then_interruption_cannot_be_certified_by_fresh_model(context):
    class MutatingRuntime(SyntheticRuntime):
        changed = False
        def generate(self, model, record):
            if not self.changed and 'sigmoid' in model.modality_gate_logits.__dict__:
                model.experts[0].up.weight.data.add_(.01)
                self.changed = True
            return super().generate(model, record)
    engine = MutatingRuntime(context, interrupt_after=64 + 3)
    with pytest.raises(ValueError, match='raw bits changed'):
        owner.evaluate(context, allow_gpu=True, runtime=engine)
    output = owner.output_path(context)
    assert len(list((output / 'records').glob('*.json'))) == 3
    closed = owner.read_json(next((output / 'attempts').glob('*/closed.json')))
    assert closed['status'] == 'failed_integrity' and closed['termination'] == 'InterruptedError'
    assert closed['records'] == 3
    assert not (output / 'summary.json').exists()
    fresh = SyntheticRuntime(context)
    with pytest.raises(ValueError, match='prior full-parameter integrity failure'):
        owner.evaluate(context, resume=True, allow_gpu=True, runtime=fresh)
    assert fresh.models == []


def test_unclosed_attempt_is_forensic_and_refuses_fresh_model_even_with_allow_gpu(context):
    output = interrupted(context)
    closure = next((output / 'attempts').glob('*/closed.json'))
    closure.unlink()  # CPU fixture simulates death before closure publication.
    before = {path.name: path.read_bytes() for path in (output / 'records').glob('*.json')}
    fresh = SyntheticRuntime(context)
    with pytest.raises(ValueError, match='unclosed model-lifetime'):
        owner.evaluate(context, resume=True, allow_gpu=True, runtime=fresh)
    assert fresh.models == []
    assert before == {path.name: path.read_bytes() for path in (output / 'records').glob('*.json')}
    assert not (output / 'summary.json').exists()


def test_actual_sigkill_leaves_unclosed_rows_untrusted(context, tmp_path):
    import os
    import subprocess
    import sys
    import time
    context_path = tmp_path / 'synthetic-context.json'; json_file(context_path, context)
    ready = tmp_path / 'kill-ready'; log = tmp_path / 'child.log'
    code = r'''
import json,runpy,signal,sys
from pathlib import Path
ns=runpy.run_path('tests/test_research_gate_evaluation.py')
owner=ns['owner'];controls=ns['controls'];Base=ns['SyntheticRuntime']
context=json.loads(Path(sys.argv[1]).read_text())
controls.MANIFEST_SHA256=context['static']['base']['manifest']['sha256']
class Waiting(Base):
    def generate(self,model,record):
        if len(self.calls)==66:
            Path(sys.argv[2]).write_text('two immutable rows committed before SIGKILL')
            while True:signal.pause()
        return super().generate(model,record)
owner.evaluate(context,allow_gpu=True,runtime=Waiting(context))
'''
    with log.open('w') as sink:
        child = subprocess.Popen([sys.executable, '-c', code, str(context_path), str(ready)],
            cwd=owner.ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            stdout=sink, stderr=sink)
        try:
            deadline = time.monotonic() + 40
            while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            assert ready.exists(), log.read_text()
            child.kill(); child.wait(timeout=10)
        finally:
            if child.poll() is None: child.kill(); child.wait(timeout=10)
    assert child.returncode == -9
    output = owner.output_path(context)
    assert len(list((output / 'records').glob('*.json'))) == 2
    assert not list((output / 'attempts').glob('*/closed.json'))
    fresh = SyntheticRuntime(context)
    with pytest.raises(ValueError, match='unclosed model-lifetime'):
        owner.evaluate(context, resume=True, allow_gpu=True, runtime=fresh)
    assert fresh.models == [] and not (output / 'generation_complete.json').exists()


def test_scope_method_replacement_after_commit_is_never_certified_by_restored_bits(context):
    class ReplacingRuntime(SyntheticRuntime):
        replaced = False
        def generate(self, model, record):
            if len(self.calls) == 65 and not self.replaced:
                model.modality_gate_logits.sigmoid = lambda: torch.ones_like(model.modality_gate_logits)
                self.replaced = True
            return super().generate(model, record)
    engine = ReplacingRuntime(context, interrupt_after=66)
    with pytest.raises(ValueError, match='control-scope failure'):
        owner.evaluate(context, allow_gpu=True, runtime=engine)
    output = owner.output_path(context)
    assert len(list((output / 'records').glob('*.json'))) == 2
    closed = owner.read_json(next((output / 'attempts').glob('*/closed.json')))
    assert closed['original_parameter_bits_pre'] == closed['original_parameter_bits_post']
    assert closed['status'] == 'failed_scope_integrity' and closed['termination'] == 'ValueError'
    assert closed['termination_verified_graceful'] is False
    assert 'sigmoid' not in engine.models[0].modality_gate_logits.__dict__
    assert not (output / 'generation_complete.json').exists() and not (output / 'summary.json').exists()
    fresh = SyntheticRuntime(context)
    with pytest.raises(ValueError, match='prior full-parameter integrity failure'):
        owner.evaluate(context, resume=True, allow_gpu=True, runtime=fresh)
    assert fresh.models == []
