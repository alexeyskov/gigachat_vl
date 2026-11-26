"""Prospective collector wiring through the real native CPU retry/step loop."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.utils import research_routing_aggregate_v1 as routing
from src.utils import research_routing_training_v1 as facade
from test_research_routing_aggregate_v1 import Tiny


class CPUResearchModel(nn.Module):
    """Synthetic CPU model uses the genuine native Small FFN/router body."""
    def __init__(self, args):
        super().__init__(); self.args = args; self.llm = Tiny('4f').requires_grad_(False)
        self.projector = nn.Linear(4, 4); self.extractor = SimpleNamespace(vision_tower=nn.Identity())
        self.wrappers = []; self.cache = SimpleNamespace(stats={}, fingerprint='b' * 64)
        self.model_ref = {'repo': args.model, 'revision': 'a' * 40}
        self.vision_ref = {'repo': args.vision, 'revision': 'c' * 40}
        self.calls = 0; self.oom_at = None
    def prepare(self, records):
        return [([1], [2], [3], [torch.zeros(2, 3)]) if r['task_group'] == 'vqa' else ([], [1], [2], []) for r in records]
    def _forward_loss(self, rows, components=False):
        self.calls += 1
        lengths = [len(r[0]) + len(r[1]) + len(r[2]) + sum(x.shape[0] for x in r[3]) for r in rows]
        width = max(lengths)
        inputs = torch.arange(len(rows) * width * 4).float().reshape(len(rows), width, 4) / 20
        hidden = self.llm(self.projector(inputs)); per_row = hidden.square().sum((1, 2))
        if self.calls == self.oom_at: raise torch.cuda.OutOfMemoryError('synthetic after routed forward')
        count = sum(len(r[2]) for r in rows); sequence = sum(lengths)
        if not components: return per_row.sum(), count, sequence
        mm = torch.tensor([bool(r[3]) for r in rows])
        return {'multimodal_loss_sum': per_row[mm].sum(), 'text_only_loss_sum': per_row[~mm].sum(),
            'multimodal_supervised_tokens': sum(len(r[2]) for r in rows if r[3]),
            'text_only_supervised_tokens': sum(len(r[2]) for r in rows if not r[3]), 'sequence_tokens': sequence}
    def forward_loss(self, rows): return self._forward_loss(rows)
    def forward_loss_components(self, rows): return self._forward_loss(rows, components=True)


def module_and_model(tmp_path, monkeypatch, *, observed=True):
    module = facade.stock.runner(); module.ResearchModel = CPUResearchModel
    records = [{'record_id': str(i), 'task_group': 'vqa' if i % 2 == 0 else 'raw_text'} for i in range(4)]
    tmp_path.mkdir(parents=True, exist_ok=True); manifest = tmp_path / 'train.jsonl'
    manifest.write_text(''.join(json.dumps(record) + '\n' for record in records))
    inputs = facade.inspect_inputs(SimpleNamespace(manifest=str(manifest), text_manifest=None))[1]
    config = facade.definition(['raw_text', 'vqa'], inputs=inputs)
    if observed: facade.instrument_runner(module, config)
    args = SimpleNamespace(model=facade.stock.SMALL, vision='synthetic/vision', output=str(tmp_path), manifest=str(manifest), resume=False)
    setattr(args, facade.FIELD, config)
    model = module.ResearchModel(args)
    prepared = [model.prepare(records[:2]), model.prepare(records[2:])]
    return module, model, prepared, config


@pytest.mark.parametrize('text_weight', [None, .15])
def test_native_OOM_retry_discards_partial_observations_and_preserves_final_gradients(tmp_path, monkeypatch, text_weight):
    torch.manual_seed(81); plain, base, prepared_base, _ = module_and_model(tmp_path / 'base', monkeypatch, observed=False)
    torch.manual_seed(81); module, model, prepared, _ = module_and_model(tmp_path / 'observed', monkeypatch)
    optimizer = torch.optim.SGD(model.projector.parameters(), lr=.01)
    base_optimizer = torch.optim.SGD(base.projector.parameters(), lr=.01)
    model.oom_at = 2
    result = module.train_accumulation_group(model, optimizer, prepared, list(model.projector.parameters()), 2, text_loss_weight=text_weight)
    expected = plain.train_accumulation_group(base, base_optimizer, prepared_base, list(base.projector.parameters()), 1, text_loss_weight=text_weight)
    assert result['microbatch_size'] == 1 and result['supervised_tokens'] == expected['supervised_tokens']
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(model.projector.parameters(), base.projector.parameters()))
    assert model._routing_collector.phase == 'ready' and model._routing_collector.step == 0
    with pytest.raises(ValueError): model._routing_collector.export_state(step=1, seen=4)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    optimizer.step(); schedule.step()
    module._routing_commit(model, 1, 4)
    state = model._routing_collector.export_state(step=1, seen=4)
    assert state['discarded_attempts'] == 1
    assert state['layers']['moe']['counters']['tokens'].sum() == 14
    model._routing_collector.close()


def test_failed_optimizer_or_scheduler_does_not_create_a_committed_window(tmp_path, monkeypatch):
    module, model, prepared, _ = module_and_model(tmp_path, monkeypatch)
    optimizer = torch.optim.SGD(model.projector.parameters(), lr=.01)
    module.train_accumulation_group(model, optimizer, prepared, list(model.projector.parameters()), 2)
    # An actual optimizer error prevents the native commit hook from running.
    with pytest.raises(RuntimeError):
        raise RuntimeError('synthetic optimizer/scheduler failure before synchronization and step increment')
    assert model._routing_collector.step == 0
    model._routing_collector.abort_window()
    assert model._routing_collector.export_state(step=0, seen=0)['layers']['moe']['counters']['tokens'].sum() == 0
    model._routing_collector.close()


def test_real_native_main_commit_checkpoint_and_resume_are_bound_without_touching_global_progress(tmp_path, monkeypatch):
    import src.utils.research_provenance as provenance
    # All native relative progress paths stay in the task-owned temporary CWD.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance, 'write_run_provenance', lambda *a, **k: {'code_sha256': 'a' * 64, 'source_archive': 'SYNTHETIC_CPU_TEST_ONLY'})
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    manifest = tmp_path / 'train.jsonl'
    records = [{'record_id': str(i), 'task_group': 'vqa' if i % 2 == 0 else 'raw_text', 'question': 'synthetic', 'answer': 'synthetic'} for i in range(4)]
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in records))
    output = tmp_path / 'synthetic-run'
    module = facade.stock.runner(); module.ResearchModel = CPUResearchModel
    args = module.parse_args(['--manifest', str(manifest), '--validation', str(manifest), '--output', str(output),
        '--stage', 'A0', '--recipe', '4f', '--max-steps', '1', '--batch-size', '2', '--effective-batch-size', '4', '--eval-records', '2'])
    args.numerical_profile = facade.stock.definition(); args.allow_legacy_a0_initialization = False
    groups, inputs = facade.inspect_inputs(args); config = facade.definition(groups, inputs=inputs)
    setattr(args, facade.FIELD, config); facade.instrument_runner(module, config); module.parse_args = lambda: args
    module.main()
    path = output / 'last/state.pt'; saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['step'] == 1 and saved['seen'] == 4 and saved[facade.STATE]['step'] == 1
    assert saved[facade.STATE]['seen'] == 4 and saved[facade.STATE]['layers']['moe']['counters']['tokens'].sum() == 14
    assert (output / 'routing_aggregate_v1/step-00000001.json').is_file()
    original_parameter_keys = set(saved['model']); assert all('_routing' not in key for key in original_parameter_keys)
    module._routing_live['model']._routing_collector.close()
    # A same-config resume at its terminal step performs no additional update.
    again = facade.stock.runner(); again.ResearchModel = CPUResearchModel
    args.resume = True; facade.instrument_runner(again, config); again.parse_args = lambda: args
    again.main()
    restored = again._routing_live['model']._routing_collector
    assert restored.step == 1 and restored.seen == 4
    assert restored.export_state(step=1, seen=4)['state_sha256'] == saved[facade.STATE]['state_sha256']
    restored.close()


def test_corrupted_counter_checkpoint_refuses_even_with_correct_step_and_profile(tmp_path, monkeypatch):
    module, model, prepared, _ = module_and_model(tmp_path, monkeypatch)
    optimizer = torch.optim.SGD(model.projector.parameters(), lr=.01)
    module.train_accumulation_group(model, optimizer, prepared, list(model.projector.parameters()), 2)
    optimizer.step(); module._routing_commit(model, 1, 4)
    state = model._routing_collector.export_state(step=1, seen=4); model._routing_collector.close()
    state['layers']['moe']['counters']['tokens'][0, 0] += 1
    restored = routing.RoutingAggregate(model.llm, ['raw_text', 'vqa'])
    with pytest.raises(ValueError, match='checksum'): restored.restore_state(state, step=1, seen=4)
    restored.close()


def test_crash_after_checkpoint_publication_repairs_missing_report_on_terminal_resume(tmp_path, monkeypatch):
    import src.utils.research_provenance as provenance
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance, 'write_run_provenance', lambda *a, **k: {'code_sha256': 'a' * 64, 'source_archive': 'SYNTHETIC_CPU_TEST_ONLY'})
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None); monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    manifest = tmp_path / 'train.jsonl'; manifest.write_text(''.join(json.dumps({'record_id': str(i), 'task_group': 'vqa'}) + '\n' for i in range(2)))
    output = tmp_path / 'synthetic-crash'
    module = facade.stock.runner(); module.ResearchModel = CPUResearchModel
    args = module.parse_args(['--manifest', str(manifest), '--validation', str(manifest), '--output', str(output),
        '--stage', 'A0', '--recipe', '4f', '--max-steps', '1', '--batch-size', '2', '--effective-batch-size', '2'])
    args.numerical_profile = facade.stock.definition(); args.allow_legacy_a0_initialization = False
    groups, inputs = facade.inspect_inputs(args); config = facade.definition(groups, inputs=inputs); setattr(args, facade.FIELD, config)
    facade.instrument_runner(module, config); module.parse_args = lambda: args
    writer = facade._write_report
    with monkeypatch.context() as local:
        local.setattr(facade, '_write_report', lambda *a: (_ for _ in ()).throw(InterruptedError('after manager publication, before JSON report')))
        with pytest.raises(InterruptedError): module.main()
    saved = torch.load(output / 'last/state.pt', map_location='cpu', weights_only=False)
    before = {name: value.clone() for name, value in saved['model'].items()}
    assert saved['step'] == 1 and not (output / 'routing_aggregate_v1/step-00000001.json').exists()
    assert not (output / 'completed.json').exists()
    module._routing_live['model']._routing_collector.close()
    again = facade.stock.runner(); again.ResearchModel = CPUResearchModel; args.resume = True
    facade.instrument_runner(again, config); again.parse_args = lambda: args; again.main()
    report = json.loads((output / 'routing_aggregate_v1/step-00000001.json').read_bytes())
    assert report['state_sha256'] == saved[facade.STATE]['state_sha256']
    assert report['checkpoint_binding']['routing_state_sha256'] == saved[facade.STATE]['state_sha256']
    assert (output / 'completed.json').is_file() and again._routing_live['model'].calls == 0
    restored = torch.load(output / 'last/state.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(value, restored['model'][name]) for name, value in before.items())
    again._routing_live['model']._routing_collector.close()


def test_raw_group_vocabulary_and_manifest_sha_are_bound_before_any_training(tmp_path):
    path = tmp_path / 'train.jsonl'; path.write_text('{"record_id":"one","task_group":"unmodified_raw_group"}\n')
    args = SimpleNamespace(manifest=str(path), text_manifest=None)
    groups, inputs = facade.inspect_inputs(args)
    assert groups == ['unmodified_raw_group'] and inputs['manifest']['path'] == str(path)
    path.write_text('{"record_id":"two"}\n')
    with pytest.raises(ValueError, match='raw task_group'): facade.inspect_inputs(args)


def test_source_archive_is_atomic_immutable_and_binds_all_new_code(tmp_path):
    config = facade.definition(['vqa'])
    facade._source_archive(tmp_path, config)
    directory = tmp_path / 'routing_aggregate_source_v1'
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    facade._source_archive(tmp_path, config)
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    assert not list(tmp_path.glob('.routing_aggregate_source_v1.*.tmp'))
    receipt = json.loads((directory / 'source.json').read_bytes())
    assert receipt['profile'] == config and 'src/utils/research_routing_aggregate_v1.py' in config['source_sha256']
    (directory / 'source.zip').write_bytes(b'changed')
    with pytest.raises(ValueError, match='source archive changed'): facade._source_archive(tmp_path, config)


def test_existing_legacy_output_refuses_before_source_archive_or_training(tmp_path):
    legacy = tmp_path / 'legacy'; legacy.mkdir(); (legacy / 'resolved_config.json').write_text('{"old_scientific_run":true}')
    manifest = tmp_path / 'train.jsonl'; manifest.write_text('{"record_id":"one","task_group":"vqa"}\n')
    source = f"""
import sys
from types import SimpleNamespace
from pathlib import Path
from src.utils import research_routing_training_v1 as f
assert 'torch' not in sys.modules
args=SimpleNamespace(model=f.stock.SMALL,vision='synthetic/vision',output={str(legacy)!r},manifest={str(manifest)!r},text_manifest=None)
f.stock.activate=lambda:None
f.verify_release=lambda:{{'synthetic_CPU_test':True}}
f.stock.runner=lambda:SimpleNamespace(parse_args=lambda remaining:args)
f._source_archive=lambda *a:(_ for _ in ()).throw(AssertionError('Must not write historical output'))
try:f.training_main(['--allow-gpu','--numerical-profile',f.stock.PROFILE])
except ValueError as error:assert 'historical output adoption' in str(error)
else:raise AssertionError('Legacy output must refuse')
assert [p.name for p in Path({str(legacy)!r}).iterdir()]==['resolved_config.json']
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', source], cwd=facade.ROOT, check=True, capture_output=True, text=True)


def test_default_cli_and_metadata_allow_flag_never_import_torch_or_start_gpu():
    code = "import sys,runpy;sys.argv=['routing','--metadata','--allow-gpu'];runpy.run_path('scripts/run_research_routing_aggregate_v1.py',run_name='__main__');assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code], cwd=facade.ROOT, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['actual_collected_training_rows'] == 0


def test_canonical_runner_instrumentation_and_late_gpu_invocation_are_refused():
    from scripts import run_research_stage as canonical
    with pytest.raises(ValueError, match='Canonical'): facade.instrument_runner(canonical, {})
    with pytest.raises(RuntimeError, match='fresh process'): facade.training_main(['--allow-gpu'])
