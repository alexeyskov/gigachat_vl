"""CPU-only controller lifecycle; GPU/token proof is a mandatory external gate."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts import research_queue as base
from scripts import research_grounding_queue as native
from scripts import research_parallel_grounding_queue as candidate
from src.utils.research_storage import BestLastCheckpointManager, DiskBudget


def records():
    groups = sorted(native.GROUPS)
    return [{'record_id': str(i), 'question': 'red?', 'answers': ['red'], 'scorer': 'exact_match',
             'task_group': groups[i % 4], 'image_paths': [f'/fixture/image-{i % 4}.png'],
             'image_hashes': [str(i % 4) * 64], 'source': {'repo': 'fixture', 'revision': 'a' * 40},
             'benchmark': 'fixture'} for i in range(512)]


def generated(rows):
    return [dict(row, prediction='red', score=1., generated_token_ids=[7], generated_tokens=1) for row in rows]


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'ROOT', tmp_path)
    monkeypatch.setattr(native, 'ROOT', tmp_path)
    monkeypatch.setattr(candidate, 'ROOT', tmp_path)
    monkeypatch.setattr(candidate, 'assert_no_live_workers', lambda spec: None)
    monkeypatch.setattr(candidate, 'adoption_evidence', lambda *a: {'actual_full512': 'verified external fixture'})
    manifest = tmp_path / 'manifest.jsonl'; jsonl(manifest, records())
    run = tmp_path / 'runs/research/sft'
    job = {'id': 'sft', 'argv': ['--stage', 'SFT', '--recipe', '4q', '--selection-metric', 'group_macro', '--output', str(run)],
           'evaluation': {'manifest': str(manifest), 'output': str(run / 'evaluation/heldout_v3'), 'checkpoint': 'best', 'max_new_tokens': 64}}
    spec = tmp_path / 'queue.json'; spec.write_text(json.dumps({'reserve_gb': 0, 'jobs': [job]}))
    calls = []
    failures = {}
    child_info = {'value': None}
    original_process_info = candidate.parallel.proc.process_info
    def process_info(pid):
        if pid == 987654321:
            return child_info['value']
        return original_process_info(pid)
    monkeypatch.setattr(candidate.parallel.proc, 'process_info', process_info)
    def partial_check(self, job, variant, output):
        calls.append(('partial_preflight', variant))
        return {'records_committed': 1}
    monkeypatch.setattr(candidate.ParallelGroundingResearchQueue, '_validate_partial_journal', partial_check)
    def artifacts():
        manager = BestLastCheckpointManager(run, DiskBudget(run, 0))
        if not (run / 'last/state.pt').exists():
            manager.save(lambda d: (d / 'state.pt').write_bytes(b'owned-fixture'), step=1, metric=.5, anticipated_bytes=128)
        (run / 'completed.json').write_text(json.dumps({'steps': 1}))
        (run / 'resolved_config.json').write_text(json.dumps({'stage': 'SFT', 'recipe': '4q', 'selection_metric': 'group_macro',
            'model': 'Tiiny/SmallThinker-4BA0.6B-Instruct', 'vision': 'google/siglip2-so400m-patch14-384'}))
    def output(variant, workers=4):
        return candidate.parallel.output_path(run, variant, workers=workers)
    def write_view(variant, workers=4):
        destination = output(variant, workers)
        jsonl(destination / 'predictions.jsonl', generated(records()))
        (destination / 'summary.json').write_text(json.dumps({'group_macro_score': 100., 'records': 512}))
        if not (destination / 'protocol.json').exists():
            (destination / 'protocol.json').write_text(json.dumps({'owner': candidate.parallel.OWNER, 'variant': variant}))
        (destination / 'execution_summary.json').write_text(json.dumps({'status': 'completed'}))
    def auditor(run_path, manifest_path, variant='none', workers=4, native_cache=None):
        assert Path(run_path) == run and Path(manifest_path) == manifest
        destination = output(variant, workers)
        rows = native.read_rows(destination / 'predictions.jsonl')
        assert len(rows) == len({row['record_id'] for row in rows}) == 512
        if failures.get('audit') == variant:
            raise ValueError('actual strict raw512 auditor refused corruption')
        from src.utils.research_checkpoint_compat import checkpoint_identity
        return {'owner': candidate.parallel.OWNER, 'status': 'passed', 'records': 512, 'worker_count': workers,
                'checkpoint_identity': checkpoint_identity(run / 'best/state.pt'), 'protocol_sha256': 'b' * 64,
                'raw_predictions_sha256': candidate.sha(destination / 'predictions.jsonl'),
                'summary_sha256': candidate.sha(destination / 'summary.json'), 'group_macro_score': 100.,
                'per_model_native64_guard_verified': True, 'full512_exact_previous_native_comparison': True}
    monkeypatch.setattr(candidate.parallel, 'validate_completed_output', auditor)
    def checkpoint_gate(self, j, record):
        calls.append('gpu_replay_best_last')
        assert record['evaluation']['status'] == 'completed'
        if failures.get('checkpoint'):
            record['failure_phase'] = 'checkpoint_validation'
            record['checkpoint_validation'] = {'status': 'failed'}
            raise RuntimeError('both retained checkpoint replay failed')
        record['checkpoint_validation'] = {'status': 'completed', 'mode': 'full_gpu_replay', 'retained_checkpoint_objects': 2}
        return True
    monkeypatch.setattr(candidate.ParallelGroundingResearchQueue, '_validate_completed_checkpoints', checkpoint_gate)
    def process(command, **kwargs):
        if Path(command[2]).name == 'run_research_stage.py':
            assert 'env' not in kwargs, 'training environment is unchanged'
            calls.append('trainer'); artifacts()
        else:
            assert Path(command[2]).name == Path(candidate.SCRIPT).name
            assert '--exec-evaluator' in command and 'preexec_fn' not in kwargs
            assert all(kwargs['env'].get(name)==value for name,value in candidate.EVALUATION_ENV.items())
            launch_path = Path(base.argv_value(command, '--exec-evaluator')); launch = candidate.read(launch_path)
            inner = launch['command']; variant = base.argv_value(inner, '--image-ablation')
            child_info['value'] = {'pid': 987654321, 'ppid': launch['parent']['pid'], 'start_time': 1234,
                                   'argv': inner, 'command_sha256': candidate.digest(inner), 'state': 'R'}
            launch_path.with_suffix('.registration.json').write_text(json.dumps({'owner': candidate.OWNER,
                'nonce': launch['nonce'], 'launch_sha256': candidate.sha(launch_path),
                'command_sha256': candidate.digest(inner), 'identity': child_info['value']}))
            calls.append((variant, '--resume' in inner))
            if failures.get('worker') == variant:
                child_info['value'] = None
                return SimpleNamespace(pid=987654321, returncode=9, poll=lambda: 9)
            write_view(variant, int(base.argv_value(inner, '--workers')))
            polls = [0]
            def poll():
                polls[0] += 1
                if polls[0] == 1: return None
                child_info['value'] = None
                return 0
            return SimpleNamespace(pid=987654321, returncode=0, poll=poll)
        return SimpleNamespace(pid=987654321, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(candidate.subprocess, 'Popen', process)
    def queue(**kwargs):
        q = candidate.ParallelGroundingResearchQueue(spec, proof_run=tmp_path / 'proof', **kwargs)
        monkeypatch.setattr(q.budget, 'ensure_space', lambda n: None)
        return q
    def state():
        return candidate.read(spec.with_suffix('.state.json'))
    return SimpleNamespace(run=run, job=job, spec=spec, manifest=manifest, calls=calls, failures=failures,
                           artifacts=artifacts, output=output, write_view=write_view, auditor=auditor, queue=queue, state=state)


def test_same_spec_train_once_explicit_parallel_schema_replay_before_ablations(setup):
    q = setup; original = q.spec.read_bytes()
    assert q.queue().run() == 0
    assert q.calls == ['trainer', ('none', False), 'gpu_replay_best_last', ('shuffle', False), ('blank', False), ('text_only', False)]
    state = q.state(); row = state['jobs']['sft']
    assert state['status'] == row['status'] == 'completed'
    assert row['training_completed'] is True and row['parallel_status'] == 'completed'
    assert row['evaluation']['execution_schema'] == 'parallel_native_b1_v1'
    assert all(row['grounding_evaluations'][v]['execution_script'] == candidate.parallel.SCRIPT for v in native.MODES)
    assert q.spec.read_bytes() == original
    paired = candidate.read(row['parallel_grounding_summary'])
    assert paired['records'] == 512 and paired['checkpoint_validation']['retained_checkpoint_objects'] == 2
    assert all(result['paired_95_percentile_ci'] == [0, 0] for result in paired['paired_image_group_bootstrap'].values())
    before = list(q.calls)
    assert q.queue().run() == 0
    assert q.calls == before + ['gpu_replay_best_last']


def test_failed_partial_view_requires_explicit_retry_never_retrains(setup):
    q = setup; q.failures['worker'] = 'blank'
    assert q.queue().run() == 1
    row = q.state()['jobs']['sft']
    assert row['status'] == 'training_completed_parallel_eval_failed' and row['training_completed'] is True
    before = list(q.calls)
    assert q.queue().run() == 1 and q.calls == before
    partial = q.output('blank'); partial.mkdir(exist_ok=True)
    (partial / 'protocol.json').write_text('{"immutable":"partial protocol"}')
    jsonl(partial / 'prediction_records/row-0000.json', [generated(records())[0]])
    previous = (partial / 'protocol.json').read_bytes()
    q.failures.clear()
    assert q.queue(retry_failed=True).run() == 0
    assert sum(call == 'trainer' for call in q.calls) == 1
    assert ('blank', True) in q.calls[len(before):]
    assert ('none', False) not in q.calls[len(before):] and ('shuffle', False) not in q.calls[len(before):]
    # The fake worker stands in for the separately tested immutable-journal CLI;
    # the controller itself neither removes nor repairs the existing protocol.
    assert (partial / 'protocol.json').read_bytes() == previous


def test_final_output_after_parent_crash_reaudited_without_generation(setup):
    q = setup; q.artifacts()
    controller = q.queue()
    controller.state['jobs']['sft'] = {'status': 'interrupted', 'training_completed': True}
    for variant in candidate.ALL_MODES:
        q.write_view(variant)
    assert controller.run() == 0
    assert q.calls == ['gpu_replay_best_last']


def test_corrupt_completed_parallel_output_refused_without_relaunch(setup):
    q = setup; assert q.queue().run() == 0
    before = list(q.calls); q.failures['audit'] = 'shuffle'
    assert q.queue().run() == 1
    assert q.calls == before + ['gpu_replay_best_last']
    assert q.state()['jobs']['sft']['status'] == 'training_completed_parallel_eval_failed'


def test_gpu_replay_failure_prevents_any_grounding_variant(setup):
    q = setup; q.failures['checkpoint'] = True
    assert q.queue().run() == 1
    assert q.calls == ['trainer', ('none', False), 'gpu_replay_best_last']
    assert q.state()['jobs']['sft']['status'] == 'training_completed_checkpoint_validation_failed'


def test_completed_native_views_and_original_receipts_preserved(setup, monkeypatch):
    q = setup; q.artifacts(); controller = q.queue()
    original_receipts = {'none': {'status': 'completed', 'output': str(q.run / 'evaluation/heldout_v3'), 'original': 'keep byte-equivalent JSON'},
                         'shuffle': {'status': 'completed', 'output': str(q.run / 'evaluation/grounding_v1/shuffle'), 'native': True}}
    for variant, receipt in original_receipts.items():
        jsonl(Path(receipt['output']) / 'predictions.jsonl', generated(records()))
    record = {'status': 'grounding_evaluating', 'training_completed': True,
              'evaluation': copy.deepcopy(original_receipts['none']),
              'grounding_evaluations': {'shuffle': copy.deepcopy(original_receipts['shuffle'])}}
    controller.state['jobs']['sft'] = record
    audits = []
    def audit(job, variant, receipt):
        audits.append(variant)
        assert receipt == original_receipts[variant]
        from src.utils.research_checkpoint_compat import checkpoint_identity
        artifact = checkpoint_identity(q.run / 'best/state.pt')
        return {'status': 'completed', 'execution_schema': 'native_serial_preserved', 'output': receipt['output'],
                'checkpoint_sha256': artifact['sha256'], 'records': 512, 'variant': variant}
    monkeypatch.setattr(controller, '_audit_native', audit)
    assert controller.run() == 0
    assert audits == ['none', 'shuffle']
    assert q.calls == ['gpu_replay_best_last', ('blank', False), ('text_only', False)]
    saved = q.state()['jobs']['sft']
    assert saved['evaluation'] == original_receipts['none']
    assert saved['grounding_evaluations']['shuffle'] == original_receipts['shuffle']
    assert candidate.read(saved['parallel_grounding_summary'])['views']['none']['execution_schema'] == 'native_serial_preserved'


def test_full_training_branch_delegates_to_original_serial_controller(setup, monkeypatch):
    q = setup; q.artifacts()
    spec = candidate.read(q.spec); spec['jobs'][0]['argv'][3] = '4f'; q.spec.write_text(json.dumps(spec))
    seen = []
    def native_eval(self, job, record):
        seen.append(job['id']); record['legacy_native_called'] = True; return True
    monkeypatch.setattr(native.GroundingResearchQueue, 'evaluate_completed_job', native_eval)
    controller = q.queue(); controller.state['jobs']['sft'] = {'status': 'interrupted', 'training_completed': True}
    assert controller.run() == 0
    assert seen == ['sft'] and q.calls == []


def test_partial_orphan_directory_without_protocol_never_repaired(setup):
    q = setup; q.artifacts(); path = q.output('none'); path.mkdir(parents=True)
    (path / 'unknown').write_text('user file')
    controller = q.queue(); controller.state['jobs']['sft'] = {'status': 'interrupted', 'training_completed': True}
    assert controller.run() == 1
    assert q.calls == [] and (path / 'unknown').read_text() == 'user file'


def test_signal_before_spawn_stops_future_work_and_preserves_completed_training(setup, monkeypatch):
    q = setup; q.artifacts(); controller = q.queue()
    controller.state['jobs']['sft'] = {'status': 'interrupted', 'training_completed': True}
    original = controller._launch_parallel
    def interrupt(*args):
        controller.interrupted_by = signal.SIGTERM
        return original(*args)
    monkeypatch.setattr(controller, '_launch_parallel', interrupt)
    assert controller.run() == 128 + signal.SIGTERM
    assert q.calls == [] and q.state()['jobs']['sft']['status'] == 'training_completed_parallel_eval_interrupted'


def test_actual_gpu_proof_gate_failure_releases_lock_and_does_not_mutate_state(setup, monkeypatch):
    q = setup
    monkeypatch.setattr(candidate, 'adoption_evidence', lambda *a: (_ for _ in ()).throw(ValueError('No actual full512 proof')))
    with pytest.raises(ValueError, match='actual full512'):
        q.queue()
    assert not q.spec.with_suffix('.state.json').exists()
    # A subsequent owner can acquire the same state lock; readiness failure
    # must not strand the existing controller's intended replacement.
    monkeypatch.setattr(candidate, 'adoption_evidence', lambda *a: {'actual_full512': True})
    controller = q.queue(); controller.lock_handle.close()


def test_worker_count_or_proof_change_refused_on_restart(setup, monkeypatch):
    q = setup; controller = q.queue(); controller.persist(); controller.lock_handle.close()
    with pytest.raises(ValueError, match='identity changed'):
        q.queue(workers=2)


@pytest.mark.parametrize('state', ['S', 'R'])
def test_live_or_pid_reused_previous_worker_never_signaled_or_duplicated(setup, monkeypatch, state):
    controller = setup.queue()
    previous = {'pid': 33, 'process_identity': {'pid': 33, 'start_time': 100}}
    monkeypatch.setattr(candidate.parallel.proc, 'process_info', lambda pid: {'pid': pid, 'start_time': 101, 'state': state, 'argv': ['foreign']})
    with pytest.raises(RuntimeError, match='reused'):
        controller._assert_no_orphan(previous)
    assert setup.calls == []
    controller.lock_handle.close()


def test_dead_previous_worker_allows_strict_journal_resume(setup, monkeypatch):
    controller = setup.queue()
    monkeypatch.setattr(candidate.parallel.proc, 'process_info', lambda pid: None)
    controller._assert_no_orphan({'pid': 33, 'process_identity': {'pid': 33, 'start_time': 100}})
    controller.lock_handle.close()


@pytest.mark.parametrize('script,flag', [('run_research_stage.py', '--output'), ('eval_research_checkpoint.py', '--run'),
                                      ('validate_research_checkpoint.py', '--run'), ('eval_research_checkpoint_parallel_b1.py', '--barrier')])
def test_live_worker_scanner_only_recognizes_owned_run_commands(tmp_path, script, flag):
    run = tmp_path / 'run'
    value = str(run / 'evaluation/attempt/barrier.json') if flag == '--barrier' else str(run)
    info = {'state': 'S', 'argv': [sys.executable, '-u', '/repo/scripts/' + script, flag, value]}
    assert candidate.worker_matches(info, [run])
    assert not candidate.worker_matches(info, [tmp_path / 'foreign'])
    assert not candidate.worker_matches(dict(info, state='Z'), [run])
    assert not candidate.worker_matches({'state': 'S', 'argv': ['unrelated', '--run', str(run)]}, [run])


def test_bootstrap_refuses_incomplete_or_duplicate_actual_rows(setup):
    q = setup; controller = q.queue(); q.artifacts()
    record = {'checkpoint_validation': {'status': 'completed'}}
    views = {}
    for variant in candidate.ALL_MODES:
        q.write_view(variant); views[variant] = {'output': str(q.output(variant))}
    jsonl(q.output('blank') / 'predictions.jsonl', generated(records())[:511])
    with pytest.raises(ValueError, match='unique512'):
        controller._write_paired(q.job, record, views)
    controller.lock_handle.close()


def test_public_actual_proof_gate_checks_native_tokens_not_cpu_readiness(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, 'readiness', lambda: {'cpu_passed': True})
    run = tmp_path / 'q4'; run.mkdir()
    config = {'recipe': '4q', 'stage': 'SFT', 'model': 'Tiiny/SmallThinker-4BA0.6B-Instruct',
              'vision': 'google/siglip2-so400m-patch14-384'}
    (run / 'resolved_config.json').write_text(json.dumps(config))
    manifest = tmp_path / 'eval'; manifest.write_text('immutable')
    result = {'owner': candidate.parallel.OWNER, 'status': 'passed', 'records': 512, 'worker_count': 4,
              'full512_exact_previous_native_comparison': False, 'per_model_native64_guard_verified': True}
    monkeypatch.setattr(candidate.parallel, 'validate_completed_output', lambda *a, **kw: result)
    with pytest.raises(ValueError, match='Actual full512'):
        candidate.adoption_evidence(run, manifest, 4)
    result['full512_exact_previous_native_comparison'] = True
    assert candidate.adoption_evidence(run, manifest, 4)['raw_actual_full512_audit'] == result
    result['records'] = 64
    with pytest.raises(ValueError, match='Actual full512'):
        candidate.adoption_evidence(run, manifest, 4)


def test_gateway_refuses_changed_parent_before_arming_or_exec(tmp_path, monkeypatch):
    launch = tmp_path / 'launch.json'
    launch.write_text(json.dumps({'owner': candidate.OWNER, 'parent': {'pid': 1, 'start_time': 1}, 'boot_id': 'wrong'}))
    monkeypatch.setattr(candidate.parallel.proc, 'arm_parent_death', lambda p: pytest.fail('must refuse before arming'))
    monkeypatch.setattr(candidate.os, 'execv', lambda *a: pytest.fail('must never execute'))
    with pytest.raises(ValueError, match='parent/source'):
        candidate.exec_evaluator(launch)


def test_native_holder_cannot_bypass_strict_actual_model_provenance_audit(setup, monkeypatch):
    q = setup; q.artifacts(); controller = q.queue()
    from src.utils.research_checkpoint_compat import checkpoint_identity
    artifact = checkpoint_identity(q.run / 'best/state.pt')
    identity = {'manifest': q.manifest, 'output': q.run / 'evaluation/heldout_v3',
                'checkpoint_artifact_identity': artifact}
    monkeypatch.setattr(controller, '_native_identity', lambda *a: identity)
    monkeypatch.setattr(controller, '_validate_evaluation', lambda *a: {'checkpoint_sha256': artifact['sha256'],
        'manifest_sha256': 'a' * 64, 'protocol_sha256': 'b' * 64, 'group_macro_score': 100.})
    monkeypatch.setattr(candidate.parallel, 'prepare_context', lambda *a: {'records': records()})
    monkeypatch.setattr(candidate.parallel, 'protocol_for', lambda *a: {'profile': 'exact native'})
    def refuse(*a, **kw):
        raise ValueError('Actual native model provenance differs')
    monkeypatch.setattr(candidate.parallel, 'module', lambda: SimpleNamespace(validate_native_cache=refuse))
    receipt = {'status': 'completed', 'original': 'native holder'}
    with pytest.raises(ValueError, match='native model provenance'):
        controller._audit_native(q.job, 'none', receipt)
    assert receipt == {'status': 'completed', 'original': 'native holder'} and q.calls == []
    controller.lock_handle.close()


def test_gateway_registers_fresh_owned_pid_before_exec_without_fork_callbacks(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, 'CODE_ROOT', tmp_path)
    script = tmp_path / candidate.SCRIPT; script.parent.mkdir(); script.write_text('frozen controller source')
    run = tmp_path / 'runs/research/sft'; run.mkdir(parents=True)
    parent = {'pid': 77, 'start_time': 100, 'argv': ['owned-controller'], 'command_sha256': 'a' * 64, 'state': 'S'}
    own = {'pid': 88, 'start_time': 200, 'ppid': 77, 'argv': ['fresh-gateway'], 'command_sha256': 'b' * 64, 'state': 'R'}
    command = [sys.executable, '-u', str(tmp_path / candidate.parallel.SCRIPT), '--run', str(run),
               '--image-ablation', 'none', '--workers', '4']
    output = candidate.parallel.output_path(run, 'none', workers=4)
    directory = output.parent / 'controller_launches'; directory.mkdir(parents=True)
    launch = directory / 'one.json'
    launch.write_text(json.dumps({'owner': candidate.OWNER, 'nonce': 'fresh-owned-nonce',
        'parent': parent, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'script_sha256': candidate.sha(script), 'command': command, 'command_sha256': candidate.digest(command),
        'execution_environment': candidate.EVALUATION_ENV}))
    for name,value in candidate.EVALUATION_ENV.items():monkeypatch.setenv(name,value)
    monkeypatch.setattr(candidate.os, 'getppid', lambda: 77)
    monkeypatch.setattr(candidate.os, 'getpid', lambda: 88)
    monkeypatch.setattr(candidate.parallel.proc, 'process_info', lambda pid: parent if pid == 77 else own)
    armed = []
    monkeypatch.setattr(candidate.parallel.proc, 'arm_parent_death', lambda pid: armed.append(pid))
    def execute(binary, argv):
        registration = candidate.read(launch.with_suffix('.registration.json'))
        assert registration['identity'] == own and registration['nonce'] == 'fresh-owned-nonce'
        assert registration['launch_sha256'] == candidate.sha(launch)
        assert binary == command[0] and argv == command and armed == [77]
        raise SystemExit(0)
    monkeypatch.setattr(candidate.os, 'execv', execute)
    with pytest.raises(SystemExit) as result:
        candidate.exec_evaluator(launch)
    assert result.value.code == 0


def test_source_proof_rejects_same_size_changes_and_path_escape(tmp_path):
    source = tmp_path / 'owned.py'; source.write_text('version A')
    sources = {'owned.py': candidate.sha(source)}
    candidate.verify_sources(tmp_path, sources, ('owned.py',))
    source.write_text('version B')
    with pytest.raises(ValueError, match='source changed'):
        candidate.verify_sources(tmp_path, sources, ('owned.py',))
    with pytest.raises(ValueError, match='incomplete'):
        candidate.verify_sources(tmp_path, {}, ('owned.py',))
    outside = tmp_path.parent / (tmp_path.name + '-outside.py'); outside.write_text('external')
    with pytest.raises(ValueError, match='source changed'):
        candidate.verify_sources(tmp_path, {'../' + outside.name: candidate.sha(outside)}, ())


@pytest.mark.parametrize('field,bad', [('status', 'HOLD'), ('tests_passed', 0), ('test_command', ''),
                                    ('checks', {'actual_full512_4q_token_proof_required_before_start': False})])
def test_unpassed_or_incomplete_readiness_is_never_adoption(tmp_path, field, bad):
    path = tmp_path / candidate.READY; path.parent.mkdir(parents=True)
    report = {'schema_version': 1, 'status': 'passed', 'tests_passed': 1, 'test_command': 'actual pytest',
              'checks': {key: True for key in candidate.CHECKS}, 'source_sha256': {}}
    report[field] = bad; path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='readiness proof'):
        candidate.readiness(tmp_path)


@pytest.mark.parametrize('corrupt', ['source', 'workers', 'variant', 'checkpoint'])
def test_partial_protocol_corruption_refused_before_generation(tmp_path, monkeypatch, corrupt):
    run = tmp_path / 'run'; output = run / 'evaluation/parallel'; output.mkdir(parents=True)
    reference = tmp_path / 'native'; reference.mkdir()
    binding = {'directory': str(reference), 'scope': 'full512'}
    expected = {'reference_binding': binding, 'source': 'fixed', 'workers': 4, 'variant': 'blank', 'checkpoint': 'fixed'}
    changed = copy.deepcopy(expected); changed[corrupt] = 'changed'
    (output / 'protocol.json').write_text(json.dumps(changed))
    context = {'records': records()}
    monkeypatch.setattr(candidate.parallel, 'prepare_context', lambda *a: context)
    monkeypatch.setattr(candidate.parallel, 'protocol_for', lambda *a, **kw: expected)
    helper = SimpleNamespace(validate_native_cache=lambda *a, **kw: {'binding': binding, 'scope': 'full512', 'rows': generated(records())},
        read_committed=lambda *a: pytest.fail('corrupt protocol must fail before reading/generating rows'))
    monkeypatch.setattr(candidate.parallel, 'module', lambda: helper)
    controller = object.__new__(candidate.ParallelGroundingResearchQueue)
    controller.workers = 4; controller.spec = {'reserve_gb': 120}
    job = {'argv': ['--output', str(run)], 'evaluation': {'manifest': str(tmp_path / 'manifest')}}
    with pytest.raises(ValueError, match='protocol differs'):
        controller._validate_partial_journal(job, 'blank', output)


@pytest.mark.parametrize('scope,terminal,committed_count,expected_error', [
    ('full512', 'failed', 1, None), ('guard64', 'failed', 1, 'cannot certify'),
    ('guard64', 'failed', 0, None), ('full512', 'running', 1, 'still running/unverified')])
def test_partial_resume_retains_strict_execution_proof_limits(tmp_path, monkeypatch, scope, terminal, committed_count, expected_error):
    run = tmp_path / 'run'; output = run / 'evaluation/parallel'; output.mkdir(parents=True)
    reference = tmp_path / 'native'; reference.mkdir()
    binding = {'directory': str(reference), 'scope': scope}
    protocol = {'reference_binding': binding}
    (output / 'protocol.json').write_text(json.dumps(protocol))
    attempt = output / 'execution/old'; attempt.mkdir(parents=True)
    (attempt / 'receipt.json').write_text(json.dumps({'status': terminal}))
    selected = records(); context = {'records': selected}
    monkeypatch.setattr(candidate.parallel, 'prepare_context', lambda *a: context)
    monkeypatch.setattr(candidate.parallel, 'protocol_for', lambda *a, **kw: protocol)
    helper = SimpleNamespace(validate_native_cache=lambda *a, **kw: {'binding': binding, 'scope': scope, 'rows': generated(selected)},
        read_committed=lambda *a: {i: {} for i in range(committed_count)}, select_guard_indices=lambda rows: list(range(64)),
        protocol_sha256=lambda p: 'a' * 64)
    monkeypatch.setattr(candidate.parallel, 'module', lambda: helper)
    monkeypatch.setattr(candidate.parallel, 'validated_parallel_guard', lambda *a: {'verified': True})
    controller = object.__new__(candidate.ParallelGroundingResearchQueue)
    controller.workers = 4; controller.spec = {'reserve_gb': 120}
    job = {'argv': ['--output', str(run)], 'evaluation': {'manifest': str(tmp_path / 'manifest')}}
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            controller._validate_partial_journal(job, 'none', output)
    else:
        assert controller._validate_partial_journal(job, 'none', output)['records_committed'] == committed_count


def test_live_controller_source_change_refused_before_new_evaluator(setup, monkeypatch):
    q = setup; q.artifacts(); controller = q.queue()
    original_sha = candidate.sha
    monkeypatch.setattr(candidate, 'sha', lambda path: 'f' * 64 if Path(path) == candidate.CODE_ROOT / candidate.SCRIPT else original_sha(path))
    with pytest.raises(ValueError, match='execution source changed'):
        controller._launch_parallel(q.job, {'training_completed': True}, 'none', {})
    assert q.calls == []
    controller.lock_handle.close()


@pytest.mark.parametrize('corrupt', ['start_time', 'ppid', 'argv', 'command_sha256', 'pid'])
def test_spawned_evaluator_identity_reuse_or_command_changes_refused(corrupt):
    parent = {'pid': 10}; gateway = ['python', 'owned-gateway']; command = ['python', 'owned-evaluator']
    info = {'pid': 11, 'ppid': 10, 'start_time': 55, 'argv': gateway,
            'command_sha256': candidate.digest(gateway), 'state': 'R'}
    assert candidate.verify_spawned_evaluator(info, parent, 11, gateway, command, 55) == info
    native_info = dict(info, argv=command, command_sha256=candidate.digest(command))
    assert candidate.verify_spawned_evaluator(native_info, parent, 11, gateway, command, 55) == native_info
    bad = copy.deepcopy(info); bad[corrupt] = ['foreign'] if corrupt == 'argv' else 'wrong' if corrupt == 'command_sha256' else 99
    with pytest.raises(RuntimeError, match='binding changed'):
        candidate.verify_spawned_evaluator(bad, parent, 11, gateway, command, 55)


def startup_fixture(tmp_path):
    parent = {'pid': 10}; gateway = ['python', 'gateway']; command = ['python', 'final']
    info = {'pid': 11, 'ppid': 10, 'start_time': 55, 'argv': command,
            'command_sha256': candidate.digest(command), 'state': 'R'}
    launch = tmp_path/'launch.json'; declaration = {'parent': parent, 'nonce': 'unique-nonce',
        'command': command, 'command_sha256': candidate.digest(command)}
    launch.write_text(json.dumps(declaration))
    registration = {'owner': candidate.OWNER, 'nonce': declaration['nonce'], 'launch_sha256': candidate.sha(launch),
                    'command_sha256': candidate.digest(command), 'identity': info}
    return parent, gateway, command, info, launch, declaration, registration


def test_startup_allows_missing_empty_gateway_then_requires_final_nonce(tmp_path, monkeypatch):
    parent, gateway, command, info, launch, declaration, registration = startup_fixture(tmp_path)
    launch.with_suffix('.registration.json').write_text(json.dumps(registration))
    reads = iter([None, dict(info, argv=[]), dict(info, argv=gateway, command_sha256=candidate.digest(gateway)), info])
    monkeypatch.setattr(candidate.parallel.proc, 'process_info', lambda pid: next(reads, info))
    monkeypatch.setattr(candidate, 'evaluator_birth', lambda pid: dict(info, argv=None))
    monkeypatch.setattr(candidate.time, 'sleep', lambda seconds: None)
    receipt = {}; persisted = []
    candidate.wait_evaluator_startup(SimpleNamespace(pid=11, poll=lambda: None), receipt, parent, gateway,
        command, launch, declaration, persist=lambda: persisted.append(copy.deepcopy(receipt)))
    assert receipt['startup_verified'] is True and receipt['process_identity'] == info
    assert [row['argv_state'] for row in receipt['startup_observations']] == ['unavailable', 'empty', 'gateway', 'final']
    assert persisted[0]['pid'] == 11 and persisted[0]['process_identity'] == {}


@pytest.mark.parametrize('bad', ['missing_nonce', 'empty_forever', 'missing_forever'])
def test_startup_grace_is_bounded15s_and_never_accepts_missing_nonce(tmp_path, monkeypatch, bad):
    parent, gateway, command, info, launch, declaration, registration = startup_fixture(tmp_path)
    clock = [0.]
    monkeypatch.setattr(candidate.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(candidate.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    live = info if bad == 'missing_nonce' else dict(info, argv=[]) if bad == 'empty_forever' else None
    monkeypatch.setattr(candidate.parallel.proc, 'process_info', lambda pid: live)
    monkeypatch.setattr(candidate, 'evaluator_birth', lambda pid: dict(info, argv=None))
    receipt = {}
    with pytest.raises(TimeoutError, match='within15s'):
        candidate.wait_evaluator_startup(SimpleNamespace(pid=11, poll=lambda: None), receipt, parent, gateway, command, launch, declaration)
    assert 15 <= clock[0] < 15.1
    assert receipt['process_identity']['start_time'] == 55 and len(receipt['startup_observations']) == 64


@pytest.mark.parametrize('mutation', ['foreign_argv', 'ppid', 'birth_empty', 'nonce'])
def test_startup_does_not_grace_foreign_argv_reuse_or_wrong_nonce(tmp_path, monkeypatch, mutation):
    parent, gateway, command, info, launch, declaration, registration = startup_fixture(tmp_path)
    receipt = {'process_identity': {'start_time': 55}}
    live = copy.deepcopy(info)
    if mutation == 'foreign_argv': live['argv'] = ['foreign-nonempty']
    if mutation == 'ppid': live['ppid'] = 99
    if mutation == 'birth_empty': live.update(start_time=56, argv=[])
    if mutation == 'nonce': registration['nonce'] = 'old-nonce'
    launch.with_suffix('.registration.json').write_text(json.dumps(registration))
    monkeypatch.setattr(candidate.parallel.proc, 'process_info', lambda pid: live)
    monkeypatch.setattr(candidate.time, 'sleep', lambda seconds: pytest.fail('foreign ownership must fail immediately'))
    if mutation == 'nonce':
        receipt['pid'] = 11
        with pytest.raises(ValueError, match='handshake differs'):
            candidate.evaluator_handshake(launch, declaration, receipt)
    else:
        with pytest.raises(RuntimeError, match='foreign nonempty|birth/PPID'):
            candidate.observe_evaluator(SimpleNamespace(pid=11, poll=lambda: None), receipt, parent, gateway, command, startup=True)


def test_after_startup_empty_argv_has_no_grace(tmp_path, monkeypatch):
    parent, gateway, command, info, launch, declaration, registration = startup_fixture(tmp_path)
    monkeypatch.setattr(candidate.parallel.proc, 'process_info', lambda pid: dict(info, argv=[]))
    with pytest.raises(RuntimeError, match='unavailable after verified startup'):
        candidate.observe_evaluator(SimpleNamespace(pid=11, poll=lambda: None), {'process_identity': info}, parent, gateway, command, startup=False)


def real_gateway_files(tmp_path, sleep_seconds=1.):
    """Only this test tree substitutes a CPU destination; no production override."""
    repo = candidate.CODE_ROOT
    script = tmp_path/candidate.SCRIPT; script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(repo))+')\n'
        'from scripts import research_parallel_grounding_queue as c\n'
        'c.CODE_ROOT=Path.cwd()\n'
        "c.exec_evaluator(Path(sys.argv[sys.argv.index('--exec-evaluator')+1]))\n")
    destination = tmp_path/candidate.parallel.SCRIPT; marker = tmp_path/'cpu-final.json'
    destination.write_text('import ctypes,json,os,sys,time\nfrom pathlib import Path\n'
        "assert 'torch' not in sys.modules\n"
        'value=ctypes.c_int();lib=ctypes.CDLL(None);assert lib.prctl(2,ctypes.byref(value),0,0,0)==0\n'
        'Path('+repr(str(marker))+').write_text(json.dumps({"pid":os.getpid(),"ppid":os.getppid(),"pdeath":value.value}))\n'
        'time.sleep('+repr(sleep_seconds)+')\n')
    run = tmp_path/'runs/research/cpu'; run.mkdir(parents=True, exist_ok=True)
    output = candidate.parallel.output_path(run, 'none', workers=4)
    launch = output.parent/'controller_launches/launch.json'; launch.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(destination), '--run', str(run), '--image-ablation', 'none', '--workers', '4']
    gateway = [sys.executable, '-u', str(script), '--exec-evaluator', str(launch)]
    return repo, marker, command, gateway, launch, script


def test_real_fresh_gateway_exec_nonce_and_transient_proc_reads(tmp_path, monkeypatch):
    repo, marker, command, gateway, launch, script = real_gateway_files(tmp_path, .5)
    parent = candidate.parallel.proc.process_info(os.getpid())
    declaration = {'owner': candidate.OWNER, 'nonce': 'real-cpu-nonce', 'parent': parent,
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'script_sha256': candidate.sha(script),
        'command': command, 'command_sha256': candidate.digest(command), 'execution_environment': candidate.EVALUATION_ENV}
    launch.write_text(json.dumps(declaration))
    process = subprocess.Popen(gateway, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env={**os.environ,**candidate.EVALUATION_ENV})
    original = candidate.parallel.proc.process_info; reads = [0]
    def info(pid):
        value = original(pid)
        if pid == process.pid and value is not None:
            reads[0] += 1
            if reads[0] == 1: return None
            if reads[0] == 2: return dict(value, argv=[])
        return value
    monkeypatch.setattr(candidate.parallel.proc, 'process_info', info)
    receipt = {}
    try:
        candidate.wait_evaluator_startup(process, receipt, parent, gateway, command, launch, declaration)
        deadline = time.monotonic()+3
        while not marker.exists() and time.monotonic() < deadline: time.sleep(.01)
        assert marker.is_file()
        actual = candidate.read(marker)
        assert actual['pid'] == process.pid and actual['ppid'] == os.getpid() and actual['pdeath'] == signal.SIGKILL
        assert receipt['startup_verified'] and candidate.evaluator_handshake(launch, declaration, receipt)
        assert ({row['argv_state'] for row in receipt['startup_observations']} |
                {receipt['last_process_observation']['argv_state']}) >= {'unavailable', 'empty', 'final'}
        process.wait(timeout=3); assert process.returncode == 0
    finally:
        if process.poll() is None: process.kill(); process.wait(timeout=3)


def test_real_parent_death_kills_the_same_gateway_pid_after_exec(tmp_path):
    repo, marker, command, gateway, launch, script = real_gateway_files(tmp_path, 60)
    controller = tmp_path/'cpu-controller.py'; verified = tmp_path/'startup-verified.json'
    controller.write_text('import json,os,sys,subprocess,time\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(repo))+')\n'
        'from scripts import research_parallel_grounding_queue as c\n'
        'parent=c.parallel.proc.process_info(os.getpid())\n'
        'command='+repr(command)+'\ngateway='+repr(gateway)+'\nlaunch=Path('+repr(str(launch))+')\n'
        'declaration={"owner":c.OWNER,"nonce":"death-cpu-nonce","parent":parent,'
        '"boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),'
        '"script_sha256":c.sha(Path('+repr(str(script))+')),"command":command,"command_sha256":c.digest(command),'
        '"execution_environment":c.EVALUATION_ENV}\n'
        'launch.write_text(json.dumps(declaration))\n'
        'process=subprocess.Popen(gateway,cwd='+repr(str(tmp_path))+',env={**os.environ,**c.EVALUATION_ENV})\nreceipt={}\n'
        'c.wait_evaluator_startup(process,receipt,parent,gateway,command,launch,declaration)\n'
        'Path('+repr(str(verified))+').write_text(json.dumps(receipt))\ntime.sleep(60)\n')
    process = subprocess.Popen([sys.executable, '-u', str(controller)], cwd=repo,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    child = None
    try:
        deadline = time.monotonic()+30
        while not verified.exists() and process.poll() is None and time.monotonic() < deadline: time.sleep(.025)
        assert verified.is_file(), process.communicate(timeout=1)
        deadline = time.monotonic()+2
        while not marker.exists() and time.monotonic() < deadline: time.sleep(.01)
        actual = candidate.read(marker); child = candidate.parallel.proc.process_info(actual['pid'])
        assert child is not None and actual['pdeath'] == signal.SIGKILL and child['ppid'] == process.pid
        process.terminate(); process.wait(timeout=3)
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            live = candidate.parallel.proc.process_info(child['pid'])
            if live is None or live.get('state') in ('Z', 'X'): break
            time.sleep(.025)
        else: pytest.fail('kernel parent-death signal did not stop owned CPU fixture child')
    finally:
        if process.poll() is None: process.kill(); process.wait(timeout=3)
        live = candidate.parallel.proc.process_info(child['pid']) if child else None
        if live is not None and live.get('state') not in ('Z', 'X') and live['start_time'] == child['start_time'] and live['argv'] == command:
            os.kill(live['pid'], signal.SIGKILL)


def test_only_evaluation_thread_environment_is_recorded_and_training_inherits_unchanged(setup,monkeypatch):
    q=setup;monkeypatch.setenv('RESEARCH_THREAD_TEST_CANARY','keep')
    assert q.queue().run()==0
    state=q.state();record=state['jobs']['sft']
    assert state['parallel_controller']['evaluation_environment']==candidate.EVALUATION_ENV
    assert all(row['execution_environment']==candidate.EVALUATION_ENV for row in record['parallel_evaluations'].values())
    assert q.calls[0]=='trainer'
