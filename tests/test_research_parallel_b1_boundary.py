"""CPU-only lifecycle/provenance tests: no service, CUDA, NVML or real signals."""
import copy
import json
from pathlib import Path
import signal
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import probe_parallel_b1_at_boundary as c


@pytest.fixture
def window(tmp_path, monkeypatch):
    spec = json.loads((c.ROOT / c.SPEC).read_text())
    source_job, target_job = c.exact_jobs(tmp_path, spec, c.TARGET_JOB)
    def write(path, value):
        path = tmp_path / path; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)); return path
    write(c.SPEC, spec)
    target_run = c.at(tmp_path, c.argv_value(target_job['argv'], '--output'))
    variant_output = target_run / 'evaluation/grounding_v1/shuffle'
    command = ['/venv/main/bin/python', '-u', str(tmp_path / 'scripts/eval_research_checkpoint.py'),
        '--run', str(target_run), '--manifest', str(c.at(tmp_path, target_job['evaluation']['manifest'])),
        '--output', str(variant_output), '--checkpoint', 'best', '--max-new-tokens', '64',
        '--image-ablation', 'shuffle', '--ablation-seed', '42']
    parent = dict(pid=44472, start_time=123, state='S', ppid=1, cwd=str(tmp_path),
        argv=['/venv/main/bin/python', '-u', c.CONTROLLER_SCRIPT, '--spec', c.SPEC, '--retry-failed'])
    child = dict(pid=70001, start_time=456, state='S', ppid=44472, cwd=str(tmp_path), argv=command)
    infos = {44472: parent, 70001: child}
    _, snapshot = c.spec_snapshot(tmp_path)
    state = {'status': 'running', 'spec_sha256': snapshot['canonical_spec_sha256'], 'jobs': {
        c.SOURCE_JOB: {'status': 'completed', 'training_completed': True},
        c.TARGET_JOB: {'status': 'grounding_evaluating', 'training_completed': True,
            'grounding_status': 'running', 'active_grounding_variant': 'shuffle',
            'grounding_evaluations': {'shuffle': {'status': 'running', 'pid': 70001,
                'command': command, 'output': str(variant_output)}}}}}
    state_path = Path(c.SPEC).with_suffix('.state.json'); write(state_path, state)
    source = {'run': str(tmp_path / c.probe.SOURCE_RUN), 'manifest_sha256': 'manifest',
        'checkpoint_identity': {'sha256': 'source4q'}, 'native_cache': 'native4q'}
    target = {'run': str(target_run), 'manifest_sha256': 'manifest',
        'checkpoint_identity': {'sha256': 'target1b'}, 'native_cache': 'native1b'}
    parallel = {'source_run': source['run'], 'manifest_sha256': 'manifest',
        'checkpoint_identity': {'sha256': 'source4q'}, 'native_cache': 'native4q'}
    bound = dict(source4q=source, target1b=target, parallel_probe=parallel)
    signals = []; verified = []; launched = []
    def kill(pid, sig):
        signals.append((pid, sig))
        if pid == parent['pid']:
            parent['state'] = 'T' if sig == signal.SIGSTOP else 'S'
    monkeypatch.setattr(c, 'process_info', lambda pid: infos.get(pid))
    monkeypatch.setattr(c.boundary, 'process_info', lambda pid: infos.get(pid))
    monkeypatch.setattr(c.boundary, 'wrapper', lambda root, pid: infos.get(pid))
    monkeypatch.setattr(c, 'readiness', lambda root: {'both_proofs': 'bound'})
    monkeypatch.setattr(c, 'contexts', lambda *args: copy.deepcopy(bound))
    monkeypatch.setattr(c.boundary, 'verify_completed_variant', lambda root, job, context, evidence:
        verified.append((job['id'], context['checkpoint_identity']['sha256'])))
    monkeypatch.setattr(c, 'host_available_bytes', lambda: 100 * 2**30)
    monkeypatch.setattr(c.DiskBudget, 'ensure_space', lambda *args: None)
    monkeypatch.setattr(c.os, 'kill', kill)
    monkeypatch.setattr(c, 'gpu_pids', lambda: (_ for _ in ()).throw(AssertionError('No actual NVML')))
    monkeypatch.setattr(c, 'wait_idle', lambda *args: None)
    def execute(root, output, log, receipt, record, context, deadline):
        record['attempted'] = True; launched.append(context)
        return {'status': 'passed', 'distinct_original_records': 64, 'aggregate_duplicate_completions': 128}
    monkeypatch.setattr(c, 'execute', execute)
    def complete():
        for name, value in {'summary.json': {'complete_manifest': True, 'records': 512, 'manifest_records': 512},
            'progress.json': {'event': 'completed', 'completed_records': 512},
            'protocol.json': {'target': '1b', 'variant': 'shuffle'},
            'predictions.jsonl': {'controlled': 'validated by mock'},
            'transformed_manifest.jsonl': {'fixture': True}}.items():
            write(variant_output / name, value)
    return SimpleNamespace(root=tmp_path, spec=spec, source_job=source_job, target_job=target_job,
        parent=parent, child=child, infos=infos, state=state, state_path=state_path, write=write,
        output=variant_output, bound=bound, signals=signals, verified=verified,
        launched=launched, complete=complete)


def test_success_separately_verifies_target1b_and_probes_source4q_then_continues(window):
    w = window; w.complete(); original = (w.root / w.state_path).read_bytes()
    result = c.run(w.root)
    assert result['status'] == 'completed'
    assert w.verified == [(c.TARGET_JOB, 'target1b')]
    assert w.launched == [w.bound['parallel_probe']]
    assert w.signals == [(44472, signal.SIGSTOP), (44472, signal.SIGCONT)]
    assert (w.root / w.state_path).read_bytes() == original
    receipt, _, _ = c.paths(w.root, 'shuffle')
    assert c.read_json(receipt)['parent_pause_outstanding'] is False


@pytest.mark.parametrize('stage', ['running', 'evaluating', 'checkpoint_validating'])
def test_future_target_phase_waits_without_retarget(window, stage):
    w = window; w.state['jobs'][c.TARGET_JOB]['status'] = stage
    assert c.phase(w.root, w.spec, w.state, w.target_job, w.parent, 'shuffle') == ('waiting', None)
    assert not w.signals


def test_explicit_blank_waits_for_shuffle_but_shuffle_never_retargets_blank(window):
    w = window
    assert c.phase(w.root, w.spec, w.state, w.target_job, w.parent, 'blank') == ('waiting', None)
    w.state['jobs'][c.TARGET_JOB]['active_grounding_variant'] = 'blank'
    w.write(w.state_path, w.state)
    result = c.run(w.root)
    assert result['reason'] == 'explicit_target_boundary_missed'
    assert not w.signals and not w.launched
    assert c.paths(w.root, 'blank')[0] != c.paths(w.root, 'shuffle')[0]
    assert c.paths(w.root, 'blank')[2] != c.paths(w.root, 'shuffle')[2]


@pytest.mark.parametrize('kind', ['wrong_target', 'swapped_recipe', 'swapped_run', 'manifest'])
def test_source_target_confusion_rejected_before_pause(window, kind):
    w = window; spec = copy.deepcopy(w.spec)
    target = next(j for j in spec['jobs'] if j['id'] == c.TARGET_JOB)
    target_id = c.SOURCE_JOB if kind == 'wrong_target' else c.TARGET_JOB
    if kind == 'swapped_recipe':
        target['argv'][target['argv'].index('--recipe') + 1] = '4q'
    if kind == 'swapped_run':
        target['argv'][target['argv'].index('--output') + 1] = c.probe.SOURCE_RUN
    if kind == 'manifest':
        target['evaluation']['manifest'] = 'other/manifest.jsonl'
    with pytest.raises(ValueError):
        c.exact_jobs(w.root, spec, target_id)
    assert not w.signals


@pytest.mark.parametrize('context_name', ['source4q', 'target1b', 'parallel_probe'])
def test_either_immutable_context_change_at_barrier_fails_and_resumes(window, monkeypatch, context_name):
    w = window; w.complete(); bound = copy.deepcopy(w.bound); count = 0
    def changed(*args):
        nonlocal count
        count += 1
        value = copy.deepcopy(bound)
        if count > 1:
            value[context_name]['tampered'] = True
        return value
    monkeypatch.setattr(c, 'contexts', changed)
    with pytest.raises(ValueError, match='context changed'):
        c.run(w.root)
    assert w.signals == [(44472, signal.SIGSTOP), (44472, signal.SIGCONT)]
    assert not w.launched


@pytest.mark.parametrize('error', [ValueError('token parity failed'), InterruptedError('SIGINT'),
                                 TimeoutError('probe deadline')])
def test_all_owned_probe_failures_finally_resume_without_evaluator_signals(window, monkeypatch, error):
    w = window; w.complete()
    def failed(*args):
        raise error
    monkeypatch.setattr(c, 'execute', failed)
    with pytest.raises(type(error), match=str(error)):
        c.run(w.root)
    assert w.signals == [(44472, signal.SIGSTOP), (44472, signal.SIGCONT)]
    assert c.read_json(c.paths(w.root, 'shuffle')[0])['status'] == 'failed'


def test_phase_race_after_stop_resumes_without_probe(window, monkeypatch):
    w = window; w.complete(); original = c.os.kill
    def shifted(pid, sig):
        original(pid, sig)
        if sig == signal.SIGSTOP:
            w.state['jobs'][c.TARGET_JOB]['active_grounding_variant'] = 'blank'
            w.write(w.state_path, w.state)
    monkeypatch.setattr(c.os, 'kill', shifted)
    assert c.run(w.root)['reason'] == 'phase_advanced_at_pause_barrier'
    assert w.signals == [(44472, signal.SIGSTOP), (44472, signal.SIGCONT)]
    assert not w.launched


def test_evaluator_pid_reuse_refuses_captured_window(window):
    w = window; w.complete()
    status, evidence = c.phase(w.root, w.spec, w.state, w.target_job, w.parent, 'shuffle')
    assert status == 'ready'
    w.child['start_time'] += 1
    assert c.phase(w.root, w.spec, w.state, w.target_job, w.parent, 'shuffle', evidence)[0] == 'missed'
    assert not w.signals


def test_controller_pid_reuse_at_pause_does_not_signal_reused_pid(window, monkeypatch):
    w = window; w.complete(); original = c.os.kill
    def reused(pid, sig):
        original(pid, sig)
        if sig == signal.SIGSTOP:
            w.parent['start_time'] += 1
    monkeypatch.setattr(c.os, 'kill', reused)
    with pytest.raises(RuntimeError, match='identity changed'):
        c.run(w.root)
    assert w.signals == [(44472, signal.SIGSTOP)]
    assert not w.launched


@pytest.mark.parametrize('when', ['initial', 'before_pause_intent'])
def test_external_controller_pause_never_claimed_or_continued(window, monkeypatch, when):
    w = window; w.complete()
    if when == 'initial':
        w.parent['state'] = 'T'
    else:
        original = c.host_available_bytes
        def external_pause():
            w.parent['state'] = 'T'
            return original()
        monkeypatch.setattr(c, 'host_available_bytes', external_pause)
    result = c.run(w.root)
    assert result['status'] == 'skipped'
    assert result['reason'] == 'controller_already_externally_paused'
    assert not w.signals and not w.launched


def test_recovery_resumes_owned_parent_before_readiness_failure(window, monkeypatch):
    w = window; w.parent['state'] = 'T'
    receipt, _, _ = c.paths(w.root, 'shuffle')
    w.write(receipt, {'owner': c.OWNER, 'parent_pause_outstanding': True,
        'controller_pid': 44472, 'controller_start_time': 123,
        'controller_command_sha256': c.command_sha(w.parent['argv']), 'attempted': True})
    monkeypatch.setattr(c, 'readiness', lambda *args: (_ for _ in ()).throw(AssertionError('No readiness yet')))
    with pytest.raises(RuntimeError, match='one-shot attempt'):
        c.run(w.root)
    assert w.signals == [(44472, signal.SIGCONT)]


@pytest.mark.parametrize('field,value', [('owner', 'foreign'), ('controller_start_time', 999),
                                      ('controller_command_sha256', 'foreign')])
def test_resume_refuses_foreign_or_reused_controller(window, field, value):
    w = window; w.parent['state'] = 'T'
    record = dict(owner=c.OWNER, parent_pause_outstanding=True, controller_pid=44472,
        controller_start_time=123, controller_command_sha256=c.command_sha(w.parent['argv']))
    record[field] = value
    c.resume_parent(w.root, record)
    assert not w.signals


def test_natural_eval_exit_wait_is_pid_safe_and_never_signals(window, monkeypatch):
    w = window; w.complete()
    _, evidence = c.phase(w.root, w.spec, w.state, w.target_job, w.parent, 'shuffle')
    w.infos[70001] = None
    with patch.object(c, 'gpu_pids', side_effect=[[70001], []]), patch.object(c.time, 'sleep'):
        # Use the original implementation despite fixture replacing it.
        ORIGINAL_WAIT_IDLE(evidence, c.time.monotonic() + 5)
    monkeypatch.setattr(c, 'gpu_pids', lambda: [999])
    with pytest.raises(RuntimeError, match='Unknown GPU'):
        ORIGINAL_WAIT_IDLE(evidence, c.time.monotonic() + 5)
    assert not w.signals


def test_natural_exit_timeout_does_not_signal_evaluator(window):
    w = window; w.complete()
    _, evidence = c.phase(w.root, w.spec, w.state, w.target_job, w.parent, 'shuffle')
    with pytest.raises(TimeoutError, match='idle window'):
        ORIGINAL_WAIT_IDLE(evidence, c.time.monotonic() - 1)
    assert not w.signals


def test_probe_command_has_fixed_source_in_script_no_target_run_or_profile_override(window):
    output = c.paths(window.root, 'shuffle')[2]
    command = c.worker_command(window.root, output)
    assert command[2:] == [str(window.root / c.probe.SCRIPT), '--output', str(output), '--allow-gpu']
    assert '--run' not in command and '--batch-size' not in command
    assert c.TARGET_JOB not in ' '.join(command)


@pytest.mark.parametrize('timeout', [0, 309, 351, 1200])
def test_pause_deadline_cap_rejected_before_any_actions(window, timeout):
    with pytest.raises(ValueError, match='310..350'):
        c.run(window.root, timeout=timeout)
    assert not window.signals


def test_preexisting_probe_output_preserved_without_controller_signals(window):
    w = window; output = c.paths(w.root, 'shuffle')[2]
    output.mkdir(parents=True); (output / 'evidence').write_text('preserve')
    with pytest.raises(ValueError, match='Fresh output'):
        c.run(w.root)
    assert (output / 'evidence').read_text() == 'preserve' and not w.signals


def test_readiness_requires_both_proofs_and_all_critical_sources(tmp_path):
    path = tmp_path / c.READY; path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'schema_version': 1, 'status': 'passed', 'tests_passed': 1,
        'test_command': 'pytest', 'checks': {key: True for key in c.CHECKS}, 'source_sha256': {}}))
    with pytest.raises(ValueError, match='required tested source'):
        ORIGINAL_READINESS(tmp_path)


def test_real_context_adapter_refuses_source_target_checkpoint_confusion(window, monkeypatch):
    w = window
    def context(root, spec, job, row):
        value = copy.deepcopy(w.bound['source4q' if job['id'] == c.SOURCE_JOB else 'target1b'])
        value['output'] = 'unused-B4-path'
        return value
    monkeypatch.setattr(c.boundary, 'context', context)
    monkeypatch.setattr(c.probe, 'prepare_context', lambda root: copy.deepcopy(w.bound['parallel_probe']))
    monkeypatch.setattr(c, 'same_checkpoint_identity', lambda a, b: a == b)
    monkeypatch.setattr(c, 'run_metadata', lambda root, value, recipe: {'recipe': recipe})
    result = ORIGINAL_CONTEXTS(w.root, w.spec, w.state, w.source_job, w.target_job)
    assert 'output' not in result['source4q'] and 'output' not in result['target1b']
    w.bound['target1b']['checkpoint_identity']['sha256'] = 'source4q'
    with pytest.raises(ValueError, match='confused'):
        ORIGINAL_CONTEXTS(w.root, w.spec, w.state, w.source_job, w.target_job)


def test_owned_nested_timeout_calls_only_identity_bound_cleanup(window, monkeypatch):
    w = window; actions = []; output = c.paths(w.root, 'shuffle')[2]
    command = c.worker_command(w.root, output)
    identity = dict(pid=80001, start_time=789, ppid=c.os.getpid(), argv=command,
        command_sha256=c.probe.digest(command), state='S')
    fake = SimpleNamespace(pid=80001, returncode=None, poll=lambda: None)
    monkeypatch.setattr(c.subprocess, 'Popen', lambda *args, **kwargs: fake)
    monkeypatch.setattr(c.probe, 'process_info', lambda pid: identity)
    monkeypatch.setattr(c.probe, 'stop_owned', lambda process, expected: actions.append((process, expected)))
    receipt, log, _ = c.paths(w.root, 'shuffle')
    with pytest.raises(TimeoutError, match='coordinator deadline'):
        ORIGINAL_EXECUTE(w.root, output, log, receipt, {}, w.bound['parallel_probe'], -1)
    assert actions == [(fake, identity)] and not w.signals


@pytest.fixture
def raw_result(tmp_path, monkeypatch):
    output = tmp_path / 'raw'; output.mkdir()
    source = {'immutable_source': '4q'}
    def identity(pid, ppid):
        argv = ['python', 'owned-probe', str(pid)]
        return dict(pid=pid, ppid=ppid, start_time=pid + 2, argv=argv,
                    command_sha256=c.probe.digest(argv), state='S')
    parent = identity(500, 400); workers = [identity(501, 500), identity(502, 500)]
    barrier = dict(owner=c.probe.OWNER, status='released', context=source, parent=parent,
                   workers=workers, nonce='fresh', deadline_monotonic=375, gpu_uuid='GPU-fixed')
    def write(name, value):
        (output / name).write_text(json.dumps(value))
    write('allocation_barrier.json', barrier)
    barrier_sha = c.probe.file_sha(output / 'allocation_barrier.json')
    generation = dict(owner=c.probe.OWNER, status='released', nonce='fresh',
                      allocation_barrier_sha256=barrier_sha, released_monotonic=110)
    write('generation_barrier.json', generation)
    results = [dict(owner=c.probe.OWNER, status='passed', worker=i, identity=workers[i],
                    predictions=[{'record_id': 'fake fixture verified by prototype'}]) for i in (0, 1)]
    for i, result in enumerate(results):
        write(f'worker{i}.result.json', result)
    metrics = dict(generation_window_seconds=9., overlap_seconds=8.,
                   aggregate_duplicate_completions=128, distinct_original_records=64,
                   aggregate_records_per_second=128 / 9)
    initial = dict(gpu_uuid='GPU-fixed', total_bytes=96 * 2**30, free_bytes=96 * 2**30, processes=[])
    observation = dict(gpu_uuid='GPU-fixed', total_bytes=96 * 2**30, free_bytes=70 * 2**30,
                       processes=[{'pid': 501, 'bytes': 6 * 2**30}, {'pid': 502, 'bytes': 6 * 2**30}])
    report = dict(owner=c.probe.OWNER, status='passed', context=source,
        allocation_barrier_sha256=barrier_sha, wall_seconds=20., probe_started_monotonic=100.,
        whole_probe_limit_seconds=300, execution_limit_seconds=275,
        observed_gpu_processes_only_owned_workers=True, no_production_profile_changes=True,
        workers=[{k: v for k, v in row.items() if k != 'predictions'} for row in results],
        initial_gpu=initial, gpu_observations=[observation], **metrics)
    write('report.json', report)
    verified = []
    monkeypatch.setattr(c.probe, 'verify_context', lambda value: verified.append(value))
    def validate(context, value, rows):
        assert context == source and rows == results
        assert value['observed_workers_done_monotonic'] == 120
        assert value['generation_barrier'] == generation
        return metrics
    monkeypatch.setattr(c.probe, 'validate_results', validate)
    return SimpleNamespace(output=output, source=source, parent=parent, report=report,
        barrier=barrier, generation=generation, results=results, metrics=metrics, write=write, verified=verified)


def test_completed_probe_revalidates_raw_workers_and_preserves_scope(raw_result):
    w = raw_result
    result = c.validate_result(w.output, w.source, w.parent)
    assert result['status'] == 'passed' and result['distinct_original_records'] == 64
    assert result['aggregate_duplicate_completions'] == 128
    assert not result['campaign_speed_claim'] and not result['production_profile_changed']
    assert set(result['raw_files_sha256']) == {'report.json', 'allocation_barrier.json',
        'generation_barrier.json', 'worker0.result.json', 'worker1.result.json'}
    assert w.verified == [w.source]


@pytest.mark.parametrize('field,value', [('context', {'immutable_source': '1b'}),
    ('status', 'failed'), ('wall_seconds', 301), ('allocation_barrier_sha256', 'forged'),
    ('whole_probe_limit_seconds', 350), ('observed_gpu_processes_only_owned_workers', False),
    ('aggregate_duplicate_completions', 512), ('no_production_profile_changes', False)])
def test_raw_report_binding_deadline_metrics_corruption_refused(raw_result, field, value):
    w = raw_result; w.report[field] = value; w.write('report.json', w.report)
    with pytest.raises(ValueError):
        c.validate_result(w.output, w.source, w.parent)


def test_raw_barrier_must_bind_the_actual_owned_parent_birth(raw_result):
    w = raw_result; expected = dict(w.parent, start_time=999)
    with pytest.raises(ValueError, match='report/context/barrier'):
        c.validate_result(w.output, w.source, expected)


@pytest.mark.parametrize('kind', ['foreign_pid', 'memory', 'uuid', 'free_floor'])
def test_raw_sampled_gpu_guards_recomputed_without_live_nvml(raw_result, kind):
    w = raw_result; snapshot = w.report['gpu_observations'][0]
    if kind == 'foreign_pid':
        snapshot['processes'][0]['pid'] = 999
    elif kind == 'memory':
        snapshot['processes'][0]['bytes'] = 13 * 2**30
    elif kind == 'uuid':
        snapshot['gpu_uuid'] = 'GPU-foreign'
    else:
        snapshot['free_bytes'] = 23 * 2**30
    w.write('report.json', w.report)
    with pytest.raises(RuntimeError):
        c.validate_result(w.output, w.source, w.parent)


ORIGINAL_WAIT_IDLE = c.wait_idle
ORIGINAL_CONTEXTS = c.contexts
ORIGINAL_READINESS = c.readiness
ORIGINAL_EXECUTE = c.execute
