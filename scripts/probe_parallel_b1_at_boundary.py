#!/usr/bin/env python3
"""One explicit 1b grounding boundary for the separate, fixed4q native-B1 probe.

Only the verified wrapper controller receives STOP/CONT. Its completed native
evaluator exits naturally. Source4q and target1b checkpoints/caches are separate
immutable contexts; no queue, scientific output or inference profile is edited.
"""
from pathlib import Path
import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_research_batch4_at_boundary as boundary
from scripts import probe_research_parallel_b1 as probe
from scripts.handoff_research_grounding import (SPEC, at, digest, options,
    process_info, read_json, spec_snapshot, verify_sources)
from scripts.probe_research_batch_at_boundary import (command_sha, gpu_pids,
    host_available_bytes, persist)
from scripts.research_queue import argv_value
from src.utils.research_checkpoint_compat import same_checkpoint_identity
from src.utils.research_storage import DiskBudget

OWNER = 'parallel_native_b1_boundary_v1'
SCRIPT = 'scripts/probe_parallel_b1_at_boundary.py'
TEST = 'tests/test_research_parallel_b1_boundary.py'
READY = 'configs/research/parallel_b1_boundary_ready.json'
PROBE_READY = 'configs/research/parallel_b1_probe_ready.json'
CONTROLLER_PID = 44472
SOURCE_JOB = 'smallthinker4__siglip2__4q__SFT_pilot_scienceqa_context_v2_balanced_s42'
TARGET_JOB = 'smallthinker4__siglip2__1b__SFT_pilot_scienceqa_context_v2_balanced_s42'
CONTROLLER_SCRIPT = 'scripts/research_grounding_queue.py'
MAX_PAUSED_SECONDS = 350
CHECKS = ('source4q_and_target1b_independently_bound',
    'target_full512_variant_checkpoint_and_raw_scores_verified',
    'only_owned_controller_paused_and_finally_resumed',
    'natural_evaluator_exit_no_child_signals',
    'pid_birth_argv_reuse_and_missed_window_refused',
    'both_immutable_contexts_and_readiness_rechecked',
    'nested300s_probe_within350s_pause_deadline',
    'raw_worker_tokens_provenance_memory_and_timing_revalidated',
    'fresh_one_shot_output_no_automatic_retarget_or_retry')
DEPENDENCIES = (SCRIPT, TEST, 'scripts/eval_research_batch4_at_boundary.py',
    'scripts/handoff_research_grounding.py', 'scripts/probe_research_batch_at_boundary.py',
    'scripts/research_queue.py', CONTROLLER_SCRIPT,
    'scripts/eval_research_checkpoint_batch4.py', 'src/eval/research_batch_inference.py',
    'src/utils/research_storage.py')


def emit(event, **fields):
    print(json.dumps(dict(event=event, time=time.time(), **fields)), flush=True)


def paths(root, variant, output=None):
    if variant not in ('shuffle', 'blank'):
        raise ValueError('Only explicitly selected shuffle or blank is supported')
    result = Path(output or ('research_results/parallel_b1_actual' if variant == 'shuffle'
                             else 'research_results/parallel_b1_actual_blank'))
    result = (root / result).resolve() if not result.is_absolute() else result.resolve()
    if result.parent != (root / 'research_results').resolve():
        raise ValueError('Probe output must be a direct research_results child')
    suffix = '1b_' + variant
    return (root / ('research_results/parallel_b1_boundary_' + suffix + '.json'),
            root / ('logs/research/parallel_b1_boundary_' + suffix + '.log'), result)


def readiness(root):
    proofs = {}
    for name, required, checks in ((READY, DEPENDENCIES, CHECKS),
            (PROBE_READY, (probe.SCRIPT, 'tests/test_research_parallel_b1.py'), None)):
        path = root / name
        if not path.is_file():
            return None
        raw = path.read_bytes(); proof = json.loads(raw)
        if (proof.get('schema_version') != 1 or proof.get('status') != 'passed' or
                type(proof.get('tests_passed')) is not int or proof['tests_passed'] < 1 or
                not isinstance(proof.get('test_command'), str) or not proof['test_command'] or
                not isinstance(proof.get('checks'), dict) or not proof['checks'] or
                any(value is not True for value in proof['checks'].values()) or
                (checks and any(proof['checks'].get(key) is not True for key in checks))):
            raise ValueError('Boundary/prototype CPU readiness checks have not passed')
        verify_sources(root, proof.get('source_sha256'), required)
        if name == PROBE_READY and (proof.get('owner') != probe.OWNER or
                proof.get('actual_gpu_validated') is not False or
                proof.get('limits', {}).get('whole_seconds') != 300):
            raise ValueError('Prototype readiness owner/scope/deadline differs')
        proofs[name] = {'sha256': digest(raw), 'sources': proof['source_sha256']}
    return proofs


def exact_jobs(root, spec, target_id):
    if target_id != TARGET_JOB or SOURCE_JOB == target_id:
        raise ValueError('Target must be the explicit1b job; source remains4q')
    by_id = {job['id']: job for job in spec['jobs']}
    source = by_id.get(SOURCE_JOB); target = by_id.get(target_id)
    if source is None or target is None:
        raise ValueError('Both explicit jobs must exist in the immutable queue')
    for job, recipe in ((source, '4q'), (target, '1b')):
        run = at(root, argv_value(job['argv'], '--output'))
        if (argv_value(job['argv'], '--recipe') != recipe or
                argv_value(job['argv'], '--stage') != 'SFT' or run.name != job['id'] or
                not job.get('evaluation')):
            raise ValueError('Source/target run, recipe or SFT evaluation mismatch')
    if at(root, argv_value(source['argv'], '--output')) != (root / probe.SOURCE_RUN).resolve():
        raise ValueError('Prototype source differs from explicit4q source')
    if at(root, source['evaluation']['manifest']) != at(root, target['evaluation']['manifest']):
        raise ValueError('Source and target require the same original512 manifest')
    return source, target


def contexts(root, spec, state, source_job, target_job):
    if state.get('jobs', {}).get(SOURCE_JOB, {}).get('status') != 'completed':
        raise ValueError('Independent4q source has not completed all standard gates')
    source = boundary.context(root, spec, source_job, state['jobs'][SOURCE_JOB])
    target = boundary.context(root, spec, target_job, state['jobs'][TARGET_JOB])
    for value, recipe in ((source, '4q'), (target, '1b')):
        value['run_metadata'] = run_metadata(root, value, recipe)
    # The reused validator's B4 output is irrelevant: no B4 evaluator is launched.
    source.pop('output', None); target.pop('output', None)
    parallel = probe.prepare_context(root)
    if (source['run'] != parallel['source_run'] or
            source['manifest_sha256'] != parallel['manifest_sha256'] or
            source['native_cache'] != parallel['native_cache'] or
            not same_checkpoint_identity(source['checkpoint_identity'], parallel['checkpoint_identity'])):
        raise ValueError('Fixed4q prototype differs from independently validated4q context')
    if (source['run'] == target['run'] or
            source['checkpoint_identity']['sha256'] == target['checkpoint_identity']['sha256'] or
            target['manifest_sha256'] != source['manifest_sha256']):
        raise ValueError('Independent1b target was confused with4q source')
    return {'source4q': source, 'target1b': target, 'parallel_probe': parallel}


def run_metadata(root, value, recipe):
    run = Path(value['run']); config = read_json(run / 'resolved_config.json')
    native = Path(value['native_cache']); provenance = read_json(native / 'model_provenance.json')
    checkpoint = Path(value['checkpoint_identity']['primary_path'])
    marker = read_json(checkpoint.parent / '.research_checkpoint.json')
    if (config.get('recipe') != recipe or config.get('stage') != 'SFT' or
            at(root, config.get('output')) != run or
            config.get('model') != 'Tiiny/SmallThinker-4BA0.6B-Instruct' or
            config.get('vision') != 'google/siglip2-so400m-patch14-384' or
            provenance.get('recipe') != recipe or provenance.get('stage') != 'SFT' or
            provenance.get('training_step') != marker.get('step') or
            provenance.get('model_ref', {}).get('repo') != config['model'] or
            provenance.get('vision_ref', {}).get('repo') != config['vision']):
        raise ValueError('Source4q/target1b config or native model provenance mismatch')
    return {'files': boundary.files_identity(run, ('resolved_config.json', 'completed.json',
        '.checkpoint_manager.json', 'checkpoint_compaction.json', 'checkpoint_roundtrip.json')),
        'native_model_provenance_sha256': digest((native / 'model_provenance.json').read_bytes()),
        'selected_marker_sha256': digest((checkpoint.parent / '.research_checkpoint.json').read_bytes())}


def phase(root, spec, state, job, controller, variant, expected=None):
    canonical = digest(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode())
    if state.get('spec_sha256') != canonical:
        raise ValueError('State/spec SHA mismatch')
    rows = state.get('jobs')
    if not isinstance(rows, dict) or any(not isinstance(v, dict) for v in rows.values()):
        raise ValueError('Malformed wrapper state')
    row = rows.get(job['id'], {})
    active = [key for key, value in rows.items() if value.get('status') in
              ('running', 'evaluating', 'checkpoint_validating', 'grounding_evaluating')]
    if state.get('status') != 'running' or active != [job['id']]:
        return 'missed', None
    if row.get('status') in ('running', 'evaluating', 'checkpoint_validating'):
        return ('missed' if expected else 'waiting'), None
    current = row.get('active_grounding_variant')
    order = ('shuffle', 'blank', 'text_only')
    if current in order and order.index(current) < order.index(variant):
        return ('missed' if expected else 'waiting'), None
    return boundary.phase(root, spec, state, job, controller, variant, expected)


def same_controller(info, record):
    return (info is not None and info['start_time'] == record.get('controller_start_time') and
            info['pid'] == record.get('controller_pid') and
            command_sha(info['argv']) == record.get('controller_command_sha256'))


def resume_parent(root, record):
    if record.get('owner') != OWNER or record.get('parent_pause_outstanding') is not True:
        return 'no_owned_pause'
    info = process_info(record.get('controller_pid'))
    declared = options(info, root, CONTROLLER_SCRIPT)
    if (not same_controller(info, record) or declared is None or '--spec' not in declared or
            at(root, declared['--spec']) != (root / SPEC).resolve()):
        return 'gone_or_identity_changed'
    if info['state'] in ('T', 't'):
        os.kill(info['pid'], signal.SIGCONT)
        return 'resumed'
    return 'already_running'


def recover_worker(root, record):
    """Terminate only a receipt-bound prototype parent; its own children have PDEATHSIG."""
    if record.get('owner') != OWNER:
        return
    expected = record.get('worker_identity')
    live = probe.process_info((expected or {}).get('pid'))
    command = record.get('worker_command', [])
    if (not probe.same_process(expected, live) or live['argv'] != command or
            command != worker_command(root, Path(record['probe_output']))):
        return
    os.kill(live['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = probe.process_info(live['pid'])
        if not probe.same_process(expected, current):
            return
        time.sleep(.1)
    if probe.same_process(expected, probe.process_info(live['pid'])):
        os.kill(live['pid'], signal.SIGKILL)


def wait_idle(evidence, deadline):
    last = 0.
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError('Natural evaluator exit exceeded the bounded idle window')
        child = process_info(evidence['evaluation_pid'])
        if child is not None and (child['start_time'] != evidence['evaluation_start_time'] or
                child['argv'] != evidence['evaluation_command']):
            raise RuntimeError('Completed evaluator PID was reused/changed')
        pids = gpu_pids()
        if any(pid != evidence['evaluation_pid'] for pid in pids):
            raise RuntimeError('Unknown GPU process; exclusive probe refused')
        if child is None and not pids:
            return
        if time.monotonic() - last >= 30:
            emit('waiting_natural_target_evaluator_exit', pid=evidence['evaluation_pid']); last = time.monotonic()
        time.sleep(.25)


def worker_command(root, output):
    return [sys.executable, '-u', str(root / probe.SCRIPT), '--output', str(output), '--allow-gpu']


def validate_result(output, source, parent_identity):
    probe.verify_context(source)
    report = read_json(output / 'report.json')
    barrier = read_json(output / 'allocation_barrier.json')
    generation = read_json(output / 'generation_barrier.json')
    results = [read_json(output / f'worker{i}.result.json') for i in (0, 1)]
    barrier_sha = probe.file_sha(output / 'allocation_barrier.json')
    elapsed = report.get('wall_seconds')
    if (report.get('owner') != probe.OWNER or report.get('status') != 'passed' or
            report.get('context') != source or barrier.get('context') != source or
            barrier.get('owner') != probe.OWNER or barrier.get('status') != 'released' or
            len(barrier.get('workers', [])) != 2 or
            not probe.same_process(parent_identity, barrier.get('parent')) or
            any(row.get('ppid') != parent_identity['pid'] for row in barrier.get('workers', [])) or
            report.get('allocation_barrier_sha256') != barrier_sha or
            generation.get('allocation_barrier_sha256') != barrier_sha or
            type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 < elapsed <= 300 or
            report.get('whole_probe_limit_seconds') != 300 or report.get('execution_limit_seconds') != 275 or
            report.get('observed_gpu_processes_only_owned_workers') is not True or
            report.get('no_production_profile_changes') is not True):
        raise ValueError('Prototype raw report/context/barrier/deadline differs')
    observed = report['probe_started_monotonic'] + elapsed
    metrics = probe.validate_results(source, dict(barrier, generation_barrier=generation,
        observed_workers_done_monotonic=observed), results)
    if any(report.get(key) != value for key, value in metrics.items()):
        raise ValueError('Raw prototype timing/parity metrics do not reproduce')
    if report.get('workers') != [{k: v for k, v in result.items() if k != 'predictions'} for result in results]:
        raise ValueError('Report worker view differs from immutable raw worker receipts')
    probe.validate_gpu(report['initial_gpu'], exclusive=True, expected_uuid=barrier.get('gpu_uuid'))
    observations = report.get('gpu_observations', [])
    if not observations:
        raise ValueError('No sampled GPU memory/isolation evidence')
    identities = {row['pid']: row for row in barrier['workers']}
    for snapshot in observations:
        probe.validate_gpu(snapshot, barrier['workers'], lookup=identities.get, expected_uuid=barrier['gpu_uuid'])
    files = ('report.json', 'allocation_barrier.json', 'generation_barrier.json',
             'worker0.result.json', 'worker1.result.json')
    return dict(status='passed', **metrics, raw_files_sha256=boundary.files_identity(output, files),
                actual_scope='two independent nativeB1 copies of fixed4q, same64 rows per worker',
                production_profile_changed=False, campaign_speed_claim=False)


def execute(root, output, log, receipt, record, source, deadline):
    if output.exists():
        raise ValueError('One-shot output already exists; artifacts are preserved')
    command = worker_command(root, output)
    record.update(status='probe_starting', attempted=True, worker_command=command,
                  worker_command_sha256=command_sha(command))
    persist(receipt, record); process = None; identity = None
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open('a') as handle:
            handle.write(json.dumps({'time': time.time(), 'command': command}) + '\n'); handle.flush()
            parent = os.getpid()
            process = subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, preexec_fn=lambda: probe.arm_parent_death(parent))
            identity = probe.process_info(process.pid)
            if identity is None or identity['argv'] != command or identity['ppid'] != parent:
                raise RuntimeError('Cannot bind the new owned prototype parent identity')
            record.update(status='running_parallel_b1_probe', worker_identity=identity)
            persist(receipt, record); last = time.monotonic()
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError('Nested probe exceeded its bounded coordinator deadline')
                if not probe.same_process(identity, probe.process_info(process.pid)):
                    raise RuntimeError('Owned prototype parent PID was reused/changed')
                if now - last >= 30:
                    DiskBudget(root, 120).ensure_space(0)
                    emit('parallel_b1_probe_running', pid=process.pid, log=str(log)); last = now
                time.sleep(.5)
            if process.returncode != 0:
                raise RuntimeError('Parallel nativeB1 probe failed; see ' + str(log))
        return validate_result(output, source, identity)
    finally:
        # No process group or evaluator PID is signaled. Prototype cleanup owns its children.
        probe.stop_owned(process, identity)


def run(root, controller_pid=CONTROLLER_PID, target_id=TARGET_JOB, variant='shuffle', output=None,
        timeout=MAX_PAUSED_SECONDS):
    if not 310 <= timeout <= MAX_PAUSED_SECONDS:
        raise ValueError('Pause deadline must be310..350seconds')
    receipt, log, output = paths(root, variant, output)
    previous = read_json(receipt) if receipt.is_file() else {}
    if previous.get('owner') == OWNER and previous.get('parent_pause_outstanding') is True:
        try:
            recover_worker(root, previous)
        finally:
            previous['resume_status'] = resume_parent(root, previous)
            previous.update(parent_pause_outstanding=False, recovered_after_crash=True)
            persist(receipt, previous)
    if previous.get('attempted'):
        raise RuntimeError('Prior one-shot attempt exists; no implicit retry or artifact reuse')
    if output.exists():
        raise ValueError('Fresh output required before waiting; existing artifacts are preserved')
    spec, snapshot = spec_snapshot(root)
    source_job, target_job = exact_jobs(root, spec, target_id)
    controller = boundary.wrapper(root, controller_pid)
    if controller is None:
        return {'owner': OWNER, 'status': 'skipped', 'reason': 'named_wrapper_not_running'}
    if controller.get('state') in ('T', 't'):
        return {'owner': OWNER, 'status': 'skipped', 'reason': 'controller_already_externally_paused'}
    anchor = dict(controller_pid=controller_pid, controller_start_time=controller['start_time'],
                  controller_command_sha256=command_sha(controller['argv']))
    record = dict(owner=OWNER, status='waiting', **anchor, **snapshot,
                  source_job_id=SOURCE_JOB, target_job_id=target_id, variant=variant,
                  parent_pause_outstanding=False, attempted=False, probe_output=str(output),
                  maximum_paused_seconds=timeout)
    expected = None; bound = None; proof = None; last = 0.
    while True:
        fresh_spec, fresh_snapshot = spec_snapshot(root)
        if fresh_snapshot != snapshot:
            raise ValueError('Immutable queue specification changed')
        live = boundary.wrapper(root, controller_pid)
        if not same_controller(live, record):
            record.update(status='skipped', reason='wrapper_changed'); persist(receipt, record); return record
        state = read_json((root / SPEC).with_suffix('.state.json'))
        status, evidence = phase(root, fresh_spec, state, target_job, live, variant, expected)
        if status == 'missed':
            record.update(status='skipped', reason='explicit_target_boundary_missed'); persist(receipt, record); return record
        if evidence is not None:
            expected = evidence
        current_proof = readiness(root)
        if current_proof is not None:
            if proof is not None and proof != current_proof:
                raise ValueError('Tested boundary/prototype readiness changed during wait')
            proof = current_proof
            if bound is None and state['jobs'][target_id].get('status') == 'grounding_evaluating':
                bound = contexts(root, fresh_spec, state, source_job, target_job)
                record.update(contexts=bound, proof=proof)
        if status == 'ready' and bound is not None and proof is not None:
            break
        if time.monotonic() - last >= 30:
            persist(receipt, record)
            emit('waiting_explicit_parallel_b1_boundary', source=SOURCE_JOB, target=target_id,
                 variant=variant, ready=proof is not None); last = time.monotonic()
        time.sleep(.5)
    if host_available_bytes() < 32 * 2 ** 30:
        raise RuntimeError('Insufficient host/cgroup headroom for two small native models')
    DiskBudget(root, 120).ensure_space(128 * 2 ** 20)
    live = process_info(controller_pid)
    if not same_controller(live, record):
        record.update(status='skipped', reason='wrapper_changed_before_pause'); persist(receipt, record); return record
    if live.get('state') in ('T', 't'):
        record.update(status='skipped', reason='controller_already_externally_paused'); persist(receipt, record); return record
    pause_started = time.monotonic(); deadline = pause_started + timeout
    record.update(status='pause_intent', parent_pause_outstanding=True, **expected,
                  pause_started_monotonic=pause_started)
    persist(receipt, record)
    try:
        os.kill(controller_pid, signal.SIGSTOP)
        while True:
            live = process_info(controller_pid)
            if not same_controller(live, record):
                raise RuntimeError('Wrapper identity changed at pause barrier')
            if live['state'] in ('T', 't'):
                break
            if time.monotonic() - pause_started >= 5:
                raise TimeoutError('Wrapper pause barrier timed out')
            time.sleep(.05)
        fresh_spec, fresh_snapshot = spec_snapshot(root)
        if fresh_snapshot != snapshot or readiness(root) != proof:
            raise ValueError('Immutable queue or tested readiness changed at pause barrier')
        state = read_json((root / SPEC).with_suffix('.state.json'))
        status, fresh = phase(root, fresh_spec, state, target_job, live, variant, expected)
        if status != 'ready' or fresh != expected:
            record.update(status='skipped', reason='phase_advanced_at_pause_barrier'); return record
        if contexts(root, fresh_spec, state, source_job, target_job) != bound:
            raise ValueError('Source4q or target1b context changed at pause barrier')
        # Critically use TARGET1b here; the GPU probe separately receives SOURCE4q.
        boundary.verify_completed_variant(root, target_job, bound['target1b'], fresh)
        if boundary.files_identity(Path(fresh['variant_output']), fresh['completed_variant_files']) != fresh['completed_variant_files']:
            raise ValueError('Completed target1b grounding files changed')
        wait_idle(fresh, min(deadline - 315, time.monotonic() + 20))
        record.update(status='paused_exclusive_window'); persist(receipt, record)
        result = execute(root, output, log, receipt, record, bound['parallel_probe'], deadline - 15)
        if spec_snapshot(root)[1] != snapshot or readiness(root) != proof:
            raise ValueError('Queue/readiness changed during owned probe')
        if contexts(root, fresh_spec, state, source_job, target_job) != bound:
            raise ValueError('Source4q or target1b context changed during owned probe')
        if boundary.files_identity(Path(fresh['variant_output']), fresh['completed_variant_files']) != fresh['completed_variant_files']:
            raise ValueError('Target1b completed grounding bytes changed during probe')
        if time.monotonic() >= deadline:
            raise TimeoutError('Total paused window exceeded its350second cap')
        record.update(status='completed', result=result); return record
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error)); raise
    finally:
        # SIGINT/SIGTERM are exceptions, so they also pass through this cleanup.
        try:
            record['resume_status'] = resume_parent(root, record)
            record['parent_pause_outstanding'] = False
        finally:
            record.update(resumed_time=time.time(), paused_seconds=time.monotonic() - pause_started)
            persist(receipt, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-pid', type=int, default=CONTROLLER_PID)
    parser.add_argument('--target-job', default=TARGET_JOB)
    parser.add_argument('--variant', choices=['shuffle', 'blank'], default='shuffle')
    parser.add_argument('--output')
    parser.add_argument('--timeout', type=int, default=MAX_PAUSED_SECONDS)
    args = parser.parse_args(argv)
    def interrupted(signum, _frame):
        raise InterruptedError('Boundary coordinator received signal ' + str(signum))
    signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
    receipt, _, _ = paths(ROOT, args.variant, args.output)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(receipt) + '.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run(ROOT, args.controller_pid, args.target_job, args.variant, args.output, args.timeout)
    emit('parallel_b1_boundary_finished', **result)
    return 0 if result['status'] in ('completed', 'skipped') else 1


if __name__ == '__main__':
    raise SystemExit(main())
