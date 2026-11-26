#!/usr/bin/env python3
"""Proposal: explicit parallel-native-B1 execution for pending Small QLoRA views.

No launch or adoption occurs on import. Startup requires a current CPU source
proof AND an independently revalidated actual full512 4q token comparison.
The immutable training specification is unchanged. Completed native views keep
their original outputs/receipts. Other models/full-training recipes retain the
serial GroundingResearchQueue implementation and its best/last replay gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT
sys.path.insert(0, str(ROOT))
from scripts import research_grounding_queue as native
from scripts import eval_research_checkpoint_parallel_b1 as parallel
from scripts.research_queue import argv_value, output_dir, resolve_path, timestamp, emit
from src.utils.research_storage import _atomic_json
from src.utils.research_checkpoint_compat import same_checkpoint_identity

OWNER = 'parallel_native_b1_grounding_controller_proposal_v1'
SCRIPT = 'scripts/research_parallel_grounding_queue.py'
TEST = 'tests/test_research_parallel_grounding_queue.py'
READY = 'configs/research/parallel_grounding_queue_ready.json'
EVALUATOR_READY = 'configs/research/parallel_b1_full_evaluator_ready.json'
EVALUATION_ENV = {'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'}
ALL_MODES = ('none', *native.MODES)
CHECKS = ('actual_full512_4q_token_proof_required_before_start',
          'same_spec_state_no_training_profile_change',
          'completed_native_views_and_receipts_preserved',
          'new_parallel_schema_and_namespace_explicit',
          'both_retained_checkpoint_gpu_replay_inherited',
          'actual512_rows_paired_bootstrap',
          'partial_journal_exact_resume_no_implicit_failure_retry',
          'full_training_and_other_models_remain_serial',
          'fresh_exec_parent_death_and_registration_guard',
          'owned_live_worker_and_pid_reuse_refusal',
          'durable_completed_training_never_relaunched')


def sha(path):
    return parallel.proc.file_sha(path)


def digest(value):
    return parallel.proc.digest(value)


def read(path):
    return json.loads(Path(path).read_text())


def verify_sources(root, sources, required):
    if not isinstance(sources, dict) or not set(required) <= set(sources):
        raise ValueError('Readiness source map is incomplete')
    for name, expected in sources.items():
        target = (root / name).resolve(strict=True)
        if not target.is_relative_to(root.resolve()) or not target.is_file() or sha(target) != expected:
            raise ValueError('Readiness code source changed: ' + name)


def readiness(root=CODE_ROOT):
    receipts = {}
    for name, required, checks in (
        (READY, (SCRIPT, TEST, 'scripts/research_queue.py', 'scripts/research_grounding_queue.py',
                 'scripts/validate_research_checkpoint.py', parallel.SCRIPT), CHECKS),
        (EVALUATOR_READY, (parallel.SCRIPT, parallel.MODULE, parallel.TEST),
                 ('full512_raw_token_score_provenance_revalidation',))):
        path = root / name
        report = read(path)
        if (report.get('schema_version') != 1 or report.get('status') != 'passed' or
                type(report.get('tests_passed')) is not int or report['tests_passed'] < 1 or
                not report.get('test_command') or not isinstance(report.get('checks'), dict) or
                any(value is not True for value in report['checks'].values()) or
                any(report['checks'].get(key) is not True for key in checks)):
            raise ValueError('Controller/evaluator CPU readiness proof is not passed')
        verify_sources(root, report.get('source_sha256'), required)
        receipts[name] = {'sha256': sha(path), 'source_sha256': report['source_sha256']}
    proposal = read(root / READY)
    evaluator = read(root / EVALUATOR_READY)
    if (proposal.get('owner') != OWNER or proposal.get('worker_counts') != [2, 4] or
            proposal.get('actual_full512_proof_required') is not True or
            evaluator.get('owner') != parallel.OWNER or evaluator.get('worker_counts') != [2, 4] or
            evaluator.get('full_manifest_records') != 512 or evaluator.get('max_new_tokens') != 64):
        raise ValueError('Proposal/evaluator scope differs from the explicit native-B1 profile')
    return receipts


def adoption_evidence(proof_run, manifest, workers):
    """Read-only actual proof; CPU readiness alone can never unlock adoption."""
    sources = readiness()
    run = Path(proof_run).resolve(strict=True)
    config = read(run / 'resolved_config.json')
    if (config.get('recipe') != '4q' or config.get('stage') != 'SFT' or
            config.get('model') != 'Tiiny/SmallThinker-4BA0.6B-Instruct' or
            config.get('vision') != 'google/siglip2-so400m-patch14-384'):
        raise ValueError('Adoption proof must be the completed SmallThinker/SigLIP2 4q SFT')
    audited = parallel.validate_completed_output(run, manifest, 'none', workers,
                                               native_cache=run / 'evaluation/heldout_v3')
    if (audited.get('owner') != parallel.OWNER or audited.get('status') != 'passed' or
            audited.get('records') != 512 or audited.get('worker_count') != workers or
            audited.get('full512_exact_previous_native_comparison') is not True or
            audited.get('per_model_native64_guard_verified') is not True):
        raise ValueError('Actual full512 previous-native token/provenance proof is required')
    return {'proof_run': str(run), 'manifest': str(Path(manifest).resolve(strict=True)),
            'manifest_sha256': sha(manifest), 'workers': workers, 'readiness': sources,
            'raw_actual_full512_audit': audited}


def worker_matches(info, runs):
    if not info or info.get('state') in ('Z', 'X'):
        return False
    argv = info.get('argv', [])
    scripts = {'run_research_stage.py', 'eval_research_checkpoint.py',
               'validate_research_checkpoint.py', Path(parallel.SCRIPT).name}
    if not any(Path(arg).name in scripts for arg in argv):
        return False
    for flag in ('--run', '--output', '--barrier'):
        value = argv_value(argv, flag)
        if value:
            target = resolve_path(value)
            if any(target == run or target.is_relative_to(run) for run in runs):
                return True
    return False


def assert_no_live_workers(spec):
    """Read only owned run command lines; never signal stale/foreign PIDs."""
    runs = [output_dir(job['argv']) for job in spec['jobs']]
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            info = parallel.proc.process_info(int(entry.name))
            if worker_matches(info, runs):
                raise RuntimeError(f'Owned research worker PID {info["pid"]} is still live; safe handoff required')


def verify_spawned_evaluator(info, parent, pid, gateway, command, expected_birth=None):
    if (not isinstance(info, dict) or info.get('pid') != pid or info.get('ppid') != parent['pid'] or
            info.get('state') in ('Z', 'X') or info.get('argv') not in (gateway, command) or
            info.get('command_sha256') != digest(info.get('argv')) or
            type(info.get('start_time')) is not int or
            (expected_birth is not None and info['start_time'] != expected_birth)):
        raise RuntimeError('Owned evaluator PID birth/PPID/gateway-or-final-command binding changed')
    return info


def evaluator_birth(pid):
    """Read raw birth/PPID when cmdline is missing or empty during exec."""
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        if fields[0] in ('Z', 'X'): return None
        return {'pid': pid, 'ppid': int(fields[1]), 'start_time': int(fields[19]),
                'state': fields[0], 'argv': None}
    except (OSError, ValueError, IndexError):
        return None


def evaluator_handshake(launch_path, declaration, receipt):
    path = Path(launch_path).with_suffix('.registration.json')
    if not path.is_file(): return False
    registration = read(path)
    own = registration.get('identity', {})
    if (registration.get('owner') != OWNER or registration.get('nonce') != declaration['nonce'] or
            registration.get('launch_sha256') != sha(launch_path) or
            registration.get('command_sha256') != declaration['command_sha256'] or
            own.get('pid') != receipt.get('pid') or own.get('ppid') != declaration['parent']['pid'] or
            own.get('start_time') != receipt.get('process_identity', {}).get('start_time')):
        raise ValueError('Fresh evaluator nonce/birth/PPID/launch handshake differs')
    return True


def observe_evaluator(process, receipt, parent, gateway, command, *, startup):
    """Grace applies only to absent/empty argv; foreign identities fail now."""
    live = parallel.proc.process_info(process.pid)
    if live is None and process.poll() is None: live = evaluator_birth(process.pid)
    argv = None if live is None else live.get('argv')
    kind = ('unavailable' if argv is None else 'empty' if argv == [] else
            'gateway' if argv == gateway else 'final' if argv == command else 'foreign')
    observed = {'monotonic': time.monotonic(), 'pid': process.pid,
                'ppid': None if live is None else live.get('ppid'),
                'start_time': None if live is None else live.get('start_time'),
                'argv_state': kind, 'argv_sha256': None if argv is None else digest(argv)}
    if startup and len(receipt.setdefault('startup_observations', [])) < 64:
        receipt['startup_observations'].append(observed)
    receipt['last_process_observation'] = observed
    expected = receipt.get('process_identity', {}).get('start_time')
    if live is not None:
        if (live.get('pid') != process.pid or live.get('ppid') != parent['pid'] or
                type(live.get('start_time')) is not int or
                (expected is not None and live['start_time'] != expected)):
            raise RuntimeError('Owned evaluator startup/live birth/PPID differs')
        if expected is None: receipt['process_identity'] = copy.deepcopy(live)
    if kind == 'foreign':
        raise RuntimeError('Owned evaluator has foreign nonempty argv; startup grace refused')
    if kind in ('gateway', 'final'):
        verify_spawned_evaluator(live, parent, process.pid, gateway, command, expected)
        receipt['process_identity'] = copy.deepcopy(live)
    if not startup:
        if live is None or kind in ('unavailable', 'empty'):
            if process.poll() is not None: return None
            raise RuntimeError('Owned evaluator identity unavailable after verified startup')
        if kind != 'final': raise RuntimeError('Owned evaluator returned to gateway after final exec')
    return kind


def wait_evaluator_startup(process, receipt, parent, gateway, command, launch_path, declaration,
                           *, persist=lambda: None, interrupted=lambda: False):
    """Require observed final exec AND fresh nonce registration within15s."""
    receipt.update(pid=process.pid, process_identity={}, status='starting')
    persist()  # Durable Popen PID before its first readable /proc sample.
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        if interrupted(): raise InterruptedError('Interrupted during evaluator startup handshake')
        if process.poll() is not None:
            raise RuntimeError('Owned evaluator exited before final exec/nonce startup handshake')
        kind = observe_evaluator(process, receipt, parent, gateway, command, startup=True)
        if (kind == 'final' and receipt.get('process_identity', {}).get('start_time') is not None and
                evaluator_handshake(launch_path, declaration, receipt)):
            receipt.update(status='running', startup_verified=True)
            persist()
            return
        time.sleep(.025)
    raise TimeoutError('Owned evaluator final exec/nonce startup handshake did not arrive within15s')


def exec_evaluator(launch_path):
    """Fresh interpreter only: validate ownership, arm PDeath, register, exec."""
    path = Path(launch_path).resolve(strict=True)
    launch = read(path)
    parent = launch.get('parent')
    if (launch.get('owner') != OWNER or launch.get('boot_id') != Path('/proc/sys/kernel/random/boot_id').read_text().strip() or
            not parallel.proc.same_process(parent, parallel.proc.process_info(os.getppid())) or
            os.getppid() != parent.get('pid') or launch.get('script_sha256') != sha(CODE_ROOT / SCRIPT)):
        raise ValueError('Fresh evaluator gateway parent/source ownership differs')
    if (launch.get('execution_environment') != EVALUATION_ENV or
            any(os.environ.get(name) != value for name, value in EVALUATION_ENV.items())):
        raise ValueError('Fresh evaluator CPU/tokenizer thread environment differs from the proved profile')
    command = launch.get('command', [])
    if (len(command) < 3 or command[:3] != [sys.executable, '-u', str(CODE_ROOT / parallel.SCRIPT)] or
            digest(command) != launch.get('command_sha256')):
        raise ValueError('Fresh evaluator gateway destination command differs')
    run = Path(argv_value(command, '--run')).resolve(strict=True)
    variant = argv_value(command, '--image-ablation')
    workers = int(argv_value(command, '--workers'))
    if path.parent != parallel.output_path(run, variant, workers=workers).parent / 'controller_launches':
        raise ValueError('Fresh evaluator gateway launch escaped the owned run namespace')
    parallel.proc.arm_parent_death(parent['pid'])
    if not parallel.proc.same_process(parent, parallel.proc.process_info(parent['pid'])):
        raise RuntimeError('Controller changed before evaluator exec')
    registration = {'owner': OWNER, 'nonce': launch['nonce'], 'launch_sha256': sha(path),
                    'command_sha256': digest(command), 'identity': parallel.proc.process_info(os.getpid())}
    _atomic_json(path.with_suffix('.registration.json'), registration)
    os.execv(command[0], command)


class ParallelGroundingResearchQueue(native.GroundingResearchQueue):
    def __init__(self, spec_path, state_path=None, retry_failed=False, *, proof_run,
                 workers=4, timeout_seconds=1200):
        if workers not in (2, 4) or type(workers) is not int or not 120 <= timeout_seconds <= 1800:
            raise ValueError('Explicit two/four workers and bounded timeout120..1800 required')
        super().__init__(spec_path, state_path, retry_failed)
        try:
            self.workers = workers
            self.timeout_seconds = timeout_seconds
            self.parallel_child_identity = None
            assert_no_live_workers(self.spec)
            manifests = {str(resolve_path(job['evaluation']['manifest'])) for job in self.spec['jobs'] if job.get('evaluation')}
            if len(manifests) != 1:
                raise ValueError('Parallel controller requires one unchanged full512 manifest')
            self.adoption = adoption_evidence(proof_run, next(iter(manifests)), workers)
            descriptor = {'schema_version': 1, 'owner': OWNER, 'workers': workers,
                          'spec_sha256': self.spec_hash, 'script_sha256': sha(CODE_ROOT / SCRIPT),
                          'parent_queue_sha256': sha(CODE_ROOT / 'scripts/research_queue.py'),
                          'native_grounding_sha256': sha(CODE_ROOT / 'scripts/research_grounding_queue.py'),
                          'evaluator_script': parallel.SCRIPT, 'evaluator_owner': parallel.OWNER,
                          'evaluation_environment': dict(EVALUATION_ENV),
                          'adoption_evidence': self.adoption,
                          'pending_profile': 'SmallThinker/SigLIP2 QLoRA only; native batch1 per process',
                          'full_training_profile': 'unchanged serial native evaluator until separate fit proof'}
            previous = self.state.get('parallel_controller')
            if previous is not None and previous != descriptor:
                raise ValueError('Parallel execution proof/worker/source identity changed; controlled migration is required')
            self.state['parallel_controller'] = descriptor
        except BaseException:
            self.lock_handle.close()
            raise

    def persist(self):
        for record in self.state.get('jobs', {}).values():
            if record.get('failure_phase') == 'parallel_evaluation' and record.get('parallel_status') in ('failed', 'interrupted'):
                record['status'] = 'training_completed_parallel_eval_' + record['parallel_status']
        super().persist()

    def _parallel_job(self, job):
        if argv_value(job['argv'], '--stage') != 'SFT' or argv_value(job['argv'], '--recipe') not in parallel.ALLOWED_RECIPES:
            return False
        config = read(output_dir(job['argv']) / 'resolved_config.json')
        return (config.get('model') == 'Tiiny/SmallThinker-4BA0.6B-Instruct' and
                config.get('vision') == 'google/siglip2-so400m-patch14-384')

    def _verify_execution_sources(self):
        expected = self.state['parallel_controller']
        for name, field in ((SCRIPT, 'script_sha256'), ('scripts/research_queue.py', 'parent_queue_sha256'),
                            ('scripts/research_grounding_queue.py', 'native_grounding_sha256')):
            if sha(CODE_ROOT / name) != expected[field]:
                raise ValueError('Loaded controller execution source changed: ' + name)
        for name, proof in self.adoption.get('readiness', {}).items():
            if sha(CODE_ROOT / name) != proof['sha256']:
                raise ValueError('Controller/evaluator source proof changed during execution')
            verify_sources(CODE_ROOT, proof['source_sha256'], ())

    def _native_identity(self, job, variant):
        return self._evaluation_identity(job) if variant == 'none' else self._grounding_identity(job, variant)

    def _audit_native(self, job, variant, receipt):
        identity = self._native_identity(job, variant)
        verified = self._validate_evaluation(job, identity) if variant == 'none' else self._validate_grounding(job, identity, variant)
        # The baseline parent checker predates strict per-row metadata/token
        # checks. Use the released native-cache checker as a second CPU audit.
        run = output_dir(job['argv'])
        context = parallel.prepare_context(run, identity['manifest'], 'none',
                    parallel.output_path(run, 'none', workers=self.workers), self.workers, self.spec.get('reserve_gb', 120))
        if variant != 'none':
            context.update(records=identity['transformed_records'], variant=variant,
                image_ablation=identity['image_ablation'], effective_manifest=str(identity['output'] / 'transformed_manifest.jsonl'),
                transformed_manifest_sha256=identity['image_ablation']['transformed_manifest_sha256'])
        audit = parallel.module().validate_native_cache(identity['output'], context['records'],
                        parallel.protocol_for(context, self.workers), scope='full512')
        return {'status': 'completed', 'execution_schema': 'native_serial_preserved', 'output': str(identity['output']),
                'native_receipt_sha256': digest(receipt), 'native_strict_binding': audit['binding'],
                'checkpoint_sha256': verified['checkpoint_sha256'],
                'checkpoint_artifact_identity': identity['checkpoint_artifact_identity'],
                'manifest_sha256': verified['manifest_sha256'], 'protocol_sha256': verified['protocol_sha256'],
                'records': 512, 'group_macro_score': verified['group_macro_score'], 'variant': variant}

    def _audit_parallel(self, job, variant):
        run = output_dir(job['argv'])
        output = parallel.output_path(run, variant, workers=self.workers)
        audit = parallel.validate_completed_output(run, resolve_path(job['evaluation']['manifest']), variant, self.workers)
        if (audit.get('owner') != parallel.OWNER or audit.get('status') != 'passed' or
                audit.get('records') != 512 or audit.get('worker_count') != self.workers or
                audit.get('per_model_native64_guard_verified') is not True or
                audit.get('raw_predictions_sha256') != sha(output / 'predictions.jsonl') or
                audit.get('summary_sha256') != sha(output / 'summary.json')):
            raise ValueError('Parallel raw512 producer/auditor schema or artifact digest differs')
        return {'status': 'completed', 'execution_schema': 'parallel_native_b1_v1',
                'execution_script': parallel.SCRIPT, 'execution_owner': parallel.OWNER,
                'output': str(output), 'summary': str(output / 'summary.json'), 'variant': variant, 'seed': 42,
                'records': 512, 'worker_count': self.workers, 'checkpoint_sha256': audit['checkpoint_identity']['sha256'],
                'checkpoint_artifact_identity': audit['checkpoint_identity'],
                'manifest_sha256': sha(resolve_path(job['evaluation']['manifest'])),
                'protocol_sha256': audit['protocol_sha256'], 'group_macro_score': audit['group_macro_score'],
                'raw512_audit': audit, 'controller_script': SCRIPT, 'controller_sha256': sha(CODE_ROOT / SCRIPT)}

    def _assert_no_orphan(self, previous):
        pid = previous.get('pid')
        identity = previous.get('process_identity')
        launch = previous.get('launch')
        if launch and Path(launch).with_suffix('.registration.json').is_file():
            registered = read(Path(launch).with_suffix('.registration.json'))
            declaration = read(launch)
            if (registered.get('owner') != OWNER or registered.get('nonce') != declaration.get('nonce') or
                    registered.get('launch_sha256') != sha(launch) or registered.get('command_sha256') != digest(declaration['command'])):
                raise ValueError('Orphan evaluator registration ownership changed')
            if identity is None:
                identity = registered['identity']
                pid = identity['pid']
        if pid is not None:
            info = parallel.proc.process_info(pid)
            if info and info.get('state') not in ('Z', 'X'):
                # A PID may transition gateway->exec while birth remains fixed.
                if identity and info.get('start_time') != identity.get('start_time'):
                    raise RuntimeError('Previous evaluator PID was reused; no process is signaled or new worker launched')
                raise RuntimeError('Previous evaluator PID is still live; no duplicate or foreign signal')

    def _validate_partial_journal(self, job, variant, output):
        """Bind partial protocol/commits before the CLI can allocate any model."""
        run = output_dir(job['argv'])
        context = parallel.prepare_context(run, resolve_path(job['evaluation']['manifest']), variant,
                                            output, self.workers, self.spec.get('reserve_gb', 120))
        protocol = read(output / 'protocol.json')
        declared = protocol.get('reference_binding', {})
        if not isinstance(declared, dict) or declared.get('scope') not in ('full512', 'guard64'):
            raise ValueError('Partial parallel journal has no immutable native reference binding')
        reference_dir = Path(declared.get('directory', '')).resolve(strict=True)
        reference = parallel.module().validate_native_cache(reference_dir, context['records'],
                        parallel.protocol_for(context, self.workers), scope=declared['scope'])
        expected = parallel.protocol_for(context, self.workers, reference_binding=reference['binding'])
        if protocol != expected:
            raise ValueError('Partial parallel checkpoint/manifest/variant/source/worker protocol differs')
        committed = parallel.module().read_committed(output, context['records'], expected)
        lookup = {row['record_id']: row for row in reference['rows']}
        reference_rows = {str(index): lookup[context['records'][index]['record_id']]
                          for index in parallel.module().select_guard_indices(context['records'])}
        if committed and parallel.validated_parallel_guard(output, context, expected, reference_rows) is None:
            raise ValueError('Partial committed rows lack a preceding verified per-model native64 guard')
        execution = output / 'execution'
        failed_attempts = []
        completed_attempts = []
        if execution.exists():
            for attempt in execution.iterdir():
                if not attempt.is_dir() or not (attempt / 'receipt.json').is_file():
                    raise ValueError('Partial execution attempt lacks owned terminal receipt; no automatic repair')
                status = read(attempt / 'receipt.json').get('status')
                if status not in ('completed', 'failed'):
                    raise ValueError('Partial execution still running/unverified; no duplicate or automatic repair')
                (completed_attempts if status == 'completed' else failed_attempts).append(attempt)
        if failed_attempts and committed and reference['scope'] != 'full512':
            covered = set()
            for attempt in completed_attempts:
                receipt = parallel.validate_cohort_attempt(attempt, context, expected)
                for member in receipt['cohort']:
                    covered.update(read(attempt / f"worker{member['logical_worker']}.result.json")['global_indices'])
            if not set(committed) <= covered:
                raise ValueError('Frozen evaluator cannot certify failed-attempt committed rows against only native64; preserve journal for explicit recovery proof')
        return {'records_committed': len(committed), 'reference_scope': reference['scope'],
                'protocol_sha256': parallel.module().protocol_sha256(expected)}

    def _launch_parallel(self, job, record, variant, previous):
        self._verify_execution_sources()
        if self.interrupted_by:
            raise InterruptedError('Interrupted before parallel evaluator preparation')
        self._assert_no_orphan(previous)
        self.budget.ensure_space(2 * 2 ** 30)
        run = output_dir(job['argv'])
        output = parallel.output_path(run, variant, workers=self.workers)
        command = [sys.executable, '-u', str(CODE_ROOT / parallel.SCRIPT), '--run', str(run),
                   '--manifest', str(resolve_path(job['evaluation']['manifest'])), '--image-ablation', variant,
                   '--workers', str(self.workers), '--allow-gpu', '--reserve-gb', str(self.spec.get('reserve_gb', 120)),
                   '--timeout-seconds', str(self.timeout_seconds)]
        if output.exists():
            if not (output / 'protocol.json').is_file():
                raise ValueError('Existing parallel output has no immutable protocol; no overwrite or cleanup')
            self._validate_partial_journal(job, variant, output)
            command.append('--resume')
        launch_dir = output.parent / 'controller_launches'
        launch_dir.mkdir(parents=True, exist_ok=True)
        launch_path = launch_dir / (uuid.uuid4().hex + '.json')
        parent = parallel.proc.process_info(os.getpid())
        declaration = {'owner': OWNER, 'nonce': uuid.uuid4().hex, 'parent': parent,
                       'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                       'command': command, 'command_sha256': digest(command), 'script_sha256': sha(CODE_ROOT / SCRIPT),
                       'execution_environment': dict(EVALUATION_ENV)}
        _atomic_json(launch_path, declaration)
        receipt = {'status': 'starting', 'execution_schema': 'parallel_native_b1_v1', 'command': command,
                   'output': str(output), 'launch': str(launch_path), 'started': timestamp(),
                   'execution_environment': dict(EVALUATION_ENV),
                   'log': str(self.log_dir / f"{job['id']}.parallel-{variant}.log")}
        record.setdefault('parallel_evaluations', {})[variant] = receipt
        record.update(status='parallel_evaluating', active_parallel_variant=variant, failure_phase='parallel_evaluation')
        self.persist()
        gateway = [sys.executable, '-u', str(CODE_ROOT / SCRIPT), '--exec-evaluator', str(launch_path)]
        with Path(receipt['log']).open('a', buffering=1) as handle:
            handle.write(json.dumps({'time': timestamp(), 'command': command, 'gateway': gateway}) + '\n')
            if self.interrupted_by:
                raise InterruptedError('Interrupted before parallel evaluator spawn')
            started = time.monotonic()
            self.process = subprocess.Popen(gateway, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, **EVALUATION_ENV})
            wait_evaluator_startup(self.process, receipt, parent, gateway, command, launch_path, declaration,
                                   persist=self.persist, interrupted=lambda: bool(self.interrupted_by))
            heartbeat = time.monotonic()
            interrupted_at = None
            while self.process.poll() is None:
                kind = observe_evaluator(self.process, receipt, parent, gateway, command, startup=False)
                if kind is None and self.process.poll() is not None: break
                if self.interrupted_by:
                    interrupted_at = interrupted_at or time.monotonic()
                    if time.monotonic() - interrupted_at > 30:
                        os.killpg(self.process.pid, signal.SIGKILL)
                if time.monotonic() - started > 2 * self.timeout_seconds + 120:
                    # A previously absent serial64 reference may precede the
                    # full cohort; each is bounded by the frozen CLI timeout.
                    raise TimeoutError('Bounded serial64+parallel evaluator controller deadline expired')
                if time.monotonic() - heartbeat >= 30:
                    self.budget.ensure_space(0)
                    emit('parallel_evaluation_running', job=job['id'], variant=variant, pid=self.process.pid,
                         worker_count=self.workers, free_gib=self.budget.report()['free_gib'])
                    self.persist(); heartbeat = time.monotonic()
                time.sleep(1)
            receipt.update(returncode=self.process.returncode, finished=timestamp(), wall_seconds=time.monotonic() - started)
            self.process = None
        if self.interrupted_by:
            raise InterruptedError('Interrupted during parallel evaluation')
        if receipt['returncode'] != 0:
            raise RuntimeError(f"Parallel evaluator returned {receipt['returncode']}; no implicit retry; inspect {receipt['log']}")
        self._verify_execution_sources()
        return receipt

    def _view(self, job, record, variant):
        holder = record.get('evaluation', {}) if variant == 'none' else record.get('grounding_evaluations', {}).get(variant, {})
        previous = record.setdefault('parallel_evaluations', {}).get(variant, {})
        if holder.get('status') == 'completed' and holder.get('execution_schema') != 'parallel_native_b1_v1':
            return self._audit_native(job, variant, holder)
        output = parallel.output_path(output_dir(job['argv']), variant, workers=self.workers)
        self._assert_no_orphan(previous)
        if (output / 'execution_summary.json').is_file():
            # Child succeeded immediately before controller death: re-audit,
            # never repeat GPU work or silently accept a corrupt final view.
            verified = {**previous, **self._audit_parallel(job, variant)}
        else:
            execution = self._launch_parallel(job, record, variant, previous)
            verified = {**execution, **self._audit_parallel(job, variant)}
        record['parallel_evaluations'][variant] = verified
        if variant == 'none':
            if holder and holder.get('execution_schema') != 'parallel_native_b1_v1':
                record.setdefault('native_partial_evaluation', copy.deepcopy(holder))
            record['evaluation'] = copy.deepcopy(verified)
        else:
            if holder and holder.get('execution_schema') != 'parallel_native_b1_v1':
                record.setdefault('native_partial_grounding', {}).setdefault(variant, copy.deepcopy(holder))
            record.setdefault('grounding_evaluations', {})[variant] = copy.deepcopy(verified)
        self.persist()
        return verified

    def _write_paired(self, job, record, views):
        original = native.read_rows(resolve_path(job['evaluation']['manifest']))
        rows = {variant: native.read_rows(Path(view['output']) / 'predictions.jsonl') for variant, view in views.items()}
        expected = {row['record_id'] for row in original}
        if len(expected) != 512 or any(len(v) != 512 or len({r['record_id'] for r in v}) != 512 or
                                      {r['record_id'] for r in v} != expected for v in rows.values()):
            raise ValueError('Paired bootstrap requires actual complete unique512 rows in every view')
        summary = {'schema_version': 1, 'owner': OWNER, 'status': 'completed',
                   'scope': 'held-out512 paired grounding diagnostics; native/parallel executions explicitly identified',
                   'original_manifest_sha256': sha(resolve_path(job['evaluation']['manifest'])), 'records': 512,
                   'views': views, 'raw_predictions_sha256': {v: sha(Path(x['output']) / 'predictions.jsonl') for v, x in views.items()},
                   'checkpoint_validation': {key: value for key, value in record['checkpoint_validation'].items()
                       if key not in ('verified', 'started', 'finished', 'wall_seconds')},
                   'paired_image_group_bootstrap': native.paired_image_bootstrap(rows['none'], {v: rows[v] for v in native.MODES}, original),
                   'controller_source_sha256': sha(CODE_ROOT / SCRIPT), 'execution_profile': self.state['parallel_controller']}
        destination = parallel.output_path(output_dir(job['argv']), 'none', workers=self.workers).parent / 'paired_summary.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and read(destination) != summary:
            raise ValueError('Existing parallel paired summary differs; no provenance overwrite')
        if not destination.exists():
            _atomic_json(destination, summary)
        # Original native paired report remains at its original path and is
        # never replaced by the new mixed-execution diagnostic.
        record['parallel_grounding_summary'] = str(destination)
        return summary

    def evaluate_completed_job(self, job, record):
        self._verify_execution_sources()
        if not self._parallel_job(job):
            record.setdefault('evaluation_execution_profile', 'unchanged_native_serial_full_or_other_model')
            return super().evaluate_completed_job(job, record)
        if ((record.get('parallel_status') == 'failed' or record.get('grounding_status') == 'failed' or
             record.get('checkpoint_validation', {}).get('status') == 'failed') and not self.retry_failed):
            raise RuntimeError('Post-training evaluation previously failed; inspect evidence and use --retry-failed')
        record.update(training_completed=True, parallel_status='running', failure_phase='parallel_evaluation')
        self.persist()
        try:
            views = {'none': self._view(job, record, 'none')}
            if self.interrupted_by:
                raise InterruptedError('Interrupted before retained checkpoint replay')
            if not self._validate_completed_checkpoints(job, record):
                return False
            record.update(failure_phase='parallel_evaluation', status='parallel_evaluating')
            for variant in native.MODES:
                if self.interrupted_by:
                    raise InterruptedError('Interrupted before next pending image view')
                record['active_parallel_variant'] = variant
                views[variant] = self._view(job, record, variant)
                if views[variant]['checkpoint_sha256'] != views['none']['checkpoint_sha256']:
                    raise ValueError('All grounding views must bind the same verified BEST checkpoint')
            self._verify_execution_sources()
            self._write_paired(job, record, views)
            kinds = {view['execution_schema'] for view in views.values()}
            profile = ('preserved_native_serial_views' if kinds == {'native_serial_preserved'} else
                       'explicit_parallel_native_b1_v1' if kinds == {'parallel_native_b1_v1'} else
                       'mixed_preserved_native_and_explicit_parallel_b1_v1')
            record.update(parallel_status='completed', grounding_status='completed', failure_phase=None,
                          evaluation_execution_profile=profile,
                          parallel_view_audits=views)
            self.persist()
            return True
        except Exception as error:
            self._stop_owned_child()
            if record.get('failure_phase') == 'checkpoint_validation':
                raise
            interrupted = bool(self.interrupted_by)
            record.update(parallel_status='interrupted' if interrupted else 'failed', failure_phase='parallel_evaluation',
                          error_type=type(error).__name__, error=str(error), finished=timestamp())
            self.persist()
            emit('parallel_grounding_failed', job=job['id'], error=str(error), status=record['status'])
            if interrupted:
                return False
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec'); parser.add_argument('--state')
    parser.add_argument('--proof-run'); parser.add_argument('--workers', type=int, choices=(2, 4), default=4)
    parser.add_argument('--timeout-seconds', type=int, default=1200)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--exec-evaluator', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.exec_evaluator:
        return exec_evaluator(args.exec_evaluator)
    if not args.spec or not args.proof_run:
        parser.error('Same immutable --spec and the actual --proof-run are required')
    queue = ParallelGroundingResearchQueue(args.spec, args.state, args.retry_failed, proof_run=args.proof_run,
                                          workers=args.workers, timeout_seconds=args.timeout_seconds)
    signal.signal(signal.SIGTERM, queue.signal); signal.signal(signal.SIGINT, queue.signal)
    return queue.run()


if __name__ == '__main__':
    raise SystemExit(main())
