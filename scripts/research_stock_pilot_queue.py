#!/usr/bin/env python3
"""Continue independent pilot branches under the explicit stock numerical profile.

The original queue, completed runs and failed legacy replay remain untouched.
Native lifecycle, compaction, scoring and strict replay checks are reused.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import signal
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import research_stock_determinism as stock

PINS = {
    'scripts/research_queue.py': '446d928181aba5f77d7e02692279741eb10d8fc3d3409af985e07d9239263bda',
    'scripts/research_grounding_queue.py': 'b99dfd413edd4176288734262197a72b3df4720e01d562e4b76f625b70d0a1d9',
}
FACADES = {
    'run_research_stage.py': 'run_research_stock_deterministic.py',
    'eval_research_checkpoint.py': 'eval_research_stock_deterministic.py',
    'validate_research_checkpoint.py': 'validate_research_stock_deterministic.py',
}
PROOF = ROOT/'research_results/selection_deterministic_3b_best170_v2/report.json'
ORIGINAL_SPEC = ROOT/'configs/research/smallthinker_pilot_scienceqa_context_v2_balanced_queue.json'
ORIGINAL_SPEC_SHA = 'c4c8181b032538f7d382c10e17af5d826141fae0df18a420dd891d55b75a7442'
RECIPES = {'3a', '3c', '2a', '2b', '2c'}


class MaintenanceBoundary(BaseException):
    """A scheduled idle boundary, with its state persisted while still locked."""


def read(path):
    return json.loads(Path(path).read_bytes())


def command_profile(command):
    result = list(command)
    if len(result) < 3 or Path(result[2]).name not in FACADES:
        raise ValueError('Unexpected executable in stock pilot queue')
    result[2] = str(ROOT/'scripts'/FACADES[Path(result[2]).name])
    result[3:3] = ['--numerical-profile', stock.PROFILE, '--allow-gpu']
    return result


def continuation_jobs():
    raw = ORIGINAL_SPEC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_SPEC_SHA:
        raise ValueError('Original immutable pilot specification changed')
    selected = [j for j in json.loads(raw)['jobs'] if any('__'+r+'__' in j['id'] for r in RECIPES)]
    names = {j['id']: j['id']+'__stockdet_v1' for j in selected}
    def rewrite(value):
        if isinstance(value, str):
            for before, after in names.items():
                value = value.replace(before, after)
            return value
        if isinstance(value, list):
            return [rewrite(x) for x in value]
        if isinstance(value, dict):
            return {k: rewrite(v) for k, v in value.items()}
        return value
    result = []
    for old in selected:
        job = rewrite(old)
        job['source_job_id'] = old['id']
        if job['argv'][job['argv'].index('--stage')+1] == 'A1':
            job['requires'] = []  # The separately pinned external A0 is already complete.
            job['argv'].append('--allow-legacy-a0-initialization')
        result.append(job)
    return result


class ExecutionHooks(ast.NodeTransformer):
    def __init__(self):
        self.commands = 0
        self.boundaries = 0

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(x, ast.Name) and x.id == 'command' for x in node.targets) and isinstance(node.value, ast.List):
            node.value = ast.Call(ast.Name('_stock_command', ast.Load()), [node.value], [])
            self.commands += 1
        return node

    def visit_For(self, node):
        node = self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.target.id == 'job' and ast.unparse(node.iter) == "self.spec['jobs']":
            # Only the run loop has interrupted_by as its first condition.
            if node.body and isinstance(node.body[0], ast.If) and 'self.interrupted_by' in ast.unparse(node.body[0].test):
                node.body.insert(0, ast.parse('self._stock_boundary(job)').body[0])
                self.boundaries += 1
        return node


def private_queues():
    modules = {}
    for name, pin in PINS.items():
        raw = (ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError('Frozen queue source changed: '+name)
        module = ModuleType('_stock_private_'+Path(name).stem)
        module.__file__ = str(ROOT/name)
        module.__dict__['_stock_command'] = command_profile
        tree = ast.parse(raw, filename=str(ROOT/name))
        if modules:
            imports = stock._Imports({'scripts.research_queue': '_private_base'})
            module.__dict__['_private_base'] = modules['base']
            tree = imports.visit(tree)
        hook = ExecutionHooks()
        tree = hook.visit(tree)
        expected = (2, 1) if not modules else (2, 0)
        if (hook.commands, hook.boundaries) != expected:
            raise ValueError('Native queue command/run-loop structure changed')
        ast.fix_missing_locations(tree)
        sys.modules[module.__name__] = module
        exec(compile(tree, str(ROOT/name), 'exec'), module.__dict__)
        modules['base' if not modules else 'grounding'] = module
    return modules


def verify_inputs(spec):
    if spec.get('numerical_profile') != stock.definition():
        raise ValueError('Queue numerical profile/source hashes changed')
    proof = read(PROOF)
    if (proof.get('owner') != 'smallthinker3b_stock_deterministic_diagnostic_v2'
        or proof.get('producer') != 'ROOT' or proof.get('status') != 'completed_diagnostic_only'
        or proof.get('three_pass_token_mismatch_records') != 0
        or proof.get('three_pass_prompt_mismatch_records') != 0
        or proof.get('actual116_completed') is not True
        or proof.get('source_inputs_unchanged') is not True
        or proof.get('runtime_environment', {}).get('deterministic_algorithms') is not True):
        raise ValueError('Actual complete three-pass stock numerical proof is required')
    if hashlib.sha256(PROOF.read_bytes()).hexdigest() != spec.get('numerical_calibration_sha256'):
        raise ValueError('Queue calibration proof digest differs')
    for item in spec.get('external_initializations', []):
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Pinned external A0 initialization changed')
    if len(spec.get('jobs', [])) != 10:
        raise ValueError('This continuation contains exactly five A1/SFT branch pairs')
    if spec['jobs'] != continuation_jobs():
        raise ValueError('Continuation must retain the exact five original branch pairs with new outputs and explicit A0 initialization')


def queue_class():
    modules = private_queues()
    parent = modules['grounding'].GroundingResearchQueue

    class StockPilotQueue(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                verify_inputs(self.spec)
                self.state['numerical_execution'] = {
                    'profile': stock.definition(), 'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'legacy_queue_unchanged': True, 'legacy_failed_3b_replay_unchanged': True,
                }
            except BaseException:
                self.lock_handle.close()
                raise
            self.stop_before_job = None

        def _stock_boundary(self, job):
            verify_inputs(self.spec)
            if job not in self.spec['jobs']:
                raise ValueError('Continuation received an unregistered job')
            if job['id'] == self.stop_before_job:
                self.state.update(status='awaiting_scheduled_maintenance', next_job=job['id'])
                self.persist()
                raise MaintenanceBoundary(job['id'])

        def _validate_evaluation(self, job, identity):
            result = super()._validate_evaluation(job, identity)
            if read(identity['output']/'protocol.json').get('numerical_profile') != stock.definition():
                raise ValueError('Ordinary generation numerical profile differs')
            return result

        def _validate_grounding(self, job, identity, variant):
            result = super()._validate_grounding(job, identity, variant)
            if read(identity['output']/'protocol.json').get('numerical_profile') != stock.definition():
                raise ValueError('Grounding generation numerical profile differs')
            return result

        def _validate_checkpoint_report(self, identity):
            if read(identity['run']/'checkpoint_roundtrip.json').get('numerical_profile') != stock.definition():
                raise ValueError('Checkpoint replay numerical profile differs')
            result = super()._validate_checkpoint_report(identity)
            result['numerical_profile'] = stock.definition()
            result['execution_facade_sha256'] = hashlib.sha256((ROOT/'scripts/validate_research_stock_deterministic.py').read_bytes()).hexdigest()
            return result

    return StockPilotQueue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--run-queue', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--stop-before-job', help='Exit successfully at this idle job boundary for ROOT maintenance')
    args = parser.parse_args()
    spec = read(args.spec)
    verify_inputs(spec)
    queue_type = queue_class()
    if not args.run_queue:
        print(json.dumps({'status': 'metadata_preflight_only', 'jobs': len(spec['jobs']), 'jobs_started': False}))
        return 0
    queue = queue_type(args.spec, args.state, args.retry_failed)
    if args.stop_before_job and args.stop_before_job not in {j['id'] for j in spec['jobs']}:
        queue.lock_handle.close()
        raise ValueError('Scheduled maintenance boundary must name a job in this continuation')
    queue.stop_before_job = args.stop_before_job
    signal.signal(signal.SIGINT, queue.signal)
    signal.signal(signal.SIGTERM, queue.signal)
    try:
        return queue.run()
    except MaintenanceBoundary as boundary:
        print(json.dumps({'status': 'awaiting_scheduled_maintenance', 'next_job': str(boundary), 'GPU_child_running': False}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
