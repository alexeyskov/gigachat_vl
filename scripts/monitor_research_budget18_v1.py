#!/usr/bin/env python3
"""Read-only, process-bound progress for the reduced 18-experiment campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.monitor_research import gpu_snapshot, ram_snapshot, with_progress_eta
from src.utils.research_storage import DiskBudget, _atomic_json

MANUAL = {'run_research_small_budget_v1.py', 'run_research_stock_deterministic.py',
          'run_research_lfm_budget_v1.py', 'run_research_giga_budget_v1.py'}


def read(path):
    path = Path(path)
    if path.stat().st_size > 16 * 2**20:
        raise ValueError('Progress metadata exceeds the bounded read limit')
    return json.loads(path.read_bytes())


def option(argv, flag):
    positions = [i for i, value in enumerate(argv[:-1]) if value == flag]
    if len(positions) > 1:
        raise ValueError('Duplicate process argument')
    return argv[positions[0] + 1] if positions else None


def identity(pid):
    base = Path('/proc') / str(pid)
    fields = (base / 'stat').read_text().rsplit(') ', 1)[1].split()
    if fields[0] in ('Z', 'X'):
        raise ProcessLookupError(pid)
    return {'pid': pid, 'birth': int(fields[19]), 'ppid': int(fields[1]),
            'argv': [v.decode() for v in (base / 'cmdline').read_bytes().split(b'\0') if v]}


def last_event(path):
    path = Path(path)
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - 2**20))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            event = json.loads(line)
            if isinstance(event, dict):
                return event
        except (ValueError, UnicodeError):
            continue
    return {}


def activities():
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            process = identity(int(entry.name))
            argv = process['argv']
            scripts = {Path(a).name for a in argv}
            if scripts & MANUAL:
                kind = 'train'
                forwarded = argv
            elif scripts.intersection({'research_budget18_queue_v1.py','research_budget18_queue_v2.py','research_budget18_queue_v3.py','research_budget18_queue_v4.py','research_budget18_queue_v5.py','research_budget18_queue_v6.py','research_budget18_queue_v7.py'}) and option(argv, '--worker-request'):
                path = Path(option(argv, '--worker-request')).resolve(strict=True)
                if not path.is_relative_to(ROOT / 'logs/research/budget18_workers'):
                    continue
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != option(argv, '--request-sha256'):
                    continue
                request = read(path)
                ready = read(path.with_name('worker_ready.json'))
                saved = ready['worker']
                parent = identity(process['ppid'])
                if (request.get('owner') != 'research_budget18_queue_v1'
                        or ready.get('nonce') != request.get('nonce')
                        or any(saved.get(k) != process[k] for k in ('pid', 'birth', 'argv'))
                        or any(request['parent'].get(k) != parent[k] for k in ('pid', 'birth', 'argv'))):
                    continue
                kind, forwarded = request['kind'], request['argv']
            else:
                continue
            destination = option(forwarded, '--output' if kind == 'train' else '--run')
            if not destination:
                continue
            run = (ROOT / destination).resolve()
            if run.parent != ROOT / 'runs/research' or not run.name.endswith('__budget18_s42'):
                continue
            current = {'run_id': run.name, 'phase': kind, 'activity_verified': True,
                       'process': process, 'source': 'live_process_birth_and_command'}
            if (run / 'resolved_config.json').is_file():
                config = read(run / 'resolved_config.json')
                current.update(stage=config.get('stage'), model=config.get('model'),
                               vision=config.get('vision'), recipe=config.get('recipe'),
                               stage_disk_reserve_gib=config.get('reserve_gb'))
            progress = run / 'metrics.jsonl' if kind == 'train' else (
                Path(option(forwarded, '--output')) / 'progress.json' if kind == 'eval' else None)
            if progress and progress.is_file():
                current.update(last_event(progress) if kind == 'train' else read(progress))
                current.update(run_id=run.name, phase=kind, activity_verified=True,
                               progress_source=str(progress), file_age_seconds=time.time() - progress.stat().st_mtime)
            if kind == 'train' and (run / 'selection_progress.json').is_file():
                selection = run / 'selection_progress.json'
                if not progress or selection.stat().st_mtime > progress.stat().st_mtime:
                    current.update(phase='internal_selection', selection=read(selection))
            found.append(with_progress_eta(current))
        except (OSError, ValueError, KeyError, TypeError, UnicodeError):
            continue
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--reserve-gb', type=float, default=120)
    args = parser.parse_args()
    if not 0 < args.interval <= 60:
        parser.error('Interval must be greater than zero and at most 60 seconds')
    output = ROOT / 'logs/research/budget18_resources.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while running:
        start = time.monotonic()
        disk = DiskBudget(ROOT, args.reserve_gb).report()
        status = {'time': time.time(), 'owner': 'budget18_process_monitor_v1',
                  'activities': activities(), 'disk': disk, 'ram': ram_snapshot(),
                  'gpus': gpu_snapshot(), 'total_configurations': 18}
        status['disk_reserve_intact'] = disk['free_bytes'] >= disk['reserve_bytes']
        state_path = ROOT / 'runs/research/budget18_queue_state_v1.json'
        if state_path.is_file():
            state = read(state_path)
            status['completed_configurations'] = sum(
                name.endswith('__SFT__budget18_s42') and row.get('status') == 'completed'
                for name, row in state.get('jobs', {}).items())
            status['queue_status'] = state.get('status')
        else:
            status['completed_configurations'] = 0
        with output.open('a') as handle:
            handle.write(json.dumps(status, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
        _atomic_json(output.with_name('budget18_resources_latest.json'), status)
        print(json.dumps(status, ensure_ascii=False, allow_nan=False), flush=True)
        if args.once:
            return
        while running and time.monotonic() - start < args.interval:
            time.sleep(min(1, args.interval - (time.monotonic() - start)))


if __name__ == '__main__':
    main()
