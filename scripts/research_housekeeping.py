#!/usr/bin/env python3
"""Explicitly scoped technical-smoke housekeeping; retain all final model tensors."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compact_research_checkpoints import compact_completed_checkpoint
from scripts.research_queue import prune_checkpoint_run
from src.utils.research_storage import DiskBudget

PATTERN = re.compile(r'^smallthinker4__siglip2__(4f|4q|[123][abc])__SFT_smoke_s42$')


def emit(event, **fields):
    print(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reserve-gb', type=float, default=120)
    p.add_argument('--interval', type=float, default=15)
    p.add_argument('--queue-state', default='configs/research/smallthinker_smoke_queue.state.json')
    args = p.parse_args()
    processed = set()
    while True:
        candidates = sorted((path.parent for path in (ROOT/'runs/research').glob('*/completed.json')
                             if PATTERN.fullmatch(path.parent.name)), key=lambda directory: (directory/'completed.json').stat().st_mtime)
        for directory in candidates:
            if str(directory) in processed:
                continue
            cfg = json.loads((directory/'resolved_config.json').read_text())
            if cfg.get('stage') != 'SFT' or not isinstance(cfg.get('max_records'), int) or not 0 < cfg['max_records'] <= 2000 or not isinstance(cfg.get('max_steps'), int) or not 0 < cfg['max_steps'] <= 100:
                raise RuntimeError(f'Refusing nontechnical run: {directory}')
            emit('compaction_started', run_dir=str(directory), disk=DiskBudget(ROOT,args.reserve_gb).report())
            result = compact_completed_checkpoint(directory, reserve_gb=args.reserve_gb)
            emit('compaction_completed_model_weights_retained', **result)
            initial = cfg.get('initial_checkpoint')
            if initial:
                source = Path(initial)
                source = ROOT/source if not source.is_absolute() else source
                # Keep lexical last/state.pt path: resolving it first would point
                # into .checkpoint_objects and lose the source run directory.
                source = source.absolute().parent.parent
                expected_name = directory.name.replace('__SFT_smoke_s42', '__A1_smoke_s42')
                if source.parent != directory.parent or source.name != expected_name:
                    raise RuntimeError(f'Refusing unexpected intermediate source: {source}')
                pruned = prune_checkpoint_run(source, [directory])
                emit('intermediate_A1_checkpoint_pruned', run_dir=str(source), **pruned)
            processed.add(str(directory))
        emit('housekeeping_poll', completed_technical_sft_runs=len(processed), disk=DiskBudget(ROOT,args.reserve_gb).report())
        state_path = ROOT/args.queue_state
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get('status') == 'completed':
                emit('housekeeping_finished', queue_status=state['status'], final_model_checkpoints_retained=True)
                return
        if len(processed) == 11:
            emit('housekeeping_finished', final_model_checkpoints_retained=True)
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
