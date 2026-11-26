#!/usr/bin/env python3
"""Remove optimizer state only from explicitly completed SFT best/last payloads.

Active training checkpoints remain fully resumable. Compacted completed model
checkpoints retain model tensors, configuration, scheduler and RNG, and explicitly
say resumable_optimizer=false. New payloads are staged before atomic pointer
replacement; unrelated files are never deleted. Validation receipts for old object
paths become stale and should be regenerated after compaction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.research_storage import (
    BestLastCheckpointManager, DiskBudget, _atomic_json, _fsync_tree,
    _fsync_directory, _tree_size,
)


def _mapping(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _tensor_storage_bytes(value):
    """Count underlying CPU tensor storages, including views, exactly once."""
    import torch
    seen = set()
    def count(node):
        if isinstance(node, torch.Tensor):
            storage = node.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(node, dict):
            return sum(count(v) for v in node.values())
        if isinstance(node, (tuple, list)):
            return sum(count(v) for v in node)
        return 0
    return count(value)


def _source_digest(path):
    """Bind a retry's source to its previously published replacement."""
    before=path.stat();digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):
            digest.update(block)
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('Source checkpoint changed during compaction')
    return digest.hexdigest()


def compact_completed_checkpoint(run_dir, reserve_gb=120, dry_run=False):
    """Compact one unique object at a time, preserving exact best/last roles.

    Publish every role referencing the current source before reclaiming its
    optimizer and moving to the next object. A partial shared-pointer publish
    is recoverable through an owned, SHA-bound replacement, with no second copy.
    """
    import torch

    directory = Path(run_dir).expanduser().resolve()
    for name in ("resolved_config.json", "completed.json", ".checkpoint_manager.json"):
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Completed managed run requires local {name}: {directory}")
    config = _mapping(directory / "resolved_config.json")
    if config.get("stage") != "SFT":
        raise ValueError("Optimizer compaction is restricted to completed SFT stages")
    output = Path(config.get("output", "")).expanduser()
    output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    if output != directory:
        raise ValueError("Resolved config output does not match selected run")
    completion = _mapping(directory / "completed.json")
    if not isinstance(completion.get("steps"), int) or completion["steps"] < 1 or completion.get("status") in {"failed", "incomplete", "interrupted"}:
        raise ValueError("Run has no successful training completion marker")
    from src.utils.research_checkpoint_compat import checkpoint_kind
    if checkpoint_kind(directory/'last/state.pt')=='bundle_v1':
        from src.utils.research_checkpoint_bundle import compact_completed_bundle
        return compact_completed_bundle(directory,reserve_gb=reserve_gb,dry_run=dry_run)
    info = _mapping(directory / ".checkpoint_manager.json")
    objects = directory / ".checkpoint_objects"
    if objects.is_symlink() or not objects.is_dir() or objects.resolve().parent != directory:
        raise ValueError("Checkpoint objects must be a local directory inside this run")
    budget = DiskBudget(directory, reserve_gb)
    manager = BestLastCheckpointManager(directory, disk_budget=budget, mode=info["mode"])
    made = []
    report = {"schema_version": 1, "run_dir": str(directory), "started": datetime.now(timezone.utc).isoformat(),
              "training_completed": True, "resumable_optimizer": False, "checkpoints": []}
    with manager._lock():
        original = {name: manager._resolve_owned_pointer(name) for name in ("best", "last")}
        if original["last"] is None:
            raise ValueError("Completed SFT has no retained last checkpoint")
        last_info = _mapping(original["last"] / ".research_checkpoint.json")
        if last_info.get("step") != completion["steps"]:
            raise ValueError("Last checkpoint step does not match training completion")
        # Deterministic role order makes partial progress predictable. A shared
        # BEST/LAST source appears only once, even though both pointers move.
        unique=list(dict.fromkeys(target for target in original.values() if target is not None))
        checked={};previous_replacements={}
        for target in unique:
            checkpoint_info=_mapping(target/'.research_checkpoint.json')
            if checkpoint_info.get('created_unix',0)>(directory/'completed.json').stat().st_mtime+1:
                raise ValueError('A newer training checkpoint appeared after completed.json; refusing active/stale run')
            state_path=target/'state.pt'
            if not state_path.is_file() or state_path.is_symlink():
                raise ValueError('Checkpoint lacks a local state.pt payload')
            payload=torch.load(state_path,map_location='cpu',weights_only=False,mmap=True)
            if not isinstance(payload,dict) or not isinstance(payload.get('model'),dict) or not payload['model']:
                raise ValueError('Unexpected checkpoint payload; model tensors are required')
            if payload.get('step')!=checkpoint_info.get('step'):
                raise ValueError('Checkpoint payload/ownership marker step mismatch')
            compacted='optimizer' not in payload and payload.get('resumable_optimizer') is False and payload.get('training_completed') is True
            if 'optimizer' not in payload and not compacted:
                raise ValueError('Checkpoint has no optimizer but lacks completed-compaction metadata')
            checked[target]=(checkpoint_info,compacted)
            provenance=payload.get('checkpoint_compaction',{})
            source=checkpoint_info.get('source_checkpoint_object')
            if (compacted and source and provenance.get('source_object')==source and
                    isinstance(provenance.get('source_state_sha256'),str)):
                previous_replacements.setdefault(source,[]).append((target,provenance['source_state_sha256']))
            del payload
        # A crash after pointer publication can leave an old optimizer object or
        # unpublished staging object. Reclaim only this manager's unreferenced
        # objects before the next disk-budget check.
        owned_before=sum(_tree_size(entry) for entry in manager.objects.iterdir() if manager._owns(entry))
        if not dry_run:manager._cleanup_owned_orphans()
        owned_after_cleanup=sum(_tree_size(entry) for entry in manager.objects.iterdir() if manager._owns(entry))
        report['recovered_orphan_bytes']=owned_before-owned_after_cleanup
        try:
            for target in unique:
                checkpoint_info,already_compacted=checked[target]
                state_path=target/'state.pt'
                before=_tree_size(target)
                if already_compacted:
                    report['checkpoints'].append({'old_object':target.name,'new_object':target.name,
                        'already_compacted':True,'before_bytes':before,'after_bytes':before})
                    continue
                candidates=previous_replacements.get(target.name,[])
                if candidates:
                    source_digest=_source_digest(state_path)
                    matching=[candidate for candidate,digest in candidates if digest==source_digest]
                    if not matching:
                        raise ValueError('Source checkpoint differs from its published compacted replacement')
                    destination=matching[0]
                    new_info=_mapping(destination/'.research_checkpoint.json')
                    if any(new_info.get(key)!=checkpoint_info.get(key) for key in ('step','metric','owner')):
                        raise ValueError('Published compacted replacement has different checkpoint roles/metadata')
                    if dry_run:
                        report['checkpoints'].append({'old_object':target.name,'new_object':destination.name,
                            'before_bytes':before,'reuses_published_object':True,'anticipated_new_bytes':0})
                        continue
                    for name in ('best','last'):
                        if manager._resolve_owned_pointer(name)==target:manager._set_pointer(name,destination)
                    manager._cleanup_owned_orphans()
                    report['checkpoints'].append({'old_object':target.name,'new_object':destination.name,
                        'step':new_info['step'],'metric':new_info.get('metric'),'before_bytes':before,
                        'after_bytes':_tree_size(destination),'reuses_published_object':True})
                    continue
                # mmap avoids materializing optimizer moments that are about to
                # be removed, and avoids CPU-RAM copies of every model tensor.
                payload = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)
                # No dtype conversion, model-key filtering, RNG removal, or
                # scheduler removal is permitted here.
                del payload["optimizer"]
                payload["training_completed"] = True
                payload["resumable_optimizer"] = False
                payload["checkpoint_compaction"] = {"operation": "remove_completed_optimizer_only",
                                                     "source_object": target.name, "original_checkpoint_bytes": before, "time": time.time()}
                anticipated = _tensor_storage_bytes(payload) + 16 * 1024 ** 2
                budget.ensure_space(math.ceil(anticipated * 1.05))
                if dry_run:
                    report["checkpoints"].append({"old_object": target.name, "anticipated_new_bytes": anticipated,
                                                   "before_bytes": before, "would_remove_optimizer": True})
                    del payload
                    continue
                payload['checkpoint_compaction']['source_state_sha256']=_source_digest(state_path)
                token = uuid.uuid4().hex
                stage = manager.objects / f".stage-compact-{token}"
                destination = manager.objects / f"object-{token}"
                stage.mkdir()
                _atomic_json(stage / ".research_checkpoint.json", checkpoint_info)
                made.append(stage)
                with (stage / "state.pt").open("wb") as handle:
                    torch.save(payload, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                del payload
                new_info = dict(checkpoint_info)
                new_info.update(optimizer_state_removed=True, training_completed=True,
                                resumable_optimizer=False, compacted_unix=time.time(),
                                source_checkpoint_object=target.name, size_bytes=_tree_size(stage))
                _atomic_json(stage / ".research_checkpoint.json", new_info)
                _fsync_tree(stage)
                budget.ensure_space(0)
                os.replace(stage, destination)
                made[-1] = destination
                _fsync_directory(manager.objects)
                # Commit this one source before staging the next. All roles
                # remain bound to the same model/step, including a shared source.
                for name in ('best','last'):
                    if manager._resolve_owned_pointer(name)==target:manager._set_pointer(name,destination)
                manager._cleanup_owned_orphans()
                report["checkpoints"].append({"old_object": target.name, "new_object": destination.name,
                                               "step": new_info["step"], "metric": new_info.get("metric"),
                                               "before_bytes": before, "after_bytes": _tree_size(destination)})
            if dry_run:
                report["status"] = "validated_dry_run"
                report["disk"] = budget.report()
                return report
            manager._cleanup_owned_orphans()
            report.update(status="completed", finished=datetime.now(timezone.utc).isoformat(), disk=budget.report())
            owned_after=sum(_tree_size(entry) for entry in manager.objects.iterdir() if manager._owns(entry))
            report['logical_bytes_reclaimed']=owned_before-owned_after
            # Keep the original reclaimed-byte audit when an idempotent retry
            # finds every object already compacted.
            if any(not item.get("already_compacted", False) for item in report["checkpoints"]) or not (directory / "checkpoint_compaction.json").exists():
                _atomic_json(directory / "checkpoint_compaction.json", report)
            return report
        except BaseException:
            # Preserve any already-published replacement after interrupted
            # pointer updates; remove only unpublished objects created here.
            referenced = {manager._resolve_owned_pointer(name) for name in ("best", "last")}
            for entry in made:
                if entry not in referenced and manager._owns(entry):
                    shutil.rmtree(entry)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--reserve-gb", type=float, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        reports = [compact_completed_checkpoint(path, args.reserve_gb, args.dry_run) for path in args.run_dir]
        print(json.dumps({"status": "ok", "runs": reports}, ensure_ascii=False))
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "refused", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
