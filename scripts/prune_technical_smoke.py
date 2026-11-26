#!/usr/bin/env python3
"""Delete explicitly selected, validated technical-smoke checkpoint weights only.

Never run automatically. Scientific pilot/primary checkpoints are refused. All
metrics, resolved configuration, completion markers and validation receipts stay
in place. Unknown files in the checkpoint object store are never removed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
# Explicitly authorized disposable two-step full-training diagnostics. The
# historical run names use '__SFT_smoke_' rather than the generic '__smoke_'.
AUTHORIZED_FULL_SMOKE_BASENAMES = {
    'smallthinker4__siglip2__4f__SFT_smoke_s42',
    'smallthinker4__siglip2__2a__SFT_smoke_s42',
    'smallthinker4__siglip2__2b__SFT_smoke_s42',
    'smallthinker4__siglip2__2c__SFT_smoke_s42',
    'smallthinker4__siglip2__3a__SFT_smoke_s42',
    'smallthinker4__siglip2__3b__SFT_smoke_s42',
    'smallthinker4__siglip2__3c__SFT_smoke_s42',
}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.research_storage import BestLastCheckpointManager, DiskBudget, _atomic_json, _tree_size


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_mapping(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _positive_integer(value, limit):
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= limit


def _validate_guard(directory):
    explicit_full_smoke = (directory.name in AUTHORIZED_FULL_SMOKE_BASENAMES and
                           directory.parent == (ROOT / 'runs/research').resolve())
    if "__smoke_" not in directory.name and not explicit_full_smoke:
        raise ValueError("Technical cleanup requires '__smoke_' or an explicitly authorized full-smoke basename under ROOT/runs/research")
    if any(word in str(directory).lower() for word in ("pilot", "primary")):
        raise ValueError("Pilot/primary scientific paths must never be pruned by this tool")
    config_path = directory / "resolved_config.json"
    completion_path = directory / "completed.json"
    manager_path = directory / ".checkpoint_manager.json"
    receipt_path = directory / "checkpoint_roundtrip.json"
    for path in (config_path, completion_path, manager_path, receipt_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Required local regular file missing: {path}")
    config = _read_mapping(config_path)
    for key, value in config.items():
        if any(term in key.lower() for term in ('manifest', 'profile', 'scope')):
            if any(term in json.dumps(value, ensure_ascii=False).lower() for term in ('pilot', 'primary')):
                raise ValueError('Pilot/primary manifest, profile or scope must never be pruned by this tool')
    if not _positive_integer(config.get("max_records"), 2000):
        raise ValueError("Technical smoke requires explicit max_records between 1 and 2000")
    if not _positive_integer(config.get("max_steps"), 100):
        raise ValueError("Technical smoke requires explicit max_steps between 1 and 100")
    if config.get("stage") not in {"A0", "A1", "SFT"}:
        raise ValueError("Unknown technical-smoke stage")
    output = Path(config.get("output", "")).expanduser()
    output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    if output != directory:
        raise ValueError("Resolved configuration output does not match the explicitly selected run")
    completion = _read_mapping(completion_path)
    if not _positive_integer(completion.get("steps"), config["max_steps"]):
        raise ValueError("Completion marker has no valid completed technical-smoke step count")
    if completion.get("status") in {"failed", "incomplete", "interrupted"}:
        raise ValueError("Technical smoke is not successfully completed")
    ownership = _read_mapping(manager_path)
    if ownership.get("schema_version") != 1 or not ownership.get("owner") or ownership.get("mode") not in {"min", "max"}:
        raise ValueError("Invalid checkpoint manager ownership marker")
    report = _read_mapping(receipt_path)
    if report.get("schema_version") != 1 or report.get("status") != "passed":
        raise ValueError("Full successful roundtrip validation is required; structural-only validation is insufficient")
    receipts = report.get("checkpoints", [report])
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("Roundtrip report has no validated checkpoints")
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise ValueError("Roundtrip checkpoint receipt is not an object")
    objects = directory / ".checkpoint_objects"
    if not objects.is_dir() or objects.is_symlink() or objects.resolve().parent != directory:
        raise ValueError("Checkpoint object store must be a local directory inside the selected run")
    return ownership, receipts, {"config_sha256": _sha256_file(config_path),
                                 "validation_receipt_sha256": _sha256_file(receipt_path)}


def _validate_checkpoint_receipt(target, receipt, owner):
    if receipt.get("schema_version") != 1 or receipt.get("status") != "passed":
        raise ValueError(f"Checkpoint roundtrip did not pass for {target}")
    if receipt.get("checkpoint_owner") != owner:
        raise ValueError("Roundtrip receipt checkpoint owner mismatch")
    if receipt.get("checkpoint_path") != str(target):
        raise ValueError("Roundtrip receipt does not identify the exact retained checkpoint object")
    if not _positive_integer(receipt.get("validated_tensor_count"), 10**12):
        raise ValueError("Roundtrip receipt has no validated tensors")
    if receipt.get("all_tensors_finite") is not True or receipt.get("expected_keys_match") is not True:
        raise ValueError("Roundtrip tensor finiteness/key validation did not pass")
    metadata = _read_mapping(target.parent / ".research_checkpoint.json")
    if receipt.get("checkpoint_step") != metadata.get("step"):
        raise ValueError("Roundtrip receipt checkpoint step mismatch")
    if not target.is_file() or target.is_symlink():
        raise ValueError("Retained checkpoint payload must be a local regular state.pt")
    stats = target.stat()
    if receipt.get("checkpoint_size_bytes") != stats.st_size or receipt.get("checkpoint_mtime_ns") != stats.st_mtime_ns:
        raise ValueError("Checkpoint changed since roundtrip validation")
    from src.utils.research_checkpoint_compat import checkpoint_kind,checkpoint_identity,same_checkpoint_identity
    if checkpoint_kind(target)=='bundle_v1':
        expected=receipt.get('checkpoint_bundle_identity')
        if not isinstance(expected,dict) or expected.get('components_fully_hashed') is not True:
            raise ValueError('Bundle cleanup requires full component-bound roundtrip identity')
        # The caller already holds this manager's exclusive lock. Taking another
        # shared lock through a new descriptor would deadlock.
        current=checkpoint_identity(target,manager_lock_held=True)
        if not same_checkpoint_identity(expected,current):
            raise ValueError('Checkpoint bundle changed since roundtrip validation')


def prune_technical_smoke(run_dir, dry_run=False):
    """Guarded, idempotent deletion of this run's owned disposable smoke weights."""
    directory = Path(run_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Run directory does not exist: {directory}")
    ownership, receipts, hashes = _validate_guard(directory)
    manager = BestLastCheckpointManager(directory, disk_budget=DiskBudget(directory, 0), mode=ownership["mode"])
    cleanup_path = directory / "technical_smoke_pruned.json"
    with manager._lock():
        previous = _read_mapping(cleanup_path) if cleanup_path.exists() else None
        if previous:
            if previous.get("owner") != manager.owner or any(previous.get(key) != value for key, value in hashes.items()):
                raise ValueError("Previous technical cleanup receipt has different ownership/config/validation")
        retained = {manager._resolve_owned_pointer(name) for name in ("best", "last")}
        retained.discard(None)
        owned = [entry for entry in manager.objects.iterdir() if manager._owns(entry)]
        if previous and previous.get("status") == "pruned":
            if retained or owned:
                raise ValueError("New checkpoints appeared after previous cleanup; fresh validation is required")
            return {**previous, "already_pruned": True, "disk": manager.budget.report()}
        if previous and previous.get("status") == "pruning":
            permitted = set(previous["owned_objects"])
            if any(entry.name not in permitted for entry in owned):
                raise ValueError("New owned checkpoints appeared during interrupted cleanup")
        else:
            if not retained:
                raise ValueError("No retained best/last checkpoint is available to validate")
            receipt_by_path = {receipt.get("checkpoint_path"): receipt for receipt in receipts}
            for target in retained:
                payload = target / "state.pt"
                receipt = receipt_by_path.get(str(payload))
                if receipt is None:
                    raise ValueError(f"No full roundtrip receipt for retained checkpoint: {payload}")
                _validate_checkpoint_receipt(payload, receipt, manager.owner)
        report = {"schema_version": 1, "owner": manager.owner, "status": "pruning",
                  "run_dir": str(directory), "scope": "disposable_technical_smoke_weights_only",
                  "owned_objects": [entry.name for entry in owned],
                  "logical_bytes_removed": sum(_tree_size(entry) for entry in owned),
                  "started": datetime.now(timezone.utc).isoformat(), **hashes}
        if dry_run:
            return {**report, "status": "validated_dry_run", "disk": manager.budget.report()}
        _atomic_json(cleanup_path, report)
        for name in ("best", "last"):
            (directory / name).unlink(missing_ok=True)
        for entry in owned:
            shutil.rmtree(entry)
        report.update(status="pruned", finished=datetime.now(timezone.utc).isoformat())
        _atomic_json(cleanup_path, report)
        return {**report, "disk": manager.budget.report()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True,
                        help="Explicit completed disposable technical-smoke run; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        # Validate every explicitly selected run before deleting the first one.
        reports = [prune_technical_smoke(path, dry_run=True) for path in args.run_dir]
        if not args.dry_run:
            reports = [prune_technical_smoke(path) for path in args.run_dir]
        print(json.dumps({"status": "ok", "runs": reports}, ensure_ascii=False))
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "refused", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
