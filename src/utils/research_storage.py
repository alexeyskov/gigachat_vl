"""Disk reservations and crash-safe, best/last-only research checkpoints.

The caller decides what to serialize: A0/QLoRA should store trainable tensors and
an immutable base-checkpoint reference, while full training must store every
changed model tensor. Optimizer, scheduler, RNG and sampler cursor belong in the
same checkpoint payload. No Hugging Face Trainer checkpoint directories are used.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid
from typing import Any, Callable, Mapping

GIB = 1024 ** 3


class InsufficientDiskSpace(RuntimeError):
    """A write would consume the explicitly reserved free disk space."""


class DiskBudget:
    """Reserve GiB of currently free space for future experimental artifacts.

    ``anticipated_bytes`` is additional peak allocation: existing checkpoints
    are already included in filesystem usage, and replacement staging must be
    included rather than assuming old files can be removed before a safe save.
    This is a preflight check, not an OS quota or a concurrent-writer lock.
    """

    def __init__(self, root: str | Path, reserve_gb: float = 120):
        if not math.isfinite(reserve_gb) or reserve_gb < 0:
            raise ValueError("reserve_gb must be finite and nonnegative")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.reserve_bytes = math.ceil(reserve_gb * GIB)

    def report(self) -> dict[str, Any]:
        usage = shutil.disk_usage(self.root)
        return {
            "root": str(self.root), "total_bytes": usage.total,
            "used_bytes": usage.used, "free_bytes": usage.free,
            "reserve_bytes": self.reserve_bytes,
            "available_after_reserve_bytes": max(0, usage.free - self.reserve_bytes),
            "free_gib": round(usage.free / GIB, 3),
            "reserve_gib": round(self.reserve_bytes / GIB, 3),
        }

    def ensure_space(self, anticipated_bytes: int = 0) -> dict[str, Any]:
        if isinstance(anticipated_bytes, bool) or int(anticipated_bytes) != anticipated_bytes or anticipated_bytes < 0:
            raise ValueError("anticipated_bytes must be a nonnegative integer")
        report = self.report()
        if report["free_bytes"] < self.reserve_bytes + anticipated_bytes:
            raise InsufficientDiskSpace(
                f"Disk reserve would be consumed at {self.root}: "
                f"free={report['free_gib']:.2f} GiB, "
                f"reserved={report['reserve_gib']:.2f} GiB, "
                f"new peak write={anticipated_bytes / GIB:.2f} GiB. "
                "No existing checkpoint or unrelated file was deleted."
            )
        return report


def _atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(path: Path) -> None:
    """Ensure caller-written payloads are durable before publishing a pointer."""
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"Checkpoint payload may not contain symlinks: {entry}")
        if entry.is_file():
            with entry.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted((p for p in path.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(path)


def _tree_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


class BestLastCheckpointManager:
    """Keep only best and last immutable checkpoint objects, with atomic pointers.

    The same object backs both pointers when a new best is saved, so its model
    and optimizer tensors are not duplicated. Before a save there can be two
    retained objects and one temporary new object; ``anticipated_bytes`` must
    cover that additional new object. An interrupted save never removes the
    prior best/last. A subsequent save removes only unreferenced objects carrying
    this manager's ownership marker. Unknown files are never removed.
    """

    def __init__(self, run_dir: str | Path, disk_budget: DiskBudget | None = None, mode: str = "min"):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.root = Path(run_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = self.root / ".checkpoint_objects"
        self.objects.mkdir(exist_ok=True)
        self.budget = disk_budget or DiskBudget(self.root)
        if self.root.stat().st_dev != self.budget.root.stat().st_dev:
            raise ValueError("DiskBudget root and checkpoint root must be on the same filesystem")
        self.mode = mode
        with self._lock():
            marker = self.root / ".checkpoint_manager.json"
            if marker.exists():
                info = json.loads(marker.read_text())
                if info.get("schema_version") != 1 or info.get("mode") != mode:
                    raise ValueError("Checkpoint manager schema/mode mismatch")
                self.owner = info["owner"]
            else:
                self.owner = uuid.uuid4().hex
                _atomic_json(marker, {"schema_version": 1, "owner": self.owner, "mode": mode})
            for name in ("best", "last"):
                if os.path.lexists(self.root / name):
                    self._resolve_owned_pointer(name)

    @contextlib.contextmanager
    def _lock(self):
        with (self.root / ".checkpoint.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _owns(self, directory: Path) -> bool:
        marker = directory / ".research_checkpoint.json"
        if directory.is_symlink() or not directory.is_dir() or not marker.is_file():
            return False
        try:
            return json.loads(marker.read_text()).get("owner") == self.owner
        except (OSError, ValueError):
            return False

    def _resolve_owned_pointer(self, name: str) -> Path | None:
        pointer = self.root / name
        if not os.path.lexists(pointer):
            return None
        if not pointer.is_symlink():
            raise ValueError(f"Refusing to replace unmanaged checkpoint path: {pointer}")
        target = pointer.resolve()
        if target.parent != self.objects or not self._owns(target):
            raise ValueError(f"Checkpoint pointer is not owned by this manager: {pointer}")
        return target

    def _set_pointer(self, name: str, target: Path) -> None:
        self._resolve_owned_pointer(name)
        temporary = self.root / f".{name}.{uuid.uuid4().hex}.link"
        try:
            temporary.symlink_to(target.relative_to(self.root), target_is_directory=True)
            os.replace(temporary, self.root / name)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)

    def _cleanup_owned_orphans(self) -> None:
        referenced = {self._resolve_owned_pointer(n) for n in ("best", "last")}
        for entry in self.objects.iterdir():
            if entry not in referenced and self._owns(entry):
                shutil.rmtree(entry)

    def describe(self) -> dict[str, Any]:
        result = {"mode": self.mode, "disk": self.budget.report()}
        for name in ("best", "last"):
            target = self._resolve_owned_pointer(name)
            result[name] = json.loads((target / ".research_checkpoint.json").read_text()) if target else None
        return result

    def save(
        self, writer: Callable[[Path], None], *, step: int, metric: float | None = None,
        anticipated_bytes: int | None = None, metadata: Mapping[str, Any] | None = None,
        is_best: bool | None = None,
    ) -> dict[str, Any]:
        """Call ``writer(staging_directory)`` then atomically advance pointers.

        The initial save needs an explicit conservative size estimate. Later
        saves can default to the last actual checkpoint size. ``metric=None``
        writes last only; best is updated only for finite measured metrics or
        explicit ``is_best=True``. Passing a training loss as validation metric
        is the caller's responsibility and should be named honestly in metadata.
        """
        if step < 0 or int(step) != step:
            raise ValueError("step must be a nonnegative integer")
        if metric is not None and not math.isfinite(metric):
            raise ValueError("Checkpoint metric must be finite")
        with self._lock():
            self._cleanup_owned_orphans()
            last = self._resolve_owned_pointer("last")
            best = self._resolve_owned_pointer("best")
            if anticipated_bytes is None:
                if last is None:
                    raise ValueError("First save requires anticipated_bytes for staging headroom")
                anticipated_bytes = _tree_size(last)
            if anticipated_bytes < 0 or int(anticipated_bytes) != anticipated_bytes:
                raise ValueError("anticipated_bytes must be a nonnegative integer")
            # Covers metadata, serialization overhead and filesystem bookkeeping.
            peak_estimate = int(anticipated_bytes) + max(16 * 1024 ** 2, math.ceil(anticipated_bytes * 0.05))
            self.budget.ensure_space(peak_estimate)
            if is_best is None:
                old_metric = json.loads((best / ".research_checkpoint.json").read_text()).get("metric") if best else None
                is_best = metric is not None and (
                    old_metric is None or (metric < old_metric if self.mode == "min" else metric > old_metric)
                )
            object_id = uuid.uuid4().hex
            staging = self.objects / f".stage-{object_id}"
            destination = self.objects / f"object-{object_id}"
            staging.mkdir()
            info = {"owner": self.owner, "schema_version": 1, "step": int(step),
                    "metric": metric, "created_unix": time.time(), "metadata": dict(metadata or {})}
            _atomic_json(staging / ".research_checkpoint.json", info)
            try:
                writer(staging)
                if not any(p.name != ".research_checkpoint.json" for p in staging.iterdir()):
                    raise ValueError("Checkpoint writer did not produce any payload")
                info["size_bytes"] = _tree_size(staging)
                _atomic_json(staging / ".research_checkpoint.json", info)
                self.budget.ensure_space(0)
                _fsync_tree(staging)
                os.replace(staging, destination)
                _fsync_directory(self.objects)
                # Best first: a crash between pointer updates retains the new
                # best and previous last, both valid and resumable checkpoints.
                if is_best:
                    self._set_pointer("best", destination)
                self._set_pointer("last", destination)
                self._cleanup_owned_orphans()
                return self.describe()
            except BaseException:
                if self._owns(staging):
                    shutil.rmtree(staging)
                raise

    def save_torch(self, payload: Any, *, checkpoint_format: str = "legacy", **kwargs) -> dict[str, Any]:
        """Serialize one explicit resume payload; torch is imported lazily."""
        if checkpoint_format not in {"legacy", "bundle"}:
            raise ValueError("checkpoint_format must be legacy or bundle")
        if checkpoint_format == "bundle":
            from src.utils.research_checkpoint_bundle import save_checkpoint_bundle, retire_best_optimizer
            self.save(lambda directory: save_checkpoint_bundle(payload,directory,compression="none"), **kwargs)
            retirement = retire_best_optimizer(self.root)
            result = self.describe()
            result["checkpoint_format"] = "bundle"
            result["optimizer_retirement"] = retirement
            return result
        import torch

        def writer(directory: Path) -> None:
            with (directory / "state.pt").open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())

        return self.save(writer, **kwargs)


def trainable_state_dict(model) -> dict[str, Any]:
    """CPU copies of trainable parameters only, without duplicated frozen bases.

    This deliberately excludes buffers. Callers must add changed buffers when
    their chosen backend has mutable buffers, and persist the base revision and
    exact module configuration needed to restore this delta checkpoint.
    """
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}
