#!/usr/bin/env python3
"""Root-launched ONLY: full-size SYNTHETIC CPU codec proof, never training.

Default --plan reads small safetensors headers only. --run-full-synthetic must be
explicitly launched under Supervisor by root after reviewing resources. No LLM,
CUDA, downloads, actual trained checkpoint transformations or source mutations.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GIB = 2 ** 30
METADATA_ENVELOPE_BYTES = 64 * 2 ** 20
FILESYSTEM_ENVELOPE_BYTES = 2 ** 20
MAX_MEMORY_SAMPLES = 32768


def snapshot_path(revision):
    return Path("/workspace/.hf_home/hub/models--Tiiny--SmallThinker-4BA0.6B-Instruct/snapshots") / revision


class BoundedWriter:
    """Hide fileno so native Torch storages cannot bypass the task size cap."""
    def __init__(self, raw, budget, maximum_bytes):
        self.raw, self.budget, self.maximum_bytes, self.bytes = raw, budget, maximum_bytes, 0
    def write(self, data):
        if self.bytes + len(data) > self.maximum_bytes:
            raise ValueError("Task-owned archive hard size cap exceeded")
        self.budget.ensure_space(len(data)); self.bytes += len(data)
        return self.raw.write(data)
    def flush(self):self.raw.flush()
    def fileno(self):raise io.UnsupportedOperation("Use guarded Python writes")


def headers(snapshot):
    from src.utils import research_lossless_delta as v1
    anchor = v1.SmallAnchor(snapshot)
    result = {}
    for shard in sorted(set(anchor.mapping.values())):
        path = Path(snapshot) / shard
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            if not 0 < size <= 16 * 2 ** 20:
                raise ValueError("Unexpected safetensors header length")
            parsed = json.loads(f.read(size))
        for name, item in parsed.items():
            if name == "__metadata__":
                continue
            if anchor.mapping.get(name) != shard or item["dtype"] != "BF16":
                raise ValueError("Pinned Small header/index/dtype differs")
            result[name] = {"shape": item["shape"], "fp32_bytes": math.prod(item["shape"]) * 4,
                            "shard": shard}
    if set(result) != set(anchor.mapping):
        raise ValueError("Missing stored Small tensors")
    return dict(sorted(result.items()))


def _atomic(path, value):
    data = (json.dumps(value, indent=2) + "\n").encode()
    # Include the old target during replacement and the entire new staging
    # record, including block rounding. The probe directory is task-owned.
    used = sum(p.stat().st_blocks * 512 for p in path.parent.iterdir()
               if p.is_file() and p.name != "synthetic_model_legacy.pt")
    additional = math.ceil(len(data) / 4096) * 4096
    if used + additional > METADATA_ENVELOPE_BYTES:
        raise ValueError("Task metadata atomic staging envelope exceeded")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Monitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.peak = 0
        self.total_samples = 0
        self.dropped_samples = 0
        self.lock = threading.Lock()
        self.phase = "startup"
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        cg = Path("/sys/fs/cgroup")
        current = int((cg / "memory.current").read_text())
        with self.lock:
            self.peak = max(self.peak, current);self.total_samples += 1
            if len(self.samples) < MAX_MEMORY_SAMPLES:
                self.samples.append({"time": time.time(), "phase": self.phase,
                                     "cgroup_current_bytes": current,
                                     "process_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
            else:self.dropped_samples += 1

    def _run(self):
        while not self.stop.wait(.25):
            self.sample()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-full-synthetic", action="store_true")
    p.add_argument("--output", default="research_results/stream_delta_full_synthetic_v1")
    p.add_argument("--max-staging-gib", type=float, default=12)
    p.add_argument("--reserve-gb", type=float, default=120)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cleanup-verified", action="store_true")
    args = p.parse_args()
    import numpy as np
    import torch
    from src.utils import research_lossless_delta as v1
    from src.utils import research_stream_delta as stream
    from src.utils.research_checkpoint_compat import load_research_payload
    from src.utils.research_storage import DiskBudget

    if not 0 < args.max_staging_gib <= 12 or args.reserve_gb < 120 or not 1 <= args.threads <= 8:
        raise ValueError("Proof requires staging<=12GiB, disk reserve>=120GiB,1..8CPU threads")
    torch.set_num_threads(args.threads)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only proof")
    snapshot = snapshot_path(v1.REVISION)
    metadata = headers(snapshot)
    original_stats = {name: v1._stat(snapshot / name) for name in v1.SHARD_SHA256}
    decoded = sum(s["fp32_bytes"] for s in metadata.values())
    largest = max(s["fp32_bytes"] for s in metadata.values())
    encoded_upper = sum(v1._zstd().ZSTD_compressBound(s["fp32_bytes"]) for s in metadata.values())
    writer_extra = 2 * encoded_upper + sum(stream.ANCHOR_SHARD_BYTES.values()) + 8 * largest + stream.METADATA_LIMIT
    memory = v1.memory_guard(writer_extra)
    plan = {"status": "plan_only", "no_full_tensors_or_archives_created": True,
            "stored_bf16_tensors": len(metadata), "stored_elements": decoded // 4,
            "fp32_model_bytes": decoded, "largest_fp32_tensor_bytes": largest,
            "includes_separately_stored_head_and_embedding": True,
            "native_tied_unique_parameter_count_for_comparison_only": 4035085824,
            "writer_extra_guard_bytes": writer_extra, "memory": memory,
            "archive_plus_metadata_limit_bytes": int(args.max_staging_gib * GIB),
            "disk_reserve_bytes": math.ceil(args.reserve_gb * GIB),
            "source_sha256": {str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "src/utils/research_stream_delta.py": stream.source_sha256(),
                              "src/utils/research_lossless_delta.py": v1.codec_sha256()},
            "scope": "all stored original tensors +tiny normal fixtures; NOT trained/research checkpoint or full optimizer proof"}
    if not args.run_full_synthetic:
        print(json.dumps(plan, indent=2)); return
    output = (ROOT / args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Task output must be new/empty; no existing artifact is overwritten")
    output.mkdir(parents=True, exist_ok=True)
    owner = uuid.uuid4().hex
    _atomic(output / "owner.json", {"owner": owner, "scope": "synthetic codec probe only", "source_sha256": plan["source_sha256"]})
    budget = DiskBudget(output, args.reserve_gb)
    budget.ensure_space(int(args.max_staging_gib * GIB))
    archive = output / "synthetic_model_legacy.pt"
    monitor = Monitor(); monitor.sample(); monitor.thread.start()
    start = time.monotonic()
    last = [0.]
    def progress(event, **kwargs):
        print(json.dumps({"event": event, "elapsed_seconds": time.monotonic() - start,
                          "free_disk_gib": budget.report()["free_gib"], **kwargs}), flush=True)
    try:
        torch.manual_seed(20261122)
        np.random.seed(20261122)
        template = {"model_ref": {"repo": v1.REPO, "revision": v1.REVISION},
                    "model": {"projector.weight": torch.randn(1536, 1152)},
                    "optimizer": {"tiny_fixture_only": {"exp_avg": torch.randn(32, 16), "exp_avg_sq": torch.rand(32, 16)}},
                    "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
                    "protocol": "FULL STORED MODEL SYNTHETIC FP32 updates; NOT TRAINED; TINY optimizer fixture only"}
        normal = template  # Small template only; never full original tensors.
        monitor.phase = "stream_encode"
        def tensors():
            anchor = v1.SmallAnchor(snapshot)
            for i, (key, shape) in enumerate(metadata.items()):
                # stream_encode_small guards BEFORE resuming this generator.
                reference = anchor.read(anchor.descriptor(key))
                update = ((torch.arange(reference.numel(), dtype=torch.float32) % 127) - 63).reshape(reference.shape) * 1e-7
                current = reference.float() + update
                yield "llm." + key, current, snapshot
                del reference, update, current
                if time.monotonic() - last[0] >= 30:
                    progress("encode", completed_tensors=i + 1, total_tensors=len(metadata));last[0] = time.monotonic()
        normal_bytes = sum(t.untyped_storage().nbytes() for _, t in v1._walk_tensors(template))
        maximum_archive = int(args.max_staging_gib * GIB) - METADATA_ENVELOPE_BYTES - FILESYSTEM_ENVELOPE_BYTES
        wrapped, receipt = stream.stream_encode_small(template, tensors(),
            expected_decoded_bytes=decoded, expected_tensor_count=len(metadata), largest_tensor_bytes=largest,
            encoded_upper_bytes=encoded_upper,
            max_encoded_bytes=maximum_archive - normal_bytes - stream.METADATA_LIMIT)
        expected = {name: {"shape": spec["shape"], "decoded_sha256": spec["decoded_sha256"]}
                    for session in wrapped.payload.sessions for name, spec in session.manifest["tensors"].items()}
        header = wrapped.header
        if header["encoded_bytes"] + header["normal_storage_bytes"] + stream.METADATA_LIMIT > maximum_archive:
            raise ValueError("Measured payload cannot fit archive staging budget")
        _atomic(output / "encoding.json", {"owner": owner, "header": header, "receipt": receipt, "expected": expected})
        progress("encoded", decoded_bytes=decoded, encoded_bytes=receipt["encoded_bytes"], tensors=len(expected))
        monitor.phase = "native_legacy_write"
        with archive.open("xb") as raw:
            stream.save_guarded_legacy(wrapped, BoundedWriter(raw, budget, maximum_archive))
            raw.flush(); os.fsync(raw.fileno())
        archive_size = archive.stat().st_size
        archive_digest = v1._file_sha(archive)
        del wrapped
        gc.collect()
        monitor.phase = "guarded_native_legacy_read"
        progress("read_start", archive_bytes=archive_size, reader_extra_guard_bytes=stream.memory_requirement(header))
        restored = load_research_payload(archive, map_location="cpu", mmap=True)
        if set(restored["model"]) != set(expected) | set(normal["model"]):
            raise ValueError("Roundtrip model key set changed")
        monitor.phase = "full_bit_verification"
        for i, (name, spec) in enumerate(expected.items()):
            v1.memory_guard(3 * largest + stream.METADATA_LIMIT)
            tensor = restored["model"][name]
            if type(tensor) is not torch.Tensor or tensor.dtype != torch.float32 or list(tensor.shape) != spec["shape"] or v1._sha(v1._raw(tensor)) != spec["decoded_sha256"]:
                raise ValueError("Full tensor bit verification failed: " + name)
            if time.monotonic() - last[0] >= 30:
                progress("verify", completed_tensors=i + 1, total_tensors=len(expected));last[0] = time.monotonic()
        for key in ("model_ref", "protocol"):
            if restored[key] != normal[key]:raise ValueError("Normal metadata changed")
        for key in ("optimizer", "rng_torch", "rng_numpy"):
            # Exact fixture comparison after native legacy has filled storages.
            if key == "rng_numpy":
                if pickle_bytes(restored[key]) != pickle_bytes(normal[key]):raise ValueError("NumPy RNG changed")
            elif key == "rng_torch":
                if v1._raw(restored[key]) != v1._raw(normal[key]):raise ValueError("Torch RNG changed")
            else:
                for n, t in normal[key]["tiny_fixture_only"].items():
                    if v1._raw(restored[key]["tiny_fixture_only"][n]) != v1._raw(t):raise ValueError("Tiny optimizer fixture changed")
        if v1._raw(restored["model"]["projector.weight"]) != v1._raw(normal["model"]["projector.weight"]):
            raise ValueError("Normal projector changed")
        del restored
        gc.collect(); monitor.sample()
        if any(v1._stat(snapshot / name) != previous for name, previous in original_stats.items()):
            raise ValueError("Original source shard stat changed during synthetic proof")
        verified = {"status": "verified_full_stored_synthetic_model_only", "owner": owner,
                    "scope": plan["scope"], "plan": plan, "header": header, "receipt": receipt,
                    "archive_path": str(archive), "archive_sha256": archive_digest, "archive_bytes": archive_size,
                    "stored_tensors_verified": len(expected), "stored_fp32_bytes_verified": decoded,
                    "all_model_and_normal_fixture_bits_exact": True,
                    "v1_source_unchanged": v1.codec_sha256() == stream.FROZEN_V1_SHA256,
                    "source_files_changed": False, "cuda_initialized": torch.cuda.is_initialized(),
                    "elapsed_seconds": time.monotonic() - start,
                    "process_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "shared_cgroup_observed_peak_bytes": monitor.peak,
                    "shared_cgroup_peak_is_not_isolated_process_memory": True,
                    "archive_deleted": False}
        _atomic(output / "verified.json", verified)
        # Only this named task-owned archive, AFTER durable verified receipt.
        if args.cleanup_verified:
            persisted = json.loads((output / "verified.json").read_text())
            if (persisted["owner"] != owner or json.loads((output / "owner.json").read_text())["owner"] != owner or
                persisted["archive_sha256"] != v1._file_sha(archive) or archive.is_symlink() or archive.parent != output):
                raise ValueError("Task-owned verified cleanup binding differs")
            archive.unlink()
            verified["archive_deleted"] = True
            verified["task_owned_archive_bytes_reclaimed"] = archive_size
            _atomic(output / "verified.json", verified)
        progress("verified", **{k: verified[k] for k in ("stored_tensors_verified", "archive_bytes", "archive_deleted", "elapsed_seconds")})
    except BaseException as error:
        _atomic(output / "failure.json", {"owner": owner, "error_type": type(error).__name__, "error": str(error),
                                          "phase": monitor.phase, "archive_preserved_if_present": archive.exists(),
                                          "scope": "synthetic proof failed; no scientific result"})
        raise
    finally:
        monitor.stop.set();monitor.thread.join(timeout=2);monitor.sample()
        _atomic(output / "memory_samples.json", {"samples": monitor.samples, "total_samples": monitor.total_samples,
                                                  "dropped_samples": monitor.dropped_samples,
                                                  "observed_peak_bytes": monitor.peak,
                                                  "scope": "shared cgroup observational samples; bounded retained list, peak always tracked"})


def pickle_bytes(value):
    import pickle
    return pickle.dumps(value, protocol=4)


if __name__ == "__main__":
    main()
