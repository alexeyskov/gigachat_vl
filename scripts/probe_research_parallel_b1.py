#!/usr/bin/env python3
"""Isolated two-process native-B1 parity/throughput prototype; never a queue worker.

No parent/controller is stopped here. A separate owned boundary coordinator must
provide an idle GPU. Both new children wait for an atomic PID/birth/argv-bound
barrier before importing the training stack or initializing CUDA. Each repeats
the SAME64 fixed records: the128 measured completions are duplicate diagnostic
work, not128 distinct scientific examples and not a campaign speed forecast.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OWNER = "parallel_native_b1_probe_v1"
SCRIPT = "scripts/probe_research_parallel_b1.py"
SOURCE_RUN = "runs/research/smallthinker4__siglip2__4q__SFT_pilot_scienceqa_context_v2_balanced_s42"
MANIFEST = "datasets/research/eval-heldout-v3/evaluation.jsonl"
GIB = 2 ** 30
PYTORCH_CAP = 8 * GIB
PROCESS_CAP = 12 * GIB
MINIMUM_FREE = 24 * GIB
MAX_SECONDS = 300
EXECUTION_SECONDS = 275  # Reserve25s for failed nvidia queries and owned-child cleanup.
BENCHMARKS = {"vqav2": "general_vqa", "gqa": "general_vqa",
              "textvqa": "ocr_documents", "docvqa": "ocr_documents",
              "chartqa": "charts_diagrams", "ai2d": "charts_diagrams",
              "scienceqa": "knowledge_reasoning", "aokvqa": "knowledge_reasoning"}
FROZEN_SOURCES = (
    "src/eval/research_inference.py", "scripts/eval_research_checkpoint.py",
    "src/eval/research_metrics.py", "scripts/run_research_stage.py",
    "src/model/research_moe.py", "src/model/smallthinker.py",
    "src/model/research_quantization.py", "src/model/research_vision.py",
    "src/model/gigachat_vl.py", "src/dataset/vision_cache.py",
    "scripts/extract_vision_features.py", "src/eval/research_ablation.py",
    "src/utils/research_checkpoint_compat.py", "src/utils/research_checkpoint_bundle.py",
    "src/utils/research_checkpoint_io.py")
ROW_FIELDS = ("record_id", "question", "image_hashes", "answers", "scorer", "task_group",
              "source", "benchmark", "choices", "scoring_protocol", "image_ablation")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2 ** 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Probe artifact must not be a symlink")
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def process_info(pid):
    """Only inspect explicitly declared parent/worker processes; no env reads."""
    if type(pid) is not int or pid < 1:
        return None
    try:
        proc = Path("/proc") / str(pid)
        data = (proc / "stat").read_text()
        fields = data[data.rfind(")") + 2:].split()
        argv = [p.decode() for p in (proc / "cmdline").read_bytes().split(b"\0") if p]
        return {"pid": pid, "start_time": int(fields[19]), "ppid": int(fields[1]),
                "argv": argv, "command_sha256": digest(argv), "state": fields[0]}
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None


def same_process(expected, actual):
    return (isinstance(expected, dict) and isinstance(actual, dict) and
            all(expected.get(key) == actual.get(key)
                for key in ("pid", "start_time", "argv", "command_sha256")) and
            actual.get("state") not in ("Z", "X"))


def gpu_snapshot():
    commands = (("--query-gpu=uuid,memory.total,memory.free", "gpu"),
                ("--query-compute-apps=pid,used_gpu_memory", "processes"))
    result = {}
    for query, name in commands:
        row = subprocess.run(["nvidia-smi", "--id=0", query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if row.returncode:
            raise RuntimeError("Cannot verify GPU isolation: " + row.stderr.strip())
        lines = [line.strip() for line in row.stdout.splitlines() if line.strip()]
        try:
            if name == "gpu":
                if len(lines) != 1:
                    raise ValueError("Expected exactly one GPU0 result")
                gpu_uuid, total, free = (x.strip() for x in lines[0].split(","))
                if not gpu_uuid.startswith("GPU-"):
                    raise ValueError("GPU0 UUID is not reported")
                result.update(gpu_uuid=gpu_uuid, total_bytes=int(total) * 2 ** 20, free_bytes=int(free) * 2 ** 20)
            else:
                result["processes"] = [{"pid": int(a.strip()), "bytes": int(b.strip()) * 2 ** 20}
                                       for a, b in (line.split(",") for line in lines)]
        except ValueError as error:
            raise RuntimeError("Unverifiable GPU memory/process response") from error
    return result


def validate_gpu(snapshot, workers=(), *, lookup=None, exclusive=False, exited_pids=(), expected_uuid=None,
                 confirm_owned_exit=None):
    lookup = process_info if lookup is None else lookup
    if expected_uuid is not None and snapshot.get("gpu_uuid") != expected_uuid:
        raise RuntimeError("NVML GPU0 UUID changed")
    if snapshot["free_bytes"] < MINIMUM_FREE:
        raise RuntimeError("Global free GPU memory fell below24GiB")
    if snapshot["total_bytes"] < MINIMUM_FREE + 2 * PROCESS_CAP:
        raise RuntimeError("GPU is smaller than the fixed two-worker budget")
    allowed = {worker["pid"]: worker for worker in workers}
    if exclusive and snapshot["processes"]:
        raise RuntimeError("Probe requires an exclusive idle GPU initially")
    if exclusive and snapshot["free_bytes"] < MINIMUM_FREE + 2 * PROCESS_CAP:
        raise RuntimeError("Initial GPU headroom must cover24GiB floor plus both12GiB worker ceilings")
    for row in snapshot["processes"]:
        if row["pid"] not in allowed:
            raise RuntimeError("Unknown GPU process; only the two owned workers are allowed")
        expected = allowed[row["pid"]]
        live = lookup(row["pid"])
        # A worker can exit AFTER the caller polled it but DURING nvidia-smi.
        # Confirm again through the parent's original Popen handle at the exact
        # failed lookup. Missing /proc, or a dying same-birth owned process, is
        # compatible with stale NVML; a reused PID or changed nonempty argv is not.
        dying_owned = (live is not None and live.get("start_time") == expected.get("start_time") and
                       live.get("ppid") == expected.get("ppid") and
                       (live.get("state") in ("Z", "X") or not live.get("argv")))
        stale_candidate = live is None or dying_owned
        confirmed_exit = stale_candidate and (row["pid"] in exited_pids or
            (confirm_owned_exit is not None and confirm_owned_exit(row["pid"]) is True))
        if not same_process(expected, live) and not confirmed_exit:
            raise RuntimeError("GPU worker PID identity changed")
        if row["bytes"] > PROCESS_CAP:
            raise RuntimeError("Owned GPU process exceeded12GiB")
    return snapshot


def selected_indices(rows):
    if len(rows) != 512 or len({r["record_id"] for r in rows}) != 512:
        raise ValueError("The preserved manifest must contain512 unique records")
    if Counter(r.get("benchmark") for r in rows) != Counter({b: 64 for b in BENCHMARKS}):
        raise ValueError("The source manifest must contain64 records per benchmark")
    counts = Counter(); result = []
    for index, row in enumerate(rows):
        if row.get("task_group") != BENCHMARKS[row["benchmark"]]:
            raise ValueError("Benchmark/task-group mapping changed")
        if counts[row["benchmark"]] < 8:
            result.append(index); counts[row["benchmark"]] += 1
    if len(result) != 64:
        raise ValueError("Selection must contain8 records from all8 benchmarks")
    return result


def public_record(row):
    return {key: row[key] for key in ("record_id", "question", "image_paths", "image_hashes") if key in row}


def frozen_image_hasher(root):
    # Importing src.dataset executes its training-oriented package __init__ and
    # imports Torch. Load this already-hash-bound pure module directly instead;
    # the parent remains safe to fork/exec and never initializes the CUDA stack.
    spec = importlib.util.spec_from_file_location("_parallel_b1_frozen_cache_hash", root / "src/dataset/vision_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.image_content_hash


def token_parity(record, generated, baseline):
    ids = generated.get("generated_token_ids")
    if (not isinstance(ids, list) or len(ids) > 64 or
            any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError("Malformed generated token IDs")
    if baseline.get("record_id") != record["record_id"] or ids != baseline.get("generated_token_ids"):
        raise ValueError("Native B1 token mismatch for record " + record["record_id"])
    if generated.get("generated_tokens") != len(ids):
        raise ValueError("Native B1 generated token count differs from actual token IDs")
    for key in ("prediction", "generated_tokens", "prompt_sha256", "prefill_tokens", "visual_tokens", "logits_to_keep"):
        if generated.get(key) != baseline.get(key):
            raise ValueError("Native B1 prompt/generation metadata mismatch: " + key)


def generate_checked_record(model, record, cached, *, generator, scorer):
    """Exactly one frozen native B1 call; references never enter the generator."""
    generated = generator(model, public_record(record), max_new_tokens=64)
    token_parity(record, generated, cached)
    score = scorer(record, generated["prediction"])
    if score != cached["score"]:
        raise ValueError("Native B1 score differs despite matched tokens")
    row = {key: record[key] for key in ROW_FIELDS if key in record}
    row.update(generated, score=score)
    return row


def prepare_context(root=ROOT):
    """CPU/read-only provenance preflight; no Torch import or model reconstruction."""
    from src.eval.research_metrics import aggregate_scores, score_prediction
    from src.utils.research_checkpoint_compat import checkpoint_identity
    run = root / SOURCE_RUN
    config = read_json(run / "resolved_config.json")
    done = read_json(run / "completed.json")
    if (config.get("model") != "Tiiny/SmallThinker-4BA0.6B-Instruct" or
            config.get("vision") != "google/siglip2-so400m-patch14-384" or
            config.get("recipe") != "4q" or config.get("stage") != "SFT" or done.get("steps") != 170):
        raise ValueError("Probe is restricted to the current completed SmallThinker4q SFT")
    manager = read_json(run / ".checkpoint_manager.json")
    checkpoint = (run / "best/state.pt").resolve(strict=True)
    if checkpoint.parent.parent != (run / ".checkpoint_objects").resolve():
        raise ValueError("Checkpoint is outside the owned object store")
    marker = read_json(checkpoint.parent / ".research_checkpoint.json")
    if (marker.get("owner") != manager.get("owner") or not manager.get("owner") or
            marker.get("training_completed") is not True or marker.get("resumable_optimizer") is not False):
        raise ValueError("Best checkpoint is not a completed owned compact artifact")
    identity = checkpoint_identity(checkpoint)
    manifest = root / MANIFEST
    raw = manifest.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    indices = selected_indices(rows)
    native = run / "evaluation/heldout_v3"
    protocol = read_json(native / "protocol.json")
    summary = read_json(native / "summary.json")
    model_provenance = read_json(native / "model_provenance.json")
    if (model_provenance.get("recipe") != "4q" or model_provenance.get("stage") != "SFT" or
            model_provenance.get("training_step") != marker.get("step") or
            model_provenance.get("model_ref", {}).get("repo") != config["model"] or
            model_provenance.get("vision_ref", {}).get("repo") != config["vision"]):
        raise ValueError("Preserved native B1 model provenance does not match the selected checkpoint")
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (protocol.get("checkpoint_sha256") != identity["sha256"] or
            protocol.get("manifest_sha256") != hashlib.sha256(raw).hexdigest() or
            protocol.get("schema_version") != 1 or protocol.get("manifest_records") != 512 or
            protocol.get("image_ablation", {}).get("variant") != "none" or
            protocol.get("selected_record_ids_sha256") != hashlib.sha256(json.dumps([r["record_id"] for r in rows]).encode()).hexdigest() or
            protocol.get("image_ablation", {}).get("source_manifest_sha256") != hashlib.sha256(raw).hexdigest() or
            protocol.get("image_ablation", {}).get("transformed_manifest_sha256") != hashlib.sha256(raw).hexdigest() or
            protocol.get("generation") != {"do_sample": False, "num_beams": 1, "use_cache": True,
                "max_new_tokens": 64, "autocast": "cuda_bf16_preserving_master_dtypes",
                "prompt": "native_user_only_image_marker_no_gold", "logits_to_keep": "1_when_supported"}):
        raise ValueError("Preserved native B1 protocol differs")
    versions = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "tokenizers", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    if protocol.get("environment") != {"python": sys.version, "packages": versions}:
        raise ValueError("Native B1 package/Python environment differs")
    sources = {name: file_sha(root / name) for name in FROZEN_SOURCES}
    if protocol.get("source_sha256") != sources:
        raise ValueError("Frozen native B1 inference/scoring source changed")
    predictions = [json.loads(line) for line in (native / "predictions.jsonl").read_text().splitlines() if line.strip()]
    if [r["record_id"] for r in predictions] != [r["record_id"] for r in rows]:
        raise ValueError("Native B1 cache order/coverage differs")
    for record, prediction in zip(rows, predictions):
        if (prediction.get("protocol_sha256") != protocol_sha or
                any(prediction.get(key) != record.get(key) for key in ROW_FIELDS) or
                prediction.get("score") != score_prediction(record, prediction.get("prediction", ""))):
            raise ValueError("Preserved native B1 cache content/score/provenance differs")
        token_parity(record, prediction, prediction)
    measured = aggregate_scores(predictions)
    if (summary.get("complete_manifest") is not True or summary.get("protocol_sha256") != protocol_sha or
            summary.get("checkpoint_sha256") != identity["sha256"] or summary.get("manifest_sha256") != hashlib.sha256(raw).hexdigest() or
            summary.get("manifest_records") != 512 or summary.get("image_ablation") != protocol["image_ablation"] or
            any(summary.get(key) != value for key, value in measured.items())):
        raise ValueError("Native B1 summary does not reproduce from immutable raw predictions")
    sources[SCRIPT] = file_sha(root / SCRIPT)
    sources["tests/test_research_parallel_b1.py"] = file_sha(root / "tests/test_research_parallel_b1.py")
    selected = [rows[i] for i in indices]
    # Canonical RGB hashes verify the actual images once before workers exist.
    image_content_hash = frozen_image_hasher(root)
    for record in selected:
        if (len(record.get("image_paths", [])) != 1 or len(record.get("image_hashes", [])) != 1 or
                image_content_hash(record["image_paths"][0]) != record["image_hashes"][0]):
            raise ValueError("Selected image/RGB hash changed")
    return {"source_run": str(run.resolve()), "checkpoint": str(checkpoint), "checkpoint_identity": identity,
            "manifest": str(manifest.resolve()), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "native_cache": str(native.resolve()), "native_protocol_sha256": protocol_sha,
            "model_provenance": model_provenance,
            "native_file_sha256": {name: file_sha(native / name) for name in
                                   ("protocol.json", "predictions.jsonl", "summary.json", "model_provenance.json")},
            "source_sha256": sources, "environment": protocol["environment"],
            "indices": indices, "record_ids": [r["record_id"] for r in selected],
            "selection": "first8 per benchmark in original512 manifest order; each worker repeats same64",
            "max_new_tokens": 64, "batch_size": 1, "workers": 2,
            "cache_dir": str((root / config["cache_dir"]).resolve())}


def verify_context(context, root=ROOT):
    from src.utils.research_checkpoint_compat import checkpoint_identity, same_checkpoint_identity
    if any(file_sha(root / name) != value for name, value in context["source_sha256"].items()):
        raise ValueError("Probe/frozen source changed")
    if file_sha(context["manifest"]) != context["manifest_sha256"]:
        raise ValueError("Probe manifest changed")
    for name, expected in context["native_file_sha256"].items():
        if file_sha(Path(context["native_cache"]) / name) != expected:
            raise ValueError("Preserved native B1 cache changed")
    if not same_checkpoint_identity(context["checkpoint_identity"], checkpoint_identity(context["checkpoint"])):
        raise ValueError("Probe checkpoint identity changed")


def arm_parent_death(parent_pid):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    if os.getppid() != parent_pid:
        os._exit(128 + signal.SIGKILL)


def barrier_worker(barrier, index, parent_pid, parent_birth, nonce, self_info, *, lookup=None):
    lookup = process_info if lookup is None else lookup
    if (barrier.get("owner") != OWNER or barrier.get("nonce") != nonce or
            barrier.get("status") != "released" or barrier.get("batch_size") != 1 or
            barrier.get("max_new_tokens") != 64 or len(barrier.get("workers", [])) != 2):
        raise ValueError("Fresh owned allocation barrier is not released")
    parent = barrier.get("parent", {})
    if (parent.get("pid") != parent_pid or parent.get("start_time") != parent_birth or
            not same_process(parent, lookup(parent_pid)) or self_info.get("ppid") != parent_pid):
        raise ValueError("Probe parent identity changed")
    workers = barrier["workers"]
    if index not in (0, 1) or not same_process(workers[index], self_info):
        raise ValueError("Worker identity/argv differs from allocation barrier")
    if len({worker["pid"] for worker in workers}) != 2:
        raise ValueError("Barrier does not bind two distinct owned children")
    for worker in workers:
        live = lookup(worker["pid"])
        if not same_process(worker, live) or live.get("ppid") != parent_pid:
            raise ValueError("A peer worker PID was replaced or reparented")
    if time.monotonic() >= barrier["deadline_monotonic"]:
        raise TimeoutError("Whole-probe300s deadline expired")
    return barrier


def stop_owned(process, expected, *, lookup=None):
    """Never signal a group, another evaluator/controller, or a reused PID."""
    lookup = process_info if lookup is None else lookup
    if process is None or process.poll() is not None or expected is None:
        return False
    if not same_process(expected, lookup(expected["pid"])):
        return False
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if same_process(expected, lookup(expected["pid"])):
            process.kill(); process.wait(timeout=2)
    return True


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Whole-probe300s deadline exceeded")


def _read_selected(context):
    rows = [json.loads(line) for line in Path(context["manifest"]).read_text().splitlines() if line.strip()]
    baseline = [json.loads(line) for line in (Path(context["native_cache"]) / "predictions.jsonl").read_text().splitlines() if line.strip()]
    records = [rows[i] for i in context["indices"]]
    expected = [baseline[i] for i in context["indices"]]
    if [r["record_id"] for r in records] != context["record_ids"]:
        raise ValueError("Selected64 manifest IDs changed")
    return records, expected


def worker_main(args):
    """No Torch or CUDA is imported before the released allocation barrier."""
    parent = process_info(args.parent_pid)
    if parent is None or parent["start_time"] != args.parent_start_time:
        raise ValueError("Worker parent identity is stale")
    arm_parent_death(args.parent_pid)
    path = Path(args.barrier).resolve(strict=True)
    if path.name != "allocation_barrier.json" or path.parent.parent != (ROOT / "research_results").resolve():
        raise ValueError("Barrier is outside the dedicated fresh probe directory")
    output = path.parent
    progress = output / f"worker{args.worker}.progress.json"
    atomic_json(progress, {"phase": "waiting_allocation_barrier", "completed_records": 0, "total_records": 64})
    deadline = time.monotonic() + 30
    while True:
        barrier = read_json(path)
        if barrier.get("status") == "released":
            break
        if barrier.get("owner") != OWNER or barrier.get("nonce") != args.nonce:
            raise ValueError("Waiting barrier ownership changed")
        _check_deadline(deadline); time.sleep(.1)
    own = process_info(os.getpid())
    barrier_worker(barrier, args.worker, args.parent_pid, args.parent_start_time, args.nonce, own)
    barrier_sha = file_sha(path)
    context = barrier["context"]
    verify_context(context)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != barrier["gpu_uuid"]:
        raise ValueError("Worker CUDA visibility is not bound to the NVML GPU0 UUID")
    validate_gpu(gpu_snapshot(), barrier["workers"], expected_uuid=barrier["gpu_uuid"])
    # Imports and initialization happen only AFTER both identities were bound.
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("Worker initialized CUDA before its allocation barrier")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(PYTORCH_CAP / total, device=0)
    from src.eval.research_inference import load_research_checkpoint, generate_prediction
    from src.eval.research_metrics import score_prediction, aggregate_scores
    atomic_json(progress, {"phase": "loading_model", "completed_records": 0, "total_records": 64,
                           "pytorch_cap_bytes": PYTORCH_CAP})
    model, provenance = load_research_checkpoint(context["checkpoint"], cache_dir=context["cache_dir"], reserve_gb=120)
    if provenance != context["model_provenance"]:
        raise ValueError("Reloaded model provenance differs from preserved native B1")
    records, expected = _read_selected(context)
    torch.cuda.synchronize()
    atomic_json(progress, {"phase": "model_ready", "completed_records": 0, "total_records": 64,
                           "allocation_barrier_sha256": barrier_sha})
    generation_path = output / "generation_barrier.json"
    while not generation_path.exists():
        _check_deadline(barrier["deadline_monotonic"])
        barrier_worker(barrier, args.worker, args.parent_pid, args.parent_start_time, args.nonce, process_info(os.getpid()))
        time.sleep(.1)
    generation = read_json(generation_path)
    if (generation.get("owner") != OWNER or generation.get("nonce") != args.nonce or
            generation.get("allocation_barrier_sha256") != barrier_sha or generation.get("status") != "released"):
        raise ValueError("Generation barrier does not bind the allocation barrier")
    validate_gpu(gpu_snapshot(), barrier["workers"], expected_uuid=barrier["gpu_uuid"])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic(); predictions = []
    for record, cached in zip(records, expected):
        _check_deadline(barrier["deadline_monotonic"])
        if file_sha(path) != barrier_sha:
            raise ValueError("Immutable allocation barrier changed")
        if not same_process(barrier["parent"], process_info(args.parent_pid)):
            raise ValueError("Probe parent PID changed")
        row = generate_checked_record(model, record, cached, generator=generate_prediction, scorer=score_prediction)
        torch.cuda.synchronize()
        predictions.append(row)
        elapsed = time.monotonic() - started
        atomic_json(progress, {"phase": "native_b1_generation", "worker": args.worker,
                               "completed_records": len(predictions), "total_records": 64,
                               "elapsed_seconds": elapsed, "batch_size": 1})
    ended = time.monotonic()
    verify_context(context)
    if file_sha(path) != barrier_sha:
        raise ValueError("Allocation barrier changed after generation")
    result = {"owner": OWNER, "status": "passed", "worker": args.worker, "identity": own,
              "allocation_barrier_sha256": barrier_sha, "context_sha256": digest(context),
              "batch_size": 1, "records": 64, "record_ids": context["record_ids"],
              "native_b1_exact_token_parity": True, "model_provenance": provenance,
              "generation_started_monotonic": started, "generation_ended_monotonic": ended,
              "generation_seconds": ended - started, "records_per_second": 64 / (ended - started),
              "peak_pytorch_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_pytorch_reserved_bytes": torch.cuda.max_memory_reserved(),
              "pytorch_cap_bytes": PYTORCH_CAP, "subset_scores": aggregate_scores(predictions),
              "predictions": predictions}
    atomic_json(output / f"worker{args.worker}.result.json", result)
    atomic_json(progress, {"phase": "completed", "completed_records": 64, "total_records": 64,
                           "elapsed_seconds": ended - started, "batch_size": 1})
    return result


def validate_results(context, barrier, results):
    from src.eval.research_metrics import aggregate_scores, score_prediction
    records, expected = _read_selected(context)
    generation = barrier.get("generation_barrier", {})
    released = generation.get("released_monotonic")
    observed = barrier.get("observed_workers_done_monotonic")
    if (generation.get("owner") != OWNER or generation.get("status") != "released" or
            generation.get("nonce") != barrier["nonce"] or
            any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                for v in (released, observed)) or not 0 <= released < observed <= barrier["deadline_monotonic"]):
        raise ValueError("Generation timing is not bound to the parent's released/observed window")
    for index, result in enumerate(results):
        if (result.get("status") != "passed" or result.get("owner") != OWNER or result.get("worker") != index or
                result.get("context_sha256") != digest(context) or result.get("batch_size") != 1 or
                result.get("records") != 64 or result.get("record_ids") != context["record_ids"] or
                result.get("native_b1_exact_token_parity") is not True or
                result.get("model_provenance") != context["model_provenance"] or
                result.get("allocation_barrier_sha256") != generation.get("allocation_barrier_sha256") or
                not same_process(barrier["workers"][index], result.get("identity"))):
            raise ValueError("Worker result provenance/parity evidence differs")
        seconds = result.get("generation_seconds")
        start = result.get("generation_started_monotonic")
        end = result.get("generation_ended_monotonic")
        if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0 or
                any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in (start, end)) or
                not released <= start < end <= observed or
                not math.isclose(seconds, end - start, rel_tol=1e-8) or
                not math.isclose(result.get("records_per_second", 0), 64 / seconds, rel_tol=1e-8) or
                result.get("peak_pytorch_allocated_bytes", PROCESS_CAP) > PYTORCH_CAP or
                result.get("peak_pytorch_reserved_bytes", PROCESS_CAP) > PYTORCH_CAP or
                result.get("pytorch_cap_bytes") != PYTORCH_CAP):
            raise ValueError("Worker timing/PyTorch memory bound is invalid")
        predictions = result.get("predictions", [])
        if len(predictions) != 64:
            raise ValueError("Worker lacks all64 predictions")
        for record, generated, cached in zip(records, predictions, expected):
            if any(generated.get(key) != record.get(key) for key in ROW_FIELDS):
                raise ValueError("Worker result order/question/image/source/scoring provenance changed")
            token_parity(record, generated, cached)
            if generated.get("score") != score_prediction(record, generated["prediction"]):
                raise ValueError("Worker score does not reproduce")
        if result.get("subset_scores") != aggregate_scores(predictions):
            raise ValueError("Worker subset metrics do not reproduce")
    starts = [r["generation_started_monotonic"] for r in results]
    ends = [r["generation_ended_monotonic"] for r in results]
    overlap = min(ends) - max(starts)
    if overlap <= 0:
        raise ValueError("Workers did not actually overlap during generation")
    seconds = max(ends) - min(starts)
    return {"generation_window_seconds": seconds, "overlap_seconds": overlap,
            "aggregate_duplicate_completions": 128, "distinct_original_records": 64,
            "aggregate_records_per_second": 128 / seconds}


def run_probe(output, *, allow_gpu=False):
    if not allow_gpu:
        raise ValueError("Actual GPU probe requires explicit --allow-gpu")
    started = time.monotonic(); deadline = started + EXECUTION_SECONDS
    output = Path(output).resolve()
    if output.parent != (ROOT / "research_results").resolve() or output.exists():
        raise ValueError("Use a fresh direct research_results child directory; existing artifacts are preserved")
    initial = validate_gpu(gpu_snapshot(), exclusive=True)
    context = prepare_context()
    _check_deadline(deadline)
    from src.utils.research_storage import DiskBudget
    DiskBudget(ROOT, 120).ensure_space(32 * 2 ** 20)
    output.mkdir()
    lock = (output / ".probe.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = process_info(os.getpid())
    if parent is None:
        raise RuntimeError("Cannot bind probe parent identity")
    nonce = secrets.token_hex(32)
    path = output / "allocation_barrier.json"
    barrier = {"owner": OWNER, "status": "preparing", "nonce": nonce, "parent": parent,
               "workers": [], "batch_size": 1, "max_new_tokens": 64, "context": context,
               "deadline_monotonic": deadline, "created_wall_time": time.time(), "gpu_uuid": initial["gpu_uuid"]}
    atomic_json(path, barrier)
    children = []; identities = []; handles = []; report = {"owner": OWNER, "status": "starting",
        "context": context, "scope": "isolated two-process nativeB1 fixed64 parity/throughput prototype",
        "initial_gpu": initial, "no_production_profile_changes": True, "probe_started_monotonic": started,
        "whole_probe_limit_seconds": MAX_SECONDS, "execution_limit_seconds": EXECUTION_SECONDS,
        "limitations": ["Each worker repeats the same64 records;128 completions are duplicate diagnostics.",
            "Matched preserved none-cache records only; no claim for all512, other image modes, recipes, encoders or campaign cost.",
            "NVML ceilings are sampled;8GiB PyTorch allocator cap does not cover all non-PyTorch CUDA allocations."]}
    report_path = output / "report.json"
    previous_handlers = {}
    def interrupted(signum, _frame):
        raise InterruptedError("Probe received signal " + str(signum))
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        for index in (0, 1):
            argv = [sys.executable, "-u", str(ROOT / SCRIPT), "--worker", str(index), "--barrier", str(path),
                    "--parent-pid", str(parent["pid"]), "--parent-start-time", str(parent["start_time"]), "--nonce", nonce]
            handle = (output / f"worker{index}.log").open("w"); handles.append(handle)
            child = subprocess.Popen(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": initial["gpu_uuid"]},
                                     preexec_fn=lambda: arm_parent_death(parent["pid"]))
            children.append(child)
            identity = process_info(child.pid)
            if identity is None or identity["argv"] != argv or identity["ppid"] != parent["pid"]:
                raise RuntimeError("Cannot bind new child PID/start/argv")
            identities.append(identity)
        barrier["workers"] = identities
        validate_gpu(gpu_snapshot(), exclusive=True, expected_uuid=initial["gpu_uuid"])
        verify_context(context)
        _check_deadline(deadline)
        if any(child.poll() is not None for child in children):
            raise RuntimeError("A worker exited before the allocation barrier")
        barrier["status"] = "released"
        atomic_json(path, barrier)
        barrier_sha = file_sha(path)
        report.update(status="running", workers=identities, allocation_barrier_sha256=barrier_sha)
        atomic_json(report_path, report)
        generation_released = False; observations = []; last_heartbeat = 0; observed_done = None
        owned_processes = {child.pid: child for child in children}
        def confirm_owned_exit(pid):
            child = owned_processes.get(pid)
            return child is not None and child.poll() == 0
        while True:
            _check_deadline(deadline)
            exited = {child.pid for child in children if child.poll() == 0}
            snapshot = validate_gpu(gpu_snapshot(), identities, exited_pids=exited, expected_uuid=initial["gpu_uuid"],
                                    confirm_owned_exit=confirm_owned_exit)
            observations.append({"monotonic": time.monotonic(), **snapshot})
            phases = []
            for index, child in enumerate(children):
                status = child.poll()
                if status not in (None, 0):
                    raise RuntimeError(f"Owned worker{index} failed with exit{status}; see its separate log")
                if status is None and not same_process(identities[index], process_info(child.pid)):
                    raise RuntimeError("Owned worker PID was reused/changed")
                progress = output / f"worker{index}.progress.json"
                phases.append(read_json(progress) if progress.exists() else {})
            if not generation_released and all(row.get("phase") == "model_ready" for row in phases):
                atomic_json(output / "generation_barrier.json", {"owner": OWNER, "status": "released", "nonce": nonce,
                    "allocation_barrier_sha256": barrier_sha, "released_monotonic": time.monotonic()})
                generation_released = True
            if all(child.poll() == 0 for child in children):
                if not generation_released:
                    raise RuntimeError("Workers exited without the generation barrier")
                observed_done = time.monotonic()
                break
            if time.monotonic() - last_heartbeat >= 30:
                public = [{key: row[key] for key in ("phase", "completed_records", "total_records", "elapsed_seconds") if key in row}
                          for row in phases]
                print(json.dumps({"event": "parallel_b1_probe", "elapsed_seconds": time.monotonic() - started,
                                  "workers": public, "gpu_free_gib": snapshot["free_bytes"] / GIB}), flush=True)
                report.update(worker_progress=public, gpu_observations=observations)
                atomic_json(report_path, report); last_heartbeat = time.monotonic()
            time.sleep(.5)
        results = [read_json(output / f"worker{i}.result.json") for i in (0, 1)]
        if any(result.get("allocation_barrier_sha256") != barrier_sha for result in results):
            raise ValueError("Worker results do not bind the released allocation barrier")
        validation_barrier = dict(barrier, generation_barrier=read_json(output / "generation_barrier.json"),
                                  observed_workers_done_monotonic=observed_done)
        metrics = validate_results(context, validation_barrier, results)
        verify_context(context)
        _check_deadline(deadline)
        elapsed = time.monotonic() - started
        report.update(status="passed", gpu_observations=observations, **metrics,
                      workers=[{key: value for key, value in result.items() if key != "predictions"} for result in results],
                      wall_seconds=elapsed, aggregate_end_to_end_records_per_second=128 / elapsed,
                      observed_gpu_processes_only_owned_workers=True)
        atomic_json(report_path, report)
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), wall_seconds=time.monotonic() - started)
        atomic_json(report_path, report)
        raise
    finally:
        cleanup_errors = []
        for index, child in enumerate(children):
            try:
                stop_owned(child, identities[index] if index < len(identities) else None)
            except Exception as error:
                cleanup_errors.append({"pid": child.pid, "error": str(error)})
        for handle in handles:
            handle.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        lock.close()
        if cleanup_errors:
            report.update(status="failed", cleanup_errors=cleanup_errors)
            atomic_json(report_path, report)
            raise RuntimeError("Owned worker cleanup did not complete; see the failed probe receipt")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="research_results/parallel_native_b1_4q_probe_v1")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--barrier", help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--parent-start-time", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--nonce", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is not None:
        if not all(value is not None for value in (args.barrier, args.parent_pid, args.parent_start_time, args.nonce)):
            parser.error("Private workers require a complete parent/barrier identity")
        return worker_main(args)
    if args.preflight_only:
        result = prepare_context()
        print(json.dumps({"status": "cpu_preflight_passed", "context_sha256": digest(result),
                          "record_ids": result["record_ids"], "no_gpu_initialized": True}), flush=True)
        return result
    return run_probe(ROOT / args.output, allow_gpu=args.allow_gpu)


if __name__ == "__main__":
    main()
