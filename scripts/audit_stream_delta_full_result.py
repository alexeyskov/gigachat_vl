#!/usr/bin/env python3
"""Read-only independent audit of the ROOT-launched synthetic codec receipts.

This does not repeat the deleted archive's round trip. It checks durable receipt,
shape/header, canonical source bytes, owner/cleanup and memory/log consistency.
No torch/model import, allocation, GPU, downloads, checkpoint rewrite or service.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30


def file_sha(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 2 ** 20), b""):
            digest.update(block)
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    return digest.hexdigest()


def main():
    base = ROOT / "research_results/stream_delta_full_synthetic_v1"
    paths = {n: base / n for n in ("owner.json", "encoding.json", "verified.json", "memory_samples.json")}
    owner, encoded, verified, memory = (json.loads(paths[n].read_text()) for n in paths)
    assert owner["owner"] == encoded["owner"] == verified["owner"]
    assert verified["status"] == "verified_full_stored_synthetic_model_only"
    assert verified["archive_deleted"] and verified["all_model_and_normal_fixture_bits_exact"]
    assert not verified["source_files_changed"] and not verified["cuda_initialized"]
    assert not (base / "failure.json").exists()
    archive = Path(verified["archive_path"])
    assert archive.parent == base and archive.name == "synthetic_model_legacy.pt" and not archive.exists()
    assert set(p.name for p in base.iterdir()) == set(paths)
    assert encoded["header"] == verified["header"] and encoded["receipt"] == verified["receipt"]
    header = verified["header"]
    for p, digest in verified["plan"]["source_sha256"].items():
        assert file_sha(ROOT / p) == digest == owner["source_sha256"][p]
    assert header["adapter_sha256"] == verified["plan"]["source_sha256"]["src/utils/research_stream_delta.py"]
    assert header["v1_sha256"] == verified["plan"]["source_sha256"]["src/utils/research_lossless_delta.py"]
    module = ast.parse((ROOT / "src/utils/research_lossless_delta.py").read_text())
    constants = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"REPO", "REVISION", "INDEX_SHA256", "CONFIG_SHA256", "SHARD_SHA256"}:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    source = {}
    tensor_headers = {}
    for item in header["anchor_files"]:
        snapshot = Path(item["snapshot"])
        assert snapshot.name == constants["REVISION"] == header["revision"]
        assert header["repo"] == constants["REPO"]
        assert file_sha(snapshot / "config.json") == constants["CONFIG_SHA256"]
        assert file_sha(snapshot / "model.safetensors.index.json") == constants["INDEX_SHA256"]
        weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        path = snapshot / item["shard"]
        assert path.stat().st_size == item["bytes"]
        digest = file_sha(path)  # Independent full canonical SHA,8MiB buffer only.
        assert digest == constants["SHARD_SHA256"][item["shard"]]
        source[item["shard"]] = {"sha256": digest, "bytes": path.stat().st_size}
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            assert size < 16 * 2 ** 20
            data = json.loads(f.read(size))
        for name, info in data.items():
            if name == "__metadata__":continue
            assert weight_map[name] == item["shard"] and info["dtype"] == "BF16"
            assert info["data_offsets"][1] - info["data_offsets"][0] == math.prod(info["shape"]) * 2
            tensor_headers["llm." + name] = info["shape"]
    assert set(tensor_headers) == set(encoded["expected"])
    for name, shape in tensor_headers.items():
        assert encoded["expected"][name]["shape"] == shape
        digest = encoded["expected"][name]["decoded_sha256"]
        assert len(digest) == 64 and int(digest, 16) >= 0
    d = sum(math.prod(shape) * 4 for shape in tensor_headers.values())
    assert len(tensor_headers) == 3299 == verified["stored_tensors_verified"] == header["encoded_tensor_count"] == header["decoder_session_count"]
    assert d == 17073838080 == verified["stored_fp32_bytes_verified"] == header["decoded_model_bytes"]
    assert max(math.prod(s) * 4 for s in tensor_headers.values()) == header["largest_decoded_tensor_bytes"] == 933494784
    assert sum(s["bytes"] for s in source.values()) == header["anchor_cache_bytes"]
    c = header["encoded_bytes"]
    assert c == encoded["receipt"]["encoded_bytes"] == 9444421470
    assert verified["archive_bytes"] == verified["task_owned_archive_bytes_reclaimed"] == 9453220558
    assert verified["archive_bytes"] < 12 * GIB - 64 * 2 ** 20 - 2 ** 20
    metadata_blocks = sum(p.stat().st_blocks * 512 for p in paths.values())
    assert metadata_blocks < 64 * 2 ** 20
    assert memory["total_samples"] == len(memory["samples"]) + memory["dropped_samples"]
    assert len(memory["samples"]) <= 32768 and memory["dropped_samples"] == 0
    peak = max(s["cgroup_current_bytes"] for s in memory["samples"])
    assert peak == memory["observed_peak_bytes"] == verified["shared_cgroup_observed_peak_bytes"]
    limit = verified["plan"]["memory"]["cgroup_limit_bytes"]
    assert limit - peak > 32 * GIB
    events = []
    log_path = ROOT / "logs/research/stream_delta_full.log"
    for line in log_path.read_text().splitlines():
        if line.startswith("{"):
            events.append(json.loads(line))
    final = next(e for e in reversed(events) if e.get("event") == "verified")
    assert final["stored_tensors_verified"] == 3299 and final["archive_deleted"] and final["archive_bytes"] == verified["archive_bytes"]
    assert final["elapsed_seconds"] == verified["elapsed_seconds"]
    ready_path = ROOT / "configs/research/grounding_queue_ready.json"
    ready = json.loads(ready_path.read_text())
    differences = [p for p, digest in ready["source_sha256"].items() if file_sha(ROOT / p) != digest]
    assert not differences
    phase_peaks = {phase: max(s["cgroup_current_bytes"] for s in memory["samples"] if s["phase"] == phase)
                   for phase in {s["phase"] for s in memory["samples"]}}
    result = {"schema_version": 1, "status": "independently_audited_receipts_sources_headers_cleanup",
              "audited_at": datetime.now(timezone.utc).isoformat(),
              "audit_code_sha256": file_sha(Path(__file__)),
              "input_receipt_sha256": {n: file_sha(p) for n, p in paths.items()},
              "log_sha256": file_sha(log_path), "canonical_ready_sha256": file_sha(ready_path),
              "all_current_ready_sources_match": True, "ready_sources_checked": len(ready["source_sha256"]),
              "bound_probe_source_sha256": verified["plan"]["source_sha256"],
              "independent_source_weight_sha256": source,
              "all3299_names_shapes_dtype_and_byte_totals_match": True,
              "owner": owner["owner"], "stored_tensors": 3299, "stored_elements": d // 4,
              "raw_fp32_model_bytes": d, "compressed_model_bytes": c,
              "archive_bytes": verified["archive_bytes"], "archive_recorded_sha256": verified["archive_sha256"],
              "compressed_bytes_over_raw_model_ratio": c / d,
              "archive_bytes_over_raw_model_ratio": verified["archive_bytes"] / d,
              "whole_uncompressed_torch_checkpoint_not_written_or_measured": True,
              "elapsed_seconds": verified["elapsed_seconds"],
              "process_maxrss_bytes": verified["process_maxrss_bytes"],
              "shared_observed_peak_bytes": peak, "cgroup_limit_bytes": limit,
              "remaining_at_observed_peak_bytes": limit - peak,
              "shared_phase_observed_peaks_bytes": phase_peaks,
              "memory_samples": len(memory["samples"]), "retained_metadata_allocated_bytes": metadata_blocks,
              "archive_removed_after_durable_verified_receipt": True,
              "archive_present_at_audit": False, "cuda_used": False,
              "limits": ["No second archive read/roundtrip can be independently repeated after intentional verified deletion; model-value equality is attested by the source-bound ROOT worker. This audit independently rehashes canonical original shards and checks every retained shape/count/digest/receipt/cleanup binding.",
                         "Actual ratio is synthetic fixed FP32 updates, not trained checkpoint compressibility or full optimizer memory proof.",
                         "Cgroup peak is250ms observed shared sampling, not an isolated-process/kernel instantaneous peak. Process maxRSS is the worker getrusage metric.",
                         "All storedhead/embed tensors counted independently;4.268B stored elements differ from native4.035B tied parameters."]}
    output = ROOT / "research_results/stream_delta_full_actual_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    text = ["# Full synthetic guarded-delta proof", "",
            "The root-launched CPU worker completed and its retained receipts were independently audited. No trained checkpoint was changed.", "",
            f"All3,299 stored original tensors ({d:,} raw FP32 bytes;15.901GiB) round-tripped exactly in{verified['elapsed_seconds']/60:.2f}minutes. The archive was{verified['archive_bytes']:,}bytes ({verified['archive_bytes']/GIB:.3f}GiB),{verified['archive_bytes']/d:.4%}of raw model-tensor bytes. No full ordinary Torch baseline file was written; this is not a ratio to a measured native checkpoint.", "",
            f"Process maxRSS:{verified['process_maxrss_bytes']/GIB:.3f}GiB. Observed shared cgroup peak:{peak/GIB:.3f}GiB of{limit/GIB:.3f}GiB, leaving{(limit-peak)/GIB:.3f}GiB. Shared sampling does not isolate this worker or prove the kernel instantaneous peak.", "",
            "The owner UUID, producer source hashes, every expected key/shape/FP32 byte count, both canonical BF16 shard SHA256s, normal-storage/header totals, memory samples, final log event and cleanup receipt agree. The task-owned archive is gone; retained metadata is about1MiB. Original source bytes match the pinned canonical digests. Current canonical readiness sources match.", "",
            "The deleted archive cannot be read again for an independent second decode. Its bit-exact model/projector/tinyoptimizer/Torch+NumPy RNG result is attested by the reviewed, hash-bound root worker; this audit verifies all remaining evidence and independently hashes both original weight shards.", "",
            "This proves full stored-model synthetic codec I/O and memory. It does not prove final trained-weight compression, full Adam-state fit, safe checkpoint-manager publication, evaluation-receipt migration or multi-TiB campaign storage sufficiency. Actual trained archival remains a proposal pending the first completed full4f checkpoint.", ""]
    output.with_suffix(".md").write_text("\n".join(text))
    print(json.dumps({"status": result["status"], "stored_tensors": 3299,
                      "archive_gib": verified["archive_bytes"] / GIB,
                      "remaining_at_shared_observed_peak_gib": (limit - peak) / GIB,
                      "archive_present": False, "gpu_used": False}))


if __name__ == "__main__":
    main()
