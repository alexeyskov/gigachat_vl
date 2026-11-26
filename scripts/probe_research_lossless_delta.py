#!/usr/bin/env python3
"""CPU-only native-Torch delta prototype on SYNTHETIC updated Small tensors.

No trained checkpoint is read, rewritten or deleted. Ratios are measurements on
deterministic artificial updates, not trained-weight compressibility estimates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from safetensors import safe_open
    from src.utils import research_lossless_delta as delta
    from src.utils.research_checkpoint_compat import load_research_payload
    from src.utils.research_storage import DiskBudget

    torch.set_num_threads(4)
    torch.manual_seed(20261122)
    if torch.cuda.is_initialized():
        raise RuntimeError("This proof must remain CPU-only")
    output = ROOT / "research_results/lossless_delta_synthetic_probe"
    output.mkdir(parents=True, exist_ok=True)
    DiskBudget(output, 120).ensure_space(256 * 2 ** 20)
    snapshot = Path("/workspace/.hf_home/hub/models--Tiiny--SmallThinker-4BA0.6B-Instruct/snapshots") / delta.REVISION
    anchor = delta.SmallAnchor(snapshot)
    keys = ["model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.block_sparse_moe.experts.0.up.weight",
            "model.layers.0.block_sparse_moe.primary_router.weight"]
    model, source = {}, []
    for key in keys:
        desc = anchor.descriptor(key)
        with safe_open(snapshot / desc["shard"], framework="pt", device="cpu") as opened:
            reference = opened.get_tensor(key)
        if sum(v.numel() * 4 for v in model.values()) + reference.numel() * 4 > 32 * 2 ** 20:
            raise ValueError("Synthetic tensor probe exceeds32MiB")
        # Deterministic artificial perturbation; NEVER label this trained.
        update = ((torch.arange(reference.numel(), dtype=torch.float32) % 127) - 63).reshape(reference.shape) * 1e-7
        model["llm." + key] = reference.float() + update
        source.append(dict(desc, bf16_tensor_sha256=hashlib.sha256(delta._raw(reference)).hexdigest()))
    model["projector.weight"] = torch.randn(32, 16)
    model["llm.model.layers.0.block_sparse_moe.visual_experts.0.up.weight"] = torch.randn(32, 16)
    model["llm.model.layers.1.block_sparse_moe.primary_router.weight"] = torch.randn(34, 1536)
    payload = {"model": model, "optimizer": {"state": {0: {"exp_avg": torch.randn(32, 16), "exp_avg_sq": torch.rand(32, 16)}}},
               "rng_torch": torch.get_rng_state(), "step": 0,
               "model_ref": {"repo": delta.REPO, "revision": delta.REVISION},
               "protocol": "SYNTHETIC pinned BF16 + deterministic FP32 perturbation; NOT TRAINED"}
    start = time.monotonic()
    prepared, receipt = delta.prepare_delta_payload(payload, snapshot)
    encode_seconds = time.monotonic() - start

    def compare(a, b):
        if isinstance(a, torch.Tensor):
            if type(b) is not torch.Tensor or a.shape != b.shape or a.dtype != b.dtype or delta._raw(a) != delta._raw(b):
                raise AssertionError("Tensor bits/type/shape/dtype changed")
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                raise AssertionError("Payload keys changed")
            for k in a:
                compare(a[k], b[k])
        elif a != b:
            raise AssertionError("Non-tensor state changed")

    baseline = output / "synthetic_native.pt"
    torch.save(payload, baseline)
    measurements = []
    for protocol, legacy in [(2, False), (4, False), (4, True)]:
        path = output / f"synthetic_delta_p{protocol}_{'legacy' if legacy else 'zip'}.pt"
        start = time.monotonic()
        torch.save(prepared, path, pickle_protocol=protocol, _use_new_zipfile_serialization=not legacy)
        save_seconds = time.monotonic() - start
        preflight = delta.preflight_delta_load(path, decoded_bytes_budget=receipt["raw_matched_tensor_bytes"],
            largest_tensor_bytes=receipt["largest_encoded_tensor_decoded_bytes"])
        start = time.monotonic()
        restored = torch.load(path, weights_only=False, map_location="cpu", mmap=not legacy)
        compare(payload, restored)
        native_seconds = time.monotonic() - start
        start = time.monotonic()
        restored = load_research_payload(path, map_location="cpu", mmap=True)
        compare(payload, restored)
        compat_seconds = time.monotonic() - start
        pickle_size = None
        if not legacy:
            with zipfile.ZipFile(path) as archive:
                pickle_size = next(i.file_size for i in archive.infolist() if i.filename.endswith("/data.pkl"))
        measurements.append({"path": str(path.relative_to(ROOT)), "pickle_protocol": protocol,
                             "serialization": "legacy" if legacy else "torch_zip", "bytes": path.stat().st_size,
                             "sha256": delta._file_sha(path), "data_pkl_bytes": pickle_size,
                             "ratio_vs_synthetic_native_whole_payload": path.stat().st_size / baseline.stat().st_size,
                             "save_seconds": save_seconds, "native_load_verify_seconds": native_seconds,
                             "unchanged_compat_load_verify_seconds": compat_seconds,
                             "all_tensor_and_untouched_state_bits_equal": True, "load_preflight": preflight})
    # Fresh interpreter: Torch's ordinary pickle import must find the decoder
    # itself. No prototype module is explicitly imported before torch.load.
    verifier = r'''
import json,sys,torch
from pathlib import Path
root=Path.cwd()/"research_results/lossless_delta_synthetic_probe"
assert "src.utils.research_lossless_delta" not in sys.modules
baseline=torch.load(root/"synthetic_native.pt",weights_only=False,mmap=True,map_location="cpu")
def eq(a,b):
    if isinstance(a,torch.Tensor):
        assert type(b) is torch.Tensor and a.shape==b.shape and a.dtype==b.dtype
        assert torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8))
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:eq(a[k],b[k])
    else:assert a==b
for filename,mmap in [("synthetic_delta_p4_zip.pt",True),("synthetic_delta_p4_legacy.pt",False)]:
    eq(baseline,torch.load(root/filename,weights_only=False,mmap=mmap,map_location="cpu"))
assert "src.utils.research_lossless_delta" in sys.modules
assert not torch.cuda.is_initialized()
print(json.dumps({"status":"passed","decoder_autoimported":True,"zip_and_legacy_plain_tensor_bits_exact":True,"cuda_initialized":False}))
'''
    child = subprocess.run([sys.executable, "-c", verifier], cwd=ROOT, capture_output=True, text=True, timeout=60)
    if child.returncode:
        raise RuntimeError("Fresh native reader failed: " + child.stderr[-4000:])
    fresh_reader = json.loads(child.stdout)
    # Existing JSON count proof only, not a full tensor/checkpoint allocation.
    count_path = ROOT / "runs/research/smallthinker4__siglip2__4f__SFT_smoke_s42/trainable_parameters.json"
    counts = json.loads(count_path.read_text())
    raw_full = sum(counts.values()) * 4
    largest = max(counts.values()) * 4
    if raw_full > delta.MAX_CHECKPOINT_DECODED_BYTES or largest > delta.MAX_TENSOR_DECODED_BYTES:
        raise ValueError("Full native Small count proof does not fit prototype bounds")
    encoded_bound = sum(delta._zstd().ZSTD_compressBound(n * 4) for n in counts.values())
    full_estimates = {
        "scope": "allocation estimates only; no full payload created/read/written",
        "count_source": str(count_path.relative_to(ROOT)), "count_source_sha256": hashlib.sha256(count_path.read_bytes()).hexdigest(),
        "original_trainable_tensor_count": len(counts), "raw_delta_eligible_max_bytes": raw_full,
        "largest_original_tensor_bytes": largest, "zstd_total_compress_bound_bytes": encoded_bound,
        "writer_extra_guard_bytes": 4 * raw_full + 6 * largest,
        "writer_required_available_with32GiB_headroom_bytes": 4 * raw_full + 6 * largest + 32 * delta.GIB,
        "worst_case_decoder_extra_guard_bytes": 3 * encoded_bound + 2 * raw_full + 6 * largest,
        "current_probe_available_bytes": receipt["memory_guard"]["available_bytes"],
        "full_writer_would_pass_current_strict_guard": receipt["memory_guard"]["available_bytes"] >= 4 * raw_full + 6 * largest + 32 * delta.GIB,
        "full_checkpoint_memory_validated": False,
    }
    sources = ["src/utils/research_lossless_delta.py", "scripts/probe_research_lossless_delta.py",
               "tests/test_research_lossless_delta.py", "src/utils/research_checkpoint_compat.py",
               "src/utils/research_checkpoint_io.py", "src/utils/research_checkpoint_bundle.py",
               "research_results/lossless_delta_anchor_metadata.json"]
    report = {"schema_version": 1, "status": "passed_tiny_synthetic_cpu_only",
              "scope": "native trusted-pickle compatibility feasibility, NOT full-checkpoint capacity proof",
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
              "source_anchor": source, "synthetic_update": "((arange %127)-63)*1e-7 added in FP32",
              "trained_checkpoint_read": False, "trained_checkpoint_rewritten": False,
              "original_source_mutated": False, "cuda_initialized": torch.cuda.is_initialized(),
              "weight_download_bytes": 0, "baseline_bytes": baseline.stat().st_size,
              "encode_seconds": encode_seconds, "writer_receipt": receipt, "measurements": measurements,
              "fresh_process_native_reader": fresh_reader,
              "full_small_allocation_estimates": full_estimates,
              "process_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "limits": [
                  "Only pinned SmallThinker;32GiB aggregate delta-decoded model/1GiB individual tensor caps; no GPT or other-family claim. Ordinary optimizer/novel storages are separate from that decode cap.",
                  "Native Torch reader requires weights_only=False and CPU map_location; compressed values allocate heap storage despite mmap=True.",
                  "Encoded values bypass Torch map_location transformations. Installed native ZIP's map_location TLS is checked to reject explicit GPU/meta/callable/dict maps; legacy has no such TLS so future callsites must explicitly constrain CPU loads there.",
                  "Protocol4 is needed to store compressed BYTES without protocol2 Latin1-to-UTF8 expansion; existing loaders need no format change.",
                  "Entire ZIP data.pkl is allocated before decoder invocation. A future caller must run external preflight_delta_load BEFORE existing loader. Decoder alone cannot guard that first allocation.",
                  "Peak estimate explicitly charges pickle/BytesIO/unpickler copies, decoded tensors and per-tensor scratch, plus32GiB headroom under host AND170.75GiB cgroup limit; page cache reclaim gets no credit.",
                  "FakeTensorMode inspection verifies shapes/anchor+encoded identities but intentionally does not decompress or verify decoded tensor values.",
                  "Source-SHA pin requires a frozen/versioned decoder module or archived exact decoder for long-term reloads. Later source edits intentionally reject old artifacts.",
                  "No actual full checkpoint, final trained compressibility ratio, memory fit at full size, concurrency throughput, disk-gain guarantee or active integration proved.",
                  "New/mismatched tensors/projector/optimizer/RNG stay ordinary exact values. Identical Tensor-object ties preserved; shared-storage views/model-nonmodel aliases refused.",
                  "The396 primary trajectory matrix still requires external multi-TiB storage; this isolated prototype changes no budgets or experiment criteria."
              ]}
    (ROOT / "research_results/lossless_delta_synthetic_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "raw_matched_bytes": receipt["raw_matched_tensor_bytes"],
                      "encoded_ratio_synthetic_only": receipt["encoded_to_raw_matched_ratio"],
                      "measurements": [{k: x[k] for k in ("pickle_protocol", "serialization", "bytes", "data_pkl_bytes", "ratio_vs_synthetic_native_whole_payload")} for x in measurements],
                      "maxrss_bytes": report["process_maxrss_bytes"], "gpu_used": False}))


if __name__ == "__main__":
    main()
