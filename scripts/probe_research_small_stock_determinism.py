#!/usr/bin/env python3
"""Tiny opt-in CUDA stock-determinism probe; no HF weights or training jobs.

Root may run with an external timeout180s after GPU work is idle. Default mode
prints CPU metadata without importing Torch. No existing file is overwritten.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORKSPACE = ":4096:8"
# This executes before any Torch import or CUDA initialization.
if os.environ.get("CUBLAS_WORKSPACE_CONFIG", WORKSPACE) != WORKSPACE:
    raise RuntimeError("This explicit profile requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = WORKSPACE


def nvml_processes():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"
    ], text=True, timeout=10)
    found = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        pid, mib = [value.strip() for value in line.split(",")]
        found[int(pid)] = int(mib) * 2**20
    return found


def atomic(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2) + "\n"
    temp = path.with_name("." + path.name + ".tmp")
    with temp.open("x") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def metadata():
    names = ["scripts/probe_research_small_stock_determinism.py", "src/model/smallthinker.py", "src/model/research_moe.py"]
    return {
        "schema_version": 1, "profile": "tiny_small_stock_determinism_probe_v1",
        "status": "cpu_metadata_only", "gpu_used": False,
        "torch_distribution_version": importlib.metadata.version("torch"),
        "workspace_config": WORKSPACE, "state_master_dtype": "float32",
        "compute_dtype": "bfloat16_autocast", "matmul_allow_tf32": True,
        "torch_threads": 1,
        "shape": {"batch": 2, "tokens": 17, "hidden": 64, "intermediate": 128, "old_experts": 8, "new_experts": 2, "top_k": 4},
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
        "limits": ["Tiny native blocks only; no actual HF weights, quantized/PEFT experts, whole-model generation or historical cause claim.",
                   "Exact repeatability is observed evidence, not proof for all shapes/backends.",
                   "Torch caching allocator capped at1GiB; own NVML charge checked against2GiB after each bounded case/repetition."],
    }


def run(args):
    if args.output is None:
        raise ValueError("--allow-gpu requires a new --output directory")
    if args.output.exists():
        raise FileExistsError("Probe output exists; preserve it and choose a new namespace")
    if nvml_processes():
        raise RuntimeError("GPU is occupied; this probe refuses concurrent CUDA processes")
    import torch
    from src.model.smallthinker import SmallThinkerConfig, SmallThinkerMoeBlock
    from src.model.research_moe import ResearchSmallThinkerMoE
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Probe requires exactly one visible CUDA GPU")
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = True
    args.output.mkdir(parents=True)
    result = metadata()
    result.update(status="running", gpu_used=True, pid=os.getpid(), started_unix=time.time(),
                  repeats=args.repeats, torch_version=torch.__version__, torch_git=torch.version.git_version,
                  cuda_version=torch.version.cuda, device_name=torch.cuda.get_device_name(0),
                  device_capability=list(torch.cuda.get_device_capability(0)), cases=[])
    atomic(args.output/"report.json", result)

    def guard():
        processes = nvml_processes()
        if set(processes) - {os.getpid()}:
            raise RuntimeError("Foreign CUDA process appeared; refusing overlapping probe")
        used = processes.get(os.getpid(), 0)
        if used > 2*2**30 or torch.cuda.max_memory_reserved() > 2**30:
            raise RuntimeError("Tiny probe exceeded its process/allocator memory budget")
        return {"nvml_process_bytes": used, "torch_peak_allocated": torch.cuda.max_memory_allocated(),
                "torch_peak_reserved": torch.cuda.max_memory_reserved()}

    def digest(tensor):
        data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    try:
        guard()
        variants = [("native", None, None)] + [(str(f)+p, f, p) for f in (1, 2, 3) for p in ("a", "b", "c")]
        for name, family, policy in variants:
            torch.manual_seed(3100 + len(result["cases"]))
            cfg = SmallThinkerConfig(vocab_size=128, hidden_size=64, num_hidden_layers=1,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                moe_ffn_hidden_size=128, moe_num_primary_experts=8, moe_num_active_primary_experts=4,
                pad_token_id=0, bos_token_id=1, eos_token_id=2)
            block = SmallThinkerMoeBlock(cfg)
            if family is not None:
                block = ResearchSmallThinkerMoE(block, family, policy, added_experts=2, router_width=32)
            block = block.float().cuda()
            pristine = {key: value.detach().clone() for key, value in block.state_dict().items()}
            values = torch.randn(2, 17, 64).cuda()
            loss_weight = torch.randn_like(values)
            if family is not None:
                visual = torch.arange(34, device="cuda").reshape(2, 17) % 3 == 0
                attention = torch.ones_like(visual); attention[-1, -1] = False
                block.set_modality_mask(visual, attention)
            for dispatch in ("eager", "grouped"):
                block.use_grouped_mm = dispatch == "grouped"
                for phase in ("train_forward_backward", "frozen_inference_cache"):
                    block.requires_grad_(phase == "train_forward_backward")
                    if family == 1:
                        block.primary_router.requires_grad_(False)
                    block.train(phase == "train_forward_backward")
                    for mode in (False, True):
                        torch.use_deterministic_algorithms(mode, warn_only=False)
                        item = {"variant": name, "dispatch": dispatch, "phase": phase,
                                "deterministic_algorithms": mode, "repetitions": [], "grouped_calls": 0}
                        started = time.monotonic()
                        original_grouped = torch._grouped_mm
                        def observed_grouped(*pos, **kw):
                            item["grouped_calls"] += 1
                            return original_grouped(*pos, **kw)
                        # Instrument only this isolated probe process, restore on every exit.
                        torch._grouped_mm = observed_grouped
                        try:
                            for repeat in range(args.repeats):
                                block.load_state_dict(pristine, strict=True)
                                block.zero_grad(set_to_none=True)
                                x = values.detach().clone().requires_grad_(phase == "train_forward_backward")
                                with torch.set_grad_enabled(phase == "train_forward_backward"), torch.autocast("cuda", dtype=torch.bfloat16):
                                    output = block(x, x)[0]
                                    if phase == "train_forward_backward":
                                        (output*loss_weight).sum().backward()
                                    else:
                                        warm_output = block(x, x)[0]
                                torch.cuda.synchronize()
                                tensors = {"output": output}
                                if phase == "frozen_inference_cache":
                                    tensors["warm_cache_output"] = warm_output
                                if x.grad is not None:
                                    tensors["input_gradient"] = x.grad
                                tensors.update({"gradient:"+key: p.grad for key, p in block.named_parameters() if p.grad is not None})
                                if not all(torch.isfinite(v).all().item() for v in tensors.values()):
                                    raise ValueError("Nonfinite tiny output or gradient")
                                item["repetitions"].append({"hashes": {key:digest(v) for key,v in tensors.items()}, "memory":guard()})
                                if phase == "frozen_inference_cache":
                                    item["repetitions"][-1]["same_cold_warm_cache_output_bits"] = torch.equal(output, warm_output)
                                    del warm_output
                                del x, output, tensors
                            reference = item["repetitions"][0]["hashes"]
                            item["all_output_and_gradient_bits_repeat_equal"] = all(row["hashes"] == reference for row in item["repetitions"])
                            if phase == "frozen_inference_cache":
                                item["all_output_and_gradient_bits_repeat_equal"] &= all(row["same_cold_warm_cache_output_bits"] for row in item["repetitions"])
                            item["status"] = "passed" if item["all_output_and_gradient_bits_repeat_equal"] else "repeat_difference_observed"
                            if dispatch == "grouped" and item["grouped_calls"] == 0:
                                raise RuntimeError("Grouped case never exercised torch._grouped_mm")
                        except Exception as exc:
                            item.update(status="unsupported_or_failed", error_type=type(exc).__name__, error=str(exc))
                        finally:
                            torch._grouped_mm = original_grouped
                        item["elapsed_seconds"] = time.monotonic()-started
                        result["cases"].append(item)
                        atomic(args.output/"progress.json", {key:item[key] for key in ("variant","dispatch","phase","deterministic_algorithms","status","elapsed_seconds")})
                        print(json.dumps({key:item[key] for key in ("variant","dispatch","phase","deterministic_algorithms","status","elapsed_seconds")}), flush=True)
                        atomic(args.output/"report.next.json", result)
                        os.replace(args.output/"report.next.json", args.output/"report.json")
                        guard()
            del block, pristine, values, loss_weight
            torch.cuda.empty_cache()
        deterministic = [case for case in result["cases"] if case["deterministic_algorithms"]]
        result.update(status="completed", finished_unix=time.time(), deterministic_cases=len(deterministic),
                      all_stock_deterministic_cases_supported_and_bit_exact=all(case["status"] == "passed" for case in deterministic),
                      final_memory=guard(), historical_cause_confirmed=False)
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc), finished_unix=time.time())
        raise
    finally:
        torch.use_deterministic_algorithms(False)
        atomic(args.output/"final.json", result)
        os.replace(args.output/"final.json", args.output/"report.json")
    return 0 if result["all_stock_deterministic_cases_supported_and_bit_exact"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 10:
        parser.error("--repeats must be between2 and10")
    if not args.allow_gpu:
        print(json.dumps(metadata(), indent=2)); return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
