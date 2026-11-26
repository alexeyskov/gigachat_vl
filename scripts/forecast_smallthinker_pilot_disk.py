#!/usr/bin/env python3
"""Read-only phase-by-phase forecast for the existing eleven SmallThinker pilots.

No trainer, retention rule, checkpoint, source model, or queue is changed.
Raw tensor counts are exact. Future pickle/ZIP overhead and unrelated growth
are explicit allowances, not measured future file sizes or compression claims.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30
MIB = 2 ** 20
SPEC = "configs/research/smallthinker_pilot_scienceqa_context_v2_balanced_queue.json"
STATE = SPEC.replace(".json", ".state.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def args_dict(argv):
    return {v[2:].replace("-", "_"): argv[i + 1]
            for i, v in enumerate(argv[:-1])
            if v.startswith("--") and not argv[i + 1].startswith("--")}


def tree_bytes(directory):
    # Pointer symlinks must never duplicate the immutable object allocation.
    files = [p for p in directory.rglob("*") if p.is_file() and not p.is_symlink()]
    return {"logical_bytes": sum(p.stat().st_size for p in files),
            "allocated_bytes": sum(p.stat().st_blocks * 512 for p in files)}


def checkpoint_sizes(trainable, projector, overhead=0):
    model = 4 * (trainable + projector)
    moments = 8 * trainable  # FP32 AdamW exp_avg and exp_avg_sq, fully initialized.
    # The manager expands the RUNNER estimate by five percent, including 128MiB.
    guard = math.ceil((4 * model + 128 * MIB) * 1.05)
    return {"model_tensor_bytes": model, "optimizer_moment_tensor_bytes": moments,
            "compact_bytes": model + overhead,
            "resumable_bytes": model + moments + overhead,
            "save_guard_anticipated_bytes": guard,
            # RNG tensors add a few KiB; the explicit overhead allowance covers
            # them here instead of presenting future serializer bytes as exact.
            "compact_guard_anticipated_bytes": math.ceil((model + overhead + 16 * MIB) * 1.05)}


def simulate(snapshot, distinct, *, extra_free=0, object_overhead=0,
             other_growth=0, compact_a1=False):
    """Follow queue order, starting with current unique objects already on disk.

    ``distinct`` names FUTURE SFT recipes whose final100/170 pointers differ.
    Existing completed results retain their observed actual pointer identities.
    The simulation continues counterfactually after first refusal to expose
    capacity needed for the entire queue; it never claims those jobs executed.
    """
    free = snapshot["disk"]["free"] + extra_free - other_growth
    reserve = snapshot["reserve_bytes"]
    events = []
    allocated = {}
    min_free = free

    def event(job, phase, guard=0, add=0, release=0):
        nonlocal free, min_free
        before = free
        free -= add
        min_free = min(min_free, free)
        events.append({"job": job["id"], "stage": job["stage"], "recipe": job["recipe"],
                       "phase": phase, "free_before_bytes": before,
                       "additional_guard_bytes": guard,
                       "guard_margin_bytes": before - reserve - guard,
                       "new_allocation_bytes": add, "released_bytes": release,
                       "free_at_physical_peak_bytes": free,
                       "free_after_bytes": free + release})
        free += release

    by_id = {j["id"]: j for j in snapshot["jobs"]}
    for job in snapshot["jobs"]:
        if job["stage"] == "A0":
            continue
        if job["stage"] == "A1":
            size = checkpoint_sizes(job["trainable_parameters"], snapshot["projector_parameters"], object_overhead)
            existing = sum(o["allocated_bytes"] for o in job["objects"])
            if job["pruned"]:
                allocated[job["id"]] = 0
                continue
            if not existing:
                event(job, "A1_save16", guard=size["save_guard_anticipated_bytes"], add=size["resumable_bytes"])
                existing = size["resumable_bytes"]
            allocated[job["id"]] = existing
            if compact_a1:
                # Alternative ONLY: a future reviewed operation, not available
                # through the current SFT-only compactor. CPU initialization of
                # SFT needs learned model tensors, not this A1 optimizer.
                event(job, "alternative_compact_completed_A1", guard=size["compact_guard_anticipated_bytes"],
                      add=size["compact_bytes"], release=existing)
                allocated[job["id"]] = size["compact_bytes"]
            continue
        if job["stage"] != "SFT":
            raise ValueError("Unknown stage")
        if not job["completed_steps"]:
            size = checkpoint_sizes(job["trainable_parameters"], snapshot["projector_parameters"], object_overhead)
            for step in job["checkpoint_steps"]:
                event(job, f"SFT_save{step}", guard=size["save_guard_anticipated_bytes"], add=size["resumable_bytes"])
            versions = 2 if job["recipe"] in distinct else 1
            if versions == 1:
                event(job, "publish_shared_best_last170_cleanup100", release=size["resumable_bytes"])
            # Best source is committed/reclaimed before touching last source.
            for i in range(versions):
                event(job, f"compact_unique_object_{i + 1}_of_{versions}",
                      guard=size["compact_guard_anticipated_bytes"],
                      add=size["compact_bytes"], release=size["resumable_bytes"])
        event(job, "post_training_evaluation", guard=64 * MIB)
        for dep in job["requires"]:
            source = by_id[dep]
            if source.get("discard_after_dependents") and allocated.get(dep, 0):
                event(job, "consumer_completed_then_prune_A1", release=allocated.pop(dep))
    failed = [e for e in events if e["guard_margin_bytes"] < 0]
    tightest = min(events, key=lambda e: e["guard_margin_bytes"])
    return {"future_distinct_best_last_recipes": sorted(distinct),
            "extra_free_bytes": extra_free, "per_new_object_overhead_allowance_bytes": object_overhead,
            "other_future_growth_allowance_bytes": other_growth,
            "alternative_compact_completed_A1": compact_a1,
            "first_refusal": failed[0] if failed else None,
            "minimum_guard_margin_bytes": tightest["guard_margin_bytes"],
            "minimum_guard_event": tightest,
            "minimum_physical_free_bytes": min_free,
            "required_additional_bytes_to_complete": max(0, -tightest["guard_margin_bytes"]),
            "counterfactual_final_free_bytes": free,
            "events": events}


def capture():
    source_paths = [SPEC, STATE, "research_results/primary_capacity.json",
                    "research_results/vision_only_derived_export.json",
                    "scripts/run_research_stage.py", "scripts/research_queue.py",
                    "scripts/research_grounding_queue.py", "scripts/compact_research_checkpoints.py",
                    "src/utils/research_storage.py", "src/utils/research_checkpoint_bundle.py",
                    "scripts/forecast_smallthinker_pilot_disk.py"]
    spec = json.loads((ROOT / SPEC).read_text())
    state_raw = (ROOT / STATE).read_bytes()
    state = json.loads(state_raw)
    capacity = json.loads((ROOT / "research_results/primary_capacity.json").read_text())
    model = next(m for m in capacity["models"] if m["id"] == "smallthinker4")
    recipe_counts = {r["recipe"]: r for r in model["recipes"]}
    projector = recipe_counts["4f"]["encoders"]["siglip2"]["frozen_projector_parameters"]
    jobs = []
    for j in spec["jobs"]:
        a = args_dict(j["argv"])
        directory = ROOT / a["output"]
        r = state["jobs"].get(j["id"], {})
        completion = directory / "completed.json"
        completed_steps = json.loads(completion.read_text())["steps"] if completion.exists() else None
        count_file = directory / "trainable_parameters.json"
        if not count_file.exists() and a["stage"] != "A0":
            count_file = ROOT / "runs/research" / f"smallthinker4__siglip2__{a['recipe']}__{a['stage']}_smoke_s42/trainable_parameters.json"
        if count_file.exists():
            counts = json.loads(count_file.read_text())
            trainable = sum(counts.values())
            source_paths.append(str(count_file.relative_to(ROOT)))
        else:
            raise ValueError(f"Missing exact count fixture {count_file}")
        if a["stage"] == "SFT" and trainable != recipe_counts[a["recipe"]]["llm_trainable_parameters"]:
            raise ValueError("Actual trainable scope differs from meta capacity proof")
        objects = []
        for p in sorted((directory / ".checkpoint_objects").glob("object-*")):
            marker = json.loads((p / ".research_checkpoint.json").read_text())
            objects.append({"object": p.name, "step": marker["step"],
                            "optimizer_state_removed": marker.get("optimizer_state_removed", False), **tree_bytes(p)})
        roles = {role: (directory / role).resolve().name for role in ("best", "last") if (directory / role).exists()}
        nrecords = sum(bool(line.strip()) for line in (ROOT / a["manifest"]).open())
        if a.get("max_records"):
            nrecords = min(nrecords, int(a["max_records"]))
        steps = math.ceil(nrecords / int(a["effective_batch_size"])) * int(a.get("epochs", 1))
        if a.get("max_steps"):
            steps = min(steps, int(a["max_steps"]))
        period = int(a["eval_every"])
        checkpoint_steps = sorted(set(range(period, steps + 1, period)) | {steps})
        if a["stage"] == "SFT" and (steps != 170 or checkpoint_steps != [100, 170]):
            raise ValueError("Forecast phase proof applies only to the current170-step recipe")
        if a["stage"] == "A1" and (steps != 16 or checkpoint_steps != [16]):
            raise ValueError("Forecast phase proof applies only to the current16-step A1 recipe")
        if a.get("checkpoint_format", "legacy") != "legacy" or a.get("optimizer_backend", "torch_adamw32") != "torch_adamw32":
            raise ValueError("Current recipe is no longer legacy FP32 AdamW")
        jobs.append({"id": j["id"], "stage": a["stage"], "recipe": a["recipe"],
                     "output": str(directory.relative_to(ROOT)), "queue_status": r.get("status"),
                     "completed_steps": completed_steps, "planned_steps": steps,
                     "checkpoint_steps": checkpoint_steps,
                     "trainable_parameters": trainable, "count_source": str(count_file.relative_to(ROOT)),
                     "roles": roles, "objects": objects,
                     "pruned": bool(r.get("checkpoint_pruned") or (directory / "checkpoint_pruned.json").exists()),
                     "discard_after_dependents": j.get("discard_checkpoint_after_dependents", False),
                     "requires": j.get("requires", [])})
    # Already completed SFTs are compact and remain untouched in every scenario.
    if any(not o["optimizer_state_removed"] for j in jobs if j["stage"] == "SFT" and j["completed_steps"] for o in j["objects"]):
        raise ValueError("New completed un-compacted SFT requires an updated snapshot model")
    donor = json.loads((ROOT / "research_results/vision_only_derived_export.json").read_text())
    originals = sum(d["original_weight_files_bytes"] for d in donor["donors"])
    donor_weight_observations = []
    for d in donor["donors"]:
        derivation_path = Path(d["path"]) / "derivation.json"
        source_paths.append(str(derivation_path.relative_to(ROOT)))
        derivation = json.loads(derivation_path.read_text())
        for name, previous in derivation["source_observations"].items():
            if not name.endswith(".safetensors"):
                continue
            p = Path(previous["resolved_path"])
            st = p.stat()
            if any(st_value != previous["stat"][key] for key, st_value in
                   [("device", st.st_dev), ("inode", st.st_ino), ("bytes", st.st_size), ("mtime_ns", st.st_mtime_ns)]):
                raise ValueError("Original donor stat identity changed since the completed bit-parity proof")
            if st.st_dev != ROOT.stat().st_dev:
                raise ValueError("Original donor is on another filesystem; cannot credit its retirement here")
            donor_weight_observations.append({"path": str(p), "bytes": st.st_size,
                                             "allocated_bytes": st.st_blocks * 512,
                                             "hardlink_count": st.st_nlink,
                                             "sha256_from_frozen_export_proof": previous["sha256"]})
    if sum(d["bytes"] for d in donor_weight_observations) != originals:
        raise ValueError("Original donor weight byte count disagrees with the export proof")
    source_hashes = {p: sha(ROOT / p) for p in sorted(set(source_paths))}
    # Record the exact state bytes parsed above, even if a heartbeat has since
    # updated its JSON. Checkpoint identities themselves must remain stable.
    source_hashes[STATE] = hashlib.sha256(state_raw).hexdigest()
    for j in jobs:
        directory = ROOT / j["output"]
        roles = {role: (directory / role).resolve().name for role in ("best", "last") if (directory / role).exists()}
        if roles != j["roles"] or any(tree_bytes(directory / ".checkpoint_objects" / o["object"])["allocated_bytes"] != o["allocated_bytes"] for o in j["objects"]):
            raise ValueError("Checkpoint snapshot changed while capturing the disk forecast; retry the read")
    return {"captured_at": datetime.now(timezone.utc).isoformat(),
            "disk": shutil.disk_usage(ROOT)._asdict(), "reserve_bytes": math.ceil(spec["reserve_gb"] * GIB),
            "projector_parameters": projector, "jobs": jobs,
            "source_sha256": source_hashes,
            "state_snapshot_sha256": hashlib.sha256(state_raw).hexdigest(),
            "derived_donors_already_in_snapshot_bytes": donor["derived_total_bytes"],
            "original_donor_weight_bytes_potentially_retirable_from_current_snapshot": originals,
            "original_donor_weight_allocated_bytes_potentially_retirable_from_current_snapshot": sum(d["allocated_bytes"] for d in donor_weight_observations),
            "original_donor_weight_observations": donor_weight_observations,
            "net_donor_saving_vs_original_only_layout_bytes": donor["potential_net_reclaim_bytes"]}


def make_report(snapshot):
    future = [j["recipe"] for j in snapshot["jobs"] if j["stage"] == "SFT" and not j["completed_steps"]]
    scenarios = {}
    for name, distinct in [("all_future_best_equals_last", set()), ("all_future_best_distinct_from_last", set(future))]:
        scenarios[name + "_raw"] = simulate(snapshot, distinct)
        scenarios[name + "_planning"] = simulate(snapshot, distinct, object_overhead=16 * MIB, other_growth=GIB)
    extra = snapshot["original_donor_weight_allocated_bytes_potentially_retirable_from_current_snapshot"]
    scenarios["distinct_planning_alternative_retire_original_donors"] = simulate(snapshot, set(future),
        extra_free=extra, object_overhead=16 * MIB, other_growth=GIB)
    scenarios["distinct_planning_alternative_compact_completed_A1"] = simulate(snapshot, set(future),
        compact_a1=True, object_overhead=16 * MIB, other_growth=GIB)
    scenarios["distinct_planning_both_alternatives"] = simulate(snapshot, set(future),
        extra_free=extra, compact_a1=True, object_overhead=16 * MIB, other_growth=GIB)
    combinations = {}
    for flags in itertools.product((False, True), repeat=len(future)):
        result = simulate(snapshot, {r for r, flag in zip(future, flags) if flag},
                          object_overhead=16 * MIB, other_growth=GIB)
        e = result["first_refusal"]
        key = f"{e['recipe']}:{e['phase']}" if e else "no_refusal"
        combinations[key] = combinations.get(key, 0) + 1
    capacities = []
    for j in snapshot["jobs"]:
        if j["stage"] in ("A1", "SFT"):
            capacities.append({"stage": j["stage"], "recipe": j["recipe"],
                               "trainable_parameters": j["trainable_parameters"],
                               **checkpoint_sizes(j["trainable_parameters"], snapshot["projector_parameters"])})
    return {"schema_version": 1, "scope": "read-only disk forecast for the current11 SmallThinker/SigLIP2 pilot trajectories",
            "snapshot": snapshot, "checkpoint_capacities": capacities, "scenarios": scenarios,
            "all_future_pointer_outcomes": {"future_recipes": future, "outcomes": 2 ** len(future),
                                           "first_refusal_counts": combinations, "probabilities_claimed": False},
            "assumptions": [
                "Existing df usage is the baseline: includes existing A0, A1, compact SFT, full donors AND derived donors; none is counted twice.",
                "Future trainable tensors and both fully initialized AdamW moments are FP32; steps/scalars/serialization/RNG are excluded from raw tensors.",
                "Planning scenarios allow16MiB per future checkpoint object and1GiB total additional non-checkpoint growth; these allowances are sensitivity inputs, not measured upper bounds.",
                "Same pinned model/encoder/data/cache: no new large downloads, datasets, vision namespaces, or embedding recomputation allocations during these pilots.",
                "SFT has precisely two writes100/170: maximum two resumable objects, never three. Distinct best/last compaction adds one model-only object to two resumable objects, then reclaims one source before compacting the next.",
                "A1 has one16-step resumable object; it remains during its entire SFT and post-training evaluations until queue status completed, then is pruned by the existing queue rule.",
                "Counterfactual timeline continues after a forecast refusal solely to determine capacity required to finish; it is not an executed experiment.",
                "Original donor retirement and completed-A1 optimizer compaction are unimplemented alternatives requiring separate review; this script makes no changes to them.",
                "Bundle best optimizer retirement does not change this170-step workload's pre-save170 guard: retirement occurs only after the new last has been published. It avoids an extra model allocation during completed-stage compaction, but that guard is not the limiting event. Lossless compression ratio has not been measured; none is credited.",
                "Future full-training GPU/RAM capacity, throughput and model quality are outside this disk-only audit. The396 primary matrix still needs its separately proved10.854TiB minimum raw compact model storage."
            ], "mutations": {"checkpoints": False, "queue": False, "source_models": False, "core_sources": False}}


def markdown(report):
    s = report["snapshot"]
    rows = ["# Current SmallThinker pilot disk forecast", "",
            f"Snapshot: {s['captured_at']}. Disk 500 GiB; free {s['disk']['free']/GIB:.3f} GiB; reserve 120 GiB. CPU-only/read-only audit.", "",
            "SFT order: " + ", ".join(j["recipe"] for j in s["jobs"] if j["stage"] == "SFT") + ".",
            "4q already retains distinct compact 100/170; 1b already retains one compact best=last 170. Their actual files are fixed in all scenarios. Precomputed 1a/1b/1c A1 files are already included in df. Final retained SFT objects: 12 if all future best=last; 21 if all future best and last differ (existing 4q/1b outcomes are fixed).", "",
            "| Future outcome | First guard refusal | Additional capacity needed to finish | Minimum physical free (counterfactual) | Final free (counterfactual) |",
            "|---|---|---:|---:|---:|"]
    for name, r in report["scenarios"].items():
        e = r["first_refusal"]
        rows.append(f"|{name}|{e['recipe']+':'+e['phase'] if e else 'none'}|{r['required_additional_bytes_to_complete']/GIB:.3f} GiB|{r['minimum_physical_free_bytes']/GIB:.3f} GiB|{r['counterfactual_final_free_bytes']/GIB:.3f} GiB|")
    rows += ["", "Planning allowance: 16 MiB per future checkpoint object plus 1 GiB total future non-checkpoint growth. Raw scenarios exclude future serializer/scalar/log overhead. Counts/tensor bytes are exact; future file sizes and best-selection outcomes are estimates.", "",
             "Two SFT saves 100/170 require at most two resumable objects. With distinct best/last the first compaction stages one model copy alongside both resumable files; its source is removed before the second compaction. A1 remains until all consumer post-training checks complete. Runner save preflight requires 1.05 × (4 × model tensor bytes + 128 MiB), so refusal can precede an actual write.", "",
             f"Both original donors and derived vision-only artifacts already exist in this snapshot. Retiring original weight blobs after a separately reviewed derived resolver could free {s['original_donor_weight_allocated_bytes_potentially_retirable_from_current_snapshot']/GIB:.6f} GiB from today's df; net saving versus the original-only pre-export layout is {s['net_donor_saving_vs_original_only_layout_bytes']/GIB:.6f} GiB. Actual deletion/reclaim: 0.", "",
             "Completed-A1 optimizer-only compaction would retain initialization model tensors and help the 4f A1 peak, but it does not remove accumulated retained SFT models. The current compactor explicitly accepts completed SFT only; no A1 change is implemented. Optional bundle best-optimizer retirement happens after publishing last and therefore does not lower the critical pre-save 170 guard here. It avoids the extra model copy during compaction; this does not change the first refusal because save 170 has the tightest guard. No compression savings are assumed.", "",
             "| Remaining SFT | Worst distinct planning save170 guard margin |",
             "|---|---:|"]
    r = report["scenarios"]["all_future_best_distinct_from_last_planning"]
    for e in r["events"]:
        if e["phase"] == "SFT_save170":
            rows.append(f"|{e['recipe']}|{e['guard_margin_bytes']/GIB:.3f} GiB|")
    rows += ["", "| Stage / recipe | Trainable parameters | Raw model + frozen projector | Raw model + FP32 Adam moments |",
             "|---|---:|---:|---:|"]
    for c in report["checkpoint_capacities"]:
        rows.append(f"|{c['stage']} {c['recipe']}|{c['trainable_parameters']:,}|{c['model_tensor_bytes']/GIB:.6f} GiB|{c['resumable_bytes']/GIB:.6f} GiB|")
    rows += ["", "All 512 future pointer assignments were enumerated without assigning probabilities: " + json.dumps(report["all_future_pointer_outcomes"]["first_refusal_counts"], sort_keys=True) + ".", "",
             "This is the 11-pilot forecast, not the 396 primary trajectories. The full primary matrix's 10.854 TiB minimum raw model-only retention remains unchanged; optimizers, staging, base weights, data and 120 GiB reserve are additional.", ""]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="research_results/smallthinker_pilot_disk_forecast")
    args = parser.parse_args()
    report = make_report(capture())
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(".md").write_text(markdown(report))
    print(json.dumps({"status": "forecast_written", "json": str(output.with_suffix('.json')),
                      "md": str(output.with_suffix('.md')), "source_mutations": False}))


if __name__ == "__main__":
    main()
