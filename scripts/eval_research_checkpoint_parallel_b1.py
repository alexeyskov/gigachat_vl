#!/usr/bin/env python3
"""Proposal: isolated disjoint native-B1 evaluation, two/four owned processes.

This is a new execution profile, never the frozen native CLI. Every generation
call remains native B1. A serial native64 cache must precede the parallel guard;
each parallel model must match all64 guard tokens before any full512 row commits.
No queue/controller/trainer is signaled. Adoption requires an actual full5124q
none comparison; this file alone is not evidence of GPU parity or speed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_research_parallel_b1 as proc

OWNER = "parallel_native_b1_evaluation_proposal_v1"
SCRIPT = "scripts/eval_research_checkpoint_parallel_b1.py"
MODULE = "src/eval/research_parallel_protocol.py"
TEST = "tests/test_research_parallel_evaluator.py"
GIB = 2 ** 30
ALLOWED_RECIPES = ("4q", "1a", "1b", "1c")
VARIANTS = ("none", "shuffle", "blank", "text_only")
PT_CAP = 8 * GIB
NVML_CAP = 12 * GIB
FREE_FLOOR = 24 * GIB


def module():
    from src.eval import research_parallel_protocol
    return research_parallel_protocol


def source_identity():
    names = (*proc.FROZEN_SOURCES, proc.SCRIPT, SCRIPT, MODULE, TEST,
             "tests/test_research_parallel_protocol.py")
    return {name: proc.file_sha(ROOT / name) for name in names}


def environment():
    versions = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "tokenizers", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "packages": versions}


def resource_profile(recipe, trainable_parameters, workers):
    if recipe not in ALLOWED_RECIPES:
        raise ValueError("Full-training profiles are resource-blocked; only SmallThinker QLoRA4q/1abc is proposed")
    if workers not in (1, 2, 4) or type(trainable_parameters) is not int or trainable_parameters <= 0:
        raise ValueError("Unsupported cohort/count metadata")
    # Explicit conservative admission estimate, not a measured1abc fit claim.
    estimate = 5 * GIB + 4 * trainable_parameters + 32 * 2 ** 20 + 3 * GIB // 2
    if estimate > PT_CAP:
        raise ValueError("Read-only metadata estimate exceeds the8GiB per-worker allocator profile")
    return {"name": "small_siglip2_qlora_b1_8gib_v1", "recipe": recipe,
            "batch_size": 1, "max_new_tokens": 64, "pytorch_cap_bytes": PT_CAP,
            "nvml_cap_bytes": NVML_CAP, "global_free_floor_bytes": FREE_FLOOR,
            "initial_gpu_free_bytes": FREE_FLOOR + workers * NVML_CAP,
            "initial_host_available_bytes": 32 * GIB + workers * 8 * GIB,
            "estimated_torch_bytes": estimate, "estimate_is_measured_fit_proof": False,
            "fit_evidence": "4q historical nativeB1 peak observed; each actual cohort remains runtime/guard-gated"}


def profiles(profile, workers):
    return [dict(profile, worker_index=index, execution_script=SCRIPT, execution_owner=OWNER,
                 cohort_workers=workers, initial_gpu_free_bytes=FREE_FLOOR+workers*NVML_CAP,
                 initial_host_available_bytes=32*GIB+workers*8*GIB,
                 native_generation_function="src.eval.research_inference.generate_prediction") for index in range(workers)]


def admit_gpu(snapshot, identities, profile, *, initial=False, confirm_owned_exit=None, lookup=None):
    if initial:
        if snapshot["processes"]:
            raise RuntimeError("An exclusive GPU is required before this owned cohort")
        if snapshot["free_bytes"] < profile["initial_gpu_free_bytes"]:
            raise RuntimeError("Initial GPU capacity does not cover the entire cohort plus24GiB reserve")
    return proc.validate_gpu(snapshot, identities, expected_uuid=profile.get("gpu_uuid"), lookup=lookup,
                             confirm_owned_exit=confirm_owned_exit)


def host_available():
    host = {k: int(v.split()[0]) * 1024 for k, v in
            (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    available = host["MemAvailable"]
    maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    if maximum != "max":
        available = min(available, max(0, int(maximum) - int(Path("/sys/fs/cgroup/memory.current").read_text())))
    return available


def output_path(run, variant, requested=None, workers=2):
    if workers not in (2,4): raise ValueError("Only explicit2/4-worker profiles are proposed")
    base = Path(run) / "evaluation/parallel_b1_v1"
    if workers == 4: base = base / "workers4"
    expected = (base / variant).resolve()
    if variant not in VARIANTS or requested is not None and Path(requested).resolve() != expected:
        raise ValueError("Output must use the separate run/evaluation/parallel_b1_v1/VARIANT namespace")
    return expected


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare_context(run, manifest, variant, output, workers, reserve_gb):
    from src.utils.research_checkpoint_compat import checkpoint_identity
    run = Path(run).resolve(strict=True)
    if run.parent != (ROOT / "runs/research").resolve():
        raise ValueError("Only owned local research runs are admitted")
    config = proc.read_json(run / "resolved_config.json")
    completion = proc.read_json(run / "completed.json")
    ownership = proc.read_json(run / ".checkpoint_manager.json")
    if (config.get("model") != "Tiiny/SmallThinker-4BA0.6B-Instruct" or
            config.get("vision") != "google/siglip2-so400m-patch14-384" or config.get("stage") != "SFT" or
            type(completion.get("steps")) is not int or completion["steps"] < 1):
        raise ValueError("Proposal is restricted to completed SmallThinker/SigLIP2 SFT")
    counts = proc.read_json(run / "trainable_parameters.json")
    if not isinstance(counts, dict) or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("Trainable-parameter resource metadata is malformed")
    profile = resource_profile(config["recipe"], sum(counts.values()), workers)
    checkpoint = (run / "best/state.pt").resolve(strict=True)
    marker = proc.read_json(checkpoint.parent / ".research_checkpoint.json")
    if (checkpoint.parent.parent != (run / ".checkpoint_objects").resolve() or
            not ownership.get("owner") or marker.get("owner") != ownership["owner"] or
            marker.get("training_completed") is not True or marker.get("resumable_optimizer") is not False):
        raise ValueError("An owned completed compact BEST artifact is required")
    identity = checkpoint_identity(checkpoint)
    if identity["total_bytes"] > 4 * sum(counts.values()) + 256 * 2 ** 20:
        raise ValueError("Checkpoint bytes exceed the declared trainable-only QLoRA admission scope")
    training = proc.read_json(run / "provenance_latest.json")
    model_provenance = {"model_ref": training["model_ref"], "vision_ref": training["vision_ref"],
                        "recipe": config["recipe"], "stage": "SFT", "training_step": marker["step"],
                        "cache_fingerprint": training["cache_fingerprint"]}
    snapshot = Path("/workspace/.hf_home/hub/models--Tiiny--SmallThinker-4BA0.6B-Instruct/snapshots") / training["model_ref"]["revision"]
    generation_path = snapshot / "generation_config.json"
    eos = proc.read_json(generation_path)["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    if any(type(i) is not int or i < 0 for i in eos):
        raise ValueError("Pinned native EOS metadata differs")
    manifest = Path(manifest).resolve(strict=True)
    original_bytes = manifest.read_bytes(); records = read_rows(manifest)
    module().validate_records(records)
    # Import the frozen transform only after CPU provenance admission. CUDA is
    # still forbidden until the child cohort allocation barrier is released.
    from src.eval.research_ablation import transform_image_ablation, write_transformed_manifest
    inputs = output.parent / "inputs" / variant
    if variant != "none":
        inputs.mkdir(parents=True, exist_ok=True)
        input_owner = {"owner": OWNER, "manifest_sha256": proc.file_sha(manifest), "variant": variant, "seed": 42}
        input_marker = inputs / "owner.json"
        if input_marker.exists() and proc.read_json(input_marker) != input_owner:
            raise ValueError("Existing intervention inputs belong to another source/variant")
        if not input_marker.exists(): proc.atomic_json(input_marker, input_owner)
    transformed, intervention = transform_image_ablation(records, variant=variant, seed=42,
        output_dir=inputs, source_manifest_sha256=proc.file_sha(manifest), reserve_bytes=math.ceil(reserve_gb * GIB))
    effective = manifest
    if variant != "none":
        effective = inputs / "transformed_manifest.jsonl"
        write_transformed_manifest(transformed, effective, reserve_bytes=math.ceil(reserve_gb * GIB))
    image_observations={};hasher=proc.frozen_image_hasher(ROOT)
    for row in transformed:
        for image,expected_hash in zip(row.get('image_paths',[]),row.get('image_hashes',[])):
            path=Path(image).resolve(strict=True)
            if str(path) not in image_observations:
                before=path.stat();actual=hasher(path);after=path.stat()
                values=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
                if actual!=expected_hash or values(before)!=values(after):
                    raise ValueError('Actual evaluation RGB pixels changed or differ from the manifest')
                image_observations[str(path)]={'canonical_rgb_sha256':actual,'stat':list(values(after))}
            elif image_observations[str(path)]['canonical_rgb_sha256']!=expected_hash:
                raise ValueError('One image path has incompatible declared RGB identities')
    sources = source_identity()
    return {"owner": OWNER, "run": str(run), "checkpoint": str(checkpoint), "checkpoint_identity": identity,
            "manifest": str(manifest), "manifest_sha256": proc.file_sha(manifest),
            "effective_manifest": str(effective), "transformed_manifest_sha256": intervention["transformed_manifest_sha256"],
            "image_ablation": intervention, "variant": variant, "records": transformed,
            "source_sha256": sources, "model_provenance": model_provenance, "profile": profile,
            "environment": environment(), "eos_token_ids": eos, "generation_config": str(generation_path),
            "generation_config_sha256": proc.file_sha(generation_path),
            "image_observations":image_observations,
            "cache_dir": str((ROOT / config["cache_dir"]).resolve()), "reserve_gb": reserve_gb,
            "metadata_sha256": {name: proc.file_sha(run / name) for name in
                                ("resolved_config.json", "completed.json", ".checkpoint_manager.json",
                                 "trainable_parameters.json", "provenance_latest.json")}}


def verify_context(context):
    from src.utils.research_checkpoint_compat import checkpoint_identity, same_checkpoint_identity
    if any(proc.file_sha(ROOT / name) != value for name, value in context["source_sha256"].items()):
        raise ValueError("Actual native/proposal execution source changed")
    if (proc.file_sha(context["manifest"]) != context["manifest_sha256"] or
            proc.file_sha(context["effective_manifest"]) != context["transformed_manifest_sha256"] or
            proc.file_sha(context["generation_config"]) != context["generation_config_sha256"]):
        raise ValueError("Manifest/intervention/native generation metadata changed")
    if not same_checkpoint_identity(context["checkpoint_identity"], checkpoint_identity(context["checkpoint"])):
        raise ValueError("Learned model checkpoint changed")
    if any(proc.file_sha(Path(context["run"]) / name) != value for name, value in context["metadata_sha256"].items()):
        raise ValueError("Run/checkpoint resource provenance changed")
    for name,observation in context['image_observations'].items():
        stat=Path(name).stat()
        if [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]!=observation['stat']:
            raise ValueError('A previously RGB-verified evaluation image changed')


def protocol_for(context, workers, *, scope="full512_parallel", reference_binding=None):
    result = module().make_protocol(context["records"], checkpoint_identity=context["checkpoint_identity"],
        manifest_sha256=context["manifest_sha256"], transformed_manifest_sha256=context["transformed_manifest_sha256"],
        source_sha256=context["source_sha256"], model_provenance=context["model_provenance"],
        worker_profiles=profiles(context["profile"], workers), variant=context["variant"], seed=42,
        environment=context["environment"], eos_token_ids=context["eos_token_ids"],
        transformed_manifest_bytes=Path(context["effective_manifest"]).read_bytes() if context["variant"] != "none" else None,
        scope=scope, reference_binding=reference_binding)
    return result


def barrier_validate(barrier, own, logical_index, nonce, *, lookup=None):
    lookup = proc.process_info if lookup is None else lookup
    if (barrier.get("owner") != OWNER or barrier.get("status") != "released" or barrier.get("nonce") != nonce or
            not 1 <= len(barrier.get("cohort", [])) <= 4 or
            not proc.same_process(barrier.get("parent"), lookup(barrier.get("parent", {}).get("pid")))):
        raise ValueError("Fresh owned parallel allocation barrier differs")
    candidates = [row for row in barrier["cohort"] if row["logical_worker"] == logical_index]
    if len(candidates) != 1 or not proc.same_process(candidates[0]["identity"], own):
        raise ValueError("Owned worker PID/birth/argv is not bound to this cohort")
    parent = barrier["parent"]["pid"]
    if own.get("ppid") != parent:
        raise ValueError("Worker parent changed")
    for row in barrier["cohort"]:
        actual = lookup(row["identity"]["pid"])
        if not proc.same_process(row["identity"], actual) or actual.get("ppid") != parent:
            raise ValueError("Peer worker PID/birth/argv changed before allocation")
    if (len({row['identity']['pid'] for row in barrier['cohort']}) != len(barrier['cohort']) or
            len({row['logical_worker'] for row in barrier['cohort']}) != len(barrier['cohort']) or
            any(type(row['logical_worker']) is not int or not 0 <= row['logical_worker'] < barrier['world_size']
                for row in barrier['cohort'])):
        raise ValueError("Duplicate or out-of-scope cohort ownership")
    if time.monotonic() >= barrier["deadline_monotonic"]:
        raise TimeoutError("Parallel evaluation cohort deadline expired")


def worker_startup_registration(args):
    """Fresh interpreter: parent-death+nonce acknowledgement before any CUDA."""
    parent = proc.process_info(args.parent_pid)
    if parent is None or parent['start_time'] != args.parent_start or os.getppid() != args.parent_pid:
        raise ValueError('Owned worker parent is stale before startup registration')
    proc.arm_parent_death(args.parent_pid)
    path = Path(args.barrier).resolve(strict=True)
    barrier = proc.read_json(path)
    if (barrier.get('owner') != OWNER or barrier.get('nonce') != args.nonce or
            barrier.get('status') != 'preparing' or
            not proc.same_process(barrier.get('parent'), proc.process_info(args.parent_pid)) or
            os.getppid() != args.parent_pid or time.monotonic() >= barrier['deadline_monotonic']):
        raise ValueError('Fresh worker initial allocation intent or parent changed')
    own = proc.process_info(os.getpid())
    if own is None or own.get('ppid') != args.parent_pid:
        raise ValueError('Fresh worker own birth/PPID unavailable')
    receipt = {'owner': OWNER, 'status': 'ready_before_allocation', 'nonce': args.nonce,
               'logical_worker': args.logical_worker, 'identity': own,
               'parent': barrier['parent'], 'initial_allocation_barrier_sha256': proc.file_sha(path),
               'protocol_sha256': barrier['protocol_sha256']}
    proc.atomic_json(path.parent / f'worker{args.logical_worker}.startup.json', receipt)
    return path


def worker_birth(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text(); fields = raw[raw.rfind(')')+2:].split()
        if fields[0] in ('Z', 'X'): return None
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]), 'start_time': int(fields[19]), 'argv': None}
    except (OSError, ValueError, IndexError): return None


def observe_starting_worker(child, command, parent, receipt, index):
    """Missing/empty argv may wait; foreign argv and birth/PPID changes cannot."""
    live = proc.process_info(child.pid)
    if live is None and child.poll() is None: live = worker_birth(child.pid)
    argv = None if live is None else live.get('argv')
    kind = 'unavailable' if argv is None else 'empty' if argv == [] else 'final' if argv == command else 'foreign'
    item = receipt['workers'][str(index)]
    observation = {'monotonic': time.monotonic(), 'argv_state': kind,
                   'start_time': None if live is None else live.get('start_time'),
                   'ppid': None if live is None else live.get('ppid')}
    if len(item['observations']) < 64: item['observations'].append(observation)
    item['last_observation'] = observation
    if live is not None:
        expected = item.get('start_time')
        if (live.get('pid') != child.pid or live.get('ppid') != parent['pid'] or
                type(live.get('start_time')) is not int or
                (expected is not None and live['start_time'] != expected)):
            raise RuntimeError('Owned inference child startup birth/PPID changed')
        if expected is None: item['start_time'] = live['start_time']
    if kind == 'foreign': raise RuntimeError('Owned inference child has foreign nonempty argv; no startup grace')
    if kind == 'final' and live.get('command_sha256') != proc.digest(command):
        raise ValueError('Owned inference child command digest differs')
    return live if kind == 'final' else None


def wait_worker_startup(children, commands, path, parent, nonce, deadline, receipt, identities=None):
    """A single bounded15s cohort startup before allocation-barrier release."""
    initial_sha = proc.file_sha(path); barrier = proc.read_json(path)
    startup_deadline = receipt['deadline_monotonic']
    identities = {} if identities is None else identities
    while time.monotonic() < startup_deadline:
        for index, child in children.items():
            if child.poll() is not None: raise RuntimeError('Owned inference child exited before startup nonce handshake')
            info = observe_starting_worker(child, commands[index], parent, receipt, index)
            ready_path = path.parent / f'worker{index}.startup.json'
            if ready_path.is_file():
                ready = proc.read_json(ready_path); item = receipt['workers'][str(index)]
                if (ready.get('owner') != OWNER or ready.get('status') != 'ready_before_allocation' or
                        ready.get('nonce') != nonce or ready.get('logical_worker') != index or
                        ready.get('parent') != parent or ready.get('protocol_sha256') != barrier['protocol_sha256'] or
                        ready.get('initial_allocation_barrier_sha256') != initial_sha or
                        ready.get('identity', {}).get('pid') != child.pid or
                        ready.get('identity', {}).get('ppid') != parent['pid'] or
                        (item.get('start_time') is not None and ready.get('identity', {}).get('start_time') != item['start_time']) or
                        ready.get('identity', {}).get('argv') != commands[index]):
                    raise ValueError('Owned inference child startup nonce/birth/command receipt differs')
                if info is not None and proc.same_process(info, ready['identity']): identities[index] = info
            if info is None: identities.pop(index, None)
        if len(identities) == len(children):
            if time.monotonic() >= startup_deadline: raise TimeoutError('Owned inference startup deadline expired at acknowledgement')
            receipt.update(status='passed', initial_allocation_barrier_sha256=initial_sha,
                           completed_monotonic=time.monotonic(),
                           startup_files_sha256={str(i):proc.file_sha(path.parent/f'worker{i}.startup.json') for i in children})
            proc.atomic_json(path.parent/'worker_startup.json', receipt)
            return identities
        proc.atomic_json(path.parent/'worker_startup.json', receipt)
        time.sleep(.025)
    raise TimeoutError('Owned inference child final argv/nonce startup handshake exceeded15s')


def stop_starting_worker(child, item, command, parent):
    """Clean up our unregistered child using bound raw birth/PPID, never argv guesses."""
    if child.poll() is not None: return True
    def owned():
        live = proc.process_info(child.pid) or worker_birth(child.pid)
        if (live is None or live.get('pid') != child.pid or live.get('ppid') != parent['pid'] or
                (item.get('start_time') is not None and live.get('start_time') != item['start_time']) or
                live.get('argv') not in (None, [], command)):
            return False
        if item.get('start_time') is None: item['start_time'] = live['start_time']
        return True
    if not owned(): return False
    child.terminate()
    try: child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if child.poll() is None and owned(): child.kill(); child.wait(timeout=5)
    return child.poll() is not None


def confirm_owned_worker_exit(pid, children, identities, *, confirmations=None, checks=None, failed_lookup=None, cohort_deadline=None):
    """At an ambiguous failed /proc lookup, wait<=1s for ORIGINAL Popen rc0.

    This is never evidence that an output is complete. A changed birth, PPID
    or nonempty argv refuses confirmation even if the old handle returned zero.
    """
    pairs=[(index,child) for index,child in children.items() if child.pid==pid]
    if len(pairs)!=1:return False
    index,child=pairs[0];expected=identities.get(index)
    if expected is None or expected.get('pid')!=pid:return False
    started=time.monotonic();deadline=min(started+1.,cohort_deadline) if cohort_deadline is not None else started+1.
    check={'pid':pid,'expected_start_time':expected['start_time'],'expected_ppid':expected['ppid'],
           'started_monotonic':started,'deadline_monotonic':deadline,'cohort_deadline_monotonic':cohort_deadline,'failed_lookup':failed_lookup,
           'samples':[],'status':'checking'}
    if checks is not None:checks.append(check)
    def compatible():
        live=proc.process_info(pid)
        raw=live if live is not None else worker_birth(pid)
        check['samples'].append({'monotonic':time.monotonic(),'lookup':live,'raw_birth':raw})
        if raw is not None and (raw.get('pid')!=pid or raw.get('start_time')!=expected.get('start_time') or
                raw.get('ppid')!=expected.get('ppid') or raw.get('argv') not in (None,[],expected.get('argv'))):
            return False
        return True
    confirmed=False
    try:
        if failed_lookup is not None:
            initial=failed_lookup.get('live')
            if (failed_lookup.get('pid')!=pid or (initial is not None and
                    (initial.get('pid')!=pid or initial.get('start_time')!=expected['start_time'] or
                     initial.get('ppid')!=expected['ppid'] or initial.get('argv') not in (None,[],expected['argv'])))):
                check['refusal_reason']='initial_failed_lookup_foreign_birth_parent_or_nonempty_argv'
                return False
        while True:
            if time.monotonic()>deadline or not compatible():return False
            status=child.poll();check['samples'][-1]['popen_returncode']=status
            if status==0:
                if not compatible():return False
                observed=time.monotonic()
                if observed>deadline:return False
                if confirmations is not None:
                    confirmations.append({'pid':pid,'start_time':expected['start_time'],'ppid':expected['ppid'],
                        'returncode':0,'observed_monotonic':observed,'wait_seconds':observed-started,
                        'reason':'ambiguous_owned_proc_confirmed_original_popen_exit'})
                check['returncode']=0;confirmed=True;return True
            if status is not None or time.monotonic()>=deadline:return False
            time.sleep(min(.01,max(0,deadline-time.monotonic())))
    finally:
        check.update(status='confirmed' if confirmed else 'refused',ended_monotonic=time.monotonic())


def worker(args):
    barrier_path = worker_startup_registration(args)
    attempt = barrier_path.parent
    while True:
        barrier = proc.read_json(barrier_path)
        if barrier.get("owner") != OWNER or barrier.get("nonce") != args.nonce:
            raise ValueError("Allocation barrier ownership changed")
        if barrier.get("status") == "released": break
        if time.monotonic() >= barrier["deadline_monotonic"]: raise TimeoutError("Allocation barrier deadline")
        time.sleep(.1)
    own = proc.process_info(os.getpid())
    barrier_validate(barrier, own, args.logical_worker, args.nonce)
    barrier_sha = proc.file_sha(barrier_path)
    context = barrier["context"]; verify_context(context)
    protocol = proc.read_json(barrier["protocol_path"])
    if module().protocol_sha256(protocol) != barrier["protocol_sha256"]:
        raise ValueError("Immutable cohort evaluation protocol changed before CUDA")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != barrier["gpu_uuid"]:
        raise ValueError("Owned worker GPU UUID visibility differs")
    identities = [row["identity"] for row in barrier["cohort"]]
    profile = dict(context["profile"], gpu_uuid=barrier["gpu_uuid"])
    admit_gpu(proc.gpu_snapshot(), identities, profile)
    import torch
    if torch.cuda.is_initialized(): raise RuntimeError("CUDA initialized before allocation barrier")
    torch.cuda.set_per_process_memory_fraction(PT_CAP / torch.cuda.get_device_properties(0).total_memory, 0)
    from src.eval.research_inference import load_research_checkpoint, generate_prediction
    from src.eval.research_metrics import score_prediction
    model, provenance = load_research_checkpoint(context["checkpoint"], cache_dir=context["cache_dir"], reserve_gb=context["reserve_gb"])
    if provenance != context["model_provenance"]:
        raise ValueError("Reloaded model provenance differs")
    runtime_eos = model.base.generation_config.eos_token_id
    runtime_eos = runtime_eos if isinstance(runtime_eos, list) else [runtime_eos]
    if runtime_eos != context["eos_token_ids"]:
        raise ValueError("Native runtime EOS configuration differs")
    output = Path(barrier["output"])
    records = context["records"]
    progress_path = attempt / f"worker{args.logical_worker}.progress.json"
    torch.cuda.synchronize()
    proc.atomic_json(progress_path, {"phase": "model_ready", "logical_worker": args.logical_worker})
    reference = barrier.get("reference_rows")
    # Each newly proved checkpoint/profile checks ALL64 in EACH independent
    # model before the parent releases full disjoint journal writes.
    if reference is not None and barrier.get("guard_required"):
        guard_rows = [];guard_started=time.monotonic()
        for index in module().select_guard_indices(records):
            if time.monotonic() >= barrier["deadline_monotonic"]: raise TimeoutError("Guard deadline")
            generated = generate_prediction(model, module().public_record(records[index]), max_new_tokens=64)
            torch.cuda.synchronize()
            proc.token_parity(records[index], generated, reference[str(index)])
            guard_rows.append({"index": index, **generated})
            proc.atomic_json(progress_path,{'phase':'native_guard64_generation','logical_worker':args.logical_worker,
                'completed_records':len(guard_rows),'total_records':64,'elapsed_seconds':time.monotonic()-guard_started,'batch_size':1})
        proc.atomic_json(attempt / f"worker{args.logical_worker}.guard.json", {"owner": OWNER,
            "status": "passed", "identity": own, "model_provenance": provenance,
            "protocol_sha256": module().protocol_sha256(protocol), "allocation_barrier_sha256": barrier_sha,
            "guard_rows": guard_rows})
        proc.atomic_json(progress_path, {"phase": "guard_completed", "logical_worker":args.logical_worker,"completed_records": 64, "total_records": 64})
    gate = attempt / "generation_barrier.json"
    while not gate.exists():
        if time.monotonic() >= barrier["deadline_monotonic"]: raise TimeoutError("Generation gate deadline")
        if not proc.same_process(barrier["parent"], proc.process_info(args.parent_pid)):
            raise ValueError("Owned parent identity changed")
        time.sleep(.1)
    generation_gate = proc.read_json(gate)
    if generation_gate.get("allocation_barrier_sha256") != barrier_sha or generation_gate.get("nonce") != args.nonce:
        raise ValueError("Generation gate binding differs")
    committed = module().read_committed(output, records, protocol)
    pending = module().pending_indices(records, protocol, committed, args.logical_worker)
    started = time.monotonic(); count = 0
    for index in pending:
        if time.monotonic() >= barrier["deadline_monotonic"]: raise TimeoutError("Native B1 generation deadline")
        if proc.file_sha(barrier_path) != barrier_sha or not proc.same_process(barrier["parent"], proc.process_info(args.parent_pid)):
            raise ValueError("Owned source/parent allocation binding changed")
        begin = time.monotonic()
        generated = generate_prediction(model, module().public_record(records[index]), max_new_tokens=64)
        torch.cuda.synchronize()
        generated["generation_seconds"] = time.monotonic() - begin
        if reference is not None and str(index) in reference:
            proc.token_parity(records[index], generated, reference[str(index)])
        module().commit_record(output, index, records[index], generated, protocol, worker_index=args.logical_worker)
        count += 1
        proc.atomic_json(progress_path, {"phase": "native_b1_generation", "logical_worker":args.logical_worker,"completed_records": count,
            "total_records": len(pending), "elapsed_seconds": time.monotonic() - started, "batch_size": 1})
    ended = time.monotonic(); verify_context(context)
    result = {"owner": OWNER, "status": "completed", "identity": own, "logical_worker": args.logical_worker,
        "model_provenance": provenance, "protocol_sha256": module().protocol_sha256(protocol),
        "allocation_barrier_sha256": barrier_sha, "generation_started_monotonic": started,
        "generation_ended_monotonic": ended, "generated_this_execution": count, "global_indices": pending,
        "journal_records_generated":count,"diagnostic_guard_records_generated":64 if barrier.get('guard_required') else 0,
        "pytorch_cap_bytes": PT_CAP, "peak_pytorch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_pytorch_reserved_bytes": torch.cuda.max_memory_reserved(), "batch_size": 1}
    proc.atomic_json(attempt / f"worker{args.logical_worker}.result.json", result)
    proc.atomic_json(progress_path, {"phase": "completed", "logical_worker":args.logical_worker,"completed_records": count, "total_records": len(pending)})


def run_cohort(context, output, protocol, active_workers, *, reference_rows=None, guard_required=False, timeout=1200):
    """Own only new inference processes. No controller/trainer/group signals."""
    if protocol['scope']=='full512_parallel' and (not guard_required or reference_rows is None):
        raise ValueError('Every newly loaded parallel model requires its own native64 guard')
    profile = dict(context["profile"], initial_gpu_free_bytes=FREE_FLOOR + len(active_workers) * NVML_CAP)
    if host_available() < 32 * GIB + len(active_workers) * 8 * GIB:
        raise RuntimeError("Insufficient host/cgroup admission headroom")
    initial = admit_gpu(proc.gpu_snapshot(), (), profile, initial=True)
    profile["gpu_uuid"] = initial["gpu_uuid"]
    attempt = output / "execution" / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    with zipfile.ZipFile(attempt / "execution_sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in context["source_sha256"]: archive.write(ROOT / name, name)
    started = time.monotonic(); cleanup_reserve = 15 + 5 * len(active_workers)
    deadline = started + timeout - cleanup_reserve
    parent = proc.process_info(os.getpid()); nonce = secrets.token_hex(32)
    initial_committed=module().read_committed(output,context['records'],protocol)
    barrier = {"owner": OWNER, "status": "preparing", "nonce": nonce, "parent": parent,
        "context": context, "cohort": [], "gpu_uuid": initial["gpu_uuid"], "deadline_monotonic": deadline,
        "output": str(output), "protocol_path": str(output / "protocol.json"),
        "reference_rows": reference_rows, "guard_required": guard_required,
        "protocol_sha256": module().protocol_sha256(protocol), "world_size": protocol['workers'],
        "timeout_seconds":timeout,"cleanup_reserve_seconds":cleanup_reserve,
        "initial_committed":sorted(initial_committed),
        "initial_committed_shard_sha256":{str(i):proc.file_sha(output/'prediction_records'/f'row-{i:04d}.json') for i in initial_committed},
        "initial_pending": {str(i):module().pending_indices(context['records'],protocol,
            initial_committed,i) for i in active_workers}}
    path = attempt / "allocation_barrier.json"; proc.atomic_json(path, barrier)
    exit_confirmations=[];exit_checks=[];latest_lookups={}
    children = {}; identities = {}; commands = {}; handles = []; samples = []; handlers = {}; result = {"owner": OWNER,
        "status": "starting", "protocol_sha256": module().protocol_sha256(protocol),
        "execution_source_sha256": context["source_sha256"], "initial_gpu": initial,
        "source_archive_sha256": proc.file_sha(attempt / "execution_sources.zip"), "started_monotonic": started,
        "owned_exit_confirmations":exit_confirmations,"owned_exit_checks":exit_checks}
    def interrupted(signum, _frame): raise InterruptedError("Owned inference cohort signal " + str(signum))
    startup_started = time.monotonic()
    startup = {'owner': OWNER, 'status': 'starting', 'nonce': nonce, 'parent': parent, 'workers': {},
               'started_monotonic': startup_started, 'deadline_monotonic': min(startup_started+15, deadline)}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM): handlers[sig] = signal.signal(sig, interrupted)
        for index in active_workers:
            if time.monotonic() >= startup['deadline_monotonic']:
                raise TimeoutError('Owned inference child cohort startup deadline expired before spawn')
            command = [sys.executable, "-u", str(ROOT / SCRIPT), "--private-worker", "--barrier", str(path),
                "--logical-worker", str(index), "--parent-pid", str(parent["pid"]),
                "--parent-start", str(parent["start_time"]), "--nonce", nonce]
            handle = (attempt / f"worker{index}.log").open("w"); handles.append(handle)
            # No preexec hook after optional CPU Torch imports. The fresh child
            # arms PDEATHSIG and verifies the parent's birth BEFORE importing
            # Torch/CUDA; death during exec/startup also fails that check.
            child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": initial["gpu_uuid"]})
            children[index] = child
            commands[index] = command
            startup['workers'][str(index)] = {'pid': child.pid, 'start_time': None, 'observations': []}
            proc.atomic_json(attempt/'worker_startup.json', startup)
            observe_starting_worker(child, command, parent, startup, index)
        wait_worker_startup(children, commands, path, parent, nonce, deadline, startup, identities)
        barrier["cohort"] = [{"logical_worker": i, "identity": identities[i]} for i in active_workers]
        barrier['worker_startup_sha256'] = proc.file_sha(attempt/'worker_startup.json')
        admit_gpu(proc.gpu_snapshot(), (), profile, initial=True); verify_context(context)
        barrier["status"] = "released"; barrier['allocation_released_monotonic'] = time.monotonic()
        proc.atomic_json(path, barrier); barrier_sha = proc.file_sha(path)
        result.update(status="running", cohort=barrier["cohort"], allocation_barrier_sha256=barrier_sha,
                      worker_startup_sha256=barrier['worker_startup_sha256'])
        proc.atomic_json(attempt / "receipt.json", result)
        released = False; last = 0.
        while True:
            if time.monotonic() >= deadline: raise TimeoutError("Owned parallel evaluation deadline expired")
            def observed_lookup(pid):
                live=proc.process_info(pid)
                latest_lookups[pid]={'monotonic':time.monotonic(),'pid':pid,'live':live}
                result['last_gpu_identity_lookup']=latest_lookups[pid]
                return live
            gpu_sample=proc.gpu_snapshot()
            try:
                snapshot = admit_gpu(gpu_sample, list(identities.values()), profile, lookup=observed_lookup,
                    confirm_owned_exit=lambda pid: confirm_owned_worker_exit(pid,children,identities,
                        confirmations=exit_confirmations,checks=exit_checks,failed_lookup=latest_lookups.get(pid),cohort_deadline=deadline))
            except BaseException:
                result['failed_gpu_snapshot']=gpu_sample
                raise
            samples.append({"monotonic": time.monotonic(), **snapshot})
            phases = []
            for index, child in children.items():
                status = child.poll()
                if status not in (None, 0): raise RuntimeError(f"Owned native B1 worker{index} failed")
                progress_path = attempt / f"worker{index}.progress.json"
                phases.append(proc.read_json(progress_path) if progress_path.exists() else {})
            expected_phase = "guard_completed" if guard_required else "model_ready"
            if not released and all(row.get("phase") == expected_phase for row in phases):
                if guard_required:
                    for index in active_workers:
                        guard = proc.read_json(attempt / f"worker{index}.guard.json")
                        if (guard.get("status") != "passed" or guard.get("protocol_sha256") != module().protocol_sha256(protocol) or
                                guard.get("model_provenance") != context["model_provenance"] or
                                not proc.same_process(identities[index], guard.get("identity"))):
                            raise ValueError("Per-model native64 guard provenance differs")
                        if [row["index"] for row in guard.get("guard_rows", [])] != module().select_guard_indices(context["records"]):
                            raise ValueError("Per-model native64 guard coverage differs")
                        for row in guard["guard_rows"]:
                            proc.token_parity(context["records"][row["index"]], row, reference_rows[str(row["index"])])
                    guard_proof={"owner": OWNER,
                        "status": "passed", "protocol_sha256": module().protocol_sha256(protocol),
                        "attempt": str(attempt), "worker_guard_files_sha256": {
                            str(i): proc.file_sha(attempt / f"worker{i}.guard.json") for i in active_workers}}
                    proc.atomic_json(attempt/'guard_passed.json',guard_proof)
                    global_guard=output/'parallel_guard_passed.json'
                    if not global_guard.exists():
                        if set(active_workers)!=set(range(protocol['workers'])):
                            raise ValueError('The initial parallel guard must cover the whole worker world')
                        proc.atomic_json(global_guard,guard_proof)
                proc.atomic_json(attempt / "generation_barrier.json", {"owner": OWNER, "status": "released", "nonce": nonce,
                    "allocation_barrier_sha256": barrier_sha, "released_monotonic": time.monotonic()})
                released = True
            if all(child.poll() == 0 for child in children.values()):
                if not released: raise RuntimeError("Owned workers exited before the generation barrier")
                observed_done = time.monotonic()
                break
            if time.monotonic() - last >= 30:
                print(json.dumps({"event": "parallel_native_b1_evaluation", "scope": protocol.get("scope"),
                    "elapsed_seconds": time.monotonic() - started, "worker_progress": phases,
                    "gpu_free_gib": snapshot["free_bytes"] / GIB}), flush=True)
                result.update(gpu_observations=samples, worker_progress=phases)
                proc.atomic_json(attempt / "receipt.json", result); last = time.monotonic()
            time.sleep(.5)
        for index in active_workers:
            row = proc.read_json(attempt / f"worker{index}.result.json")
            if (row.get("status") != "completed" or row.get("protocol_sha256") != module().protocol_sha256(protocol) or
                    row.get("allocation_barrier_sha256") != barrier_sha or row.get("model_provenance") != context["model_provenance"] or
                    not proc.same_process(identities[index], row.get("identity")) or
                    row.get("peak_pytorch_reserved_bytes", PT_CAP + 1) > PT_CAP):
                raise ValueError("Owned worker completion/provenance/resource receipt differs")
        verify_context(context)
        if time.monotonic() >= deadline: raise TimeoutError('Cohort final validation exceeded its deadline')
        result.update(status="completed", gpu_observations=samples, observed_workers_done_monotonic=observed_done,
                      ended_monotonic=time.monotonic())
        proc.atomic_json(attempt / "receipt.json", result)
        validate_cohort_attempt(attempt, context, protocol)
        return result
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error), gpu_observations=samples)
        proc.atomic_json(attempt / "receipt.json", result)
        raise
    finally:
        failures = []
        for index, child in children.items():
            try:
                stopped = (proc.stop_owned(child, identities[index]) if index in identities else
                           stop_starting_worker(child, startup['workers'][str(index)], commands[index], parent))
                if not stopped and child.poll() is None: failures.append('Unverified child ownership; no signal sent: '+str(child.pid))
            except Exception as error: failures.append(str(error))
        for handle in handles: handle.close()
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        if failures:
            result.update(status="failed", cleanup_errors=failures); proc.atomic_json(attempt / "receipt.json", result)
            raise RuntimeError("Owned inference cleanup failed")


def validate_startup_proof(attempt, barrier, receipt):
    path = Path(attempt)/'worker_startup.json'
    report = proc.read_json(path); digest = proc.file_sha(path)
    cohort = {str(member['logical_worker']):member['identity'] for member in barrier['cohort']}
    initial = json.loads(json.dumps(barrier)); initial.update(status='preparing', cohort=[])
    initial.pop('worker_startup_sha256', None)
    initial.pop('allocation_released_monotonic', None)
    initial_sha = hashlib.sha256((json.dumps(initial,sort_keys=True,indent=2)+'\n').encode()).hexdigest()
    if (receipt.get('worker_startup_sha256') != digest or barrier.get('worker_startup_sha256') != digest or
            report.get('owner') != OWNER or report.get('status') != 'passed' or
            report.get('nonce') != barrier['nonce'] or report.get('parent') != barrier['parent'] or
            set(report.get('workers', {})) != set(cohort) or
            set(report.get('startup_files_sha256', {})) != set(cohort) or
            report.get('initial_allocation_barrier_sha256') != initial_sha):
        raise ValueError('Raw startup cohort/nonce/initial allocation proof differs')
    started = report.get('started_monotonic'); deadline = report.get('deadline_monotonic'); completed = report.get('completed_monotonic')
    if (any(type(value) not in (int,float) or not math.isfinite(value) for value in (started,deadline,completed)) or
            not receipt['started_monotonic'] <= started <= completed < deadline or
            not math.isclose(deadline,min(started+15,barrier['deadline_monotonic']),abs_tol=1e-6)):
        raise ValueError('Raw startup whole-cohort15s timing differs')
    allocation = barrier.get('allocation_released_monotonic')
    if type(allocation) not in (int,float) or not math.isfinite(allocation) or not completed <= allocation < barrier['deadline_monotonic']:
        raise ValueError('Raw startup did not precede allocation release')
    for index, identity in cohort.items():
        item = report['workers'][index]; ready_path = Path(attempt)/f'worker{index}.startup.json'
        ready = proc.read_json(ready_path)
        if (proc.file_sha(ready_path) != report['startup_files_sha256'][index] or
                item.get('pid') != identity['pid'] or item.get('start_time') != identity['start_time'] or
                ready.get('owner') != OWNER or ready.get('status') != 'ready_before_allocation' or
                ready.get('nonce') != barrier['nonce'] or ready.get('logical_worker') != int(index) or
                not proc.same_process(ready.get('identity'), identity) or
                ready.get('identity', {}).get('ppid') != barrier['parent']['pid'] or
                identity.get('ppid') != barrier['parent']['pid'] or
                ready.get('parent') != barrier['parent'] or ready.get('protocol_sha256') != barrier['protocol_sha256'] or
                ready.get('initial_allocation_barrier_sha256') != initial_sha):
            raise ValueError('Raw per-worker startup nonce/birth/PPID/final argv proof differs')
        observations = item.get('observations', [])
        last = item.get('last_observation', {})
        if (not observations or len(observations) > 64 or last.get('argv_state') != 'final' or
                last.get('start_time') != identity['start_time'] or last.get('ppid') != identity['ppid'] or
                any(row.get('argv_state') not in ('unavailable','empty','final') or
                    row.get('start_time') not in (None,identity['start_time']) or
                    row.get('ppid') not in (None,identity['ppid']) for row in observations)):
            raise ValueError('Raw startup observations contain ambiguous/foreign identity')
    return report


def validate_cohort_attempt(attempt, context, protocol):
    """CPU-only raw process/resource/source/timing validation, never generation."""
    attempt=Path(attempt).resolve(strict=True)
    receipt=proc.read_json(attempt/'receipt.json')
    barrier_path=attempt/'allocation_barrier.json';barrier=proc.read_json(barrier_path)
    gate=proc.read_json(attempt/'generation_barrier.json');barrier_sha=proc.file_sha(barrier_path)
    if (receipt.get('owner')!=OWNER or receipt.get('status')!='completed' or
            receipt.get('protocol_sha256')!=module().protocol_sha256(protocol) or
            receipt.get('execution_source_sha256')!=context['source_sha256'] or
            receipt.get('cohort')!=barrier.get('cohort') or barrier.get('context')!=context or
            barrier.get('protocol_sha256')!=module().protocol_sha256(protocol) or barrier.get('owner')!=OWNER or
            barrier.get('status')!='released' or receipt.get('allocation_barrier_sha256')!=barrier_sha or
            gate.get('owner')!=OWNER or gate.get('status')!='released' or gate.get('nonce')!=barrier.get('nonce') or
            gate.get('allocation_barrier_sha256')!=barrier_sha):
        raise ValueError('Raw cohort protocol/owner/source/barrier receipt differs')
    startup = validate_startup_proof(attempt, barrier, receipt)
    if barrier['allocation_released_monotonic'] > gate.get('released_monotonic', -math.inf):
        raise ValueError('Startup proof did not precede generation/allocation work')
    if Path(barrier['output']).resolve()/'execution'!=attempt.parent:
        raise ValueError('Raw cohort escaped its owned output')
    archive=attempt/'execution_sources.zip'
    if proc.file_sha(archive)!=receipt.get('source_archive_sha256'):
        raise ValueError('Executed source archive SHA differs')
    with zipfile.ZipFile(archive) as source_zip:
        if len(source_zip.namelist())!=len(context['source_sha256']) or set(source_zip.namelist())!=set(context['source_sha256']):
            raise ValueError('Executed source archive members differ or duplicate names')
        import hashlib
        if any(hashlib.sha256(source_zip.read(name)).hexdigest()!=value for name,value in context['source_sha256'].items()):
            raise ValueError('Executed source archive contents differ')
    identities={row['identity']['pid']:row['identity'] for row in barrier['cohort']}
    if len(identities)!=len(barrier['cohort']):raise ValueError('Raw cohort PID ownership is duplicated')
    for confirmation in receipt.get('owned_exit_confirmations',[]):
        expected=identities.get(confirmation.get('pid'),{})
        elapsed=confirmation.get('wait_seconds');observed=confirmation.get('observed_monotonic')
        if (not expected or confirmation.get('start_time')!=expected['start_time'] or
                confirmation.get('ppid')!=expected['ppid'] or confirmation.get('returncode')!=0 or
                confirmation.get('reason')!='ambiguous_owned_proc_confirmed_original_popen_exit' or
                type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not 0<=elapsed<=1.1 or
                type(observed) not in (int,float) or not math.isfinite(observed) or
                not receipt['started_monotonic']<=observed<=receipt['ended_monotonic']):
            raise ValueError('Raw owned-exit confirmation lacks bounded original-Popen ownership proof')
    for check in receipt.get('owned_exit_checks',[]):
        expected=identities.get(check.get('pid'),{})
        if (not expected or check.get('expected_start_time')!=expected['start_time'] or
                check.get('expected_ppid')!=expected['ppid'] or check.get('status')!='confirmed' or check.get('returncode')!=0 or
                not check.get('samples') or check.get('cohort_deadline_monotonic')!=barrier['deadline_monotonic'] or
                not math.isclose(check['deadline_monotonic'],min(check['started_monotonic']+1.,barrier['deadline_monotonic']),abs_tol=1e-6) or
                not receipt['started_monotonic']<=check['started_monotonic']<=check['ended_monotonic']<=receipt['ended_monotonic'] or
                check['ended_monotonic']-check['started_monotonic']>1.1):
            raise ValueError('Raw bounded exit-check timing/ownership/returncode differs')
        failed=check.get('failed_lookup',{})
        ambiguous=failed.get('live')
        if (failed.get('pid')!=check['pid'] or type(failed.get('monotonic')) not in (int,float) or
                not math.isfinite(failed['monotonic']) or
                not receipt['started_monotonic']<=failed['monotonic']<=check['started_monotonic'] or
                (ambiguous is not None and
                (ambiguous.get('pid')!=check['pid'] or ambiguous.get('start_time')!=expected['start_time'] or ambiguous.get('ppid')!=expected['ppid'] or
                 ambiguous.get('argv') not in (None,[],expected['argv']) or
                 (ambiguous.get('state') not in ('Z','X') and ambiguous.get('argv'))))):
            raise ValueError('Raw exit check did not start from an ambiguous owned identity lookup')
        if not any(sample.get('popen_returncode')==0 for sample in check['samples']):
            raise ValueError('Raw exit check has no original Popen zero status')
        for sample in check['samples']:
            raw=sample.get('lookup') if sample.get('lookup') is not None else sample.get('raw_birth')
            if (type(sample.get('monotonic')) not in (int,float) or not math.isfinite(sample['monotonic']) or
                    not check['started_monotonic']<=sample['monotonic']<=check['ended_monotonic'] or
                    (raw is not None and (raw.get('pid')!=check['pid'] or raw.get('start_time')!=expected['start_time'] or
                        raw.get('ppid')!=expected['ppid'] or raw.get('argv') not in (None,[],expected['argv'])))):
                raise ValueError('Raw exit-check retry has a foreign/reused identity or invalid time')
    limit=barrier.get('timeout_seconds');reserve=barrier.get('cleanup_reserve_seconds')
    if (type(limit) is not int or not 120<=limit<=1800 or reserve!=15+5*len(identities) or
            type(receipt.get('started_monotonic')) not in (int,float) or
            not math.isclose(barrier['deadline_monotonic']-receipt['started_monotonic'],limit-reserve,abs_tol=1e-6)):
        raise ValueError('Raw cohort deadline is not bound to its declared timeout/cleanup reserve')
    selected=set(protocol['selected_global_indices']);prior=barrier.get('initial_committed')
    if (not isinstance(prior,list) or len(prior)!=len(set(prior)) or
            any(type(i) is not int or i not in selected for i in prior)):
        raise ValueError('Initial committed index set is malformed')
    if set(barrier.get('initial_committed_shard_sha256',{}))!={str(i) for i in prior}:
        raise ValueError('Initial committed shard binding differs')
    output=Path(barrier['output'])
    for i in prior:
        if proc.file_sha(output/'prediction_records'/f'row-{i:04d}.json')!=barrier['initial_committed_shard_sha256'][str(i)]:
            raise ValueError('An already committed row changed during the resumed cohort')
    pending=barrier.get('initial_pending',{})
    logical={member['logical_worker'] for member in barrier['cohort']}
    if len(logical)!=len(identities) or set(pending)!={str(i) for i in logical}:
        raise ValueError('Initial pending logical workers differ')
    current=module().read_committed(output,context['records'],protocol)
    for index in logical:
        expected=[i for i in protocol['selected_global_indices'] if i not in prior and
                  (i%protocol['workers'] if protocol['scope']=='full512_parallel' else 0)==index]
        if not expected or pending[str(index)]!=expected or any(i not in current for i in expected):
            raise ValueError('Pending indices violate unique disjoint assignment or current journal coverage')
    if set(prior)|{i for values in pending.values() for i in values}!=selected:
        raise ValueError('The admitted pending cohort does not cover the remaining selected records')
    if protocol['scope']=='full512_parallel':
        validate_attempt_guard(attempt,context,protocol,barrier['reference_rows'])
    profile=dict(context['profile'],gpu_uuid=barrier['gpu_uuid'],
                 initial_gpu_free_bytes=FREE_FLOOR+len(identities)*NVML_CAP)
    admit_gpu(receipt['initial_gpu'],[],profile,initial=True)
    samples=receipt.get('gpu_observations',[])
    if not samples:raise ValueError('Raw cohort lacks GPU memory/isolation observations')
    observed_pids=set()
    for sample in samples:
        proc.validate_gpu(sample,list(identities.values()),lookup=identities.get,expected_uuid=barrier['gpu_uuid'])
        observed_pids.update(row['pid'] for row in sample['processes'])
    if observed_pids!=set(identities):raise ValueError('Not every owned CUDA process was observed')
    observed=receipt.get('observed_workers_done_monotonic');released=gate.get('released_monotonic')
    ended=receipt.get('ended_monotonic');deadline=barrier['deadline_monotonic']
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (observed,released,ended)) or not released<observed<=ended<deadline:
        raise ValueError('Parent-observed cohort completion/release/deadline differs')
    for member in barrier['cohort']:
        index=member['logical_worker'];identity=member['identity']
        row=proc.read_json(attempt/f'worker{index}.result.json')
        start=row.get('generation_started_monotonic');end=row.get('generation_ended_monotonic')
        if (row.get('owner')!=OWNER or row.get('status')!='completed' or row.get('logical_worker')!=index or
                not proc.same_process(identity,row.get('identity')) or identity.get('ppid')!=barrier['parent']['pid'] or
                row.get('model_provenance')!=context['model_provenance'] or row.get('batch_size')!=1 or
                row.get('protocol_sha256')!=module().protocol_sha256(protocol) or
                row.get('allocation_barrier_sha256')!=barrier_sha or
                row.get('global_indices')!=barrier['initial_pending'][str(index)] or
                row.get('generated_this_execution')!=len(row['global_indices']) or
                row.get('journal_records_generated')!=len(row['global_indices']) or
                row.get('diagnostic_guard_records_generated')!=(64 if barrier.get('guard_required') else 0) or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end)) or not released<=start<=end<=observed or
                any(type(row.get(key)) is not int or not 0<=row[key]<=PT_CAP for key in ('peak_pytorch_allocated_bytes','peak_pytorch_reserved_bytes')) or
                row.get('pytorch_cap_bytes')!=PT_CAP):
            raise ValueError('Raw worker token-execution/resource/provenance/timing receipt differs')
    return receipt


def validate_attempt_guard(attempt, context, protocol, reference_rows):
    attempt=Path(attempt).resolve(strict=True)
    proof=proc.read_json(attempt/'guard_passed.json')
    if proof.get("owner") != OWNER or proof.get("status") != "passed" or proof.get("protocol_sha256") != module().protocol_sha256(protocol):
        raise ValueError("Persisted parallel native64 guard scope/source changed")
    if Path(proof['attempt']).resolve()!=attempt:raise ValueError('Guard attempt binding differs')
    barrier_path = attempt / "allocation_barrier.json"
    barrier = proc.read_json(barrier_path); barrier_sha = proc.file_sha(barrier_path)
    if (barrier.get("owner") != OWNER or barrier.get("context") != context or
            barrier.get("protocol_sha256") != module().protocol_sha256(protocol) or
            barrier.get("reference_rows") != reference_rows or barrier.get("guard_required") is not True):
        raise ValueError("Guard allocation/provenance/reference binding differs")
    expected_workers = {row['logical_worker'] for row in barrier['cohort']}
    if set(map(int, proof.get("worker_guard_files_sha256", {}))) != expected_workers:
        raise ValueError("Parallel guard did not cover every independent model")
    cohort = {row["logical_worker"]: row["identity"] for row in barrier["cohort"]}
    if set(cohort) != expected_workers: raise ValueError("Guard cohort coverage differs")
    for index in sorted(expected_workers):
        guard_path = attempt / f"worker{index}.guard.json"
        if proc.file_sha(guard_path) != proof["worker_guard_files_sha256"][str(index)]:
            raise ValueError("Saved per-model guard receipt changed")
        row = proc.read_json(guard_path)
        if (row.get("owner") != OWNER or row.get("status") != "passed" or
                row.get("protocol_sha256") != module().protocol_sha256(protocol) or
                row.get("allocation_barrier_sha256") != barrier_sha or
                row.get("model_provenance") != context["model_provenance"] or
                not proc.same_process(cohort[index], row.get("identity")) or
                [r["index"] for r in row.get("guard_rows", [])] != module().select_guard_indices(context["records"])):
            raise ValueError("Saved per-model guard provenance/coverage differs")
        for generated in row["guard_rows"]:
            proc.token_parity(context["records"][generated["index"]], generated, reference_rows[str(generated["index"])])
    return proof


def validated_parallel_guard(output, context, protocol, reference_rows):
    path=output/'parallel_guard_passed.json'
    if not path.exists():return None
    proof=proc.read_json(path);attempt=Path(proof['attempt']).resolve(strict=True)
    if attempt.parent!=(output/'execution').resolve():raise ValueError('Initial guard escaped its owned output')
    verified=validate_attempt_guard(attempt,context,protocol,reference_rows)
    if proof!=verified or set(map(int,proof['worker_guard_files_sha256']))!=set(range(protocol['workers'])):
        raise ValueError('The initial whole-world guard changed or is incomplete')
    return proof


def reference_cache(context, output, workers, *, supplied=None, allow_gpu=False, timeout=1200):
    preliminary = protocol_for(context, workers)
    old_default = Path(context["run"]) / "evaluation" / ("heldout_v3" if context["variant"] == "none" else "grounding_v1/" + context["variant"])
    external = Path(supplied).resolve(strict=True) if supplied is not None else old_default if (old_default / "summary.json").exists() else None
    if external is not None:
        source_protocol = proc.read_json(external / "protocol.json")
        scope = "guard64" if source_protocol.get("scope") == "guard64_serial_reference" else "full512"
        return module().validate_native_cache(external, context["records"], preliminary, scope=scope)
    guard_output = output.parent / "native_guard" / context["variant"]
    serial_protocol = protocol_for(context, 1, scope="guard64_serial_reference")
    module().init_journal(guard_output, serial_protocol, resume=(guard_output / "protocol.json").exists())
    committed = module().read_committed(guard_output, context["records"], serial_protocol)
    if len(committed) != 64:
        if not allow_gpu:
            raise RuntimeError("A previous native64 cache is required; explicit GPU admission builds it before any parallel phase")
        serial_context = dict(context, profile=resource_profile(context["profile"]["recipe"],
            sum(proc.read_json(Path(context["run"]) / "trainable_parameters.json").values()), 1))
        run_cohort(serial_context, guard_output, serial_protocol, [0], timeout=timeout)
    module().merge_completed(guard_output, context["records"], serial_protocol)
    return module().validate_native_cache(guard_output, context["records"], preliminary, scope="guard64")


def evaluate(args):
    run = Path(args.run).resolve(strict=True)
    output = output_path(run, args.image_ablation, args.output, args.workers)
    if output.exists() and not args.resume:
        raise ValueError("Existing parallel artifacts require explicit --resume; frozen native outputs are preserved")
    if not args.preflight_only and not args.allow_gpu:
        raise ValueError("Actual proposal evaluation requires explicit --allow-gpu")
    if args.preflight_only and args.image_ablation != 'none':
        raise ValueError("Read-only preflight supports none; interventions are materialized only during an explicit evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = (output.parent / ("." + args.image_ablation + ".parallel.lock")).open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        context = prepare_context(run, args.manifest, args.image_ablation, output, args.workers, args.reserve_gb)
        if args.preflight_only:
            preliminary = protocol_for(context, args.workers)
            reference = reference_cache(context, output, args.workers, supplied=args.native_cache, allow_gpu=False)
            result = {"status": "cpu_preflight_passed", "resource_profile": context["profile"],
                      "checkpoint_sha256": context["checkpoint_identity"]["sha256"],
                      "reference_scope": reference["scope"], "records": 512,
                      "protocol_source_sha256": preliminary["source_sha256"], "actual_gpu_validated": False}
            print(json.dumps(result), flush=True); return result
        from src.utils.research_storage import DiskBudget
        DiskBudget(ROOT, args.reserve_gb).ensure_space(2 * GIB)
        reference = reference_cache(context, output, args.workers, supplied=args.native_cache,
                                    allow_gpu=args.allow_gpu, timeout=args.timeout_seconds)
        protocol = protocol_for(context, args.workers, reference_binding=reference["binding"])
        module().init_journal(output, protocol, resume=args.resume)
        guard_indices = module().select_guard_indices(context["records"])
        reference_by_id = {row["record_id"]: row for row in reference["rows"]}
        reference_rows = {str(i): reference_by_id[context["records"][i]["record_id"]] for i in guard_indices}
        committed = module().read_committed(output, context["records"], protocol)
        guard = validated_parallel_guard(output, context, protocol, reference_rows)
        if committed and guard is None:
            raise ValueError("Committed full-evaluation rows have no preceding verified parallel guard")
        active = [i for i in range(args.workers) if module().pending_indices(context["records"], protocol, committed, i)]
        execution = None
        if active:
            execution = run_cohort(context, output, protocol, active, reference_rows=reference_rows,
                guard_required=True, timeout=args.timeout_seconds)
        guard = validated_parallel_guard(output, context, protocol, reference_rows)
        if guard is None: raise ValueError("No completed per-checkpoint parallel native64 guard")
        verify_context(context)
        full_reference = reference if reference["scope"] == "full512" else None
        _, summary = module().merge_completed(output, context["records"], protocol, reference=full_reference)
        result = {"owner": OWNER, "status": "completed", "protocol_sha256": module().protocol_sha256(protocol),
            "summary_sha256": proc.file_sha(output / "summary.json"), "records": 512,
            "worker_count": args.workers, "reference_binding": reference["binding"],
            "per_model_parallel_native64_guard_verified": True,
            "full512_previous_native_cache_comparison": full_reference is not None,
            "guard_proof_sha256": proc.file_sha(output / "parallel_guard_passed.json"),
            "execution": execution, "production_controller_changed": False,
            "adoption_status": "isolated proposal; no controller integration before actual first4qnone full512 proof"}
        result_path = output / "execution_summary.json"
        if result_path.exists():
            # The immutable prediction summary is unchanged; new resume attempts
            # receive their own execution receipts instead of overwriting history.
            result_path = output / f"execution_summary_resume_{uuid.uuid4().hex}.json"
        proc.atomic_json(result_path, result)
        return result
    finally:
        lock.close()


def validate_completed_output(run, manifest, variant='none', workers=2, native_cache=None):
    """Read-only complete-output verifier used by an independent boundary owner.

    No GPU query/model construction/process signal is performed. Existing
    merged views must be present before journal recomputation is allowed.
    """
    output=output_path(run,variant,workers=workers)
    for name in ('protocol.json','predictions.jsonl','summary.json','execution_summary.json','parallel_guard_passed.json'):
        if not (output/name).is_file() or (output/name).is_symlink():
            raise ValueError('Missing completed parallel output: '+name)
    if variant!='none' and not (output.parent/'inputs'/variant/'owner.json').is_file():
        raise ValueError('Completed intervention inputs are missing')
    context=prepare_context(run,manifest,variant,output,workers,120)
    disk_protocol=proc.read_json(output/'protocol.json')
    declared=disk_protocol.get('reference_binding',{})
    directory=Path(declared.get('directory','')).resolve(strict=True)
    if native_cache is not None and Path(native_cache).resolve(strict=True)!=directory:
        raise ValueError('Requested completed reference cache differs')
    for name in ('protocol.json','summary.json','predictions.jsonl'):
        if not (directory/name).is_file():raise ValueError('Previous native reference is incomplete')
    reference=module().validate_native_cache(directory,context['records'],protocol_for(context,workers),scope=declared.get('scope'))
    protocol=protocol_for(context,workers,reference_binding=reference['binding'])
    if disk_protocol!=protocol:raise ValueError('Completed parallel protocol/source/profile differs')
    lookup={row['record_id']:row for row in reference['rows']}
    reference_rows={str(i):lookup[context['records'][i]['record_id']] for i in module().select_guard_indices(context['records'])}
    guard=validated_parallel_guard(output,context,protocol,reference_rows)
    if guard is None:raise ValueError('Completed output has no parallel native64 guard')
    rows,summary=module().merge_completed(output,context['records'],protocol,
        reference=reference if reference['scope']=='full512' else None)
    execution=proc.read_json(output/'execution_summary.json')
    if (execution.get('owner')!=OWNER or execution.get('status')!='completed' or execution.get('records')!=512 or
            execution.get('worker_count')!=workers or execution.get('protocol_sha256')!=module().protocol_sha256(protocol) or
            execution.get('summary_sha256')!=proc.file_sha(output/'summary.json') or
            execution.get('guard_proof_sha256')!=proc.file_sha(output/'parallel_guard_passed.json') or
            execution.get('reference_binding')!=reference['binding'] or
            execution.get('per_model_parallel_native64_guard_verified') is not True or
            execution.get('full512_previous_native_cache_comparison')!=(reference['scope']=='full512')):
        raise ValueError('Completed execution/summary/guard/reference receipt differs')
    attempts=[];covered=set()
    for attempt in sorted((output/'execution').iterdir()):
        receipt_path=attempt/'receipt.json'
        if not receipt_path.is_file():raise ValueError('Unowned/incomplete execution attempt is not silently ignored')
        receipt=proc.read_json(receipt_path)
        if receipt.get('status')=='completed':
            validated=validate_cohort_attempt(attempt,context,protocol)
            for member in validated['cohort']:
                row=proc.read_json(attempt/f"worker{member['logical_worker']}.result.json")
                covered.update(row['global_indices'])
            attempts.append({'attempt':str(attempt),'status':'completed','receipt_sha256':proc.file_sha(receipt_path)})
        elif receipt.get('status')=='failed':
            # Interrupted committed records remain immutable. A full old cache
            # can independently validate every such token; otherwise resource
            # provenance is incomplete and this read-only verifier will refuse.
            attempts.append({'attempt':str(attempt),'status':'failed','receipt_sha256':proc.file_sha(receipt_path)})
        else:raise ValueError('An inference attempt is still running or unverified')
    if covered!=set(range(512)) and reference['scope']!='full512':
        raise ValueError('Not all resumed rows have completed owned execution receipts or a full native token reference')
    verify_context(context)
    return {'owner':OWNER,'status':'passed','records':len(rows),'worker_count':workers,
        'checkpoint_identity':context['checkpoint_identity'],'protocol_sha256':module().protocol_sha256(protocol),
        'raw_predictions_sha256':proc.file_sha(output/'predictions.jsonl'),
        'summary_sha256':proc.file_sha(output/'summary.json'),'group_macro_score':summary['group_macro_score'],
        'full512_exact_previous_native_comparison':reference['scope']=='full512',
        'per_model_native64_guard_verified':True,'execution_attempts':attempts,
        'actual_scope':'SmallThinker/SigLIP2 proposed QLoRA nativeB1 profile; no other recipes or campaign-speed claim'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run"); parser.add_argument("--manifest", default=str(ROOT / proc.MANIFEST))
    parser.add_argument("--image-ablation", choices=VARIANTS, default="none")
    parser.add_argument("--workers", choices=(2, 4), type=int, default=2)
    parser.add_argument("--output"); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--native-cache"); parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--reserve-gb", type=float, default=120); parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--private-worker", action="store_true", help=argparse.SUPPRESS)
    for name, dtype in (("barrier", str), ("logical-worker", int), ("parent-pid", int), ("parent-start", int), ("nonce", str)):
        parser.add_argument("--" + name, type=dtype, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.private_worker:
        if any(getattr(args, key) is None for key in ("barrier", "logical_worker", "parent_pid", "parent_start", "nonce")):
            parser.error("Private workers require all parent/barrier fields")
        return worker(args)
    if not args.run or args.reserve_gb < 120 or not 120 <= args.timeout_seconds <= 1800:
        parser.error("Run, reserve>=120GiB and bounded120..1800s timeout are required")
    return evaluate(args)


if __name__ == "__main__":
    main()
