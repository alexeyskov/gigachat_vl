#!/usr/bin/env python3
"""Run a finite research job queue sequentially, with explicit failures and resume.

Spec example::

  {"queue_id":"smallthinker-smoke", "reserve_gb":120,
   "stop_on_failure":true,
   "jobs":[{"id":"a0-bf16", "requires":[],
            "argv":["--stage","A0","--recipe","4f",
                    "--manifest","datasets/research/captions.train.jsonl",
                    "--validation","datasets/research/captions.validation.jsonl",
                    "--output","runs/research/smoke-a0-bf16"]}]}

``argv`` contains run_research_stage.py arguments, never a shell command. Every
job writes its own best/last checkpoint and completed.json. Re-running this queue
resumes an interrupted job if last/state.pt exists; a failed job is not retried
unless --retry-failed is explicitly supplied. A successful smoke is a pipeline
check, not completion of the scientific experiment matrix.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.research_storage import DiskBudget, BestLastCheckpointManager, _atomic_json, _tree_size


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **fields):
    print(json.dumps({"time": timestamp(), "event": event, **fields}, ensure_ascii=False), flush=True)


def argv_value(argv, flag):
    values = []
    for i, argument in enumerate(argv):
        if argument == flag:
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise ValueError(f"Missing argument for {flag}")
            values.append(argv[i + 1])
        elif argument.startswith(flag + "="):
            values.append(argument[len(flag) + 1:])
    if len(values) > 1:
        raise ValueError(f"Duplicate {flag} arguments")
    return values[0] if values else None


def output_dir(argv):
    value = argv_value(argv, "--output")
    if not value:
        raise ValueError("Every queue job needs --output")
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def resolve_path(value):
    path = Path(value).expanduser()
    return (ROOT/path).resolve() if not path.is_absolute() else path.resolve()


def validate_spec(spec):
    if not isinstance(spec, dict) or not isinstance(spec.get("jobs"), list) or not spec["jobs"]:
        raise ValueError("Spec requires a nonempty jobs list")
    seen = set()
    outputs = set()
    evaluation_outputs = set()
    for job in spec["jobs"]:
        if not isinstance(job, dict) or not isinstance(job.get("id"), str) or not job["id"]:
            raise ValueError("Each job needs a nonempty string id")
        if not all(c.isalnum() or c in "-_." for c in job["id"]):
            raise ValueError("Job IDs may contain letters, digits, dots, underscores and dashes")
        if job["id"] in seen:
            raise ValueError(f"Duplicate job id: {job['id']}")
        argv = job.get("argv")
        if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            raise ValueError(f"Job {job['id']} argv must be a list of strings")
        if not argv or not argv[0].startswith("--"):
            raise ValueError("argv must contain trainer arguments, not an executable")
        dependencies = job.get("requires", [])
        if not isinstance(dependencies, list) or not all(isinstance(d, str) for d in dependencies):
            raise ValueError("requires must be a list of job IDs")
        if any(dep not in seen for dep in dependencies):
            raise ValueError(f"Job {job['id']} references a missing or later dependency")
        if job.get("compact_completed", False) and argv_value(argv, "--stage") != "SFT":
            raise ValueError("Only completed SFT checkpoints may compact optimizer state")
        if job.get("discard_checkpoint_after_dependents", False) and argv_value(argv, "--stage") not in {"A0", "A1"}:
            raise ValueError("Only intermediate A0/A1 checkpoints may be discarded; final SFT checkpoints are retained")
        directory = output_dir(argv)
        evaluation = job.get("evaluation")
        if evaluation is not None:
            if not isinstance(evaluation,dict) or not all(isinstance(evaluation.get(key),str) and evaluation[key] for key in ("manifest","output")):
                raise ValueError("evaluation requires nonempty manifest/output paths")
            if evaluation.get("checkpoint","best") not in {"best","last"}:
                raise ValueError("evaluation checkpoint must be best or last")
            tokens = evaluation.get("max_new_tokens",64)
            if isinstance(tokens,bool) or not isinstance(tokens,int) or tokens<=0:
                raise ValueError("evaluation max_new_tokens must be a positive integer")
            destination = resolve_path(evaluation["output"])
            if destination==directory or any(destination.is_relative_to(directory/name) for name in ("best","last",".checkpoint_objects")):
                raise ValueError("evaluation output must not overwrite checkpoint/training paths")
            if destination in evaluation_outputs:
                raise ValueError("Evaluation output reused across jobs")
            evaluation_outputs.add(destination)
        if directory in outputs:
            raise ValueError(f"Output reused across jobs: {directory}")
        outputs.add(directory)
        seen.add(job["id"])
    return spec


def artifacts_for(job, allow_pruned=False):
    directory = output_dir(job["argv"])
    completion = directory / "completed.json"
    last = directory / "last/state.pt"
    receipt = directory / "checkpoint_pruned.json"
    pruned = False
    if allow_pruned and receipt.is_file():
        pruning = json.loads(receipt.read_text())
        owner = json.loads((directory / ".checkpoint_manager.json").read_text())["owner"]
        if pruning.get("owner") != owner or pruning.get("source_run") != str(directory):
            raise ValueError("Pruning receipt ownership/source mismatch")
        config = json.loads((directory / "resolved_config.json").read_text())
        if config.get("stage") not in {"A0", "A1"}:
            raise ValueError("Only an intermediate A0/A1 pruning receipt can replace a checkpoint")
        if pruning.get("status") == "pruning":
            prune_checkpoint_run(directory, pruning["consumer_runs"])
            pruning = json.loads(receipt.read_text())
        pruned = pruning.get("status") == "pruned"
    if not completion.is_file() or (not last.is_file() and not pruned):
        raise RuntimeError(f"Successful job {job['id']} lacks completed.json or last/state.pt")
    info = json.loads(completion.read_text())
    if not isinstance(info, dict) or not isinstance(info.get("steps"), int) or info["steps"] < 1:
        raise RuntimeError(f"Invalid completed.json for {job['id']}")
    if info.get("status") in {"failed", "incomplete", "interrupted"}:
        raise RuntimeError(f"Job {job['id']} reports {info['status']}")
    if last.is_file():
        from src.utils.research_checkpoint_compat import checkpoint_kind,checkpoint_identity
        if checkpoint_kind(last)=="bundle_v1":checkpoint_identity(last)
    return {"output": str(directory), "completed": info, "last": str(last) if last.is_file() else None, "checkpoint_pruned": pruned,
            "best": str(directory / "best/state.pt") if (directory / "best/state.pt").is_file() else None}


def prune_checkpoint_run(source_run, consumer_runs):
    """Discard only an intermediate checkpoint after validated consumers finish.

    This is also the standalone guarded pruning API. Each consumer must be a
    completed run, retain a manager-owned last/state.pt, and name this source in
    its resolved initialization config. Final SFT sources are always refused.
    Metrics, configs, completion markers, unknown files, and consumer checkpoints
    are retained. An ownership receipt makes an interrupted deletion retryable.
    """
    import shutil

    source = Path(source_run).expanduser().resolve()
    marker_path = source / ".checkpoint_manager.json"
    if not marker_path.is_file() or not (source / "completed.json").is_file():
        raise ValueError(f"Source lacks managed checkpoint ownership/completion: {source}")
    config = json.loads((source / "resolved_config.json").read_text())
    if config.get("stage") not in {"A0", "A1"}:
        raise ValueError("Only intermediate A0/A1 checkpoints may be pruned; final SFT is retained")
    consumers = [Path(p).expanduser().resolve() for p in consumer_runs]
    if not consumers or source in consumers:
        raise ValueError("Pruning requires distinct completed consumer runs")
    for consumer in consumers:
        if not (consumer / "completed.json").is_file() or not (consumer / "last/state.pt").is_file():
            raise ValueError(f"Consumer lacks completion and retained last checkpoint: {consumer}")
        completion = json.loads((consumer / "completed.json").read_text())
        if completion.get("steps", 0) < 1 or completion.get("status") in {"failed", "incomplete", "interrupted"}:
            raise ValueError(f"Consumer is not successfully completed: {consumer}")
        consumer_info = json.loads((consumer / ".checkpoint_manager.json").read_text())
        consumer_manager = BestLastCheckpointManager(consumer, disk_budget=DiskBudget(consumer, 0), mode=consumer_info["mode"])
        consumer_manager._resolve_owned_pointer("last")
        from src.utils.research_checkpoint_compat import checkpoint_kind,checkpoint_identity
        if checkpoint_kind(consumer/"last/state.pt")=="bundle_v1":checkpoint_identity(consumer/"last/state.pt")
        consumer_config = json.loads((consumer / "resolved_config.json").read_text())
        references_source = False
        for key in ("initial_checkpoint", "alignment_checkpoint"):
            if consumer_config.get(key):
                reference = Path(consumer_config[key]).expanduser()
                reference = (ROOT / reference).absolute() if not reference.is_absolute() else reference.absolute()
                # Lexical check also works after source pointers were removed by
                # a previously interrupted prune; resolution handles object refs.
                if reference.is_relative_to(source) or reference.resolve().is_relative_to(source):
                    references_source = True
        if not references_source:
            raise ValueError(f"Consumer config does not reference source checkpoint: {consumer}")
    info = json.loads(marker_path.read_text())
    manager = BestLastCheckpointManager(source, disk_budget=DiskBudget(source, 0), mode=info["mode"])
    receipt_path = source / "checkpoint_pruned.json"
    with manager._lock():
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if previous and previous.get("owner") != manager.owner:
            raise ValueError("Pruning receipt ownership mismatch")
        if previous and previous.get("status") == "pruned":
            return previous
        # Validate both pointers before removing either. An interrupted prune
        # may legitimately have removed one or both already.
        for name in ("best", "last"):
            manager._resolve_owned_pointer(name)
        owned = [entry for entry in manager.objects.iterdir() if manager._owns(entry)]
        receipt = {"schema_version": 1, "owner": manager.owner, "status": "pruning",
                   "source_run": str(source), "consumer_runs": [str(p) for p in consumers],
                   "started": previous.get("started", timestamp()) if previous else timestamp(),
                   "logical_bytes_removed": sum(_tree_size(entry) for entry in owned)}
        _atomic_json(receipt_path, receipt)
        for name in ("best", "last"):
            (source / name).unlink(missing_ok=True)
        for entry in owned:
            shutil.rmtree(entry)
        receipt.update(status="pruned", finished=timestamp())
        _atomic_json(receipt_path, receipt)
        return receipt


class ResearchQueue:
    def __init__(self, spec_path, state_path=None, retry_failed=False):
        self.spec_path = Path(spec_path).expanduser().resolve()
        self.spec = validate_spec(json.loads(self.spec_path.read_text()))
        self.spec_hash = hashlib.sha256(json.dumps(self.spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.state_path = Path(state_path).expanduser().resolve() if state_path else self.spec_path.with_suffix(".state.json")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_handle = self.state_path.with_suffix(self.state_path.suffix + ".lock").open("a")
        try:
            fcntl.flock(self.lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_handle.close()
            raise RuntimeError(f"Another queue process already holds {self.state_path}")
        self.retry_failed = retry_failed
        self.budget = DiskBudget(ROOT, self.spec.get("reserve_gb", 120))
        self.interrupted_by = None
        self.process = None
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state.get("spec_sha256") != self.spec_hash:
                self.lock_handle.close()
                raise ValueError("Queue spec changed; use a new state path/queue spec instead of mixing experiments")
        else:
            self.state = {"schema_version": 1, "queue_id": self.spec.get("queue_id", self.spec_path.stem),
                          "spec_sha256": self.spec_hash, "created": timestamp(), "jobs": {}}
        self.log_dir = ROOT / "logs/research/queue"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def persist(self):
        self.state["updated"] = timestamp()
        _atomic_json(self.state_path, self.state)

    def compact_completed_job(self,job,record):
        """Keep completed training durable before retiring optimizer state."""
        if not job.get("compact_completed") or record.get("compaction"):
            return
        record.update(training_completed=True,failure_phase="checkpoint_compaction")
        self.persist()
        from scripts.compact_research_checkpoints import compact_completed_checkpoint
        try:
            report = compact_completed_checkpoint(output_dir(job["argv"]),reserve_gb=self.spec.get("reserve_gb",120))
            artifacts = artifacts_for(job)
        except Exception as error:
            record.update(status="training_completed_compaction_failed",error_type=type(error).__name__,error=str(error),finished=timestamp())
            self.persist()
            raise
        record.update(compaction=report,artifacts=artifacts,failure_phase=None)
        self.persist()
        emit("completed_optimizer_compacted", job=job["id"], report=record["compaction"])

    def signal(self, signum, _frame):
        self.interrupted_by = signum
        emit("queue_signal", signal=signum)
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def prune_finished_intermediates(self):
        for source in self.spec["jobs"]:
            if not source.get("discard_checkpoint_after_dependents", False):
                continue
            record = self.state["jobs"].get(source["id"], {})
            if record.get("status") != "completed" or record.get("checkpoint_pruned"):
                continue
            consumers = [job for job in self.spec["jobs"] if source["id"] in job.get("requires", [])]
            if not consumers or not all(self.state["jobs"].get(job["id"], {}).get("status") == "completed" for job in consumers):
                continue
            receipt = prune_checkpoint_run(output_dir(source["argv"]), [output_dir(job["argv"]) for job in consumers])
            record["checkpoint_pruned"] = True
            record["prune_receipt"] = receipt
            self.persist()
            emit("intermediate_checkpoint_pruned", job=source["id"], logical_bytes_removed=receipt["logical_bytes_removed"])

    def _evaluation_identity(self,job):
        configuration = job["evaluation"]
        manifest = resolve_path(configuration["manifest"])
        raw = manifest.read_bytes()
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        ids = [r["record_id"] for r in records]
        if not records or len(ids)!=len(set(ids)):
            raise ValueError("Evaluation manifest is empty or has duplicate IDs")
        checkpoint = (output_dir(job["argv"])/configuration.get("checkpoint","best")/"state.pt").resolve(strict=True)
        from src.utils.research_checkpoint_compat import checkpoint_identity
        last_heartbeat = time.monotonic()
        def progress(_done,_total):
            nonlocal last_heartbeat
            if self.interrupted_by:raise InterruptedError("Queue interrupted while identifying evaluation checkpoint")
            if time.monotonic()-last_heartbeat>=30:
                self.budget.ensure_space(0)
                emit("evaluation_checkpoint_hashing",job=job["id"],free_gib=self.budget.report()["free_gib"])
                last_heartbeat=time.monotonic()
        artifact=checkpoint_identity(checkpoint,progress=progress)
        return {"checkpoint_sha256":artifact["sha256"],"manifest_sha256":hashlib.sha256(raw).hexdigest(),
                "records":records,"manifest":manifest,"checkpoint":checkpoint,"checkpoint_artifact_identity":artifact,
                "output":resolve_path(configuration["output"])}

    def _validate_evaluation(self,job,identity):
        from src.eval.research_metrics import score_prediction,aggregate_scores
        destination = identity["output"]
        summary = json.loads((destination/"summary.json").read_text())
        protocol = json.loads((destination/"protocol.json").read_text())
        # Image interventions have separate scientific meaning and must never
        # satisfy the ordinary post-training benchmark hook.
        for report in (summary, protocol):
            ablation = report.get("image_ablation", {})
            if not isinstance(ablation, dict) or ablation.get("variant", "none") != "none":
                raise ValueError("Standard queue evaluation requires the original images; image ablation results belong in separate reports")
        for key in ("checkpoint_sha256","manifest_sha256"):
            if summary.get(key)!=identity[key] or protocol.get(key)!=identity[key]:
                raise ValueError(f"Evaluation {key} does not match current actual artifacts")
        artifact=identity.get("checkpoint_artifact_identity",{})
        if artifact.get("format")=="bundle_v1":
            from src.utils.research_checkpoint_compat import same_checkpoint_identity
            for report in (summary,protocol):
                if not same_checkpoint_identity(artifact,report.get("checkpoint_bundle_identity",{})):
                    raise ValueError("Evaluation does not bind every current bundle component")
        count = len(identity["records"])
        if (summary.get("complete_manifest") is not True or summary.get("records")!=count or
                summary.get("manifest_records")!=count or summary.get("complete_four_group_coverage") is not True):
            raise ValueError("Evaluation summary is incomplete or has incorrect manifest/four-group coverage")
        generation = protocol.get("generation",{})
        if (generation.get("max_new_tokens")!=job["evaluation"].get("max_new_tokens",64) or
                generation.get("do_sample") is not False or generation.get("use_cache") is not True):
            raise ValueError("Evaluation generation protocol differs from queue configuration")
        protocol_sha = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        if summary.get("protocol_sha256")!=protocol_sha:
            raise ValueError("Evaluation protocol digest mismatch")
        predictions = [json.loads(line) for line in (destination/"predictions.jsonl").read_text().splitlines() if line.strip()]
        expected = {r["record_id"]:r for r in identity["records"]}
        predicted_ids = [r["record_id"] for r in predictions]
        if len(predicted_ids)!=count or len(set(predicted_ids))!=count or set(predicted_ids)!=set(expected):
            raise ValueError("Evaluation raw predictions do not cover exactly the manifest")
        for row in predictions:
            source = expected[row["record_id"]]
            if row.get("protocol_sha256")!=protocol_sha or row.get("task_group")!=source["task_group"] or row.get("scorer")!=source["scorer"]:
                raise ValueError("Evaluation prediction provenance/task differs from manifest")
            actual = score_prediction(source,row["prediction"])
            if not math.isfinite(float(row["score"])) or not math.isclose(float(row["score"]),actual,abs_tol=1e-12):
                raise ValueError("Evaluation raw score does not match prediction/references")
        computed = aggregate_scores(predictions,require_four_groups=True)
        if (summary.get("groups")!=computed["groups"] or
                not isinstance(summary.get("group_macro_score"),(int,float)) or
                not math.isclose(summary["group_macro_score"],computed["group_macro_score"],abs_tol=1e-9)):
            raise ValueError("Evaluation summary macro/groups differ from verified raw predictions")
        return {"status":"completed","output":str(destination),"summary":str(destination/"summary.json"),
                "checkpoint_sha256":identity["checkpoint_sha256"],"manifest_sha256":identity["manifest_sha256"],
                "protocol_sha256":protocol_sha,"records":count,"group_macro_score":computed["group_macro_score"],
                "verified":timestamp()}

    def evaluate_completed_job(self,job,record):
        """Evaluate only after successful trainer artifacts and optional compaction."""
        if not job.get("evaluation"):
            return True
        record["training_completed"] = True
        try:
            self.budget.ensure_space(int(job["evaluation"].get("anticipated_new_bytes",64*2**20)))
            identity = self._evaluation_identity(job)
            receipt = record.get("evaluation",{})
            if receipt.get("status")=="completed":
                record["evaluation"] = self._validate_evaluation(job,identity)
                self.persist()
                emit("job_evaluation_already_completed",job=job["id"],evaluation=record["evaluation"])
                return True
            configuration = job["evaluation"]
            command = [sys.executable,"-u",str(ROOT/"scripts/eval_research_checkpoint.py"),
                       "--run",str(output_dir(job["argv"])),"--manifest",str(identity["manifest"]),
                       "--output",str(identity["output"]),"--checkpoint",configuration.get("checkpoint","best"),
                       "--max-new-tokens",str(configuration.get("max_new_tokens",64)),
                       "--reserve-gb",str(self.spec.get("reserve_gb",120))]
            if (identity["output"]/"protocol.json").is_file():
                command.append("--resume")
            evaluation = {"status":"running","started":timestamp(),"command":command,
                          "log":str(self.log_dir/f"{job['id']}.evaluation.log"),"output":str(identity["output"])}
            record.update(status="evaluating",evaluation=evaluation)
            self.persist()
            emit("job_evaluation_started",job=job["id"],log=evaluation["log"])
            with Path(evaluation["log"]).open("a",buffering=1) as log_handle:
                log_handle.write(json.dumps({"time":timestamp(),"command":command})+"\n")
                self.process = subprocess.Popen(command,cwd=ROOT,stdout=log_handle,stderr=subprocess.STDOUT,start_new_session=True)
                evaluation["pid"] = self.process.pid
                self.persist()
                heartbeat = time.monotonic()
                interrupted_at = None
                while self.process.poll() is None:
                    if self.interrupted_by:
                        interrupted_at = interrupted_at or time.monotonic()
                        if time.monotonic()-interrupted_at>30:
                            os.killpg(self.process.pid,signal.SIGKILL)
                    if time.monotonic()-heartbeat>=30:
                        self.budget.ensure_space(0)
                        emit("job_evaluation_running",job=job["id"],pid=self.process.pid,free_gib=self.budget.report()["free_gib"])
                        heartbeat = time.monotonic()
                    time.sleep(1)
                evaluation.update(returncode=self.process.returncode,finished=timestamp())
                self.process = None
            if self.interrupted_by:
                raise InterruptedError("Queue interrupted during generation evaluation")
            if evaluation["returncode"]!=0:
                raise RuntimeError(f"Evaluator returned {evaluation['returncode']}; inspect {evaluation['log']}; no implicit retry")
            # Recompute actual file identity after the child, including compaction
            # or concurrent changes; a stale summary cannot mark this job complete.
            final_identity = self._evaluation_identity(job)
            if any(final_identity[key]!=identity[key] for key in ("checkpoint_sha256","manifest_sha256")):
                raise ValueError("Checkpoint or manifest changed during evaluation")
            verified = self._validate_evaluation(job,final_identity)
            record["evaluation"].update(verified)
            self.persist()
            emit("job_evaluation_completed",job=job["id"],evaluation=record["evaluation"])
            return True
        except Exception as error:
            if self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(self.process.pid,signal.SIGTERM)
                    self.process.wait(timeout=10)
                except (ProcessLookupError,subprocess.TimeoutExpired):
                    if self.process.poll() is None:
                        os.killpg(self.process.pid,signal.SIGKILL)
                        self.process.wait(timeout=10)
                finally:
                    self.process = None
            interrupted = bool(self.interrupted_by)
            status = "training_completed_eval_interrupted" if interrupted else "training_completed_eval_failed"
            record.update(status=status,finished=timestamp(),error_type=type(error).__name__,error=str(error))
            record.setdefault("evaluation",{}).update(status="interrupted" if interrupted else "failed",error=str(error))
            self.persist()
            emit("job_evaluation_failed",job=job["id"],status=status,error=str(error))
            if interrupted:
                return False
            raise

    def run(self):
        self.state["status"] = "running"
        self.persist()
        emit("queue_started", queue_id=self.state["queue_id"], jobs=len(self.spec["jobs"]), disk=self.budget.report())
        try:
            for job in self.spec["jobs"]:
                if self.interrupted_by:
                    break
                name = job["id"]
                previous = self.state["jobs"].get(name, {})
                if previous.get("status") == "completed":
                    existing_artifacts = artifacts_for(job, allow_pruned=bool(job.get("discard_checkpoint_after_dependents") or (output_dir(job["argv"]) / "checkpoint_pruned.json").is_file()))
                    if existing_artifacts.get("checkpoint_pruned"):
                        previous["checkpoint_pruned"] = True
                        self.persist()
                    if job.get("evaluation"):
                        previous["artifacts"] = existing_artifacts
                        self.compact_completed_job(job,previous)
                        if not self.evaluate_completed_job(job,previous):
                            break
                        previous["status"] = "completed"
                        self.persist()
                    emit("job_already_completed", job=name)
                    continue
                if previous.get("status") in {"failed","training_completed_eval_failed","training_completed_compaction_failed"} and not self.retry_failed:
                    raise RuntimeError(f"Job {name} previously failed; inspect logs and use --retry-failed after fixing it")
                dependencies = job.get("requires", [])
                missing = [dep for dep in dependencies if self.state["jobs"].get(dep, {}).get("status") != "completed"]
                if missing:
                    self.state["jobs"][name] = {"status": "blocked_dependency", "requires": missing, "updated": timestamp()}
                    self.persist()
                    if self.spec.get("stop_on_failure", True):
                        raise RuntimeError(f"Dependencies not complete for {name}: {missing}")
                    continue
                if previous.get("training_completed"):
                    previous["artifacts"] = artifacts_for(job)
                    self.compact_completed_job(job,previous)
                    if not self.evaluate_completed_job(job,previous):
                        break
                    previous["status"] = "completed"
                    self.persist()
                    self.prune_finished_intermediates()
                    continue
                directory = output_dir(job["argv"])
                argv = list(job["argv"])
                if (directory / "last/state.pt").is_file() and "--resume" not in argv:
                    argv.append("--resume")
                command = [sys.executable, "-u", str(ROOT / "scripts/run_research_stage.py"), *argv]
                record = {"status": "running", "started": timestamp(), "command": command,
                          "output": str(directory), "log": str(self.log_dir / f"{name}.log")}
                self.state["jobs"][name] = record
                self.persist()
                try:
                    # Keep the reserve for checkpoints; stage-specific additional
                    # allocation can be set explicitly in the queue specification.
                    self.budget.ensure_space(int(job.get("anticipated_new_bytes", 2 * 1024 ** 3)))
                    emit("job_started", job=name, stage=argv_value(argv, "--stage"), recipe=argv_value(argv, "--recipe"),
                         log=record["log"], disk=self.budget.report())
                    with Path(record["log"]).open("a", buffering=1) as log_handle:
                        log_handle.write(json.dumps({"time": timestamp(), "command": command}) + "\n")
                        self.process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle,
                                                        stderr=subprocess.STDOUT, start_new_session=True)
                        record["pid"] = self.process.pid
                        self.persist()
                        heartbeat = time.monotonic()
                        interrupted_at = None
                        while self.process.poll() is None:
                            if self.interrupted_by:
                                interrupted_at = interrupted_at or time.monotonic()
                                if time.monotonic() - interrupted_at > 30:
                                    os.killpg(self.process.pid, signal.SIGKILL)
                            if time.monotonic() - heartbeat >= 30:
                                emit("job_running", job=name, pid=self.process.pid, free_gib=self.budget.report()["free_gib"])
                                heartbeat = time.monotonic()
                            time.sleep(1)
                        code = self.process.returncode
                        self.process = None
                    record.update(returncode=code, finished=timestamp())
                    if self.interrupted_by:
                        record["status"] = "interrupted"
                        self.persist()
                        break
                    if code != 0:
                        raise RuntimeError(f"Trainer returned {code}; inspect {record['log']}")
                    record["artifacts"] = artifacts_for(job)
                    # A compactor may permanently retire optimizer state before
                    # returning. Persist successful training first so recovery
                    # never tries to resume that non-resumable completed stage.
                    record["training_completed"] = True
                    self.persist()
                    self.compact_completed_job(job,record)
                    if not self.evaluate_completed_job(job,record):
                        break
                    record["status"] = "completed"
                    self.persist()
                    emit("job_completed", job=name, artifacts=record["artifacts"], disk=self.budget.report())
                    self.prune_finished_intermediates()
                except Exception as error:
                    failed_status = ("training_completed_compaction_failed" if record.get("training_completed") and record.get("failure_phase")=="checkpoint_compaction"
                                     else "training_completed_eval_failed" if record.get("training_completed") and job.get("evaluation") else "failed")
                    record.update(status=failed_status,
                                  finished=timestamp(), error_type=type(error).__name__, error=str(error))
                    self.persist()
                    emit("job_failed", job=name, error=record["error"], log=record["log"])
                    if self.spec.get("stop_on_failure", True):
                        raise
            self.prune_finished_intermediates()
            if self.interrupted_by:
                self.state["status"] = "interrupted"
            else:
                statuses = [self.state["jobs"].get(j["id"], {}).get("status") for j in self.spec["jobs"]]
                self.state["status"] = "completed" if all(s == "completed" for s in statuses) else "failed"
            self.persist()
            emit("queue_finished", queue_id=self.state["queue_id"], status=self.state["status"],
                 note="Smoke/pilot success does not constitute completion of the scientific matrix.")
            return 128 + self.interrupted_by if self.interrupted_by else (0 if self.state["status"] == "completed" else 1)
        except Exception as error:
            self.state.update(status="failed", error_type=type(error).__name__, error=str(error))
            self.persist()
            emit("queue_failed", error=str(error))
            return 1
        finally:
            fcntl.flock(self.lock_handle, fcntl.LOCK_UN)
            self.lock_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec")
    parser.add_argument("--prune-run", help="Guarded pruning of a completed intermediate A0/A1 run")
    parser.add_argument("--consumer-run", action="append", default=[], help="Completed consumer retaining its checkpoint; repeatable")
    parser.add_argument("--state")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.prune_run:
        if args.spec or not args.consumer_run:
            parser.error("--prune-run needs --consumer-run and cannot be combined with --spec")
        emit("intermediate_checkpoint_pruned", **prune_checkpoint_run(args.prune_run, args.consumer_run))
        return
    if not args.spec:
        parser.error("--spec or --prune-run is required")
    queue = ResearchQueue(args.spec, args.state, args.retry_failed)
    signal.signal(signal.SIGTERM, queue.signal)
    signal.signal(signal.SIGINT, queue.signal)
    raise SystemExit(queue.run())


if __name__ == "__main__":
    main()
