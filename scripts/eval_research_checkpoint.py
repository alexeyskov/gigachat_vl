#!/usr/bin/env python3
"""Greedy held-out generation/scoring; refuses to share an occupied GPU."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.eval.research_inference import (
    GENERATION_SCHEMA_VERSION, generate_prediction, load_research_checkpoint, sha256_file,
)
from src.eval.research_metrics import aggregate_scores, score_prediction
from src.eval.research_ablation import VARIANTS, transform_image_ablation, write_transformed_manifest
from src.utils.research_checkpoint_compat import checkpoint_kind, checkpoint_identity, same_checkpoint_identity


def assert_gpu_idle(gpu=0):
    result = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,used_gpu_memory",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Cannot verify GPU availability: "+result.stderr.strip())
    occupants = [line.strip() for line in result.stdout.splitlines()
                 if line.strip() and line.split(",")[0].strip() != str(os.getpid())]
    if occupants:
        raise RuntimeError("GPU is occupied; generation evaluation must wait: "+"; ".join(occupants))


def atomic_json(path,payload):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    os.replace(temporary,path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--checkpoint",choices=["best","last"],default="best")
    parser.add_argument("--max-new-tokens",type=int,default=64)
    parser.add_argument("--max-records",type=int)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--cache-dir",default="cache/vision")
    parser.add_argument("--reserve-gb",type=float,default=120)
    parser.add_argument("--image-ablation",choices=VARIANTS,default="none")
    parser.add_argument("--ablation-seed",type=int,default=42)
    return parser.parse_args(argv)


def _read_predictions(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    result = []
    for index,line in enumerate(lines):
        try:
            value = json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if index != len(lines)-1 or line.endswith(b"\n"):
                raise ValueError("Corrupt completed prediction line; refusing silent repair.")
            # A killed writer may leave only its final append incomplete.
            with path.open("r+b") as handle:
                handle.truncate(sum(len(x) for x in lines[:index]))
            continue
        if not isinstance(value,dict) or "record_id" not in value:
            raise ValueError("Prediction record is malformed.")
        result.append(value)
        if index==len(lines)-1 and not line.endswith(b"\n"):
            # A complete JSON value without its trailing delimiter is valid,
            # but the next append must not concatenate two JSON objects.
            with path.open("ab") as handle:
                handle.write(b"\n");handle.flush();os.fsync(handle.fileno())
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.max_new_tokens <= 0 or (args.max_records is not None and args.max_records <= 0):
        raise ValueError("Token/record limits must be positive.")
    assert_gpu_idle(args.gpu)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    lock = (output/".evaluation.lock").open("a+")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This evaluation output is already being written.")
    events = output/"progress.jsonl"
    def event(**payload):
        data = dict(time=time.time(),**payload)
        with events.open("a") as handle:
            handle.write(json.dumps(data,ensure_ascii=False)+"\n")
        atomic_json(output/"progress.json",data)
        print(json.dumps(data,ensure_ascii=False),flush=True)
    checkpoint = (Path(args.run)/args.checkpoint/"state.pt").resolve(strict=True)
    manifest_path = Path(args.manifest)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    records = [json.loads(line) for line in manifest_bytes.splitlines() if line.strip()]
    ids = [record["record_id"] for record in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError("Evaluation manifest is empty or contains duplicate record IDs.")
    protocol_file = output/"protocol.json"
    if protocol_file.exists():
        previous = json.loads(protocol_file.read_text())
        variant = previous.get("image_ablation",{}).get("variant","none")
        seed = previous.get("image_ablation",{}).get("seed",42)
        if variant != args.image_ablation or seed != args.ablation_seed:
            raise ValueError("Resume image ablation variant/seed differs; choose a separate output directory.")
    transformed,ablation_protocol = transform_image_ablation(records,variant=args.image_ablation,
        seed=args.ablation_seed,output_dir=output,source_manifest_sha256=manifest_sha,
        reserve_bytes=int(args.reserve_gb*1024**3))
    if args.image_ablation != "none":
        actual_digest = write_transformed_manifest(transformed,output/"transformed_manifest.jsonl",
            reserve_bytes=int(args.reserve_gb*1024**3))
        if actual_digest != ablation_protocol["transformed_manifest_sha256"]:
            raise ValueError("Actual transformed manifest digest differs.")
    selected = transformed[:args.max_records] if args.max_records else transformed
    event(event="hashing_checkpoint",path=str(checkpoint),bytes=checkpoint.stat().st_size)
    before = checkpoint.stat()
    last_hash_update = [time.monotonic()]
    def hash_progress(done,total):
        if time.monotonic()-last_hash_update[0]>=10:
            event(event="hashing_checkpoint",bytes_completed=done,bytes_total=total)
            last_hash_update[0] = time.monotonic()
    bundle_identity = checkpoint_identity(checkpoint,progress=hash_progress) if checkpoint_kind(checkpoint)=='bundle_v1' else None
    checkpoint_sha = bundle_identity['sha256'] if bundle_identity is not None else sha256_file(checkpoint,progress=hash_progress)
    after = checkpoint.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError("Checkpoint changed during hashing.")
    sources = [ROOT/"src/eval/research_inference.py",Path(__file__).resolve(),
               ROOT/"src/eval/research_metrics.py",ROOT/"scripts/run_research_stage.py",
               ROOT/"src/model/research_moe.py",ROOT/"src/model/smallthinker.py",
               ROOT/"src/model/research_quantization.py",ROOT/"src/model/research_vision.py",
               ROOT/"src/model/gigachat_vl.py",ROOT/"src/dataset/vision_cache.py",
               ROOT/"scripts/extract_vision_features.py",ROOT/"src/eval/research_ablation.py",
               ROOT/"src/utils/research_checkpoint_compat.py",ROOT/"src/utils/research_checkpoint_bundle.py",
               ROOT/"src/utils/research_checkpoint_io.py"]
    versions = {}
    for package in ("torch","transformers","peft","bitsandbytes","tokenizers","huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    protocol = {"schema_version":GENERATION_SCHEMA_VERSION,"checkpoint_sha256":checkpoint_sha,
                "manifest_sha256":manifest_sha,"manifest_records":len(records),
                "image_ablation":ablation_protocol,
                "selected_record_ids_sha256":hashlib.sha256(json.dumps([r["record_id"] for r in selected]).encode()).hexdigest(),
                "generation":{"do_sample":False,"num_beams":1,"use_cache":True,
                              "max_new_tokens":args.max_new_tokens,"autocast":"cuda_bf16_preserving_master_dtypes",
                              "prompt":"native_user_only_image_marker_no_gold","logits_to_keep":"1_when_supported"},
                "source_sha256":{str(path.relative_to(ROOT)):sha256_file(path) for path in sources}}
    if bundle_identity is not None:protocol["checkpoint_bundle_identity"]=bundle_identity
    protocol["environment"] = dict(python=sys.version,packages=versions)
    protocol_sha = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    protocol_file = output/"protocol.json"
    prediction_file = output/"predictions.jsonl"
    if args.resume:
        if not protocol_file.exists() or json.loads(protocol_file.read_text()) != protocol:
            raise ValueError("Resume protocol/checkpoint/manifest/code differs.")
        rows = _read_predictions(prediction_file)
    else:
        if protocol_file.exists() or prediction_file.exists():
            raise ValueError("Existing evaluation results require --resume; refusing overwrite.")
        atomic_json(protocol_file,protocol)
        rows = []
    done = {row["record_id"] for row in rows}
    if len(done)!=len(rows) or not done.issubset({r["record_id"] for r in selected}):
        raise ValueError("Saved prediction IDs duplicate or differ from selected manifest.")
    if any(row.get("protocol_sha256")!=protocol_sha for row in rows):
        raise ValueError("Saved predictions have mixed provenance.")
    started = time.monotonic()
    remaining = [record for record in selected if record["record_id"] not in done]
    if remaining:
        event(event="loading_model",completed_records=len(rows),selected_records=len(selected))
        model,model_provenance = load_research_checkpoint(checkpoint,cache_dir=args.cache_dir,reserve_gb=args.reserve_gb)
        atomic_json(output/"model_provenance.json",model_provenance)
        for offset,record in enumerate(remaining):
            begin = time.monotonic()
            generated = generate_prediction(model,record,max_new_tokens=args.max_new_tokens)
            row = {key:record[key] for key in ("record_id","question","image_hashes","answers","scorer","task_group","source","benchmark","choices","scoring_protocol","image_ablation") if key in record}
            row.update(generated,score=score_prediction(record,generated["prediction"]),
                       protocol_sha256=protocol_sha,generation_seconds=time.monotonic()-begin)
            with prediction_file.open("a") as handle:
                handle.write(json.dumps(row,ensure_ascii=False)+"\n")
                handle.flush();os.fsync(handle.fileno())
            rows.append(row)
            elapsed = time.monotonic()-started
            event(event="generated",record_id=record["record_id"],completed_records=len(rows),
                  selected_records=len(selected),manifest_records=len(records),elapsed_seconds=elapsed,
                  eta_seconds=elapsed/(offset+1)*(len(remaining)-offset-1),
                  prediction=generated["prediction"],score=row["score"])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=manifest_sha:
        raise ValueError("Evaluation manifest changed during generation.")
    if args.image_ablation != "none" and hashlib.sha256((output/"transformed_manifest.jsonl").read_bytes()).hexdigest()!=ablation_protocol["transformed_manifest_sha256"]:
        raise ValueError("Transformed ablation manifest changed during generation.")
    complete = len(rows)==len(records)
    metrics = aggregate_scores(rows,require_four_groups=complete)
    if not complete:
        metrics["group_macro_score"] = None
    metrics.update(complete_manifest=complete,manifest_records=len(records),
                   protocol_sha256=protocol_sha,checkpoint_sha256=checkpoint_sha,
                   manifest_sha256=manifest_sha,image_ablation=ablation_protocol)
    if bundle_identity is not None:
        if not same_checkpoint_identity(bundle_identity,checkpoint_identity(checkpoint,verify_components=False)):
            raise ValueError("Bundle checkpoint components changed during generation.")
        metrics["checkpoint_bundle_identity"]=bundle_identity
        metrics["checkpoint_artifact_total_bytes"]=bundle_identity["total_bytes"]
    atomic_json(output/("summary.json" if complete else "partial_summary.json"),metrics)
    event(event="completed" if complete else "partial_completed",**metrics)
    fcntl.flock(lock,fcntl.LOCK_UN)
    lock.close()
    return metrics


if __name__=="__main__":
    main()
