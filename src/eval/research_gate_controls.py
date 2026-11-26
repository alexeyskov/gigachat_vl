"""Versioned, inference-only P07 controls for the verified SmallThinker c path.

No checkpoint/logit value is written.  Exact endpoint gates are supplied by an
instance-local Parameter.sigmoid override, which both frozen Small dispatch
paths call.  This deliberately excludes packed/shared-expert implementations.
Importing the metadata/preflight helpers does not import Torch or load a model.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
import types

ROOT = Path(__file__).resolve().parents[2]
OWNER = "research_small_gate_control_p07_v1"
CONDITIONS = ("learned", "all_open", "visual_closed", "text_closed", "all_closed")
RECIPES = ("1c", "2c", "3c")
MANIFEST = "datasets/research/eval-heldout-v3/evaluation.jsonl"
MANIFEST_SHA256 = "9fcb6783de70f44a94505eefa4ee90eef5759319c1db59850bd351777f38f5f4"
PUBLIC_FIELDS = frozenset(("record_id", "question", "image_paths", "image_hashes"))
FROZEN_SOURCES = {
    "src/model/research_moe.py": "f8cbfd6a8fdb7044d25d031f0a028055cb114a7ebd71a9b4f873dcd752674c6c",
    "src/model/smallthinker.py": "7d5aaa551b98e7958f1a69930c797f7906d7c770084b156ba36cc5d577cbdd26",
    "src/eval/research_inference.py": "47576ba901c5e6ba3dc95b8c8777872d36c3ea73f2a4e0c6d74a0955df75744d",
    "src/eval/research_metrics.py": "440424ed720dbf17fce9a12ccb4ec6565f30298c6c40e9b47fd0ab60a8ed6844",
    "scripts/run_research_stage.py": "f7fd22d1c94fb337af2be60d3324415b2b2c8cd3d62e9799f2da4c05ace87cfc",
    "src/eval/research_caption_generation.py": "754c261266ab3544d81560abd90447d6abc2ad6b74b987321d007815407abefa",
}
_ACTIVE = set()
_LOCK = threading.Lock()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    path = Path(path)
    before = path.stat()
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            result.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("P07 source changed while hashing")
    return result.hexdigest()


def verify_frozen_sources():
    for name, expected in FROZEN_SOURCES.items():
        if file_sha(ROOT / name) != expected:
            raise ValueError("P07 verified Small source differs: " + name)
    return dict(FROZEN_SOURCES)


def tensor_identity(tensor):
    """Raw-bit fingerprint; no casts or NaN canonicalization."""
    import torch
    detached = tensor.detach().contiguous().cpu()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(detached.view(torch.uint8).numpy().tobytes()).hexdigest()}


def applied_gates(logits, condition):
    import torch
    if condition not in CONDITIONS:
        raise ValueError("Unknown P07 gate condition")
    result = logits.detach().sigmoid().clone()
    if condition == "all_open":
        result.fill_(1)
    elif condition == "visual_closed":
        result[:, 1] = 0
    elif condition == "text_closed":
        result[:, 0] = 0
    elif condition == "all_closed":
        result.zero_()
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite applied P07 gates")
    return result


def gate_registry(model, *, recipe=None):
    """Reject every path except exact native Small E+K c wrappers."""
    verify_frozen_sources()
    import torch
    from src.model.research_moe import ResearchSmallThinkerMoE
    from src.model.smallthinker import SmallThinkerExpert, SmallThinkerMoeBlock
    if recipe is not None and recipe not in RECIPES:
        raise ValueError("P07 supports only Small 1c/2c/3c")
    if any(module.training for module in model.modules()):
        raise ValueError("P07 requires the entire model in eval mode")
    result = []
    seen = set()
    for name, block in model.named_modules():
        if isinstance(block, SmallThinkerMoeBlock):
            raise ValueError("P07 refuses an unwrapped native Small block")
        if not isinstance(block, ResearchSmallThinkerMoE):
            continue
        if (type(block) is not ResearchSmallThinkerMoE or block.policy != "c" or
                block.family not in (1, 2, 3) or getattr(block.config, "model_type", None) != "smallthinker" or
                block.warmup_force_new or block.training):
            raise ValueError("P07 requires eval-only exact Small c wrappers, without warmup/shared/packed paths")
        if recipe is not None and block.family != int(recipe[0]):
            raise ValueError("P07 recipe and installed router family differ")
        for method in ("forward", "_shared_output", "_run_expert", "_grouped_original_dispatch"):
            if method in block.__dict__:
                raise ValueError("P07 refuses an overridden Small dispatch method")
        experts = list(block.experts) + list(block.visual_experts)
        if len(block.experts) != block.old_count or len(block.visual_experts) != block.added_count or block.added_count <= 0:
            raise ValueError("P07 E+K expert census differs")
        if any(type(expert) is not SmallThinkerExpert for expert in experts):
            raise ValueError("P07 expert contract is not verified Small biasless ReGLU")
        for expert in experts:
            for layer in (expert.up, expert.gate, expert.down):
                if getattr(layer, "bias", None) is not None:
                    raise ValueError("P07 supports only biasless Small experts")
        logits = block.modality_gate_logits
        if (not isinstance(logits, torch.nn.Parameter) or logits.dtype != torch.float32 or
                tuple(logits.shape) != (len(experts), 2) or logits.device.type == "meta" or
                not torch.isfinite(logits).all() or logits.requires_grad or
                "sigmoid" in logits.__dict__ or id(logits) in seen):
            raise ValueError("P07 requires unique finite frozen FP32 gate parameters with native sigmoid")
        seen.add(id(logits))
        result.append((name, block, logits))
    if not result:
        raise ValueError("P07 found no verified Small c gates")
    # A native LLM must have every configured MoE layer; standalone block fixtures
    # deliberately have no num_hidden_layers contract at their root.
    config = getattr(model, "config", None)
    if config is not None and not isinstance(model, ResearchSmallThinkerMoE):
        if getattr(config, "model_type", None) != "smallthinker" or len(result) != config.num_hidden_layers:
            raise ValueError("P07 backbone/layer census differs")
    return result


def _tensor_version(tensor):
    # An inference_mode-created result has no version counter.  It remains
    # private to this closure and every returned endpoint is a fresh clone;
    # exit still verifies its raw-bit checksum.
    try:
        return tensor._version
    except RuntimeError:
        return None


@contextmanager
def gate_control(model, condition, *, recipe=None):
    """Temporary inference scope. No Parameter replacement/value writes.

    The scope is exclusive within this process and must run without autograd.
    Learned is a literal identity: it does not install any sigmoid override.
    Every call checks the override tensor checksum; exit checks original gate
    identity/version/rawbits and removes all installed overrides even on error.
    """
    import torch
    if torch.is_grad_enabled():
        raise ValueError("P07 controls require torch.no_grad/inference_mode")
    if condition not in CONDITIONS:
        raise ValueError("Unknown P07 gate condition")
    entries = gate_registry(model, recipe=recipe)
    ids = {id(parameter) for _, _, parameter in entries}
    with _LOCK:
        if _ACTIVE:
            raise ValueError("P07 gate scopes cannot nest or overlap")
        _ACTIVE.update(ids)
    saved, installed, details, applied_saved = [], [], [], []
    try:
        for name, block, parameter in entries:
            original = tensor_identity(parameter)
            saved.append((name, block, parameter, parameter._version, parameter.data_ptr(), original))
            learned = parameter.detach().sigmoid()
            applied = applied_gates(parameter, condition)
            applied_identity = tensor_identity(applied)
            applied_saved.append((applied, applied_identity))
            if condition != "learned":
                # Return a fresh clone so no downstream operation can mutate the
                # scope's private endpoint tensor. This is E+K by 2, not weights.
                def forced(self, _applied=applied, _version=_tensor_version(applied)):
                    if torch.is_grad_enabled() or self.requires_grad:
                        raise ValueError("P07 controlled sigmoid is inference-only")
                    if self.device != _applied.device or self.dtype != _applied.dtype:
                        raise ValueError("P07 gate moved inside active scope")
                    if _tensor_version(_applied) != _version:
                        raise ValueError("P07 applied gate tensor changed")
                    return _applied.clone()
                parameter.sigmoid = types.MethodType(forced, parameter)
                installed.append((parameter, parameter.sigmoid))
            details.append({"module": name, "old_experts": block.old_count, "added_experts": block.added_count,
                            "gate_logit_tensor": original, "learned_gate_tensor": tensor_identity(learned),
                            "applied_gate_tensor": applied_identity,
                            "applied_values": applied.cpu().tolist(), "columns": ["text", "visual"]})
        receipt = {"owner": OWNER, "condition": condition, "recipe": recipe,
                   "gate_application": "g * Expert(g * x); frozen routing implementation unchanged",
                   "routing_caveat": "Later layer inputs and top-k may change; whole-model routing is not held fixed",
                   "ideal_expert_scaling": "biasless ReGLU expert branch is degree three in positive scalar g, including the outer gate",
                   "parameter_values_written": False, "gates": details, "gates_sha256": digest(details)}
        yield receipt
    finally:
        changed_overrides = []
        for parameter, method in installed:
            if parameter.__dict__.get("sigmoid") is not method:
                changed_overrides.append(id(parameter))
            parameter.__dict__.pop("sigmoid", None)
        with _LOCK:
            _ACTIVE.difference_update(ids)
        corrupt = []
        for name, block, parameter, version, pointer, original in saved:
            if (block.modality_gate_logits is not parameter or parameter._version != version or
                    parameter.data_ptr() != pointer or parameter.requires_grad or tensor_identity(parameter) != original):
                corrupt.append(name)
        if corrupt or changed_overrides or any(tensor_identity(value) != identity for value, identity in applied_saved):
            raise ValueError("P07 original gate parameters changed inside scope: " + repr(corrupt))


def public_record(record):
    """Copy only question/image/identity fields; scorer/gold never reach model."""
    value = {key: record[key] for key in PUBLIC_FIELDS}
    if (not isinstance(value["record_id"], str) or not value["record_id"] or
            not isinstance(value["question"], str) or not value["question"].strip() or
            not isinstance(value["image_paths"], list) or len(value["image_paths"]) != 1 or
            not isinstance(value["image_hashes"], list) or len(value["image_hashes"]) != 1):
        raise ValueError("P07 requires one-image fixed heldout public records")
    if "[RESEARCH_IMAGE]" in value["question"]:
        raise ValueError("P07 question collides with reserved marker")
    if any(not isinstance(path, str) or not Path(path).is_absolute() for path in value["image_paths"]):
        raise ValueError("P07 audited absolute image paths required")
    image_hash = value["image_hashes"][0]
    if not isinstance(image_hash, str) or len(image_hash) != 64 or any(c not in "0123456789abcdef" for c in image_hash):
        raise ValueError("P07 canonical image hash missing")
    return {key: list(item) if isinstance(item, list) else item for key, item in value.items()}


def prepare_context(run, *, checkpoint="best", condition="learned", manifest=None):
    """Read-only CPU preflight. No Torch, model construction or gate execution."""
    verify_frozen_sources()
    if condition not in CONDITIONS:
        raise ValueError("Unknown P07 gate condition")
    from src.eval.research_caption_generation import checkpoint_context, environment_context, _file, _rows, stat
    manifest = Path(manifest) if manifest is not None else ROOT / MANIFEST
    manifest = manifest.resolve(strict=True)
    if file_sha(manifest) != MANIFEST_SHA256:
        raise ValueError("P07 requires the unchanged fixed heldout-v3 512 manifest")
    full = _rows(manifest)
    records = [public_record(record) for record in full]
    groups = {record.get("task_group") for record in full}
    if (len(records) != 512 or len({row["record_id"] for row in records}) != 512 or groups !=
            {"general_vqa", "ocr_documents", "charts_diagrams", "knowledge_reasoning"}):
        raise ValueError("P07 requires complete four-group 512 rows")
    model = checkpoint_context(run, checkpoint)
    if model["config"].get("recipe") not in RECIPES or model["config"].get("model") != "Tiiny/SmallThinker-4BA0.6B-Instruct":
        raise ValueError("P07 supports only completed Small 1c/2c/3c SFT")
    dependencies = ("src/model/research_quantization.py", "src/model/research_vision.py",
        "src/model/gigachat_vl.py", "src/dataset/vision_cache.py", "scripts/extract_vision_features.py",
        "src/utils/research_checkpoint_compat.py", "src/utils/research_checkpoint_bundle.py",
        "src/utils/research_checkpoint_io.py", "src/utils/research_storage.py", "RESEARCH_PIPELINE.md")
    sources = {name: _file(ROOT / name) for name in (*FROZEN_SOURCES, *dependencies,
        "src/eval/research_gate_controls.py", "scripts/eval_research_gate_control.py")}
    images = {}
    for record in records:
        path = record["image_paths"][0]
        previous = images.get(path)
        item = {"path": path, "canonical_rgb_sha256": record["image_hashes"][0], "stat": stat(path)}
        if previous is not None and previous != item:
            raise ValueError("P07 image path/hash census conflicts")
        images[path] = item
    static = {"schema_version": 1, "owner": OWNER, "scope": "P07 mechanistic pilot; P06 corpus routing and primary completion remain open",
              "condition": condition, "checkpoint": model, "manifest": _file(manifest, MANIFEST_SHA256),
              "record_order_sha256": digest([row["record_id"] for row in records]),
              "public_records_sha256": digest(records), "records": 512, "images": list(images.values()),
              "sources": sources, "environment": environment_context(), "image_ablation": "none",
              "generation": {"batch_size": 1, "do_sample": False, "num_beams": 1, "use_cache": True,
                             "max_new_tokens": 64, "implementation": "frozen research_inference native B1",
                             "gold_policy": "exact public four-field projection only; fixed scoring after generation"},
              "gate_control": {"endpoint_values": "exact 0/1, without logit writes", "column_order": ["text", "visual"],
                               "application": "g * Expert(g * x)", "global_topk_fixed": False}}
    return {"static": static, "records": records}


def assert_context(context, *, rehash=False):
    """Future owner hook: fail closed if role, checkpoint, sources or inputs drift."""
    from src.eval.research_caption_generation import environment_context, stat, _checkpoint_identity
    static = context["static"]
    model = static["checkpoint"]
    verify_frozen_sources()
    files = [static["manifest"], *static["sources"].values(), *model["files"].values(),
             *(role["marker_file"] for role in model["roles"].values())]
    for item in files:
        if stat(item["path"]) != item["stat"] or (rehash and file_sha(item["path"]) != item["sha256"]):
            raise ValueError("P07 input/source/provenance changed: " + item["path"])
    for image in static["images"]:
        if stat(image["path"]) != image["stat"]:
            raise ValueError("P07 image file changed")
    for name, role in model["roles"].items():
        pointer = Path(model["run"]) / name
        if not pointer.is_symlink() or str(pointer.readlink()) != role["pointer"] or str(pointer.resolve()) != role["object"]:
            raise ValueError("P07 checkpoint role changed")
    for member in model["artifact"]["members"]:
        if stat(member["path"]) != model["member_stats"][member["path"]]:
            raise ValueError("P07 checkpoint member changed")
    if rehash and _checkpoint_identity(model["checkpoint_path"]) != model["artifact"]:
        raise ValueError("P07 checkpoint content changed")
    if environment_context() != static["environment"]:
        raise ValueError("P07 runtime environment changed")
    if digest(context["records"]) != static["public_records_sha256"]:
        raise ValueError("P07 public records changed")


def control_protocol(context, receipt, model_provenance):
    """Bind a future raw journal to observed gates, never to a legacy eval cache."""
    static = context["static"]
    config = static["checkpoint"]["config"]
    role = static["checkpoint"]["role"]
    step = static["checkpoint"]["roles"][role]["marker"]["step"]
    if (receipt.get("owner") != OWNER or receipt.get("condition") != static["condition"] or
            receipt.get("recipe") != config["recipe"] or receipt.get("parameter_values_written") is not False or
            digest(receipt.get("gates")) != receipt.get("gates_sha256")):
        raise ValueError("P07 observed gate receipt differs from requested condition/checkpoint")
    if (model_provenance.get("recipe") != config["recipe"] or model_provenance.get("stage") != "SFT" or
            model_provenance.get("training_step") != step or
            model_provenance.get("model_ref", {}).get("repo") != config["model"] or
            model_provenance.get("vision_ref", {}).get("repo") != config["vision"]):
        raise ValueError("P07 loaded checkpoint provenance differs")
    for field in ("model_ref", "vision_ref"):
        revision = model_provenance.get(field, {}).get("revision")
        if not isinstance(revision, str) or len(revision) not in (40, 64) or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("P07 loaded source revision must be pinned")
    fingerprint = model_provenance.get("cache_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("P07 loaded vision cache fingerprint missing")
    return {"schema_version": 1, "owner": OWNER, "static": static,
            "observed_gate_control": receipt, "loaded_model": model_provenance,
            "namespace": "evaluation/gate_controls_p07_v1/" + static["condition"],
            "legacy_cache_compatible": False, "scientific_completion_claim": False}


def validate_resume(previous, expected):
    if not isinstance(previous, dict) or previous.get("owner") != OWNER or previous != expected:
        raise ValueError("P07 resume condition/gates/checkpoint/input/code differs; separate namespace required")
    return True


class NativeGateRuntime:
    """Scoped observer around unchanged native generation, under gate_control."""
    def load(self, checkpoint, cache_dir, reserve_gb):
        from src.eval.research_inference import load_research_checkpoint
        return load_research_checkpoint(checkpoint, cache_dir=cache_dir, reserve_gb=reserve_gb)

    def generate(self, model, record):
        from unittest.mock import patch
        from src.eval import research_inference as native
        record = public_record(record)
        original = native.build_generation_inputs
        captured = []
        def observe(actual_model, actual_record, **kwargs):
            if actual_model is not model or actual_record != record or kwargs.get("max_new_tokens") != 64:
                raise ValueError("P07 native generation public input/budget changed")
            prepared = original(actual_model, actual_record, **kwargs)
            captured.append({key: prepared[key] for key in ("prefix_token_ids", "suffix_token_ids", "prompt_sha256", "visual_tokens", "prefill_tokens")})
            return prepared
        with patch.object(native, "build_generation_inputs", observe):
            result = native.generate_prediction(model, record, max_new_tokens=64)
        if len(captured) != 1 or captured[0]["prompt_sha256"] != result["prompt_sha256"]:
            raise ValueError("P07 actual native prompt observation differs")
        return {**result, **captured[0], "prompt_token_ids_sha256": digest(captured[0])}
