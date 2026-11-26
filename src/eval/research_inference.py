"""Gold-independent native generation from research best/last checkpoints."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.model.research_moe import set_modality_mask


IMAGE_MARKER = "[RESEARCH_IMAGE]"
GENERATION_SCHEMA_VERSION = 1


def sha256_file(path, *, progress=None):
    path = Path(path)
    digest = hashlib.sha256()
    total = path.stat().st_size
    completed = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8*2**20):
            digest.update(chunk)
            completed += len(chunk)
            if progress is not None:
                progress(completed, total)
    return digest.hexdigest()


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def freeze_for_generation(model):
    """Preserve FP32 masters; inference uses bf16 autocast only on CUDA."""
    model.requires_grad_(False)
    model.eval()
    for candidate in (getattr(model, "llm", None), getattr(model, "base", None)):
        if candidate is not None:
            if hasattr(candidate, "gradient_checkpointing_disable"):
                candidate.gradient_checkpointing_disable()
            candidate.config.use_cache = True
    extractor = getattr(model, "extractor", None)
    if extractor is not None:
        for name in ("vision_tower", "vision_projector"):
            module = getattr(extractor, name, None)
            if module is not None:
                module.requires_grad_(False).eval()
    return model


@torch.no_grad()
def build_generation_inputs(model, record, *, max_new_tokens=64):
    """Read only the question and image fields, never any answer/reference field."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    paths = record.get("image_paths", [])
    if len(paths) > 1:
        raise NotImplementedError("The research evaluation profile currently supports one image per record.")
    question = record["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Evaluation question must be nonempty text.")
    if IMAGE_MARKER in question:
        raise ValueError("Question collides with the reserved image insertion marker.")
    content = (IMAGE_MARKER+"\n" if paths else "")+question
    messages = [{"role": "user", "content": content}]
    prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if getattr(model.base.config, "model_type", None) == "gpt_oss":
        # Match the project's explicit Harmony direct-answer/final channel.
        prompt += "<|channel|>final<|message|>"
    if paths:
        if prompt.count(IMAGE_MARKER) != 1:
            raise ValueError("Native chat template did not preserve one image marker.")
        before, after = prompt.split(IMAGE_MARKER, 1)
    else:
        before, after = "", prompt
    prefix_ids = model.tokenizer.encode(before, add_special_tokens=False)
    suffix_ids = model.tokenizer.encode(after, add_special_tokens=False)
    cache_kwargs = {}
    if "image_hashes" in record:
        hashes = record["image_hashes"]
        if len(hashes) != len(paths):
            raise ValueError("Image path/hash count differs.")
        cache_kwargs["image_hashes"] = hashes
    features = model.cache.get_many(paths, model.encode, **cache_kwargs) if paths else []
    if len(features) != len(paths):
        raise ValueError("Cached feature/image counts differ.")
    for feature in features:
        if feature.ndim != 2 or not feature.shape[0] or not torch.isfinite(feature).all():
            raise ValueError("Nonfinite or malformed visual features.")
    embedding = model.base.get_input_embeddings()
    device = embedding.weight.device
    chunks = [embedding(torch.tensor(prefix_ids, device=device, dtype=torch.long))]
    visual_tokens = 0
    with _autocast(device):
        for feature in features:
            dtype = next(model.projector.parameters()).dtype
            chunks.append(model.projector(feature.to(device=device, dtype=dtype)))
            visual_tokens += feature.shape[0]
    chunks.append(embedding(torch.tensor(suffix_ids, device=device, dtype=torch.long)))
    inputs = torch.cat(chunks, dim=0).unsqueeze(0)
    total = inputs.shape[1]
    maximum = int(getattr(model.args, "max_length", 4096))
    model_maximum = int(getattr(model.base.config, "max_position_embeddings", maximum))
    context_limit = min(maximum, model_maximum)
    if total+max_new_tokens > context_limit:
        raise ValueError(f"Evaluation prompt+generation exceeds context: {total}+{max_new_tokens}>{context_limit}; no silent truncation.")
    modality = torch.tensor([False]*len(prefix_ids)+[True]*visual_tokens+[False]*len(suffix_ids),
                            device=device, dtype=torch.bool).unsqueeze(0)
    attention = torch.ones((1,total), device=device, dtype=torch.bool)
    return {
        "inputs_embeds": inputs, "attention_mask": attention, "modality_mask": modality,
        "prompt": prompt, "prefix_token_ids": prefix_ids, "suffix_token_ids": suffix_ids,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "visual_tokens": visual_tokens, "prefill_tokens": total,
    }


@torch.no_grad()
def generate_prediction(model, record, *, max_new_tokens=64):
    prepared = build_generation_inputs(model, record, max_new_tokens=max_new_tokens)
    set_modality_mask(model.base, prepared["modality_mask"], prepared["attention_mask"])
    kwargs = dict(inputs_embeds=prepared["inputs_embeds"], attention_mask=prepared["attention_mask"],
                  do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True,
                  return_dict_in_generate=True, output_scores=True,
                  pad_token_id=model.tokenizer.pad_token_id)
    signature = inspect.signature(model.base.forward)
    if "logits_to_keep" in signature.parameters:
        kwargs["logits_to_keep"] = 1
    with _autocast(prepared["inputs_embeds"].device):
        generated = model.base.generate(**kwargs)
    # inputs_embeds generation usually returns only generated IDs. Counting
    # returned score steps also handles native implementations with dummy IDs.
    count = len(generated.scores)
    if count > max_new_tokens or generated.sequences.ndim != 2 or generated.sequences.shape[0] != 1:
        raise ValueError("Unexpected native generation output shape/length.")
    if generated.scores:
        valid = [torch.isfinite(scores).any(-1).all() & ~torch.isnan(scores).any() & ~torch.isposinf(scores).any()
                 for scores in generated.scores]
        if not torch.stack(valid).all():
            raise ValueError("Native generation produced invalid logits.")
    ids = generated.sequences[0, -count:].detach().cpu().tolist() if count else []
    prediction = model.tokenizer.decode(ids, skip_special_tokens=True).strip()
    return {
        "prediction": prediction, "generated_token_ids": ids, "generated_tokens": count,
        "prompt_sha256": prepared["prompt_sha256"], "prefill_tokens": prepared["prefill_tokens"],
        "visual_tokens": prepared["visual_tokens"], "logits_to_keep": kwargs.get("logits_to_keep"),
    }


def _validate_learned_state(model, state):
    expected = model.state_dict()
    if not state or not any(key.startswith("projector.") for key in state):
        raise ValueError("Research checkpoint has no alignment projector.")
    unknown = set(state)-set(expected)
    if unknown:
        raise ValueError(f"Unrecognized checkpoint tensors: {sorted(unknown)[:10]}")
    required = {name for name,parameter in model.named_parameters() if parameter.requires_grad}
    missing = required-set(state)
    if missing:
        raise ValueError(f"Checkpoint omits learned parameters required by its training recipe: {sorted(missing)[:10]}")
    for key,value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f"Checkpoint tensor shape/type mismatch: {key}")
        if value.is_floating_point() or value.is_complex():
            flat = value.reshape(-1)
            for chunk in flat.split(4*2**20):
                if not torch.isfinite(chunk).all():
                    raise ValueError(f"Nonfinite checkpoint tensor: {key}")


def load_research_checkpoint(checkpoint, *, cache_dir="cache/vision", reserve_gb=120):
    """Build the trained architecture from pinned metadata and restore its delta."""
    from huggingface_hub import snapshot_download
    import scripts.run_research_stage as runner
    from src.utils.research_checkpoint_compat import load_research_payload
    payload = load_research_payload(Path(checkpoint), map_location="cpu", mmap=True)
    config = dict(payload["config"])
    references = [payload.get("model_ref"), payload.get("vision_ref")]
    if any(not isinstance(ref, dict) or not ref.get("repo") or not re.fullmatch(r"[0-9a-f]{40,64}", str(ref.get("revision", ""))) for ref in references):
        raise ValueError("Evaluation requires pinned model and vision repository revisions.")
    pinned = {ref["repo"]:ref for ref in references}
    if config["model"] not in pinned or config["vision"] not in pinned:
        raise ValueError("Checkpoint model references do not match its architecture config.")
    def resolve_pinned(repo):
        if repo not in pinned:
            raise ValueError("Unexpected unpinned model requested during evaluation.")
        ref = pinned[repo]
        path = snapshot_download(repo, revision=ref["revision"], local_files_only=True,
                                 allow_patterns=["*.json","*.safetensors","*.model","*.jinja","vocab.json","merges.txt"])
        return path, dict(ref)
    config.update(cache_dir=str(cache_dir), reserve_gb=float(reserve_gb))
    args = SimpleNamespace(**config)
    # This process-local injection prevents a later inventory edit from silently
    # changing the base weights underlying an adapter/delta checkpoint.
    with patch.object(runner, "resolve_model", resolve_pinned):
        model = runner.ResearchModel(args)
    _validate_learned_state(model, payload["model"])
    result = model.load_state_dict(payload["model"], strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unrecognized restored checkpoint keys: {result.unexpected_keys[:10]}")
    fingerprint = payload.get("cache_fingerprint")
    if fingerprint is not None and fingerprint != model.cache.fingerprint:
        raise ValueError("Checkpoint and reconstructed vision cache fingerprints differ.")
    provenance = {"model_ref":payload["model_ref"], "vision_ref":payload["vision_ref"],
                  "recipe":config["recipe"], "stage":config["stage"],
                  "training_step":payload["step"], "cache_fingerprint":model.cache.fingerprint}
    del payload
    freeze_for_generation(model)
    return model, provenance
