import gc
import importlib.util
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from transformers.activations import ACT2FN
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
    Qwen2_5_VLForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.utils.quantization_config import QuantizationMethod
from peft import LoraConfig, PeftModel, get_peft_model


IMAGE_TOKEN = "[image_token]"
GIGACHAT_ROLE_SEP = "<|role_sep|>\n"
GIGACHAT_MESSAGE_SEP = "<|message_sep|>\n\n"
IGNORE_INDEX = -100
DEFAULT_PATCH_VISION_IMAGE_SIZE = 1024
DEFAULT_PATCH_VISION_PATCH_SIZE = 32
_MISTRAL_FIXED_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _resolve_torch_device(device: Optional[Any] = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device was requested but CUDA is unavailable: {device}")
        index = resolved.index
        if index is None:
            index = torch.cuda.current_device()
        return torch.device(f"cuda:{index}")

    return resolved


def single_device_map(device: Optional[Any] = None):
    resolved = _resolve_torch_device(device)
    if resolved.type == "cuda":
        return {"": resolved.index if resolved.index is not None else torch.cuda.current_device()}
    return None


def single_device(device: Optional[Any] = None):
    return _resolve_torch_device(device)


def _has_4bit_parameters(module: nn.Module) -> bool:
    for param in module.parameters():
        if param.__class__.__name__ == "Params4bit":
            return True
    return False


def _set_requires_grad_on_tensor_outputs(output: Any) -> Any:
    if torch.is_tensor(output):
        if output.is_floating_point() or output.is_complex():
            output.requires_grad_(True)
        return output
    if isinstance(output, tuple):
        return tuple(_set_requires_grad_on_tensor_outputs(item) for item in output)
    if isinstance(output, list):
        return [_set_requires_grad_on_tensor_outputs(item) for item in output]
    if isinstance(output, dict):
        return {key: _set_requires_grad_on_tensor_outputs(value) for key, value in output.items()}
    return output


def _register_output_require_grads_hook(module: nn.Module) -> None:
    def make_outputs_require_grad(_module, _inputs, output):
        return _set_requires_grad_on_tensor_outputs(output)

    module.register_forward_hook(make_outputs_require_grad)


def _maybe_register_input_require_grads_hook(model: nn.Module) -> bool:
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
            return True
        except NotImplementedError:
            pass

    if hasattr(model, "get_input_embeddings"):
        try:
            input_embeddings = model.get_input_embeddings()
        except NotImplementedError:
            input_embeddings = None
        if input_embeddings is not None:
            _register_output_require_grads_hook(input_embeddings)
            return True

    vision_embedding_paths = [
        "patch_embed",
        "patch_embedder",
        "vision_model.embeddings",
        "embeddings",
    ]
    for path in vision_embedding_paths:
        current = model
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        if isinstance(current, nn.Module):
            _register_output_require_grads_hook(current)
            return True

    return False


def _prepare_model_for_kbit_training_no_fp32_cast(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False

    _maybe_register_input_require_grads_hook(model)

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except TypeError:
            model.gradient_checkpointing_enable()

    return model


@contextmanager
def _without_transformers_allocator_warmup():
    try:
        from transformers import modeling_utils
    except Exception:
        yield
        return

    original = getattr(modeling_utils, "caching_allocator_warmup", None)
    if original is None:
        yield
        return

    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    try:
        yield
    finally:
        modeling_utils.caching_allocator_warmup = original


def _get_by_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            raise AttributeError(path)
        cur = getattr(cur, part)
    return cur


def _first_existing_path(obj: Any, candidates: List[str]) -> Any:
    for path in candidates:
        try:
            return _get_by_path(obj, path)
        except AttributeError:
            pass
    raise AttributeError(f"None of the candidate paths exist: {candidates}")


def _maybe_tensor_to(
    x: Any,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
):
    if torch.is_tensor(x):
        if dtype is not None and x.is_floating_point():
            return x.to(device=device, dtype=dtype)
        return x.to(device=device)
    return x


def _candidate_safetensor_files(model_dir: Path) -> List[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text())
        except Exception:
            index_data = {}
        weight_map = index_data.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        if shard_names:
            return [model_dir / shard_name for shard_name in shard_names]

    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        return [single_file]

    shard_files = sorted(model_dir.glob("model-*.safetensors"))
    if shard_files:
        return shard_files

    raise FileNotFoundError(f"No safetensors weights found under {model_dir}")


def _load_prefixed_safetensor_state_dict(
    model_dir: Path,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}

    for weight_file in _candidate_safetensor_files(model_dir):
        with safe_open(str(weight_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    state_dict[key[len(prefix) :]] = f.get_tensor(key)

    if not state_dict:
        raise KeyError(
            f"No tensors with prefix '{prefix}' were found in {model_dir}"
        )

    return state_dict


def _load_prefixed_safetensor_state_dict_from_candidates(
    model_dir: Path,
    prefixes: List[str],
) -> Dict[str, torch.Tensor]:
    last_error: Optional[Exception] = None
    for prefix in prefixes:
        try:
            return _load_prefixed_safetensor_state_dict(model_dir, prefix)
        except Exception as e:
            last_error = e
    raise RuntimeError(
        f"Could not find any of the prefixes {prefixes} in {model_dir}"
    ) from last_error


def _load_gemma4_vision_modules(
    model_dir: str,
    cfg,
    device: torch.device,
    dtype: torch.dtype,
):
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4MultimodalEmbedder,
        Gemma4VisionModel,
    )

    model_path = Path(model_dir)
    vision_tower = Gemma4VisionModel(cfg.vision_config)
    vision_projector = Gemma4MultimodalEmbedder(cfg.vision_config, cfg.text_config)

    vision_state = _load_prefixed_safetensor_state_dict(
        model_path,
        prefix="model.vision_tower.",
    )
    projector_state = _load_prefixed_safetensor_state_dict(
        model_path,
        prefix="model.embed_vision.",
    )

    vision_tower.load_state_dict(vision_state, strict=True)
    vision_projector.load_state_dict(projector_state, strict=True)

    del vision_state
    del projector_state

    vision_tower.to(device=device, dtype=dtype)
    vision_projector.to(device=device, dtype=dtype)
    return vision_tower, vision_projector


def _load_qwen35_vision_module(
    model_dir: str,
    cfg,
    device: torch.device,
    dtype: torch.dtype,
):
    model_type = getattr(cfg, "model_type", None)
    if model_type == "qwen3_5_moe":
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel

        vision_cls = Qwen3_5MoeVisionModel
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        vision_cls = Qwen3_5VisionModel

    model_path = Path(model_dir)
    vision_tower = vision_cls(cfg.vision_config)
    vision_state = _load_prefixed_safetensor_state_dict_from_candidates(
        model_path,
        prefixes=["model.visual.", "visual."],
    )
    vision_tower.load_state_dict(vision_state, strict=True)
    del vision_state
    vision_tower.to(device=device, dtype=dtype)
    return vision_tower


def _resolve_vision_attn_implementation(attn_implementation: Optional[str]) -> Optional[str]:
    name = str(attn_implementation or "auto").strip().lower()
    if name in {"none", "default"}:
        return None
    if name == "auto":
        if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
            return "flash_attention_2"
        return "sdpa"
    if name in {"eager", "sdpa", "flash_attention_2", "flash_attention_3"}:
        return name
    raise ValueError(
        "Unsupported vision_attn_implementation="
        f"{attn_implementation!r}. Expected 'auto', 'sdpa', 'eager', "
        "'flash_attention_2', 'flash_attention_3', or 'none'."
    )


def _apply_vision_attn_implementation(cfg: Any, attn_implementation: Optional[str]) -> None:
    resolved = _resolve_vision_attn_implementation(attn_implementation)
    if resolved is None:
        return

    for target in (cfg, getattr(cfg, "vision_config", None)):
        if target is not None:
            setattr(target, "_attn_implementation", resolved)


def _load_siglip_vision_module(
    model_dir: str,
    cfg,
    device: torch.device,
    dtype: torch.dtype,
):
    model_type = getattr(cfg, "model_type", None)
    if model_type in {"siglip", "siglip_vision_model"}:
        from transformers import SiglipVisionModel

        vision_cls = SiglipVisionModel
    elif model_type in {"siglip2", "siglip2_vision_model"}:
        from transformers import Siglip2VisionModel

        vision_cls = Siglip2VisionModel
    else:
        raise ValueError(f"Unsupported SigLIP model_type={model_type}")

    vision_tower = vision_cls.from_pretrained(
        model_dir,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    vision_tower.to(device=device, dtype=dtype)
    return vision_tower


def _load_aimv2_vision_module(
    model_dir: str,
    cfg,
    device: torch.device,
    dtype: torch.dtype,
):
    model_type = getattr(cfg, "model_type", None)
    if model_type not in {"aimv2", "aimv2_vision_model"}:
        raise ValueError(f"Unsupported AIMv2 model_type={model_type}")

    from transformers.models.aimv2.modeling_aimv2 import Aimv2Model, Aimv2VisionModel

    if model_type == "aimv2":
        model_path = Path(model_dir)
        if model_path.exists():
            vision_tower = Aimv2VisionModel(cfg.vision_config)
            vision_tower.to(dtype=dtype)
            vision_state = _load_prefixed_safetensor_state_dict_from_candidates(
                model_path,
                prefixes=["vision_model.", "model.vision_model."],
            )
            vision_tower.load_state_dict(vision_state, strict=True)
        else:
            src = Aimv2Model.from_pretrained(
                model_dir,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
            vision_tower = src.vision_model
            if hasattr(src, "text_model"):
                del src.text_model
            if hasattr(src, "visual_projection"):
                del src.visual_projection
            if hasattr(src, "text_projection"):
                del src.text_projection
            del src
    else:
        vision_tower = Aimv2VisionModel.from_pretrained(
            model_dir,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    vision_tower.to(device=device, dtype=dtype)
    return vision_tower


def _toggle_gradient_checkpointing(module: Optional[nn.Module], enabled: bool) -> bool:
    if module is None:
        return False

    toggled = False
    method_name = (
        "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
    )
    method = getattr(module, method_name, None)
    if callable(method):
        try:
            method()
        except TypeError:
            method({})
        toggled = True

    if not toggled:
        for submodule in module.modules():
            if hasattr(submodule, "gradient_checkpointing"):
                try:
                    setattr(submodule, "gradient_checkpointing", enabled)
                    toggled = True
                except Exception:
                    pass

    config = getattr(module, "config", None)
    if config is not None and hasattr(config, "use_cache") and enabled:
        config.use_cache = False

    return toggled


def _normalize_qwen_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image.size}")

    new_width = width
    new_height = height

    if min(width, height) < 56:
        scale = 56.0 / min(width, height)
        new_width = int(round(new_width * scale))
        new_height = int(round(new_height * scale))

    max_ratio = 200.0
    if new_width / new_height >= max_ratio:
        new_height = max(new_height, int(new_width / max_ratio) + 1)
    if new_height / new_width >= max_ratio:
        new_width = max(new_width, int(new_height / max_ratio) + 1)

    new_width = max(56, new_width)
    new_height = max(56, new_height)

    if (new_width, new_height) != image.size:
        image = image.resize((new_width, new_height), Image.BICUBIC)

    return image


def _prepare_single_qwen_image(processor, image: Image.Image) -> Dict[str, torch.Tensor]:
    current = _normalize_qwen_image(image)
    image_token = getattr(processor, "image_token", "<|image_pad|>")

    for _ in range(4):
        image_batch = processor(
            images=[current],
            text=[image_token],
            return_tensors="pt",
            padding=True,
        )
        pixel_values = image_batch["pixel_values"]
        grid = image_batch.get("image_grid_thw", None)
        if grid is None:
            grid = image_batch.get("grid_thw", None)
        if grid is None:
            raise KeyError(
                "Qwen processor did not return image_grid_thw/grid_thw. "
                "Check transformers version."
            )

        if pixel_values.dim() > 2:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])

        expected_rows = int(grid.prod().item())

        if pixel_values.size(0) == expected_rows and pixel_values.size(0) >= 8:
            return {
                "pixel_values": pixel_values,
                "grid": grid,
            }

        current = current.resize(
            (max(56, current.width * 2), max(56, current.height * 2)),
            Image.BICUBIC,
        )

    raise RuntimeError(
        "Qwen vision preprocessing produced inconsistent patch rows "
        f"(got {pixel_values.size(0)}, expected {expected_rows}) "
        f"for image size {image.size} after retries."
    )


def _prepare_qwen_images_batch(
    processor,
    images: List[Image.Image],
) -> Dict[str, torch.Tensor]:
    if not images:
        raise ValueError("images cannot be empty")

    image_token = getattr(processor, "image_token", "<|image_pad|>")
    normalized_images = [_normalize_qwen_image(image) for image in images]

    image_batch = processor(
        images=normalized_images,
        text=[image_token] * len(normalized_images),
        return_tensors="pt",
        padding=True,
    )
    pixel_values = image_batch["pixel_values"]
    grid = image_batch.get("image_grid_thw", None)
    if grid is None:
        grid = image_batch.get("grid_thw", None)
    if grid is None:
        raise KeyError(
            "Qwen processor did not return image_grid_thw/grid_thw. "
            "Check transformers version."
        )

    if pixel_values.dim() == 3:
        expected_rows = [int(x) for x in grid.prod(dim=-1).tolist()]
        if pixel_values.size(0) != len(expected_rows):
            raise RuntimeError(
                "Qwen batched preprocessing returned a 3D tensor whose first "
                f"dimension does not match the grid batch: "
                f"pixel_values_shape={tuple(pixel_values.shape)}, "
                f"grid_shape={tuple(grid.shape)}."
            )
        if any(rows > pixel_values.size(1) for rows in expected_rows):
            raise RuntimeError(
                "Qwen batched preprocessing returned too few padded rows for "
                f"the grid: pixel_values_shape={tuple(pixel_values.shape)}, "
                f"grid={grid.tolist()}."
            )
    elif pixel_values.dim() > 2:
        pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])

    if pixel_values.dim() == 2:
        pixel_values = _pad_qwen_flat_pixel_values(pixel_values, grid)
    elif pixel_values.dim() != 3:
        raise RuntimeError(
            "Qwen batched preprocessing produced unexpected pixel_values shape "
            f"{tuple(pixel_values.shape)} for {len(images)} images."
        )

    return {
        "pixel_values": pixel_values,
        "grid": grid,
    }


def _pad_qwen_flat_pixel_values(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    if pixel_values.dim() != 2:
        raise ValueError(
            "Expected flat Qwen pixel_values with shape (rows, dim), "
            f"got {tuple(pixel_values.shape)}"
        )
    if image_grid_thw.dim() != 2:
        raise ValueError(
            "Expected image_grid_thw with shape (batch, 3), "
            f"got {tuple(image_grid_thw.shape)}"
        )

    row_counts = [int(x) for x in image_grid_thw.prod(dim=-1).tolist()]
    expected_rows = sum(row_counts)
    if pixel_values.size(0) != expected_rows:
        raise RuntimeError(
            "Qwen preprocessing produced inconsistent patch rows "
            f"(got {tuple(pixel_values.shape)}, expected_rows={expected_rows}) "
            f"for grid={image_grid_thw.tolist()}."
        )

    if not row_counts:
        return pixel_values.new_empty((0, 0, pixel_values.size(-1)))

    max_rows = max(row_counts)
    chunks = list(torch.split(pixel_values, row_counts, dim=0))
    padded_chunks = []
    for chunk, rows in zip(chunks, row_counts):
        if rows < max_rows:
            pad = chunk.new_zeros((max_rows - rows, chunk.size(-1)))
            chunk = torch.cat([chunk, pad], dim=0)
        padded_chunks.append(chunk)

    return torch.stack(padded_chunks, dim=0)


def _compact_qwen_padded_pixel_values(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    if pixel_values.dim() != 3:
        raise ValueError(
            "Expected padded Qwen pixel_values with shape (batch, rows, dim), "
            f"got {tuple(pixel_values.shape)}"
        )
    if image_grid_thw.dim() != 2 or image_grid_thw.size(0) != pixel_values.size(0):
        raise ValueError(
            "Expected image_grid_thw with shape (batch, 3), "
            f"got {tuple(image_grid_thw.shape)} for pixel_values={tuple(pixel_values.shape)}"
        )

    chunks = []
    max_rows = pixel_values.size(1)
    for i in range(pixel_values.size(0)):
        rows_i = int(image_grid_thw[i].prod().item())
        if rows_i > max_rows:
            raise RuntimeError(
                "Qwen padded pixel_values has fewer rows than the grid requires. "
                f"sample={i}, rows={rows_i}, max_rows={max_rows}"
            )
        chunks.append(pixel_values[i, :rows_i])

    if not chunks:
        return pixel_values.new_empty((0, pixel_values.size(-1)))
    return torch.cat(chunks, dim=0)


def _token_content(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("content")
    if isinstance(value, list):
        return [_token_content(v) for v in value]
    return value


def _read_tokenizer_chat_template(
    tokenizer_dir: Path,
    tokenizer_config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    tokenizer_config = tokenizer_config or {}
    chat_template = tokenizer_config.get("chat_template")
    if chat_template:
        return chat_template

    chat_template_path = tokenizer_dir / "chat_template.jinja"
    if chat_template_path.exists():
        return chat_template_path.read_text(encoding="utf-8")

    return None


def _set_tokenizer_chat_template(tokenizer, chat_template: Optional[str]) -> None:
    if not chat_template:
        return

    tokenizer.chat_template = chat_template
    if hasattr(tokenizer, "init_kwargs"):
        tokenizer.init_kwargs["chat_template"] = chat_template


def _should_fix_mistral_regex(tokenizer_name: str, tokenizer_config: Dict[str, Any]) -> bool:
    name = str(tokenizer_name).lower()
    if any(x in name for x in ["mistral", "gigachat"]):
        return True

    config_path = Path(tokenizer_name) / "config.json"
    if config_path.exists():
        try:
            model_config = json.loads(config_path.read_text())
        except Exception:
            model_config = {}

        if model_config.get("model_type") in {
            "mistral",
            "mistral3",
            "ministral",
            "pixtral",
            "voxtral",
            "deepseek_v3",
        }:
            return True

    tokenizer_class = str(tokenizer_config.get("tokenizer_class", "")).lower()
    return "mistral" in tokenizer_class


def _maybe_patch_mistral_regex(tokenizer) -> bool:
    try:
        import tokenizers
    except Exception:
        return False

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None or getattr(backend, "pre_tokenizer", None) is None:
        return False

    split_pretokenizer = tokenizers.pre_tokenizers.Split(
        pattern=tokenizers.Regex(_MISTRAL_FIXED_REGEX),
        behavior="isolated",
    )
    current_pretokenizer = backend.pre_tokenizer

    if isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Sequence):
        current_pretokenizer[0] = split_pretokenizer
    else:
        if isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Metaspace):
            current_pretokenizer = tokenizers.pre_tokenizers.ByteLevel(
                add_prefix_space=False,
                use_regex=False,
            )
        backend.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
            [split_pretokenizer, current_pretokenizer]
        )

    setattr(tokenizer, "fix_mistral_regex", True)
    if hasattr(tokenizer, "init_kwargs"):
        tokenizer.init_kwargs["fix_mistral_regex"] = True
    return True


def _preview_loading_keys(keys: Any, limit: int = 10) -> List[str]:
    if keys is None:
        return []
    if isinstance(keys, set):
        keys = sorted(keys)
    elif not isinstance(keys, (list, tuple)):
        keys = list(keys)
    return list(keys[:limit])


def _looks_like_benign_mtp_unexpected_keys(
    unexpected_keys: Any,
    model_config: Any,
) -> bool:
    if not unexpected_keys:
        return False

    keys = list(unexpected_keys)
    num_hidden_layers = getattr(model_config, "num_hidden_layers", None)
    num_nextn_predict_layers = getattr(model_config, "num_nextn_predict_layers", 0)
    if num_hidden_layers is None or not num_nextn_predict_layers:
        return False

    mtp_prefix = f"model.layers.{num_hidden_layers}."
    if not all(k.startswith(mtp_prefix) for k in keys):
        return False

    mtp_markers = (
        ".shared_head.",
        ".eh_proj.",
        ".enorm.",
        ".hnorm.",
        ".embed_tokens.",
    )
    return any(marker in k for k in keys for marker in mtp_markers)


@torch.no_grad()
def _mean_embedding_norm(embedding: nn.Module, chunk_size: int = 4096) -> float:
    weight = getattr(embedding, "weight", None)
    if weight is None:
        return 1.0

    norm_sum = torch.zeros((), device=weight.device, dtype=torch.float32)
    count = 0
    for chunk in weight.detach().split(chunk_size, dim=0):
        norm_sum += chunk.float().norm(dim=-1).sum()
        count += chunk.size(0)

    if count == 0:
        return 1.0
    return float((norm_sum / count).clamp_min(1e-6).item())


def _load_fast_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerFast:
    tokenizer_dir = Path(tokenizer_name)
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    tokenizer_config_file = tokenizer_dir / "tokenizer_config.json"
    special_tokens_map_file = tokenizer_dir / "special_tokens_map.json"

    if tokenizer_file.exists():
        tokenizer_config: Dict[str, Any] = {}
        special_tokens_map: Dict[str, Any] = {}

        if tokenizer_config_file.exists():
            tokenizer_config = json.loads(tokenizer_config_file.read_text())
        if special_tokens_map_file.exists():
            special_tokens_map = json.loads(special_tokens_map_file.read_text())

        bos_token = _token_content(
            special_tokens_map.get("bos_token", tokenizer_config.get("bos_token", "<s>"))
        )
        eos_token = _token_content(
            special_tokens_map.get("eos_token", tokenizer_config.get("eos_token", "</s>"))
        )
        unk_token = _token_content(
            special_tokens_map.get("unk_token", tokenizer_config.get("unk_token"))
        )
        pad_token = _token_content(
            special_tokens_map.get("pad_token", tokenizer_config.get("pad_token"))
        )
        sep_token = _token_content(
            special_tokens_map.get("sep_token", tokenizer_config.get("sep_token"))
        )
        cls_token = _token_content(
            special_tokens_map.get("cls_token", tokenizer_config.get("cls_token"))
        )
        mask_token = _token_content(
            special_tokens_map.get("mask_token", tokenizer_config.get("mask_token"))
        )
        additional_special_tokens = _token_content(
            special_tokens_map.get(
                "additional_special_tokens",
                tokenizer_config.get("additional_special_tokens", []),
            )
        )

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token or eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            model_max_length=tokenizer_config.get("model_max_length"),
            clean_up_tokenization_spaces=tokenizer_config.get(
                "clean_up_tokenization_spaces", True
            ),
        )

        chat_template = _read_tokenizer_chat_template(tokenizer_dir, tokenizer_config)
        _set_tokenizer_chat_template(tokenizer, chat_template)

        tokenizer.padding_side = tokenizer_config.get("padding_side", "right")
        tokenizer.truncation_side = tokenizer_config.get("truncation_side", "right")

        if _should_fix_mistral_regex(tokenizer_name, tokenizer_config):
            _maybe_patch_mistral_regex(tokenizer)
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
        if tokenizer_dir.exists():
            _set_tokenizer_chat_template(
                tokenizer,
                _read_tokenizer_chat_template(tokenizer_dir),
            )

    tokenizer.padding_side = getattr(tokenizer, "padding_side", "right") or "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _build_multimodal_user_text(text: str, num_images: int = 1) -> str:
    if num_images < 0:
        raise ValueError(f"num_images must be >= 0, got {num_images}")

    image_prefix = ""
    if num_images > 0:
        image_prefix = "\n".join([IMAGE_TOKEN] * num_images)

    return text if not image_prefix else f"{image_prefix}\n{text}"


def _normalize_chat_template_mode(chat_template_mode: Any) -> str:
    if isinstance(chat_template_mode, bool):
        return "tokenizer" if chat_template_mode else "short"

    mode = str(chat_template_mode or "tokenizer").lower()
    aliases = {
        "auto": "tokenizer",
        "hf": "tokenizer",
        "default": "tokenizer",
        "full": "tokenizer",
        "tokenizer": "tokenizer",
        "compact": "short",
        "minimal": "short",
        "short": "short",
        "plain": "plain",
        "legacy": "plain",
    }
    if mode not in aliases:
        raise ValueError(
            "Unsupported chat_template_mode. Expected one of "
            "'tokenizer', 'short', or 'plain', got "
            f"{chat_template_mode!r}."
        )
    return aliases[mode]


def _build_short_gigachat_prompt(tokenizer, user_text: str) -> str:
    bos_token = getattr(tokenizer, "bos_token", None) or ""
    return (
        f"{bos_token}"
        f"user{GIGACHAT_ROLE_SEP}{user_text}{GIGACHAT_MESSAGE_SEP}"
        f"assistant{GIGACHAT_ROLE_SEP}"
    )


def _build_chat_prompt(
    tokenizer,
    text: str,
    num_images: int = 1,
    chat_template_mode: str = "tokenizer",
) -> str:
    user_text = _build_multimodal_user_text(text, num_images=num_images)
    mode = _normalize_chat_template_mode(chat_template_mode)

    if mode == "short":
        return _build_short_gigachat_prompt(tokenizer, user_text)

    if mode == "plain":
        return f"User: {user_text}\nAssistant:"

    if getattr(tokenizer, "chat_template", None):
        try:
            messages = [{"role": "user", "content": user_text}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return f"User: {user_text}\nAssistant:"


def _build_chat_training_texts(
    tokenizer,
    question: str,
    answer: str,
    num_images: int = 1,
    chat_template_mode: str = "tokenizer",
) -> tuple[str, str]:
    user_text = _build_multimodal_user_text(question, num_images=num_images)
    mode = _normalize_chat_template_mode(chat_template_mode)

    if mode == "short":
        prompt = _build_short_gigachat_prompt(tokenizer, user_text)
        return prompt, prompt + str(answer) + GIGACHAT_MESSAGE_SEP

    if mode == "plain":
        prompt = f"User: {user_text}\nAssistant:"
        eos_token = tokenizer.eos_token or ""
        return prompt, prompt + str(answer) + eos_token

    if getattr(tokenizer, "chat_template", None):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": str(answer)},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            return prompt, full_text
        except Exception:
            pass

    prompt = f"User: {user_text}\nAssistant:"
    eos_token = tokenizer.eos_token or ""
    return prompt, prompt + str(answer) + eos_token


def _single_token_id(tokenizer, text: str) -> Optional[int]:
    token_id = tokenizer.convert_tokens_to_ids(text)
    if isinstance(token_id, int) and token_id >= 0:
        return int(token_id)

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) == 1:
        return int(ids[0])

    return None


def _chat_stop_token_ids(tokenizer) -> List[int]:
    stop_ids = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        stop_ids.append(int(eos_token_id))

    for token in (
        "<|message_sep|>\n\n",
        "<|message_sep|>\n",
        "<|message_sep|>",
    ):
        token_id = _single_token_id(tokenizer, token)
        if token_id is not None:
            stop_ids.append(token_id)

    deduped = []
    seen = set()
    for token_id in stop_ids:
        if token_id in seen:
            continue
        seen.add(token_id)
        deduped.append(token_id)
    return deduped


class _RepeatedNGramStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        *,
        start_length: int,
        ngram_size: int = 8,
        max_occurrences: int = 3,
    ):
        self.start_length = int(start_length)
        self.ngram_size = int(ngram_size)
        self.max_occurrences = int(max_occurrences)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: Optional[torch.FloatTensor],
        **kwargs,
    ) -> bool:
        if self.ngram_size <= 0 or self.max_occurrences <= 1:
            return False

        for row in input_ids:
            generated = row[self.start_length :]
            if generated.numel() < self.ngram_size * self.max_occurrences:
                continue

            tail = tuple(generated[-self.ngram_size :].tolist())
            occurrences = 0
            for idx in range(0, generated.numel() - self.ngram_size + 1):
                if tuple(generated[idx : idx + self.ngram_size].tolist()) == tail:
                    occurrences += 1
                    if occurrences >= self.max_occurrences:
                        return True

        return False


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(module.parameters())
    return param.device, param.dtype


def _clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _replace_linear_layers_with_bnb_4bit(
    module: nn.Module,
    compute_dtype: torch.dtype,
    device: torch.device,
    exclude_module_names: Optional[List[str]] = None,
    module_prefix: str = "",
) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit vision QLoRA requires CUDA.")

    try:
        import bitsandbytes as bnb
    except Exception as e:
        raise RuntimeError(
            "bitsandbytes is required for 4-bit vision QLoRA."
        ) from e

    exclude_module_names = exclude_module_names or []
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{module_prefix}.{child_name}" if module_prefix else child_name
        if any(excluded in full_name for excluded in exclude_module_names):
            continue

        if isinstance(child, nn.Linear):
            quantized = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                quant_type="nf4",
                compress_statistics=True,
            )
            quantized.load_state_dict(
                {key: value.detach().cpu() for key, value in child.state_dict().items()},
                strict=True,
            )
            quantized.requires_grad_(False)
            quantized.to(device)
            setattr(module, child_name, quantized)
            del child
            replaced += 1
        else:
            replaced += _replace_linear_layers_with_bnb_4bit(
                child,
                compute_dtype=compute_dtype,
                device=device,
                exclude_module_names=exclude_module_names,
                module_prefix=full_name,
            )
    return replaced


def _pil_to_normalized_tensor(
    image: Image.Image,
    max_side_size: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")

    image = _resize_pil_image_max_side(image, max_side_size)
    width, height = image.size

    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.reshape(height, width, 3).permute(2, 0, 1)
    tensor = data.to(dtype=torch.float32).div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _resize_pil_image_max_side(
    image: Image.Image,
    max_side_size: Optional[int] = None,
) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")

    image = image.convert("RGB")
    if max_side_size is None:
        return image

    max_side_size = int(max_side_size)
    if max_side_size <= 0:
        raise ValueError(f"max_side_size must be > 0, got {max_side_size}")

    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_side_size:
        return image

    scale = float(max_side_size) / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.BICUBIC)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _resolve_dtype_choice(
    dtype_name: str,
    default_dtype: torch.dtype,
) -> torch.dtype:
    name = str(dtype_name or "auto").lower()
    aliases = {
        "auto": default_dtype,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in aliases:
        raise ValueError(
            "Unsupported dtype choice "
            f"{dtype_name!r}. Expected one of: 'auto', 'bf16', 'fp16', 'fp32'."
        )
    return aliases[name]


def _resolve_causal_lm_backbone(model: nn.Module) -> nn.Module:
    return _first_existing_path(
        model,
        [
            "base_model.model.model",
            "base_model.model.transformer",
            "base_model.model.gpt_neox",
            "base_model.model.backbone",
            "model",
            "transformer",
            "gpt_neox",
            "backbone",
        ],
    )


def _clone_shared_tensors_in_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Clone duplicate tensor storages so generic safetensors saving can proceed."""
    seen_storages = set()
    for key, value in list(state_dict.items()):
        if not isinstance(value, torch.Tensor):
            continue
        try:
            storage = value.untyped_storage()
            storage_key = (
                value.device.type,
                value.device.index,
                storage.data_ptr(),
                storage.nbytes(),
            )
        except Exception:
            continue

        if storage_key in seen_storages:
            state_dict[key] = value.clone()
        else:
            seen_storages.add(storage_key)
    return state_dict


class VisualTokenExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, initializer_range: float):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=initializer_range)
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class VLExpertMLPWrapper(nn.Module):
    def __init__(self, base_mlp: nn.Module, config: Any):
        super().__init__()
        self.base_mlp = base_mlp
        hidden_size = getattr(config, "hidden_size")
        intermediate_size = getattr(config, "moe_intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = getattr(base_mlp, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = getattr(config, "intermediate_size")
        self.visual_expert = VisualTokenExpert(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            initializer_range=getattr(config, "initializer_range", 0.02),
        )
        self.visual_token_mask: Optional[torch.Tensor] = None

    def _resolve_mask(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        if self.visual_token_mask is None:
            return None

        expected_shape = hidden_states.shape[:-1]
        mask = self.visual_token_mask
        if tuple(mask.shape) == tuple(expected_shape):
            return mask.to(device=hidden_states.device, dtype=torch.bool)

        if mask.numel() == hidden_states.numel() // hidden_states.size(-1):
            return mask.reshape(expected_shape).to(device=hidden_states.device, dtype=torch.bool)

        return None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_out = self.base_mlp(hidden_states)
        mask = self._resolve_mask(hidden_states)
        if mask is None or not torch.any(mask):
            return base_out

        flat_mask = mask.reshape(-1)
        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        token_indices = flat_mask.nonzero(as_tuple=False).flatten()
        selected_hidden = flat_hidden.index_select(0, token_indices)

        expert_device, expert_dtype = _module_device_dtype(self.visual_expert)
        selected_hidden = selected_hidden.to(device=expert_device, dtype=expert_dtype)
        selected_delta = self.visual_expert(selected_hidden).to(
            device=base_out.device,
            dtype=base_out.dtype,
        )

        flat_delta = torch.zeros(
            flat_hidden.size(0),
            flat_hidden.size(1),
            device=base_out.device,
            dtype=base_out.dtype,
        )
        flat_delta.index_copy_(0, token_indices.to(base_out.device), selected_delta)
        return base_out + flat_delta.reshape_as(base_out)


class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QFormerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        mlp_hidden = hidden_size * mlp_ratio

        self.self_attn_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

        self.cross_attn_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_kv_norm = nn.LayerNorm(hidden_size)

        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(
        self,
        queries: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        x = self.self_attn_norm(queries)
        self_attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        queries = queries + self_attn_out

        q = self.cross_attn_norm(queries)
        kv = self.cross_kv_norm(encoder_hidden_states)
        cross_attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        queries = queries + cross_attn_out

        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries


class QFormerProjector(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_queries: int = 32,
        num_heads: int = 8,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_size = llm_dim

        self.vision_proj = nn.Linear(vision_dim, llm_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, llm_dim) * 0.02)
        self.block = QFormerBlock(
            hidden_size=llm_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.output_norm = nn.LayerNorm(llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        image_features = self.vision_proj(x)
        queries = self.query_tokens.expand(image_features.size(0), -1, -1)
        queries = self.block(queries, image_features)
        queries = self.output_norm(queries)

        if squeeze:
            return queries[0]
        return queries


class PatchLLMVisionEncoder(nn.Module):
    def __init__(
        self,
        vision_llm_name: str,
        image_size: int = DEFAULT_PATCH_VISION_IMAGE_SIZE,
        patch_size: int = DEFAULT_PATCH_VISION_PATCH_SIZE,
        conv_hidden_size: int = 256,
        use_maxpool: bool = False,
        use_qlora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_path: Optional[str] = None,
        freeze: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {image_size}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {patch_size}")

        base_patch_grid_size = max(1, _ceil_div(image_size, patch_size))
        if use_maxpool and base_patch_grid_size % 2 != 0:
            base_patch_grid_size += 1

        self.vision_llm_name = vision_llm_name
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.conv_hidden_size = int(conv_hidden_size)
        self.use_maxpool = bool(use_maxpool)
        self.use_qlora = bool(use_qlora)
        self.freeze = bool(freeze)
        self.base_grid_size = (
            base_patch_grid_size // 2 if self.use_maxpool else base_patch_grid_size
        )
        self.num_base_tokens = self.base_grid_size * self.base_grid_size

        device = device or single_device()
        dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
        self.dtype = dtype

        vision_quant_config = None
        vision_device_map = None
        if self.use_qlora and device.type == "cuda":
            vision_quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            vision_device_map = single_device_map(device)

        self.vision_llm = AutoModelForCausalLM.from_pretrained(
            vision_llm_name,
            dtype=dtype,
            quantization_config=vision_quant_config,
            device_map=vision_device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if hasattr(self.vision_llm.config, "use_cache"):
            self.vision_llm.config.use_cache = False

        if vision_quant_config is not None:
            if (
                not getattr(self.vision_llm, "is_loaded_in_4bit", False)
                and _has_4bit_parameters(self.vision_llm)
            ):
                self.vision_llm.is_loaded_in_4bit = True
            self.vision_llm = _prepare_model_for_kbit_training_no_fp32_cast(self.vision_llm)
        elif not self.use_qlora:
            self.vision_llm.to(device=device, dtype=dtype)

        self.hidden_size = getattr(self.vision_llm.config, "hidden_size", None)
        if self.hidden_size is None:
            self.hidden_size = getattr(self.vision_llm.config, "n_embd", None)
        if self.hidden_size is None:
            raise RuntimeError(
                f"Could not infer hidden size from vision LLM {vision_llm_name}."
            )

        if self.use_qlora or lora_path is not None:
            if lora_path is not None:
                self.vision_llm = PeftModel.from_pretrained(
                    self.vision_llm,
                    lora_path,
                    is_trainable=not self.freeze,
                )
            else:
                target_modules = _resolve_lora_target_modules(
                    self.vision_llm,
                    num_lora_layers=-1,
                    exclude_substrings=["lm_head"],
                )
                lora_cfg = LoraConfig(
                    task_type="CAUSAL_LM",
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    target_modules=target_modules,
                )
                self.vision_llm = get_peft_model(self.vision_llm, lora_cfg)

        backbone = self.vision_llm_backbone
        backbone_config = getattr(backbone, "config", None)
        if backbone_config is not None and hasattr(backbone_config, "use_cache"):
            backbone_config.use_cache = False

        conv_layers: List[nn.Module] = [
            nn.Conv2d(
                3,
                self.conv_hidden_size,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.conv_hidden_size,
                self.conv_hidden_size,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.conv_hidden_size,
                self.conv_hidden_size,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GELU(),
        ]
        if self.use_maxpool:
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv = nn.Sequential(*conv_layers)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_base_tokens, self.conv_hidden_size)
        )
        self.to_llm = nn.Sequential(
            nn.LayerNorm(self.conv_hidden_size),
            nn.Linear(self.conv_hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        self.output_norm = nn.LayerNorm(self.hidden_size)

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        self.conv.to(device=device, dtype=dtype)
        self.pos_embed.data = self.pos_embed.data.to(device=device, dtype=dtype)
        self.to_llm.to(device=device, dtype=dtype)
        self.output_norm.to(device=device, dtype=dtype)
        self.set_trainable(not self.freeze)

    @property
    def vision_llm_backbone(self) -> nn.Module:
        return _resolve_causal_lm_backbone(self.vision_llm)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.vision_llm.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None):
        method = getattr(self.vision_llm, "gradient_checkpointing_enable", None)
        if callable(method):
            try:
                if gradient_checkpointing_kwargs is None:
                    method()
                else:
                    method(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except TypeError:
                method()

        config = getattr(self.vision_llm, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False

    def gradient_checkpointing_disable(self):
        method = getattr(self.vision_llm, "gradient_checkpointing_disable", None)
        if callable(method):
            method()

    def set_trainable(self, trainable: bool) -> None:
        self.freeze = not trainable
        self.conv.requires_grad_(trainable)
        self.pos_embed.requires_grad_(trainable)
        self.to_llm.requires_grad_(trainable)
        self.output_norm.requires_grad_(trainable)

        if self.use_qlora or isinstance(self.vision_llm, PeftModel):
            for name, param in self.vision_llm.named_parameters():
                is_adapter_param = "lora_" in name or "modules_to_save" in name
                param.requires_grad = bool(trainable and is_adapter_param)
        else:
            self.vision_llm.requires_grad_(trainable)

        if not trainable:
            self.vision_llm.eval()

    def non_llm_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict = {"pos_embed": self.pos_embed.detach().cpu()}
        for prefix, module in [
            ("conv", self.conv),
            ("to_llm", self.to_llm),
            ("output_norm", self.output_norm),
        ]:
            for key, value in module.state_dict().items():
                state_dict[f"{prefix}.{key}"] = value.detach().cpu()
        return state_dict

    def load_non_llm_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        expected_keys = set(self.non_llm_state_dict().keys())
        provided_keys = set(state_dict.keys())
        missing_keys = sorted(expected_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_keys)
        if missing_keys:
            raise RuntimeError(f"Missing patch-LLM vision encoder weights: {missing_keys}")
        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected patch-LLM vision encoder weights: {unexpected_keys}"
            )

        pos_embed = state_dict["pos_embed"]
        if tuple(pos_embed.shape) != tuple(self.pos_embed.shape):
            raise RuntimeError(
                "Patch-LLM position embedding shape mismatch. "
                f"checkpoint={tuple(pos_embed.shape)}, current={tuple(self.pos_embed.shape)}"
            )
        self.pos_embed.data.copy_(pos_embed.to(
            device=self.pos_embed.device,
            dtype=self.pos_embed.dtype,
        ))

        for prefix, module in [
            ("conv", self.conv),
            ("to_llm", self.to_llm),
            ("output_norm", self.output_norm),
        ]:
            module_state = {
                key[len(prefix) + 1 :]: value
                for key, value in state_dict.items()
                if key.startswith(f"{prefix}.")
            }
            module.load_state_dict(module_state, strict=True)

    def _pad_pixel_values_to_patch_grid(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        patch_grid_h = max(1, _ceil_div(int(height), self.patch_size))
        patch_grid_w = max(1, _ceil_div(int(width), self.patch_size))

        if self.use_maxpool:
            if patch_grid_h % 2 != 0:
                patch_grid_h += 1
            if patch_grid_w % 2 != 0:
                patch_grid_w += 1

        target_h = patch_grid_h * self.patch_size
        target_w = patch_grid_w * self.patch_size
        pad_h = target_h - height
        pad_w = target_w - width
        if pad_h > 0 or pad_w > 0:
            pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h), value=0.0)
        return pixel_values

    def _position_embeddings_for_grid(
        self,
        grid_h: int,
        grid_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if grid_h <= 0 or grid_w <= 0:
            raise RuntimeError(f"Invalid Patch-LLM feature grid: {(grid_h, grid_w)}")

        pos = self.pos_embed.reshape(
            1,
            self.base_grid_size,
            self.base_grid_size,
            self.conv_hidden_size,
        ).permute(0, 3, 1, 2)

        if grid_h != self.base_grid_size or grid_w != self.base_grid_size:
            pos = F.interpolate(
                pos.float(),
                size=(grid_h, grid_w),
                mode="bicubic",
                align_corners=False,
            ).to(dtype=dtype)
        else:
            pos = pos.to(dtype=dtype)

        return pos.to(device=device).permute(0, 2, 3, 1).reshape(
            1,
            grid_h * grid_w,
            self.conv_hidden_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        conv_device, conv_dtype = _module_device_dtype(self.conv)
        pixel_values = pixel_values.to(device=conv_device, dtype=conv_dtype)
        pixel_values = self._pad_pixel_values_to_patch_grid(pixel_values)

        features = self.conv(pixel_values)
        grid_h, grid_w = int(features.size(-2)), int(features.size(-1))
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self._position_embeddings_for_grid(
            grid_h=grid_h,
            grid_w=grid_w,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        inputs_embeds = self.to_llm(tokens)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device,
        )

        if self.freeze:
            self.vision_llm.eval()

        try:
            outputs = self.vision_llm_backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
        except TypeError:
            outputs = self.vision_llm_backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs[0]
        return self.output_norm(hidden_states)


class LLMProjector(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        connector_llm_name: str,
        freeze_connector_llm: bool = True,
        connector_llm_use_qlora: bool = False,
        connector_lora_r: int = 16,
        connector_lora_alpha: int = 32,
        connector_lora_dropout: float = 0.05,
        connector_llm_lora_path: Optional[str] = None,
        connector_dtype: Optional[torch.dtype] = None,
        connector_device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.connector_llm_name = connector_llm_name
        self.freeze_connector_llm = freeze_connector_llm
        self.connector_llm_use_qlora = bool(
            connector_llm_use_qlora or connector_llm_lora_path is not None
        )
        self.connector_lora_r = int(connector_lora_r)
        self.connector_lora_alpha = int(connector_lora_alpha)
        self.connector_lora_dropout = float(connector_lora_dropout)
        self.connector_llm_lora_path = connector_llm_lora_path
        connector_device = connector_device or single_device()

        connector_quant_config = None
        connector_device_map = None
        if self.connector_llm_use_qlora and connector_device.type == "cuda":
            connector_quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=connector_dtype or torch.bfloat16,
            )
            connector_device_map = single_device_map(connector_device)
        self.connector_is_quantized = connector_quant_config is not None

        self.connector_llm = AutoModelForCausalLM.from_pretrained(
            connector_llm_name,
            dtype=connector_dtype,
            quantization_config=connector_quant_config,
            device_map=connector_device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.connector_config = self.connector_llm.config
        if hasattr(self.connector_config, "use_cache"):
            self.connector_config.use_cache = False

        if connector_quant_config is not None:
            if (
                not getattr(self.connector_llm, "is_loaded_in_4bit", False)
                and _has_4bit_parameters(self.connector_llm)
            ):
                self.connector_llm.is_loaded_in_4bit = True
            self.connector_llm = _prepare_model_for_kbit_training_no_fp32_cast(
                self.connector_llm
            )

        if connector_llm_lora_path is not None:
            self.connector_llm = PeftModel.from_pretrained(
                self.connector_llm,
                connector_llm_lora_path,
                is_trainable=not self.freeze_connector_llm,
            )
        elif self.connector_llm_use_qlora:
            target_modules = _resolve_lora_target_modules(
                self.connector_llm,
                num_lora_layers=-1,
                exclude_substrings=["lm_head"],
            )
            lora_cfg = LoraConfig(
                task_type="CAUSAL_LM",
                r=self.connector_lora_r,
                lora_alpha=self.connector_lora_alpha,
                lora_dropout=self.connector_lora_dropout,
                bias="none",
                target_modules=target_modules,
            )
            self.connector_llm = get_peft_model(self.connector_llm, lora_cfg)

        connector_model_config = getattr(self.connector_model, "config", None)
        if connector_model_config is not None and hasattr(connector_model_config, "use_cache"):
            connector_model_config.use_cache = False
        connector_hidden_size = getattr(self.connector_config, "hidden_size", None)
        if connector_hidden_size is None:
            connector_hidden_size = getattr(self.connector_config, "n_embd", None)
        if connector_hidden_size is None:
            raise RuntimeError(
                f"Could not infer connector hidden size from {connector_llm_name}."
            )

        self.input_norm = nn.LayerNorm(vision_dim)
        self.text_in_proj = nn.Linear(llm_dim, connector_hidden_size)
        self.vision_in_proj = nn.Linear(vision_dim, connector_hidden_size)
        self.output_norm = nn.LayerNorm(connector_hidden_size)
        self.llm_out_proj = nn.Linear(connector_hidden_size, llm_dim)

        self.set_connector_trainable(not self.freeze_connector_llm)

    @property
    def connector_model(self) -> nn.Module:
        return _first_existing_path(
            self.connector_llm,
            [
                "base_model.model.model",
                "base_model.model.transformer",
                "base_model.model.gpt_neox",
                "base_model.model.backbone",
                "model",
                "transformer",
                "gpt_neox",
                "backbone",
            ],
        )

    def move_projection_layers(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        for module in [
            self.input_norm,
            self.text_in_proj,
            self.vision_in_proj,
            self.output_norm,
            self.llm_out_proj,
        ]:
            module.to(device=device, dtype=dtype)

        if not self.connector_is_quantized:
            self.connector_llm.to(device=device, dtype=dtype)

    def set_connector_trainable(self, trainable: bool) -> None:
        if self.connector_llm_use_qlora or isinstance(self.connector_llm, PeftModel):
            for name, param in self.connector_llm.named_parameters():
                is_adapter_param = "lora_" in name or "modules_to_save" in name
                param.requires_grad = bool(trainable and is_adapter_param)
        else:
            self.connector_llm.requires_grad_(trainable)

        if not trainable:
            self.connector_llm.eval()
        self._freeze_unused_connector_token_layers()

    def _freeze_unused_connector_token_layers(self) -> None:
        """The connector receives inputs_embeds and never predicts connector tokens."""
        input_embeddings = None
        if hasattr(self.connector_llm, "get_input_embeddings"):
            try:
                input_embeddings = self.connector_llm.get_input_embeddings()
            except Exception:
                input_embeddings = None

        output_embeddings = None
        if hasattr(self.connector_llm, "get_output_embeddings"):
            try:
                output_embeddings = self.connector_llm.get_output_embeddings()
            except Exception:
                output_embeddings = None

        if isinstance(input_embeddings, nn.Module):
            input_embeddings.requires_grad_(False)
        if isinstance(output_embeddings, nn.Module):
            output_embeddings.requires_grad_(False)

        lm_head = getattr(self.connector_llm, "lm_head", None)
        if isinstance(lm_head, nn.Module) and lm_head is not output_embeddings:
            lm_head.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_connector_llm:
            self.connector_llm.eval()
        return self

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Enable gradient checkpointing on the connector LLM when it is trainable."""
        if not self.freeze_connector_llm:
            _toggle_gradient_checkpointing(self.connector_llm, enabled=True)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on the connector LLM when it is trainable."""
        if not self.freeze_connector_llm:
            _toggle_gradient_checkpointing(self.connector_llm, enabled=False)

    def forward(
        self,
        x: torch.Tensor,
        text_prefix_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        if x.dim() != 3:
            raise RuntimeError(
                "LLMProjector expects vision features with shape "
                f"(tokens, dim) or (batch, tokens, dim), got {tuple(x.shape)}."
            )

        vision_inputs = self.vision_in_proj(self.input_norm(x))
        connector_inputs = vision_inputs
        prefix_len = 0

        if text_prefix_embeds is not None:
            if text_prefix_embeds.dim() == 2:
                text_prefix_embeds = text_prefix_embeds.unsqueeze(0)
            if text_prefix_embeds.dim() != 3:
                raise RuntimeError(
                    "text_prefix_embeds must have shape "
                    f"(tokens, dim) or (batch, tokens, dim), got "
                    f"{tuple(text_prefix_embeds.shape)}."
                )
            if text_prefix_embeds.size(0) != x.size(0):
                raise RuntimeError(
                    "text_prefix_embeds batch size must match vision features. "
                    f"text_batch={text_prefix_embeds.size(0)}, vision_batch={x.size(0)}"
                )

            text_prefix_embeds = text_prefix_embeds.to(
                device=vision_inputs.device,
                dtype=vision_inputs.dtype,
            )
            text_inputs = self.text_in_proj(text_prefix_embeds)
            prefix_len = text_inputs.size(1)
            connector_inputs = torch.cat([text_inputs, vision_inputs], dim=1)

        attention_mask = torch.ones(
            connector_inputs.shape[:2],
            dtype=torch.long,
            device=connector_inputs.device,
        )

        if self.freeze_connector_llm:
            self.connector_llm.eval()

        try:
            outputs = self.connector_model(
                inputs_embeds=connector_inputs,
                attention_mask=attention_mask,
                use_cache=False,
            )
        except TypeError:
            outputs = self.connector_model(
                inputs_embeds=connector_inputs,
                attention_mask=attention_mask,
            )

        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs[0]

        if prefix_len:
            hidden_states = hidden_states[:, prefix_len:, :]

        y = self.llm_out_proj(self.output_norm(hidden_states))
        if squeeze:
            return y[0]
        return y

    def connector_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict = {}
        for prefix, module in [
            ("input_norm", self.input_norm),
            ("text_in_proj", self.text_in_proj),
            ("vision_in_proj", self.vision_in_proj),
            ("output_norm", self.output_norm),
            ("llm_out_proj", self.llm_out_proj),
        ]:
            for key, value in module.state_dict().items():
                state_dict[f"{prefix}.{key}"] = value.detach().cpu()

        if not self.freeze_connector_llm and not self.connector_llm_use_qlora:
            for key, value in self.connector_model.state_dict().items():
                state_dict[f"connector_model.{key}"] = value.detach().cpu()
        return state_dict

    def load_connector_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        connector_model_state = {
            key[len("connector_model.") :]: value
            for key, value in state_dict.items()
            if key.startswith("connector_model.")
        }
        projector_state = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("connector_model.")
        }

        missing_keys, unexpected_keys = self.load_state_dict(projector_state, strict=False)
        if connector_model_state:
            self.connector_model.load_state_dict(connector_model_state, strict=True)

        required_keys = {
            "input_norm.weight",
            "input_norm.bias",
            "vision_in_proj.weight",
            "vision_in_proj.bias",
            "output_norm.weight",
            "output_norm.bias",
            "llm_out_proj.weight",
            "llm_out_proj.bias",
        }
        missing_required = sorted(key for key in missing_keys if key in required_keys)
        if missing_required:
            raise RuntimeError(f"Missing LLM projector weights: {missing_required}")
        unexpected_keys = [
            key for key in unexpected_keys if not key.startswith("connector_llm.")
        ]
        if unexpected_keys:
            raise RuntimeError(f"Unexpected LLM projector weights: {unexpected_keys}")


def _infer_projector_type_from_state_dict(state_dict: Dict[str, Any]) -> Optional[str]:
    keys = set(state_dict.keys())
    if any(key.startswith("net.") for key in keys):
        return "mlp"
    if "vision_proj.weight" in keys or "query_tokens" in keys:
        return "qformer"
    if (
        "vision_in_proj.weight" in keys
        or "text_in_proj.weight" in keys
        or "llm_out_proj.weight" in keys
    ):
        return "llm"
    return None


def _resolve_checkpoint_sidecar_path(
    checkpoint_path: Path,
    relative_name: str,
) -> Optional[Path]:
    candidates = [
        checkpoint_path / relative_name,
        checkpoint_path.parent / relative_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_lora_layers_to_transform(
    num_hidden_layers: Optional[int],
    num_lora_layers: int,
) -> Optional[List[int]]:
    if num_lora_layers == -1:
        return None

    if num_hidden_layers is None:
        raise RuntimeError(
            "Could not infer `num_hidden_layers` from the base LLM config, "
            "so LoRA layer restriction is unavailable."
        )

    if num_lora_layers < 1:
        raise ValueError(
            f"num_lora_layers must be -1 or >= 1, got {num_lora_layers}"
        )

    if num_lora_layers > num_hidden_layers:
        raise ValueError(
            f"num_lora_layers={num_lora_layers} exceeds num_hidden_layers={num_hidden_layers}"
        )

    return list(range(num_lora_layers))


def _resolve_lora_target_modules(
    model: nn.Module,
    num_lora_layers: int,
    exclude_substrings: Optional[List[str]] = None,
) -> Any:
    exclude_substrings = exclude_substrings or []

    if num_lora_layers == -1 and not exclude_substrings:
        return "all-linear"

    try:
        from transformers.pytorch_utils import Conv1D
        linear_classes = (nn.Linear, Conv1D)
    except Exception:
        linear_classes = (nn.Linear,)

    linear_module_names = set()
    for name, module in model.named_modules():
        if any(excluded in name for excluded in exclude_substrings):
            continue
        if isinstance(module, linear_classes):
            linear_module_names.add(name)

    if hasattr(model, "get_output_embeddings"):
        try:
            output_emb = model.get_output_embeddings()
        except Exception:
            output_emb = None
        if output_emb is not None:
            for name, module in model.named_modules():
                if module is output_emb:
                    linear_module_names.discard(name)
                    break

    if not linear_module_names:
        raise RuntimeError("Could not resolve any linear target modules for LoRA.")

    return linear_module_names


def _resolve_llm_decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidates = []
    if hasattr(model, "get_base_model"):
        try:
            candidates.append(model.get_base_model())
        except Exception:
            pass
    candidates.append(model)
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        nested = getattr(base_model, "model", None)
        if nested is not None:
            candidates.append(nested)

    for candidate in candidates:
        for path in ["model.layers", "layers"]:
            try:
                layers = _get_by_path(candidate, path)
            except AttributeError:
                continue
            if isinstance(layers, nn.ModuleList):
                return layers

    raise RuntimeError("Could not locate decoder layers in the LLM.")


def _resolve_vl_expert_layer_indices(
    num_hidden_layers: Optional[int],
    num_lora_layers: int,
    vl_expert_layers: Optional[List[int]] = None,
) -> List[int]:
    if num_hidden_layers is None:
        raise RuntimeError(
            "Could not infer `num_hidden_layers` from the base LLM config, "
            "so VL expert layer selection is unavailable."
        )

    if vl_expert_layers is not None:
        normalized_layers = []
        seen_layers = set()
        for layer_idx in vl_expert_layers:
            layer_idx = int(layer_idx)
            if layer_idx < 0 or layer_idx >= num_hidden_layers:
                raise ValueError(
                    f"VL expert layer index {layer_idx} is out of range for "
                    f"num_hidden_layers={num_hidden_layers}."
                )
            if layer_idx not in seen_layers:
                normalized_layers.append(layer_idx)
                seen_layers.add(layer_idx)
        return normalized_layers

    if num_lora_layers == -1:
        return list(range(num_hidden_layers))

    return _resolve_lora_layers_to_transform(
        num_hidden_layers=num_hidden_layers,
        num_lora_layers=num_lora_layers,
    ) or []


class GigaChatVL(nn.Module):
    """Multimodal wrapper that connects a donor vision encoder to GigaChat.

    The model keeps the base causal LM as the text decoder, encodes images with
    one of the supported vision backends, projects visual features into the LLM
    embedding space, and replaces ``[image_token]`` placeholders with those
    projected visual tokens. The class supports projector-only alignment,
    LLM LoRA/QLoRA-style adaptation, optional donor-vision LoRA/QLoRA, and
    optional visual experts for selected LLM layers.

    Args:
        llm_name: Hugging Face id or local path for the base GigaChat/LLM
            checkpoint.
        vision_name: Hugging Face id or local path for the donor vision model
            or patch-LLM visual encoder.
        tokenizer_name: Optional tokenizer path. Defaults to ``llm_name``.
        chat_template_mode: Prompt formatting mode. ``"tokenizer"`` uses the
            checkpoint chat template, ``"short"`` keeps only compact GigaChat
            role tokens, and ``"plain"`` uses ``User:/Assistant:`` text.
        max_image_side: Optional maximum input image side before any backend
            processor runs. Aspect ratio is preserved and smaller images are
            not upscaled.
        use_4bit_llm: Whether to load the base LLM in 4-bit on CUDA devices.
        freeze_vision: Whether to freeze the donor vision stack. Set this to
            ``False`` when training new donor-vision LoRA/QLoRA adapters.
        vision_backend: Optional explicit backend name. Supported values are
            inferred from config when omitted; use ``"patch_llm"`` for the
            trainable patch-LLM encoder.
        vision_image_size: Maximum image side for the patch-LLM backend.
        vision_patch_size: Patch grid size for the patch-LLM backend.
        vision_conv_hidden_size: Hidden size of the patch-LLM convolutional
            stem.
        vision_use_maxpool: Whether the patch-LLM convolutional stem uses a
            max-pooling step.
        vision_llm_use_qlora: Whether the patch-LLM internal LLM is loaded with
            QLoRA adapters.
        vision_llm_dtype: Dtype for the patch-LLM internal LLM. One of
            ``"auto"``, ``"bf16"``, ``"fp16"``, or ``"fp32"``.
        vision_attn_implementation: Attention implementation requested for
            donor vision models, for example ``"auto"``, ``"sdpa"``, or
            ``"flash_attention_2"`` when supported by the model.
        vision_use_lora: Whether to attach LoRA adapters to the donor vision
            tower.
        vision_use_qlora: Whether to quantize donor vision linear layers to
            4-bit and attach LoRA adapters. Implies ``vision_use_lora``.
        vision_lora_r: Rank for donor-vision LoRA adapters.
        vision_lora_alpha: Alpha scaling for donor-vision LoRA adapters.
        vision_lora_dropout: Dropout used by donor-vision LoRA adapters.
        vision_lora_path: Optional existing donor-vision LoRA adapter path.
            Required when ``freeze_vision=True`` and vision LoRA is enabled.
        vision_projector_path: Optional donor-native projector weights, used by
            backends that expose a separate vision projector.
        vision_encoder_path: Optional saved patch-LLM/custom vision encoder
            state dict.
        vision_llm_lora_path: Optional saved LoRA adapter for the patch-LLM
            internal LLM.
        lora_r: Rank for LoRA adapters attached to the base LLM.
        lora_alpha: Alpha scaling for base LLM LoRA adapters.
        lora_dropout: Dropout used by base LLM LoRA adapters.
        num_lora_layers: Number of initial LLM layers that receive LoRA
            adapters. Use ``-1`` to target all eligible layers.
        train_llm_lora: Whether base LLM LoRA adapters are trainable. Set this
            to ``False`` for projector-only alignment stages.
        normalize_visual_embeddings: Whether to L2-normalize projected visual
            tokens to the mean norm of the base LLM token embeddings. Disabled
            by default to preserve raw projector behavior and checkpoint
            compatibility.
        lora_path: Optional existing base LLM LoRA adapter path.
        projector_path: Optional saved GigaChatVL projector state dict.
        projector_type: Visual connector type. Supported values are
            ``"mlp"``, ``"qformer"``, and ``"llm"``.
        projector_num_queries: Number of query tokens used by the QFormer
            connector.
        connector_llm_name: Hugging Face id or local path for the small LLM
            used when ``projector_type="llm"``.
        connector_llm_dtype: Dtype for the connector LLM. One of ``"auto"``,
            ``"bf16"``, ``"fp16"``, or ``"fp32"``.
        connector_use_text_prefix: Whether the connector LLM should see text
            embeddings before each ``[image_token]`` placeholder. Disabled by
            default, so the connector receives visual tokens only.
        freeze_connector_llm: Whether to freeze the connector LLM base weights.
        connector_llm_use_qlora: Whether to load the connector LLM with QLoRA
            adapters.
        connector_lora_r: Rank for connector LLM LoRA adapters.
        connector_lora_alpha: Alpha scaling for connector LLM LoRA adapters.
        connector_lora_dropout: Dropout used by connector LLM LoRA adapters.
        connector_llm_lora_path: Optional existing connector LLM LoRA adapter
            path.
        enable_vl_experts: Whether to add visual experts to selected LLM
            layers.
        vl_expert_layers: Layer indices that receive visual experts. ``None``
            follows the default LoRA layer selection, and an empty list disables
            visual experts even if the flag is present.
        vl_experts_path: Optional saved visual expert state dict.
        device: Optional torch device or device string. Defaults to the current
            CUDA device when available, otherwise CPU.

    Raises:
        ValueError: If an unsupported parameter combination is requested, such
            as training new donor-vision LoRA adapters while
            ``freeze_vision=True``.
    """

    def __init__(
        self,
        llm_name: str = "ai-sage/GigaChat3.1-10B-A1.8B-bf16",
        vision_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        tokenizer_name: Optional[str] = None,
        chat_template_mode: str = "tokenizer",
        max_image_side: Optional[int] = None,
        use_4bit_llm: bool = True,
        freeze_vision: bool = True,
        vision_backend: Optional[str] = None,
        vision_image_size: int = DEFAULT_PATCH_VISION_IMAGE_SIZE,
        vision_patch_size: int = DEFAULT_PATCH_VISION_PATCH_SIZE,
        vision_conv_hidden_size: int = 256,
        vision_use_maxpool: bool = False,
        vision_llm_use_qlora: bool = False,
        vision_llm_dtype: str = "auto",
        vision_attn_implementation: str = "auto",
        vision_use_lora: bool = False,
        vision_use_qlora: bool = False,
        vision_lora_r: int = 16,
        vision_lora_alpha: int = 32,
        vision_lora_dropout: float = 0.05,
        vision_lora_path: Optional[str] = None,
        vision_projector_path: Optional[str] = None,
        vision_encoder_path: Optional[str] = None,
        vision_llm_lora_path: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_lora_layers: int = -1,
        train_llm_lora: bool = True,
        normalize_visual_embeddings: bool = False,
        lora_path: Optional[str] = None,
        projector_path: Optional[str] = None,
        projector_type: str = "qformer",
        projector_num_queries: int = 32,
        connector_llm_name: Optional[str] = None,
        connector_llm_dtype: str = "auto",
        connector_use_text_prefix: bool = False,
        freeze_connector_llm: bool = True,
        connector_llm_use_qlora: bool = False,
        connector_lora_r: int = 16,
        connector_lora_alpha: int = 32,
        connector_lora_dropout: float = 0.05,
        connector_llm_lora_path: Optional[str] = None,
        enable_vl_experts: bool = False,
        vl_expert_layers: Optional[List[int]] = None,
        vl_experts_path: Optional[str] = None,
        device: Optional[Any] = None,
    ):
        super().__init__()

        self.model_device = single_device(device)
        self.llm_name = llm_name
        self.chat_template_mode = _normalize_chat_template_mode(chat_template_mode)
        self.max_image_side = None
        if max_image_side is not None:
            self.max_image_side = int(max_image_side)
            if self.max_image_side <= 0:
                raise ValueError(f"max_image_side must be > 0, got {max_image_side}")
        self.freeze_vision = freeze_vision
        self.vision_name = vision_name
        self.vision_backend_override = vision_backend.lower() if vision_backend is not None else None
        self.vision_image_size = int(vision_image_size)
        self.vision_patch_size = int(vision_patch_size)
        self.vision_conv_hidden_size = int(vision_conv_hidden_size)
        self.vision_use_maxpool = bool(vision_use_maxpool)
        self.vision_llm_use_qlora = bool(vision_llm_use_qlora)
        self.vision_llm_dtype = str(vision_llm_dtype or "auto").lower()
        self.vision_attn_implementation = str(
            vision_attn_implementation or "auto"
        ).lower()
        self.vision_use_lora = bool(
            vision_use_lora or vision_use_qlora or vision_lora_path is not None
        )
        self.vision_use_qlora = bool(vision_use_qlora)
        self.vision_lora_r = int(vision_lora_r)
        self.vision_lora_alpha = int(vision_lora_alpha)
        self.vision_lora_dropout = float(vision_lora_dropout)
        self.vision_lora_path = vision_lora_path
        self.vision_projector_path = vision_projector_path
        self.vision_encoder_path = vision_encoder_path
        self.vision_llm_lora_path = vision_llm_lora_path
        if self.vision_use_lora and self.freeze_vision and self.vision_lora_path is None:
            raise ValueError(
                "Vision LoRA/QLoRA training requires freeze_vision=False. "
                "Use freeze_vision=True only when loading an existing vision_lora adapter "
                "with vision_lora_path."
            )
        self.num_lora_layers = num_lora_layers
        self.train_llm_lora = bool(train_llm_lora)
        self.normalize_visual_embeddings = bool(normalize_visual_embeddings)
        self.vl_expert_layers = vl_expert_layers
        self.enable_vl_experts = enable_vl_experts or vl_experts_path is not None
        self.vl_expert_layer_indices: List[int] = []
        self.quantization_method = None
        self.is_loaded_in_4bit = False
        self.hf_device_map = {}

        # LLM tokenizer
        self.tokenizer = _load_fast_tokenizer(tokenizer_name or llm_name)

        if IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [IMAGE_TOKEN]}
            )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        # LLM loading
        llm_quant_config = None
        if use_4bit_llm and self.model_device.type == "cuda":
            llm_quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif use_4bit_llm and self.model_device.type != "cuda":
            print(
                "Warning: use_4bit_llm=True was requested, but 4-bit loading is "
                f"only enabled for CUDA devices. Loading full-precision LLM on {self.model_device}."
            )

        llm_dtype = torch.bfloat16 if self.model_device.type == "cuda" else torch.float32
        llm_device_map = single_device_map(self.model_device) if llm_quant_config is not None else None

        with _without_transformers_allocator_warmup():
            llm_load = AutoModelForCausalLM.from_pretrained(
                llm_name,
                dtype=llm_dtype,
                quantization_config=llm_quant_config,
                device_map=llm_device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                output_loading_info=True,
            )
        if isinstance(llm_load, tuple):
            self.llm, llm_loading_info = llm_load
        else:
            self.llm = llm_load
            llm_loading_info = {}

        unexpected_llm_keys = llm_loading_info.get("unexpected_keys", [])
        missing_llm_keys = llm_loading_info.get("missing_keys", [])
        benign_mtp_extras = (
            not missing_llm_keys
            and _looks_like_benign_mtp_unexpected_keys(unexpected_llm_keys, self.llm.config)
        )
        if benign_mtp_extras:
            print(
                "Note: base LLM checkpoint contains an extra MTP block "
                f"({len(list(unexpected_llm_keys))} tensors under "
                f"`model.layers.{self.llm.config.num_hidden_layers}.*`). "
                "Transformers `DeepseekV3ForCausalLM` does not load this auxiliary "
                "MTP block for standard causal LM training/inference, so these "
                "unexpected keys are expected and can be ignored."
            )
        elif unexpected_llm_keys or missing_llm_keys:
            print(
                "Warning: base LLM was loaded with non-empty loading info. "
                "This often means an architecture mismatch, which can severely hurt "
                "training quality. "
                f"unexpected={_preview_loading_keys(unexpected_llm_keys)}, "
                f"missing={_preview_loading_keys(missing_llm_keys)}"
            )

        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.config.use_cache = False

        if llm_quant_config is not None:
            # Some custom model loaders quantize weights but forget to expose
            # `is_loaded_in_4bit`, which makes PEFT cast the whole model to fp32.
            if not getattr(self.llm, "is_loaded_in_4bit", False) and _has_4bit_parameters(
                self.llm
            ):
                self.llm.is_loaded_in_4bit = True

            if getattr(self.llm, "is_loaded_in_4bit", False):
                self.quantization_method = QuantizationMethod.BITS_AND_BYTES
                self.is_loaded_in_4bit = True
                self.hf_device_map = {"llm": self.model_device.index}
                self.llm = _prepare_model_for_kbit_training_no_fp32_cast(self.llm)
            else:
                print(
                    "Warning: 4-bit quantization was requested, but the loaded LLM "
                    "does not expose 4-bit parameters. Skipping "
                    "k-bit preparation to avoid fp32 OOM."
                )
                for param in self.llm.parameters():
                    param.requires_grad = False
        elif device is not None:
            self.llm.to(device=self.model_device, dtype=llm_dtype)

        self.llm_hidden_size = self.llm.config.hidden_size
        self.visual_embedding_target_norm = _mean_embedding_norm(
            self.llm.get_input_embeddings()
        )

        if self.enable_vl_experts:
            self._install_vl_experts()

        if lora_path is not None:
            self.llm = PeftModel.from_pretrained(
                self.llm,
                lora_path,
                is_trainable=False,
            )
            self._set_vl_experts_trainable(False)
        else:
            layers_to_transform = _resolve_lora_layers_to_transform(
                getattr(self.llm.config, "num_hidden_layers", None),
                num_lora_layers=self.num_lora_layers,
            )
            target_modules = _resolve_lora_target_modules(
                self.llm,
                num_lora_layers=self.num_lora_layers,
                exclude_substrings=[".visual_expert."] if self.enable_vl_experts else None,
            )
            lora_kwargs = {
                "task_type": "CAUSAL_LM",
                "r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "bias": "none",
                "target_modules": target_modules,
            }
            if layers_to_transform is not None:
                lora_kwargs["layers_to_transform"] = layers_to_transform
                lora_kwargs["layers_pattern"] = "layers"
            lora_cfg = LoraConfig(**lora_kwargs)
            self.llm = get_peft_model(self.llm, lora_cfg)
            self.set_llm_lora_trainable(self.train_llm_lora)
            self._set_vl_experts_trainable(True)

        if vl_experts_path is not None:
            if not self.enable_vl_experts:
                raise RuntimeError("vl_experts_path was provided but VL experts are disabled.")
            self.load_vl_experts(vl_experts_path)

        # Vision side
        self.vision_processor = None
        self.vision_backend = None
        self.vision_source_model = None
        self.vision_tower = None
        self.vision_projector = None
        self.vision_hidden_size = None
        self.gemma_image_placeholder = None
        self.projector_type = projector_type.lower()
        self.projector_num_queries = projector_num_queries
        self.connector_llm_name = connector_llm_name
        self.connector_llm_dtype = str(connector_llm_dtype or "auto").lower()
        self.connector_use_text_prefix = bool(connector_use_text_prefix)
        self.freeze_connector_llm = freeze_connector_llm
        self.connector_llm_use_qlora = bool(
            connector_llm_use_qlora or connector_llm_lora_path is not None
        )
        self.connector_lora_r = int(connector_lora_r)
        self.connector_lora_alpha = int(connector_lora_alpha)
        self.connector_lora_dropout = float(connector_lora_dropout)
        self.connector_llm_lora_path = connector_llm_lora_path
        if self.projector_type not in {"qformer", "mlp", "llm"}:
            raise ValueError(
                f"Unsupported projector_type={projector_type}. "
                "Supported values: 'qformer', 'mlp', 'llm'."
            )

        self._load_vision_backend(vision_name)

        # Connector
        if self.projector_type == "qformer":
            self.projector = QFormerProjector(
                vision_dim=self.vision_hidden_size,
                llm_dim=self.llm_hidden_size,
                num_queries=self.projector_num_queries,
            )
        elif self.projector_type == "mlp":
            self.projector = MLPProjector(
                vision_dim=self.vision_hidden_size,
                llm_dim=self.llm_hidden_size,
            )
        else:
            if self.connector_llm_name is None:
                raise ValueError(
                    "projector_type='llm' requires connector_llm_name, for example "
                    "'Qwen/Qwen3-0.6B'."
                )
            connector_dtype = _resolve_dtype_choice(
                self.connector_llm_dtype,
                self.llm.get_input_embeddings().weight.dtype,
            )
            self.projector = LLMProjector(
                vision_dim=self.vision_hidden_size,
                llm_dim=self.llm_hidden_size,
                connector_llm_name=self.connector_llm_name,
                freeze_connector_llm=self.freeze_connector_llm,
                connector_llm_use_qlora=self.connector_llm_use_qlora,
                connector_lora_r=self.connector_lora_r,
                connector_lora_alpha=self.connector_lora_alpha,
                connector_lora_dropout=self.connector_lora_dropout,
                connector_llm_lora_path=self.connector_llm_lora_path,
                connector_dtype=connector_dtype,
                connector_device=self.llm.get_input_embeddings().weight.device,
            )
        projector_device = self.llm.get_input_embeddings().weight.device
        projector_dtype = self.llm.get_input_embeddings().weight.dtype
        if isinstance(self.projector, LLMProjector):
            self.projector.move_projection_layers(
                device=projector_device,
                dtype=_resolve_dtype_choice(self.connector_llm_dtype, projector_dtype),
            )
        else:
            self.projector.to(
                device=projector_device,
                dtype=projector_dtype,
            )
        if projector_path is not None:
            projector_state = torch.load(projector_path, map_location="cpu")
            checkpoint_projector_type = _infer_projector_type_from_state_dict(projector_state)
            if (
                checkpoint_projector_type is not None
                and checkpoint_projector_type != self.projector_type
            ):
                raise RuntimeError(
                    "Projector type mismatch while loading checkpoint. "
                    f"checkpoint_projector_type={checkpoint_projector_type}, "
                    f"current_projector_type={self.projector_type}"
                )
            try:
                self._load_projector_state_dict(projector_state)
            except RuntimeError as e:
                checkpoint_vision_dim = None
                current_vision_dim = None
                vision_proj_weight = projector_state.get("vision_proj.weight")
                if torch.is_tensor(vision_proj_weight) and vision_proj_weight.dim() == 2:
                    checkpoint_vision_dim = int(vision_proj_weight.shape[1])
                if checkpoint_vision_dim is None:
                    mlp_weight = projector_state.get("net.0.weight")
                    if torch.is_tensor(mlp_weight) and mlp_weight.dim() == 2:
                        checkpoint_vision_dim = int(mlp_weight.shape[1])
                if checkpoint_vision_dim is None:
                    llm_connector_weight = projector_state.get("vision_in_proj.weight")
                    if (
                        torch.is_tensor(llm_connector_weight)
                        and llm_connector_weight.dim() == 2
                    ):
                        checkpoint_vision_dim = int(llm_connector_weight.shape[1])

                current_weight = getattr(getattr(self.projector, "vision_proj", None), "weight", None)
                if current_weight is None and hasattr(self.projector, "net"):
                    try:
                        current_weight = self.projector.net[0].weight
                    except Exception:
                        current_weight = None
                if current_weight is None:
                    current_weight = getattr(
                        getattr(self.projector, "vision_in_proj", None),
                        "weight",
                        None,
                    )
                if torch.is_tensor(current_weight) and current_weight.dim() == 2:
                    current_vision_dim = int(current_weight.shape[1])

                raise RuntimeError(
                    f"Failed to load projector weights into {type(self.projector).__name__}. "
                    "This usually means either the checkpoint was trained with a different "
                    "projector architecture, or it was trained with a different vision "
                    "encoder than the one you are loading now. "
                    f"checkpoint_vision_dim={checkpoint_vision_dim}, "
                    f"current_vision_dim={current_vision_dim}, "
                    f"current_vision_name={vision_name}"
                ) from e

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return _clone_shared_tensors_in_state_dict(state_dict)

    def _vision_tower_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        device, dtype = _module_device_dtype(self.vision_tower)
        if getattr(self, "vision_use_qlora", False):
            dtype = getattr(self, "vision_compute_dtype", dtype)
        return device, dtype

    def _configure_donor_vision_trainability(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if self.vision_tower is None:
            return

        self.vision_compute_dtype = dtype
        use_adapter = bool(
            self.vision_use_lora
            or self.vision_use_qlora
            or self.vision_lora_path is not None
        )

        if use_adapter:
            if self.freeze_vision and self.vision_lora_path is None:
                raise ValueError(
                    "Vision LoRA/QLoRA training requires freeze_vision=False. "
                    "Use freeze_vision=True only when loading an existing vision_lora adapter "
                    "for inference."
                )

            target_modules = _resolve_lora_target_modules(
                self.vision_tower,
                num_lora_layers=0,
            )

            if self.vision_use_qlora:
                replaced = _replace_linear_layers_with_bnb_4bit(
                    self.vision_tower,
                    compute_dtype=dtype,
                    device=device,
                    exclude_module_names=[
                        # Gemma4 casts pixel inputs to input_proj.weight.dtype.
                        # Linear4bit stores weights as uint8, which would turn
                        # pixels into Byte tensors before LoRA dropout.
                        "patch_embedder.input_proj",
                    ],
                )
                if replaced == 0:
                    raise RuntimeError(
                        "Could not replace any vision Linear layers with 4-bit layers."
                    )
                self.vision_tower.is_loaded_in_4bit = True
                _clear_cuda_cache()
                self.vision_tower = _prepare_model_for_kbit_training_no_fp32_cast(
                    self.vision_tower
                )

            if self.vision_lora_path is not None:
                self.vision_tower = PeftModel.from_pretrained(
                    self.vision_tower,
                    self.vision_lora_path,
                    is_trainable=not self.freeze_vision,
                )
            else:
                lora_cfg = LoraConfig(
                    r=self.vision_lora_r,
                    lora_alpha=self.vision_lora_alpha,
                    lora_dropout=self.vision_lora_dropout,
                    bias="none",
                    target_modules=target_modules,
                )
                self.vision_tower = get_peft_model(self.vision_tower, lora_cfg)

            for name, param in self.vision_tower.named_parameters():
                is_adapter_param = "lora_" in name or "modules_to_save" in name
                param.requires_grad = bool((not self.freeze_vision) and is_adapter_param)
        else:
            self.vision_tower.requires_grad_(not self.freeze_vision)

        if self.vision_projector is not None:
            self.vision_projector.requires_grad_(not self.freeze_vision)

        if self.freeze_vision:
            self.vision_tower.eval()
            if self.vision_projector is not None:
                self.vision_projector.eval()

    def drop_frozen_vision_modules(self) -> None:
        """
        Free frozen donor vision modules when training only from precomputed features.

        After calling this, batches must provide `vision_precomputed_features`;
        online image encoding through `prepare_vision_inputs`/`encode_images` is no
        longer available.
        """
        if not self.freeze_vision:
            raise ValueError(
                "drop_frozen_vision_modules() is only safe when freeze_vision=True."
            )

        self.vision_tower = None
        self.vision_source_model = None
        self.vision_projector = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_vision_projector_state_dict(self, path: str) -> None:
        if self.vision_projector is None:
            raise RuntimeError(
                "A vision_projector checkpoint was provided, but this backend "
                "does not expose a donor vision_projector."
            )
        state_dict = torch.load(path, map_location="cpu")
        self.vision_projector.load_state_dict(state_dict, strict=True)

    def projector_state_dict(self) -> Dict[str, torch.Tensor]:
        if isinstance(self.projector, LLMProjector):
            return self.projector.connector_state_dict()
        return {
            key: value.detach().cpu()
            for key, value in self.projector.state_dict().items()
        }

    def _load_projector_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if isinstance(self.projector, LLMProjector):
            self.projector.load_connector_state_dict(state_dict)
        else:
            self.projector.load_state_dict(state_dict)

    def vision_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.vision_backend == "patch_llm" and isinstance(self.vision_tower, PatchLLMVisionEncoder):
            return self.vision_tower.non_llm_state_dict()
        if self.vision_tower is None:
            return {}
        return {
            key: value.detach().cpu()
            for key, value in self.vision_tower.state_dict().items()
        }

    def save_vision_encoder(self, output_dir: str) -> None:
        if self.vision_backend != "patch_llm" or not isinstance(
            self.vision_tower,
            PatchLLMVisionEncoder,
        ):
            if (
                self.vision_tower is not None
                and isinstance(self.vision_tower, PeftModel)
            ):
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                self.vision_tower.save_pretrained(str(output_path / "vision_lora"))

            if self.vision_projector is not None and not self.freeze_vision:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        key: value.detach().cpu()
                        for key, value in self.vision_projector.state_dict().items()
                    },
                    output_path / "vision_projector.pt",
                )
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.vision_encoder_state_dict(),
                "config": {
                    "vision_backend": self.vision_backend,
                    "vision_name": self.vision_name,
                    "vision_image_size": self.vision_image_size,
                    "vision_max_side_size": self.vision_image_size,
                    "vision_patch_size": self.vision_patch_size,
                    "vision_conv_hidden_size": self.vision_conv_hidden_size,
                    "vision_use_maxpool": self.vision_use_maxpool,
                    "vision_llm_use_qlora": self.vision_llm_use_qlora,
                    "vision_llm_dtype": self.vision_llm_dtype,
                },
            },
            output_path / "vision_encoder.pt",
        )

        if isinstance(self.vision_tower.vision_llm, PeftModel):
            self.vision_tower.vision_llm.save_pretrained(
                str(output_path / "vision_llm_lora")
            )
        elif not self.freeze_vision:
            self.vision_tower.vision_llm.save_pretrained(
                str(output_path / "vision_llm")
            )

    def save_connector_llm(self, output_dir: str) -> None:
        if not isinstance(self.projector, LLMProjector):
            return
        if not isinstance(self.projector.connector_llm, PeftModel):
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.projector.connector_llm.save_pretrained(
            str(output_path / "connector_llm_lora")
        )

    def load_vision_encoder(self, path: str) -> None:
        if self.vision_backend != "patch_llm" or not isinstance(
            self.vision_tower,
            PatchLLMVisionEncoder,
        ):
            raise RuntimeError("Patch-LLM vision encoder must be initialized before loading.")

        payload = torch.load(path, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        self.vision_tower.load_non_llm_state_dict(state_dict)

    def _install_vl_experts(self) -> None:
        layers = _resolve_llm_decoder_layers(self.llm)
        layer_indices = _resolve_vl_expert_layer_indices(
            num_hidden_layers=len(layers),
            num_lora_layers=self.num_lora_layers,
            vl_expert_layers=self.vl_expert_layers,
        )

        expert_device, expert_dtype = _module_device_dtype(self.llm.get_input_embeddings())
        installed_indices = []

        for layer_idx in layer_indices:
            layer = layers[layer_idx]
            if not hasattr(layer, "mlp"):
                raise RuntimeError(f"Layer {layer_idx} does not expose an `mlp` module.")

            if isinstance(layer.mlp, VLExpertMLPWrapper):
                wrapper = layer.mlp
            else:
                wrapper = VLExpertMLPWrapper(layer.mlp, self.llm.config)
                wrapper.visual_expert.to(device=expert_device, dtype=expert_dtype)
                layer.mlp = wrapper

            installed_indices.append(layer_idx)

        self.vl_expert_layer_indices = installed_indices

    def _iter_vl_expert_wrappers(self):
        if not self.enable_vl_experts:
            return

        layers = _resolve_llm_decoder_layers(self.llm)
        for layer_idx in self.vl_expert_layer_indices:
            mlp = layers[layer_idx].mlp
            if not isinstance(mlp, VLExpertMLPWrapper):
                raise RuntimeError(
                    f"Expected layer {layer_idx} to contain a VLExpertMLPWrapper."
                )
            yield layer_idx, mlp

    def _set_vl_expert_visual_token_mask(
        self,
        visual_token_mask: Optional[torch.Tensor],
    ) -> None:
        if not self.enable_vl_experts:
            return

        if visual_token_mask is not None:
            visual_token_mask = visual_token_mask.detach().to(dtype=torch.bool)
            if not torch.any(visual_token_mask):
                visual_token_mask = None

        for _, wrapper in self._iter_vl_expert_wrappers():
            wrapper.visual_token_mask = visual_token_mask

    def _set_vl_experts_trainable(self, trainable: bool) -> None:
        if not self.enable_vl_experts:
            return

        for _, wrapper in self._iter_vl_expert_wrappers():
            wrapper.visual_expert.requires_grad_(trainable)

    def set_llm_lora_trainable(self, trainable: bool) -> None:
        self.train_llm_lora = bool(trainable)

        if not isinstance(self.llm, PeftModel):
            self.llm.requires_grad_(self.train_llm_lora)
            return

        for name, param in self.llm.named_parameters():
            is_adapter_param = "lora_" in name or "modules_to_save" in name
            if is_adapter_param:
                param.requires_grad = self.train_llm_lora

    def vl_experts_state_dict(self) -> Dict[str, torch.Tensor]:
        if not self.enable_vl_experts:
            return {}

        state_dict: Dict[str, torch.Tensor] = {}
        for layer_idx, wrapper in self._iter_vl_expert_wrappers():
            for key, value in wrapper.visual_expert.state_dict().items():
                state_dict[f"layers.{layer_idx}.visual_expert.{key}"] = value.detach().cpu()
        return state_dict

    def save_vl_experts(self, path: str) -> None:
        if not self.enable_vl_experts:
            raise RuntimeError("VL experts are disabled; nothing to save.")

        torch.save(
            {
                "layers": self.vl_expert_layer_indices,
                "state_dict": self.vl_experts_state_dict(),
            },
            path,
        )

    def load_vl_experts(self, path: str) -> None:
        if not self.enable_vl_experts:
            raise RuntimeError("Enable VL experts before loading VL expert weights.")

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            payload_layers = payload.get("layers")
        else:
            state_dict = payload
            payload_layers = None

        if payload_layers is not None:
            payload_layers = [int(layer_idx) for layer_idx in payload_layers]
            if payload_layers != self.vl_expert_layer_indices:
                raise RuntimeError(
                    "VL expert layer mismatch while loading checkpoint. "
                    f"checkpoint_layers={payload_layers}, "
                    f"current_layers={self.vl_expert_layer_indices}"
                )

        for layer_idx, wrapper in self._iter_vl_expert_wrappers():
            prefix = f"layers.{layer_idx}.visual_expert."
            layer_state = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if not layer_state:
                raise KeyError(f"No VL expert weights found for layer {layer_idx}.")
            wrapper.visual_expert.load_state_dict(layer_state, strict=True)

    def build_vlm_meta(self) -> Dict[str, Any]:
        return {
            "image_token": IMAGE_TOKEN,
            "image_token_id": self.image_token_id,
            "llm_name": self.llm_name,
            "llm_hidden_size": self.llm_hidden_size,
            "num_lora_layers": self.num_lora_layers,
            "train_llm_lora": self.train_llm_lora,
            "chat_template_mode": getattr(self, "chat_template_mode", "tokenizer"),
            "max_image_side": getattr(self, "max_image_side", None),
            "normalize_visual_embeddings": self.normalize_visual_embeddings,
            "visual_embedding_target_norm": self.visual_embedding_target_norm,
            "enable_vl_experts": self.enable_vl_experts,
            "vl_expert_layers": self.vl_expert_layer_indices,
            "vl_expert_layers_config": self.vl_expert_layers,
            "projector_type": getattr(self, "projector_type", None),
            "projector_num_queries": getattr(self.projector, "num_queries", None),
            "connector_llm_name": getattr(self, "connector_llm_name", None),
            "connector_llm_dtype": getattr(self, "connector_llm_dtype", "auto"),
            "connector_use_text_prefix": bool(
                getattr(self, "connector_use_text_prefix", False)
            ),
            "freeze_connector_llm": getattr(self, "freeze_connector_llm", True),
            "connector_llm_use_qlora": getattr(self, "connector_llm_use_qlora", False),
            "connector_lora_r": getattr(self, "connector_lora_r", 16),
            "connector_lora_alpha": getattr(self, "connector_lora_alpha", 32),
            "connector_lora_dropout": getattr(self, "connector_lora_dropout", 0.05),
            "vision_hidden_size": self.vision_hidden_size,
            "vision_backend": self.vision_backend,
            "vision_backend_override": self.vision_backend_override,
            "vision_name": self.vision_name,
            "freeze_vision": self.freeze_vision,
            "vision_image_size": self.vision_image_size,
            "vision_max_side_size": self.vision_image_size,
            "vision_patch_size": self.vision_patch_size,
            "vision_conv_hidden_size": self.vision_conv_hidden_size,
            "vision_use_maxpool": self.vision_use_maxpool,
            "vision_llm_use_qlora": self.vision_llm_use_qlora,
            "vision_llm_dtype": self.vision_llm_dtype,
            "vision_use_lora": self.vision_use_lora,
            "vision_use_qlora": self.vision_use_qlora,
            "vision_lora_r": self.vision_lora_r,
            "vision_lora_alpha": self.vision_lora_alpha,
            "vision_lora_dropout": self.vision_lora_dropout,
        }

    def save_training_setup(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tokenizer_dir = output_path / "tokenizer"
        vision_processor_dir = output_path / "vision_processor"
        meta_path = output_path / "vlm_meta.json"

        self.tokenizer.save_pretrained(str(tokenizer_dir))

        if self.vision_processor is not None and hasattr(self.vision_processor, "save_pretrained"):
            self.vision_processor.save_pretrained(str(vision_processor_dir))

        meta_path.write_text(
            json.dumps(self.build_vlm_meta(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_training_checkpoint(
        cls,
        checkpoint_dir: str,
        llm_name: Optional[str] = None,
        vision_name: Optional[str] = None,
        use_4bit_llm: bool = True,
        freeze_vision: bool = True,
        vision_backend: Optional[str] = None,
        vision_image_size: Optional[int] = None,
        vision_patch_size: Optional[int] = None,
        vision_conv_hidden_size: Optional[int] = None,
        vision_use_maxpool: Optional[bool] = None,
        vision_llm_use_qlora: Optional[bool] = None,
        vision_llm_dtype: Optional[str] = None,
        vision_use_lora: Optional[bool] = None,
        vision_use_qlora: Optional[bool] = None,
        vision_lora_r: Optional[int] = None,
        vision_lora_alpha: Optional[int] = None,
        vision_lora_dropout: Optional[float] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_lora_layers: Optional[int] = None,
        train_llm_lora: Optional[bool] = None,
        chat_template_mode: Optional[str] = None,
        max_image_side: Optional[int] = None,
        normalize_visual_embeddings: Optional[bool] = None,
        projector_type: Optional[str] = None,
        projector_num_queries: Optional[int] = None,
        connector_llm_name: Optional[str] = None,
        connector_llm_dtype: Optional[str] = None,
        connector_use_text_prefix: Optional[bool] = None,
        freeze_connector_llm: Optional[bool] = None,
        connector_llm_use_qlora: Optional[bool] = None,
        connector_lora_r: Optional[int] = None,
        connector_lora_alpha: Optional[int] = None,
        connector_lora_dropout: Optional[float] = None,
        enable_vl_experts: Optional[bool] = None,
        vl_expert_layers: Optional[List[int]] = None,
        device: Optional[Any] = None,
    ) -> "GigaChatVL":
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

        meta_path = _resolve_checkpoint_sidecar_path(checkpoint_path, "vlm_meta.json")
        if meta_path is None:
            raise FileNotFoundError(
                "Could not find `vlm_meta.json` next to the checkpoint or in its parent "
                f"experiment directory: {checkpoint_path}"
            )

        tokenizer_dir = _resolve_checkpoint_sidecar_path(checkpoint_path, "tokenizer")
        vision_processor_dir = _resolve_checkpoint_sidecar_path(
            checkpoint_path,
            "vision_processor",
        )
        vision_encoder_path = _resolve_checkpoint_sidecar_path(
            checkpoint_path,
            "vision_encoder.pt",
        )
        vision_llm_lora_path = _resolve_checkpoint_sidecar_path(
            checkpoint_path,
            "vision_llm_lora",
        )
        vision_llm_full_path = _resolve_checkpoint_sidecar_path(
            checkpoint_path,
            "vision_llm",
        )
        vision_lora_path = _resolve_checkpoint_sidecar_path(
            checkpoint_path,
            "vision_lora",
        )
        vision_projector_path = _resolve_checkpoint_sidecar_path(
            checkpoint_path,
            "vision_projector.pt",
        )
        connector_llm_lora_path = _resolve_checkpoint_sidecar_path(
            checkpoint_path,
            "connector_llm_lora",
        )
        vl_experts_path = _resolve_checkpoint_sidecar_path(checkpoint_path, "vl_experts.pt")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        resolved_llm_name = llm_name or meta.get("llm_name")
        resolved_vision_name = vision_name or meta.get("vision_name")
        resolved_vision_backend = (
            vision_backend
            or meta.get("vision_backend_override")
            or ("patch_llm" if meta.get("vision_backend") == "patch_llm" else None)
        )
        if (
            resolved_vision_backend == "patch_llm"
            and vision_name is None
            and vision_llm_full_path is not None
        ):
            resolved_vision_name = str(vision_llm_full_path)
        resolved_projector_type = projector_type or meta.get("projector_type") or "qformer"
        resolved_num_lora_layers = num_lora_layers
        if resolved_num_lora_layers is None:
            resolved_num_lora_layers = meta.get("num_lora_layers", -1)
        resolved_train_llm_lora = train_llm_lora
        if resolved_train_llm_lora is None:
            resolved_train_llm_lora = bool(meta.get("train_llm_lora", True))
        resolved_chat_template_mode = _normalize_chat_template_mode(
            chat_template_mode or meta.get("chat_template_mode", "tokenizer")
        )
        resolved_max_image_side = max_image_side
        if resolved_max_image_side is None:
            resolved_max_image_side = meta.get("max_image_side")
        resolved_normalize_visual_embeddings = normalize_visual_embeddings
        if resolved_normalize_visual_embeddings is None:
            resolved_normalize_visual_embeddings = bool(
                meta.get("normalize_visual_embeddings", False)
            )
        resolved_projector_num_queries = projector_num_queries
        if resolved_projector_num_queries is None:
            resolved_projector_num_queries = meta.get("projector_num_queries")
        if resolved_projector_num_queries is None:
            resolved_projector_num_queries = 32
        resolved_connector_llm_name = connector_llm_name or meta.get("connector_llm_name")
        resolved_connector_llm_dtype = (
            connector_llm_dtype
            or meta.get("connector_llm_dtype")
            or "auto"
        )
        resolved_connector_use_text_prefix = connector_use_text_prefix
        if resolved_connector_use_text_prefix is None:
            resolved_connector_use_text_prefix = bool(
                meta.get(
                    "connector_use_text_prefix",
                    meta.get("connector_uses_text_prefix", False),
                )
            )
        resolved_freeze_connector_llm = freeze_connector_llm
        if resolved_freeze_connector_llm is None:
            resolved_freeze_connector_llm = bool(meta.get("freeze_connector_llm", True))
        resolved_connector_llm_use_qlora = connector_llm_use_qlora
        if resolved_connector_llm_use_qlora is None:
            resolved_connector_llm_use_qlora = bool(
                meta.get("connector_llm_use_qlora", False)
                or connector_llm_lora_path is not None
            )
        resolved_connector_lora_r = connector_lora_r
        if resolved_connector_lora_r is None:
            resolved_connector_lora_r = int(meta.get("connector_lora_r", 16))
        resolved_connector_lora_alpha = connector_lora_alpha
        if resolved_connector_lora_alpha is None:
            resolved_connector_lora_alpha = int(meta.get("connector_lora_alpha", 32))
        resolved_connector_lora_dropout = connector_lora_dropout
        if resolved_connector_lora_dropout is None:
            resolved_connector_lora_dropout = float(
                meta.get("connector_lora_dropout", 0.05)
            )
        resolved_vision_image_size = vision_image_size
        if resolved_vision_image_size is None:
            resolved_vision_image_size = meta.get("vision_max_side_size")
        if resolved_vision_image_size is None:
            resolved_vision_image_size = meta.get("vision_image_size")
        if resolved_vision_image_size is None:
            resolved_vision_image_size = DEFAULT_PATCH_VISION_IMAGE_SIZE
        resolved_vision_patch_size = vision_patch_size
        if resolved_vision_patch_size is None:
            resolved_vision_patch_size = meta.get(
                "vision_patch_size",
                DEFAULT_PATCH_VISION_PATCH_SIZE,
            )
        resolved_vision_conv_hidden_size = vision_conv_hidden_size
        if resolved_vision_conv_hidden_size is None:
            resolved_vision_conv_hidden_size = meta.get("vision_conv_hidden_size", 256)
        resolved_vision_use_maxpool = vision_use_maxpool
        if resolved_vision_use_maxpool is None:
            resolved_vision_use_maxpool = bool(meta.get("vision_use_maxpool", False))
        resolved_vision_llm_use_qlora = vision_llm_use_qlora
        if resolved_vision_llm_use_qlora is None:
            resolved_vision_llm_use_qlora = bool(
                meta.get("vision_llm_use_qlora", False)
                or vision_llm_lora_path is not None
            )
        resolved_vision_llm_dtype = vision_llm_dtype or meta.get("vision_llm_dtype") or "auto"
        resolved_vision_use_lora = vision_use_lora
        if resolved_vision_use_lora is None:
            resolved_vision_use_lora = bool(
                meta.get("vision_use_lora", False)
                or vision_lora_path is not None
            )
        resolved_vision_use_qlora = vision_use_qlora
        if resolved_vision_use_qlora is None:
            resolved_vision_use_qlora = bool(meta.get("vision_use_qlora", False))
        resolved_vision_lora_r = vision_lora_r
        if resolved_vision_lora_r is None:
            resolved_vision_lora_r = int(meta.get("vision_lora_r", 16))
        resolved_vision_lora_alpha = vision_lora_alpha
        if resolved_vision_lora_alpha is None:
            resolved_vision_lora_alpha = int(meta.get("vision_lora_alpha", 32))
        resolved_vision_lora_dropout = vision_lora_dropout
        if resolved_vision_lora_dropout is None:
            resolved_vision_lora_dropout = float(meta.get("vision_lora_dropout", 0.05))
        resolved_enable_vl_experts = enable_vl_experts
        if resolved_enable_vl_experts is None:
            resolved_enable_vl_experts = bool(meta.get("enable_vl_experts", False))
        resolved_vl_expert_layers = vl_expert_layers
        if resolved_vl_expert_layers is None:
            resolved_vl_expert_layers = meta.get("vl_expert_layers_config")
        if resolved_vl_expert_layers is None and bool(resolved_enable_vl_experts):
            resolved_vl_expert_layers = meta.get("vl_expert_layers")

        if resolved_llm_name is None:
            raise RuntimeError(
                "Could not infer base LLM path from training metadata. Pass llm_name explicitly."
            )
        if resolved_vision_name is None:
            raise RuntimeError(
                "Could not infer vision model path from training metadata. Pass vision_name explicitly."
            )
        if resolved_projector_type == "llm" and resolved_connector_llm_name is None:
            raise RuntimeError(
                "Could not infer connector LLM path from training metadata. "
                "Pass connector_llm_name explicitly."
            )
        if (
            resolved_projector_type == "llm"
            and bool(resolved_connector_llm_use_qlora)
            and connector_llm_lora_path is None
        ):
            raise FileNotFoundError(
                "Could not find connector LLM QLoRA adapter directory "
                "`connector_llm_lora` next to the checkpoint or in its parent "
                f"experiment directory: {checkpoint_path}"
            )
        if bool(resolved_vision_use_lora) and vision_lora_path is None:
            raise FileNotFoundError(
                "Could not find donor vision LoRA adapter directory `vision_lora` "
                "next to the checkpoint or in its parent experiment directory: "
                f"{checkpoint_path}"
            )

        model = cls(
            llm_name=resolved_llm_name,
            vision_name=resolved_vision_name,
            tokenizer_name=str(tokenizer_dir) if tokenizer_dir is not None else resolved_llm_name,
            chat_template_mode=resolved_chat_template_mode,
            max_image_side=resolved_max_image_side,
            use_4bit_llm=use_4bit_llm,
            freeze_vision=freeze_vision,
            vision_backend=resolved_vision_backend,
            vision_image_size=int(resolved_vision_image_size),
            vision_patch_size=int(resolved_vision_patch_size),
            vision_conv_hidden_size=int(resolved_vision_conv_hidden_size),
            vision_use_maxpool=bool(resolved_vision_use_maxpool),
            vision_llm_use_qlora=bool(resolved_vision_llm_use_qlora),
            vision_llm_dtype=str(resolved_vision_llm_dtype),
            vision_use_lora=bool(resolved_vision_use_lora),
            vision_use_qlora=bool(resolved_vision_use_qlora),
            vision_lora_r=int(resolved_vision_lora_r),
            vision_lora_alpha=int(resolved_vision_lora_alpha),
            vision_lora_dropout=float(resolved_vision_lora_dropout),
            vision_lora_path=str(vision_lora_path) if vision_lora_path is not None else None,
            vision_projector_path=(
                str(vision_projector_path)
                if vision_projector_path is not None
                else None
            ),
            vision_encoder_path=str(vision_encoder_path) if vision_encoder_path is not None else None,
            vision_llm_lora_path=(
                str(vision_llm_lora_path)
                if vision_llm_lora_path is not None
                else None
            ),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_lora_layers=int(resolved_num_lora_layers),
            train_llm_lora=bool(resolved_train_llm_lora),
            normalize_visual_embeddings=bool(resolved_normalize_visual_embeddings),
            projector_type=resolved_projector_type,
            projector_num_queries=int(resolved_projector_num_queries),
            connector_llm_name=resolved_connector_llm_name,
            connector_llm_dtype=str(resolved_connector_llm_dtype),
            connector_use_text_prefix=bool(resolved_connector_use_text_prefix),
            freeze_connector_llm=bool(resolved_freeze_connector_llm),
            connector_llm_use_qlora=bool(resolved_connector_llm_use_qlora),
            connector_lora_r=int(resolved_connector_lora_r),
            connector_lora_alpha=int(resolved_connector_lora_alpha),
            connector_lora_dropout=float(resolved_connector_lora_dropout),
            connector_llm_lora_path=(
                str(connector_llm_lora_path)
                if connector_llm_lora_path is not None
                else None
            ),
            enable_vl_experts=bool(resolved_enable_vl_experts),
            vl_expert_layers=resolved_vl_expert_layers,
            vl_experts_path=(
                str(vl_experts_path)
                if bool(resolved_enable_vl_experts) and vl_experts_path is not None
                else None
            ),
            device=device,
        )

        if vision_processor_dir is not None:
            model.vision_processor = AutoProcessor.from_pretrained(str(vision_processor_dir))

        return model

    def _load_vision_backend(self, vision_name: str):
        vision_device = getattr(self, "model_device", single_device())
        vision_dtype = torch.bfloat16 if vision_device.type == "cuda" else torch.float32

        if self.vision_backend_override in {"patch_llm", "conv_llm"}:
            patch_llm_dtype = _resolve_dtype_choice(
                self.vision_llm_dtype,
                default_dtype=vision_dtype,
            )
            self.vision_backend = "patch_llm"
            self.vision_processor = None
            self.vision_source_model = None
            self.vision_projector = None
            self.vision_tower = PatchLLMVisionEncoder(
                vision_llm_name=vision_name,
                image_size=self.vision_image_size,
                patch_size=self.vision_patch_size,
                conv_hidden_size=self.vision_conv_hidden_size,
                use_maxpool=self.vision_use_maxpool,
                use_qlora=self.vision_llm_use_qlora,
                lora_r=self.vision_lora_r,
                lora_alpha=self.vision_lora_alpha,
                lora_dropout=self.vision_lora_dropout,
                lora_path=self.vision_llm_lora_path,
                freeze=self.freeze_vision,
                device=vision_device,
                dtype=patch_llm_dtype,
            )
            self.vision_hidden_size = self.vision_tower.hidden_size
            if self.vision_encoder_path is not None:
                self.load_vision_encoder(self.vision_encoder_path)
            return

        cfg = AutoConfig.from_pretrained(vision_name, trust_remote_code=False)
        vision_attn_implementation = getattr(self, "vision_attn_implementation", "auto")
        if vision_device.type != "cuda" and str(vision_attn_implementation).lower() == "auto":
            vision_attn_implementation = "sdpa"
        _apply_vision_attn_implementation(
            cfg,
            vision_attn_implementation,
        )
        model_type = getattr(cfg, "model_type", None)

        # Qwen2.5-VL
        if model_type == "qwen2_5_vl":
            self.vision_backend = "qwen2_5_vl"
            self.vision_processor = AutoProcessor.from_pretrained(vision_name)

            src = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vision_name,
                dtype=vision_dtype,
                low_cpu_mem_usage=True,
            )
            self.vision_tower = _first_existing_path(src, ["model.visual", "visual"])
            self.vision_tower.to(device=vision_device, dtype=vision_dtype)
            self.vision_source_model = None

            if hasattr(src, "lm_head"):
                del src.lm_head
            if hasattr(src, "model") and hasattr(src.model, "language_model"):
                del src.model.language_model
            del src
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.vision_hidden_size = getattr(cfg.vision_config, "out_hidden_size", None)
            if self.vision_hidden_size is None:
                self.vision_hidden_size = getattr(cfg.vision_config, "hidden_size")

            self._configure_donor_vision_trainability(
                device=vision_device,
                dtype=vision_dtype,
            )
            if self.vision_projector_path is not None:
                self._load_vision_projector_state_dict(self.vision_projector_path)
            return

        # Qwen3 / Qwen3.5 family
        if model_type in {"qwen3_vl", "qwen3_5", "qwen3_5_moe", "qwen3_5_vl"}:
            self.vision_backend = model_type
            self.vision_processor = AutoProcessor.from_pretrained(vision_name)
            self.vision_source_model = None

            if model_type in {"qwen3_5", "qwen3_5_moe"}:
                try:
                    self.vision_tower = _load_qwen35_vision_module(
                        model_dir=vision_name,
                        cfg=cfg,
                        device=vision_device,
                        dtype=vision_dtype,
                    )
                except Exception as e:
                    if model_type == "qwen3_5_moe":
                        raise RuntimeError(
                            "Failed to load qwen3_5_moe vision-only weights directly. "
                            "Refusing to fall back to full donor loading because this "
                            "checkpoint is too large for the intended vision-only path."
                        ) from e

                    print(
                        "Warning: failed to load Qwen3.5 vision-only module directly "
                        f"from weights, falling back to full donor load: {e}"
                    )
                    src = AutoModelForImageTextToText.from_pretrained(
                        vision_name,
                        dtype=vision_dtype,
                        low_cpu_mem_usage=True,
                    )

                    try:
                        self.vision_tower = _first_existing_path(
                            src,
                            [
                                "model.visual",
                                "visual",
                                "model.vision_tower",
                                "vision_tower",
                                "model.vision_model",
                                "vision_model",
                            ],
                        )
                    except Exception:
                        self.vision_tower = None

                    if self.vision_tower is not None:
                        self.vision_tower.to(device=vision_device, dtype=vision_dtype)

                    if hasattr(src, "lm_head"):
                        del src.lm_head
                    if hasattr(src, "model") and hasattr(src.model, "language_model"):
                        del src.model.language_model
                    if hasattr(src, "language_model"):
                        del src.language_model
                    del src
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                src = AutoModelForImageTextToText.from_pretrained(
                    vision_name,
                    dtype=vision_dtype,
                    low_cpu_mem_usage=True,
                )

                try:
                    self.vision_tower = _first_existing_path(
                        src,
                        [
                            "model.visual",
                            "visual",
                            "model.vision_tower",
                            "vision_tower",
                            "model.vision_model",
                            "vision_model",
                        ],
                    )
                except Exception:
                    self.vision_tower = None

                if self.vision_tower is not None:
                    self.vision_tower.to(device=vision_device, dtype=vision_dtype)

                if hasattr(src, "lm_head"):
                    del src.lm_head
                if hasattr(src, "model") and hasattr(src.model, "language_model"):
                    del src.model.language_model
                if hasattr(src, "language_model"):
                    del src.language_model
                del src
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.vision_hidden_size = getattr(cfg.vision_config, "out_hidden_size", None)
            if self.vision_hidden_size is None:
                self.vision_hidden_size = getattr(
                    getattr(cfg, "text_config", None),
                    "hidden_size",
                    None,
                )
            if self.vision_hidden_size is None:
                self.vision_hidden_size = getattr(cfg.vision_config, "hidden_size")

            self._configure_donor_vision_trainability(
                device=vision_device,
                dtype=vision_dtype,
            )
            if self.vision_projector_path is not None:
                self._load_vision_projector_state_dict(self.vision_projector_path)
            return

        # SigLIP / SigLIP2
        if model_type in {"siglip", "siglip_vision_model", "siglip2", "siglip2_vision_model"}:
            self.vision_backend = "siglip2" if model_type.startswith("siglip2") else "siglip"
            try:
                self.vision_processor = AutoProcessor.from_pretrained(vision_name)
            except Exception:
                self.vision_processor = AutoImageProcessor.from_pretrained(vision_name)
            self.vision_source_model = None
            self.vision_tower = _load_siglip_vision_module(
                model_dir=vision_name,
                cfg=cfg,
                device=vision_device,
                dtype=vision_dtype,
            )

            vision_config = getattr(cfg, "vision_config", cfg)
            self.vision_hidden_size = getattr(vision_config, "hidden_size")
            self._configure_donor_vision_trainability(
                device=vision_device,
                dtype=vision_dtype,
            )
            if self.vision_projector_path is not None:
                self._load_vision_projector_state_dict(self.vision_projector_path)
            return

        # AIMv2
        if model_type in {"aimv2", "aimv2_vision_model"}:
            self.vision_backend = "aimv2"
            try:
                self.vision_processor = AutoProcessor.from_pretrained(vision_name)
            except Exception:
                self.vision_processor = AutoImageProcessor.from_pretrained(vision_name)
            self.vision_source_model = None
            self.vision_tower = _load_aimv2_vision_module(
                model_dir=vision_name,
                cfg=cfg,
                device=vision_device,
                dtype=vision_dtype,
            )

            vision_config = getattr(cfg, "vision_config", cfg)
            self.vision_hidden_size = getattr(vision_config, "hidden_size")
            self._configure_donor_vision_trainability(
                device=vision_device,
                dtype=vision_dtype,
            )
            if self.vision_projector_path is not None:
                self._load_vision_projector_state_dict(self.vision_projector_path)
            return

        # Gemma4
        if model_type == "gemma4":
            self.vision_backend = "gemma4"
            self.vision_processor = AutoProcessor.from_pretrained(vision_name)
            self.vision_source_model = None

            try:
                self.vision_tower, self.vision_projector = _load_gemma4_vision_modules(
                    model_dir=vision_name,
                    cfg=cfg,
                    device=vision_device,
                    dtype=vision_dtype,
                )
            except Exception as e:
                print(
                    "Warning: failed to load Gemma4 vision-only modules directly "
                    f"from weights, falling back to full donor load: {e}"
                )
                src = AutoModelForImageTextToText.from_pretrained(
                    vision_name,
                    dtype=vision_dtype,
                    low_cpu_mem_usage=True,
                )

                try:
                    self.vision_tower = _first_existing_path(
                        src,
                        [
                            "model.vision_tower",
                            "vision_tower",
                            "model.vision_model",
                            "vision_model",
                            "model.vision_encoder",
                            "vision_encoder",
                        ],
                    )
                except Exception:
                    self.vision_tower = None

                try:
                    self.vision_projector = _first_existing_path(
                        src,
                        [
                            "model.embed_vision",
                            "embed_vision",
                            "model.multi_modal_projector",
                            "multi_modal_projector",
                        ],
                    )
                except Exception:
                    self.vision_projector = None

                if self.vision_tower is not None:
                    self.vision_tower.to(device=vision_device, dtype=vision_dtype)
                if self.vision_projector is not None:
                    self.vision_projector.to(device=vision_device, dtype=vision_dtype)

                if hasattr(src, "lm_head"):
                    del src.lm_head
                if hasattr(src, "model") and hasattr(src.model, "language_model"):
                    del src.model.language_model
                if hasattr(src, "language_model"):
                    del src.language_model
                del src
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.vision_hidden_size = cfg.text_config.hidden_size

            self.gemma_image_placeholder = getattr(self.vision_processor, "image_token", None)
            if self.gemma_image_placeholder is None:
                self.gemma_image_placeholder = getattr(
                    getattr(self.vision_processor, "tokenizer", None),
                    "image_token",
                    None,
                )

            if self.gemma_image_placeholder is None:
                raise RuntimeError(
                    "Gemma4 processor does not expose image_token. Update transformers."
                )

            self._configure_donor_vision_trainability(
                device=vision_device,
                dtype=vision_dtype,
            )
            if self.vision_projector_path is not None:
                self._load_vision_projector_state_dict(self.vision_projector_path)
            return

        raise ValueError(
            f"Unsupported vision backend: {vision_name} (model_type={model_type}). "
            "Supported: qwen2_5_vl, qwen3_vl, qwen3_5, qwen3_5_moe, qwen3_5_vl, "
            "siglip, siglip2, aimv2, gemma4, patch_llm."
        )

    def prepare_vision_inputs(self, images: List[Any]) -> Dict[str, torch.Tensor]:
        max_image_side = getattr(self, "max_image_side", None)
        if max_image_side is not None:
            images = [
                _resize_pil_image_max_side(image, max_side_size=max_image_side)
                for image in images
            ]

        # Qwen family
        if self.vision_backend in {
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_5_vl",
        }:
            for image in images:
                if not isinstance(image, Image.Image):
                    raise TypeError(
                        f"Qwen vision backend expects PIL images in collator, got {type(image)}"
                    )

            try:
                image_batch = _prepare_qwen_images_batch(self.vision_processor, images)
            except Exception:
                pixel_values_parts = []
                grid_parts = []
                for image in images:
                    image_batch = _prepare_single_qwen_image(self.vision_processor, image)
                    pixel_values_parts.append(image_batch["pixel_values"])
                    grid_parts.append(image_batch["grid"])
                grid = torch.cat(grid_parts, dim=0)
                image_batch = {
                    "pixel_values": _pad_qwen_flat_pixel_values(
                        torch.cat(pixel_values_parts, dim=0),
                        grid,
                    ),
                    "grid": grid,
                }

            return {
                "vision_pixel_values": image_batch["pixel_values"],
                "vision_image_grid_thw": image_batch["grid"],
            }

        # SigLIP / SigLIP2
        if self.vision_backend in {"siglip", "siglip2"}:
            for image in images:
                if not isinstance(image, Image.Image):
                    raise TypeError(
                        f"SigLIP vision backend expects PIL images in collator, got {type(image)}"
                    )

            try:
                batch = self.vision_processor(
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
            except TypeError:
                batch = self.vision_processor(
                    images=images,
                    return_tensors="pt",
                )
            return {f"vision_{k}": v for k, v in batch.items()}

        # AIMv2
        if self.vision_backend == "aimv2":
            for image in images:
                if not isinstance(image, Image.Image):
                    raise TypeError(
                        f"AIMv2 vision backend expects PIL images in collator, got {type(image)}"
                    )

            vision_config = getattr(self.vision_tower, "config", None)
            is_native_aimv2 = bool(getattr(vision_config, "is_native", False))
            if not is_native_aimv2:
                try:
                    batch = self.vision_processor(
                        images=images,
                        return_tensors="pt",
                    )
                    return {f"vision_{k}": v for k, v in batch.items()}
                except Exception:
                    pass

            pixel_values_parts = []
            original_hw = []
            for image in images:
                single = self.vision_processor(
                    images=image,
                    return_tensors="pt",
                )
                pixel_values = single["pixel_values"]
                if pixel_values.dim() != 4 or pixel_values.size(0) != 1:
                    raise RuntimeError(
                        "AIMv2 processor returned unexpected pixel_values shape: "
                        f"{tuple(pixel_values.shape)}"
                    )
                pixel_values = pixel_values[0]
                pixel_values_parts.append(pixel_values)
                original_hw.append((int(pixel_values.size(-2)), int(pixel_values.size(-1))))

            patch_size = int(
                getattr(getattr(self.vision_tower, "config", None), "patch_size", 14)
            )
            max_h = max(h for h, _ in original_hw)
            max_w = max(w for _, w in original_hw)
            max_h = _ceil_div(max_h, patch_size) * patch_size
            max_w = _ceil_div(max_w, patch_size) * patch_size
            max_grid_h = max_h // patch_size
            max_grid_w = max_w // patch_size

            padded_pixel_values = []
            patch_masks = []
            for pixel_values, (height, width) in zip(pixel_values_parts, original_hw):
                pad_h = max_h - int(pixel_values.size(-2))
                pad_w = max_w - int(pixel_values.size(-1))
                padded_pixel_values.append(
                    F.pad(pixel_values, (0, pad_w, 0, pad_h), value=0.0)
                )

                grid_h = int(height) // patch_size
                grid_w = int(width) // patch_size
                mask_2d = torch.zeros(
                    max_grid_h,
                    max_grid_w,
                    dtype=torch.bool,
                )
                mask_2d[:grid_h, :grid_w] = True
                patch_masks.append(mask_2d.flatten())

            return {
                "vision_pixel_values": torch.stack(padded_pixel_values, dim=0),
                "vision_patch_attention_mask": torch.stack(patch_masks, dim=0),
            }

        # Gemma4
        if self.vision_backend == "gemma4":
            prompts = [self.gemma_image_placeholder for _ in images]
            nested_images = [[image] for image in images]
            batch = self.vision_processor(
                images=nested_images,
                text=prompts,
                return_tensors="pt",
                padding=True,
            )
            return {f"vision_{k}": v for k, v in batch.items()}

        # Trainable patch + small LLM vision encoder
        if self.vision_backend == "patch_llm":
            pixel_values_parts = []
            patch_grid_hw = []
            for image in images:
                x = _pil_to_normalized_tensor(
                    image=image,
                    max_side_size=self.vision_image_size,
                )
                height, width = x.shape[-2:]
                grid_h = max(1, _ceil_div(int(height), self.vision_patch_size))
                grid_w = max(1, _ceil_div(int(width), self.vision_patch_size))
                if self.vision_use_maxpool:
                    if grid_h % 2 != 0:
                        grid_h += 1
                    if grid_w % 2 != 0:
                        grid_w += 1
                patch_grid_hw.append((grid_h, grid_w))
                pixel_values_parts.append(x)

            max_h = max(grid_h for grid_h, _ in patch_grid_hw) * self.vision_patch_size
            max_w = max(grid_w for _, grid_w in patch_grid_hw) * self.vision_patch_size
            padded_pixel_values = []
            for x in pixel_values_parts:
                pad_h = max_h - x.size(-2)
                pad_w = max_w - x.size(-1)
                x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
                padded_pixel_values.append(x)

            return {
                "vision_pixel_values": torch.stack(padded_pixel_values, dim=0),
                "vision_patch_grid_hw": torch.tensor(
                    patch_grid_hw,
                    dtype=torch.long,
                ),
            }

        raise RuntimeError(f"Unknown vision backend: {self.vision_backend}")

    def _encode_images_qwen25(
        self,
        vision_pixel_values,
        vision_image_grid_thw,
    ) -> List[torch.Tensor]:
        return self._project_image_features(
            self._encode_vision_features_qwen25(
                vision_pixel_values=vision_pixel_values,
                vision_image_grid_thw=vision_image_grid_thw,
            )
        )

    def _encode_vision_features_qwen25(
        self,
        vision_pixel_values,
        vision_image_grid_thw,
    ) -> List[torch.Tensor]:
        if torch.is_tensor(vision_pixel_values) and vision_pixel_values.dim() == 3:
            return self._encode_vision_features_qwen25(
                vision_pixel_values=_compact_qwen_padded_pixel_values(
                    vision_pixel_values,
                    vision_image_grid_thw,
                ),
                vision_image_grid_thw=vision_image_grid_thw,
            )

        if isinstance(vision_pixel_values, list):
            out = []
            for pixel_values_i, grid_i in zip(vision_pixel_values, vision_image_grid_thw):
                out.extend(
                    self._encode_vision_features_qwen25(
                        vision_pixel_values=pixel_values_i,
                        vision_image_grid_thw=grid_i,
                    )
                )
            return out

        vision_device, vision_dtype = self._vision_tower_device_dtype()

        pixel_values = _maybe_tensor_to(vision_pixel_values, vision_device, vision_dtype)
        image_grid_thw = _maybe_tensor_to(vision_image_grid_thw, vision_device)
        if image_grid_thw.dim() == 1:
            expected_rows = int(image_grid_thw.prod().item())
        else:
            expected_rows = int(image_grid_thw.prod(dim=-1).sum().item())

        if pixel_values.dim() != 2 or pixel_values.size(0) != expected_rows:
            raise RuntimeError(
                "Inconsistent Qwen2.5 vision input before visual tower. "
                f"pixel_values_shape={tuple(pixel_values.shape)}, "
                f"image_grid_thw={image_grid_thw.tolist()}, "
                f"expected_rows={expected_rows}"
            )

        ctx = torch.no_grad() if self.freeze_vision else nullcontext()
        with ctx:
            try:
                feats = self.vision_tower(pixel_values, grid_thw=image_grid_thw)
            except TypeError:
                feats = self.vision_tower(pixel_values, image_grid_thw=image_grid_thw)
            except RuntimeError as e:
                raise RuntimeError(
                    "Qwen2.5 vision tower failed. "
                    f"pixel_values_shape={tuple(pixel_values.shape)}, "
                    f"image_grid_thw={image_grid_thw.tolist()}"
                ) from e

        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        elif hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state

        if feats.dim() != 2:
            raise RuntimeError(
                f"Unexpected Qwen2.5-VL visual output shape: {tuple(feats.shape)}"
            )

        spatial_merge_size = getattr(self.vision_tower, "spatial_merge_size", 1)
        merge_length = spatial_merge_size ** 2
        token_counts = [
            int(x) for x in (image_grid_thw.prod(dim=-1) // merge_length).tolist()
        ]
        return list(torch.split(feats, token_counts, dim=0))

    def _encode_images_qwen3_family(
        self,
        vision_pixel_values,
        vision_image_grid_thw,
    ) -> List[torch.Tensor]:
        return self._project_image_features(
            self._encode_vision_features_qwen3_family(
                vision_pixel_values=vision_pixel_values,
                vision_image_grid_thw=vision_image_grid_thw,
            )
        )

    def _encode_vision_features_qwen3_family(
        self,
        vision_pixel_values,
        vision_image_grid_thw,
    ) -> List[torch.Tensor]:
        if torch.is_tensor(vision_pixel_values) and vision_pixel_values.dim() == 3:
            return self._encode_vision_features_qwen3_family(
                vision_pixel_values=_compact_qwen_padded_pixel_values(
                    vision_pixel_values,
                    vision_image_grid_thw,
                ),
                vision_image_grid_thw=vision_image_grid_thw,
            )

        if isinstance(vision_pixel_values, list):
            out = []
            for pixel_values_i, grid_i in zip(vision_pixel_values, vision_image_grid_thw):
                out.extend(
                    self._encode_vision_features_qwen3_family(
                        vision_pixel_values=pixel_values_i,
                        vision_image_grid_thw=grid_i,
                    )
                )
            return out

        if self.vision_tower is None:
            raise RuntimeError("This Qwen3-family checkpoint does not expose a raw visual tower.")

        vision_device, vision_dtype = self._vision_tower_device_dtype()

        pixel_values = _maybe_tensor_to(vision_pixel_values, vision_device, vision_dtype)
        image_grid_thw = _maybe_tensor_to(vision_image_grid_thw, vision_device)

        ctx = torch.no_grad() if self.freeze_vision else nullcontext()

        with ctx:
            try:
                feats = self.vision_tower(pixel_values, grid_thw=image_grid_thw)
            except TypeError:
                feats = self.vision_tower(pixel_values, image_grid_thw=image_grid_thw)

        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        elif hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state

        spatial_merge_size = getattr(self.vision_tower, "spatial_merge_size", 1)
        merge_length = spatial_merge_size ** 2
        token_counts = [
            int(x) for x in (image_grid_thw.prod(dim=-1) // merge_length).tolist()
        ]

        if feats.dim() == 3:
            if feats.size(0) != len(token_counts):
                raise RuntimeError(
                    f"Unexpected qwen3-family batch shape: {tuple(feats.shape)} "
                    f"vs token_counts={token_counts}"
                )
            chunks = [feats[i, :token_counts[i]] for i in range(feats.size(0))]
        elif feats.dim() == 2:
            chunks = list(torch.split(feats, token_counts, dim=0))
        else:
            raise RuntimeError(f"Unexpected qwen3-family feature shape: {tuple(feats.shape)}")

        return chunks

    def _encode_images_siglip(
        self,
        vision_inputs: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        return self._project_image_features(self._encode_vision_features_siglip(vision_inputs))

    def _encode_vision_features_siglip(
        self,
        vision_inputs: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        if self.vision_tower is None:
            raise RuntimeError("SigLIP backend requires a vision_tower.")

        src_device, src_dtype = self._vision_tower_device_dtype()

        prepared = {}
        for k, v in vision_inputs.items():
            base_key = k.replace("vision_", "", 1)
            prepared[base_key] = _maybe_tensor_to(v, src_device, src_dtype)

        if "pixel_values" not in prepared:
            raise RuntimeError("SigLIP backend expects pixel_values from the processor.")

        forward_kwargs = {"pixel_values": prepared["pixel_values"]}
        if self.vision_backend == "siglip2":
            if "pixel_attention_mask" not in prepared or "spatial_shapes" not in prepared:
                raise RuntimeError(
                    "SigLIP2 backend expects pixel_attention_mask and spatial_shapes "
                    "from the processor."
                )
            forward_kwargs["pixel_attention_mask"] = prepared["pixel_attention_mask"]
            forward_kwargs["spatial_shapes"] = prepared["spatial_shapes"]

        ctx = torch.no_grad() if self.freeze_vision else nullcontext()
        with ctx:
            vision_outputs = self.vision_tower(**forward_kwargs)

        image_hidden_states = vision_outputs.last_hidden_state
        if image_hidden_states.dim() != 3:
            raise RuntimeError(
                f"Unexpected SigLIP hidden state shape: {tuple(image_hidden_states.shape)}"
            )

        attention_mask = prepared.get("pixel_attention_mask")
        chunks = []
        for i in range(image_hidden_states.size(0)):
            x = image_hidden_states[i]
            if (
                attention_mask is not None
                and attention_mask.dim() == 2
                and attention_mask.size(1) == x.size(0)
            ):
                x = x[attention_mask[i].to(device=x.device, dtype=torch.bool)]
            chunks.append(x)

        return chunks

    def _encode_images_aimv2(
        self,
        vision_inputs: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        return self._project_image_features(self._encode_vision_features_aimv2(vision_inputs))

    def _encode_vision_features_aimv2(
        self,
        vision_inputs: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        if self.vision_tower is None:
            raise RuntimeError("AIMv2 backend requires a vision_tower.")

        src_device, src_dtype = self._vision_tower_device_dtype()

        prepared = {}
        for k, v in vision_inputs.items():
            base_key = k.replace("vision_", "", 1)
            prepared[base_key] = _maybe_tensor_to(v, src_device, src_dtype)

        if "pixel_values" not in prepared:
            raise RuntimeError("AIMv2 backend expects pixel_values from the processor.")

        forward_kwargs = {"pixel_values": prepared["pixel_values"]}
        patch_attention_mask = prepared.get("patch_attention_mask")
        if patch_attention_mask is not None:
            if patch_attention_mask.dim() != 2:
                raise RuntimeError(
                    "AIMv2 patch_attention_mask must have shape (batch, patches), "
                    f"got {tuple(patch_attention_mask.shape)}"
                )
            mask_dtype = src_dtype if torch.is_floating_point(prepared["pixel_values"]) else torch.float32
            forward_kwargs["attention_mask"] = (
                (~patch_attention_mask.to(device=src_device, dtype=torch.bool))[
                    :, None, None, :
                ].to(dtype=mask_dtype)
                * torch.finfo(mask_dtype).min
            )

        ctx = torch.no_grad() if self.freeze_vision else nullcontext()
        with ctx:
            vision_outputs = self.vision_tower(**forward_kwargs)

        image_hidden_states = vision_outputs.last_hidden_state
        if image_hidden_states.dim() != 3:
            raise RuntimeError(
                f"Unexpected AIMv2 hidden state shape: {tuple(image_hidden_states.shape)}"
            )

        chunks = []
        for i in range(image_hidden_states.size(0)):
            x = image_hidden_states[i]
            if (
                patch_attention_mask is not None
                and patch_attention_mask.dim() == 2
                and patch_attention_mask.size(1) == x.size(0)
            ):
                x = x[patch_attention_mask[i].to(device=x.device, dtype=torch.bool)]
            chunks.append(x)
        return chunks

    def _encode_images_gemma4(
        self,
        vision_inputs: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        return self._project_image_features(self._encode_vision_features_gemma4(vision_inputs))

    def _encode_vision_features_gemma4(
        self,
        vision_inputs: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        if self.vision_tower is None or self.vision_projector is None:
            raise RuntimeError("Gemma4 backend requires vision_tower and vision_projector.")

        src_device, src_dtype = self._vision_tower_device_dtype()

        prepared = {}
        for k, v in vision_inputs.items():
            base_key = k.replace("vision_", "", 1)
            prepared[base_key] = _maybe_tensor_to(v, src_device, src_dtype)

        image_position_ids = prepared.get("image_position_ids")
        if image_position_ids is None:
            raise RuntimeError(
                "Gemma4 backend expects image_position_ids from the processor."
            )

        ctx = torch.no_grad() if self.freeze_vision else nullcontext()
        with ctx:
            try:
                vision_outputs = self.vision_tower(
                    pixel_values=prepared["pixel_values"],
                    pixel_position_ids=image_position_ids,
                    return_dict=True,
                )
            except TypeError:
                vision_outputs = self.vision_tower(
                    pixel_values=prepared["pixel_values"],
                    image_position_ids=image_position_ids,
                    return_dict=True,
                )
            image_hidden_states = self.vision_projector(vision_outputs.last_hidden_state)

        if image_hidden_states.dim() == 4:
            if image_hidden_states.size(1) != 1:
                raise NotImplementedError(
                    "Only 1 image per sample is supported for Gemma4 backend."
                )
            chunks = [image_hidden_states[i, 0] for i in range(image_hidden_states.size(0))]
        elif image_hidden_states.dim() == 3:
            chunks = [image_hidden_states[i] for i in range(image_hidden_states.size(0))]
        elif image_hidden_states.dim() == 2:
            chunks = None
        else:
            raise RuntimeError(
                f"Unexpected Gemma4 image_hidden_states shape: "
                f"{tuple(image_hidden_states.shape)}"
            )

        image_token_id = self.vision_processor.tokenizer.convert_tokens_to_ids(
            self.gemma_image_placeholder
        )
        token_counts = (prepared["input_ids"] == image_token_id).sum(dim=1).tolist()
        token_counts = [int(x) for x in token_counts]

        if chunks is None:
            total_tokens = sum(token_counts)
            if image_hidden_states.size(0) != total_tokens:
                raise RuntimeError(
                    "Gemma4 image features do not match placeholder count. "
                    f"feature_rows={image_hidden_states.size(0)}, "
                    f"token_counts={token_counts}"
                )
            chunks = list(torch.split(image_hidden_states, token_counts, dim=0))
        elif len(chunks) != len(token_counts):
            raise RuntimeError(
                "Gemma4 image feature batch does not match token count batch. "
                f"num_chunks={len(chunks)}, token_counts={token_counts}"
            )

        out = []
        for i, n in enumerate(token_counts):
            if n <= 0:
                raise RuntimeError(
                    f"Could not infer Gemma4 image token count for sample {i}."
            )

            x = chunks[i][:n]
            out.append(x)
        return out

    def _encode_images_patch_llm(
        self,
        vision_pixel_values: torch.Tensor,
        vision_patch_grid_hw: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        return self._project_image_features(
            self._encode_vision_features_patch_llm(
                vision_pixel_values=vision_pixel_values,
                vision_patch_grid_hw=vision_patch_grid_hw,
            )
        )

    def _encode_vision_features_patch_llm(
        self,
        vision_pixel_values: torch.Tensor,
        vision_patch_grid_hw: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if self.vision_tower is None or not isinstance(self.vision_tower, PatchLLMVisionEncoder):
            raise RuntimeError("Patch-LLM backend requires a PatchLLMVisionEncoder.")

        if vision_pixel_values.dim() != 4:
            raise RuntimeError(
                "Patch-LLM backend expects vision_pixel_values with shape "
                f"(batch, 3, H, W), got {tuple(vision_pixel_values.shape)}."
            )

        out = []
        ctx = torch.no_grad() if self.freeze_vision else nullcontext()

        for i in range(vision_pixel_values.size(0)):
            pixel_values_i = vision_pixel_values[i : i + 1]
            if vision_patch_grid_hw is not None:
                grid_h = int(vision_patch_grid_hw[i, 0].item())
                grid_w = int(vision_patch_grid_hw[i, 1].item())
                crop_h = grid_h * self.vision_patch_size
                crop_w = grid_w * self.vision_patch_size
                pixel_values_i = pixel_values_i[:, :, :crop_h, :crop_w]

            with ctx:
                hidden_states_i = self.vision_tower(pixel_values_i)

            if hidden_states_i.dim() != 3 or hidden_states_i.size(0) != 1:
                raise RuntimeError(
                    "Unexpected Patch-LLM hidden state shape: "
                    f"{tuple(hidden_states_i.shape)}"
                )

            x = hidden_states_i[0]
            out.append(x)
        return out

    def _project_single_image_features(
        self,
        vision_features: torch.Tensor,
        text_prefix_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(self.projector, LLMProjector):
            projector_device = self.projector.vision_in_proj.weight.device
            projector_dtype = self.projector.vision_in_proj.weight.dtype
        else:
            projector_device, projector_dtype = _module_device_dtype(self.projector)
        llm_embed_weight = self.llm.get_input_embeddings().weight
        x = vision_features.to(
            device=projector_device,
            dtype=projector_dtype,
        )
        if isinstance(self.projector, LLMProjector):
            if text_prefix_embeds is not None:
                text_prefix_embeds = text_prefix_embeds.to(
                    device=projector_device,
                    dtype=projector_dtype,
                )
            y = self.projector(
                x,
                text_prefix_embeds=text_prefix_embeds,
            )
        else:
            y = self.projector(x)

        y = y.to(
            device=llm_embed_weight.device,
            dtype=llm_embed_weight.dtype,
        )
        if getattr(self, "normalize_visual_embeddings", False):
            y_float = y.float()
            y_norm = y_float.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            target_norm = torch.tensor(
                getattr(self, "visual_embedding_target_norm", 1.0),
                device=y.device,
                dtype=torch.float32,
            )
            y = (y_float * (target_norm / y_norm)).to(dtype=llm_embed_weight.dtype)
        return y

    def _project_image_features(
        self,
        flat_vision_features: List[torch.Tensor],
        text_prefix_embeds_per_image: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        if text_prefix_embeds_per_image is None:
            text_prefix_embeds_per_image = [None] * len(flat_vision_features)
        if len(text_prefix_embeds_per_image) != len(flat_vision_features):
            raise RuntimeError(
                "text_prefix_embeds_per_image must match flat_vision_features. "
                f"prefixes={len(text_prefix_embeds_per_image)}, "
                f"features={len(flat_vision_features)}"
            )

        out = []
        for x, text_prefix_embeds in zip(
            flat_vision_features,
            text_prefix_embeds_per_image,
        ):
            y = self._project_single_image_features(
                vision_features=x,
                text_prefix_embeds=text_prefix_embeds,
            )
            out.append(y)
        return out

    def project_vision_features(
        self,
        flat_vision_features: List[torch.Tensor],
        text_prefix_embeds_per_image: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        return self._project_image_features(
            list(flat_vision_features),
            text_prefix_embeds_per_image=text_prefix_embeds_per_image,
        )

    def encode_images(
        self,
        vision_pixel_values: Optional[torch.Tensor] = None,
        vision_image_grid_thw: Optional[torch.Tensor] = None,
        **vision_kwargs,
    ) -> List[torch.Tensor]:
        return self.project_vision_features(
            self.encode_vision_features(
                vision_pixel_values=vision_pixel_values,
                vision_image_grid_thw=vision_image_grid_thw,
                **vision_kwargs,
            )
        )

    def encode_vision_features(
        self,
        vision_pixel_values: Optional[torch.Tensor] = None,
        vision_image_grid_thw: Optional[torch.Tensor] = None,
        **vision_kwargs,
    ) -> List[torch.Tensor]:
        if self.vision_backend == "qwen2_5_vl":
            if vision_pixel_values is None or vision_image_grid_thw is None:
                raise ValueError(
                    "Qwen2.5-VL backend expects vision_pixel_values and vision_image_grid_thw."
                )
            return self._encode_vision_features_qwen25(
                vision_pixel_values=vision_pixel_values,
                vision_image_grid_thw=vision_image_grid_thw,
            )

        if self.vision_backend in {"qwen3_vl", "qwen3_5", "qwen3_5_moe", "qwen3_5_vl"}:
            if vision_pixel_values is None or vision_image_grid_thw is None:
                raise ValueError(
                    "Qwen3-family backend expects vision_pixel_values and vision_image_grid_thw."
                )
            return self._encode_vision_features_qwen3_family(
                vision_pixel_values=vision_pixel_values,
                vision_image_grid_thw=vision_image_grid_thw,
            )

        if self.vision_backend in {"siglip", "siglip2"}:
            siglip_inputs = dict(vision_kwargs)
            if vision_pixel_values is not None:
                siglip_inputs["vision_pixel_values"] = vision_pixel_values
            if vision_image_grid_thw is not None:
                siglip_inputs["vision_image_grid_thw"] = vision_image_grid_thw
            return self._encode_vision_features_siglip(siglip_inputs)

        if self.vision_backend == "aimv2":
            aimv2_inputs = dict(vision_kwargs)
            if vision_pixel_values is not None:
                aimv2_inputs["vision_pixel_values"] = vision_pixel_values
            if vision_image_grid_thw is not None:
                aimv2_inputs["vision_image_grid_thw"] = vision_image_grid_thw
            return self._encode_vision_features_aimv2(aimv2_inputs)

        if self.vision_backend == "gemma4":
            gemma_inputs = dict(vision_kwargs)
            if vision_pixel_values is not None:
                gemma_inputs["vision_pixel_values"] = vision_pixel_values
            if vision_image_grid_thw is not None:
                gemma_inputs["vision_image_grid_thw"] = vision_image_grid_thw
            return self._encode_vision_features_gemma4(gemma_inputs)

        if self.vision_backend == "patch_llm":
            if vision_pixel_values is None:
                vision_pixel_values = vision_kwargs.get("vision_pixel_values")
            if vision_pixel_values is None:
                raise ValueError(
                    "Patch-LLM backend expects vision_pixel_values."
                )
            return self._encode_vision_features_patch_llm(
                vision_pixel_values=vision_pixel_values,
                vision_patch_grid_hw=vision_kwargs.get("vision_patch_grid_hw"),
            )

        raise RuntimeError(f"Unknown vision backend: {self.vision_backend}")

    def _group_image_features_by_sample(
        self,
        flat_image_features: List[torch.Tensor],
        num_images_per_sample: List[int],
    ) -> List[List[torch.Tensor]]:
        grouped: List[List[torch.Tensor]] = []
        offset = 0

        for count in num_images_per_sample:
            if count < 0:
                raise ValueError(f"num_images_per_sample must be >= 0, got {count}")
            grouped.append(flat_image_features[offset : offset + count])
            offset += count

        if offset != len(flat_image_features):
            raise RuntimeError(
                "Mismatch between flat image features and num_images_per_sample. "
                f"num_features={len(flat_image_features)}, consumed={offset}, "
                f"counts={num_images_per_sample}"
            )

        return grouped

    def _merge_single_sample_text_and_images(
        self,
        ids_i: torch.Tensor,
        embeds_i: torch.Tensor,
        labels_i: Optional[torch.Tensor],
        sample_image_features: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        image_positions = (ids_i == self.image_token_id).nonzero(as_tuple=False).flatten()
        num_images = len(sample_image_features)

        if image_positions.numel() != num_images:
            raise ValueError(
                f"Number of {IMAGE_TOKEN} placeholders must match number of images "
                f"in the sample. placeholders={image_positions.numel()}, images={num_images}"
            )

        if num_images == 0:
            visual_token_mask = torch.zeros(
                ids_i.size(0),
                dtype=torch.bool,
                device=ids_i.device,
            )
            return ids_i, embeds_i, labels_i, visual_token_mask

        merged_ids_parts = []
        merged_embed_parts = []
        merged_label_parts = [] if labels_i is not None else None
        merged_visual_mask_parts = []
        cursor = 0

        for pos, img_feats in zip(image_positions.tolist(), sample_image_features):
            merged_ids_parts.append(ids_i[cursor:pos])
            merged_embed_parts.append(embeds_i[cursor:pos])
            merged_visual_mask_parts.append(
                torch.zeros(
                    pos - cursor,
                    dtype=torch.bool,
                    device=ids_i.device,
                )
            )
            if merged_label_parts is not None:
                merged_label_parts.append(labels_i[cursor:pos])

            if isinstance(self.projector, LLMProjector):
                text_prefix_embeds = None
                if getattr(self, "connector_use_text_prefix", False):
                    text_prefix_mask = ids_i[:pos] != self.image_token_id
                    text_prefix_embeds = embeds_i[:pos][text_prefix_mask]
                img_feats = self._project_single_image_features(
                    vision_features=img_feats,
                    text_prefix_embeds=text_prefix_embeds,
                )

            merged_ids_parts.append(
                torch.full(
                    (img_feats.size(0),),
                    self.image_token_id,
                    dtype=ids_i.dtype,
                    device=ids_i.device,
                )
            )
            merged_embed_parts.append(img_feats)
            merged_visual_mask_parts.append(
                torch.ones(
                    img_feats.size(0),
                    dtype=torch.bool,
                    device=ids_i.device,
                )
            )
            if merged_label_parts is not None:
                merged_label_parts.append(
                    torch.full(
                        (img_feats.size(0),),
                        IGNORE_INDEX,
                        dtype=labels_i.dtype,
                        device=labels_i.device,
                    )
                )
            cursor = pos + 1

        merged_ids_parts.append(ids_i[cursor:])
        merged_embed_parts.append(embeds_i[cursor:])
        merged_visual_mask_parts.append(
            torch.zeros(
                ids_i.size(0) - cursor,
                dtype=torch.bool,
                device=ids_i.device,
            )
        )
        if merged_label_parts is not None:
            merged_label_parts.append(labels_i[cursor:])

        merged_ids = torch.cat(merged_ids_parts, dim=0)
        merged_embeds = torch.cat(merged_embed_parts, dim=0)
        merged_labels = (
            torch.cat(merged_label_parts, dim=0) if merged_label_parts is not None else None
        )
        visual_token_mask = torch.cat(merged_visual_mask_parts, dim=0)
        return merged_ids, merged_embeds, merged_labels, visual_token_mask

    def _merge_text_and_image(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        image_features_per_sample: List[List[torch.Tensor]],
    ):
        embed_tokens = self.llm.get_input_embeddings()
        base_embeds = embed_tokens(input_ids)

        batch_embeds = []
        batch_masks = []
        batch_labels = []
        batch_visual_masks = []
        max_len = 0

        for i in range(input_ids.size(0)):
            seq_len_i = int(attention_mask[i].sum().item())

            ids_i = input_ids[i, :seq_len_i]
            embeds_i = base_embeds[i, :seq_len_i]
            labels_i = labels[i, :seq_len_i] if labels is not None else None
            _, merged_embeds, merged_labels, visual_token_mask = self._merge_single_sample_text_and_images(
                ids_i=ids_i,
                embeds_i=embeds_i,
                labels_i=labels_i,
                sample_image_features=image_features_per_sample[i],
            )

            merged_mask = torch.ones(
                merged_embeds.size(0),
                dtype=attention_mask.dtype,
                device=merged_embeds.device,
            )

            batch_embeds.append(merged_embeds)
            batch_masks.append(merged_mask)
            batch_labels.append(merged_labels)
            batch_visual_masks.append(visual_token_mask)
            max_len = max(max_len, merged_embeds.size(0))

        padded_embeds = []
        padded_masks = []
        padded_labels = []
        padded_visual_masks = []

        hidden_size = batch_embeds[0].size(-1)
        embed_dtype = batch_embeds[0].dtype
        embed_device = batch_embeds[0].device

        for embeds_i, mask_i, labels_i, visual_mask_i in zip(
            batch_embeds,
            batch_masks,
            batch_labels,
            batch_visual_masks,
        ):
            pad_len = max_len - embeds_i.size(0)

            if pad_len > 0:
                pad_embeds = torch.zeros(
                    pad_len,
                    hidden_size,
                    dtype=embed_dtype,
                    device=embed_device,
                )
                pad_mask = torch.zeros(
                    pad_len,
                    dtype=mask_i.dtype,
                    device=mask_i.device,
                )

                embeds_i = torch.cat([embeds_i, pad_embeds], dim=0)
                mask_i = torch.cat([mask_i, pad_mask], dim=0)
                visual_pad_mask = torch.zeros(
                    pad_len,
                    dtype=torch.bool,
                    device=visual_mask_i.device,
                )
                visual_mask_i = torch.cat([visual_mask_i, visual_pad_mask], dim=0)

                if labels_i is not None:
                    pad_labels = torch.full(
                        (pad_len,),
                        IGNORE_INDEX,
                        dtype=labels_i.dtype,
                        device=labels_i.device,
                    )
                    labels_i = torch.cat([labels_i, pad_labels], dim=0)

            padded_embeds.append(embeds_i)
            padded_masks.append(mask_i)
            padded_labels.append(labels_i)
            padded_visual_masks.append(visual_mask_i)

        inputs_embeds = torch.stack(padded_embeds, dim=0)
        attention_mask = torch.stack(padded_masks, dim=0)
        visual_token_mask = torch.stack(padded_visual_masks, dim=0)

        if labels is not None:
            labels = torch.stack(padded_labels, dim=0)
        else:
            labels = None

        return inputs_embeds, attention_mask, labels, visual_token_mask

    def build_chat_prompt(self, text: str, num_images: int = 1) -> str:
        """Build an inference prompt for this model's tokenizer.

        Args:
            text: User text prompt.
            num_images: Number of image placeholders to prepend.

        Returns:
            Formatted chat prompt ending at the assistant generation prefix.
        """
        return _build_chat_prompt(
            self.tokenizer,
            text,
            num_images=num_images,
            chat_template_mode=getattr(self, "chat_template_mode", "tokenizer"),
        )

    def build_chat_training_texts(
        self,
        question: str,
        answer: str,
        num_images: int = 1,
    ) -> tuple[str, str]:
        """Build prompt/full text pair used for supervised label masking.

        Args:
            question: User-side training prompt.
            answer: Assistant answer used as supervised target.
            num_images: Number of image placeholders to prepend.

        Returns:
            A ``(prompt, full_text)`` tuple where ``full_text`` should start
            with ``prompt``.
        """
        return _build_chat_training_texts(
            self.tokenizer,
            question=question,
            answer=answer,
            num_images=num_images,
            chat_template_mode=getattr(self, "chat_template_mode", "tokenizer"),
        )

    def _build_generation_inputs(
        self,
        text: str,
        image: Optional[Any],
    ):
        if image is None:
            images: List[Image.Image] = []
        elif isinstance(image, Image.Image):
            images = [image.convert("RGB")]
        elif isinstance(image, (list, tuple)):
            images = []
            for item in image:
                if not isinstance(item, Image.Image):
                    raise TypeError(
                        "Every item in image list must be PIL.Image.Image. "
                        f"Got {type(item)}"
                    )
                images.append(item.convert("RGB"))
        else:
            raise TypeError(
                f"image must be PIL.Image.Image, list[Image.Image], or None, got {type(image)}"
            )

        prompt = self.build_chat_prompt(text, num_images=len(images))
        text_batch = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = text_batch["input_ids"]
        attention_mask = text_batch["attention_mask"]
        llm_device = self.llm.get_input_embeddings().weight.device

        if not images:
            return {
                "input_ids": input_ids.to(llm_device),
                "attention_mask": attention_mask.to(llm_device),
                "prompt_length": int(attention_mask[0].sum().item()),
            }

        vision_batch = self.prepare_vision_inputs(images)
        if isinstance(self.projector, LLMProjector):
            flat_image_features = self.encode_vision_features(**vision_batch)
        else:
            flat_image_features = self.encode_images(**vision_batch)
        grouped_image_features = self._group_image_features_by_sample(
            flat_image_features=flat_image_features,
            num_images_per_sample=[len(images)],
        )
        input_ids = input_ids.to(llm_device)
        attention_mask = attention_mask.to(llm_device)

        embed_tokens = self.llm.get_input_embeddings()
        base_embeds = embed_tokens(input_ids)
        expanded_input_ids, merged_embeds, _, visual_token_mask = self._merge_single_sample_text_and_images(
            ids_i=input_ids[0],
            embeds_i=base_embeds[0],
            labels_i=None,
            sample_image_features=grouped_image_features[0],
        )
        expanded_input_ids = expanded_input_ids.unsqueeze(0)
        inputs_embeds = merged_embeds.unsqueeze(0)

        attention_mask = torch.ones(
            expanded_input_ids.shape,
            dtype=attention_mask.dtype,
            device=expanded_input_ids.device,
        )

        return {
            "input_ids": expanded_input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "visual_token_mask": visual_token_mask.unsqueeze(0),
            "prompt_length": expanded_input_ids.size(1),
        }

    @torch.no_grad()
    def debug_multimodal_inputs(
        self,
        text: str,
        image: Optional[Any],
    ) -> Dict[str, Any]:
        """Return diagnostics for the multimodal prompt/image merge path.

        Args:
            text: User text prompt.
            image: Optional PIL image or list of PIL images.

        Returns:
            A dictionary with token counts, visual token counts, tensor shapes,
            feature norms, and connector/backend metadata. This method is
            intended for notebook debugging and does not change model state.
        """
        if image is None:
            images: List[Image.Image] = []
        elif isinstance(image, Image.Image):
            images = [image]
        elif isinstance(image, (list, tuple)):
            images = list(image)
        else:
            raise TypeError(
                f"image must be PIL.Image.Image, list[Image.Image], or None, got {type(image)}"
            )

        prompt = self.build_chat_prompt(text, num_images=len(images))
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prepared = self._build_generation_inputs(text=text, image=image)

        visual_token_mask = prepared.get("visual_token_mask")
        visual_token_count = (
            int(visual_token_mask.sum().item())
            if torch.is_tensor(visual_token_mask)
            else 0
        )
        input_ids = prepared.get("input_ids")
        input_image_token_count = (
            int((input_ids == self.image_token_id).sum().item())
            if torch.is_tensor(input_ids)
            else 0
        )

        visual_embed_norm_mean = None
        text_embed_norm_mean = None
        visual_to_text_norm_ratio = None
        visual_norm_warning = None
        if "inputs_embeds" in prepared and torch.is_tensor(visual_token_mask):
            inputs_embeds = prepared["inputs_embeds"]
            mask = visual_token_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if mask.any():
                visual_embed_norm_mean = float(inputs_embeds[mask].norm(dim=-1).mean().item())
            if (~mask).any():
                text_embed_norm_mean = float(inputs_embeds[~mask].norm(dim=-1).mean().item())
            if (
                visual_embed_norm_mean is not None
                and text_embed_norm_mean is not None
                and text_embed_norm_mean > 0
            ):
                visual_to_text_norm_ratio = visual_embed_norm_mean / text_embed_norm_mean
                if visual_to_text_norm_ratio > 3.0 or visual_to_text_norm_ratio < 0.33:
                    visual_norm_warning = (
                        "visual embeddings norm is far from text embeddings; "
                        "try normalize_visual_embeddings=True and retrain/load a "
                        "checkpoint trained with the same setting"
                    )

        connector_llm = None
        connector_is_peft = False
        connector_is_quantized = False
        if isinstance(getattr(self, "projector", None), LLMProjector):
            connector_llm = self.projector.connector_llm
            connector_is_peft = isinstance(connector_llm, PeftModel)
            connector_is_quantized = bool(getattr(self.projector, "connector_is_quantized", False))

        return {
            "vision_backend": self.vision_backend,
            "vision_name": self.vision_name,
            "projector_type": self.projector_type,
            "chat_template_mode": getattr(self, "chat_template_mode", "tokenizer"),
            "max_image_side": getattr(self, "max_image_side", None),
            "connector_llm_name": getattr(self, "connector_llm_name", None),
            "connector_llm_dtype": getattr(self, "connector_llm_dtype", "auto"),
            "connector_use_text_prefix": bool(
                getattr(self, "connector_use_text_prefix", False)
            ),
            "connector_llm_use_qlora": bool(getattr(self, "connector_llm_use_qlora", False)),
            "connector_is_peft": connector_is_peft,
            "connector_is_quantized": connector_is_quantized,
            "normalize_visual_embeddings": bool(
                getattr(self, "normalize_visual_embeddings", False)
            ),
            "visual_embedding_target_norm": float(
                getattr(self, "visual_embedding_target_norm", 1.0)
            ),
            "num_images": len(images),
            "image_token": IMAGE_TOKEN,
            "image_token_id": int(self.image_token_id),
            "prompt_image_token_count": int(
                sum(1 for token_id in prompt_ids if token_id == self.image_token_id)
            ),
            "input_image_token_count": input_image_token_count,
            "visual_token_count": visual_token_count,
            "prompt_token_length": len(prompt_ids),
            "merged_sequence_length": int(prepared["attention_mask"].shape[-1]),
            "has_inputs_embeds": "inputs_embeds" in prepared,
            "inputs_embeds_shape": (
                tuple(prepared["inputs_embeds"].shape)
                if "inputs_embeds" in prepared
                else None
            ),
            "attention_mask_shape": tuple(prepared["attention_mask"].shape),
            "visual_embed_norm_mean": visual_embed_norm_mean,
            "text_embed_norm_mean": text_embed_norm_mean,
            "visual_to_text_norm_ratio": visual_to_text_norm_ratio,
            "visual_norm_warning": visual_norm_warning,
        }

    @torch.no_grad()
    def debug_visual_embedding_effect(
        self,
        text: str,
        image: Any,
    ) -> Dict[str, Any]:
        """Compare next-token logits with visual embeddings vs image token ids.

        Args:
            text: User text prompt.
            image: PIL image or list of PIL images.

        Returns:
            A dictionary with logit-difference statistics. If the visual
            embeddings are connected to the LLM forward pass, the differences
            should be non-zero. Values close to zero mean the model is seeing
            the image-token ids rather than the projected visual embeddings.
        """
        prepared = self._build_generation_inputs(text=text, image=image)
        if "inputs_embeds" not in prepared:
            return {
                "has_inputs_embeds": False,
                "visual_token_count": 0,
                "max_abs_logit_diff": 0.0,
                "mean_abs_logit_diff": 0.0,
            }

        device = self.llm.get_input_embeddings().weight.device
        input_ids = prepared["input_ids"].to(device)
        attention_mask = prepared["attention_mask"].to(device)
        inputs_embeds = prepared["inputs_embeds"].to(device)
        visual_token_mask = prepared.get("visual_token_mask")
        self._set_vl_expert_visual_token_mask(visual_token_mask)
        try:
            visual_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            token_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        finally:
            self._set_vl_expert_visual_token_mask(None)

        visual_logits = visual_outputs.logits[:, -1, :].float()
        token_logits = token_outputs.logits[:, -1, :].float()
        diff = (visual_logits - token_logits).abs()

        return {
            "has_inputs_embeds": True,
            "visual_token_count": (
                int(visual_token_mask.sum().item())
                if torch.is_tensor(visual_token_mask)
                else 0
            ),
            "max_abs_logit_diff": float(diff.max().item()),
            "mean_abs_logit_diff": float(diff.mean().item()),
        }

    def unfreeze_last_vision_blocks(self, n_blocks: int = 4):
        if self.vision_tower is None:
            raise RuntimeError(
                "This backend does not expose a raw vision tower in a stable way."
            )

        block_container = None
        for name in ["blocks", "layers", "encoder.layers"]:
            try:
                block_container = _get_by_path(self.vision_tower, name)
                break
            except AttributeError:
                pass

        if block_container is None:
            raise RuntimeError("Could not find vision blocks/layers to unfreeze.")

        for block in block_container[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    def gradient_checkpointing_enable(self):
        _toggle_gradient_checkpointing(self.llm, enabled=True)

        if self.vision_tower is not None:
            vision_trainable = any(p.requires_grad for p in self.vision_tower.parameters())
            if vision_trainable:
                _toggle_gradient_checkpointing(self.vision_tower, enabled=True)

        if isinstance(self.projector, LLMProjector):
            self.projector.gradient_checkpointing_enable()

        return self

    def gradient_checkpointing_disable(self):
        _toggle_gradient_checkpointing(self.llm, enabled=False)

        if self.vision_tower is not None:
            vision_trainable = any(p.requires_grad for p in self.vision_tower.parameters())
            if vision_trainable:
                _toggle_gradient_checkpointing(self.vision_tower, enabled=False)

        if isinstance(self.projector, LLMProjector):
            self.projector.gradient_checkpointing_disable()

        return self

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen donor vision modules in eval mode so that attention
        # dropout (and any other train/eval-sensitive layers) inside the
        # frozen vision encoder do not produce stochastic features during
        # training. Without this override, Trainer.train() recursively sets
        # every submodule to training mode, undoing the eval() call made in
        # _configure_donor_vision_trainability().
        if self.freeze_vision:
            if self.vision_tower is not None:
                self.vision_tower.eval()
            if self.vision_projector is not None:
                self.vision_projector.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        vision_pixel_values: Optional[torch.Tensor] = None,
        vision_image_grid_thw: Optional[torch.Tensor] = None,
        vision_precomputed_features: Optional[List[torch.Tensor]] = None,
        vision_num_images_per_sample: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        vision_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k.startswith("vision_") and v is not None
        }
        if vision_pixel_values is not None:
            vision_kwargs["vision_pixel_values"] = vision_pixel_values
        if vision_image_grid_thw is not None:
            vision_kwargs["vision_image_grid_thw"] = vision_image_grid_thw

        if vision_num_images_per_sample is None:
            if vision_kwargs:
                num_images_per_sample = [1] * input_ids.size(0)
            else:
                num_images_per_sample = [0] * input_ids.size(0)
        elif torch.is_tensor(vision_num_images_per_sample):
            num_images_per_sample = [int(x) for x in vision_num_images_per_sample.tolist()]
        else:
            num_images_per_sample = [int(x) for x in vision_num_images_per_sample]

        total_num_images = sum(num_images_per_sample)
        if total_num_images == 0:
            self._set_vl_expert_visual_token_mask(None)
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                num_items_in_batch=num_items_in_batch,
            )

        project_images_during_merge = isinstance(self.projector, LLMProjector)
        if vision_precomputed_features is not None:
            flat_image_features = list(vision_precomputed_features)
            if not project_images_during_merge:
                flat_image_features = self.project_vision_features(flat_image_features)
        else:
            if project_images_during_merge:
                flat_image_features = self.encode_vision_features(**vision_kwargs)
            else:
                flat_image_features = self.encode_images(**vision_kwargs)

        image_features_per_sample = self._group_image_features_by_sample(
            flat_image_features=flat_image_features,
            num_images_per_sample=num_images_per_sample,
        )

        inputs_embeds, attention_mask, labels, visual_token_mask = self._merge_text_and_image(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_features_per_sample=image_features_per_sample,
        )

        self._set_vl_expert_visual_token_mask(visual_token_mask)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            num_items_in_batch=num_items_in_batch,
        )


class GigaChatVLForInference(GigaChatVL):
    def __init__(
        self,
        checkpoint_dir: str,
        llm_name: Optional[str] = None,
        vision_name: Optional[str] = None,
        use_4bit_llm: bool = True,
        vision_backend: Optional[str] = None,
        vision_image_size: Optional[int] = None,
        vision_patch_size: Optional[int] = None,
        vision_conv_hidden_size: Optional[int] = None,
        vision_use_maxpool: Optional[bool] = None,
        vision_llm_use_qlora: Optional[bool] = None,
        vision_llm_dtype: Optional[str] = None,
        vision_use_lora: Optional[bool] = None,
        vision_use_qlora: Optional[bool] = None,
        vision_lora_r: Optional[int] = None,
        vision_lora_alpha: Optional[int] = None,
        vision_lora_dropout: Optional[float] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_lora_layers: Optional[int] = None,
        chat_template_mode: Optional[str] = None,
        max_image_side: Optional[int] = None,
        normalize_visual_embeddings: Optional[bool] = None,
        projector_type: Optional[str] = None,
        projector_num_queries: Optional[int] = None,
        connector_llm_name: Optional[str] = None,
        connector_llm_dtype: Optional[str] = None,
        connector_use_text_prefix: Optional[bool] = None,
        freeze_connector_llm: Optional[bool] = None,
        connector_llm_use_qlora: Optional[bool] = None,
        connector_lora_r: Optional[int] = None,
        connector_lora_alpha: Optional[int] = None,
        connector_lora_dropout: Optional[float] = None,
        enable_vl_experts: Optional[bool] = None,
        vl_expert_layers: Optional[List[int]] = None,
        device: Optional[Any] = None,
    ):
        checkpoint_path = Path(checkpoint_dir)
        meta_path = checkpoint_path / "vlm_meta.json"
        lora_dir = checkpoint_path / "llm_lora"
        projector_path = checkpoint_path / "projector.pt"
        vl_experts_path = checkpoint_path / "vl_experts.pt"
        vision_encoder_path = checkpoint_path / "vision_encoder.pt"
        vision_llm_lora_path = checkpoint_path / "vision_llm_lora"
        vision_llm_full_path = checkpoint_path / "vision_llm"
        vision_lora_path = checkpoint_path / "vision_lora"
        vision_projector_path = checkpoint_path / "vision_projector.pt"
        connector_llm_lora_path = checkpoint_path / "connector_llm_lora"
        tokenizer_dir = checkpoint_path / "tokenizer"
        vision_processor_dir = checkpoint_path / "vision_processor"

        if not meta_path.exists():
            raise FileNotFoundError(f"Missing checkpoint metadata: {meta_path}")
        if not lora_dir.exists():
            raise FileNotFoundError(f"Missing LoRA adapter directory: {lora_dir}")
        if not projector_path.exists():
            raise FileNotFoundError(f"Missing projector weights: {projector_path}")

        meta: Dict[str, Any] = {}
        adapter_config: Dict[str, Any] = {}
        meta = json.loads(meta_path.read_text())
        adapter_config_path = lora_dir / "adapter_config.json"
        if adapter_config_path.exists():
            adapter_config = json.loads(adapter_config_path.read_text())

        resolved_llm_name = llm_name or meta.get("llm_name") or adapter_config.get("base_model_name_or_path")
        meta_llm_name = meta.get("llm_name")
        resolved_vision_backend = (
            vision_backend
            or meta.get("vision_backend_override")
            or ("patch_llm" if meta.get("vision_backend") == "patch_llm" else None)
        )
        resolved_vision_name = vision_name or meta.get("vision_name")
        if (
            resolved_vision_backend == "patch_llm"
            and vision_name is None
            and vision_llm_full_path.exists()
        ):
            resolved_vision_name = str(vision_llm_full_path)
        meta_vision_name = meta.get("vision_name")
        projector_state = torch.load(projector_path, map_location="cpu")
        inferred_projector_type = _infer_projector_type_from_state_dict(projector_state)
        resolved_projector_type = (
            projector_type
            or meta.get("projector_type")
            or inferred_projector_type
            or "qformer"
        )
        resolved_projector_num_queries = projector_num_queries
        if resolved_projector_num_queries is None:
            resolved_projector_num_queries = meta.get("projector_num_queries")
        if resolved_projector_num_queries is None and torch.is_tensor(projector_state.get("query_tokens")):
            resolved_projector_num_queries = int(projector_state["query_tokens"].shape[1])
        if resolved_projector_num_queries is None:
            resolved_projector_num_queries = 32
        resolved_connector_llm_name = connector_llm_name or meta.get("connector_llm_name")
        resolved_connector_llm_dtype = (
            connector_llm_dtype
            or meta.get("connector_llm_dtype")
            or "auto"
        )
        resolved_connector_use_text_prefix = connector_use_text_prefix
        if resolved_connector_use_text_prefix is None:
            resolved_connector_use_text_prefix = bool(
                meta.get(
                    "connector_use_text_prefix",
                    meta.get("connector_uses_text_prefix", False),
                )
            )
        resolved_freeze_connector_llm = freeze_connector_llm
        if resolved_freeze_connector_llm is None:
            resolved_freeze_connector_llm = bool(meta.get("freeze_connector_llm", True))
        resolved_connector_llm_use_qlora = connector_llm_use_qlora
        if resolved_connector_llm_use_qlora is None:
            resolved_connector_llm_use_qlora = bool(
                meta.get("connector_llm_use_qlora", False)
                or connector_llm_lora_path.exists()
            )
        resolved_connector_lora_r = connector_lora_r
        if resolved_connector_lora_r is None:
            resolved_connector_lora_r = int(meta.get("connector_lora_r", 16))
        resolved_connector_lora_alpha = connector_lora_alpha
        if resolved_connector_lora_alpha is None:
            resolved_connector_lora_alpha = int(meta.get("connector_lora_alpha", 32))
        resolved_connector_lora_dropout = connector_lora_dropout
        if resolved_connector_lora_dropout is None:
            resolved_connector_lora_dropout = float(
                meta.get("connector_lora_dropout", 0.05)
            )
        resolved_vision_image_size = vision_image_size
        if resolved_vision_image_size is None:
            resolved_vision_image_size = meta.get("vision_max_side_size")
        if resolved_vision_image_size is None:
            resolved_vision_image_size = meta.get("vision_image_size")
        if resolved_vision_image_size is None:
            resolved_vision_image_size = DEFAULT_PATCH_VISION_IMAGE_SIZE
        resolved_vision_patch_size = vision_patch_size
        if resolved_vision_patch_size is None:
            resolved_vision_patch_size = meta.get(
                "vision_patch_size",
                DEFAULT_PATCH_VISION_PATCH_SIZE,
            )
        resolved_vision_conv_hidden_size = vision_conv_hidden_size
        if resolved_vision_conv_hidden_size is None:
            resolved_vision_conv_hidden_size = meta.get("vision_conv_hidden_size", 256)
        resolved_vision_use_maxpool = vision_use_maxpool
        if resolved_vision_use_maxpool is None:
            resolved_vision_use_maxpool = bool(meta.get("vision_use_maxpool", False))
        resolved_vision_llm_use_qlora = vision_llm_use_qlora
        if resolved_vision_llm_use_qlora is None:
            resolved_vision_llm_use_qlora = bool(
                meta.get("vision_llm_use_qlora", False) or vision_llm_lora_path.exists()
            )
        resolved_vision_llm_dtype = vision_llm_dtype or meta.get("vision_llm_dtype") or "auto"
        resolved_vision_use_lora = vision_use_lora
        if resolved_vision_use_lora is None:
            resolved_vision_use_lora = bool(
                meta.get("vision_use_lora", False) or vision_lora_path.exists()
            )
        resolved_vision_use_qlora = vision_use_qlora
        if resolved_vision_use_qlora is None:
            resolved_vision_use_qlora = bool(meta.get("vision_use_qlora", False))
        resolved_vision_lora_r = vision_lora_r
        if resolved_vision_lora_r is None:
            resolved_vision_lora_r = int(meta.get("vision_lora_r", 16))
        resolved_vision_lora_alpha = vision_lora_alpha
        if resolved_vision_lora_alpha is None:
            resolved_vision_lora_alpha = int(meta.get("vision_lora_alpha", 32))
        resolved_vision_lora_dropout = vision_lora_dropout
        if resolved_vision_lora_dropout is None:
            resolved_vision_lora_dropout = float(meta.get("vision_lora_dropout", 0.05))
        resolved_num_lora_layers = num_lora_layers
        if resolved_num_lora_layers is None:
            resolved_num_lora_layers = meta.get("num_lora_layers", -1)
        resolved_chat_template_mode = _normalize_chat_template_mode(
            chat_template_mode or meta.get("chat_template_mode", "tokenizer")
        )
        resolved_max_image_side = max_image_side
        if resolved_max_image_side is None:
            resolved_max_image_side = meta.get("max_image_side")
        resolved_normalize_visual_embeddings = normalize_visual_embeddings
        if resolved_normalize_visual_embeddings is None:
            resolved_normalize_visual_embeddings = bool(
                meta.get("normalize_visual_embeddings", False)
            )
        resolved_enable_vl_experts = enable_vl_experts
        if resolved_enable_vl_experts is None:
            resolved_enable_vl_experts = bool(
                meta.get("enable_vl_experts", False) or vl_experts_path.exists()
            )
        if resolved_enable_vl_experts and not vl_experts_path.exists():
            raise FileNotFoundError(f"Missing VL expert weights: {vl_experts_path}")
        resolved_vl_expert_layers = vl_expert_layers
        if resolved_vl_expert_layers is None:
            resolved_vl_expert_layers = meta.get("vl_expert_layers_config")
        if resolved_vl_expert_layers is None and bool(resolved_enable_vl_experts):
            resolved_vl_expert_layers = meta.get("vl_expert_layers")

        if resolved_llm_name is None:
            raise RuntimeError(
                "Could not infer base LLM path from checkpoint metadata or adapter config. "
                "Pass llm_name explicitly."
            )
        if resolved_vision_name is None:
            raise RuntimeError(
                "Could not infer vision model path from checkpoint metadata. "
                "Pass vision_name explicitly."
            )
        if resolved_vision_backend == "patch_llm" and not vision_encoder_path.exists():
            raise FileNotFoundError(
                f"Missing Patch-LLM vision encoder weights: {vision_encoder_path}"
            )
        if (
            resolved_vision_backend == "patch_llm"
            and bool(resolved_vision_llm_use_qlora)
            and not vision_llm_lora_path.exists()
        ):
            raise FileNotFoundError(
                f"Missing Patch-LLM vision QLoRA adapter directory: {vision_llm_lora_path}"
            )
        if resolved_projector_type == "llm" and resolved_connector_llm_name is None:
            raise RuntimeError(
                "Could not infer connector LLM path from checkpoint metadata. "
                "Pass connector_llm_name explicitly."
            )
        if (
            resolved_projector_type == "llm"
            and bool(resolved_connector_llm_use_qlora)
            and not connector_llm_lora_path.exists()
        ):
            raise FileNotFoundError(
                f"Missing connector LLM QLoRA adapter directory: {connector_llm_lora_path}"
            )
        if bool(resolved_vision_use_lora) and not vision_lora_path.exists():
            raise FileNotFoundError(
                f"Missing donor vision LoRA adapter directory: {vision_lora_path}"
            )

        if vision_name is not None and meta_vision_name is not None and vision_name != meta_vision_name:
            print(
                "Warning: explicit vision_name differs from checkpoint metadata. "
                f"meta_vision_name={meta_vision_name}, explicit_vision_name={vision_name}. "
                "Projector weights will only load if the checkpoint was trained with the same vision donor."
            )
        if llm_name is not None and meta_llm_name is not None and llm_name != meta_llm_name:
            print(
                "Warning: explicit llm_name differs from checkpoint metadata. "
                f"meta_llm_name={meta_llm_name}, explicit_llm_name={llm_name}. "
                "LoRA adapters should be loaded on the same base LLM checkpoint they were trained on."
            )

        super().__init__(
            llm_name=resolved_llm_name,
            vision_name=resolved_vision_name,
            tokenizer_name=str(tokenizer_dir) if tokenizer_dir.exists() else resolved_llm_name,
            chat_template_mode=resolved_chat_template_mode,
            max_image_side=resolved_max_image_side,
            use_4bit_llm=use_4bit_llm,
            freeze_vision=True,
            vision_backend=resolved_vision_backend,
            vision_image_size=int(resolved_vision_image_size),
            vision_patch_size=int(resolved_vision_patch_size),
            vision_conv_hidden_size=int(resolved_vision_conv_hidden_size),
            vision_use_maxpool=bool(resolved_vision_use_maxpool),
            vision_llm_use_qlora=bool(resolved_vision_llm_use_qlora),
            vision_llm_dtype=str(resolved_vision_llm_dtype),
            vision_use_lora=bool(resolved_vision_use_lora),
            vision_use_qlora=bool(resolved_vision_use_qlora),
            vision_lora_r=int(resolved_vision_lora_r),
            vision_lora_alpha=int(resolved_vision_lora_alpha),
            vision_lora_dropout=float(resolved_vision_lora_dropout),
            vision_lora_path=str(vision_lora_path) if vision_lora_path.exists() else None,
            vision_projector_path=(
                str(vision_projector_path)
                if vision_projector_path.exists()
                else None
            ),
            vision_encoder_path=str(vision_encoder_path) if vision_encoder_path.exists() else None,
            vision_llm_lora_path=(
                str(vision_llm_lora_path)
                if vision_llm_lora_path.exists()
                else None
            ),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_lora_layers=int(resolved_num_lora_layers),
            normalize_visual_embeddings=bool(resolved_normalize_visual_embeddings),
            lora_path=str(lora_dir),
            projector_path=str(projector_path) if projector_path.exists() else None,
            projector_type=resolved_projector_type,
            projector_num_queries=int(resolved_projector_num_queries),
            connector_llm_name=resolved_connector_llm_name,
            connector_llm_dtype=str(resolved_connector_llm_dtype),
            connector_use_text_prefix=bool(resolved_connector_use_text_prefix),
            freeze_connector_llm=bool(resolved_freeze_connector_llm),
            connector_llm_use_qlora=bool(resolved_connector_llm_use_qlora),
            connector_lora_r=int(resolved_connector_lora_r),
            connector_lora_alpha=int(resolved_connector_lora_alpha),
            connector_lora_dropout=float(resolved_connector_lora_dropout),
            connector_llm_lora_path=(
                str(connector_llm_lora_path)
                if connector_llm_lora_path.exists()
                else None
            ),
            enable_vl_experts=bool(resolved_enable_vl_experts),
            vl_expert_layers=resolved_vl_expert_layers,
            vl_experts_path=(
                str(vl_experts_path) if bool(resolved_enable_vl_experts) else None
            ),
            device=device,
        )

        if vision_processor_dir.exists():
            self.vision_processor = AutoProcessor.from_pretrained(str(vision_processor_dir))

        self.eval()

    @torch.inference_mode()
    def inference(
        self,
        text: str,
        image: Optional[Any] = None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_token_ids: Optional[List[int]] = None,
        no_repeat_ngram_size: int = 6,
        stop_repeated_ngram_size: int = 8,
        stop_repeated_ngram_occurrences: int = 3,
    ) -> str:
        prepared = self._build_generation_inputs(text=text, image=image)

        device = self.llm.get_input_embeddings().weight.device
        attention_mask = prepared["attention_mask"].to(device)
        generation_kwargs = {
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": (
                stop_token_ids
                if stop_token_ids is not None
                else _chat_stop_token_ids(self.tokenizer)
            ),
            "repetition_penalty": repetition_penalty,
        }
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
        if stop_repeated_ngram_size and stop_repeated_ngram_size > 0:
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [
                    _RepeatedNGramStoppingCriteria(
                        start_length=prepared["prompt_length"],
                        ngram_size=int(stop_repeated_ngram_size),
                        max_occurrences=int(stop_repeated_ngram_occurrences),
                    )
                ]
            )
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        self._set_vl_expert_visual_token_mask(prepared.get("visual_token_mask"))
        try:
            if "inputs_embeds" in prepared:
                outputs = self.llm.generate(
                    input_ids=prepared["input_ids"].to(device),
                    inputs_embeds=prepared["inputs_embeds"].to(device),
                    **generation_kwargs,
                )
            else:
                outputs = self.llm.generate(
                    input_ids=prepared["input_ids"].to(device),
                    **generation_kwargs,
                )
        finally:
            self._set_vl_expert_visual_token_mask(None)

        prompt_length = int(prepared["prompt_length"])
        generated_ids = outputs[0, prompt_length:] if outputs.size(1) > prompt_length else outputs[0]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
