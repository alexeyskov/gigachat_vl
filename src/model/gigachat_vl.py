import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
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
)
from transformers.utils.quantization_config import QuantizationMethod
from peft import LoraConfig, PeftModel, get_peft_model


IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
_MISTRAL_FIXED_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def single_device_map():
    if torch.cuda.is_available():
        return {"": torch.cuda.current_device()}
    return None


def single_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _has_4bit_parameters(module: nn.Module) -> bool:
    for param in module.parameters():
        if param.__class__.__name__ == "Params4bit":
            return True
    return False


def _prepare_model_for_kbit_training_no_fp32_cast(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    model_path = Path(model_dir)
    vision_tower = Qwen3_5VisionModel(cfg.vision_config)
    vision_state = _load_prefixed_safetensor_state_dict_from_candidates(
        model_path,
        prefixes=["model.visual.", "visual."],
    )
    vision_tower.load_state_dict(vision_state, strict=True)
    del vision_state
    vision_tower.to(device=device, dtype=dtype)
    return vision_tower


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


def _token_content(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("content")
    if isinstance(value, list):
        return [_token_content(v) for v in value]
    return value


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

        chat_template = tokenizer_config.get("chat_template")
        if chat_template is not None:
            tokenizer.chat_template = chat_template
            if hasattr(tokenizer, "init_kwargs"):
                tokenizer.init_kwargs["chat_template"] = chat_template

        tokenizer.padding_side = tokenizer_config.get("padding_side", "right")
        tokenizer.truncation_side = tokenizer_config.get("truncation_side", "right")

        if _should_fix_mistral_regex(tokenizer_name, tokenizer_config):
            _maybe_patch_mistral_regex(tokenizer)
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)

    tokenizer.padding_side = getattr(tokenizer, "padding_side", "right") or "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _build_chat_prompt(tokenizer, text: str, num_images: int = 1) -> str:
    if num_images < 0:
        raise ValueError(f"num_images must be >= 0, got {num_images}")

    image_prefix = ""
    if num_images > 0:
        image_prefix = "\n".join([IMAGE_TOKEN] * num_images)

    user_text = text if not image_prefix else f"{image_prefix}\n{text}"

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


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(module.parameters())
    return param.device, param.dtype


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


def _infer_projector_type_from_state_dict(state_dict: Dict[str, Any]) -> Optional[str]:
    keys = set(state_dict.keys())
    if any(key.startswith("net.") for key in keys):
        return "mlp"
    if "vision_proj.weight" in keys or "query_tokens" in keys:
        return "qformer"
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
    def __init__(
        self,
        llm_name: str = "ai-sage/GigaChat3.1-10B-A1.8B-bf16",
        vision_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        tokenizer_name: Optional[str] = None,
        use_4bit_llm: bool = True,
        freeze_vision: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_lora_layers: int = -1,
        lora_path: Optional[str] = None,
        projector_path: Optional[str] = None,
        projector_type: str = "qformer",
        projector_num_queries: int = 32,
        enable_vl_experts: bool = False,
        vl_expert_layers: Optional[List[int]] = None,
        vl_experts_path: Optional[str] = None,
    ):
        super().__init__()

        self.llm_name = llm_name
        self.freeze_vision = freeze_vision
        self.vision_name = vision_name
        self.num_lora_layers = num_lora_layers
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
        if use_4bit_llm and torch.cuda.is_available():
            llm_quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        llm_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        llm_device_map = single_device_map() if llm_quant_config is not None else None

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
                self.hf_device_map = {"llm": torch.cuda.current_device()}
                self.llm = _prepare_model_for_kbit_training_no_fp32_cast(self.llm)
            else:
                print(
                    "Warning: 4-bit quantization was requested, but the loaded LLM "
                    "does not expose 4-bit parameters. Skipping "
                    "k-bit preparation to avoid fp32 OOM."
                )
                for param in self.llm.parameters():
                    param.requires_grad = False

        self.llm_hidden_size = self.llm.config.hidden_size

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
            lora_cfg = LoraConfig(
                task_type="CAUSAL_LM",
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_modules,
                layers_to_transform=layers_to_transform,
                layers_pattern="layers",
            )
            self.llm = get_peft_model(self.llm, lora_cfg)
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
        if self.projector_type not in {"qformer", "mlp"}:
            raise ValueError(
                f"Unsupported projector_type={projector_type}. "
                "Supported values: 'qformer', 'mlp'."
            )

        self._load_vision_backend(vision_name)

        # Connector
        if self.projector_type == "qformer":
            self.projector = QFormerProjector(
                vision_dim=self.vision_hidden_size,
                llm_dim=self.llm_hidden_size,
                num_queries=self.projector_num_queries,
            )
        else:
            self.projector = MLPProjector(
                vision_dim=self.vision_hidden_size,
                llm_dim=self.llm_hidden_size,
            )
        self.projector.to(
            device=self.llm.get_input_embeddings().weight.device,
            dtype=self.llm.get_input_embeddings().weight.dtype,
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
                self.projector.load_state_dict(projector_state)
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

                current_weight = getattr(getattr(self.projector, "vision_proj", None), "weight", None)
                if current_weight is None and hasattr(self.projector, "net"):
                    try:
                        current_weight = self.projector.net[0].weight
                    except Exception:
                        current_weight = None
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
            "enable_vl_experts": self.enable_vl_experts,
            "vl_expert_layers": self.vl_expert_layer_indices,
            "vl_expert_layers_config": self.vl_expert_layers,
            "projector_type": getattr(self, "projector_type", None),
            "projector_num_queries": getattr(self.projector, "num_queries", None),
            "vision_hidden_size": self.vision_hidden_size,
            "vision_backend": self.vision_backend,
            "vision_name": self.vision_name,
            "freeze_vision": self.freeze_vision,
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
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_lora_layers: Optional[int] = None,
        projector_type: Optional[str] = None,
        projector_num_queries: Optional[int] = None,
        enable_vl_experts: Optional[bool] = None,
        vl_expert_layers: Optional[List[int]] = None,
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
        vl_experts_path = _resolve_checkpoint_sidecar_path(checkpoint_path, "vl_experts.pt")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        resolved_llm_name = llm_name or meta.get("llm_name")
        resolved_vision_name = vision_name or meta.get("vision_name")
        resolved_projector_type = projector_type or meta.get("projector_type") or "qformer"
        resolved_num_lora_layers = num_lora_layers
        if resolved_num_lora_layers is None:
            resolved_num_lora_layers = meta.get("num_lora_layers", -1)
        resolved_projector_num_queries = projector_num_queries
        if resolved_projector_num_queries is None:
            resolved_projector_num_queries = meta.get("projector_num_queries")
        if resolved_projector_num_queries is None:
            resolved_projector_num_queries = 32
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

        model = cls(
            llm_name=resolved_llm_name,
            vision_name=resolved_vision_name,
            tokenizer_name=str(tokenizer_dir) if tokenizer_dir is not None else resolved_llm_name,
            use_4bit_llm=use_4bit_llm,
            freeze_vision=freeze_vision,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_lora_layers=int(resolved_num_lora_layers),
            projector_type=resolved_projector_type,
            projector_num_queries=int(resolved_projector_num_queries),
            enable_vl_experts=bool(resolved_enable_vl_experts),
            vl_expert_layers=resolved_vl_expert_layers,
            vl_experts_path=(
                str(vl_experts_path)
                if bool(resolved_enable_vl_experts) and vl_experts_path is not None
                else None
            ),
        )

        if vision_processor_dir is not None:
            model.vision_processor = AutoProcessor.from_pretrained(str(vision_processor_dir))

        return model

    def _load_vision_backend(self, vision_name: str):
        cfg = AutoConfig.from_pretrained(vision_name, trust_remote_code=False)
        model_type = getattr(cfg, "model_type", None)

        vision_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        vision_device = single_device()

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

            if self.freeze_vision:
                self.vision_tower.requires_grad_(False)
            else:
                self.vision_tower.requires_grad_(True)
            return

        # Qwen3 / Qwen3.5 family
        if model_type in {"qwen3_vl", "qwen3_5", "qwen3_5_vl"}:
            self.vision_backend = model_type
            self.vision_processor = AutoProcessor.from_pretrained(vision_name)
            self.vision_source_model = None

            if model_type == "qwen3_5":
                try:
                    self.vision_tower = _load_qwen35_vision_module(
                        model_dir=vision_name,
                        cfg=cfg,
                        device=vision_device,
                        dtype=vision_dtype,
                    )
                except Exception as e:
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

            if self.vision_tower is not None:
                self.vision_tower.requires_grad_(not self.freeze_vision)
            if self.vision_projector is not None:
                self.vision_projector.requires_grad_(not self.freeze_vision)
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
            self.vision_tower.requires_grad_(not self.freeze_vision)
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

            if self.vision_tower is not None:
                self.vision_tower.requires_grad_(not self.freeze_vision)
            if self.vision_projector is not None:
                self.vision_projector.requires_grad_(not self.freeze_vision)
            return

        raise ValueError(
            f"Unsupported vision backend: {vision_name} (model_type={model_type}). "
            "Supported: qwen2_5_vl, qwen3_vl, qwen3_5, qwen3_5_vl, "
            "siglip, siglip2, gemma4."
        )

    def prepare_vision_inputs(self, images: List[Any]) -> Dict[str, torch.Tensor]:
        # Qwen family
        if self.vision_backend in {"qwen2_5_vl", "qwen3_vl", "qwen3_5", "qwen3_5_vl"}:
            pixel_values_parts = []
            grid_parts = []

            for image in images:
                if not isinstance(image, Image.Image):
                    raise TypeError(
                        f"Qwen vision backend expects PIL images in collator, got {type(image)}"
                    )

                image_batch = _prepare_single_qwen_image(self.vision_processor, image)
                pixel_values = image_batch["pixel_values"]
                grid = image_batch["grid"]

                pixel_values_parts.append(pixel_values)
                grid_parts.append(grid)

            max_rows = max(x.size(0) for x in pixel_values_parts)
            feat_dim = pixel_values_parts[0].size(1)
            padded_pixel_values = []
            for x in pixel_values_parts:
                pad_rows = max_rows - x.size(0)
                if pad_rows > 0:
                    pad = torch.zeros(
                        pad_rows,
                        feat_dim,
                        dtype=x.dtype,
                    )
                    x = torch.cat([x, pad], dim=0)
                padded_pixel_values.append(x)

            return {
                "vision_pixel_values": torch.stack(padded_pixel_values, dim=0),
                "vision_image_grid_thw": torch.cat(grid_parts, dim=0),
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

        raise RuntimeError(f"Unknown vision backend: {self.vision_backend}")

    def _encode_images_qwen25(
        self,
        vision_pixel_values,
        vision_image_grid_thw,
    ) -> List[torch.Tensor]:
        if torch.is_tensor(vision_pixel_values) and vision_pixel_values.dim() == 3:
            out = []
            for i in range(vision_pixel_values.size(0)):
                grid_i = vision_image_grid_thw[i : i + 1]
                rows_i = int(grid_i.prod().item())
                pixel_values_i = vision_pixel_values[i, :rows_i]
                out.extend(
                    self._encode_images_qwen25(
                        vision_pixel_values=pixel_values_i,
                        vision_image_grid_thw=grid_i,
                    )
                )
            return out

        if isinstance(vision_pixel_values, list):
            out = []
            for pixel_values_i, grid_i in zip(vision_pixel_values, vision_image_grid_thw):
                out.extend(
                    self._encode_images_qwen25(
                        vision_pixel_values=pixel_values_i,
                        vision_image_grid_thw=grid_i,
                    )
                )
            return out

        vision_device = next(self.vision_tower.parameters()).device
        vision_dtype = next(self.vision_tower.parameters()).dtype

        pixel_values = _maybe_tensor_to(vision_pixel_values, vision_device, vision_dtype)
        image_grid_thw = _maybe_tensor_to(vision_image_grid_thw, vision_device)
        expected_rows = int(image_grid_thw.prod().item())

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
        chunks = list(torch.split(feats, token_counts, dim=0))

        out = []
        projector_device, projector_dtype = _module_device_dtype(self.projector)
        for x in chunks:
            y = self.projector(
                x.to(
                    device=projector_device,
                    dtype=projector_dtype,
                )
            )
            y = y.to(
                device=self.llm.get_input_embeddings().weight.device,
                dtype=self.llm.get_input_embeddings().weight.dtype,
            )
            out.append(y)
        return out

    def _encode_images_qwen3_family(
        self,
        vision_pixel_values,
        vision_image_grid_thw,
    ) -> List[torch.Tensor]:
        if torch.is_tensor(vision_pixel_values) and vision_pixel_values.dim() == 3:
            out = []
            for i in range(vision_pixel_values.size(0)):
                grid_i = vision_image_grid_thw[i : i + 1]
                rows_i = int(grid_i.prod().item())
                pixel_values_i = vision_pixel_values[i, :rows_i]
                out.extend(
                    self._encode_images_qwen3_family(
                        vision_pixel_values=pixel_values_i,
                        vision_image_grid_thw=grid_i,
                    )
                )
            return out

        if isinstance(vision_pixel_values, list):
            out = []
            for pixel_values_i, grid_i in zip(vision_pixel_values, vision_image_grid_thw):
                out.extend(
                    self._encode_images_qwen3_family(
                        vision_pixel_values=pixel_values_i,
                        vision_image_grid_thw=grid_i,
                    )
                )
            return out

        if self.vision_tower is None:
            raise RuntimeError("This Qwen3-family checkpoint does not expose a raw visual tower.")

        vision_device = next(self.vision_tower.parameters()).device
        vision_dtype = next(self.vision_tower.parameters()).dtype

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

        out = []
        projector_device, projector_dtype = _module_device_dtype(self.projector)
        for x in chunks:
            y = self.projector(
                x.to(
                    device=projector_device,
                    dtype=projector_dtype,
                )
            )
            y = y.to(
                device=self.llm.get_input_embeddings().weight.device,
                dtype=self.llm.get_input_embeddings().weight.dtype,
            )
            out.append(y)
        return out

    def _encode_images_siglip(
        self,
        vision_inputs: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        if self.vision_tower is None:
            raise RuntimeError("SigLIP backend requires a vision_tower.")

        src_device = next(self.vision_tower.parameters()).device
        src_dtype = next(self.vision_tower.parameters()).dtype

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

        out = []
        projector_device, projector_dtype = _module_device_dtype(self.projector)
        for x in chunks:
            y = self.projector(
                x.to(
                    device=projector_device,
                    dtype=projector_dtype,
                )
            )
            y = y.to(
                device=self.llm.get_input_embeddings().weight.device,
                dtype=self.llm.get_input_embeddings().weight.dtype,
            )
            out.append(y)
        return out

    def _encode_images_gemma4(
        self,
        vision_inputs: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        if self.vision_tower is None or self.vision_projector is None:
            raise RuntimeError("Gemma4 backend requires vision_tower and vision_projector.")

        src_device = next(self.vision_tower.parameters()).device
        src_dtype = next(self.vision_tower.parameters()).dtype

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
        projector_device, projector_dtype = _module_device_dtype(self.projector)
        for i, n in enumerate(token_counts):
            if n <= 0:
                raise RuntimeError(
                    f"Could not infer Gemma4 image token count for sample {i}."
                )

            x = chunks[i][:n]
            y = self.projector(
                x.to(
                    device=projector_device,
                    dtype=projector_dtype,
                )
            )
            y = y.to(
                device=self.llm.get_input_embeddings().weight.device,
                dtype=self.llm.get_input_embeddings().weight.dtype,
            )
            out.append(y)
        return out

    def encode_images(
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
            return self._encode_images_qwen25(
                vision_pixel_values=vision_pixel_values,
                vision_image_grid_thw=vision_image_grid_thw,
            )

        if self.vision_backend in {"qwen3_vl", "qwen3_5", "qwen3_5_vl"}:
            if vision_pixel_values is None or vision_image_grid_thw is None:
                raise ValueError(
                    "Qwen3-family backend expects vision_pixel_values and vision_image_grid_thw."
                )
            return self._encode_images_qwen3_family(
                vision_pixel_values=vision_pixel_values,
                vision_image_grid_thw=vision_image_grid_thw,
            )

        if self.vision_backend in {"siglip", "siglip2"}:
            siglip_inputs = dict(vision_kwargs)
            if vision_pixel_values is not None:
                siglip_inputs["vision_pixel_values"] = vision_pixel_values
            if vision_image_grid_thw is not None:
                siglip_inputs["vision_image_grid_thw"] = vision_image_grid_thw
            return self._encode_images_siglip(siglip_inputs)

        if self.vision_backend == "gemma4":
            gemma_inputs = dict(vision_kwargs)
            if vision_pixel_values is not None:
                gemma_inputs["vision_pixel_values"] = vision_pixel_values
            if vision_image_grid_thw is not None:
                gemma_inputs["vision_image_grid_thw"] = vision_image_grid_thw
            return self._encode_images_gemma4(gemma_inputs)

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

        prompt = _build_chat_prompt(self.tokenizer, text, num_images=len(images))
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

        return self

    def gradient_checkpointing_disable(self):
        _toggle_gradient_checkpointing(self.llm, enabled=False)

        if self.vision_tower is not None:
            _toggle_gradient_checkpointing(self.vision_tower, enabled=False)

        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        vision_pixel_values: Optional[torch.Tensor] = None,
        vision_image_grid_thw: Optional[torch.Tensor] = None,
        vision_num_images_per_sample: Optional[torch.Tensor] = None,
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
            )

        if self.vision_backend == "qwen2_5_vl":
            flat_image_features = self.encode_images(
                vision_pixel_values=vision_pixel_values,
                vision_image_grid_thw=vision_image_grid_thw,
            )
        elif self.vision_backend in {"qwen3_vl", "qwen3_5", "qwen3_5_vl"}:
            flat_image_features = self.encode_images(
                vision_pixel_values=vision_pixel_values,
                vision_image_grid_thw=vision_image_grid_thw,
            )
        elif self.vision_backend in {"siglip", "siglip2"}:
            flat_image_features = self.encode_images(**vision_kwargs)
        elif self.vision_backend == "gemma4":
            flat_image_features = self.encode_images(**vision_kwargs)
        else:
            raise RuntimeError(f"Unknown vision backend: {self.vision_backend}")

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
        )


class GigaChatVLForInference(GigaChatVL):
    def __init__(
        self,
        checkpoint_dir: str,
        llm_name: Optional[str] = None,
        vision_name: Optional[str] = None,
        use_4bit_llm: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_lora_layers: Optional[int] = None,
        projector_type: Optional[str] = None,
        projector_num_queries: Optional[int] = None,
        enable_vl_experts: Optional[bool] = None,
        vl_expert_layers: Optional[List[int]] = None,
    ):
        checkpoint_path = Path(checkpoint_dir)
        meta_path = checkpoint_path / "vlm_meta.json"
        lora_dir = checkpoint_path / "llm_lora"
        projector_path = checkpoint_path / "projector.pt"
        vl_experts_path = checkpoint_path / "vl_experts.pt"
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
        resolved_vision_name = vision_name or meta.get("vision_name")
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
        resolved_num_lora_layers = num_lora_layers
        if resolved_num_lora_layers is None:
            resolved_num_lora_layers = meta.get("num_lora_layers", -1)
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

        if vision_name is not None and meta_vision_name is not None and vision_name != meta_vision_name:
            print(
                "Warning: explicit vision_name differs from checkpoint metadata. "
                f"meta_vision_name={meta_vision_name}, explicit_vision_name={vision_name}. "
                "Projector weights will only load if the checkpoint was trained with the same vision donor."
            )

        super().__init__(
            llm_name=resolved_llm_name,
            vision_name=resolved_vision_name,
            tokenizer_name=str(tokenizer_dir) if tokenizer_dir.exists() else resolved_llm_name,
            use_4bit_llm=use_4bit_llm,
            freeze_vision=True,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_lora_layers=int(resolved_num_lora_layers),
            lora_path=str(lora_dir),
            projector_path=str(projector_path) if projector_path.exists() else None,
            projector_type=resolved_projector_type,
            projector_num_queries=int(resolved_projector_num_queries),
            enable_vl_experts=bool(resolved_enable_vl_experts),
            vl_expert_layers=resolved_vl_expert_layers,
            vl_experts_path=(
                str(vl_experts_path) if bool(resolved_enable_vl_experts) else None
            ),
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
    ) -> str:
        prepared = self._build_generation_inputs(text=text, image=image)
        self._set_vl_expert_visual_token_mask(prepared.get("visual_token_mask"))

        device = self.llm.get_input_embeddings().weight.device
        attention_mask = prepared["attention_mask"].to(device)
        generation_kwargs = {
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

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

        generated_ids = outputs[0, prepared["prompt_length"] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
