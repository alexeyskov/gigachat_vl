import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image

from transformers import (
    AutoConfig,
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


def _build_chat_prompt(tokenizer, text: str, include_image: bool) -> str:
    user_text = f"{IMAGE_TOKEN}\n{text}" if include_image else text

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
        lora_path: Optional[str] = None,
        projector_path: Optional[str] = None,
    ):
        super().__init__()

        self.llm_name = llm_name
        self.freeze_vision = freeze_vision
        self.vision_name = vision_name
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

        if lora_path is not None:
            self.llm = PeftModel.from_pretrained(
                self.llm,
                lora_path,
                is_trainable=False,
            )
        else:
            lora_cfg = LoraConfig(
                task_type="CAUSAL_LM",
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules="all-linear",
            )
            self.llm = get_peft_model(self.llm, lora_cfg)
        self.llm_hidden_size = self.llm.config.hidden_size

        # Vision side
        self.vision_processor = None
        self.vision_backend = None
        self.vision_source_model = None
        self.vision_tower = None
        self.vision_projector = None
        self.vision_hidden_size = None
        self.gemma_image_placeholder = None
        self.projector_type = "qformer"

        self._load_vision_backend(vision_name)

        # Connector
        self.projector = QFormerProjector(
            vision_dim=self.vision_hidden_size,
            llm_dim=self.llm_hidden_size,
        )
        self.projector.to(
            device=self.llm.get_input_embeddings().weight.device,
            dtype=self.llm.get_input_embeddings().weight.dtype,
        )
        if projector_path is not None:
            projector_state = torch.load(projector_path, map_location="cpu")
            try:
                self.projector.load_state_dict(projector_state)
            except RuntimeError as e:
                raise RuntimeError(
                    "Failed to load projector weights into QFormerProjector. "
                    "This usually means the checkpoint was trained with a different "
                    "projector architecture, for example the older MLPProjector."
                ) from e

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
            self.vision_source_model = None
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

        # Gemma4
        if model_type == "gemma4":
            self.vision_backend = "gemma4"
            self.vision_processor = AutoProcessor.from_pretrained(vision_name)

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
            self.vision_source_model = None
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
            "Supported: qwen2_5_vl, qwen3_vl, qwen3_5, qwen3_5_vl, gemma4."
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

        # Gemma4
        if self.vision_backend == "gemma4":
            prompts = [self.gemma_image_placeholder for _ in images]
            batch = self.vision_processor(
                images=images,
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

        ctx = torch.no_grad() if self.freeze_vision else nullcontext()
        with ctx:
            vision_outputs = self.vision_tower(
                pixel_values=prepared["pixel_values"],
                return_dict=True,
            )
            image_hidden_states = self.vision_projector(vision_outputs.last_hidden_state)

        # Expected shape: [B, num_images, seq, H]
        if image_hidden_states.dim() == 4:
            if image_hidden_states.size(1) != 1:
                raise NotImplementedError(
                    "Only 1 image per sample is supported for Gemma4 backend."
                )
            image_hidden_states = image_hidden_states[:, 0]
        elif image_hidden_states.dim() != 3:
            raise RuntimeError(
                f"Unexpected Gemma4 image_hidden_states shape: "
                f"{tuple(image_hidden_states.shape)}"
            )

        image_token_id = self.vision_processor.tokenizer.convert_tokens_to_ids(
            self.gemma_image_placeholder
        )
        token_counts = (prepared["input_ids"] == image_token_id).sum(dim=1).tolist()
        token_counts = [int(x) for x in token_counts]

        out = []
        projector_device, projector_dtype = _module_device_dtype(self.projector)
        for i, n in enumerate(token_counts):
            if n <= 0:
                raise RuntimeError(
                    f"Could not infer Gemma4 image token count for sample {i}."
                )

            x = image_hidden_states[i, :n]
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

        if self.vision_backend == "gemma4":
            return self._encode_images_gemma4(vision_kwargs)

        raise RuntimeError(f"Unknown vision backend: {self.vision_backend}")

    def _merge_text_and_image(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        image_features_per_sample: List[torch.Tensor],
    ):
        embed_tokens = self.llm.get_input_embeddings()
        base_embeds = embed_tokens(input_ids)

        batch_embeds = []
        batch_masks = []
        batch_labels = []
        max_len = 0

        for i in range(input_ids.size(0)):
            seq_len_i = int(attention_mask[i].sum().item())

            ids_i = input_ids[i, :seq_len_i]
            embeds_i = base_embeds[i, :seq_len_i]

            image_pos = (ids_i == self.image_token_id).nonzero(as_tuple=False).flatten()
            if image_pos.numel() != 1:
                raise ValueError(
                    f"Every sample must contain exactly one {IMAGE_TOKEN}. "
                    f"Found {image_pos.numel()} in sample {i}."
                )

            pos = int(image_pos.item())
            img_feats = image_features_per_sample[i]

            merged_embeds = torch.cat(
                [
                    embeds_i[:pos],
                    img_feats,
                    embeds_i[pos + 1 :],
                ],
                dim=0,
            )

            merged_mask = torch.ones(
                merged_embeds.size(0),
                dtype=attention_mask.dtype,
                device=merged_embeds.device,
            )

            if labels is not None:
                labels_i = labels[i, :seq_len_i]
                merged_labels = torch.cat(
                    [
                        labels_i[:pos],
                        torch.full(
                            (img_feats.size(0),),
                            IGNORE_INDEX,
                            dtype=labels_i.dtype,
                            device=labels_i.device,
                        ),
                        labels_i[pos + 1 :],
                    ],
                    dim=0,
                )
            else:
                merged_labels = None

            batch_embeds.append(merged_embeds)
            batch_masks.append(merged_mask)
            batch_labels.append(merged_labels)
            max_len = max(max_len, merged_embeds.size(0))

        padded_embeds = []
        padded_masks = []
        padded_labels = []

        hidden_size = batch_embeds[0].size(-1)
        embed_dtype = batch_embeds[0].dtype
        embed_device = batch_embeds[0].device

        for embeds_i, mask_i, labels_i in zip(batch_embeds, batch_masks, batch_labels):
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

        inputs_embeds = torch.stack(padded_embeds, dim=0)
        attention_mask = torch.stack(padded_masks, dim=0)

        if labels is not None:
            labels = torch.stack(padded_labels, dim=0)
        else:
            labels = None

        return inputs_embeds, attention_mask, labels

    def _build_generation_inputs(
        self,
        text: str,
        image: Optional[Image.Image],
    ):
        include_image = image is not None
        prompt = _build_chat_prompt(self.tokenizer, text, include_image=include_image)
        text_batch = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = text_batch["input_ids"]
        attention_mask = text_batch["attention_mask"]
        llm_device = self.llm.get_input_embeddings().weight.device

        if image is None:
            return {
                "input_ids": input_ids.to(llm_device),
                "attention_mask": attention_mask.to(llm_device),
                "prompt_length": int(attention_mask[0].sum().item()),
            }

        vision_batch = self.prepare_vision_inputs([image])
        image_features_per_sample = self.encode_images(**vision_batch)
        input_ids = input_ids.to(llm_device)
        attention_mask = attention_mask.to(llm_device)

        embed_tokens = self.llm.get_input_embeddings()
        base_embeds = embed_tokens(input_ids)
        img_feats = image_features_per_sample[0]

        image_pos = (input_ids[0] == self.image_token_id).nonzero(as_tuple=False).flatten()
        if image_pos.numel() != 1:
            raise ValueError(
                f"Prompt must contain exactly one {IMAGE_TOKEN}. Found {image_pos.numel()}."
            )

        pos = int(image_pos.item())
        expanded_input_ids = torch.cat(
            [
                input_ids[0, :pos],
                torch.full(
                    (img_feats.size(0),),
                    self.image_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
                input_ids[0, pos + 1 :],
            ],
            dim=0,
        ).unsqueeze(0)

        inputs_embeds = torch.cat(
            [
                base_embeds[0, :pos],
                img_feats,
                base_embeds[0, pos + 1 :],
            ],
            dim=0,
        ).unsqueeze(0)

        attention_mask = torch.ones(
            expanded_input_ids.shape,
            dtype=attention_mask.dtype,
            device=expanded_input_ids.device,
        )

        return {
            "input_ids": expanded_input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        vision_pixel_values: Optional[torch.Tensor] = None,
        vision_image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        vision_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k.startswith("vision_") and v is not None
        }

        if self.vision_backend == "qwen2_5_vl":
            image_features_per_sample = self.encode_images(
                vision_pixel_values=vision_pixel_values,
                vision_image_grid_thw=vision_image_grid_thw,
            )
        elif self.vision_backend in {"qwen3_vl", "qwen3_5", "qwen3_5_vl"}:
            image_features_per_sample = self.encode_images(
                vision_pixel_values=vision_pixel_values,
                vision_image_grid_thw=vision_image_grid_thw,
            )
        elif self.vision_backend == "gemma4":
            image_features_per_sample = self.encode_images(**vision_kwargs)
        else:
            raise RuntimeError(f"Unknown vision backend: {self.vision_backend}")

        inputs_embeds, attention_mask, labels = self._merge_text_and_image(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_features_per_sample=image_features_per_sample,
        )

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
    ):
        checkpoint_path = Path(checkpoint_dir)
        meta_path = checkpoint_path / "vlm_meta.json"
        lora_dir = checkpoint_path / "llm_lora"
        projector_path = checkpoint_path / "projector.pt"
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

        super().__init__(
            llm_name=resolved_llm_name,
            vision_name=resolved_vision_name,
            tokenizer_name=str(tokenizer_dir) if tokenizer_dir.exists() else resolved_llm_name,
            use_4bit_llm=use_4bit_llm,
            freeze_vision=True,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_path=str(lora_dir),
            projector_path=str(projector_path) if projector_path.exists() else None,
        )

        if vision_processor_dir.exists():
            self.vision_processor = AutoProcessor.from_pretrained(str(vision_processor_dir))

        self.eval()

    @torch.inference_mode()
    def inference(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        if image is not None and not isinstance(image, Image.Image):
            raise TypeError(f"image must be PIL.Image.Image or None, got {type(image)}")

        if image is not None:
            image = image.convert("RGB")

        prepared = self._build_generation_inputs(text=text, image=image)

        device = self.llm.get_input_embeddings().weight.device
        attention_mask = prepared["attention_mask"].to(device)
        generation_kwargs = {
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
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
