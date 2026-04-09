import json
from contextlib import nullcontext
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
from peft import LoraConfig, get_peft_model


IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100


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


class GigaChatVL(nn.Module):
    def __init__(
        self,
        llm_name: str = "ai-sage/GigaChat3.1-10B-A1.8B-bf16",
        vision_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        use_4bit_llm: bool = True,
        freeze_vision: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        self.freeze_vision = freeze_vision
        self.vision_name = vision_name
        self.quantization_method = None
        self.is_loaded_in_4bit = False
        self.hf_device_map = {}

        # LLM tokenizer
        tokenizer_dir = Path(llm_name)
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

            bos_token = special_tokens_map.get(
                "bos_token", tokenizer_config.get("bos_token", "<s>")
            )
            eos_token = special_tokens_map.get(
                "eos_token", tokenizer_config.get("eos_token", "</s>")
            )

            if isinstance(bos_token, dict):
                bos_token = bos_token.get("content")
            if isinstance(eos_token, dict):
                eos_token = eos_token.get("content")

            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_file),
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=eos_token,
                model_max_length=tokenizer_config.get("model_max_length"),
                clean_up_tokenization_spaces=tokenizer_config.get(
                    "clean_up_tokenization_spaces", True
                ),
            )
        else:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(llm_name)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            dtype=llm_dtype,
            quantization_config=llm_quant_config,
            device_map=llm_device_map,
            low_cpu_mem_usage=True,
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

        self._load_vision_backend(vision_name)

        # Connector
        self.projector = MLPProjector(
            vision_dim=self.vision_hidden_size,
            llm_dim=self.llm_hidden_size,
        )
        self.projector.to(
            device=self.llm.get_input_embeddings().weight.device,
            dtype=self.llm.get_input_embeddings().weight.dtype,
        )

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
        for x in chunks:
            y = self.projector(
                x.to(
                    device=self.projector.net[0].weight.device,
                    dtype=self.projector.net[0].weight.dtype,
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
        for x in chunks:
            y = self.projector(
                x.to(
                    device=self.projector.net[0].weight.device,
                    dtype=self.projector.net[0].weight.dtype,
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
        for i, n in enumerate(token_counts):
            if n <= 0:
                raise RuntimeError(
                    f"Could not infer Gemma4 image token count for sample {i}."
                )

            x = image_hidden_states[i, :n]
            y = self.projector(
                x.to(
                    device=self.projector.net[0].weight.device,
                    dtype=self.projector.net[0].weight.dtype,
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
