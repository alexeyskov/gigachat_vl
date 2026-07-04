#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.gigachat_vl import (  # noqa: E402
    _load_gemma4_vision_modules,
    _load_prefixed_safetensor_state_dict_from_candidates,
    _load_qwen35_vision_module,
    _load_siglip_vision_module,
    _maybe_tensor_to,
    _prepare_single_qwen_image,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff", ".tif"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute donor vision features for images and save one .pt file per image. "
            "Supports Qwen2.5-VL, Qwen3.5, SigLIP/SigLIP2, and Gemma4 backends used in this project."
        )
    )
    parser.add_argument(
        "--vision-model",
        required=True,
        help="Path or HF id of the model from which to take the vision encoder.",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory with input images. The script walks it recursively.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where .pt files and manifest.jsonl will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images to process per batch. Default: 8.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Compute dtype for the vision encoder. Default: auto.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use. Default: auto.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files in the output directory.",
    )
    parser.add_argument(
        "--save-input-metadata",
        action="store_true",
        help="Also store backend-specific metadata such as image_grid_thw.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "fp32":
        return torch.float32

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def iter_image_paths(images_dir: Path) -> List[Path]:
    paths = [
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)


def batched(seq: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_qwen25_vision_module(model_dir: str, cfg, device: torch.device, dtype: torch.dtype):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    model_path = Path(model_dir)
    vision_tower = Qwen2_5_VisionTransformerPretrainedModel(cfg.vision_config)
    vision_state = _load_prefixed_safetensor_state_dict_from_candidates(
        model_path,
        prefixes=["model.visual.", "visual."],
    )
    vision_tower.load_state_dict(vision_state, strict=True)
    del vision_state
    vision_tower.to(device=device, dtype=dtype)
    return vision_tower


class VisionFeatureExtractor:
    def __init__(self, vision_model: str, device: torch.device, dtype: torch.dtype):
        self.vision_model = vision_model
        self.device = device
        self.dtype = dtype
        self.config = AutoConfig.from_pretrained(vision_model, trust_remote_code=False)
        self.model_type = getattr(self.config, "model_type", None)
        try:
            self.processor = AutoProcessor.from_pretrained(vision_model)
        except Exception:
            self.processor = AutoImageProcessor.from_pretrained(vision_model)
        self.backend = self._resolve_backend(self.model_type)
        self.vision_tower = None
        self.vision_projector = None
        self._load_modules()

    @staticmethod
    def _resolve_backend(model_type: str) -> str:
        if model_type == "qwen2_5_vl":
            return "qwen2_5_vl"
        if model_type == "qwen3_5":
            return "qwen3_5"
        if model_type in {"siglip", "siglip_vision_model"}:
            return "siglip"
        if model_type in {"siglip2", "siglip2_vision_model"}:
            return "siglip2"
        if model_type == "gemma4":
            return "gemma4"
        raise ValueError(
            f"Unsupported model_type={model_type}. "
            "Supported backends: qwen2_5_vl, qwen3_5, siglip, siglip2, gemma4."
        )

    def _load_modules(self):
        if self.backend == "qwen2_5_vl":
            try:
                self.vision_tower = load_qwen25_vision_module(
                    self.vision_model,
                    self.config,
                    self.device,
                    self.dtype,
                )
            except Exception as e:
                print(
                    "Warning: failed to load Qwen2.5 vision-only module directly "
                    f"from weights, falling back to full donor load: {e}"
                )
                src = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.vision_model,
                    dtype=self.dtype,
                    low_cpu_mem_usage=True,
                )
                self.vision_tower = src.model.visual.to(device=self.device, dtype=self.dtype)
                if hasattr(src, "lm_head"):
                    del src.lm_head
                if hasattr(src, "model") and hasattr(src.model, "language_model"):
                    del src.model.language_model
                del src
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.vision_tower.eval()
            return

        if self.backend == "qwen3_5":
            self.vision_tower = _load_qwen35_vision_module(
                self.vision_model,
                self.config,
                self.device,
                self.dtype,
            )
            self.vision_tower.eval()
            return

        if self.backend in {"siglip", "siglip2"}:
            try:
                self.processor = AutoProcessor.from_pretrained(self.vision_model)
            except Exception:
                self.processor = AutoImageProcessor.from_pretrained(self.vision_model)
            self.vision_tower = _load_siglip_vision_module(
                self.vision_model,
                self.config,
                self.device,
                self.dtype,
            )
            self.vision_tower.eval()
            return

        if self.backend == "gemma4":
            self.vision_tower, self.vision_projector = _load_gemma4_vision_modules(
                self.vision_model,
                self.config,
                self.device,
                self.dtype,
            )
            self.vision_tower.eval()
            self.vision_projector.eval()
            return

        raise RuntimeError(f"Unexpected backend: {self.backend}")

    @torch.inference_mode()
    def encode_batch(self, images: List[Image.Image]) -> Tuple[List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
        if self.backend in {"qwen2_5_vl", "qwen3_5"}:
            return self._encode_qwen_batch(images)
        if self.backend in {"siglip", "siglip2"}:
            return self._encode_siglip_batch(images)
        if self.backend == "gemma4":
            return self._encode_gemma_batch(images)
        raise RuntimeError(f"Unsupported backend: {self.backend}")

    def _encode_qwen_batch(self, images: List[Image.Image]):
        pixel_values_parts = []
        grid_parts = []
        for image in images:
            prepared = _prepare_single_qwen_image(self.processor, image)
            pixel_values_parts.append(prepared["pixel_values"])
            grid_parts.append(prepared["grid"])

        pixel_values = torch.cat(pixel_values_parts, dim=0)
        image_grid_thw = torch.cat(grid_parts, dim=0)
        pixel_values = _maybe_tensor_to(pixel_values, self.device, self.dtype)
        image_grid_thw = _maybe_tensor_to(image_grid_thw, self.device)

        try:
            outputs = self.vision_tower(pixel_values, grid_thw=image_grid_thw)
        except TypeError:
            outputs = self.vision_tower(pixel_values, image_grid_thw=image_grid_thw)

        features = getattr(outputs, "pooler_output", None)
        if features is None:
            features = getattr(outputs, "last_hidden_state", None)
        if features is None:
            raise RuntimeError("Qwen vision tower did not return pooler_output/last_hidden_state.")
        if features.dim() != 2:
            raise RuntimeError(f"Unexpected Qwen feature shape: {tuple(features.shape)}")

        merge_size = getattr(self.vision_tower, "spatial_merge_size", 1) ** 2
        token_counts = [
            int(x)
            for x in (image_grid_thw.prod(dim=-1) // merge_size).tolist()
        ]
        chunks = list(torch.split(features, token_counts, dim=0))
        metadata = {"image_grid_thw": [grid.cpu() for grid in image_grid_thw]}
        return chunks, metadata

    def _encode_siglip_batch(self, images: List[Image.Image]):
        try:
            batch = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
            )
        except TypeError:
            batch = self.processor(
                images=images,
                return_tensors="pt",
            )

        prepared = {
            key: _maybe_tensor_to(value, self.device, self.dtype)
            for key, value in batch.items()
        }
        forward_kwargs = {"pixel_values": prepared["pixel_values"]}
        if self.backend == "siglip2":
            forward_kwargs["pixel_attention_mask"] = prepared["pixel_attention_mask"]
            forward_kwargs["spatial_shapes"] = prepared["spatial_shapes"]

        outputs = self.vision_tower(**forward_kwargs)
        hidden_states = outputs.last_hidden_state
        if hidden_states.dim() != 3:
            raise RuntimeError(f"Unexpected SigLIP feature shape: {tuple(hidden_states.shape)}")

        attention_mask = prepared.get("pixel_attention_mask")
        chunks = []
        valid_token_counts = []
        for i in range(hidden_states.size(0)):
            x = hidden_states[i]
            if (
                attention_mask is not None
                and attention_mask.dim() == 2
                and attention_mask.size(1) == x.size(0)
            ):
                x = x[attention_mask[i].to(device=x.device, dtype=torch.bool)]
            chunks.append(x)
            valid_token_counts.append(torch.tensor(x.size(0), dtype=torch.long))

        metadata = {"valid_token_count": valid_token_counts}
        return chunks, metadata

    def _encode_gemma_batch(self, images: List[Image.Image]):
        image_token = getattr(self.processor, "image_token", None)
        if image_token is None:
            image_token = getattr(getattr(self.processor, "tokenizer", None), "image_token", None)
        if image_token is None:
            raise RuntimeError("Gemma4 processor does not expose image_token.")

        batch = self.processor(
            images=[[image] for image in images],
            text=[image_token for _ in images],
            return_tensors="pt",
            padding=True,
        )

        prepared = {
            key: _maybe_tensor_to(value, self.device, self.dtype)
            for key, value in batch.items()
        }
        image_position_ids = prepared["image_position_ids"]

        try:
            outputs = self.vision_tower(
                pixel_values=prepared["pixel_values"],
                pixel_position_ids=image_position_ids,
                return_dict=True,
            )
        except TypeError:
            outputs = self.vision_tower(
                pixel_values=prepared["pixel_values"],
                image_position_ids=image_position_ids,
                return_dict=True,
            )

        image_hidden_states = self.vision_projector(outputs.last_hidden_state)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)
        token_counts = (batch["input_ids"] == image_token_id).sum(dim=1).tolist()
        token_counts = [int(x) for x in token_counts]

        if image_hidden_states.dim() == 2:
            chunks = list(torch.split(image_hidden_states, token_counts, dim=0))
        elif image_hidden_states.dim() == 3:
            chunks = [image_hidden_states[i, : token_counts[i]] for i in range(len(token_counts))]
        elif image_hidden_states.dim() == 4:
            if image_hidden_states.size(1) != 1:
                raise NotImplementedError("Only one image per sample is supported for Gemma4.")
            chunks = [image_hidden_states[i, 0, : token_counts[i]] for i in range(len(token_counts))]
        else:
            raise RuntimeError(
                f"Unexpected Gemma4 image feature shape: {tuple(image_hidden_states.shape)}"
            )

        metadata = {
            "image_token_count": [torch.tensor(x, dtype=torch.long) for x in token_counts],
        }
        return chunks, metadata


def save_feature_file(
    output_dir: Path,
    images_dir: Path,
    image_path: Path,
    feature: torch.Tensor,
    backend: str,
    vision_model: str,
    metadata: Dict[str, torch.Tensor],
):
    rel_path = image_path.relative_to(images_dir)
    target_path = (output_dir / rel_path).with_suffix(".pt")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "vision_model": vision_model,
        "vision_backend": backend,
        "source_path": str(image_path),
        "source_relpath": str(rel_path),
        "features": feature.detach().cpu(),
        "num_tokens": int(feature.shape[0]),
        "hidden_size": int(feature.shape[1]),
    }
    payload.update(metadata)

    torch.save(payload, target_path)
    return target_path, rel_path


def main():
    args = parse_args()
    images_dir = Path(args.images_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Input image directory does not exist: {images_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_image_paths(images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found under {images_dir}")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"Found {len(image_paths)} images")
    print(f"Using device={device}, dtype={dtype}, vision_model={args.vision_model}")

    extractor = VisionFeatureExtractor(
        vision_model=args.vision_model,
        device=device,
        dtype=dtype,
    )
    print(f"Resolved backend={extractor.backend}")

    manifest_path = output_dir / "manifest.jsonl"
    processed = 0
    skipped = 0

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for batch_paths in batched(image_paths, args.batch_size):
            batch_images = []
            batch_image_paths = []

            for image_path in batch_paths:
                rel_path = image_path.relative_to(images_dir)
                target_path = (output_dir / rel_path).with_suffix(".pt")
                if target_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                try:
                    batch_images.append(open_rgb_image(image_path))
                    batch_image_paths.append(image_path)
                except (OSError, UnidentifiedImageError, ValueError) as e:
                    print(f"Skipping unreadable image {image_path}: {e}")
                    skipped += 1

            if not batch_images:
                continue

            try:
                features, extra_metadata = extractor.encode_batch(batch_images)
            except Exception as e:
                print(f"Failed to encode batch starting at {batch_image_paths[0]}: {e}")
                skipped += len(batch_image_paths)
                continue

            for idx, (image_path, feature) in enumerate(zip(batch_image_paths, features)):
                metadata = {}
                if args.save_input_metadata:
                    for key, values in extra_metadata.items():
                        metadata[key] = values[idx]

                target_path, rel_path = save_feature_file(
                    output_dir=output_dir,
                    images_dir=images_dir,
                    image_path=image_path,
                    feature=feature,
                    backend=extractor.backend,
                    vision_model=args.vision_model,
                    metadata=metadata,
                )

                manifest.write(
                    json.dumps(
                        {
                            "source_relpath": str(rel_path),
                            "source_path": str(image_path),
                            "feature_path": str(target_path.relative_to(output_dir)),
                            "vision_backend": extractor.backend,
                            "vision_model": args.vision_model,
                            "num_tokens": int(feature.shape[0]),
                            "hidden_size": int(feature.shape[1]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                processed += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"Processed {processed}/{len(image_paths)} images "
                f"(skipped={skipped})",
                flush=True,
            )

    print(f"Done. Processed={processed}, skipped={skipped}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
