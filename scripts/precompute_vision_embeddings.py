import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.finevision import load_images_from_example
from src.dataset.precomputed_embeddings import (
    PrecomputedVisionEmbeddingDataset,
    embeddings_root_for_dataset,
    visual_encoder_dir_name,
)
from src.dataset.unified_vlm_dataset import SupportedDatasets
from src.model.gigachat_vl import (
    DEFAULT_PATCH_VISION_IMAGE_SIZE,
    DEFAULT_PATCH_VISION_PATCH_SIZE,
    GigaChatVL,
    single_device,
)


VISION_PATH = "/media/alexey/HDDLargeData/models/VLM/Qwen3.5-9B/"
DEVICE = "cuda"  # "auto" | "cpu" | "cuda"
VISION_ATTN_IMPLEMENTATION = "auto"  # "auto" | "flash_attention_2" | "sdpa" | "eager"

GLOBAL_SEED = 42
GLOBAL_SHUFFLE_BUFFER = 1000

SAVE_DTYPE = "bf16"  # "bf16" | "fp16" | "fp32"
OVERWRITE = False
LOG_EVERY = 100
BATCH_SIZE = 16  # Number of images per vision forward.
PROFILE_TIMINGS = True
PROGRESS_BAR = True
EMPTY_CACHE_EVERY_N_BATCHES = 50

dataset_specs = [
    {
        "config": SupportedDatasets.LLAVA_PRETRAIN_RU.value,
        "limit": None,
        "dataset_root": "/media/alexey/HDDLargeData/datasets/llm/VL/Maya/",
    },
    {
        "config": SupportedDatasets.MSCOCO_CAPTION_ML.value,
        "limit": None,
        "dataset_root": "/media/alexey/HDDLargeData/datasets/llm/VL/mscoco-multilingual-30k/",
    },
    {
        "config": SupportedDatasets.RUSTITW_OCR.value,
        "limit": None,
        "dataset_root": "/media/alexey/HDDLargeData/datasets/llm/VL/rustitw_ocr/",
    },
    {
        "config": SupportedDatasets.GQA_RU.value,
        "limit": None,
        "dataset_root": "/media/alexey/HDDLargeData/datasets/llm/VL/GRA-ru/",
    },
]


def _save_dtype() -> torch.dtype:
    name = str(SAVE_DTYPE).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported SAVE_DTYPE={SAVE_DTYPE!r}")


def _build_dataset_for_spec(
    spec: Dict[str, Any],
    visual_encoder: str,
) -> Iterable[Dict[str, Any]]:
    config = spec["config"]
    limit: Optional[int] = spec.get("limit")
    dataset_root: Optional[str] = spec.get("dataset_root")
    load_kwargs: Dict[str, Any] = spec.get("load_kwargs", {})
    dataset_kwargs: Dict[str, Any] = spec.get("dataset_kwargs", {})

    if dataset_root is None:
        raise ValueError(
            f"dataset_root is required for precomputing embeddings: {config.name}"
        )
    if config.load_raw_func is None or config.dataset_class is None:
        raise ValueError(f"Dataset {config.name} is not configured for loading")

    raw_ds = config.load_raw_func(
        config=config,
        limit=limit,
        shuffle_buffer=GLOBAL_SHUFFLE_BUFFER,
        seed=GLOBAL_SEED,
        dataset_root=dataset_root,
        **load_kwargs,
    )

    dataset = config.dataset_class(
        raw_hf_iterable=raw_ds,
        dataset_root=dataset_root,
        seed=GLOBAL_SEED,
        skip_missing_images=True,
        **dataset_kwargs,
    )

    return PrecomputedVisionEmbeddingDataset(
        dataset=dataset,
        dataset_root=dataset_root,
        visual_encoder=visual_encoder,
        require_exists=False,
    )


def _write_metadata(
    output_dir: Path,
    model: GigaChatVL,
    spec: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "gigachat_vl_precomputed_vision_features_v1",
        "projector_applied": False,
        "vision_name": VISION_PATH,
        "vision_dir_name": visual_encoder_dir_name(VISION_PATH),
        "vision_backend": model.vision_backend,
        "vision_hidden_size": model.vision_hidden_size,
        "vision_device": str(getattr(model, "model_device", "unknown")),
        "vision_attn_implementation": getattr(
            getattr(model.vision_tower, "config", None),
            "_attn_implementation",
            None,
        ),
        "save_dtype": SAVE_DTYPE,
        "dataset_name": spec["config"].name,
        "dataset_limit": spec.get("limit"),
        "global_seed": GLOBAL_SEED,
        "global_shuffle_buffer": GLOBAL_SHUFFLE_BUFFER,
        "created_at_unix": int(time.time()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_feature(path: Path, feature: torch.Tensor, model: GigaChatVL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature = feature.detach().to(dtype=_save_dtype(), device="cpu").contiguous()
    payload = {
        "features": feature,
        "projector_applied": False,
        "vision_name": VISION_PATH,
        "vision_backend": model.vision_backend,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_precompute_device() -> torch.device:
    if DEVICE is None or str(DEVICE).lower() == "auto":
        return single_device()
    return single_device(DEVICE)


def _estimated_total_samples(spec: Dict[str, Any]) -> Optional[int]:
    limit = spec.get("limit")
    if limit is not None:
        return int(limit)

    total_samples = getattr(spec["config"], "total_samples", None)
    if total_samples is None:
        return None
    total_samples = int(total_samples)
    return total_samples if total_samples > 0 else None


def _progress_write(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def build_vision_feature_extractor() -> GigaChatVL:
    model = GigaChatVL.__new__(GigaChatVL)
    torch.nn.Module.__init__(model)

    model.model_device = _resolve_precompute_device()
    model.freeze_vision = True
    model.vision_name = VISION_PATH
    model.vision_backend_override = None
    model.vision_image_size = DEFAULT_PATCH_VISION_IMAGE_SIZE
    model.vision_patch_size = DEFAULT_PATCH_VISION_PATCH_SIZE
    model.vision_conv_hidden_size = 256
    model.vision_use_maxpool = False
    model.vision_llm_use_qlora = False
    model.vision_llm_dtype = "auto"
    model.vision_attn_implementation = VISION_ATTN_IMPLEMENTATION
    model.vision_use_lora = False
    model.vision_use_qlora = False
    model.vision_lora_r = 16
    model.vision_lora_alpha = 32
    model.vision_lora_dropout = 0.05
    model.vision_lora_path = None
    model.vision_projector_path = None
    model.vision_encoder_path = None
    model.vision_llm_lora_path = None
    model.vision_processor = None
    model.vision_backend = None
    model.vision_source_model = None
    model.vision_tower = None
    model.vision_projector = None
    model.vision_hidden_size = None
    model.gemma_image_placeholder = None

    model._load_vision_backend(VISION_PATH)
    model.eval()
    return model


def precompute_for_spec(model: GigaChatVL, spec: Dict[str, Any]) -> None:
    config = spec["config"]
    dataset_root = spec["dataset_root"]
    output_dir = embeddings_root_for_dataset(dataset_root, VISION_PATH)
    _write_metadata(output_dir, model, spec)

    dataset = _build_dataset_for_spec(spec, visual_encoder=VISION_PATH)

    seen = 0
    processed = 0
    saved = 0
    skipped_existing = 0
    skipped_no_image = 0
    skipped_bad_image = 0
    started_at = time.time()
    load_time = 0.0
    prepare_time = 0.0
    forward_time = 0.0
    save_time = 0.0
    batch_images = []
    batch_paths = []
    flushed_batches = 0
    progress_total = _estimated_total_samples(spec)
    pbar = None

    print(f"\nDataset: {config.name}")
    print(f"Embeddings: {output_dir}")
    print(f"Vision device: {getattr(model, 'model_device', 'unknown')}")
    print(
        "Vision attention: "
        f"{getattr(getattr(model.vision_tower, 'config', None), '_attn_implementation', None)}"
    )
    if not OVERWRITE:
        print(
            "Resume: OVERWRITE=False, existing .pt files are skipped. "
            "Keep dataset_specs, GLOBAL_SEED, and GLOBAL_SHUFFLE_BUFFER unchanged "
            "between runs."
        )

    if PROGRESS_BAR and tqdm is not None:
        pbar = tqdm(
            total=progress_total,
            desc=config.name,
            unit="sample",
            dynamic_ncols=True,
        )

    def update_progress(force: bool = False) -> None:
        elapsed = max(time.time() - started_at, 1e-6)
        postfix = {
            "processed": processed,
            "saved": saved,
            "existing": skipped_existing,
            "pending_img": len(batch_images),
            "sample/s": f"{seen / elapsed:.2f}",
        }
        if PROFILE_TIMINGS:
            postfix.update(
                {
                    "load": f"{load_time:.1f}s",
                    "prep": f"{prepare_time:.1f}s",
                    "vision": f"{forward_time:.1f}s",
                    "save": f"{save_time:.1f}s",
                }
            )

        if pbar is not None:
            if force or seen % LOG_EVERY == 0:
                pbar.set_postfix(postfix)
            return

        if force or seen % LOG_EVERY == 0:
            message = (
                f"{config.name}: seen={seen:,} processed={processed:,} "
                f"saved={saved:,} existing={skipped_existing:,} "
                f"pending_images={len(batch_images):,} speed={seen / elapsed:.2f} sample/s"
            )
            if progress_total is not None:
                message += f" total~={progress_total:,}"
            if PROFILE_TIMINGS:
                message += (
                    f" | load={load_time:.1f}s prep={prepare_time:.1f}s "
                    f"vision={forward_time:.1f}s save={save_time:.1f}s"
                )
            print(message)

    def flush_batch() -> None:
        nonlocal saved, batch_images, batch_paths, prepare_time, forward_time, save_time
        nonlocal flushed_batches

        if not batch_images:
            return

        _sync_cuda()
        t0 = time.perf_counter()
        vision_batch = model.prepare_vision_inputs(batch_images)
        _sync_cuda()
        prepare_time += time.perf_counter() - t0

        _sync_cuda()
        t0 = time.perf_counter()
        with torch.inference_mode():
            features = model.encode_vision_features(**vision_batch)
        _sync_cuda()
        forward_time += time.perf_counter() - t0

        if len(features) != len(batch_paths):
            raise RuntimeError(
                "Vision feature count does not match embedding path count: "
                f"features={len(features)}, paths={len(batch_paths)}"
            )

        t0 = time.perf_counter()
        for path, feature in zip(batch_paths, features):
            if OVERWRITE or not path.exists():
                _save_feature(path, feature, model)
                saved += 1
        save_time += time.perf_counter() - t0

        batch_images = []
        batch_paths = []
        flushed_batches += 1
        if (
            torch.cuda.is_available()
            and EMPTY_CACHE_EVERY_N_BATCHES > 0
            and flushed_batches % EMPTY_CACHE_EVERY_N_BATCHES == 0
        ):
            torch.cuda.empty_cache()
        update_progress()

    try:
        for sample in dataset:
            seen += 1
            if pbar is not None:
                pbar.update(1)

            paths = sample.get("vision_embedding_paths")
            if not paths:
                skipped_no_image += 1
                update_progress()
                continue

            path_objs = [Path(path) for path in paths]
            if not OVERWRITE and all(path.exists() for path in path_objs):
                skipped_existing += 1
                processed += 1
                update_progress()
                continue

            try:
                t0 = time.perf_counter()
                images = load_images_from_example(sample)
                load_time += time.perf_counter() - t0
            except Exception as e:
                skipped_bad_image += 1
                _progress_write(f"Skipping sample with unreadable image: {e}")
                update_progress()
                continue

            if len(images) != len(path_objs):
                raise RuntimeError(
                    "Image count does not match embedding path count: "
                    f"images={len(images)}, paths={len(path_objs)}"
                )

            batch_images.extend(images)
            batch_paths.extend(path_objs)
            processed += 1
            if len(batch_images) >= BATCH_SIZE:
                flush_batch()
            update_progress()

        flush_batch()
    finally:
        if pbar is not None:
            update_progress(force=True)
            pbar.close()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    message = (
        f"Done {config.name}: seen={seen:,}, processed={processed:,}, saved={saved:,}, "
        f"existing={skipped_existing:,}, no_image={skipped_no_image:,}, "
        f"bad_image={skipped_bad_image:,}"
    )
    if PROFILE_TIMINGS:
        message += (
            f" | load={load_time:.1f}s prep={prepare_time:.1f}s "
            f"vision={forward_time:.1f}s save={save_time:.1f}s"
        )
    print(message)


def main() -> None:
    model = build_vision_feature_extractor()

    for spec in dataset_specs:
        precompute_for_spec(model, spec)


if __name__ == "__main__":
    main()
