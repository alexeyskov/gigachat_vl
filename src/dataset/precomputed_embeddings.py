import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset


def visual_encoder_dir_name(visual_encoder: str) -> str:
    name = Path(str(visual_encoder).rstrip("/")).name
    if not name:
        name = str(visual_encoder)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "visual_encoder"


def embeddings_root_for_dataset(dataset_root: str, visual_encoder: str) -> Path:
    return Path(dataset_root) / "embeddings" / visual_encoder_dir_name(visual_encoder)


def embedding_path_for_sample(
    dataset_root: str,
    visual_encoder: str,
    sample_index: int,
    image_index: int = 0,
) -> Path:
    return (
        embeddings_root_for_dataset(dataset_root, visual_encoder)
        / f"{sample_index:012d}_{image_index:02d}.pt"
    )


def num_images_in_sample(sample: Dict[str, Any]) -> int:
    raw_images = sample.get("images", None)
    if raw_images is None:
        raw_images = sample.get("image", None)

    if raw_images is None:
        return 0
    if isinstance(raw_images, (list, tuple)):
        return len(raw_images)
    return 1


def _normalize_metadata_value(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip().rstrip("/")
    if not text:
        return None

    path = Path(text).expanduser()
    if path.is_absolute() or path.exists():
        try:
            return str(path.resolve(strict=False)).rstrip("/")
        except Exception:
            pass

    return text


def _check_metadata_matches(
    path: str,
    field: str,
    actual: Any,
    expected: Optional[str],
) -> None:
    actual_norm = _normalize_metadata_value(actual)
    expected_norm = _normalize_metadata_value(expected)
    if actual_norm is None or expected_norm is None:
        return
    if actual_norm != expected_norm:
        raise ValueError(
            f"Precomputed vision embedding metadata mismatch for {path}: "
            f"{field}={actual!r}, expected {expected!r}."
        )


def load_precomputed_vision_feature(
    path: str,
    expected_vision_name: Optional[str] = None,
    expected_vision_backend: Optional[str] = None,
) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        _check_metadata_matches(
            path=path,
            field="vision_name",
            actual=payload.get("vision_name"),
            expected=expected_vision_name,
        )
        _check_metadata_matches(
            path=path,
            field="vision_backend",
            actual=payload.get("vision_backend"),
            expected=expected_vision_backend,
        )
        if "features" in payload:
            payload = payload["features"]
        elif "vision_features" in payload:
            payload = payload["vision_features"]

    if not torch.is_tensor(payload):
        raise TypeError(f"Precomputed vision embedding is not a tensor: {path}")

    return payload.detach().cpu()


class PrecomputedVisionEmbeddingDataset(IterableDataset):
    """
    Adds per-image embedding paths to unified VLM samples.

    The wrapped samples still keep their original `image` field. The collator can
    then choose precomputed features when the vision tower is frozen, or fall back
    to the image and compute features online.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        dataset_root: Optional[str],
        visual_encoder: Optional[str],
        require_exists: bool = True,
    ):
        self.dataset = dataset
        self.dataset_root = dataset_root
        self.visual_encoder = visual_encoder
        self.require_exists = require_exists

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for sample_index, sample in enumerate(self.dataset):
            if (
                not self.dataset_root
                or not self.visual_encoder
                or not isinstance(sample, dict)
            ):
                yield sample
                continue

            num_images = num_images_in_sample(sample)
            if num_images <= 0:
                yield sample
                continue

            paths: List[str] = [
                str(
                    embedding_path_for_sample(
                        dataset_root=self.dataset_root,
                        visual_encoder=self.visual_encoder,
                        sample_index=sample_index,
                        image_index=image_index,
                    )
                )
                for image_index in range(num_images)
            ]

            if self.require_exists and not all(Path(path).exists() for path in paths):
                yield sample
                continue

            sample = dict(sample)
            sample["vision_embedding_paths"] = paths
            yield sample
