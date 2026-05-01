from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from huggingface_hub import snapshot_download, login

from src.dataset.dataset_base import DatasetConfig


def download_pixmo_ask_model_anything_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Downloads dnth/pixmo-ask-model-anything-images parquet files.

    This repository contains embedded images in parquet files, so no separate
    image downloading is needed.

    Expected local structure:
        dataset_root/
        └── data/
            ├── train-00000-of-xxxxx.parquet
            ├── train-00001-of-xxxxx.parquet
            └── ...

    Dataset size:
        153,592 rows
        ~15.9 GB
    """
    repo_id = "dnth/pixmo-ask-model-anything-images"

    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    if hf_token:
        login(token=hf_token)

    has_parquet_files = any((dataset_root_path / "data").glob("*.parquet"))

    if not has_parquet_files or force_redownload:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dataset_root_path),
            local_dir_use_symlinks=False,
            force_download=force_redownload,
            resume_download=True,
            allow_patterns=["data/*.parquet"],
        )


def load_pixmo_ask_model_anything_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads local PixMo Ask Model Anything parquet shards in streaming mode.
    """
    if dataset_root is None:
        raise ValueError(
            "For PIXMO_ASK_MODEL_ANYTHING_EN dataset_root is required. "
            "Expected dataset_root/data/train-xxxxx.parquet files."
        )

    dataset_root_path = Path(dataset_root)
    data_dir = dataset_root_path / "data"

    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No PixMo Ask Model Anything parquet files found under {data_dir}. "
            "Run download_pixmo_ask_model_anything_en(...) first."
        )

    ds = load_dataset(
        "parquet",
        data_files=[str(p) for p in parquet_files],
        streaming=True,
        split="train",
    )

    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if limit is not None:
        ds = ds.take(limit)

    return ds


class PixMoAskModelAnythingEnIterableDataset(TorchIterableDataset):
    """
    Converts dnth/pixmo-ask-model-anything-images samples into the unified format:
        {"image": PIL.Image, "question": str, "answer": str}

    Expected useful raw fields:
        - image
        - question
        - answer
    """

    def __init__(
        self,
        raw_hf_iterable,
        dataset_root: Optional[str] = None,
        seed: int = 42,
        skip_missing_images: bool = True,
    ):
        self.seed = seed
        self.raw_hf_iterable = raw_hf_iterable
        self.dataset_root = dataset_root
        self.skip_missing_images = skip_missing_images

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self.raw_hf_iterable:
            if not isinstance(row, dict):
                continue

            image = row.get("image")
            if image is None and self.skip_missing_images:
                continue

            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()

            if not question or not answer:
                continue

            yield {
                "image": image,
                "question": question,
                "answer": answer,
            }
