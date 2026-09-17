import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from src.dataset.dataset_base import DatasetConfig, CAPTIONING_QUESTION_TEMPLATES_EN
from src.dataset.huggingface_utils import download_parquet_files


def download_pixmo_cap_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = 50,
) -> None:
    """
    Downloads a subset of dnth/pixmo-cap-images parquet shards.

    The original repository contains 381 train shards:
        data/train-00000-of-00381.parquet
        data/train-00001-of-00381.parquet
        ...

    Full dataset:
        584,650 rows
        ~187 GB

    Approximate subset size:
        rows_per_shard ~= 584650 / 381 ~= 1534
        50 shards ~= 76k samples

    Expected local structure:
        dataset_root/
        └── data/
            ├── train-00000-of-00381.parquet
            ├── train-00001-of-00381.parquet
            └── ...
    """
    download_parquet_files(
        repo_id="dnth/pixmo-cap-images",
        dataset_root=dataset_root,
        force_redownload=force_redownload,
        hf_token=hf_token,
        max_parquet_files=max_parquet_files,
    )


def load_pixmo_cap_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 1000,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads local PixMo-Cap parquet shards in streaming mode.

    This loader intentionally expects local parquet files, because the full
    repository is large. Use download_pixmo_cap_en(..., num_shards=N) first.
    """
    if dataset_root is None:
        raise ValueError(
            "For PIXMO_CAP_EN dataset_root is required. "
            "Expected dataset_root/data/train-xxxxx-of-00381.parquet files."
        )

    dataset_root_path = Path(dataset_root)
    data_dir = dataset_root_path / "data"

    parquet_files = sorted(data_dir.glob("train-*-of-00381.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No PixMo-Cap parquet shards found under {data_dir}. "
            "Run download_pixmo_cap_en(...) first."
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


class PixMoCapEnIterableDataset(TorchIterableDataset):
    """
    Converts dnth/pixmo-cap-images samples into the unified format:
        {"image": PIL.Image, "question": str, "answer": str}

    Expected useful raw fields:
        - image
        - caption
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
        self.random = random.Random(seed)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self.raw_hf_iterable:
            if not isinstance(row, dict):
                continue

            image = row.get("image")
            if image is None and self.skip_missing_images:
                continue

            answer = row.get("caption")
            if not answer:
                continue

            question = self.random.choice(CAPTIONING_QUESTION_TEMPLATES_EN)

            yield {
                "image": image,
                "question": question,
                "answer": answer,
            }
