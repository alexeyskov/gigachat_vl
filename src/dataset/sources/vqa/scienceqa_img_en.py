from pathlib import Path
from typing import Optional

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from huggingface_hub import snapshot_download, login

from src.dataset.dataset_base import DatasetConfig


def download_scienceqa_img_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Downloads lmms-lab/ScienceQA-IMG parquet files from the data/ folder.

    Expected local structure:
    dataset_root/
    └── data/
        ├── train-*.parquet
        ├── validation-*.parquet
        └── test-*.parquet
    """
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    if hf_token:
        login(token=hf_token)

    has_parquet_files = any((dataset_root_path / "data").glob("*.parquet"))

    if not has_parquet_files or force_redownload:
        snapshot_download(
            repo_id="lmms-lab/ScienceQA-IMG",
            repo_type="dataset",
            local_dir=str(dataset_root_path),
            local_dir_use_symlinks=False,
            force_download=force_redownload,
            resume_download=True,
            allow_patterns=["data/*.parquet"],
        )


def load_scienceqa_img_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 500,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads local ScienceQA-IMG parquet files in streaming mode.

    All local parquet shards under data/ are loaded together, regardless of
    whether they belong to train, validation, or test.
    """
    if dataset_root is None:
        raise ValueError(
            "For SCIENCEQA_IMG_EN dataset_root is required. "
            "Expected dataset_root/data/*.parquet files."
        )

    dataset_root_path = Path(dataset_root)
    data_dir = dataset_root_path / "data"

    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No ScienceQA-IMG parquet files found under {data_dir}. "
            "Run download_scienceqa_img_en(...) first."
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
