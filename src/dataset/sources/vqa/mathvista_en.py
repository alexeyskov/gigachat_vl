from pathlib import Path
from typing import Optional

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from src.dataset.dataset_base import DatasetConfig
from src.dataset.huggingface_utils import download_parquet_files


def download_mathvista_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = None,
) -> None:
    """
    Downloads AI4Math/MathVista parquet files from the data/ folder.

    Expected local structure:
    dataset_root/
    └── data/
        └── *.parquet
    """
    download_parquet_files(
        repo_id="AI4Math/MathVista",
        dataset_root=dataset_root,
        force_redownload=force_redownload,
        hf_token=hf_token,
        max_parquet_files=max_parquet_files,
    )


def load_mathvista_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 500,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads local MathVista parquet files in streaming mode.
    """
    if dataset_root is None:
        raise ValueError(
            "For MATHVISTA_EN dataset_root is required. "
            "Expected dataset_root/data/*.parquet files."
        )

    dataset_root_path = Path(dataset_root)
    data_dir = dataset_root_path / "data"

    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No MathVista parquet files found under {data_dir}. "
            "Run download_mathvista_en(...) first."
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
