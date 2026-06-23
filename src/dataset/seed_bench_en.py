from pathlib import Path
from typing import Optional, Sequence

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from huggingface_hub import hf_hub_download, login

from src.dataset.dataset_base import DatasetConfig


def download_seed_bench_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    num_shards: Optional[int] = None,
    shard_indices: Optional[Sequence[int]] = None,
) -> None:
    """
    Downloads a subset of lmms-lab/SEED-Bench parquet shards.

    The original repository contains 273 test shards:
        data/test-00000-of-00273.parquet
        data/test-00001-of-00273.parquet
        ...

    Expected local structure:
        dataset_root/
        └── data/
            ├── test-00000-of-00273.parquet
            ├── test-00001-of-00273.parquet
            └── ...
    """
    repo_id = "lmms-lab/SEED-Bench"
    total_shards = 273

    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    if hf_token:
        login(token=hf_token)

    if shard_indices is None:
        if num_shards is None:
            shard_indices = list(range(total_shards))
        else:
            if num_shards <= 0:
                raise ValueError("num_shards must be positive")
            if num_shards > total_shards:
                raise ValueError(f"num_shards cannot exceed {total_shards}")
            shard_indices = list(range(num_shards))
    else:
        shard_indices = list(shard_indices)

    for shard_idx in shard_indices:
        if shard_idx < 0 or shard_idx >= total_shards:
            raise ValueError(
                f"Invalid shard index {shard_idx}; expected 0 <= idx < {total_shards}"
            )

    for shard_idx in shard_indices:
        filename = f"data/test-{shard_idx:05d}-of-00273.parquet"
        local_path = dataset_root_path / filename

        if local_path.exists() and not force_redownload:
            continue

        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=dataset_root_path,
            local_dir_use_symlinks=False,
            force_download=force_redownload,
            resume_download=True,
        )


def load_seed_bench_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 500,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads local SEED-Bench parquet shards in streaming mode.

    This loader intentionally expects local parquet files, because the full
    repository is large. Use download_seed_bench_en(..., num_shards=N) first.
    """
    if dataset_root is None:
        raise ValueError(
            "For SEED_BENCH_EN dataset_root is required. "
            "Expected dataset_root/data/test-xxxxx-of-00273.parquet files."
        )

    dataset_root_path = Path(dataset_root)
    data_dir = dataset_root_path / "data"

    parquet_files = sorted(data_dir.glob("test-*-of-00273.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No SEED-Bench parquet shards found under {data_dir}. "
            "Run download_seed_bench_en(...) first."
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
