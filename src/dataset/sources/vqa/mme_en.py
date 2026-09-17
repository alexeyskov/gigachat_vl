from pathlib import Path
from typing import Optional

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.dataset_base import DatasetConfig
from src.dataset.huggingface_utils import download_parquet_files


def download_mme_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = None,
) -> None:
    """
    Downloads lmms-lab/MME parquet shards from the data/ folder.

    Expected local structure:
    dataset_root/
    └── data/
        ├── test-00000-of-00004-*.parquet
        ├── test-00001-of-00004-*.parquet
        ├── test-00002-of-00004-*.parquet
        └── test-00003-of-00004-*.parquet
    """
    download_parquet_files(
        repo_id="lmms-lab/MME",
        dataset_root=dataset_root,
        force_redownload=force_redownload,
        hf_token=hf_token,
        max_parquet_files=max_parquet_files,
    )


def load_mme_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 500,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads local MME parquet shards in streaming mode.
    """
    if dataset_root is None:
        raise ValueError(
            "For MME dataset_root is required. "
            "Expected dataset_root/data/test-*.parquet files."
        )

    dataset_root_path = Path(dataset_root)
    data_dir = dataset_root_path / "data"

    parquet_files = sorted(data_dir.glob("test-*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No MME parquet files found under {data_dir}. "
            "Run download_mme(...) first."
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


class MMEEnIterableDataset(TorchIterableDataset):
    """Converts MME rows into English binary VQA training samples."""

    def __init__(self, raw_hf_iterable, dataset_root=None, seed=42, skip_missing_images=True):
        self.raw_hf_iterable = raw_hf_iterable
        self.skip_missing_images = skip_missing_images

    def __iter__(self):
        for row in self.raw_hf_iterable:
            image = row.get("image")
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip().lower()
            if image is None and self.skip_missing_images:
                continue
            if question and answer in {"yes", "no"}:
                yield {"image": image, "question": question, "answer": answer.title()}
