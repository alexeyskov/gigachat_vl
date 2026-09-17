from pathlib import Path
import re
from typing import Optional

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.dataset_base import DatasetConfig
from src.dataset.huggingface_utils import download_parquet_files


IMAGE_TAG_PATTERN = re.compile(r"<\s*/?\s*img\s*>|<\s*image_\d+\s*>", re.IGNORECASE)


def download_mm_vet_v2_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = None,
) -> None:
    """
    Downloads whyu/mm-vet-v2 parquet files from the data/ folder.

    Expected local structure:
    dataset_root/
    └── data/
        └── *.parquet
    """
    download_parquet_files(
        repo_id="whyu/mm-vet-v2",
        dataset_root=dataset_root,
        force_redownload=force_redownload,
        hf_token=hf_token,
        max_parquet_files=max_parquet_files,
    )


def load_mm_vet_v2_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 500,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads local mm-vet-v2 parquet files in streaming mode.
    """
    if dataset_root is None:
        raise ValueError(
            "For MM_VET_V2_EN dataset_root is required. "
            "Expected dataset_root/data/*.parquet files."
        )

    dataset_root_path = Path(dataset_root)
    data_dir = dataset_root_path / "data"

    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No mm-vet-v2 parquet files found under {data_dir}. "
            "Run download_mm_vet_v2_en(...) first."
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


class MMVetV2EnIterableDataset(TorchIterableDataset):
    """Converts single-image MM-Vet-v2 rows into English VQA samples."""

    def __init__(self, raw_hf_iterable, dataset_root=None, seed=42, skip_missing_images=True):
        self.raw_hf_iterable = raw_hf_iterable
        self.skip_missing_images = skip_missing_images

    def __iter__(self):
        for row in self.raw_hf_iterable:
            if any(row.get(f"image_{index}") is not None for index in range(1, 18)):
                continue
            image = row.get("image_0")
            question = IMAGE_TAG_PATTERN.sub(" ", str(row.get("question", "")))
            question = re.sub(r"\s+", " ", question).strip()
            answer = str(row.get("answer", "")).strip()
            if image is None and self.skip_missing_images:
                continue
            if "<OR>" in answer:
                answer = next((part.strip() for part in answer.split("<OR>") if part.strip()), "")
            if "<AND>" in answer:
                answer = " and ".join(part.strip() for part in answer.split("<AND>") if part.strip())
            if question and answer:
                yield {"image": image, "question": question, "answer": answer}
