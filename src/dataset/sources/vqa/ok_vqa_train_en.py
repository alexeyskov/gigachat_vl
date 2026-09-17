from pathlib import Path
from typing import Optional
from collections import Counter

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.dataset_base import DatasetConfig
from src.dataset.huggingface_utils import download_parquet_files


def download_ok_vqa_train_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = None,
) -> None:
    """
    Downloads Multimodal-Fatima/OK-VQA_train parquet files from the data/ folder.

    Expected local structure:
    dataset_root/
    └── data/
        └── *.parquet
    """
    download_parquet_files(
        repo_id="Multimodal-Fatima/OK-VQA_train",
        dataset_root=dataset_root,
        force_redownload=force_redownload,
        hf_token=hf_token,
        max_parquet_files=max_parquet_files,
    )


def load_ok_vqa_train_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 500,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads local OK-VQA_train parquet files in streaming mode.
    """
    if dataset_root is None:
        raise ValueError(
            "For OK_VQA_TRAIN_EN dataset_root is required. "
            "Expected dataset_root/data/*.parquet files."
        )

    dataset_root_path = Path(dataset_root)
    data_dir = dataset_root_path / "data"

    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No OK-VQA_train parquet files found under {data_dir}. "
            "Run download_ok_vqa_train_en(...) first."
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

def _select_consensus_answer(answers) -> Optional[str]:
    cleaned = []

    for item in answers or []:
        value = item.get("answer", "") if isinstance(item, dict) else item
        value = str(value).strip()

        if value:
            cleaned.append(value)

    if not cleaned:
        return None

    normalized = [
        " ".join(answer.split()).casefold()
        for answer in cleaned
    ]

    winner, _ = Counter(normalized).most_common(1)[0]

    for original, normalized_answer in zip(cleaned, normalized):
        if normalized_answer == winner:
            return original

    return None

class OKVQATrainEnIterableDataset(TorchIterableDataset):
    """Converts OK-VQA train rows into English open-ended VQA samples."""

    def __init__(self, raw_hf_iterable, dataset_root=None, seed=42, skip_missing_images=True):
        self.raw_hf_iterable = raw_hf_iterable
        self.skip_missing_images = skip_missing_images

    def __iter__(self):
        for row in self.raw_hf_iterable:
            image = row.get("image")
            question = str(row.get("question", "")).strip()
            if image is None and self.skip_missing_images:
                continue
            answer = _select_consensus_answer(row.get("answers"))
            if question and answer:
                yield {
                    "image": image,
                    "question": question,
                    "answer": answer,
                }
