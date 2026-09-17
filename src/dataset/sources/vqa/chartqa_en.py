import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.dataset_base import DatasetConfig
from src.dataset.huggingface_utils import download_parquet_files

CHARTQA_QUESTION_TEMPLATES_EN = [
    "Use the chart to answer the question.\n\nQuestion: {question}",
    "Answer the question based on the chart.\n\nQuestion: {question}",
    "Read the chart and answer the question.\n\nQuestion: {question}",
]


def download_chartqa_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = None,
) -> None:
    download_parquet_files(
        repo_id="lmms-lab/ChartQA",
        dataset_root=dataset_root,
        force_redownload=force_redownload,
        hf_token=hf_token,
        max_parquet_files=max_parquet_files,
        parquet_pattern="data/*.parquet",
    )


def load_chartqa_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 100,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads local ChartQA parquet files in streaming mode.
    """
    if dataset_root is None:
        raise ValueError(
            "For CHARTQA_EN dataset_root is required. "
            "Expected dataset_root/data/*.parquet files."
        )

    dataset_root_path = Path(dataset_root)
    data_dir = dataset_root_path / "data"

    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No ChartQA parquet files found under {data_dir}. "
            "Run download_chartqa_en(...) first."
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


class ChartQAEnIterableDataset(TorchIterableDataset):
    """
    Converts lmms-lab/ChartQA samples into the unified format:
        {"image": PIL.Image, "question": str, "answer": str}

    Expected raw fields:
        - question
        - answer
        - image
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

            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()

            if not question or not answer:
                continue

            if self.random.random() < 0.4:
                question_template = self.random.choice(CHARTQA_QUESTION_TEMPLATES_EN)
                question = question_template.format(question=question)

            yield {
                "image": image,
                "question": question,
                "answer": answer,
            }