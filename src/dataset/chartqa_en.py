import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from huggingface_hub import snapshot_download, login

from src.dataset.dataset_base import DatasetConfig

CHARTQA_QUESTION_TEMPLATES_EN = [
    "Answer the question based on the chart. Give only a short, direct answer.\n\nQuestion: {question}",
    "Look at the chart and answer the question briefly.\n\nQuestion: {question}",
    "Use the information shown in the chart to answer. Keep the answer concise.\n\nQuestion: {question}",
    "Based on the chart, provide the final answer only.\n\nQuestion: {question}",
    "Read the chart and answer the question with a short answer.\n\nQuestion: {question}",
    "Use only the visual information in the chart. Respond with the final answer.\n\nQuestion: {question}",
    "Answer the question using the chart. If calculation is needed, return only the final value.\n\nQuestion: {question}",
]


def download_chartqa_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Downloads lmms-lab/ChartQA locally.

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
    else:
        login()

    if not any(dataset_root_path.iterdir()) or force_redownload:
        snapshot_download(
            repo_id="lmms-lab/ChartQA",
            repo_type="dataset",
            local_dir=str(dataset_root_path),
            local_dir_use_symlinks=False,
            force_download=force_redownload,
            resume_download=True,
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

            question_template = self.random.choice(CHARTQA_QUESTION_TEMPLATES_EN)

            yield {
                "image": image,
                "question": question_template.format(question=question),
                "answer": answer,
            }
