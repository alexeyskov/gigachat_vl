import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.dataset_base import DatasetConfig
from src.dataset.huggingface_utils import download_parquet_files


INFOGRAPHICVQA_QUESTION_TEMPLATES_EN = [
    "Use the infographic to answer the question.\n\nQuestion: {question}",
    "Answer the question based on the infographic.\n\nQuestion: {question}",
    "Read the infographic and answer the question.\n\nQuestion: {question}",
]


def download_infographicvqa_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = None,
) -> None:
    download_parquet_files(
        repo_id="lmms-lab/DocVQA",
        dataset_root=dataset_root,
        force_redownload=force_redownload,
        hf_token=hf_token,
        max_parquet_files=max_parquet_files,
        parquet_pattern="InfographicVQA/train-*.parquet",
    )


def load_infographicvqa_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 100,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    if dataset_root is None:
        raise ValueError(
            "For INFOGRAPHICVQA_EN dataset_root is required. "
            "Expected dataset_root/InfographicVQA/train-*.parquet files."
        )

    data_dir = Path(dataset_root) / "InfographicVQA"

    parquet_files = sorted(
        data_dir.glob("train-*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No InfographicVQA train parquet files found under {data_dir}. "
            "Run download_infographicvqa_en(...) first."
        )

    ds = load_dataset(
        "parquet",
        data_files=[str(path) for path in parquet_files],
        streaming=True,
        split="train",
    )

    if shuffle_buffer > 0:
        ds = ds.shuffle(
            seed=seed,
            buffer_size=shuffle_buffer,
        )

    if limit is not None:
        ds = ds.take(limit)

    return ds


class InfographicVQAEnIterableDataset(TorchIterableDataset):
    def __init__(
        self,
        raw_hf_iterable,
        dataset_root: Optional[str] = None,
        seed: int = 42,
        skip_missing_images: bool = True,
    ):
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

            question = str(
                row.get("question", "")
            ).strip()

            if not question:
                continue

            answers = row.get("answers")

            if not answers:
                continue

            answers = [
                str(answer).strip()
                for answer in answers
                if answer is not None
                and str(answer).strip()
            ]

            if not answers:
                continue

            answer = self.random.choice(answers)

            if self.random.random() < 0.4:
                template = self.random.choice(
                    INFOGRAPHICVQA_QUESTION_TEMPLATES_EN
                )
                question = template.format(
                    question=question
                )

            yield {
                "image": image,
                "question": question,
                "answer": answer,
            }