import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from huggingface_hub import snapshot_download, login

from src.dataset.dataset_base import DatasetConfig

DOCVQA_QUESTION_TEMPLATES_EN = [
    "Answer the question using the image. Give only a short, direct answer.\n\nQuestion: {question}",
    "Look at the image and answer the question briefly.\n\nQuestion: {question}",
    "Use the visible text and layout in the image to answer. Keep the answer concise.\n\nQuestion: {question}",
    "Based on the image, provide the exact answer when possible.\n\nQuestion: {question}",
    "Read the image and answer the question with the final answer only.\n\nQuestion: {question}",
    "Use only the information visible in the image. Respond with a short answer.\n\nQuestion: {question}",
    "Answer the question based on the document, chart, table, or infographic shown in the image. Keep it short.\n\nQuestion: {question}",
]


def download_docvqa_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Downloads lmms-lab/DocVQA locally.

    Expected local structure:
    dataset_root/
    ├── DocVQA/
    │   ├── train-*.parquet
    │   ├── validation-*.parquet
    │   └── test-*.parquet
    └── InfographicVQA/
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
            repo_id="lmms-lab/DocVQA",
            repo_type="dataset",
            local_dir=str(dataset_root_path),
            local_dir_use_symlinks=False,
            force_download=force_redownload,
            resume_download=True,
        )


def _load_docvqa_subset(
    dataset_root: str,
    subset: str,
    limit: Optional[int],
    shuffle_buffer: int,
    seed: int,
) -> HFDataset:
    subset_dir = Path(dataset_root) / subset

    if not subset_dir.exists():
        raise FileNotFoundError(f"DocVQA subset directory not found: {subset_dir}")

    parquet_files = sorted(subset_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {subset_dir}")

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


def load_docvqa_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 100,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    if dataset_root is None:
        raise ValueError(
            "For DOCVQA_EN dataset_root is required. "
            "Expected dataset_root/DocVQA/*.parquet files."
        )

    return _load_docvqa_subset(
        dataset_root=dataset_root,
        subset="DocVQA",
        limit=limit,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
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
            "Expected dataset_root/InfographicVQA/*.parquet files."
        )

    return _load_docvqa_subset(
        dataset_root=dataset_root,
        subset="InfographicVQA",
        limit=limit,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )


class DocVQAEnIterableDataset(TorchIterableDataset):
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
            if not question:
                continue

            answers = row.get("answers", None)
            if not answers:
                continue

            answers = [
                str(answer).strip()
                for answer in answers
                if answer is not None and str(answer).strip()
            ]
            if not answers:
                continue

            answer = self.random.choice(answers)
            question_template = self.random.choice(DOCVQA_QUESTION_TEMPLATES_EN)

            yield {
                "image": image,
                "question": question_template.format(question=question),
                "answer": answer,
            }


class InfographicVQAEnIterableDataset(DocVQAEnIterableDataset):
    pass
