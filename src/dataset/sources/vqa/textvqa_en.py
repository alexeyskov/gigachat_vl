from collections import Counter
import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.dataset_base import DatasetConfig
from src.dataset.huggingface_utils import download_parquet_files


TEXTVQA_QUESTION_TEMPLATES_EN = [
    (
        "Answer the question using the image. Read any visible text carefully. "
        "Give only a short, direct answer.\n\nQuestion: {question}"
    ),
    (
        "Look at the image, including any text visible in it, and answer briefly."
        "\n\nQuestion: {question}"
    ),
    (
        "Read the text in the image as needed and answer the question. "
        "Return only the final answer.\n\nQuestion: {question}"
    ),
    (
        "Use the visual information and visible text in the image to answer. "
        "Keep the answer concise.\n\nQuestion: {question}"
    ),
    (
        "Answer the question based on the image. Pay attention to words, letters, "
        "numbers, signs, or labels that may be visible.\n\nQuestion: {question}"
    ),
]


def download_textvqa_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = None,
) -> None:
    """
    Downloads TextVQA train parquet shards.

    Expected local structure:
        dataset_root/
        └── data/
            ├── train-00000-of-00020.parquet
            ├── train-00001-of-00020.parquet
            └── ...
    """
    download_parquet_files(
        repo_id="lmms-lab-encoder/textvqa",
        dataset_root=dataset_root,
        force_redownload=force_redownload,
        hf_token=hf_token,
        max_parquet_files=max_parquet_files,
        parquet_pattern="data/train-*.parquet",
    )


def load_textvqa_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 500,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    if dataset_root is None:
        raise ValueError(
            "For TEXTVQA_EN dataset_root is required. "
            "Expected dataset_root/data/train-*.parquet files."
        )

    data_dir = Path(dataset_root) / "data"

    parquet_files = sorted(data_dir.glob("train-*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No TextVQA train parquet files found under {data_dir}. "
            "Run download_textvqa_en(...) first."
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


def _normalize_answer(answer: str) -> str:
    return " ".join(answer.strip().split()).casefold()


def _select_consensus_answer(answers: Any) -> Optional[str]:
    if not isinstance(answers, (list, tuple)):
        return None

    cleaned_answers = [
        str(answer).strip()
        for answer in answers
        if answer is not None and str(answer).strip()
    ]

    if not cleaned_answers:
        return None

    normalized_answers = [
        _normalize_answer(answer)
        for answer in cleaned_answers
    ]

    counts = Counter(normalized_answers)
    winner, _ = counts.most_common(1)[0]

    for original, normalized in zip(
        cleaned_answers,
        normalized_answers,
    ):
        if normalized == winner:
            return original

    return None


class TextVQAEnIterableDataset(TorchIterableDataset):
    """
    Converts TextVQA rows into:

        {
            "image": ...,
            "question": str,
            "answer": str,
        }

    `ocr_tokens` are intentionally ignored: the model should read text
    visually instead of receiving external OCR output.
    """

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

            question = str(row.get("question", "")).strip()
            if not question:
                continue

            answer = _select_consensus_answer(row.get("answers"))
            if not answer:
                continue

            if self.random.random() < 0.5:
                template = self.random.choice(TEXTVQA_QUESTION_TEMPLATES_EN)
                question = template.format(question=question)

            yield {
                "image": image,
                "question": question,
                "answer": answer,
            }