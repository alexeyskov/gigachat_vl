import random
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.dataset_base import DatasetConfig
from src.dataset.huggingface_utils import download_parquet_files

A_OKVQA_QUESTION_TEMPLATES_EN = [
    (
        "{question}\n\n"
        "Choices:\n{choices}"
    ),
    (
        "Answer the question using the image and explain your reasoning.\n\n"
        "Question: {question}\n\n"
        "Choices:\n{choices}"
    ),
    (
        "Look at the image and determine the best answer. "
        "Briefly explain how you arrived at it.\n\n"
        "Question: {question}\n\n"
        "Choices:\n{choices}"
    ),
    (
        "Use the visual information and your general knowledge to answer the question. "
        "Provide a short explanation before the final answer.\n\n"
        "Question: {question}\n\n"
        "Choices:\n{choices}"
    ),
    (
        "Choose the most appropriate answer based on the image. "
        "Explain your reasoning briefly.\n\n"
        "{question}\n\n"
        "Choices:\n{choices}"
    ),
]

A_OKVQA_ANSWER_TEMPLATES_EN = [
    (
        "Reasoning: {rationale}\n"
        "Answer: {answer}"
    ),
    (
        "Explanation: {rationale}\n"
        "Final answer: {answer}"
    ),
    (
        "{rationale}\n\n"
        "Answer: {answer}"
    ),
]

def download_a_okvqa_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = None,
) -> None:
    download_parquet_files(
        repo_id="HuggingFaceM4/A-OKVQA",
        dataset_root=dataset_root,
        force_redownload=force_redownload,
        hf_token=hf_token,
        max_parquet_files=max_parquet_files,
        parquet_pattern="data/train-*.parquet",
    )


def load_a_okvqa_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 500,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    if dataset_root is None:
        raise ValueError(
            "For A_OKVQA_EN dataset_root is required. "
            "Expected dataset_root/data/train-*.parquet files."
        )

    data_dir = Path(dataset_root) / "data"

    parquet_files = sorted(
        data_dir.glob("train-*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No A-OKVQA train parquet files found under {data_dir}. "
            "Run download_a_okvqa_en(...) first."
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


class AOKVQAEnIterableDataset(TorchIterableDataset):
    """
    Converts A-OKVQA train rows into reasoning VQA samples.

    Raw fields:
        image
        question
        choices
        correct_choice_idx
        rationales

    Unified output:
        {
            "image": image,
            "question": str,
            "answer": str,
        }
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

    def __iter__(self):
        for row in self.raw_hf_iterable:
            if not isinstance(row, dict):
                continue

            image = row.get("image")

            if image is None and self.skip_missing_images:
                continue

            question = str(
                row.get("question", "")
            ).strip()

            raw_choices = row.get("choices")

            if not isinstance(raw_choices, (list, tuple)):
                continue

            choices = [
                str(choice).strip()
                for choice in raw_choices
            ]

            if (
                not question
                or not choices
                or any(not choice for choice in choices)
            ):
                continue

            try:
                answer_index = int(
                    row.get("correct_choice_idx")
                )
            except (TypeError, ValueError):
                continue

            if not 0 <= answer_index < len(choices):
                continue

            rationales = [
                str(rationale).strip()
                for rationale in (row.get("rationales") or [])
                if rationale is not None
                and str(rationale).strip()
            ]

            if not rationales:
                continue

            rationale = self.random.choice(
                rationales
            )

            choice_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            formatted_choices = "\n".join(
                f"{choice_letters[index]}. {choice}"
                for index, choice in enumerate(choices)
            )

            question_template = self.random.choice(
                A_OKVQA_QUESTION_TEMPLATES_EN
            )

            formatted_question = question_template.format(
                question=question,
                choices=formatted_choices,
            )

            final_answer = choices[answer_index]

            answer_template = self.random.choice(
                A_OKVQA_ANSWER_TEMPLATES_EN
            )

            formatted_answer = answer_template.format(
                rationale=rationale,
                answer=final_answer,
            )

            yield {
                "image": image,
                "question": formatted_question,
                "answer": formatted_answer,
            }