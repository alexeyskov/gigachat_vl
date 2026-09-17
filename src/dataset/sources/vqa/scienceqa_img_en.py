import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.dataset_base import DatasetConfig
from src.dataset.huggingface_utils import download_parquet_files

SCIENCEQA_QUESTION_TEMPLATES_EN = [
    (
        "Question: {question}\n\n"
        "{context}"
        "Choices:\n{choices}\n\n"
        "Explain your reasoning and give the final answer."
    ),
    (
        "Use the image and the provided information to solve the following problem.\n\n"
        "Question: {question}\n\n"
        "{context}"
        "Choices:\n{choices}\n\n"
        "Provide your reasoning followed by the final answer."
    ),
    (
        "{question}\n\n"
        "{context}"
        "Choices:\n{choices}\n\n"
        "Work through the problem and state the final answer."
    ),
    (
        "Answer the following question using the image and any provided context. "
        "Explain how you reached the answer.\n\n"
        "Question: {question}\n\n"
        "{context}"
        "Choices:\n{choices}"
    ),
    (
        "Determine the correct answer to the following problem. "
        "Give a short explanation before the final answer.\n\n"
        "Question: {question}\n\n"
        "{context}"
        "Choices:\n{choices}"
    ),
]


SCIENCEQA_REASONING_ANSWER_TEMPLATES_EN = [
    (
        "Lecture: {lecture}\n"
        "Solution: {solution}\n"
        "Answer: {answer}"
    ),
    (
        "Relevant knowledge: {lecture}\n"
        "Reasoning: {solution}\n"
        "Final answer: {answer}"
    ),
    (
        "{lecture}\n\n"
        "Reasoning: {solution}\n"
        "Answer: {answer}"
    ),
]


SCIENCEQA_SOLUTION_ONLY_ANSWER_TEMPLATES_EN = [
    (
        "Reasoning: {solution}\n"
        "Answer: {answer}"
    ),
    (
        "Explanation: {solution}\n"
        "Final answer: {answer}"
    ),
    (
        "{solution}\n\n"
        "Answer: {answer}"
    ),
]

def download_scienceqa_img_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = None,
) -> None:
    """
    Downloads lmms-lab/ScienceQA-IMG parquet files from the data/ folder.

    Expected local structure:
    dataset_root/
    └── data/
        ├── train-*.parquet
        ├── validation-*.parquet
        └── test-*.parquet
    """
    download_parquet_files(
        repo_id="lmms-lab/ScienceQA-IMG",
        dataset_root=dataset_root,
        force_redownload=force_redownload,
        hf_token=hf_token,
        max_parquet_files=max_parquet_files,
        parquet_pattern="data/*.parquet",
    )


def load_scienceqa_img_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 500,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads local ScienceQA-IMG parquet files in streaming mode.

    All local parquet shards under data/ are loaded together, regardless of
    whether they belong to train, validation, or test.
    """
    if dataset_root is None:
        raise ValueError(
            "For SCIENCEQA_IMG_EN dataset_root is required. "
            "Expected dataset_root/data/*.parquet files."
        )

    dataset_root_path = Path(dataset_root)
    data_dir = dataset_root_path / "data"

    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No ScienceQA-IMG parquet files found under {data_dir}. "
            "Run download_scienceqa_img_en(...) first."
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


class ScienceQAImgEnIterableDataset(TorchIterableDataset):
    """
    Converts ScienceQA-IMG rows into English visual reasoning samples.

    Raw fields used:
        - image
        - question
        - choices
        - answer
        - hint
        - lecture
        - solution

    Input:
        image + question + optional hint/context + choices

    Target:
        lecture + solution + final answer

    Unified format:
        {
            "image": PIL.Image,
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

            hint = str(
                row.get("hint", "") or ""
            ).strip()

            lecture = str(
                row.get("lecture", "") or ""
            ).strip()

            solution = str(
                row.get("solution", "") or ""
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
                    row.get("answer")
                )
            except (TypeError, ValueError):
                continue

            if not 0 <= answer_index < len(choices):
                continue

            # We use ScienceQA specifically as a reasoning dataset.
            # If neither lecture nor solution is available, the sample
            # does not add reasoning supervision.
            if not lecture and not solution:
                continue

            choice_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            formatted_choices = "\n".join(
                f"{choice_letters[index]}. {choice}"
                for index, choice in enumerate(choices)
            )

            formatted_context = (
                f"Context: {hint}\n\n"
                if hint
                else ""
            )

            question_template = self.random.choice(
                SCIENCEQA_QUESTION_TEMPLATES_EN
            )

            formatted_question = question_template.format(
                question=question,
                context=formatted_context,
                choices=formatted_choices,
            )

            final_answer = (
                f"{choice_letters[answer_index]}. "
                f"{choices[answer_index]}"
            )

            if lecture and solution:
                answer_template = self.random.choice(
                    SCIENCEQA_REASONING_ANSWER_TEMPLATES_EN
                )

                formatted_answer = answer_template.format(
                    lecture=lecture,
                    solution=solution,
                    answer=final_answer,
                )

            elif solution:
                answer_template = self.random.choice(
                    SCIENCEQA_SOLUTION_ONLY_ANSWER_TEMPLATES_EN
                )

                formatted_answer = answer_template.format(
                    solution=solution,
                    answer=final_answer,
                )

            else:
                # Rare fallback: lecture exists but solution does not.
                formatted_answer = (
                    f"Relevant knowledge: {lecture}\n"
                    f"Final answer: {final_answer}"
                )

            yield {
                "image": image,
                "question": formatted_question,
                "answer": formatted_answer,
            }