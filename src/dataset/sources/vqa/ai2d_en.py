from pathlib import Path
from typing import Optional

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.dataset_base import DatasetConfig
from src.dataset.huggingface_utils import download_parquet_files


def download_ai2d_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = None,
) -> None:
    """
    Downloads lmms-lab-encoder/ai2d parquet shards.

    The repository contains the dataset under the test split, but we use it
    as training data.

    Expected local structure:
        dataset_root/
        └── data/
            ├── test-00000-of-00002.parquet
            └── test-00001-of-00002.parquet
    """
    download_parquet_files(
        repo_id="lmms-lab-encoder/ai2d",
        dataset_root=dataset_root,
        force_redownload=force_redownload,
        hf_token=hf_token,
        max_parquet_files=max_parquet_files,
        parquet_pattern="data/test-*.parquet",
    )


def load_ai2d_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 500,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads local AI2D parquet shards in streaming mode.
    """
    if dataset_root is None:
        raise ValueError(
            "For AI2D_EN dataset_root is required. "
            "Expected dataset_root/data/test-*.parquet files."
        )

    data_dir = Path(dataset_root) / "data"

    parquet_files = sorted(
        data_dir.glob("test-*.parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No AI2D parquet files found under {data_dir}. "
            "Run download_ai2d_en(...) first."
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


class AI2DEnIterableDataset(TorchIterableDataset):
    """
    Converts AI2D rows into multiple-choice visual reasoning samples.

    Raw schema:
        question: str
        options: list[str]
        answer: str  # index of the correct option
        image: PIL.Image

    Unified schema:
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

            raw_options = row.get("options")

            if not isinstance(raw_options, (list, tuple)):
                continue

            options = [
                str(option).strip()
                for option in raw_options
            ]

            if not question or not options or any(
                not option for option in options
            ):
                continue

            try:
                answer_index = int(row.get("answer"))
            except (TypeError, ValueError):
                continue

            if not 0 <= answer_index < len(options):
                continue

            choice_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            formatted_choices = "\n".join(
                f"{choice_letters[index]}. {option}"
                for index, option in enumerate(options)
            )

            formatted_question = (
                f"{question}\n"
                f"Choices:\n"
                f"{formatted_choices}"
            )

            yield {
                "image": image,
                "question": formatted_question,
                "answer": options[answer_index],
            }