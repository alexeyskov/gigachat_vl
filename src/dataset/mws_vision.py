import re
import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from huggingface_hub import hf_hub_download
from huggingface_hub import login

from src.dataset.dataset_base import DatasetConfig


def download_mws_vision(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Downloads MTSAIR/MWS-Vision-Bench (Russian version).

    Final structure:
    dataset_root/
    └── data/
        └── train-00000-of-00001.parquet
    """
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    if hf_token:
        login(token=hf_token)
    else:
        login()

    parquet_path = Path("data/train-00000-of-00001.parquet")

    if not (dataset_root_path / parquet_path).exists() or force_redownload:
        hf_hub_download(
            repo_id="MTSAIR/MWS-Vision-Bench",
            filename="data/train-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=dataset_root_path,
            local_dir_use_symlinks=False,
            force_download=force_redownload,
            resume_download=True,
        )


def load_mws_vision(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads MTSAIR/MWS-Vision-Bench in streaming mode.
    """
    if dataset_root is not None:
        dataset_root = Path(dataset_root)
        parquet_files = sorted(dataset_root.rglob("*.parquet"))
        if parquet_files:
            ds = load_dataset(
                "parquet",
                data_files=[str(p) for p in parquet_files],
                streaming=True,
                split="train",
            )
        else:
            ds = load_dataset(
                "MTSAIR/MWS-Vision-Bench",
                split="train",
                streaming=True,
            )
    else:
        ds = load_dataset(
            "MTSAIR/MWS-Vision-Bench",
            split="train",
            streaming=True,
        )

    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if limit is not None:
        ds = ds.take(limit)

    return ds


class MWSVisionIterableDataset(TorchIterableDataset):
    """
    Converts raw MWS-Vision-Bench samples into unified format:
    {"image": PIL.Image, "question": str, "answer": str}
    """

    def __init__(
        self,
        raw_hf_iterable,
        dataset_root: Optional[str] = None,
        seed: int = 42,
        skip_missing_images: bool = True,
    ):
        self.raw_hf_iterable = raw_hf_iterable
        self.seed = seed
        self.skip_missing_images = skip_missing_images
        self.random = random.Random(seed)

    def _is_bounding_box_task(self, question: str) -> bool:
        pattern = r"x1\s*,\s*y1\s*,\s*x2\s*,\s*y2"
        return bool(re.search(pattern, question, re.IGNORECASE))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self.raw_hf_iterable:
            image = row.get("image")
            if image is None and self.skip_missing_images:
                continue

            question = row.get("question", "").strip()
            answers = row.get("answers", [])

            if not question:
                continue

            if isinstance(answers, str):
                answers = [answers]
            elif not isinstance(answers, (list, tuple)):
                answers = [str(answers)]

            answers = [str(a).strip() for a in answers if str(a).strip()]

            if not answers:
                continue

            if len(answers) == 4 and self._is_bounding_box_task(question):
                x1, y1, x2, y2 = answers
                answer = f"({x1}, {y1}, {x2}, {y2})"
            else:
                answer = self.random.choice(answers)

            yield {
                "image": image,
                "question": question,
                "answer": answer,
            }
