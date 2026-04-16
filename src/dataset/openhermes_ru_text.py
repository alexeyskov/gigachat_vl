import os
import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from huggingface_hub import snapshot_download
from huggingface_hub import login

from src.dataset.dataset_base import DatasetConfig

def download_openhermes_ru_text(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Downloads d0rj/OpenHermes-2.5-ru dataset to the specified directory using snapshot_download.

    Expected structure after download:
    dataset_root/
    └──data
       ├── train-00000-of-00006.parquet
       ├── train-00001-of-00006.parquet
    ...
    """
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    if hf_token:
        login(token=hf_token)
    else:
        login()

    if not any(dataset_root_path.iterdir()) or force_redownload:
        snapshot_download(
            repo_id="d0rj/OpenHermes-2.5-ru",
            repo_type="dataset",
            local_dir=str(dataset_root_path),
            local_dir_use_symlinks=False,
            force_download=force_redownload,
            resume_download=True,
        )


def load_openhermes_ru_text(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads the OpenHermes-2.5-ru dataset.

    Behavior:
    - If dataset_root is provided and contains parquet files → load locally
    - Otherwise → stream directly from Hugging Face Hub
    """
    if dataset_root is not None:
        dataset_root = Path(dataset_root)
        parquet_files = sorted(dataset_root.rglob("train-*.parquet"))
        print(parquet_files)
        if parquet_files:
            data_files = [str(p) for p in parquet_files]
            ds = load_dataset(
                "parquet",
                data_files=data_files,
                streaming=True,
                split="train",
            )
        else:
            ds = load_dataset(
                "d0rj/OpenHermes-2.5-ru",
                split="train",
                streaming=True,
            )
    else:
        ds = load_dataset(
            "d0rj/OpenHermes-2.5-ru",
            split="train",
            streaming=True,
        )

    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if limit is not None:
        ds = ds.take(limit)

    return ds


class OpenHermesRuIterableDataset(TorchIterableDataset):
    """
    Converts raw OpenHermes-2.5-ru samples into the unified format:
    {"image": None, "question": str, "answer": str}

    Expects 'conversations' column with list of dicts: [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]
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
        self.random = random.Random(seed)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self.raw_hf_iterable:
            if not isinstance(row, dict):
                continue

            conversations = row.get("conversations")
            if not conversations or not isinstance(conversations, list) or len(conversations) < 2:
                continue

            human_msg = conversations[0]
            gpt_msg = conversations[1]

            if human_msg.get("from") != "human" or gpt_msg.get("from") != "gpt":
                continue

            question = str(human_msg.get("value", "")).strip()
            answer = str(gpt_msg.get("value", "")).strip()

            if not question or not answer:
                continue

            yield {
                "image": None,
                "question": question,
                "answer": answer,
            }