import os
import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset, concatenate_datasets
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.finevision import open_image
from dataset.dataset_base import DatasetConfig, OCR_QUESTION_TEMPLATES

def load_rustitw_ocr(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads real parts from both train/real/info.csv and test/real/info.csv
    and concatenates them into one streaming dataset.

    Expected structure:
    dataset_root/
    ├── train/
    │   └── real/
    │       ├── images/
    │       ├── info.csv
    │       └── info_raw.csv
    └── test/
        └── real/
            ├── images/
            ├── info.csv
            └── info_raw.csv
    """
    if dataset_root is None:
        raise ValueError(
            "For RUSTITW_RU dataset_root is required "
            "(must contain train/real/info.csv and test/real/info.csv)"
        )

    dataset_root = Path(dataset_root).resolve()

    parts = [
        {
            "csv": dataset_root / "train/real/info.csv",
            "images_dir": dataset_root / "train/real/images",
            "split": "train"
        },
        {
            "csv": dataset_root / "test/real/info.csv",
            "images_dir": dataset_root / "test/real/images",
            "split": "test"
        },
    ]

    datasets = []
    for part in parts:
        if not part["csv"].exists():
            raise FileNotFoundError(f"CSV file not found: {part['csv']}")

        ds = load_dataset(
            "csv",
            data_files=str(part["csv"]),
            streaming=True,
            split="train",
        )

        def add_full_image_path(example, images_dir=part["images_dir"]):
            image_name = example.get("image_name")
            if image_name:
                example["image_path"] = str(images_dir / image_name)
            else:
                example["image_path"] = None
            return example

        ds = ds.map(add_full_image_path)
        datasets.append(ds)

    if len(datasets) > 1:
        combined_ds = concatenate_datasets(datasets, axis=0)
    else:
        combined_ds = datasets[0]

    if shuffle_buffer > 0:
        combined_ds = combined_ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if limit is not None:
        combined_ds = combined_ds.take(limit)

    return combined_ds



class RusTitWOCRIterableDataset(TorchIterableDataset):
    """
    Converts RusTitW real samples into unified format:
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
        self.skip_missing_images = skip_missing_images
        self.seed = seed
        self.random = random.Random(seed)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self.raw_hf_iterable:
            image_path = row.get("image_path")
            text = row.get("text", "").strip()

            if not image_path or not text:
                continue

            if self.skip_missing_images and not os.path.exists(image_path):
                continue

            try:
                image = open_image(image_path)
            except Exception:
                continue

            question = self.random.choice(OCR_QUESTION_TEMPLATES)

            yield {
                "image": image,
                "question": question,
                "answer": text,
            }
