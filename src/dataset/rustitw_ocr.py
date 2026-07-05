import os
import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, List

from datasets import load_dataset, concatenate_datasets
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.finevision import open_image
from src.dataset.dataset_base import DatasetConfig, OCR_QUESTION_TEMPLATES


def _rustitw_parts_for_root(dataset_root: Path) -> List[Dict[str, Any]]:
    parts = []
    for split in ["train", "test"]:
        csv_path = dataset_root / split / "real" / "info.csv"
        if csv_path.exists():
            parts.append(
                {
                    "csv": csv_path,
                    "images_dir": dataset_root / split / "real" / "images",
                    "split": split,
                }
            )
    return parts


def _find_rustitw_parts(dataset_root: Path) -> List[Dict[str, Any]]:
    direct_parts = _rustitw_parts_for_root(dataset_root)
    if direct_parts:
        if len(direct_parts) < 2:
            found_splits = ", ".join(part["split"] for part in direct_parts)
            print(
                "Warning: RusTitW OCR loader found only these direct splits: "
                f"{found_splits}. Continuing with available data."
            )
        return direct_parts

    candidate_roots = []
    seen_roots = set()
    for csv_path in sorted(dataset_root.rglob("info.csv")):
        if csv_path.parent.name != "real":
            continue
        split_dir = csv_path.parent.parent
        if split_dir.name not in {"train", "test"}:
            continue
        candidate_root = split_dir.parent
        if candidate_root in seen_roots:
            continue
        candidate_roots.append(candidate_root)
        seen_roots.add(candidate_root)

    candidate_roots.sort(key=lambda path: len(path.relative_to(dataset_root).parts))
    best_partial_parts = None
    best_partial_root = None
    for candidate_root in candidate_roots:
        parts = _rustitw_parts_for_root(candidate_root)
        if len(parts) == 2:
            print(f"Resolved RusTitW OCR nested dataset root: {candidate_root}")
            return parts
        if parts and best_partial_parts is None:
            best_partial_parts = parts
            best_partial_root = candidate_root

    if best_partial_parts is not None:
        found_splits = ", ".join(part["split"] for part in best_partial_parts)
        print(
            "Warning: RusTitW OCR loader found only these nested splits under "
            f"{best_partial_root}: {found_splits}. Continuing with available data."
        )
        return best_partial_parts

    found_info_csv = [str(path) for path in sorted(dataset_root.rglob("info.csv"))[:20]]
    details = (
        "\nFound info.csv candidates:\n" + "\n".join(found_info_csv)
        if found_info_csv
        else "\nNo info.csv files were found under dataset_root."
    )
    raise FileNotFoundError(
        "RusTitW OCR dataset was not found. Expected either "
        f"{dataset_root}/train/real/info.csv and {dataset_root}/test/real/info.csv, "
        "or the same train/test/real layout inside a nested directory."
        f"{details}"
    )


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

    parts = _find_rustitw_parts(dataset_root)

    datasets = []
    for part in parts:
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
