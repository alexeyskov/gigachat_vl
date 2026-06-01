import os
import shutil
from itertools import chain
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from huggingface_hub import snapshot_download
from huggingface_hub import login

from src.dataset.finevision import open_image
from src.dataset.dataset_base import DatasetConfig


GQA_IMAGE_PARQUET_DIRS = ["train_balanced_images", "testdev_balanced_images"]


def _extract_gqa_images_from_parquets(
    dataset_root_path: Path,
    force_redownload: bool = False,
    cleanup_parquets: bool = False,
) -> None:
    images_dir = dataset_root_path / "images"
    images_dir.mkdir(exist_ok=True)

    found_parquets = False
    print(f"Extracting GQA images to {images_dir} ...")

    for subdir_name in GQA_IMAGE_PARQUET_DIRS:
        subdir_path = dataset_root_path / subdir_name
        if not subdir_path.exists():
            continue

        parquet_files = sorted(subdir_path.glob("*.parquet"))
        if parquet_files:
            found_parquets = True

        for parquet_path in parquet_files:
            print(f"  Processing {parquet_path.name} ...")

            ds = load_dataset(
                "parquet",
                data_files=str(parquet_path),
                streaming=True,
                split="train",
            )

            for row in ds:
                img_id = row["id"]
                image = row["image"]
                img_path = images_dir / f"{img_id}.jpg"

                if not img_path.exists() or force_redownload:
                    image.save(str(img_path), format="JPEG", quality=92)

        if cleanup_parquets and subdir_path.exists():
            shutil.rmtree(subdir_path, ignore_errors=True)

    if not found_parquets:
        raise ValueError(
            "GQA image parquet directories were not found under "
            f"{dataset_root_path}. Expected one of: {GQA_IMAGE_PARQUET_DIRS}"
        )

    print(
        f"Image extraction completed. "
        f"{len(list(images_dir.iterdir())):,} images available in {images_dir}"
    )


def download_gqa_ru(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Downloads deepvk/GQA-ru dataset and extracts all images into a flat images/ folder.
    
    Final structure after download:
    dataset_root/
    ├── train_balanced_instructions/
    │   └── train-00000-of-00001.parquet
    ├── testdev_balanced_instructions/     (optional)
    └── images/                            # extracted JPG files
            ├── n161313.jpg
            ├── n235859.jpg
            └── ...
    """
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root_path / "images"

    if hf_token:
        login(token=hf_token)
    else:
        login()

    # 1. Download the entire dataset using snapshot_download (no HF cache)
    if not any(dataset_root_path.iterdir()) or force_redownload:
        snapshot_download(
            repo_id="deepvk/GQA-ru",
            repo_type="dataset",
            local_dir=str(dataset_root_path),
            local_dir_use_symlinks=False,
            force_download=force_redownload,
            resume_download=True,
        )

    # 2. Extract images from parquet files into flat images/ directory
    if not images_dir.exists() or force_redownload or not any(images_dir.iterdir()):
        _extract_gqa_images_from_parquets(
            dataset_root_path=dataset_root_path,
            force_redownload=force_redownload,
            cleanup_parquets=True,
        )


def load_gqa_ru(
    config: DatasetConfig,
    limit: Optional[int] = None,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads both train and testdev instructions from GQA-ru in streaming mode 
    and concatenates them into a single dataset.
    
    Requires dataset_root with 'images/' folder and both:
    - train_balanced_instructions/
    - testdev_balanced_instructions/
    """
    if dataset_root is None:
        raise ValueError("For GQA_RU dataset_root is required (must contain 'images/' folder "
        "and both 'train_balanced_instructions/' and 'testdev_balanced_instructions/' folders)")

    instructions_dirs = [
        os.path.join(dataset_root, "train_balanced_instructions"),
        os.path.join(dataset_root, "testdev_balanced_instructions"),
    ]
    dataset_root_path = Path(dataset_root)
    images_dir = dataset_root_path / "images"

    if not images_dir.exists() or not any(images_dir.iterdir()):
        _extract_gqa_images_from_parquets(
            dataset_root_path=dataset_root_path,
            force_redownload=False,
            cleanup_parquets=False,
        )

    all_parquet_files = []

    for instr_dir in instructions_dirs:
        if not os.path.exists(instr_dir):
            raise ValueError(f"Instructions directory not found: {instr_dir}")
        
        parquet_files = [
            os.path.join(instr_dir, f)
            for f in os.listdir(instr_dir)
            if f.endswith(".parquet")
        ]
        all_parquet_files.extend(parquet_files)

    if not all_parquet_files:
        raise ValueError(f"No parquet files found in instructions directories under {dataset_root}")

    ds = load_dataset(
        "parquet",
        data_files=all_parquet_files,
        streaming=True,
        split="train",
    )

    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if limit is not None:
        ds = ds.take(limit)

    return ds


class GQARUIterableDataset(TorchIterableDataset):
    """
    Converts raw GQA-ru instructions into the unified format:
    {"image": PIL.Image, "question": str, "answer": str}
    
    Images are loaded from the pre-extracted images/{imageId}.jpg files.
    """
    def __init__(
        self,
        raw_hf_iterable,
        dataset_root: str,
        seed: int = 42,
        skip_missing_images: bool = True,
    ):
        self.raw_hf_iterable = raw_hf_iterable
        self.dataset_root = dataset_root
        self.seed = seed
        self.skip_missing_images = skip_missing_images

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self.raw_hf_iterable:
            image_id = row.get("imageId")
            if not image_id:
                continue

            image_path = os.path.join(self.dataset_root, "images", f"{image_id}.jpg")

            if self.skip_missing_images and not os.path.exists(image_path):
                continue

            try:
                image = open_image(image_path)
            except Exception:
                continue

            question = row.get("question", "").strip()
            # Prefer fullAnswer (more detailed), fallback to short answer
            answer = row.get("fullAnswer", "").strip() or row.get("answer", "").strip()

            if not question or not answer:
                continue

            yield {
                "image": image,
                "question": question,
                "answer": answer,
            }
