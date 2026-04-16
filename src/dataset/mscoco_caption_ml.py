import zipfile
import os
import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from huggingface_hub import hf_hub_download
from huggingface_hub import login
from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from dataset.dataset_base import DatasetConfig, CAPTIONING_QUESTION_TEMPLATES
from src.dataset.finevision import open_image

def download_mscoco_caption_ml(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Downloads piyushsinghpasi/mscoco-multilingual-30k (test.zip) 
    
    Example structure:
    dataset_root/
    ├── test/
    │   ├── COCO_val2014_000000XXXXXX.jpg
    │   ├── ...
    │   └── metadata.csv
    """
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    zip_filename = "test.zip"
    zip_path = dataset_root_path / zip_filename
    test_dir = dataset_root_path / "test"

    if hf_token:
        login(token=hf_token)
    else:
        login()

    if not test_dir.exists() or force_redownload or not any(test_dir.iterdir()):
        if not zip_path.exists() or force_redownload:
            hf_hub_download(
                repo_id="piyushsinghpasi/mscoco-multilingual-30k",
                filename=zip_filename,
                repo_type="dataset",
                local_dir=dataset_root_path,
                local_dir_use_symlinks=False,
                force_download=force_redownload,
                resume_download=True,
            )

        test_dir.mkdir(exist_ok=True)

        print("Unzipping test.zip...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(path=dataset_root_path)

        zip_path.unlink()


def load_mscoco_caption_ml(
    config: DatasetConfig,
    limit: Optional[int] = None,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Load the MSCOCO Multilingual 30k dataset.

    Behavior:
    - If dataset_root is provided AND the local test/ folder with metadata.csv exists,
      load from local disk.
    - Otherwise, load directly from Hugging Face Hub (streaming=True).

    Returns raw HF IterableDataset with original columns:
    ['file_name', 'image_name', 'caption', 'Russian', 'French', ...]
    """
    if dataset_root is not None and Path(dataset_root).exists():
        test_dir = Path(dataset_root) / "test"
        metadata_path = test_dir / "metadata.csv"
        images_dir = test_dir
        if metadata_path.exists() and images_dir.exists():
            ds = load_dataset(
                "csv",
                data_files=str(metadata_path),
                streaming=True,
                split="train",
            )

            def make_absolute_path(example):
                example["file_name"] = str(images_dir / example["file_name"])
                return example

            ds = ds.map(make_absolute_path)
    else:
        ds = load_dataset(
            config.name,
            split="test",
            streaming=True,
        )

    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if limit is not None:
        ds = ds.take(limit)

    return ds

class MSCOCOCaptionMlIterableDataset(TorchIterableDataset):
    """
    Converts raw MSCOCO Caption RU samples into the unified format:
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
        self.dataset_root = dataset_root
        self.seed = seed
        self.skip_missing_images = skip_missing_images
        self.random = random.Random(seed)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self.raw_hf_iterable:
            if "image" in row and row["image"] is not None:
                image = row["image"]
            elif "file_name" in row and row["file_name"]:
                image_path = row["file_name"]
                if self.skip_missing_images and not os.path.exists(image_path):
                    continue
                try:
                    image = open_image(image_path)
                except Exception:
                    continue
            else:
                continue

            russian_caption = row.get("Russian", "").strip()

            if not russian_caption:
                continue

            question = self.random.choice(CAPTIONING_QUESTION_TEMPLATES)

            yield {
                "image": image,
                "question": question,
                "answer": russian_caption,
            }