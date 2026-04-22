import os
import zipfile
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, Union

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from huggingface_hub import hf_hub_download
from huggingface_hub import login

from src.dataset.finevision import open_image
from src.dataset.dataset_base import DatasetConfig


def download_llava_instruct_ru(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Downloads translated instructions from deepvk/LLaVA-Instruct-ru and images
    from adamo1139/llava-instruct-150k-with-images.

    Example structure:
    dataset_root/
    ├── llava_instruct_ru_train.json
    ├── llava_instruct_ru_val.json
    └── train2017/
        ├── 000000000009.jpg
        ├── 000000000025.jpg
        ...
    """
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    json_filenames = [
        "llava_instruct_ru_train.json",
        "llava_instruct_ru_val.json"
    ]
    zip_filename = "train2017.zip"
    images_dir = dataset_root / "train2017"

    zip_path = dataset_root / zip_filename

    if hf_token:
        login(token=hf_token)
    else:
        login()

    for json_filename in json_filenames:
        json_path = dataset_root / json_filename
        if not json_path.exists() or force_redownload:
            hf_hub_download(
                repo_id="deepvk/LLaVA-Instruct-ru",
                filename=json_filename,
                repo_type="dataset",
                local_dir=dataset_root,
                local_dir_use_symlinks=False,
                force_download=force_redownload,
                resume_download=True,
            )

    if not images_dir.exists() or force_redownload or not any(images_dir.iterdir()):
        if not zip_path.exists() or force_redownload:
            hf_hub_download(
                repo_id="adamo1139/llava-instruct-150k-with-images",
                filename=zip_filename,
                repo_type="dataset",
                local_dir=dataset_root,
                local_dir_use_symlinks=False,
                force_download=force_redownload,
                resume_download=True,
            )

        images_dir.mkdir(exist_ok=True)

        print("Unziping...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(path=dataset_root)

        zip_path.unlink()


def load_llava_instruct_ru(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads both train and val JSON files into a single streaming dataset.
    """
    missing_files_error_msg = (
        "For LLaVA_INSTRUCT_RU dataset_root is required (the folder containing "
        "llava_instruct_ru_train.json, llava_instruct_ru_val.json + train2017/ folder)"
    )

    if dataset_root is None:
        raise ValueError(missing_files_error_msg)

    json_files = [
        os.path.join(dataset_root, "llava_instruct_ru_train.json"),
        os.path.join(dataset_root, "llava_instruct_ru_val.json"),
    ]
    images_path = os.path.join(dataset_root, "train2017")

    if not all(os.path.exists(f) for f in json_files) or not os.path.exists(images_path):
        raise ValueError(missing_files_error_msg)

    ds = load_dataset(
        "json",
        data_files=json_files,
        streaming=True,
        split="train"
    )

    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if limit is not None:
        ds = ds.take(limit)

    return ds

class LLaVAInstructRuIterableDataset(TorchIterableDataset):
    """
    Converts raw LLaVA-Instruct-ru samples into the unified format:
    {"image": PIL.Image, "question": str, "answer": str}

    Takes only the first turn from conversations (human → gpt).
    Images are loaded from train2017/{filename}.jpg
    """
    def __init__(
        self,
        raw_hf_iterable: Union[HFDataset, str, os.PathLike],
        dataset_root: str,
        seed: int = 42,
        skip_missing_images: bool = True,
    ):
        self.seed = seed
        self.raw_hf_iterable = self._resolve_raw_iterable(raw_hf_iterable)
        self.dataset_root = dataset_root
        self.skip_missing_images = skip_missing_images

    def _resolve_raw_iterable(
        self,
        raw_hf_iterable: Union[HFDataset, str, os.PathLike],
    ) -> HFDataset:
        if isinstance(raw_hf_iterable, (str, os.PathLike)):
            json_path = os.fspath(raw_hf_iterable)
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"LLaVA instruct json file not found: {json_path}")

            ds = load_dataset(
                "json",
                data_files=json_path,
                streaming=True,
                split="train",
            )
            return ds.shuffle(seed=self.seed, buffer_size=10_000)

        return raw_hf_iterable

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self.raw_hf_iterable:
            if not isinstance(row, dict):
                continue

            conversations = row.get("conversations", [])
            if len(conversations) < 2:
                continue

            image_rel_path = row.get("image")
            if not image_rel_path:
                continue

            image_filename = os.path.basename(image_rel_path)
            image_path = os.path.join(self.dataset_root, "train2017", image_filename)

            if self.skip_missing_images and not os.path.exists(image_path):
                continue

            try:
                image = open_image(image_path)
            except Exception:
                continue

            question = conversations[0].get("value", "").strip().replace("<image>", "").replace("\n", "")
            answer = conversations[1].get("value", "").strip()

            if not question or not answer:
                continue

            yield {
                "image": image,
                "question": question,
                "answer": answer,
            }
