import os
import zipfile
from pathlib import Path
from typing import Optional, Iterator, Dict, Any

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from huggingface_hub import hf_hub_download
from huggingface_hub import login

from src.dataset.finevision import open_image
from src.dataset.dataset_config import DatasetConfig

def download_llava_pretrain_ru(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Downloads RU part of maya-multimodal/pretrain + original images LLaVA-Pretrain.
    
    Example structure:
    dataset_root/
    ├── maya_russian_blip_laion_cc_sbu_558k.json
    └── images/
        ├── 0000/
        ├── 0001/
        ...
        └── 0053/
            └── *.jpg
    """
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    json_filename = "maya_russian_blip_laion_cc_sbu_558k.json"
    zip_filename = "images.zip"
    images_dir = dataset_root_path / "images"

    json_path = dataset_root_path / json_filename
    zip_path = dataset_root_path / zip_filename

    if hf_token:
        login(token=hf_token)
    else:
        login()

    if not json_path.exists() or force_redownload:
        hf_hub_download(
            repo_id="maya-multimodal/pretrain",
            filename=json_filename,
            repo_type="dataset",
            local_dir=dataset_root_path,
            local_dir_use_symlinks=False,
            force_download=force_redownload,
            resume_download=True,
        )

    if not images_dir.exists() or force_redownload or not any(images_dir.iterdir()):
        if not zip_path.exists() or force_redownload:
            hf_hub_download(
                repo_id="liuhaotian/LLaVA-Pretrain",
                filename=zip_filename,
                repo_type="dataset",
                local_dir=dataset_root_path,
                local_dir_use_symlinks=False,
                force_download=force_redownload,
                resume_download=True,
            )

        images_dir.mkdir(exist_ok=True)

        print("Unziping...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(path=dataset_root)

        zip_path.unlink()


def load_llava_pretrain_ru(
    config: DatasetConfig,
    limit: Optional[int] = None,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    missing_files_error_msg = (
        "For LLAVA_PRETRAIN_RU dataset_root is required (the folder containing "
        "maya_russian_blip_laion_cc_sbu_558k.json + images folder)"
    )

    if dataset_root is None:
        raise ValueError(missing_files_error_msg)

    russian_json_filename = "maya_russian_blip_laion_cc_sbu_558k.json"
    json_path = os.path.join(dataset_root, russian_json_filename)
    images_path = os.path.join(dataset_root, "images")

    if not os.path.exists(json_path) or not os.path.exists(images_path):
        raise ValueError(missing_files_error_msg)

    ds = load_dataset(
        "json",
        data_files=json_path,
        streaming=True,
        split="train"
    )

    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if limit is not None:
        ds = ds.take(limit)

    return ds

class LLaVAPretrainRuIterableDataset(TorchIterableDataset):
    """
    Convert raw samples maya_russian_blip_laion_cc_sbu_558k.json
    into {"image": PIL.Image, "question": str, "answer": str}
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
            conversations = row.get("conversations", [])
            if len(conversations) < 2:
                continue

            image_rel_path = row.get("image")
            if not image_rel_path:
                continue

            image_path = os.path.join(self.dataset_root, "images", image_rel_path)

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