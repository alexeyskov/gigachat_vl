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
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    missing_files_error_msg = (
        "For LLAVA_PRETRAIN_RU dataset_root is required (the folder containing "
        "maya_russian_blip_laion_cc_sbu_558k.json plus either the original shard "
        "folders like 00000/, 00001/, ... or an images/ folder)"
    )

    if dataset_root is None:
        raise ValueError(missing_files_error_msg)

    russian_json_filename = "maya_russian_blip_laion_cc_sbu_558k.json"
    json_path = os.path.join(dataset_root, russian_json_filename)

    if not os.path.exists(json_path):
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
                raise FileNotFoundError(f"LLaVA pretrain json file not found: {json_path}")

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

            image_path = self._resolve_image_path(image_rel_path)

            if image_path is None:
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

    def _resolve_image_path(self, image_rel_path: str) -> Optional[str]:
        candidates = [
            os.path.join(self.dataset_root, "images", image_rel_path),
            os.path.join(self.dataset_root, image_rel_path),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        if self.skip_missing_images:
            return None

        return candidates[0]
