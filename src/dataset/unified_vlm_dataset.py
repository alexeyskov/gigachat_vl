import random
from enum import Enum
from typing import Any, List, Dict, Optional, Callable, Literal

from datasets import interleave_datasets, Features, Value
from datasets import Image as DatasetsImage
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset


from src.dataset.dataset_base import DatasetConfig
from src.dataset.llava_pretrain_ru import (
    download_llava_pretrain_ru,
    load_llava_pretrain_ru, 
    LLaVAPretrainRuIterableDataset
)
from src.dataset.mscoco_caption_ml import (
    download_mscoco_caption_ml,
    load_mscoco_caption_ml,
    MSCOCOCaptionMlIterableDataset
)
from src.dataset.gqa_ru import (
    download_gqa_ru,
    load_gqa_ru,
    GQARUIterableDataset
)
from src.dataset.llava_instruct_ru import (
    download_llava_instruct_ru,
    load_llava_instruct_ru, 
    LLaVAInstructRuIterableDataset
)
from src.dataset.rustitw_ocr import (
    load_rustitw_ocr,
    RusTitWOCRIterableDataset
)
from src.dataset.openhermes_ru_text import (
    download_openhermes_ru_text,
    load_openhermes_ru_text,
    OpenHermesRuIterableDataset
)

class SupportedDatasets(Enum):
    LLAVA_PRETRAIN_RU = DatasetConfig(
        name="maya-multimodal/pretrain",
        total_samples=550_000,
        load_raw_func=load_llava_pretrain_ru,
        dataset_class=LLaVAPretrainRuIterableDataset,
        download_func=download_llava_pretrain_ru,
        requires_download=True,
    )

    MSCOCO_CAPTION_ML = DatasetConfig(
        name="piyushsinghpasi/mscoco-multilingual-30k",
        total_samples=30_000,
        load_raw_func=load_mscoco_caption_ml,
        dataset_class=MSCOCOCaptionMlIterableDataset,
        download_func=download_mscoco_caption_ml,
        requires_download=False,
    )

    GQA_RU = DatasetConfig(
        name="deepvk/GQA-ru",
        total_samples=52_216,
        load_raw_func=load_gqa_ru,
        dataset_class=GQARUIterableDataset,
        download_func=download_gqa_ru,
        requires_download=True,
    )

    LLAVA_INSTRUCT_RU = DatasetConfig(
        name="deepvk/LLaVA-Instruct-ru",
        total_samples=143_980,
        load_raw_func=load_llava_instruct_ru,
        dataset_class=LLaVAInstructRuIterableDataset,
        download_func=download_llava_instruct_ru,
        requires_download=True,
    )

    RUSTITW_OCR = DatasetConfig(
        name="rustitw_ru",
        total_samples=28_000,
        load_raw_func=load_rustitw_ocr,
        dataset_class=RusTitWOCRIterableDataset,
        download_func=None,
        requires_download=True,
    )

    OPENHERMES_RU_TEXT = DatasetConfig(
        name="d0rj/OpenHermes-2.5-ru",
        total_samples=1_000_000,
        load_raw_func=load_openhermes_ru_text,
        dataset_class=OpenHermesRuIterableDataset,
        download_func=download_openhermes_ru_text,
        requires_download=True,
    )


class MixedTorchIterableDataset(TorchIterableDataset):
    """
    Easy interleaving of multiple TorchIterableDatasets with support for probabilities and stopping_strategy.
    Works as fast as possible, without Arrow/HF overhead.
    """
    def __init__(
        self,
        datasets: List[TorchIterableDataset],
        probabilities: Optional[List[float]] = None,
        seed: int = 42,
        stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'all_exhausted',
    ):
        self.datasets = datasets
        self.probabilities = probabilities
        self.seed = seed
        self.stopping_strategy = stopping_strategy
        self.random = random.Random(seed)

    def __iter__(self):
        iters = [iter(ds) for ds in self.datasets]
        probs = self.probabilities[:] if self.probabilities is not None else None

        while iters:
            if probs is None:
                idx = self.random.randint(0, len(iters) - 1)
            else:
                idx = self.random.choices(range(len(iters)), weights=probs, k=1)[0]

            try:
                yield next(iters[idx])
            except StopIteration:
                if self.stopping_strategy == 'first_exhausted':
                    return
                del iters[idx]
                if probs is not None:
                    del probs[idx]
                if not iters:
                    return

def load_merged_dataset(
    dataset_specs: List[Dict[str, Any]],
    global_seed: int = 42,
    global_shuffle_buffer: int = 10000,
    interleave_stopping_strategy: Literal['first_exhausted', 'all_exhausted']='all_exhausted',
    interleave_balance_probabilities: bool = False
) -> MixedTorchIterableDataset:
    """
    Creates a single lazy streaming IterableDataset by mixing several datasets.

    Args:
        dataset_specs: List of dicts, one per dataset. Example:
            [
                {
                    "config": VLSource.LLAVA_PRETRAIN_RU.value,
                    "limit": 200_000,          # or None (use full dataset)
                    "dataset_root": "data/llava_pretrain_ru",   # required for datasets with requires_download=True
                },
                {
                    "config": VLSource.MSCOCO_CAPTION_RU.value,
                    "limit": None,
                    "dataset_root": "data/mscoco_caption_ru",
                },
            ]
        global_seed: Seed used for all shuffles and random operations.
        global_shuffle_buffer: Buffer size used inside each load_raw_func.
    Returns:
        A single streaming IterableDataset in the unified format:
        {"image": PIL.Image, "question": str, "answer": str}
    """
    if not dataset_specs:
        raise ValueError("dataset_specs cannot be empty")

    custom_datasets: List[TorchIterableDataset] = []
    effective_sizes = []

    for spec in dataset_specs:
        config: DatasetConfig = spec["config"]
        limit: Optional[int] = spec.get("limit")
        dataset_root: Optional[str] = spec.get("dataset_root")

        if config.load_raw_func is None or config.dataset_class is None:
            raise ValueError(f"Dataset {config.name} is not fully configured for loading")

        # 1. Get raw HF iterable
        raw_ds = config.load_raw_func(
            config=config,
            limit=limit,
            shuffle_buffer=global_shuffle_buffer,
            seed=global_seed,
            dataset_root=dataset_root,
        )

        # 2. Wrap with the dataset-specific converter (LLaVAPretrainRuIterableDataset / MSCOCOCaptionRuIterableDataset etc.)
        #    This step turns raw data into the unified {"image", "question", "answer"} format
        custom_ds: TorchIterableDataset = config.dataset_class(
            raw_hf_iterable=raw_ds,
            dataset_root=dataset_root,
            seed=global_seed,
            skip_missing_images=True,
        )

        custom_datasets.append(custom_ds)

        if interleave_balance_probabilities:
            size = config.total_samples
            if limit is not None:
                size = min(size, limit) if size is not None else limit
            if size != None:
                effective_sizes.append(size)

    if len(custom_datasets) == 1:
        return custom_datasets[0]

    probabilities = None
    if interleave_balance_probabilities and len(effective_sizes) == len(custom_datasets):
        total = sum(effective_sizes)
        if total > 0:
            probabilities = [size / total for size in effective_sizes]

    return MixedTorchIterableDataset(
        datasets=custom_datasets,
        probabilities=probabilities,
        seed=global_seed,
        stopping_strategy=interleave_stopping_strategy,
    )