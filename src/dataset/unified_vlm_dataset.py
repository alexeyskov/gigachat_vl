from enum import Enum
from typing import Any, List, Dict, Optional, Callable, Literal

from datasets import interleave_datasets, Features, Value
from datasets import Image as DatasetsImage
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset


from src.dataset.dataset_config import DatasetConfig
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

UNIFIED_VLM_FEATURES = Features({
    "image":    DatasetsImage(),
    "question": Value("string"),
    "answer":   Value("string"),
})

def load_merged_dataset(
    dataset_specs: List[Dict[str, Any]],
    global_seed: int = 42,
    global_shuffle_buffer: int = 10000,
    interleave_stopping_strategy: Literal['first_exhausted', 'all_exhausted']='all_exhausted',
    interleave_balance_probabilities: bool = False
) -> HFDataset:
    """
    Creates a single lazy streaming IterableDataset by mixing several datasets.

    Args:
        dataset_specs: List of dicts, one per dataset. Example:
            [
                {
                    "config": SupportedDatasets.LLAVA_PRETRAIN_RU.value,
                    "limit": 200_000,          # or None (use full dataset)
                    "dataset_root": "data/llava_pretrain_ru",   # required for datasets with requires_download=True
                },
                {
                    "config": SupportedDatasets.MSCOCO_CAPTION_ML.value,
                    "dataset_root": "data/mscoco_caption_ru",
                },
            ]
        global_seed: Seed used for all shuffles and random operations.
        global_shuffle_buffer: Buffer size used inside each load_raw_func.
        interleave_stopping_strategy: stopping_strategy for interleave_datasets
        interleave_balance_probabilities: If True, and all datasets have size information (config.total_samples or limit),
            then probabilities will be calculated in proportion to their actual size.
    Returns:
        A single streaming IterableDataset in the unified format:
        {"image": PIL.Image, "question": str, "answer": str}
    """
    if not dataset_specs:
        raise ValueError("dataset_specs cannot be empty")

    wrapped_hf_datasets: List[HFDataset] = []
    probabilities: Optional[List[float]] = None
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

        # 3. wrap Torch IterableDataset with HF IterableDataset
        def make_generator(custom_iterable: TorchIterableDataset) -> Callable:
            def generator():
                yield from custom_iterable
            return generator

        hf_ds = HFDataset.from_generator(
            make_generator(custom_ds),
            features=UNIFIED_VLM_FEATURES,
        )

        wrapped_hf_datasets.append(hf_ds)

        if interleave_balance_probabilities:
            size = config.total_samples
            if limit is not None:
                size = min(size, limit) if size is not None else limit
            if size != None:
                effective_sizes.append(size)

    if interleave_balance_probabilities and len(effective_sizes) == len(wrapped_hf_datasets):
        total = sum(effective_sizes)
        if total > 0:
            probabilities = [size / total for size in effective_sizes]
        else:
            probabilities = None
    else:
        probabilities = None

    # 4. Mix all wrapped datasets with interleave_datasets
    if len(wrapped_hf_datasets) == 1:
        final_ds = wrapped_hf_datasets[0]
    else:
        final_ds = interleave_datasets(
            wrapped_hf_datasets,
            probabilities=probabilities,
            seed=global_seed,
            stopping_strategy=interleave_stopping_strategy,
        )

    return final_ds