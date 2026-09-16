import random
from typing import Any, List, Dict, Optional, Literal

from torch.utils.data import IterableDataset as TorchIterableDataset


from src.dataset.dataset_base import DatasetConfig
from src.dataset.precomputed_embeddings import PrecomputedVisionEmbeddingDataset

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
        stopping_strategy: Literal[
            "first_exhausted", "all_exhausted"
        ] = "all_exhausted",
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
                if self.stopping_strategy == "first_exhausted":
                    return
                del iters[idx]
                if probs is not None:
                    del probs[idx]
                if not iters:
                    return


def load_merged_dataset(
    dataset_specs: List[Dict[str, Any]],
    global_seed: int = 42,
    global_shuffle_buffer: int = 100,
    interleave_stopping_strategy: Literal[
        "first_exhausted", "all_exhausted"
    ] = "all_exhausted",
    interleave_balance_probabilities: bool = False,
    skip_missing_datasets: bool = False,
) -> MixedTorchIterableDataset:
    """
    Creates a single lazy streaming IterableDataset by mixing several datasets.

    Args:
        dataset_specs: List of dicts, one per dataset. Example:
            [
                {
                    "config": SupportedDatasets.LLAVA_PRETRAIN_RU.value,
                    "limit": 200_000,          # or None (use full dataset)
                    "dataset_root": "data/llava_pretrain_ru",
                },
                {
                    "config": SupportedDatasets.MSCOCO_CAPTION_RU.value,
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
        load_kwargs = dict(spec.get("load_kwargs") or {})
        dataset_kwargs = dict(spec.get("dataset_kwargs") or {})

        if config.load_raw_func is None or config.dataset_class is None:
            raise ValueError(
                f"Dataset {config.name} is not fully configured for loading"
            )

        if bool(spec.get("download", False)):
            if config.download_func is None:
                raise ValueError(
                    f"Dataset {config.name} does not provide a download function. "
                    "Prepare dataset_root manually and remove `download=True` from the spec."
                )
            config.download_func(dataset_root)

        # 1. Get raw HF iterable
        try:
            raw_ds = config.load_raw_func(
                config=config,
                limit=limit,
                shuffle_buffer=global_shuffle_buffer,
                seed=global_seed,
                dataset_root=dataset_root,
                **load_kwargs,
            )
        except FileNotFoundError as e:
            if skip_missing_datasets or bool(spec.get("skip_if_missing", False)):
                print(
                    f"Warning: skipping dataset {config.name} because local files "
                    f"are missing under dataset_root={dataset_root!r}: {e}"
                )
                continue
            raise

        # 2. Wrap with the dataset-specific converter (LLaVAPretrainRuIterableDataset / MSCOCOCaptionRuIterableDataset etc.)
        #    This step turns raw data into the unified {"image", "question", "answer"} format
        custom_ds: TorchIterableDataset = config.dataset_class(
            raw_hf_iterable=raw_ds,
            dataset_root=dataset_root,
            seed=global_seed,
            skip_missing_images=True,
            **dataset_kwargs,
        )

        visual_encoder = spec.get("visual_encoder")
        if visual_encoder is not None:
            custom_ds = PrecomputedVisionEmbeddingDataset(
                dataset=custom_ds,
                dataset_root=dataset_root,
                visual_encoder=visual_encoder,
                require_exists=bool(spec.get("require_precomputed_exists", True)),
            )

        custom_datasets.append(custom_ds)

        if interleave_balance_probabilities:
            size = config.total_samples
            if limit is not None:
                size = min(size, limit) if size is not None else limit
            if size != None:
                effective_sizes.append(size)

    if not custom_datasets:
        raise ValueError(
            "No datasets could be loaded. Check dataset_root paths or disable "
            "skip_missing_datasets."
        )

    if len(custom_datasets) == 1:
        return custom_datasets[0]

    probabilities = None
    if interleave_balance_probabilities and len(effective_sizes) == len(
        custom_datasets
    ):
        total = sum(effective_sizes)
        if total > 0:
            probabilities = [size / total for size in effective_sizes]

    return MixedTorchIterableDataset(
        datasets=custom_datasets,
        probabilities=probabilities,
        seed=global_seed,
        stopping_strategy=interleave_stopping_strategy,
    )
