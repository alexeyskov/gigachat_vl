from pathlib import Path
from time import perf_counter
from typing import Optional

from datasets import load_dataset
from datasets import IterableDataset as HFDataset
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.dataset_base import DatasetConfig
from src.dataset.huggingface_utils import download_parquet_files


def download_seed_bench_en(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
    max_parquet_files: Optional[int] = None,
) -> None:
    """
    Downloads lmms-lab/SEED-Bench parquet shards.

    The original repository contains 273 test shards:
        data/test-00000-of-00273.parquet
        data/test-00001-of-00273.parquet
        ...

    Set max_parquet_files to download a deterministic prefix for a smoke test.

    Expected local structure:
        dataset_root/
        └── data/
            ├── test-00000-of-00273.parquet
            ├── test-00001-of-00273.parquet
            └── ...
    """
    download_parquet_files(
        repo_id="lmms-lab/SEED-Bench",
        dataset_root=dataset_root,
        force_redownload=force_redownload,
        hf_token=hf_token,
        max_parquet_files=max_parquet_files,
    )


def load_seed_bench_en(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 500,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads local SEED-Bench parquet shards in streaming mode.

    This loader intentionally expects local parquet files, because the full
    repository is large. Use download_seed_bench_en(..., max_parquet_files=N)
    to prepare a subset.
    """
    if dataset_root is None:
        raise ValueError(
            "For SEED_BENCH_EN dataset_root is required. "
            "Expected dataset_root/data/test-xxxxx-of-00273.parquet files."
        )

    dataset_root_path = Path(dataset_root)
    data_dir = dataset_root_path / "data"

    parquet_files = sorted(data_dir.glob("test-*-of-00273.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No SEED-Bench parquet shards found under {data_dir}. "
            "Run download_seed_bench_en(...) first."
        )

    ds = load_dataset(
        "parquet",
        data_files=[str(p) for p in parquet_files],
        streaming=True,
        split="train",
    )

    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if limit is not None:
        ds = ds.take(limit)

    return ds


class SeedBenchEnIterableDataset(TorchIterableDataset):
    """Converts SEED-Bench rows into English multiple-choice VQA samples."""

    _OPTION_KEYS = ("choice_a", "choice_b", "choice_c", "choice_d")

    def __init__(
        self,
        raw_hf_iterable,
        dataset_root=None,
        seed=42,
        skip_missing_images=True,
        log_slow_samples_after_seconds: Optional[float] = None,
    ):
        self.raw_hf_iterable = raw_hf_iterable
        self.skip_missing_images = skip_missing_images
        self.log_slow_samples_after_seconds = log_slow_samples_after_seconds

    def __iter__(self):
        iterator = iter(self.raw_hf_iterable)
        while True:
            started_at = perf_counter()
            try:
                row = next(iterator)
            except StopIteration:
                return
            raw_row_latency = perf_counter() - started_at
            image = row.get("image")
            if isinstance(image, list):
                image = image[0] if image else None
            if (
                self.log_slow_samples_after_seconds is not None
                and raw_row_latency >= self.log_slow_samples_after_seconds
            ):
                print(
                    "Slow SEED-Bench row: "
                    f"{raw_row_latency:.3f}s, "
                    f"question_id={row.get('question_id')!r}, "
                    f"data_id={row.get('data_id')!r}, "
                    f"data_type={row.get('data_type')!r}, "
                    f"image_size={getattr(image, 'size', None)!r}"
                )
            question = str(row.get("question", "")).strip()
            choices = [str(row.get(key, "")).strip() for key in self._OPTION_KEYS]
            answer_letter = str(row.get("answer", "")).strip().upper()
            if image is None and self.skip_missing_images:
                continue
            if not question or not all(choices) or answer_letter not in "ABCD":
                continue
            answer_index = ord(answer_letter) - ord("A")
            formatted_choices = "\n".join(
                f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)
            )
            yield {
                "image": image,
                "question": f"{question}\nChoices:\n{formatted_choices}",
                "answer": choices[answer_index],
            }
