import random
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, List

from datasets import load_dataset, get_dataset_config_names
from datasets import IterableDataset as HFDataset
from datasets import interleave_datasets
from torch.utils.data import IterableDataset as TorchIterableDataset
from huggingface_hub import snapshot_download, login

from src.dataset.dataset_base import DatasetConfig


REPO_ID = "mnezhinskii/ru-vlm-reasoning-sft"

REASONING_TEMPLATES = [
    {
        "question": (
            "Реши задачу по изображению.\n"
            "Ответ верни строго в формате:\n"
            "Рассуждение: <краткое рассуждение>\n"
            "Ответ: <итоговый ответ>\n\n"
            "Вопрос: {question}"
        ),
        "answer": "Рассуждение: {reasoning}\nОтвет: {answer}",
    },
    {
        "question": (
            "Посмотри на изображение и ответь на вопрос.\n"
            "Нужны ровно две строки ответа:\n"
            "Ход мысли: <краткое рассуждение>\n"
            "Итог: <финальный ответ>\n\n"
            "Вопрос: {question}"
        ),
        "answer": "Ход мысли: {reasoning}\nИтог: {answer}",
    },
    {
        "question": (
            "Проанализируй изображение и реши задачу.\n"
            "Сначала напиши блок 'Объяснение:', затем блок 'Финальный ответ:'.\n\n"
            "{question}"
        ),
        "answer": "Объяснение: {reasoning}\nФинальный ответ: {answer}",
    },
    {
        "question": (
            "Найди решение по изображению.\n"
            "Строго используй два раздела:\n"
            "[Разбор]\n"
            "<краткое рассуждение>\n"
            "[Ответ]\n"
            "<финальный ответ>\n\n"
            "Задача: {question}"
        ),
        "answer": "[Разбор]\n{reasoning}\n[Ответ]\n{answer}",
    },
    {
        "question": (
            "Реши задачу по изображению.\n"
            "Сначала после слов 'Короткое рассуждение:' запиши краткое объяснение,\n"
            "потом после слов 'Решение:' запиши только финальный ответ.\n\n"
            "Вопрос: {question}"
        ),
        "answer": "Короткое рассуждение: {reasoning}\nРешение: {answer}",
    },
    {
        "question": (
            "Дай ответ на вопрос по изображению.\n"
            "Рассуждение не пиши.\n"
            "Верни результат только в виде строки 'Ответ: <значение>'.\n\n"
            "Вопрос: {question}"
        ),
        "answer": "Ответ: {answer}",
        "reasoning_required": False,
    },
    {
        "question": (
            "Посмотри на изображение и реши задачу.\n"
            "Нужен только итог без пояснений.\n"
            "Запиши его после слов 'Итоговый ответ:'.\n\n"
            "{question}"
        ),
        "answer": "Итоговый ответ: {answer}",
        "reasoning_required": False,
    },
]


def download_ru_vlm_reasoning_sft(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    """
    Downloads mnezhinskii/ru-vlm-reasoning-sft parquet files.

    Expected local structure:
    dataset_root/
    ├── README.md
    ├── mme_en/
    │   └── *.parquet
    ├── seed_bench_en/
    │   └── *.parquet
    └── ...
    """
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    if hf_token:
        login(token=hf_token)

    has_parquet_files = any(dataset_root_path.rglob("*.parquet"))

    if not has_parquet_files or force_redownload:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(dataset_root_path),
            local_dir_use_symlinks=False,
            force_download=force_redownload,
            resume_download=True,
            allow_patterns=["*.parquet", "README.md"],
        )


def load_ru_vlm_reasoning_sft(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 1000,
    seed: int = 42,
    dataset_root: Optional[str] = None,
) -> HFDataset:
    """
    Loads the published ru-vlm-reasoning-sft dataset.

    Behavior:
    - If dataset_root is provided and contains parquet files, loads all parquet
      shards recursively as one streaming dataset.
    - Otherwise loads all configs from Hugging Face and interleaves them.
    """
    if dataset_root is not None:
        dataset_root_path = Path(dataset_root)
        parquet_files = sorted(dataset_root_path.rglob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {dataset_root_path}. "
                "Run download_ru_vlm_reasoning_sft(...) first."
            )

        ds = load_dataset(
            "parquet",
            data_files=[str(p) for p in parquet_files],
            streaming=True,
            split="train",
        )
    else:
        config_names = get_dataset_config_names(REPO_ID)
        streams: List[HFDataset] = []
        for config_name in config_names:
            streams.append(
                load_dataset(
                    REPO_ID,
                    config_name,
                    streaming=True,
                    split="train",
                )
            )

        if not streams:
            raise ValueError(f"No configs found for dataset {REPO_ID}")

        ds = streams[0] if len(streams) == 1 else interleave_datasets(streams)

    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if limit is not None:
        ds = ds.take(limit)

    return ds


class RuVLMReasoningSFTIterableDataset(TorchIterableDataset):
    """
    Converts ru-vlm-reasoning-sft rows into the unified format:
    {"image": PIL.Image | None, "question": str, "answer": str}

    If reasoning_ru is present, wraps the sample into an instruction template
    that usually expects reasoning followed by the final answer. A small share
    of templates requests only the final answer. If reasoning_ru is absent,
    the sample is returned as a plain question-answer pair.
    """

    def __init__(
        self,
        raw_hf_iterable,
        dataset_root: Optional[str] = None,
        seed: int = 42,
        skip_missing_images: bool = True,
    ):
        self.raw_hf_iterable = raw_hf_iterable
        self.seed = seed
        self.skip_missing_images = skip_missing_images
        self.random = random.Random(seed)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self.raw_hf_iterable:
            if not isinstance(row, dict):
                continue

            question = str(row.get("question_ru", "")).strip()
            answer = str(row.get("answer_ru", "")).strip()
            reasoning = str(row.get("reasoning_ru", "") or "").strip()

            if not question or not answer:
                continue

            image = row.get("image")
            if image is None and self.skip_missing_images:
                continue

            if reasoning:
                template = self.random.choices(
                    REASONING_TEMPLATES,
                    weights=[1, 1, 1, 1, 1, 0.5, 0.5],
                    k=1,
                )[0]
                wrapped_question = template["question"].format(question=question)
                if template.get("reasoning_required", True):
                    wrapped_answer = template["answer"].format(
                        reasoning=reasoning,
                        answer=answer,
                    )
                else:
                    wrapped_answer = template["answer"].format(answer=answer)
            else:
                wrapped_question = question
                wrapped_answer = answer

            yield {
                "image": image,
                "question": wrapped_question,
                "answer": wrapped_answer,
            }
