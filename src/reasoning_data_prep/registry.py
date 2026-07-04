from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from src.dataset.mme_en import download_mme_en, load_mme_en
from src.dataset.mm_vet_v2_en import download_mm_vet_v2_en, load_mm_vet_v2_en
from src.dataset.mws_vision import download_mws_vision, load_mws_vision
from src.dataset.mathvista_en import download_mathvista_en, load_mathvista_en
from src.dataset.scienceqa_img_en import (
    download_scienceqa_img_en,
    load_scienceqa_img_en,
)
from src.dataset.ok_vqa_train_en import (
    download_ok_vqa_train_en,
    load_ok_vqa_train_en,
)
from src.dataset.seed_bench_en import download_seed_bench_en, load_seed_bench_en
from src.reasoning_data_prep.datasets.mme_en import build_mme_en_reasoning_request
from src.reasoning_data_prep.datasets.mm_vet_v2_en import (
    build_mm_vet_v2_en_reasoning_request,
)
from src.reasoning_data_prep.datasets.mathvista_en import (
    build_mathvista_en_reasoning_request,
)
from src.reasoning_data_prep.datasets.ok_vqa_train_en import (
    build_ok_vqa_train_en_reasoning_request,
)
from src.reasoning_data_prep.datasets.mws_vision import (
    build_mws_vision_reasoning_request,
)
from src.reasoning_data_prep.datasets.scienceqa_img_en import (
    build_scienceqa_img_en_reasoning_request,
)
from src.reasoning_data_prep.datasets.seed_bench_en import (
    build_seed_bench_en_reasoning_request,
)
from src.reasoning_data_prep.schemas import (
    PreparedReasoningRecord,
    ReasoningRequest,
)


ReasoningBuildResult = ReasoningRequest | PreparedReasoningRecord


@dataclass
class ReasoningDatasetConfig:
    name: str
    download_func: Callable[..., None]
    load_func: Callable[..., Iterable[dict[str, Any]]]
    build_request_func: Callable[[dict[str, Any], str], ReasoningBuildResult]


SUPPORTED_REASONING_DATASETS = {
    "mws_vision": ReasoningDatasetConfig(
        name="mws_vision",
        download_func=download_mws_vision,
        load_func=load_mws_vision,
        build_request_func=build_mws_vision_reasoning_request,
    ),
    "mme_en": ReasoningDatasetConfig(
        name="mme_en",
        download_func=download_mme_en,
        load_func=load_mme_en,
        build_request_func=build_mme_en_reasoning_request,
    ),
    "seed_bench_en": ReasoningDatasetConfig(
        name="seed_bench_en",
        download_func=download_seed_bench_en,
        load_func=load_seed_bench_en,
        build_request_func=build_seed_bench_en_reasoning_request,
    ),
    "scienceqa_img_en": ReasoningDatasetConfig(
        name="scienceqa_img_en",
        download_func=download_scienceqa_img_en,
        load_func=load_scienceqa_img_en,
        build_request_func=build_scienceqa_img_en_reasoning_request,
    ),
    "mm_vet_v2_en": ReasoningDatasetConfig(
        name="mm_vet_v2_en",
        download_func=download_mm_vet_v2_en,
        load_func=load_mm_vet_v2_en,
        build_request_func=build_mm_vet_v2_en_reasoning_request,
    ),
    "mathvista_en": ReasoningDatasetConfig(
        name="mathvista_en",
        download_func=download_mathvista_en,
        load_func=load_mathvista_en,
        build_request_func=build_mathvista_en_reasoning_request,
    ),
    "ok_vqa_train_en": ReasoningDatasetConfig(
        name="ok_vqa_train_en",
        download_func=download_ok_vqa_train_en,
        load_func=load_ok_vqa_train_en,
        build_request_func=build_ok_vqa_train_en_reasoning_request,
    ),
}


def get_reasoning_dataset_config(name: str) -> ReasoningDatasetConfig:
    if name not in SUPPORTED_REASONING_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_REASONING_DATASETS))
        raise KeyError(f"Unknown reasoning dataset: {name}. Supported: {supported}")
    return SUPPORTED_REASONING_DATASETS[name]
