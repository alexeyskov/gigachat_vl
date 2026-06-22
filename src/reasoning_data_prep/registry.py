from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from src.dataset.mws_vision import download_mws_vision, load_mws_vision
from src.reasoning_data_prep.datasets.mws_vision import (
    build_mws_vision_reasoning_request,
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
}


def get_reasoning_dataset_config(name: str) -> ReasoningDatasetConfig:
    if name not in SUPPORTED_REASONING_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_REASONING_DATASETS))
        raise KeyError(f"Unknown reasoning dataset: {name}. Supported: {supported}")
    return SUPPORTED_REASONING_DATASETS[name]
