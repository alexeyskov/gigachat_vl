import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from PIL import Image
from pydantic import BaseModel, Field


@dataclass
class ReasoningRequest:
    source_dataset: str
    source_sample_id: str
    image: Optional[Image.Image]
    user_message: str
    system_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningStructuredOutput(BaseModel):
    question_ru: str = Field(
        description=(
            "Question in Russian, cleaned from output-format requirements, "
            "length restrictions, markdown requirements, and 'write nothing else' clauses."
        ),
    )
    reasoning_ru: str = Field(
        description=(
            "Short reasoning in Russian from the assistant's perspective while solving "
            "the visual task. Do not mention that the correct answer was provided."
        ),
    )
    answer_ru: str = Field(
        description="Final answer in Russian, preserving the meaning of the provided correct answer.",
    )


@dataclass
class PreparedReasoningRecord:
    source_dataset: str
    source_sample_id: str
    image: Optional[Image.Image]
    question_ru: str
    answer_ru: str
    reasoning_ru: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_fallback_source_sample_id(
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    normalized_payload = payload or {}
    serialized_payload = json.dumps(
        normalized_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(serialized_payload.encode("utf-8")).hexdigest()

    return digest
