import asyncio
import base64
import io
from typing import Any, Optional

from openai import AsyncOpenAI
from PIL import Image

from src.reasoning_data_prep.schemas import (
    ReasoningRequest,
    ReasoningStructuredOutput,
)
from src.reasoning_data_prep.settings import SETTINGS


class TeacherClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.model = model or SETTINGS.TEACHER_MODEL
        self.client = AsyncOpenAI(
            api_key=api_key or SETTINGS.TEACHER_API_KEY.get_secret_value(),
            base_url=base_url or SETTINGS.TEACHER_BASE_URL,
            timeout=timeout or SETTINGS.TEACHER_TIMEOUT_SECONDS,
        )

    async def generate_reasoning_batch(
        self,
        requests: list[ReasoningRequest],
        retry_delay_seconds: float = 3.0,
        max_retries: int = 3,
    ) -> list[ReasoningStructuredOutput | None]:
        results: list[ReasoningStructuredOutput | None] = [None] * len(requests)
        pending_indices = list(range(len(requests)))
        messages_by_index = [_build_messages(request) for request in requests]

        for attempt in range(max_retries + 1):
            if not pending_indices:
                break

            batch_results = await asyncio.gather(
                *(
                    self._generate_reasoning(messages_by_index[index])
                    for index in pending_indices
                ),
                return_exceptions=True,
            )

            failed_indices: list[int] = []
            for index, result in zip(pending_indices, batch_results):
                if isinstance(result, Exception):
                    failed_indices.append(index)
                    continue
                results[index] = result

            pending_indices = failed_indices
            if pending_indices and attempt < max_retries:
                await asyncio.sleep(retry_delay_seconds)

        return results

    async def _generate_reasoning(
        self,
        messages: list[dict[str, Any]],
    ) -> ReasoningStructuredOutput:
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=ReasoningStructuredOutput,
        )

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("Teacher model returned an empty structured response")
        return parsed


def _build_messages(request: ReasoningRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if request.system_message:
        messages.append({"role": "system", "content": request.system_message})

    content: list[dict[str, Any]] = [{"type": "text", "text": request.user_message}]
    image_url = _pil_image_to_data_url(request.image)
    if image_url is not None:
        content.insert(0, {"type": "image_url", "image_url": {"url": image_url}})
    messages.append({"role": "user", "content": content})
    return messages


def _pil_image_to_data_url(image: Optional[Image.Image]) -> Optional[str]:
    if image is None:
        return None

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
