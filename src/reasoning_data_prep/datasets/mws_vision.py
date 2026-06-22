import re
from typing import Any

from src.reasoning_data_prep.schemas import (
    PreparedReasoningRecord,
    ReasoningRequest,
    build_fallback_source_sample_id,
)


DATASET_NAME = "mws_vision"
MAX_REASONING_ANSWER_CHARS = 1000


def build_mws_vision_reasoning_request(
    sample: dict[str, Any],
) -> ReasoningRequest | PreparedReasoningRecord:
    question = str(sample.get("question", "")).strip()
    if not question:
        raise ValueError("MWS-Vision sample is missing question")

    normalized_answers = _normalize_answers(sample.get("answers", []), question)
    if not normalized_answers:
        raise ValueError("MWS-Vision sample has no usable answers")

    primary_answer = normalized_answers[0]
    source_sample_id = str(sample.get("id"))

    if _should_skip_reasoning(question, primary_answer):
        return PreparedReasoningRecord(
            source_dataset=DATASET_NAME,
            source_sample_id=source_sample_id,
            image=sample.get("image"),
            question_ru=question,
            answer_ru=primary_answer,
            reasoning_ru=None,
        )

    user_message_lines = [
        "По изображению, вопросу и правильному ответу составь обучающий пример.",
        "Верни question_ru: исходный вопрос без инструкций о формате, длине ответа, markdown-разметке, запрете писать лишнее или требовании отвечать только кратко.",
        "Сохрани смысл вопроса и не добавляй новые требования.",
        "Напиши reasoning_ru: краткое рассуждение от лица ассистента, который решает задачу по изображению.",
        "Не упоминай, что правильный ответ был дан заранее.",
        "Верни финальный ответ без изменений.",
        "",
        f"Исходный вопрос: {question}",
        f"Правильный ответ: {primary_answer}",
    ]

    return ReasoningRequest(
        source_dataset=DATASET_NAME,
        source_sample_id=source_sample_id,
        image=sample.get("image"),
        system_message=(
            "Ты готовишь русскоязычные обучающие данные для VLM. "
            "Верни question_ru, reasoning_ru и answer_ru строго по заданной схеме."
        ),
        user_message="\n".join(user_message_lines),
    )


def _normalize_answers(
    answers: Any,
    question: str,
) -> list[str]:
    if isinstance(answers, str):
        normalized_answers = [answers]
    elif isinstance(answers, (list, tuple)):
        normalized_answers = [str(answer).strip() for answer in answers]
    elif answers is None:
        normalized_answers = []
    else:
        normalized_answers = [str(answers).strip()]

    normalized_answers = [answer for answer in normalized_answers if answer]
    if len(normalized_answers) == 4 and _is_bounding_box_task(question):
        x1, y1, x2, y2 = normalized_answers
        return [f"({x1}, {y1}, {x2}, {y2})"]
    return normalized_answers


def _is_bounding_box_task(question: str) -> bool:
    pattern = r"x1\s*,\s*y1\s*,\s*x2\s*,\s*y2"
    return bool(re.search(pattern, question, re.IGNORECASE))


def _is_markdown_request(question: str) -> bool:
    pattern = (
        r"\bmark\s*down\b|"
        r"\bmarkdown\b|"
        r"\bmakrdown\b|"
        r"\bmd\b|"
        r"марк\s*даун\w*|"
        r"маркдаун\w*|"
        r"маркадун\w*|"
        r"маркдоун\w*|"
        r"маркдовн\w*"
    )
    return bool(re.search(pattern, question, re.IGNORECASE))


def _should_skip_reasoning(
    question: str,
    answer: str,
) -> bool:
    return _is_markdown_request(question) or len(answer) >= MAX_REASONING_ANSWER_CHARS
