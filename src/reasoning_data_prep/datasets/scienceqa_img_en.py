from typing import Any

from src.reasoning_data_prep.schemas import (
    ReasoningRequest,
    build_fallback_source_sample_id,
)


DATASET_NAME = "scienceqa_img_en"


def build_scienceqa_img_en_reasoning_request(
    sample: dict[str, Any],
) -> ReasoningRequest:
    question = str(sample.get("question", "")).strip()
    if not question:
        raise ValueError("ScienceQA-IMG sample is missing question")

    choices = _extract_choices(sample.get("choices"))
    if len(choices) < 2:
        raise ValueError("ScienceQA-IMG sample must contain at least two choices")

    answer_index = _normalize_answer_index(sample.get("answer"))
    if answer_index is None or answer_index < 0 or answer_index >= len(choices):
        raise ValueError("ScienceQA-IMG sample has invalid answer index")

    correct_answer_text = choices[answer_index]
    hint = str(sample.get("hint", "")).strip()
    source_sample_id = _build_source_sample_id(sample, question, choices, answer_index)

    user_message_lines = [
        "По изображению, английскому вопросу, подсказке, вариантам ответа и правильному ответу составь русскоязычный обучающий пример.",
        "Верни question_ru: естественный перевод исходного вопроса на русский язык вместе с вариантами ответа.",
        "Обязательно учитывай подсказку при формулировке question_ru.",
        "Всегда сохраняй варианты ответа в финальном question_ru и переводи их на русский.",
        "Не используй нумерацию, буквы или индексы вариантов. Просто перечисли варианты ответа в естественной форме.",
        "Не добавляй новые требования и не искажай учебную постановку задачи.",
        "Напиши reasoning_ru: краткое рассуждение на русском от лица ассистента, который решает задачу по изображению и подсказке.",
        "Не упоминай, что правильный ответ был дан заранее.",
        "Верни answer_ru как краткий содержательный финальный ответ на русском по смыслу правильного варианта.",
        "",
        f"English question: {question}",
        f"Hint: {hint}" if hint else "Hint: (none)",
        "Answer choices:",
    ]

    for idx, choice in enumerate(choices):
        user_message_lines.append(f"{idx}. {choice}")

    user_message_lines.extend(
        [
            f"Correct answer index: {answer_index}",
            f"Correct answer text: {correct_answer_text}",
        ]
    )

    return ReasoningRequest(
        source_dataset=DATASET_NAME,
        source_sample_id=source_sample_id,
        image=sample.get("image"),
        system_message=(
            "Ты готовишь русскоязычные обучающие данные для VLM. "
            "Верни question_ru, reasoning_ru и answer_ru строго по заданной схеме."
        ),
        user_message="\n".join(user_message_lines),
        metadata={
            "task": sample.get("task"),
            "grade": sample.get("grade"),
            "subject": sample.get("subject"),
            "topic": sample.get("topic"),
            "category": sample.get("category"),
            "skill": sample.get("skill"),
        },
    )


def _extract_choices(raw_choices: Any) -> list[str]:
    if not isinstance(raw_choices, (list, tuple)):
        return []

    choices: list[str] = []
    for choice in raw_choices:
        value = str(choice).strip()
        if value:
            choices.append(value)
    return choices


def _normalize_answer_index(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_source_sample_id(
    sample: dict[str, Any],
    question: str,
    choices: list[str],
    answer_index: int,
) -> str:
    for key in ("question_id", "data_id", "id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value)

    return build_fallback_source_sample_id(
        {
            "question": question,
            "choices": choices,
            "answer_index": answer_index,
            "hint": sample.get("hint"),
        }
    )
