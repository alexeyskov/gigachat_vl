from typing import Any

from src.reasoning_data_prep.schemas import (
    PreparedReasoningRecord,
    ReasoningRequest,
    build_fallback_source_sample_id,
)


DATASET_NAME = "mme_en"


def build_mme_en_reasoning_request(
    sample: dict[str, Any],
) -> ReasoningRequest | PreparedReasoningRecord:
    question = str(sample.get("question", "")).strip()
    if not question:
        raise ValueError("MME sample is missing question")

    answer_en = sample.get("answer")
    if answer_en is None:
        raise ValueError("MME sample has unsupported answer; expected yes/no")

    source_sample_id = sample.get("question_id")

    user_message_lines = [
        "По изображению, английскому вопросу и правильному ответу составь русскоязычный обучающий пример.",
        "Верни question_ru: естественный перевод исходного вопроса на русский язык.",
        "Если в вопросе есть указание ответить yes/no, переведи естественно на русский.",
        "Сохрани исходный формат бинарного вопроса и не добавляй новые требования.",
        "Верни answer_ru только как 'Да' или 'Нет' в соответствии с правильным ответом.",
        "Напиши reasoning_ru: краткое рассуждение на русском от лица ассистента, который решает задачу по изображению.",
        "Не упоминай, что правильный ответ был дан заранее.",
        "",
        f"English question: {question}",
        f"Correct answer: {answer_en}",
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
        metadata={
            "category": sample.get("category"),
        },
    )
