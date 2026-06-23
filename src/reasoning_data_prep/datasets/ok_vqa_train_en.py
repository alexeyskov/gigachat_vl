import random
from typing import Any

from src.reasoning_data_prep.schemas import ReasoningRequest


DATASET_NAME = "ok_vqa_train_en"


def build_ok_vqa_train_en_reasoning_request(
    sample: dict[str, Any],
) -> ReasoningRequest:
    question = str(sample.get("question", "")).strip()
    if not question:
        raise ValueError("OK-VQA_train sample is missing question")

    answers = _extract_answers(sample.get("answers"))
    if not answers:
        raise ValueError("OK-VQA_train sample has no usable answers")

    answer = random.choice(answers)
    source_sample_id = str(sample.get("id", "")).strip()
    if not source_sample_id:
        raise ValueError("OK-VQA_train sample is missing id")

    user_message_lines = [
        "По изображению, английскому вопросу и правильному ответу составь русскоязычный обучающий пример.",
        "Верни question_ru: естественный перевод исходного вопроса на русский язык.",
        "Сохрани смысл вопроса и не добавляй новые требования.",
        "Напиши reasoning_ru: краткое рассуждение на русском от лица ассистента, который решает задачу по изображению.",
        "Не упоминай, что правильный ответ был дан заранее.",
        "Верни answer_ru как перевод, максимально близкий к исходному правильному ответу по смыслу, форме и краткости.",
        "",
        f"English question: {question}",
        f"Correct answer: {answer}",
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


def _extract_answers(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []

    answers: list[str] = []
    for item in value:
        if isinstance(item, dict):
            normalized = str(item.get("answer", "")).strip()
        else:
            normalized = str(item).strip()

        if normalized:
            answers.append(normalized)

    return answers
