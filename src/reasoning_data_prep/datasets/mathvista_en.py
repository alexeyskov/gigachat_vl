from typing import Any

from src.reasoning_data_prep.schemas import (
    ReasoningRequest,
    build_fallback_source_sample_id,
)


DATASET_NAME = "mathvista_en"


def build_mathvista_en_reasoning_request(
    sample: dict[str, Any],
) -> ReasoningRequest:
    question = str(sample.get("question", "")).strip()
    if not question:
        raise ValueError("MathVista sample is missing question")

    answer = str(sample.get("answer", "")).strip()
    if not answer:
        raise ValueError("MathVista sample is missing answer")

    query = str(sample.get("query", "")).strip()
    question_type = str(sample.get("question_type", "")).strip()
    answer_type = str(sample.get("answer_type", "")).strip()
    choices = _extract_choices(sample.get("choices"))
    source_sample_id = _build_source_sample_id(sample, question, answer)

    if question_type == "multi_choice" and choices:
        user_message_lines = [
            "По изображению, английскому вопросу, дополнительной инструкции и правильному ответу составь русскоязычный обучающий пример.",
            "Верни question_ru: естественный перевод исходного вопроса на русский язык вместе с вариантами ответа.",
            "Сохрани варианты ответа в финальном question_ru и переведи их на русский.",
            "Не используй буквы, индексы или нумерацию вариантов. Просто перечисли варианты ответа в естественной форме.",
            "Учитывай дополнительную инструкцию из query, если она есть, и естественно включи её в формулировку question_ru.",
            "Напиши reasoning_ru: краткое рассуждение на русском от лица ассистента, который решает задачу по изображению.",
            "Не упоминай, что правильный ответ был дан заранее.",
            "Верни answer_ru как перевод, максимально близкий к исходному правильному ответу по смыслу, форме и краткости.",
            "",
            f"English question: {question}",
            f"Additional instruction: {query}" if query else "Additional instruction: (none)",
            "Answer choices:",
        ]
        for choice in choices:
            user_message_lines.append(f"- {choice}")
        user_message_lines.append(f"Correct answer: {answer}")
    else:
        user_message_lines = [
            "По изображению, английскому вопросу, дополнительной инструкции и правильному ответу составь русскоязычный обучающий пример.",
            "Верни question_ru: естественный перевод исходного вопроса на русский язык.",
            "Учитывай дополнительную инструкцию из query, если она есть, и естественно включи её в формулировку question_ru.",
            "Сохрани смысл вопроса и не добавляй новые требования.",
            "Напиши reasoning_ru: краткое рассуждение на русском от лица ассистента, который решает задачу по изображению.",
            "Не упоминай, что правильный ответ был дан заранее.",
            "Верни answer_ru как перевод, максимально близкий к исходному правильному ответу по смыслу, форме и краткости.",
            "",
            f"English question: {question}",
            f"Additional instruction: {query}" if query else "Additional instruction: (none)",
            f"Correct answer: {answer}",
        ]

    return ReasoningRequest(
        source_dataset=DATASET_NAME,
        source_sample_id=source_sample_id,
        image=sample.get("decoded_image"),
        system_message=(
            "Ты готовишь русскоязычные обучающие данные для VLM. "
            "Верни question_ru, reasoning_ru и answer_ru строго по заданной схеме."
        ),
        user_message="\n".join(user_message_lines),
        metadata={
            "question_type": question_type,
            "answer_type": answer_type,
            "unit": sample.get("unit"),
            "precision": sample.get("precision"),
        },
    )


def _extract_choices(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []

    choices: list[str] = []
    for item in value:
        normalized = str(item).strip()
        if normalized:
            choices.append(normalized)
    return choices


def _build_source_sample_id(
    sample: dict[str, Any],
    question: str,
    answer: str,
) -> str:
    for key in ("pid", "id", "question_id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value)

    return build_fallback_source_sample_id(
        {
            "question": question,
            "answer": answer,
            "query": sample.get("query"),
        }
    )
