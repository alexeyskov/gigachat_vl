import random
import re
from typing import Any

from src.reasoning_data_prep.schemas import ReasoningRequest


DATASET_NAME = "mm_vet_v2_en"
IMAGE_TAG_PATTERN = re.compile(r"<\s*/?\s*img\s*>|<\s*image_\d+\s*>", re.IGNORECASE)


def build_mm_vet_v2_en_reasoning_request(
    sample: dict[str, Any],
) -> ReasoningRequest:
    if _has_extra_images(sample):
        raise ValueError("MM-Vet-v2 sample uses multiple images; only image_0 is supported")

    question = _normalize_question(sample.get("question"))
    if not question:
        raise ValueError("MM-Vet-v2 sample is missing question")

    answer = _normalize_answer(sample.get("answer"))
    if not answer:
        raise ValueError("MM-Vet-v2 sample is missing usable answer")

    source_sample_id = str(sample.get("id", "")).strip()
    if not source_sample_id:
        raise ValueError("MM-Vet-v2 sample is missing id")

    user_message_lines = [
        "По изображению, английскому вопросу и правильному ответу составь русскоязычный обучающий пример.",
        "Верни question_ru: естественный перевод исходного вопроса на русский язык.",
        "Не сохраняй технические теги изображений, служебную разметку или специальные маркеры.",
        "Сохрани смысл вопроса и не добавляй новые требования.",
        "Напиши reasoning_ru: краткое рассуждение на русском от лица ассистента, который решает задачу по изображению.",
        "Никогда не упоминай, что правильный ответ был дан заранее и не допускай утечки информации из ответа в сформированный вопрос",
        "Верни answer_ru как перевод, максимально близкий к исходному правильному ответу по смыслу, форме и краткости.",
        "Не расширяй ответ, не добавляй пояснения и не заменяй короткий ответ более длинной переформулировкой без необходимости.",
        "",
        f"English question: {question}",
        f"Correct answer: {answer}",
    ]

    return ReasoningRequest(
        source_dataset=DATASET_NAME,
        source_sample_id=source_sample_id,
        image=sample.get("image_0"),
        system_message=(
            "Ты готовишь русскоязычные обучающие данные для VLM. "
            "Верни question_ru, reasoning_ru и answer_ru строго по заданной схеме."
        ),
        user_message="\n".join(user_message_lines),
        metadata={
            "added_in": sample.get("added_in"),
        },
    )


def _has_extra_images(sample: dict[str, Any]) -> bool:
    for idx in range(1, 18):
        value = sample.get(f"image_{idx}")
        if value is not None:
            return True
    return False


def _normalize_question(value: Any) -> str:
    question = str(value or "")
    question = IMAGE_TAG_PATTERN.sub(" ", question)
    question = re.sub(r"\s+", " ", question).strip()
    return question


def _normalize_answer(value: Any) -> str:
    answer = str(value or "").strip()
    if not answer:
        return ""

    if "<OR>" in answer:
        variants = [part.strip() for part in answer.split("<OR>") if part.strip()]
        if variants:
            answer = random.choice(variants)

    if "<AND>" in answer:
        parts = [part.strip() for part in answer.split("<AND>") if part.strip()]
        answer = " and ".join(parts)

    answer = re.sub(r"\s+", " ", answer).strip()
    return answer
