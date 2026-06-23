import random
from typing import Any

from src.reasoning_data_prep.schemas import ReasoningRequest


DATASET_NAME = "seed_bench_en"
OPTION_KEYS = ("choice_a", "choice_b", "choice_c", "choice_d")


def build_seed_bench_en_reasoning_request(
    sample: dict[str, Any],
) -> ReasoningRequest:
    question = str(sample.get("question", "")).strip()
    if not question:
        raise ValueError("SEED-Bench sample is missing question")

    choices = _extract_choices(sample)
    if len(choices) != 4:
        raise ValueError("SEED-Bench sample must contain four answer choices")

    answer_letter = str(sample.get("answer", "")).strip().upper()
    if answer_letter not in choices:
        raise ValueError("SEED-Bench sample has unsupported answer letter")

    correct_option_text = choices[answer_letter]
    source_sample_id = str(sample.get("question_id", "")).strip()
    if not source_sample_id:
        raise ValueError("SEED-Bench sample is missing question_id")

    answer_mode = "letter_only" if random.random() < 0.5 else "letter_plus_text"
    user_message_lines = [
        "По изображению, английскому вопросу с вариантами ответа и правильному ответу составь русскоязычный обучающий пример.",
        "Верни question_ru: естественный перевод исходного вопроса на русский язык вместе с вариантами ответа.",
        "Сохрани формат multiple choice: вопрос и четыре варианта ответа A/B/C/D, но переведи всё на русский естественно.",
        "Не добавляй новые требования и не меняй количество вариантов.",
        "Напиши reasoning_ru: краткое рассуждение на русском от лица ассистента, который решает задачу по изображению.",
        "Не упоминай, что правильный ответ был дан заранее.",
        "",
        f"English question: {question}",
        "Answer choices:",
        f"A. {choices['A']}",
        f"B. {choices['B']}",
        f"C. {choices['C']}",
        f"D. {choices['D']}",
        f"Correct answer letter: {answer_letter}",
        f"Correct answer text: {correct_option_text}",
    ]
    if answer_mode == "letter_only":
        user_message_lines.insert(
            3,
            "Допиши в конце question_ru естественную инструкцию ответить только одной буквой: A, B, C или D.",
        )
        user_message_lines.insert(
            7,
            "Верни answer_ru только как одну букву правильного ответа: A, B, C или D.",
        )
    else:
        user_message_lines.insert(
            3,
            "Допиши в конце question_ru естественную инструкцию выбрать правильный вариант ответа.",
        )
        user_message_lines.insert(
            7,
            "Верни answer_ru в формате 'C. <переведённый правильный вариант ответа>', то есть буква плюс текст правильного варианта на русском.",
        )

    return ReasoningRequest(
        source_dataset=DATASET_NAME,
        source_sample_id=source_sample_id,
        image=_extract_image(sample),
        system_message=(
            "Ты готовишь русскоязычные обучающие данные для VLM. "
            "Верни question_ru, reasoning_ru и answer_ru строго по заданной схеме."
        ),
        user_message="\n".join(user_message_lines),
        metadata={
            "data_type": sample.get("data_type"),
            "data_id": sample.get("data_id"),
            "question_type_id": sample.get("question_type_id"),
            "answer_mode": answer_mode,
        },
    )


def _extract_choices(sample: dict[str, Any]) -> dict[str, str]:
    letters = ("A", "B", "C", "D")
    choices: dict[str, str] = {}

    for letter, key in zip(letters, OPTION_KEYS):
        value = str(sample.get(key, "")).strip()
        if not value:
            continue
        choices[letter] = value

    return choices


def _extract_image(sample: dict[str, Any]) -> Any:
    image = sample.get("image")
    if isinstance(image, list):
        if not image:
            return None
        return image[0]
    return image
