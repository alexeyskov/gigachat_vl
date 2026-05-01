from dataclasses import dataclass
from typing import Optional, Callable, Type

from torch.utils.data import IterableDataset as TorchIterableDataset


@dataclass
class DatasetConfig:
    """Configuration for each Russian VLM dataset"""

    name: str
    total_samples: Optional[int]

    # Core functions and classes
    load_raw_func: Callable  # function that returns raw HF iterable (streaming)
    dataset_class: Type[
        TorchIterableDataset
    ]  # class that converts raw -> {"image": PIL, "question": str, "answer": str}
    download_func: Optional[Callable] = None  # optional download function

    # Metadata
    requires_download: bool = (
        False  # True = needs manual download before use (local files required)
    )


CAPTIONING_QUESTION_TEMPLATES_EN = [
    "Describe this image in detail.",
    "Give a detailed description of this image.",
    "What is shown in this image? Describe it in detail.",
    "Describe everything visible in this image.",
    "Tell me what you see in this picture.",
    "Provide a comprehensive description of the scene.",
    "Describe the image as accurately and thoroughly as possible.",
    "What can you observe in this image?",
]

CAPTIONING_QUESTION_TEMPLATES = [
    "Опиши это изображение",
    "Что изображено на этой картинке?",
    "Расскажи, что ты видишь на этом изображении.",
    "Опиши данное изображение.",
    "Что находится на этой фотографии?",
    "Дай подробное описание этого изображения",
    "Расскажи, что показано на картинке.",
    "Опиши сцену, изображённую на фото.",
    "Что ты можешь рассказать об этом изображении?",
    "Опиши всё, что видно на этом изображении.",
]

OCR_QUESTION_TEMPLATES = [
    "Какой текст написан на этом изображении?",
    "Что написано на этой картинке?",
    "Прочитай текст на изображении.",
    "Какой текст изображён на фото?",
    "Что здесь написано?",
    "Расскажи, какой текст виден на этом изображении.",
    "Прочитай, что написано на картинке.",
    "Какой текст присутствует на этом фото?",
    "Опиши текст, который ты видишь на изображении.",
    "Что написано на этой фотографии?",
]
