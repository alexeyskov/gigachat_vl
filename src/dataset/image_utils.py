import io
from typing import Any, Dict, List

from PIL import Image


def open_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, str):
        return Image.open(value).convert("RGB")

    if isinstance(value, dict):
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")

    raise ValueError(f"Unsupported image field type: {type(value)}")


def load_images_from_example(example: Dict[str, Any]) -> List[Image.Image]:
    raw_images = (
        example["images"]
        if example.get("images") is not None
        else example.get("image")
    )
    if raw_images is None:
        return []
    if isinstance(raw_images, (list, tuple)):
        return [open_image(image) for image in raw_images]
    return [open_image(raw_images)]
