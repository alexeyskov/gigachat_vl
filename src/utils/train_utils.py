import os

import json
import torch

from src.model.gigachat_vl import GigaChatVL, IMAGE_TOKEN


def save_artifacts(model: GigaChatVL, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    vision_processor_dir = os.path.join(output_dir, "vision_processor")
    lora_dir = os.path.join(output_dir, "llm_lora")
    projector_path = os.path.join(output_dir, "projector.pt")
    meta_path = os.path.join(output_dir, "vlm_meta.json")

    model.tokenizer.save_pretrained(tokenizer_dir)

    if model.vision_processor is not None and hasattr(model.vision_processor, "save_pretrained"):
        model.vision_processor.save_pretrained(vision_processor_dir)

    model.llm.save_pretrained(lora_dir)
    torch.save(model.projector.state_dict(), projector_path)

    meta = {
        "image_token": IMAGE_TOKEN,
        "image_token_id": model.image_token_id,
        "llm_name": model.llm_name,
        "llm_hidden_size": model.llm_hidden_size,
        "vision_hidden_size": model.vision_hidden_size,
        "vision_backend": model.vision_backend,
        "vision_name": model.vision_name,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
