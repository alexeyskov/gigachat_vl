import os

import torch

from src.model.gigachat_vl import GigaChatVL


def save_artifacts(model: GigaChatVL, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    lora_dir = os.path.join(output_dir, "llm_lora")
    projector_path = os.path.join(output_dir, "projector.pt")
    vl_experts_path = os.path.join(output_dir, "vl_experts.pt")
    model.save_training_setup(output_dir)

    model.llm.save_pretrained(lora_dir)
    torch.save(model.projector.state_dict(), projector_path)
    if getattr(model, "enable_vl_experts", False):
        model.save_vl_experts(vl_experts_path)


def save_training_setup(model: GigaChatVL, output_dir: str):
    model.save_training_setup(output_dir)
