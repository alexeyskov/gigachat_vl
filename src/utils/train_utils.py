import os

import torch
from transformers import TrainerCallback
from peft import PeftModel

from src.model.gigachat_vl import GigaChatVL


def _unwrap_model(model):
    return getattr(model, "module", model)


def save_artifacts(model: GigaChatVL, output_dir: str):
    model = _unwrap_model(model)
    os.makedirs(output_dir, exist_ok=True)

    lora_dir = os.path.join(output_dir, "llm_lora")
    full_llm_dir = os.path.join(output_dir, "llm")
    projector_path = os.path.join(output_dir, "projector.pt")
    vl_experts_path = os.path.join(output_dir, "vl_experts.pt")
    model.save_training_setup(output_dir)

    if getattr(model, "use_llm_lora", True) or isinstance(model.llm, PeftModel):
        model.llm.save_pretrained(lora_dir)
    else:
        model.llm.save_pretrained(
            full_llm_dir,
            state_dict=model.llm_base_state_dict(),
        )
    torch.save(model.projector_state_dict(), projector_path)
    model.save_vision_encoder(output_dir)
    model.save_connector_llm(output_dir)
    if getattr(model, "enable_vl_experts", False):
        model.save_vl_experts(vl_experts_path)


def save_training_setup(model: GigaChatVL, output_dir: str):
    model = _unwrap_model(model)
    model.save_training_setup(output_dir)


class SaveVLMArtifactsCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if not args.should_save:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        checkpoint_dir = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}",
        )
        save_artifacts(model, checkpoint_dir)
        return control
