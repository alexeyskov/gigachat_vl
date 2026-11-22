"""Native DeepSeek-V2 multimodal models and training support."""

import math
from typing import Any, Optional

import torch
from transformers import Trainer

from src.model.gigachat_vl import GigaChatVL, GigaChatVLForInference


DEFAULT_DEEPSEEK_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"


class DeepSeekVL(GigaChatVL):
    """DeepSeek-V2-Lite with learned visual access for its original experts.

    The initial training stage freezes the language and vision backbones while
    training the projector, added visual experts, and visual-access gates.
    Set ``train_llm_lora=True`` with ``use_llm_lora=False`` to subsequently
    finetune the full language backbone. That option retains the base wrapper's
    historical name but controls full training when LoRA is disabled.
    """

    def __init__(
        self,
        llm_name: str = DEFAULT_DEEPSEEK_NAME,
        use_4bit_llm: bool = False,
        use_llm_lora: bool = False,
        text_visual_gates: bool = True,
        **kwargs: Any,
    ):
        if text_visual_gates and (use_4bit_llm or use_llm_lora):
            raise ValueError(
                "Gated DeepSeek uses native unquantized expert weights. "
                "Set use_4bit_llm=False and use_llm_lora=False."
            )
        kwargs.setdefault("llm_trust_remote_code", False)
        kwargs.setdefault("chat_template_mode", "tokenizer")
        kwargs.setdefault("enable_vl_experts", True)
        kwargs.setdefault("num_lora_layers", -1)
        kwargs.setdefault("train_llm_lora", False)
        kwargs.setdefault("text_visual_gate_init", 0.9)
        kwargs.setdefault("text_visual_gate_reg_weight", 1e-3 if text_visual_gates else 0.0)
        super().__init__(
            llm_name=llm_name,
            use_4bit_llm=use_4bit_llm,
            use_llm_lora=use_llm_lora,
            text_visual_gates=text_visual_gates,
            **kwargs,
        )

    @classmethod
    def from_training_checkpoint(
        cls,
        checkpoint_dir: str,
        use_4bit_llm: bool = False,
        **kwargs: Any,
    ) -> "DeepSeekVL":
        kwargs.setdefault("llm_trust_remote_code", False)
        return super().from_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            use_4bit_llm=use_4bit_llm,
            **kwargs,
        )


class DeepSeekVLForInference(GigaChatVLForInference):
    """Load DeepSeek VL artifacts, including gate settings from metadata."""

    def __init__(
        self,
        checkpoint_dir: str,
        use_4bit_llm: bool = False,
        **kwargs: Any,
    ):
        kwargs.setdefault("llm_trust_remote_code", False)
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            use_4bit_llm=use_4bit_llm,
            **kwargs,
        )


class DeepSeekVLTrainer(Trainer):
    """Train visual-access gates with a separate LR and log their values.

    Uses standard parameter-group optimizers selected through TrainingArguments.
    Optimizers that construct their own parameters or replace the whole model
    are rejected because they would silently discard the gate learning rate.
    """

    def __init__(self, *args: Any, gate_learning_rate: float = 1e-3, **kwargs: Any):
        if not math.isfinite(gate_learning_rate) or gate_learning_rate <= 0:
            raise ValueError("gate_learning_rate must be a finite positive number.")
        training_args = kwargs.get("args", args[1] if len(args) > 1 else None)
        if getattr(training_args, "label_smoothing_factor", 0.0) != 0.0:
            raise ValueError(
                "DeepSeekVLTrainer requires label_smoothing_factor=0 so labels "
                "reach the model's causal loss and visual gate regularizer."
            )
        if kwargs.get("compute_loss_func") is not None:
            raise ValueError(
                "Custom compute_loss_func is unsupported by DeepSeekVLTrainer: "
                "the model must receive labels to compute the visual gate regularizer."
            )
        self.gate_learning_rate = float(gate_learning_rate)
        super().__init__(*args, **kwargs)
        if self.compute_loss_func is not None:
            raise ValueError("Custom compute_loss_func is unsupported by DeepSeekVLTrainer.")

    def create_optimizer(self, model: Optional[torch.nn.Module] = None):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model if model is None else model
        decay_names = set(self.get_decay_parameter_names(opt_model))
        decay_params, no_decay_params, gate_params = [], [], []
        for name, parameter in opt_model.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".text_visual_gate." in name:
                gate_params.append(parameter)
            elif name in decay_names:
                decay_params.append(parameter)
            else:
                no_decay_params.append(parameter)

        groups = [
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "group_name": group_name,
            }
            for parameters, learning_rate, weight_decay, group_name in (
                (decay_params, self.args.learning_rate, self.args.weight_decay, "base_decay"),
                (no_decay_params, self.args.learning_rate, 0.0, "base_no_decay"),
                (gate_params, self.gate_learning_rate, 0.0, "visual_gates"),
            )
            if parameters
        ]
        if not groups:
            raise ValueError("No trainable parameters were found.")

        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        optimizer_kwargs = dict(optimizer_kwargs)
        if {"params", "model", "optimizer_dict"}.intersection(optimizer_kwargs):
            raise ValueError(
                "This optimizer overrides parameter groups. Use adamw_torch "
                "to preserve the separate visual gate learning rate."
            )
        if not isinstance(optimizer_cls, type) or not issubclass(optimizer_cls, torch.optim.Optimizer):
            raise ValueError("DeepSeekVLTrainer requires a standard parameter-group optimizer.")
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        logs = dict(logs)
        model = getattr(self.model, "module", self.model)
        if getattr(model, "text_visual_gates", False):
            with torch.no_grad():
                for layer_idx, wrapper in model._iter_vl_expert_wrappers():
                    gate = getattr(wrapper, "text_visual_gate", None)
                    if gate is None:
                        continue
                    for kind, values in gate.gate_values().items():
                        if values.numel() == 0:
                            continue
                        values = values.detach().float()
                        prefix = f"visual_gates/layer_{layer_idx}/{kind}"
                        logs[f"{prefix}_mean"] = values.mean().item()
                        logs[f"{prefix}_open_fraction"] = (values > 0).float().mean().item()
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group.get("group_name") == "visual_gates":
                    logs["gate_learning_rate"] = float(group["lr"])
                    break
        super().log(logs, start_time=start_time)
