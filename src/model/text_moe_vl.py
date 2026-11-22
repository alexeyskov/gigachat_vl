"""Text-only MoE backbones connected to the existing multimodal pipeline."""

from src.model.gigachat_vl import GigaChatVL, GigaChatVLForInference
from src.model.text_moe_backbones import TEXT_MOE_NAMES


class TextMoEVL(GigaChatVL):
    """Frozen-backbone alignment by default; enable full training explicitly.

    Set train_llm_lora=True with use_llm_lora=False for full backbone training.
    Visual experts are optional. DeepSeek's per-expert input gates are unsupported.
    """

    backbone_type = None

    def __init__(self, llm_name=None, use_4bit_llm=False, use_llm_lora=False, **kwargs):
        if kwargs.get("text_visual_gates", False):
            raise ValueError("text_visual_gates are supported only by DeepSeek-V2.")
        if self.backbone_type in {"gpt_oss", "lfm2_moe"} and (use_4bit_llm or use_llm_lora):
            raise ValueError("Set use_4bit_llm=False and use_llm_lora=False for packed MoE experts.")
        kwargs.setdefault("llm_trust_remote_code", False)
        kwargs.setdefault("chat_template_mode", "tokenizer")
        kwargs.setdefault("projector_type", "mlp")
        kwargs.setdefault("enable_vl_experts", True)
        kwargs.setdefault("train_llm_lora", False)
        super().__init__(
            llm_name=llm_name or TEXT_MOE_NAMES[self.backbone_type],
            use_4bit_llm=use_4bit_llm, use_llm_lora=use_llm_lora, **kwargs,
        )
        if self.llm.config.model_type != self.backbone_type:
            raise ValueError(f"Expected {self.backbone_type}, got {self.llm.config.model_type}.")

    @classmethod
    def from_training_checkpoint(cls, checkpoint_dir, use_4bit_llm=False, **kwargs):
        return super().from_training_checkpoint(
            checkpoint_dir, use_4bit_llm=use_4bit_llm, **kwargs,
        )


class GptOssVL(TextMoEVL):
    backbone_type = "gpt_oss"


class SmallThinkerVL(TextMoEVL):
    backbone_type = "smallthinker"


class LFM2MoEVL(TextMoEVL):
    backbone_type = "lfm2_moe"


class TextMoEVLForInference(GigaChatVLForInference):
    def __init__(self, checkpoint_dir, use_4bit_llm=False, **kwargs):
        super().__init__(checkpoint_dir, use_4bit_llm=use_4bit_llm, **kwargs)


class GptOssVLForInference(TextMoEVLForInference):
    pass


class SmallThinkerVLForInference(TextMoEVLForInference):
    pass


class LFM2MoEVLForInference(TextMoEVLForInference):
    pass
