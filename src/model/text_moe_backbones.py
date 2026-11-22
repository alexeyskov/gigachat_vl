"""Loading and chat-format rules shared by training and checkpoint inference."""

from transformers import AutoConfig, PretrainedConfig


TEXT_MOE_NAMES = {
    "gpt_oss": "openai/gpt-oss-20b",
    "smallthinker": "Tiiny/SmallThinker-4BA0.6B-Instruct",
    "lfm2_moe": "LiquidAI/LFM2.5-8B-A1B",
}


def text_moe_load_kwargs(
    llm_name, *, use_4bit_llm, use_llm_lora, text_visual_gates,
):
    raw_config, _ = PretrainedConfig.get_config_dict(llm_name)
    model_type = raw_config.get("model_type")
    if model_type not in TEXT_MOE_NAMES:
        return {}
    if text_visual_gates:
        raise ValueError("Learned text_visual_gates currently support only native DeepSeek-V2.")
    if model_type in {"gpt_oss", "lfm2_moe"} and (use_4bit_llm or use_llm_lora):
        raise ValueError(
            f"{model_type} experiments use native expert weights: set use_4bit_llm=False "
            "and use_llm_lora=False. The generic linear LoRA path does not adapt packed experts."
        )
    if model_type == "smallthinker":
        from src.model.smallthinker import register_smallthinker

        register_smallthinker()
    config = AutoConfig.from_pretrained(llm_name, trust_remote_code=False)
    result = {"config": config, "trust_remote_code": False}
    quantization = raw_config.get("quantization_config") or {}
    if model_type == "gpt_oss" and quantization.get("quant_method") == "mxfp4":
        from transformers import Mxfp4Config

        # The official checkpoint is MXFP4. Dequantization provides a regular
        # differentiable PyTorch path, including gradients to frozen-base inputs.
        result["quantization_config"] = Mxfp4Config(dequantize=True)
    return result


def apply_text_moe_chat_template(tokenizer, messages, *, model_type, add_generation_prompt):
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{model_type} requires its checkpoint's chat template.")
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
    )
    if model_type == "gpt_oss" and add_generation_prompt:
        # Official Harmony's generation prompt ends at the assistant role. VQA
        # labels contain direct answers, so use the final channel consistently.
        text += "<|channel|>final<|message|>"
    return text
