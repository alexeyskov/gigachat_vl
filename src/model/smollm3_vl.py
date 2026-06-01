from typing import Any, Dict, List, Optional

from src.model.gigachat_vl import (
    GigaChatVL,
    GigaChatVLForInference,
    _build_multimodal_user_text,
)


DEFAULT_SMOLLM3_NAME = "HuggingFaceTB/SmolLM3-3B"


class _SmolLM3ChatMixin:
    """SmolLM3-specific chat prompt helpers.

    SmolLM3 supports an extended thinking mode. For VLM alignment experiments
    the default here is ``/no_think`` so the model learns direct visual answers
    instead of reasoning traces.
    """

    smollm_enable_thinking: bool
    smollm_system_prompt: Optional[str]

    def _smollm_messages(
        self,
        user_text: str,
        assistant_text: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.smollm_system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": str(self.smollm_system_prompt),
                }
            )

        messages.append({"role": "user", "content": user_text})
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": str(assistant_text)})
        return messages

    def _apply_smollm_chat_template(
        self,
        messages: List[Dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=bool(self.smollm_enable_thinking),
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def build_chat_prompt(self, text: str, num_images: int = 1) -> str:
        """Build a SmolLM3 inference prompt.

        Args:
            text: User text prompt.
            num_images: Number of image placeholders to prepend.

        Returns:
            Formatted SmolLM3 chat prompt ending at the assistant generation
            prefix.
        """
        user_text = _build_multimodal_user_text(text, num_images=num_images)

        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self._apply_smollm_chat_template(
                    self._smollm_messages(user_text),
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        system = f"System: {self.smollm_system_prompt}\n" if self.smollm_system_prompt else ""
        return f"{system}User: {user_text}\nAssistant:"

    def build_chat_training_texts(
        self,
        question: str,
        answer: str,
        num_images: int = 1,
    ) -> tuple[str, str]:
        """Build SmolLM3 prompt/full text pair for supervised training.

        Args:
            question: User-side training prompt.
            answer: Assistant answer used as supervised target.
            num_images: Number of image placeholders to prepend.

        Returns:
            A ``(prompt, full_text)`` tuple where ``full_text`` starts with
            ``prompt``.
        """
        user_text = _build_multimodal_user_text(question, num_images=num_images)

        if getattr(self.tokenizer, "chat_template", None):
            try:
                prompt = self._apply_smollm_chat_template(
                    self._smollm_messages(user_text),
                    add_generation_prompt=True,
                )
                full_text = self._apply_smollm_chat_template(
                    self._smollm_messages(user_text, assistant_text=str(answer)),
                    add_generation_prompt=False,
                )
                return prompt, full_text
            except Exception:
                pass

        prompt = self.build_chat_prompt(question, num_images=num_images)
        eos_token = self.tokenizer.eos_token or ""
        return prompt, prompt + str(answer) + eos_token


class SmolLM3VL(_SmolLM3ChatMixin, GigaChatVL):
    """VLM wrapper that uses HuggingFaceTB/SmolLM3-3B as the base LLM.

    This class intentionally reuses the GigaChatVL multimodal pipeline:
    supported vision encoders, projectors, connector LLMs, LoRA/QLoRA, visual
    experts, precomputed embeddings, and the existing data collator all work the
    same way. Only the base LLM default and chat prompt formatting are changed.

    Args:
        llm_name: Hugging Face id or local path for SmolLM3. Defaults to
            ``"HuggingFaceTB/SmolLM3-3B"``.
        smollm_enable_thinking: Whether to enable SmolLM3 extended thinking in
            generated chat templates.
        smollm_system_prompt: Optional system prompt. Defaults to
            ``"/no_think"`` for direct VLM answers.
        **kwargs: Forwarded to :class:`GigaChatVL`.
    """

    def __init__(
        self,
        llm_name: str = DEFAULT_SMOLLM3_NAME,
        smollm_enable_thinking: bool = False,
        smollm_system_prompt: Optional[str] = "/no_think",
        **kwargs: Any,
    ):
        self.smollm_enable_thinking = bool(smollm_enable_thinking)
        self.smollm_system_prompt = smollm_system_prompt
        super().__init__(llm_name=llm_name, **kwargs)


class SmolLM3VLForInference(_SmolLM3ChatMixin, GigaChatVLForInference):
    """Inference loader for SmolLM3VL checkpoints.

    Args:
        checkpoint_dir: Directory produced by ``save_artifacts`` or
            ``SaveVLMArtifactsCallback``.
        llm_name: Optional Hugging Face id or local path for SmolLM3. When
            omitted, the value is restored from checkpoint metadata.
        smollm_enable_thinking: Whether to enable SmolLM3 extended thinking in
            generated chat templates.
        smollm_system_prompt: Optional system prompt. Defaults to
            ``"/no_think"`` for direct VLM answers.
        **kwargs: Forwarded to :class:`GigaChatVLForInference`.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        llm_name: Optional[str] = None,
        smollm_enable_thinking: bool = False,
        smollm_system_prompt: Optional[str] = "/no_think",
        **kwargs: Any,
    ):
        self.smollm_enable_thinking = bool(smollm_enable_thinking)
        self.smollm_system_prompt = smollm_system_prompt
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            llm_name=llm_name,
            **kwargs,
        )
