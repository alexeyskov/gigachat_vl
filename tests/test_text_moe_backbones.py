"""Offline CPU smoke tests; no pretrained weights or image downloads."""

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GptOssConfig, GptOssForCausalLM, Lfm2MoeConfig, Lfm2MoeForCausalLM,
    PreTrainedTokenizerFast,
)

from src.model.gigachat_vl import GigaChatVL, GigaChatVLForInference
from src.model.smallthinker import SmallThinkerConfig, SmallThinkerForCausalLM
from src.model.text_moe_vl import GptOssVL, LFM2MoEVL, SmallThinkerVL
from src.utils.train_utils import save_artifacts


def tiny_backbone(kind):
    common = dict(
        vocab_size=64, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    if kind == "gpt_oss":
        return GptOssForCausalLM(GptOssConfig(
            **common, head_dim=8, intermediate_size=12, num_local_experts=4,
            num_experts_per_tok=2, experts_per_token=2, max_position_embeddings=64,
            initial_context_length=64, rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            layer_types=["full_attention", "sliding_attention"], sliding_window=4,
        ))
    if kind == "smallthinker":
        return SmallThinkerForCausalLM(SmallThinkerConfig(
            **common, head_dim=8, moe_ffn_hidden_size=12,
            moe_num_primary_experts=4, moe_num_active_primary_experts=2,
        ))
    return Lfm2MoeForCausalLM(Lfm2MoeConfig(
        **common, intermediate_size=24, moe_intermediate_size=12, num_experts=4,
        num_experts_per_tok=2, num_dense_layers=1, layer_types=["conv", "full_attention"],
    ))


@pytest.mark.parametrize("kind,model_class", [
    ("gpt_oss", GptOssVL), ("smallthinker", SmallThinkerVL), ("lfm2_moe", LFM2MoEVL),
])
def test_alignment_and_checkpoint_roundtrip(tmp_path, monkeypatch, kind, model_class):
    torch.manual_seed(31)
    base_dir = tmp_path / "base"
    tiny_backbone(kind).save_pretrained(base_dir)
    vocab = {f"word_{i}": i for i in range(64)}
    for i, token in {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>", 63: "[image_token]"}.items():
        del vocab[f"word_{i}"]
        vocab[token] = i
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    # A local chat template keeps this test independent of network downloads.
    template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    if kind == "gpt_oss":
        template = (
            "{% for m in messages %}<|start|>{{ m['role'] }}"
            "{% if m['role'] == 'assistant' %}<|channel|>final{% endif %}"
            "<|message|>{{ m['content'] }}"
            "{% if m['role'] == 'assistant' %}<|return|>{% else %}<|end|>{% endif %}"
            "{% endfor %}{% if add_generation_prompt %}<|start|>assistant{% endif %}"
        )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>",
        eos_token="<eos>", unk_token="<unk>",
        additional_special_tokens=["[image_token]"], chat_template=template,
    )
    tokenizer.save_pretrained(base_dir)

    def offline_vision(self, _name):
        self.vision_hidden_size = 6
        self.vision_backend = "offline_test"

    monkeypatch.setattr(GigaChatVL, "_load_vision_backend", offline_vision)
    model = model_class(llm_name=str(base_dir), vision_name="offline", device="cpu")
    model.gradient_checkpointing_enable()
    model.train()
    batch = dict(
        input_ids=torch.tensor([[1, 63, 7, 8], [1, 9, 10, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        labels=torch.tensor([[-100, -100, 7, 8], [-100, 9, 10, -100]]),
        vision_precomputed_features=[torch.randn(3, 6)],
        vision_num_images_per_sample=torch.tensor([1, 0]),
    )
    result = model(**batch)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.projector.net[0].weight.grad.abs().sum() > 0
    for _, wrapper in model._iter_vl_expert_wrappers():
        assert wrapper.visual_expert.down_proj.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in wrapper.base_mlp.parameters())
    prompt, full = model.build_chat_training_texts("question", "answer")
    assert full.startswith(prompt)
    if kind == "gpt_oss":
        assert prompt.endswith("<|channel|>final<|message|>")
    model.gradient_checkpointing_disable()
    model.eval()
    with torch.no_grad():
        for _, wrapper in model._iter_vl_expert_wrappers():
            wrapper.visual_expert.down_proj.weight.normal_(std=0.1)
        expected = model(**batch).logits
    artifact_dir = tmp_path / "artifacts"
    save_artifacts(model, str(artifact_dir))
    meta = json.loads((artifact_dir / "vlm_meta.json").read_text())
    assert meta["llm_model_type"] == kind
    for loaded in (
        model_class.from_training_checkpoint(str(artifact_dir), device="cpu"),
        GigaChatVLForInference(str(artifact_dir), use_4bit_llm=False, device="cpu"),
    ):
        loaded.eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(**batch).logits, expected)
        assert loaded.build_chat_prompt("question") == prompt
        assert "base_mlp" not in " ".join(loaded.llm_base_state_dict())
