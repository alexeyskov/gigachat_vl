"""CPU integration checks using random, tiny native DeepSeek-V2 models.

No model weights, tokenizer files, or vision towers are downloaded. These tests
exercise the real Transformers decoder and the repository's multimodal wrapper.
"""

import copy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers.models.deepseek_v2.modeling_deepseek_v2")

from torch import nn
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

from src.model.gigachat_vl import GigaChatVL


def tiny_config():
    config = DeepseekV2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        first_k_dense_replace=1,
        n_routed_experts=4,
        n_shared_experts=2,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        kv_lora_rank=8,
        q_lora_rank=None,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        max_position_embeddings=64,
        initializer_range=0.1,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    return config


def tiny_vlm(*, gated=True):
    # Bypass only external checkpoint/tokenizer/vision loading. The expert
    # installation, merge, forward, trainability, and saving paths are real.
    model = GigaChatVL.__new__(GigaChatVL)
    nn.Module.__init__(model)
    model.llm = DeepseekV2ForCausalLM(tiny_config())
    model.model_device = torch.device("cpu")
    model.llm_hidden_size = model.llm.config.hidden_size
    model.num_lora_layers = -1
    model.vl_expert_layers = None
    model.enable_vl_experts = True
    model.vl_expert_layer_indices = []
    model.text_visual_gates = gated
    model.text_visual_gate_init = 0.9
    model.text_visual_gate_reg_weight = 0.01
    model.llm_trust_remote_code = False
    model.use_llm_lora = False
    model.train_llm_lora = True
    model.projector = nn.Linear(6, model.llm_hidden_size)
    model.normalize_visual_embeddings = False
    model.image_token_id = 63
    model.freeze_vision = True
    model.vision_tower = None
    model.vision_projector = None
    model._install_vl_experts()
    return model


def multimodal_batch():
    # A multimodal and a text-only sample in one padded batch. Expanding the
    # first sample's image produces three visual positions and three padding
    # slots in the other sample, exercising masks after sequence expansion.
    return {
        "input_ids": torch.tensor([[1, 63, 7, 8], [1, 9, 10, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, -100, 7, 8], [-100, 9, 10, -100]]),
        "vision_precomputed_features": [torch.randn(3, 6)],
        "vision_num_images_per_sample": torch.tensor([1, 0]),
    }


def gate_parameters(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".text_visual_gate." in name
    }


def test_frozen_text_base_still_trains_masks_projector_and_visual_experts():
    torch.manual_seed(20)
    model = tiny_vlm()
    model.set_llm_lora_trainable(False)
    model._set_vl_experts_trainable(True)
    model.train()

    result = model(**multimodal_batch())
    assert torch.isfinite(result.loss)
    result.loss.backward()

    gates = gate_parameters(model)
    assert gates
    for parameter in gates.values():
        assert parameter.requires_grad
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert model.projector.weight.grad.abs().sum() > 0
    for _, wrapper in model._iter_vl_expert_wrappers():
        assert wrapper.visual_expert.down_proj.weight.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in wrapper.base_mlp.parameters())


def test_text_only_forward_clears_previous_visual_mask_and_matches_base():
    torch.manual_seed(21)
    model = tiny_vlm().eval()
    reference = DeepseekV2ForCausalLM(tiny_config()).eval()
    reference.load_state_dict(model.llm_base_state_dict(), strict=True)
    model._set_vl_expert_visual_token_mask(torch.ones(1, 4, dtype=torch.bool))
    batch = {
        "input_ids": torch.tensor([[1, 5, 6, 7]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, 5, 6, 7]]),
    }
    actual = model(**batch)
    expected = reference(**batch, use_cache=False)
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.loss, expected.loss)
    actual.loss.backward()
    assert all(parameter.grad is None for parameter in gate_parameters(model).values())
    assert all(wrapper.visual_token_mask is None for _, wrapper in model._iter_vl_expert_wrappers())


def test_frozen_text_only_training_batch_allows_noop_backward():
    torch.manual_seed(29)
    model = tiny_vlm()
    model.set_llm_lora_trainable(False)
    model._set_vl_experts_trainable(True)
    model.gradient_checkpointing_enable()
    model.train()
    batch = {
        "input_ids": torch.tensor([[1, 5, 6, 7]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, 5, 6, 7]]),
    }
    expected = model.llm(**batch, use_cache=False).loss.detach()
    actual = model(**batch)
    torch.testing.assert_close(actual.loss.detach(), expected)
    assert actual.loss.requires_grad
    actual.loss.backward()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_open_gates_reproduce_native_decoder_and_plain_export_is_loadable():
    torch.manual_seed(22)
    model = tiny_vlm().eval()
    reference = DeepseekV2ForCausalLM(tiny_config()).eval()
    plain_state = model.llm_base_state_dict()
    assert not any("text_visual_gate" in key or "visual_expert" in key for key in plain_state)
    assert not any("base_mlp" in key for key in plain_state)
    reference.load_state_dict(plain_state, strict=True)
    with torch.no_grad():
        for parameter in gate_parameters(model).values():
            parameter.fill_(100.0)
    inputs = torch.randn(1, 5, model.llm_hidden_size)
    model._set_vl_expert_visual_token_mask(torch.tensor([[False, True, True, False, False]]))
    actual = model.llm(inputs_embeds=inputs, use_cache=False).logits
    expected = reference(inputs_embeds=inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_gate_and_visual_weights_round_trip_in_expert_sidecar(tmp_path):
    torch.manual_seed(23)
    model = tiny_vlm()
    with torch.no_grad():
        for parameter in gate_parameters(model).values():
            parameter.uniform_(-2, 2)
        for _, wrapper in model._iter_vl_expert_wrappers():
            wrapper.visual_expert.down_proj.weight.normal_()
    checkpoint = tmp_path / "vl_experts.pt"
    model.save_vl_experts(str(checkpoint))
    restored = tiny_vlm()
    restored.load_vl_experts(str(checkpoint))
    expected = model.vl_experts_state_dict()
    actual = restored.vl_experts_state_dict()
    assert any("text_visual_gate" in key for key in expected)
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_legacy_visual_sidecar_cannot_silently_initialize_enabled_gates(tmp_path):
    legacy = tiny_vlm(gated=False)
    checkpoint = tmp_path / "vl_experts.pt"
    legacy.save_vl_experts(str(checkpoint))
    legacy_restored = tiny_vlm(gated=False)
    legacy_restored.load_vl_experts(str(checkpoint))
    with pytest.raises((KeyError, RuntimeError, ValueError), match="(?i)gate|mask"):
        tiny_vlm(gated=True).load_vl_experts(str(checkpoint))


def test_checkpointing_preserves_stochastic_gate_gradients():
    torch.manual_seed(24)
    eager = tiny_vlm()
    checkpointed = copy.deepcopy(eager)
    eager.train()
    checkpointed.train()
    checkpointed.gradient_checkpointing_enable()
    batch = multimodal_batch()

    gradients = []
    for model in (eager, checkpointed):
        torch.manual_seed(25)
        output = model(**batch)
        output.loss.backward()
        gradients.append({
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        })
    assert gradients[0].keys() == gradients[1].keys()
    assert any("text_visual_gate" in key for key in gradients[0])
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-5, atol=1e-7)


def test_cached_decode_matches_full_recompute_with_visual_prompt():
    torch.manual_seed(26)
    model = tiny_vlm().eval()
    with torch.no_grad():
        for _, wrapper in model._iter_vl_expert_wrappers():
            wrapper.visual_expert.down_proj.weight.normal_(std=0.2)
    prompt = torch.randn(1, 4, model.llm_hidden_size)
    prompt_mask = torch.tensor([[False, True, True, False]])
    model._set_vl_expert_visual_token_mask(prompt_mask)
    token = torch.tensor([[6]])
    with torch.no_grad():
        prefill = model.llm(inputs_embeds=prompt, use_cache=True)
        cached = model.llm(
            input_ids=token,
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        all_embeds = torch.cat([prompt, model.llm.get_input_embeddings()(token)], dim=1)
        recomputed = model.llm(inputs_embeds=all_embeds, use_cache=False)
    torch.testing.assert_close(cached.logits[:, -1], recomputed.logits[:, -1], rtol=1e-5, atol=1e-6)


def test_single_visual_token_prompt_is_not_reapplied_during_cached_generation():
    torch.manual_seed(28)
    model = tiny_vlm().eval()
    with torch.no_grad():
        for _, wrapper in model._iter_vl_expert_wrappers():
            wrapper.visual_expert.down_proj.weight.normal_(std=0.2)
    prompt = torch.randn(1, 1, model.llm_hidden_size)
    prompt_mask = torch.ones(1, 1, dtype=torch.bool)
    seen_visual = []
    first_wrapper = next(model._iter_vl_expert_wrappers())[1]

    def observe_mask(module, _inputs):
        seen_visual.append(module.visual_token_mask is not None and bool(module.visual_token_mask.any()))

    handle = first_wrapper.register_forward_pre_hook(observe_mask)
    with torch.no_grad(), model._generation_visual_token_mask(prompt_mask):
        generated = model.llm.generate(
            input_ids=torch.tensor([[model.image_token_id]]),
            inputs_embeds=prompt,
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=0,
            return_dict_in_generate=True,
            output_scores=True,
        )
    handle.remove()
    assert seen_visual == [True, False, False]
    assert first_wrapper.visual_token_mask is None

    model._set_vl_expert_visual_token_mask(prompt_mask)
    with torch.no_grad():
        for step, cached_score in enumerate(generated.scores):
            text_ids = generated.sequences[:, 1:step + 1]
            full_embeddings = torch.cat([prompt, model.llm.get_input_embeddings()(text_ids)], dim=1)
            recomputed = model.llm(inputs_embeds=full_embeddings, use_cache=False).logits[:, -1]
            torch.testing.assert_close(cached_score, recomputed, rtol=1e-5, atol=1e-6)


def test_beam_expansion_repeats_visual_origin_masks_per_sample():
    model = tiny_vlm().eval()
    source_mask = torch.tensor([[False, True, False], [True, False, False]])
    model._set_vl_expert_visual_token_mask(source_mask)
    for _, wrapper in model._iter_vl_expert_wrappers():
        expanded = wrapper._resolve_mask(torch.randn(6, 3, model.llm_hidden_size))
        torch.testing.assert_close(expanded, source_mask.repeat_interleave(3, dim=0))


def test_trainer_assigns_every_trainable_parameter_once_with_gate_lr(tmp_path):
    from transformers import TrainingArguments

    from src.model.deepseek_vl import DeepSeekVLTrainer

    model = tiny_vlm()
    model.set_llm_lora_trainable(False)
    model._set_vl_experts_trainable(True)
    args = TrainingArguments(
        output_dir=str(tmp_path),
        learning_rate=2e-5,
        weight_decay=0.1,
        optim="adamw_torch",
        report_to="none",
        use_cpu=True,
    )
    trainer = DeepSeekVLTrainer(model=model, args=args, gate_learning_rate=3e-3)
    optimizer = trainer.create_optimizer()
    assigned_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    gate_ids = {id(parameter) for parameter in gate_parameters(model).values()}
    assert len(assigned_ids) == len(set(assigned_ids))
    assert set(assigned_ids) == expected_ids
    for group in optimizer.param_groups:
        group_ids = {id(parameter) for parameter in group["params"]}
        if group_ids & gate_ids:
            assert group_ids == gate_ids
            assert group["lr"] == 3e-3
            assert group["weight_decay"] == 0.0
        else:
            assert group["lr"] == args.learning_rate
    assert trainer.create_optimizer() is optimizer


def test_trainer_accumulates_text_only_and_multimodal_batches_with_frozen_base(tmp_path):
    from transformers import TrainingArguments

    from src.model.deepseek_vl import DeepSeekVLTrainer

    torch.manual_seed(30)
    model = tiny_vlm()
    model.set_llm_lora_trainable(False)
    model._set_vl_experts_trainable(True)
    model.gradient_checkpointing_enable()
    original_base = {key: value.clone() for key, value in model.llm_base_state_dict().items()}
    original_projector = model.projector.weight.detach().clone()
    original_gates = {name: parameter.detach().clone() for name, parameter in gate_parameters(model).items()}
    batches = [
        {
            "input_ids": torch.tensor([[1, 5, 6, 7]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.tensor([[-100, 5, 6, 7]]),
        },
        multimodal_batch(),
    ]
    args = TrainingArguments(
        output_dir=str(tmp_path),
        learning_rate=1e-3,
        optim="adamw_torch",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=1,
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=True,
        use_cpu=True,
    )
    trainer = DeepSeekVLTrainer(
        model=model,
        args=args,
        train_dataset=batches,
        data_collator=lambda examples: examples[0],
        gate_learning_rate=3e-3,
    )
    result = trainer.train()
    assert result.global_step == 1
    for key, value in model.llm_base_state_dict().items():
        torch.testing.assert_close(value, original_base[key], rtol=0, atol=0)
    assert not torch.equal(model.projector.weight, original_projector)
    assert any(not torch.equal(parameter, original_gates[name]) for name, parameter in gate_parameters(model).items())


def test_deepseek_default_arguments_select_native_unquantized_backend(monkeypatch):
    from src.model.deepseek_vl import DEFAULT_DEEPSEEK_NAME, DeepSeekVL

    captured = {}

    def capture_init(self, **kwargs):
        nn.Module.__init__(self)
        captured.update(kwargs)

    monkeypatch.setattr(GigaChatVL, "__init__", capture_init)
    DeepSeekVL()
    assert captured["llm_name"] == DEFAULT_DEEPSEEK_NAME
    assert captured["llm_trust_remote_code"] is False
    assert captured["use_4bit_llm"] is False
    assert captured["use_llm_lora"] is False
    assert captured["text_visual_gates"] is True
    assert captured["enable_vl_experts"] is True
    assert captured["chat_template_mode"] == "tokenizer"


@pytest.mark.parametrize("option", ["use_4bit_llm", "use_llm_lora"])
def test_deepseek_rejects_unsupported_quantized_or_lora_gate_path(option):
    from src.model.deepseek_vl import DeepSeekVL

    with pytest.raises(ValueError, match="(?i)unquantized|native"):
        DeepSeekVL(**{option: True})


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("gated", [False, True])
def test_checkpoint_loaders_restore_gate_configuration(tmp_path, monkeypatch, inference, gated):
    from src.model.deepseek_vl import DeepSeekVL, DeepSeekVLForInference

    (tmp_path / "llm").mkdir()
    torch.save({}, tmp_path / "projector.pt")
    torch.save({}, tmp_path / "vl_experts.pt")
    metadata = {
        "llm_name": "local-test-deepseek",
        "vision_name": "local-test-vision",
        "use_llm_lora": False,
        "projector_type": "mlp",
        "enable_vl_experts": True,
        "vl_expert_layers": [1],
        "text_visual_gates": gated,
        "text_visual_gate_init": 0.73,
        "text_visual_gate_reg_weight": 0.004 if gated else 0.0,
        "llm_trust_remote_code": False,
    }
    (tmp_path / "vlm_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    captured = {}

    def capture_init(self, **kwargs):
        nn.Module.__init__(self)
        self.freeze_vision = False
        captured.update(kwargs)

    # Exercise the actual loader's sidecar discovery and metadata resolution,
    # replacing only heavyweight checkpoint instantiation.
    monkeypatch.setattr(GigaChatVL, "__init__", capture_init)
    if inference:
        DeepSeekVLForInference(str(tmp_path))
    else:
        DeepSeekVL.from_training_checkpoint(str(tmp_path))
    assert captured["text_visual_gates"] is gated
    assert captured["text_visual_gate_init"] == metadata["text_visual_gate_init"]
    assert captured["text_visual_gate_reg_weight"] == metadata["text_visual_gate_reg_weight"]
    assert captured["llm_trust_remote_code"] is False
    assert captured["use_4bit_llm"] is False
    assert captured["vl_expert_layers"] == [1]


def test_real_constructor_and_artifact_roundtrip_without_external_weights(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from src.model.deepseek_vl import DeepSeekVL, DeepSeekVLForInference
    from src.utils.train_utils import save_artifacts

    torch.manual_seed(27)
    base_dir = tmp_path / "base"
    DeepseekV2ForCausalLM(tiny_config()).save_pretrained(base_dir)
    vocab = {f"word_{index}": index for index in range(64)}
    for index, name in {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>", 63: "[image_token]"}.items():
        del vocab[f"word_{index}"]
        vocab[name] = index
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=["[image_token]"],
        chat_template=(
            "{{ bos_token }}{% for message in messages %}"
            "{{ message['role'] }} {{ message['content'] }} "
            "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        ),
    )
    tokenizer.save_pretrained(base_dir)

    def load_dummy_vision(self, vision_name):
        self.vision_hidden_size = 6
        self.vision_backend = "offline_test"

    monkeypatch.setattr(GigaChatVL, "_load_vision_backend", load_dummy_vision)
    model = DeepSeekVL(
        llm_name=str(base_dir),
        vision_name="offline-test-vision",
        projector_type="mlp",
        text_visual_gate_init=0.73,
        text_visual_gate_reg_weight=0.004,
        device="cpu",
    ).eval()
    with torch.no_grad():
        for parameter in gate_parameters(model).values():
            parameter.uniform_(-0.9, 1.5)
        for _, wrapper in model._iter_vl_expert_wrappers():
            wrapper.visual_expert.down_proj.weight.normal_(std=0.1)
    batch = multimodal_batch()
    with torch.no_grad():
        expected = model(**batch)

    artifact_dir = tmp_path / "artifacts"
    save_artifacts(model, str(artifact_dir))
    metadata = json.loads((artifact_dir / "vlm_meta.json").read_text(encoding="utf-8"))
    assert metadata["llm_trust_remote_code"] is False
    assert metadata["text_visual_gates"] is True
    assert metadata["text_visual_gate_init"] == 0.73
    assert metadata["text_visual_gate_reg_weight"] == 0.004

    resumed = DeepSeekVL.from_training_checkpoint(str(artifact_dir), device="cpu").eval()
    inference = DeepSeekVLForInference(str(artifact_dir), device="cpu")
    for loaded in (resumed, inference):
        assert loaded.image_token_id == 63
        assert loaded.text_visual_gates is True
        assert loaded.text_visual_gate_init == 0.73
        assert loaded.text_visual_gate_reg_weight == 0.004
        assert loaded.llm_trust_remote_code is False
        for key, value in model.projector_state_dict().items():
            torch.testing.assert_close(loaded.projector_state_dict()[key], value, rtol=0, atol=0)
        for key, value in model.vl_experts_state_dict().items():
            torch.testing.assert_close(loaded.vl_experts_state_dict()[key], value, rtol=0, atol=0)
        with torch.no_grad():
            actual = loaded(**batch)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-5, atol=1e-6)
