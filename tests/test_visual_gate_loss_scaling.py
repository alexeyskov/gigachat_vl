"""Regression checks for gate penalties across mixed microbatch accumulation."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from transformers import TrainingArguments

from src.model.deepseek_vl import DeepSeekVLTrainer
from test_deepseek_vl_integration import gate_parameters, multimodal_batch, tiny_vlm


def penalty_delta(model, batch, num_items_in_batch=None):
    weight = model.text_visual_gate_reg_weight
    torch.manual_seed(173)
    with_penalty = model(**batch, num_items_in_batch=num_items_in_batch).loss
    model.text_visual_gate_reg_weight = 0.0
    try:
        torch.manual_seed(173)
        without_penalty = model(**batch, num_items_in_batch=num_items_in_batch).loss
    finally:
        model.text_visual_gate_reg_weight = weight
    return with_penalty - without_penalty


def split_batch(batch):
    offset = 0
    for index, image_count in enumerate(batch["vision_num_images_per_sample"].tolist()):
        sample = {
            name: batch[name][index:index + 1]
            for name in ("input_ids", "attention_mask", "labels", "vision_num_images_per_sample")
        }
        if image_count:
            sample["vision_precomputed_features"] = batch["vision_precomputed_features"][offset:offset + image_count]
        offset += image_count
        yield sample


@pytest.mark.parametrize("mixed", [False, True])
def test_gate_penalty_is_invariant_to_microbatch_splitting(mixed, tmp_path):
    model = tiny_vlm().train()
    model.text_visual_gate_reg_weight = 0.5
    batch = multimodal_batch()
    if not mixed:
        batch["input_ids"][1] = torch.tensor([1, 63, 9, 10])
        batch["attention_mask"][1] = 1
        batch["labels"][1] = torch.tensor([-100, -100, -100, 10])
        batch["vision_num_images_per_sample"][1] = 1
        batch["vision_precomputed_features"].append(torch.randn(2, 6))

    arguments = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to="none",
        gradient_accumulation_steps=2,
    )
    trainer = DeepSeekVLTrainer(model=model, args=arguments)
    microbatches = list(split_batch(batch))
    denominator = trainer._get_num_items_in_batch(microbatches, torch.device("cpu"))
    assert trainer._loss_shifts_labels is True
    torch.testing.assert_close(denominator, batch["labels"][..., 1:].ne(-100).sum())

    # Position zero is excluded even when callers supply a supervised label there.
    count_probe = {**microbatches[0], "labels": microbatches[0]["labels"].clone()}
    count_probe["labels"][0, 0] = 1
    probe_count = trainer._get_num_items_in_batch([count_probe], torch.device("cpu"))
    torch.testing.assert_close(probe_count, count_probe["labels"][..., 1:].ne(-100).sum())

    full_penalty = penalty_delta(model, batch)
    accumulated_penalty = sum(penalty_delta(model, sample, denominator) for sample in microbatches)
    torch.testing.assert_close(accumulated_penalty, full_penalty, rtol=1e-5, atol=1e-6)

    visual_rows = batch["vision_num_images_per_sample"].gt(0)
    visual_targets = batch["labels"][visual_rows, 1:].ne(-100).sum()
    expected = (
        model.text_visual_gate_reg_weight
        * model._text_visual_gate_regularization_loss()
        * visual_targets / denominator
    )
    torch.testing.assert_close(full_penalty, expected, rtol=1e-5, atol=1e-6)


def test_global_penalty_weights_individual_experts_equally():
    model = tiny_vlm()
    for layer_index, wrapper in model._iter_vl_expert_wrappers():
        with torch.no_grad():
            for parameter in wrapper.text_visual_gate.parameters():
                parameter.fill_(-20 if layer_index == 0 else 20)
    expected_probabilities = []
    for parameter in gate_parameters(model).values():
        expected_probabilities.append(
            torch.sigmoid(parameter.float() - (2.0 / 3.0) * torch.log(torch.tensor(1.0 / 11.0)))
        )
    expected = torch.cat(expected_probabilities).mean()
    torch.testing.assert_close(model._text_visual_gate_regularization_loss(), expected)
