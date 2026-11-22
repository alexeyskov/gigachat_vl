"""Migration checks that do not load model checkpoints."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")


def test_server_uses_metadata_and_unquantized_loading_by_default(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from src.api import server

    monkeypatch.setattr("sys.argv", ["server"])
    args = server.parse_args()
    assert args.checkpoint_dir == "runs/deepseek_v2_lite_vl_gated_alignment"
    assert args.llm_path is None
    assert args.vision_path is None
    assert args.use_4bit_llm is False

    captured = {}
    loaded = object()

    def loader(**kwargs):
        captured.update(kwargs)
        return loaded

    monkeypatch.setattr(server, "GigaChatVLForInference", loader)
    monkeypatch.setattr(server, "model", None)
    server.load_model(args.checkpoint_dir)
    assert server.model is loaded
    assert captured == {
        "checkpoint_dir": args.checkpoint_dir,
        "llm_name": None,
        "vision_name": None,
        "use_4bit_llm": False,
    }


def test_quantization_rejects_visual_gate_checkpoint_before_loading_weights(tmp_path):
    from quant_model import check_input_checkpoint

    (tmp_path / "llm").mkdir()
    (tmp_path / "projector.pt").touch()
    (tmp_path / "vlm_meta.json").write_text(
        json.dumps({"text_visual_gates": True, "use_llm_lora": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="text_visual_gates=True"):
        check_input_checkpoint(tmp_path)


def test_disabled_deepseek_gates_default_to_no_gate_penalty(monkeypatch):
    from torch import nn
    from src.model.deepseek_vl import DeepSeekVL
    from src.model.gigachat_vl import GigaChatVL

    captured = {}

    def capture_init(self, **kwargs):
        nn.Module.__init__(self)
        captured.update(kwargs)

    monkeypatch.setattr(GigaChatVL, "__init__", capture_init)
    DeepSeekVL(text_visual_gates=False)
    assert captured["text_visual_gates"] is False
    assert captured["text_visual_gate_reg_weight"] == 0.0


@pytest.mark.parametrize("custom_loss", [False, True])
def test_trainer_rejects_losses_that_hide_labels_from_gates(tmp_path, custom_loss):
    from torch import nn
    from transformers import TrainingArguments
    from src.model.deepseek_vl import DeepSeekVLTrainer

    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to="none",
        label_smoothing_factor=0.0 if custom_loss else 0.1,
    )
    kwargs = {"compute_loss_func": lambda *args, **kwargs: None} if custom_loss else {}
    with pytest.raises(ValueError, match="compute_loss_func|label_smoothing_factor"):
        DeepSeekVLTrainer(model=nn.Linear(2, 2), args=args, **kwargs)
