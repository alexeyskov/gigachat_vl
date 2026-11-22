import argparse
import os
import sys

import torch
from transformers import Trainer, TrainingArguments, set_seed

from src.dataset import SupportedDatasets, VLMDataCollator, load_merged_dataset
from src.model.deepseek_vl import DEFAULT_DEEPSEEK_NAME, DeepSeekVL, DeepSeekVLTrainer
from src.model.text_moe_backbones import TEXT_MOE_NAMES
from src.model.text_moe_vl import GptOssVL, LFM2MoEVL, SmallThinkerVL
from src.utils.train_utils import SaveVLMArtifactsCallback, save_artifacts


EXP_NAME = "deepseek_v2_lite_vl_gated_alignment"
PROJECT_ROOT = "./"
OUTPUT_DIR = os.path.join("runs", EXP_NAME)

for p in [PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.append(p)

LLM_PATH = DEFAULT_DEEPSEEK_NAME
VISION_PATH = "./models/Qwen2.5-VL-7B-Instruct/"
# A projector trained for GigaChat must be realigned to the DeepSeek backbone.
PROJECTOR_PATH = None

BACKBONE_CLASSES = {
    "deepseek": DeepSeekVL,
    "gpt_oss": GptOssVL,
    "smallthinker": SmallThinkerVL,
    "lfm2_moe": LFM2MoEVL,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Multimodal alignment of text MoE backbones.")
    parser.add_argument("--backbone", choices=BACKBONE_CLASSES, default="deepseek")
    parser.add_argument("--llm_path", default=None, help="Optional local checkpoint or Hugging Face ID.")
    parser.add_argument("--vision_path", default=VISION_PATH)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=25_000)
    parser.add_argument("--train_llm", action="store_true", help="Also train all language weights.")
    return parser.parse_args(argv)

dataset_specs_full = [
    {
        "config": SupportedDatasets.LLAVA_PRETRAIN_RU.value,
        "limit": 10_000,
        "dataset_root": "./datasets/Maya/",
        "visual_encoder": VISION_PATH,
        "download": True,
    },
    {
        "config": SupportedDatasets.MSCOCO_CAPTION_ML.value,
        "limit": 10_000,
        "dataset_root": "./datasets/mscoco-multilingual-30k/",
        "visual_encoder": VISION_PATH,
    },
    {
        "config": SupportedDatasets.RUSTITW_OCR.value,
        "limit": 10_000,
        "dataset_root": "./datasets/rustitw_ocr/",
        "visual_encoder": VISION_PATH,
    },
    {
        "config": SupportedDatasets.OPENHERMES_RU_TEXT.value,
        "limit": 5_000,
        "dataset_root": "./datasets/OpenHermes-2.5-ru/",
    },
    {
        "config": SupportedDatasets.GQA_RU.value,
        "limit": 7_000,
        "dataset_root": "./datasets/GQA-ru/",
        "visual_encoder": VISION_PATH,
        "download": True,
    },
    {
        "config": SupportedDatasets.LLAVA_INSTRUCT_RU.value,
        "limit": 5_000,
        "dataset_root": "./datasets/LLaVA-Instruct-ru",
        "visual_encoder": VISION_PATH,
        "download": True,
    },
    {
        "config": SupportedDatasets.RU_VLM_REASONING_SFT.value,
        "limit": 3_500,
        "dataset_root": "./datasets/ru-vlm-reasoning-sft",
        "visual_encoder": VISION_PATH,
        "download": True,
    },
]

dataset_specs_connector = [
    {
        "config": SupportedDatasets.LLAVA_PRETRAIN_RU.value,
        "limit": 100_000,
        "dataset_root": "./datasets/Maya/",
        "visual_encoder": VISION_PATH,
        "download": True,
    },
    {
        "config": SupportedDatasets.MSCOCO_CAPTION_ML.value,
        "limit": 30_000,
        "dataset_root": "./datasets/mscoco-multilingual-30k/",
        "visual_encoder": VISION_PATH,
    },
    {
        "config": SupportedDatasets.RUSTITW_OCR.value,
        "limit": 50_000,
        "dataset_root": "./datasets/rustitw_ocr/",
        "visual_encoder": VISION_PATH,
    },
]


if __name__ == "__main__":
    cli_args = parse_args()
    model_class = BACKBONE_CLASSES[cli_args.backbone]
    LLM_PATH = cli_args.llm_path or (
        DEFAULT_DEEPSEEK_NAME if cli_args.backbone == "deepseek" else TEXT_MOE_NAMES[cli_args.backbone]
    )
    VISION_PATH = cli_args.vision_path
    EXP_NAME = (
        EXP_NAME if cli_args.backbone == "deepseek" else f"{cli_args.backbone}_vl_alignment"
    )
    OUTPUT_DIR = cli_args.output_dir or os.path.join("runs", EXP_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset_specs = dataset_specs_full

    MAX_STEPS = cli_args.max_steps
    LR = 1e-5
    GATE_LR = 1e-3
    WEIGHT_DECAY = 0.0
    WARMUP_RATIO = 0.03
    warmup_steps = int(MAX_STEPS * WARMUP_RATIO)

    TRAIN_BS = 1
    GRAD_ACCUM = 16
    MAX_LENGTH = 4096

    SHUFFLE_BUFFER = 1000
    LOGGING_STEPS = 25
    SAVE_STEPS = 2500
    SAVE_TOTAL_LIMIT = 2
    SEED = 42

    USE_4BIT_LLM = False
    USE_LLM_LORA = False
    # Start with frozen language weights; gates, visual experts and projector train.
    TRAIN_LLM = cli_args.train_llm
    FREEZE_VISION = True
    TRAIN_PROJECTOR = True
    USE_PRECOMPUTED_VISION_EMBEDDINGS = False
    REQUIRE_PRECOMPUTED_VISION_EMBEDDINGS = USE_PRECOMPUTED_VISION_EMBEDDINGS
    DROP_FROZEN_VISION_AFTER_PROJECTOR = FREEZE_VISION and USE_PRECOMPUTED_VISION_EMBEDDINGS

    ENABLE_VL_EXPERTS = True
    VL_EXPERT_LAYERS = None
    TEXT_VISUAL_GATES = cli_args.backbone == "deepseek"
    TEXT_VISUAL_GATE_INIT = 0.9
    TEXT_VISUAL_GATE_REG_WEIGHT = 1e-3 if TEXT_VISUAL_GATES else 0.0

    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if USE_PRECOMPUTED_VISION_EMBEDDINGS and not FREEZE_VISION:
        raise ValueError("Precomputed vision embeddings can only be used with FREEZE_VISION=True.")

    image_dataset_names = {
        SupportedDatasets.LLAVA_PRETRAIN_RU.value.name,
        SupportedDatasets.MSCOCO_CAPTION_ML.value.name,
        SupportedDatasets.RUSTITW_OCR.value.name,
        SupportedDatasets.GQA_RU.value.name,
        SupportedDatasets.LLAVA_INSTRUCT_RU.value.name,
        SupportedDatasets.MWS_VISION.value.name,
        SupportedDatasets.PIXMO_CAP_EN.value.name,
        SupportedDatasets.PIXMO_ASK_MODEL_ANYTHING_EN.value.name,
        SupportedDatasets.DOCVQA_EN.value.name,
        SupportedDatasets.INFOGRAPHICVQA_EN.value.name,
        SupportedDatasets.CHARTQA_EN.value.name,
    }

    for spec in dataset_specs:
        if spec["config"].name in image_dataset_names:
            if USE_PRECOMPUTED_VISION_EMBEDDINGS:
                spec["visual_encoder"] = VISION_PATH
            else:
                spec.pop("visual_encoder", None)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if not use_bf16:
        raise RuntimeError(
            "This script is configured for bf16 CUDA training. "
            "Run it on a CUDA device with sufficient memory for the selected stage."
        )

    model = model_class(
        llm_name=LLM_PATH,
        vision_name=VISION_PATH,
        use_4bit_llm=USE_4BIT_LLM,
        freeze_vision=FREEZE_VISION,
        chat_template_mode="tokenizer",
        max_image_side=1520,
        projector_type="mlp",
        projector_path=PROJECTOR_PATH,
        use_llm_lora=USE_LLM_LORA,
        train_llm_lora=TRAIN_LLM,
        num_lora_layers=-1,
        normalize_visual_embeddings=True,
        enable_vl_experts=ENABLE_VL_EXPERTS,
        vl_expert_layers=VL_EXPERT_LAYERS,
        text_visual_gates=TEXT_VISUAL_GATES,
        text_visual_gate_init=TEXT_VISUAL_GATE_INIT,
        text_visual_gate_reg_weight=TEXT_VISUAL_GATE_REG_WEIGHT,
    )

    if not TRAIN_PROJECTOR:
        model.projector.requires_grad_(False)

    model.gradient_checkpointing_enable()

    if hasattr(model.llm, "print_trainable_parameters"):
        try:
            model.llm.print_trainable_parameters()
        except Exception as e:
            print(f"print_trainable_parameters failed: {e}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"VLM trainable params: {trainable_params:,} / {all_params:,} "
        f"({100 * trainable_params / all_params:.4f}%)"
    )
    print(
        f"Training mode: native bf16 {cli_args.backbone}, train_llm={TRAIN_LLM}, "
        f"freeze_vision={FREEZE_VISION}, train_projector={TRAIN_PROJECTOR}, "
        f"enable_vl_experts={ENABLE_VL_EXPERTS}, vl_expert_layers={model.vl_expert_layer_indices}, "
        f"text_visual_gates={TEXT_VISUAL_GATES}, gate_lr={GATE_LR}"
    )

    train_dataset = load_merged_dataset(
        dataset_specs=dataset_specs,
        global_seed=SEED,
        global_shuffle_buffer=SHUFFLE_BUFFER,
        interleave_stopping_strategy="all_exhausted",
        interleave_balance_probabilities=False,
    )

    if DROP_FROZEN_VISION_AFTER_PROJECTOR:
        model.drop_frozen_vision_modules()

    collator = VLMDataCollator(
        model=model,
        max_length=MAX_LENGTH,
        require_precomputed_vision_embeddings=REQUIRE_PRECOMPUTED_VISION_EMBEDDINGS,
    )

    training_args = TrainingArguments(
        optim="adamw_torch",
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=TRAIN_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_grad_norm=1.0,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=True,
        fp16=False,
        tf32=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )

    trainer_class = DeepSeekVLTrainer if TEXT_VISUAL_GATES else Trainer
    trainer_kwargs = {"gate_learning_rate": GATE_LR} if TEXT_VISUAL_GATES else {}
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[SaveVLMArtifactsCallback()],
        **trainer_kwargs,
    )

    trainer.train()

    save_artifacts(model, OUTPUT_DIR)
    print(f"Saved final artifacts to: {OUTPUT_DIR}")
