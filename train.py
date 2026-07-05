import json
import os
import sys

import torch
from transformers import Trainer, TrainingArguments, set_seed

from src.dataset.finevision import VLMDataCollator
from src.dataset.unified_vlm_dataset import SupportedDatasets, load_merged_dataset
from src.model.gigachat_vl import GigaChatVL
from src.utils.train_utils import SaveVLMArtifactsCallback, save_artifacts


NEED_PATH_GIGACHAT = False

EXP_NAME = "gigachat_vl_qwen25_full_bf16_vl_experts"
PROJECT_ROOT = "./"
OUTPUT_DIR = os.path.join("runs", EXP_NAME)

for p in [PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.append(p)

os.makedirs(OUTPUT_DIR, exist_ok=True)

LLM_PATH = "./models/GigaChat3.1-10B-A1.8B-bf16"
VISION_PATH = "./models/Qwen2.5-VL-7B-Instruct/"
PROJECTOR_PATH = "./models/projector.pt"

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
    dataset_specs = dataset_specs_full

    if NEED_PATH_GIGACHAT:
        config_path = os.path.join(LLM_PATH, "config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        print(
            "Before:",
            cfg.get("routed_scaling_factor"),
            type(cfg.get("routed_scaling_factor")),
        )

        if "routed_scaling_factor" in cfg and isinstance(cfg["routed_scaling_factor"], int):
            cfg["routed_scaling_factor"] = float(cfg["routed_scaling_factor"])

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        print(
            "After:",
            cfg.get("routed_scaling_factor"),
            type(cfg.get("routed_scaling_factor")),
        )
        print("Patched:", config_path)

    MAX_STEPS = 25_000
    LR = 1e-5
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
    TRAIN_LLM = True
    FREEZE_VISION = True
    TRAIN_PROJECTOR = True
    USE_PRECOMPUTED_VISION_EMBEDDINGS = False
    REQUIRE_PRECOMPUTED_VISION_EMBEDDINGS = USE_PRECOMPUTED_VISION_EMBEDDINGS
    DROP_FROZEN_VISION_AFTER_PROJECTOR = FREEZE_VISION and USE_PRECOMPUTED_VISION_EMBEDDINGS

    ENABLE_VL_EXPERTS = True
    VL_EXPERT_LAYERS = None

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
            "Run it on the RTX 6000 Pro server."
        )

    model = GigaChatVL(
        llm_name=LLM_PATH,
        vision_name=VISION_PATH,
        use_4bit_llm=USE_4BIT_LLM,
        freeze_vision=FREEZE_VISION,
        chat_template_mode="short",
        max_image_side=1520,
        projector_type="mlp",
        projector_path=PROJECTOR_PATH,
        use_llm_lora=USE_LLM_LORA,
        train_llm_lora=TRAIN_LLM,
        num_lora_layers=-1,
        normalize_visual_embeddings=True,
        enable_vl_experts=ENABLE_VL_EXPERTS,
        vl_expert_layers=VL_EXPERT_LAYERS,
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
        "Training mode: full bf16 LLM, "
        f"freeze_vision={FREEZE_VISION}, train_projector={TRAIN_PROJECTOR}, "
        f"enable_vl_experts={ENABLE_VL_EXPERTS}, vl_expert_layers={model.vl_expert_layer_indices}"
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
        optim="paged_adamw_8bit",
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[SaveVLMArtifactsCallback()],
    )

    trainer.train()

    save_artifacts(model, OUTPUT_DIR)
    print(f"Saved final artifacts to: {OUTPUT_DIR}")
