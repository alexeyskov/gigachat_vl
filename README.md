# gigachat_vl
GigaChat VL

## LLM connector projector

In addition to `projector_type="mlp"` and `projector_type="qformer"`, the model
can use a small causal LM as the visual connector:

```python
model = GigaChatVL(
    llm_name=LLM_PATH,
    vision_name=VISION_PATH,
    projector_type="llm",
    connector_llm_name="Qwen/Qwen3-0.6B",
    freeze_connector_llm=True,
)
```

The connector receives visual features via `inputs_embeds` and projects its
hidden states back to the GigaChat hidden size. With `freeze_connector_llm=True`,
only the input/output projection layers and norms are trained and saved in
`projector.pt`. Set `freeze_connector_llm=False` to finetune and save the
connector backbone as part of the projector checkpoint.

To train the connector with QLoRA, leave the donor vision encoder frozen and
enable connector LoRA:

```python
model = GigaChatVL(
    llm_name=LLM_PATH,
    vision_name=VISION_PATH,
    freeze_vision=True,
    projector_type="llm",
    connector_llm_name="Qwen/Qwen3-0.6B",
    freeze_connector_llm=False,
    connector_llm_use_qlora=True,
    connector_lora_r=16,
    connector_lora_alpha=32,
    connector_lora_dropout=0.05,
)
```

`save_artifacts(...)` writes connector adapters to `connector_llm_lora/`.
During inference the loader restores them automatically from the checkpoint
metadata when `connector_llm_use_qlora=True`.

`GigaChatVLForInference` can be forced onto a specific device:

```python
model = GigaChatVLForInference(
    checkpoint_dir=CHECKPOINT_DIR,
    llm_name=LLM_PATH,
    vision_name=VISION_PATH,
    device="cpu",        # or "cuda", "cuda:0"
    use_4bit_llm=False,  # 4-bit loading is CUDA-only
)
```

## Trainable Patch-LLM Vision Encoder

For experiments without a pretrained vision tower, set `vision_backend="patch_llm"`.
In this mode `vision_name` points to a small causal LM, for example Qwen3-0.6B.

```python
model = GigaChatVL(
    llm_name=LLM_PATH,
    vision_name="Qwen/Qwen3-0.6B",
    vision_backend="patch_llm",
    freeze_vision=False,
    vision_image_size=1024,  # max side size, aspect ratio is preserved
    vision_patch_size=32,
    vision_conv_hidden_size=256,
    vision_use_maxpool=False,
    vision_llm_use_qlora=True,
    vision_llm_dtype="bf16",  # "auto", "bf16", "fp16", or "fp32"
    projector_type="mlp",
)
```

The image is resized only when its longest side is larger than
`vision_image_size`; aspect ratio is preserved. Smaller images are not upscaled.
After that, pixels are normalized, padded to a multiple of `vision_patch_size`,
and split by a stride-`P` patch convolution. A 1024-pixel wide image with `P=32`
therefore produces 32 patch columns. The conv tokens are refined with two local
convolutions, enriched with learned 2D-grid position embeddings interpolated
from the `vision_image_size` max-side reference grid, projected into the small
LM hidden size, and then contextualized by the small LM.
`vision_llm_use_qlora=True` loads that small LM in 4-bit on CUDA and trains
LoRA adapters plus the conv/MLP/position-embedding front-end. Set
`vision_llm_use_qlora=False` with `freeze_vision=False` for full finetuning;
`vision_llm_dtype` controls the small LM and 4-bit compute dtype.
`vision_use_maxpool=False` is the recommended default for OCR and small-object
tasks; enabling it halves the patch grid but drops fine detail.

## Donor Vision Encoder LoRA / QLoRA

For a pretrained donor vision encoder, such as the Gemma4, Qwen VL, SigLIP, or
AIMv2 vision tower, keep the connector as a plain MLP and train the donor vision
tower with LoRA/QLoRA:

```python
model = GigaChatVL(
    llm_name=LLM_PATH,
    vision_name=VISION_PATH,
    use_4bit_llm=True,
    freeze_vision=False,
    projector_type="mlp",
    vision_use_qlora=True,
    vision_lora_r=16,
    vision_lora_alpha=32,
    vision_lora_dropout=0.05,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    num_lora_layers=NUM_LORA_LAYERS,
)
```

`vision_use_qlora=True` replaces donor vision tower linear layers with 4-bit
NF4 layers on CUDA, freezes the base weights, and trains LoRA adapters. Use
`vision_use_lora=True` instead for bf16/fp16 LoRA without 4-bit quantization.
For Gemma4, the donor `embed_vision` projector is saved as `vision_projector.pt`
when `freeze_vision=False`; the LoRA adapter is saved to `vision_lora/`.

AIMv2 can be used directly as a donor vision encoder:

```python
model = GigaChatVL(
    llm_name=LLM_PATH,
    vision_name="apple/aimv2-large-patch14-native",
    freeze_vision=True,
    projector_type="mlp",
)
```

## Precomputed Vision Features

For frozen vision towers, you can precompute raw visual features once and reuse
them during training. The script writes files to
`<dataset_root>/embeddings/<visual_encoder_name>/`:

```bash
python scripts/precompute_vision_embeddings.py
```

The precompute script should use `limit=None` for image datasets so every image
that may later appear in training has a cached feature. Training specs can still
use smaller `limit` values. Increase `BATCH_SIZE` in the script if GPU
utilization is low and there is free VRAM.

Then pass the same visual encoder path in each dataset spec:

```python
dataset_specs = [
    {
        "config": SupportedDatasets.LLAVA_PRETRAIN_RU.value,
        "limit": 100_000,
        "dataset_root": "/path/to/Maya",
        "visual_encoder": VISION_PATH,
    },
]
```

`VLMDataCollator` uses precomputed features only when `model.freeze_vision=True`.
If a feature file is missing, or the vision tower is trainable, images are
encoded online as before. The saved tensors are taken before the trainable MLP
projector, so the projector can still be trained.

To make training fail fast instead of silently falling back to online vision
encoding, enable the strict collator mode:

```python
collator = VLMDataCollator(
    model=model,
    max_length=MAX_LENGTH,
    require_precomputed_vision_embeddings=True,
)
```

When every image dataset has precomputed features, the frozen donor vision tower
can also be dropped after the model has built its projector:

```python
model.drop_frozen_vision_modules()
```

## SmolTalk2 Text Data

`HuggingFaceTB/smoltalk2` is available through the unified dataset loader as
text-only training data:

```python
from src.dataset.unified_vlm_dataset import SupportedDatasets

dataset_specs = [
    {
        "config": SupportedDatasets.SMOLTALK2_SFT.value,
        "limit": 100_000,
        "load_kwargs": {
            # Optional: omit this to interleave all SFT splits.
            "splits": [
                "smoltalk-multilingual-8languages_lang_5_no_think",
                "OpenHermes-2.5_no_think",
            ],
        },
        "dataset_kwargs": {
            "strip_thinking": True,
            "emit_all_assistant_turns": True,
        },
    },
]
```

The wrapper reads OpenAI-style `messages` and emits
`{"image": None, "question": ..., "answer": ...}` samples compatible with
`VLMDataCollator`.
