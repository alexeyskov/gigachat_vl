# gigachat_vl
GigaChat VL

## Additional text MoE experiments

The CUDA training entrypoint also supports these official text backbones:

| `--backbone` | Default checkpoint | Python class |
| --- | --- | --- |
| `gpt_oss` | `openai/gpt-oss-20b` | `GptOssVL` |
| `smallthinker` | `Tiiny/SmallThinker-4BA0.6B-Instruct` | `SmallThinkerVL` |
| `lfm2_moe` | `LiquidAI/LFM2.5-8B-A1B` | `LFM2MoEVL` |

Install into your CUDA environment (install the appropriate CUDA PyTorch build
first), then run one experiment:

```bash
pip install -r requirements-moe.txt
python train.py --backbone gpt_oss --vision_path Qwen/Qwen2.5-VL-7B-Instruct
python train.py --backbone smallthinker --vision_path Qwen/Qwen2.5-VL-7B-Instruct
python train.py --backbone lfm2_moe --vision_path Qwen/Qwen2.5-VL-7B-Instruct
```

`--llm_path` accepts a local checkpoint or overrides the default Hugging Face ID.
`--output_dir` overrides `runs/<backbone>_vl_alignment`. For a short GPU smoke run,
append `--max_steps 2`. Dataset roots and mixtures are still configured in
`train.py`; prepare those datasets on the VPS before starting. Running without
`--backbone` preserves the DeepSeek experiment.

These experiments freeze the language backbone and donor vision encoder and
train a new MLP projector plus added visual experts. Their default path uses
native bf16 on CUDA; `--train_llm` additionally trains all language weights.
DeepSeek's learned input gates are not enabled for these backbones.

Model-specific details:

- **GPT-OSS:** uses the official `openai/gpt-oss-20b` checkpoint. Its MXFP4 expert
  weights are dequantized for a differentiable training path. Budget roughly
  **42 GB for bf16 language weights alone**, plus vision, activations, and
  optimizer state. This is not the checkpoint's compact MXFP4 inference path.
  The official Harmony template is used with a `final` channel prefix for direct
  visual answers. Router scores returned by the FFN are preserved.
- **SmallThinker:** the project supplies a checkpoint-compatible implementation
  on the same Transformers version as the other models. It preserves ReGLU
  experts and the router's pre-attention input, and adds causal training loss,
  gradient checkpointing, and native cached generation. It supports the
  published 4B checkpoint's all-RoPE, full-attention layout. Loading does not
  require the older remote implementation or `trust_remote_code=True`.
- **LFM2.5:** uses native `Lfm2MoeForCausalLM`, including convolution state in the
  generation cache. Visual experts wrap `feed_forward` blocks in both dense and
  MoE layers. The official chat template is preserved; the post-trained base
  model has reasoning behavior, while visual SFT examples supervise the provided
  answers.

GPT-OSS and LFM2-MoE reject the generic LLM LoRA/4-bit flags because their packed
expert parameters are not ordinary linear layers. SmallThinker can use the
existing linear LoRA path. All three reuse the current vision backends,
precomputed features, datasets, and artifact format.

Python usage and checkpoint loading:

```python
from src.model.text_moe_vl import GptOssVL, SmallThinkerVL, LFM2MoEVL
from src.model.gigachat_vl import GigaChatVLForInference
from src.utils.train_utils import save_artifacts

model = SmallThinkerVL(vision_name="Qwen/Qwen2.5-VL-7B-Instruct")
save_artifacts(model, "runs/smallthinker_vl_alignment")

model = GigaChatVLForInference(
    checkpoint_dir="runs/smallthinker_vl_alignment",
    use_4bit_llm=False,
    device="cuda",
)
# Training can resume from saved model artifacts using:
# model = SmallThinkerVL.from_training_checkpoint("runs/smallthinker_vl_alignment")
```

The existing API server restores the model type and chat behavior from the saved
language config. For example:

```bash
python -m src.api.server --checkpoint_dir runs/smallthinker_vl_alignment
```

Local validation uses small randomly initialized CPU models; full checkpoint
loading, CUDA kernels, and end-to-end GPU training must be validated on the VPS.

## DeepSeek-V2-Lite with learned visual access

`train.py` now starts a DeepSeek-V2-Lite-Chat experiment using the existing
vision encoder, projector, datasets, and checkpoint format. Install the native
DeepSeek dependencies with:

```bash
pip install -r requirements-deepseek.txt
```

This path pins Transformers 5.15.0 and loads the native DeepSeek-V2 classes with
`llm_trust_remote_code=False`. It does not use the older model repository's
remote Python implementation. The default is
[`deepseek-ai/DeepSeek-V2-Lite-Chat`](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat),
which has 16B total parameters and about 2.4B active parameters per token.
The [model configuration](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json)
has 27 decoder layers: one dense layer followed by MoE layers with 64 routed
experts, top-6 routing, and two shared experts. Active parameters describe
computation; all expert weights still need storage. Bf16 language weights alone
require roughly 32 GB, before the vision tower, activations, gradients, and
optimizer state.

```python
from src.model.deepseek_vl import DeepSeekVL

model = DeepSeekVL(
    vision_name=VISION_PATH,
    projector_path=None,
    projector_type="mlp",
    freeze_vision=True,
    use_4bit_llm=False,
    use_llm_lora=False,
    train_llm_lora=False,
    text_visual_gates=True,
    text_visual_gate_init=0.9,
    text_visual_gate_reg_weight=1e-3,
)
```

The original language experts already receive visual-token hidden states in
the previous implementation. Learned gates now control that direct access.
Each routed expert and each shared expert has its own scalar gate in every
selected layer; the first dense FFN has its own gate. For an expert `e`, token
`t`, image-origin indicator `m[t]`, and current hidden state `h[t]`, its input is:

```text
expert_input[e, t] = h[t] * (1 - m[t] + m[t] * z[e])
```

The router still chooses experts from the original hidden states. At image
positions, an open gate passes the current, contextualized visual hidden state;
a closed gate blocks it. Text-position inputs pass unchanged. The added visual
expert continues to receive unmasked hidden states at image-origin positions.
There is no second copy of the image embeddings: image tokens are inserted
once and their evolving hidden states are gated in each layer. During cached
decoding, new text tokens use the normal language path; visual context remains
available through attention to the cache.

Gates use hard-concrete samples during training and deterministic clipped
values during inference, following the
[hard-concrete L0 formulation](https://arxiv.org/abs/1712.01312).
`text_visual_gate_init=0.9` specifies the initial
deterministic gate value, not the probability of sampling a nonzero gate.
The regularizer adds `text_visual_gate_reg_weight` times the expected fraction
of open gates to the language-model loss. This lets task supervision justify
visual access while the regularizer encourages unnecessary access to close.
The coefficient is an experimental starting point; specialization is learned,
not guaranteed.
The penalty is averaged over all expert gates and weighted by supervised
targets from image-bearing samples, using the same token denominator as the
causal loss across gradient accumulation. Text-only samples contribute no gate
penalty. Closing a gate does not change top-k selection or skip the expert's
matrix multiplications in this implementation.

A closed gate means an expert does not directly process image-origin token
states. It does **not** guarantee that the expert is independent of the image:
shared self-attention can already carry visual information into text-position
states. Strict isolation would require separate attention streams or explicit
control of visual contributions before attention mixing.

The initial configuration in `train.py` freezes the language backbone and vision
tower and trains the new projector, visual experts, and gates. A GigaChat
projector is not reused: alignment must be learned for DeepSeek. The
`DeepSeekVLTrainer` uses `gate_learning_rate=1e-3`, separate from the remaining
parameters' learning rate, with no gate weight decay. Its logs include each
layer's deterministic gate mean, fraction of nonzero gates, and gate LR.
The normal warmup/cosine scheduler applies to both optimizer groups.

After alignment, continue from the saved model for a stage that also adapts the
language experts:

```python
model = DeepSeekVL.from_training_checkpoint(
    checkpoint_dir="runs/deepseek_v2_lite_vl_gated_alignment",
    use_4bit_llm=False,
    use_llm_lora=False,
    train_llm_lora=True,
)
```

Here `train_llm_lora=True` enables full language-model training because
`use_llm_lora=False`; the argument retains its historical name. Use this model
with `DeepSeekVLTrainer` and a new output directory. Full training needs
substantially more memory than frozen-backbone alignment. Keep text-only
examples in the training mixture and evaluate both text and visual tasks when
choosing the gate regularization strength.
The pinned native backend does not add the original remote implementation's
router balancing auxiliary loss. The router is frozen in the initial stage;
full-backbone experiments must account for this difference when evaluating
expert utilization.

Gated DeepSeek currently rejects LLM LoRA and 4-bit loading. Native routed
expert weights are packed three-dimensional parameters, so the repository's
linear-layer LoRA/QLoRA path cannot be assumed to cover them. Existing GigaChat
and SmolLM configurations remain available through their original classes.

Inference restores gate settings and weights from the saved artifacts:

```python
from src.model.deepseek_vl import DeepSeekVLForInference

model = DeepSeekVLForInference(
    checkpoint_dir="runs/deepseek_v2_lite_vl_gated_alignment",
    device="cuda",
)
```

The API server also reads model paths and gate settings from checkpoint
metadata and defaults to unquantized loading:

```bash
python -m src.api.server --checkpoint_dir runs/deepseek_v2_lite_vl_gated_alignment
```

`--llm_path` and `--vision_path` are optional overrides. `--use_4bit_llm` is
available for compatible legacy checkpoints; `quant_model.py` rejects gated
DeepSeek checkpoints before attempting conversion.

The native API was checked with a small randomly initialized DeepSeek-V2 model
on CPU, including visual-style `inputs_embeds`, loss/backward, and cached
generation. This is not validation of a complete 16B checkpoint or an end-to-end
GPU training run.

## LLM connector projector

In addition to `projector_type="mlp"` and `projector_type="qformer"`, the model
can use a small causal LM as the visual connector:

```python
model = GigaChatVL(
    llm_name=LLM_PATH,
    vision_name=VISION_PATH,
    chat_template_mode="short",
    max_image_side=1520,
    projector_type="llm",
    connector_llm_name="Qwen/Qwen3-0.6B",
    connector_llm_dtype="fp16",
    freeze_connector_llm=True,
)
```

By default the connector receives only raw visual features via `inputs_embeds`.
It returns contextualized visual-token hidden states, projected back to the
GigaChat hidden size. Set `connector_use_text_prefix=True` to also pass the
text-token embeddings that appear before each `[image_token]` placeholder.
With `freeze_connector_llm=True`, only the input/output projection layers and
norms are trained and saved in `projector.pt`. Set `freeze_connector_llm=False`
to finetune and save the connector backbone as part of the projector checkpoint.

For a first connector-only fp16 alignment stage, freeze the donor vision stack
and base LLM LoRA adapters while training the Qwen connector:

```python
model = GigaChatVL(
    llm_name=LLM_PATH,
    vision_name=VISION_PATH,
    chat_template_mode="short",
    max_image_side=1520,
    use_4bit_llm=True,
    freeze_vision=True,
    train_llm_lora=False,
    projector_type="llm",
    connector_llm_name="Qwen/Qwen3-0.6B",
    connector_llm_dtype="fp16",
    freeze_connector_llm=False,
    connector_llm_use_qlora=False,
    normalize_visual_embeddings=True,
)
```

Use `chat_template_mode="short"` for GigaChat VLM training to avoid the large
default developer/system preamble from the tokenizer chat template.
Use `max_image_side=1520` to downscale large input images before the selected
vision processor runs; aspect ratio is preserved and smaller images are not
upscaled.

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
from src.dataset import SupportedDatasets

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
