# Evaluation Guide for GigaChat-VL

## 1. Install VLMEvalKit

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```  
This installs VLMEvalKit in editable mode (required for your custom datasets to be registered automatically).

## 2. Start the Model Server

Launch the OpenAI-compatible server:

```bash
python -m src.api.server
```

```bash
python -m src.api.server \
  --checkpoint_dir /path/to/checkpoints \
  --llm_path /path/to/llm \
  --vision_path /path/to/vision \
  --port 8000
```

The server runs by default on http://localhost:8000/v1.

## 3. Run Evaluation

Once the server is up, run the evaluation:

```bash
python src/eval/run_eval.py \
  --model GigachatVL \
  --base-url http://localhost:8000/v1 \
  --data MMBenchRU \
  --api-mode \
  --work-dir /path/to/gigachat_vl_eval \
  --judge exact_matching
```

### What each flag does:

| Flag                  | Description |
|-----------------------|-------------|
| `--model GigachatVL`  | **(required)** Internal model name used by VLMEvalKit |
| `--base-url`          | **(required)**  URL of your OpenAI-compatible API server |
| `--data`              | **(required)** Dataset to evaluate |
| `--api-mode`          | **(required)** Use HTTP API instead of loading the model locally |
| `--work-dir`          | Folder for logs, predictions, and final results (Excel + JSONL) |
| `--judge exact_matching` | Evaluation method (exact string match for MCQ answers) |
| `--mode`              | default to 'all', choices are ['all', 'infer', 'eval'] When mode set to "all", will perform both inference and evaluation; when set to "infer", will only perform the inference |

## Datasets

### Automatic Dataset Handling
- On the **first run** of any dataset, VLMEvalKit **automatically downloads** it (if needed), converts it to the internal `.tsv` format, and prepares images.
- All data is cached locally in `~/LMUData/` (or the path set via the `LMUData` environment variable).
- Subsequent runs load the cached version instantly.

```bash
export LMUData=/custom/path/to/data   # optional
```

### Built-in Russian Datasets
Currently available Russian / Russian-translated datasets:

- `MMMB_ru` — Russian subset of the multilingual MMMB benchmark
- `MMBench_dev_ru` — Russian translation of MMBench dev split

Full list of all built-in datasets can be found [here](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe).

### Custom Datasets
- `MMBenchRU` — Russian translation of MMBench
