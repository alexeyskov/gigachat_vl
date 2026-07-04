# Run Chat UI for GigaChat-VL

## 1. Start the Model Server

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
ё
The server runs by default on http://localhost:8000/v1.

## 2. Start the Web UI (Hugging Face Chat UI)

**On Linux:**

```bash
docker run -d \
  -p 3000:3000 \
  -e OPENAI_BASE_URL=http://172.17.0.1:8000/v1 \
  -e OPENAI_API_KEY=sk-anything \
  -v chat-ui-data:/data \
  --name hf-chat \
  ghcr.io/huggingface/chat-ui-db:latest
```

**On macOS / Windows:**

```bash
docker run -d \
  -p 3000:3000 \
  -e OPENAI_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=sk-anything \
  -v chat-ui-data:/data \
  --name hf-chat \
  ghcr.io/huggingface/chat-ui-db:latest
```

Open browser → http://localhost:3000

## 3. UI Configuration

- Go to **Settings** (gear icon)
- Select model **`gigachat-vl`**
- Enable **Multimodal support**