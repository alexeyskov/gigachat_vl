from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import json
from fastapi.responses import StreamingResponse
import uvicorn
import json
import argparse
from PIL import Image
import base64
import io

from src.model.gigachat_vl import GigaChatVLForInference

def parse_args():
    parser = argparse.ArgumentParser(description="GigaChat-VL OpenAI-compatible server")
    parser.add_argument("--checkpoint_dir", type=str, default="/media/alexey/SSDData/experiments/gigachat_vl/outputs/gigachat_vl_finevision_local/")
    parser.add_argument("--llm_path", type=str, default="/media/alexey/HDDLargeData/models/VLM/GigaChat3.1-10B-A1.8B-bf16")
    parser.add_argument("--vision_path", type=str, default="/media/alexey/HDDLargeData/models/VLM/gemma-4-31B-it/")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()

model = None

def load_model(checkpoint_dir: str, llm_path: str, vision_path: str):
    global model
    print("Loading GigaChatVLForInference model...")
    model = GigaChatVLForInference(
        checkpoint_dir=checkpoint_dir,
        llm_name=llm_path,
        vision_name=vision_path,
        use_4bit_llm=True,
    )
    print("Model loaded successfully!")

app = FastAPI(title="GigaChat-VL OpenAI-compatible server")


def process_openai_message(messages: list) -> tuple[str, Image.Image | None]:
    """Parse OpenAI chat format and extract prompt + optional image."""
    # Take the last user message
    user_msg = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
    if not user_msg or "content" not in user_msg:
        return "No user message found", None

    content = user_msg["content"]
    prompt_parts: list[str] = []
    image: Image.Image | None = None

    if isinstance(content, str):
        prompt_parts.append(content)
    else:
        for part in content:
            if part.get("type") == "text":
                prompt_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part["image_url"].get("url", "")
                if url.startswith("data:image"):
                    if "," in url:
                        base64_data = url.split(",", 1)[1]
                    else:
                        base64_data = url
                    image_bytes = base64.b64decode(base64_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prompt = "\n".join(prompt_parts).strip()
    return prompt, image


@app.get("/v1/models")
async def list_models():
    """Required by OpenAI-compatible clients (including VLMEvalKit)."""
    return {
        "object": "list",
        "data": [{"id": "gigachat-vl", "object": "model", "created": 0, "owned_by": "custom"}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        stream = body.get("stream", False)
        messages = body.get("messages", [])

        # processing title generation request
        if len(messages) > 0 and messages[0].get("role") == "system":
            system_content = messages[0].get("content", "")
            if "chat thread titling assistant" in system_content and "Produce a very short, descriptive title" in system_content:
                user_msg = messages[-1].get("content", "Chat")
                title = user_msg[:30].strip() if isinstance(user_msg, str) else "New Chat"
                response_text = title[:50]

                if stream:
                    return StreamingResponse(
                        iter([f"data: {{\"choices\": [{{\"delta\": {{\"content\": \"{response_text}\"}}}}]}}\n\n"]),
                        media_type="text/event-stream"
                    )
                else:
                    return {
                        "model": "gigachat-vl",
                        "choices": [{"message": {"role": "assistant", "content": response_text}}]
                    }

        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)

        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            temperature = 0.7

        try:
            top_p = float(top_p)
        except (TypeError, ValueError):
            top_p = 1.0

        do_sample = temperature > 0.0

        prompt, image = process_openai_message(messages)

        response_text = model.inference(
            text=prompt,
            image=image,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        if stream:
            async def fake_stream():
                yield f'data: {{"choices": [{{"delta": {{"content": "{response_text}"}}}}]}}\n\n'
                yield 'data: [DONE]\n\n'
            return StreamingResponse(fake_stream(), media_type="text/event-stream")
        else:
            return {
                "model": "gigachat-vl",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    args = parse_args()
    
    load_model(args.checkpoint_dir, args.llm_path, args.vision_path)
    
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")