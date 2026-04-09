import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset
from PIL import Image
from datasets import load_dataset, interleave_datasets

from src.model.gigachat_vl import GigaChatVL, IMAGE_TOKEN, IGNORE_INDEX


def open_image(x: Any) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, str):
        return Image.open(x).convert("RGB")

    if isinstance(x, dict):
        if x.get("path") is not None:
            return Image.open(x["path"]).convert("RGB")
        if x.get("bytes") is not None:
            import io
            return Image.open(io.BytesIO(x["bytes"])).convert("RGB")

    raise ValueError(f"Unsupported image field type: {type(x)}")


def build_prompt(tokenizer, question: str) -> str:
    user_text = f"{IMAGE_TOKEN}\n{question}"

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": user_text}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return f"User: {user_text}\nAssistant:"


class FineVisionIterableDataset(IterableDataset):
    """
    Converts FineVision rows:
      {
        "images": [PIL.Image, ...],
        "texts": [{"user": ..., "assistant": ...}, ...],
        ...
      }

    into flat training samples:
      {
        "image": PIL.Image,
        "question": str,
        "answer": str,
      }
    """

    def __init__(
        self,
        hf_iterable,
        shuffle_conversations: bool = False,
        seed: int = 42,
        skip_multi_image: bool = True,
        max_turns_per_row: Optional[int] = None,
    ):
        self.hf_iterable = hf_iterable
        self.shuffle_conversations = shuffle_conversations
        self.seed = seed
        self.skip_multi_image = skip_multi_image
        self.max_turns_per_row = max_turns_per_row

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = random.Random(self.seed)

        for row in self.hf_iterable:
            images = row.get("images", None)
            texts = row.get("texts", None)

            if not images or not texts:
                continue

            if self.skip_multi_image and len(images) != 1:
                continue

            image = images[0]

            turns = list(texts)
            if self.shuffle_conversations:
                rng.shuffle(turns)

            if self.max_turns_per_row is not None:
                turns = turns[: self.max_turns_per_row]

            for turn in turns:
                if not isinstance(turn, dict):
                    continue

                user = turn.get("user", None)
                assistant = turn.get("assistant", None)

                if not user or not assistant:
                    continue

                yield {
                    "image": image,
                    "question": str(user).strip(),
                    "answer": str(assistant).strip(),
                }


@dataclass
class VLMDataCollator:
    model: GigaChatVL
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        tokenizer = self.model.tokenizer

        images = []
        prompts = []
        full_texts = []

        for ex in features:
            try:
                image = open_image(ex["image"])
            except Exception as e:
                print(f"Skipping sample with unreadable image: {e}")
                continue
            question = ex["question"]
            answer = ex["answer"]

            prompt = build_prompt(tokenizer, question)
            full_text = prompt + answer + tokenizer.eos_token

            images.append(image)
            prompts.append(prompt)
            full_texts.append(full_text)

        if not images:
            raise RuntimeError("All samples in the batch were filtered out due to image errors.")

        text_batch = tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        labels = text_batch["input_ids"].clone()
        labels[text_batch["attention_mask"] == 0] = IGNORE_INDEX

        for i, prompt in enumerate(prompts):
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]

            prefix_len = min(len(prompt_ids), labels.size(1))
            labels[i, :prefix_len] = IGNORE_INDEX

        vision_batch = self.model.prepare_vision_inputs(images)

        return {
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
            "labels": labels,
            **vision_batch,
        }


def load_finevision_streaming(
    subsets: List[str],
    shuffle_buffer: int,
    seed: int,
    dataset_root: Optional[str] = None,
):
    streams = []
    for subset in subsets:
        if dataset_root is not None:
            subset_dir = os.path.join(dataset_root, subset)
            data_files = sorted(
                os.path.join(subset_dir, name)
                for name in os.listdir(subset_dir)
                if name.endswith(".parquet")
            )
            if not data_files:
                raise FileNotFoundError(
                    f"No parquet shards found for subset '{subset}' in {subset_dir}"
                )

            ds = load_dataset(
                "parquet",
                data_files={"train": data_files},
                split="train",
                streaming=True,
            )
        else:
            ds = load_dataset(
                "HuggingFaceM4/FineVision",
                name=subset,
                split="train",
                streaming=True,
            )
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        streams.append(ds)

    if len(streams) == 1:
        return streams[0]

    return interleave_datasets(streams)
