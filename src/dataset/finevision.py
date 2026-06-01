import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset
from PIL import Image
from datasets import load_dataset, interleave_datasets

from src.model.gigachat_vl import (
    GigaChatVL,
    IGNORE_INDEX,
    _build_chat_prompt,
    _build_chat_training_texts,
)
from src.dataset.precomputed_embeddings import load_precomputed_vision_feature


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


def build_prompt(tokenizer, question: str, num_images: int = 1) -> str:
    return _build_chat_prompt(tokenizer, question, num_images=num_images)


def build_prompt_and_full_text(
    tokenizer,
    question: str,
    answer: str,
    num_images: int = 1,
) -> tuple[str, str]:
    return _build_chat_training_texts(
        tokenizer,
        question=question,
        answer=answer,
        num_images=num_images,
    )


def _warn_limited(obj: Any, key: str, message: str, limit: int = 20) -> None:
    attr = f"_warn_count_{key}"
    count = int(getattr(obj, attr, 0))
    if count < limit:
        print(message)
    elif count == limit:
        print(f"Further `{key}` warnings are suppressed.")
    setattr(obj, attr, count + 1)


def _truncate_prompt_answer_ids(
    prompt_ids: List[int],
    answer_ids: List[int],
    *,
    max_length: int,
    image_token_id: int,
    num_images: int,
) -> Optional[tuple[List[int], List[int]]]:
    if max_length <= 1:
        raise ValueError(f"max_length must be > 1, got {max_length}")
    if not answer_ids:
        return None

    if len(prompt_ids) + len(answer_ids) <= max_length:
        return list(prompt_ids), list(answer_ids)

    min_prompt_tokens = 64 if num_images == 0 else 128
    min_prompt_tokens = min(min_prompt_tokens, len(prompt_ids), max_length - 1)
    answer_budget = max_length - min_prompt_tokens
    if answer_budget <= 0:
        return None

    kept_answer_ids = list(answer_ids[:answer_budget])
    prompt_budget = max_length - len(kept_answer_ids)
    if prompt_budget <= 0:
        return None

    if len(prompt_ids) <= prompt_budget:
        kept_prompt_ids = list(prompt_ids)
    elif num_images <= 0:
        kept_prompt_ids = list(prompt_ids[-prompt_budget:])
    else:
        image_positions = [
            idx for idx, token_id in enumerate(prompt_ids)
            if token_id == image_token_id
        ]
        if len(image_positions) != num_images:
            return None

        first_image_pos = image_positions[0]
        last_image_pos = image_positions[-1]
        image_span_len = last_image_pos - first_image_pos + 1
        if image_span_len > prompt_budget:
            return None

        # The image token is usually near the beginning of the user message,
        # while the assistant generation prompt is at the end. For very long
        # questions, keep both sides and drop the middle of the user text.
        if last_image_pos + 1 <= prompt_budget:
            first_span = (0, last_image_pos + 1)
        else:
            tail_reserve = min(64, max(0, prompt_budget - image_span_len))
            image_context_budget = prompt_budget - tail_reserve
            start = max(0, last_image_pos + 1 - image_context_budget)
            first_span = (start, last_image_pos + 1)

        tail_budget = prompt_budget - (first_span[1] - first_span[0])
        tail_start = max(first_span[1], len(prompt_ids) - tail_budget)
        spans = [first_span]
        if tail_budget > 0 and tail_start < len(prompt_ids):
            spans.append((tail_start, len(prompt_ids)))

        kept_prompt_ids = []
        for start, end in spans:
            if end <= start:
                continue
            kept_prompt_ids.extend(prompt_ids[start:end])
        if kept_prompt_ids.count(image_token_id) != num_images:
            return None

    return kept_prompt_ids, kept_answer_ids


def load_images_from_example(ex: Dict[str, Any]) -> List[Image.Image]:
    if "images" in ex and ex["images"] is not None:
        raw_images = ex["images"]
    else:
        raw_images = ex.get("image")

    if raw_images is None:
        return []

    if isinstance(raw_images, (list, tuple)):
        return [open_image(x) for x in raw_images]

    return [open_image(raw_images)]


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
        "image": PIL.Image | List[PIL.Image] | None,
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

            image = images[0] if len(images) == 1 else list(images)

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
    require_precomputed_vision_embeddings: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenizer = self.model.tokenizer

        flat_images = []
        flat_precomputed_features = []
        num_images_per_sample = []
        input_id_rows = []
        label_rows = []
        records = []

        for ex in features:
            if "question" not in ex or "answer" not in ex:
                continue

            question = ex["question"]
            answer = ex["answer"]

            precomputed_paths = ex.get("vision_embedding_paths")
            if isinstance(precomputed_paths, str):
                precomputed_paths = [precomputed_paths]
            if precomputed_paths is not None:
                precomputed_paths = [str(path) for path in precomputed_paths]

            usable_precomputed_paths = None
            if (
                getattr(self.model, "freeze_vision", False)
                and precomputed_paths
                and all(os.path.exists(path) for path in precomputed_paths)
            ):
                usable_precomputed_paths = precomputed_paths

            raw_images = ex["images"] if ex.get("images") is not None else ex.get("image")
            has_raw_images = (
                bool(raw_images)
                if isinstance(raw_images, (list, tuple))
                else raw_images is not None
            )
            if (
                self.require_precomputed_vision_embeddings
                and getattr(self.model, "freeze_vision", False)
                and has_raw_images
                and usable_precomputed_paths is None
            ):
                raise RuntimeError(
                    "Precomputed vision embeddings are required, but this sample "
                    "does not provide usable `vision_embedding_paths`. Add "
                    "`visual_encoder=VISION_PATH` to the corresponding dataset spec "
                    "and make sure embeddings exist under "
                    "`<dataset_root>/embeddings/<visual_encoder_name>/`."
                )

            images = None
            if usable_precomputed_paths is None:
                try:
                    images = load_images_from_example(ex)
                except Exception as e:
                    print(f"Skipping sample with unreadable image: {e}")
                    continue

            records.append(
                {
                    "example": ex,
                    "question": question,
                    "answer": answer,
                    "images": images,
                    "precomputed_paths": usable_precomputed_paths,
                }
            )

        has_images = any(
            bool(record["precomputed_paths"])
            or bool(record["images"])
            for record in records
        )
        use_precomputed = has_images and all(
            (not record["images"] and not record["precomputed_paths"])
            or bool(record["precomputed_paths"])
            for record in records
        )

        for record in records:
            question = record["question"]
            answer = record["answer"]
            precomputed_paths = record["precomputed_paths"]
            images = record["images"]
            if use_precomputed:
                num_images = len(precomputed_paths) if precomputed_paths else 0
            else:
                if images is None:
                    try:
                        images = load_images_from_example(record["example"])
                    except Exception as e:
                        print(f"Skipping sample with unreadable image: {e}")
                        continue
                num_images = len(images)

            if hasattr(self.model, "build_chat_training_texts"):
                prompt, full_text = self.model.build_chat_training_texts(
                    question=question,
                    answer=answer,
                    num_images=num_images,
                )
            else:
                prompt, full_text = build_prompt_and_full_text(
                    tokenizer,
                    question=question,
                    answer=answer,
                    num_images=num_images,
                )
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            full_ids = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            prefix_len = len(prompt_ids)
            if len(full_ids) < prefix_len or full_ids[:prefix_len] != prompt_ids:
                _warn_limited(
                    self,
                    "unsafe_label_mask",
                    "Skipping sample because prompt tokens are not a prefix of "
                    "full chat tokens. This would make label masking unsafe.",
                )
                continue
            answer_ids = full_ids[prefix_len:]
            truncated = _truncate_prompt_answer_ids(
                prompt_ids,
                answer_ids,
                max_length=self.max_length,
                image_token_id=getattr(self.model, "image_token_id", -1),
                num_images=num_images,
            )
            if truncated is None:
                _warn_limited(
                    self,
                    "no_supervised_tokens_after_truncation",
                    "Skipping sample with no supervised answer tokens after truncation. "
                    f"prompt_tokens={len(prompt_ids)}, answer_tokens={len(answer_ids)}, "
                    f"max_length={self.max_length}",
                )
                continue
            kept_prompt_ids, kept_answer_ids = truncated

            if use_precomputed and precomputed_paths:
                try:
                    loaded_features = [
                        load_precomputed_vision_feature(
                            path,
                            expected_vision_name=getattr(self.model, "vision_name", None),
                            expected_vision_backend=getattr(self.model, "vision_backend", None),
                        )
                        for path in precomputed_paths
                    ]
                except Exception as e:
                    print(f"Skipping sample with unreadable precomputed vision embedding: {e}")
                    continue
                flat_precomputed_features.extend(loaded_features)
            elif not use_precomputed and images:
                flat_images.extend(images)

            num_images_per_sample.append(num_images)
            input_ids_i = kept_prompt_ids + kept_answer_ids
            labels_i = [IGNORE_INDEX] * len(kept_prompt_ids) + kept_answer_ids
            input_id_rows.append(input_ids_i)
            label_rows.append(labels_i)

        if not input_id_rows:
            raise RuntimeError("All samples in the batch were filtered out during collation.")

        max_seq_len = max(len(input_ids_i) for input_ids_i in input_id_rows)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        input_ids = torch.full(
            (len(input_id_rows), max_seq_len),
            int(pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(input_id_rows), max_seq_len),
            dtype=torch.long,
        )
        labels = torch.full(
            (len(input_id_rows), max_seq_len),
            IGNORE_INDEX,
            dtype=torch.long,
        )
        for i, (input_ids_i, labels_i) in enumerate(zip(input_id_rows, label_rows)):
            seq_len = len(input_ids_i)
            input_ids[i, :seq_len] = torch.tensor(input_ids_i, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(labels_i, dtype=torch.long)

        vision_batch = {}
        if flat_precomputed_features:
            vision_batch = {
                "vision_precomputed_features": flat_precomputed_features,
            }
        elif flat_images:
            vision_batch = self.model.prepare_vision_inputs(flat_images)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "vision_num_images_per_sample": torch.tensor(
                num_images_per_sample,
                dtype=torch.long,
            ),
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
