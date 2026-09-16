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
