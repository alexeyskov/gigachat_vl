import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from datasets import interleave_datasets, load_dataset
from datasets import IterableDataset as HFDataset
from huggingface_hub import login, snapshot_download
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.dataset.dataset_base import DatasetConfig


SMOLTALK2_REPO_ID = "HuggingFaceTB/smoltalk2"


def download_smoltalk2(
    dataset_root: str,
    force_redownload: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    dataset_root_path = Path(dataset_root)
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    if hf_token:
        login(token=hf_token)
    else:
        login()

    if not any(dataset_root_path.iterdir()) or force_redownload:
        snapshot_download(
            repo_id=SMOLTALK2_REPO_ID,
            repo_type="dataset",
            local_dir=str(dataset_root_path),
            local_dir_use_symlinks=False,
            force_download=force_redownload,
            resume_download=True,
        )


def _as_split_list(splits: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if splits is None:
        return None
    if isinstance(splits, str):
        return [splits]
    return [str(split) for split in splits]


def _interleave_if_needed(ds: Any, seed: int) -> HFDataset:
    if isinstance(ds, HFDataset):
        return ds

    if isinstance(ds, list):
        if not ds:
            raise ValueError("SmolTalk2 split list resolved to no datasets.")
        if len(ds) == 1:
            return ds[0]
        return interleave_datasets(ds, seed=seed, stopping_strategy="all_exhausted")

    if hasattr(ds, "values"):
        streams = list(ds.values())
        if not streams:
            raise ValueError("SmolTalk2 dataset dict has no splits.")
        if len(streams) == 1:
            return streams[0]
        return interleave_datasets(streams, seed=seed, stopping_strategy="all_exhausted")

    raise TypeError(f"Unsupported SmolTalk2 dataset object: {type(ds)}")


def _find_local_parquet_files(
    dataset_root: Path,
    subset: str,
    splits: Optional[List[str]],
) -> List[str]:
    parquet_files = sorted(dataset_root.rglob("*.parquet"))
    if not parquet_files:
        return []

    subset_lower = subset.lower()
    filtered = [
        path
        for path in parquet_files
        if subset_lower in "/".join(part.lower() for part in path.parts)
    ]
    if not filtered:
        filtered = parquet_files

    if splits:
        split_lowers = [split.lower() for split in splits]
        split_filtered = [
            path
            for path in filtered
            if any(split in path.name.lower() or split in str(path).lower() for split in split_lowers)
        ]
        if split_filtered:
            filtered = split_filtered

    return [str(path) for path in filtered]


def load_smoltalk2(
    config: Optional[DatasetConfig] = None,
    limit: Optional[int] = None,
    shuffle_buffer: int = 10000,
    seed: int = 42,
    dataset_root: Optional[str] = None,
    subset: str = "SFT",
    splits: Optional[Union[str, Sequence[str]]] = None,
) -> HFDataset:
    split_list = _as_split_list(splits)

    if dataset_root is not None:
        parquet_files = _find_local_parquet_files(
            dataset_root=Path(dataset_root),
            subset=subset,
            splits=split_list,
        )
        if parquet_files:
            ds = load_dataset(
                "parquet",
                data_files=parquet_files,
                streaming=True,
                split="train",
            )
        else:
            ds = load_dataset(
                SMOLTALK2_REPO_ID,
                subset,
                split=split_list if split_list is not None else None,
                streaming=True,
            )
    else:
        ds = load_dataset(
            SMOLTALK2_REPO_ID,
            subset,
            split=split_list if split_list is not None else None,
            streaming=True,
        )

    ds = _interleave_if_needed(ds, seed=seed)

    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if limit is not None:
        ds = ds.take(limit)

    return ds


def load_smoltalk2_sft(**kwargs) -> HFDataset:
    return load_smoltalk2(subset="SFT", **kwargs)


def load_smoltalk2_mid(**kwargs) -> HFDataset:
    return load_smoltalk2(subset="Mid", **kwargs)


def load_smoltalk2_preference(**kwargs) -> HFDataset:
    return load_smoltalk2(subset="Preference", **kwargs)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {None, "text"}:
                    parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return str(content).strip()


def _normalize_role(role: Any) -> str:
    role = str(role or "").strip().lower()
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "bot": "assistant",
        "model": "assistant",
    }
    return role_map.get(role, role)


def _extract_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    candidates = [
        row.get("messages"),
        row.get("chosen"),
        row.get("chosen_messages"),
        row.get("conversation"),
        row.get("conversations"),
    ]

    raw_messages = next(
        (candidate for candidate in candidates if isinstance(candidate, list)),
        None,
    )

    if raw_messages is None:
        prompt = row.get("prompt") or row.get("instruction") or row.get("question")
        response = row.get("response") or row.get("answer") or row.get("output")
        if prompt is not None and response is not None:
            raw_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]

    if not isinstance(raw_messages, list):
        return None

    messages = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue

        role = _normalize_role(message.get("role", message.get("from")))
        content = _content_to_text(message.get("content", message.get("value")))
        if not role or not content:
            continue

        messages.append({"role": role, "content": content})

    return messages or None


def _strip_thinking_trace(text: str) -> str:
    text = str(text)
    text = re.sub(
        r"<think\b[^>]*>.*?</think\s*>\s*",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"<think\b[^>]*>.*$",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"</think\s*>\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _format_context(messages: List[Dict[str, str]]) -> str:
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"].strip()

    labels = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
        "tool": "Tool",
        "developer": "Developer",
    }
    lines = []
    for message in messages:
        label = labels.get(message["role"], message["role"].title())
        lines.append(f"{label}: {message['content']}")
    return "\n".join(lines).strip()


class SmolTalk2IterableDataset(TorchIterableDataset):
    """
    Converts HuggingFaceTB/smoltalk2 rows into the unified training format:
        {"image": None, "question": str, "answer": str}

    SmolTalk2 uses OpenAI-style `messages`: [{"role": ..., "content": ...}, ...].
    For multi-turn rows, each assistant message can become a separate sample:
    previous turns are placed in the prompt context, and the current assistant
    message is used as the supervised answer.
    """

    def __init__(
        self,
        raw_hf_iterable,
        dataset_root: Optional[str] = None,
        seed: int = 42,
        skip_missing_images: bool = True,
        emit_all_assistant_turns: bool = True,
        include_system_prompt: bool = True,
        strip_thinking: bool = False,
        max_context_messages: Optional[int] = None,
    ):
        self.raw_hf_iterable = raw_hf_iterable
        self.emit_all_assistant_turns = emit_all_assistant_turns
        self.include_system_prompt = include_system_prompt
        self.strip_thinking = strip_thinking
        self.max_context_messages = max_context_messages

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for row in self.raw_hf_iterable:
            if not isinstance(row, dict):
                continue

            messages = _extract_messages(row)
            if not messages:
                continue

            assistant_indices = [
                idx
                for idx, message in enumerate(messages)
                if message["role"] == "assistant" and message["content"].strip()
            ]
            if not assistant_indices:
                continue
            if not self.emit_all_assistant_turns:
                assistant_indices = assistant_indices[-1:]

            for assistant_idx in assistant_indices:
                context_messages = messages[:assistant_idx]
                if not self.include_system_prompt:
                    context_messages = [
                        message
                        for message in context_messages
                        if message["role"] != "system"
                    ]

                if not any(message["role"] == "user" for message in context_messages):
                    continue

                if self.max_context_messages is not None:
                    context_messages = context_messages[-self.max_context_messages :]

                question = _format_context(context_messages)
                answer = messages[assistant_idx]["content"].strip()
                if self.strip_thinking:
                    answer = _strip_thinking_trace(answer)

                if not question or not answer:
                    continue

                yield {
                    "image": None,
                    "question": question,
                    "answer": answer,
                }
