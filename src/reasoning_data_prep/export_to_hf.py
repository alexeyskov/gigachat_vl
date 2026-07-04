import sys
sys.path.append("/home/ymayma/projects/vlm-research/gigachat_vl")


import argparse
import json
from pathlib import Path
from typing import Iterator, Optional

from datasets import Dataset, Features, Image, Value
from huggingface_hub import HfApi
from tqdm import tqdm

from src.reasoning_data_prep.settings import SETTINGS

"""
Converts one staged local reasoning dataset into Hugging Face friendly parquet shards.

Expected input layout:
    <input-dir>/
        records.jsonl
        images/
            *.png

Created output layout:
    <input-dir>/hf_export/
        train-00000.parquet
        train-00001.parquet
        ...

If --push-to-hub is used, the generated parquet files are uploaded to a folder
inside the target dataset repo with the same name as <input-dir>.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export staged reasoning records to HF-friendly parquet shards.",
    )
    parser.add_argument("--input-dir", required=False, default="/home/ymayma/projects/vlm-research/data/reasoning/mme_en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--export-batch-size", type=int, default=512)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--repo-type", default="dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_batch_size <= 0:
        raise ValueError("--export-batch-size must be positive")

    input_dir = Path(args.input_dir)
    records_path = input_dir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"records.jsonl not found: {records_path}")

    export_dir = input_dir / "hf_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    total_records = _count_records(records_path)
    shard_index = 0
    batch: list[dict] = []
    with tqdm(total=total_records, desc=f"Exporting {input_dir.name}", unit="record") as pbar:
        for row in _iter_rows(records_path, input_dir):
            batch.append(row)
            pbar.update(1)
            if len(batch) >= args.export_batch_size:
                _write_parquet_shard(
                    rows=batch,
                    export_dir=export_dir,
                    split=args.split,
                    shard_index=shard_index,
                )
                shard_index += 1
                batch = []

        if batch:
            _write_parquet_shard(
                rows=batch,
                export_dir=export_dir,
                split=args.split,
                shard_index=shard_index,
            )

    if args.push_to_hub:
        repo_id = args.repo_id or SETTINGS.HF_REPO_ID
        if repo_id is None:
            raise ValueError("HF_REPO_ID must be set in env or passed via --repo-id")
        _upload_export_dir(
            export_dir=export_dir,
            dataset_dir_name=input_dir.name,
            repo_id=repo_id,
            repo_type=args.repo_type,
        )


def _iter_rows(records_path: Path, input_dir: Path) -> Iterator[dict]:
    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            image_payload = _load_image_payload(record.get("image_path"), input_dir)

            yield {
                "record_id": record["record_id"],
                "source_dataset": record["source_dataset"],
                "source_sample_id": record["source_sample_id"],
                "image": image_payload,
                "question_ru": record["question_ru"],
                "reasoning_ru": record.get("reasoning_ru"),
                "answer_ru": record["answer_ru"],
                "metadata_json": json.dumps(
                    record.get("metadata", {}),
                    ensure_ascii=False,
                ),
            }


def _count_records(records_path: Path) -> int:
    total = 0
    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def _load_image_payload(
    image_path: Optional[str],
    input_dir: Path,
) -> Optional[dict[str, object]]:
    if not image_path:
        return None

    resolved = (input_dir / image_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Image file not found: {resolved}")

    return {
        "bytes": resolved.read_bytes(),
        "path": None,
    }


def _write_parquet_shard(
    rows: list[dict],
    export_dir: Path,
    split: str,
    shard_index: int,
) -> None:
    dataset = Dataset.from_list(rows, features=_features())
    shard_path = export_dir / f"{split}-{shard_index:05d}.parquet"
    dataset.to_parquet(str(shard_path))


def _features() -> Features:
    return Features(
        {
            "record_id": Value("string"),
            "source_dataset": Value("string"),
            "source_sample_id": Value("string"),
            "image": Image(),
            "question_ru": Value("string"),
            "reasoning_ru": Value("string"),
            "answer_ru": Value("string"),
            "metadata_json": Value("string"),
        }
    )


def _upload_export_dir(
    export_dir: Path,
    dataset_dir_name: str,
    repo_id: str,
    repo_type: str,
) -> None:
    if SETTINGS.HF_TOKEN is None:
        raise ValueError("HF_TOKEN is required to push export to hub")

    api = HfApi(token=SETTINGS.HF_TOKEN.get_secret_value())
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(export_dir),
        path_in_repo=dataset_dir_name,
    )


if __name__ == "__main__":
    main()
