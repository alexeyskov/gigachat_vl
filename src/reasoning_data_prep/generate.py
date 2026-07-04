import argparse
import asyncio
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Optional

from src.reasoning_data_prep.registry import get_reasoning_dataset_config
from src.reasoning_data_prep.schemas import (
    PreparedReasoningRecord,
    ReasoningRequest,
)
from src.reasoning_data_prep.settings import SETTINGS
from src.reasoning_data_prep.teacher_client import TeacherClient

"""
Builds a local staged reasoning dataset for one source dataset.

Output layout under:
    <output-dir>/<dataset-name>/

Files created:
    records.jsonl
        Final staged records with question/reasoning/answer and a relative image path.
    images/
        Saved image files referenced by records.jsonl.
    state/completed_ids.txt
        Source sample ids already written successfully, used for resume.
    intermediate/errors.jsonl
        Failed source samples or failed teacher requests.

This script does not export to Hugging Face parquet. That is handled later by
export_to_hf.py.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Russian reasoning records from a source VLM dataset.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--retry-delay-seconds", type=float, default=3.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--teacher-timeout-seconds", type=float, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--max-source-shards", type=int, default=4)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    config = get_reasoning_dataset_config(args.dataset)

    output_dir = Path(args.output_dir) / config.name
    state_dir = output_dir / "state"
    intermediate_dir = output_dir / "intermediate"
    images_dir = output_dir / "images"
    state_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    completed_ids_path = state_dir / "completed_ids.txt"
    errors_path = intermediate_dir / "errors.jsonl"
    records_path = output_dir / "records.jsonl"
    completed_ids = _load_completed_ids(completed_ids_path)

    if args.download:
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required when --download is set")
        download_kwargs = {
            "dataset_root": args.dataset_root,
            "force_redownload": args.force_redownload,
            "hf_token": _settings_hf_token(),
        }
        if _supports_kwarg(config.download_func, "num_shards") and args.max_source_shards is not None:
            download_kwargs["num_shards"] = args.max_source_shards

        config.download_func(**download_kwargs)

    dataset = config.load_func(
        limit=None,
        dataset_root=args.dataset_root,
        seed=args.seed,
    )
    client = TeacherClient()
    pending_requests: list[ReasoningRequest] = []

    for sample in dataset:
        if _limit_reached(args.limit, completed_ids):
            break

        try:
            build_result = config.build_request_func(sample)
            source_sample_id = build_result.source_sample_id
            if source_sample_id in completed_ids:
                continue

            if isinstance(build_result, PreparedReasoningRecord):
                _write_record(
                    build_result,
                    records_path=records_path,
                    images_dir=images_dir,
                    completed_ids_path=completed_ids_path,
                    completed_ids=completed_ids,
                )
            elif isinstance(build_result, ReasoningRequest):
                pending_requests.append(build_result)
                remaining_slots = _remaining_slots(args.limit, completed_ids)
                should_flush = len(pending_requests) >= args.batch_size
                if remaining_slots is not None and len(pending_requests) >= remaining_slots:
                    should_flush = True

                if should_flush:
                    requests_to_flush = pending_requests
                    if remaining_slots is not None:
                        requests_to_flush = pending_requests[:remaining_slots]

                    await _flush_requests(
                        client=client,
                        requests=requests_to_flush,
                        records_path=records_path,
                        images_dir=images_dir,
                        errors_path=errors_path,
                        completed_ids_path=completed_ids_path,
                        completed_ids=completed_ids,
                        retry_delay_seconds=args.retry_delay_seconds,
                        max_retries=args.max_retries,
                    )
                    pending_requests.clear()

                    if _limit_reached(args.limit, completed_ids):
                        break
                continue
            else:
                raise TypeError(f"Unexpected build result: {type(build_result)}")
        except Exception as e:
            _append_jsonl(
                errors_path,
                {
                    "error": str(e),
                    "sample": _safe_jsonable(sample),
                },
            )

    if pending_requests and not _limit_reached(args.limit, completed_ids):
        requests_to_flush = pending_requests
        remaining_slots = _remaining_slots(args.limit, completed_ids)
        if remaining_slots is not None:
            requests_to_flush = pending_requests[:remaining_slots]

        await _flush_requests(
            client=client,
            requests=requests_to_flush,
            records_path=records_path,
            images_dir=images_dir,
            errors_path=errors_path,
            completed_ids_path=completed_ids_path,
            completed_ids=completed_ids,
            retry_delay_seconds=args.retry_delay_seconds,
            max_retries=args.max_retries,
        )


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _flush_requests(
    client: TeacherClient,
    requests: list[ReasoningRequest],
    records_path: Path,
    images_dir: Path,
    errors_path: Path,
    completed_ids_path: Path,
    completed_ids: set[str],
    retry_delay_seconds: float,
    max_retries: int,
) -> None:
    outputs = await client.generate_reasoning_batch(
        requests,
        retry_delay_seconds=retry_delay_seconds,
        max_retries=max_retries,
    )

    for request, output in zip(requests, outputs):
        if output is None:
            _append_jsonl(
                errors_path,
                {
                    "error": "Teacher request failed after retries",
                    "source_sample_id": request.source_sample_id,
                    "source_dataset": request.source_dataset,
                },
            )
            continue

        record = PreparedReasoningRecord(
            source_dataset=request.source_dataset,
            source_sample_id=request.source_sample_id,
            image=request.image,
            question_ru=output.question_ru,
            reasoning_ru=output.reasoning_ru,
            answer_ru=output.answer_ru,
            metadata=request.metadata,
        )
        _write_record(
            record,
            records_path=records_path,
            images_dir=images_dir,
            completed_ids_path=completed_ids_path,
            completed_ids=completed_ids,
        )


def _write_record(
    record: PreparedReasoningRecord,
    records_path: Path,
    images_dir: Path,
    completed_ids_path: Path,
    completed_ids: set[str],
) -> None:
    record_id = _build_record_id(record)
    image_rel_path = _save_record_image(
        record=record,
        record_id=record_id,
        images_dir=images_dir,
    )
    _append_jsonl(records_path, _record_to_jsonable(record, record_id, image_rel_path))
    _append_line(completed_ids_path, record.source_sample_id)
    completed_ids.add(record.source_sample_id)


def _save_record_image(
    record: PreparedReasoningRecord,
    record_id: str,
    images_dir: Path,
) -> Optional[str]:
    if record.image is None:
        return None

    image_path = images_dir / f"{record_id}.png"
    record.image.convert("RGB").save(image_path, format="PNG")
    return str(Path("images") / image_path.name)


def _build_record_id(record: PreparedReasoningRecord) -> str:
    payload = f"{record.source_dataset}:{record.source_sample_id}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _record_to_jsonable(
    record: PreparedReasoningRecord,
    record_id: str,
    image_path: Optional[str],
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "source_dataset": record.source_dataset,
        "source_sample_id": record.source_sample_id,
        "image_path": image_path,
        "question_ru": record.question_ru,
        "reasoning_ru": record.reasoning_ru,
        "answer_ru": record.answer_ru,
        "metadata": record.metadata,
    }


def _safe_jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return repr(value)


def _settings_hf_token() -> Optional[str]:
    if SETTINGS.HF_TOKEN is None:
        return None
    return SETTINGS.HF_TOKEN.get_secret_value()


def _remaining_slots(
    limit: Optional[int],
    completed_ids: set[str],
) -> Optional[int]:
    if limit is None:
        return None
    return max(limit - len(completed_ids), 0)


def _limit_reached(
    limit: Optional[int],
    completed_ids: set[str],
) -> bool:
    if limit is None:
        return False
    return len(completed_ids) >= limit


def _supports_kwarg(func: Any, arg_name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    return arg_name in signature.parameters


if __name__ == "__main__":
    asyncio.run(main())
