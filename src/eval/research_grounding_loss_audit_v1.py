"""Read-only answer-token diagnostics; never changes a training objective."""
from __future__ import annotations

import json
import re
from collections import Counter


def answer_character_fields(answer: str) -> list[str]:
    """Mark JSON digits and labels, excluding numeric-looking label strings."""
    from src.eval.research_detection_v1 import parse_prediction

    objects, error = parse_prediction(answer)
    if error:
        raise ValueError(error)
    fields = ['format'] * len(answer)
    boxes = list(re.finditer(r'"bbox_2d"\s*:\s*\[([^\]]*)\]', answer))
    labels = list(re.finditer(r'"label"\s*:\s*"((?:[^"\\]|\\.)*)"', answer))
    if len(boxes) != len(objects) or len(labels) != len(objects):
        raise ValueError('JSON fields cannot be aligned exactly')
    for match in boxes:
        numbers = list(re.finditer(r'\d+', match.group(1)))
        if len(numbers) != 4:
            raise ValueError('Four integer coordinate spans required')
        for number in numbers:
            start = match.start(1) + number.start()
            end = match.start(1) + number.end()
            fields[start:end] = ['coordinate'] * (end - start)
    for match in labels:
        fields[match.start(1):match.end(1)] = ['label'] * len(match.group(1))
    return fields


def token_fields(answer: str, assistant_suffix: str, offsets) -> dict:
    """Classify offset spans without treating token fractions as gradients."""
    if not assistant_suffix.startswith(answer):
        raise ValueError('Native assistant suffix must preserve the entire JSON')
    fields = answer_character_fields(answer)
    fields += ['terminator'] * (len(assistant_suffix) - len(answer))
    counts = Counter()
    coordinate_overlap = 0
    covered = set()
    for start, end in offsets:
        if not 0 <= start <= end <= len(assistant_suffix):
            raise ValueError('Tokenizer offsets exceed the native assistant suffix')
        kinds = set(fields[start:end])
        counts[next(iter(kinds)) if len(kinds) == 1 else 'mixed' if kinds else 'zero_offset_special'] += 1
        if 'coordinate' in kinds:
            coordinate_overlap += 1
        covered.update(range(start, end))
    missing = [i for i, kind in enumerate(fields) if kind == 'coordinate' and i not in covered]
    if missing:
        raise ValueError('Coordinate digits lost from supervised token offsets')
    return dict(tokens=len(offsets), fields=dict(counts), coordinate_overlap_tokens=coordinate_overlap,
                coordinate_digits=sum(f == 'coordinate' for f in fields), all_coordinate_digits_covered=True)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('Nonempty token audit required')
    counts = Counter()
    for row in rows:
        counts.update(row['fields'])
    total = sum(row['tokens'] for row in rows)
    overlap = sum(row['coordinate_overlap_tokens'] for row in rows)
    return dict(records=len(rows), assistant_tokens=total, token_fields=dict(counts),
                coordinate_overlap_tokens=overlap, coordinate_token_fraction=overlap / total,
                mean_answer_tokens=total / len(rows), max_answer_tokens=max(row['tokens'] for row in rows),
                all_coordinate_digits_covered=all(row['all_coordinate_digits_covered'] for row in rows),
                scope='Target token counts only; neither measured CE components nor gradient contributions')
