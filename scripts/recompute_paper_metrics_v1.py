#!/usr/bin/env python3
"""Recompute all final metrics from raw responses and references; no GPU/deps."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_pure_source(name, relative):
    # Avoid package __init__ files that register datasets and import Torch.
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vqa_scoring = load_pure_source('_paper_vqa_scoring', 'src/eval/research_metrics.py')
dataset_name = 'src.dataset.research_detection_v1'
sentinel = object()
previous = sys.modules.get(dataset_name, sentinel)
try:
    sys.modules[dataset_name] = load_pure_source('_paper_box_validation', 'src/dataset/research_detection_v1.py')
    detection_scoring = load_pure_source('_paper_detection_scoring', 'src/eval/research_detection_v1.py')
finally:
    if previous is sentinel:
        sys.modules.pop(dataset_name, None)
    else:
        sys.modules[dataset_name] = previous
aggregate_scores, score_prediction = vqa_scoring.aggregate_scores, vqa_scoring.score_prediction
aggregate_detection, score_record = detection_scoring.aggregate, detection_scoring.score_record


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def index_records(records, expected_count):
    result = {row['record_id']: row for row in records}
    if len(result) != len(records) or len(result) != expected_count:
        raise ValueError('Duplicate, missing or unexpected evaluation records')
    return result


def rescore(references, predictions, detection=False):
    observed = index_records(predictions, len(references))
    if set(observed) != set(references):
        raise ValueError('Prediction/reference IDs differ')
    scored = []
    for row in predictions:
        record = references[row['record_id']]
        if detection:
            scored.append(score_record(record, row['prediction']))
        else:
            scored.append(dict(record, prediction=row['prediction'],
                               score=score_prediction(record, row['prediction'])))
    return aggregate_detection(scored) if detection else aggregate_scores(scored)


def verify_metrics(actual, original, label):
    """Check every recomputed metric, including group denominators, not just F1."""
    for key, value in actual.items():
        if key not in original:
            raise ValueError(f'{label}: original metric missing {key}')
        expected = original[key]
        if isinstance(value, dict):
            verify_metrics(value, expected, label+'.'+key)
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError(f'{label}: recomputed metric differs: {key}')
        elif value != expected:
            raise ValueError(f'{label}: recomputed field differs: {key}')


def recompute(bundle):
    bundle = Path(bundle).resolve(strict=True)
    specifications = read(bundle / 'reports/all_final_metrics.json')['rows']
    if len(specifications) != 18 or len({item['id'] for item in specifications}) != 18:
        raise ValueError('Exactly 18 distinct final variants required')
    refs = bundle / 'evaluation_manifests'
    vqa = index_records(rows(refs / 'internal_selection_116.jsonl'), 116)
    validation = index_records(rows(refs / 'detection_validation.jsonl'), 64)
    test = index_records(rows(refs / 'detection_test.jsonl'), 64)
    results = []
    for item in specifications:
        run = (bundle / item['run']).resolve(strict=True)
        if not run.is_relative_to(bundle):
            raise ValueError('Run path leaves the downloaded bundle')
        metrics = {}
        for name, records, is_detection in (
                ('before_internal116', vqa, False), ('after_internal116', vqa, False),
                ('after_detection_validation', validation, True),
                ('heldout_detection_test', test, True)):
            directory = run / 'evaluation'
            actual = rescore(records, rows(directory / (name+'.predictions.jsonl')), is_detection)
            verify_metrics(actual, read(directory / (name+'.metrics.json')), item['id']+'.'+name)
            metrics[name] = actual
        results.append(dict(id=item['id'], model=item['model'], vision=item['vision'], method=item['method'],
                            VQA_before=metrics['before_internal116']['group_macro_score'],
                            VQA_after=metrics['after_internal116']['group_macro_score'],
                            detection_val_F1_IoU50=metrics['after_detection_validation']['primary_score'],
                            detection_test_F1_IoU50=metrics['heldout_detection_test']['primary_score'],
                            negative_query_accuracy=metrics['heldout_detection_test']['negative_query_accuracy'],
                            metrics=metrics))
    return dict(status='all_18_variants_recomputed_and_matched', inference_rerun=False,
                method='Raw responses scored against ground truth; saved per-record scores ignored',
                official_benchmark_scores=False, rows=results)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = recompute(args.bundle)
    output = args.output or args.bundle / 'recomputed_metrics'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    table = [{key: value for key, value in row.items() if key != 'metrics'} for row in result['rows']]
    with (output / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(dict(status=result['status'], variants=len(table), output=str(output))))


if __name__ == '__main__':
    main()
