import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'paper_rescore', Path(__file__).resolve().parents[1] / 'scripts/recompute_paper_metrics_v1.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_rescoring_ignores_saved_score_and_uses_independent_references():
    records = {}
    predictions = []
    for index, group in enumerate(('general_vqa', 'ocr_documents', 'charts_diagrams', 'knowledge_reasoning')):
        record_id = str(index)
        records[record_id] = dict(record_id=record_id, task_group=group, scorer='exact_match', answers=['correct'])
        predictions.append(dict(record_id=record_id, prediction='wrong', answers=['wrong'], score=1))
    metrics = module.rescore(records, predictions)
    assert metrics['records'] == 4
    assert metrics['group_macro_score'] == 0
    with pytest.raises(ValueError, match='Duplicate'):
        module.rescore(records, predictions + [predictions[0]])


def test_changed_metric_or_denominator_is_rejected():
    with pytest.raises(ValueError, match='differs'):
        module.verify_metrics({'score': 25.0}, {'score': 30.0}, 'example')
    with pytest.raises(ValueError, match='differs'):
        module.verify_metrics({'groups': {'vqa': {'records': 29}}},
                              {'groups': {'vqa': {'records': 28}}}, 'example')
