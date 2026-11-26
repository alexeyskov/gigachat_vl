from scripts.audit_research_internal116_format_v1 import canonical_score
from src.eval.research_metrics import score_prediction
import pytest


def science():
    return dict(task_group='knowledge_reasoning',scorer='exact_match',answers=['Answer: C'],
        question='Question: Which?\nChoices:\nA. red\nB. green\nC. blue\nAnswer with the letter.',source={'config':'scienceqa'})


def test_explicit_letter_format_without_mining_explanations():
    row=science();assert score_prediction(row,'C')==0
    assert canonical_score(row,'C')[0]==1 and canonical_score(row,'Answer: C')[0]==1
    assert canonical_score(row,'After reasoning, Answer: C')[0]==0
    assert canonical_score(row,'D')[0]==0


def test_ambiguous_reference_fails_closed():
    row=science();row['answers']=['Answer: D']
    with pytest.raises(ValueError,match='reference'):canonical_score(row,'D')


def test_vqa_punctuation_only_and_unchanged_document_literal_metric():
    vqa=dict(task_group='general_vqa',scorer='exact_match',answers=['No.'])
    assert score_prediction(vqa,'no')==0 and canonical_score(vqa,'no')[0]==1
    assert canonical_score(vqa,'yes')[0]==0
    doc=dict(task_group='ocr_documents',scorer='anls',answers=['No.'])
    assert canonical_score(doc,'no')[0]==score_prediction(doc,'no')
