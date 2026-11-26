from collections import Counter
import pytest
from scripts.build_research_mix import build,WEIGHTS


def record(group,i):return {'record_id':f'{group}-{i}','task_group':group}


def test_finite_mix_deterministic_group_and_text_proportions():
    mm=[record(g,i) for g in WEIGHTS for i in range(25)]
    text=[record('text_instruction',i) for i in range(20)]
    chosen,audit=build(mm,text)
    assert chosen==build(mm,text)[0]
    counts=Counter(r['task_group'] for r in chosen)
    assert counts==dict(general_vqa=35,ocr_documents=30,charts_diagrams=20,knowledge_reasoning=15,text_instruction=18)
    assert audit['total_records']==118
    assert max(Counter(r['record_id'] for r in chosen).values())<=2


def test_missing_group_and_unbounded_oversampling_rejected():
    with pytest.raises(ValueError):build([record('general_vqa',i) for i in range(10)],[])
    mm=[record(g,i) for g in WEIGHTS for i in range(25)]
    with pytest.raises(ValueError):build(mm,[])
