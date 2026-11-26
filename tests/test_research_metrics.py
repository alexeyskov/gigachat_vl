import pytest
from src.eval.research_metrics import normalize_vqa,vqa_accuracy,anls,chart_relaxed,mcq_answer,score_prediction,aggregate_scores


def test_vqa_consensus_not_simplified_rule():
    for count,expected in [(0,0),(1,.3),(2,.6),(3,.9),(4,1),(10,1)]:
        assert vqa_accuracy('yes',['yes']*count+['no']*(10-count))==pytest.approx(expected)
    assert vqa_accuracy('1',['one']*10)==1
    assert vqa_accuracy('1',['one']*10,protocol='vqav2_official_legacy')==0
    with pytest.raises(ValueError):vqa_accuracy('yes',['yes'])


def test_official_normalization_rules():
    assert normalize_vqa('The TWO, cats!')=='2 cats'
    assert normalize_vqa('1,000.25')=='1000.25'
    assert normalize_vqa('dont')=="don't"
    assert normalize_vqa('a black-and-white cat')=='black and white cat'


def test_anls_strict_half_distance_boundary_and_aliases():
    assert anls('bc',['ab'])==0
    assert anls('ac',['ab'])==0
    assert anls('abcd',['abce'])==.75
    assert anls('ABC',['other','abc'])==1


def test_chart_percent_and_zero_protocol():
    assert chart_relaxed('104',['100'])==1
    assert chart_relaxed('106',['100'])==0
    assert chart_relaxed('0.25',['25%'])==1
    assert chart_relaxed('nan',['25%'])==0
    assert chart_relaxed('0.0',['0'])==0
    assert chart_relaxed('0',['0'])==1


def test_mcq_does_not_mine_arbitrary_explanation():
    choices=['alpha','beta','gamma']
    assert mcq_answer('(B)',choices)=='B'
    assert mcq_answer('Answer: B',choices)=='B'
    assert mcq_answer('beta',choices)=='B'
    assert mcq_answer('A is wrong; B is correct',choices) is None
    assert mcq_answer('D',choices) is None
    assert score_prediction({'scorer':'mcq_accuracy','answers':['B'],'choices':choices},'B')==1


def test_group_macro_instead_of_record_micro_average():
    rows=[]
    for group,values in [('general_vqa',[1]*10),('ocr_documents',[0]),('charts_diagrams',[0]),('knowledge_reasoning',[0])]:
        for i,value in enumerate(values):rows.append({'record_id':group+str(i),'task_group':group,'scorer':'exact_match','score':value})
    assert aggregate_scores(rows)['group_macro_score']==25
    with pytest.raises(ValueError):aggregate_scores(rows[:10])
