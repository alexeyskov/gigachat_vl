"""Synthetic units only; no datasets, images, model packages or GPU."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('primary_protocol_cpu',ROOT/'src/dataset/research_primary_protocol.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def source(config='vqav2'):
    return {'repo':'HuggingFaceM4/FineVision','revision':'3c380a731a3429c1d04693d6ec16d7e683def84c',
        'config':config,'split':'train','shard':config+'/train-00000.parquet'}
def row():return {'images':[{'fixture':'no image bytes'}],
    'texts':[{'user':'Question one?','assistant':'First answer'},{'user':'Follow up?','assistant':'Second answer'}]}
def independent(identity='row',image='a'*64):
    return p.normalise_finevision(row(),source_id='vqav2',source=source(),original_id=identity,image_hashes=[image])
def eligible(unit):
    return dict(unit,partition='train',component_id='component',audit_status=dict.fromkeys(p.AUDITS,True))


def test_independent_qa_never_inherits_sibling_answers():
    units=independent();assert len(units)==2 and all(u['kind']=='independent_qa' for u in units)
    records=p.select_exposures([eligible(u) for u in units],2)
    assert all(r['messages']==[{'role':'user','content':r['question']}] for r in records)
    assert all(len(u['turns'])==1 for u in units)


def test_genuine_llava_requires_order_verification_and_preserves_only_prior_history():
    kwargs={'source_id':'llava_instruct','source':source('LLaVA_Instruct_150K'),
        'original_id':'dialogue','image_hashes':['a'*64]}
    with pytest.raises(ValueError,match='dialogue provenance'):p.normalise_finevision(row(),**kwargs)
    unit=p.normalise_finevision(row(),dialogue_verified=True,**kwargs)[0]
    records=p.select_exposures([eligible(unit)],4)
    assert len(records)==4 and {r['turn_index'] for r in records}=={0,1}
    for r in records:
        assert r['sampling_unit_weight']==r['supervised_target_turns']==1
        assert r['messages'][-1]=={'role':'user','content':r['question']}
        if r['turn_index']==0:assert len(r['messages'])==1
        else:assert r['messages'][:-1]==[{'role':'user','content':'Question one?'},{'role':'assistant','content':'First answer'}]
    assert len({r['sampling_unit_id'] for r in records})==1


def test_dialogue_length_does_not_change_sampling_mass():
    args={'source_id':'llava_instruct','source':source('LLaVA_Instruct_150K'),'image_hashes':['b'*64],
        'dialogue_verified':True}
    short=p.normalise_finevision(dict(row(),texts=row()['texts'][:1]),original_id='short',**args)[0]
    long=p.normalise_finevision(dict(row(),texts=row()['texts']*5),original_id='long',**args)[0]
    records=p.select_exposures([eligible(short),eligible(long)],8)
    assert sum(r['sampling_unit_id']==short['unit_id'] for r in records)==4
    assert sum(r['sampling_unit_id']==long['unit_id'] for r in records)==4


def test_gqa_joins_exact_original_image_id_not_position():
    src={'repo':'lmms-lab/GQA','revision':'a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8',
        'config':'train_balanced_instructions','image_config':'train_balanced_images','split':'train'}
    instruction={'id':'q17','imageId':'image91','question':'What?','answer':'Yes','isBalanced':True}
    unit=p.normalise_gqa(instruction,{'id':'image91'},source=src,question_id='q17',image_hashes=['a'*64])
    assert unit['original_image_id']=='image91'
    with pytest.raises(ValueError,match='join'):p.normalise_gqa(instruction,{'id':'image90'},source=src,question_id='q17',image_hashes=['a'*64])
    with pytest.raises(ValueError,match='TRAIN'):p.normalise_gqa(instruction,{'id':'image91'},source=dict(src,split='testdev'),question_id='q17',image_hashes=['a'*64])


def test_document_alias_requires_original_document_mapping():
    with pytest.raises(ValueError,match='document component'):
        p.normalise_finevision(row(),source_id='docvqa',source=source('docvqa'),original_id='page1',image_hashes=['a'*64])


def test_split_transitively_groups_rgb_document_dialogue_before_flatten():
    a=independent('a','a'*64)[0];b=independent('b','a'*64)[0];c=independent('c','c'*64)[0]
    b['document_ids']=['original/document9'];c['document_ids']=['original/document9']
    values=p.split_components([a,b,c]);assert len({v['component_id'] for v in values.values()})==1
    assert len({v['partition'] for v in values.values()})==1
    assert values==p.split_components([c,b,a])


def test_science_equal_source_capacity_counterexample_and_proportional_fix():
    assert 6149*4<37500
    q=p.proportional_quotas({'scienceqa':6149,'aokvqa':20000},75000)
    assert sum(q.values())==75000 and q['scienceqa']<=24596
    assert q==p.proportional_quotas({'aokvqa':20000,'scienceqa':6149},75000)


def test_all_fixed_groups_and_declared_sources_are_required():
    counts={s:100000 for v in p.GROUPS.values() for s in v['sources']}
    quotas=p.primary_quotas(counts)
    assert [v['exposures'] for v in quotas.values()]==[175000,150000,100000,75000]
    assert sum(sum(v['sources'].values()) for v in quotas.values())==500000
    counts.pop('gqa')
    with pytest.raises(ValueError,match='direct GQA'):p.primary_quotas(counts)


@pytest.mark.parametrize('counts,target',[({'source':1},5),({'scienceqa':0},1),({'a':1,'b':100000},1)])
def test_pool_capacity_and_source_coverage_fail_closed(counts,target):
    with pytest.raises(ValueError):p.proportional_quotas(counts,target)


def test_sampling_deterministic_capped_and_never_samples_validation():
    units=[eligible(u) for u in independent()]
    records=p.select_exposures(units,8)
    assert records==p.select_exposures(list(reversed(units)),8)
    assert max(r['unit_occurrence'] for r in records)==3
    with pytest.raises(ValueError):p.select_exposures(units,9)
    with pytest.raises(ValueError,match='audit'):p.select_exposures([dict(units[0],partition='validation')],1)


def test_duplicate_normalized_content_under_different_original_ids_refused():
    first=eligible(independent('original-one')[0]);second=eligible(independent('original-two')[0])
    assert first['unit_id']!=second['unit_id'] and first['dedupe_key']==second['dedupe_key']
    with pytest.raises(ValueError,match='normalized content'):p.select_exposures([first,second],2)


@pytest.mark.parametrize('operation',['split','sample'])
@pytest.mark.parametrize('bad_seed',[123,42.0,'42'])
def test_data_seed_fixed_across_model_initialization_seeds(operation,bad_seed):
    units=[eligible(independent()[0])]
    with pytest.raises(ValueError,match='seed42'):
        if operation=='split':p.split_components(units,seed=bad_seed)
        else:p.select_exposures(units,1,seed=bad_seed)


@pytest.mark.parametrize('kind',['image','revision'])
def test_source_schema_hashes_are_hexadecimal_and_revision_pinned(kind):
    src=source();image='a'*64
    if kind=='image':image='Z'*64
    else:src['revision']='unversioned-main'
    with pytest.raises(ValueError):p.normalise_finevision(row(),source_id='vqav2',source=src,
        original_id='malformed fixture',image_hashes=[image])


@pytest.mark.parametrize('audit',p.AUDITS)
def test_every_preregistered_audit_is_mandatory(audit):
    u=eligible(independent()[0]);u['audit_status'][audit]=False
    with pytest.raises(ValueError,match='audit'):p.select_exposures([u],1)


def test_no_torch_import_and_exact_budget_definitions():
    import sys
    assert 'torch' not in sys.modules
    policy=p.policy()
    assert policy['sft_text_train_exposures']==88236
    assert policy['A0_train_exposures']==200000 and policy['A1_train_exposures']==10000
    assert policy['A1_additional_unique_images']==0 and not policy['primary_manifest_exists']
