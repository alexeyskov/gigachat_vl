"""CPU scorer contracts; all supplied responses/log likelihoods are synthetic."""
from copy import deepcopy
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('primary_text_metric_tests',ROOT/'src/eval/research_text_metrics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.mark.parametrize('response,value',[('Work.\n#### 1,000\n','1000'),('#### -3.50','-3.5'),
    ('A calculation gave 999.\n#### .25\n\n','0.25'),('#### 1e3','1000'),('#### 0\r\n','0')])
def test_gsm_exact_final_finite_decimal(response,value):
    assert m.gsm_final_decimal(response)==Decimal(value)


@pytest.mark.parametrize('response',['The answer is 42','Work:42','#### 42\nTrailing commentary',
    '#### 41\n#### 42','#### NaN','#### Infinity','#### -Infinity','#### one',' #### 42',
    '42 #### 42','####\n42','#### 42\n```','', '#### 42 dollars'])
def test_gsm_rejects_missing_invalid_ambiguous_or_nonfinal_marker(response):
    assert m.gsm_final_decimal(response) is None


def test_gsm_no_last_number_rescue():
    record={'record_id':'g','reference':{'numeric_answer':'42'}}
    assert m.score_gsm(record,'The answer is 42')['correct'] is False
    assert m.score_gsm(record,'Reasoning123.\n#### 42')['correct'] is True


def mmlu_raw():
    census={'prompt_token_ids_sha256':'a'*64,'candidate_continuation_token_lengths':dict.fromkeys(m.CHOICES,1)}
    raw={'candidate_log_likelihoods':dict.fromkeys(m.CHOICES,-1.),
        'candidate_token_ids':{c:[m.TOKEN_IDS['smallthinker4'][c]] for c in m.CHOICES},
        'candidate_prefix_token_ids_sha256':dict.fromkeys(m.CHOICES,'a'*64),'length_normalization':False}
    record={'record_id':'m','subject':'algebra','reference':{'answer_letter':'A'}}
    return record,raw,census


def test_mmlu_supplied_likelihood_ties_and_no_logits_invention():
    record,raw,census=mmlu_raw()
    original=deepcopy(raw)
    result=m.score_mmlu(record,raw,'smallthinker4',census)
    assert result['selected_choice']=='A' and result['correct'] and result['tie_choices']==list(m.CHOICES)
    assert raw==original
    raw['candidate_log_likelihoods']['D']=-.1
    assert m.score_mmlu(record,raw,'smallthinker4',census)['selected_choice']=='D'


@pytest.mark.parametrize('bad',['missing','extra','nan','inf','positive','bool','candidate_id','prefix','length_normalized','census_boundary'])
def test_mmlu_invalid_supplied_logits_or_boundaries_rejected(bad):
    record,raw,census=mmlu_raw()
    if bad=='missing':raw['candidate_log_likelihoods'].pop('A')
    elif bad=='extra':raw['candidate_log_likelihoods']['E']=-4
    elif bad in ('nan','inf','positive','bool'):raw['candidate_log_likelihoods']['A']={'nan':float('nan'),'inf':float('inf'),'positive':2.,'bool':False}[bad]
    elif bad=='candidate_id':raw['candidate_token_ids']['A']=[999]
    elif bad=='prefix':raw['candidate_prefix_token_ids_sha256']['A']='b'*64
    elif bad=='length_normalized':raw['length_normalization']=True
    else:census['candidate_continuation_token_lengths']['A']=2
    with pytest.raises(ValueError):m.score_mmlu(record,raw,'smallthinker4',census)


def test_mmlu_equal_subject_macro_differs_from_record_weighted_mean():
    rows=[{'subject':'small','correct':True}]+[{'subject':'large','correct':False} for _ in range(3)]
    result=m.aggregate_mmlu(rows,require_full_suite=False)
    assert result['subject_macro_accuracy']==.5 and result['all_record_accuracy']==.25
    with pytest.raises(ValueError):m.aggregate_mmlu(rows)


def test_ifeval_distinct_prompt_instruction_strict_loose_denominators():
    rows=[{'instruction_ids':['a','b'],'strict':{'follow_all_instructions':False,'follow_instruction_list':[True,False]},
        'loose':{'follow_all_instructions':True,'follow_instruction_list':[True,True]}},
        {'instruction_ids':['a'],'strict':{'follow_all_instructions':True,'follow_instruction_list':[True]},
        'loose':{'follow_all_instructions':True,'follow_instruction_list':[True]}}]
    result=m.aggregate_ifeval(rows)
    assert result['strict']['prompt_accuracy']==.5 and result['strict']['instruction_accuracy']==2/3
    assert result['loose']['prompt_accuracy']==result['loose']['instruction_accuracy']==1.
    rows[0]['strict']['follow_instruction_list']=[False]
    with pytest.raises(ValueError,match='skip'):m.aggregate_ifeval(rows)


def tiny_suite():
    records=[{'record_id':task,'task':task,'source':{'repo':'pinned','revision':'1'*40,'original_id':'key:1'},
        'input_text':'Text\u2028with Unicode separator','prompt':'Text prompt','question':'Text prompt',
        'reference':{'numeric_answer':'42'} if task=='gsm8k' else {'instruction_id_list':['x'],'kwargs':[{}]}}
        for task in ('gsm8k','ifeval')]
    rows=[{'record_id':r['record_id'],'prompt_sha256':'b'*64,'prompt_token_ids_sha256':'c'*64,'prompt_tokens':10} for r in records]
    generation={'gsm8k':{'max_new_tokens':1024,'do_sample':False},'ifeval':{'max_new_tokens':2048,'do_sample':False}}
    suite={'records':records,'protocol_sha256':'d'*64,'manifest_sha256':'e'*64,
        'protocol':{'evaluation':{'generation':generation}},
        'census':{'smallthinker4':{'revision':'f'*40,'records':rows}}}
    producer={'schema_version':1,'owner':'research_primary_text_responses_v1','suite_protocol_sha256':'d'*64,
        'manifest_sha256':'e'*64,'tasks':['gsm8k','ifeval'],'model_id':'smallthinker4','model_revision':'f'*40,
        'stage':'baseline_before_vision','numerical_profile':{'weights':'nf4','compute':'bf16'},
        'checkpoint_identity':{'pinned_source':'synthetic fixture, no model'},
        'evidence_scope':'synthetic_scorer_validation','generation':generation}
    responses=[{'record_id':r['record_id'],'task':r['task'],'source':r['source'],
        'input_text_sha256':m.sha(r['input_text']),'response_contract_sha256':m.object_sha(producer),
        'prompt_sha256':'b'*64,'prompt_token_ids_sha256':'c'*64,'prompt_tokens':10,
        'response':'#### 42' if r['task']=='gsm8k' else 'Synthetic response'} for r in records]
    return suite,responses,producer


@pytest.mark.parametrize('field',['task','source','input_text_sha256','response_contract_sha256','prompt_sha256','prompt_token_ids_sha256','prompt_tokens','reference'])
def test_raw_row_source_prompt_gold_or_contract_corruption_refused(field):
    suite,responses,producer=tiny_suite()
    responses[0][field]='changed'
    with pytest.raises(ValueError):m.validate_response_rows(suite,responses,producer,tasks=['gsm8k','ifeval'])


@pytest.mark.parametrize('field',['owner','manifest_sha256','suite_protocol_sha256','model_revision','model_id','stage','numerical_profile','checkpoint_identity','generation','tasks','evidence_scope'])
def test_raw_producer_scope_precision_and_provenance_corruption_refused(field):
    suite,responses,producer=tiny_suite();producer[field]='changed'
    with pytest.raises(ValueError):m.validate_response_rows(suite,responses,producer,tasks=['gsm8k','ifeval'])


@pytest.mark.parametrize('bad',['missing','duplicate','reorder','extra'])
def test_raw_response_ids_exact_coverage_order(bad):
    suite,responses,producer=tiny_suite()
    if bad=='missing':responses.pop()
    elif bad=='duplicate':responses[1]=deepcopy(responses[0])
    elif bad=='reorder':responses.reverse()
    else:responses.append(deepcopy(responses[0]))
    with pytest.raises(ValueError):m.validate_response_rows(suite,responses,producer,tasks=['gsm8k','ifeval'])


def test_missing_official_adapter_is_fatal_not_denominator_skip():
    suite,responses,producer=tiny_suite()
    with pytest.raises(RuntimeError,match='no instruction'):m.score_rows(suite,responses,producer,tasks=['gsm8k','ifeval'])


def test_scoring_supplied_synthetic_rows_preserves_raw_inputs_and_scope():
    class FakeOfficial:
        def score(self,record,response):
            assert response=='Synthetic response'
            return {'record_id':record['record_id'],'task':'ifeval','instruction_ids':['x'],
                'strict':{'follow_all_instructions':False,'follow_instruction_list':[False]},
                'loose':{'follow_all_instructions':True,'follow_instruction_list':[True]}}
    suite,responses,producer=tiny_suite();before=deepcopy(responses)
    rows,summary=m.score_rows(suite,responses,producer,tasks=['gsm8k','ifeval'],ifeval=FakeOfficial())
    assert responses==before and len(rows)==2 and summary['tasks']['gsm8k']['accuracy']==1
    assert summary['response_producer']['evidence_scope']=='synthetic_scorer_validation'
    assert not summary['model_generation_performed'] and not summary['model_logits_invented']
    assert not summary['primary_completion_claim'] and summary['cross_task_composite'] is None


def test_jsonl_preserves_unicode_line_separators(tmp_path):
    p=tmp_path/'unicode.jsonl';expected=[{'prompt':'A\u2028B\u0085C\u2029D'},{'response':'another'}]
    p.write_text(''.join(m.canonical(r)+'\n' for r in expected))
    assert m.read_jsonl(p)==expected


def test_actual_frozen_bundle_loads_without_model_code():
    before='torch' in sys.modules
    suite=m.load_suite(ROOT/'datasets/research/primary-text-en-v1')
    assert len(suite['records'])==1680 and len(suite['census'])==4
    assert ('torch' in sys.modules)==before


@pytest.mark.parametrize('bad',['protocol','member'])
def test_actual_bundle_corruption_rejected(tmp_path,bad):
    original=ROOT/'datasets/research/primary-text-en-v1/protocol.json'
    (tmp_path/'protocol.json').write_bytes(original.read_bytes() if bad=='member' else b'{}')
    (tmp_path/'gsm8k.jsonl').write_bytes(b'CORRUPT')
    with pytest.raises(ValueError):m.load_suite(tmp_path)


def checkpoint_descriptor(kind):
    members=[{'role':'state','relative_path':'state.pt','path':'/fixture/state.pt',
        'bytes':10,'mtime_ns':123,'sha256':'a'*64}]
    if kind=='bundle_v1':
        members[0]['role']='manifest'
        members.append({'role':'model','relative_path':'model_tensors.pt','path':'/fixture/model_tensors.pt',
            'bytes':20,'mtime_ns':124,'sha256':'b'*64})
    identity={'format':kind,'sha256':members[0]['sha256'],'primary_path':members[0]['path'],
        'primary_bytes':10,'primary_mtime_ns':123,'total_bytes':sum(r['bytes'] for r in members),
        'components_fully_hashed':True,'members':members}
    if kind=='bundle_v1':
        stable={'format':kind,'members':[{k:r[k] for k in ('role','relative_path','bytes','sha256')} for r in members]}
        identity['sha256']=m.sha(b'RESEARCH_CHECKPOINT_ARTIFACT_V1\0'+json.dumps(stable,sort_keys=True,separators=(',',':')).encode())
    return identity


def test_baseline_source_descriptor_must_be_actual_pinned_inventory():
    model={'repo':'pinned/model','revision':'f'*40}
    descriptor={'format':'pinned_baseline',**model,'weight_inventory_sha256':'a'*64}
    m.validate_checkpoint_descriptor(descriptor,'baseline_before_vision',model)
    with pytest.raises(ValueError):m.validate_checkpoint_descriptor({'REQUIRED':'placeholder'},'baseline_before_vision',model)
    descriptor['revision']='0'*40
    with pytest.raises(ValueError):m.validate_checkpoint_descriptor(descriptor,'baseline_before_vision',model)


@pytest.mark.parametrize('kind',['legacy_single_file','bundle_v1'])
def test_managed_checkpoint_descriptor_is_compound_but_no_weight_read(kind):
    descriptor=checkpoint_descriptor(kind)
    m.validate_checkpoint_descriptor(descriptor,'SFT',{},owner='fixture-owner',step=10)


@pytest.mark.parametrize('bad',['digest','total','member','duplicate','not_hashed','owner','step'])
def test_managed_checkpoint_descriptor_corruption_refused(bad):
    descriptor=checkpoint_descriptor('bundle_v1');owner='fixture-owner';step=10
    if bad=='digest':descriptor['sha256']='0'*64
    elif bad=='total':descriptor['total_bytes']+=1
    elif bad=='member':descriptor['members'][1]['sha256']='c'*64
    elif bad=='duplicate':descriptor['members'].append(dict(descriptor['members'][1]))
    elif bad=='not_hashed':descriptor['components_fully_hashed']=False
    elif bad=='owner':owner=None
    else:step=0
    with pytest.raises(ValueError):m.validate_checkpoint_descriptor(descriptor,'SFT',{},owner=owner,step=step)
