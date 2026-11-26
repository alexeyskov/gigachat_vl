"""Synthetic stdlib-only semantic dedup, graph and receipt interfaces."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('independent_primary_v2_tests',ROOT/'src/dataset/research_primary_protocol_v2.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
REGISTRY='a'*64


def unit(identity='independent',*,source='vqav2',turns=None,image='b'*64,documents=()):
    config=p.v1.FINEVISION_CONFIGS[source]
    turns=turns or [('Same question?','Same answer')]
    return p.normalise_finevision({'images':[{}],'texts':[{'user':q,'assistant':a} for q,a in turns]},
        source_id=source,source={'repo':'HuggingFaceM4/FineVision','revision':'3c380a731a3429c1d04693d6ec16d7e683def84c',
            'config':config,'split':'train'},original_id=identity,image_hashes=[image],
        document_ids=documents,dialogue_verified=source=='llava_instruct')[0]


def dedup(values,**kwargs):return p.deduplicate_targets(values,source_registry_sha256=REGISTRY,**kwargs)
def qualify(values,receipt):
    # Synthetic receipt descriptors ONLY, never actual source audit evidence.
    graph=p.seal_component_graph(values)
    audits={a:{'status':'passed','receipt_sha256':'d'*64,'joint_pool_sha256':receipt['joint_pool_sha256'],
        'joint_graph_sha256':graph['joint_graph_sha256'],'source_registry_sha256':REGISTRY} for a in p.AUDITS}
    eligible=p.bind_external_eligibility(values,receipt,graph,audit_receipts=audits)
    result=copy.deepcopy(values)
    for u in result:u['audit_status']=dict.fromkeys(p.AUDITS,True)
    return result,eligible


def test_equivalent_one_turn_dialogue_and_independent_qa_one_owner():
    a=unit();b=unit('dialogue',source='llava_instruct')
    assert p.emitted_target(a,0)==p.emitted_target(b,0)
    assert a['emitted_context_target_keys']==b['emitted_context_target_keys']
    assert a['dedupe_key']==b['dedupe_key']
    values,receipt=dedup([a,b])
    assert sum(len(u['eligible_turn_indices']) for u in values)==1
    assert receipt==dedup([b,a])[1]
    assert len(values)==2 # loser graph links/provenance never silently dropped


def test_multiturn_first_target_duplicate_retains_other_target_and_original_history():
    dialogue=unit('dialogue',source='llava_instruct',turns=[('Same question?','Same answer'),('Then what?','Second answer')])
    # Choose a deterministic fixture ID whose independent first target wins.
    for index in range(100):
        a=unit(str(index));values,receipt=dedup([a,dialogue])
        d=next(u for u in values if u['unit_id']==dialogue['unit_id'])
        if d['eligible_turn_indices']==[1]:break
    else:raise AssertionError('No synthetic winner fixture')
    eligible,binding=qualify(values,receipt);d=next(u for u in eligible if u['unit_id']==dialogue['unit_id'])
    records=p.select_exposures([d],4,target_receipt=receipt,eligibility_receipt=binding)
    assert len(records)==4 and all(r['turn_index']==1 for r in records)
    assert records[0]['messages'][:-1]==[{'role':'user','content':'Same question?'},{'role':'assistant','content':'Same answer'}]
    assert all(r['eligible_original_turn_indices']==[1] for r in records)


def test_same_current_qa_with_genuine_prior_history_is_distinct():
    a=unit();b=unit('dialogue',source='llava_instruct',turns=[('Earlier question?','Earlier answer'),('Same question?','Same answer')])
    assert a['emitted_context_target_keys'][0]!=b['emitted_context_target_keys'][1]
    values,receipt=dedup([a,b])
    assert sum(len(u['eligible_turn_indices']) for u in values)==3


def test_document_metadata_does_not_hide_same_rgb_model_input_or_merge_different_pages():
    a=unit('page_a',source='docvqa',documents=('doc-a',))
    b=unit('page_b',source='docvqa',documents=('doc-b',))
    c=unit('page_c',source='docvqa',documents=('doc-a',),image='e'*64)
    values,receipt=dedup([a,b,c])
    assert a['emitted_context_target_keys']==b['emitted_context_target_keys']
    assert a['emitted_context_target_keys']!=c['emitted_context_target_keys']
    assert sum(len(u['eligible_turn_indices']) for u in values)==2
    assert len({v['component_id'] for v in p.split_components(values).values()})==1


@pytest.mark.parametrize('operation',['split','dedup','sample'])
@pytest.mark.parametrize('mutation',['question','answer','image','document','source','exact_key','semantic_key'])
def test_mutated_content_or_stale_keys_refused_every_entrypoint(operation,mutation):
    values,receipt=dedup([unit()]);values,binding=qualify(values,receipt);u=values[0]
    if mutation in ('question','answer'):u['turns'][0]['user' if mutation=='question' else 'assistant']+=' changed'
    if mutation=='image':u['image_hashes']=['e'*64]
    if mutation=='document':u['document_ids']=['different-document']
    if mutation=='source':u['source']['revision']='f'*40
    if mutation=='exact_key':u['emitted_context_target_exact_sha256'][0]='e'*64
    if mutation=='semantic_key':u['emitted_context_target_keys'][0]='e'*64
    with pytest.raises(ValueError):
        if operation=='split':p.split_components(values)
        elif operation=='dedup':dedup(values)
        else:p.select_exposures(values,1,target_receipt=receipt,eligibility_receipt=binding)


def test_rebinding_changed_content_resets_owner_receipt_compatibility():
    values,receipt=dedup([unit()]);values,binding=qualify(values,receipt)
    changed=unit(turns=[('Changed?','Changed answer')])
    changed.update(eligible_turn_indices=[0],target_ownership_receipt_sha256=receipt['sha256'],audit_status=dict.fromkeys(p.AUDITS,True))
    with pytest.raises(ValueError,match='receipt differs'):
        p.select_exposures([changed],1,target_receipt=receipt,eligibility_receipt=binding)


def test_complete_rgb_document_external_text_graph_before_holdout_including_loser():
    a=unit();b=unit('dialogue',source='llava_instruct');values,receipt=dedup([a,b])
    caption=p.bind_component_node(source_id='pixmo_repack',source={'repo':'dnth/pixmo-cap-images','revision':'a'*40,'split':'train'},
        original_id='caption',content={'caption':'Exact caption'},image_hashes=['b'*64],conversation_key='d'*64)
    text=p.bind_component_node(source_id='smoltalk',source={'repo':'HuggingFaceTB/smoltalk','revision':'a'*40,'split':'train'},
        original_id='text',content={'messages':[{'role':'system','content':'Exact task'},{'role':'user','content':'Question'}]},conversation_key='d'*64)
    split=p.split_components(values,component_nodes=[caption,text])
    assert len(split)==4 and len({v['component_id'] for v in split.values()})==1
    assert split==p.split_components(list(reversed(values)),component_nodes=[text,caption])
    text['content']['messages'][0]['content']='Changed task'
    with pytest.raises(ValueError,match='changed'):p.split_components(values,component_nodes=[caption,text])


def test_global_ownership_unit_and_turn_maps_are_not_mutable():
    values,receipt=dedup([unit()]);values,binding=qualify(values,receipt)
    values[0]['eligible_turn_indices']=[]
    with pytest.raises(ValueError,match='owned turn'):p.select_exposures(values,1,target_receipt=receipt,eligibility_receipt=binding)
    values,receipt=dedup([unit()]);receipt['target_owners']={}
    with pytest.raises(ValueError,match='Immutable receipt'):p.verify_target_receipt(values,receipt)


def test_global_pool_receipt_not_interchangeable_with_per_source_receipt():
    a=unit();b=unit('other',image='e'*64);values,receipt=dedup([a,b])
    with pytest.raises(ValueError):p.verify_target_receipt(values[:1],receipt,full_pool=True)
    _,other=dedup([a]);bad=copy.deepcopy(values);bad[0]['target_ownership_receipt_sha256']=other['sha256']
    with pytest.raises(ValueError):p.verify_target_receipt(bad,receipt)


@pytest.mark.parametrize('audit',p.AUDITS)
def test_all_external_audit_receipts_must_bind_pool_and_registry(audit):
    values,receipt=dedup([unit()]);graph=p.seal_component_graph(values)
    audits={a:{'status':'passed','receipt_sha256':'d'*64,'joint_pool_sha256':receipt['joint_pool_sha256'],
        'joint_graph_sha256':graph['joint_graph_sha256'],'source_registry_sha256':REGISTRY} for a in p.AUDITS}
    audits[audit]['source_registry_sha256']='e'*64
    with pytest.raises(ValueError,match='audit receipt missing'):p.bind_external_eligibility(values,receipt,graph,audit_receipts=audits)


def test_rebound_external_text_changes_graph_and_rejects_old_audit_scope_even_same_split_ids():
    values,receipt=dedup([unit()])
    def text(content):
        return p.bind_component_node(source_id='smoltalk',source={'repo':'HuggingFaceTB/smoltalk','revision':'a'*40,'split':'train'},
            original_id='same-original-id',content={'messages':[{'role':'system','content':content}]},conversation_key='d'*64)
    old=text('Original essential system task');new=text('Changed essential system task')
    before=p.seal_component_graph(values,component_nodes=[old]);after=p.seal_component_graph(values,component_nodes=[new])
    assert old['unit_id']==new['unit_id'] and before['assignments']==after['assignments']
    assert before['joint_graph_sha256']!=after['joint_graph_sha256']
    audits={a:{'status':'passed','receipt_sha256':'d'*64,'joint_pool_sha256':receipt['joint_pool_sha256'],
        'joint_graph_sha256':before['joint_graph_sha256'],'source_registry_sha256':REGISTRY} for a in p.AUDITS}
    with pytest.raises(ValueError,match='audit receipt missing'):
        p.bind_external_eligibility(values,receipt,after,component_nodes=[new],audit_receipts=audits)
    with pytest.raises(ValueError,match='graph/source/content'):
        p.bind_external_eligibility(values,receipt,before,component_nodes=[new],audit_receipts=audits)
    bound=p.bind_external_eligibility(values,receipt,before,component_nodes=[old],audit_receipts=audits)
    assert bound['joint_graph_sha256']==before['joint_graph_sha256'] and not bound['primary_training_authorized']


def test_sampling_requires_receipts_preserves_unit_mass_cap_and_input():
    a=unit('short',source='llava_instruct',turns=[('short','answer')])
    b=unit('long',source='llava_instruct',turns=[('one','a'),('two','b'),('three','c')])
    original=copy.deepcopy([a,b]);values,receipt=dedup([a,b]);values,binding=qualify(values,receipt)
    records=p.select_exposures(values,8,target_receipt=receipt,eligibility_receipt=binding)
    assert [a,b]==original and len(records)==8
    assert {sum(r['sampling_unit_id']==u['unit_id'] for r in records) for u in values}=={4}
    assert records==p.select_exposures(list(reversed(values)),8,target_receipt=receipt,eligibility_receipt=binding)
    with pytest.raises(ValueError,match='cap4'):p.select_exposures(values,9,target_receipt=receipt,eligibility_receipt=binding)
    with pytest.raises(TypeError):p.select_exposures(values,1)


def test_external_descriptors_do_not_silently_complete_candidate_audits():
    values,receipt=dedup([unit()]);_,binding=qualify(values,receipt)
    assert all(flag is False for flag in values[0]['audit_status'].values())
    with pytest.raises(ValueError,match='audit binding'):
        p.select_exposures(values,1,target_receipt=receipt,eligibility_receipt=binding)


@pytest.mark.parametrize('field',['partition','component_id'])
def test_resealed_eligibility_projection_cannot_override_immutable_graph(field):
    chosen=unit()
    if field=='partition':
        for index in range(1000):
            candidate=unit('validation-fixture-'+str(index))
            if p.split_components([candidate])[candidate['unit_id']]['partition']=='validation':chosen=candidate;break
        else:raise AssertionError('No deterministic validation fixture')
    values,receipt=dedup([chosen]);values,binding=qualify(values,receipt)
    altered=copy.deepcopy(binding);altered['assignments'][0][field]='train' if field=='partition' else 'e'*64
    altered=p._sealed({k:v for k,v in altered.items() if k!='sha256'})
    with pytest.raises(ValueError,match='component assignment'):
        p.select_exposures(values,1,target_receipt=receipt,eligibility_receipt=altered)


@pytest.mark.parametrize('seed',[123,42.0,'42',True])
def test_seed_strict_all_v2_operations(seed):
    u=unit()
    with pytest.raises(ValueError):dedup([u],seed=seed)
    with pytest.raises(ValueError):p.split_components([u],seed=seed)


def test_no_torch_real_eligibility_or_training_claim_and_numerical_budgets_unchanged():
    assert 'torch' not in sys.modules
    result=p.policy();old=p.v1.policy()
    for key in ('A0_train_exposures','A0_repeat_cap','A1_train_exposures','sft_mm_train_exposures','sft_text_train_exposures','groups'):
        assert result[key]==old[key]
    assert result['profile']!=old['profile'] and not result['GPU_training_authorized']
    assert not result['primary_manifest_exists'] and not result['actual_eligible_pool_counts_known']
    assert not result['external_immutable_audit_files_verified']
