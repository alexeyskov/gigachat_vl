"""CPU-only scientific-data contracts; no weights or model inference."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import os
from unittest.mock import patch
import hashlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('text_suite_test_data', ROOT/'src/eval/research_text_data.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def raw_pool(task, count, subjects=('algebra',)):
    if task == 'gsm8k': return [{'question':f'Question {i}?', 'answer':f'Work {i}.\n#### {i}'} for i in range(count)]
    if task == 'mmlu': return [{'question':f'Question {i}?','subject':subjects[i%len(subjects)],
        'choices':['Alpha','Beta','Gamma','Delta'],'answer':i%4} for i in range(count)]
    return [{'key':i,'prompt':f'Write {i} words.', 'instruction_id_list':['length_constraints:number_words'],
        'kwargs':[{'num_words':i,'relation':'at least'}]} for i in range(count)]


def asset(task):
    return {'repo':d.REPOS[task][0],'revision':d.REPOS[task][1],'filename':task+'.jsonl','sha256':'a'*64}


@pytest.fixture
def small_pools(monkeypatch):
    subjects = ['algebra','history']
    monkeypatch.setattr(d,'COUNTS',{'gsm8k':12,'mmlu':40,'ifeval':9})
    monkeypatch.setattr(d,'CAPS',{'gsm8k':5,'mmlu':32,'ifeval':4})
    pools = {t:d.build_pool(t,raw_pool(t,d.COUNTS[t],subjects),asset(t),subjects) for t in d.TASKS}
    return pools, subjects


def test_selection_reference_independent_and_source_order(small_pools):
    pools, subjects = small_pools
    selected = d.select_records(pools,subjects)
    modified = deepcopy(pools)
    for pool in modified.values():
        for row in pool: row['reference']={'wrong_answer':'CHANGED GOLD, NEVER SELECT FROM THIS'}
    assert [r['record_id'] for r in selected] == [r['record_id'] for r in d.select_records(modified,subjects)]
    for task in d.TASKS:
        indexes=[r['source']['row_index'] for r in selected if r['task']==task]
        assert indexes == sorted(indexes)
    assert all(sum(r['subject']==subject for r in selected)==16 for subject in subjects)


@pytest.mark.parametrize('task',d.TASKS)
def test_gold_changes_do_not_change_ids_or_prompts(monkeypatch,task):
    monkeypatch.setitem(d.COUNTS,task,3)
    original=raw_pool(task,3)
    changed=deepcopy(original)
    for row in changed:
        if task=='gsm8k': row['answer']='Different work.\n#### 999999'
        elif task=='mmlu': row['answer']=(row['answer']+1)%4
        else: row['instruction_id_list']=['keywords:existence']; row['kwargs']=[{'keywords':['secret_gold']}]
    a=d.build_pool(task,original,asset(task),['algebra'])
    b=d.build_pool(task,changed,asset(task),['algebra'])
    assert [(r['record_id'],r['prompt'],d.selection_key(r['source'])) for r in a] == [
        (r['record_id'],r['prompt'],d.selection_key(r['source'])) for r in b]


def test_exclusions_protect_all_released_rows_not_just_selected(small_pools):
    pools,subjects=small_pools
    selected=d.select_records(pools,subjects)
    protected=d.exclusion_rows(pools)
    assert len(protected)==61 and len(selected)==41
    assert {r['record_id'] for r in selected} < {r['record_id'] for r in protected}
    assert all('reference' not in r and not r['training_overlap_screen_performed'] for r in protected)
    assert all(r['heldout_only_despite_nominal_train_split']==(r['task']=='ifeval') for r in protected)
    for row in protected:
        assert row['raw_input_sha256']==d.digest(row['input_text'])
        assert row['normalised_input_sha256']==d.digest(d.normalise(row['input_text']))


@pytest.mark.parametrize('bad',['count','empty','answer','revision','duplicate_key','options'])
def test_bad_sources_rejected(monkeypatch,bad):
    task='ifeval' if bad=='duplicate_key' else 'mmlu' if bad=='options' else 'gsm8k'
    monkeypatch.setitem(d.COUNTS,task,2)
    rows=raw_pool(task,2); info=asset(task)
    if bad=='count': rows.pop()
    elif bad=='empty': rows[0]['question']=''
    elif bad=='answer': rows[0]['answer']='No numeric final'
    elif bad=='revision': info['revision']='b'*40
    elif bad=='duplicate_key': rows[1]['key']=rows[0]['key']
    else: rows[0]['choices']=None
    with pytest.raises(ValueError): d.build_pool(task,rows,info,['algebra'])


@pytest.mark.parametrize('answer,expected',[('Compute.\n#### 1,000','1E+3'),('#### -3.50','-3.5'),('#### .25','0.25')])
def test_numeric_gold_parse(answer,expected): assert d.numeric_reference(answer)==expected


@pytest.mark.parametrize('answer',['No final','#### NaN','#### Infinity','#### 2\n#### 3','#### xyz'])
def test_numeric_bad_gold_parse(answer):
    with pytest.raises(ValueError): d.numeric_reference(answer)


def test_public_input_is_explicit_gold_allowlist(small_pools):
    pools,_=small_pools
    row=pools['mmlu'][0]
    public=d.public_model_input(row)
    changed=deepcopy(row)
    changed.update({'reference':{'answer':'GOLDSECRET'},'answer':'GOLDSECRET','system':'GOLDSECRET'})
    assert public==d.public_model_input(changed)
    assert set(public)=={'messages','image_paths','image_hashes'}
    assert 'A. Alpha' in public['messages'][0]['content']
    assert 'D. Delta' in public['messages'][0]['content']
    public['messages'][0]['content']='mutated'
    assert row['prompt']!='mutated'


class TinyTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['tokenize'] is False and kwargs['add_generation_prompt'] is True
        assert kwargs['date_string']==kwargs['current_date']=='2026-11-22'
        assert kwargs['strftime_now']('%Y-%m-%d')=='2026-11-22'
        return '<assistant>'+messages[0]['content']
    def encode(self,text,**kwargs):
        assert kwargs.get('add_special_tokens') is False
        assert not kwargs.get('truncation',False)
        return list(text.encode())


def test_census_all_rows_gold_filtered_no_truncation(small_pools):
    pools,subjects=small_pools
    records=d.select_records(pools,subjects)
    a=d.context_census(records,TinyTokenizer(),'smallthinker4',32768)
    changed=deepcopy(records)
    for row in changed: row['reference']={'secret':'ignored'}
    assert a==d.context_census(changed,TinyTokenizer(),'smallthinker4',32768)
    assert len(a['records'])==41
    assert a['per_task']['ifeval']['max_required_context_tokens']>=2048
    assert all(r['candidate_continuation_token_lengths']==dict.fromkeys('ABCD',1)
               for r in a['records'] if r['task']=='mmlu')


def test_gpt_final_channel_explicit(small_pools):
    row=small_pools[0]['gsm8k'][0]
    assert d.render_prompt(TinyTokenizer(),row,'gptoss20').endswith('<|channel|>final<|message|>')
    assert not d.render_prompt(TinyTokenizer(),row,'smallthinker4').endswith('<|channel|>final<|message|>')


def test_mmlu_unstable_token_prefix_fails(small_pools):
    class Bad(TinyTokenizer):
        def encode(self,text,**kwargs):
            ids=super().encode(text,**kwargs)
            return [999]+ids[1:] if text.endswith(tuple('ABCD')) else ids
    with pytest.raises(ValueError,match='prefix'): d.context_census(d.select_records(*small_pools),Bad(),'x',32768)


def test_context_choice_before_scores_no_silent_drop():
    census=[{'max_required_context_tokens':4300,'native_context_limit':32768} for _ in range(4)]
    result=d.choose_context(census)
    assert result['chosen_context_limit']==8192 and result['changed_from_proposal']
    assert result['all_selected_records_fit'] and not result['GPU_memory_fit_proven']
    with pytest.raises(ValueError): d.choose_context(census[:3])
    census[0]['native_context_limit']=4096
    with pytest.raises(ValueError): d.choose_context(census)


def test_immutable_bundle_idempotent_bytes(tmp_path):
    output=tmp_path/'suite'; files={'records.jsonl':b'{}\n'}; protocol={'name':'suite'}
    first=d.write_immutable_bundle(output,files,protocol,reserve_bytes=0)
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in output.iterdir()}
    assert d.write_immutable_bundle(output,files,protocol,reserve_bytes=0)==first
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in output.iterdir()}
    assert first['files_sha256']['records.jsonl']==d.digest(files['records.jsonl'])


@pytest.mark.parametrize('corrupt',['content','protocol','extra','missing','symlink'])
def test_immutable_bundle_refuses_corruption(tmp_path,corrupt):
    output=tmp_path/'suite'; files={'records.jsonl':b'{}\n'}; protocol={'name':'suite'}
    d.write_immutable_bundle(output,files,protocol,reserve_bytes=0)
    if corrupt=='content': (output/'records.jsonl').write_bytes(b'BAD')
    elif corrupt=='protocol': (output/'protocol.json').write_text('{}')
    elif corrupt=='extra': (output/'extra').touch()
    elif corrupt=='missing': (output/'records.jsonl').unlink()
    else:
        (tmp_path/'outside').write_bytes(files['records.jsonl'])
        (output/'records.jsonl').unlink(); (output/'records.jsonl').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError): d.write_immutable_bundle(output,files,protocol,reserve_bytes=0)


@pytest.mark.parametrize('name',['../bad','sub/file','protocol.json'])
def test_immutable_bundle_rejects_unsafe_members(tmp_path,name):
    with pytest.raises(ValueError): d.write_immutable_bundle(tmp_path/'suite',{name:b'x'},{},reserve_bytes=0)
    assert not (tmp_path/'suite').exists()


def test_immutable_bundle_disk_reserve(tmp_path):
    with pytest.raises(RuntimeError): d.write_immutable_bundle(tmp_path/'suite',{'x':b'x'},{},reserve_bytes=10**30)
    assert not (tmp_path/'suite').exists()


def test_fixed_released_counts_and_caps():
    assert d.COUNTS=={'gsm8k':1319,'mmlu':14042,'ifeval':541}
    assert sum(d.COUNTS.values())==15902 and sum(d.CAPS.values())==1680
    assert all(len(r[1])==40 for r in d.REPOS.values())


@pytest.fixture
def cli():
    # Importing the standalone CLI must not alter the parent test environment.
    original_path=sys.path.copy()
    with patch.dict(os.environ,dict(os.environ),clear=True):
        spec=importlib.util.spec_from_file_location('text_suite_test_cli',ROOT/'scripts/prepare_research_text_eval.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sys.path[:]=original_path
    return module


def assets_receipt(tmp_path):
    raw=b'{"pinned":"tokenizer metadata"}'
    path=tmp_path/'config.json'; path.write_bytes(raw)
    row={'kind':'tokenizer','id':'giga31','repo':'ai-sage/GigaChat3.1-10B-A1.8B-bf16',
        'revision':'189fff27a1dee68473960c3d5bca53e0e07a3191','repo_type':'model',
        'filename':'config.json','bytes':len(raw),'sha256':d.digest(raw),'declared_sha256':None,
        'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),
        'path':str(path),'cached_before':False}
    return {'schema_version':1,'status':'passed','new_content_bytes':len(raw),
        'metadata_allowance_bytes':8*2**20,'cap_bytes':250*2**20,'files':[row]}


def test_asset_validation_pinned_git_blob_and_LFS(cli,tmp_path):
    receipt=assets_receipt(tmp_path); path=tmp_path/'receipt.json'
    path.write_text(json.dumps(receipt)); assert cli.verify_assets(path)==receipt
    receipt['files'][0]['declared_sha256']=receipt['files'][0]['sha256']
    path.write_text(json.dumps(receipt)); assert cli.verify_assets(path)==receipt


@pytest.mark.parametrize('corrupt',['repo','revision','weight','blob','lfs','local_bytes','duplicate','budget','accounting','status'])
def test_asset_validation_corruption_refused(cli,tmp_path,corrupt):
    receipt=assets_receipt(tmp_path); row=receipt['files'][0]
    if corrupt=='repo': row['repo']='foreign/model'
    elif corrupt=='revision': row['revision']='0'*40
    elif corrupt=='weight': row['filename']='model-00001-of-00004.safetensors'
    elif corrupt=='blob': row['git_blob_sha1']='0'*40
    elif corrupt=='lfs': row['declared_sha256']='0'*64
    elif corrupt=='local_bytes': Path(row['path']).write_bytes(b'changed')
    elif corrupt=='duplicate': receipt['files'].append(deepcopy(row));receipt['new_content_bytes']*=2
    elif corrupt=='budget': receipt['metadata_allowance_bytes']=250*2**20
    elif corrupt=='accounting': receipt['new_content_bytes']+=1
    else: receipt['status']='download_in_progress'
    path=tmp_path/'receipt.json';path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError): cli.verify_assets(path)
