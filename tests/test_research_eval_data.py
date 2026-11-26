from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.eval.research_eval_data import convert_row,overlap_candidates,SHORT_ANSWER,MCQ_ANSWER


SOURCE={'repo':'test/mirror','revision':'sha','config':'default','split':'validation','shard':'heldout.parquet'}
IMAGE={'bytes':b'placeholder','path':'4352.png'}


def test_vqa_retains_all_ten_answers_and_original_ids(tmp_path):
    row={'image':IMAGE,'question':'Where is he looking?','answers':[{'answer':'down'}]*7+[{'answer':'left'}]*3,'question_id':42,'image_id':99}
    record=convert_row('vqav2',row,0,SOURCE,tmp_path)
    assert record['answers']==['down']*7+['left']*3
    assert record['scorer']=='vqa_accuracy' and record['source']['original_id']=='42'
    assert record['scoring_protocol']=='vqav2_official_legacy'
    assert record['question'].endswith(SHORT_ANSWER)
    row['answers']=row['answers'][:9]
    with pytest.raises(ValueError,match='ten'):
        convert_row('vqav2',row,0,SOURCE,tmp_path)


def test_docvqa_rejects_training_split(tmp_path):
    with pytest.raises(ValueError,match='not validation'):
        convert_row('docvqa',{'image':IMAGE,'data_split':'train'},0,SOURCE,tmp_path)


def test_scienceqa_pid_verified_hint_allowed_lecture_solution_absent(tmp_path):
    problem={'question':'Which object is alive?','choices':['rock','tree'],'answer':1,'hint':'Consider the picture.'}
    (tmp_path/'ScienceQA-original-problems.json').write_text(json.dumps({'123':problem}))
    (tmp_path/'ScienceQA-original-pid_splits.json').write_text(json.dumps({'test':['123']}))
    row={**problem,'image':IMAGE,'lecture':'DO_NOT_INCLUDE_LECTURE','solution':'DO_NOT_INCLUDE_SOLUTION'}
    record=convert_row('scienceqa',row,0,{**SOURCE,'split':'test'},tmp_path)
    assert record['source']['original_id']=='123'
    assert 'Context: Consider the picture.' in record['question']
    assert 'DO_NOT_INCLUDE' not in record['question']
    assert record['answers']==['B'] and record['answer_index']==1
    assert record['question'].endswith(MCQ_ANSWER)
    row['answer']=0
    with pytest.raises(ValueError,match='PID mismatch'):
        convert_row('scienceqa',row,0,{**SOURCE,'split':'test'},tmp_path)


def test_chartqa_original_annotation_and_image_verified(tmp_path):
    (tmp_path/'ChartQA-original-test_human.json').write_text(json.dumps([{'query':'How many?','label':'14','imgname':'official.png'}]))
    record=convert_row('chartqa',{'image':IMAGE,'query':'How many?','label':['14'],'human_or_machine':0},0,SOURCE,tmp_path)
    assert record['source']['original_image_id']=='official.png'
    assert record['source']['original_id']=='test_human.json:0'
    assert record['scorer']=='chart_relaxed'


def test_ai2d_choices_are_reindexed_not_confused_with_diagram_labels(tmp_path):
    record=convert_row('ai2d',{'image':IMAGE,'question':'Which label points to milk?','options':['c','D','b','a'],'answer':'1'},0,SOURCE,tmp_path)
    assert record['answers']==['B'] and record['answer_index']==1
    assert 'B. D' in record['question']
    assert record['source']['original_question_id'] is None
    assert record['source']['original_image_id']=='4352.png'


def test_exact_and_near_overlap_reports_do_not_modify_training_refs():
    refs=[{'manifest':'train.jsonl','record_id':'train-1','partition':'train'}]
    index={'images':{'same':{'phash':'0000000000000000','references':refs}}}
    exact=overlap_candidates({'image_hashes':['same'],'image_phashes':['0000000000000000']},index)
    assert [x['kind'] for x in exact]==['exact']
    near=overlap_candidates({'image_hashes':['different'],'image_phashes':['0000000000000001']},index)
    assert near[0]['kind']=='near_duplicate_candidate' and near[0]['distance']==1
    assert not overlap_candidates({'image_hashes':['far'],'image_phashes':['ffffffffffffffff']},index)
    assert refs==[{'manifest':'train.jsonl','record_id':'train-1','partition':'train'}]


def test_protocol_derivative_preserves_selected_ids_and_images(tmp_path):
    import hashlib
    from src.eval.research_eval_data import derive_scoring_protocols
    parent=tmp_path/'parent';parent.mkdir();out=tmp_path/'out'
    rows=[{'record_id':'vqa','benchmark':'vqav2','image_paths':['existing.image'],'answers':['yes']*10},
          {'record_id':'text','benchmark':'textvqa','image_paths':['other.image'],'answers':['word']*10}]
    text=''.join(json.dumps(row)+'\n' for row in rows)
    (parent/'evaluation.jsonl').write_text(text)
    (parent/'resolved_config.json').write_text('{}');(parent/'README.md').write_text('Original source protocol.\n')
    (parent/'audit.json').write_text(json.dumps({'manifest_fingerprint':'parenthash','file_sha256':{'evaluation.jsonl':hashlib.sha256(text.encode()).hexdigest()},'selected_records':2}))
    result=derive_scoring_protocols(parent,out)
    actual=[json.loads(line) for line in (out/'evaluation.jsonl').read_text().splitlines()]
    assert [r['record_id'] for r in actual]==['vqa','text']
    assert [r['image_paths'] for r in actual]==[['existing.image'],['other.image']]
    assert [r['scoring_protocol'] for r in actual]==['vqav2_official_legacy','textvqa_evalai']
    assert result['derived_from_eval_manifest_fingerprint']=='parenthash'
    assert derive_scoring_protocols(parent,out)==result


def test_future_sft_derivative_drops_exact_groups_only_and_preserves_captions(tmp_path,monkeypatch):
    import hashlib
    from src.eval.research_eval_data import derive_scienceqa_disjoint_training
    import scripts.audit_research_manifest
    monkeypatch.setattr(scripts.audit_research_manifest,'audit_prepared',lambda path:{'status':'test_schema_checked'})
    parent=tmp_path/'parent';parent.mkdir();official=tmp_path/'official';official.mkdir();out=tmp_path/'out'
    def row(name,image,repo,config):
        return {'record_id':name,'image_hashes':[image],'source':{'repo':repo,'config':config}}
    records={('alignment','train'):[row('caption','different-caption','caption/repo','data')],
             ('sft','train'):[row('exact','heldout','qa/repo','scienceqa'),row('near','near-only','qa/repo','scienceqa')],
             ('sft','validation'):[row('exact-val','heldout','qa/repo','scienceqa')]}
    audit={'manifest_id':'parent','source_counts':{'caption/repo/data':{'records':1},'qa/repo/scienceqa':{'records':3}}}
    for stage in ['alignment','sft','text']:
        for partition in ['train','validation']:
            text=''.join(json.dumps(r)+'\n' for r in records.get((stage,partition),[]))
            (parent/f'{stage}_{partition}.jsonl').write_text(text)
            audit[f'{stage}_{partition}_sha256']=hashlib.sha256(text.encode()).hexdigest()
    (parent/'audit.json').write_text(json.dumps(audit))
    (parent/'specs.json').write_text(json.dumps({'specs':[{'repo':'caption/repo','config':'data','limit':1},{'repo':'qa/repo','config':'scienceqa','limit':3}]}))
    reference=json.dumps({'image_hashes':['heldout'],'source':{'original_id':'official-pid','split':'test'}})+'\n'
    (official/'scienceqa_full_heldout_images.jsonl').write_text(reference)
    (official/'audit.json').write_text(json.dumps({'official_release_split_row_counts_match':True,'manifest_fingerprint':'full-original-test-val',
        'file_sha256':{'scienceqa_full_heldout_images.jsonl':hashlib.sha256(reference.encode()).hexdigest()}}))
    result=derive_scienceqa_disjoint_training(parent,official,out)
    assert (out/'alignment_train.jsonl').read_bytes()==(parent/'alignment_train.jsonl').read_bytes()
    assert [json.loads(x)['record_id'] for x in (out/'sft_train.jsonl').read_text().splitlines()]==['near']
    assert (out/'sft_validation.jsonl').read_text()==''
    assert result['exact_excluded_records']==2
    assert result['scienceqa_exclusion_lineage']['near_only_exclusions_applied'] is False
    assert result['primary_full_eval_decontamination_complete'] is False
