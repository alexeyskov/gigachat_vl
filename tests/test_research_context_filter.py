"""Context screening preserves data and rejects unusable supervision explicitly."""
import json
from pathlib import Path

import pytest

from scripts.filter_research_context import MANIFESTS,derive,digest,measure_record,preflight_rows
from src.dataset.vision_cache import canonical_json


def tokenize(row):return list(range(row.get('prompt_tokens',2))),[],list(range(row.get('answer_tokens',4)))

def record(identifier,split='train',image=False,**fields):
    return {'record_id':identifier,'partition':split,'source':{'repo':'test/source','config':'cfg'},
            'image_paths':['shared-image.jpg'] if image else [],'image_hashes':['samehash'] if image else [],
            'question':'café','answer':'target',**fields}


def make_parent(tmp_path,extra_text=(),overflow_mm=False):
    parent=tmp_path/'parent';parent.mkdir()
    specs={'seed':42,'specs':[{'repo':'test/source','config':'cfg','limit':6+len(extra_text)}]}
    audit={'manifest_id':digest(canonical_json(specs).encode()),'source_counts':{'test/source/cfg':{'records':6+len(extra_text),'requested':6+len(extra_text)}}}
    for name in MANIFESTS:
        split=name.removesuffix('.jsonl').rsplit('_',1)[1]
        row=record(name,split,image=not name.startswith('text_'))
        if overflow_mm and name=='sft_train.jsonl':row['prompt_tokens']=8
        rows=[row]+(list(extra_text) if name=='text_train.jsonl' else [])
        raw=''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in rows).encode()
        (parent/name).write_bytes(raw)
        audit[name.removesuffix('.jsonl')+'_sha256']=digest(raw)
        audit[name.removesuffix('.jsonl')+'_records']=len(rows)
    (parent/'specs.json').write_text(canonical_json(specs)+'\n')
    (parent/'audit.json').write_text(json.dumps(audit))
    return parent


def derive_small(parent,out):
    return derive(parent,out,tokenize,{'model_ref':{'repo':'tiny','revision':'fixed'}},max_length=10,vision_tokens=3,reserve_bytes=0,run_source_audit=False)


def test_only_zero_position_text_is_dropped_and_other_bytes_unchanged(tmp_path):
    parent=make_parent(tmp_path,[record('overflow',prompt_tokens=10),record('one-target',prompt_tokens=9,answer_tokens=25)])
    before={name:(parent/name).read_bytes() for name in MANIFESTS}
    out=tmp_path/'derived';audit=derive_small(parent,out)
    assert audit['context_excluded_record_ids']==['overflow']
    assert audit['context_excluded_records_by_manifest']=={'text_train.jsonl':1}
    assert [json.loads(x)['record_id'] for x in (out/'text_train.jsonl').read_text().splitlines()]==['text_train.jsonl','one-target']
    for name in MANIFESTS:
        assert (parent/name).read_bytes()==before[name]
        if not name.startswith('text_'):assert (out/name).read_bytes()==before[name]
    assert (out/'parent_specs.json').read_bytes()==(parent/'specs.json').read_bytes()
    assert audit['source_counts']['test/source/cfg']['records']==7
    assert audit['context_preflight']['text_train.jsonl']['answer_truncated_records']==1
    assert audit['context_filter_lineage']['policy']['arbitrary_target_cap'] is None


def test_idempotent_and_changed_output_rejected(tmp_path):
    parent=make_parent(tmp_path);out=tmp_path/'derived'
    original=derive_small(parent,out)
    before={p.name:p.read_bytes() for p in out.iterdir()}
    assert derive_small(parent,out)==original
    assert {p.name:p.read_bytes() for p in out.iterdir()}==before
    (out/'alignment_train.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Immutable'):derive_small(parent,out)


def test_multimodal_overflow_is_fatal_before_output_created(tmp_path):
    parent=make_parent(tmp_path,overflow_mm=True);out=tmp_path/'derived'
    with pytest.raises(ValueError,match='multimodal context overflow is fatal'):derive_small(parent,out)
    assert not out.exists()
    assert not list(tmp_path.glob('derived.staging-*'))


def test_parent_tampering_and_policy_changes_are_refused(tmp_path):
    parent=make_parent(tmp_path);out=tmp_path/'derived';derive_small(parent,out)
    with pytest.raises(FileExistsError,match='different provenance'):
        derive(parent,out,tokenize,{'model_ref':{'repo':'other'}},max_length=10,vision_tokens=3,reserve_bytes=0,run_source_audit=False)
    (parent/'text_train.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Parent manifest changed'):derive_small(parent,out)


def test_expanded_vision_cost_and_supervised_token_fraction():
    row=record('visual',image=True,prompt_tokens=4,answer_tokens=20)
    stats=measure_record(row,tokenize,max_length=10,vision_tokens=3)
    assert stats['expanded_prompt_tokens']==7 and stats['supervised_target_tokens']==3
    assert stats['original_target_tokens']==20 and stats['answer_truncated']
    report=preflight_rows([row,record('text',answer_tokens=7)],tokenize,max_length=10,vision_tokens=3)
    assert report['text_target_tokens']==7 and report['multimodal_target_tokens']==3
    assert report['text_supervised_token_fraction']==.7 and report['expanded_prompt_overflow']==[]
