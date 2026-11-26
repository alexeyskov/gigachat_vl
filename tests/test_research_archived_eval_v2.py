"""Actual compressed fixtures and unchanged after-decode memory guards."""
import copy
import json
import subprocess
import sys

import pytest

from src.utils import research_archived_eval_v2 as phased
from tests import test_research_archived_eval_v1 as previous
from test_research_trained_delta_probe import fixture
from test_research_checkpoint_archival import prepared


@pytest.fixture
def archived(prepared,monkeypatch):
    helper=phased.implementation()
    monkeypatch.setattr(previous,'helper',helper)
    return previous.archived.__wrapped__(prepared,monkeypatch)


def test_fresh_profile_reads_no_torch_or_checkpoint_bodies():
    source="from src.utils import research_archived_eval_v2 as a;import sys,json;p=a.execution_profile();assert 'torch' not in sys.modules;print(json.dumps(p))"
    result=subprocess.run([sys.executable,'-c',source],cwd=phased.ROOT,capture_output=True,text=True,check=True)
    profile=json.loads(result.stdout)
    assert profile['CPU_payload_decodes_per_load']==1
    assert profile['headroom_bytes']==32*2**30
    assert profile['constructor_allowance_bytes']==16*2**30
    assert profile['constructor_allowance_charged_with_decoder_scratch'] is False
    assert profile['allocator_or_clean_file_cache_release_credit_bytes']==0


def test_actual_payload_aliases_rng_and_full_signature_stay_exact(archived,monkeypatch):
    previous.test_metadata_and_full_context_do_not_decode_then_real_single_payload_roundtrip(archived,monkeypatch)


def test_actual_native_constructor_still_follows_exactly_one_decode(archived,monkeypatch):
    previous.test_frozen_compatibility_loader_called_once_before_native_constructor(archived,monkeypatch)


def test_decode_and_constructor_guards_are_disjoint_but_floor_is_not_lowered(archived,monkeypatch):
    from src.utils import research_lossless_delta as codec
    import src.utils.research_checkpoint_compat as compat
    helper=phased.implementation();context=helper.checkpoint_context(archived['run'])
    plan=context['archive_admission'];value=plan['transactions']['best']
    assert value['load_extra_bytes']==value['decoder_extra_bytes']+value['archive_file_bytes']
    assert value['constructor_phase_extra_bytes']==helper.CONSTRUCTOR_ALLOWANCE+helper.METADATA_LIMIT
    events=[];guard=codec.memory_guard;load=compat.load_research_payload
    def observed_guard(extra,*args,**kwargs):
        events.append(('guard',extra));return guard(extra,*args,**kwargs)
    def observed_load(*args,**kwargs):
        result=load(*args,**kwargs);events.append(('decode_returned',None));return result
    monkeypatch.setattr(codec,'memory_guard',observed_guard)
    monkeypatch.setattr(compat,'load_research_payload',observed_load)
    restored=helper.load_payload(context['checkpoint_path'],plan)
    boundary=events.index(('decode_returned',None))
    assert ('guard',value['load_extra_bytes']) in events[:boundary]
    assert events[boundary+1]==('guard',value['constructor_phase_extra_bytes'])
    assert previous.probe.payload_signature(restored)==archived['expected_signature']


@pytest.mark.parametrize('field',['load_extra_bytes','constructor_phase_extra_bytes'])
def test_caller_cannot_relax_either_projection(archived,field):
    helper=phased.implementation();context=helper.checkpoint_context(archived['run'])
    modified=copy.deepcopy(context);modified['archive_admission']['transactions']['best'][field]=0
    with pytest.raises(ValueError,match='projection'):helper.assert_context(modified)


def test_old_profile_context_is_not_relabelled(archived):
    helper=phased.implementation();context=helper.checkpoint_context(archived['run'])
    context['archive_admission']['execution_profile']['owner']='owned_small_archived_evaluation_v1'
    with pytest.raises(ValueError,match='profile'):helper.assert_context(context)
