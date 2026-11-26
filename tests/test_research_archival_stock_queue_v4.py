"""No weights/services: exact original result, scoped advice and inherited isolation."""
import json,subprocess,sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scripts import research_archival_stock_queue_v4 as q

def fixture(result=None):
 events=[];result={'all_payload_fields_and_backing_bits_exact':True,'archive_sha256':'f'*64} if result is None else result
 def original(*a,**k):events.append('derive_verified');return result
 publisher=SimpleNamespace(_derive_unlocked=original);queue=SimpleNamespace(state={},persist=lambda:events.append('persist'))
 def advice(actual,run,*,selected_source):
  assert actual is queue and run=='owned_run' and selected_source=='owned_source'
  events.append('retained_owned_advice');return {'actual_available_after_bytes':80*2**30,'reclaimable_cache_credit_assumed_bytes':0}
 q.install_advice(publisher,{'queue':queue},advice,lambda *a,**k:events.append('emit'))
 return publisher,queue,events,result

def test_postderive_same_owned_advice_after_original_verification_preserves_return_and_order():
 publisher,queue,events,result=fixture();returned=publisher._derive_unlocked('owned_source','owned_stage',{'run':'owned_run'})
 assert returned is result and events==['derive_verified','retained_owned_advice','persist','emit']
 row=queue.state['archival_post_derive_cache_advice'][0]
 assert row['derived_archive_advised'] is False and row['verified_archive_sha256']=='f'*64

@pytest.mark.parametrize('result',[{}, {'all_payload_fields_and_backing_bits_exact':False},None])
def test_unverified_result_or_original_failure_never_advise(result):
 publisher=SimpleNamespace(_derive_unlocked=Mock(side_effect=RuntimeError('decode failed')) if result is None else Mock(return_value=result))
 advice=Mock();q.install_advice(publisher,{'queue':object()},advice,Mock())
 with pytest.raises((ValueError,RuntimeError)):publisher._derive_unlocked('source','stage',{'run':'owned_run'})
 advice.assert_not_called()

def test_advice_failure_stops_before_publication_without_mutating_result():
 result={'all_payload_fields_and_backing_bits_exact':True};publisher=SimpleNamespace(_derive_unlocked=lambda *a,**k:result)
 queue=SimpleNamespace(state={},persist=Mock());q.install_advice(publisher,{'queue':queue},Mock(side_effect=MemoryError('floor32')),Mock())
 with pytest.raises(MemoryError):publisher._derive_unlocked('source','stage',{'run':'owned_run'})
 assert result=={'all_payload_fields_and_backing_bits_exact':True} and not queue.state;queue.persist.assert_not_called()

def test_history_bounded_and_no_stage_passed_to_advice():
 publisher,queue,_,_=fixture()
 for _ in range(20):publisher._derive_unlocked('owned_source','must_never_advise_this_stage',{'run':'owned_run'})
 assert len(queue.state['archival_post_derive_cache_advice'])==16

def test_real_private_v3_stack_keeps_parent_hidden_child_proxy_and_native_recovery():
 code="""
import os,sys,json
from scripts import research_archival_stock_queue_v4 as q
assert 'torch' not in sys.modules
os.environ['CUDA_VISIBLE_DEVICES']='';module=q.load_controller();cls=module.queue_class()
import torch
assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
assert 'run' not in cls.__dict__ and 'compact_completed_job' not in cls.__dict__
assert cls.compact_completed_job.__globals__['archival']._derive_unlocked.__module__==q.__name__
assert hasattr(cls.run.__globals__['subprocess'],'module')
print(json.dumps({'parent_cuda_initialized':False,'training_and_verified_journal_recovery_inherited':True}))
"""
 r=subprocess.run([sys.executable,'-c',code],cwd=q.ROOT,capture_output=True,text=True);assert r.returncode==0,r.stderr
 assert json.loads(r.stdout)['parent_cuda_initialized'] is False

def test_previous_v3_release_and_all_dependencies_unchanged():
 sources=q.source_binding();assert q.sha(q.ROOT/q.V3)==q.V3_SHA and q.sha(q.V3_READY)==q.V3_READY_SHA
 assert sources[q.SOURCE]==q.sha(q.ROOT/q.SOURCE)
