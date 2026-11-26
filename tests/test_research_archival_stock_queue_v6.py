"""Completed metadata must remain byte/stat immutable; pending recovery retained."""
import json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scripts import research_archival_stock_queue_v6 as q

@pytest.fixture
def current(tmp_path):
 report=tmp_path/'report.json';journal=tmp_path/'journal.json';report.write_text('{"status":"completed","disk":{"free":123}}');journal.write_text('{"phase":"completed"}')
 read=lambda p:json.loads(Path(p).read_bytes());publisher=SimpleNamespace(REPORT=report.name,JOURNAL=journal.name,read=read,
  eligibility=lambda run:{'roles':{'best':{'marker':{'checkpoint_archival':{'protocol':'fixture'}}},'last':{'marker':{'checkpoint_archival':{'protocol':'fixture'}}}}},
  validate_completed_archival=Mock(side_effect=lambda run:read(report)),publish_completed_checkpoint=Mock(return_value='pending_original'))
 q.install_readonly(publisher);return SimpleNamespace(run=tmp_path,report=report,journal=journal,publisher=publisher)

def test_completed_report_and_journal_exact_bytes_stats_and_disk_not_rewritten(current):
 before={p:(p.read_bytes(),p.stat()) for p in (current.report,current.journal)}
 result=current.publisher.publish_completed_checkpoint(current.run,reserve_gb=120)
 assert result['disk']=={'free':123} and current.publisher.validate_completed_archival.call_count==1
 assert before=={p:(p.read_bytes(),p.stat()) for p in before}

@pytest.mark.parametrize('pending',['journal','raw_role'])
def test_pending_transactions_and_raw_roles_keep_original_recovery(current,pending):
 if pending=='journal':current.journal.write_text('{"phase":"verified"}')
 else:current.publisher.eligibility=lambda run:{'roles':{'best':{'marker':{}}}}
 assert current.publisher.publish_completed_checkpoint(current.run)=='pending_original'
 current.publisher.validate_completed_archival.assert_not_called()

def test_corrupt_completed_validation_never_falls_back_or_overwrites(current):
 before=current.report.read_bytes();current.publisher.validate_completed_archival.side_effect=ValueError('source/role lineage corrupt')
 with pytest.raises(ValueError,match='lineage'):current.publisher.publish_completed_checkpoint(current.run)
 assert current.report.read_bytes()==before

def test_failed_or_unknown_report_refuses_without_rewriting(current):
 current.report.write_text('{"status":"failed"}');before=current.report.read_bytes()
 with pytest.raises(ValueError,match='preserved'):current.publisher.publish_completed_checkpoint(current.run)
 assert current.report.read_bytes()==before;current.publisher.validate_completed_archival.assert_not_called()

def test_unexpected_metadata_mutation_during_validation_refuses(current):
 def mutate(run):current.report.write_text('{"status":"completed","disk":{"free":999}}');return json.loads(current.report.read_bytes())
 current.publisher.validate_completed_archival.side_effect=mutate
 with pytest.raises(ValueError,match='changed'):current.publisher.publish_completed_checkpoint(current.run)

def test_private_full_stack_still_CPU_only_and_preserves_training_recovery_and_gpu_proof():
 code="""
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts import research_archival_stock_queue_v6 as q
m=q.load_controller('eed4f7b577d1335adad956a464849edf2202e1a89e34c661f7cd8e03ca05b807');c=m.queue_class()
import torch
assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
assert 'run' not in c.__dict__ and 'compact_completed_job' not in c.__dict__
assert c.compact_completed_job.__globals__['archival'].publish_completed_checkpoint.__module__==q.__name__
print(json.dumps({'CPU_only':True,'same_training_and_pending_recovery':True}))
"""
 r=subprocess.run([sys.executable,'-c',code],cwd=q.ROOT,capture_output=True,text=True);assert r.returncode==0,r.stderr
 assert json.loads(r.stdout)['CPU_only']
