"""The fifth route is CPU orchestration only; original GPU policies stay frozen."""
import json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scripts import research_archival_stock_queue_v7 as q

def fixture(tmp_path,monkeypatch,routed=True):
 run=tmp_path/'run';job={'id':'job','argv':['--output',str(run)]};queue=SimpleNamespace(spec={'jobs':[job]},_routed=lambda j:routed,_guard=Mock());live={'queue':queue};old=SimpleNamespace(Popen=Mock(return_value='old'));spawn=q._Spawn(old,live);mock=Mock(return_value=SimpleNamespace(pid=123));monkeypatch.setattr(q.subprocess,'Popen',mock);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');command=[sys.executable,'-u',str(q.VALIDATOR),'--run',str(run),'--device','cuda','--numerical-profile','stock_deterministic_small_v1','--allow-gpu'];return spawn,old,mock,command

def test_exact_archived_validator_orchestration_is_CPU_parent(tmp_path,monkeypatch):
 spawn,old,mock,command=fixture(tmp_path,monkeypatch);spawn.Popen(command,cwd=q.ROOT)
 assert mock.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and os.environ['CUDA_VISIBLE_DEVICES']=='';old.Popen.assert_not_called()

def test_raw_NoBenefit_or_foreign_run_cannot_use_new_route(tmp_path,monkeypatch):
 spawn,old,mock,command=fixture(tmp_path,monkeypatch,False)
 with pytest.raises(ValueError,match='registered'):spawn.Popen(command)
 mock.assert_not_called();old.Popen.assert_not_called()

def test_old_training_eval_and_raw_validator_delegated_unchanged(tmp_path,monkeypatch):
 spawn,old,mock,command=fixture(tmp_path,monkeypatch)
 for name in ('run_research_stock_deterministic.py','eval_research_stock_deterministic.py','validate_research_stock_deterministic.py'):
  oldcommand=[*command];oldcommand[2]=str(q.ROOT/'scripts'/name);assert spawn.Popen(oldcommand)=='old';assert old.Popen.call_args.args[0]==oldcommand
 mock.assert_not_called()

def test_wrong_profile_or_parent_visibility_refused(tmp_path,monkeypatch):
 spawn,_,mock,command=fixture(tmp_path,monkeypatch);command[command.index('--numerical-profile')+1]='other'
 with pytest.raises(ValueError):spawn.Popen(command)
 command[command.index('--numerical-profile')+1]='stock_deterministic_small_v1';monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
 with pytest.raises(ValueError):spawn.Popen(command)
 mock.assert_not_called()

def test_real_private_class_CPU_mask_and_training_methods_inherited():
 code="""
import os,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts import research_archival_stock_queue_v7 as q
m=q.load_controller('eed4f7b577d1335adad956a464849edf2202e1a89e34c661f7cd8e03ca05b807');c=m.queue_class()
import torch
assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
assert 'run' not in c.__dict__ and 'compact_completed_job' not in c.__dict__
assert isinstance(c.run.__globals__['subprocess'],q._Spawn)
print(json.dumps({'CPU_only':True,'native_training_inherited':True}))
"""
 r=subprocess.run([sys.executable,'-c',code],cwd=q.ROOT,capture_output=True,text=True);assert r.returncode==0,r.stderr
 assert json.loads(r.stdout)['native_training_inherited']
