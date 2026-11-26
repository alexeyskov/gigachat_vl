import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scripts import probe_research_selection_deterministic as v2


def test_clone_is_private_and_frozen_native_bindings_unchanged():
 from scripts import probe_research_selection_replay as native
 old=(native.OWNER,native.READY,native.NativeRuntime,native.SOURCES,native.sha(Path(native.__file__)))
 d=v2.clone_native()
 assert d.OWNER==v2.OWNER and d.OUTPUT==v2.OUTPUT and d.READY==v2.READY and len(d.SOURCES)==25
 assert d.__file__==v2.__file__ and d.start_watchdog.__globals__['OWNER']==v2.OWNER
 assert (native.OWNER,native.READY,native.NativeRuntime,native.SOURCES,native.sha(Path(native.__file__)))==old


def test_stock_runtime_sets_strict_torch_determinism_and_fast_restore_poll(monkeypatch):
 d=v2.clone_native();base=d.NativeRuntime.__mro__[1];calls=[]
 monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
 torch=SimpleNamespace(use_deterministic_algorithms=Mock());monkeypatch.setitem(__import__('sys').modules,'torch',torch)
 monkeypatch.setattr(base,'load',lambda self,plan,check:calls.append(check) or ('model',{}))
 class Guard:
  def check(self):pass
  def fast_check(self):pass
 guard=Guard();assert d.NativeRuntime().load({},guard.check)==('model',{})
 assert calls==[guard.fast_check];torch.use_deterministic_algorithms.assert_called_once_with(True,warn_only=False)


def test_workspace_required_before_any_torch_policy(monkeypatch):
 d=v2.clone_native();monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
 with pytest.raises(RuntimeError,match='precede Torch'):d.NativeRuntime().load({},lambda:None)


def test_unsupported_stock_deterministic_op_has_no_fallback(monkeypatch):
 d=v2.clone_native();monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
 torch=SimpleNamespace(use_deterministic_algorithms=Mock(side_effect=RuntimeError('unsupported deterministic op')))
 monkeypatch.setitem(__import__('sys').modules,'torch',torch)
 with pytest.raises(RuntimeError,match='unsupported'):d.NativeRuntime().load({},lambda:None)


def test_repeated_scope_is_all_records_not_only_old_mismatch(monkeypatch):
 d=v2.clone_native();base=d.NativeRuntime.__mro__[1];seen=[]
 monkeypatch.setattr(base,'repeat',lambda self,model,selection,ids,check:seen.extend(ids))
 rows=[{'record_id':str(i)} for i in range(116)]
 d.NativeRuntime().repeat(None,{'records':rows},[{'record_id':'1'}],lambda:None)
 assert seen==rows


def test_repeat_observer_uses_explicit_full116_phase(monkeypatch):
 d=v2.clone_native();base=d.NativeRuntime.__mro__[1];calls=[]
 monkeypatch.setattr(base,'observed_generate',lambda self,*a,**k:calls.append(k) or 'same_output')
 assert d.NativeRuntime().observed_generate(None,{},None,phase='mismatch_repeat_2',max_new_tokens=64)=='same_output'
 assert calls[0]['phase']=='full116_repeat_2'


def test_completed_report_requires_two_complete_passes_and_compares_all_ids(tmp_path):
 d=v2.clone_native()
 first=[{'record_id':str(i),'generation':{'generated_token_ids':[i],'prompt_sha256':str(i)}} for i in range(116)]
 d.write(tmp_path/'replayed_selection.json',{'predictions':first})
 repeats=[[{'record_id':r['record_id'],'generation':copy.deepcopy(r['generation'])} for r in first] for _ in range(2)]
 repeats[1][90]['generation']['generated_token_ids']=[999]
 report={'owner':d.OWNER,'status':'completed_diagnostic_only','repeated_mismatches':repeats}
 d.write(tmp_path/'report.json',report)
 saved=d.read(tmp_path/'report.json')
 assert saved['three_pass_token_mismatch_records']==1 and saved['three_pass_prompt_mismatch_records']==0
 assert 'repeated_mismatches' not in saved and len(saved['repeated_all116'])==2 and not saved['legacy_validation_passed']
 assert saved['numerical_profile']==v2.PROFILE

@pytest.mark.parametrize('mutation',['count','order'])
def test_incomplete_or_reordered_repeat_refuses_completed_report(tmp_path,mutation):
 d=v2.clone_native();first=[{'record_id':str(i),'generation':{'generated_token_ids':[i],'prompt_sha256':str(i)}} for i in range(116)]
 d.write(tmp_path/'replayed_selection.json',{'predictions':first});repeats=[copy.deepcopy(first) for _ in range(2)]
 if mutation=='count':repeats[1].pop()
 else:repeats[1].reverse()
 with pytest.raises(ValueError):d.write(tmp_path/'report.json',{'owner':d.OWNER,'status':'completed_diagnostic_only','repeated_mismatches':repeats})
 assert not (tmp_path/'report.json').exists()


def test_repeat_file_name_never_claims_mismatch_only_scope(tmp_path):
 d=v2.clone_native();d.write(tmp_path/'mismatch_repeat_1.json',[])
 assert (tmp_path/'full116_repeat_1.json').is_file() and not (tmp_path/'mismatch_repeat_1.json').exists()

@pytest.mark.parametrize('args',[['--mismatch-repeats','2'],['--full116-repeats','1']])
def test_cli_cannot_shrink_repeated_scope(args):
 with pytest.raises(ValueError):v2.main(args)


def test_cli_sets_workspace_before_clone_or_torch_and_keeps_fixed_three_passes(monkeypatch):
 calls=[]
 def clone():
  assert __import__('os').environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
  return SimpleNamespace(main=lambda args:calls.append(args) or 0)
 monkeypatch.setattr(v2,'clone_native',clone);monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
 assert v2.main(['--full116-repeats','2'])==0 and calls==[['--mismatch-repeats','2']]


def test_gpu_cli_refuses_process_that_already_imported_torch(monkeypatch):
 monkeypatch.setitem(__import__('sys').modules,'torch',SimpleNamespace())
 with pytest.raises(RuntimeError,match='Fresh process'):v2.main(['--allow-gpu'])


def test_actual_v2_watchdog_uses_v2_owner_and_exact_own_parent(tmp_path):
 import subprocess,sys
 nonce='c'*64
 code="""
import os,sys,time,subprocess
from pathlib import Path
from scripts import probe_research_selection_deterministic as v2
from scripts import probe_research_parallel_b1 as proc
d=v2.clone_native();path=Path(sys.argv[1]);identity=proc.process_info(os.getpid());nonce=sys.argv[3]
d.write(path,{'owner':d.OWNER,'nonce':nonce,'deadline':time.monotonic()+2,'process':identity,'status':'active'})
subprocess.Popen([sys.executable,str(Path(v2.__file__).resolve()),'--deadline-watchdog',str(path)])
time.sleep(30)
"""
 child=subprocess.Popen([sys.executable,'-c',code,str(tmp_path/'lease.json'),'--nonce',nonce])
 try:
  assert child.wait(timeout=5)==-15
  receipt=json.loads((tmp_path/'deadline_watchdog_ready.json').read_text())
  assert receipt['owner']==v2.OWNER and receipt['watched']['pid']==child.pid
 finally:
  if child.poll() is None:child.terminate();child.wait(timeout=3)


def test_resource_phase_labels_full116_repeat_scope(monkeypatch):
 d=v2.clone_native();base=d.NativeRuntime.__mro__[1]
 monkeypatch.setattr(base,'repeat',lambda *a:[])
 class Guard:
  phase='old_mismatch_label'
  def check(self):pass
 guard=Guard();runtime=d.NativeRuntime();runtime.repeat_number=2
 runtime.repeat(None,{'records':[]},[],guard.check)
 assert guard.phase=='full116_repeat_2'
