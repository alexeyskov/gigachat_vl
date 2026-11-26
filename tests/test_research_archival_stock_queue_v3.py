"""Parent-only CUDA isolation and exact child admission, no scientific writes."""
import json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scripts import research_archival_stock_queue_v3 as q

def command(script):return [sys.executable,'-u',str(q.ROOT/'scripts'/script),'--numerical-profile','stock_deterministic_small_v1','--allow-gpu']

@pytest.mark.parametrize('script',q.GPU_SCRIPTS)
def test_exact_three_gpu_children_restore_only_own_environment(script,monkeypatch):
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');before=dict(os.environ);env=q.child_environment(command(script),dict(os.environ,OTHER_TEST_FIELD='preserved'))
 assert env['CUDA_VISIBLE_DEVICES']=='0' and env['OTHER_TEST_FIELD']=='preserved' and dict(os.environ)==before

@pytest.mark.parametrize('argv',[[sys.executable,'-c','pass'],['nvidia-smi','--query-compute-apps=pid'],command(q.GPU_SCRIPTS[2])+['--device','cpu'],command(q.GPU_SCRIPTS[2])+['--cpu-structural-only']])
def test_cpu_commands_always_hide_cuda_even_supplied_gpu_env(argv,monkeypatch):
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');assert q.child_environment(argv,{'CUDA_VISIBLE_DEVICES':'0'})['CUDA_VISIBLE_DEVICES']==''

@pytest.mark.parametrize('argv',[[sys.executable,'-u','/tmp/run_research_stock_deterministic.py','--allow-gpu'],[sys.executable,'-u','foreign.py','--device','cuda:1'],[sys.executable,'-u','foreign.py','--device=cuda:2'],command(q.GPU_SCRIPTS[0])[:3],command(q.GPU_SCRIPTS[0])[:4]+['wrong','--allow-gpu']])
def test_unknown_or_implicit_python_gpu_commands_refused(argv,monkeypatch):
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
 with pytest.raises(ValueError):q.child_environment(argv)

def test_proxy_scoped_to_private_modules_and_does_not_patch_global_popen(monkeypatch):
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');real=subprocess.Popen;fake=Mock(return_value='owned_child');module=SimpleNamespace(Popen=fake,STDOUT='unchanged')
 proxy=q._Subprocess(module);assert proxy.Popen(command(q.GPU_SCRIPTS[1]),cwd=q.ROOT)=='owned_child'
 assert fake.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES']=='0' and proxy.STDOUT=='unchanged' and subprocess.Popen is real

def test_parent_drift_or_initialized_cuda_refuses_before_spawn(monkeypatch):
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
 with pytest.raises(RuntimeError,match='hidden'):q.child_environment(command(q.GPU_SCRIPTS[0]))
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True)))
 with pytest.raises(RuntimeError,match='initialized'):q.child_environment(command(q.GPU_SCRIPTS[0]))

def test_fresh_process_bootstrap_before_real_torch_keeps_cuda_uninitialized():
 code="""
import os,sys,json
from scripts import research_archival_stock_queue_v3 as q
assert 'torch' not in sys.modules
os.environ['CUDA_VISIBLE_DEVICES']='0';q.bootstrap()
module=q.load_controller()
import torch
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
cls=module.queue_class()
assert 'run' not in cls.__dict__ and 'compact_completed_job' not in cls.__dict__
spawns=[value for base in cls.__mro__ for value in base.__dict__.values() if callable(value) and hasattr(value,'__code__') and 'Popen' in value.__code__.co_names]
assert len(spawns)>=4
assert all(isinstance(value.__globals__['subprocess'],q._Subprocess) for value in spawns)
assert not torch.cuda.is_initialized()
print(json.dumps({'parent_cuda_initialized':torch.cuda.is_initialized(),'parent_cuda_available':torch.cuda.is_available(),'training_and_postprocess_methods_inherited':True}))
"""
 result=subprocess.run([sys.executable,'-c',code],cwd=q.ROOT,capture_output=True,text=True)
 assert result.returncode==0,result.stderr
 assert json.loads(result.stdout)['parent_cuda_initialized'] is False

def test_bootstrap_refuses_late_import_even_without_initialized_cuda(monkeypatch):
 monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False)))
 with pytest.raises(RuntimeError,match='fresh process'):q.bootstrap()

def test_source_release_is_same_v2_and_same_immutable_spec():
 assert q.sha(q.ROOT/q.V2)==q.V2_SHA and q.sha(q.V2_READY)==q.V2_READY_SHA
 sources=q.source_binding();assert sources[q.V2]==q.V2_SHA and sources[q.SOURCE]==q.sha(q.ROOT/q.SOURCE)
 source=(q.ROOT/q.SOURCE).read_text();assert 'os.environ[\'CUDA_VISIBLE_DEVICES\']=\'\'' in source
