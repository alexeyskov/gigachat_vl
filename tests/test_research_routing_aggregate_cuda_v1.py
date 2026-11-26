"""CPU fixtures only for the ROOT-owned opt-in actual CUDA probe."""
import importlib.util,json,os,subprocess,sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
def module():
 path=ROOT/'scripts/probe_research_routing_aggregate_cuda_v1.py'
 spec=importlib.util.spec_from_file_location('_tiny_routing_cuda_fixture',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.mark.parametrize('recipe,recompute,failed,frozen',[('4f',False,False,False),('3c',True,True,False),('4f',True,False,True)])
def test_real_native_CPU_fixture_exact_gradient_RNG_update_and_lifecycle(recipe,recompute,failed,frozen):
 m=module();before=torch.are_deterministic_algorithms_enabled()
 try:
  torch.use_deterministic_algorithms(True,warn_only=False)
  result=m.pair(recipe,device='cpu',recompute=recompute,synthetic_OOM=failed,frozen=frozen)
 finally:torch.use_deterministic_algorithms(before)
 assert result['status']=='passed' and result['valid_tokens']==22
 assert result['CPU_CUDA_RNG_exact'] and result['post_optimizer_parameter_bits_exact']
 assert result['synthetic_OOM_rollback']==failed
 assert result['frozen_base_with_trainable_projector']==frozen
 assert result['actual_grouped_mm_calls']=={'original':0,'observed':0}
 assert result['output_all_gradient_hashes']['output']
 if frozen:assert not result['all_named_parameter_grad_none']['projector.weight']


def test_default_CLI_metadata_is_no_Torch_no_GPU_or_subprocess_probe(tmp_path):
 source="""
import sys,json
from scripts import probe_research_routing_aggregate_cuda_v1 as p
p.nvml_processes=lambda:(_ for _ in ()).throw(AssertionError('No GPU query in metadata'))
assert 'torch' not in sys.modules
assert p.main([])==0
assert 'torch' not in sys.modules
"""
 result=subprocess.run([sys.executable,'-c',source],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,check=True)
 value=json.loads(result.stdout)
 assert value['status']=='metadata_only_GPU_not_used' and value['stock_numerical_profile']['torch_threads']==8
 assert value['actual_GPU_validated'] is False and value['scientific_rows']==0
 assert len(value['variants'])==11
 assert 'Actual bnb NF4' in value['NF4_scope']


def test_actual_cli_refuses_preimported_Torch_before_GPU_query_or_output(tmp_path):
 m=module();output=tmp_path/'never-created'
 with pytest.raises(RuntimeError,match='fresh process'):m.main(['--allow-gpu','--output',str(output)])
 assert not output.exists()


def test_quantized_CPU_fixture_refuses_dense_substitution():
 m=module()
 with pytest.raises(ValueError,match='Real NF4'):m.tiny('4q',device='cpu')
 with pytest.raises(ValueError,match='Real NF4'):m.tiny('1c',device='cpu')


def test_probe_fails_if_frozen_collector_ready_or_sources_change(monkeypatch,tmp_path):
 m=module();changed=tmp_path/'changed-ready.json';changed.write_text('{}')
 monkeypatch.setattr(m.release,'READY',changed)
 with pytest.raises(ValueError,match='Frozen collector'):m.source_binding()


def test_atomic_report_refuses_overwrite_of_incomplete_staging(tmp_path):
 m=module();path=tmp_path/'report.json';m.atomic(path,{'status':'running'})
 assert json.loads(path.read_bytes())=={'status':'running'}
 (tmp_path/'.report.json.tmp').write_text('owned interrupted staging')
 with pytest.raises(FileExistsError):m.atomic(path,{'status':'passed'})
 assert json.loads(path.read_bytes())=={'status':'running'}


def test_dispatch_and_callback_failures_restore_process_local_grouped_method(monkeypatch):
 m=module()
 # Tiny CPU path does not alter Torch's global grouped-mm entry point at all.
 initial=getattr(torch,'_grouped_mm',None)
 monkeypatch.setattr(m.release.routing,'RoutingAggregate',lambda *a,**k:(_ for _ in ()).throw(ValueError('test install refusal')))
 with pytest.raises(ValueError,match='install refusal'):m.pair('3c',device='cpu')
 assert getattr(torch,'_grouped_mm',None) is initial
