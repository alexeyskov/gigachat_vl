"""CPU-only admission/namespace/composition checks; actual CUDA proof is ROOT-owned."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('derived_cuda_probe',ROOT/'scripts/probe_research_derived_vision_cuda_v1.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_default_actual_metadata_opens_no_original_weight_body_and_no_torch(monkeypatch):
    original=Path.open
    def checked(path,*args,**kwargs):
        if path.suffix=='.safetensors' and '.hf_home' in str(path):
            raise AssertionError('Metadata must not open original mixed weight bodies')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',checked)
    value=probe.plan()
    assert value['status']=='metadata_preflight_only' and value['paired_records']==4
    assert value['pruning_authorized'] is False and value['scientific_caption_score_claimed'] is False
    assert value['optimizer_updates_per_donor']==1
    assert {x['binding']['export_ref']['feature_dim'] for x in value['donors']}=={3584,2560}


def test_actual_default_cli_imports_no_torch_and_creates_no_gpu_job():
    source=("import runpy,sys;sys.argv=['probe'];runpy.run_path('scripts/probe_research_derived_vision_cuda_v1.py',run_name='__main__');"
            "assert 'torch' not in sys.modules")
    result=subprocess.run([sys.executable,'-c',source],cwd=ROOT,check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)['gpu_used'] is False


def test_existing_output_and_occupied_gpu_refused_before_model_or_output_creation(tmp_path,monkeypatch):
    existing=tmp_path/'old';existing.mkdir();(existing/'evidence').write_bytes(b'preserve')
    with pytest.raises(ValueError,match='NEW'):
        probe.run(SimpleNamespace(output=existing))
    assert (existing/'evidence').read_bytes()==b'preserve'
    monkeypatch.setattr(probe,'processes',lambda:{123456:1})
    new=tmp_path/'never_created'
    with pytest.raises(RuntimeError,match='occupied'):
        probe.run(SimpleNamespace(output=new))
    assert not new.exists()


def test_cache_reopen_uses_actual_existing_namespace_and_forces_disk_roundtrip(tmp_path):
    import torch
    from src.dataset.vision_cache import VisionFeatureCache
    original=VisionFeatureCache(tmp_path/'own','fixed',reserve_bytes=0,ram_max_bytes=1024)
    features=torch.arange(24,dtype=torch.bfloat16).reshape(3,8)
    original.get_many(['synthetic'],lambda _: [features],image_hashes=['rgb'])
    reopened=probe.reopen_disk_cache(original,VisionFeatureCache)
    # Respect the real parent filesystem reserve without writing more features.
    assert reopened.root==original.root and reopened.ram_max_bytes==0
    restored=reopened.get_many(['synthetic'],lambda _:(_ for _ in ()).throw(AssertionError('Musthitdisk')),
        image_hashes=['rgb'],read_only=True)
    assert torch.equal(restored[0],features)
    assert reopened.stats['hits']==1 and reopened.stats['encode_calls']==0


def test_private_stock_derived_composition_refuses_bad_profile_before_cuda_and_keeps_canonical():
    import torch
    value=probe.plan();binding=value['donors'][0]['binding']
    canonical=sys.modules.get('scripts.run_research_stage')
    module=probe.composed_runner(binding)
    assert sys.modules.get('scripts.run_research_stage') is canonical
    args=SimpleNamespace(model=probe.stock.SMALL,vision=binding['upstream_ref']['repo'],
        vision_preprocessing_profile=binding['profile_ref']['name'],numerical_profile={'wrong':True})
    with pytest.raises(ValueError,match='numerical profile'):
        module.ResearchModel(args)
    assert not torch.cuda.is_initialized()


def test_changed_fixture_refused_without_touching_fixed_source(tmp_path,monkeypatch):
    from PIL import Image
    p=tmp_path/'changed.png';Image.new('RGB',(112,84),'red').save(p)
    monkeypatch.setattr(probe,'FIXTURES',(str(p),probe.FIXTURES[1]))
    with pytest.raises(ValueError,match='fixtures changed'):
        probe.plan()
