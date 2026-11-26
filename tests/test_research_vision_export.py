"""Tiny CPU serialization/crash fixtures; actual native parity is a separate run."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import src.utils.research_vision_export as export


@pytest.fixture
def source(tmp_path,monkeypatch):
    snapshot=tmp_path/'source';snapshot.mkdir()
    specification=export.DONORS['qwen25vision']
    for name in specification['required_assets']:(snapshot/name).write_text('{}')
    (snapshot/'config.json').write_text(json.dumps({'model_type':'qwen2_5_vl'}))
    (snapshot/'vocab.json').write_text('{"a":0}')
    (snapshot/'private-unrelated-file.txt').write_text('do not copy')
    tensors={'visual.proj.weight':torch.tensor([[0.,-0.,1.],[2.,-3.,4.]],dtype=torch.bfloat16),
             'visual.scalar':torch.tensor(.75,dtype=torch.float32),
             'model.language_model.weight':torch.ones(5,6,dtype=torch.bfloat16)}
    save_file(tensors,str(snapshot/'mixed.safetensors'))
    index={'weight_map':{key:'mixed.safetensors' for key in tensors}}
    (snapshot/'model.safetensors.index.json').write_text(json.dumps(index))
    shapes={key:list(value.shape) for key,value in tensors.items() if key.startswith('visual.')}
    monkeypatch.setattr(export,'_expected_shapes',lambda snapshot,backend,prefixes:copy.deepcopy(shapes))
    plan=export.plan_export(snapshot,'qwen25vision','a'*40)
    return snapshot,plan,tensors,shapes


def destination(tmp_path):return tmp_path/'derived'


def test_exact_vision_only_raw_bits_dtype_shapes_index_and_controlled_asset_copy(source,tmp_path):
    snapshot,plan,tensors,_=source
    before={path.name:path.read_bytes() for path in snapshot.iterdir()}
    result=export.export_vision_snapshot(plan,destination(tmp_path),reserve_gb=0)
    path=Path(result['path']);manifest=export.verify_derived_artifact(path,plan)
    assert not result['reused'] and manifest['artifact_scope']==export.SCOPE
    assert manifest['native_validation']['status']=='not_run'
    assert manifest['source_files_deleted'] is False
    with safe_open(str(path/'model.safetensors'),framework='pt') as reader:
        assert set(reader.keys())==set(plan['tensor_headers'])
        for name in reader.keys():
            copied=reader.get_tensor(name)
            assert copied.dtype==tensors[name].dtype and copied.shape==tensors[name].shape
            assert export.tensor_sha(copied)==export.tensor_sha(tensors[name])
    index=json.loads((path/'model.safetensors.index.json').read_text())
    assert index['weight_map']=={key:'model.safetensors' for key in plan['tensor_headers']}
    assert not (path/'private-unrelated-file.txt').exists()
    assert all(not entry.is_symlink() for entry in path.iterdir())
    assert {entry.name:entry.read_bytes() for entry in snapshot.iterdir()}==before
    assert manifest['artifact_total_bytes']==sum(entry.stat().st_size for entry in path.iterdir())


def test_immutable_idempotence_reuses_only_verified_identical_payload(source,tmp_path):
    _,plan,_,_=source
    first=export.export_vision_snapshot(plan,destination(tmp_path),reserve_gb=0)
    path=Path(first['path']);before={entry.name:export.file_sha(entry) for entry in path.iterdir()}
    second=export.export_vision_snapshot(plan,destination(tmp_path),reserve_gb=0)
    assert second['reused'] and second['manifest']==first['manifest']
    assert {entry.name:export.file_sha(entry) for entry in path.iterdir()}==before
    (path/'config.json').write_text('corrupted')
    with pytest.raises(ValueError,match='checksum'):export.export_vision_snapshot(plan,destination(tmp_path),reserve_gb=0)
    assert (path/'config.json').read_text()=='corrupted'


def test_missing_required_tensor_and_ambiguous_prefix_rejected_before_export(source,tmp_path):
    snapshot,_,tensors,_=source
    changed={key:value for key,value in tensors.items() if key!='visual.scalar'}
    save_file(changed,str(snapshot/'mixed.safetensors'))
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{key:'mixed.safetensors' for key in changed}}))
    with pytest.raises(ValueError,match='missing'):export.plan_export(snapshot,'qwen25vision','a'*40)
    changed=dict(tensors);changed['model.visual.other']=torch.ones(1)
    save_file(changed,str(snapshot/'mixed.safetensors'))
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{key:'mixed.safetensors' for key in changed}}))
    with pytest.raises(ValueError,match='prefix group'):export.plan_export(snapshot,'qwen25vision','a'*40)


def test_source_index_cannot_hide_an_unmapped_vision_tensor(source):
    snapshot,_,_,_=source
    index=json.loads((snapshot/'model.safetensors.index.json').read_text())
    index['weight_map'].pop('visual.scalar')
    (snapshot/'model.safetensors.index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError,match='index/header'):export.plan_export(snapshot,'qwen25vision','a'*40)


def test_unexpected_language_tensor_rejected_even_if_file_checksum_is_rewritten(source,tmp_path):
    _,plan,_,_=source
    result=export.export_vision_snapshot(plan,destination(tmp_path),reserve_gb=0);path=Path(result['path'])
    with safe_open(str(path/'model.safetensors'),framework='pt') as reader:tensors={name:reader.get_tensor(name).clone() for name in reader.keys()}
    tensors['model.language_model.weight']=torch.ones(1)
    save_file(tensors,str(path/'model.safetensors'))
    manifest=json.loads((path/'derivation.json').read_text())
    manifest['output_files']['model.safetensors']['sha256']=export.file_sha(path/'model.safetensors')
    (path/'derivation.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='non-vision'):export.verify_derived_artifact(path,plan)


@pytest.mark.parametrize('phase',['after_weights','before_publish'])
def test_source_change_aborts_without_publishing_or_deleting_source(source,tmp_path,phase):
    snapshot,plan,tensors,_=source
    def mutate(event,stage):
        if event!=phase:return
        other=snapshot/'replacement.safetensors'
        changed={key:value.clone() for key,value in tensors.items()};changed['visual.scalar']+=1
        save_file(changed,str(other));os.replace(other,snapshot/'mixed.safetensors')
    with pytest.raises(ValueError,match='source changed'):
        export.export_vision_snapshot(plan,destination(tmp_path),reserve_gb=0,_fault_hook=mutate)
    assert (snapshot/'mixed.safetensors').is_file()
    assert not (destination(tmp_path)/('qwen25vision-'+'a'*40)).exists()
    assert not list(destination(tmp_path).glob('*.staging-*'))


@pytest.mark.parametrize('phase',['after_weights','before_publish'])
def test_actual_process_crash_leaves_no_published_partial_and_recovers_only_owned_staging(source,tmp_path,phase):
    snapshot,plan,_,shapes=source;out=destination(tmp_path)
    code='''
import json,os,sys
from pathlib import Path
import src.utils.research_vision_export as e
shapes=json.loads(sys.argv[3]);e._expected_shapes=lambda *args:shapes
p=e.plan_export(sys.argv[1],'qwen25vision','a'*40)
def crash(event,stage):
    if event==sys.argv[4]:os._exit(73)
e.export_vision_snapshot(p,sys.argv[2],reserve_gb=0,_fault_hook=crash)
'''
    result=subprocess.run([sys.executable,'-c',code,str(snapshot),str(out),json.dumps(shapes),phase],
                          cwd=export.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=30)
    assert result.returncode==73,result.stderr
    assert not (out/('qwen25vision-'+'a'*40)).exists()
    assert list(out.glob('*.staging-*'))
    recovered=export.export_vision_snapshot(plan,out,reserve_gb=0)
    assert Path(recovered['path']).is_dir() and not list(out.glob('*.staging-*'))


def test_unknown_staging_content_preserved_and_existing_destination_never_overwritten(source,tmp_path):
    _,plan,_,_=source;out=destination(tmp_path);out.mkdir()
    unknown=out/('.qwen25vision-'+'a'*40+'.staging-foreign');unknown.mkdir();(unknown/'user.txt').write_text('preserve')
    with pytest.raises(ValueError,match='Unknown staging'):export.export_vision_snapshot(plan,out,reserve_gb=0)
    assert (unknown/'user.txt').read_text()=='preserve'
    source_dir=tmp_path/'publish-source';source_dir.mkdir();(source_dir/'new.txt').write_text('new')
    target=tmp_path/'publish-target';target.mkdir();(target/'old.txt').write_text('old')
    with pytest.raises(FileExistsError):export._publish_no_replace(source_dir,target)
    assert (target/'old.txt').read_text()=='old' and (source_dir/'new.txt').read_text()=='new'


def test_budget_reserves_full_new_payload_before_any_tensor_write(source,tmp_path,monkeypatch):
    _,plan,_,_=source;calls=[]
    def reject(self,anticipated_bytes=0):
        calls.append(anticipated_bytes)
        if anticipated_bytes:raise RuntimeError('budget rejected')
        return {}
    monkeypatch.setattr(export.DiskBudget,'ensure_space',reject)
    with pytest.raises(RuntimeError,match='budget rejected'):
        export.export_vision_snapshot(plan,destination(tmp_path),reserve_gb=120)
    assert plan['anticipated_new_peak_bytes'] in calls
    assert not list(destination(tmp_path).glob('*.staging-*'))


def test_production_validator_cannot_reuse_an_unvalidated_serialization_fixture(source,tmp_path):
    _,plan,_,_=source
    export.export_vision_snapshot(plan,destination(tmp_path),reserve_gb=0)
    with pytest.raises(ValueError,match='no passed native parity'):
        export.export_vision_snapshot(plan,destination(tmp_path),reserve_gb=0,validator=lambda *args:{'status':'passed'})


def test_export_cannot_write_into_original_source_tree(source):
    snapshot,plan,_,_=source
    with pytest.raises(ValueError,match='outside'):export.export_vision_snapshot(plan,snapshot/'derived',reserve_gb=0)


def test_gemma_monolithic_source_requires_both_prefixes_and_creates_only_new_vision_index(tmp_path,monkeypatch):
    snapshot=tmp_path/'gemma-source';snapshot.mkdir()
    for name in export.DONORS['gemma4vision']['required_assets']:(snapshot/name).write_text('{}')
    tensors={'model.vision_tower.weight':torch.ones(2,3,dtype=torch.bfloat16),
             'model.embed_vision.weight':torch.tensor([-0.,1.],dtype=torch.float32),
             'model.language_model.weight':torch.ones(3,3),
             'model.audio_tower.weight':torch.zeros(3,3)}
    shapes={key:list(value.shape) for key,value in tensors.items() if key.startswith(('model.vision_tower.','model.embed_vision.'))}
    monkeypatch.setattr(export,'_expected_shapes',lambda *args:shapes)
    save_file(tensors,str(snapshot/'model.safetensors'))
    plan=export.plan_export(snapshot,'gemma4vision','b'*40)
    assert plan['source']['source_index_present'] is False
    result=export.export_vision_snapshot(plan,tmp_path/'derived',reserve_gb=0)
    index=json.loads((Path(result['path'])/'model.safetensors.index.json').read_text())
    assert set(index['weight_map'])==set(shapes)
    tensors.pop('model.embed_vision.weight');save_file(tensors,str(snapshot/'model.safetensors'))
    with pytest.raises(ValueError,match='prefix group'):export.plan_export(snapshot,'gemma4vision','b'*40)


def test_qwen_alternate_native_model_visual_prefix_is_preserved(source,tmp_path,monkeypatch):
    snapshot,_,tensors,_=source
    tensors={key.replace('visual.','model.visual.',1) if key.startswith('visual.') else key:value for key,value in tensors.items()}
    shapes={key:list(value.shape) for key,value in tensors.items() if key.startswith('model.visual.')}
    monkeypatch.setattr(export,'_expected_shapes',lambda *args:shapes)
    save_file(tensors,str(snapshot/'mixed.safetensors'))
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{key:'mixed.safetensors' for key in tensors}}))
    plan=export.plan_export(snapshot,'qwen25vision','a'*40)
    assert plan['source']['prefixes']==['model.visual.']
    result=export.export_vision_snapshot(plan,destination(tmp_path),reserve_gb=0)
    assert set(result['manifest']['tensors'])==set(shapes)


def test_incomplete_parity_claim_cannot_publish_a_production_validated_artifact(source,tmp_path):
    _,plan,_,_=source
    with pytest.raises(ValueError,match='complete proof'):
        export.export_vision_snapshot(plan,destination(tmp_path),reserve_gb=0,validator=lambda *args:{'status':'passed'})
    assert not (destination(tmp_path)/('qwen25vision-'+'a'*40)).exists()


def test_cgroup_limit_wins_over_large_host_memory_and_cache_is_not_credited(tmp_path):
    meminfo=tmp_path/'meminfo';meminfo.write_text(f'MemTotal: {251*2**20} kB\nMemAvailable: {200*2**20} kB\n')
    cgroup=tmp_path/'cgroup';cgroup.mkdir()
    (cgroup/'memory.max').write_text(str(170*2**30));(cgroup/'memory.current').write_text(str(160*2**30))
    (cgroup/'memory.stat').write_text('inactive_file '+str(130*2**30)+'\n')
    with pytest.raises(MemoryError,match='Cgroup remaining'):
        export.cpu_memory_budget(2**30,meminfo_path=meminfo,cgroup_root=cgroup)
    (cgroup/'memory.current').write_text(str(50*2**30))
    result=export.cpu_memory_budget(2**30,meminfo_path=meminfo,cgroup_root=cgroup)
    assert result['cgroup_limit_bytes']==170*2**30
    assert result['required_available_bytes']==33*2**30 and result['cache_credit_applied'] is False
    meminfo.write_text(f'MemTotal: {251*2**20} kB\nMemAvailable: {20*2**20} kB\n')
    with pytest.raises(MemoryError,match='MemAvailable'):
        export.cpu_memory_budget(2**30,meminfo_path=meminfo,cgroup_root=cgroup)
