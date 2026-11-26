"""Tiny SYNTHETIC managed fixtures; no actual scientific checkpoint is changed."""
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from src.utils import research_lossless_delta as v1, research_stream_delta as stream
from src.utils import research_trained_delta_probe as probe
from src.utils.research_checkpoint_compat import load_research_payload


def put(path, value):
    Path(path).write_bytes(probe.canonical(value)+b'\n')


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    snapshot=tmp_path/v1.REVISION; snapshot.mkdir()
    key='model.layers.0.self_attn.q_proj.weight'; original=torch.linspace(-1,1,24).reshape(4,6).bfloat16()
    save_file({key:original},snapshot/'model.safetensors')
    put(snapshot/'config.json',{'model_type':'smallthinker','tie_word_embeddings':False})
    put(snapshot/'model.safetensors.index.json',{'weight_map':{key:'model.safetensors'}})
    monkeypatch.setattr(v1,'CONFIG_SHA256',probe.file_sha(snapshot/'config.json'))
    monkeypatch.setattr(v1,'INDEX_SHA256',probe.file_sha(snapshot/'model.safetensors.index.json'))
    monkeypatch.setattr(v1,'SHARD_SHA256',{'model.safetensors':probe.file_sha(snapshot/'model.safetensors')})
    monkeypatch.setattr(stream,'ANCHOR_SHARD_BYTES',{'model.safetensors':(snapshot/'model.safetensors').stat().st_size})
    monkeypatch.setattr(probe,'BASE_ELEMENTS',24)
    run=tmp_path/'completed_full4f'; run.mkdir(); (run/'.checkpoint.lock').touch()
    objects=run/'.checkpoint_objects';objects.mkdir();target=objects/'object-raw';target.mkdir()
    config={'model':v1.REPO,'recipe':'4f','stage':'SFT','checkpoint_format':'legacy','output':str(run)}
    value=original.float()+torch.arange(24).reshape(4,6)*1e-7
    value.view(torch.int32).flatten()[:4]=torch.tensor([0,-2147483648,2143289345,2143289346],dtype=torch.int32)
    backing=torch.arange(31,dtype=torch.int16)
    payload={'model':{'llm.'+key:value,'projector.weight':torch.randn(3,5)},
        'scheduler':{'_step_count':171,'base_lrs':[1e-4], 'nested':{'int_keys':{1:[True,None]}}},
        'rng_torch':torch.get_rng_state(), 'rng_numpy':np.random.get_state(), 'rng_python':(3,(1,2,3),None),
        'rng_cuda':[torch.arange(12,dtype=torch.uint8)],'step':170,'seen':10831,'active_microbatch_size':4,
        'config':config,'model_ref':{'repo':v1.REPO,'revision':v1.REVISION},
        'vision_ref':{'repo':'fixture/frozen_vision','revision':'a'*40}, 'cache_fingerprint':'b'*64,
        'normal_backing':backing,'normal_view':backing[3:12], 'normal_same_object':backing,
        'negative_zero':-0., 'metadata_bytes':b'\x00\xff', 'training_completed':True,'resumable_optimizer':False,
        'checkpoint_compaction':{'operation':'remove_completed_optimizer_only','source_object':'object-old','source_state_sha256':'d'*64}}
    source=target/'state.pt';torch.save(payload,source)
    marker={'owner':'fixture_owner','step':170,'metric':31.2,'created_unix':0.,'training_completed':True,
        'resumable_optimizer':False,'optimizer_state_removed':True,'source_checkpoint_object':'object-old'}
    put(target/'.research_checkpoint.json',marker)
    (run/'best').symlink_to(Path('.checkpoint_objects')/target.name,target_is_directory=True)
    (run/'last').symlink_to(Path('.checkpoint_objects')/target.name,target_is_directory=True)
    put(run/'resolved_config.json',config);put(run/'completed.json',{'steps':170,'records':10831})
    put(run/'.checkpoint_manager.json',{'owner':'fixture_owner','mode':'max','schema_version':1})
    put(run/'checkpoint_compaction.json',{'status':'completed','run_dir':str(run),'training_completed':True,
        'resumable_optimizer':False,'checkpoints':[{'new_object':target.name}]})
    return {'run':run,'snapshot':snapshot,'source':source,'payload':payload,'key':'llm.'+key,'marker':target/'.research_checkpoint.json'}


def execute(fixture,tmp_path,**kwargs):
    return probe.run_probe(fixture['run'],tmp_path/'derived',snapshot=fixture['snapshot'],max_staging_bytes=192*2**20,**kwargs)


def test_default_closed_gate_no_import_torch_and_no_files(tmp_path):
    command=[sys.executable,'-c',
        "import sys; from src.utils.research_trained_delta_probe import preflight; "
        f"r=preflight({str(tmp_path/'absent')!r}); assert r['status']=='gate_closed'; assert 'torch' not in sys.modules"]
    result=subprocess.run(command,cwd=probe.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not list(tmp_path.iterdir())


def test_open_preflight_is_structural_not_archive_or_quality(fixture,tmp_path):
    before=probe.file_sha(fixture['source'])
    result=probe.preflight(fixture['run'],snapshot=fixture['snapshot'])
    assert result['status']=='preflight_gate_open_only' and not result['archive_created'] and not result['payload_values_verified']
    assert result['census']['original_fp32_bytes']==96
    assert result['writer_extra_before_source_values_bytes']>=fixture['source'].stat().st_size
    assert result['source_mmap_file_pages_charged'] and result['memory_headroom_bytes']==32*2**30
    assert probe.file_sha(fixture['source'])==before and not (tmp_path/'derived').exists()


def test_actual_tiny_native_roundtrip_all_payload_fields_and_bits(fixture,tmp_path):
    expected=probe.payload_signature(fixture['payload']); before=probe.gate(fixture['run'])
    source_sha=probe.file_sha(fixture['source']);report=execute(fixture,tmp_path)
    assert report['status']=='verified_actual_trained_derived_copy_only'
    assert report['source_payload_sha256']==expected['sha256']==report['restored_payload_sha256']
    assert report['all_payload_fields_and_backing_bits_exact'] and report['scheduler_rng_config_compaction_metadata_preserved']
    assert not report['source_modified'] and not report['manager_roles_modified'] and not report['optimizer_removed_by_probe']
    assert not report['gpu_reconstruction_validated'] and not report['cuda_initialized']
    restored=load_research_payload(report['archive_path'],map_location='cpu',mmap=True)
    assert probe.payload_signature(restored)==expected
    assert restored['normal_same_object'] is restored['normal_backing']
    assert restored['normal_view'].storage_offset()==3
    assert restored['normal_view'].untyped_storage()._cdata==restored['normal_backing'].untyped_storage()._cdata
    assert list(restored['model'])==list(fixture['payload']['model'])
    assert probe.gate(fixture['run'])==before and probe.file_sha(fixture['source'])==source_sha
    assert (tmp_path/'derived'/'verified.json').is_file() and (tmp_path/'derived'/'memory_samples.json').is_file()


def test_verified_cleanup_deletes_only_owned_derived_after_durable_receipt(fixture,tmp_path,monkeypatch):
    original=Path.unlink; seen=[]
    def unlink(self,*args,**kwargs):
        if self.name=='derived_full4f_legacy.pt':
            report=json.loads((self.parent/'verified.json').read_text())
            assert report['archive_sha256']==probe.file_sha(self) and report['archive_deleted'] is False
            assert (self.parent/'owner.json').is_file(); seen.append(str(self))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',unlink)
    report=execute(fixture,tmp_path,cleanup_verified=True)
    assert report['archive_deleted'] and len(seen)==1 and not Path(report['archive_path']).exists()
    assert fixture['source'].is_file()


@pytest.mark.parametrize('field,value',[('recipe','4q'),('stage','A1'),('model','other/model'),('checkpoint_format','bundle')])
def test_wrong_training_profile_gate_closed(fixture,field,value):
    path=fixture['run']/'resolved_config.json';config=json.loads(path.read_text());config[field]=value;put(path,config)
    assert probe.preflight(fixture['run'],snapshot=fixture['snapshot'])['status']=='gate_closed'


@pytest.mark.parametrize('mutation',['completion_missing','optimizer_not_compacted','foreign_owner','missing_compaction_entry','last_step_wrong','newer_checkpoint'])
def test_incomplete_or_unowned_source_gate_closed(fixture,mutation):
    run=fixture['run'];marker=json.loads(fixture['marker'].read_text())
    if mutation=='completion_missing':(run/'completed.json').unlink()
    if mutation=='optimizer_not_compacted':marker['resumable_optimizer']=True
    if mutation=='foreign_owner':marker['owner']='foreign'
    if mutation=='missing_compaction_entry':
        report=json.loads((run/'checkpoint_compaction.json').read_text());report['checkpoints']=[];put(run/'checkpoint_compaction.json',report)
    if mutation=='last_step_wrong':marker['step']=169
    if mutation=='newer_checkpoint':marker['created_unix']=10**20
    put(fixture['marker'],marker)
    assert probe.preflight(run,snapshot=fixture['snapshot'])['status']=='gate_closed'


@pytest.mark.parametrize('mutation',['missing_key','extra_key','bf16','alias','optimizer'])
def test_partial_dtype_alias_or_optimizer_payload_never_accepted(fixture,mutation,tmp_path):
    payload=fixture['payload'];key=fixture['key']
    if mutation=='missing_key':payload['model'].pop(key)
    if mutation=='extra_key':payload['model']['llm.extra']=torch.ones(2)
    if mutation=='bf16':payload['model'][key]=payload['model'][key].bfloat16()
    if mutation=='alias':payload['bad_model_alias']=payload['model'][key]
    if mutation=='optimizer':payload['optimizer']={'state':{}}
    torch.save(payload,fixture['source'])
    with pytest.raises(ValueError):execute(fixture,tmp_path)
    assert not (tmp_path/'derived').exists()


def test_target_rng_corruption_is_detected_full_payload_not_only_model(fixture,tmp_path,monkeypatch):
    from src.utils import research_checkpoint_compat as compat
    original=compat.load_research_payload
    def load(path,**kwargs):
        result=original(path,**kwargs)
        if Path(path).name=='derived_full4f_legacy.pt':result['rng_torch'][0]^=1
        return result
    monkeypatch.setattr(compat,'load_research_payload',load)
    before=probe.file_sha(fixture['source'])
    with pytest.raises(ValueError,match='Complete payload'):execute(fixture,tmp_path)
    assert (tmp_path/'derived'/'failure.json').is_file() and not (tmp_path/'derived'/'verified.json').exists()
    assert (tmp_path/'derived'/'derived_full4f_legacy.pt').is_file()
    assert probe.file_sha(fixture['source'])==before


def test_source_mmap_release_before_cache_advice_and_decoder(fixture,tmp_path,monkeypatch):
    calls=[]
    def advice(fd,offset,length,flag):
        assert probe.stat(Path(f'/proc/self/fd/{fd}'))==probe.stat(fixture['source'])
        assert flag==probe.os.POSIX_FADV_DONTNEED and offset==length==0
        probe._no_source_mappings(fixture['source']);calls.append(fd)
    monkeypatch.setattr(probe.os,'posix_fadvise',advice)
    report=execute(fixture,tmp_path,advise_source=True)
    assert len(calls)==1 and report['cache_advice']['source_stat_unchanged']


def test_live_source_mmap_refuses_advice(fixture):
    payload=load_research_payload(fixture['source'],map_location='cpu',mmap=True)
    with pytest.raises(ValueError,match='still live'):probe.advise_source_cache(fixture['source'],probe.stat(fixture['source']))
    del payload;gc.collect();probe._no_source_mappings(fixture['source'])


def test_mapping_census_detects_same_inode_through_different_hardlink(fixture,tmp_path):
    alias=tmp_path/'same inode with spaces.pt';probe.os.link(fixture['source'],alias)
    payload=load_research_payload(alias,map_location='cpu',mmap=True)
    with pytest.raises(ValueError,match='still live'):probe._no_source_mappings(fixture['source'])
    del payload;gc.collect();probe._no_source_mappings(fixture['source'])


def test_existing_output_and_manager_namespace_refused(fixture,tmp_path):
    output=tmp_path/'derived';output.mkdir();(output/'keep').write_text('keep')
    with pytest.raises(FileExistsError):execute(fixture,tmp_path)
    assert (output/'keep').read_text()=='keep'
    with pytest.raises(ValueError,match='separate'):
        probe.run_probe(fixture['run'],fixture['run']/'derived',snapshot=fixture['snapshot'])


def test_hard_writer_cap_preserves_source_and_failed_probe(fixture,tmp_path,monkeypatch):
    original=probe._BoundedWriter
    class TinyCap(original):
        def __init__(self,raw,budget,maximum):super().__init__(raw,budget,32)
    monkeypatch.setattr(probe,'_BoundedWriter',TinyCap)
    before=probe.file_sha(fixture['source'])
    with pytest.raises(ValueError,match='hard staging cap'):execute(fixture,tmp_path)
    assert probe.file_sha(fixture['source'])==before
    assert (tmp_path/'derived'/'failure.json').is_file() and not (tmp_path/'derived'/'verified.json').exists()


def test_signature_catches_hidden_backing_bytes_scheduler_and_float_bits(fixture):
    payload=fixture['payload']; first=probe.payload_signature(payload)
    payload['normal_backing'][0]=99
    assert probe.payload_signature(payload)['sha256']!=first['sha256']
    payload['normal_backing'][0]=0
    assert probe.payload_signature(payload)==first
    payload['negative_zero']=0.
    assert probe.payload_signature(payload)['sha256']!=first['sha256']


def test_closed_gate_run_flag_refuses_without_creating_output(tmp_path):
    with pytest.raises(probe.GateClosed):probe.run_probe(tmp_path/'absent',tmp_path/'derived')
    assert not (tmp_path/'derived').exists()
