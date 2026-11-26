"""Tiny synthetic HF symlink layout; no trained source rewrite or GPU."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.utils import research_lossless_delta as codec
from src.utils import research_trained_delta_probe as frozen
from src.utils import research_trained_delta_probe_v2 as adapter
from test_research_trained_delta_probe import fixture


@pytest.fixture
def cached(fixture, monkeypatch):
    snapshot = fixture['snapshot']; blobs = snapshot.parent/'blobs'; blobs.mkdir()
    for name in ('config.json','model.safetensors.index.json','model.safetensors'):
        path = snapshot/name; target = blobs/name; path.rename(target); path.symlink_to(target)
    # Tiny tests retain the original production implementation/guards. Only the
    # same synthetic element-count substitution as the frozen tests is needed.
    implementation = adapter._implementation
    def tiny():
        module = implementation(); module.BASE_ELEMENTS = frozen.BASE_ELEMENTS; return module
    monkeypatch.setattr(adapter,'_implementation',tiny)
    return fixture


def test_real_isolated_implementation_only_replaces_anchor_reader():
    module = adapter._implementation()
    assert module is not frozen and frozen._anchor_headers is not module._anchor_headers
    assert module._json.__code__.co_code == frozen._json.__code__.co_code
    assert module.gate.__code__.co_code == frozen.gate.__code__.co_code
    assert module.run_probe.__code__.co_code == frozen.run_probe.__code__.co_code
    assert frozen.file_sha(frozen.__file__) == adapter.FROZEN_SHA256


def test_default_closed_gate_no_files_and_no_torch(tmp_path):
    child = subprocess.run([sys.executable,'-c',
        'import sys; from src.utils import research_trained_delta_probe_v2 as a; '
        f"r=a.preflight({str(tmp_path/'absent')!r}); assert r['status']=='gate_closed'; "
        "assert 'torch' not in sys.modules; assert r['adapter_protocol']==a.OWNER"],
        cwd=adapter.ROOT,capture_output=True,text=True)
    assert child.returncode == 0, child.stderr
    assert not list(tmp_path.iterdir())


def test_metadata_symlinks_only_pinned_and_full_census_without_tensor_reads(cached, monkeypatch):
    with pytest.raises(ValueError,match='local bounded JSON'): frozen._anchor_headers(cached['snapshot'])
    def never(*args,**kwargs): raise AssertionError('Tensor data/full shard hashing is forbidden for metadata census')
    monkeypatch.setattr(codec.SmallAnchor,'read',never)
    monkeypatch.setattr(codec.SmallAnchor,'descriptor',never)
    monkeypatch.setattr(codec,'_file_sha',never)
    result = adapter.anchor_census(cached['snapshot'])
    assert result['unique_native_elements'] == 24 and result['stored_bf16_elements'] == 24
    assert result['unique_native_tensors'] == result['stored_tensors'] == 1
    assert not result['tensor_values_read'] and not result['full_shard_bytes_hashed']
    assert result['config']['is_symlink'] and result['index']['is_symlink']
    shard = result['shards']['model.safetensors']
    assert shard['is_symlink'] and shard['header_sha256'] and not shard['full_file_sha256_checked_in_metadata_census']
    assert shard['expected_full_file_sha256'] == codec.SHARD_SHA256['model.safetensors']


@pytest.mark.parametrize('name',['config.json','model.safetensors.index.json'])
def test_changed_canonical_metadata_rejected_before_header_read(cached,name):
    target = (cached['snapshot']/name).resolve(); target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(ValueError,match='metadata bytes/identity'): adapter.anchor_census(cached['snapshot'])


def test_run_json_symlink_still_refused(cached):
    path = cached['run']/'completed.json'; target = path.with_name('other.json'); path.rename(target); path.symlink_to(target)
    result = adapter.preflight(cached['run'],snapshot=cached['snapshot'])
    assert result['status'] == 'gate_closed' and 'local bounded JSON' in result['reason']


def test_complete_tiny_roundtrip_through_same_loader_preserves_roles(cached,tmp_path):
    before = frozen.gate(cached['run']); source_sha = frozen.file_sha(cached['source'])
    report = adapter.run_probe(cached['run'],tmp_path/'derived_v2',snapshot=cached['snapshot'],max_staging_bytes=192*2**20)
    assert report['status'] == 'verified_actual_trained_derived_copy_only'
    assert report['all_payload_fields_and_backing_bits_exact'] and not report['archive_deleted']
    assert report['metadata_adapter']['only_override'] == '_anchor_headers'
    assert report['metadata_adapter']['source_sha256'] == adapter.source_binding()
    assert report['metadata_adapter']['anchor_metadata']['unique_native_elements'] == 24
    assert not report['metadata_adapter']['scientific_publication_authorized']
    assert report['source_payload_sha256'] == report['restored_payload_sha256']
    assert frozen.gate(cached['run']) == before and frozen.file_sha(cached['source']) == source_sha
    assert frozen._json(tmp_path/'derived_v2'/'verified.json') == report


def test_v2_cleanup_requires_durable_extended_metadata_receipt(cached,tmp_path,monkeypatch):
    unlink = Path.unlink; seen = []
    def observed(path,*args,**kwargs):
        if path.name == 'derived_full4f_legacy.pt':
            report = frozen._json(path.parent/'verified.json')
            assert report['metadata_adapter']['protocol'] == adapter.OWNER
            assert report['metadata_adapter']['source_sha256'] == adapter.source_binding()
            assert not report['archive_deleted']; seen.append(path)
        return unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',observed)
    report = adapter.run_probe(cached['run'],tmp_path/'derived_v2',snapshot=cached['snapshot'],
        max_staging_bytes=192*2**20,cleanup_verified=True)
    assert len(seen) == 1 and report['archive_deleted'] and cached['source'].is_file()


@pytest.mark.parametrize('mutation',['bad_dtype','offset_gap','missing_key','duplicate_json_key'])
def test_malformed_header_refused_without_values(cached,mutation):
    import struct
    path = (cached['snapshot']/'model.safetensors').resolve(); raw = path.read_bytes()
    size = struct.unpack('<Q',raw[:8])[0]; header = json.loads(raw[8:8+size]); key = next(iter(header))
    if mutation == 'bad_dtype': header[key]['dtype'] = 'F32'
    if mutation == 'offset_gap': header[key]['data_offsets'] = [2,50]
    if mutation == 'missing_key': header = {}
    serialized = json.dumps(header).encode()
    if mutation == 'duplicate_json_key': serialized = b'{"duplicate":{},"duplicate":{}}'
    path.write_bytes(struct.pack('<Q',len(serialized))+serialized+raw[8+size:])
    with pytest.raises(ValueError): adapter.anchor_census(cached['snapshot'])


def test_wrong_revision_refused(cached):
    with pytest.raises(ValueError,match='revision'): adapter.anchor_census(cached['snapshot'].parent)


def test_frozen_file_binding_tamper_refused(monkeypatch):
    original = frozen.file_sha
    monkeypatch.setattr(frozen,'file_sha',lambda path:'0'*64 if Path(path)==Path(frozen.__file__) else original(path))
    with pytest.raises(ValueError,match='Frozen'): adapter.source_binding()
