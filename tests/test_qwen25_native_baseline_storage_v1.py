"""Destructive cleanup is confined to owned temporary weights, never shared HF files."""
import hashlib,json,os
from pathlib import Path
import pytest
from scripts import retire_qwen25_native_baseline_source_v1 as module


def fixture(tmp_path,monkeypatch):
    out=tmp_path/'results';out.mkdir();source=tmp_path/'cache/baselines/qwen25_native_int4_v1/source';source.mkdir(parents=True)
    owner=dict(owner='qwen25_native_int4_internal116_v1',repo='Qwen/Qwen2.5-VL-7B-Instruct',revision='a'*40)
    (source/'.baseline_owner.json').write_text(json.dumps(owner));files=[]
    for i in range(5):
        path=source/f'model-{i:05}-of-00005.safetensors';path.write_bytes(b'weight'+bytes([i]))
        files.append(dict(name=path.name,path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),borrowed_shared_file=False))
    shared=tmp_path/'shared.json';shared.write_text('{}');link=source/'config.json';link.symlink_to(shared)
    files.append(dict(name=link.name,path=str(link),bytes=2,sha256=hashlib.sha256(b'{}').hexdigest(),borrowed_shared_file=True))
    manifest=out/'source_content_hashes.json';manifest.write_text(json.dumps(dict(repo=owner['repo'],revision=owner['revision'],files=files)))
    journal=out/'predictions.jsonl';journal.write_text('{}\n'*116)
    (out/'protocol.json').write_text(json.dumps(dict(source_content_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))
    (out/'summary.json').write_text(json.dumps(dict(status='completed',records=116,predictions_sha256=hashlib.sha256(journal.read_bytes()).hexdigest())))
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'OUT',out)
    from scripts import research_archival_stock_queue_v5
    monkeypatch.setattr(research_archival_stock_queue_v5,'scan_anchor_users',lambda ids:[])
    return source,shared,out


def test_private_shards_retire_but_shared_metadata_remains(tmp_path,monkeypatch):
    source,shared,out=fixture(tmp_path,monkeypatch);result=module.retire()
    assert result['files']==5 and result['protected_shared_files_deleted']==0
    assert not list(source.glob('*.safetensors'))
    assert shared.read_text()=='{}' and (source/'config.json').is_symlink()
    assert (out/'predictions.jsonl').exists() and (out/'source_content_hashes.json').exists()


def test_shared_hardlink_refuses_entire_retirement(tmp_path,monkeypatch):
    source,shared,out=fixture(tmp_path,monkeypatch)
    os.link(next(source.glob('*.safetensors')),tmp_path/'shared_weight')
    with pytest.raises(ValueError,match='shared link'):module.retire()
    assert len(list(source.glob('*.safetensors')))==5


def test_incomplete_baseline_cannot_retire_weights(tmp_path,monkeypatch):
    source,shared,out=fixture(tmp_path,monkeypatch)
    summary=json.loads((out/'summary.json').read_bytes());summary['records']=115
    (out/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='Complete116'):module.retire()
    assert len(list(source.glob('*.safetensors')))==5
