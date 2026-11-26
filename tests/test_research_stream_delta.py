"""Streaming prefix guard ordering on tiny native Torch legacy files."""
import hashlib
import io
import importlib.util
import json
import pickle
import pickletools
from pathlib import Path
import sys
import subprocess
import os

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from src.utils import research_lossless_delta as v1
from src.utils import research_stream_delta as stream
from src.utils.research_checkpoint_compat import load_research_payload


@pytest.fixture
def anchor(tmp_path, monkeypatch):
    path = tmp_path / v1.REVISION
    path.mkdir()
    value = torch.randn(512, 512).bfloat16()
    key = "model.layers.0.self_attn.q_proj.weight"
    second = key.replace(".q_proj.", ".k_proj.")
    save_file({key: value, second: value + 1}, path / "model.safetensors")
    (path / "config.json").write_text('{"model_type":"smallthinker"}')
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "model.safetensors", second: "model.safetensors"}}))
    monkeypatch.setattr(v1, "CONFIG_SHA256", hashlib.sha256((path / "config.json").read_bytes()).hexdigest())
    monkeypatch.setattr(v1, "INDEX_SHA256", hashlib.sha256((path / "model.safetensors.index.json").read_bytes()).hexdigest())
    monkeypatch.setattr(v1, "SHARD_SHA256", {"model.safetensors": hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()})
    monkeypatch.setattr(stream, "ANCHOR_SHARD_BYTES", {"model.safetensors": (path / "model.safetensors").stat().st_size})
    return path, key, value


def make_payload(anchor):
    _, key, value = anchor
    tensor = value.float() + torch.randn_like(value, dtype=torch.float32) * 1e-4
    return {"model_ref": {"repo": v1.REPO, "revision": v1.REVISION},
            "model": {"llm." + key: tensor, "projector.weight": torch.randn(16, 16)},
            "optimizer": {"state": {0: {"exp_avg": torch.randn(16, 16), "exp_avg_sq": torch.ones(16, 16)}}},
            "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(), "step": 170}


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert type(b) is torch.Tensor and a.shape == b.shape and a.dtype == b.dtype
        assert v1._raw(a) == v1._raw(b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def prepared(anchor):
    payload = make_payload(anchor)
    encoded, _ = v1.prepare_delta_payload(payload, anchor[0])
    return payload, encoded


def test_native_unchanged_loader_exact_and_no_frames_or_zip(anchor, tmp_path):
    payload, encoded = prepared(anchor)
    p = tmp_path / "stream.pt"
    header = stream.save_guarded_legacy(encoded, p)
    assert not p.read_bytes().startswith(b"PK")
    # Torch legacy's five pickle records precede ordinary raw storages.
    file = io.BytesIO(p.read_bytes())
    for _ in range(5):
        start = file.tell()
        ops = list(pickletools.genops(file))
        assert not any(op.name == "FRAME" for op, _, _ in ops)
        assert file.tell() > start
    equal(payload, torch.load(p, weights_only=False, map_location="cpu", mmap=False))
    equal(payload, load_research_payload(p, map_location="cpu", mmap=True))
    assert header["normal_storage_bytes"] > 0
    assert header["v1_sha256"] == stream.FROZEN_V1_SHA256


def test_prefix_guard_precedes_large_read_and_first_decode(anchor, monkeypatch):
    _, encoded = prepared(anchor)
    file = io.BytesIO()
    stream.save_guarded_legacy(encoded, file)
    events = []
    original_guard = v1.memory_guard
    original_decode = v1._restore_tensor
    def guard(*a, **k):
        events.append("guard")
        return original_guard(*a, **k)
    def decode(*a, **k):
        events.append("decode")
        return original_decode(*a, **k)
    class Reader(io.BytesIO):
        def read(self, n=-1):
            if n > 128 * 1024:
                events.append("large_read")
            return super().read(n)
        def readinto(self, buffer):
            if len(buffer) > 128 * 1024:
                events.append("large_read")
            return super().readinto(buffer)
    monkeypatch.setattr(v1, "memory_guard", guard)
    monkeypatch.setattr(v1, "_restore_tensor", decode)
    torch.load(Reader(file.getvalue()), weights_only=False, map_location="cpu", mmap=False)
    assert "large_read" in events
    assert events.index("guard") < events.index("large_read") < events.index("decode")


def test_refused_prefix_reads_no_large_blob_or_tensor_decode(anchor, monkeypatch):
    _, encoded = prepared(anchor)
    file = io.BytesIO()
    stream.save_guarded_legacy(encoded, file)
    events = []
    class Reader(io.BytesIO):
        def read(self, n=-1):
            if n > 128 * 1024:
                events.append("large_read")
            return super().read(n)
        def readinto(self, buffer):
            if len(buffer) > 128 * 1024:
                events.append("large_read")
            return super().readinto(buffer)
    def refuse(*a, **k):
        events.append("refused_guard")
        raise MemoryError("test preallocation refusal")
    monkeypatch.setattr(v1, "memory_guard", refuse)
    with pytest.raises(MemoryError, match="preallocation"):
        torch.load(Reader(file.getvalue()), weights_only=False, map_location="cpu", mmap=False)
    assert events == ["refused_guard"]


def test_zip_protocol_and_default_framed_writer_refused(anchor):
    _, encoded = prepared(anchor)
    wrapped = stream.GuardedPayload(encoded)
    for kwargs in [{}, {"_use_new_zipfile_serialization": False, "pickle_protocol": 4},
                   {"_use_new_zipfile_serialization": False, "pickle_protocol": 2}]:
        with pytest.raises(ValueError):
            torch.save(wrapped, io.BytesIO(), **kwargs)
    with pytest.raises(ValueError):
        stream.save_guarded_legacy(encoded, io.BytesIO(), _use_new_zipfile_serialization=True)
    with pytest.raises(ValueError):
        stream.save_guarded_legacy(encoded, io.BytesIO(), pickle_protocol=2)


def test_identical_tensor_aliases_preserved_normal_views_untouched(anchor):
    payload = make_payload(anchor)
    payload["model"]["tied_alias"] = payload["model"]["llm." + anchor[1]]
    backing = torch.randn(40)
    payload["normal_view"] = backing[3:13]
    payload["normal_backing"] = backing
    encoded, _ = v1.prepare_delta_payload(payload, anchor[0])
    file = io.BytesIO()
    stream.save_guarded_legacy(encoded, file)
    file.seek(0)
    restored = torch.load(file, weights_only=False, map_location="cpu")
    equal(payload, restored)
    assert restored["model"]["tied_alias"] is restored["model"]["llm." + anchor[1]]
    assert restored["normal_view"].untyped_storage().data_ptr() == restored["normal_backing"].untyped_storage().data_ptr()


def test_stream_builder_guards_before_generator_and_matches_shape_bytes(anchor, monkeypatch):
    payload = make_payload(anchor)
    tensor = payload["model"].pop("llm." + anchor[1])
    events = []
    original_guard = v1.memory_guard
    def guard(*a, **k):
        events.append("guard")
        return original_guard(*a, **k)
    def iterator():
        events.append("tensor_generator")
        yield "llm." + anchor[1], tensor, anchor[0]
    monkeypatch.setattr(v1, "memory_guard", guard)
    wrapped, receipt = stream.stream_encode_small(payload, iterator(),
        expected_decoded_bytes=tensor.numel() * 4, expected_tensor_count=1,
        largest_tensor_bytes=tensor.numel() * 4,
        encoded_upper_bytes=v1._zstd().ZSTD_compressBound(tensor.numel() * 4), max_encoded_bytes=2 ** 22)
    assert events.index("guard") < events.index("tensor_generator")
    file = io.BytesIO()
    stream.save_guarded_legacy(wrapped, file)
    file.seek(0)
    restored = torch.load(file, weights_only=False, map_location="cpu")
    payload["model"]["llm." + anchor[1]] = tensor
    equal(payload, restored)
    assert receipt["input_tensor_count"] == 1


def test_header_source_aggregate_and_normal_metadata_limits(anchor):
    _, encoded = prepared(anchor)
    header = stream.GuardedPayload(encoded).header
    changed = dict(header, normal_storage_bytes=stream.NORMAL_STORAGE_LIMIT + 1)
    with pytest.raises(ValueError, match="limit"):
        stream.memory_requirement(changed)
    with pytest.raises(ValueError, match="source"):
        stream.memory_requirement(dict(header, adapter_sha256="0" * 64))
    # Direct reducer invocation is refused without native legacy call context.
    with pytest.raises(ValueError, match="legacy loader context"):
        stream._restore_preflight(header, "0" * 64)


def test_multiple_bounded_v1_sessions_global_counts_and_exact_load(anchor):
    payload = make_payload(anchor)
    first = payload["model"].pop("llm." + anchor[1])
    second_key = anchor[1].replace(".q_proj.", ".k_proj.")
    second = anchor[2].float() + 1 + 1e-5
    iterator = [("llm." + anchor[1], first, anchor[0]), ("llm." + second_key, second, anchor[0])]
    size = first.numel() * 4
    wrapped, _ = stream.stream_encode_small(payload, iterator, expected_decoded_bytes=2 * size,
        expected_tensor_count=2, largest_tensor_bytes=size,
        encoded_upper_bytes=2 * v1._zstd().ZSTD_compressBound(size), max_encoded_bytes=4 * size)
    assert wrapped.header["decoder_session_count"] == 2
    assert wrapped.header["decoded_model_bytes"] == 2 * size
    for session in wrapped.payload.sessions:
        assert len(session.manifest["tensors"]) == 1
    file = io.BytesIO();stream.save_guarded_legacy(wrapped, file);file.seek(0)
    restored = torch.load(file, weights_only=False, map_location="cpu")
    payload["model"].update({"llm." + anchor[1]: first, "llm." + second_key: second})
    equal(payload, restored)


def test_full_probe_bounded_writer_caps_ordinary_storage_and_metadata_headers(anchor):
    path = Path(__file__).parents[1] / "scripts/probe_research_stream_delta_full.py"
    spec = importlib.util.spec_from_file_location("stream_full_probe", path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Budget:
        def __init__(self):self.checked = 0
        def ensure_space(self, size):self.checked += size
    payload, encoded = prepared(anchor)
    file = io.BytesIO();budget = Budget()
    bound = module.BoundedWriter(file, budget, 4 * 2 ** 20)
    stream.save_guarded_legacy(encoded, bound)
    assert bound.bytes == len(file.getvalue()) == budget.checked
    file.seek(0);equal(payload, torch.load(file, weights_only=False, map_location="cpu"))
    with pytest.raises(ValueError, match="hard size cap"):
        stream.save_guarded_legacy(encoded, module.BoundedWriter(io.BytesIO(), Budget(), 100))
    parsed = module.headers(anchor[0])
    assert len(parsed) == 2 and all(p["fp32_bytes"] == 512 * 512 * 4 for p in parsed.values())


def test_full_probe_main_tiny_fixture_receipt_before_owned_cleanup(anchor, tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/probe_research_stream_delta_full.py"
    spec = importlib.util.spec_from_file_location("stream_full_probe_main", path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module, "snapshot_path", lambda revision: anchor[0])
    output = tmp_path / "task_owned"
    unrelated = tmp_path / "unrelated.pt";unrelated.write_bytes(b"preserve unrelated data")
    source_before = (anchor[0] / "model.safetensors").read_bytes()
    monkeypatch.setattr(sys, "argv", [str(path), "--run-full-synthetic", "--output", str(output), "--cleanup-verified"])
    module.main()
    verified = json.loads((output / "verified.json").read_text())
    assert verified["stored_tensors_verified"] == 2 and verified["all_model_and_normal_fixture_bits_exact"]
    assert verified["archive_deleted"] and verified["task_owned_archive_bytes_reclaimed"] > 0
    assert not (output / "synthetic_model_legacy.pt").exists()
    assert (output / "encoding.json").is_file() and (output / "memory_samples.json").is_file()
    assert unrelated.read_bytes() == b"preserve unrelated data"
    assert (anchor[0] / "model.safetensors").read_bytes() == source_before
    with pytest.raises(ValueError, match="new/empty"):
        module.main()


def test_fake_tensor_load_remains_structural_and_normal_storages_distinct(anchor, tmp_path):
    from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
    payload, encoded = prepared(anchor)
    p = tmp_path / "stream.pt";stream.save_guarded_legacy(encoded, p)
    with FakeTensorMode():
        result = load_research_payload(p, map_location="cpu", mmap=True)
        for name, original in payload["model"].items():
            assert isinstance(result["model"][name], FakeTensor)
            assert result["model"][name].shape == original.shape


def test_anchor_size_is_checked_before_guard_or_large_hash_read(anchor, tmp_path, monkeypatch):
    _, encoded = prepared(anchor)
    p = tmp_path / "stream.pt";stream.save_guarded_legacy(encoded, p)
    source = anchor[0] / "model.safetensors"
    with source.open("ab") as f:f.write(b"unexpected extra bytes")
    events = []
    monkeypatch.setattr(v1, "memory_guard", lambda *a, **k: events.append("guard"))
    with pytest.raises(ValueError, match="anchor file size"):
        torch.load(p, weights_only=False, map_location="cpu")
    assert events == []


def test_full_probe_metadata_atomic_peak_cap_and_bounded_samples(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/probe_research_stream_delta_full.py"
    spec = importlib.util.spec_from_file_location("stream_probe_caps", path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    target = tmp_path / "record.json"
    module._atomic(target, {"safe": True})
    before = target.read_bytes()
    monkeypatch.setattr(module, "METADATA_ENVELOPE_BYTES", 4096)
    with pytest.raises(ValueError, match="atomic staging"):
        module._atomic(target, {"replacement": True})
    assert target.read_bytes() == before
    assert not target.with_suffix(".json.tmp").exists()
    monkeypatch.setattr(module, "MAX_MEMORY_SAMPLES", 2)
    monitor = module.Monitor()
    for _ in range(5):monitor.sample()
    assert len(monitor.samples) == 2 and monitor.total_samples == 5 and monitor.dropped_samples == 3
    assert monitor.peak > 0


def test_stale_zip_tls_does_not_change_actual_cpu_legacy_request(anchor, tmp_path):
    payload, encoded = prepared(anchor)
    p = tmp_path / "stream.pt";stream.save_guarded_legacy(encoded, p)
    tls = torch.serialization._serialization_tls
    previous = tls.map_location
    sentinel = lambda storage, location: storage
    try:
        tls.map_location = sentinel
        equal(payload, load_research_payload(p, map_location="cpu", mmap=True))
        assert tls.map_location is sentinel  # No TLS mutation/global patch.
    finally:
        tls.map_location = previous


@pytest.mark.parametrize("location", ["meta", {"cpu": "meta"}, lambda storage, loc: storage])
def test_actual_non_cpu_legacy_request_refused_before_big_read(anchor, tmp_path, monkeypatch, location):
    _, encoded = prepared(anchor)
    p = tmp_path / "stream.pt";stream.save_guarded_legacy(encoded, p)
    calls = []
    monkeypatch.setattr(v1, "memory_guard", lambda *a, **k: calls.append("guard"))
    with pytest.raises(ValueError, match="legacy supports CPU"):
        torch.load(p, weights_only=False, map_location=location)
    assert calls == []


def test_fresh_process_failed_zip_map_then_guarded_cpu_legacy_is_exact(anchor, tmp_path):
    payload, encoded = prepared(anchor)
    p = tmp_path / "stream.pt";stream.save_guarded_legacy(encoded, p)
    baseline = tmp_path / "baseline.pt";torch.save(payload, baseline)
    zip_path = tmp_path / "old_zip.pt";torch.save(encoded, zip_path, pickle_protocol=4)
    code = r'''
import sys,json,hashlib
from pathlib import Path
import torch,numpy as np
from src.utils import research_lossless_delta as v1
from src.utils import research_stream_delta as stream
from src.utils.research_checkpoint_compat import load_research_payload
root,archive,baseline,oldzip=map(Path,sys.argv[1:])
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v1.CONFIG_SHA256=h(root/'config.json');v1.INDEX_SHA256=h(root/'model.safetensors.index.json')
v1.SHARD_SHA256={'model.safetensors':h(root/'model.safetensors')}
stream.ANCHOR_SHARD_BYTES={'model.safetensors':(root/'model.safetensors').stat().st_size}
expected=torch.load(baseline,map_location='cpu',weights_only=False)
try:torch.load(oldzip,map_location='meta',weights_only=False)
except ValueError:pass
else:raise AssertionError('Expected frozen ZIP CPU-only refusal')
assert torch.serialization._serialization_tls.map_location=='meta'
actual=load_research_payload(archive,map_location='cpu',mmap=True)
assert torch.serialization._serialization_tls.map_location=='meta'
def eq(a,b):
    if isinstance(a,torch.Tensor):assert type(b) is torch.Tensor and a.shape==b.shape and a.dtype==b.dtype and v1._raw(a)==v1._raw(b)
    elif isinstance(a,np.ndarray):assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:eq(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):eq(x,y)
    else:assert a==b
eq(expected,actual)
assert not torch.cuda.is_initialized()
print(json.dumps({'fresh_process':True,'actual_failed_zip_tls_reproduced':True,'guarded_cpu_legacy_exact':True,'global_tls_untouched':True,'gpu_used':False}))
'''
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    child = subprocess.run([sys.executable, "-c", code, str(anchor[0]), str(p), str(baseline), str(zip_path)],
        cwd=Path(__file__).parents[1], env=environment, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["guarded_cpu_legacy_exact"]
