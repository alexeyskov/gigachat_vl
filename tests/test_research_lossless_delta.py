"""Trusted native-Torch round trips, no production checkpoints or CUDA."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.utils import research_lossless_delta as delta
from src.utils.research_checkpoint_compat import load_research_payload


@pytest.fixture
def anchor(tmp_path, monkeypatch):
    path = tmp_path / delta.REVISION
    path.mkdir()
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.linspace(-1, 1, 24).reshape(4, 6).bfloat16(),
        "model.layers.0.block_sparse_moe.primary_router.weight": torch.arange(12).reshape(3, 4).bfloat16(),
    }
    save_file(tensors, path / "model.safetensors")
    (path / "config.json").write_text(json.dumps({"model_type": "smallthinker"}))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))
    monkeypatch.setattr(delta, "INDEX_SHA256", hashlib.sha256((path / "model.safetensors.index.json").read_bytes()).hexdigest())
    monkeypatch.setattr(delta, "CONFIG_SHA256", hashlib.sha256((path / "config.json").read_bytes()).hexdigest())
    monkeypatch.setattr(delta, "SHARD_SHA256", {"model.safetensors": hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()})
    return path, tensors


def payload(anchor):
    _, tensors = anchor
    model = {"llm." + k: v.float() + torch.linspace(-1e-6, 1e-6, v.numel()).reshape(v.shape) for k, v in tensors.items()}
    model.update({"projector.weight": torch.arange(9, dtype=torch.float32).reshape(3, 3),
                  "llm.model.layers.0.block_sparse_moe.visual_experts.0.up.weight": torch.ones(4, 6),
                  "llm.model.layers.1.block_sparse_moe.primary_router.weight": torch.ones(5, 4)})
    return {"model": model, "optimizer": {"state": {0: {"exp_avg": torch.randn(4, 6), "exp_avg_sq": torch.ones(4, 6)}}},
            "rng_torch": torch.get_rng_state(), "step": 170, "config": {"recipe": "3b"},
            "model_ref": {"repo": delta.REPO, "revision": delta.REVISION}}


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert type(b) is torch.Tensor
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    else:
        assert a == b


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("protocol", [2, 4])
def test_native_and_unchanged_research_loader_exact_roundtrip(anchor, tmp_path, legacy, protocol):
    original = payload(anchor)
    prepared, receipt = delta.prepare_delta_payload(original, anchor[0])
    assert receipt["unique_encoded_tensors"] == 2
    assert len(receipt["normal_model_tensors"]) == 3
    p = tmp_path / "state.pt"
    torch.save(prepared, p, _use_new_zipfile_serialization=not legacy, pickle_protocol=protocol)
    preflight = delta.preflight_delta_load(p, decoded_bytes_budget=receipt["raw_matched_tensor_bytes"],
        largest_tensor_bytes=receipt["largest_encoded_tensor_decoded_bytes"])
    assert preflight["initial_allocation_guard_outside_loader"]
    equal(original, torch.load(p, weights_only=False, map_location="cpu", mmap=not legacy))
    equal(original, load_research_payload(p, map_location="cpu", mmap=True))
    assert isinstance(original["model"][next(iter(original["model"]))], torch.Tensor)


def test_exact_object_ties_preserved_but_storage_views_refused(anchor, tmp_path):
    original = payload(anchor)
    k = next(iter(original["model"]))
    original["model"]["tied_alias"] = original["model"][k]
    prepared, _ = delta.prepare_delta_payload(original, anchor[0])
    p = tmp_path / "tie.pt"
    torch.save(prepared, p)
    restored = torch.load(p, weights_only=False, mmap=True)
    assert restored["model"][k] is restored["model"]["tied_alias"]
    original["model"]["view"] = original["model"][k].view(-1)
    with pytest.raises(ValueError, match="aliases/views"):
        delta.prepare_delta_payload(original, anchor[0])


def test_alias_between_optimizer_and_model_refused(anchor):
    original = payload(anchor)
    original["optimizer"]["bad_alias"] = next(iter(original["model"].values()))
    with pytest.raises(ValueError, match="aliases/views"):
        delta.prepare_delta_payload(original, anchor[0])


def test_nan_payload_signed_zero_and_infinities_bit_exact(anchor, tmp_path):
    original = payload(anchor)
    t = original["model"][next(iter(original["model"]))]
    t.view(torch.int32).flatten()[:6] = torch.tensor([0, -2147483648, 2139095040, -8388608, 2143289345, 2143289346], dtype=torch.int32)
    prepared, _ = delta.prepare_delta_payload(original, anchor[0])
    p = tmp_path / "nan.pt"
    torch.save(prepared, p)
    equal(original, torch.load(p, weights_only=False, mmap=True))


def test_missing_anchor_or_same_size_mutation_after_hash_cache_refused(anchor, tmp_path):
    original = payload(anchor)
    prepared, _ = delta.prepare_delta_payload(original, anchor[0])
    p = tmp_path / "state.pt"
    torch.save(prepared, p)
    shard = anchor[0] / "model.safetensors"
    backup = shard.read_bytes()
    shard.unlink()
    with pytest.raises(FileNotFoundError):
        torch.load(p, weights_only=False, mmap=True)
    shard.write_bytes(backup[:-1] + bytes([backup[-1] ^ 1]))
    with pytest.raises(ValueError, match="Anchor"):
        torch.load(p, weights_only=False, mmap=True)


def test_checksum_code_and_budget_failures_refused(anchor):
    original = payload(anchor)
    prepared, _ = delta.prepare_delta_payload(original, anchor[0])
    session = prepared.session
    name = next(iter(session.manifest["tensors"]))
    proxy = prepared.payload["model"][name]
    with pytest.raises(ValueError, match="checksum"):
        delta._restore_tensor(session, name, proxy.data[:-1] + bytes([proxy.data[-1] ^ 1]))
    m = dict(session.manifest, codec_sha256="0" * 64)
    with pytest.raises(ValueError, match="source/version"):
        delta._new_session(m)
    m = dict(session.manifest, decoded_bytes=delta.MAX_CHECKPOINT_DECODED_BYTES + 4)
    with pytest.raises(ValueError):
        delta._new_session(m)


def test_decoded_checksum_and_trailing_zstd_frame_refused(anchor):
    prepared, _ = delta.prepare_delta_payload(payload(anchor), anchor[0])
    name = next(iter(prepared.session.manifest["tensors"]))
    proxy = prepared.payload["model"][name]
    spec = prepared.session.manifest["tensors"][name]
    spec["decoded_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Reconstructed"):
        delta._restore_tensor(prepared.session, name, proxy.data)
    with pytest.raises(ValueError):
        delta._decompress(proxy.data + b"extra", spec["decoded_bytes"])


def test_native_router_exact_key_shape_change_and_novel_tensors(anchor):
    original = payload(anchor)
    name = "llm.model.layers.0.block_sparse_moe.primary_router.weight"
    _, receipt = delta.prepare_delta_payload(original, anchor[0])
    assert receipt["unique_encoded_tensors"] == 2
    original["model"][name] = torch.ones(5, 4)
    _, receipt = delta.prepare_delta_payload(original, anchor[0])
    assert receipt["unique_encoded_tensors"] == 1
    assert {"name": name, "reason": "shape_or_dtype_mismatch"} in receipt["normal_model_tensors"]


def test_fake_tensor_inspection_is_structural_only(anchor, tmp_path):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    original = payload(anchor)
    prepared, _ = delta.prepare_delta_payload(original, anchor[0])
    p = tmp_path / "fake.pt"
    torch.save(prepared, p)
    with FakeTensorMode():
        restored = load_research_payload(p, map_location="cpu", mmap=True)
        for name, t in restored["model"].items():
            assert isinstance(t, FakeTensor)
            assert t.shape == original["model"][name].shape
            assert t.dtype == original["model"][name].dtype


def test_other_snapshot_and_index_rejected(anchor, tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        delta.SmallAnchor(tmp_path)
    p = anchor[0] / "model.safetensors.index.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="SHA"):
        delta.SmallAnchor(anchor[0])


def test_per_tensor_and_aggregate_allocation_limits_before_reconstruction(anchor, monkeypatch):
    original = payload(anchor)
    monkeypatch.setattr(delta, "MAX_TENSOR_DECODED_BYTES", 4)
    with pytest.raises(ValueError, match="allocation limit"):
        delta.prepare_delta_payload(original, anchor[0])
    monkeypatch.setattr(delta, "MAX_TENSOR_DECODED_BYTES", delta.GIB)
    monkeypatch.setattr(delta, "MAX_CHECKPOINT_DECODED_BYTES", 4)
    with pytest.raises(ValueError, match="budget"):
        delta.prepare_delta_payload(original, anchor[0])


def test_external_preflight_rejects_before_reading_large_pickle(anchor, tmp_path, monkeypatch):
    prepared, _ = delta.prepare_delta_payload(payload(anchor), anchor[0])
    p = tmp_path / "state.pt"
    torch.save(prepared, p, pickle_protocol=4)
    monkeypatch.setattr(delta, "MAX_PICKLE_BYTES", 10)
    with pytest.raises(ValueError, match="Pickle byte budget"):
        delta.preflight_delta_load(p)


def test_memory_guard_honors_smaller_cgroup_limit_without_page_cache_credit(monkeypatch):
    original = Path.read_text
    def text(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return "MemTotal: 300000000 kB\nMemAvailable: 200000000 kB\n"
        if str(path) == "/sys/fs/cgroup/memory.max":
            return str(170 * delta.GIB)
        if str(path) == "/sys/fs/cgroup/memory.current":
            return str(140 * delta.GIB)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", text)
    with pytest.raises(MemoryError, match="CPU memory"):
        delta.memory_guard(1)


def test_other_model_reference_refused_even_when_tensor_names_match(anchor):
    original = payload(anchor)
    original["model_ref"] = {"repo": "openai/gpt-oss-20b", "revision": "0" * 40}
    with pytest.raises(ValueError, match="pinned SmallThinker"):
        delta.prepare_delta_payload(original, anchor[0])


@pytest.mark.parametrize("offset", [0, 3])
def test_contiguous_views_with_larger_backing_storage_refused(anchor, offset):
    original = payload(anchor)
    k = next(iter(original["model"]))
    original["model"][k] = torch.randn(36)[offset:offset + 24].reshape(4, 6)
    assert original["model"][k].is_contiguous()
    with pytest.raises(ValueError, match="views/oversized"):
        delta.prepare_delta_payload(original, anchor[0])


def test_native_zip_non_cpu_map_location_refused(anchor, tmp_path):
    prepared, _ = delta.prepare_delta_payload(payload(anchor), anchor[0])
    p = tmp_path / "state.pt"
    torch.save(prepared, p, pickle_protocol=4)
    for location in ["meta", {"cpu": "meta"}, lambda storage, loc: storage]:
        with pytest.raises(ValueError, match="CPU map_location only"):
            torch.load(p, weights_only=False, mmap=True, map_location=location)


def test_preexisting_mutated_shard_refused_against_canonical_sha(anchor):
    shard = anchor[0] / "model.safetensors"
    data = shard.read_bytes()
    shard.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="canonical"):
        delta.prepare_delta_payload(payload(anchor), anchor[0])
