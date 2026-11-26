"""Isolated guarded native-legacy adapter for frozen lossless-delta v1.

Trusted local checkpoints only. No trainer/importer is modified. Framing is
disabled on an isolated Pickler INSTANCE: protocol4 BINBYTES remain efficient,
while a small preflight reducer executes before compressed payload bytes are
read. Ordinary CPU storages are filled later by native Torch legacy loading.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import pickle
import types

from src.utils import research_lossless_delta as v1

FROZEN_V1_SHA256 = "e53d1a39082ed022ef07d908b148bd18fa2798af65df42f16d0cbb98e53175a3"
GIB = 2 ** 30
METADATA_LIMIT = 64 * 2 ** 20
HEADER_LIMIT = 8192
NORMAL_STORAGE_LIMIT = 64 * GIB
AGGREGATE_LIMIT = 96 * GIB
READ_AHEAD_ALLOWANCE = 2 ** 20
ANCHOR_SHARD_BYTES = {"model-00001-of-00002.safetensors": 4999721256,
                      "model-00002-of-00002.safetensors": 3537628448}
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_sha256():
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if digest != _SOURCE_SHA256 or v1.codec_sha256() != FROZEN_V1_SHA256:
        raise ValueError("Streaming adapter or frozen v1 source changed")
    return digest


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class _NoFramePickler(pickle._Pickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.proto != 4 or not hasattr(self, "framer"):
            raise ValueError("Streaming delta requires installed protocol4 Pickler contract")
        # Per-instance only; CPython's global pickle implementation is intact.
        self.framer.start_framing = lambda: None


def _dump(obj, file, protocol=None, **kwargs):
    return _NoFramePickler(file, protocol=protocol, **kwargs).dump(obj)


def _pickle_module():
    module = types.ModuleType("research_stream_delta_pickle")
    module.Pickler, module.dump = _NoFramePickler, _dump
    return module


def _writer_contract():
    """Reject accidental native ZIP / standard framed Pickler even if bypassed."""
    frame = inspect.currentframe()
    try:
        while frame is not None:
            if frame.f_globals.get("__name__") == "torch.serialization":
                if frame.f_code.co_name == "_save":
                    raise ValueError("Guarded delta refuses ZIP; native legacy serialization is mandatory")
                if frame.f_code.co_name == "_legacy_save":
                    if frame.f_locals.get("pickle_protocol") != 4 or not isinstance(frame.f_locals.get("pickler"), _NoFramePickler):
                        raise ValueError("Guarded delta requires protocol4 and the isolated no-frame Pickler")
                    return
            frame = frame.f_back
    finally:
        del frame
    raise ValueError("Guarded delta must be written by its native legacy writer")


def _legacy_cpu_context():
    """Inspect the ACTUAL legacy request, not ZIP's possibly stale Torch TLS."""
    import torch
    frame = inspect.currentframe()
    try:
        while frame is not None:
            if frame.f_globals.get("__name__") == "torch.serialization" and frame.f_code.co_name == "_legacy_load":
                location = frame.f_locals.get("map_location")
                if location is None or (isinstance(location, str) and location == "cpu") or (isinstance(location, torch.device) and location.type == "cpu"):
                    return
                raise ValueError("Guarded native legacy supports CPU map_location only")
            frame = frame.f_back
    finally:
        del frame
    raise ValueError("Guarded payload requires installed native legacy loader context")


class _LegacySession(v1._Session):
    def __reduce__(self):
        return (_restore_legacy_session, (self.manifest,))


def _restore_legacy_session(manifest):
    _legacy_cpu_context()
    # Preserve frozen v1's manifest/source/size validation and Tensor decoder.
    # Only its ZIP-TLS reducer is replaced with actual legacy-call validation.
    return _LegacySession(manifest)


def _combined(sessions):
    descriptors = {}
    for session in sessions:
        for name, spec in session.manifest["tensors"].items():
            if name in descriptors:
                raise ValueError("Duplicate tensor across decoder sessions")
            descriptors[name] = spec
    return descriptors


def _payload_statistics(payload, sessions, *, encoded):
    """Header-bound structural sizes, without reading normal storage values."""
    import numpy as np
    import torch
    descriptors = _combined(sessions)
    decoded_ids = set()
    if not encoded:
        decoded_ids = {id(payload["model"][name]) for name in descriptors}
    normal_storages = {}
    encoded_seen = set()
    metadata_bytes = 0
    visiting = set()

    def walk(value):
        nonlocal metadata_bytes
        if isinstance(value, v1._TensorProxy):
            if not encoded or value.session not in sessions or value.name not in value.session.manifest["tensors"]:
                raise ValueError("Unexpected encoded proxy/session")
            spec = descriptors[value.name]
            if len(value.data) != spec["encoded_bytes"] or v1._sha(value.data) != spec["encoded_sha256"]:
                raise ValueError("Encoded proxy byte binding changed")
            encoded_seen.add(value.name)
            return ("delta_tensor", spec["shape"], "torch.float32")
        if isinstance(value, torch.Tensor):
            if not encoded and id(value) in decoded_ids:
                return ("delta_tensor", list(value.shape), str(value.dtype))
            from torch._subclasses.fake_tensor import FakeTensor
            structural_fake = not encoded and isinstance(value, FakeTensor)
            if (value.device.type != "cpu" and not structural_fake) or value.layout != torch.strided:
                raise ValueError("Streaming prototype normal tensors require CPU dense storage")
            storage = value.untyped_storage()
            # Storage identity also works for structural FakeTensor inspection;
            # Fake storages all expose data_ptr0 and must not collapse together.
            key = (storage._cdata, storage.nbytes())
            normal_storages[key] = storage.nbytes()
            return ("normal_tensor", list(value.shape), str(value.dtype), value.stride(), value.storage_offset(), storage.nbytes())
        if isinstance(value, np.ndarray):
            metadata_bytes += value.nbytes
            if metadata_bytes > METADATA_LIMIT:
                raise ValueError("Non-Torch metadata allocation limit exceeded")
            return ("ndarray", value.dtype.str, list(value.shape), hashlib.sha256(value.tobytes()).hexdigest())
        if isinstance(value, (str, bytes, bytearray)):
            metadata_bytes += len(value.encode() if isinstance(value, str) else value)
            if metadata_bytes > METADATA_LIMIT:
                raise ValueError("Non-Torch metadata allocation limit exceeded")
            return ("bytes", len(value), hashlib.sha256(bytes(value)).hexdigest()) if not isinstance(value, str) else value
        if value is None or isinstance(value, (bool, int, float)):
            if isinstance(value, int) and value.bit_length() > 1024:
                raise ValueError("Oversized integer metadata refused")
            return value
        if isinstance(value, (dict, tuple, list)):
            identity = id(value)
            if identity in visiting:
                raise ValueError("Cyclic checkpoint containers refused")
            visiting.add(identity)
            try:
                if isinstance(value, dict):
                    return ("mapping", [(walk(k), walk(v)) for k, v in value.items()])
                items = [walk(v) for v in value]
                return ("tuple" if isinstance(value, tuple) else "list", items)
            finally:
                visiting.remove(identity)
        raise ValueError("Unsupported non-Torch checkpoint metadata type")

    structure = walk(payload)
    if encoded and encoded_seen != set(descriptors):
        raise ValueError("Missing encoded model proxy")
    serialized = _canonical(structure)
    manifest = _canonical([s.manifest for s in sessions])
    metadata_size = len(serialized) + len(manifest) + metadata_bytes
    if metadata_size > METADATA_LIMIT or len(normal_storages) > 65536:
        raise ValueError("Checkpoint structural metadata limit exceeded")
    return {"normal_storage_bytes": sum(normal_storages.values()),
            "largest_normal_storage_bytes": max(normal_storages.values(), default=0),
            "normal_storage_count": len(normal_storages),
            "metadata_bytes": metadata_size,
            "structure_sha256": hashlib.sha256(serialized).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest).hexdigest()}


def _header(payload, sessions):
    stats = _payload_statistics(payload, sessions, encoded=True)
    descriptors = _combined(sessions)
    largest = max(s["decoded_bytes"] for s in descriptors.values())
    first_blob = next(iter(descriptors.values()))["encoded_bytes"]
    v1_guard = max(3 * s.manifest["encoded_bytes"] + 2 * s.manifest["decoded_bytes"] +
                   6 * max(d["decoded_bytes"] for d in s.manifest["tensors"].values()) +
                   next(iter(s.manifest["tensors"].values()))["encoded_bytes"] for s in sessions)
    anchors = {(s["anchor"]["snapshot"], s["anchor"]["shard"]) for s in descriptors.values()}
    if len({snapshot for snapshot, _ in anchors}) != 1:
        raise ValueError("One pinned snapshot per streaming payload is required")
    anchor_files = [{"snapshot": snapshot, "shard": shard, "bytes": ANCHOR_SHARD_BYTES[shard]}
                    for snapshot, shard in sorted(anchors)]
    header = {"schema_version": 1, "adapter_sha256": source_sha256(), "v1_sha256": FROZEN_V1_SHA256,
              "repo": v1.REPO, "revision": v1.REVISION,
              "decoded_model_bytes": sum(s["decoded_bytes"] for s in descriptors.values()),
              "encoded_bytes": sum(s["encoded_bytes"] for s in descriptors.values()),
              "largest_decoded_tensor_bytes": largest, "first_encoded_blob_bytes": first_blob,
              "encoded_tensor_count": len(descriptors), "decoder_session_count": len(sessions),
              "max_v1_decoder_extra_bytes": v1_guard, "anchor_files": anchor_files,
              "anchor_cache_bytes": sum(a["bytes"] for a in anchor_files), **stats}
    _validate_header(header)
    return header


def _validate_header(header):
    if len(_canonical(header)) > HEADER_LIMIT or header.get("schema_version") != 1:
        raise ValueError("Streaming preflight header size/schema limit exceeded")
    if header.get("adapter_sha256") != source_sha256() or header.get("v1_sha256") != FROZEN_V1_SHA256:
        raise ValueError("Streaming decoder source binding changed")
    if (header.get("repo"), header.get("revision")) != (v1.REPO, v1.REVISION):
        raise ValueError("Streaming delta is pinned SmallThinker only")
    files = header["anchor_files"]
    if not 0 < len(files) <= 2 or len({(a["snapshot"], a["shard"]) for a in files}) != len(files):
        raise ValueError("Invalid source anchor file list")
    for item in files:
        if (Path(item["snapshot"]).name != v1.REVISION or item["shard"] not in ANCHOR_SHARD_BYTES or
            item["bytes"] != ANCHOR_SHARD_BYTES[item["shard"]] or
            (Path(item["snapshot"]) / item["shard"]).stat().st_size != item["bytes"]):
            raise ValueError("Pinned source anchor file size/path changed before read")
    if header["anchor_cache_bytes"] != sum(a["bytes"] for a in files):
        raise ValueError("Source anchor cache allowance differs")
    limits = {"decoded_model_bytes": v1.MAX_CHECKPOINT_DECODED_BYTES,
              "encoded_bytes": v1.MAX_PICKLE_BYTES, "largest_decoded_tensor_bytes": v1.MAX_TENSOR_DECODED_BYTES,
              "first_encoded_blob_bytes": v1.MAX_TENSOR_DECODED_BYTES + 2 ** 20,
              "normal_storage_bytes": NORMAL_STORAGE_LIMIT, "largest_normal_storage_bytes": NORMAL_STORAGE_LIMIT,
              "metadata_bytes": METADATA_LIMIT, "encoded_tensor_count": 10000, "normal_storage_count": 65536,
              "decoder_session_count": 10000, "max_v1_decoder_extra_bytes": 256 * GIB}
    limits["anchor_cache_bytes"] = sum(ANCHOR_SHARD_BYTES.values())
    for key, limit in limits.items():
        value = header[key]
        if type(value) is not int or value < 0 or value > limit:
            raise ValueError("Streaming aggregate/tensor/normal storage limit exceeded: " + key)
    if not header["decoded_model_bytes"] or header["decoded_model_bytes"] + header["encoded_bytes"] + header["normal_storage_bytes"] > AGGREGATE_LIMIT:
        raise ValueError("Streaming total allocation limit exceeded")


def memory_requirement(header):
    _validate_header(header)
    d, c, n = (header[k] for k in ("decoded_model_bytes", "encoded_bytes", "normal_storage_bytes"))
    scratch = 6 * header["largest_decoded_tensor_bytes"]
    # Pickle memo retains ALL compressed bytes until normal storage loading ends.
    resident = d + c + n + header["anchor_cache_bytes"] + 2 * header["metadata_bytes"] + READ_AHEAD_ALLOWANCE
    streaming_live = resident + scratch
    # Frozen v1's first decoder guard must also pass after reading its firstblob.
    v1_first_guard = header["max_v1_decoder_extra_bytes"]
    return max(streaming_live, resident + v1_first_guard)


class _PreflightProxy:
    def __init__(self, header):
        self.header = header

    def __reduce__(self):
        return (_restore_preflight, (self.header, hashlib.sha256(_canonical(self.header)).hexdigest()))


def _restore_preflight(header, digest):
    _legacy_cpu_context()
    if hashlib.sha256(_canonical(header)).hexdigest() != digest:
        raise ValueError("Streaming preflight header checksum changed")
    guard = v1.memory_guard(memory_requirement(header))
    return {"header": header, "guard": guard}


class _EncodedCarrier:
    def __init__(self, payload):
        self.payload = payload

    def __reduce__(self):
        return (_restore_carrier, (self.payload.sessions, self.payload.payload))


def _restore_carrier(sessions, payload):
    return sessions, payload


class _EncodedGroup:
    def __init__(self, payload, sessions):
        self.payload, self.sessions = payload, tuple(sessions)


class GuardedPayload:
    def __init__(self, payload):
        if isinstance(payload, v1.DeltaPayload):
            payload = _EncodedGroup(payload.payload, (payload.session,))
        if not isinstance(payload, _EncodedGroup):
            raise ValueError("Expected a frozen v1 encoded payload")
        if any(not isinstance(s, _LegacySession) for s in payload.sessions):
            rebound = {id(s): _LegacySession(s.manifest) for s in payload.sessions}
            copied = dict(payload.payload)
            copied["model"] = dict(copied["model"])
            aliases = {}
            for name, value in copied["model"].items():
                if isinstance(value, v1._TensorProxy):
                    if id(value) not in aliases:
                        aliases[id(value)] = v1._TensorProxy(rebound[id(value.session)], value.name, value.data)
                    copied["model"][name] = aliases[id(value)]
            payload = _EncodedGroup(copied, tuple(rebound[id(s)] for s in payload.sessions))
        self.payload = payload
        self.header = _header(payload.payload, payload.sessions)

    def __reduce__(self):
        _writer_contract()
        # Header reducer executes BEFORE unpickling carrier/session/blob args.
        return (_finish_guarded, (_PreflightProxy(self.header), _EncodedCarrier(self.payload)))


def _finish_guarded(preflight, carrier):
    sessions, payload = carrier
    if any(s.seen != set(s.manifest["tensors"]) for s in sessions) or payload.get("model_ref") != {"repo": v1.REPO, "revision": v1.REVISION}:
        raise ValueError("Incomplete decoder session or wrong model binding")
    stats = _payload_statistics(payload, sessions, encoded=False)
    if any(stats[k] != preflight["header"][k] for k in stats):
        raise ValueError("Loaded payload structure/normal allocation differs from its preflight header")
    return payload


def save_guarded_legacy(payload, file, *, pickle_protocol=4, _use_new_zipfile_serialization=False):
    import torch
    if _use_new_zipfile_serialization or pickle_protocol != 4:
        raise ValueError("Guarded writer requires native legacy and pickle_protocol=4")
    wrapped = payload if isinstance(payload, GuardedPayload) else GuardedPayload(payload)
    torch.save(wrapped, file, pickle_protocol=4, _use_new_zipfile_serialization=False,
               pickle_module=_pickle_module())
    return dict(wrapped.header, reader_extra_guard_bytes=memory_requirement(wrapped.header),
                writer="native_torch_legacy_protocol4_no_frames")


def stream_encode_small(template, tensor_iterator, *, expected_decoded_bytes,
                        expected_tensor_count, largest_tensor_bytes, encoded_upper_bytes,
                        max_encoded_bytes):
    """Encode one original F32 tensor at a time; retain only compressed bytes.

    Caller supplies metadata-only bounds before iterator reads/allocates values.
    No full FP32 model is held by this builder. All tensors must map exactly to
    frozen v1's canonical BF16 anchors. Normal template tensors stay unchanged.
    """
    import torch
    source_sha256()
    if template.get("model_ref") != {"repo": v1.REPO, "revision": v1.REVISION} or not isinstance(template.get("model"), dict):
        raise ValueError("Pinned SmallThinker template required")
    if not 0 < expected_decoded_bytes <= v1.MAX_CHECKPOINT_DECODED_BYTES or not 0 < largest_tensor_bytes <= v1.MAX_TENSOR_DECODED_BYTES:
        raise ValueError("Streaming writer tensor/aggregate budget exceeded")
    if not 0 < expected_tensor_count <= 10000 or not 0 < max_encoded_bytes <= v1.MAX_PICKLE_BYTES or encoded_upper_bytes < expected_decoded_bytes:
        raise ValueError("Invalid streaming writer bound")
    descriptors, compressed = {}, {}
    total = encoded = 0
    iterator = iter(tensor_iterator)
    anchors = {}
    guard_extra = 2 * encoded_upper_bytes + sum(ANCHOR_SHARD_BYTES.values()) + 8 * largest_tensor_bytes + METADATA_LIMIT
    initial_guard = v1.memory_guard(guard_extra)
    while True:
        # This runs BEFORE next(generator) can read an anchor or create FP32.
        v1.memory_guard(max(0, guard_extra - encoded))
        try:
            name, tensor, snapshot = next(iterator)
        except StopIteration:
            break
        if name in descriptors or name in template["model"]:
            raise ValueError("Duplicate streaming model key")
        snapshot = str(snapshot)
        if snapshot not in anchors:
            anchors[snapshot] = v1.SmallAnchor(snapshot)
        anchor = anchors[snapshot]
        key = anchor.reference_key(name)
        if key is None or tensor.dtype != torch.float32 or tensor.device.type != "cpu" or tensor.requires_grad or not tensor.is_contiguous():
            raise ValueError("Streaming tensor must be an original detached CPU FP32 parameter")
        size = tensor.numel() * 4
        if tensor.storage_offset() or tensor.untyped_storage().nbytes() != size or size > largest_tensor_bytes or total + size > expected_decoded_bytes:
            raise ValueError("Streaming tensor ownership/size bound changed")
        desc = anchor.descriptor(key)
        if desc["shape"] != list(tensor.shape) or desc["safetensors_dtype"] != "BF16":
            raise ValueError("Streaming original shape/dtype changed")
        reference = anchor.read(desc)
        bits = torch.bitwise_xor(tensor.reshape(-1).view(torch.int32), reference.float().reshape(-1).view(torch.int32))
        shuffled = bits.view(torch.uint8).reshape(-1, 4).t().contiguous().numpy().tobytes()
        data = v1._compress(shuffled)
        total += size
        encoded += len(data)
        if encoded > min(max_encoded_bytes, encoded_upper_bytes):
            raise ValueError("Measured compressed bytes exceed task staging budget")
        descriptors[name] = {"shape": list(tensor.shape), "decoded_bytes": size,
                             "decoded_sha256": v1._sha(v1._raw(tensor)), "encoded_bytes": len(data),
                             "encoded_sha256": v1._sha(data), "anchor": desc,
                             "anchor_tensor_sha256": v1._sha(v1._raw(reference))}
        compressed[name] = data
        del tensor, reference, bits, shuffled, data
    if len(descriptors) != expected_tensor_count or total != expected_decoded_bytes:
        raise ValueError("Streaming input count/bytes differ from metadata preflight")
    manifest = {"version": 1, "repo": v1.REPO, "revision": v1.REVISION,
                "codec_sha256": FROZEN_V1_SHA256, "decoded_bytes": total,
                "encoded_bytes": encoded, "tensors": descriptors}
    result = dict(template)
    result["model"] = dict(template["model"])
    # Independent bounded v1 sessions keep the frozen decoder's deliberately
    # ZIP-sized guard local to one tensor. The new prefix guards global totals.
    sessions = []
    for name, data in compressed.items():
        single = dict(manifest, decoded_bytes=descriptors[name]["decoded_bytes"],
                      encoded_bytes=len(data), tensors={name: descriptors[name]})
        session = _LegacySession(single)
        sessions.append(session)
        result["model"][name] = v1._TensorProxy(session, name, data)
    wrapped = GuardedPayload(_EncodedGroup(result, sessions))
    return wrapped, {"decoded_bytes": total, "encoded_bytes": encoded,
                     "input_tensor_count": len(descriptors), "initial_writer_guard": initial_guard,
                     "reader_extra_guard_bytes": memory_requirement(wrapped.header)}
