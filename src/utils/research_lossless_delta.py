"""Isolated, trusted-pickle SmallThinker FP32 reference-delta prototype.

Not enabled by any trainer or resolver. Encoded values are BYTES in data.pkl,
not Torch storages: legacy unpickling executes reconstructors before filling
normal tensor storages. CPU map_location only; encoded tensors are real CPU
allocations even when untouched tensors use mmap. weights_only=False is required.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

REPO = "Tiiny/SmallThinker-4BA0.6B-Instruct"
REVISION = "a24932bb880c85c04c794283202a9194b1b42709"
INDEX_SHA256 = "52e7f8942e0237e0b35d793c375b5c15a57a455a51fafe1b0110e91d52529f5d"
CONFIG_SHA256 = "40adb346175e86f3840b0b16224acce3f01d0002b2bba76ffd2818f3fce1e876"
SHARD_SHA256 = {
    "model-00001-of-00002.safetensors": "6cefd487ca75d8283cb2d3c28a6cd66527455ea926cc39a07816a5715ed87992",
    "model-00002-of-00002.safetensors": "20be6f7b5051c683b6e0de1e9794ee1c5ff9cc8380f74246463eb34870061cda",
}
GIB = 2 ** 30
MAX_CHECKPOINT_DECODED_BYTES = 32 * GIB
MAX_TENSOR_DECODED_BYTES = GIB  # Small's largest tied embedding is~0.869GiB.
MAX_PICKLE_BYTES = 32 * GIB
_HASH_CACHE = {}
_ZSTD = None
_IMPORTED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def codec_sha256():
    current = _sha(Path(__file__).read_bytes())
    if current != _IMPORTED_SOURCE_SHA256:
        raise ValueError("Lossless-delta source changed after module import")
    return current


def _stat(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _file_sha(path):
    """Once per immutable stat identity per process, with post-read stat check."""
    path = Path(path).resolve(strict=True)
    before = _stat(path)
    key = (str(path), before)
    if key not in _HASH_CACHE:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda: f.read(8 * 2 ** 20), b""):
                h.update(b)
        if _stat(path) != before:
            raise ValueError("Anchor changed while hashing")
        _HASH_CACHE[key] = h.hexdigest()
    return _HASH_CACHE[key]


def _zstd():
    """Use the already installed libzstd stable public C ABI; no new package."""
    global _ZSTD
    if _ZSTD is None:
        lib = ctypes.CDLL(ctypes.util.find_library("zstd") or "libzstd.so.1")
        lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        lib.ZSTD_compressBound.restype = ctypes.c_size_t
        lib.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.ZSTD_compress.restype = ctypes.c_size_t
        lib.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        lib.ZSTD_decompress.restype = ctypes.c_size_t
        lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        lib.ZSTD_isError.restype = ctypes.c_uint
        lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        lib.ZSTD_getErrorName.restype = ctypes.c_char_p
        lib.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
        lib.ZSTD_findFrameCompressedSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.ZSTD_findFrameCompressedSize.restype = ctypes.c_size_t
        lib.ZSTD_versionString.restype = ctypes.c_char_p
        _ZSTD = lib
    return _ZSTD


def _checked(value):
    if _zstd().ZSTD_isError(value):
        raise ValueError("zstd: " + _zstd().ZSTD_getErrorName(value).decode())
    return value


def _compress(data, level=3):
    lib = _zstd()
    capacity = _checked(lib.ZSTD_compressBound(len(data)))
    out = ctypes.create_string_buffer(capacity)
    size = _checked(lib.ZSTD_compress(out, capacity, data, len(data), level))
    return out.raw[:size]


def _decompress(data, expected):
    lib = _zstd()
    # Refuse bombs, truncation, concatenated frames and trailing bytes BEFORE
    # allocating the output; metadata also binds the compressed byte SHA.
    if lib.ZSTD_getFrameContentSize(data, len(data)) != expected:
        raise ValueError("zstd decoded size does not match the tensor bound")
    if _checked(lib.ZSTD_findFrameCompressedSize(data, len(data))) != len(data):
        raise ValueError("Expected one complete zstd frame without trailing bytes")
    out = ctypes.create_string_buffer(expected)
    if _checked(lib.ZSTD_decompress(out, expected, data, len(data))) != expected:
        raise ValueError("zstd output size changed")
    return out.raw


def memory_guard(extra_bytes, *, headroom_bytes=32 * GIB):
    """Conservative no-cache-credit host AND cgroup v2 remaining-memory check."""
    host = {k: int(v.split()[0]) * 1024 for k, v in
            (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    available = host["MemAvailable"]
    current = limit = None
    cg = Path("/sys/fs/cgroup")
    if (cg / "memory.max").exists():
        value = (cg / "memory.max").read_text().strip()
        if value != "max":
            limit = int(value)
            current = int((cg / "memory.current").read_text())
            available = min(available, max(0, limit - current))
    result = {"available_bytes": available, "host_available_bytes": host["MemAvailable"],
              "cgroup_limit_bytes": limit, "cgroup_current_bytes": current,
              "extra_bytes": extra_bytes, "headroom_bytes": headroom_bytes,
              "reclaimable_cache_credit_bytes": 0}
    if extra_bytes < 0 or available < headroom_bytes + extra_bytes:
        raise MemoryError("Lossless-delta CPU memory preflight refused: " + json.dumps(result))
    return result


def preflight_delta_load(path, *, decoded_bytes_budget=MAX_CHECKPOINT_DECODED_BYTES,
                         largest_tensor_bytes=MAX_TENSOR_DECODED_BYTES, mmap=True):
    """Optional OUTSIDE-LOADER guard, needed BEFORE native data.pkl allocation.

    The unchanged torch.load reader allocates the ZIP's full pickle record before
    calling our reducers. A decoder guard alone cannot prevent that allocation.
    Lower tensor budgets may come from the writer's trusted receipt; the decoder
    separately validates its actual manifest. Defaults are deliberately strict.
    Legacy streams conservatively charge their entire file, including storages.
    """
    if not 0 < decoded_bytes_budget <= MAX_CHECKPOINT_DECODED_BYTES or not 0 < largest_tensor_bytes <= MAX_TENSOR_DECODED_BYTES:
        raise ValueError("Unsupported SmallThinker load budget")
    path = Path(path)
    before = _stat(path)
    with path.open("rb") as f:
        magic = f.read(4)
    normal_heap_bytes = 0
    if magic.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(path) as archive:
            records = [i for i in archive.infolist() if i.filename.endswith("/data.pkl")]
            if len(records) != 1 or records[0].compress_type != zipfile.ZIP_STORED:
                raise ValueError("Expected one uncompressed native Torch pickle record")
            size = records[0].file_size
            if not mmap:
                normal_heap_bytes = 2 * sum(i.file_size for i in archive.infolist() if "/data/" in i.filename)
    elif magic.startswith(b"\x80"):
        size = path.stat().st_size
    else:
        raise ValueError("Only native Torch ZIP/legacy prototype files are supported")
    if size > MAX_PICKLE_BYTES or _stat(path) != before:
        raise ValueError("Pickle byte budget exceeded or checkpoint changed")
    return dict(memory_guard(3 * size + 2 * decoded_bytes_budget + 6 * largest_tensor_bytes + normal_heap_bytes),
                pickle_record_bytes=size, normal_heap_allowance_bytes=normal_heap_bytes,
                initial_allocation_guard_outside_loader=True)


class SmallAnchor:
    """Metadata-pinned original BF16 SmallThinker safetensors, read on CPU."""
    def __init__(self, snapshot):
        self.snapshot = Path(snapshot).resolve(strict=True)
        if Path(snapshot).name != REVISION:
            raise ValueError("Only the pinned SmallThinker snapshot is supported")
        self.config = self.snapshot / "config.json"
        self.index = self.snapshot / "model.safetensors.index.json"
        if _file_sha(self.config) != CONFIG_SHA256 or _file_sha(self.index) != INDEX_SHA256:
            raise ValueError("Pinned SmallThinker config/index SHA mismatch")
        self.mapping = json.loads(self.index.read_text())["weight_map"]
        if any(Path(n).name != n or not n.endswith(".safetensors") for n in self.mapping.values()):
            raise ValueError("Unsafe anchor shard filename")
        if set(self.mapping.values()) != set(SHARD_SHA256):
            raise ValueError("Pinned index references unexpected anchor shards")

    def reference_key(self, name):
        if not name.startswith("llm."):
            return None
        key = name.removeprefix("llm.")
        return key if key in self.mapping else None

    def descriptor(self, key):
        from safetensors import safe_open
        if key not in self.mapping:
            raise ValueError("Missing tensor in pinned anchor index")
        shard = self.mapping[key]
        path = self.snapshot / shard
        digest = _file_sha(path)
        if digest != SHARD_SHA256[shard]:
            raise ValueError("Anchor shard SHA differs from pinned canonical HF LFS metadata")
        before = _stat(path)
        with safe_open(path, framework="pt", device="cpu") as opened:
            tensor = opened.get_slice(key)
            shape, dtype = list(tensor.get_shape()), tensor.get_dtype()
        if _stat(path) != before:
            raise ValueError("Anchor changed while reading tensor headers")
        return {"snapshot": str(self.snapshot), "repo": REPO, "revision": REVISION,
                "index_sha256": INDEX_SHA256, "config_sha256": CONFIG_SHA256,
                "shard": shard, "shard_sha256": digest, "key": key,
                "shape": shape, "safetensors_dtype": dtype}

    def read(self, desc):
        from safetensors import safe_open
        if self.descriptor(desc["key"]) != desc:
            raise ValueError("Anchor identity/header/shard SHA changed")
        path = self.snapshot / desc["shard"]
        before = _stat(path)
        with safe_open(path, framework="pt", device="cpu") as opened:
            value = opened.get_tensor(desc["key"])
        if _stat(path) != before:
            raise ValueError("Anchor changed while reading tensor")
        return value


def _raw(tensor):
    return tensor.detach().contiguous().reshape(-1).view(__import__("torch").uint8).numpy().tobytes()


class _Session:
    def __init__(self, manifest):
        if manifest["codec_sha256"] != codec_sha256() or manifest["version"] != 1:
            raise ValueError("Lossless-delta decoder source/version differs")
        if manifest["repo"] != REPO or manifest["revision"] != REVISION or sys.byteorder != "little":
            raise ValueError("Only pinned SmallThinker on little-endian CPU is supported")
        descriptors = manifest["tensors"]
        if not descriptors or len(descriptors) > 10000:
            raise ValueError("Invalid number of encoded tensors")
        total = sum(s["decoded_bytes"] for s in descriptors.values())
        encoded = sum(s["encoded_bytes"] for s in descriptors.values())
        largest = max(s["decoded_bytes"] for s in descriptors.values())
        if total != manifest["decoded_bytes"] or encoded != manifest["encoded_bytes"]:
            raise ValueError("Session tensor byte totals disagree")
        if not 0 < total <= MAX_CHECKPOINT_DECODED_BYTES or not 0 < encoded <= MAX_PICKLE_BYTES or largest > MAX_TENSOR_DECODED_BYTES:
            raise ValueError("SmallThinker decoded allocation budget exceeded")
        for s in descriptors.values():
            if (any(type(v) is not int or v < 0 for v in s["shape"]) or
                type(s["decoded_bytes"]) is not int or type(s["encoded_bytes"]) is not int or
                not 0 < s["decoded_bytes"] <= MAX_TENSOR_DECODED_BYTES or
                s["decoded_bytes"] != math.prod(s["shape"]) * 4 or
                not 0 < s["encoded_bytes"] <= _zstd().ZSTD_compressBound(s["decoded_bytes"])):
                raise ValueError("Invalid encoded tensor shape/size bound")
        self.manifest = manifest
        self.seen = set()
        self.anchors = {}
        self.guard = None

    def __reduce__(self):
        return (_new_session, (self.manifest,))


def _new_session(manifest):
    import torch
    # Native ZIP publishes its current map_location here in installed Torch.
    # Legacy does not: its CPU-only callsite requirement remains explicit.
    location = getattr(getattr(torch.serialization, "_serialization_tls", None), "map_location", None)
    if location is not None and not (isinstance(location, str) and location == "cpu") and not (isinstance(location, torch.device) and location.type == "cpu"):
        raise ValueError("Delta values support CPU map_location only; callable/dict/GPU/meta maps are unsupported")
    return _Session(manifest)


class _TensorProxy:
    def __init__(self, session, name, data):
        self.session, self.name, self.data = session, name, data

    def __reduce__(self):
        return (_restore_tensor, (self.session, self.name, self.data))


def _restore_tensor(session, name, data):
    import torch
    if name in session.seen or name not in session.manifest["tensors"]:
        raise ValueError("Duplicate/unlisted tensor reconstruction")
    spec = session.manifest["tensors"][name]
    if not isinstance(data, bytes) or len(data) != spec["encoded_bytes"] or _sha(data) != spec["encoded_sha256"]:
        raise ValueError("Encoded tensor checksum/length changed")
    anchor_desc = spec["anchor"]
    path = anchor_desc["snapshot"]
    if path not in session.anchors:
        session.anchors[path] = SmallAnchor(path)
    anchor = session.anchors[path]
    if anchor.descriptor(anchor_desc["key"]) != anchor_desc:
        raise ValueError("Anchor metadata/shard changed")
    from torch._guards import detect_fake_mode
    if detect_fake_mode() is not None:
        # Structural inspection ONLY: no decompression or value/SHA proof.
        result = torch.empty(spec["shape"], dtype=torch.float32, device="cpu")
    else:
        if session.guard is None:
            m = session.manifest
            # data.pkl/BytesIO/unpickler may coexist with compressed bytes and
            # decoded heap tensors. Credit neither mmap nor page-cache reclaim.
            session.guard = memory_guard(3 * m["encoded_bytes"] + 2 * m["decoded_bytes"] +
                                         6 * max(s["decoded_bytes"] for s in m["tensors"].values()))
        reference = anchor.read(anchor_desc)
        if reference.dtype != torch.bfloat16 or _sha(_raw(reference)) != spec["anchor_tensor_sha256"]:
            raise ValueError("Original BF16 tensor checksum changed")
        shuffled = _decompress(data, spec["decoded_bytes"])
        lanes = torch.frombuffer(bytearray(shuffled), dtype=torch.uint8).reshape(4, -1)
        delta = lanes.t().contiguous().view(torch.int32).reshape(-1)
        result = torch.bitwise_xor(delta, reference.float().reshape(-1).view(torch.int32)).view(torch.float32).reshape(spec["shape"])
        if _sha(_raw(result)) != spec["decoded_sha256"]:
            raise ValueError("Reconstructed FP32 tensor checksum changed")
    session.seen.add(name)
    return result


class DeltaPayload:
    """Writer-side wrapper; ordinary torch.load returns the original dict."""
    def __init__(self, payload, session):
        self.payload, self.session = payload, session

    def __reduce__(self):
        return (_finish_payload, (self.session, self.payload))


def _finish_payload(session, payload):
    if session.seen != set(session.manifest["tensors"]):
        raise ValueError("Incomplete delta tensor reconstruction")
    if payload.get("model_ref") != {"repo": REPO, "revision": REVISION}:
        raise ValueError("Decoded payload is not bound to pinned SmallThinker")
    return payload


def _walk_tensors(value, prefix=()):
    import torch
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _walk_tensors(v, prefix + (k,))
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from _walk_tensors(v, prefix + (i,))


def prepare_delta_payload(payload, snapshot):
    """Encode matching original parameters only; return wrapper and small receipt.

    Original payload/tensors are not mutated. Requires dense contiguous CPU F32
    parameters. Repeated references to the identical model Tensor preserve ties;
    shared-storage views and model/nonmodel storage aliases are refused.
    """
    import torch
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("Expected trusted research payload with model mapping")
    if payload.get("model_ref") != {"repo": REPO, "revision": REVISION}:
        raise ValueError("Only payloads explicitly bound to pinned SmallThinker are supported")
    anchor = SmallAnchor(snapshot)
    candidates, normal = {}, []
    for name, tensor in payload["model"].items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Model entries must be named tensors")
        key = anchor.reference_key(name)
        if key is None:
            normal.append({"name": name, "reason": "novel_or_unmatched_key"})
            continue
        desc = anchor.descriptor(key)
        if list(tensor.shape) != desc["shape"] or desc["safetensors_dtype"] != "BF16" or tensor.dtype != torch.float32:
            normal.append({"name": name, "reason": "shape_or_dtype_mismatch"})
            continue
        if tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.layout != torch.strided or tensor.requires_grad:
            raise ValueError("Encoded parameters require detached contiguous dense CPU tensors")
        if tensor.storage_offset() != 0 or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size():
            raise ValueError("Encoded tensor views/oversized backing storages are refused")
        if not 0 < tensor.numel() * 4 <= MAX_TENSOR_DECODED_BYTES:
            raise ValueError("Encoded tensor exceeds1GiB allocation limit")
        candidates[name] = (tensor, desc)
    if not candidates:
        raise ValueError("No original pinned BF16 tensors match the payload")
    ids = {id(t) for t, _ in candidates.values()}
    storage_uses = {}
    for path, t in _walk_tensors(payload):
        if t.device.type != "cpu" or t.layout != torch.strided:
            continue
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        storage_uses.setdefault(key, []).append((path, t))
    for uses in storage_uses.values():
        selected = [(p, t) for p, t in uses if id(t) in ids]
        if selected and (len({id(t) for _, t in uses}) != 1 or any(len(p) != 2 or p[0] != "model" for p, _ in uses)):
            raise ValueError("Encoded tensor storage aliases/views outside identical model references are refused")
    unique = {id(t): (name, t, desc) for name, (t, desc) in reversed(list(candidates.items()))}
    raw_total = sum(t.numel() * 4 for _, t, _ in unique.values())
    largest = max(t.numel() * 4 for _, t, _ in unique.values())
    if raw_total > MAX_CHECKPOINT_DECODED_BYTES:
        raise ValueError("SmallThinker checkpoint decoded budget exceeds32GiB")
    # Worst-case compressed bytes and Torch ZIP pickle buffer copies; original
    # model/optimizer residency is already reflected in current cgroup usage.
    guard = memory_guard(4 * raw_total + 6 * largest)
    descriptors, encoded = {}, {}
    for identity, (name, tensor, desc) in unique.items():
        reference = anchor.read(desc)
        bits = torch.bitwise_xor(tensor.reshape(-1).view(torch.int32), reference.float().reshape(-1).view(torch.int32))
        shuffled = bits.view(torch.uint8).reshape(-1, 4).t().contiguous().numpy().tobytes()
        data = _compress(shuffled)
        descriptors[name] = {"shape": list(tensor.shape), "decoded_bytes": tensor.numel() * 4,
                             "decoded_sha256": _sha(_raw(tensor)), "encoded_bytes": len(data),
                             "encoded_sha256": _sha(data), "anchor": desc,
                             "anchor_tensor_sha256": _sha(_raw(reference))}
        encoded[identity] = (name, data)
    manifest = {"version": 1, "repo": REPO, "revision": REVISION,
                "codec_sha256": codec_sha256(), "decoded_bytes": raw_total,
                "encoded_bytes": sum(s["encoded_bytes"] for s in descriptors.values()),
                "tensors": descriptors}
    session = _Session(manifest)
    proxies = {identity: _TensorProxy(session, name, data) for identity, (name, data) in encoded.items()}
    transformed = dict(payload)
    transformed["model"] = {name: proxies.get(id(t), t) for name, t in payload["model"].items()}
    alias_names = [s["name"] for s in normal if id(payload["model"][s["name"]]) in proxies]
    normal = [s for s in normal if s["name"] not in alias_names]
    receipt = {"status": "writer_proxy_prepared_only", "repo": REPO, "revision": REVISION,
               "codec_sha256": manifest["codec_sha256"], "unique_encoded_tensors": len(descriptors),
               "encoded_model_references": len(candidates) + len(alias_names), "normal_model_tensors": normal,
               "identical_tensor_alias_names_preserved": alias_names,
               "raw_matched_tensor_bytes": raw_total, "encoded_bytes": manifest["encoded_bytes"],
               "encoded_to_raw_matched_ratio": manifest["encoded_bytes"] / raw_total,
               "zstd_version": _zstd().ZSTD_versionString().decode(), "memory_guard": guard,
               "max_checkpoint_decoded_bytes": MAX_CHECKPOINT_DECODED_BYTES,
               "max_tensor_decoded_bytes": MAX_TENSOR_DECODED_BYTES,
               "largest_encoded_tensor_decoded_bytes": largest,
               "optimizer_and_rng_encoded": False, "cpu_map_location_only": True}
    return DeltaPayload(transformed, session), receipt
