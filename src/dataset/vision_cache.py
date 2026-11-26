"""Versioned, image-content addressed frozen vision feature cache.

No model-specific projection or language state belongs here. ``encode_fn`` is
called in the caller's GPU process and only receives unique cache misses.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Sequence
import hashlib
import json
import os
import shutil
import tempfile
import fcntl

SCHEMA_VERSION = 1


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def encoder_fingerprint(config: dict) -> str:
    payload = {"cache_schema_version": SCHEMA_VERSION, **config}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def image_content_hash(image_or_path) -> str:
    from PIL import Image, ImageOps
    close = isinstance(image_or_path, (str, Path))
    image = Image.open(image_or_path) if close else image_or_path
    try:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        h = hashlib.sha256()
        h.update(f"RGB:{rgb.width}:{rgb.height}:".encode())
        h.update(rgb.tobytes())
        return h.hexdigest()
    finally:
        if close:
            image.close()


def ensure_disk_space(path, required_bytes: int = 0, reserve_bytes: int = 120 * 1024**3):
    path = Path(path)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free - int(required_bytes) < reserve_bytes:
        raise OSError(f"Disk reserve reached: free={free}, required={required_bytes}, reserve={reserve_bytes}")


def _tensor_checksum(tensor):
    # uint8 avoids numpy's lack of bfloat16 support.
    return hashlib.sha256(tensor.contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


class VisionFeatureCache:
    def __init__(self, root, fingerprint: str, *, reserve_bytes=120 * 1024**3,
                 ram_max_bytes=2 * 1024**3):
        self.fingerprint = fingerprint
        self.root = Path(root) / fingerprint
        self.root.mkdir(parents=True, exist_ok=True)
        self.reserve_bytes = reserve_bytes
        self.ram_max_bytes = ram_max_bytes
        self._ram = OrderedDict()
        self._ram_bytes = 0
        self.stats = {"requests": 0, "hits": 0, "misses": 0,
                      "encoded_images": 0, "encode_calls": 0, "bytes_written": 0}

    def key(self, image_hash):
        return hashlib.sha256((self.fingerprint + image_hash).encode()).hexdigest()

    def path_for_hash(self, image_hash):
        key = self.key(image_hash)
        return self.root / key[:2] / (key + ".pt")

    def _remember(self, image_hash, tensor):
        size = tensor.numel() * tensor.element_size()
        if size > self.ram_max_bytes:
            return
        if image_hash in self._ram:
            old = self._ram.pop(image_hash)
            self._ram_bytes -= old.numel() * old.element_size()
        while self._ram and self._ram_bytes + size > self.ram_max_bytes:
            _, old = self._ram.popitem(last=False)
            self._ram_bytes -= old.numel() * old.element_size()
        self._ram[image_hash] = tensor
        self._ram_bytes += size

    def _read(self, image_hash):
        import torch
        if image_hash in self._ram:
            self._ram.move_to_end(image_hash)
            return self._ram[image_hash]
        path = self.path_for_hash(image_hash)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("fingerprint") != self.fingerprint or payload.get("image_hash") != image_hash:
            raise ValueError(f"Feature metadata mismatch: {path}")
        tensor = payload.get("features")
        if not torch.is_tensor(tensor) or tensor.ndim != 2 or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid feature tensor: {path}")
        if payload.get("checksum") != _tensor_checksum(tensor):
            raise ValueError(f"Feature checksum mismatch: {path}")
        with torch.inference_mode(False):
            tensor = tensor.detach().clone()
        self._remember(image_hash, tensor)
        return tensor

    def _write(self, image_hash, tensor):
        import torch
        metadata = {}
        if isinstance(tensor, dict):
            metadata = tensor.get("metadata", {})
            tensor = tensor["features"]
        # clone outside inference_mode: downstream MLP must be able to save inputs.
        with torch.inference_mode(False):
            tensor = tensor.detach().to("cpu").clone()
        if tensor.ndim != 2 or not torch.isfinite(tensor).all():
            raise ValueError("Vision encoder must return finite [num_tokens, hidden_size] tensors")
        path = self.path_for_hash(image_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        ensure_disk_space(self.root, tensor.numel()*tensor.element_size()+8192, self.reserve_bytes)
        payload = {"schema_version": SCHEMA_VERSION, "fingerprint": self.fingerprint,
                   "image_hash": image_hash, "features": tensor, "checksum": _tensor_checksum(tensor),
                   "shape": list(tensor.shape), "storage_dtype": str(tensor.dtype),
                   "valid_token_count": tensor.shape[0], "metadata": metadata}
        fd, temp = tempfile.mkstemp(prefix=".feature-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        self.stats["bytes_written"] += path.stat().st_size
        self._remember(image_hash, tensor)
        return tensor

    def get_many(self, image_paths: Sequence[str], encode_fn: Callable, *, read_only=False,
                 image_hashes: Sequence[str] | None = None):
        if image_hashes is not None and len(image_hashes) != len(image_paths):
            raise ValueError("image_paths/image_hashes length mismatch")
        hashes = list(image_hashes) if image_hashes is not None else [image_content_hash(p) for p in image_paths]
        self.stats["requests"] += len(hashes)
        unique = dict(zip(hashes, image_paths))
        found = {h: self._read(h) for h in unique}
        missing = sorted(h for h, value in found.items() if value is None)
        self.stats["hits"] += len(unique)-len(missing)
        self.stats["misses"] += len(missing)
        if missing and read_only:
            raise FileNotFoundError(f"read_only vision cache is missing {len(missing)} unique images")
        if missing:
            # Sorted acquisition prevents cycles between concurrent batches.
            with ExitStack() as stack:
                for h in missing:
                    path = self.path_for_hash(h)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    lock = stack.enter_context(open(str(path)+".lock", "a+b"))
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                for h in missing:
                    found[h] = self._read(h)
                remaining = [h for h in missing if found[h] is None]
                if remaining:
                    tensors = list(encode_fn([unique[h] for h in remaining]))
                    if len(tensors) != len(remaining):
                        raise ValueError("Encoder result count does not match unique miss count")
                    self.stats["encode_calls"] += 1
                    self.stats["encoded_images"] += len(remaining)
                    for h, tensor in zip(remaining, tensors):
                        found[h] = self._write(h, tensor)
        return [found[h] for h in hashes]
