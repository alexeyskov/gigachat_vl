"""HF-cache metadata adapter for the frozen, derived-copy-only trained probe.

Only the anchor-header reader is replaced in an isolated instance of the frozen
probe. Run/manager JSON keeps its original no-symlink rule. No decoder, tensor
math, source gate, CPU/disk guard, or scientific pointer is changed here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

from src.utils import research_trained_delta_probe as frozen

ROOT = frozen.ROOT
OWNER = 'completed_full4f_small_derived_copy_probe_hf_metadata_v2'
FROZEN_SHA256 = '67da3a9ab87df1999a11a124354cf2e8623a6ecc3bfd30ca67729ac44b0215d3'
SNAPSHOT = Path('/workspace/.hf_home/hub/models--Tiiny--SmallThinker-4BA0.6B-Instruct/snapshots')/frozen.REVISION
SOURCES = ('src/utils/research_trained_delta_probe_v2.py', 'scripts/probe_research_trained_small_delta_v2.py',
    'tests/test_research_trained_delta_probe_v2.py', 'src/utils/research_trained_delta_probe.py',
    'scripts/probe_research_trained_small_delta.py', 'src/utils/research_lossless_delta.py',
    'src/utils/research_stream_delta.py', 'src/utils/research_checkpoint_compat.py',
    'src/utils/research_checkpoint_io.py', 'src/utils/research_checkpoint_bundle.py', 'src/utils/research_storage.py')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON key in pinned metadata')
        result[key] = value
    return result


def _bounded_metadata(path, expected_sha):
    """HF symlinks only: resolve, bound, and verify actual bytes every time."""
    path = Path(path); target = path.resolve(strict=True)
    before = frozen.stat(target)
    if not target.is_file() or not 0 < before['st_size'] <= 16*2**20:
        raise ValueError('Bounded regular pinned HF metadata required')
    with target.open('rb') as handle: raw = handle.read(16*2**20+1)
    if (len(raw) != before['st_size'] or frozen.stat(target) != before or
            path.resolve(strict=True) != target or hashlib.sha256(raw).hexdigest() != expected_sha):
        raise ValueError('Pinned HF metadata bytes/identity changed')
    value = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(value, dict): raise ValueError('Pinned HF metadata must be a JSON object')
    return value, {'path': str(path), 'resolved_path': str(target), 'is_symlink': path.is_symlink(),
        'stat': before, 'sha256': expected_sha}


def anchor_census(snapshot=None):
    """Read config/index and both headers only; never tensor values or Torch."""
    from src.utils import research_lossless_delta as v1
    snapshot = Path(snapshot) if snapshot is not None else SNAPSHOT
    if snapshot.name != frozen.REVISION: raise ValueError('Pinned Small snapshot revision required')
    snapshot = snapshot.resolve(strict=True)
    config, config_binding = _bounded_metadata(snapshot/'config.json', v1.CONFIG_SHA256)
    index, index_binding = _bounded_metadata(snapshot/'model.safetensors.index.json', v1.INDEX_SHA256)
    mapping = index.get('weight_map')
    if (not isinstance(mapping, dict) or not mapping or any(type(k) is not str or not k for k in mapping) or
            any(type(name) is not str or Path(name).name != name for name in mapping.values()) or
            set(mapping.values()) != set(v1.SHARD_SHA256)):
        raise ValueError('Pinned index shard mapping differs')
    shapes = {}; shards = {}; stored_elements = 0
    for name in sorted(v1.SHARD_SHA256):
        path = snapshot/name; target = path.resolve(strict=True); before = frozen.stat(target)
        if not target.is_file(): raise ValueError('Pinned anchor shard must be a regular resolved file')
        with target.open('rb') as handle:
            prefix = handle.read(8)
            if len(prefix) != 8: raise ValueError('Truncated safetensors header length')
            size = struct.unpack('<Q', prefix)[0]
            if not 0 < size <= 16*2**20 or 8+size > before['st_size']:
                raise ValueError('Bounded safetensors header required')
            raw = handle.read(size)
        if len(raw) != size: raise ValueError('Truncated safetensors header')
        header = json.loads(raw, object_pairs_hook=_unique); ranges = []; count = 0; elements = 0
        if not isinstance(header, dict): raise ValueError('Safetensors header must be an object')
        for key, entry in header.items():
            if key == '__metadata__': continue
            if key in shapes or mapping.get(key) != name or not isinstance(entry, dict) or entry.get('dtype') != 'BF16':
                raise ValueError('Pinned BF16 index/header keys differ')
            shape = entry.get('shape'); offsets = entry.get('data_offsets')
            if (not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape) or
                    not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int or n < 0 for n in offsets) or
                    offsets[1]-offsets[0] != math.prod(shape)*2 or offsets[1] > before['st_size']-8-size):
                raise ValueError('Pinned tensor shape/offset bound differs')
            shapes[key] = shape; count += 1; elements += math.prod(shape); ranges.append(offsets)
        ranges.sort(); cursor = 0
        for start, end in ranges:
            if start != cursor: raise ValueError('Overlapping or missing safetensors tensor bytes')
            cursor = end
        if cursor != before['st_size']-8-size: raise ValueError('Unaccounted safetensors payload bytes')
        if frozen.stat(target) != before or path.resolve(strict=True) != target:
            raise ValueError('Anchor changed while reading metadata headers')
        stored_elements += elements
        shards[name] = {'path': str(path), 'resolved_path': str(target), 'is_symlink': path.is_symlink(),
            'stat': before, 'header_bytes': size, 'header_sha256': hashlib.sha256(prefix+raw).hexdigest(),
            'expected_full_file_sha256': v1.SHARD_SHA256[name], 'full_file_sha256_checked_in_metadata_census': False,
            'stored_tensors': count, 'stored_bf16_elements': elements}
    if set(shapes) != set(mapping): raise ValueError('Incomplete anchor headers')
    expected = {'llm.'+name: shape for name, shape in shapes.items()}
    removed_tied = None
    if config.get('tie_word_embeddings') is True:
        embed = expected.get('llm.model.embed_tokens.weight'); head = expected.get('llm.lm_head.weight')
        if embed is None or head != embed: raise ValueError('Declared tied head/embedding metadata differs')
        removed_tied = 'llm.lm_head.weight'; expected.pop(removed_tied)
    elements = sum(math.prod(shape) for shape in expected.values())
    if elements != frozen.BASE_ELEMENTS: raise ValueError('Pinned unique original parameter census differs')
    # Recheck small canonical bytes after every shard; metadata symlink replacement
    # cannot silently bind a report to a different config/index.
    if _bounded_metadata(snapshot/'config.json', v1.CONFIG_SHA256)[1] != config_binding or _bounded_metadata(
            snapshot/'model.safetensors.index.json', v1.INDEX_SHA256)[1] != index_binding:
        raise ValueError('Pinned config/index identity changed during header census')
    return {'schema_version': 1, 'protocol': OWNER, 'status': 'pinned_anchor_headers_verified_without_tensor_values',
        'snapshot': str(snapshot), 'repo': frozen.REPO, 'revision': frozen.REVISION,
        'config': config_binding, 'index': index_binding, 'shards': shards, 'shapes': expected,
        'stored_tensors': len(shapes), 'stored_bf16_elements': stored_elements,
        'unique_native_tensors': len(expected), 'unique_native_elements': elements,
        'removed_tied_head': removed_tied, 'tensor_values_read': False, 'full_shard_bytes_hashed': False,
        'decoder_later_requires_full_canonical_shard_sha256': True}


def source_binding():
    result = {name: frozen.file_sha(ROOT/name) for name in SOURCES}
    if result['src/utils/research_trained_delta_probe.py'] != FROZEN_SHA256:
        raise ValueError('Frozen derived-copy probe source changed')
    return result


def _implementation():
    """No global monkeypatch: only this private instance gets the new reader."""
    source_binding()
    spec = importlib.util.spec_from_file_location('_isolated_trained_probe_hf_metadata_v2', frozen.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module._anchor_headers = lambda snapshot: anchor_census(snapshot)['shapes']
    return module


def _profile(census, sources):
    public = {key: value for key, value in census.items() if key != 'shapes'}
    return {'protocol': OWNER, 'only_override': '_anchor_headers', 'frozen_probe_sha256': FROZEN_SHA256,
        'source_sha256': sources, 'anchor_metadata': public, 'run_json_symlinks_allowed': False,
        'decoder_or_native_loader_changed': False, 'scientific_publication_authorized': False}


def preflight(run, *, role='best', snapshot=None, max_staging_bytes=24*frozen.GIB):
    sources = source_binding(); implementation = _implementation()
    result = implementation.preflight(run, role=role, snapshot=snapshot, max_staging_bytes=max_staging_bytes)
    if result['status'] == 'gate_closed':
        return {**result, 'adapter_protocol': OWNER, 'adapter_source_sha256': sources}
    census = anchor_census(snapshot); result['metadata_adapter'] = _profile(census, sources)
    if sources != source_binding(): raise ValueError('Adapter sources changed during preflight')
    return result


def run_probe(run, output, *, role='best', snapshot=None, max_staging_bytes=24*frozen.GIB,
              reserve_gb=120, advise_source=False, cleanup_verified=False, progress=None):
    """Same frozen probe; extend its proof before optional derived-file cleanup."""
    sources = source_binding(); census = anchor_census(snapshot); implementation = _implementation()
    result = implementation.run_probe(run, output, role=role, snapshot=snapshot,
        max_staging_bytes=max_staging_bytes, reserve_gb=reserve_gb, advise_source=advise_source,
        cleanup_verified=False, progress=progress)
    if sources != source_binding() or census != anchor_census(snapshot):
        raise ValueError('Adapter source/anchor metadata changed during actual probe; derived artifact preserved')
    result['metadata_adapter'] = _profile(census, sources)
    output = Path(output).resolve(); implementation._atomic(output/'verified.json', result, frozen.METADATA_LIMIT)
    if cleanup_verified:
        persisted = frozen._json(output/'verified.json'); archive = Path(result['archive_path'])
        if (persisted != result or frozen._json(output/'owner.json')['owner'] != result['owner'] or
                archive.is_symlink() or archive.parent != output or frozen.file_sha(archive) != result['archive_sha256']):
            raise ValueError('V2 durable owner/archive/metadata proof differs; cleanup refused')
        archive.unlink(); result['archive_deleted'] = True
        implementation._atomic(output/'verified.json', result, frozen.METADATA_LIMIT)
    return result
