"""Isolated derived-copy measurement of a completed full4f Small checkpoint.

No manager pointer, scientific object, optimizer, or source byte is rewritten.
The default preflight creates nothing. The optional probe writes a new task-owned
archive and verifies the complete payload through the unchanged CPU loader.
Only trusted local Torch payloads are supported; ordinary pickle is required for
the original scheduler, RNG and metadata. This is not an untrusted-file sandbox.
"""
from __future__ import annotations

import contextlib
import fcntl
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import struct
import threading
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPO = 'Tiiny/SmallThinker-4BA0.6B-Instruct'
REVISION = 'a24932bb880c85c04c794283202a9194b1b42709'
BASE_ELEMENTS = 4_035_085_824
GIB = 2**30
NORMAL_LIMIT = 64*2**20
METADATA_LIMIT = 64*2**20
SOURCE_LIMIT = 24*GIB
OWNER = 'completed_full4f_small_derived_copy_probe_v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def stat(path):
    value = Path(path).stat()
    return {name: getattr(value, name) for name in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def file_sha(path):
    path = Path(path); before = stat(path); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''): digest.update(block)
    if stat(path) != before: raise ValueError('File changed while hashing: '+str(path))
    return digest.hexdigest()


def _json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16*2**20:
        raise ValueError('Required local bounded JSON file: '+str(path))
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict): raise ValueError('Expected JSON object: '+str(path))
    return value


class GateClosed(ValueError):
    pass


@contextlib.contextmanager
def _source_lock(run):
    path = Path(run)/'.checkpoint.lock'
    if path.is_symlink() or not path.is_file(): raise GateClosed('Completed run has no existing local manager lock')
    with path.open('rb') as handle:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def gate(run, role='best'):
    """Metadata-only gate. Never instantiate a manager or create a lock/file."""
    run = Path(run).expanduser().resolve()
    if role not in ('best', 'last'): raise GateClosed('Only retained best or last is accepted')
    if not run.is_dir(): raise GateClosed('Full4f run does not exist yet')
    if any(token in run.name.lower() for token in ('smoke', 'fastbench', 'technical')):
        raise GateClosed('Technical/smoke runs are outside this trained probe')
    try:
        config = _json(run/'resolved_config.json'); completion = _json(run/'completed.json')
        manager = _json(run/'.checkpoint_manager.json'); compaction = _json(run/'checkpoint_compaction.json')
        resolved_output = Path(config.get('output', ''))
        resolved_output = (resolved_output if resolved_output.is_absolute() else ROOT/resolved_output).resolve()
        if (config.get('model') != REPO or config.get('stage') != 'SFT' or config.get('recipe') != '4f' or
                config.get('checkpoint_format', 'legacy') != 'legacy' or resolved_output != run):
            raise GateClosed('Only an actual full4f pinned Small SFT run with raw legacy_single_file checkpoint is accepted')
        if (type(completion.get('steps')) is not int or completion['steps'] < 1 or
                completion.get('status') in ('failed', 'incomplete', 'interrupted')):
            raise GateClosed('No successful training completion')
        if (compaction.get('status') != 'completed' or compaction.get('training_completed') is not True or
                compaction.get('resumable_optimizer') is not False or Path(compaction.get('run_dir', '')).resolve() != run):
            raise GateClosed('Completed optimizer compaction receipt is required')
        if not isinstance(manager.get('owner'), str) or not manager['owner'] or manager.get('mode') not in ('min', 'max'):
            raise GateClosed('Manager ownership/mode is missing')
        objects = run/'.checkpoint_objects'
        if objects.is_symlink() or not objects.is_dir(): raise GateClosed('Local manager object directory required')
        roles = {}
        for name in ('best', 'last'):
            pointer = run/name
            if not pointer.is_symlink(): raise GateClosed('Both best and last must be owned manager pointers')
            target = pointer.resolve(strict=True)
            if target.parent != objects or target.is_symlink() or not target.name.startswith('object-'):
                raise GateClosed('Retained pointer escaped owned manager objects')
            marker = _json(target/'.research_checkpoint.json')
            if (marker.get('owner') != manager['owner'] or marker.get('training_completed') is not True or
                    marker.get('resumable_optimizer') is not False or marker.get('optimizer_state_removed') is not True or
                    marker.get('created_unix', math.inf) > (run/'completed.json').stat().st_mtime+1):
                raise GateClosed('Retained source is not completed/compacted and manager-owned')
            if name == 'last' and marker.get('step') != completion['steps']:
                raise GateClosed('LAST step differs from completed training')
            source = target/'state.pt'
            if source.is_symlink() or not source.is_file() or not 0 < source.stat().st_size <= SOURCE_LIMIT:
                raise GateClosed('Bounded local raw checkpoint required')
            roles[name] = {'object': target.name, 'pointer': os.readlink(pointer), 'marker': marker,
                'marker_sha256': file_sha(target/'.research_checkpoint.json'), 'source': str(source), 'stat': stat(source)}
        selected = roles[role]
        if not any(item.get('new_object') == selected['object'] for item in compaction.get('checkpoints', [])):
            raise GateClosed('Selected object is absent from completed compaction receipt')
        small_files = ('resolved_config.json', 'completed.json', '.checkpoint_manager.json', 'checkpoint_compaction.json')
        return {'status': 'gate_open_metadata_only', 'run': str(run), 'role': role, 'config': config,
            'completion': completion, 'manager': manager, 'roles': roles, 'selected': selected,
            'metadata_sha256': {name: file_sha(run/name) for name in small_files}}
    except (OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, GateClosed): raise
        raise GateClosed('Incomplete or malformed completed full4f source: '+str(error)) from error


def _anchor_headers(snapshot):
    from src.utils import research_lossless_delta as v1
    anchor = v1.SmallAnchor(snapshot); result = {}
    for shard in sorted(set(anchor.mapping.values())):
        with (Path(snapshot)/shard).open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < size <= 16*2**20: raise ValueError('Anchor header bound exceeded')
            header = json.loads(stream.read(size))
        for name, entry in header.items():
            if name == '__metadata__': continue
            if anchor.mapping.get(name) != shard or entry['dtype'] != 'BF16': raise ValueError('Anchor index/header differs')
            result['llm.'+name] = entry['shape']
    if set(result) != {'llm.'+name for name in anchor.mapping}: raise ValueError('Incomplete anchor headers')
    config = _json(Path(snapshot)/'config.json')
    if config.get('tie_word_embeddings') is True and 'llm.model.embed_tokens.weight' in result:
        result.pop('llm.lm_head.weight', None)
    if sum(math.prod(shape) for shape in result.values()) != BASE_ELEMENTS:
        raise ValueError('Native full4f unique original parameter census differs')
    return result


def _raw_zip_preflight(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist(); pickles = [item for item in members if item.filename.endswith('/data.pkl')]
        if (not 0 < len(members) <= 20000 or len(pickles) != 1 or pickles[0].file_size > METADATA_LIMIT or
                any(item.compress_type != zipfile.ZIP_STORED for item in members)):
            raise ValueError('Only bounded ordinary uncompressed Torch ZIP source is accepted')
        return {'members': len(members), 'data_pickle_bytes': pickles[0].file_size,
                'source_mmap_file_cache_allowance_bytes': Path(path).stat().st_size}


def _census(payload, expected):
    import torch
    if (not isinstance(payload, dict) or not isinstance(payload.get('model'), dict) or
            payload.get('model_ref') != {'repo': REPO, 'revision': REVISION} or 'optimizer' in payload or
            payload.get('training_completed') is not True or payload.get('resumable_optimizer') is not False):
        raise ValueError('Payload must be pinned full Small and already optimizer-compacted')
    model = payload['model']
    if any(not isinstance(name,str) for name in model):raise ValueError('Model keys must be strings')
    if set(name for name in model if name.startswith('llm.')) != set(expected):
        raise ValueError('Full4f original model keys missing/extra; partial/adapter checkpoints refused')
    storage_uses = {}; normal = {}; model_elements = 0; largest = 0
    def walk(value, selected=False):
        nonlocal model_elements, largest
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage(); key = (storage._cdata, storage.nbytes())
            storage_uses.setdefault(key, []).append(selected)
            if not selected: normal[key] = storage.nbytes()
        elif isinstance(value, dict):
            for key, child in value.items(): walk(child, value is model and key in expected)
        elif isinstance(value, (tuple, list)):
            for child in value: walk(child)
    walk(payload)
    for name, shape in expected.items():
        tensor = model[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or list(tensor.shape) != shape or
                tensor.layout != torch.strided or not tensor.is_contiguous() or tensor.requires_grad or tensor.storage_offset() or
                tensor.untyped_storage().nbytes() != tensor.numel()*4):
            raise ValueError('Original full4f tensor must be owned contiguous FP32: '+name)
        key = (tensor.untyped_storage()._cdata, tensor.untyped_storage().nbytes())
        if len(storage_uses[key]) != 1: raise ValueError('Encoded aliases/views are refused; no tie is silently changed')
        model_elements += tensor.numel(); largest = max(largest, tensor.numel()*4)
    if not 0 < largest <= GIB or model_elements != BASE_ELEMENTS or sum(normal.values()) > NORMAL_LIMIT:
        raise ValueError('Full4f model or small normal-storage budget differs')
    return {'original_tensors': len(expected), 'original_fp32_bytes': model_elements*4,
            'largest_original_tensor_bytes': largest, 'normal_storage_bytes': sum(normal.values()),
            'normal_storage_limit_bytes': NORMAL_LIMIT, 'model_key_order': list(model)}


def payload_signature(payload):
    """Every scalar/container and every raw backing-storage byte, with aliases."""
    import numpy as np
    import torch
    objects = {}; storages = {}; visiting = set(); tensor_count = 0
    def walk(value):
        nonlocal tensor_count
        if isinstance(value, torch.Tensor):
            if type(value) is not torch.Tensor or value.device.type != 'cpu' or value.layout != torch.strided:
                raise ValueError('Trusted plain CPU dense tensors only')
            if value.is_conj() or value.is_neg() or value.is_quantized or any(name is not None for name in getattr(value,'names',())):
                raise ValueError('Lazy/quantized/named tensor metadata is outside this probe')
            if id(value) in objects: return ['object_ref', objects[id(value)]]
            number = objects.setdefault(id(value), len(objects)); tensor_count += 1
            storage = value.untyped_storage(); key = (storage._cdata, storage.nbytes())
            if key not in storages:
                backing = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
                digest = hashlib.sha256(memoryview(backing.numpy())).hexdigest()
                storages[key] = [len(storages), storage.nbytes(), digest]
            return ['tensor', number, str(value.dtype), list(value.shape), list(value.stride()), value.storage_offset(),
                    value.requires_grad, storages[key]]
        if isinstance(value, np.ndarray):
            if id(value) in objects: return ['object_ref', objects[id(value)]]
            native_pickle_base = isinstance(value.base,bytes) and len(value.base)==value.nbytes
            if value.dtype.hasobject or (value.base is not None and not native_pickle_base) or not (value.flags.c_contiguous or value.flags.f_contiguous):
                raise ValueError('NumPy metadata views are outside this small-normal-buffer probe')
            number = objects.setdefault(id(value), len(objects))
            return ['ndarray', number, value.dtype.str, list(value.shape), list(value.strides),
                    hashlib.sha256(value.tobytes(order='A')).hexdigest()]
        if type(value) in (dict, list, tuple):
            if id(value) in visiting: raise ValueError('Cyclic checkpoint containers refused')
            if id(value) in objects: return ['object_ref', objects[id(value)]]
            number = objects.setdefault(id(value), len(objects)); visiting.add(id(value))
            try:
                items = [[walk(key), walk(child)] for key, child in value.items()] if type(value) is dict else [walk(child) for child in value]
                return [type(value).__name__, number, items]
            finally: visiting.remove(id(value))
        if value is None: return ['none']
        if type(value) in (str, bool, int): return [type(value).__name__, value]
        if type(value) is float: return ['float64_bits', struct.pack('!d', value).hex()]
        if type(value) in (bytes, bytearray): return [type(value).__name__, len(value), hashlib.sha256(value).hexdigest()]
        raise ValueError('Unsupported checkpoint metadata type: '+str(type(value)))
    structure = walk(payload); encoded = canonical(structure)
    if len(encoded) > METADATA_LIMIT//2: raise ValueError('Full payload hash manifest exceeds metadata allowance')
    return {'structure': structure, 'sha256': hashlib.sha256(encoded).hexdigest(), 'tensor_references': tensor_count,
        'unique_storages': len(storages), 'storage_bytes': sum(item[1] for item in storages.values())}


def _clone_template(payload, selected):
    """Copy normal storages in full; preserve normal views/object/container ties."""
    import numpy as np
    import torch
    memo = {}; storages = {}; total = 0; visiting = set()
    def clone(value):
        nonlocal total
        if id(value) in memo: return memo[id(value)]
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage(); key = (storage._cdata, storage.nbytes())
            if key not in storages:
                total += storage.nbytes()
                if total > NORMAL_LIMIT: raise ValueError('Normal storage cloning bound exceeded')
                storages[key] = storage.clone()
            result = torch.empty(0, dtype=value.dtype).set_(storages[key], value.storage_offset(), value.shape, value.stride())
            memo[id(value)] = result; return result
        if isinstance(value, np.ndarray):
            result = value.copy(order='A'); memo[id(value)] = result; return result
        if type(value) in (dict, list, tuple):
            if id(value) in visiting: raise ValueError('Cyclic checkpoint containers refused')
            visiting.add(id(value))
            try:
                if type(value) is dict:
                    result = {}; memo[id(value)] = result
                    for key, child in value.items():
                        if value is payload['model'] and key in selected: continue
                        result[clone(key)] = clone(child)
                else:
                    items = [clone(child) for child in value]; result = tuple(items) if type(value) is tuple else items
                    memo[id(value)] = result
                return result
            finally: visiting.remove(id(value))
        return value
    return clone(payload)


def _encode_source(payload, expected, census, snapshot, max_archive, progress):
    from src.utils import research_lossless_delta as v1, research_stream_delta as stream
    signature = payload_signature(payload); template = _clone_template(payload, set(expected)); last = time.monotonic()
    def tensors():
        nonlocal last
        for index, name in enumerate(expected):
            yield name, payload['model'][name], snapshot
            if progress and time.monotonic()-last >= 30:
                progress('encoding', tensors=index+1, total=len(expected)); last = time.monotonic()
    upper = sum(v1._zstd().ZSTD_compressBound(math.prod(shape)*4) for shape in expected.values())
    wrapped, receipt = stream.stream_encode_small(template, tensors(), expected_decoded_bytes=census['original_fp32_bytes'],
        expected_tensor_count=len(expected), largest_tensor_bytes=census['largest_original_tensor_bytes'],
        encoded_upper_bytes=upper, max_encoded_bytes=max_archive-census['normal_storage_bytes']-stream.METADATA_LIMIT)
    model = wrapped.payload.payload['model']
    wrapped.payload.payload['model'] = {name: model[name] for name in census['model_key_order']}
    wrapped = stream.GuardedPayload(wrapped.payload)
    return wrapped, receipt, signature


def _no_source_mappings(source):
    info=Path(source).stat(); device=(os.major(info.st_dev),os.minor(info.st_dev))
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(None,5)
        if len(fields)<5:raise ValueError('Malformed process mapping census')
        mapped_device=tuple(int(part,16) for part in fields[3].split(':'))
        if int(fields[4])==info.st_ino and mapped_device==device:
            raise ValueError('Original ZIP mmap is still live; refusing decode/cache advice')


def advise_source_cache(source, source_stat):
    """Read-only advisory, only after this process holds zero source mmap views."""
    source = Path(source); _no_source_mappings(source)
    if source.is_symlink() or stat(source) != source_stat: raise ValueError('Owned source identity changed before cache advice')
    with source.open('rb') as handle: os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if stat(source) != source_stat: raise ValueError('Cache advice changed source identity')
    return {'operation': 'POSIX_FADV_DONTNEED', 'scope': 'selected completed manager-owned raw source only',
            'bytes_changed': False, 'own_source_mmaps_absent': True, 'source_stat_unchanged': True}


def preflight(run, *, role='best', snapshot=None, max_staging_bytes=24*GIB):
    """Default result is honest gate-closed until the actual first full4f exists."""
    try:
        with _source_lock(run):
            selected = gate(run, role)
            source = Path(selected['selected']['source']); zip_info = _raw_zip_preflight(source)
            from src.utils import research_lossless_delta as v1, research_stream_delta as stream
            import torch
            from torch._subclasses.fake_tensor import FakeTensorMode
            snapshot = Path(snapshot) if snapshot else Path('/workspace/.hf_home/hub/models--Tiiny--SmallThinker-4BA0.6B-Instruct/snapshots')/REVISION
            expected = _anchor_headers(snapshot)
            v1.memory_guard(2*zip_info['data_pickle_bytes']+METADATA_LIMIT)
            with FakeTensorMode(): payload = torch.load(source, weights_only=False, map_location='cpu', mmap=True)
            census = _census(payload, expected); del payload; gc.collect()
            upper = sum(v1._zstd().ZSTD_compressBound(math.prod(shape)*4) for shape in expected.values())
            extra = source.stat().st_size+2*upper+sum(stream.ANCHOR_SHARD_BYTES.values())+8*census['largest_original_tensor_bytes']+census['normal_storage_bytes']+METADATA_LIMIT
            return {'status': 'preflight_gate_open_only', 'archive_created': False, 'payload_values_verified': False,
                'gate': selected, 'census': census, 'zip': zip_info, 'snapshot': str(snapshot),
                'writer_extra_before_source_values_bytes': extra, 'encoded_upper_bytes': upper,
                'source_mmap_file_pages_charged': True, 'memory_headroom_bytes': 32*GIB,
                'max_staging_bytes': max_staging_bytes, 'scope': 'derived-copy measurement only; no publication or GPU parity'}
    except GateClosed as error:
        return {'status': 'gate_closed', 'reason': str(error), 'archive_created': False, 'run': str(Path(run).resolve()),
                'scope': 'actual completed compacted owned full4f Small checkpoint is required'}


def _atomic(path, value, maximum):
    raw = canonical(value)+b'\n'; path = Path(path)
    used = sum(item.stat().st_blocks*512 for item in path.parent.iterdir() if item.is_file() and item.suffix != '.pt')
    if used+math.ceil(len(raw)/4096)*4096 > maximum: raise ValueError('Task metadata staging envelope exceeded')
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as handle: handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally: temporary.unlink(missing_ok=True)


class _BoundedWriter:
    def __init__(self, raw, budget, maximum): self.raw=raw; self.budget=budget; self.maximum=maximum; self.count=0
    def write(self, value):
        if self.count+len(value)>self.maximum: raise ValueError('Derived archive hard staging cap reached')
        self.budget.ensure_space(len(value)); written=self.raw.write(value); self.count+=written; return written
    def flush(self): self.raw.flush()
    def fileno(self): raise io.UnsupportedOperation('Native legacy writes must remain bounded')


class _MemorySamples:
    def __init__(self):
        self.samples=[]; self.peak=0; self.total=0; self.phase='startup'; self.stop=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)
    def sample(self):
        path=Path('/sys/fs/cgroup/memory.current')
        current=int(path.read_text()) if path.is_file() else None
        self.total+=1
        if current is not None:self.peak=max(self.peak,current)
        if len(self.samples)<8192:self.samples.append({'time':time.time(),'phase':self.phase,'cgroup_current_bytes':current,
            'process_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    def _run(self):
        while not self.stop.wait(.5):self.sample()


def run_probe(run, output, *, role='best', snapshot=None, max_staging_bytes=24*GIB,
              reserve_gb=120, advise_source=False, cleanup_verified=False, progress=None):
    """Explicit caller action only; creates derived files, never moves a role."""
    if not 0 < max_staging_bytes <= 24*GIB or not math.isfinite(reserve_gb) or reserve_gb < 120:
        raise ValueError('Require staging<=24GiB and disk reserve>=120GiB')
    from src.utils import research_lossless_delta as v1, research_stream_delta as stream
    from src.utils.research_checkpoint_compat import load_research_payload
    from src.utils.research_storage import DiskBudget
    import torch
    if torch.cuda.is_initialized(): raise ValueError('Probe must stay CPU-only')
    plan = preflight(run, role=role, snapshot=snapshot, max_staging_bytes=max_staging_bytes)
    if plan['status'] != 'preflight_gate_open_only': raise GateClosed(plan.get('reason', 'Full4f gate is closed'))
    output = Path(output).resolve(); run = Path(run).resolve()
    if output == run or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError('Derived probe output must be separate from source run')
    if output.exists(): raise FileExistsError('Use a new task output; existing proof is never overwritten')
    owner = uuid.uuid4().hex; started = time.monotonic(); phase = 'preflight'
    maximum_archive = max_staging_bytes-METADATA_LIMIT-2**20
    if maximum_archive <= stream.METADATA_LIMIT: raise ValueError('Staging cap cannot hold metadata envelopes')
    output.mkdir(parents=True, exist_ok=False); budget = DiskBudget(output, reserve_gb)
    source_names=(str(Path(__file__).relative_to(ROOT)), 'scripts/probe_research_trained_small_delta.py',
        'src/utils/research_lossless_delta.py','src/utils/research_stream_delta.py','src/utils/research_checkpoint_compat.py',
        'src/utils/research_checkpoint_io.py','src/utils/research_checkpoint_bundle.py','src/utils/research_storage.py')
    sources={name:file_sha(ROOT/name) for name in source_names}
    budget.ensure_space(max_staging_bytes)
    _atomic(output/'owner.json', {'owner': owner, 'protocol': OWNER, 'source_sha256': sources}, METADATA_LIMIT)
    archive = output/'derived_full4f_legacy.pt'
    guards = []; cache_advice = None
    monitor=_MemorySamples(); monitor.sample(); monitor.thread.start()
    try:
        budget.ensure_space(max_staging_bytes)
        with _source_lock(run):
            if gate(run, role) != plan['gate']: raise ValueError('Managed source changed after preflight')
            source = Path(plan['gate']['selected']['source']); before = stat(source)
            phase = 'guard_before_source_values'
            monitor.phase=phase
            guards.append({'phase': phase, **v1.memory_guard(plan['writer_extra_before_source_values_bytes'])})
            source_sha = file_sha(source)
            payload = load_research_payload(source, map_location='cpu', mmap=True)
            expected = _anchor_headers(plan['snapshot']); census = _census(payload, expected)
            if census != plan['census'] or payload.get('config') != plan['gate']['config'] or payload.get('step') != plan['gate']['selected']['marker']['step']:
                raise ValueError('Source payload differs from complete full4f metadata')
            compaction = payload.get('checkpoint_compaction', {})
            if (compaction.get('operation') != 'remove_completed_optimizer_only' or
                    compaction.get('source_object') != plan['gate']['selected']['marker'].get('source_checkpoint_object') or
                    not isinstance(compaction.get('source_state_sha256'), str) or len(compaction['source_state_sha256']) != 64):
                raise ValueError('Payload compaction provenance differs from owned source')
            phase = 'complete_payload_hash_and_stream_encode'
            monitor.phase=phase
            wrapped, receipt, expected_signature = _encode_source(payload, expected, census, plan['snapshot'], maximum_archive, progress)
            header = wrapped.header
            # All ordinary tensors are independent small clones in wrapped.
            del payload; gc.collect(); _no_source_mappings(source)
            if stat(source) != before: raise ValueError('Source changed while encoding')
            if advise_source: cache_advice = advise_source_cache(source, before)
            phase = 'guard_before_archive_write'
            monitor.phase=phase
            guards.append({'phase': phase, **v1.memory_guard(header['encoded_bytes']+header['normal_storage_bytes']+2*METADATA_LIMIT)})
            _atomic(output/'expected_payload.json', expected_signature, METADATA_LIMIT)
            _atomic(output/'encoding.json', {'owner': owner, 'header': header, 'receipt': receipt, 'source_sha256': source_sha,
                'source_stat': before, 'plan': plan, 'cache_advice': cache_advice}, METADATA_LIMIT)
            phase = 'native_guarded_legacy_write'
            monitor.phase=phase
            with archive.open('xb') as raw:
                stream.save_guarded_legacy(wrapped, _BoundedWriter(raw, budget, maximum_archive))
                raw.flush(); os.fsync(raw.fileno())
            archive_sha = file_sha(archive); archive_bytes = archive.stat().st_size
            del wrapped; gc.collect(); _no_source_mappings(source)
            phase = 'guard_before_unchanged_cpu_load'
            monitor.phase=phase
            # Source/target file cache are charged by actual current cgroup,
            # never credited as free; add worst reader heaps +32GiB headroom.
            guards.append({'phase': phase, **v1.memory_guard(stream.memory_requirement(header))})
            phase = 'unchanged_cpu_loader_full_payload_verify'
            monitor.phase=phase
            restored = load_research_payload(archive, map_location='cpu', mmap=True)
            actual_signature = payload_signature(restored)
            if actual_signature != expected_signature: raise ValueError('Complete payload/storage/metadata raw-bit signature changed')
            del restored; gc.collect()
            if gate(run, role) != plan['gate'] or stat(source) != before or file_sha(source) != source_sha:
                raise ValueError('Original managed source/roles changed during derived proof')
            if any(file_sha(ROOT/name)!=sha for name,sha in sources.items()):raise ValueError('Probe/decoder/unchanged loader source changed during proof')
            v1.codec_sha256();stream.source_sha256()
            report = {'status': 'verified_actual_trained_derived_copy_only', 'protocol': OWNER, 'owner': owner,
                'run': str(run), 'role': role, 'source_sha256': source_sha, 'source_stat': before,
                'source_roles_before_after': plan['gate']['roles'], 'source_payload_sha256': expected_signature['sha256'],
                'restored_payload_sha256': actual_signature['sha256'], 'all_payload_fields_and_backing_bits_exact': True,
                'scheduler_rng_config_compaction_metadata_preserved': True, 'normal_storage_views_aliases_preserved': True,
                'encoded_aliases_refused': True, 'archive_path': str(archive), 'archive_bytes': archive_bytes,
                'archive_sha256': archive_sha, 'archive_to_raw_source_ratio': archive_bytes/before['st_size'],
                'archive_deleted': False, 'source_modified': False, 'manager_roles_modified': False,
                'optimizer_removed_by_probe': False, 'gpu_reconstruction_validated': False,
                'cuda_initialized': torch.cuda.is_initialized(), 'seconds': time.monotonic()-started,
                'process_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'shared_cgroup_observed_peak_bytes': monitor.peak, 'shared_peak_is_not_isolated_process_or_instantaneous_kernel_peak': True,
                'memory_guards': guards, 'cache_advice': cache_advice, 'code_sha256': sources,
                'runtime': {'torch':torch.__version__,'numpy':__import__('numpy').__version__},
                'normal_storage_bytes': census['normal_storage_bytes'], 'decoded_model_bytes': census['original_fp32_bytes'],
                'scope': 'Actual trained compression/I/O measurement only; no publication, migration or scientific metric claim'}
            _atomic(output/'verified.json', report, METADATA_LIMIT)
            if cleanup_verified:
                persisted = _json(output/'verified.json')
                if (_json(output/'owner.json')['owner'] != owner or persisted.get('owner') != owner or
                        persisted.get('archive_sha256') != file_sha(archive) or archive.is_symlink() or archive.parent != output):
                    raise ValueError('Task-owned archive cleanup proof differs')
                archive.unlink(); report['archive_deleted'] = True
                _atomic(output/'verified.json', report, METADATA_LIMIT)
            return report
    except BaseException as error:
        _atomic(output/'failure.json', {'owner': owner, 'phase': phase, 'error_type': type(error).__name__, 'error': str(error),
            'archive_preserved_if_present': archive.exists(), 'source_replacement_attempted': False}, METADATA_LIMIT)
        raise
    finally:
        monitor.stop.set(); monitor.thread.join(timeout=2); monitor.sample()
        _atomic(output/'memory_samples.json', {'owner':owner,'samples':monitor.samples,'total_samples':monitor.total,
            'dropped_samples':monitor.total-len(monitor.samples),'shared_observed_peak_bytes':monitor.peak}, METADATA_LIMIT)
