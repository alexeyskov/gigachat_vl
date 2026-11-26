"""Version2 signal-safe immutable caption preparation; v1 stays frozen.

This module uses only the standard library and Pillow. It does not load models,
download data, or claim that the existing pilot validation pool is a benchmark.
"""
from __future__ import annotations

import ctypes
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import threading
import uuid

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
QUESTION = 'Describe this image in detail.'
NAMESPACE = 'research-caption-pilot-image-rank-v1'
PUBLIC_FIELDS = frozenset(('record_id', 'image_paths', 'image_hashes', 'question'))
FILES = frozenset(('generation.jsonl', 'references.jsonl', 'protocol.json', 'audit.json'))
HEX = re.compile(r'^[0-9a-f]{64}$')
REVISION = re.compile(r'^[0-9a-f]{40}$')
MEMORY_RESERVE_BYTES = 2*2**30
PIXEL_STRIP_BYTES = 8*2**20
V2_PROFILE = 'caption_pilot_en_v2_signal_safe'


@contextmanager
def _defer_submit_stop_signals():
    """Finish ThreadPoolExecutor's started-thread registration before stop.

    A main-thread pthread mask alone is insufficient: a different unmasked
    thread can receive the signal and schedule Python's main-thread callback.
    The temporary nonraising handlers cover that case too. The original mask
    and both handlers are restored before invoking the original stop handler.
    """
    if threading.current_thread() is not threading.main_thread():
        # Python signal callbacks execute in the main thread, so they cannot
        # interrupt another thread's submit bookkeeping.
        yield
        return
    signals=(signal.SIGINT,signal.SIGTERM)
    previous_mask=signal.pthread_sigmask(signal.SIG_BLOCK,set())
    previous_handlers={signum:signal.getsignal(signum) for signum in signals}
    deferred=[]
    def capture(signum,frame):
        if not any(item[0]==signum for item in deferred):deferred.append((signum,frame))
    try:
        for signum in signals:signal.signal(signum,capture)
        signal.pthread_sigmask(signal.SIG_BLOCK,set(signals))
        yield
    finally:
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask)
        finally:
            try:signal.signal(signal.SIGINT,previous_handlers[signal.SIGINT])
            finally:signal.signal(signal.SIGTERM,previous_handlers[signal.SIGTERM])
        for signum,frame in deferred:
            handler=previous_handlers[signum]
            if callable(handler):handler(signum,frame)
            # Default stop is process-directed: the original main mask may
            # block this signal while a pre-existing thread remains eligible.
            elif handler==signal.SIG_DFL:os.kill(os.getpid(),signum)
            # SIG_IGN retains its original semantics.


def _restore_audit_worker_mask(mask):
    # A thread started inside the protected submit inherits its temporary mask.
    # Restore the caller's original policy before this dedicated worker decodes.
    signal.pthread_sigmask(signal.SIG_SETMASK,mask)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''): result.update(block)
    return result.hexdigest()


def _stat(path):
    value = Path(path).stat()
    return {key: getattr(value, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def _reject_constant(value):
    raise ValueError('Nonfinite JSON constant: '+value)


def _manifest_rows(path, receipts):
    """Hash the exact streamed bytes and recheck stat and SHA after consumption."""
    path = Path(path).resolve(strict=True); before = _stat(path); digest = hashlib.sha256(); count = 0
    with path.open('rb') as stream:
        while True:
            raw = stream.readline(16*2**20+1)
            if not raw: break
            if len(raw) > 16*2**20: raise ValueError('Manifest row exceeds 16 MiB')
            digest.update(raw)
            if not raw.strip(): raise ValueError('Blank manifest row: '+str(path))
            row = json.loads(raw, parse_constant=_reject_constant)
            if not isinstance(row, dict): raise ValueError('Manifest row must be an object')
            count += 1
            yield row
    if _stat(path) != before or file_sha(path) != digest.hexdigest() or _stat(path) != before:
        raise ValueError('Source manifest changed during audit: '+str(path))
    receipts.append({'path': str(path), 'sha256': digest.hexdigest(), 'stat': before, 'records': count})


def _verify_sources(receipts):
    for receipt in receipts:
        path = receipt['path']
        if (_stat(path) != receipt['stat'] or file_sha(path) != receipt['sha256'] or
                _stat(path) != receipt['stat']):
            raise ValueError('Source manifest changed during audit: '+path)


def _memory_available_bytes():
    """Current host/cgroup headroom, without importing Torch or reading env."""
    values = {key: int(value.split()[0])*1024 for key, value in
              (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    available = values['MemAvailable']
    maximum = Path('/sys/fs/cgroup/memory.max')
    current = Path('/sys/fs/cgroup/memory.current')
    if maximum.is_file() and current.is_file():
        limit = maximum.read_text().strip()
        if limit != 'max': available = min(available, max(0, int(limit)-int(current.read_text())))
    return available


class _DecodeMemory:
    """Conservative admission estimate, not an OS memory allocation quota."""
    def __init__(self, reserve_bytes):
        self.reserve = reserve_bytes
        self.capacity = max(0, _memory_available_bytes()-reserve_bytes)
        self.reserved = 0
        self.condition = threading.Condition()

    @contextmanager
    def acquire(self, required, stopped):
        with self.condition:
            while True:
                if stopped.is_set(): raise CancelledError('Pixel audit cancelled')
                available = _memory_available_bytes()
                if required > self.capacity or (self.reserved == 0 and available < required+self.reserve):
                    raise MemoryError('Preserve caption pixel-audit host/cgroup memory headroom')
                if self.reserved+required <= self.capacity and available >= required+self.reserve:
                    self.reserved += required
                    break
                self.condition.wait(timeout=.1)
        try:
            yield
        finally:
            with self.condition:
                self.reserved -= required
                self.condition.notify_all()


def _decode_pixels(path, before, stopped, memory):
    """One worker: canonical RGB bytes, bounded strips, exact old hash function."""
    if stopped.is_set(): raise CancelledError('Pixel audit cancelled')
    if _stat(path) != before: raise ValueError('Image changed before decoding: '+str(path))
    try:
        with Image.open(path) as original:
            # Allow original/EXIF/RGB decoder buffers plus strip copies. This
            # estimate gates concurrent admissions; Pillow/OS peaks still vary.
            estimate = original.width*original.height*24+2*PIXEL_STRIP_BYTES+2**20
            with memory.acquire(estimate, stopped):
                if stopped.is_set(): raise CancelledError('Pixel audit cancelled')
                with ImageOps.exif_transpose(original) as oriented:
                    with oriented.convert('RGB') as rgb:
                        digest = hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode())
                        rows_per_strip = max(1, PIXEL_STRIP_BYTES//max(1, rgb.width*3))
                        for top in range(0, rgb.height, rows_per_strip):
                            if stopped.is_set(): raise CancelledError('Pixel audit cancelled')
                            with rgb.crop((0, top, rgb.width, min(top+rows_per_strip, rgb.height))) as strip:
                                digest.update(strip.tobytes())
                        actual = digest.hexdigest(); size = list(rgb.size)
    except (CancelledError, MemoryError):
        raise
    except Exception as error:
        raise ValueError('Cannot decode canonical RGB pixels: '+str(path)) from error
    if _stat(path) != before: raise ValueError('Image changed while decoding: '+str(path))
    return {'sha256': actual, 'size': size, 'stat': before}


class _Pixels:
    def __init__(self, project_root, progress=None, *, workers=4, memory_reserve_bytes=MEMORY_RESERVE_BYTES):
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError('Pixel audit workers must be an integer from1 to4')
        if type(memory_reserve_bytes) is not int or memory_reserve_bytes < 0:
            raise ValueError('Pixel-audit memory reserve must be nonnegative integer bytes')
        self.root = Path(project_root).resolve(); self.files = {}; self.progress = progress
        self.workers = workers; self.max_pending = 2*workers
        self.expected = {}; self.pending = {}; self.stopped = threading.Event(); self.closed = False
        self.shutdown_diagnostics = None
        self.memory = _DecodeMemory(memory_reserve_bytes)
        initial_mask=signal.pthread_sigmask(signal.SIG_BLOCK,set())
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='caption-pixels',
            initializer=_restore_audit_worker_mask,initargs=(initial_mask,))

    def __enter__(self):
        try:
            if self.progress:
                self.progress({'event': 'pixel_audit_config', 'workers': self.workers,
                               'max_pending': self.max_pending, 'memory_reserve_bytes': self.memory.reserve})
        except BaseException:
            self.close(cancel=True)
            raise
        return self

    def __exit__(self, error_type, _error, _traceback):
        self.close(cancel=error_type is not None)

    def close(self, *, cancel=False):
        if self.closed: return
        if cancel:
            self.stopped.set()
            for future in self.pending.values(): future.cancel()
            with self.memory.condition: self.memory.condition.notify_all()
        # Joining is intentional: when an exception returns to the caller no
        # audit thread or decoded Pillow image remains alive in the background.
        self.executor.shutdown(wait=True, cancel_futures=cancel)
        self.closed = True
        self.shutdown_diagnostics = {'event':'pixel_audit_workers_closed','worker_threads':len(self.executor._threads),
            'alive_worker_threads':sum(thread.is_alive() for thread in self.executor._threads),
            'pending_futures':len(self.pending),'all_futures_done':all(future.done() for future in self.pending.values()),
            'cancel_requested':cancel}
        if self.shutdown_diagnostics['alive_worker_threads'] or not self.shutdown_diagnostics['all_futures_done']:
            raise RuntimeError('Caption worker shutdown did not join every registered worker/future')
        if self.progress:self.progress(self.shutdown_diagnostics)

    def _collect_first(self):
        key = next(iter(self.pending)); future = self.pending[key]
        receipt = future.result()
        if receipt['sha256'] != self.expected[key]:
            raise ValueError('Canonical RGB image hash mismatch: '+key)
        self.files[key] = receipt
        del self.pending[key]
        if self.progress and len(self.files) % 1000 == 0:
            self.progress({'event': 'pixel_audit', 'distinct_files': len(self.files)})

    def check(self, value, expected):
        if self.closed: raise RuntimeError('Pixel audit is already closed')
        if not isinstance(value, str) or not value or not isinstance(expected, str) or not HEX.fullmatch(expected):
            raise ValueError('Image path and canonical SHA256 are required')
        path = Path(value); path = (path if path.is_absolute() else self.root/path).resolve(strict=True)
        key = str(path)
        if key in self.expected:
            if self.expected[key] != expected: raise ValueError('Canonical RGB image hash mismatch: '+key)
        else:
            # Submission/verification happens only in the calling thread. A
            # repeated resolved path gets exactly one decode, even while queued.
            while len(self.pending) >= self.max_pending: self._collect_first()
            before = _stat(path)
            with _defer_submit_stop_signals():
                self.expected[key] = expected
                self.pending[key] = self.executor.submit(_decode_pixels, path, before, self.stopped, self.memory)
        return key

    def finish(self):
        try:
            while self.pending: self._collect_first()
            for path in sorted(self.files):
                if _stat(path) != self.files[path]['stat']: raise ValueError('Image changed during pool audit: '+path)
        except BaseException:
            self.close(cancel=True)
            raise
        self.close()
        if self.progress: self.progress({'event': 'pixel_audit_complete', 'distinct_files': len(self.files)})
        return {'distinct_files': len(self.files), 'files_sha256': hashlib.sha256(canonical(self.files)).hexdigest(),
                'canonical_hash': 'SHA256(ASCII RGB:<width>:<height>: + EXIF-transposed RGB bytes)'}


def _images(row, pixels, *, single):
    paths = row.get('image_paths'); hashes = row.get('image_hashes')
    if not isinstance(paths, list) or not isinstance(hashes, list) or len(paths) != len(hashes):
        raise ValueError('Aligned image_paths/image_hashes lists are required')
    if single and len(paths) != 1: raise ValueError('Caption pool must have exactly one image per record')
    return [pixels.check(path, sha) for path, sha in zip(paths, hashes)], hashes


def _identity(row, expected_partition):
    if not isinstance(row.get('record_id'), str) or not row['record_id']:
        raise ValueError('Nonempty source record_id required')
    if row.get('partition') != expected_partition:
        raise ValueError('Unexpected source partition; caption selection uses only internal validation and exclusions only train')


def build_caption_bundle_data(pool, exclusions, *, seed=42, cap=1024, namespace=NAMESPACE,
                              project_root=ROOT, progress=None, workers=4,
                              memory_reserve_bytes=MEMORY_RESERVE_BYTES):
    """Audit the ENTIRE source pool and each exclusion, then select image groups."""
    if type(seed) is not int or type(cap) is not int or cap < 2 or not isinstance(namespace, str) or not namespace:
        raise ValueError('Integer seed, image cap >=2 and nonempty namespace required')
    exclusions = [Path(path).resolve(strict=True) for path in exclusions]
    if not exclusions or len(set(exclusions)) != len(exclusions):
        raise ValueError('Declare at least one unique train exclusion manifest')
    pool = Path(pool).resolve(strict=True)
    if pool in exclusions: raise ValueError('Caption pool cannot be a train exclusion')
    with _Pixels(project_root, progress, workers=workers, memory_reserve_bytes=memory_reserve_bytes) as pixels:
        return _audited_bundle(pool, exclusions, pixels, seed=seed, cap=cap, namespace=namespace)


def _audited_bundle(pool, exclusions, pixels, *, seed, cap, namespace):
    receipts = []; groups = {}; seen = set()
    for row in _manifest_rows(pool, receipts):
        _identity(row, 'validation')
        if row.get('task_group') != 'captioning' or row.get('language') != 'en':
            raise ValueError('Caption pool must be English captioning validation')
        if row['record_id'] in seen: raise ValueError('Duplicate caption source record_id')
        seen.add(row['record_id']); paths, hashes = _images(row, pixels, single=True)
        answer = row.get('answer'); source = row.get('source')
        if not isinstance(answer, str) or not answer.strip(): raise ValueError('An actual nonempty caption reference is required')
        if (not isinstance(source, dict) or not isinstance(source.get('repo'), str) or not source['repo'] or
                not isinstance(source.get('revision'), str) or not REVISION.fullmatch(source['revision'])):
            raise ValueError('Every reference must retain its pinned source repo/revision')
        group = groups.setdefault(hashes[0], {'paths': set(), 'records': []})
        group['paths'].add(paths[0]); group['records'].append({'source_record_id': row['record_id'],
            'reference': answer, 'source': source})
    if len(groups) < 2: raise ValueError('Caption evaluation needs at least two distinct image groups')
    exclusion_receipts = []
    for path in exclusions:
        rows = 0; hashes_seen = set(); text_rows = 0
        for row in _manifest_rows(path, receipts):
            _identity(row, 'train'); _, hashes = _images(row, pixels, single=False); rows += 1
            if not hashes: text_rows += 1
            if set(hashes) & groups.keys(): raise ValueError('Caption validation/train image overlap: '+str(path))
            hashes_seen.update(hashes)
        exclusion_receipts.append({'path': str(path), 'records': rows, 'text_only_records': text_rows,
            'distinct_images': len(hashes_seen), 'overlap_images': 0})
    pixel_receipt = pixels.finish(); _verify_sources(receipts)
    audit = {'schema_version': 1, 'status': 'passed', 'pool_records': len(seen), 'pool_images': len(groups),
        'source_manifests': receipts, 'exclusions': exclusion_receipts, 'pixels': pixel_receipt,
        'audited_unselected_pool': True, 'all_exclusion_pixels_verified': True, 'overlap_policy': 'refuse any overlap; never silently drop'}
    def rank(image_hash):
        return hashlib.sha256(canonical({'namespace': namespace, 'seed': seed, 'image_hash': image_hash})).hexdigest(), image_hash
    selected = sorted(groups, key=rank)[:cap]; public = []; references = []
    for image_hash in selected:
        group = groups[image_hash]; records = sorted(group['records'], key=lambda row: row['source_record_id'])
        record_id = 'caption-'+image_hash
        public.append({'record_id': record_id, 'image_paths': [min(group['paths'])], 'image_hashes': [image_hash], 'question': QUESTION})
        references.append({'record_id': record_id, 'image_hash': image_hash, 'references': [row['reference'] for row in records],
            'reference_records': records})
    options = {'question': QUESTION, 'rank_namespace': namespace, 'seed': seed, 'cap': cap,
        'unit': 'unique canonical RGB image', 'all_actual_references_per_selected_image': True,
        'dataset_scope': 'internal pilot captioning validation; not the primary benchmark',
        'public_fields': sorted(PUBLIC_FIELDS), 'source_manifests': receipts}
    return {'generation': public, 'references': references, 'audit': audit, 'options': options}


def _write(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _publish_directory(source, target):
    # Linux atomic no-replace publication; never replace an existing partial bundle.
    rename = getattr(ctypes.CDLL(None, use_errno=True), 'renameat2', None)
    if rename is None: raise RuntimeError('Atomic renameat2(RENAME_NOREPLACE) required')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        value = ctypes.get_errno()
        if value == errno.EEXIST: raise FileExistsError('Caption output already exists: '+str(target))
        raise OSError(value, os.strerror(value))


def prepare_caption_bundle(pool, exclusions, output, *, seed=42, cap=1024, namespace=NAMESPACE,
                           project_root=ROOT, resume=False, reserve_bytes=120*2**30, progress=None,
                           preparation_source_paths=(), workers=4, memory_reserve_bytes=MEMORY_RESERVE_BYTES,
                           preparation_profile=V2_PROFILE):
    if preparation_profile!=V2_PROFILE:raise ValueError('Only the explicit signal-safe v2 preparation profile is supported')
    output = Path(output)
    if output.is_symlink(): raise ValueError('Caption output cannot be a symlink')
    output = output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not resume: raise FileExistsError('Existing bundle requires explicit --resume')
    if output.exists() and (not output.is_dir() or set(path.name for path in output.iterdir()) != FILES):
        raise ValueError('Incomplete or unexpected caption bundle; refusing resume')
    if output.exists():
        existing=json.loads((output/'protocol.json').read_bytes())
        if existing.get('preparation_source_sha256',{}).get(str(Path(__file__).resolve()))!=file_sha(__file__):
            raise ValueError('Existing caption preparation source differs; use its archived source or a new explicit v2 output')
    data = build_caption_bundle_data(pool, exclusions, seed=seed, cap=cap, namespace=namespace,
        project_root=project_root, progress=progress, workers=workers, memory_reserve_bytes=memory_reserve_bytes)
    encoded = {'generation.jsonl': b''.join(canonical(row)+b'\n' for row in data['generation']),
               'references.jsonl': b''.join(canonical(row)+b'\n' for row in data['references']),
               'audit.json': canonical(data['audit'])+b'\n'}
    source_paths = sorted({str(Path(__file__).resolve()), *(str(Path(path).resolve(strict=True)) for path in preparation_source_paths)})
    protocol = {'schema_version': 1, 'protocol': 'research_caption_pilot_eval_v2', 'status': 'complete',
        'records': len(data['generation']), 'reference_records': sum(len(row['references']) for row in data['references']),
        'options': data['options'], 'files_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in encoded.items()},
        'preparation_source_sha256': {path: file_sha(path) for path in source_paths},
        'generation_references_separate': True, 'checkpoint_binding': 'none: generation has not run'}
    protocol.update(preparation_profile=V2_PROFILE,
        stop_signal_contract='defer SIGINT/SIGTERM through executor started-thread registration and caller pending-future bookkeeping; restore masks/handlers before replay')
    encoded['protocol.json'] = canonical(protocol)+b'\n'
    _verify_sources(data['options']['source_manifests'])
    if output.exists():
        for name, raw in encoded.items():
            if (output/name).is_symlink() or (output/name).read_bytes() != raw:
                raise ValueError('Existing caption bundle differs from same-protocol source audit: '+name)
        return {'status': 'reused', 'output': str(output), 'records': protocol['records'], 'protocol_sha256': file_sha(output/'protocol.json')}
    required = sum(len(raw) for raw in encoded.values())+2**20
    if shutil.disk_usage(output.parent).free < reserve_bytes+required:
        raise OSError('Preserve caption-preparation disk reserve plus staging bytes')
    staging = output.parent/('.'+output.name+'.staging-'+uuid.uuid4().hex)
    staging.mkdir()
    try:
        for name, raw in encoded.items(): _write(staging/name, raw)
        _fsync_dir(staging); _publish_directory(staging, output); _fsync_dir(output.parent)
    finally:
        # Only this invocation's staging directory is eligible for cleanup.
        if staging.exists(): shutil.rmtree(staging)
    return {'status': 'prepared', 'output': str(output), 'records': protocol['records'], 'protocol_sha256': file_sha(output/'protocol.json')}
