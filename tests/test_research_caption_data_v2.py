"""Actual Pillow fixtures and immutable publication, with no model imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import CancelledError

from PIL import Image, ImageOps
import pytest

from src.eval import research_caption_data_v2 as data


def image(path, color, *, exif=False):
    raw = Image.new('RGB', (7, 4), color)
    if exif:
        values = Image.Exif(); values[274] = 6
        raw.save(path, exif=values)
    else: raw.save(path)
    with Image.open(path) as saved:
        rgb = ImageOps.exif_transpose(saved).convert('RGB')
        sha = hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode()+rgb.tobytes()).hexdigest()
    return str(path), sha


def row(record_id, pixels, *, partition='validation', answer=None):
    return {'record_id': record_id, 'image_paths': [pixels[0]] if pixels else [],
        'image_hashes': [pixels[1]] if pixels else [], 'partition': partition,
        'task_group': 'captioning' if partition == 'validation' else 'arbitrary_train_task', 'language': 'en',
        'answer': answer or 'REFERENCE_CANARY_'+record_id, 'question': 'Do not use the source question or gold.',
        'messages': [{'role': 'assistant', 'content': 'GOLD_MESSAGE_CANARY'}],
        'source': {'repo': 'fixture/captions', 'revision': 'a'*40, 'row_id': record_id, 'split': 'train'}}


def manifest(path, rows):
    path.write_bytes(b''.join(data.canonical(value)+b'\n' for value in rows))
    return path


@pytest.fixture
def corpus(tmp_path):
    pixels = [image(tmp_path/f'image{i}.png', (i*31, 20, 40)) for i in range(5)]
    pool = manifest(tmp_path/'pool.jsonl', [row(str(i), value) for i, value in enumerate(pixels[:4])])
    train = manifest(tmp_path/'train.jsonl', [row('train', pixels[4], partition='train'), row('text', None, partition='train')])
    return pool, train, pixels


def test_group_all_actual_references_and_remove_gold(corpus, tmp_path):
    pool, train, pixels = corpus
    with pool.open() as stream: rows = [json.loads(line) for line in stream]
    duplicate_path, duplicate_hash = image(tmp_path/'same_pixels.png', (0, 20, 40))
    assert duplicate_hash == pixels[0][1]
    rows += [row('second_reference', (duplicate_path, duplicate_hash), answer='ANOTHER_ACTUAL_REFERENCE')]
    manifest(pool, rows)
    result = data.build_caption_bundle_data(pool, [train])
    assert len(result['generation']) == 4
    refs = next(value for value in result['references'] if value['image_hash'] == pixels[0][1])
    assert refs['references'] == ['REFERENCE_CANARY_0', 'ANOTHER_ACTUAL_REFERENCE']
    assert [value['source_record_id'] for value in refs['reference_records']] == ['0', 'second_reference']
    assert all(value['source']['revision'] == 'a'*40 for value in refs['reference_records'])
    for record in result['generation']:
        assert set(record) == data.PUBLIC_FIELDS
        assert record['question'] == 'Describe this image in detail.'
    public = data.canonical(result['generation'])
    assert b'REFERENCE' not in public and b'GOLD' not in public and b'source question' not in public
    assert result['audit']['exclusions'][0]['text_only_records'] == 1


def test_deterministic_rank_is_image_based_and_input_order_independent(corpus):
    pool, train, _ = corpus
    first = data.build_caption_bundle_data(pool, [train], cap=2)
    rows = [json.loads(line) for line in pool.read_text().splitlines()]
    manifest(pool, list(reversed(rows)))
    second = data.build_caption_bundle_data(pool, [train], cap=2)
    assert first['generation'] == second['generation']
    assert first['references'] == second['references']
    full = data.build_caption_bundle_data(pool, [train], cap=1024)
    assert first['generation'] == full['generation'][:2]


def test_corrupt_unselected_pool_image_is_not_silently_dropped(corpus):
    pool, train, pixels = corpus
    selected = data.build_caption_bundle_data(pool, [train], cap=2)['generation']
    selected_hashes = {value['image_hashes'][0] for value in selected}
    unselected = next(value for value in pixels[:4] if value[1] not in selected_hashes)
    Path(unselected[0]).write_bytes(b'corrupted image')
    with pytest.raises(ValueError, match='Cannot decode'): data.build_caption_bundle_data(pool, [train], cap=2)


def test_real_pixel_mismatch_in_any_train_exclusion_rejected(corpus):
    pool, train, pixels = corpus
    image(Path(pixels[4][0]), (254, 0, 0))
    with pytest.raises(ValueError, match='hash mismatch'): data.build_caption_bundle_data(pool, [train])


def test_train_overlap_uses_pixels_even_with_different_paths(corpus, tmp_path):
    pool, train, _ = corpus
    duplicate = image(tmp_path/'duplicate.png', (0, 20, 40))
    manifest(train, [row('overlap', duplicate, partition='train')])
    with pytest.raises(ValueError, match='overlap'): data.build_caption_bundle_data(pool, [train], cap=2)


def test_all_declared_exclusion_manifests_audited(corpus, tmp_path):
    pool, train, pixels = corpus
    second = manifest(tmp_path/'other_train.jsonl', [row('overlap', pixels[0], partition='train')])
    with pytest.raises(ValueError, match='overlap'): data.build_caption_bundle_data(pool, [train, second])


def test_exif_transposition_is_canonical(tmp_path):
    oriented = image(tmp_path/'orientation.jpg', (0, 42, 80), exif=True)
    other = image(tmp_path/'other.png', (0, 40, 80))
    pool = manifest(tmp_path/'pool.jsonl', [row('oriented', oriented), row('other', other)])
    train = manifest(tmp_path/'train.jsonl', [row('text', None, partition='train')])
    result = data.build_caption_bundle_data(pool, [train])
    with Image.open(oriented[0]) as raw:
        untransposed = hashlib.sha256(f'RGB:{raw.width}:{raw.height}:'.encode()+raw.convert('RGB').tobytes()).hexdigest()
    assert oriented[1] != untransposed
    assert oriented[1] in {value['image_hashes'][0] for value in result['generation']}


def test_manifest_change_while_streaming_rejected(corpus):
    pool, _, _ = corpus; receipts = []
    iterator = data._manifest_rows(pool, receipts); next(iterator)
    pool.write_bytes(pool.read_bytes()+b'\n')
    with pytest.raises(ValueError): list(iterator)
    assert receipts == []


def test_late_manifest_change_after_first_read_rejected(corpus, monkeypatch):
    pool, train, _ = corpus; original = data._Pixels.finish
    def finish(self):
        result = original(self)
        pool.write_bytes(pool.read_bytes().replace(b'REFERENCE_CANARY_0', b'CHANGED_REFERENCE_0'))
        return result
    monkeypatch.setattr(data._Pixels, 'finish', finish)
    with pytest.raises(ValueError, match='manifest changed'): data.build_caption_bundle_data(pool, [train])


def test_immutable_bundle_explicit_resume_and_full_checks(corpus, tmp_path):
    pool, train, _ = corpus; output = tmp_path/'bundle'
    first = data.prepare_caption_bundle(pool, [train], output, reserve_bytes=0)
    assert first['status'] == 'prepared' and first['records'] == 4
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    with pytest.raises(FileExistsError): data.prepare_caption_bundle(pool, [train], output, reserve_bytes=0)
    resumed = data.prepare_caption_bundle(pool, [train], output, resume=True, reserve_bytes=0)
    assert resumed['status'] == 'reused'
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()} == before
    with pytest.raises(ValueError, match='differs'): data.prepare_caption_bundle(pool, [train], output, cap=2, resume=True, reserve_bytes=0)
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['options']['dataset_scope'].startswith('internal pilot')
    assert protocol['checkpoint_binding'].startswith('none:')
    for name, sha in protocol['files_sha256'].items(): assert data.file_sha(output/name) == sha


def test_changed_source_and_output_refuse_resume(corpus, tmp_path):
    pool, train, _ = corpus; output = tmp_path/'bundle'
    data.prepare_caption_bundle(pool, [train], output, reserve_bytes=0)
    (output/'generation.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='differs'): data.prepare_caption_bundle(pool, [train], output, resume=True, reserve_bytes=0)


def test_partial_bundle_not_resumed_or_cleaned(corpus, tmp_path):
    pool, train, _ = corpus; output = tmp_path/'bundle'; output.mkdir()
    (output/'generation.jsonl').write_text('partial')
    with pytest.raises(ValueError, match='Incomplete'): data.prepare_caption_bundle(pool, [train], output, resume=True, reserve_bytes=0)
    assert (output/'generation.jsonl').read_text() == 'partial'


def test_publish_failure_cleans_only_own_staging(corpus, tmp_path, monkeypatch):
    pool, train, _ = corpus; output = tmp_path/'bundle'
    foreign = tmp_path/'.bundle.staging-foreign'; foreign.mkdir(); (foreign/'keep').write_text('keep')
    def fail(*args): raise RuntimeError('simulated before publish')
    monkeypatch.setattr(data, '_publish_directory', fail)
    with pytest.raises(RuntimeError, match='simulated'): data.prepare_caption_bundle(pool, [train], output, reserve_bytes=0)
    assert not output.exists()
    assert list(tmp_path.glob('.bundle.staging-*')) == [foreign]


@pytest.mark.parametrize('mutation', ['partition', 'language', 'revision', 'duplicate'])
def test_invalid_reference_provenance_not_accepted(corpus, mutation):
    pool, train, _ = corpus; rows = [json.loads(line) for line in pool.read_text().splitlines()]
    if mutation == 'partition': rows[0]['partition'] = 'test'
    if mutation == 'language': rows[0]['language'] = 'ru'
    if mutation == 'revision': rows[0]['source']['revision'] = 'main'
    if mutation == 'duplicate': rows.append(rows[0])
    manifest(pool, rows)
    with pytest.raises(ValueError): data.build_caption_bundle_data(pool, [train])


def test_source_code_binding_and_disk_reserve(corpus, tmp_path):
    pool, train, _ = corpus; output = tmp_path/'bundle'; helper = tmp_path/'helper.py'; helper.write_text('# version1')
    with pytest.raises(OSError, match='disk reserve'):
        data.prepare_caption_bundle(pool, [train], output, reserve_bytes=2**62)
    data.prepare_caption_bundle(pool, [train], output, reserve_bytes=0, preparation_source_paths=[helper])
    helper.write_text('# version2')
    with pytest.raises(ValueError, match='differs'):
        data.prepare_caption_bundle(pool, [train], output, reserve_bytes=0, preparation_source_paths=[helper], resume=True)


def test_import_does_not_load_torch():
    result = subprocess.run([sys.executable, '-c', "import sys; import src.eval.research_caption_data_v2; assert 'torch' not in sys.modules"],
        cwd=data.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def audit_threads():
    return [thread for thread in threading.enumerate() if thread.name.startswith('caption-pixels')]


def test_workers1_and4_make_byte_identical_bundles_despite_completion_order(corpus, tmp_path, monkeypatch):
    pool, train, _ = corpus
    original = data._decode_pixels
    def decode(path, *args):
        time.sleep((4-int(Path(path).stem[-1]))*.002)
        return original(path, *args)
    monkeypatch.setattr(data, '_decode_pixels', decode)
    first = data.build_caption_bundle_data(pool, [train], workers=1)
    second = data.build_caption_bundle_data(pool, [train], workers=4)
    assert data.canonical(first) == data.canonical(second)
    one, four = tmp_path/'serial_bundle', tmp_path/'parallel_bundle'
    data.prepare_caption_bundle(pool, [train], one, workers=1, reserve_bytes=0)
    data.prepare_caption_bundle(pool, [train], four, workers=4, reserve_bytes=0)
    assert {p.name: p.read_bytes() for p in one.iterdir()} == {p.name: p.read_bytes() for p in four.iterdir()}
    assert audit_threads() == []


def test_four_actual_workers_and_duplicate_pending_paths_decode_once(corpus, monkeypatch):
    pool, train, pixels = corpus
    pool_rows = [json.loads(line) for line in pool.read_text().splitlines()]
    pool_rows += [row('duplicate-reference', pixels[0])]
    manifest(pool, pool_rows)
    manifest(train, [row('train', pixels[4], partition='train'), row('same-train-path', pixels[4], partition='train')])
    original = data._decode_pixels; lock = threading.Lock(); barrier = threading.Barrier(4, timeout=3)
    state = {'active': 0, 'peak': 0, 'calls': {}}
    def decode(path, *args):
        with lock:
            state['active'] += 1; state['peak'] = max(state['peak'], state['active'])
            state['calls'][str(path)] = state['calls'].get(str(path), 0)+1
        try:
            if str(path) in {value[0] for value in pixels[:4]}: barrier.wait()
            return original(path, *args)
        finally:
            with lock: state['active'] -= 1
    monkeypatch.setattr(data, '_decode_pixels', decode)
    result = data.build_caption_bundle_data(pool, [train], workers=4)
    assert state['peak'] == 4 and state['active'] == 0
    assert state['calls'] == {str(Path(value[0]).resolve()): 1 for value in pixels}
    assert result['audit']['pixels']['distinct_files'] == 5 and audit_threads() == []


def many_pixels(tmp_path, count=12):
    return [image(tmp_path/f'queued-{index}.png', (index*11 % 256, 32, 54)) for index in range(count)]


def test_pending_futures_bounded_and_all_images_verified(tmp_path, monkeypatch):
    pixels = many_pixels(tmp_path, 18)
    pool = manifest(tmp_path/'pool.jsonl', [row(str(i), p) for i, p in enumerate(pixels[:-1])])
    train = manifest(tmp_path/'train.jsonl', [row('train', pixels[-1], partition='train')])
    seen = []; original = data._Pixels._collect_first
    def collect(self):
        seen.append(len(self.pending)); return original(self)
    monkeypatch.setattr(data._Pixels, '_collect_first', collect)
    result = data.build_caption_bundle_data(pool, [train], workers=4, cap=2)
    assert max(seen) == 8 and all(count <= 8 for count in seen)
    assert result['audit']['pixels']['distinct_files'] == 18
    assert result['audit']['audited_unselected_pool'] and result['audit']['all_exclusion_pixels_verified']
    assert audit_threads() == []


def test_manifest_error_cancels_queued_tasks_and_joins_started_workers(tmp_path, monkeypatch):
    values = many_pixels(tmp_path, 9); lock = threading.Lock(); state = {'started': 0, 'ended': 0}
    def waiting_decode(path, before, stopped, memory):
        with lock: state['started'] += 1
        try:
            assert stopped.wait(timeout=3), 'source failure must cancel active work'
            raise CancelledError('cancelled test decode')
        finally:
            with lock: state['ended'] += 1
    monkeypatch.setattr(data, '_decode_pixels', waiting_decode)
    rows = [row(str(i), value) for i, value in enumerate(values)]
    rows[-1]['language'] = 'ru'
    pool = manifest(tmp_path/'pool.jsonl', rows)
    train = manifest(tmp_path/'train.jsonl', [row('text', None, partition='train')])
    with pytest.raises(ValueError, match='English captioning'):
        data.build_caption_bundle_data(pool, [train], workers=4)
    assert 1 <= state['started'] <= 4 and state['ended'] == state['started']
    assert audit_threads() == []


@pytest.mark.parametrize('failure', [KeyboardInterrupt, ValueError])
def test_any_caller_exception_cancels_futures_and_leaves_no_worker(tmp_path, monkeypatch, failure):
    values = many_pixels(tmp_path, 8)
    def waiting_decode(path, before, stopped, memory):
        assert stopped.wait(timeout=3)
        raise CancelledError('cancelled')
    monkeypatch.setattr(data, '_decode_pixels', waiting_decode)
    pixels = data._Pixels(tmp_path, workers=4)
    with pytest.raises(failure):
        with pixels:
            for path, digest in values: pixels.check(path, digest)
            raise failure('simulated interruption')
    assert pixels.closed and pixels.stopped.is_set()
    assert all(future.done() for future in pixels.pending.values())
    assert any(future.cancelled() for future in pixels.pending.values())
    assert audit_threads() == []


def test_worker_decode_error_cancels_remaining_queue_and_joins(tmp_path, monkeypatch):
    values = many_pixels(tmp_path, 8)
    def decode(path, before, stopped, memory):
        if str(path) == values[0][0]: raise ValueError('Cannot decode canonical RGB pixels: corrupt')
        assert stopped.wait(timeout=3)
        raise CancelledError('cancelled')
    monkeypatch.setattr(data, '_decode_pixels', decode)
    pixels = data._Pixels(tmp_path, workers=4)
    with pytest.raises(ValueError, match='Cannot decode'):
        with pixels:
            for path, digest in values: pixels.check(path, digest)
            pixels.finish()
    assert pixels.closed and all(future.done() for future in pixels.pending.values())
    assert any(future.cancelled() for future in pixels.pending.values())
    assert audit_threads() == []


def test_changed_file_while_queued_is_rejected_before_decode(corpus, monkeypatch):
    _, _, values = corpus; original = data._decode_pixels
    entered, release = threading.Event(), threading.Event()
    def decode(path, *args):
        if str(path) == values[0][0]:
            entered.set(); assert release.wait(timeout=3)
        return original(path, *args)
    monkeypatch.setattr(data, '_decode_pixels', decode)
    with pytest.raises(ValueError, match='changed before decoding'):
        with data._Pixels(data.ROOT, workers=1) as pixels:
            pixels.check(*values[0]); assert entered.wait(timeout=3)
            pixels.check(*values[1]); image(Path(values[1][0]), (250, 251, 252)); release.set()
            pixels.finish()
    assert audit_threads() == []


def test_duplicate_pending_path_with_conflicting_hash_refused(corpus):
    _, _, values = corpus
    with pytest.raises(ValueError, match='hash mismatch'):
        with data._Pixels(data.ROOT, workers=4) as pixels:
            pixels.check(*values[0]); pixels.check(values[0][0], 'f'*64)
    assert audit_threads() == []


def test_insufficient_cgroup_headroom_refuses_decode_cleanly(corpus, monkeypatch):
    _, _, values = corpus
    monkeypatch.setattr(data, '_memory_available_bytes', lambda: 2**30)
    with pytest.raises(MemoryError, match='memory headroom'):
        with data._Pixels(data.ROOT, workers=4) as pixels:
            pixels.check(*values[0]); pixels.finish()
    assert audit_threads() == []


@pytest.mark.parametrize('workers', [0, 5, True, 1.5])
def test_workers_are_explicitly_bounded(workers):
    with pytest.raises(ValueError, match='workers'):
        data._Pixels(data.ROOT, workers=workers)


def test_cli_default_four_workers_and_signal_handlers_restore(tmp_path, monkeypatch, capsys):
    import signal
    from scripts import prepare_research_caption_eval_v2 as cli
    seen = {}
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def prepare(*args, **kwargs):
        seen.update(kwargs)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    monkeypatch.setattr(cli, 'prepare_caption_bundle', prepare)
    assert cli.main(['--output', str(tmp_path/'bundle')]) == 128+signal.SIGTERM
    assert seen['workers'] == 4 and seen['memory_reserve_bytes'] == 2*2**30
    assert all(signal.getsignal(sig) == handler for sig, handler in before.items())
    assert '"workers_joined": true' in capsys.readouterr().out


def test_strip_hash_is_exact_full_rgb_bytes_across_exif_and_row_boundaries(corpus, tmp_path, monkeypatch):
    _, _, values = corpus
    values = values+[image(tmp_path/'oriented-strip.jpg', (1, 42, 80), exif=True)]
    monkeypatch.setattr(data, 'PIXEL_STRIP_BYTES', 17)
    with data._Pixels(data.ROOT, workers=4) as pixels:
        for value in values: pixels.check(*value)
        receipt = pixels.finish()
    assert receipt['distinct_files'] == len(values)
    assert all(pixels.files[path]['sha256'] == expected for path, expected in values)
    assert audit_threads() == []


def test_progress_exception_on_enter_closes_executor_without_workers():
    def fail(event): raise RuntimeError('progress consumer failed')
    pixels = data._Pixels(data.ROOT, fail, workers=4)
    with pytest.raises(RuntimeError, match='progress consumer'):
        with pixels: pytest.fail('failed progress must prevent audit entry')
    assert pixels.closed and audit_threads() == []


def test_fresh_actual_cli_four_workers_resume_one_and_never_import_torch(corpus, tmp_path):
    pool, train, _ = corpus; output = tmp_path/'cli_bundle'
    code = ("import runpy,sys\n"
            "script=sys.argv.pop(1)\n"
            "try: runpy.run_path(script,run_name='__main__')\n"
            "except SystemExit as result: assert result.code==0\n"
            "assert 'torch' not in sys.modules\n")
    command = [sys.executable, '-c', code, str(data.ROOT/'scripts/prepare_research_caption_eval_v2.py'),
               '--pool', str(pool), '--exclude-train', str(train), '--output', str(output),
               '--workers', '4', '--reserve-gb', '0', '--memory-reserve-gb', '0']
    result = subprocess.run(command, cwd=data.ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert '"workers": 4' in result.stdout and '"status": "prepared"' in result.stdout
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    command[command.index('--workers')+1] = '1'; command.append('--resume')
    resumed = subprocess.run(command, cwd=data.ROOT, capture_output=True, text=True, timeout=20)
    assert resumed.returncode == 0, resumed.stderr
    assert '"status": "reused"' in resumed.stdout
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()} == before


def test_actual_sigterm_during_worker_wait_cancels_and_joins_all_threads(corpus, tmp_path):
    pool, train, _ = corpus; output = tmp_path/'interrupted_bundle'
    code = """import os,signal,sys,threading
from concurrent.futures import CancelledError
from scripts import prepare_research_caption_eval_v2 as cli
from src.eval import research_caption_data_v2 as data
entered=threading.Event()
def decode(path,before,stopped,memory):
    entered.set()
    assert stopped.wait(5)
    raise CancelledError('signal cancelled active worker')
data._decode_pixels=decode
def sender():
    assert entered.wait(5)
    os.kill(os.getpid(),signal.SIGTERM)
thread=threading.Thread(target=sender);thread.start()
result=cli.main(sys.argv[1:]);thread.join()
assert result==128+signal.SIGTERM
assert not any(t.name.startswith('caption-pixels') for t in threading.enumerate())
assert 'torch' not in sys.modules
"""
    command = [sys.executable, '-c', code, '--pool', str(pool), '--exclude-train', str(train),
               '--output', str(output), '--workers', '4', '--reserve-gb', '0']
    result = subprocess.run(command, cwd=data.ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert '"workers_joined": true' in result.stdout and not output.exists()


@pytest.mark.parametrize('stop_name',['SIGTERM','SIGINT'])
@pytest.mark.parametrize('protected',[True,False])
def test_real_signal_in_started_but_unregistered_submit_gap_is_deferred(corpus,tmp_path,stop_name,protected):
    pool,train,_=corpus;output=tmp_path/'interrupted_submit_gap'
    code=r'''import json,os,signal,sys,threading,time
from concurrent.futures import CancelledError
from contextlib import nullcontext
from scripts import prepare_research_caption_eval_v2 as cli
from src.eval import research_caption_data_v2 as data
stop_signal=getattr(signal,sys.argv[1]);protected=sys.argv[2]=='protected';args=sys.argv[3:]
if not protected:data._defer_submit_stop_signals=nullcontext
# A pre-existing unmasked thread proves why main-thread pthread_sigmask alone
# cannot defer Python callbacks. Preserve a separate caller mask bit as well.
signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGINT,signal.SIGTERM})
signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGUSR1})
initial_mask=signal.pthread_sigmask(signal.SIG_BLOCK,set())
initial_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
sender_stop=threading.Event();sender=threading.Thread(target=lambda:sender_stop.wait(5),name='preexisting-signal-recipient')
sender.start();holder={};entered=threading.Event();gap={};worker_masks=[]
original_init=data._Pixels.__init__
def init(self,*args,**kwargs):
    original_init(self,*args,**kwargs);holder['pixels']=self
data._Pixels.__init__=init
def decode(path,before,stopped,memory):
    worker_masks.append(signal.pthread_sigmask(signal.SIG_BLOCK,set()))
    entered.set();assert stopped.wait(5)
    raise CancelledError('main stop cancelled actual worker')
data._decode_pixels=decode
original_start=threading.Thread.start
def start(thread,*args,**kwargs):
    result=original_start(thread,*args,**kwargs)
    if thread.name.startswith('caption-pixels') and not gap:
        pixels=holder['pixels'];assert thread not in pixels.executor._threads
        assert entered.wait(3);gap['started_unregistered']=True;holder['gap_thread']=thread
        signal.pthread_kill(sender.ident,stop_signal)
        # Force main Python signal dispatch while Thread.start has not yet
        # returned to ThreadPoolExecutor._adjust_thread_count registration.
        for _ in range(8):time.sleep(.001)
        gap['continued_before_registration']=True
    return result
threading.Thread.start=start
try:
    result=cli.main(args)
finally:
    threading.Thread.start=original_start
    sender_stop.set();sender.join(timeout=3)
pixels=holder['pixels'];diagnostic=pixels.shutdown_diagnostics
assert result==128+stop_signal
assert pixels.closed and pixels.stopped.is_set()
if protected:
    assert gap=={'started_unregistered':True,'continued_before_registration':True}
    assert len(pixels.pending)==1 and len(pixels.executor._threads)==1 and diagnostic['worker_threads']==1
else:
    # Removing only the fix deterministically reproduces the original broken
    # registration: a real worker started, yet executor owns/joins zero threads.
    assert gap=={'started_unregistered':True}
    assert len(pixels.pending)==0 and len(pixels.executor._threads)==0 and diagnostic['worker_threads']==0
    holder['gap_thread'].join(timeout=3)  # Test-only cleanup of its unregistered worker.
assert diagnostic['alive_worker_threads']==0 and diagnostic['all_futures_done']
assert all(not thread.is_alive() for thread in pixels.executor._threads)
assert not any(thread.name.startswith('caption-pixels') for thread in threading.enumerate())
assert signal.pthread_sigmask(signal.SIG_BLOCK,set())==initial_mask
assert all(signal.getsignal(sig)==handler for sig,handler in initial_handlers.items())
assert worker_masks==[initial_mask]
assert 'torch' not in sys.modules
print(json.dumps({'test':'actual_started_unregistered_gap','signal':stop_signal,'protection':protected,'diagnostic':diagnostic}))
'''
    command=[sys.executable,'-c',code,stop_name,'protected' if protected else 'regression_control','--pool',str(pool),'--exclude-train',str(train),'--output',str(output),
        '--workers','4','--reserve-gb','0','--preparation-profile',data.V2_PROFILE]
    result=subprocess.run(command,cwd=data.ROOT,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr+'\n'+result.stdout
    assert '"workers_joined": true' in result.stdout and 'actual_started_unregistered_gap' in result.stdout
    assert f'"worker_threads": {1 if protected else 0}' in result.stdout and not output.exists()


def test_submit_exception_restores_masks_handlers_and_joins(corpus,monkeypatch):
    import signal
    _,_,values=corpus;pixels=data._Pixels(data.ROOT,workers=4)
    before_mask=signal.pthread_sigmask(signal.SIG_BLOCK,set())
    handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def fail(*args,**kwargs):raise ValueError('submit fixture failure')
    monkeypatch.setattr(pixels.executor,'submit',fail)
    with pytest.raises(ValueError,match='submit fixture'):
        with pixels:pixels.check(*values[0])
    assert signal.pthread_sigmask(signal.SIG_BLOCK,set())==before_mask
    assert all(signal.getsignal(sig)==handler for sig,handler in handlers.items())
    assert pixels.closed and pixels.shutdown_diagnostics['alive_worker_threads']==0 and audit_threads()==[]


def test_default_term_replay_preserves_process_delivery_when_main_mask_blocks():
    import signal
    code=r'''import signal,threading,time
from src.eval.research_caption_data_v2 import _defer_submit_stop_signals
signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM})
signal.signal(signal.SIGTERM,signal.SIG_DFL)
ready=threading.Event()
def recipient():
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM});ready.set()
    while True:time.sleep(.01)
thread=threading.Thread(target=recipient,daemon=True);thread.start();assert ready.wait(2)
with _defer_submit_stop_signals():
    signal.pthread_kill(thread.ident,signal.SIGTERM)
    for _ in range(10):time.sleep(.001)
    print('BODY_BOOKKEEPING_COMPLETED',flush=True)
print('UNEXPECTED_AFTER_DEFAULT_STOP',flush=True)
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=data.ROOT,capture_output=True,text=True,timeout=10)
    assert result.returncode==-signal.SIGTERM,result.stderr+'\n'+result.stdout
    assert 'BODY_BOOKKEEPING_COMPLETED' in result.stdout and 'UNEXPECTED_AFTER_DEFAULT_STOP' not in result.stdout


def test_explicit_v2_profile_preserves_pixels_rank_references_and_v1_default(corpus,tmp_path):
    from src.eval import research_caption_data as legacy_data
    pool,train,_=corpus;legacy=tmp_path/'legacy';v2=tmp_path/'caption-eval-pilot-en-v2'
    legacy_data.prepare_caption_bundle(pool,[train],legacy,reserve_bytes=0)
    data.prepare_caption_bundle(pool,[train],v2,reserve_bytes=0,preparation_profile=data.V2_PROFILE)
    assert json.loads((legacy/'protocol.json').read_text())['protocol']=='research_caption_pilot_eval_v1'
    protocol=json.loads((v2/'protocol.json').read_text())
    assert protocol['protocol']=='research_caption_pilot_eval_v2' and protocol['preparation_profile']==data.V2_PROFILE
    for name in ('generation.jsonl','references.jsonl','audit.json'):assert (v2/name).read_bytes()==(legacy/name).read_bytes()
    assert data.prepare_caption_bundle(pool,[train],v2,reserve_bytes=0,preparation_profile=data.V2_PROFILE,resume=True)['status']=='reused'
    with pytest.raises(ValueError,match='profile'):
        data.prepare_caption_bundle(pool,[train],v2,reserve_bytes=0,preparation_profile=None,resume=True)


def test_existing_v1_bundle_refuses_new_source_resume_before_any_decode(monkeypatch):
    existing=data.ROOT/'datasets/research/caption-eval-pilot-en-v1'
    before={path.name:data.file_sha(path) for path in existing.iterdir()}
    def fail(*args,**kwargs):pytest.fail('Old v1 source mismatch must refuse before image audit')
    monkeypatch.setattr(data,'_decode_pixels',fail)
    with pytest.raises(ValueError,match='source differs'):
        data.prepare_caption_bundle(data.ROOT/'datasets/research/pilot-en-v1/alignment_validation.jsonl',
            [data.ROOT/'datasets/research/pilot-en-v1/alignment_train.jsonl'],existing,resume=True)
    assert {path.name:data.file_sha(path) for path in existing.iterdir()}==before


def test_explicit_v2_cli_new_output_and_no_torch(corpus,tmp_path):
    pool,train,_=corpus;output=tmp_path/'caption-eval-pilot-en-v2'
    code=("import sys; from scripts.prepare_research_caption_eval_v2 import main; "
          "assert main(sys.argv[1:])==0; assert 'torch' not in sys.modules")
    result=subprocess.run([sys.executable,'-c',code,'--pool',str(pool),'--exclude-train',str(train),'--output',str(output),
        '--reserve-gb','0','--preparation-profile',data.V2_PROFILE],cwd=data.ROOT,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert json.loads((output/'protocol.json').read_text())['protocol']=='research_caption_pilot_eval_v2'


def test_archived_v1_code_bundle_and_caption_owner_proofs_remain_exact():
    archive=data.ROOT/'research_results/caption_data_v1_before_signal_fix/source_index.json'
    receipt=json.loads(archive.read_text())
    for entry in receipt['files'].values():assert data.file_sha(data.ROOT/entry['archive'])==entry['sha256']
    bundle=data.ROOT/'datasets/research/caption-eval-pilot-en-v1'
    assert {path.name:data.file_sha(path) for path in bundle.iterdir()}=={
        name:entry['sha256'] for name,entry in receipt['full_immutable_v1_bundle_file_hashes'].items()}
    assert receipt['files']['src/eval/research_caption_generation.py']['sha256']=='754c261266ab3544d81560abd90447d6abc2ad6b74b987321d007815407abefa'
    assert receipt['files']['configs/research/caption_evaluator_ready.json']['sha256']=='04f980e43f8ccbad05939aef1cc18040ca0475092bb84bbbc407fb0d1a4cc526'
    for name in ('src/eval/research_caption_data.py','scripts/prepare_research_caption_eval.py','tests/test_research_caption_data.py'):
        assert data.file_sha(data.ROOT/name)==receipt['files'][name]['sha256']
