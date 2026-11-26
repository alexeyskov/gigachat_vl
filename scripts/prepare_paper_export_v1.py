#!/usr/bin/env python3
"""Curate article artifacts and losslessly compressed, proof-bound final weights."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = ROOT / 'exports/paper_v1'
GiB = 2**30
BLOCK = 8 * 2**20
RESERVE = 32 * GiB


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(BLOCK):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    stat = Path(path).stat()
    # Creating a hardlink changes ctime/nlink, not these immutable content fields.
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def disk_guard(directory, extra=0):
    free = shutil.disk_usage(directory).free
    if free < RESERVE + extra:
        raise OSError(f'Export needs {extra/GiB:.2f}GiB extra plus 32GiB reserve; free={free/GiB:.2f}GiB')


def choose_weights(rows, profile):
    if profile == 'metrics':
        return []
    if profile == 'all-final':
        return list(rows)
    selected = []
    for model in ('smallthinker4', 'lfm25', 'giga31'):
        candidates = [row for row in rows if row['model'] == model]
        if not candidates:
            raise ValueError('Missing final model family: ' + model)
        # Internal selection metric only. Detection test never selects a checkpoint.
        selected.append(min(candidates, key=lambda row: (-row['VQA_after'], row['checkpoint_bytes'], row['id'])))
    return selected


def final_rows():
    plan = read(ROOT / 'configs/research/detection_mix_plan_v1.json')
    state = read(ROOT / 'runs/research/detection_mix_v1/queue_state.json')
    if state['status'] != 'completed' or len(plan['jobs']) != 18:
        raise ValueError('All 18 final experiments must have completed')
    result = []
    for job in plan['jobs']:
        run = Path(job['output'])
        if run.parent != ROOT / 'runs/research/detection_mix_v1':
            raise ValueError('Unexpected final run directory')
        proof = read(run / 'native_reload_proof.json')
        done = read(run / 'completed.json')
        if (proof['status'] != 'passed' or proof['mismatches']
                or proof['checkpoint_sha256'] != done['checkpoint_sha256']
                or done['steps'] != 56 or done['records'] != 3529):
            raise ValueError('Independent final replay missing: ' + job['id'])
        checkpoint = (run / 'last/state.pt').resolve(strict=True)
        owner = read(run / '.checkpoint_manager.json')['owner']
        if (checkpoint.parent.parent != run / '.checkpoint_objects'
                or read(checkpoint.parent / '.research_checkpoint.json')['owner'] != owner
                or (run / 'best').resolve() != (run / 'last').resolve()):
            raise ValueError('Final checkpoint is not one owned best/last object')
        before = read(run / 'evaluation/before_internal116.metrics.json')
        after = read(run / 'evaluation/after_internal116.metrics.json')
        test = read(run / 'evaluation/heldout_detection_test.metrics.json')
        result.append(dict(id=job['id'], model=job['model_key'], vision=job['vision_key'], method=job['method'],
                           run=str(run.relative_to(ROOT)), checkpoint=str(checkpoint.relative_to(ROOT)),
                           checkpoint_bytes=checkpoint.stat().st_size, checkpoint_sha256=proof['checkpoint_sha256'],
                           VQA_before=before['group_macro_score'], VQA_after=after['group_macro_score'],
                           detection_test_F1_IoU50=test['primary_score'],
                           detection_test_JSON_valid=test['JSON_and_coordinate_valid_rate'],
                           negative_query_accuracy=test['negative_query_accuracy']))
    return result


class DigestReader:
    def __init__(self, handle, on_progress=None):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.count = 0
        self.on_progress = on_progress
        self.advised = 0

    def read(self, count=-1):
        chunk = self.handle.read(count)
        self.digest.update(chunk)
        self.count += len(chunk)
        if self.count - self.advised >= 128 * 2**20:
            if self.on_progress:
                self.on_progress(self.count)
            # Read-only advice for already-consumed, clean source pages.
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(self.handle.fileno(), self.advised, self.count - self.advised, os.POSIX_FADV_DONTNEED)
            self.advised = self.count
        return chunk


def zstd_tool():
    tool = shutil.which('zstd') or str(Path(sys.executable).with_name('zstd'))
    if not Path(tool).is_file():
        raise RuntimeError('zstd is required on the exporting server')
    return tool


def verify_archive(archive, expected):
    """Decode and SHA every archived file without writing a second raw checkpoint."""
    child = subprocess.Popen([zstd_tool(), '-q', '-d', '-c', str(archive)], stdout=subprocess.PIPE)
    found = set()
    decoded_bytes = 0
    last_progress = time.monotonic()
    try:
        # Large buffers avoid repeatedly concatenating 10KiB tar chunks while
        # hashing multi-GiB tensors. Memory stays bounded regardless of model size.
        with tarfile.open(fileobj=child.stdout, mode='r|', bufsize=BLOCK) as source:
            for member in source:
                if not member.isfile() or member.name not in expected or member.name in found:
                    raise ValueError('Unexpected/duplicate member in export archive')
                wanted = expected[member.name]
                if member.size != wanted['bytes']:
                    raise ValueError('Archive member size differs')
                stream = source.extractfile(member)
                digest = hashlib.sha256()
                while chunk := stream.read(BLOCK):
                    digest.update(chunk)
                    decoded_bytes += len(chunk)
                    if time.monotonic() - last_progress > 25:
                        emit('verifying_bytes', archive=Path(archive).name,
                             decoded_GiB=round(decoded_bytes/GiB, 2))
                        last_progress = time.monotonic()
                if digest.hexdigest() != wanted['sha256']:
                    raise ValueError('Archive does not reconstruct the original bytes: ' + member.name)
                found.add(member.name)
        # Drain trailing tar padding and the zstd checksum before waiting.
        while child.stdout.read(BLOCK):
            pass
        if child.wait() or found != set(expected):
            raise ValueError('Incomplete archive or zstd integrity error')
    finally:
        child.stdout.close()
        if child.poll() is None:
            child.terminate()
            child.wait()


def archive_files(entries, output, expected_sha=None, threads=8):
    """Native tar.zst; preserve checkpoint byte identity, dtype and model semantics."""
    output = Path(output)
    receipt_path = output.with_suffix(output.suffix + '.json')
    expected_sha = expected_sha or {}
    if output.exists() and receipt_path.exists():
        receipt = read(receipt_path)
        if (receipt['archive_bytes'] != output.stat().st_size
                or set(receipt['members']) != {name for name, _ in entries}
                or any(receipt['members'][name]['sha256'] != value for name, value in expected_sha.items())):
            raise ValueError('Existing export receipt differs')
        return receipt
    output.parent.mkdir(parents=True, exist_ok=True)
    disk_guard(output.parent, sum(path.stat().st_size for _, path in entries))
    temporary = output.with_name(output.name + '.partial')
    # Only this locked export's unpublished staging file is eligible for cleanup.
    temporary.unlink(missing_ok=True)
    snapshots = {name: identity(path) for name, path in entries}
    members = {}
    last_progress = 0

    def progress(count):
        nonlocal last_progress
        disk_guard(output.parent)
        if time.monotonic() - last_progress > 25:
            emit('compressing', archive=output.name, current_file_GiB=round(count/GiB, 2))
            last_progress = time.monotonic()

    emit('archive_started', archive=output.name, raw_GiB=round(sum(s[2] for s in snapshots.values())/GiB, 3))
    with temporary.open('xb') as handle:
        child = subprocess.Popen([zstd_tool(), '-q', '-3', '-T' + str(threads), '-c'], stdin=subprocess.PIPE, stdout=handle)
        try:
            with tarfile.open(fileobj=child.stdin, mode='w|', bufsize=BLOCK, copybufsize=BLOCK) as target:
                for name, path in entries:
                    if Path(name).is_absolute() or '..' in Path(name).parts or identity(path) != snapshots[name]:
                        raise ValueError('Unsafe/changed export source')
                    info = target.gettarinfo(str(path), arcname=name)
                    if not info.isfile():
                        raise ValueError('Only regular payload files may enter archives')
                    with path.open('rb') as source:
                        reader = DigestReader(source, progress)
                        target.addfile(info, reader)
                    digest = reader.digest.hexdigest()
                    if reader.count != snapshots[name][2] or identity(path) != snapshots[name]:
                        raise ValueError('Export source changed while reading')
                    if name in expected_sha and digest != expected_sha[name]:
                        raise ValueError('Source SHA differs from native GPU proof')
                    members[name] = dict(bytes=reader.count, sha256=digest)
            child.stdin.close()
            if child.wait():
                raise RuntimeError('zstd compression failed')
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            if not child.stdin.closed:
                child.stdin.close()
            if child.poll() is None:
                child.terminate()
                child.wait()
    emit('verifying_archive', archive=output.name)
    verify_archive(temporary, members)
    archive_sha = sha(temporary)
    temporary.replace(output)
    receipt = dict(status='byte_exact_verified', compression='tar+zstd_level3', members=members,
                   raw_bytes=sum(v['bytes'] for v in members.values()), archive_bytes=output.stat().st_size,
                   archive_sha256=archive_sha, no_quantization_or_dtype_conversion=True)
    atomic(receipt_path, receipt)
    if hasattr(os, 'posix_fadvise'):
        with output.open('rb') as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    emit('archive_ready', archive=output.name, compressed_GiB=round(receipt['archive_bytes']/GiB, 3),
         fraction=round(receipt['archive_bytes']/receipt['raw_bytes'], 4))
    return receipt


def copy_metadata(run, target):
    for source in sorted(run.rglob('*')):
        relative = source.relative_to(run)
        destination = target / relative
        if source.is_symlink():
            if source.name not in ('best', 'last') or source.readlink().is_absolute():
                raise ValueError('Unexpected metadata symlink')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(destination):
                if not destination.is_symlink() or destination.readlink() != source.readlink():
                    raise ValueError('Existing metadata role pointer differs')
            else:
                destination.symlink_to(source.readlink(), target_is_directory=True)
        elif source.is_file() and (source.suffix in ('.json', '.jsonl', '.csv', '.md', '.txt') or source.name == '.checkpoint.lock'):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def evaluation_assets(selected):
    names = ['datasets/research/budget18-en-v1/release.json',
             'datasets/research/budget18-en-v1/sft_validation.jsonl',
             'datasets/research/budget18-en-v1/internal_selection_116.jsonl',
             'datasets/research/detection-v1/release.json',
             'datasets/research/detection-v1/audit.json',
             'datasets/research/detection-v1/detection_validation.jsonl',
             'datasets/research/detection-v1/detection_test.jsonl']
    records = []
    for name in (names[2], names[-2], names[-1]):
        records.extend(json.loads(line) for line in (ROOT / name).read_text().splitlines())
    images = {Path(path).resolve(strict=True) for record in records for path in record['image_paths']}
    files = {ROOT / name for name in names} | images
    image_hashes = {value for record in records for value in record['image_hashes']}
    for row in selected:
        run = ROOT / row['run']
        config = read(run / 'resolved_config.json')
        fingerprint = read(run / 'initialization.json')['cache_fingerprint']
        cache = Path(config['cache_dir'])
        cache = cache if cache.is_absolute() else ROOT / cache
        candidates = [p for p in (cache / fingerprint, cache / 'derived_vision_v1' / fingerprint) if p.is_dir()]
        if len(candidates) != 1:
            raise ValueError('Exact cached feature namespace not found')
        for image_hash in image_hashes:
            key = hashlib.sha256((fingerprint + image_hash).encode()).hexdigest()
            files.add((candidates[0] / key[:2] / (key + '.pt')).resolve(strict=True))
    for path in files:
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Evaluation asset outside the experiment workspace')
    return sorted((str(path.relative_to(ROOT)), path) for path in files)


def build(profile):
    rows = final_rows()
    selected = choose_weights(rows, profile)
    output = EXPORT_ROOT / profile
    output.mkdir(parents=True, exist_ok=True)
    selection = {'all-final': 'one final checkpoint for every one of the 18 trained variants',
                 'metrics': 'predictions and references for all 18 variants; no weights',
                 'best': 'highest after-stage internal116 VQA per model; ties by smaller checkpoint then id; OD test not used'}[profile]
    contract = dict(version=1, profile=profile, selection=selection,
                    final_rows=rows, selected_ids=[r['id'] for r in selected])
    owner = output / '.export_owner.json'
    if owner.exists():
        previous = read(owner)
        if any(previous[key] != contract[key] for key in ('version', 'profile', 'final_rows', 'selected_ids')):
            raise ValueError('Existing export contract differs; create a new revision')
    atomic(owner, contract)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        existing = read(manifest_path)
        if existing.get('status') == 'ready':
            for item in existing['archives']:
                if (output / item['file']).stat().st_size != item['archive_bytes']:
                    raise ValueError('Published export archive size changed')
            emit('reusing_verified_payloads', profile=profile)
    # Completed archives survive retries; reserve space only for pending payloads.
    if profile == 'best':
        pending_bytes = 0
        for row in selected:
            archive = output / 'weights' / (row['id'] + '.tar.zst')
            if not (archive.is_file() and archive.with_suffix('.zst.json').is_file()):
                pending_bytes += row['checkpoint_bytes']
        disk_guard(output, pending_bytes + 2*GiB)
    for row in rows:
        copy_metadata(ROOT / row['run'], output / row['run'])
    for source in sorted((ROOT / 'configs').rglob('*')):
        if source.is_file() and source.suffix in ('.json', '.txt') and not source.name.endswith('.state.json'):
            target = output / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ['RESEARCH_PIPELINE.md', 'DETECTION_PIPELINE.md', 'BENCHMARKS.md', 'STORAGE_RETENTION.md',
                 'research_results/model_inventory.json', 'research_results/environment.lock.txt']:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    reports = output / 'reports'
    reports.mkdir(exist_ok=True)
    source_reports = ROOT / 'research_results/detection_mix_v1'
    for name in ['FINAL_RESULTS.md', 'comparison_final_20261126.csv', 'final_summary.json',
                 'geometry_collapse_diagnostic.json', 'geometry_collapse_diagnostic.md']:
        shutil.copy2(source_reports / name, reports / name)
    # Preserve the external baseline and the previous stage's compact reports,
    # without bringing their former checkpoints or training caches along.
    for name in ('capacity_eval_v5', 'qwen25_native_int4_internal116_v1'):
        copy_metadata(ROOT / 'research_results' / name,
                      output / 'research_results' / name)
    references = output / 'evaluation_manifests'
    references.mkdir(exist_ok=True)
    for name in ('datasets/research/budget18-en-v1/internal_selection_116.jsonl',
                 'datasets/research/detection-v1/detection_validation.jsonl',
                 'datasets/research/detection-v1/detection_test.jsonl'):
        shutil.copy2(ROOT / name, references / Path(name).name)
    atomic(reports / 'all_final_metrics.json', dict(metric_scale='0-100', VQA='internal116 single-teacher-reference macro, not official benchmark',
                                                 detection='positive-query macro F1@IoU0.5 on64test, not COCO AP', rows=rows))
    atomic(reports / 'weight_coverage.json', [dict(id=r['id'], VQA_after=r['VQA_after'],
                                                weights_in_export=r['id'] in contract['selected_ids']) for r in rows])
    archives = []
    if profile == 'best':
        for row in selected:
            archive = output / 'weights' / (row['id'] + '.tar.zst')
            receipt = archive_files([(row['checkpoint'], ROOT / row['checkpoint'])], archive,
                                    {row['checkpoint']: row['checkpoint_sha256']})
            archives.append(dict(file=str(archive.relative_to(output)), **receipt))
    elif profile == 'all-final':
        # A zero-extra-storage view of all final weights; rsync compresses them
        # in transit. No second185GiB server copy and no original weights removed.
        for row in selected:
            source = ROOT / row['checkpoint']
            target = output / row['checkpoint']
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not os.path.samefile(source, target):
                    raise ValueError('Unexpected existing exported checkpoint')
            else:
                os.link(source, target)
    if selected:
        assets = evaluation_assets(selected)
        receipt = archive_files(assets, output / 'evaluation_assets.tar.zst')
        archives.append(dict(file='evaluation_assets.tar.zst', **receipt))
    selected_summary = [{k: r[k] for k in ('id', 'model', 'vision', 'method', 'VQA_after', 'checkpoint', 'checkpoint_bytes', 'checkpoint_sha256')} for r in selected]
    readme = f'''# Пакет результатов для статьи

Профиль: {profile}. Ответы и проверки всех 18 финальных конфигураций включены.
Выбранные веса: {len(selected)}. Политика: {selection}.

По умолчанию sync.sh скачивает all-final: один финальный checkpoint каждого из 18 вариантов.
Веса сжимаются rsync при передаче, а локально сохраняются в исходном формате, без переквантования.
Для пересчёта метрик по сохранённым ответам достаточно metrics (--metrics-only), без GPU и весов.
Из Git-копии исходников:

```bash
python3 scripts/recompute_paper_metrics_v1.py --bundle /путь/к/скачанному/пакету
```

Скрипт заново оценивает raw predictions по evaluation_manifests и проверяет совпадение всех 18 строк.
Before-stage метрики сохранены для сравнения; весов прежнего этапа в пакете нет.

Для повторной генерации нужны финальные веса, Git-код и соответствующие базовые HF модели.
Точные repo/revision сохранены в конфигурациях. QLoRA-чекпоинты требуют frozen базовую LLM.
Включены только оценочные изображения и их исходные BF16 features; training dataset/cache и
frozen vision encoder weights не копируются. Распаковка оценочных assets из корня пакета:

```bash
zstd -dc evaluation_assets.tar.zst | tar -xf -
```

Нужны zstd и tar. Для пересчёта готовых ответов распаковка assets не требуется.
Исторические абсолютные пути в JSON не переписаны: для генерации настройте пути загрузчика.
Исходный SHA256 каждого checkpoint из native GPU proof записан в manifest.
Best/last указывают на один физический файл. Внутренний профиль best содержит лишь три выбранных
чекпоинта и не подходит для повторной генерации всех 18 вариантов.

VQA116 — внутренняя выборка с одним teacher reference, не официальный benchmark. Detection F1 — не COCO AP.
На первых 14 вариантах выявлены грубые рамки почти на весь кадр; отрицательные запросы провалены у всех 18.
'''
    (output / 'README.md').write_text(readme)
    files = [p for p in output.rglob('*') if p.is_file() and not p.is_symlink() and p.name != 'manifest.json']
    download_bytes = sum(p.stat().st_size for p in files)
    manifest = dict(status='ready', created_UTC=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                    profile=profile, final_metric_rows=18, selected_weights=selected_summary, selection=contract['selection'],
                    archives=archives, download_bytes=download_bytes,
                    original_selected_weight_bytes=sum(r['checkpoint_bytes'] for r in selected),
                    checkpoint_bytes_changed=False, original_server_weights_deleted=False,
                    all_final_weights_included=profile == 'all-final')
    atomic(manifest_path, manifest)
    emit('export_ready', profile=profile, weights=len(selected), download_GiB=round(download_bytes/GiB, 3))
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--profile', choices=('best', 'metrics', 'all-final'), default='all-final')
    args = parser.parse_args()
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    with (EXPORT_ROOT / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build(args.profile)


if __name__ == '__main__':
    main()
