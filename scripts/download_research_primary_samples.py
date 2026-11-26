#!/usr/bin/env python3
"""Fetch a fixed, bounded set of TRAIN source samples; never authorize training."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.research_storage import DiskBudget, _atomic_json

OWNER = 'primary_source_bounded_train_samples_v1'
PLAN = ROOT / 'configs/research/primary_source_sample_download.json'
PLAN_SHA256 = 'a39f70004c388d3e6934f0ec51ddcd418297b50fe279e673dd6150a603fe7602'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(8 * 2**20):
            result.update(chunk)
    return result.hexdigest()


def identity(path):
    value = Path(path).stat()
    return tuple(getattr(value, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def emit(event, **fields):
    print(json.dumps(dict(time=time.time(), event=event, **fields)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if sha(PLAN) != PLAN_SHA256:
        raise ValueError('Pinned source download plan changed')
    plan = json.loads(PLAN.read_bytes())
    if (plan['owner'] != OWNER or plan['max_new_bytes'] != 3 * 2**30 or
            plan['reserve_gib'] != 120 or len(plan['files']) != 7 or
            plan['primary_training_authorized'] is not False):
        raise ValueError('Bounded TRAIN-only source plan differs')
    budget = DiskBudget(ROOT, 120)
    selected, missing = [], 0
    for row in plan['files']:
        inventory = ROOT / row['inventory']
        if sha(inventory) != row['inventory_sha256']:
            raise ValueError('Pinned source inventory changed')
        info = json.loads(inventory.read_bytes())
        if info['repo'] != row['repo'] or info['revision'] != row['revision']:
            raise ValueError('Source revision differs from inventory')
        match = [entry for entry in info['files'] if entry['path'] == row['path']]
        parts = PurePosixPath(row['path'])
        if (len(match) != 1 or match[0]['size'] != row['bytes'] or
                parts.is_absolute() or '..' in parts.parts or
                not parts.name.startswith('train-') or parts.suffix != '.parquet'):
            raise ValueError('Unexpected TRAIN file/path/size')
        directory = ROOT / 'datasets/research/raw' / row['repo'].replace('/', '--')
        path = directory / row['path']
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError('Raw TRAIN sample path cannot traverse symlinks')
        exists = path.is_file() and path.stat().st_size == row['bytes']
        if path.exists() and not exists:
            raise ValueError('Unexpected existing sample size; preserve it for inspection')
        missing += 0 if exists else row['bytes']
        selected.append((row, directory, path, exists))
    if sum(row['bytes'] for row, *_ in selected) != plan['declared_total_bytes'] or missing > plan['max_new_bytes']:
        raise ValueError('Download total exceeds fixed plan cap')
    budget.ensure_space(math.ceil(missing * 1.1) + 64 * 2**20)
    emit('bounded_train_download_preflight', files=len(selected), missing_bytes=missing,
         max_new_bytes=plan['max_new_bytes'], download_enabled=args.download, disk=budget.report())
    if not args.download:
        return
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    target = ROOT / 'research_results/primary_source_sample_download.json'
    proof = {'schema_version': 1, 'owner': OWNER, 'status': 'running', 'started_unix': time.time(),
        'plan_sha256': PLAN_SHA256, 'script_sha256': sha(__file__), 'files': [],
        'declared_download_bytes': missing, 'primary_training_authorized': False,
        'eligible_pool_counts_known': False, 'scientific_train_manifest_created': False,
        'image_encoder_or_GPU_used': False}
    _atomic_json(target, proof)
    for row, directory, path, exists in selected:
        budget.ensure_space(math.ceil((0 if exists else row['bytes']) * 1.1) + 64 * 2**20)
        emit('train_source_file_start', repo=row['repo'], path=row['path'], bytes=row['bytes'], cache_hit=exists)
        started = time.monotonic()
        if not exists:
            fetched = Path(hf_hub_download(row['repo'], row['path'], repo_type='dataset',
                revision=row['revision'], local_dir=str(directory)))
            if fetched.resolve() != path.resolve():
                raise ValueError('HF returned an unexpected local path')
        if path.is_symlink() or path.stat().st_size != row['bytes']:
            raise ValueError('Downloaded file size/path mismatch')
        before = identity(path)
        checksum = sha(path)
        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
        schema = str(parquet.schema_arrow)
        summary = {'rows': metadata.num_rows, 'row_groups': metadata.num_row_groups,
            'schema_arrow': schema, 'image_or_text_column_bodies_read': False}
        if identity(path) != before:
            raise ValueError('TRAIN sample changed while hashing/reading its footer')
        item = {**row, 'local_path': str(path), 'local_full_file_sha256': checksum,
            'cache_hit': exists, 'elapsed_seconds': time.monotonic() - started, 'parquet': summary,
            'source_revision_pinned': True, 'full_source_pool_downloaded': False}
        proof['files'].append(item)
        proof['disk'] = budget.ensure_space()
        _atomic_json(target, proof)
        emit('train_source_file_completed', repo=row['repo'], path=row['path'], rows=metadata.num_rows,
            elapsed_seconds=item['elapsed_seconds'], disk=proof['disk'])
    if sha(PLAN) != PLAN_SHA256 or sha(__file__) != proof['script_sha256']:
        raise ValueError('Download plan/script changed during execution')
    proof.update(status='completed', finished_unix=time.time(), disk=budget.ensure_space())
    _atomic_json(target, proof)
    emit('bounded_train_samples_completed', files=len(proof['files']), disk=proof['disk'], primary_training_authorized=False)


if __name__ == '__main__':
    main()
