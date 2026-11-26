#!/usr/bin/env python3
"""Retire only the exact released, recomputable, outside-budget feature list."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import plan_research_budget18_cache_retirement_v1 as plan
from src.utils.research_storage import DiskBudget, _atomic_json
from src.utils.research_lossless_delta import memory_guard

RELEASE = plan.OUT / 'release.json'
RELEASE_SHA = '720fafa48bb0b5cda91e388e61b33f212d7c21e573dbcf0fa37cf7309f0f04a7'
OUT = ROOT / 'research_results/budget18_cache_actual_retirement_v1'


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def guard():
    memory_guard(0, headroom_bytes=32 * 2**30)
    DiskBudget(ROOT, 120).ensure_space(0)


def closure():
    if sha_file(plan.DATA / 'release.json') != plan.RELEASE:
        raise ValueError('Active data release changed')
    release = json.loads((plan.DATA / 'release.json').read_text())
    protected = set()
    for name, item in release['files'].items():
        if not name.endswith('.jsonl'):
            continue
        if sha_file(item['path']) != item['sha256']:
            raise ValueError('Active protected manifest changed')
        for record in rows(item['path']):
            protected.update(record.get('image_hashes', []))
    report = json.loads((plan.OUT / 'report.json').read_text())
    for name, item in report['inputs'].items():
        if 'grounding_v1/blank/' in name:
            if sha_file(item['path']) != item['sha256']:
                raise ValueError('Fixed blank intervention source changed')
            if name.endswith('transformed_manifest.jsonl'):
                for record in rows(item['path']):
                    protected.update(record['image_hashes'])
    if len(protected) != 7616:
        raise ValueError('Expected complete budget+blank closure')
    return protected


def validate_candidate(row, protected):
    path = Path(row['path'])
    base = ROOT / 'cache/vision' / plan.SIG
    if (path.is_symlink() or not path.is_relative_to(base)
            or row['classification'] != 'candidate_recomputable_old_feature'
            or row['metadata']['image_hash'] in protected):
        raise ValueError('Only ordinary outside-budget owned features may retire')
    current, metadata = plan.header(path)
    if current != row['stat'] or metadata != row['metadata']:
        raise ValueError('Candidate identity or metadata changed')
    sources = row['available_original_images']
    if not sources or not any(Path(s['path']).is_file() and not Path(s['path']).is_symlink()
                              and plan.stat_row(s['path']) == s['stat'] for s in sources):
        raise ValueError('An unchanged original image must remain available')
    physical_sha = sha_file(path)
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if '/data/' in m.filename]
        if len(members) != 1:
            raise ValueError('One exact BF16 backing required')
        digest = hashlib.sha256()
        with archive.open(members[0]) as handle:
            for chunk in iter(lambda: handle.read(2**20), b''):
                digest.update(chunk)
        if digest.hexdigest() != metadata['stored_feature_tensor_checksum']:
            raise ValueError('Actual feature body checksum differs')
    if plan.stat_row(path) != current:
        raise ValueError('Candidate changed during complete hashing')
    return dict(path=str(path), stat=current, whole_file_sha256=physical_sha,
                tensor_checksum_verified=True, image_hash=metadata['image_hash'])


def scan_users(candidates):
    from scripts.research_archival_stock_queue_v5 import scan_anchor_users
    identities = [(r['stat']['st_dev'], r['stat']['st_ino']) for r in candidates]
    # Bound every argv below the OS per-argument limit of the frozen scanner.
    return [scan_anchor_users(set(identities[i:i+1000]))
            for i in range(0, len(identities), 1000)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if sha_file(RELEASE) != RELEASE_SHA:
        raise ValueError('Exact reviewed retirement release required')
    release = json.loads(RELEASE.read_text())
    for name in ('candidate_features.jsonl', 'protected_image_closure.jsonl', 'report.json'):
        item = release['bindings'][name]
        if sha_file(item['path']) != item['sha256']:
            raise ValueError('Retirement plan changed')
    protected = closure()
    candidates = list(rows(plan.OUT / 'candidate_features.jsonl'))
    if len(candidates) != 17981 or len({r['path'] for r in candidates}) != len(candidates):
        raise ValueError('Exact unique reviewed candidates required')
    if not args.execute:
        print(json.dumps({'status': 'passed_plan_preflight_no_deletion', 'candidates': len(candidates),
                          'protected_RGB': len(protected)}));return
    if OUT.exists():
        raise ValueError('Existing execution receipt requires explicit audited recovery')
    guard();OUT.mkdir();started=time.time();disk_before=DiskBudget(ROOT,120).report()
    _atomic_json(OUT/'intent.json', {'owner':'ROOT_budget18_unused_cache_retirement_v1',
                 'plan_release_sha256':RELEASE_SHA,'script_sha256':sha_file(__file__),
                 'only_recomputable_old_feature_files':True,'candidate_count':17981,
                 'original_images_models_scientific_weights_untouched':True})
    scan_before = scan_users(candidates)
    checked=[]
    with (OUT/'full_feature_hashes.jsonl').open('x') as journal:
        for index,row in enumerate(candidates):
            if index % 100 == 0:
                guard();print(json.dumps({'event':'hashing_owned_old_features','done':index,'total':17981}),flush=True)
            verified=validate_candidate(row,protected);checked.append(verified)
            journal.write(json.dumps(verified,sort_keys=True)+'\n')
        journal.flush();os.fsync(journal.fileno())
    if closure()!=protected:raise ValueError('Protected closure changed before deletion')
    scan_after_hash = scan_users(candidates)
    _atomic_json(OUT/'preflight.json',{'status':'passed_all_files','feature_hashes_sha256':sha_file(OUT/'full_feature_hashes.jsonl'),
                 'uid_scans_before':scan_before,'uid_scans_after_hash':scan_after_hash,'candidate_count':len(checked)})
    reclaimed=0;directories=set()
    with (OUT/'retired.jsonl').open('x') as journal:
        for index,row in enumerate(checked):
            if index % 100 == 0:guard()
            path=Path(row['path'])
            if plan.stat_row(path)!=row['stat'] or row['image_hash'] in protected:
                raise ValueError('Candidate changed before unlink')
            _atomic_json(OUT/'last_unlink_intent.json',dict(index=index,**row))
            path.unlink();reclaimed+=row['stat']['st_blocks']*512;directories.add(path.parent)
            journal.write(json.dumps({'index':index,**row},sort_keys=True)+'\n');journal.flush();os.fsync(journal.fileno())
            if index%100==0:print(json.dumps({'event':'retiring_owned_old_features','done':index+1,'total':17981}),flush=True)
    for path in directories:
        fd=os.open(path,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    # Every protected feature that was warm in the reviewed plan must survive.
    for row in rows(plan.OUT/'protected_features.jsonl'):
        if not Path(row['path']).is_file():raise ValueError('A protected warm feature disappeared')
    _atomic_json(OUT/'completed.json',{'owner':'ROOT_budget18_unused_cache_retirement_v1','status':'completed',
                 'files_retired':len(checked),'allocated_bytes_retired':reclaimed,
                 'elapsed_seconds':time.time()-started,'plan_release_sha256':RELEASE_SHA,
                 'full_feature_hashes_sha256':sha_file(OUT/'full_feature_hashes.jsonl'),
                 'retired_journal_sha256':sha_file(OUT/'retired.jsonl'),
                 'protected_RGB':len(protected),'old_scientific_checkpoints_deleted':False,
                 'source_images_or_models_deleted':False,'disk_before':disk_before,
                 'disk_after':DiskBudget(ROOT,120).report()})
    print((OUT/'completed.json').read_text(),flush=True)


if __name__=='__main__':main()
