#!/usr/bin/env python3
"""Retire only this baseline's private downloaded shards after completed116."""
from __future__ import annotations
import hashlib,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'research_results/qwen25_native_int4_internal116_v1'


def retire():
    from src.utils.research_storage import _atomic_json
    from scripts.research_archival_stock_queue_v5 import scan_anchor_users
    if (OUT/'source_retired.json').exists():return json.loads((OUT/'source_retired.json').read_bytes())
    summary=json.loads((OUT/'summary.json').read_bytes());protocol=json.loads((OUT/'protocol.json').read_bytes())
    manifest_path=OUT/'source_content_hashes.json';manifest=json.loads(manifest_path.read_bytes())
    if summary.get('status')!='completed' or summary.get('records')!=116:raise ValueError('Complete116 baseline required before retirement')
    if (hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=protocol['source_content_manifest_sha256']
        or hashlib.sha256((OUT/'predictions.jsonl').read_bytes()).hexdigest()!=summary['predictions_sha256']):
        raise ValueError('Baseline source/prediction proof changed')
    source=ROOT/'cache/baselines/qwen25_native_int4_v1/source'
    owner=json.loads((source/'.baseline_owner.json').read_bytes())
    if owner!={k:manifest[k] for k in ('repo','revision')}|{'owner':'qwen25_native_int4_internal116_v1'}:
        raise ValueError('Private baseline source ownership differs')
    files=[row for row in manifest['files'] if not row['borrowed_shared_file']]
    if len(files)!=5 or any(not row['name'].endswith('.safetensors') for row in files):raise ValueError('Expected exact five private shards')
    paths={}
    for row in files:
        path=Path(row['path'])
        if path.parent!=source or path.is_symlink() or path.name!=row['name']:raise ValueError('Not an owned private baseline shard')
        stat=path.stat();identity=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if stat.st_nlink!=1 or stat.st_size!=row['bytes']:raise ValueError('Private shard acquired a shared link or changed size')
        paths[path]=dict(row,identity=identity)
    scans=scan_anchor_users({tuple(row['identity'][:2]) for row in paths.values()})
    before=shutil.disk_usage(ROOT).free
    _atomic_json(OUT/'source_retirement_intent.json',dict(owner=owner,files=list(paths.values()),all_UID_fd_maps_scan=scans,
        free_bytes_before=before,protected_shared_files_deleted=0,created_unix=time.time()))
    for path,row in paths.items():
        hasher=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*2**20),b''):hasher.update(chunk)
        stat=path.stat()
        if ((stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)!=tuple(row['identity'])
            or hasher.hexdigest()!=row['sha256']):raise ValueError('Private source identity/content changed before retirement')
        scan_anchor_users({tuple(row['identity'][:2])});path.unlink()
    result=dict(status='retired_completed_baseline_private_shards',files=5,
        retired_bytes=sum(row['bytes'] for row in files),protected_shared_files_deleted=0,
        free_bytes_before=before,free_bytes_after=shutil.disk_usage(ROOT).free,created_unix=time.time(),
        retained='predictions, metrics, protocols, full source content hashes and borrowed shared metadata')
    _atomic_json(OUT/'source_retired.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':retire()
