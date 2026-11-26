"""Local source snapshots for each training invocation, without account secrets.

This records filesystem source bytes at capture time. It cannot reconstruct old
code from an already-running process or prove the implementation of external
libraries; the independently recorded environment lock identifies those packages.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import uuid
import zipfile

from src.utils.research_storage import DiskBudget, _atomic_json, _fsync_directory


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_bytes(root):
    paths = sorted(set(root.glob('requirements*.txt')) |
                   set((root/'src').rglob('*.py')) |
                   set((root/'scripts').glob('*.py')))
    if not paths:
        raise ValueError('No research source files found for provenance')
    content = {}
    for path in paths:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Source must be a regular local file: {path}')
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Source changed while capturing provenance: {path}')
        content[str(path.relative_to(root))] = raw
    return content


def capture_source_snapshot(root, *, reserve_gb=120):
    root = Path(root).expanduser().resolve()
    content = _source_bytes(root)
    hashes = {name:_digest(raw) for name,raw in content.items()}
    code_sha = _digest(json.dumps(hashes,sort_keys=True).encode())
    directory = root/'research_results/source_snapshots'
    budget = DiskBudget(root,reserve_gb)
    budget.ensure_space(sum(map(len,content.values()))+1024**2)
    directory.mkdir(parents=True,exist_ok=True)
    archive = directory/(code_sha+'.zip')
    if archive.exists():
        if archive.is_symlink():raise ValueError('Source snapshot archive may not be a symlink')
        with zipfile.ZipFile(archive) as saved:
            if set(saved.namelist()) != set(content) or any(_digest(saved.read(name)) != hashes[name] for name in content):
                raise ValueError('Existing source snapshot does not match its content address')
    else:
        fd,temporary = tempfile.mkstemp(dir=directory,prefix='.source-',suffix='.zip')
        try:
            with os.fdopen(fd,'w+b') as handle:
                with zipfile.ZipFile(handle,'w',compression=zipfile.ZIP_DEFLATED) as saved:
                    for name,raw in content.items():
                        info = zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
                        info.compress_type = zipfile.ZIP_DEFLATED
                        info.create_system = 3
                        info.external_attr = 0o644 << 16
                        saved.writestr(info,raw)
                handle.flush();os.fsync(handle.fileno())
            # Identical captures have identical content identity. Atomic replace
            # leaves either complete ZIP valid even with concurrent CPU probes.
            os.replace(temporary,archive)
            _fsync_directory(directory)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    return {'code_sha256':code_sha,'source_sha256':hashes,
            'source_archive':str(archive.relative_to(root)),
            'source_archive_sha256':_digest(archive.read_bytes()),
            'capture_scope':'filesystem Python sources and requirements at invocation capture time'}


def write_run_provenance(root, run_dir, config, *, reserve_gb=120,
                         model_ref=None, vision_ref=None, cache_fingerprint=None):
    """Keep original provenance plus a separate receipt for every resume attempt.

    No environment variables, HF cache, token files, datasets or weights are
    copied into the source archive. The config must already be the resolved,
    credential-free training configuration used for checkpoint identity.
    """
    root = Path(root).expanduser().resolve()
    run = Path(run_dir).expanduser().resolve()
    if not run.is_relative_to(root):raise ValueError('Run provenance must stay inside the workspace')
    encoded = json.dumps(config,sort_keys=True,allow_nan=False).encode()
    source = capture_source_snapshot(root,reserve_gb=reserve_gb)
    try:
        commit = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,
                                text=True,timeout=5,check=True).stdout.strip()
    except (OSError,subprocess.SubprocessError):commit = None
    locks = {}
    for name in ('research_results/environment.lock.txt','research_results/environment.json'):
        path = root/name
        if path.is_file() and not path.is_symlink():locks[name] = _digest(path.read_bytes())
    attempt = uuid.uuid4().hex
    receipt = {'schema_version':1,'attempt_id':attempt,'pid':os.getpid(),
               'captured_utc':datetime.now(timezone.utc).isoformat(),
               'run':str(run.relative_to(root)),'resolved_config_sha256':_digest(encoded),
               'git_commit':commit,'environment_files_sha256':locks,
               'model_ref':model_ref or {},'vision_ref':vision_ref or {},'cache_fingerprint':cache_fingerprint,**source}
    run.mkdir(parents=True,exist_ok=True)
    attempts = run/'provenance_attempts'
    attempts.mkdir(exist_ok=True)
    budget = DiskBudget(root,reserve_gb)
    budget.ensure_space(1024**2)
    _atomic_json(attempts/(attempt+'.json'),receipt)
    original = run/'provenance.json'
    # Root trainers own their run directory; a single process executes per run.
    if not original.exists():_atomic_json(original,receipt)
    _atomic_json(run/'provenance_latest.json',receipt)
    return receipt
