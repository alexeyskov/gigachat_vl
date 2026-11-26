"""Common payload loading and whole-artifact identity for legacy/bundle files.

Legacy SHA256 stays exactly the SHA256 of the original state.pt bytes. Bundle
identity is domain-separated and binds its manifest plus every referenced model
and optimizer component. It must never be replaced by hashing the tiny manifest
alone. Member stats allow receipts to detect later object/retirement changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.utils.research_checkpoint_bundle import (
    MAGIC, MODEL_FILE, OPTIMIZER_FILE, inspect_checkpoint_bundle, load_checkpoint_bundle, _reader_lock,
)


def _primary(path):
    path=Path(path).expanduser()
    return (path/'state.pt' if path.is_dir() else path).resolve(strict=True)


def checkpoint_kind(path):
    path=_primary(path)
    with path.open('rb') as handle:
        return 'bundle_v1' if handle.read(len(MAGIC))==MAGIC else 'legacy_single_file'


def load_research_payload(path,*,map_location='cpu',mmap=True):
    """Autodetect and integrity-validate a legacy or split-bundle payload."""
    return load_checkpoint_bundle(path,map_location=map_location,mmap=mmap)


def _file_identity(path,progress=None,completed=0,total=None,verify_hash=True,expected_sha=None):
    before=path.stat();digest=hashlib.sha256()
    if verify_hash:
        with path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(8*1024**2),b''):
                digest.update(chunk);completed+=len(chunk)
                if progress is not None:progress(completed,total or before.st_size)
        actual=digest.hexdigest()
        if expected_sha is not None and actual!=expected_sha:
            raise ValueError(f'Bundle component SHA256 changed: {path.name}')
    else:
        if expected_sha is None:
            # The manifest is tiny and always read in full; only immutable large
            # components may use their already-attested descriptor digest.
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
        else:actual=expected_sha
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError(f'Checkpoint member changed while identifying: {path}')
    return {'path':str(path),'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':actual},completed


def _checkpoint_identity_unlocked(path,*,progress=None,verify_components=True):
    """Return whole-artifact content SHA256 and member binding metadata.

    ``verify_components=False`` is only for frequent status reports comparing a
    previously validated receipt: it hashes the small manifest, checks component
    ownership/format/size, and reuses descriptor hashes. It is not proof that the
    current component bytes were validated. Evaluation/restore/validation must
    use the default full verification.
    """
    primary=_primary(path);kind=checkpoint_kind(primary)
    if kind=='legacy_single_file':
        member,_=_file_identity(primary,progress=progress)
        return {'format':kind,'sha256':member['sha256'],'primary_path':str(primary),
                'primary_bytes':member['bytes'],'primary_mtime_ns':member['mtime_ns'],
                'total_bytes':member['bytes'],'components_fully_hashed':True,
                'members':[dict(role='state',relative_path='state.pt',**member)]}
    inspected=inspect_checkpoint_bundle(primary,verify_hashes=False)
    manifest=inspected['manifest'];directory=primary.parent
    paths=[('manifest','state.pt',primary,None),('model',MODEL_FILE,directory/MODEL_FILE,manifest['model']['sha256'])]
    if manifest['resumable_optimizer']:
        paths.append(('optimizer',OPTIMIZER_FILE,directory/OPTIMIZER_FILE,manifest['optimizer']['sha256']))
    total=sum(path.stat().st_size for _,_,path,_ in paths)
    members=[];done=0
    for role,relative,member_path,expected in paths:
        item,done=_file_identity(member_path,progress=progress,completed=done,total=total,
                                  verify_hash=verify_components or role=='manifest',expected_sha=expected)
        members.append(dict(role=role,relative_path=relative,**item))
    stable={'format':kind,'members':[{k:item[k] for k in ('role','relative_path','bytes','sha256')} for item in members]}
    digest=hashlib.sha256(b'RESEARCH_CHECKPOINT_ARTIFACT_V1\0'+json.dumps(stable,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'format':kind,'sha256':digest,'primary_path':str(primary),
            'primary_bytes':members[0]['bytes'],'primary_mtime_ns':members[0]['mtime_ns'],
            'total_bytes':total,'components_fully_hashed':bool(verify_components),'members':members}


def checkpoint_identity(path,*,progress=None,verify_components=True,manager_lock_held=False):
    """Bind complete artifact bytes; callers holding the manager lock may opt out.

    ``manager_lock_held=True`` requires an already-held manager shared/exclusive
    lock. It avoids acquiring a second descriptor and deadlocking an exclusive
    housekeeping operation against itself.
    """
    primary=_primary(path)
    if manager_lock_held:
        return _checkpoint_identity_unlocked(primary,progress=progress,verify_components=verify_components)
    with _reader_lock(primary):
        return _checkpoint_identity_unlocked(primary,progress=progress,verify_components=verify_components)


def same_checkpoint_identity(expected,current):
    """Compare exact content/member stats, ignoring whether status reread hashes."""
    keys=('format','sha256','primary_path','primary_bytes','primary_mtime_ns','total_bytes','members')
    return all(expected.get(key)==current.get(key) for key in keys)
