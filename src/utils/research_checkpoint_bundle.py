"""Optional split model/optimizer checkpoint bundles.

Write only into a fresh BestLastCheckpointManager-owned staging directory.
Model/RNG/config/scheduler and optimizer are separate lossless components. A tiny
versioned state.pt manifest binds exact relative names, SHA256, ownership, step,
provenance and bundle UUID. Native or streaming gzip/zstd component I/O reuses
research_checkpoint_io; compressed tensors are never mmap'ed.

Retiring an old BEST's optimizer is safe only while a different owned LAST has a
verified fully resumable bundle. The manifest is committed before unlinking the
optimizer: a crash can leave an orphan component, never a false resume promise.
No scientific model tensors are removed, quantized, or converted.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from src.utils.research_checkpoint_io import checkpoint_format, load_checkpoint, save_checkpoint
from src.utils.research_storage import (
    BestLastCheckpointManager, DiskBudget, _atomic_json, _fsync_directory, _tree_size,
)

MAGIC = b'RESEARCH_CHECKPOINT_BUNDLE_V1\n'
MODEL_FILE = 'model.state.pt'
OPTIMIZER_FILE = 'optimizer.state.pt'
MANIFEST_FILE = 'state.pt'


class CheckpointBundleError(ValueError):
    """Ownership, component integrity, provenance, or bundle contract failed."""


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def _owned_marker(directory):
    path = directory/'.research_checkpoint.json'
    if not path.is_file() or path.is_symlink():
        raise CheckpointBundleError('Bundle directory lacks its local manager ownership marker')
    info = json.loads(path.read_text())
    if not isinstance(info,dict) or not info.get('owner') or info.get('schema_version') != 1:
        raise CheckpointBundleError('Invalid checkpoint ownership marker')
    return info


def _manifest_path(path):
    path = Path(path).expanduser()
    if path.is_dir(): path = path/MANIFEST_FILE
    return path


def _read_manifest(path):
    if path.is_symlink():
        raise CheckpointBundleError('Bundle manifest must not be a symlink')
    with path.open('rb') as handle:
        header = handle.read(len(MAGIC))
        if header != MAGIC: return None
        # A component manifest should stay tiny even for a very large model.
        data = handle.read(1024**2+1)
        if len(data)>1024**2:
            raise CheckpointBundleError('Bundle manifest exceeds 1 MiB')
    try: manifest = json.loads(data)
    except ValueError as error: raise CheckpointBundleError('Malformed bundle manifest JSON') from error
    if not isinstance(manifest,dict) or manifest.get('format')!='research_checkpoint_bundle' or manifest.get('version')!=1:
        raise CheckpointBundleError('Unsupported checkpoint bundle format/version')
    return manifest


def _write_manifest(path,manifest):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    encoded = json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(encoded)>1024**2: raise CheckpointBundleError('Bundle manifest exceeds 1 MiB')
    try:
        with temporary.open('xb') as handle:
            handle.write(MAGIC);handle.write(encoded);handle.write(b'\n')
            handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
        _fsync_directory(path.parent)
    finally: temporary.unlink(missing_ok=True)


def _component(directory,descriptor,expected_name,verify_hash=True):
    if not isinstance(descriptor,dict) or descriptor.get('path')!=expected_name:
        raise CheckpointBundleError(f'Component must use exact owned relative name {expected_name}')
    # Fixed basenames reject absolute paths, ../ traversal, alternate encodings,
    # nested directories, and accidental references to another checkpoint.
    path = directory/descriptor['path']
    if not path.is_file() or path.is_symlink() or path.resolve().parent!=directory.resolve():
        raise CheckpointBundleError(f'Component missing or escapes its owned checkpoint: {expected_name}')
    digest = descriptor.get('sha256')
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise CheckpointBundleError('Invalid component SHA256 digest')
    if not isinstance(descriptor.get('bytes'),int) or descriptor['bytes']<0 or path.stat().st_size!=descriptor['bytes']:
        raise CheckpointBundleError(f'Component size mismatch: {expected_name}')
    if verify_hash and _sha256(path)!=digest:
        raise CheckpointBundleError(f'Component SHA256 mismatch: {expected_name}')
    fmt = checkpoint_format(path)
    if fmt['compression']!=descriptor.get('compression') or fmt['serialization']!=descriptor.get('serialization'):
        raise CheckpointBundleError(f'Component format mismatch: {expected_name}')
    return path


def _provenance(payload):
    return {key:payload.get(key) for key in ('model_ref','vision_ref','manifest_sha256','cache_fingerprint')}


def _tensor_storages(value):
    import torch
    found = set()
    def visit(item):
        if isinstance(item,torch.Tensor):
            storage=item.untyped_storage()
            if storage.nbytes():found.add((str(item.device),storage.data_ptr(),storage.nbytes()))
        elif isinstance(item,dict):
            for child in item.values():visit(child)
        elif isinstance(item,(tuple,list)):
            for child in item:visit(child)
    visit(value)
    return found


def _optimizer_schema(optimizer):
    """Validate native or allocator-aware optimizer envelopes without copying.

    The CPU-gradient optimizer deliberately wraps its native bitsandbytes state
    so restoration can rebuild paged allocations and pending CPU gradients.
    Preserve that envelope verbatim; treating its native subset as the complete
    optimizer would silently lose resume state.
    """
    if not isinstance(optimizer,dict):
        raise CheckpointBundleError('Resumable bundle requires an optimizer state dictionary')
    schema='torch_state_dict'
    native=optimizer
    if 'research_optimizer' in optimizer or 'native_optimizer' in optimizer:
        metadata=optimizer.get('research_optimizer')
        if not isinstance(metadata,dict) or metadata.get('version')!=1 or metadata.get('backend')!='cpu_gradient_paged_adamw' or metadata.get('optim_bits') not in (8,32):
            raise CheckpointBundleError('Unsupported allocator-aware optimizer envelope')
        native=optimizer.get('native_optimizer')
        schema='cpu_gradient_paged_adamw_v1'
    if not isinstance(native,dict) or not isinstance(native.get('state'),dict) or not isinstance(native.get('param_groups'),list) or not native['param_groups']:
        raise CheckpointBundleError('Resumable bundle requires optimizer.state_dict with parameter groups')
    groups=native['param_groups']
    if any(not isinstance(group,dict) or not isinstance(group.get('params'),list) for group in groups):
        raise CheckpointBundleError('Optimizer parameter groups lack serialized parameter indices')
    if schema!='torch_state_dict':
        import torch
        indices={index for group in groups for index in group['params']}
        shapes=metadata.get('parameter_shapes')
        gradients=optimizer.get('cpu_gradients')
        if not isinstance(shapes,dict) or set(shapes)!=indices or any(not isinstance(shape,(list,tuple)) or any(not isinstance(size,int) or size<0 for size in shape) for shape in shapes.values()):
            raise CheckpointBundleError('Allocator-aware optimizer parameter shapes differ from groups')
        if not set(native['state']).issubset(indices) or not isinstance(gradients,dict):
            raise CheckpointBundleError('Allocator-aware optimizer has unknown state/gradient indices')
        if any(index not in indices or not isinstance(value,torch.Tensor) or tuple(value.shape)!=tuple(shapes[index]) for index,value in gradients.items()):
            raise CheckpointBundleError('Allocator-aware optimizer pending CPU gradients differ from parameter shapes')
    return schema,len(groups)


def _verify_optimizer_manifest(optimizer,manifest):
    schema,count=_optimizer_schema(optimizer)
    if count!=manifest['optimizer_param_group_count'] or schema!=manifest.get('optimizer_schema','torch_state_dict'):
        raise CheckpointBundleError('Optimizer component schema/parameter-group mismatch')


def save_checkpoint_bundle(payload,directory,compression='none',optimizer_compression=None):
    """Write a fresh owned bundle with no full uncompressed staging component.

    Cross-component tensor-storage aliases are refused, because separate files
    cannot preserve such aliases. Standard optimizer.state_dict() stores moment
    tensors independently; model-internal tied tensors/views remain intact.
    """
    if not isinstance(payload,dict) or not isinstance(payload.get('model'),dict) or not payload['model']:
        raise CheckpointBundleError('Bundle payload requires a nonempty model state dictionary')
    directory=Path(directory).expanduser().resolve()
    marker=_owned_marker(directory)
    if marker.get('step')!=payload.get('step'):
        raise CheckpointBundleError('Payload step differs from manager checkpoint step')
    if any((directory/name).exists() or (directory/name).is_symlink() for name in (MODEL_FILE,OPTIMIZER_FILE,MANIFEST_FILE)):
        raise CheckpointBundleError('Bundle writer requires a fresh owned staging directory')
    model_payload=dict(payload)
    optimizer=model_payload.pop('optimizer',None)
    resumable=optimizer is not None
    optimizer_schema=None;optimizer_groups=0
    if resumable:
        optimizer_schema,optimizer_groups=_optimizer_schema(optimizer)
        if payload.get('resumable_optimizer') is False:
            raise CheckpointBundleError('Payload contains optimizer but explicitly says nonresumable')
        if _tensor_storages(model_payload) & _tensor_storages(optimizer):
            raise CheckpointBundleError('Cross-component shared tensor storage requires legacy single-file serialization')
    elif payload.get('resumable_optimizer') is not False:
        raise CheckpointBundleError('Missing optimizer must be explicitly marked resumable_optimizer=false')
    bundle_id=uuid.uuid4().hex
    common={'bundle_id':bundle_id,'checkpoint_owner':marker['owner'],'checkpoint_step':marker['step']}
    manifest={'format':'research_checkpoint_bundle','version':1,**common,
              'provenance':_provenance(payload),'resumable_optimizer':resumable,
              'training_completed':bool(payload.get('training_completed',False)),
              'optimizer_param_group_count':optimizer_groups,'optimizer_schema':optimizer_schema,
              'created_unix':time.time(),'model':None,'optimizer':None}
    # Validate JSON provenance before creating large component files.
    json.dumps(manifest,allow_nan=False)
    made=[]
    try:
        for name,value,codec,key in [
            (MODEL_FILE,{'payload':model_payload,**common},compression,'model'),
            (OPTIMIZER_FILE,{'optimizer':optimizer,**common},optimizer_compression or compression,'optimizer')]:
            if key=='optimizer' and not resumable:continue
            path=directory/name
            info=save_checkpoint(value,path,compression=codec)
            made.append(path)
            manifest[key]={'path':name,'sha256':_sha256(path),'bytes':path.stat().st_size,
                           'compression':info['compression'],'serialization':info['serialization']}
        _write_manifest(directory/MANIFEST_FILE,manifest)
        return manifest
    except BaseException:
        # Only files created by this invocation in the fresh owned directory.
        for path in made:path.unlink(missing_ok=True)
        raise


def inspect_checkpoint_bundle(path,verify_hashes=True):
    """Validate manifest/component integrity without loading model/Adam tensors."""
    path=_manifest_path(path)
    manifest=_read_manifest(path)
    if manifest is None:return {'kind':'legacy_single_file','path':str(path)}
    directory=path.parent.resolve()
    marker=_owned_marker(directory)
    if manifest.get('checkpoint_owner')!=marker['owner'] or manifest.get('checkpoint_step')!=marker['step']:
        raise CheckpointBundleError('Bundle manifest owner/step differs from manager ownership')
    if not isinstance(manifest.get('bundle_id'),str) or len(manifest['bundle_id'])!=32:
        raise CheckpointBundleError('Invalid bundle UUID')
    _component(directory,manifest.get('model'),MODEL_FILE,verify_hashes)
    if manifest.get('resumable_optimizer') is True:
        _component(directory,manifest.get('optimizer'),OPTIMIZER_FILE,verify_hashes)
        if not isinstance(manifest.get('optimizer_param_group_count'),int) or manifest['optimizer_param_group_count']<1:
            raise CheckpointBundleError('Resumable manifest lacks validated optimizer parameter groups')
        if manifest.get('optimizer_schema','torch_state_dict') not in ('torch_state_dict','cpu_gradient_paged_adamw_v1'):
            raise CheckpointBundleError('Unsupported optimizer component schema')
    elif manifest.get('resumable_optimizer') is False:
        if manifest.get('optimizer') is not None:
            raise CheckpointBundleError('Nonresumable manifest still references an optimizer')
    else:raise CheckpointBundleError('Manifest lacks explicit optimizer resumability')
    return {'kind':'bundle','path':str(path),'manifest':manifest,'marker':marker,
            'manifest_sha256':_sha256(path),'fully_resumable':manifest['resumable_optimizer'] is True}


@contextmanager
def _reader_lock(path):
    directory=path.parent.resolve()
    # Normal manager objects live immediately below .checkpoint_objects. A
    # staging-directory load does not acquire another manager lock recursively.
    run=directory.parent.parent if directory.parent.name=='.checkpoint_objects' else None
    if run is not None and (run/'.checkpoint_manager.json').is_file():
        with (run/'.checkpoint.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_SH)
            try:yield
            finally:fcntl.flock(handle,fcntl.LOCK_UN)
    else:yield


def load_checkpoint_bundle(path,map_location='cpu',mmap=True):
    """Reconstruct exact model/optimizer/RNG payload, or load a legacy file."""
    path=_manifest_path(path)
    with _reader_lock(path):
        inspected=inspect_checkpoint_bundle(path)
        if inspected['kind']=='legacy_single_file':
            return load_checkpoint(path,map_location=map_location,mmap=mmap)
        manifest=inspected['manifest'];directory=path.parent.resolve()
        model_component=load_checkpoint(directory/MODEL_FILE,map_location=map_location,mmap=mmap)
        common={key:manifest[key] for key in ('bundle_id','checkpoint_owner','checkpoint_step')}
        if not isinstance(model_component,dict) or any(model_component.get(k)!=v for k,v in common.items()):
            raise CheckpointBundleError('Model component bundle provenance mismatch')
        payload=model_component.get('payload')
        if not isinstance(payload,dict) or not isinstance(payload.get('model'),dict) or _provenance(payload)!=manifest['provenance'] or payload.get('step')!=manifest['checkpoint_step']:
            raise CheckpointBundleError('Model payload manifest/provenance mismatch')
        if manifest['resumable_optimizer']:
            component=load_checkpoint(directory/OPTIMIZER_FILE,map_location=map_location,mmap=mmap)
            if not isinstance(component,dict) or any(component.get(k)!=v for k,v in common.items()):
                raise CheckpointBundleError('Optimizer component bundle provenance mismatch')
            optimizer=component.get('optimizer')
            _verify_optimizer_manifest(optimizer,manifest)
            payload['optimizer']=optimizer
        payload['resumable_optimizer']=manifest['resumable_optimizer']
        if manifest.get('training_completed'):payload['training_completed']=True
        if manifest.get('optimizer_retirement'):
            payload['optimizer_retirement']=manifest['optimizer_retirement']
        return payload


def _verify_component_structure_without_materializing(directory,manifest):
    """Validate wrapper provenance and optimizer schema using fake tensor storages.

    SHA256 already verifies real on-disk bytes. FakeTensorMode reconstructs the
    complete pickle structure without allocating a second full model/optimizer
    in CPU RAM. Compressed streams are still drained for codec integrity by the
    I/O helper; no raw compressed storage is mmap'ed.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    common={key:manifest[key] for key in ('bundle_id','checkpoint_owner','checkpoint_step')}
    with FakeTensorMode():
        model=load_checkpoint(directory/MODEL_FILE,map_location='cpu',mmap=True)
        if not isinstance(model,dict) or any(model.get(k)!=v for k,v in common.items()):
            raise CheckpointBundleError('Model component bundle provenance mismatch')
        payload=model.get('payload')
        if not isinstance(payload,dict) or not isinstance(payload.get('model'),dict) or not payload['model'] or payload.get('step')!=manifest['checkpoint_step'] or _provenance(payload)!=manifest['provenance']:
            raise CheckpointBundleError('Model payload manifest/provenance mismatch')
        if manifest['resumable_optimizer']:
            optimizer=load_checkpoint(directory/OPTIMIZER_FILE,map_location='cpu',mmap=True)
            if not isinstance(optimizer,dict) or any(optimizer.get(k)!=v for k,v in common.items()):
                raise CheckpointBundleError('Optimizer component bundle provenance mismatch')
            state=optimizer.get('optimizer')
            _verify_optimizer_manifest(state,manifest)


def retire_best_optimizer(run_dir):
    """Retire only a distinct old BEST optimizer after validating resumable LAST.

    SHA-verified component files and fake-tensor structural reconstruction verify
    optimizer schema/provenance without materializing another full optimizer in
    CPU RAM. Normal restoration additionally reconstructs all actual tensors.
    """
    run=Path(run_dir).expanduser().resolve()
    info_path=run/'.checkpoint_manager.json'
    if not info_path.is_file():raise CheckpointBundleError('Run has no existing checkpoint manager')
    manager_info=json.loads(info_path.read_text())
    manager=BestLastCheckpointManager(run,DiskBudget(run,0),mode=manager_info['mode'])
    with manager._lock():
        best=manager._resolve_owned_pointer('best');last=manager._resolve_owned_pointer('last')
        if best is None or last is None:return {'status':'skipped_missing_role'}
        if best==last:return {'status':'skipped_shared_best_last','model_weights_unchanged':True}
        best_inspection=inspect_checkpoint_bundle(best/MANIFEST_FILE)
        last_inspection=inspect_checkpoint_bundle(last/MANIFEST_FILE)
        if best_inspection['kind']!='bundle' or last_inspection['kind']!='bundle':
            return {'status':'skipped_legacy_single_file','model_weights_unchanged':True}
        if not last_inspection['fully_resumable']:
            raise CheckpointBundleError('Cannot retire BEST optimizer: LAST is not fully resumable')
        _verify_component_structure_without_materializing(best,best_inspection['manifest'])
        _verify_component_structure_without_materializing(last,last_inspection['manifest'])
        manifest=best_inspection['manifest']
        old_hash=best_inspection['manifest_sha256']
        if manifest['resumable_optimizer']:
            descriptor=manifest['optimizer']
            _component(best,descriptor,OPTIMIZER_FILE)
            manifest=dict(manifest)
            manifest['optimizer']=None
            manifest['resumable_optimizer']=False
            manifest['optimizer_retirement']={'reason':'best_is_no_longer_last','retired_unix':time.time(),
                'verified_last_object':last.name,'verified_last_manifest_sha256':last_inspection['manifest_sha256'],
                'previous_optimizer':descriptor,'previous_manifest_sha256':old_hash}
            # Commit point first: subsequent readers never expect the soon-to-be
            # deleted file. A crash before unlink merely leaves reclaimable data.
            _write_manifest(best/MANIFEST_FILE,manifest)
        else:
            retirement=manifest.get('optimizer_retirement')
            if not isinstance(retirement,dict):return {'status':'skipped_nonresumable_without_retirement','model_weights_unchanged':True}
            descriptor=retirement['previous_optimizer']
        optimizer_path=best/OPTIMIZER_FILE
        reclaimed=0
        if optimizer_path.exists() or optimizer_path.is_symlink():
            _component(best,descriptor,OPTIMIZER_FILE)
            reclaimed=optimizer_path.stat().st_size
            optimizer_path.unlink()
            _fsync_directory(best)
        marker=_owned_marker(best)
        marker.update(resumable_optimizer=False,optimizer_retired=True,
                      manifest_sha256=_sha256(best/MANIFEST_FILE),size_bytes=_tree_size(best))
        _atomic_json(best/'.research_checkpoint.json',marker)
        return {'status':'retired' if reclaimed else 'already_retired','best_object':best.name,'last_object':last.name,
                'optimizer_bytes_reclaimed':reclaimed,'last_fully_resumable':True,'best_resumable_optimizer':False,
                'model_component_sha256_unchanged':manifest['model']['sha256'],
                'model_weights_unchanged':True,'invalidates_prior_roundtrip_receipt':True,
                'best_manifest_sha256':_sha256(best/MANIFEST_FILE)}


def checkpoint_bundle_capacity(serialized_model_bytes,serialized_optimizer_bytes):
    """Conservative component-file peak arithmetic, not a hardware fit claim.

    After every role transition retires a distinct BEST optimizer, the next
    save holds BEST model + LAST resume bundle + new resume staging bundle.
    Input sizes must be actual serialized component sizes for the chosen codec;
    raw parameter counts or a small compression probe are not substitutes.
    """
    if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in (serialized_model_bytes,serialized_optimizer_bytes)):
        raise ValueError('Serialized component sizes must be nonnegative integers')
    m,o=serialized_model_bytes,serialized_optimizer_bytes
    overhead=3*(1024**2+16*1024)  # bounded manifests and ownership/housekeeping allowance
    return {'model_bytes':m,'optimizer_bytes':o,'one_resumable_bundle_bytes':m+o,
            'legacy_three_resume_payloads_bytes':3*(m+o),
            'bundle_best_model_plus_last_and_staging_bytes':m+2*(m+o)+overhead,
            'manifest_and_owner_allowance_bytes':overhead,
            'assumes_retirement_before_next_save':True,
            'does_not_include':['120 GiB disk reserve','source model download','images','vision cache','unrelated runs'],
            'does_not_claim_current_disk_fit':True}


def compact_completed_bundle(run_dir,reserve_gb=120,dry_run=False):
    """Drop optimizer components only after a completed SFT, without model rewrite."""
    run=Path(run_dir).expanduser().resolve()
    config=json.loads((run/'resolved_config.json').read_text())
    completion=json.loads((run/'completed.json').read_text())
    if config.get('stage')!='SFT' or completion.get('steps',0)<1 or completion.get('status') in {'failed','interrupted','incomplete'}:
        raise CheckpointBundleError('Bundle compaction requires a successfully completed SFT stage')
    info=json.loads((run/'.checkpoint_manager.json').read_text())
    manager=BestLastCheckpointManager(run,DiskBudget(run,reserve_gb),mode=info['mode'])
    report={'schema_version':1,'run_dir':str(run),'training_completed':True,'resumable_optimizer':False,
            'checkpoint_format':'bundle','checkpoints':[],'started':time.time()}
    with manager._lock():
        best=manager._resolve_owned_pointer('best');last=manager._resolve_owned_pointer('last')
        if last is None or _owned_marker(last)['step']!=completion['steps']:
            raise CheckpointBundleError('Completed SFT does not match its retained LAST checkpoint')
        checked=[]
        for target in sorted({best,last}-{None}):
            inspection=inspect_checkpoint_bundle(target/MANIFEST_FILE)
            if inspection['kind']!='bundle':raise CheckpointBundleError('Bundle compaction does not rewrite legacy files')
            manifest=inspection['manifest'];marker=_owned_marker(target)
            if marker.get('created_unix',0)>(run/'completed.json').stat().st_mtime+1:
                raise CheckpointBundleError('A newer checkpoint appeared after SFT completion')
            _verify_component_structure_without_materializing(target,manifest)
            checked.append((target,inspection,manifest,marker))
        for target,inspection,manifest,marker in checked:
            descriptor=manifest.get('optimizer')
            if descriptor is None:
                descriptor=(manifest.get('optimizer_retirement') or {}).get('previous_optimizer')
            before=_tree_size(target)
            already_completed=manifest.get('training_completed') is True and not manifest['resumable_optimizer']
            manifest=dict(manifest)
            if manifest['resumable_optimizer']:
                manifest['optimizer']=None;manifest['resumable_optimizer']=False
                manifest['optimizer_retirement']={'reason':'completed_sft_stage','retired_unix':time.time(),
                    'previous_optimizer':descriptor,'previous_manifest_sha256':inspection['manifest_sha256']}
            manifest['training_completed']=True
            manager.budget.ensure_space(1024**2)
            if dry_run:
                report['checkpoints'].append({'old_object':target.name,'before_bytes':before,'optimizer_bytes_to_remove':0 if descriptor is None else descriptor['bytes']})
                continue
            if not already_completed:_write_manifest(target/MANIFEST_FILE,manifest)
            optimizer_path=target/OPTIMIZER_FILE
            if optimizer_path.exists() or optimizer_path.is_symlink():
                if descriptor is None:raise CheckpointBundleError('Unknown optimizer component cannot be deleted')
                _component(target,descriptor,OPTIMIZER_FILE)
                optimizer_path.unlink();_fsync_directory(target)
            marker.update(resumable_optimizer=False,optimizer_retired=True,training_completed=True,
                          manifest_sha256=_sha256(target/MANIFEST_FILE),size_bytes=_tree_size(target))
            _atomic_json(target/'.research_checkpoint.json',marker)
            report['checkpoints'].append({'old_object':target.name,'new_object':target.name,
                'before_bytes':before,'after_bytes':_tree_size(target),
                'model_component_sha256_unchanged':manifest['model']['sha256'],
                'invalidates_prior_roundtrip_receipt':True})
        if dry_run:
            report['status']='validated_dry_run'
            return report
        report['logical_bytes_reclaimed']=sum(row['before_bytes']-row['after_bytes'] for row in report['checkpoints'])
        report.update(status='completed',finished=time.time(),disk=manager.budget.report())
        _atomic_json(run/'checkpoint_compaction.json',report)
        return report
