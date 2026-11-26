"""Prospective completed-Small archival publication, deliberately gate closed.

No import constructs a manager, reads model values, or changes a checkpoint.
Only a measured actual4f probe plus a separately produced ROOT GPU parity proof
can enable publication. Trusted local payloads only; this is not a pickle sandbox.
"""
from __future__ import annotations
import contextlib
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import uuid

from src.utils import research_trained_delta_probe as probe

ROOT=Path(__file__).resolve().parents[2]
OWNER='prospective_completed_small_archival_v1'
ACTIVATION_PROTOCOL='actual_trained_small_archival_activation_v1'
GPU_PROTOCOL='actual_trained_small_archive_gpu_parity_v1'
FULL_RECIPES=('4f','2a','2b','2c','3a','3b','3c')
ACTIVATION=ROOT/'configs/research/trained_small_archival_activation.json'
REPORT='checkpoint_archival.json'
JOURNAL='checkpoint_archival_transaction.json'
STAGING_LIMIT=24*2**30
METADATA_LIMIT=64*2**20
NORMAL_LIMIT=2*2**30
MIN_SAVING_BYTES=2**20
CODE_FILES=('src/utils/research_checkpoint_archival.py','src/utils/research_trained_delta_probe_v2.py',
    'src/utils/research_trained_delta_probe.py','src/utils/research_lossless_delta.py',
    'src/utils/research_stream_delta.py','src/utils/research_checkpoint_io.py',
    'src/utils/research_checkpoint_compat.py','src/utils/research_checkpoint_bundle.py',
    'src/utils/research_storage.py')
_IMPORTED_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ArchivalGateClosed(ValueError):pass
class NoBenefit(ValueError):pass


def read(path):return probe._json(path)
def sha(path):return probe.file_sha(path)
def canonical(value):return probe.canonical(value)


def code_binding():
    result={name:sha(ROOT/name) for name in CODE_FILES}
    if result['src/utils/research_checkpoint_archival.py']!=_IMPORTED_SOURCE_SHA256:
        raise ValueError('Archival helper source changed after import')
    return result


def activation_evidence(path=None):
    """No boolean command-line bypass. Missing real calibration stays closed."""
    path=Path(path) if path is not None else ACTIVATION
    try:
        activation=read(path)
        if (activation.get('schema_version')!=1 or activation.get('protocol')!=ACTIVATION_PROTOCOL or
            activation.get('status')!='passed' or activation.get('code_sha256')!=code_binding()):
            raise ArchivalGateClosed('Actual trained calibration/codec activation proof is not passed')
        p=Path(activation['trained_probe_report']).resolve(strict=True)
        g=Path(activation['root_gpu_parity_report']).resolve(strict=True)
        if sha(p)!=activation.get('trained_probe_report_sha256') or sha(g)!=activation.get('root_gpu_parity_report_sha256'):
            raise ArchivalGateClosed('Actual trained CPU/GPU evidence SHA differs')
        cpu=read(p);gpu=read(g)
        if (cpu.get('protocol')!=probe.OWNER or cpu.get('status')!='verified_actual_trained_derived_copy_only' or
            cpu.get('decoded_model_bytes')!=probe.BASE_ELEMENTS*4 or
            cpu.get('all_payload_fields_and_backing_bits_exact') is not True or
            cpu.get('scheduler_rng_config_compaction_metadata_preserved') is not True or
            cpu.get('source_modified') is not False or cpu.get('manager_roles_modified') is not False or
            cpu.get('source_payload_sha256')!=cpu.get('restored_payload_sha256') or
            not 0<cpu.get('archive_bytes',0)<cpu.get('source_stat',{}).get('st_size',0)):
            raise ArchivalGateClosed('Actual full trained4f CPU equality AND measured benefit are required')
        from src.utils import research_trained_delta_probe_v2 as adapter
        if cpu.get('metadata_adapter',{}).get('source_sha256')!=adapter.source_binding():
            raise ArchivalGateClosed('Actual trained CPU evidence must bind the reviewed HF metadata v2 adapter')
        config=read(Path(cpu['run'])/'resolved_config.json')
        if config.get('model')!=probe.REPO or config.get('stage')!='SFT' or config.get('recipe')!='4f':
            raise ArchivalGateClosed('Calibration must use actual full4f Small SFT')
        required={'schema_version':1,'protocol':GPU_PROTOCOL,'status':'passed','producer':'ROOT',
            'device':'cuda','run':cpu['run'],'source_state_sha256':cpu['source_sha256'],
            'archive_sha256':cpu['archive_sha256'],'source_payload_sha256':cpu['source_payload_sha256'],
            'restored_payload_sha256':cpu['restored_payload_sha256'],
            'all_model_tensors_exact':True,'generated_token_ids_exact':True,'selection_replay_exact':True,
            'source_authoritative':True,'publication_performed':False}
        if any(gpu.get(key)!=value for key,value in required.items()) or type(gpu.get('generation_records')) is not int or gpu['generation_records']<64:
            raise ArchivalGateClosed('ROOT full reconstruction/generation/internal-selection GPU parity is absent')
        if gpu.get('code_sha256')!=code_binding():raise ArchivalGateClosed('GPU parity decoder/loader source changed')
        return {'path':str(path.resolve()),'sha256':sha(path),'cpu_report_sha256':sha(p),
            'gpu_report_sha256':sha(g),'calibration_source_sha256':cpu['source_sha256'],
            'calibration_archive_sha256':cpu['archive_sha256'],'code_sha256':code_binding()}
    except (OSError,KeyError,TypeError,ValueError) as error:
        if isinstance(error,ArchivalGateClosed):raise
        raise ArchivalGateClosed('Actual trained4f derived probe plus ROOT GPU parity activation is required: '+str(error)) from error


def _roles(run,owner):
    objects=run/'.checkpoint_objects'
    if objects.is_symlink() or not objects.is_dir():raise ArchivalGateClosed('Local manager objects required')
    result={}
    for role in ('best','last'):
        pointer=run/role
        if not pointer.is_symlink():raise ArchivalGateClosed('Both retained roles must be manager-owned symlinks')
        target=pointer.resolve(strict=True)
        if target.parent!=objects or not target.name.startswith('object-') or target.is_symlink():
            raise ArchivalGateClosed('Retained role escaped manager objects')
        marker=read(target/'.research_checkpoint.json')
        if (marker.get('owner')!=owner or marker.get('training_completed') is not True or
                marker.get('resumable_optimizer') is not False or marker.get('optimizer_state_removed') is not True):
            raise ArchivalGateClosed('Retained state must already be owned/completed/optimizer-compacted')
        state=target/'state.pt'
        if state.is_symlink() or not state.is_file() or not 0<state.stat().st_size<=STAGING_LIMIT:
            raise ArchivalGateClosed('Local bounded retained payload required')
        result[role]={'object':target.name,'marker':marker,'marker_sha256':sha(target/'.research_checkpoint.json'),
                      'state':str(state),'stat':probe.stat(state)}
    return result


def eligibility(run,*,allow_archived=True):
    """Metadata only, no locks/files/model values. No evaluation may be relabelled."""
    run=Path(run).resolve()
    if any(word in run.name.lower() for word in ('smoke','technical','fastbench')):
        raise ArchivalGateClosed('Technical runs are outside scientific archival publication')
    config=read(run/'resolved_config.json');done=read(run/'completed.json');manager=read(run/'.checkpoint_manager.json')
    compact=read(run/'checkpoint_compaction.json')
    output=Path(config.get('output',''));output=(output if output.is_absolute() else ROOT/output).resolve()
    if (config.get('model')!=probe.REPO or config.get('stage')!='SFT' or config.get('recipe') not in FULL_RECIPES or
        config.get('checkpoint_format','legacy')!='legacy' or output!=run):
        raise ArchivalGateClosed('Only completed full Small SFT raw single-file checkpoints are supported')
    if (type(done.get('steps')) is not int or done['steps']<1 or done.get('status') in ('failed','incomplete','interrupted') or
        manager.get('mode') not in ('min','max') or not manager.get('owner') or
        compact.get('status')!='completed' or compact.get('training_completed') is not True or
        compact.get('resumable_optimizer') is not False or Path(compact.get('run_dir','')).resolve()!=run):
        raise ArchivalGateClosed('Durable successful training and optimizer compaction required')
    if not (run/'.checkpoint.lock').is_file() or (run/'.checkpoint.lock').is_symlink():
        raise ArchivalGateClosed('Existing local manager lock required')
    roles=_roles(run,manager['owner'])
    if roles['last']['marker'].get('step')!=done['steps']:raise ArchivalGateClosed('LAST/completion step differs')
    for item in roles.values():
        marker=item['marker'];lineage=marker.get('checkpoint_archival')
        if marker.get('created_unix',math.inf)>(run/'completed.json').stat().st_mtime+1:
            raise ArchivalGateClosed('A later training object appeared after completion')
        if lineage:
            if not allow_archived or lineage.get('protocol')!=OWNER:raise ArchivalGateClosed('Unknown/unsupported archival lineage')
        else:
            if not any(row.get('new_object')==item['object'] for row in compact.get('checkpoints',[])):
                raise ArchivalGateClosed('Raw retained object lacks compaction lineage')
            if set(p.name for p in Path(item['state']).parent.iterdir())!={'state.pt','.research_checkpoint.json'}:
                raise ArchivalGateClosed('Unknown source-object files are preserved, not deleted')
    metadata=('resolved_config.json','completed.json','.checkpoint_manager.json','checkpoint_compaction.json','trainable_parameters.json')
    return {'run':str(run),'config':config,'completion':done,'manager':manager,'roles':roles,
            'metadata_sha256':{name:sha(run/name) for name in metadata}}


def _no_evaluation(run):
    if any(os.path.lexists(run/name) for name in ('evaluation','checkpoint_roundtrip.json')):
        raise ArchivalGateClosed('First ordinary evaluation/replay already began; raw source remains authoritative, probe only')
    # No signal, GPU query or training import. A CLI preparing inputs may not yet
    # have made its evaluation directory, so check known run consumers as well.
    for procdir in Path('/proc').iterdir():
        if not procdir.name.isdecimal():continue
        try:argv=[os.fsdecode(p) for p in (procdir/'cmdline').read_bytes().split(b'\0') if p]
        except OSError:continue
        if not any(Path(arg).name in ('run_research_stage.py','eval_research_checkpoint.py',
            'eval_research_checkpoint_parallel_b1.py','validate_research_checkpoint.py') for arg in argv):continue
        for flag in ('--run','--output'):
            if flag in argv and argv.index(flag)+1<len(argv):
                value=Path(argv[argv.index(flag)+1]);value=(value if value.is_absolute() else ROOT/value).resolve()
                if value==run:raise ArchivalGateClosed('An owned training/evaluation consumer is live; publication refused')


def _anchor_headers(snapshot):
    """Pinned HF metadata may be cache symlinks; run/transaction JSON may not."""
    from src.utils import research_trained_delta_probe_v2 as adapter
    census=adapter.anchor_census(snapshot)
    return census['shapes'],read((Path(snapshot)/'config.json').resolve(strict=True))


def _full_shapes(snapshot,recipe,vision_dim=None):
    base,cfg=_anchor_headers(snapshot);expected=dict(base)
    if recipe!='4f':
        h=int(cfg['hidden_size']);ffn=int(cfg['moe_ffn_hidden_size']);old=int(cfg['moe_num_primary_experts'])
        for layer in range(int(cfg['num_hidden_layers'])):
            prefix=f'llm.model.layers.{layer}.block_sparse_moe.'
            if base.get(prefix+'primary_router.weight')!=[old,h]:raise ValueError('Pinned native router shape differs')
            if recipe[0]=='2':expected[prefix+'primary_router.weight']=[old+2,h]
            for index in range(2):
                for name,shape in (('up',[ffn,h]),('gate',[ffn,h]),('down',[h,ffn])):
                    expected[prefix+f'visual_experts.{index}.{name}.weight']=shape
            if recipe[0]=='3':
                for name,out in (('router_new',2),('router_mass',1)):
                    for suffix,shape in (('up.weight',[128,h+old]),('up.bias',[128]),('down.weight',[out,128]),('down.bias',[out])):
                        expected[prefix+name+'.'+suffix]=shape
            if recipe[1]=='c':expected[prefix+'modality_gate_logits']=[old+2,2]
    return base,expected


def _census(payload,plan,snapshot):
    import torch
    if (not isinstance(payload,dict) or not isinstance(payload.get('model'),dict) or 'optimizer' in payload or
        payload.get('training_completed') is not True or payload.get('resumable_optimizer') is not False or
        payload.get('model_ref')!={'repo':probe.REPO,'revision':probe.REVISION} or payload.get('config')!=plan['config']):
        raise ValueError('Complete already-compacted pinned Small payload required')
    base,expected=_full_shapes(snapshot,plan['config']['recipe']);model=payload['model']
    llm={name:value for name,value in model.items() if name.startswith('llm.')}
    if set(llm)!=set(expected):raise ValueError('Incomplete/extra full model keys; no adapter or missing novel state')
    declared=read(Path(plan['run'])/'trainable_parameters.json')
    if declared!={name:math.prod(shape) for name,shape in expected.items()}:
        raise ValueError('Trainable full-parameter census differs from the exact architecture')
    hidden=read((Path(snapshot)/'config.json').resolve(strict=True))['hidden_size']
    projector={name:value for name,value in model.items() if not name.startswith('llm.')}
    if set(projector)!={'projector.0.weight','projector.0.bias','projector.2.weight','projector.2.bias'}:
        raise ValueError('Complete frozen MLP projector must be retained')
    if any(not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.device.type!='cpu' for value in projector.values()):
        raise ValueError('Frozen projector must retain its full CPU FP32 tensors')
    first=list(projector['projector.0.weight'].shape)
    if (len(first)!=2 or first[0]!=hidden or not 0<first[1]<=8192 or list(projector['projector.0.bias'].shape)!=[hidden] or
        list(projector['projector.2.weight'].shape)!=[hidden,hidden] or list(projector['projector.2.bias'].shape)!=[hidden]):
        raise ValueError('MLP projector shapes differ')
    selected={name:shape for name,shape in base.items() if expected.get(name)==shape}
    for name,shape in expected.items():
        value=model[name]
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or list(value.shape)!=shape:
            raise ValueError('Full FP32 model shape/dtype differs: '+name)
    uses={};normal={}
    def visit(value,encoded=False):
        if isinstance(value,torch.Tensor):
            storage=value.untyped_storage();key=(storage._cdata,storage.nbytes());uses.setdefault(key,[]).append(encoded)
            if not encoded:normal[key]=storage.nbytes()
        elif isinstance(value,dict):
            for key,child in value.items():visit(child,value is model and key in selected)
        elif isinstance(value,(list,tuple)):
            for child in value:visit(child)
    visit(payload)
    for name in selected:
        tensor=model[name];storage=tensor.untyped_storage();key=(storage._cdata,storage.nbytes())
        if (tensor.requires_grad or not tensor.is_contiguous() or tensor.storage_offset() or
            storage.nbytes()!=tensor.numel()*4 or len(uses[key])!=1):raise ValueError('Encoded original aliases/views are refused')
    largest=max(math.prod(shape)*4 for shape in selected.values())
    if sum(normal.values())>NORMAL_LIMIT or largest>2**30:raise ValueError('Normal storage/encoded tensor bound exceeded')
    if payload.get('step') not in [item['marker']['step'] for item in plan['roles'].values()]:raise ValueError('Payload retained step differs')
    return {'selected':selected,'original_bytes':sum(math.prod(s)*4 for s in selected.values()),
        'normal_bytes':sum(normal.values()),'largest':largest,'model_order':list(model)}


def _atomic(path,value):
    probe._atomic(Path(path),value,METADATA_LIMIT)


def _clone_normal(payload,selected):
    """The frozen clone algorithm, with a separately bounded novel-model budget."""
    import numpy as np
    import torch
    memo={};storages={};total=0;visiting=set()
    def clone(value):
        nonlocal total
        if id(value) in memo:return memo[id(value)]
        if isinstance(value,torch.Tensor):
            storage=value.untyped_storage();key=(storage._cdata,storage.nbytes())
            if key not in storages:
                total+=storage.nbytes()
                if total>NORMAL_LIMIT:raise ValueError('Ordinary novel/projector/RNG storage budget exceeded')
                storages[key]=storage.clone()
            result=torch.empty(0,dtype=value.dtype).set_(storages[key],value.storage_offset(),value.shape,value.stride())
            memo[id(value)]=result;return result
        if isinstance(value,np.ndarray):
            result=value.copy(order='A');memo[id(value)]=result;return result
        if type(value) in (dict,list,tuple):
            if id(value) in visiting:raise ValueError('Cyclic checkpoint metadata refused')
            visiting.add(id(value))
            try:
                if type(value) is dict:
                    result={};memo[id(value)]=result
                    for key,child in value.items():
                        if value is payload['model'] and key in selected:continue
                        result[clone(key)]=clone(child)
                else:
                    children=[clone(child) for child in value];result=tuple(children) if type(value) is tuple else children
                    memo[id(value)]=result
                return result
            finally:visiting.remove(id(value))
        return value
    return clone(payload)


def _derive_unlocked(source,stage,plan,snapshot,budget,maximum,progress=None):
    """Caller already holds EX. Never acquire the compatibility reader SH lock."""
    from src.utils import research_lossless_delta as v1,research_stream_delta as stream
    from src.utils.research_checkpoint_io import load_checkpoint
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    if torch.cuda.is_initialized():raise ValueError('Archival publication is CPU only')
    source=Path(source);before=probe.stat(source);zip_info=probe._raw_zip_preflight(source)
    v1.memory_guard(2*zip_info['data_pickle_bytes']+METADATA_LIMIT)
    with FakeTensorMode():fake=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    census=_census(fake,plan,snapshot);del fake;gc.collect()
    upper=sum(v1._zstd().ZSTD_compressBound(math.prod(shape)*4) for shape in census['selected'].values())
    extra=before['st_size']+2*upper+sum(stream.ANCHOR_SHARD_BYTES.values())+8*census['largest']+census['normal_bytes']+METADATA_LIMIT
    guards=[{'phase':'before_source_values',**v1.memory_guard(extra)}]
    source_sha=sha(source);payload=load_checkpoint(source,map_location='cpu',mmap=True)
    if _census(payload,plan,snapshot)!=census:raise ValueError('Real/FakeTensor complete model census differs')
    marker=read(source.parent/'.research_checkpoint.json');compaction=payload.get('checkpoint_compaction',{})
    if (payload.get('step')!=marker['step'] or compaction.get('operation')!='remove_completed_optimizer_only' or
        compaction.get('source_object')!=marker.get('source_checkpoint_object') or
        type(compaction.get('source_state_sha256')) is not str or len(compaction['source_state_sha256'])!=64):
        raise ValueError('Exact retained-object step/compaction payload provenance differs')
    expected=probe.payload_signature(payload);template=_clone_normal(payload,set(census['selected']));last=time.monotonic()
    def tensors():
        nonlocal last
        for index,name in enumerate(census['selected']):
            yield name,payload['model'][name],snapshot
            if progress and time.monotonic()-last>=30:
                progress('archival_encoding',tensors=index+1,total=len(census['selected']));last=time.monotonic()
    wrapped,receipt=stream.stream_encode_small(template,tensors(),expected_decoded_bytes=census['original_bytes'],
        expected_tensor_count=len(census['selected']),largest_tensor_bytes=census['largest'],encoded_upper_bytes=upper,
        max_encoded_bytes=maximum-census['normal_bytes']-stream.METADATA_LIMIT)
    model=wrapped.payload.payload['model'];wrapped.payload.payload['model']={name:model[name] for name in census['model_order']}
    del model,template,payload;wrapped=stream.GuardedPayload(wrapped.payload);gc.collect();probe._no_source_mappings(source)
    header=wrapped.header
    guards.append({'phase':'before_archive_write',**v1.memory_guard(header['encoded_bytes']+header['normal_storage_bytes']+2*METADATA_LIMIT)})
    target=stage/'state.pt'
    with target.open('xb') as handle:
        stream.save_guarded_legacy(wrapped,probe._BoundedWriter(handle,budget,maximum));handle.flush();os.fsync(handle.fileno())
    del wrapped;gc.collect();probe._no_source_mappings(source)
    guards.append({'phase':'before_exact_decode',**v1.memory_guard(stream.memory_requirement(header))})
    restored=load_checkpoint(target,map_location='cpu',mmap=True);actual=probe.payload_signature(restored)
    del restored;gc.collect()
    if expected!=actual:raise ValueError('Whole payload/backing bits/alias/RNG/config equality failed')
    if probe.stat(source)!=before or sha(source)!=source_sha:raise ValueError('Owned raw source changed during derivation')
    return {'source_state_sha256':source_sha,'archive_sha256':sha(target),'archive_bytes':target.stat().st_size,
        'payload_sha256':expected['sha256'],'all_payload_fields_and_backing_bits_exact':True,
        'decoded_original_bytes':census['original_bytes'],'normal_storage_bytes':census['normal_bytes'],
        'encoding_receipt':receipt,'reader_header':header,'memory_guards':guards}


def preflight(run,*,activation=None):
    """No writes/Torch on default closed activation; no boolean force override."""
    try:
        evidence=activation_evidence(activation);plan=eligibility(run)
        if any(not value['marker'].get('checkpoint_archival') for value in plan['roles'].values()):_no_evaluation(Path(plan['run']))
        return {'status':'eligible_metadata_only','activation':evidence,'plan':plan,'publication_performed':False}
    except (OSError,ValueError,KeyError,TypeError) as error:
        return {'status':'gate_closed','reason':str(error),'publication_performed':False,'run':str(Path(run).resolve())}


@contextlib.contextmanager
def _exclusive(run):
    path=run/'.checkpoint.lock'
    if path.is_symlink() or not path.is_file():raise ValueError('Existing local manager lock required')
    with path.open('rb') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def _sync(directory):
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _role_names(run,owner):
    return {role:value['object'] for role,value in _roles(run,owner).items()}


def _set_role(run,role,destination):
    if role not in ('best','last'):raise ValueError('Only retained role pointers may change')
    if not destination.startswith('object-archive-') or len(destination)!=len('object-archive-')+32:
        raise ValueError('Only this transaction archival destination is accepted')
    temp=run/f'.{role}.archive.{destination.removeprefix("object-archive-")}.link'
    expected=Path('.checkpoint_objects')/destination
    # Deterministic task/role name makes a hard-kill before os.replace retryable.
    # A foreign same-name file or differently aimed symlink is never removed.
    if os.path.lexists(temp):
        if not temp.is_symlink() or os.readlink(temp)!=str(expected):raise ValueError('Unknown role temporary is preserved')
        temp.unlink()
    made=False
    try:
        temp.symlink_to(expected,target_is_directory=True);made=True
        os.replace(temp,run/role);_sync(run)
    finally:
        if made and temp.is_symlink() and os.readlink(temp)==str(expected):temp.unlink()


def _journal_valid(journal,plan,evidence):
    if (journal.get('schema_version')!=1 or journal.get('protocol')!=OWNER or
        journal.get('run')!=plan['run'] or journal.get('manager_owner')!=plan['manager']['owner'] or
        journal.get('metadata_sha256')!=plan['metadata_sha256'] or journal.get('activation')!=evidence):
        raise ValueError('Archival transaction owner/config/completion/source activation differs')
    transaction=journal.get('transaction')
    if (type(transaction) is not str or len(transaction)!=32 or any(c not in '0123456789abcdef' for c in transaction) or
        journal.get('stage')!='.stage-archive-'+transaction or journal.get('destination')!='object-archive-'+transaction or
        type(journal.get('source_object')) is not str or Path(journal['source_object']).name!=journal['source_object'] or
        not journal['source_object'].startswith('object-') or journal['source_object']==journal['destination']):
        raise ValueError('Unsafe task-owned journal object names')
    source_roles=journal.get('source_roles')
    if (not isinstance(source_roles,dict) or set(source_roles)!={'best','last'} or
        journal.get('roles')!=[role for role in ('best','last') if source_roles[role]==journal['source_object']]):
        raise ValueError('Journal historical role association differs')
    if not journal['roles']:raise ValueError('Journal cannot publish an unretained object')
    marker=journal.get('source_marker',{})
    if (marker.get('owner')!=plan['manager']['owner'] or marker.get('checkpoint_archival') or
        hashlib.sha256(canonical(marker)+b'\n').hexdigest()!=journal.get('source_marker_canonical_sha256')):
        raise ValueError('Journal original complete marker differs')
    for role,current in _role_names(Path(plan['run']),plan['manager']['owner']).items():
        allowed={source_roles[role]}
        if role in journal['roles']:allowed.add(journal['destination'])
        if current not in allowed:raise ValueError('Historical BEST/LAST changed outside this exact transaction')


def _known_directory(directory):
    if directory.is_symlink() or not directory.is_dir():raise ValueError('Local owned object directory required')
    names={path.name for path in directory.iterdir()}
    if not names<={'state.pt','.research_checkpoint.json'}:raise ValueError('Unknown files preserved in archival object')
    if any(path.is_symlink() or not path.is_file() for path in directory.iterdir()):
        raise ValueError('Unknown nested/symlink object entries preserved')
    return names


def _target_verified(run,journal):
    destination=run/'.checkpoint_objects'/journal['destination'];stage=run/'.checkpoint_objects'/journal['stage']
    if os.path.lexists(destination) and os.path.lexists(stage):raise ValueError('Ambiguous duplicate archival target/stage')
    target=destination if os.path.lexists(destination) else stage
    if _known_directory(target)!={'state.pt','.research_checkpoint.json'}:raise ValueError('Verified target is incomplete')
    from src.utils import research_lossless_delta as v1
    v1.memory_guard((target/'state.pt').stat().st_size+METADATA_LIMIT)
    marker=read(target/'.research_checkpoint.json');expected=dict(journal['source_marker'])
    expected['checkpoint_archival']=journal['target_lineage']
    if marker!=expected or sha(target/'state.pt')!=journal['derived']['archive_sha256']:
        raise ValueError('Verified target bytes/complete source marker/lineage changed')
    return target


def _cleanup_exact(directory,marker,state_sha,*,allow_partial=False):
    """Delete only recorded individual files; never recurse or delete unknowns."""
    if not os.path.lexists(directory):return
    names=_known_directory(directory)
    if not allow_partial and names!={'state.pt','.research_checkpoint.json'}:raise ValueError('Incomplete source before cleanup')
    if '.research_checkpoint.json' in names and read(directory/'.research_checkpoint.json')!=marker:
        raise ValueError('Cleanup object ownership/whole marker differs')
    if 'state.pt' in names:
        from src.utils import research_lossless_delta as v1
        v1.memory_guard((directory/'state.pt').stat().st_size+METADATA_LIMIT)
        if sha(directory/'state.pt')!=state_sha:raise ValueError('Cleanup object bytes differ')
    # Remove model first. The durable journal supplies the marker/ownership if
    # killed between these two unlinks; unknown entries are always refused.
    for name in ('state.pt','.research_checkpoint.json'):
        (directory/name).unlink(missing_ok=True)
    directory.rmdir();_sync(directory.parent)


def _resume_transaction(run,plan,evidence,journal,budget,snapshot,maximum,derive,progress):
    _journal_valid(journal,plan,evidence);objects=run/'.checkpoint_objects'
    source=objects/journal['source_object'];stage=objects/journal['stage'];destination=objects/journal['destination']
    phase=journal.get('phase')
    if phase=='no_benefit':
        _cleanup_exact(stage,journal['staging_marker'],journal['derived']['archive_sha256'],allow_partial=True)
        raise NoBenefit('Previous exact derived object did not save the required bytes; raw roles retained')
    if phase=='planned':
        if os.path.lexists(destination):raise ValueError('Unverified published destination is never accepted')
        if (sha(source/'state.pt')!=journal['source_state_sha256'] or
            sha(source/'.research_checkpoint.json')!=journal['source_marker_sha256'] or
            read(source/'.research_checkpoint.json')!=journal['source_marker']):
            raise ValueError('Raw source changed since durable transaction intent')
        if os.path.lexists(stage):
            # An interrupted unpublished writer can leave a partial state.pt.
            # Its marker was durably owned before the writer ever ran.
            names=_known_directory(stage)
            if names and ('.research_checkpoint.json' not in names or read(stage/'.research_checkpoint.json')!=journal['staging_marker']):
                raise ValueError('Unknown/unowned partial stage preserved')
            for name in ('state.pt','.research_checkpoint.json'):(stage/name).unlink(missing_ok=True)
            stage.rmdir();_sync(objects)
        budget.ensure_space(maximum+METADATA_LIMIT+2**20)
        stage.mkdir();_atomic(stage/'.research_checkpoint.json',journal['staging_marker']);_sync(objects)
        derived=derive(source/'state.pt',stage,plan,snapshot,budget,maximum,progress)
        if (derived.get('source_state_sha256')!=journal['source_state_sha256'] or
            derived.get('all_payload_fields_and_backing_bits_exact') is not True or
            derived.get('archive_sha256')!=sha(stage/'state.pt') or
            derived.get('archive_bytes')!=(stage/'state.pt').stat().st_size):
            raise ValueError('Full exact CPU derivation/source/target proof differs')
        # Charge the complete allowed permanent journal/report/marker envelope,
        # not just state.pt. A smaller model file alone is not a net saving.
        if derived['archive_bytes']+METADATA_LIMIT+MIN_SAVING_BYTES>=journal['source_stat']['st_size']:
            journal.update(phase='no_benefit',derived=derived);_atomic(run/JOURNAL,journal)
            _cleanup_exact(stage,journal['staging_marker'],derived['archive_sha256'])
            raise NoBenefit('Measured archive offers no required saving; raw best/last remain authoritative')
        lineage={'protocol':OWNER,'transaction':journal['transaction'],'source_object':journal['source_object'],
            'source_state_sha256':journal['source_state_sha256'],'source_marker_sha256':journal['source_marker_sha256'],
            'payload_sha256':derived['payload_sha256'],'archive_sha256':derived['archive_sha256'],
            'archive_bytes':derived['archive_bytes'],'activation_sha256':evidence['sha256'],
            'decode_code_sha256':evidence['code_sha256'],'anchor_metadata':journal['anchor_metadata']}
        journal.update(phase='derived',derived=derived,target_lineage=lineage);_atomic(run/JOURNAL,journal)
        phase='derived'
    if phase=='derived':
        if os.path.lexists(destination) or _known_directory(stage)!={'state.pt','.research_checkpoint.json'}:
            raise ValueError('Unpublished exact derived stage is ambiguous/incomplete')
        marker=dict(journal['source_marker']);marker['checkpoint_archival']=journal['target_lineage']
        current=read(stage/'.research_checkpoint.json')
        if current not in (journal['staging_marker'],marker) or sha(stage/'state.pt')!=journal['derived']['archive_sha256']:
            raise ValueError('Derived target bytes/staging or final marker changed')
        _atomic(stage/'.research_checkpoint.json',marker);_sync(stage)
        journal['phase']='verified';_atomic(run/JOURNAL,journal);phase='verified'
    if phase in ('verified','publishing'):
        from src.utils import research_trained_delta_probe_v2 as adapter
        from src.utils import research_lossless_delta as v1
        # Hashing can fault raw/derived pages back into the cgroup. No claim
        # that an existing clean page is reclaimable or already resident.
        v1.memory_guard(journal['source_stat']['st_size']+journal['derived']['archive_bytes']+METADATA_LIMIT)
        if evidence!=activation_evidence(evidence['path']):raise ValueError('Activation/source profile changed before publication')
        current_anchor=adapter.anchor_census(snapshot);current_anchor.pop('shapes')
        if current_anchor!=journal['anchor_metadata']:raise ValueError('Canonical anchor metadata changed before publication')
        if (read(source/'.research_checkpoint.json')!=journal['source_marker'] or
            sha(source/'.research_checkpoint.json')!=journal['source_marker_sha256'] or
            probe.stat(source/'state.pt')!=journal['source_stat'] or sha(source/'state.pt')!=journal['source_state_sha256']):
            raise ValueError('Original owned raw source changed before role publication')
        target=_target_verified(run,journal)
        if target==stage:
            os.rename(stage,destination);_sync(objects)
        journal['phase']='publishing';_atomic(run/JOURNAL,journal)
        for role in journal['roles']:
            current=_role_names(run,plan['manager']['owner'])[role]
            if current==journal['source_object']:_set_role(run,role,journal['destination'])
            elif current!=journal['destination']:raise ValueError('Concurrent retained role change refused')
        journal['phase']='cleanup_started';_atomic(run/JOURNAL,journal);phase='cleanup_started'
    if phase in ('cleanup_started','completed'):
        _target_verified(run,journal)
        roles=_role_names(run,plan['manager']['owner'])
        if any(roles[role]!=journal['destination'] for role in journal['roles']) or journal['source_object'] in roles.values():
            raise ValueError('All intended roles must be durable before deleting raw source')
        _cleanup_exact(source,journal['source_marker'],journal['source_state_sha256'],allow_partial=True)
        journal['phase']='completed';_atomic(run/JOURNAL,journal)
        return journal
    raise ValueError('Unknown archival transaction phase')


def publish_completed_checkpoint(run,*,activation=None,snapshot=None,reserve_gb=120,
                                 max_staging_bytes=STAGING_LIMIT,progress=None):
    """Prospective ONLY. Hard actual-trained/GPU activation gate has no bypass."""
    if not math.isfinite(reserve_gb) or reserve_gb<120 or not 0<max_staging_bytes<=STAGING_LIMIT:
        raise ValueError('Require120GiB disk reserve and staging<=24GiB')
    evidence=activation_evidence(activation);initial=eligibility(run);run=Path(initial['run'])
    from src.utils import research_trained_delta_probe_v2 as adapter
    snapshot=Path(snapshot) if snapshot is not None else adapter.SNAPSHOT
    anchor=adapter.anchor_census(snapshot);public_anchor={key:value for key,value in anchor.items() if key!='shapes'}
    from src.utils.research_storage import DiskBudget
    budget=DiskBudget(run,reserve_gb);maximum=max_staging_bytes-METADATA_LIMIT-2**20
    if maximum<=METADATA_LIMIT:raise ValueError('Insufficient hard staging envelope')
    with _exclusive(run):
        plan=eligibility(run)
        if evidence!=activation_evidence(activation):raise ValueError('Activation changed before exclusive publication')
        journal=read(run/JOURNAL) if os.path.lexists(run/JOURNAL) else None
        pending=journal and journal.get('phase') not in ('completed',)
        raw=any(not item['marker'].get('checkpoint_archival') for item in plan['roles'].values())
        if raw or pending:_no_evaluation(run)
        if os.path.lexists(run/REPORT):
            report=read(run/REPORT)
            if report.get('protocol')!=OWNER or report.get('manager_owner')!=plan['manager']['owner']:
                raise ValueError('Unknown archival report preserved')
        else:report={'schema_version':1,'protocol':OWNER,'run':str(run),'manager_owner':plan['manager']['owner'],
            'activation':evidence,'transactions':[],'training_completed':True,'resumable_optimizer':False}
        while True:
            plan=eligibility(run)
            if journal:
                journal=_resume_transaction(run,plan,evidence,journal,budget,snapshot,maximum,_derive_unlocked,progress)
                if not any(row['transaction']==journal['transaction'] for row in report['transactions']):
                    report['transactions'].append(journal)
                report.update(status='publishing',roles=_role_names(run,plan['manager']['owner']))
                _atomic(run/REPORT,report);journal=None
            plan=eligibility(run)
            sources=[value for value in plan['roles'].values() if not value['marker'].get('checkpoint_archival')]
            if not sources:break
            _no_evaluation(run);value=sources[0];transaction=uuid.uuid4().hex
            source_marker=value['marker'];stage_marker=dict(source_marker)
            stage_marker['checkpoint_archival_staging']={'protocol':OWNER,'transaction':transaction}
            from src.utils import research_lossless_delta as v1
            hash_guard=v1.memory_guard(value['stat']['st_size']+METADATA_LIMIT)
            journal={'schema_version':1,'protocol':OWNER,'run':str(run),'manager_owner':plan['manager']['owner'],
                'transaction':transaction,'phase':'planned','source_object':value['object'],
                'source_roles':_role_names(run,plan['manager']['owner']),
                'roles':[role for role,item in plan['roles'].items() if item['object']==value['object']],
                'source_marker':source_marker,'source_marker_sha256':value['marker_sha256'],
                'source_marker_canonical_sha256':hashlib.sha256(canonical(source_marker)+b'\n').hexdigest(),
                'source_state_sha256':sha(value['state']),'source_stat':value['stat'],
                'stage':'.stage-archive-'+transaction,'destination':'object-archive-'+transaction,
                'staging_marker':stage_marker,'metadata_sha256':plan['metadata_sha256'],'activation':evidence,
                'anchor_metadata':public_anchor,'source_hash_memory_guard':hash_guard}
            _atomic(run/JOURNAL,journal)
        report.update(status='completed',roles=_role_names(run,plan['manager']['owner']),disk=budget.report(),
            payload_or_step_or_metric_changed=False,optimizer_removed_by_archiver=False)
        _completed_roles_verified(eligibility(run),report,read(run/JOURNAL),evidence)
        _atomic(run/REPORT,report)
        return report


def _completed_roles_verified(plan,report,journal,evidence):
    run=Path(plan['run'])
    if (report.get('status')!='completed' or report.get('protocol')!=OWNER or report.get('run')!=str(run) or
        report.get('manager_owner')!=plan['manager']['owner'] or report.get('activation')!=evidence or
        report.get('roles')!=_role_names(run,plan['manager']['owner']) or
        journal.get('phase')!='completed' or
        not any(row==journal for row in report.get('transactions',[]))):
        raise ValueError('Completed archival journal/report/activation/roles differ')
    _journal_valid(journal,plan,evidence)
    for value in {item['object']:item for item in plan['roles'].values()}.values():
        from src.utils import research_lossless_delta as v1
        v1.memory_guard(value['stat']['st_size']+METADATA_LIMIT)
        lineage=value['marker'].get('checkpoint_archival',{})
        candidates=[row for row in report['transactions'] if row.get('destination')==value['object']]
        if len(candidates)!=1:raise ValueError('Retained archive lacks one exact source transaction')
        transaction=candidates[0]
        expected=dict(transaction['source_marker']);expected['checkpoint_archival']=transaction['target_lineage']
        if (transaction.get('activation')!=evidence or transaction.get('metadata_sha256')!=plan['metadata_sha256'] or
            transaction.get('manager_owner')!=plan['manager']['owner'] or transaction.get('run')!=str(run) or
            transaction.get('phase')!='completed' or value['marker']!=expected or lineage.get('protocol')!=OWNER or
            lineage.get('archive_sha256')!=sha(value['state']) or
            transaction.get('derived',{}).get('all_payload_fields_and_backing_bits_exact') is not True or
            transaction['derived'].get('payload_sha256')!=lineage.get('payload_sha256')):
            raise ValueError('Archived physical bytes/full marker/whole payload lineage differ')


def validate_completed_archival(run):
    """Archive-aware postprocessing check; never sends legacy files to mmap."""
    plan=eligibility(run);run=Path(plan['run'])
    with _exclusive(run):
        plan=eligibility(run);report=read(run/REPORT);journal=read(run/JOURNAL)
        evidence=activation_evidence(report.get('activation',{}).get('path'))
        _completed_roles_verified(plan,report,journal,evidence)
        return report
