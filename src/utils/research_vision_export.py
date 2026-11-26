"""Task-owned immutable vision-only exports from pinned, local HF snapshots.

No original file is modified or deleted. Full donor files are hashed, but only
the exact native vision state tensors are deserialized. Derived artifacts retain
original names/dtypes/bits and are not full multimodal checkpoints.
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.utils.research_storage import DiskBudget,_atomic_json,_fsync_tree,_fsync_directory

ROOT=Path(__file__).resolve().parents[2]
OWNER_KIND='research_vision_derived_export_v1'
SCOPE='vision_only_donor_weights_not_full_multimodal_checkpoint'
DONORS={
 'qwen25vision':{'repo':'Qwen/Qwen2.5-VL-7B-Instruct','backend':'qwen2_5_vl',
    'prefix_groups':(('model.visual.','visual.'),),
    'required_assets':('config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json'),
    'optional_assets':('vocab.json','merges.txt','chat_template.json','chat_template.jinja','special_tokens_map.json','added_tokens.json')},
 'gemma4vision':{'repo':'google/gemma-4-E4B-it','backend':'gemma4',
    'prefix_groups':(('model.vision_tower.',),('model.embed_vision.',)),
    'required_assets':('config.json','processor_config.json','tokenizer_config.json','tokenizer.json'),
    'optional_assets':('chat_template.jinja','chat_template.json','special_tokens_map.json','added_tokens.json','tokenizer.model')},
}
LICENSE_ASSETS=('LICENSE','LICENSE.txt','LICENSE.md','NOTICE','NOTICE.txt')
CODE_FILES=('src/utils/research_vision_export.py','scripts/export_research_vision_only.py',
            'src/model/research_vision.py','scripts/extract_vision_features.py',
            'src/model/gigachat_vl.py','src/utils/research_storage.py')
_DTYPE_BYTES={'BOOL':1,'U8':1,'I8':1,'I16':2,'U16':2,'F16':2,'BF16':2,
              'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8,'F8_E4M3':1,'F8_E5M2':1}


def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def file_sha(path,progress=None):
    h=hashlib.sha256();done=0;last=time.monotonic();total=Path(path).stat().st_size
    with Path(path).open('rb') as handle:
        while chunk:=handle.read(16*2**20):
            h.update(chunk);done+=len(chunk)
            if progress and time.monotonic()-last>=10:
                progress('hashing_source',file=Path(path).name,bytes_completed=done,total_bytes=total);last=time.monotonic()
    return h.hexdigest()


def tensor_sha(tensor):
    if tensor.device.type!='cpu':raise ValueError('Export/checksum tensors must stay on CPU')
    return hashlib.sha256(memoryview(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()).cast('B')).hexdigest()


def _codes():return {name:file_sha(ROOT/name) for name in CODE_FILES}
def _software():return {name:version(name) for name in ('torch','transformers','safetensors','tokenizers')}
def _stat(path):
    value=Path(path).stat()
    return {'device':value.st_dev,'inode':value.st_ino,'bytes':value.st_size,'mtime_ns':value.st_mtime_ns}


def _expected_shapes(snapshot,backend,prefixes):
    """Independent native meta-state contract; no donor language model exists."""
    from transformers import AutoConfig
    config=AutoConfig.from_pretrained(str(snapshot),trust_remote_code=False,local_files_only=True)
    if config.model_type!=backend:raise ValueError('Source config backend differs from requested donor')
    with torch.device('meta'):
        if backend=='qwen2_5_vl':
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
            modules=[Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)]
        elif backend=='gemma4':
            from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel,Gemma4MultimodalEmbedder
            modules=[Gemma4VisionModel(config.vision_config),Gemma4MultimodalEmbedder(config.vision_config,config.text_config)]
        else:raise ValueError('Unsupported strict vision-only export backend')
    if not all(parameter.device.type=='meta' for module in modules for parameter in module.parameters()):
        raise RuntimeError('Native shape validation materialized weights')
    return {prefix+name:list(tensor.shape) for prefix,module in zip(prefixes,modules)
            for name,tensor in module.state_dict().items()}


def _relative_file(name):
    path=Path(name)
    if path.is_absolute() or '..' in path.parts or len(path.parts)!=1:
        raise ValueError(f'Unexpected source file path: {name}')
    return name


def plan_export(snapshot,donor_id,source_revision,*,source_repo=None,progress=None):
    if donor_id not in DONORS:raise ValueError('Only Qwen2.5/Gemma4 vision exports are supported')
    if len(source_revision)!=40 or any(c not in '0123456789abcdef' for c in source_revision):
        raise ValueError('Export requires a pinned 40-character source commit')
    specification=DONORS[donor_id];snapshot=Path(snapshot).resolve()
    if source_repo not in (None,specification['repo']):raise ValueError('Donor repo does not match strict export scope')
    if (snapshot/'derivation.json').exists():raise ValueError('Source must be the original pinned donor snapshot')
    assets=list(specification['required_assets'])
    if any(not (snapshot/name).is_file() for name in assets):raise FileNotFoundError('Required source config/processor/tokenizer asset is missing')
    assets+=sorted(name for name in specification['optional_assets']+LICENSE_ASSETS if (snapshot/name).is_file() and name not in assets)
    index=snapshot/'model.safetensors.index.json'
    weight_map=None
    if index.exists():
        weight_map=json.loads(index.read_text())['weight_map']
        selected={name:_relative_file(file) for name,file in weight_map.items()
                  if any(name.startswith(prefix) for group in specification['prefix_groups'] for prefix in group)}
        files=sorted(set(selected.values()))
    else:selected=None;files=['model.safetensors']
    if not files:raise ValueError('No required donor vision tensor prefix in source index')
    if any(not (snapshot/name).is_file() for name in files):raise FileNotFoundError('A required local vision source weight shard is missing')
    tensors={};all_keys=0
    for name in files:
        with safe_open(str(snapshot/name),framework='pt',device='cpu') as reader:
            all_keys+=len(reader.keys())
            for key in reader.keys():
                if not any(key.startswith(prefix) for group in specification['prefix_groups'] for prefix in group):continue
                if key in tensors:raise ValueError('Duplicate source vision tensor across shards')
                view=reader.get_slice(key);dtype=view.get_dtype();shape=view.get_shape()
                if dtype not in _DTYPE_BYTES:raise ValueError(f'Unsupported source storage dtype: {dtype}')
                tensors[key]={'source_file':name,'safetensors_dtype':dtype,'shape':shape,
                              'bytes':math.prod(shape)*_DTYPE_BYTES[dtype]}
    if selected is not None and selected!={key:value['source_file'] for key,value in tensors.items()}:
        raise ValueError('Source index/header vision tensor keys or shard assignments differ')
    prefixes=[]
    for group in specification['prefix_groups']:
        matching=[prefix for prefix in group if any(key.startswith(prefix) for key in tensors)]
        if len(matching)!=1:raise ValueError('Each required vision prefix group must match exactly once')
        prefixes.append(matching[0])
    expected=_expected_shapes(snapshot,specification['backend'],prefixes)
    if set(tensors)!=set(expected):
        raise ValueError(f'Native required vision tensor keys differ: missing={sorted(set(expected)-set(tensors))}, unexpected={sorted(set(tensors)-set(expected))}')
    if any(value['shape']!=expected[key] for key,value in tensors.items()):raise ValueError('Source native vision tensor shape differs')
    source_files={name:{'sha256':file_sha(snapshot/name,progress),'bytes':(snapshot/name).stat().st_size,
                        'resolved_path':str((snapshot/name).resolve()),'stat':_stat(snapshot/name)}
                  for name in files+assets+(['model.safetensors.index.json'] if index.exists() else [])}
    identity={'repo':specification['repo'],'revision':source_revision,'backend':specification['backend'],
              'prefixes':prefixes,'source_files':{name:{'sha256':value['sha256'],'bytes':value['bytes']} for name,value in source_files.items()},
              'tensor_headers':tensors,'source_index_present':index.exists()}
    return {'donor_id':donor_id,'snapshot':str(snapshot),'source':identity,'source_identity_sha256':canonical_sha(identity),
            'source_observations':source_files,'assets':assets,'weight_files':files,'tensor_headers':tensors,
            'excluded_source_tensor_keys':all_keys-len(tensors),'export_code_sha256':_codes(),'software':_software(),
            'tensor_bytes':sum(value['bytes'] for value in tensors.values()),
            'anticipated_new_peak_bytes':sum(value['bytes'] for value in tensors.values())+sum(source_files[name]['bytes'] for name in assets)+4*2**20}


def _source_unchanged(plan,progress=None):
    snapshot=Path(plan['snapshot'])
    for name,expected in plan['source_observations'].items():
        if _stat(snapshot/name)!=expected['stat'] or file_sha(snapshot/name,progress)!=expected['sha256']:
            raise ValueError(f'Original source changed during export: {name}')
    if _codes()!=plan['export_code_sha256']:raise ValueError('Exporter/native-loader source changed during export')


def _check_raw_tensors(path,expected,prefixes):
    with safe_open(str(path),framework='pt',device='cpu') as reader:
        keys=set(reader.keys())
        if any(not any(key.startswith(prefix) for prefix in prefixes) for key in keys):
            raise ValueError('Unexpected language/audio/non-vision tensor in derived artifact')
        if keys!=set(expected):raise ValueError('Derived safetensors required keys differ')
        for name,record in expected.items():
            view=reader.get_slice(name)
            if view.get_dtype()!=record['safetensors_dtype'] or view.get_shape()!=record['shape']:
                raise ValueError(f'Derived tensor dtype/shape differs: {name}')
            if tensor_sha(reader.get_tensor(name))!=record['sha256_raw_bytes']:
                raise ValueError(f'Derived tensor bits differ: {name}')


def verify_derived_artifact(path,plan):
    path=Path(path)
    if path.is_symlink() or not path.is_dir():raise ValueError('Derived target must be a real immutable directory')
    manifest=json.loads((path/'derivation.json').read_text())
    if (manifest.get('artifact_scope')!=SCOPE or manifest.get('source_identity_sha256')!=plan['source_identity_sha256']
        or manifest.get('export_code_sha256')!=plan['export_code_sha256'] or manifest.get('software')!=plan['software']):
        raise ValueError('Existing derived artifact source/code/environment identity differs')
    expected_files=set(manifest['output_files'])|{'derivation.json'}
    if {entry.name for entry in path.iterdir()}!=expected_files:raise ValueError('Derived artifact contains unexpected files')
    for name in expected_files:
        _relative_file(name)
        if (path/name).is_symlink() or not (path/name).is_file():raise ValueError('Derived files must be independent regular copies')
    headers=plan['tensor_headers'];records=manifest['tensors']
    if set(records)!=set(headers):raise ValueError('Derived manifest has extra/missing native vision tensors')
    for key,header in headers.items():
        if any(records[key].get(name)!=value for name,value in header.items()):raise ValueError('Derived manifest tensor header differs')
    index=json.loads((path/'model.safetensors.index.json').read_text())
    if index['weight_map']!={key:'model.safetensors' for key in records}:
        raise ValueError('Derived vision-only index contains unexpected/missing mappings')
    _check_raw_tensors(path/'model.safetensors',records,plan['source']['prefixes'])
    if canonical_sha(records)!=manifest['tensors_sha256']:raise ValueError('Derived aggregate tensor checksum differs')
    for name,metadata in manifest['output_files'].items():
        if (path/name).stat().st_size!=metadata['bytes'] or file_sha(path/name)!=metadata['sha256']:
            raise ValueError(f'Derived artifact file checksum differs: {name}')
    if sum(entry.stat().st_size for entry in path.iterdir())!=manifest['artifact_total_bytes']:
        raise ValueError('Derived artifact byte total differs')
    return manifest


def _publish_no_replace(source,destination):
    libc=ctypes.CDLL(None,use_errno=True)
    function=getattr(libc,'renameat2',None)
    if function is None:raise NotImplementedError('Atomic no-replace directory publish requires Linux renameat2')
    function.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint)
    function.restype=ctypes.c_int
    if function(-100,os.fsencode(source),-100,os.fsencode(destination),1):
        number=ctypes.get_errno()
        if number==errno.EEXIST:raise FileExistsError(destination)
        raise OSError(number,os.strerror(number))
    _fsync_directory(Path(destination).parent)


def _stage_cleanup(stage,owner,target_name):
    if stage.is_symlink() or not stage.is_dir():raise ValueError('Refusing non-owned staging cleanup')
    marker=stage/'export.pending.json'
    if marker.is_file():data=json.loads(marker.read_text())
    elif (stage/'derivation.json').is_file():
        manifest=json.loads((stage/'derivation.json').read_text())
        if manifest.get('artifact_scope')!=SCOPE:raise ValueError('Unknown completed staging is preserved')
        data={'kind':OWNER_KIND,'owner':manifest.get('export_owner'),'target_name':manifest.get('target_name'),
              'allowed_files':list(manifest['output_files'])+['derivation.json']}
    elif stage.name.startswith('.'+target_name+'.staging-'+owner+'-'):
        # The random root owner in the private staging name survives a kill
        # during the first atomic marker write. Still preserve unknown content.
        assets={name for spec in DONORS.values() for name in spec['required_assets']+spec['optional_assets']}
        data={'kind':OWNER_KIND,'owner':owner,'target_name':target_name,
              'allowed_files':sorted(assets|set(LICENSE_ASSETS)|{'model.safetensors','model.safetensors.index.json','derivation.json','README.md'})}
    else:raise ValueError('Unknown staging directory is preserved')
    if data.get('owner')!=owner or data.get('target_name')!=target_name or data.get('kind')!=OWNER_KIND:
        raise ValueError('Staging ownership does not match; directory preserved')
    allowed=set(data['allowed_files'])|{'export.pending.json'}
    def allowed_name(name):
        return name in allowed or any(re.fullmatch(re.escape('.'+base+'.')+r'[0-9a-f]{32}\.tmp',name) for base in allowed)
    if any(entry.is_symlink() or not entry.is_file() or not allowed_name(entry.name) for entry in stage.iterdir()):
        raise ValueError('Unexpected content in staging directory is preserved')
    shutil.rmtree(stage)


def export_vision_snapshot(plan,output_root,*,reserve_gb=120,validator=None,progress=None,_fault_hook=None):
    """Atomically publish independent raw-bit-preserving weights and metadata.

    The production script supplies native CPU state/feature parity validation.
    Without a validator, the manifest explicitly records that parity is pending;
    this low-level mode supports serialization tests, not production validation.
    """
    if plan.get('donor_id') not in DONORS or canonical_sha(plan['source'])!=plan['source_identity_sha256']:
        raise ValueError('Invalid or changed pinned source plan')
    specification=DONORS[plan['donor_id']]
    if (plan['source']['repo']!=specification['repo'] or plan['source']['backend']!=specification['backend']
        or plan['tensor_headers']!=plan['source']['tensor_headers']):raise ValueError('Source plan changed export scope')
    for name in plan['assets']+plan['weight_files']:_relative_file(name)
    allowed_assets=set(specification['required_assets']+specification['optional_assets']+LICENSE_ASSETS)
    if not set(plan['assets'])<=allowed_assets:raise ValueError('Unexpected source asset requested')
    output_root=Path(output_root).resolve();snapshot=Path(plan['snapshot'])
    from huggingface_hub.constants import HF_HUB_CACHE
    if (output_root.is_relative_to(Path(HF_HUB_CACHE).resolve()) or output_root.is_relative_to(snapshot)
        or snapshot.is_relative_to(output_root)):
        raise ValueError('Derived artifacts must be outside the original HF/source tree')
    budget=DiskBudget(output_root,reserve_gb=reserve_gb);budget.ensure_space(0)
    target_name=plan['donor_id']+'-'+plan['source']['revision'];target=output_root/target_name
    _relative_file(target_name)
    lock_path=output_root/('.'+target_name+'.export.lock')
    if lock_path.is_symlink():raise ValueError('Export lock must not be a symlink')
    lock=lock_path.open('a+')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        owner_path=output_root/'.vision-export-owner.json'
        if owner_path.is_symlink():raise ValueError('Export owner marker must not be a symlink')
        if owner_path.exists():
            owner_data=json.loads(owner_path.read_text())
            if owner_data.get('kind')!=OWNER_KIND:raise ValueError('Export root belongs to another task')
        else:
            owner_data={'kind':OWNER_KIND,'owner':uuid.uuid4().hex};_atomic_json(owner_path,owner_data)
        owner=owner_data['owner']
        if target.exists() or target.is_symlink():
            _source_unchanged(plan,progress)
            manifest=verify_derived_artifact(target,plan)
            if validator is not None and manifest.get('native_validation',{}).get('status')!='passed':
                raise ValueError('Existing immutable artifact has no passed native parity validation')
            return {'path':str(target),'reused':True,'manifest':manifest,'disk':budget.report()}
        for stale in output_root.glob('.'+target_name+'.staging-*'):_stage_cleanup(stale,owner,target_name)
        budget.ensure_space(plan['anticipated_new_peak_bytes'])
        stage=Path(tempfile.mkdtemp(prefix='.'+target_name+'.staging-'+owner+'-',dir=output_root))
        allowed=set(plan['assets'])|{'model.safetensors','model.safetensors.index.json','derivation.json','README.md'}
        _atomic_json(stage/'export.pending.json',{'kind':OWNER_KIND,'owner':owner,'target_name':target_name,'allowed_files':sorted(allowed)})
        try:
            for name in plan['assets']:
                shutil.copyfile(snapshot/name,stage/name)
                if file_sha(stage/name)!=plan['source_observations'][name]['sha256']:
                    raise ValueError('Copied source asset checksum differs')
            records={};tensors={}
            for file in plan['weight_files']:
                with safe_open(str(snapshot/file),framework='pt',device='cpu') as reader:
                    for name,header in plan['tensor_headers'].items():
                        if header['source_file']!=file:continue
                        value=reader.get_tensor(name)
                        records[name]=dict(header,sha256_raw_bytes=tensor_sha(value),torch_dtype=str(value.dtype))
                        tensors[name]=value
            if progress:progress('writing_vision_only_weights',donor=plan['donor_id'],tensor_bytes=plan['tensor_bytes'],tensor_count=len(tensors))
            budget.ensure_space(plan['anticipated_new_peak_bytes']-sum(path.stat().st_size for path in stage.iterdir()))
            save_file(tensors,str(stage/'model.safetensors'),metadata={'format':'pt'})
            del tensors;gc.collect()
            if _fault_hook:_fault_hook('after_weights',stage)
            _atomic_json(stage/'model.safetensors.index.json',{'metadata':{'total_size':plan['tensor_bytes'],'artifact_scope':SCOPE},
                'weight_map':{name:'model.safetensors' for name in sorted(records)}})
            readme=(f"# Vision-only derived donor artifact\n\nSource: `{plan['source']['repo']}` at `{plan['source']['revision']}`.\n\n"
                "Only native frozen visual tower tensors (and Gemma's frozen donor multimodal embedder) are included. "
                "Original config and processor/tokenizer assets are copied exactly for the strict `build_frozen_vision` loader. "
                "The original config retains full donor architecture metadata; this directory has no donor language/audio weights "
                "and is not a complete multimodal checkpoint.\n\n"
                "All tensor names, storage dtypes and raw bits are preserved. The vision-only index references only this directory's "
                "own model.safetensors. Source HF files remain intact. Current production resolution and feature-cache namespaces "
                "are unchanged; derived local paths acquire separate fingerprints until an explicitly reviewed resolver is added.\n\n"
                "derivation.json records source/code/file/tensor checksums and validation scope. Original upstream model terms continue to apply; "
                "this export grants no additional model rights.\n")
            (stage/'README.md').write_text(readme)
            _check_raw_tensors(stage/'model.safetensors',records,plan['source']['prefixes'])
            validation=validator(snapshot,stage,plan) if validator else {'status':'not_run','native_parity_required_before_production':True}
            if validator and (validation.get('status')!='passed'
                or validation.get('backend')!=plan['source']['backend']
                or validation.get('strict_native_load_state_dict') is not True
                or validation.get('all_native_state_keys_and_bits_equal') is not True
                or validation.get('all_parameters_frozen_eval_cpu') is not True
                or validation.get('native_state_key_count')!=len(records)
                or validation.get('llm_instantiated') is not False or validation.get('gpu_initialized') is not False
                or len(validation.get('fixtures',[]))!=2 or len(validation.get('features',[]))!=2
                or any(row.get('max_abs_difference')!=0. for row in validation.get('features',[]))):
                raise ValueError('Native vision parity validator did not provide the required complete proof')
            _source_unchanged(plan,progress)
            for file in plan['weight_files']:
                with safe_open(str(snapshot/file),framework='pt',device='cpu') as reader:
                    for name,record in records.items():
                        if record['source_file']==file and tensor_sha(reader.get_tensor(name))!=record['sha256_raw_bytes']:
                            raise ValueError('Source vision tensor bits changed during export')
            output_files={name:{'sha256':file_sha(stage/name),'bytes':(stage/name).stat().st_size} for name in sorted(allowed-{'derivation.json'})}
            manifest={'schema_version':1,'artifact_scope':SCOPE,'donor_id':plan['donor_id'],
                'export_owner':owner,'target_name':target_name,
                'source':plan['source'],'source_identity_sha256':plan['source_identity_sha256'],
                'source_observations':plan['source_observations'],'export_code_sha256':plan['export_code_sha256'],
                'software':plan['software'],'tensors':records,'tensors_sha256':canonical_sha(records),
                'tensor_count':len(records),'tensor_bytes':plan['tensor_bytes'],
                'tensor_elements':sum(math.prod(record['shape']) for record in records.values()),
                'excluded_source_tensor_keys':plan['excluded_source_tensor_keys'],'output_files':output_files,
                'source_cache_unchanged':True,'source_files_deleted':False,'native_validation':validation,
                'artifact_total_bytes':0,'cache_namespace':'derived local artifact; production resolver not modified'}
            subtotal=sum(value['bytes'] for value in output_files.values())
            for _ in range(10):
                size=subtotal+len((json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
                if size==manifest['artifact_total_bytes']:break
                manifest['artifact_total_bytes']=size
            _atomic_json(stage/'derivation.json',manifest)
            (stage/'export.pending.json').unlink()
            verify_derived_artifact(stage,plan)
            budget.ensure_space(0);_fsync_tree(stage)
            if _fault_hook:_fault_hook('before_publish',stage)
            if any(_stat(snapshot/name)!=value['stat'] for name,value in plan['source_observations'].items()):
                raise ValueError('Original source changed immediately before publish')
            if _codes()!=plan['export_code_sha256']:raise ValueError('Export code changed immediately before publish')
            _publish_no_replace(stage,target)
            if progress:progress('vision_only_artifact_published',path=str(target),bytes=manifest['artifact_total_bytes'],free_gib=budget.report()['free_gib'])
            return {'path':str(target),'reused':False,'manifest':manifest,'disk':budget.report()}
        except BaseException:
            if stage.exists():
                if not (stage/'export.pending.json').exists():
                    _atomic_json(stage/'export.pending.json',{'kind':OWNER_KIND,'owner':owner,'target_name':target_name,'allowed_files':sorted(allowed)})
                _stage_cleanup(stage,owner,target_name)
            raise


def _capture_native(path,fixtures):
    from src.model.research_vision import build_frozen_vision
    vision=build_frozen_vision(str(path),torch.device('cpu'),torch.float32)
    modules={'vision_tower':vision.vision_tower}
    if vision.vision_projector is not None:modules['vision_projector']=vision.vision_projector
    state={};parameters=0
    for component,module in modules.items():
        if module.training or any(parameter.requires_grad or parameter.device.type!='cpu' for parameter in module.parameters()):
            raise ValueError('Native donor is not entirely frozen/eval/CPU')
        if any(any(word in name for word in ('language_model','embed_tokens','lm_head','audio_tower')) for name,_ in module.named_modules()):
            raise ValueError('A donor language/audio module was instantiated')
        parameters+=sum(parameter.numel() for parameter in module.parameters())
        for name,tensor in module.state_dict().items():
            state[component+'.'+name]={'dtype':str(tensor.dtype),'shape':list(tensor.shape),'sha256_raw_bytes':tensor_sha(tensor)}
    encoded=vision.encode_paths(fixtures)
    return {'backend':vision.backend,'feature_dim':vision.feature_dim,'parameters':parameters,'state':state,
            'features':[row['features'].clone() for row in encoded],'metadata':[row['metadata'] for row in encoded],
            'fingerprint_sha256':canonical_sha(vision.fingerprint_config)}


def cpu_memory_budget(anticipated_extra_bytes,*,meminfo_path='/proc/meminfo',cgroup_root='/sys/fs/cgroup'):
    """Require32GiB headroom against both host availability and the real cgroup.

    Existing file cache is observable but never added back as available memory.
    A large host MemTotal cannot override a smaller container memory.max.
    """
    values={line.split(':',1)[0]:int(line.split(':',1)[1].strip().split()[0])*1024
            for line in Path(meminfo_path).read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    if 'MemAvailable' not in values:raise RuntimeError('Cannot establish actual host available memory')
    cgroup=Path(cgroup_root)
    if not (cgroup/'memory.max').is_file() or not (cgroup/'memory.current').is_file():
        raise RuntimeError('Cannot establish actual cgroup-v2 memory limit/current usage')
    maximum=(cgroup/'memory.max').read_text().strip();limit=None if maximum=='max' else int(maximum)
    current=int((cgroup/'memory.current').read_text())
    remaining=None if limit is None else max(0,limit-current)
    reserve=32*2**30;required=reserve+int(anticipated_extra_bytes)
    statistics={}
    if (cgroup/'memory.stat').is_file():
        statistics={key:int(value) for key,value in (line.split() for line in (cgroup/'memory.stat').read_text().splitlines())}
    report={'host_total_bytes':values.get('MemTotal'),'host_available_bytes':values['MemAvailable'],
            'cgroup_limit_bytes':limit,'cgroup_current_bytes':current,'cgroup_remaining_bytes':remaining,
            'cgroup_reclaimable_file_cache_observed_bytes':statistics.get('inactive_file',0),
            'cache_credit_applied':False,'headroom_reserve_bytes':reserve,
            'anticipated_extra_bytes':int(anticipated_extra_bytes),'required_available_bytes':required}
    if values['MemAvailable']<required:raise MemoryError('Host MemAvailable cannot preserve32GiB headroom plus vision peak: '+json.dumps(report))
    if remaining is not None and remaining<required:raise MemoryError('Cgroup remaining memory cannot preserve32GiB headroom plus vision peak: '+json.dumps(report))
    return report


def _wait_cpu_memory(anticipated_extra_bytes,progress=None):
    started=time.monotonic()
    while True:
        try:return cpu_memory_budget(anticipated_extra_bytes)
        except MemoryError as error:
            if time.monotonic()-started>=300:raise
            gc.collect()
            if progress:progress('waiting_cpu_memory_headroom',error=str(error),elapsed_wait_seconds=time.monotonic()-started)
            time.sleep(10)


def native_cpu_parity(snapshot,derived,plan,fixtures,progress=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1') or torch.cuda.is_initialized():
        raise RuntimeError('Actual vision parity must run with CUDA hidden and uninitialized')
    if len(fixtures)!=2 or any(not Path(path).is_file() for path in fixtures):raise ValueError('Exactly two existing RGB fixtures are required')
    # Two FP32 native models + three raw tensor-sized temporaries is deliberately
    # conservative even though captures release each model before the next load.
    anticipated=8*sum(math.prod(header['shape']) for header in plan['tensor_headers'].values())+3*plan['tensor_bytes']+2*2**30
    source_memory=_wait_cpu_memory(anticipated,progress)
    if progress:progress('native_source_vision_cpu_loading',donor=plan['donor_id'])
    original=_capture_native(snapshot,fixtures);gc.collect()
    derived_memory=_wait_cpu_memory(anticipated,progress)
    if progress:progress('native_derived_vision_cpu_loading',donor=plan['donor_id'])
    exported=_capture_native(derived,fixtures);gc.collect()
    if original['state']!=exported['state']:raise ValueError('Native source/derived state keys, dtypes, shapes or parameter bits differ')
    if original['metadata']!=exported['metadata']:raise ValueError('Native source/derived image metadata differ')
    features=[]
    for source,changed,metadata in zip(original['features'],exported['features'],original['metadata']):
        if source.dtype!=changed.dtype or source.shape!=changed.shape or tensor_sha(source)!=tensor_sha(changed):
            raise ValueError('Native source/derived RGB features are not bit-identical')
        features.append({'shape':list(source.shape),'dtype':str(source.dtype),'sha256_raw_bytes':tensor_sha(source),
                         'max_abs_difference':float((source-changed).abs().max()),'metadata_sha256':canonical_sha(metadata)})
    if torch.cuda.is_initialized():raise RuntimeError('Vision parity unexpectedly initialized CUDA')
    return {'status':'passed','scope':'actual native strict vision-only CPU source/derived parity',
            'backend':original['backend'],'feature_dim':original['feature_dim'],'vision_parameter_count':original['parameters'],
            'strict_native_load_state_dict':True,'all_native_state_keys_and_bits_equal':True,
            'native_state_key_count':len(original['state']),'native_state_sha256':canonical_sha(original['state']),
            'fixtures':[{'path':str(path),'sha256':file_sha(path)} for path in fixtures],'features':features,
            'source_fingerprint_sha256':original['fingerprint_sha256'],'derived_fingerprint_sha256':exported['fingerprint_sha256'],
            'derived_fingerprint_scope':'prepublish staging path; final directory obtains its own local cache namespace',
            'fingerprint_namespace_equal':original['fingerprint_sha256']==exported['fingerprint_sha256'],
            'all_parameters_frozen_eval_cpu':True,'llm_instantiated':False,'gpu_initialized':False,
            'cpu_memory_guards':{'before_source':source_memory,'before_derived':derived_memory},
            'limits':'Native vision CPU equivalence, not VLM quality, GPU kernels, or production derived resolution/cache compatibility'}
