"""Opt-in, local-only derived vision contract; canonical frozen modules untouched.

Metadata resolution never opens original donor weights, imports Torch or calls
the Hub. Full exported-byte verification is mandatory before model construction.
This prototype supplies no CUDA/pruning authorization or current queue migration.
"""
from __future__ import annotations

import copy
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from types import ModuleType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
OWNER='research_derived_vision_local_v1'
DEFAULT_EXPORT_ROOT=ROOT/'models/research-vision-derived'
SOURCE='src/utils/research_derived_vision_v1.py'
TRAIN_CLI='scripts/run_research_derived_vision_v1.py'
EVAL_CLI='scripts/eval_research_derived_vision_v1.py'
TEST='tests/test_research_derived_vision_v1.py'
FROZEN={
    'scripts/run_research_stage.py':'f7fd22d1c94fb337af2be60d3324415b2b2c8cd3d62e9799f2da4c05ace87cfc',
    'src/eval/research_inference.py':'47576ba901c5e6ba3dc95b8c8777872d36c3ea73f2a4e0c6d74a0955df75744d',
    'src/model/research_vision.py':'6f8d2aeaa2daea022ffedaed53b7fc2daecfa18388165ef8f23876664ebee46d',
    'src/dataset/vision_cache.py':'47bfa012cffdc88ab960f036c18407ad1f7215b69a78267b623dc33bebd4d808',
    'src/utils/research_checkpoint_compat.py':'4dcf5c73f8e719d8a995f2578fb65be9e6fca48ad80705cb95028ed2bc3141b6',
    'src/utils/research_storage.py':'ad67bd545ee86076dab713c5316e85dcede65147ea96e9f778ba25a5fc1894b3',
    'src/model/gigachat_vl.py':'12f6b1070ccc60f9c53a045e53e866b2526618a658e0f28f002dec4e21675a95',
    'scripts/extract_vision_features.py':'ceb245f5a5d0c0d777b60225eb07e4ce2478903b4e7bc65274d3650a2de88258',
    'src/model/smallthinker.py':'7d5aaa551b98e7958f1a69930c797f7906d7c770084b156ba36cc5d577cbdd26',
    'src/model/research_moe.py':'f8cbfd6a8fdb7044d25d031f0a028055cb114a7ebd71a9b4f873dcd752674c6c',
    'src/model/research_quantization.py':'b588541b3232ca522acaeae2d1949cc2aac29cf9608c2d8f99afc8faad129830',
    'src/model/visual_expert_gates.py':'93f7e69f1903c3cee9a7999fa6b93428a8b5df42cf4cf1f79e62ea999609bb4d',
    'src/model/text_moe_backbones.py':'3ad524222e500acb6526256224df875f529f4b1288b1025a29e24e79298c20c8',
    'src/eval/research_selection.py':'91c05924df717ce46ef22787dfa5b4b0e45e35533096769373be62ceda3cade2',
    'src/eval/research_metrics.py':'440424ed720dbf17fce9a12ccb4ec6565f30298c6c40e9b47fd0ab60a8ed6844',
    'third_party/vqa_normalization/tables.json':'ba9b45e88fcf697c1de6e550158ab9fbbe4d8456e4da34f3c61079e35823bce8',
}
DONORS={
 'Qwen/Qwen2.5-VL-7B-Instruct':{
    'revision':'cc594898137f460bfe9f0759e9844b3ce807cfb5','donor':'qwen25vision',
    'backend':'qwen2_5_vl','feature_dim':3584,
    'derivation_sha256':'453ae3410d505d02fb76fb391b36858e3276c25bbc670b997b9b18d7633d0efd'},
 'google/gemma-4-E4B-it':{
    'revision':'4ecc2ba63bcac23ba4a767fc87025b1450f84c9c','donor':'gemma4vision',
    'backend':'gemma4','feature_dim':2560,
    'derivation_sha256':'3d3cfd4f2360a3fc159af24c6a888ea9b846a7e7dd931b94df834756974cdfe5'},
}
PROFILE_BUDGETS={'qwen_native_bounded_729':729,'qwen_native_bounded_1024':1024,'qwen_native_bounded_2048':2048}
CONFIG_FIELDS=('derived_vision_schema','vision_export_ref','vision_profile_ref',
               'vision_cache_contract','vision_cache_fingerprint','derived_vision_code_sha256')
MAX_METADATA=2*2**20
_BYTES={'BOOL':1,'U8':1,'I8':1,'I16':2,'U16':2,'F16':2,'BF16':2,'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def _pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('Duplicate JSON field refused')
        result[key]=value
    return result
def _regular(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Independent regular exported file required: '+str(path))
    return path
def stat(path):
    value=_regular(path).stat()
    return {name:getattr(value,name) for name in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
def sha(path,check=None):
    path=_regular(path);before=stat(path);h=hashlib.sha256()
    with path.open('rb') as handle:
        while chunk:=handle.read(2**20):
            if check is not None:check()
            h.update(chunk)
    if stat(path)!=before:raise ValueError('Export changed while hashing')
    return h.hexdigest()
def read(path):
    path=_regular(path)
    if path.stat().st_size>MAX_METADATA:raise ValueError('Bounded JSON metadata required')
    return json.loads(path.read_bytes(),object_pairs_hook=_pairs)
def software():return {name:version(name) for name in ('torch','transformers','safetensors','tokenizers')}
def code_binding():
    for name,h in FROZEN.items():
        if sha(ROOT/name)!=h:raise ValueError('Byte-pinned canonical source changed: '+name)
    return {**FROZEN,**{name:sha(ROOT/name) for name in (SOURCE,TRAIN_CLI,EVAL_CLI,TEST)}}
def _safe_name(name):
    if not isinstance(name,str) or Path(name).name!=name or name in ('.','..') or '\\' in name:
        raise ValueError('Export filename escaped its immutable directory')
    return name


def read_weight_header(path):
    """Only bounded safetensors header, never tensor bodies or original blobs."""
    path=_regular(path);before=stat(path)
    with path.open('rb') as handle:
        size=handle.read(8)
        if len(size)!=8:raise ValueError('Missing safetensors header length')
        length=struct.unpack('<Q',size)[0]
        if not 2<=length<=MAX_METADATA or length+8>before['st_size']:raise ValueError('Invalid bounded safetensors header')
        raw=handle.read(length)
    if len(raw)!=length or stat(path)!=before:raise ValueError('Export header changed/truncated')
    header=json.loads(raw,object_pairs_hook=_pairs);header.pop('__metadata__',None)
    return header,length


def resolve_derived_vision(repo,*,revision=None,export_root=DEFAULT_EXPORT_ROOT,preprocessing_profile=None):
    """Resolve only the two released exports. No Hub lookup or original-file read."""
    if repo not in DONORS:raise ValueError('Derived resolver supports only declared Qwen2.5/Gemma4 donors')
    spec=DONORS[repo];revision=revision or spec['revision']
    if revision!=spec['revision']:raise ValueError('Derived donor source revision differs')
    if preprocessing_profile is not None and (spec['backend']!='qwen2_5_vl' or preprocessing_profile not in PROFILE_BUDGETS):
        raise ValueError('Unsupported explicit derived preprocessing profile')
    directory=Path(export_root).expanduser()/f"{spec['donor']}-{revision}"
    if directory.is_symlink() or not directory.is_dir() or any(p.is_symlink() for p in directory.parents):
        raise ValueError('Immutable local export directory must not traverse symlinks')
    directory=directory.resolve();manifest_path=directory/'derivation.json';manifest=read(manifest_path)
    if sha(manifest_path)!=spec['derivation_sha256']:raise ValueError('Released export derivation digest differs')
    source=manifest['source'];native=manifest['native_validation']
    if (manifest.get('artifact_scope')!='vision_only_donor_weights_not_full_multimodal_checkpoint' or
        source.get('repo')!=repo or source.get('revision')!=revision or source.get('backend')!=spec['backend'] or
        manifest.get('donor_id')!=spec['donor'] or manifest.get('source_identity_sha256')!=digest(source) or
        manifest.get('tensors_sha256')!=digest(manifest['tensors']) or manifest.get('software')!=software() or
        native.get('status')!='passed' or native.get('all_native_state_keys_and_bits_equal') is not True or
        native.get('strict_native_load_state_dict') is not True or native.get('feature_dim')!=spec['feature_dim'] or
        native.get('llm_instantiated') is not False or native.get('gpu_initialized') is not False):
        raise ValueError('Released source/export/native CPU contract differs')
    for name,h in manifest['export_code_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Exported native extraction source differs')
    outputs=manifest['output_files'];expected=set(outputs)|{'derivation.json'}
    if {p.name for p in directory.iterdir()}!=expected:raise ValueError('Unexpected/missing derived export files')
    observations={}
    for name,item in outputs.items():
        path=_regular(directory/_safe_name(name));observations[name]=stat(path)
        if path.stat().st_size!=item['bytes']:raise ValueError('Exported file size differs')
        if name!='model.safetensors' and item['bytes']<=MAX_METADATA and sha(path)!=item['sha256']:
            raise ValueError('Exported config/processor metadata differs')
    config=read(directory/'config.json')
    if config.get('model_type')!=spec['backend']:raise ValueError('Exported donor config backend differs')
    index=read(directory/'model.safetensors.index.json');tensors=manifest['tensors']
    if index.get('weight_map')!={name:'model.safetensors' for name in tensors}:raise ValueError('Self-contained vision-only index differs')
    prefixes=source['prefixes']
    header,length=read_weight_header(directory/'model.safetensors')
    if set(header)!=set(tensors) or any(not any(name.startswith(prefix) for prefix in prefixes) for name in header):
        raise ValueError('Derived header contains missing/unexpected non-vision tensors')
    intervals=[]
    for name,value in header.items():
        declared=tensors[name];dtype=value.get('dtype');shape=value.get('shape');offset=value.get('data_offsets')
        if (dtype!=declared['safetensors_dtype'] or shape!=declared['shape'] or dtype not in _BYTES or
            not isinstance(offset,list) or len(offset)!=2 or any(type(v) is not int for v in offset) or
            offset[0]<0 or offset[1]-offset[0]!=math.prod(shape)*_BYTES[dtype]):
            raise ValueError('Derived header dtype/shape/offset differs')
        intervals.append(tuple(offset))
    cursor=0
    for start,end in sorted(intervals):
        if start!=cursor:raise ValueError('Derived tensor spans overlap or contain gaps')
        cursor=end
    if cursor+length+8!=observations['model.safetensors']['st_size']:raise ValueError('Derived payload size/header differs')
    upstream_ref={'repo':repo,'revision':revision}
    export_ref={'schema_version':1,'source_ref':upstream_ref,'donor':spec['donor'],'backend':spec['backend'],
        'derivation_sha256':spec['derivation_sha256'],'source_identity_sha256':manifest['source_identity_sha256'],
        'tensors_sha256':manifest['tensors_sha256'],'feature_dim':spec['feature_dim'],
        'output_files':copy.deepcopy(outputs),'software':software()}
    profile_ref={'schema_version':1,'name':preprocessing_profile or 'native',
        'max_visual_tokens':PROFILE_BUDGETS.get(preprocessing_profile),
        'extraction_code_sha256':FROZEN['src/model/research_vision.py'],
        'feature_point':'native_spatial_merger_output_before_our_mlp' if spec['backend']=='qwen2_5_vl' else 'frozen_donor_multimodal_embedder_output_before_our_mlp',
        'compute_dtype':'bf16','storage_dtype':'bf16','image_decode_policy':'PIL_EXIF_transpose_RGB'}
    cache_contract={'name':'derived_vision_cache_v1','schema_version':1,'upstream_ref':upstream_ref,
        'export_ref':export_ref,'profile_ref':profile_ref,'no_legacy_cache_reuse':True}
    # Match the frozen cache class's own schema augmentation, with a distinct name.
    fingerprint=digest({'cache_schema_version':1,**cache_contract})
    return {'owner':OWNER,'path':str(directory),'upstream_ref':upstream_ref,'export_ref':export_ref,
        'profile_ref':profile_ref,'cache_contract':cache_contract,'cache_fingerprint':fingerprint,
        'manifest_stat':stat(manifest_path),'file_stats':observations,'metadata_and_header_verified':True,
        'whole_export_bytes_verified':False,'original_weight_files_opened':False,
        'actual_CUDA_integration_verified':False,'pruning_authorized':False}


def assert_binding(binding,*,full=False,check=None):
    profile=binding['profile_ref']['name'];profile=None if profile=='native' else profile
    directory=Path(binding['path'])
    current=resolve_derived_vision(binding['upstream_ref']['repo'],revision=binding['upstream_ref']['revision'],
        export_root=directory.parent,preprocessing_profile=profile)
    if current!=binding:raise ValueError('Derived source/export/profile/stat binding changed')
    if full:
        for name,item in binding['export_ref']['output_files'].items():
            if sha(directory/name,check)!=item['sha256']:raise ValueError('Full derived file checksum differs')
    return {'full_derived_files_verified':full,'original_weight_files_opened':False}


def config_contract(binding):
    return {'derived_vision_schema':OWNER,'vision_export_ref':copy.deepcopy(binding['export_ref']),
        'vision_profile_ref':copy.deepcopy(binding['profile_ref']),
        'vision_cache_contract':copy.deepcopy(binding['cache_contract']),
        'vision_cache_fingerprint':binding['cache_fingerprint'],'derived_vision_code_sha256':code_binding()}


def configure_args(args,binding):
    if args.vision!=binding['upstream_ref']['repo']:raise ValueError('Model vision repo differs from derived source')
    expected_profile=None if binding['profile_ref']['name']=='native' else binding['profile_ref']['name']
    if getattr(args,'vision_preprocessing_profile',None)!=expected_profile:raise ValueError('Model processor profile differs')
    for name,value in config_contract(binding).items():
        if hasattr(args,name) and getattr(args,name)!=value:raise ValueError('Saved derived config differs: '+name)
        setattr(args,name,copy.deepcopy(value))
    return args


def validate_payload_binding(payload,binding):
    expected=config_contract(binding)
    model_ref=payload.get('model_ref')
    if (not isinstance(model_ref,dict) or set(model_ref)!={'repo','revision'} or
        model_ref.get('repo')!=payload.get('config',{}).get('model') or
        not re.fullmatch('[0-9a-f]{40,64}',str(model_ref.get('revision','')))):
        raise ValueError('Checkpoint pinned language model_ref missing or differs')
    if payload.get('vision_ref')!=binding['upstream_ref']:raise ValueError('Checkpoint upstream vision_ref differs')
    if payload.get('cache_fingerprint')!=binding['cache_fingerprint']:raise ValueError('Checkpoint derived cache fingerprint differs')
    for name,value in expected.items():
        if payload.get('config',{}).get(name)!=value or payload.get(name)!=value:
            raise ValueError('Checkpoint config/payload derived binding absent or differs: '+name)
    if payload['config'].get('vision')!=binding['upstream_ref']['repo']:
        raise ValueError('Checkpoint config vision differs from source pin')
    return True


def bind_checkpoint_payload(payload,binding):
    result=dict(payload)
    for name,value in config_contract(binding).items():
        if name in result and result[name]!=value:raise ValueError('Existing payload derived binding differs: '+name)
        result[name]=copy.deepcopy(value)
    validate_payload_binding(result,binding)
    return result


class DeferredConstructionCache:
    """No directory creation/read/write; any constructor feature use is a bug."""
    def __init__(self,*args,**kwargs):self.touched=False
    def __getattr__(self,name):
        self.touched=True
        raise RuntimeError('Frozen constructor attempted feature-cache access before derived replacement: '+name)


def _local_language_resolver(repo,expected=None):
    if expected is None:
        inventory=read(ROOT/'research_results/model_inventory.json');rows=[r for r in inventory if r['repo']==repo]
        if len(rows)!=1:raise ValueError('Exactly one pinned language inventory row required')
        expected={'repo':repo,'revision':rows[0]['sha']}
    if (expected.get('repo')!=repo or set(expected)!={'repo','revision'} or
        not re.fullmatch('[0-9a-f]{40,64}',str(expected.get('revision','')))):
        raise ValueError('Pinned language source reference required')
    from huggingface_hub import hf_hub_download
    config=hf_hub_download(repo,'config.json',revision=expected['revision'],local_files_only=True)
    return str(Path(config).parent),dict(expected)


def _private_runner():
    code_binding();path=ROOT/'scripts/run_research_stage.py';raw=path.read_bytes()
    module=ModuleType('scripts._derived_vision_v1_private_runner')
    module.__file__=str(path);module.__package__='scripts'
    exec(compile(raw,str(path),'exec'),module.__dict__)
    if sha(path)!=FROZEN['scripts/run_research_stage.py']:raise ValueError('Frozen runner changed during private clone')
    return module


def install_private_runner_hooks(module,binding,*,model_repo,model_ref=None):
    """Only a private module receives hooks; no canonical global is patched."""
    base=module.ResearchModel;manager=module.BestLastCheckpointManager
    if module.__name__=='scripts.run_research_stage':raise ValueError('Canonical runner cannot receive derived hooks')
    if model_repo==binding['upstream_ref']['repo']:raise ValueError('Vision-only export cannot be the language backbone')
    def resolver(repo):
        if repo==binding['upstream_ref']['repo']:
            assert_binding(binding);return binding['path'],dict(binding['upstream_ref'])
        if repo!=model_repo:raise ValueError('Unrecognized repository requested by private model')
        return _local_language_resolver(repo,model_ref)
    module.resolve_model=resolver;module.VisionFeatureCache=DeferredConstructionCache
    class DerivedResearchModel(base):
        def __init__(self,args):
            configure_args(args,binding)
            base.__init__(self,args)
            if not isinstance(self.cache,DeferredConstructionCache) or self.cache.touched:
                raise RuntimeError('Constructor touched/replaced deferred feature cache')
            if self.vision_ref!=binding['upstream_ref']:raise ValueError('Constructor changed upstream vision_ref')
            if (self.extractor.backend!=binding['export_ref']['backend'] or
                self.extractor.feature_dim!=binding['export_ref']['feature_dim']):
                raise ValueError('Actual derived encoder backend/feature width differs')
            from src.dataset.vision_cache import VisionFeatureCache
            self.cache=VisionFeatureCache(Path(args.cache_dir)/'derived_vision_v1',binding['cache_fingerprint'],
                reserve_bytes=int(args.reserve_gb*2**30),ram_max_bytes=2*2**30)
            self.vision_export_ref=copy.deepcopy(binding['export_ref']);self.vision_profile_ref=copy.deepcopy(binding['profile_ref'])
    class DerivedManager(manager):
        def save_torch(self,payload,*args,**kwargs):
            assert_binding(binding)
            return super().save_torch(bind_checkpoint_payload(payload,binding),*args,**kwargs)
    original_validate=module.validate_model_binding
    def validate(payload,model):
        validate_payload_binding(payload,binding);return original_validate(payload,model)
    module.ResearchModel=DerivedResearchModel;module.BestLastCheckpointManager=DerivedManager
    module.validate_model_binding=validate
    return module


def make_training_runner(args,binding,*,model_ref=None):
    if not math.isfinite(args.reserve_gb) or args.reserve_gb<120:raise ValueError('Preserve120GiB disk reserve')
    assert_binding(binding,full=True,check=host_guard)
    configure_args(args,binding)
    return install_private_runner_hooks(_private_runner(),binding,model_repo=args.model,model_ref=model_ref)


def host_guard():
    available=None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):available=int(line.split()[1])*1024
    maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if maximum!='max':available=min(available,int(maximum)-int(Path('/sys/fs/cgroup/memory.current').read_text()))
    if available is None or available<32*2**30:raise MemoryError('Derived loading preserves32GiB actual host/cgroup headroom; no cache credit')
    return available


def load_derived_checkpoint(checkpoint,*,export_root=DEFAULT_EXPORT_ROOT,cache_dir='cache/vision',reserve_gb=120):
    """NEW opt-in loader; saved export/source/profile/cache is authoritative."""
    if not math.isfinite(reserve_gb) or reserve_gb<120:raise ValueError('Preserve120GiB disk reserve')
    code_binding();host_guard()
    from src.utils.research_checkpoint_compat import load_research_payload
    from src.eval.research_inference import _validate_learned_state,freeze_for_generation
    payload=load_research_payload(Path(checkpoint),map_location='cpu',mmap=True)
    config=copy.deepcopy(payload['config']);ref=payload.get('vision_ref',{})
    binding=resolve_derived_vision(ref.get('repo'),revision=ref.get('revision'),export_root=export_root,
        preprocessing_profile=config.get('vision_preprocessing_profile'))
    validate_payload_binding(payload,binding)
    args=SimpleNamespace(**config);args.cache_dir=str(cache_dir);args.reserve_gb=float(reserve_gb)
    module=make_training_runner(args,binding,model_ref=payload.get('model_ref'))
    model=module.ResearchModel(args)
    module.validate_model_binding(payload,model);_validate_learned_state(model,payload['model'])
    result=model.load_state_dict(payload['model'],strict=False)
    if result.unexpected_keys:raise ValueError('Unknown derived-checkpoint state keys')
    provenance={'model_ref':payload['model_ref'],'vision_ref':payload['vision_ref'],'recipe':config['recipe'],
        'stage':config['stage'],'training_step':payload['step'],'cache_fingerprint':model.cache.fingerprint,
        'derived_vision_schema':OWNER,'vision_export_ref':binding['export_ref'],'vision_profile_ref':binding['profile_ref']}
    del payload
    return freeze_for_generation(model),provenance


def metadata_preflight(repo,*,revision=None,export_root=DEFAULT_EXPORT_ROOT,preprocessing_profile=None):
    binding=resolve_derived_vision(repo,revision=revision,export_root=export_root,preprocessing_profile=preprocessing_profile)
    return {'schema_version':1,'owner':OWNER,'status':'metadata_header_preflight_only',
        'binding':binding,'config_contract':config_contract(binding),'Torch_imported':'torch' in sys.modules,
        'whole_export_bytes_verified':False,'GPU_launched':False,'actual_CUDA_integration_verified':False,
        'pruning_authorized':False}
