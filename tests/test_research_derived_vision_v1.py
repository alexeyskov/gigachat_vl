"""Local-only metadata, real tiny models, private hooks and strict resume tests."""
import copy
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
from types import ModuleType,SimpleNamespace

import pytest
from src.utils import research_derived_vision_v1 as d


@pytest.fixture
def export(tmp_path,monkeypatch):
    repo='Qwen/Qwen2.5-VL-7B-Instruct';revision=d.DONORS[repo]['revision']
    root=tmp_path/'exports';directory=root/('qwen25vision-'+revision);directory.mkdir(parents=True)
    def write(name,value):
        path=directory/name;path.write_bytes(value if isinstance(value,bytes) else d.canonical(value))
        return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    key='visual.proj.weight';header={key:{'dtype':'F32','shape':[2,2],'data_offsets':[0,16]}}
    raw=d.canonical(header);weights=struct.pack('<Q',len(raw))+raw+struct.pack('<4f',1.,2.,3.,4.)
    outputs={'config.json':write('config.json',{'model_type':'qwen2_5_vl'}),
        'preprocessor_config.json':write('preprocessor_config.json',{'patch_size':14,'merge_size':2}),
        'model.safetensors':write('model.safetensors',weights),
        'model.safetensors.index.json':write('model.safetensors.index.json',{'weight_map':{key:'model.safetensors'}})}
    source={'repo':repo,'revision':revision,'backend':'qwen2_5_vl','prefixes':['visual.'],
        'source_files':{'original_missing.safetensors':{'sha256':'a'*64,'bytes':999}}}
    tensors={key:{'safetensors_dtype':'F32','shape':[2,2]}}
    manifest={'artifact_scope':'vision_only_donor_weights_not_full_multimodal_checkpoint','donor_id':'qwen25vision',
        'source':source,'source_identity_sha256':d.digest(source),'tensors':tensors,'tensors_sha256':d.digest(tensors),
        'software':d.software(),'export_code_sha256':{},'output_files':outputs,
        'native_validation':{'status':'passed','all_native_state_keys_and_bits_equal':True,
            'strict_native_load_state_dict':True,'feature_dim':3,'llm_instantiated':False,'gpu_initialized':False}}
    write('derivation.json',manifest)
    spec=dict(d.DONORS[repo],feature_dim=3,derivation_sha256=d.sha(directory/'derivation.json'))
    monkeypatch.setitem(d.DONORS,repo,spec)
    return repo,root,directory,manifest


def resolve(export,**kwargs):
    repo,root,_,_=export
    return d.resolve_derived_vision(repo,export_root=root,**kwargs)


def test_fresh_import_and_metadata_preflight_never_import_torch_or_network():
    result=subprocess.run([sys.executable,'-c',"import sys; from src.utils import research_derived_vision_v1 as d; "
        "assert 'torch' not in sys.modules; r=d.metadata_preflight('google/gemma-4-E4B-it'); "
        "assert r['GPU_launched'] is False and 'torch' not in sys.modules"],cwd=d.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_original_absent_and_network_trap_metadata_only_reads_export_header(export,monkeypatch):
    _,_,directory,_=export;original=Path.open;reads=[]
    class HeaderOnly:
        def __init__(self,handle):self.handle=handle
        def __enter__(self):return self
        def __exit__(self,*args):self.handle.close()
        def read(self,n=-1):
            assert 0<=n<=d.MAX_METADATA
            reads.append(n);return self.handle.read(n)
    def opened(path,*args,**kwargs):
        path=Path(path)
        assert 'original_missing' not in str(path)
        handle=original(path,*args,**kwargs)
        return HeaderOnly(handle) if path==directory/'model.safetensors' else handle
    monkeypatch.setattr(Path,'open',opened)
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('Network forbidden'))
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k:pytest.fail('Network socket forbidden'))
    b=resolve(export)
    assert b['upstream_ref']['repo']==export[0] and b['metadata_and_header_verified']
    assert not b['whole_export_bytes_verified'] and reads[0]==8 and len(reads)==2
    assert not (directory/'original_missing.safetensors').exists()


@pytest.mark.parametrize('mutation',['source_revision','unknown_repo','profile','missing_file','symlink','config','index','header','weight_size','manifest'])
def test_wrong_source_export_config_header_and_profile_refuse(export,mutation):
    repo,root,directory,_=export
    if mutation=='source_revision':
        with pytest.raises(ValueError):d.resolve_derived_vision(repo,revision='0'*40,export_root=root)
        return
    if mutation=='unknown_repo':
        with pytest.raises(ValueError):d.resolve_derived_vision('some/model',export_root=root)
        return
    if mutation=='profile':
        with pytest.raises(ValueError):resolve(export,preprocessing_profile='unknown')
        return
    path=directory/('config.json' if mutation in ('config','symlink') else 'model.safetensors.index.json' if mutation=='index' else 'model.safetensors' if mutation in ('header','weight_size') else 'derivation.json')
    if mutation=='missing_file':path.unlink()
    elif mutation=='symlink':
        raw=path.read_bytes();path.unlink();copy=directory.parent/'external.json';copy.write_bytes(raw);path.symlink_to(copy)
    elif mutation=='weight_size':path.write_bytes(path.read_bytes()+b'!')
    else:
        raw=bytearray(path.read_bytes());raw[min(15,len(raw)-1)]^=1;path.write_bytes(raw)
    with pytest.raises((ValueError,KeyError,UnicodeDecodeError)):resolve(export)


def test_full_export_verification_detects_same_size_tensor_body_change(export):
    b=resolve(export);proof=d.assert_binding(b,full=True)
    assert proof['full_derived_files_verified'] and not proof['original_weight_files_opened']
    path=Path(b['path'])/'model.safetensors';raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
    # A freshly resolved metadata binding still cannot certify edited tensor bits.
    edited=resolve(export)
    with pytest.raises(ValueError,match='checksum'):d.assert_binding(edited,full=True)


def test_content_addressed_cache_profile_and_relocation_are_explicit(export,tmp_path):
    import shutil
    b=resolve(export);c=resolve(export,preprocessing_profile='qwen_native_bounded_729')
    assert b['cache_fingerprint']!=c['cache_fingerprint'] and b['export_ref']==c['export_ref']
    target=tmp_path/'relocated';shutil.copytree(export[1],target)
    relocated=d.resolve_derived_vision(export[0],export_root=target)
    assert b['cache_fingerprint']==relocated['cache_fingerprint']
    assert b['export_ref']==relocated['export_ref'] and b['path']!=relocated['path']
    assert b['cache_contract']['no_legacy_cache_reuse'] is True


def args(binding,tmp_path):
    return SimpleNamespace(model='Tiiny/SmallThinker-4BA0.6B-Instruct',vision=binding['upstream_ref']['repo'],
        vision_preprocessing_profile=None,cache_dir=str(tmp_path/'cache'),reserve_gb=120.)


def payload(binding,tmp_path):
    a=d.configure_args(args(binding,tmp_path),binding)
    return d.bind_checkpoint_payload({'config':vars(a),'vision_ref':binding['upstream_ref'],
        'model_ref':{'repo':a.model,'revision':'d'*40},
        'cache_fingerprint':binding['cache_fingerprint']},binding)


@pytest.mark.parametrize('field',['model_ref','vision_ref','cache_fingerprint',*d.CONFIG_FIELDS])
def test_saved_export_source_profile_cache_and_code_binding_strict(export,tmp_path,field):
    b=resolve(export);p=payload(b,tmp_path);assert d.validate_payload_binding(p,b)
    p[field]={'foreign':True}
    with pytest.raises(ValueError):d.validate_payload_binding(p,b)
    p=payload(b,tmp_path);p['config'][field]={'foreign':True}
    if field not in ('model_ref','vision_ref','cache_fingerprint'):
        with pytest.raises(ValueError):d.validate_payload_binding(p,b)


def test_legacy_payload_and_wrong_config_never_silently_migrate(export,tmp_path):
    b=resolve(export);p=payload(b,tmp_path);p.pop('vision_export_ref')
    with pytest.raises(ValueError):d.validate_payload_binding(p,b)
    a=args(b,tmp_path);a.vision_export_ref={'foreign':True}
    with pytest.raises(ValueError):d.configure_args(a,b)


def fake_runner(binding,*,touch=False):
    module=ModuleType('synthetic_private_runner')
    class Base:
        def __init__(self,a):
            path,self.vision_ref=module.resolve_model(a.vision)
            assert path==binding['path']
            self.extractor=SimpleNamespace(backend=binding['export_ref']['backend'],feature_dim=binding['export_ref']['feature_dim'])
            self.cache=module.VisionFeatureCache(a.cache_dir,'legacy_fingerprint')
            if touch:self.cache.get_many([],None)
    class Manager:
        def save_torch(self,payload,*args,**kwargs):return payload
    module.ResearchModel=Base;module.BestLastCheckpointManager=Manager
    module.validate_model_binding=lambda p,m:True
    return module


def test_private_constructor_never_reads_writes_or_retags_legacy_cache(export,tmp_path):
    b=resolve(export);a=args(b,tmp_path);old=Path(a.cache_dir)/'legacy_fingerprint';old.mkdir(parents=True)
    sentinel=old/'original.pt';sentinel.write_bytes(b'immutable old feature bytes')
    m=fake_runner(b);d.install_private_runner_hooks(m,b,model_repo=a.model);model=m.ResearchModel(a)
    assert model.vision_ref==b['upstream_ref'] and model.vision_export_ref==b['export_ref']
    assert model.cache.root==Path(a.cache_dir)/'derived_vision_v1'/b['cache_fingerprint']
    assert model.cache.stats['requests']==model.cache.stats['bytes_written']==0
    assert sentinel.read_bytes()==b'immutable old feature bytes' and list(old.iterdir())==[sentinel]
    saved=m.BestLastCheckpointManager().save_torch({'config':vars(a),'vision_ref':model.vision_ref,
        'model_ref':{'repo':a.model,'revision':'d'*40},'cache_fingerprint':model.cache.fingerprint})
    assert d.validate_payload_binding(saved,b)
    with pytest.raises(ValueError):m.validate_model_binding(dict(saved,vision_export_ref={}),model)


def test_constructor_feature_use_and_canonical_hooking_refuse(export,tmp_path):
    b=resolve(export);m=fake_runner(b,touch=True);d.install_private_runner_hooks(m,b,model_repo='language')
    with pytest.raises(RuntimeError,match='before derived replacement'):m.ResearchModel(args(b,tmp_path))
    m=fake_runner(b);m.__name__='scripts.run_research_stage'
    with pytest.raises(ValueError,match='Canonical'):d.install_private_runner_hooks(m,b,model_repo='language')


def test_private_runner_clone_preserves_canonical_globals():
    import scripts.run_research_stage as canonical
    before={key:getattr(canonical,key) for key in ('ResearchModel','resolve_model','VisionFeatureCache','BestLastCheckpointManager')}
    private=d._private_runner()
    assert private.__name__!='scripts.run_research_stage' and private not in sys.modules.values()
    assert all(getattr(canonical,key) is value for key,value in before.items())


def test_frozen_native_loader_math_tiny_real_vision_only_index_without_original(tmp_path,monkeypatch):
    import torch
    from safetensors.torch import save_file
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
    from src.model.research_vision import _StrictVisionFeatureExtractor
    config=Qwen2_5_VLVisionConfig(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
        patch_size=14,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=16)
    torch.manual_seed(42);native=Qwen2_5_VisionTransformerPretrainedModel(config).eval()
    state={'visual.'+k:v for k,v in native.state_dict().items()};save_file(state,str(tmp_path/'model.safetensors'))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'model.safetensors' for k in state}}))
    extractor=object.__new__(_StrictVisionFeatureExtractor)
    extractor.backend='qwen2_5_vl';extractor.vision_model=str(tmp_path);extractor.config=SimpleNamespace(vision_config=config)
    extractor.device=torch.device('cpu');extractor.dtype=torch.float32
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('Native direct export load must stay offline'))
    extractor._load_modules()
    assert extractor.loaded_weight_files==['model.safetensors']
    assert not any(p.requires_grad for p in extractor.vision_tower.parameters())
    for key,value in native.state_dict().items():assert torch.equal(value,extractor.vision_tower.state_dict()[key])
    pixels=torch.randn(4,3*2*14*14);grid=torch.tensor([[1,2,2]])
    with torch.no_grad():
        left=native(pixels,grid_thw=grid);right=extractor.vision_tower(pixels,grid_thw=grid)
    assert torch.equal(left.pooler_output,right.pooler_output) and left.pooler_output.shape[-1]==16
    assert not torch.cuda.is_initialized()


def test_real_tiny_MoE_checkpoint_new_loader_restores_tokens_and_new_cache_only(export,tmp_path,monkeypatch):
    import torch
    from tests.test_research_inference import TinyResearchModel,record
    from src.eval.research_inference import freeze_for_generation,generate_prediction
    from src.utils.research_storage import BestLastCheckpointManager
    b=resolve(export);a=args(b,tmp_path);a.model='synthetic/CPU-native-MoE';a.stage='SFT';a.recipe='3c';a.max_length=64
    model_ref={'repo':a.model,'revision':'d'*40}
    language_calls=[]
    def language(repo,expected=None):
        assert repo==a.model and expected==model_ref
        language_calls.append(repo);return str(tmp_path),model_ref
    monkeypatch.setattr(d,'_local_language_resolver',language)
    monkeypatch.setattr(d,'host_guard',lambda:100*2**30)
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('No network in checkpoint replay'))
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k:pytest.fail('No network socket in checkpoint replay'))
    def module():
        m=ModuleType('synthetic_private_native_MoE')
        class Model(TinyResearchModel):
            def __init__(self,args):
                super().__init__(wrapped=True);self.args=args
                self.model_ref=m.resolve_model(args.model)[1];path,self.vision_ref=m.resolve_model(args.vision)
                assert path==b['path']
                self.extractor=SimpleNamespace(backend='qwen2_5_vl',feature_dim=3)
                self.cache=m.VisionFeatureCache(args.cache_dir,'legacy_never_used')
                self.encoder_calls=0
            def encode(self,paths):
                self.encoder_calls+=1
                return [torch.tensor([[1.,2.,3.],[4.,5.,6.]]) for _ in paths]
        m.ResearchModel=Model;m.BestLastCheckpointManager=BestLastCheckpointManager
        m.validate_model_binding=lambda p,model:None
        return m
    monkeypatch.setattr(d,'_private_runner',module)
    torch.manual_seed(5);private=d.make_training_runner(a,b,model_ref=model_ref)
    source=private.ResearchModel(a)
    p={'config':copy.deepcopy(vars(a)),'model_ref':model_ref,'vision_ref':source.vision_ref,
       'cache_fingerprint':source.cache.fingerprint,'step':1,'model':source.state_dict()}
    checkpoint=tmp_path/'source.pt';torch.save(d.bind_checkpoint_payload(p,b),checkpoint)
    source=freeze_for_generation(source);first=generate_prediction(source,record(),max_new_tokens=4)
    assert source.encoder_calls==1
    restored,provenance=d.load_derived_checkpoint(checkpoint,export_root=export[1],cache_dir=a.cache_dir)
    second=generate_prediction(restored,record(),max_new_tokens=4)
    assert first==second and restored.encoder_calls==0 and restored.cache.stats['hits']==1
    assert provenance['vision_ref']==b['upstream_ref'] and provenance['vision_export_ref']==b['export_ref']
    assert len(language_calls)==2
    for name,tensor in source.state_dict().items():assert torch.equal(tensor,restored.state_dict()[name])
    assert not any(p.requires_grad for p in restored.parameters()) and not torch.cuda.is_initialized()
    assert not (Path(a.cache_dir)/'legacy_never_used').exists()


def test_manager_does_not_normalize_existing_wrong_export_payload(export,tmp_path):
    b=resolve(export);p=payload(b,tmp_path);p['vision_profile_ref']={'foreign':True}
    with pytest.raises(ValueError,match='Existing payload'):d.bind_checkpoint_payload(p,b)


def test_language_resolution_is_pinned_config_only_and_local_only(tmp_path,monkeypatch):
    import huggingface_hub
    path=tmp_path/'config.json';path.write_text('{}')
    calls=[]
    def local(repo,filename,**kwargs):
        assert filename=='config.json' and kwargs=={'revision':'a'*40,'local_files_only':True}
        calls.append(repo);return str(path)
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',local)
    ref={'repo':'synthetic/language','revision':'a'*40}
    resolved,source=d._local_language_resolver(ref['repo'],ref)
    assert resolved==str(tmp_path) and source==ref and calls==[ref['repo']]
    with pytest.raises(ValueError):d._local_language_resolver(ref['repo'],dict(ref,revision='main'))


def test_actual_tiny_full_strict_encoder_processor_one_RGB_no_original_no_network(tmp_path,monkeypatch):
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig,Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from src.model.research_vision import build_frozen_vision
    config=Qwen2_5_VLVisionConfig(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
        patch_size=14,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=16)
    Qwen2_5_VLConfig(vision_config=config.to_dict()).save_pretrained(tmp_path)
    Qwen2VLImageProcessor(min_pixels=3136,max_pixels=3136).save_pretrained(tmp_path)
    tokens=['<unk>','<|image_pad|>','<|video_pad|>','<|vision_start|>','<|vision_end|>']
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({name:i for i,name in enumerate(tokens)},unk_token='<unk>')),
        unk_token='<unk>',pad_token='<unk>',additional_special_tokens=tokens[1:])
    tokenizer.save_pretrained(tmp_path)
    torch.manual_seed(1);native=Qwen2_5_VisionTransformerPretrainedModel(config)
    state={'visual.'+k:v for k,v in native.state_dict().items()};save_file(state,str(tmp_path/'model.safetensors'))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'model.safetensors' for k in state}}))
    path=tmp_path/'synthetic.png';Image.new('RGB',(112,84),(31,57,91)).save(path)
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('Strict local processor must never access network'))
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k:pytest.fail('Strict local processor must never open network socket'))
    encoder=build_frozen_vision(str(tmp_path),'cpu',torch.float32)
    first=encoder.encode_paths([str(path)])[0];second=encoder.encode_paths([str(path)])[0]
    assert first['features'].shape[-1]==16 and torch.isfinite(first['features']).all()
    assert torch.equal(first['features'],second['features']) and first['metadata']==second['metadata']
    assert not any(p.requires_grad for p in encoder.vision_tower.parameters())
    assert not torch.cuda.is_initialized()
