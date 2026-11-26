"""Private stage eligibility and unchanged caption math; synthetic CPU fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.eval import research_caption_generation as legacy
from src.eval import research_caption_generation_v2 as v2
from src.eval import research_caption_stages as stages
from tests.test_research_caption_generation import example,FakeRuntime,fake_scores,interval,write


def configure_stage(example, monkeypatch, stage='A0', recipe='4q'):
    import torch
    protocol=json.loads((example.bundle/'protocol.json').read_text())
    protocol.update(protocol='research_caption_pilot_eval_v2',preparation_profile='caption_pilot_en_v2_signal_safe')
    protocol['preparation_source_sha256']={str(stages.ROOT/name):legacy.file_sha(stages.ROOT/name) for name in
        ('src/eval/research_caption_data_v2.py','scripts/prepare_research_caption_eval_v2.py')}
    write(example.bundle/'protocol.json',protocol)
    monkeypatch.setattr(stages,'DATA_PROTOCOL_SHA256',legacy.file_sha(example.bundle/'protocol.json'))
    monkeypatch.setattr(stages,'RECORDS',3)
    monkeypatch.setattr(stages,'_memory_guard',lambda:64*2**30)
    config=json.loads((example.run/'resolved_config.json').read_text());config.update(stage=stage,recipe=recipe)
    write(example.run/'resolved_config.json',config)
    (example.run/'checkpoint_compaction.json').unlink()
    marker=example.source.parent/'.research_checkpoint.json'
    write(marker,{'owner':'fixture_owner','step':170,'created_unix':0.})
    torch.save({'fixture':'synthetic eligibility only; runtime injected explicitly'},example.source)
    archive=example.root/'source.zip';archive.write_bytes(b'synthetic source archive bytes')
    provenance={'resolved_config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        'model_ref':{'repo':'fixture/model','revision':'a'*40},'vision_ref':{'repo':'fixture/vision','revision':'b'*40},
        'cache_fingerprint':'f'*64,'code_sha256':'c'*64,'source_archive':str(archive),'source_archive_sha256':legacy.file_sha(archive),
        'capture_scope':'synthetic CPU metadata fixture; not a real training run'}
    write(example.run/'provenance_latest.json',provenance);write(example.run/'provenance.json',provenance)
    original=stages._private_owner
    def cpu_private():
        owner=original();owner.gpu_idle=lambda:None;return owner
    monkeypatch.setattr(stages,'_private_owner',cpu_private)
    example.context=stages.prepare_context(example.run,example.bundle,assets=example.assets)
    example.output=stages.output_path(example.context)
    return example


@pytest.fixture
def current(example,monkeypatch):return configure_stage(example,monkeypatch)


class StageRuntime(FakeRuntime):
    def load(self,*args):
        self.loads+=1
        context=self.example.context;model=context['static']['checkpoint'];prov=model['stage_training_provenance']
        return object(),{'stage':model['config']['stage'],'recipe':model['config']['recipe'],
            'training_step':model['roles'][model['role']]['marker']['step'],
            **{key:prov[key] for key in ('model_ref','vision_ref','cache_fingerprint')}}


def globals_snapshot():
    return {name:{field:getattr(module,field) for field in ('OWNER','DATA_PROTOCOL_SHA256','RECORDS')}
        for name,module in (('legacy',legacy),('v2',v2))}


@pytest.mark.parametrize('stage,recipe',[('A0','4f'),('A0','4q'),('A1','4f'),('A1','1c')])
def test_completed_raw_stages_need_no_compaction_and_keep_frozen_math(example,monkeypatch,stage,recipe):
    current=configure_stage(example,monkeypatch,stage,recipe);before=globals_snapshot();owner=stages._private_owner()
    for name in ('prepare_context','evaluate','_row','_references','committed_rows'):
        assert getattr(owner,name).__code__.co_code==getattr(legacy,name).__code__.co_code
    assert owner.NativeRuntime.load.__code__.co_code==legacy.NativeRuntime.load.__code__.co_code
    assert owner.NativeRuntime.generate.__code__.co_code==legacy.NativeRuntime.generate.__code__.co_code
    result=stages.evaluate(current.context,allow_gpu=True,runtime=StageRuntime(current),scorer=fake_scores,bootstrap=interval)
    assert result['owner']==stages.OWNER and result['gold_used_only_after_complete_generation'] is True
    assert result['records']==3 and result['checkpoint_roles']==current.context['static']['checkpoint']['roles']
    assert current.output.parts[-2:]==(stage,'best') and globals_snapshot()==before


def test_historical_best_and_exact_terminal_last_are_both_bound(current):
    target=current.run/'.checkpoint_objects/object-historical';target.mkdir()
    import torch
    torch.save({'fixture':'historical retained role'},target/'state.pt')
    write(target/'.research_checkpoint.json',{'owner':'fixture_owner','step':100,'created_unix':0.})
    (current.run/'best').unlink();(current.run/'best').symlink_to(Path('.checkpoint_objects')/target.name)
    context=stages.prepare_context(current.run,current.bundle,assets=current.assets)
    assert context['static']['checkpoint']['roles']['best']['marker']['step']==100
    assert context['static']['checkpoint']['roles']['last']['marker']['step']==170


@pytest.mark.parametrize('failure',['no_completed','wrong_last','owner','outside','retired','SFT','source_config','source_archive','source_ref','nonzip'])
def test_stage_roles_source_and_retired_payloads_refuse_without_recreation(current,failure):
    if failure=='no_completed':(current.run/'completed.json').unlink()
    elif failure=='wrong_last':
        path=current.source.parent/'.research_checkpoint.json';value=json.loads(path.read_text());value['step']=169;write(path,value)
    elif failure=='owner':
        path=current.source.parent/'.research_checkpoint.json';value=json.loads(path.read_text());value['owner']='foreign';write(path,value)
    elif failure=='outside':
        (current.run/'best').unlink();(current.run/'best').symlink_to(current.root)
    elif failure=='retired':current.source.unlink()
    elif failure=='SFT':
        path=current.run/'resolved_config.json';value=json.loads(path.read_text());value['stage']='SFT';write(path,value)
    elif failure=='source_config':
        path=current.run/'provenance_latest.json';value=json.loads(path.read_text());value['resolved_config_sha256']='0'*64;write(path,value)
    elif failure=='source_archive':(current.root/'source.zip').write_bytes(b'changed')
    elif failure=='source_ref':
        path=current.run/'provenance_latest.json';value=json.loads(path.read_text());value['model_ref']['revision']='main';write(path,value)
    else:current.source.write_bytes(b'not a native mmap ZIP')
    before={str(path):path.stat().st_size for path in current.run.rglob('*') if path.is_file()}
    with pytest.raises((ValueError,FileNotFoundError)):stages.prepare_context(current.run,current.bundle,assets=current.assets)
    assert before=={str(path):path.stat().st_size for path in current.run.rglob('*') if path.is_file()}


def test_no_stage_or_role_cache_retag_and_resume_only_pending(current):
    original=globals_snapshot();engine=StageRuntime(current,fail_after=1)
    with pytest.raises(InterruptedError):stages.evaluate(current.context,allow_gpu=True,runtime=engine,scorer=fake_scores,bootstrap=interval)
    next_engine=StageRuntime(current)
    result=stages.evaluate(current.context,resume=True,allow_gpu=True,runtime=next_engine,scorer=fake_scores,bootstrap=interval)
    assert next_engine.calls==[row['record_id'] for row in current.records[1:]]
    with pytest.raises(ValueError,match='namespace'):stages.evaluate(current.context,current.run/'evaluation/caption_pilot_en_v2')
    changed=copy.deepcopy(current.context);changed['static']['checkpoint']['config']['stage']='A1'
    with pytest.raises(ValueError):stages.evaluate(changed,current.output,resume=True,score_only=True)
    assert globals_snapshot()==original and result['caption_scores']['images']==3


def test_gold_content_first_opened_after_complete_durable_stage_journal(current,monkeypatch):
    original=Path.open;reads=[]
    def opened(path,*args,**kwargs):
        if path==current.train:raise AssertionError('Train gold opened for caption evaluation')
        if path==current.bundle/'references.jsonl':
            assert (current.output/'generation_complete.json').exists()
            assert len(list((current.output/'records').glob('*.json')))==3
            reads.append(str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)
    stages.evaluate(current.context,allow_gpu=True,runtime=StageRuntime(current),scorer=fake_scores,bootstrap=interval)
    assert reads


def test_loaded_stage_revision_cache_or_required_profile_mismatch_refused_before_generation(current):
    for key in ('stage','recipe','model_ref','cache_fingerprint'):
        class Wrong(StageRuntime):
            def load(self,*args):
                model,value=super().load(*args)
                value[key]={'repo':'fixture/model','revision':'e'*40} if key=='model_ref' else 'wrong'
                return model,value
        engine=Wrong(current)
        with pytest.raises(ValueError):stages.evaluate(current.context,allow_gpu=True,runtime=engine,scorer=fake_scores,bootstrap=interval)
        assert engine.calls==[] and not (current.output/'protocol.json').exists()


def test_headroom_refusal_and_no_torch_import_on_clean_module_import(monkeypatch):
    monkeypatch.setattr(stages,'_available_host_bytes',lambda:2**30)
    with pytest.raises(MemoryError):stages._memory_guard()
    result=subprocess.run([sys.executable,'-c',"import sys; import src.eval.research_caption_stages; assert 'torch' not in sys.modules; assert not any('stage_caption_private' in x for x in sys.modules)"],cwd=stages.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_sealed_source_drift_refused_and_global_profiles_never_patched(current,monkeypatch):
    original=globals_snapshot();monkeypatch.setattr(stages,'V2_SHA256','0'*64)
    with pytest.raises(ValueError,match='sealed source'):stages._private_owner()
    assert globals_snapshot()==original


def tiny_native_factory(args):
    """CPU native reconstruction fixture; never claims actual NF4 CUDA load."""
    import torch
    from types import SimpleNamespace
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from src.model.smallthinker import SmallThinkerConfig,SmallThinkerForCausalLM
    from src.model.research_moe import install_research_moe,configure_research_trainability
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();torch.manual_seed(15)
            config=SmallThinkerConfig(vocab_size=32,hidden_size=8,num_hidden_layers=1,
                num_attention_heads=2,num_key_value_heads=1,head_dim=4,moe_ffn_hidden_size=8,
                moe_num_primary_experts=3,moe_num_active_primary_experts=2,
                pad_token_id=0,bos_token_id=1,eos_token_id=2,max_position_embeddings=512)
            self.llm=SmallThinkerForCausalLM(config)
            self.projector=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.GELU(),torch.nn.Linear(8,8))
            self.args=SimpleNamespace(max_length=512)
            vocabulary={'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,'user':4,'assistant':5,
                **{f'token{i}':i for i in range(6,32)}}
            tokenizer=Tokenizer(WordLevel(vocabulary,unk_token='<unk>'));tokenizer.pre_tokenizer=Whitespace()
            self.tokenizer=PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token='<pad>',bos_token='<bos>',eos_token='<eos>',unk_token='<unk>',
                chat_template="{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}")
            if args.stage=='A0':self.llm.requires_grad_(False)
            else:
                self.projector.requires_grad_(False)
                if args.recipe=='1c':
                    self.llm.requires_grad_(False)
                    install_research_moe(self.llm,1,'c',added_experts=1,router_width=4)
                    configure_research_trainability(self.llm,stage='A1')
            class Cache:
                fingerprint='f'*64
                def get_many(self,paths,encode,**kwargs):
                    return [torch.tensor([[1.,2.,3.],[4.,5.,6.],[1.,0.,1.]]) for _ in paths]
            self.cache=Cache()
        @property
        def base(self):return self.llm
        def encode(self,paths):raise AssertionError('Synthetic CPU feature cache must hit')
    return Tiny()


def native_checkpoint(current,monkeypatch,stage,recipe):
    import torch
    from types import SimpleNamespace
    import huggingface_hub
    import scripts.run_research_stage as runner
    args=SimpleNamespace(stage=stage,recipe=recipe)
    native=tiny_native_factory(args)
    state={name:parameter.detach().clone() for name,parameter in native.named_parameters()
        if parameter.requires_grad or name.startswith('projector.')}
    payload={'model':state,'step':170,'config':{'model':'fixture/model','vision':'fixture/vision','stage':stage,'recipe':recipe},
        'model_ref':{'repo':'fixture/model','revision':'a'*40},'vision_ref':{'repo':'fixture/vision','revision':'b'*40},'cache_fingerprint':'f'*64}
    path=current.root/f'native-{stage}-{recipe}.pt';torch.save(payload,path)
    monkeypatch.setattr(runner,'ResearchModel',lambda config:tiny_native_factory(config))
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *args,**kwargs:str(current.native.parent))
    return path,payload


@pytest.mark.parametrize('stage,recipe',[('A0','4f'),('A0','4q'),('A1','4f'),('A1','1c')])
def test_frozen_loader_restores_projector_and_all_native_required_stage_keys(current,monkeypatch,stage,recipe):
    import torch
    path,payload=native_checkpoint(current,monkeypatch,stage,recipe)
    model,provenance=stages._private_owner().NativeRuntime().load(path,'unused',120)
    assert provenance['stage']==stage and provenance['recipe']==recipe
    assert not model.training and all(not parameter.requires_grad for parameter in model.parameters())
    for name,value in payload['model'].items():torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
    if stage=='A0':assert all(name.startswith('projector.') for name in payload['model'])
    if stage=='A1' and recipe=='1c':assert any('visual_experts' in name for name in payload['model'])


@pytest.mark.parametrize('missing',['P_all','P_one','A1_llm','A1_visual_expert'])
def test_partial_projector_or_required_A1_state_refused_by_unchanged_loader(current,monkeypatch,missing):
    import torch
    stage='A0' if missing.startswith('P') else 'A1';recipe='1c' if missing=='A1_visual_expert' else '4f'
    path,payload=native_checkpoint(current,monkeypatch,stage,recipe)
    if missing=='P_all':payload['model']={name:value for name,value in payload['model'].items() if not name.startswith('projector.')}
    elif missing=='P_one':payload['model'].pop(next(name for name in payload['model'] if name.startswith('projector.')))
    elif missing=='A1_llm':payload['model'].pop(next(name for name in payload['model'] if name.startswith('llm.')))
    else:payload['model'].pop(next(name for name in payload['model'] if 'visual_experts' in name))
    torch.save(payload,path)
    with pytest.raises(ValueError,match='projector|omits learned parameters'):
        stages._private_owner().NativeRuntime().load(path,'unused',120)


def test_actual_tiny_native_caption_generation_has_no_optimizer_update_or_weight_mutation(current,monkeypatch):
    import torch
    path,payload=native_checkpoint(current,monkeypatch,'A1','4f')
    runtime=stages._private_owner().NativeRuntime();model,_=runtime.load(path,'unused',120)
    before={name:value.detach().clone() for name,value in model.state_dict().items()}
    def forbidden(*args,**kwargs):raise AssertionError('Caption inference must never update an optimizer')
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    result=runtime.generate(model,current.records[0])
    assert 1<=result['generated_tokens']<=256 and result['visual_tokens']==3
    assert result['prefill_tokens']==len(result['prefix_token_ids'])+len(result['suffix_token_ids'])+3
    assert not torch.cuda.is_initialized()
    for name,value in model.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
