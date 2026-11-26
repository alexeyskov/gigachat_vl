"""Stock caption facade tests use tiny files/models, never campaign weights."""
import copy,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
import pytest
from src.eval import research_caption_stock_generation as s
from src.eval import research_caption_generation as old
from tests.test_research_caption_generation import example,FakeRuntime,fake_scores,interval,write
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def current(example,monkeypatch):
 config=old.read_json(example.run/'resolved_config.json');config.update(stage='SFT',recipe='3a',model=s.stock.SMALL,numerical_profile=s.stock.definition())
 write(example.run/'resolved_config.json',config)
 with zipfile.ZipFile(example.source,'w') as z:z.writestr('synthetic_metadata_only','not actual tensors')
 protocol=old.read_json(example.bundle/'protocol.json');protocol.update(protocol='research_caption_pilot_eval_v2',preparation_profile='caption_pilot_en_v2_signal_safe')
 protocol['preparation_source_sha256']={str(ROOT/n):s.sha(ROOT/n) for n in ('src/eval/research_caption_data_v2.py','scripts/prepare_research_caption_eval_v2.py')}
 write(example.bundle/'protocol.json',protocol);expected=s.sha(example.bundle/'protocol.json');clone=s._clone
 def fixture_clone(name,*a,**k):
  m=clone(name,*a,**k)
  if name in (s.V2,s.STAGES):m.DATA_PROTOCOL_SHA256=expected;m.RECORDS=3
  if name==s.STAGES:m._memory_guard=lambda:64*2**30
  return m
 monkeypatch.setattr(s,'_clone',fixture_clone);monkeypatch.setattr(s,'RECORDS',3)
 module=s._module
 def fixture_module(stage):
  m=module(stage);factory=m._private_owner
  def owner():
   value=factory();value.gpu_idle=lambda:None;return value
  m._private_owner=owner;return m
 monkeypatch.setattr(s,'_module',fixture_module)
 example.context=s.prepare_context(example.run,example.bundle,assets=example.assets);example.output=s.output_path(example.context)
 return example

def make_stage(x,stage):
 config=old.read_json(x.run/'resolved_config.json');config.update(stage=stage)
 write(x.run/'resolved_config.json',config);(x.run/'checkpoint_compaction.json').unlink()
 write(x.source.parent/'.research_checkpoint.json',{'owner':'fixture_owner','step':170,'created_unix':0.})
 archive=x.root/'source.zip'
 with zipfile.ZipFile(archive,'w') as z:z.writestr('fixture.py','# synthetic stage source')
 provenance={'resolved_config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest(),
  'model_ref':{'repo':s.stock.SMALL,'revision':'a'*40},'vision_ref':{'repo':'fixture/vision','revision':'b'*40},
  'cache_fingerprint':'f'*64,'code_sha256':'c'*64,'source_archive':str(archive),'source_archive_sha256':s.sha(archive)}
 write(x.run/'provenance_latest.json',provenance)
 x.context=s.prepare_context(x.run,x.bundle,assets=x.assets);x.output=s.output_path(x.context)
 return x

class Runtime(FakeRuntime):
 def load(self,*a):
  self.loads+=1;config=self.example.context['static']['checkpoint']['config']
  return object(),{'recipe':config['recipe'],'stage':config['stage'],'training_step':170,'cache_fingerprint':'f'*64,
   'model_ref':{'repo':s.stock.SMALL,'revision':'a'*40},'vision_ref':{'repo':'fixture/vision','revision':'b'*40},'numerical_profile':s.stock.definition()}
 def describe(self,*a):
  value=super().describe(*a);value['stock_execution']=s.execution_profile();return value

def evaluate(x,**kwargs):return s.evaluate(x.context,scorer=fake_scores,bootstrap=interval,**kwargs)

def test_metadata_actual_stock_sft_no_torch_gold_or_weights():
 code="""
import pathlib,sys
from scripts import eval_research_caption_stock_checkpoint as c
original=pathlib.Path.open
def opened(p,*a,**k):
 assert p.name not in ('state.pt','references.jsonl') and p.suffix!='.safetensors','body/gold read'
 return original(p,*a,**k)
pathlib.Path.open=opened
assert c.main(['--run','runs/research/smallthinker4__siglip2__3a__SFT_pilot_scienceqa_context_v2_balanced_s42__stockdet_v1'])==0
assert 'torch' not in sys.modules
"""
 result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
 assert result.returncode==0,result.stderr
 result=json.loads(result.stdout);assert result['plan']['records']==372 and not result['Torch_imported'] and not result['GPU_used']
 assert result['plan']['checkpoint_content_hash_computed'] is False and result['plan']['full_RGB_checkpoint_and_source_audit_pending'] is True

def test_same_sealed_sources_global_values_and_public_bundle_preserved(current):
 before={n:s.sha(ROOT/n) for n in (s.V1,s.V2,s.STAGES,*s.PARENTS)}
 canonical={key:getattr(old,key) for key in ('OWNER','SOURCES','DATA_PROTOCOL_SHA256','RECORDS')}
 module=s._module('SFT');owner=module._private_owner()
 assert owner.__dict__ is not old.__dict__ and owner._stock_native.value is None
 for name in ('_references','_row','committed_rows','evaluate'):assert getattr(owner,name).__code__.co_code==getattr(old,name).__code__.co_code
 assert before=={n:s.sha(ROOT/n) for n in before} and canonical=={key:getattr(old,key) for key in canonical}
 assert owner.__name__ not in sys.modules

@pytest.mark.parametrize('stage',['A0','A1'])
def test_future_stock_a0_and_retained_a1_native_zip_completed_without_sft_compaction(current,stage):
 make_stage(current,stage);plan=s.metadata_plan(current.run,current.bundle)
 assert plan['stage']==stage and current.context['static']['checkpoint']['completed_stage']==stage
 assert current.output.parts[-3:]==(s.NAMESPACE,stage,'best')
 result=evaluate(current,allow_gpu=True,runtime=Runtime(current))
 assert result['records']==3 and result['owner']==s.OWNER and result['gold_used_only_after_complete_generation']

@pytest.mark.parametrize('mutation',['legacy','archived','missing','outside_exclusion','separate_text'])
def test_invalid_or_legacy_retired_and_unsupported_data_refuse_metadata_no_model(current,mutation):
 config=old.read_json(current.run/'resolved_config.json')
 if mutation=='legacy':config.pop('numerical_profile')
 elif mutation=='archived':
  p=current.source.parent/'.research_checkpoint.json';v=old.read_json(p);v['checkpoint_archival']={'protocol':'real_archive_needs_new_profile'};write(p,v)
 elif mutation=='missing':(current.run/'best').unlink()
 elif mutation=='outside_exclusion':config['manifest_sha256']='0'*64
 else:config['text_manifest']='not_audited_separately.jsonl'
 write(current.run/'resolved_config.json',config)
 with pytest.raises((ValueError,FileNotFoundError)):s.metadata_plan(current.run,current.bundle)
 assert not current.output.exists()

def test_sft_raw_preflight_bound_sources_and_stock_profile(current):
 static=current.context['static'];assert static['stock_caption_profile']==s.execution_profile()
 assert static['checkpoint']['config']['stage']=='SFT' and static['owner']==s.OWNER
 for n in (*s.NEW,s.V1,s.V2,s.STAGES):assert static['source_files'][n]['sha256']==s.sha(ROOT/n)
 assert static['generation']['batch_size']==1 and static['generation']['max_new_tokens']==256

def test_resume_only_pending_gold_deferred_and_no_legacy_namespace(current,monkeypatch):
 original=Path.open;reads=[]
 def opened(p,*a,**k):
  assert p!=current.train,'Training gold read'
  if p==current.bundle/'references.jsonl':
   assert (current.output/'generation_complete.json').is_file()
   assert len(list((current.output/'records').glob('*.json')))==3;reads.append(p)
  return original(p,*a,**k)
 monkeypatch.setattr(Path,'open',opened)
 with pytest.raises(InterruptedError):evaluate(current,allow_gpu=True,runtime=Runtime(current,fail_after=1))
 first=(current.output/'records/000000.json').read_bytes();assert not reads
 runtime=Runtime(current);result=evaluate(current,resume=True,allow_gpu=True,runtime=runtime)
 assert runtime.calls==[r['record_id'] for r in current.records[1:]] and reads
 assert first==(current.output/'records/000000.json').read_bytes() and result['max_new_tokens']==256
 with pytest.raises(ValueError,match='namespace'):s.evaluate(current.context,current.run/'evaluation/caption_pilot_en_v2',score_only=True)

@pytest.mark.parametrize('mutation',['static_profile','protocol_owner','source'])
def test_resume_rejects_mixed_owner_profile_or_source_before_model(current,mutation):
 with pytest.raises(InterruptedError):evaluate(current,allow_gpu=True,runtime=Runtime(current,fail_after=1))
 path=current.output/'protocol.json';v=old.read_json(path)
 if mutation=='static_profile':v['static'].pop('stock_caption_profile')
 elif mutation=='protocol_owner':v['owner']='research_caption_native_b1_v2'
 else:v['static']['source_files'][s.NEW[0]]['sha256']='0'*64
 write(path,v);runtime=Runtime(current)
 with pytest.raises(ValueError):evaluate(current,resume=True,allow_gpu=True,runtime=runtime)
 assert runtime.loads==0 and runtime.calls==[]

def test_stock_load_before_constructor_and_loaded_payload_profile_enforced(current,monkeypatch):
 from unittest.mock import Mock
 owner=s._module('SFT')._private_owner();base=owner.NativeRuntime.__mro__[1];active=Mock();check=Mock()
 monkeypatch.setattr(s.stock,'activate',active);monkeypatch.setattr(s.stock,'assert_active',check)
 def load(self,*a):active.assert_called_once();return 'model',{'numerical_profile':s.stock.definition()}
 monkeypatch.setattr(base,'load',load)
 assert owner.NativeRuntime().load('unused','cache',120)[0]=='model';check.assert_called_once()
 monkeypatch.setattr(base,'load',lambda *a:('model',{}))
 with pytest.raises(ValueError,match='exact stock'):owner.NativeRuntime().load('unused','cache',120)

def test_missing_loaded_stock_provenance_refused_before_commit(current):
 class Bad(Runtime):
  def load(self,*a):
   model,value=super().load(*a);value.pop('numerical_profile');return model,value
 runtime=Bad(current)
 with pytest.raises(ValueError,match='profile'):evaluate(current,allow_gpu=True,runtime=runtime)
 assert runtime.calls==[] and not (current.output/'protocol.json').exists()

def test_score_only_incomplete_never_reads_gold_or_loads_model(current,monkeypatch):
 original=Path.open
 def opened(p,*a,**k):
  assert p!=current.bundle/'references.jsonl';return original(p,*a,**k)
 monkeypatch.setattr(Path,'open',opened);runtime=Runtime(current)
 with pytest.raises(ValueError,match='Score-only'):evaluate(current,score_only=True,runtime=runtime)
 assert runtime.loads==0

def test_native_tiny_private_observer_exact_and_no_weight_change(current):
 import torch
 from types import SimpleNamespace
 from tests.test_research_caption_stages import tiny_native_factory
 model=tiny_native_factory(SimpleNamespace(stage='A1',recipe='4f'));owner=s._module('SFT')._private_owner();runtime=owner.NativeRuntime()
 flags=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),torch.backends.cuda.matmul.allow_tf32,torch.get_num_threads())
 before={n:v.detach().clone() for n,v in model.state_dict().items()}
 try:
  s.stock.activate();result=runtime.generate(model,current.records[0])
  reference=old.NativeRuntime().generate(model,current.records[0]);assert result==reference
  assert 1<=result['generated_tokens']<=256 and result['visual_tokens']==3
  assert result['prefill_tokens']==len(result['prefix_token_ids'])+len(result['suffix_token_ids'])+3
  assert owner._stock_native.value is not None and not torch.cuda.is_initialized()
  for n,v in model.state_dict().items():torch.testing.assert_close(v,before[n],atol=0,rtol=0)
 finally:
  torch.use_deterministic_algorithms(flags[0],warn_only=flags[1]);torch.backends.cuda.matmul.allow_tf32=flags[2];torch.set_num_threads(flags[3])

def test_real_tiny_stock_payload_restore_exact_and_legacy_config_refused_before_constructor(current):
 import torch
 from types import SimpleNamespace
 from unittest.mock import Mock
 from tests.test_research_caption_stages import tiny_native_factory
 args=SimpleNamespace(stage='A1',recipe='4f');original=tiny_native_factory(args)
 state={n:p.detach().clone() for n,p in original.named_parameters() if p.requires_grad or n.startswith('projector.')}
 config={'model':s.stock.SMALL,'vision':'fixture/vision','stage':'A1','recipe':'4f','numerical_profile':s.stock.definition()}
 payload={'model':state,'step':170,'config':config,'model_ref':{'repo':s.stock.SMALL,'revision':'a'*40},
  'vision_ref':{'repo':'fixture/vision','revision':'b'*40},'cache_fingerprint':'f'*64}
 path=current.root/'tiny_actual_stock.pt';torch.save(payload,path)
 constructor=Mock(side_effect=lambda ignored:(s.stock.assert_active() or tiny_native_factory(args)))
 native=s.stock.inference(private_runner=SimpleNamespace(ResearchModel=constructor,resolve_model=lambda ignored:None))
 owner=s._module('SFT')._private_owner();owner._stock_native.value=native
 flags=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),torch.backends.cuda.matmul.allow_tf32,torch.get_num_threads())
 try:
  model,loaded=owner.NativeRuntime().load(path,'unused',120)
  assert loaded['numerical_profile']==s.stock.definition() and loaded['stage']=='A1' and not torch.cuda.is_initialized()
  assert constructor.call_count==1 and all(not p.requires_grad for p in model.parameters())
  for name,value in state.items():torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
  config.pop('numerical_profile');torch.save(payload,path);constructor.reset_mock()
  with pytest.raises(ValueError,match='Missing, legacy'):owner.NativeRuntime().load(path,'unused',120)
  constructor.assert_not_called()
 finally:
  torch.use_deterministic_algorithms(flags[0],warn_only=flags[1]);torch.backends.cuda.matmul.allow_tf32=flags[2];torch.set_num_threads(flags[3])

def test_cli_fresh_process_guard_before_torch(monkeypatch):
 from scripts import eval_research_caption_stock_checkpoint as cli
 monkeypatch.setitem(sys.modules,'torch',object())
 with pytest.raises(RuntimeError,match='fresh process'):cli.main(['--allow-gpu','--run','unused'])
