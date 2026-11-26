"""Caption archive-consumer wiring; synthetic helper fixtures never decode weights."""
import copy,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.eval import research_caption_stock_archived_v2 as v
from src.eval import research_caption_stock_generation as raw
from src.eval import research_caption_generation as old
from tests.test_research_caption_stock_generation import current as raw_current
from tests.test_research_caption_generation import example,FakeRuntime,fake_scores,interval,write

class Archive:
 def __init__(self,x):self.x=x;self.inspect_calls=0;self.context_calls=0;self.inference_calls=[];self.loads=0
 def execution_profile(self):return {'owner':'synthetic_archive_contract_fixture','CPU_payload_decodes_per_load':1,'headroom_bytes':32*2**30}
 def context(self,full):
  value=copy.deepcopy(self.x.base_context['static']['checkpoint'])
  for name in ('best','last'):
   path=Path(value['roles'][name]['object'])/'.research_checkpoint.json'
   value['roles'][name].update(marker=old.read_json(path),marker_file=old._file(path))
  value.update(metadata_only=not full,archive_admission={'owner':'synthetic_fixture','physical_archives_verified':full,'execution_profile':self.execution_profile()})
  value['artifact']['components_fully_hashed']=full
  return value
 def inspect(self,run,role='best'):self.inspect_calls+=1;return self.context(False)
 def checkpoint_context(self,run,role='best'):self.context_calls+=1;return self.context(True)
 def inference(self,context):
  self.inference_calls.append(context)
  def load(*a,**k):self.loads+=1;return 'synthetic_model',self.provenance(context)
  return SimpleNamespace(load_research_checkpoint=load)
 def provenance(self,context):
  config=context['config']
  return {'recipe':config['recipe'],'stage':config['stage'],'training_step':170,'cache_fingerprint':'f'*64,
   'model_ref':{'repo':raw.stock.SMALL,'revision':'a'*40},'vision_ref':{'repo':'fixture/vision','revision':'b'*40},
   'numerical_profile':raw.stock.definition(),'archived_evaluation':self.execution_profile(),'archived_physical_sha256':context['artifact']['sha256']}

@pytest.fixture
def current(raw_current,monkeypatch):
 x=raw_current;x.base_context=copy.deepcopy(x.context);archive=Archive(x);x.archive=archive
 monkeypatch.setattr(v,'_shared_release',lambda:{'path':'synthetic_fixture','sha256':'f'*64,'source_sha256':{}})
 monkeypatch.setattr(v,'_archive',lambda:archive);factory=v.factory
 def fixture_factory(bound_context=None):
  g=factory(bound_context);g._clone=raw._clone;g.RECORDS=3;module=g._module
  def configured(stage):
   m=module(stage);original=m._private_owner
   def owner():
    o=original();o.gpu_idle=lambda:None;return o
   m._private_owner=owner;return m
  g._module=configured;return g
 monkeypatch.setattr(v,'factory',fixture_factory)
 x.context=v.prepare_context(x.run,x.bundle,assets=x.assets);x.output=v.factory().output_path(x.context)
 return x

def archived(x):
 p=x.source.parent/'.research_checkpoint.json';marker=old.read_json(p);marker['checkpoint_archival']={'protocol':'synthetic_owned_archive'};write(p,marker)
 x.context=v.prepare_context(x.run,x.bundle,assets=x.assets);x.output=v.factory().output_path(x.context);return x

class Runtime(FakeRuntime):
 def load(self,*a):
  self.loads+=1;model=self.example.context['static']['checkpoint']
  if model.get('archive_admission'):return object(),self.example.archive.provenance(model)
  config=model['config'];return object(),{'recipe':config['recipe'],'stage':config['stage'],'training_step':170,
   'cache_fingerprint':'f'*64,'model_ref':{'repo':raw.stock.SMALL,'revision':'a'*40},'vision_ref':{'repo':'fixture/vision','revision':'b'*40},'numerical_profile':raw.stock.definition()}
 def describe(self,*a):
  value=super().describe(*a);value['stock_execution']=v.factory().execution_profile();return value
def evaluate(x,**kwargs):return v.evaluate(x.context,scorer=fake_scores,bootstrap=interval,**kwargs)

def test_unpinned_shared_helper_is_closed_before_import_or_work(monkeypatch):
 monkeypatch.setattr(v,'HELPER_READY_SHA',None)
 with pytest.raises(ValueError,match='closed'):v._shared_release()

def test_shared_ready_exact_bytes_and_every_dependency_sha_required(tmp_path,monkeypatch):
 path=tmp_path/'ready.json';source='src/utils/research_archived_eval_v1.py'
 path.write_text(json.dumps({'source_sha256':{source:v.sha(v.ROOT/source)}}))
 monkeypatch.setattr(v,'HELPER_READY',path);monkeypatch.setattr(v,'HELPER_READY_SHA',v.sha(path))
 assert v._shared_release()['source_sha256'][source]==v.sha(v.ROOT/source)
 path.write_text(path.read_text()+' ')
 with pytest.raises(ValueError,match='readiness'):v._shared_release()
 path.write_text(json.dumps({'source_sha256':{source:'0'*64}}));monkeypatch.setattr(v,'HELPER_READY_SHA',v.sha(path))
 with pytest.raises(ValueError,match='dependency'):v._shared_release()

def test_raw_branch_preserves_original_guards_and_uses_new_namespace(current):
 assert not current.archive.inspect_calls and not current.archive.context_calls
 assert current.context['static']['owner']==v.OWNER and current.output.parts[-3:]==(v.NAMESPACE,'SFT','best')
 assert current.context['static']['stock_caption_profile']['archive_admission']['execution_profile']==current.archive.execution_profile()
 result=evaluate(current,allow_gpu=True,runtime=Runtime(current));assert result['records']==3 and not current.archive.inference_calls

def test_archive_metadata_delegates_only_metadata_and_actual_context_requires_full(current):
 archived(current);assert current.archive.context_calls==1
 calls=current.archive.context_calls;plan=v.metadata_plan(current.run,current.bundle)
 assert current.archive.inspect_calls==2 and current.archive.context_calls==calls
 assert plan['checkpoint_content_hash_computed'] is False and plan['checkpoint']['metadata_only'] is True
 assert current.context['static']['checkpoint']['metadata_only'] is False

def test_archived_runtime_uses_bound_shared_single_decode_not_original_double_peek(current,monkeypatch):
 archived(current);factory=v.factory(current.context);owner=factory._module('SFT')._private_owner()
 monkeypatch.setattr(raw.stock,'activate',lambda:None);monkeypatch.setattr(raw.stock,'assert_active',lambda:None)
 monkeypatch.setattr(raw.stock,'inference',lambda:(_ for _ in ()).throw(AssertionError('Old double-peek loader used')))
 model,loaded=owner.NativeRuntime().load(current.source,'cache',120)
 assert model=='synthetic_model' and current.archive.loads==1 and len(current.archive.inference_calls)==1
 assert loaded['archived_physical_sha256']==current.context['static']['checkpoint']['artifact']['sha256']

def test_archive_runtime_without_bound_context_refuses_before_shared_or_native_load(current,monkeypatch):
 archived(current);owner=v.factory()._module('SFT')._private_owner()
 with pytest.raises(ValueError,match='fully admitted'):owner.NativeRuntime().load(current.source,'cache',120)
 assert current.archive.loads==0 and not current.archive.inference_calls

def test_archive_bound_checkpoint_mismatch_refused_before_shared_load(current):
 archived(current);owner=v.factory(current.context)._module('SFT')._private_owner()
 with pytest.raises(ValueError,match='checkpoint changed'):owner.NativeRuntime().load(current.root/'foreign.pt','cache',120)
 assert not current.archive.inference_calls and current.archive.loads==0

def test_legacy_archive_run_profile_rejected_before_inspection(current):
 archived(current);count=current.archive.inspect_calls;path=current.run/'resolved_config.json';config=old.read_json(path);config.pop('numerical_profile');write(path,config)
 with pytest.raises(ValueError,match='Missing, legacy'):v.metadata_plan(current.run,current.bundle)
 assert current.archive.inspect_calls==count

def test_archive_helper_config_projection_must_match_caption_run(current,monkeypatch):
 archived(current);inspect=current.archive.inspect
 def wrong(*a,**k):
  value=inspect(*a,**k);value['config']['recipe']='2c';return value
 monkeypatch.setattr(current.archive,'inspect',wrong)
 with pytest.raises(ValueError,match='matching completed SFT'):v.metadata_plan(current.run,current.bundle)

@pytest.mark.parametrize('field',['archived_evaluation','archived_physical_sha256'])
def test_archive_actual_provenance_required_before_first_commit(current,field):
 archived(current)
 class Bad(Runtime):
  def load(self,*a):model,p=super().load(*a);p.pop(field);return model,p
 engine=Bad(current)
 with pytest.raises(ValueError,match='provenance'):evaluate(current,allow_gpu=True,runtime=engine)
 assert engine.calls==[] and not (current.output/'protocol.json').exists()

def test_archive_journal_resume_pending_only_and_gold_after_complete(current,monkeypatch):
 archived(current);original=Path.open;gold=[]
 def opened(p,*a,**k):
  assert p!=current.train
  if p==current.bundle/'references.jsonl':
   assert (current.output/'generation_complete.json').exists();gold.append(p)
  return original(p,*a,**k)
 monkeypatch.setattr(Path,'open',opened)
 with pytest.raises(InterruptedError):evaluate(current,allow_gpu=True,runtime=Runtime(current,fail_after=1))
 saved=(current.output/'records/000000.json').read_bytes();assert not gold
 engine=Runtime(current);result=evaluate(current,allow_gpu=True,resume=True,runtime=engine)
 assert engine.calls==[r['record_id'] for r in current.records[1:]] and saved==(current.output/'records/000000.json').read_bytes()
 assert result['gold_used_only_after_complete_generation'] and gold

def test_raw_v1_outputs_never_retagged_or_resumed_as_v2(current):
 with pytest.raises(ValueError,match='namespace'):v.evaluate(current.context,current.run/'evaluation'/raw.NAMESPACE/'SFT/best',score_only=True)
 assert not (current.run/'evaluation'/raw.NAMESPACE).exists()

def test_archive_marker_cannot_be_silently_treated_as_raw_after_context(current):
 archived(current);wrong=copy.deepcopy(current.context);wrong['static']['checkpoint'].pop('archive_admission')
 owner=v.factory(wrong)._module('SFT')._private_owner()
 with pytest.raises(ValueError,match='fully admitted'):owner.NativeRuntime().load(current.source,'cache',120)
 assert not current.archive.inference_calls

def test_native_math_and_full_gold_gate_bytecode_reused_without_canonical_edits(current):
 before={n:raw.sha(raw.ROOT/n) for n in (raw.V1,raw.V2,raw.STAGES,*v._raw_release()['source_sha256'])}
 owner=v.factory()._module('SFT')._private_owner()
 for name in ('evaluate','committed_rows','_references','_row'):assert getattr(owner,name).__code__.co_code==getattr(old,name).__code__.co_code
 assert before=={n:raw.sha(raw.ROOT/n) for n in before}

def test_cli_refuses_preimported_torch_before_actual_generation(monkeypatch):
 from scripts import eval_research_caption_stock_archived_v2 as cli
 monkeypatch.setitem(sys.modules,'torch',object())
 with pytest.raises(RuntimeError,match='fresh process'):cli.main(['--run','unused','--allow-gpu'])
