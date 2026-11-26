"""Archive text contracts with tiny owned metadata and gold-free CPU fixtures."""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.eval import research_text_stock_archived_v2 as s
from tests import test_research_text_generation as old

@pytest.fixture
def current(tmp_path,monkeypatch):
 owner=s.producer();monkeypatch.setattr(old,'g',owner);x=old.setup.__wrapped__(tmp_path,monkeypatch)
 x.context['model']['repo']=s.stock.SMALL
 run=tmp_path/'archived_run';target=run/'.checkpoint_objects/object-owned';target.mkdir(parents=True)
 config={'stage':'SFT','recipe':'3c','model':s.stock.SMALL,'vision':'fixture/vision','output':str(run),'numerical_profile':s.stock.definition()}
 for name,value in [('resolved_config.json',config),('completed.json',{'steps':3}),('.checkpoint_manager.json',{'owner':'own'}),('checkpoint_archival.json',{'synthetic_fixture':True})]:
  (run/name).write_text(json.dumps(value))
 (target/'state.pt').write_bytes(b'owned tiny nonTensor fixture')
 (target/'.research_checkpoint.json').write_text(json.dumps({'owner':'own','step':3,'checkpoint_archival':{'protocol':'synthetic_fixture'}}))
 for role in ('best','last'):(run/role).symlink_to(Path('.checkpoint_objects')/target.name)
 from src.utils.research_checkpoint_compat import checkpoint_identity
 artifact=checkpoint_identity(target/'state.pt');files={n:{'path':str(run/n),'stat':owner.stat(run/n),'sha256':owner.sha(run/n)} for n in ('resolved_config.json','completed.json','.checkpoint_manager.json','checkpoint_archival.json')}
 roles={name:{'object':str(target),'marker_file':{'path':str(target/'.research_checkpoint.json'),'stat':owner.stat(target/'.research_checkpoint.json'),'sha256':owner.sha(target/'.research_checkpoint.json')}} for name in ('best','last')}
 base={'run':str(run),'role':'best','checkpoint_path':str(target/'state.pt'),'config':config,'files':files,'roles':roles,'member_stats':{str(target/'state.pt'):owner.stat(target/'state.pt')},'artifact':artifact}
 real=s.shared;counts={'inspect':0,'context':0,'inference':0,'load':0}
 def context(full):
  value=copy.deepcopy(base);value['metadata_only']=not full;value['artifact']['components_fully_hashed']=full;value['archive_admission']={'physical_archives_verified':full};return value
 def inspect(*a,**k):counts['inspect']+=1;return context(False)
 def admit(*a,**k):counts['context']+=1;return context(True)
 def loaded():
  return {'model_ref':{'repo':s.stock.SMALL,'revision':'f'*40},'recipe':'3c','stage':'SFT','training_step':3,'numerical_profile':s.stock.definition(),
   'archived_evaluation':real.execution_profile(),'archived_physical_sha256':artifact['sha256'],'archived_publication_sha256':files['checkpoint_archival.json']['sha256']}
 def inference(value=None):
  counts['inference']+=1
  def load(*a,**k):counts['load']+=1;return SimpleNamespace(args=SimpleNamespace(max_length=4096),tokenizer=object()),loaded()
  return SimpleNamespace(load_research_checkpoint=load)
 monkeypatch.setattr(s,'shared',SimpleNamespace(READY=real.READY,verify_release=real.verify_release,source_binding=real.source_binding,execution_profile=real.execution_profile,
  inspect=inspect,checkpoint_context=admit,inference=inference,stat=real.stat,_publisher=lambda:SimpleNamespace(REPORT='checkpoint_archival.json')))
 # Factory created before fixture admission is intentionally replaced with the
 # same exact source factory after only the shared runtime entry points change.
 owner=s.producer();x.owner=owner;x.plan=owner.learned_plan(run,'best',x.context);x.run=run;x.counts=counts;x.loaded=loaded
 x.output=s.output_path(x.plan);return x

class Runtime(old.FakeRuntime):
 def __init__(self,x,**k):super().__init__(**k);self.x=x
 def load_learned(self,*a):self.loads+=1;return object(),self.x.loaded()
def evaluate(x,**k):return x.owner.evaluate(x.context,x.plan,**k)

def test_same_1680_public_suite_gold_free_and_baselines_explicitly_remain_previous_owner():
 p=s.producer();context=p.public_context(s.ROOT/'datasets/research/primary-text-public-en-v1','smallthinker4')
 assert len(context['records'])==1680 and p._stock_native.value is None and p.OWNER==s.OWNER
 for precision in ('bf16','nf4','full_fp32_master_bf16_autocast'):
  with pytest.raises(ValueError,match='preserved stock text'):p.baseline_plan(context,precision)

def test_archived_metadata_expected_sha_no_full_admission_or_decode(current):
 assert current.counts=={'inspect':1,'context':0,'inference':0,'load':0}
 assert current.plan['archive_metadata']['artifact']['components_fully_hashed'] is False
 assert current.plan['numerical_profile']['weights']=='fp32' and current.plan['numerical_profile']['stock_execution']==s.execution_profile()

def test_raw_stage_and_legacy_stock_config_refused_before_loader(current,monkeypatch):
 monkeypatch.setattr(s.shared,'inspect',Mock(side_effect=ValueError('Only owned archived SFT accepted')))
 with pytest.raises(ValueError):current.owner.learned_plan(current.run,'best',current.context)
 path=current.run/'resolved_config.json';config=json.loads(path.read_bytes());config.pop('numerical_profile');path.write_text(json.dumps(config))
 with pytest.raises(ValueError,match='Missing, legacy'):current.owner.learned_plan(current.run,'best',current.context)

def test_bound_shared_single_load_no_old_stock_double_peek(current,monkeypatch):
 monkeypatch.setattr(s.stock,'activate',lambda:None);monkeypatch.setattr(s.stock,'assert_active',lambda:None)
 monkeypatch.setattr(s.stock,'inference',Mock(side_effect=AssertionError('Old double peek used')))
 native=current.owner.NativeTextRuntime()
 class Bridge(Runtime):
  def load_learned(self,*a):self.loads+=1;return native.load_learned(*a)
 result=evaluate(current,allow_gpu=True,runtime=Bridge(current))
 assert result['records']==3 and current.counts=={'inspect':1,'context':1,'inference':1,'load':1}

def test_archive_native_load_without_bound_plan_refuses_before_admission(current):
 with pytest.raises(ValueError,match='Bound archived'):current.owner.NativeTextRuntime().load_learned(current.plan['checkpoint'],'cache',120)
 assert current.counts['context']==0 and current.counts['load']==0

def test_changed_full_admission_projection_refuses_before_inference(current,monkeypatch):
 admit=s.shared.checkpoint_context
 def wrong(*a,**k):value=admit(*a,**k);value['artifact']['sha256']='0'*64;return value
 monkeypatch.setattr(s.shared,'checkpoint_context',wrong);native=current.owner.NativeTextRuntime()
 class Bridge(Runtime):
  def load_learned(self,*a):return native.load_learned(*a)
 with pytest.raises(ValueError,match='admission changed'):evaluate(current,allow_gpu=True,runtime=Bridge(current))
 assert not current.counts['inference'] and not (current.output/'generation_protocol.json').exists()

def test_pending_only_resume_and_new_namespace_keep_raw_contract_and_gold_deferred(current):
 with pytest.raises(InterruptedError):evaluate(current,allow_gpu=True,runtime=Runtime(current,fail_after=1))
 first=(current.output/'records/000000.json').read_bytes();engine=Runtime(current);result=evaluate(current,resume=True,allow_gpu=True,runtime=engine)
 assert engine.calls==[r['record_id'] for r in current.records[1:]] and first==(current.output/'records/000000.json').read_bytes()
 assert result['scoring_performed'] is False and result['gold_reference_files_opened'] is False
 assert current.output.parts[-3:]==('evaluation',s.NAMESPACE,'best')
 producer=current.owner.read(current.output/'producer.json');assert producer['owner']=='research_primary_text_responses_v1'
 with pytest.raises(ValueError,match='namespace'):current.owner.evaluate(current.context,current.plan,current.run/'evaluation/old-text',resume=True,allow_gpu=True,runtime=Runtime(current))

@pytest.mark.parametrize('change',['source','profile','owner','archive_metadata'])
def test_resume_drift_refused_before_model(current,change):
 with pytest.raises(InterruptedError):evaluate(current,allow_gpu=True,runtime=Runtime(current,fail_after=1))
 p=current.output/'generation_protocol.json';value=current.owner.read(p)
 if change=='source':value['static']['sources']={'wrong':'0'*64}
 elif change=='profile':value['static']['plan']['numerical_profile'].pop('stock_execution')
 elif change=='owner':value['owner']='research_primary_text_stock_native_b1_v1'
 else:value['static']['plan']['archive_metadata']['artifact']['sha256']='0'*64
 p.write_text(json.dumps(value));engine=Runtime(current)
 with pytest.raises(ValueError):evaluate(current,resume=True,allow_gpu=True,runtime=engine)
 assert not engine.loads and not engine.calls

def test_actual_archive_metadata_file_mutation_refused_before_model(current):
 path=current.run/'checkpoint_archival.json';path.write_text('{"different":true}')
 engine=Runtime(current)
 with pytest.raises(ValueError,match='metadata changed'):evaluate(current,allow_gpu=True,runtime=engine)
 assert not engine.loads

def test_real_tiny_MMLU_last_prefix_LL_and_IFEval_whitespace_parity_unchanged():
 import torch
 p=s.producer();model=old.tiny_native_model();flags=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),torch.backends.cuda.matmul.allow_tf32,torch.get_num_threads())
 try:
  s.stock.activate();runtime=p.NativeTextRuntime();row=old.row('mmlu');census=old.tiny_census(row)
  actual=runtime.candidate_likelihoods(model,row,census,'smallthinker4');reference=old.g.NativeTextRuntime().candidate_likelihoods(model,row,census,'smallthinker4')
  assert actual==reference and actual['length_normalization'] is False
  row=old.row('ifeval');result=runtime.generate(model,row,old.tiny_census(row))
  assert result['response']=='  raw response with whitespace\n' and result['generated_token_ids']==[9,2]
  assert not torch.cuda.is_initialized()
 finally:
  torch.use_deterministic_algorithms(flags[0],warn_only=flags[1]);torch.backends.cuda.matmul.allow_tf32=flags[2];torch.set_num_threads(flags[3])

def test_cli_gpu_requires_fresh_process_and_explicit_profile(monkeypatch):
 from scripts import eval_research_text_stock_archived_v2 as cli
 monkeypatch.setitem(sys.modules,'torch',object())
 with pytest.raises(RuntimeError,match='fresh process'):cli.main(['--run','unused','--allow-gpu'])
