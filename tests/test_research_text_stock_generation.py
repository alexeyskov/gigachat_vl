"""Focused stock facade contracts; no actual model weights/CUDA or old edits."""
import copy,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.eval import research_text_stock_generation as s
ROOT=Path(__file__).resolve().parents[1]


def test_private_factory_keeps_sealed_source_and_public_bundle_unchanged():
 source=s.sha(ROOT/s.SOURCE);ready=s.sha(s.PARENT_READY)
 public=ROOT/'datasets/research/primary-text-public-en-v1';before={p.name:(s.sha(p),p.stat().st_mtime_ns) for p in public.iterdir()}
 p=s.producer();context=p.public_context(public,'smallthinker4')
 assert len(context['records'])==1680 and p.OWNER==s.OWNER and p._stock_native.value is None
 assert s.sha(ROOT/s.SOURCE)==source and s.sha(s.PARENT_READY)==ready
 assert before=={p.name:(s.sha(p),p.stat().st_mtime_ns) for p in public.iterdir()}


def test_actual_metadata_cli_no_torch_gold_or_weight_body_reads():
 code="""
import pathlib,sys
from scripts import eval_research_text_stock_checkpoint as cli
root=pathlib.Path.cwd();original=pathlib.Path.open
forbidden=root/'datasets/research/primary-text-en-v1'
def guard(path,*args,**kwargs):
 assert not path.resolve().is_relative_to(forbidden),'gold opened'
 assert not str(path).endswith(('.safetensors','state.pt')),'model weights opened'
 return original(path,*args,**kwargs)
pathlib.Path.open=guard
assert cli.main(['--baseline-precision','full_fp32_master_bf16_autocast'])==0
assert 'torch' not in sys.modules
"""
 result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,cwd=ROOT)
 assert result.returncode==0,result.stderr
 r=json.loads(result.stdout);assert r['records']==1680 and not r['Torch_imported'] and not r['GPU_used']
 assert r['plan']['numerical_profile']['weights']=='fp32'
 assert r['plan']['numerical_profile']['stock_execution']['stock_numerical_definition']==s.stock.definition()

@pytest.mark.parametrize('precision',['bf16','nf4','full_fp32_master_bf16_autocast'])
def test_explicit_baseline_dtype_preserved_with_separate_execution_profile(precision):
 p=s.producer();context=p.public_context(ROOT/'datasets/research/primary-text-public-en-v1','smallthinker4');plan=p.baseline_plan(context,precision)
 assert plan['precision']==precision and plan['numerical_profile']['weights']==('fp32' if precision==p.FULL_INIT_CONTROL else precision)
 assert plan['numerical_profile']['stock_execution']==s.execution_profile() and plan['weight_downloads_allowed'] is False


def test_unsupported_backbone_or_precision_refuses():
 p=s.producer();context=p.public_context(ROOT/'datasets/research/primary-text-public-en-v1','smallthinker4')
 with pytest.raises(ValueError):p.baseline_plan(context,'auto')
 with pytest.raises(ValueError):p.baseline_plan(dict(context,model=dict(context['model'],model_id='gptoss20')),'bf16')


def test_legacy_learned_config_rejected_before_any_payload_load(tmp_path):
 p=s.producer();run=tmp_path/'legacy';run.mkdir();(run/'resolved_config.json').write_text(json.dumps({'model':s.stock.SMALL,'stage':'SFT'}))
 context=p.public_context(ROOT/'datasets/research/primary-text-public-en-v1','smallthinker4')
 with pytest.raises(ValueError,match='Missing, legacy'):p.learned_plan(run,'best',context)


def test_current_stock_learned_metadata_uses_same_payload_stat_and_profile():
 p=s.producer();context=p.public_context(ROOT/'datasets/research/primary-text-public-en-v1','smallthinker4')
 run=ROOT/'runs/research/smallthinker4__siglip2__3a__SFT_pilot_scienceqa_context_v2_balanced_s42__stockdet_v1'
 plan=p.learned_plan(run,'best',context)
 assert plan['kind']=='learned' and plan['stage']=='SFT' and plan['numerical_profile']['weights']=='fp32'
 assert plan['numerical_profile']['stock_execution']==s.execution_profile() and p._stock_native.value is None


def test_load_wiring_enforces_stock_and_uses_private_inference(monkeypatch):
 p=s.producer();base=p.NativeTextRuntime.__mro__[1];activate=Mock();assertion=Mock()
 monkeypatch.setattr(s.stock,'activate',activate);monkeypatch.setattr(s.stock,'assert_active',assertion)
 monkeypatch.setattr(base,'load_baseline',lambda self,plan:('model',{'precision':plan['precision']}))
 plan={'precision':p.FULL_INIT_CONTROL,'numerical_profile':{'stock_execution':s.execution_profile()}}
 model,loaded=p.NativeTextRuntime().load_baseline(plan)
 assert model=='model' and loaded['stock_execution']==s.execution_profile();activate.assert_called_once();assertion.assert_called_once()
 with pytest.raises(ValueError):p.NativeTextRuntime().load_baseline({'precision':p.FULL_INIT_CONTROL})


def test_loaded_learned_profile_is_required_no_implicit_upgrade(monkeypatch):
 p=s.producer();base=p.NativeTextRuntime.__mro__[1]
 monkeypatch.setattr(s.stock,'activate',lambda:None);monkeypatch.setattr(s.stock,'assert_active',lambda:None)
 monkeypatch.setattr(base,'load_learned',lambda *a:('model',{'recipe':'3a'}))
 with pytest.raises(ValueError,match='lacks exact stock'):p.NativeTextRuntime().load_learned('checkpoint','cache',120)


def test_real_tiny_native_mmlu_and_raw_ifeval_same_math_under_stock_flags():
 import torch
 import test_research_text_generation as old
 p=s.producer();model=old.tiny_native_model();oldflags=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),torch.backends.cuda.matmul.allow_tf32,torch.get_num_threads())
 try:
  s.stock.activate();runtime=p.NativeTextRuntime()
  row=old.row('mmlu');actual=runtime.candidate_likelihoods(model,row,old.tiny_census(row),'smallthinker4')
  reference=old.g.NativeTextRuntime().candidate_likelihoods(model,row,old.tiny_census(row),'smallthinker4')
  assert actual==reference and actual['length_normalization'] is False and actual['log_probability_reduction_dtype']=='float32'
  row=old.row('ifeval');generated=runtime.generate(model,row,old.tiny_census(row))
  assert generated['response']=='  raw response with whitespace\n' and generated['generated_token_ids']==[9,2]
  assert p._stock_native.value is not None and not torch.cuda.is_initialized()
 finally:
  torch.use_deterministic_algorithms(oldflags[0],warn_only=oldflags[1]);torch.backends.cuda.matmul.allow_tf32=oldflags[2];torch.set_num_threads(oldflags[3])


def test_gpu_cli_requires_explicit_numeric_profile_and_fresh_process(monkeypatch):
 from scripts import eval_research_text_stock_checkpoint as cli
 monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
 with pytest.raises(RuntimeError,match='fresh process'):cli.main(['--allow-gpu','--baseline-precision','bf16'])

@pytest.fixture
def journal(tmp_path,monkeypatch):
 import test_research_text_generation as old
 p=s.producer();monkeypatch.setattr(old,'g',p)
 x=old.setup.__wrapped__(tmp_path,monkeypatch)
 x.context['model']['repo']=s.stock.SMALL;x.plan.update(repo=s.stock.SMALL)
 x.plan['numerical_profile']={'weights':'nf4','compute':'native bf16 autocast','stock_execution':s.execution_profile()}
 return p,x,old


def test_stock_journal_resume_only_uncommitted_rows_and_preserves_contract(journal):
 p,x,old=journal
 with pytest.raises(InterruptedError):p.evaluate(x.context,x.plan,x.output,allow_gpu=True,runtime=old.FakeRuntime(fail_after=1))
 before=(x.output/'records/000000.json').read_bytes();runtime=old.FakeRuntime()
 result=p.evaluate(x.context,x.plan,x.output,resume=True,allow_gpu=True,runtime=runtime)
 assert result['records']==3 and result['scoring_performed'] is False
 assert runtime.calls==[r['record_id'] for r in x.records[1:]] and (x.output/'records/000000.json').read_bytes()==before
 producer=p.read(x.output/'producer.json');assert producer['numerical_profile']['stock_execution']==s.execution_profile()
 assert producer['evidence_scope']=='synthetic_scorer_validation' and p.read(x.output/'generation_protocol.json')['owner']==s.OWNER

@pytest.mark.parametrize('mutation',['legacy_owner','numerical_profile','source'])
def test_legacy_mixed_or_changed_resume_refuses_before_model(journal,mutation):
 p,x,old=journal
 with pytest.raises(InterruptedError):p.evaluate(x.context,x.plan,x.output,allow_gpu=True,runtime=old.FakeRuntime(fail_after=1))
 file=x.output/'generation_protocol.json';protocol=p.read(file)
 if mutation=='legacy_owner':protocol['owner']='research_primary_text_native_b1_v1'
 elif mutation=='numerical_profile':protocol['static']['plan']['numerical_profile'].pop('stock_execution')
 else:protocol['static']['sources']={'foreign':'0'*64}
 file.write_text(json.dumps(protocol));runtime=old.FakeRuntime()
 with pytest.raises(ValueError):p.evaluate(x.context,x.plan,x.output,resume=True,allow_gpu=True,runtime=runtime)
 assert runtime.loads==0 and runtime.calls==[]


def test_full_baseline_stock_flags_before_native_load_and_fp32_promotion(monkeypatch):
 import torch
 from torch import nn
 import transformers
 from src.model import research_config
 p=s.producer();flags=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),torch.backends.cuda.matmul.allow_tf32,torch.get_num_threads())
 class Tiny(nn.Module):
  def __init__(self):
   super().__init__();self.embed=nn.Embedding(8,4,dtype=torch.bfloat16);self.head=nn.Linear(4,8,bias=False,dtype=torch.bfloat16);self.head.weight=self.embed.weight
   self.norm=nn.LayerNorm(4,dtype=torch.bfloat16);self.register_buffer('float_buffer',torch.ones(2,dtype=torch.bfloat16));self.config=SimpleNamespace(model_type='smallthinker')
  def to(self,*a,**kw):return self  # Unit fixture staysCPU; no actual GPU/weights.
 tiny=Tiny();loader=Mock(side_effect=lambda *a,**k:(s.stock.assert_active() or tiny,{}))
 monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',loader)
 monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:SimpleNamespace(eos_token='eos',pad_token=None))
 monkeypatch.setattr(research_config,'load_research_native_config',lambda *a:(tiny.config,{'synthetic':True}))
 monkeypatch.setattr(p,'snapshot_identity',lambda path:{})
 plan={'kind':'pinned_baseline','model_id':'smallthinker4','repo':s.stock.SMALL,'revision':'f'*40,'snapshot':'not_opened',
  'precision':p.FULL_INIT_CONTROL,'missing_weights':[],'numerical_profile':{'weights':'fp32','stock_execution':s.execution_profile()}}
 try:
  model,loaded=p.NativeTextRuntime().load_baseline(plan)
  assert loader.call_count==1 and all(v.dtype==torch.float32 and not v.requires_grad for v in model.parameters())
  assert model.base.float_buffer.dtype==torch.float32 and model.base.embed.weight is model.base.head.weight
  assert loaded['initialization_control']['Module_float_includes_floating_buffers'] and loaded['stock_execution']==s.execution_profile()
  assert not torch.cuda.is_initialized()
 finally:
  torch.use_deterministic_algorithms(flags[0],warn_only=flags[1]);torch.backends.cuda.matmul.allow_tf32=flags[2];torch.set_num_threads(flags[3])


def test_owned_stock_a1_plan_supported_and_retired_roles_never_reconstructed(tmp_path):
 p=s.producer();context=p.public_context(ROOT/'datasets/research/primary-text-public-en-v1','smallthinker4');run=tmp_path/'A1';target=run/'.checkpoint_objects/object-owned';target.mkdir(parents=True)
 config={'model':s.stock.SMALL,'stage':'A1','recipe':'3c','output':str(run),'numerical_profile':s.stock.definition()}
 for name,value in [('resolved_config.json',config),('completed.json',{'steps':3}),('.checkpoint_manager.json',{'owner':'own'})]:
  (run/name).write_text(json.dumps(value))
 (target/'state.pt').write_bytes(b'metadata_only_not_loaded');(target/'.research_checkpoint.json').write_text(json.dumps({'owner':'own','step':3}))
 for role in ('best','last'):(run/role).symlink_to(Path('.checkpoint_objects')/target.name)
 plan=p.learned_plan(run,'best',context);assert plan['stage']=='A1' and plan['numerical_profile']['weights']=='fp32'
 (run/'best').unlink();(run/'last').unlink()
 with pytest.raises(ValueError,match='role symlink'):p.learned_plan(run,'best',context)
 assert not (run/'best').exists() and p._stock_native.value is None
