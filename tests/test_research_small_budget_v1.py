"""Budget/source/transition gates and actual exact native tiny save; CPU only."""
import copy,io,json,os,subprocess,sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
import pytest,torch
from torch import nn
from src.utils import research_small_budget_v1 as b,research_stock_determinism as stock

def args(tmp_path,stage='A0',recipe='1b'):
 return SimpleNamespace(model=stock.SMALL,vision=b.SIG,vision_preprocessing_profile=None,recipe=recipe,stage=stage,seed=42,epochs=1,effective_batch_size=64,max_length=4096,eval_every=10000,max_records=None,max_steps=None,checkpoint_format='legacy',reserve_gb=120.,alignment_checkpoint=None,initial_checkpoint=None,resume=False,selection_manifest=None,text_validation=None,text_manifest=None,text_fraction=.15,text_loss_weight=.15 if stage=='SFT' else None,selection_metric='group_macro' if stage=='SFT' else 'ce',selection_per_group=32,manifest=str(b.DATA/b.stage_manifest(stage)),validation=str(b.DATA/('sft_validation.jsonl' if stage=='SFT' else 'alignment_validation.jsonl')),output=str(tmp_path/'run'))

def test_actual_frozen_data_inputs_counts_and_stock_numeric_profile(tmp_path):
 a=b.configure(args(tmp_path),b.RELEASE)
 assert a.numerical_profile==stock.definition() and a.budget18_expected_steps==64 and a.budget18_inputs['manifest']['sha256']=='e7fa219fd3885fed8df0816710f8b2b3ab13526742701cd10d12529721d966fb'
 assert b.validate_execution(vars(a)) and a.budget18_initialization_origin is None

@pytest.mark.parametrize('field,value',[('recipe','2c'),('seed',43),('epochs',2),('effective_batch_size',32),('max_steps',1),('eval_every',1),('max_length',2048),('reserve_gb',119.)])
def test_no_truncation_rebrand_or_budget_change(tmp_path,field,value):
 a=args(tmp_path);setattr(a,field,value)
 with pytest.raises(ValueError):b.configure(a,b.RELEASE)

def test_4f_A1_and_foreign_input_path_refused(tmp_path):
 with pytest.raises(ValueError,match='A1 disallowed'):b.configure(args(tmp_path,'A1','4f'),b.RELEASE)
 a=args(tmp_path);p=tmp_path/'alignment_train.jsonl';p.write_bytes(Path(a.manifest).read_bytes());a.manifest=str(p)
 with pytest.raises(ValueError,match='manifests'):b.configure(a,b.RELEASE)

def P():return nn.Sequential(nn.Linear(3,4),nn.GELU(),nn.Linear(4,4))
def state(projector):return {'projector.'+n:v.detach().clone() for n,v in projector.named_parameters()}
@pytest.mark.parametrize('key',b.PKEYS)
def test_frozen_partial_projector_is_refused_before_native_load(key):
 p=P().requires_grad_(False);s=state(p);s.pop(key)
 with pytest.raises(ValueError,match='All four'):b.require_complete_P(s,p)

def test_projector_dtype_shape_finiteness_and_complete_state():
 p=P();s=state(p);assert b.require_complete_P(s,p)
 for change in ('dtype','shape','nonfinite'):
  bad=copy.deepcopy(s)
  if change=='dtype':bad['projector.0.weight']=bad['projector.0.weight'].bfloat16()
  elif change=='shape':bad['projector.0.weight']=bad['projector.0.weight'][:2]
  else:bad['projector.0.weight'][0,0]=float('nan')
  with pytest.raises(ValueError):b.require_complete_P(bad,p)

def source_A0(tmp_path,monkeypatch):
 root=tmp_path/'fresh';root.mkdir();obj=root/'.checkpoint_objects'/'o64';obj.mkdir(parents=True);(obj/'state.pt').write_bytes(b'fixture-not-loaded')
 for role in ('best','last'):(root/role).symlink_to(obj,target_is_directory=True)
 a=args(tmp_path);config=vars(a).copy();config.update(output=str(root),numerical_profile=stock.definition(),manifest_sha256=b.sha(b.DATA/'alignment_train.jsonl'),validation_sha256=b.sha(b.DATA/'alignment_validation.jsonl'))
 (root/'resolved_config.json').write_text(json.dumps(config));(root/'completed.json').write_text(json.dumps({'steps':64,'records':4096}));monkeypatch.setattr(b,'FRESH_A0',root)
 target=args(tmp_path,'A1');target.initial_checkpoint=str(root/'best/state.pt');return root,target,config

def test_ONLY_exact_fresh_stock_A0_origin_admitted_without_rewriting(tmp_path,monkeypatch):
 root,a,config=source_A0(tmp_path,monkeypatch);before=(root/'resolved_config.json').read_bytes();a=b.configure(a,b.RELEASE)
 assert a.budget18_initialization_origin['kind']=='explicit_fresh_existing_stock_A0_origin_no_config_or_payload_rewrite'
 assert (root/'resolved_config.json').read_bytes()==before
 config['manifest_sha256']='0'*64;(root/'resolved_config.json').write_text(json.dumps(config))
 with pytest.raises(ValueError,match='manifest'):b.initial_origin(a)

def test_old_SFT_initialization_and_legacy_resume_refused(tmp_path,monkeypatch):
 root,a,config=source_A0(tmp_path,monkeypatch);config['stage']='SFT';(root/'resolved_config.json').write_text(json.dumps(config))
 with pytest.raises(ValueError,match='no SFT'):b.initial_origin(a)
 with pytest.raises(ValueError,match='legacy'):b.validate_execution(config)

def test_private_manager_saves_complete_native_model_optimizer_RNG_exact(tmp_path):
 a=b.configure(args(tmp_path),b.RELEASE);module=ModuleType('_small_budget_real_tiny_save');destination=tmp_path/'state.pt';seen={}
 class Native:
  def __init__(self):self.root=tmp_path
  def save_torch(self,payload,**kwargs):seen.update(payload=payload,kwargs=kwargs);torch.save(payload,destination);return 'saved'
 module.BestLastCheckpointManager=Native;module.parse_args=lambda argv=None:a;module.validate_initial_checkpoint=lambda *a:True;module.validate_resume_config=lambda *a:True;module.validate_model_binding=lambda *a:True
 module.ResearchModel=nn.Module
 b.patch_module(module,b.RELEASE);p=P();payload={'config':vars(a),'step':64,'model':state(p),'optimizer':{'moment':torch.arange(7,dtype=torch.uint8)},'scheduler':{'last_epoch':64},'rng_torch':torch.get_rng_state()}
 assert module.BestLastCheckpointManager().save_torch(payload,step=64,metric=1.)=='saved'
 restored=torch.load(destination,weights_only=False);assert seen['payload'] is payload and set(restored)==set(payload)
 assert all(torch.equal(v,restored['model'][k]) for k,v in payload['model'].items()) and torch.equal(restored['optimizer']['moment'],payload['optimizer']['moment']) and torch.equal(restored['rng_torch'],payload['rng_torch'])
 assert seen['kwargs']['anticipated_bytes']>destination.stat().st_size and (tmp_path/'checkpoint_storage_budget.json').is_file()

def test_process_local_runner_hook_restored_on_error(monkeypatch):
 original=stock.runner
 with pytest.raises(RuntimeError):
  with b.patched_stock_runner(b.RELEASE):
   assert stock.runner is not original;raise RuntimeError('fixture')
 assert stock.runner is original

def test_default_metadata_fresh_process_does_not_import_Torch_or_construct_GPU():
 code="from src.utils.research_small_budget_v1 import main; main(['--metadata']); import sys; assert 'torch' not in sys.modules"
 run=subprocess.run([sys.executable,'-c',code],cwd=b.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
 assert run.returncode==0,run.stderr
 assert json.loads(run.stdout)['numerical_profile']==stock.definition()

def test_A1_new_expert_router_and_resume_all_learned_tensors_are_mandatory():
 class Tiny(nn.Module):
  def __init__(self):
   super().__init__();self.projector=P().requires_grad_(False);self.visual_experts=nn.Linear(4,4);self.router_adapter=nn.Linear(4,2);self.old=nn.Linear(4,4)
 model=Tiny();full={n:p.detach().clone() for n,p in model.named_parameters()};initial={n:v for n,v in full.items() if n.startswith(('projector.','visual_experts.','router_adapter.'))}
 required=set(initial);assert b.require_learned_state(model,initial,required)
 broken=dict(initial);broken.pop('visual_experts.bias')
 with pytest.raises(ValueError,match='Missing learned'):b.require_learned_state(model,broken,required)
 with pytest.raises(ValueError,match='Missing learned'):b.require_learned_state(model,initial)
 assert b.require_learned_state(model,full)

def test_fixed_clock_and_pinned_actualSmall_template_prompt_equivalence():
 from transformers import AutoTokenizer
 assets=json.loads((b.ROOT/'datasets/research/text-eval-inputs/assets.json').read_bytes());row=next(r for r in assets['files'] if r['id']=='smallthinker4' and r['filename']=='tokenizer_config.json');path=Path(row['path']).parent;tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
 assert not any(term in tokenizer.chat_template for term in ('strftime','date_string','current_date'))
 messages=[{'role':'user','content':'Describe this image.'}]
 assert tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)==b.fixed_clock(tokenizer).apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
