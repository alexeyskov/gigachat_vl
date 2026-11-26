"""New LFM budget wiring/complete restores; genuine tiny native CPU kernels only."""
import copy,io,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM,Lfm2MoeConfig
from src.utils import research_lfm_budget_v1 as b
from src.model.research_moe import install_research_moe,configure_research_trainability,set_modality_mask

@pytest.fixture(autouse=True)
def flags():
 old=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),torch.backends.cuda.matmul.allow_tf32,torch.get_num_threads(),os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
 yield
 torch.use_deterministic_algorithms(old[0],warn_only=old[1]);torch.backends.cuda.matmul.allow_tf32=old[2];torch.set_num_threads(old[3])
 if old[4] is None:os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
 else:os.environ['CUBLAS_WORKSPACE_CONFIG']=old[4]

def args(**kw):
 values=dict(reserve_gb=120,model=b.MODEL,vision=b.VISION,vision_preprocessing_profile=None,recipe='3c',stage='SFT',seed=42,epochs=1,effective_batch_size=64,eval_every=10000,checkpoint_format='legacy',alignment_checkpoint=None,max_steps=None,max_records=None,optimizer_backend='cpu_grad_bnb_paged_adamw8',text_loss_weight=.15,max_length=4096,text_fraction=.15,prompt_schema_version=2,text_manifest=None,text_validation=None,selection_manifest=None)
 values.update(kw);return SimpleNamespace(**values)

class Tokenizer:
 def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False,**kwargs):return ''.join(x['role']+':'+x['content']+'\n' for x in messages)+('assistant:' if add_generation_prompt else '')
 def encode(self,value,add_special_tokens=False):return list(value.encode())

def records(tmp_path,**rowkw):
 row=dict(record_id='r1',question='question?',answer='answer.',image_paths=['not-read.jpg'],image_hashes=['a'*64]);row.update(rowkw)
 for name in ('train','val'):(tmp_path/(name+'.jsonl')).write_text(json.dumps(row)+'\n')
 return args(manifest=str(tmp_path/'train.jsonl'),validation=str(tmp_path/'val.jsonl'))

class Tiny(nn.Module):
 def __init__(self,stage='SFT',recipe='3c'):
  super().__init__();self.args=args(stage=stage,recipe=recipe)
  cfg=Lfm2MoeConfig(vocab_size=32,hidden_size=16,intermediate_size=24,moe_intermediate_size=12,num_hidden_layers=3,num_dense_layers=1,num_experts=4,num_experts_per_tok=2,num_attention_heads=2,num_key_value_heads=1,layer_types=['conv','full_attention','conv'],max_position_embeddings=128)
  self.llm=AutoModelForCausalLM.from_config(cfg,attn_implementation='eager',experts_implementation='eager');self.base=self.llm
  self.projector=nn.Sequential(nn.Linear(5,16),nn.GELU(),nn.Linear(16,16));self.projector.requires_grad_(stage=='A0');self.llm.requires_grad_(stage!='A0')
  if stage!='A0' and recipe!='4f':
   if stage=='A1':self.llm.requires_grad_(False)
   install_research_moe(self.llm,int(recipe[0]),recipe[1]);configure_research_trainability(self.llm,stage=stage)
 def delta(self):return {n:p.detach().clone() for n,p in self.named_parameters() if p.requires_grad or n.startswith('projector.')}

@pytest.mark.parametrize('script',['run','eval','validate'])
def test_actual_metadata_entry_does_not_import_Torch_or_construct_model(script):
 code="import runpy,sys;sys.argv=['x','--metadata'];runpy.run_path('scripts/"+script+"_research_lfm_budget_v1.py',run_name='__main__');assert 'torch' not in sys.modules"
 result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,check=True)
 assert json.loads(result.stdout)['numerical_profile']['model']==b.MODEL


def test_exact_profile_is_distinct_from_Small_and_legacy_resume_refuses():
 from src.utils import research_stock_determinism as stock
 assert b.definition()!=stock.definition() and b.definition()['full_optimizer_moment_bits']==8
 with pytest.raises(ValueError,match='exact LFM'):b.validate_config({'model':b.MODEL,'numerical_profile':stock.definition()})
 module=b.runner();cfg={'model':b.MODEL,'numerical_profile':b.definition(),'budget_context':{'all_full_targets_fit':True,'files':{}}}
 module.validate_resume_config(cfg,copy.deepcopy(cfg))
 bad=dict(cfg,numerical_profile=None)
 with pytest.raises(ValueError):module.validate_resume_config(bad,cfg)

@pytest.mark.parametrize('recipe,stage,backend',[('4f','A1','cpu_grad_bnb_paged_adamw8'),('3c','A1','torch_adamw32'),('1b','SFT','torch_adamw32'),('4f','A0','torch_adamw32')])
def test_exact_stage_optimizer_and_final_only(recipe,stage,backend):
 a=args(recipe=recipe,stage=stage,optimizer_backend=backend);b.validate_budget_args(a)
 a.eval_every=100
 with pytest.raises(ValueError,match='final-only'):b.validate_budget_args(a)


def test_whole_native_context_retains_system_and_full_target_rejects_truncation(tmp_path):
 module=b.runner();a=records(tmp_path,messages=[{'role':'system','content':'Summarize carefully.'},{'role':'user','content':'question?'}]);receipt=b.context_preflight(module,a,tokenizer=Tokenizer())
 assert receipt['counts']['manifest']['records']==1 and receipt['all_full_targets_fit']
 assert receipt['max_expanded_tokens']['manifest']>729+len('Summarize carefully.')
 a.max_length=receipt['max_expanded_tokens']['manifest']-1
 with pytest.raises(ValueError,match='FULL input'):b.context_preflight(module,a,tokenizer=Tokenizer())


def test_context_hash_mutation_and_Qwen_upper_bound(tmp_path):
 a=records(tmp_path);a.vision=b.QWEN;a.vision_preprocessing_profile=b.QWEN_PROFILE;b.validate_budget_args(a)
 receipt=b.context_preflight(b.runner(),a,tokenizer=Tokenizer());assert not receipt['vision_tokens_exact'] and receipt['vision_tokens_per_image_upper_bound']==729
 b.validate_context({'budget_context':receipt});Path(a.manifest).write_text('{}\n')
 with pytest.raises(ValueError,match='manifest changed'):b.validate_context({'budget_context':receipt})

@pytest.mark.parametrize('recipe',['1b','4f','3c'])
def test_genuine_native_tiny_backward_complete_delta_restore_exact(recipe):
 torch.manual_seed(4);model=Tiny(recipe=recipe);ids=torch.tensor([[1,3,4,5,2]]);set_modality_mask(model.base,torch.tensor([[False,True,True,False,False]]))
 loss=model.base(input_ids=ids,labels=ids,use_cache=False).loss;loss.backward();torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01).step()
 state=model.delta();b.validate_complete_state(model,state);torch.manual_seed(4);again=Tiny(recipe=recipe);b.validate_complete_state(again,state);again.load_state_dict(state,strict=False)
 assert all(torch.equal(p,dict(model.named_parameters())[n]) for n,p in again.named_parameters())
 missing=dict(state);missing.pop(next(n for n in state if n.startswith('llm.')))
 with pytest.raises(ValueError,match='omits learned'):b.validate_complete_state(again,missing)
 partial=dict(state);partial.pop('projector.0.bias')
 with pytest.raises(ValueError,match='Complete alignment'):b.validate_complete_state(again,partial)


def test_A1_partial_initialization_requires_every_changed_new_branch_but_not_unchanged_base():
 A1=Tiny('A1','3c');state=A1.delta();SFT=Tiny('SFT','3c');required=b.required_initial_names(SFT,{'stage':'A1','recipe':'3c'})
 b.validate_complete_state(SFT,state,required)
 bad=dict(state);bad.pop(next(n for n in state if 'visual_experts' in n))
 with pytest.raises(ValueError,match='omits learned'):b.validate_complete_state(SFT,bad,required)
 with pytest.raises(ValueError,match='omits learned'):b.validate_complete_state(SFT,state)


def test_source_archive_is_exact_and_idempotent(tmp_path):
 first=b.source_archive(tmp_path,reserve_gb=0);assert b.source_archive(tmp_path,reserve_gb=0)==first
 (tmp_path/'budget_lfm_execution_v1/source.zip').write_bytes(b'changed')
 with pytest.raises(ValueError,match='sidecar'):b.source_archive(tmp_path,reserve_gb=0)


def test_checkpoint_unique_storage_bound_preserves_complete_payload_aliases_RNG():
 import numpy as np
 tensor=torch.arange(12,dtype=torch.float32);payload={'model':{'x':tensor,'alias':tensor},'optimizer':{'moments':torch.ones(12)},'scheduler':{'last_epoch':1},'rng_torch':torch.get_rng_state(),'rng_numpy':np.random.get_state(),'metadata':{'stage':'SFT'}}
 size=b.payload_storage_bytes(payload);assert size<128*2**20+100000
 stream=io.BytesIO();torch.save(payload,stream);assert len(stream.getvalue())<size
 restored=torch.load(io.BytesIO(stream.getvalue()),weights_only=False,map_location='cpu');assert torch.equal(restored['model']['x'],tensor) and restored['metadata']==payload['metadata'] and restored['model']['alias'].data_ptr()==restored['model']['x'].data_ptr()


def test_private_entrypoint_enforces_allowGPU_after_actual_CPU_preflight(tmp_path,monkeypatch):
 a=records(tmp_path);native=b.runner();fake=SimpleNamespace(**native.__dict__);monkeypatch.setattr(b,'runner',lambda *a,**k:fake)
 monkeypatch.setattr(b,'context_preflight',lambda *a,**k:{'all_full_targets_fit':True,'files':{},'total_optimizer_steps':1})
 argv=['--model',b.MODEL,'--stage','A0','--recipe','4f','--manifest',a.manifest,'--validation',a.validation,'--output',str(tmp_path/'run'),'--eval-every','10000','--effective-batch-size','64']
 result=b.train_main(['--preflight','--technical-smoke',*argv]);assert result['numerical_profile']==b.definition()
 with pytest.raises(ValueError,match='allow-gpu'):b.train_main(['--technical-smoke',*argv])
 assert not torch.cuda.is_initialized()


def test_source_proof_binds_every_actual_manifest_and_refuses_mutation(tmp_path):
 a=records(tmp_path);context=b.context_preflight(b.runner(),a,tokenizer=Tokenizer())
 proof=tmp_path/'proof.json';proof.write_text(json.dumps({'status':'passed','profile':'budget18-en-v1','files':{Path(v['path']).name:v for v in context['files'].values()}}))
 bound=b.bind_data_proof(proof,context);assert bound['all_context_manifest_SHA_present']
 b.validate_context({'budget_context':context,'budget_data_proof':bound})
 proof.write_text('{}')
 with pytest.raises(ValueError,match='source proof changed'):b.validate_context({'budget_context':context,'budget_data_proof':bound})
 with pytest.raises(ValueError,match='Passed exact'):b.bind_data_proof(proof,context)


def test_native_template_clock_is_fixed_for_all_consumers():
 class Clock:
  def apply_chat_template(self,*a,**k):return k
 t=b.fixed_clock(Clock());values=t.apply_chat_template([],date_string='foreign',current_date='tomorrow')
 assert values['date_string']==values['current_date']=='2026-11-22'
 assert values['strftime_now']('%Y-%m-%d')=='2026-11-22'
 assert b.fixed_clock(t) is t and b.definition()['fixed_native_template_date_UTC']=='2026-11-22'


def test_real_native_R2_A0_to_SFT_initialization_restores_complete_P_only(tmp_path):
 a=records(tmp_path);context=b.context_preflight(b.runner(),a,tokenizer=Tokenizer());proof=tmp_path/'proof.json'
 proof.write_text(json.dumps({'status':'passed','profile':'budget18-en-v1','files':{Path(v['path']).name:v for v in context['files'].values()}}));bound=b.bind_data_proof(proof,context)
 previous=dict(model=b.MODEL,stage='A0',recipe='4f',numerical_profile=b.definition(),budget_context=context,budget_data_proof=bound)
 aligned=Tiny('A0','4f');state=aligned.delta();model=Tiny('SFT','4f');model.args.numerical_profile=b.definition();model.args.budget_context=context;model.args.budget_data_proof=bound
 model.model_ref={'repo':b.MODEL,'revision':b.REVISION};model.vision_ref={'repo':b.VISION,'revision':'a'*40};model.cache=SimpleNamespace(fingerprint='fixture')
 before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('llm.')}
 module=b.runner();assert module.validate_initial_checkpoint({'config':previous,'model':state},model,model.args)
 assert model._budget_initial_required==set(state)
 b.validate_complete_state(model,state,model._budget_initial_required);model.load_state_dict(state,strict=False)
 assert all(torch.equal(p,before[n]) for n,p in model.named_parameters() if n.startswith('llm.'))
 assert all(torch.equal(p,state[n]) for n,p in model.named_parameters() if n.startswith('projector.'))
 technical=dict(previous,budget_technical_smoke=True)
 with pytest.raises(ValueError,match='Technical smoke'):module.validate_initial_checkpoint({'config':technical,'model':state},model,model.args)
 bad=dict(state);bad.pop('projector.2.bias')
 with pytest.raises(ValueError,match='Complete alignment'):module.validate_initial_checkpoint({'config':previous,'model':bad},model,model.args)


def test_constructor_refuses_missing_profile_before_model_or_cuda_work():
 module=b.runner()
 with pytest.raises(ValueError,match='exact LFM'):module.ResearchModel(SimpleNamespace(model=b.MODEL,vision=b.VISION))
 assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('reserve',[119,float('nan'),float('inf')])
def test_explicit_profile_cannot_lower_disk_floor(reserve):
 with pytest.raises(ValueError,match='120GiB'):b.validate_budget_args(args(reserve_gb=reserve))
