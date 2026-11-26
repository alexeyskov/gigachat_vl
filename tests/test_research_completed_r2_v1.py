import json,random
from pathlib import Path
from types import ModuleType,SimpleNamespace
import numpy as np
import pytest
import torch
from src.utils import research_completed_r2_v1 as policy
from src.utils.research_storage import BestLastCheckpointManager,DiskBudget,_atomic_json
from src.utils.research_budget_storage_v1 import install_manager

def fixture(tmp_path,monkeypatch):
 from scripts.run_research_stage import write_selection_artifacts
 from src.eval import research_selection
 config=dict(model='Tiiny/SmallThinker-4BA0.6B-Instruct',recipe='4f',stage='SFT',seed=42,epochs=1,effective_batch_size=64,eval_every=10000,max_steps=None,max_records=None,selection_metric='group_macro',selection_manifest=None,text_validation=None,text_loss_weight=.15,budget18_expected_steps=56,output=str(tmp_path),reserve_gb=120.)
 args=SimpleNamespace(**config);policy.attach(args);config=vars(args).copy();_atomic_json(tmp_path/'resolved_config.json',config)
 (tmp_path/'metrics.jsonl').write_text(json.dumps({'event':'train','step':56,'completed_records':3529})+'\n')
 result={'protocol':{'name':'internal_group_macro_v1','selection_manifest_sha256':'a'*64},'metrics':{'records':116,'group_macro_score':25.,'groups':{n:{'records':29,'score':25.} for n in ('general_vqa','ocr_documents','charts_diagrams','knowledge_reasoning')}},'predictions':[{'record_id':str(i),'prediction':'x','score':25.} for i in range(116)]}
 monkeypatch.setattr(research_selection,'evaluate_internal_group_macro',lambda *a,**k:result)
 module=ModuleType('_tiny_completed_R2');module.main=lambda:None;module.parse_args=lambda:args;module.BestLastCheckpointManager=BestLastCheckpointManager;module.evaluate_text_only_ce=lambda *a:{'text_only_records':34,'text_only_supervised_tokens':100,'text_only_heldout_ce':2.};module.write_selection_artifacts=write_selection_artifacts
 install_manager(module);policy.install_module(module)
 model=SimpleNamespace(args=args);module._completed_R2_group_macro(model,None);module.evaluate_text_only_ce(model,[{} for _ in range(34)],1)
 base=torch.randn(4,4);state={'llm.weight':base,'llm.tie':base,'llm.view':base[1:],'projector.0.weight':torch.randn(4,4),'projector.0.bias':torch.randn(4),'projector.2.weight':torch.randn(4,4),'projector.2.bias':torch.randn(4)}
 payload={'model':state,'optimizer':{'moments':torch.ones(4)},'scheduler':{'last_epoch':56},'step':56,'seen':3529,'config':config,'selection_protocol':result['protocol'],'selection_statistics':result['metrics'],'selection_value':25.,'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':[],'elapsed_seconds':42.5,'model_ref':{'repo':config['model'],'revision':'a'*40},'vision_ref':{'repo':'vision','revision':'b'*40},'manifest_sha256':'c'*64}
 manager=module.BestLastCheckpointManager(tmp_path,disk_budget=DiskBudget(tmp_path,0),mode='max');return module,manager,payload

def save(manager,payload):return manager.save_torch(payload,step=56,metric=25.,anticipated_bytes=2**20,checkpoint_format='legacy')
def test_native_final_model_bits_RNG_scheduler_aliases_and_original_payload_preserved(tmp_path,monkeypatch):
 module,manager,payload=fixture(tmp_path,monkeypatch);original_optimizer=payload['optimizer'];save(manager,payload)
 assert payload['optimizer'] is original_optimizer and 'training_completed' not in payload
 assert (tmp_path/'best').resolve()==(tmp_path/'last').resolve()
 restored=torch.load(tmp_path/'last/state.pt',map_location='cpu',mmap=True,weights_only=False)
 assert 'optimizer' not in restored and restored['training_completed'] and not restored['resumable_optimizer'] and restored['scheduler']==payload['scheduler']
 for n,t in payload['model'].items():assert torch.equal(t.view(torch.uint8),restored['model'][n].view(torch.uint8))
 assert restored['model']['llm.tie'].untyped_storage()._cdata==restored['model']['llm.view'].untyped_storage()._cdata
 assert restored['rng_python']==payload['rng_python'] and np.array_equal(restored['rng_numpy'][1],payload['rng_numpy'][1]) and torch.equal(restored['rng_torch'],payload['rng_torch'])
 budget=json.loads((tmp_path/'checkpoint_storage_budget.json').read_bytes());assert budget['unique_tensor_storage_bytes']<sum(t.numel()*t.element_size() for t in payload['model'].values())+8192
 assert (tmp_path/'last/.completed_r2_final.json').is_file()

@pytest.mark.parametrize('mutation',['step','seen','selection','text','config'])
def test_incomplete_or_mismatched_final_evidence_refuses_before_publication(tmp_path,monkeypatch,mutation):
 module,manager,payload=fixture(tmp_path,monkeypatch)
 if mutation in ('step','seen'):payload[mutation]-=1
 elif mutation=='selection':(tmp_path/'selection/last.predictions.jsonl').write_text('corrupt\n')
 elif mutation=='text':
  p=tmp_path/policy.DIRECTORY/'end_validation.json';x=json.loads(p.read_bytes());x['text']['text_only_records']=33;_atomic_json(p,x)
 else:payload['config']=dict(payload['config'],recipe='1b')
 with pytest.raises((ValueError,KeyError)):save(manager,payload)
 assert not (tmp_path/'last').exists()

def test_published_final_without_completed_marker_recovers_no_optimizer_replay(tmp_path,monkeypatch):
 module,manager,payload=fixture(tmp_path,monkeypatch);save(manager,payload);assert not (tmp_path/'completed.json').exists()
 with pytest.raises(ValueError,match='invocation'):policy.recover_if_published(tmp_path,expected_args=dict(payload['config'],seed=123))
 # The production main wrapper must return before entering canonical main;
 # this tiny module deliberately has no model/optimizer/training constructors.
 assert module.main() is None;done=json.loads((tmp_path/'completed.json').read_bytes());assert done['steps']==56 and done['records']==3529 and done['optimizer_resume_performed'] is False
 assert policy.recover_if_published(tmp_path)

def test_crash_between_best_and_last_pointer_recovered_only_matching_owned_object(tmp_path,monkeypatch):
 _,manager,payload=fixture(tmp_path,monkeypatch);save(manager,payload);(tmp_path/'last').unlink()
 assert policy.recover_if_published(tmp_path) and (tmp_path/'best').resolve()==(tmp_path/'last').resolve()
 assert json.loads((tmp_path/policy.DIRECTORY/'recovered.json').read_bytes())['repaired_missing_roles']==['last']

def test_actual_crash_after_immutable_object_rename_before_first_pointer_recovers_without_updates(tmp_path,monkeypatch):
 _,manager,payload=fixture(tmp_path,monkeypatch)
 monkeypatch.setattr(manager,'_set_pointer',lambda *a:(_ for _ in ()).throw(SystemExit('power loss before firstpointer')))
 with pytest.raises(SystemExit):save(manager,payload)
 assert not (tmp_path/'best').exists() and not (tmp_path/'last').exists()
 assert len(list((tmp_path/'.checkpoint_objects').glob('object-*')))==1
 assert policy.recover_if_published(tmp_path) and (tmp_path/'best').resolve()==(tmp_path/'last').resolve()
 assert json.loads((tmp_path/'completed.json').read_bytes())['optimizer_resume_performed'] is False

@pytest.mark.parametrize('mutation',['state_bytes','intent','source','foreign_role'])
def test_recovery_corruption_refuses_without_fabricating_completion(tmp_path,monkeypatch,mutation):
 _,manager,payload=fixture(tmp_path,monkeypatch);save(manager,payload)
 if mutation=='state_bytes':
  p=tmp_path/'last/state.pt'
  with p.open('r+b') as f:f.seek(200);b=f.read(1);f.seek(200);f.write(bytes([b[0]^1]))
 elif mutation=='intent':
  p=tmp_path/policy.DIRECTORY/'intent.json';x=json.loads(p.read_bytes());x['config_sha256']='0'*64;_atomic_json(p,x)
 elif mutation=='source':(tmp_path/policy.DIRECTORY/'source.py').write_text('different')
 else:(tmp_path/'last').unlink();(tmp_path/'last').symlink_to(tmp_path/'outside')
 with pytest.raises((ValueError,FileNotFoundError)):policy.recover_if_published(tmp_path)
 assert not (tmp_path/'completed.json').exists()

def test_only4fSFT_policy_and_private_backend_scope(tmp_path,monkeypatch):
 _,manager,payload=fixture(tmp_path,monkeypatch)
 for cfg in (dict(payload['config'],stage='A1'),dict(payload['config'],recipe='1c'),dict(payload['config'],budget18_expected_steps=55)):
  with pytest.raises(ValueError):policy.eligible(cfg)
 module=ModuleType('scripts.run_research_stage')
 with pytest.raises(ValueError):policy.install_module(module)
 original=lambda:'raw';backend=SimpleNamespace(patch_module=original)
 with policy.patched_backend(backend):assert backend.patch_module is not original
 assert backend.patch_module is original
