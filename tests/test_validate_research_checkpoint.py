from pathlib import Path
import importlib.util
import json
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.validate_research_checkpoint import structural_checks,select_validation_records,cpu_native_full_checks


def test_structural_checks_distinguish_missing_and_nonfinite():
    config={'stage':'SFT'}
    payload={'config':config,'model':{'llm.x':torch.ones(3),'projector.0.weight':torch.ones(2,3)}}
    result=structural_checks(payload,config,{'llm.x':3})
    assert result['validated_tensor_count']==2 and result['expected_keys_match']
    payload['model']['llm.x'][0]=float('nan')
    with pytest.raises(ValueError,match='Non-finite'):
        structural_checks(payload,config,{'llm.x':3})


def test_validation_selection_reproduces_sft_text_mix_and_seed():
    import random
    records={'vision':[{'id':i} for i in range(8)],'text':[{'id':i} for i in range(8,12)]}
    config={'stage':'SFT','validation':'vision','text_manifest':'train-text','text_validation':'text','eval_records':5,'seed':42}
    chosen=select_validation_records(config,lambda path:list(records[path]))
    assert chosen==random.Random(42).sample(records['vision']+records['text'],5)


@pytest.mark.parametrize('recipe',['4f','3b','2c'])
def test_cpu_native_full_reconstructs_tiny_backbone_and_projector(tmp_path,recipe):
    from src.model.smallthinker import SmallThinkerConfig,SmallThinkerForCausalLM
    from src.model.research_moe import install_research_moe
    from torch import nn
    config=SmallThinkerConfig(vocab_size=32,hidden_size=8,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=4,
        moe_ffn_hidden_size=12,moe_num_primary_experts=2,moe_num_active_primary_experts=1)
    config.save_pretrained(tmp_path)
    model=SmallThinkerForCausalLM(config)
    if recipe[0] in '23': install_research_moe(model,int(recipe[0]),recipe[1],added_experts=2)
    projector=nn.Sequential(nn.Linear(5,8),nn.GELU(),nn.Linear(8,8))
    saved={'llm.'+name:param.detach().clone() for name,param in model.named_parameters()}
    saved.update({'projector.'+name:tensor.detach().clone() for name,tensor in projector.state_dict().items()})
    cfg={'model':str(tmp_path),'recipe':recipe,'stage':'SFT'}
    result=cpu_native_full_checks({'model':saved},cfg)
    assert result['validation_mode']=='cpu_native_full_reconstruction'
    assert result['native_logits_shape']==[1,4,32]
    assert result['native_logits_roundtrip_max_abs_diff']==0
    assert result['heldout_ce_replayed'] is False
    assert result['projector_frozen']


def test_restore_comparison_rejects_lossy_dtype_cast():
    from scripts.validate_research_checkpoint import _tensor_equal_after_restore
    with pytest.raises(ValueError,match='differs'):
        _tensor_equal_after_restore(torch.tensor([1.],dtype=torch.bfloat16),torch.tensor([1.001]),'weight')


def test_cpu_structural_receipt_is_never_full_pass(tmp_path):
    from scripts.validate_research_checkpoint import validate_run
    from src.utils.research_storage import BestLastCheckpointManager,DiskBudget
    cfg={'stage':'SFT'}
    (tmp_path/'resolved_config.json').write_text(json.dumps(cfg))
    (tmp_path/'trainable_parameters.json').write_text(json.dumps({'llm.x':3}))
    (tmp_path/'completed.json').write_text(json.dumps({'steps':1}))
    manager=BestLastCheckpointManager(tmp_path,DiskBudget(tmp_path,0))
    payload={'step':1,'config':cfg,'model':{'llm.x':torch.ones(3),'projector.0.weight':torch.ones(2,3)}}
    manager.save_torch(payload,step=1,metric=.5,anticipated_bytes=10000)
    result=validate_run(tmp_path,cpu_structural_only=True)
    assert result['status']=='structural_only'
    assert result['checkpoints'][0]['status']=='structural_only'
    assert result['checkpoints'][0]['checkpoint_path']==str((tmp_path/'last/state.pt').resolve())
    assert result['checkpoints'][0]['checkpoint_size_bytes']==(tmp_path/'last/state.pt').stat().st_size


def internal_records():
    return [{'record_id':group,'image_paths':[group+'.png'],'question':'question','answer':'yes',
             'task_group':group,'partition':'validation','source':{'repo':'train','split':'train'}}
            for group in ['general_vqa','ocr_documents','charts_diagrams','knowledge_reasoning']]


def test_group_selection_replay_uses_full_manifest_not_ce_record_cap():
    records=internal_records()
    config={'stage':'SFT','selection_metric':'group_macro','validation':'internal','seed':42,
            'eval_records':1,'selection_per_group':1}
    selected=select_validation_records(config,lambda _:records+records)
    assert len(selected['records'])==4
    assert selected['protocol']['name']=='internal_teacher_reference_group_macro'


def test_group_macro_full_replay_against_saved_marker_and_protocol(monkeypatch):
    from torch import nn
    from src.eval.research_selection import build_internal_validation,PROTOCOL_NAME
    from scripts.validate_research_checkpoint import full_model_checks
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.llm=nn.Linear(2,2)
            self.projector=nn.Linear(2,2).requires_grad_(False)
    model=Tiny()
    selected=build_internal_validation(internal_records(),per_group=1)
    payload={'model':{k:v.detach().clone() for k,v in model.state_dict().items()},
             'config':{'stage':'SFT','selection_metric':'group_macro','selection_max_new_tokens':2},
             'selection_metric':PROTOCOL_NAME,'selection_protocol':selected['protocol']}
    monkeypatch.setattr('src.eval.research_selection.generate_prediction',lambda *a,**k:{'prediction':'yes'})
    receipt=full_model_checks(model,payload,selected,1,100.,.001,.0001)
    assert receipt['selection_metric_replayed']
    assert receipt['group_macro_score_replayed']==100
    assert receipt['heldout_ce_replayed'] is False
    assert receipt['official_benchmark_score'] is False
    assert model.training and not model.projector.weight.requires_grad
    with pytest.raises(ValueError,match='group macro replay mismatch'):
        full_model_checks(model,payload,selected,1,0.,.001,.0001)
    payload['selection_protocol']=dict(selected['protocol'],selection_manifest_sha256='wrong')
    with pytest.raises(ValueError,match='differs from saved selection'):
        full_model_checks(model,payload,selected,1,100.,.001,.0001)


def test_group_selection_data_sha_and_cpu_structural_do_not_claim_quality(tmp_path):
    import hashlib
    from scripts.validate_research_checkpoint import validate_run,check_manifest_hashes
    from src.utils.research_storage import BestLastCheckpointManager,DiskBudget
    from src.eval.research_selection import build_internal_validation,PROTOCOL_NAME
    manifest=tmp_path/'internal.jsonl'
    rows=internal_records()
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    cfg={'stage':'SFT','selection_metric':'group_macro','selection_manifest':str(manifest),
         'selection_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (tmp_path/'resolved_config.json').write_text(json.dumps(cfg))
    (tmp_path/'trainable_parameters.json').write_text(json.dumps({'llm.x':3}))
    (tmp_path/'completed.json').write_text(json.dumps({'steps':1}))
    manager=BestLastCheckpointManager(tmp_path,DiskBudget(tmp_path,0),mode='max')
    payload={'step':1,'config':cfg,'model':{'llm.x':torch.ones(3),'projector.0.weight':torch.ones(2,3)},
             'selection_metric':PROTOCOL_NAME,'selection_protocol':build_internal_validation(rows,per_group=1)['protocol']}
    manager.save_torch(payload,step=1,metric=100.,anticipated_bytes=10000)
    result=validate_run(tmp_path,cpu_structural_only=True)
    assert result['status']=='structural_only'
    assert result['checkpoints'][0]['selection_metric_replayed'] is False
    assert result['checkpoints'][0]['official_benchmark_score'] is False
    manifest.write_text(manifest.read_text()*2)
    with pytest.raises(ValueError,match='selection_manifest changed'):
        check_manifest_hashes(cfg)
