"""CPU source/transition isolation and real native family1c gradient contracts."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from src.utils import research_small_budget_v1 as v1
from src.utils import research_small_budget_v2 as v2
from src.utils import research_stock_determinism as stock


def args(tmp_path, stage='A1', recipe='1c'):
    return SimpleNamespace(model=stock.SMALL, vision=v1.SIG,
        vision_preprocessing_profile=None, recipe=recipe, stage=stage, seed=42,
        epochs=1, effective_batch_size=64, max_length=4096, eval_every=10000,
        max_records=None, max_steps=None, checkpoint_format='legacy', reserve_gb=120.,
        alignment_checkpoint=None, initial_checkpoint=str(v1.FRESH_A0/'best/state.pt'),
        resume=False, selection_manifest=None, text_validation=None, text_manifest=None,
        text_fraction=.15, text_loss_weight=.15 if stage=='SFT' else None,
        selection_metric='group_macro' if stage=='SFT' else 'ce', selection_per_group=32,
        manifest=str(v1.DATA/v1.stage_manifest(stage)),
        validation=str(v1.DATA/('sft_validation.jsonl' if stage=='SFT' else 'alignment_validation.jsonl')),
        output=str(tmp_path/'run'))


def save_source(path, config, *, steps, records):
    path.mkdir();obj=path/'.checkpoint_objects'/'retained';obj.mkdir(parents=True)
    (obj/'state.pt').write_bytes(b'not-read-by-metadata-transition')
    for label in ('best','last'):(path/label).symlink_to(obj,target_is_directory=True)
    config=copy.deepcopy(config);config['output']=str(path)
    config['manifest_sha256']=v1.sha(config['manifest'])
    config['validation_sha256']=v1.sha(config['validation'])
    (path/'resolved_config.json').write_text(json.dumps(config))
    (path/'completed.json').write_text(json.dumps({'steps':steps,'records':records}))
    return path/'best/state.pt'


def test_v1_exact_bytes_globals_and_stock_numerics_preserved():
    before=v1.definition();old_factory=stock.runner
    private=v2.implementation();definition=private.definition()
    assert hashlib.sha256((v2.ROOT/v2.BASE).read_bytes()).hexdigest()==v2.BASE_SHA
    assert definition['owner']==v2.OWNER and definition['supported_recipe']=='1c'
    assert private.OWNER!=v1.OWNER and private is not v1
    assert v1.definition()==before and stock.runner is old_factory
    assert definition['stock_numerical_profile_unchanged'] is True


@pytest.mark.parametrize('recipe', ['1b','4f','3c','2c'])
def test_only_new_1c_dispatch_and_no_full_router_family(tmp_path,recipe):
    with pytest.raises(ValueError,match='only runs native1c'):
        v2.implementation().configure(args(tmp_path,recipe=recipe),v1.RELEASE)


def test_actual_completed_fresh_stock_NF4_A0_admitted_for1c_without_rewrite(tmp_path):
    before=(v1.FRESH_A0/'resolved_config.json').read_bytes()
    target=v2.implementation().configure(args(tmp_path),v1.RELEASE)
    assert target.numerical_profile==stock.definition()
    assert target.budget18_initialization_origin['transition']=='shared_exact_NF4_1b_A0_to_native_1c_A1'
    assert target.budget18_initialization_origin['recipe']=='1b'
    assert target.budget18_initialization_origin['target_recipe']=='1c'
    assert target.budget18_expected_steps==4
    assert before==(v1.FRESH_A0/'resolved_config.json').read_bytes()


@pytest.mark.parametrize('vision,profile',[(v1.SIG,None),(v1.QWEN,'qwen_native_bounded_729')])
def test_v1_budget_A0_can_be_shared_with_same_vision_and1c(tmp_path,vision,profile):
    source=args(tmp_path,stage='A0',recipe='1b');source.initial_checkpoint=None
    source.vision,source.vision_preprocessing_profile=vision,profile
    configured=v1.configure(source,v1.RELEASE)
    checkpoint=save_source(tmp_path/'source',vars(configured),steps=64,records=4096)
    target=args(tmp_path);target.vision,target.vision_preprocessing_profile=vision,profile
    target.initial_checkpoint=str(checkpoint)
    result=v2.implementation().configure(target,v1.RELEASE)
    assert result.budget18_initialization_origin['kind']=='budget18_predecessor'
    assert result.budget18_execution['owner']==v2.OWNER
    assert result.budget18_initialization_origin['recipe']=='1b'


def test_BF16_4f_A0_and_legacy_A1_not_relabelled(tmp_path):
    source=args(tmp_path,stage='A0',recipe='4f');source.initial_checkpoint=None
    configured=v1.configure(source,v1.RELEASE)
    checkpoint=save_source(tmp_path/'wrong',vars(configured),steps=64,records=4096)
    target=args(tmp_path);target.initial_checkpoint=str(checkpoint)
    with pytest.raises(ValueError,match='NF4 A0'):v2.implementation().configure(target,v1.RELEASE)
    config=vars(configured)
    config=dict(config,stage='A1',recipe='3c',manifest=str(v1.DATA/'alignment_a1_train.jsonl'))
    old=save_source(tmp_path/'oldA1',config,steps=4,records=256)
    target=args(tmp_path,stage='SFT');target.initial_checkpoint=str(old)
    with pytest.raises(ValueError):v2.implementation().configure(target,v1.RELEASE)


def test_own_1c_A1_to_SFT_and_complete_gate_keys(tmp_path):
    private=v2.implementation();a1=private.configure(args(tmp_path),v1.RELEASE)
    checkpoint=save_source(tmp_path/'ownA1',vars(a1),steps=4,records=256)
    target=args(tmp_path,stage='SFT');target.initial_checkpoint=str(checkpoint)
    target=private.configure(target,v1.RELEASE)
    assert target.budget18_expected_steps==56
    assert target.budget18_initialization_origin['recipe']=='1c'
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.projector=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.GELU(),torch.nn.Linear(4,4)).requires_grad_(False)
            self.visual_experts=torch.nn.Linear(4,4);self.router_adapter=torch.nn.Linear(4,2)
            self.modality_gate_logits=torch.nn.Parameter(torch.zeros(2,2));self.old=torch.nn.Linear(4,4)
    model=Tiny();state={n:p.detach().clone() for n,p in model.named_parameters() if not n.startswith('old.')}
    assert private.require_learned_state(model,state,set(state))
    missing=dict(state);missing.pop('modality_gate_logits')
    with pytest.raises(ValueError,match='Missing learned'):private.require_learned_state(model,missing,set(state))
    with pytest.raises(ValueError,match='Missing learned'):private.require_learned_state(model,state)


def test_v2_sidecar_distinct_and_contains_exact_v1_source(tmp_path):
    private=v2.implementation();receipt=private.sidecar(tmp_path/'new')
    directory=tmp_path/'new'/'budget18_execution_small_1c_v2'
    assert directory.is_dir() and not (tmp_path/'new'/'budget18_execution_v1').exists()
    import zipfile
    with zipfile.ZipFile(directory/'source.zip') as z:
        assert hashlib.sha256(z.read(v2.BASE)).hexdigest()==v2.BASE_SHA
        assert set(v2.OWN).issubset(z.namelist())
    assert private.sidecar(tmp_path/'new')==receipt


def test_real_native_family1c_freezes_old_router_and_learns_gates_not_family3():
    from src.model.smallthinker import SmallThinkerConfig,SmallThinkerMoeBlock
    from src.model.research_moe import ResearchSmallThinkerMoE
    torch.manual_seed(42)
    config=SmallThinkerConfig(vocab_size=32,hidden_size=8,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=4,moe_ffn_hidden_size=12,
        moe_num_primary_experts=4,moe_num_active_primary_experts=2,
        moe_primary_router_apply_softmax=False,pad_token_id=0,bos_token_id=1,eos_token_id=2)
    native=SmallThinkerMoeBlock(config).requires_grad_(False)
    module=ResearchSmallThinkerMoE(native,1,'c',added_experts=2)
    module.warmup_force_new=True
    module.set_modality_mask(torch.tensor([[True,False,True,False]]))
    inputs=torch.randn(1,4,8,requires_grad=True)
    output,_=module(inputs,inputs);output.square().sum().backward()
    assert module.family==1 and module.modality_gate_logits.grad is not None
    assert module.primary_router.weight.requires_grad is False and module.primary_router.weight.grad is None
    assert module.router_adapter.down.weight.grad is not None
    assert module.modality_gate_logits.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


def test_new_metadata_fresh_process_noTorch_and_original_v1_unaffected():
    code="from src.utils.research_small_budget_v2 import main;main(['--metadata']);import sys;assert 'torch' not in sys.modules"
    result=subprocess.run([sys.executable,'-c',code],cwd=v2.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    metadata=json.loads(result.stdout)
    assert metadata['numerical_profile']==stock.definition()
    assert metadata['budget_execution']['owner']==v2.OWNER

