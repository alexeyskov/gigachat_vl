import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from src.utils import research_small_budget_v3 as small
from src.utils import research_lfm_budget_v3 as lfm
from src.utils import research_giga_budget_v3 as giga
from src.utils.research_resource_vision_v3 import derived, composed
from src.utils import research_derived_vision_v1 as original_vision


def test_profiles_declare_resource_change_and_unchanged_compute():
    for module in (small, lfm, giga):
        profile = module.definition()
        assert profile['minimum_disk_reserve_GiB'] == 32
        assert profile['R2_final_save'] == 'completed_model_only_no_optimizer'
    for module in (lfm, giga):
        profile = module.definition()
        assert profile['full_master_dtype'] == 'float32'
        assert profile['compute_dtype'] == 'bfloat16'
        assert profile['recipes'] == ['1b', '4f', '1c']


@pytest.mark.parametrize('module', [lfm, giga.implementation()])
@pytest.mark.parametrize('recipe,optimizer', [('1b','torch_adamw32'),
                                            ('1c','torch_adamw32'),
                                            ('4f','cpu_grad_bnb_paged_adamw8')])
def test_actual_budget_guard_accepts32_and_correct_optimizer_only(module, recipe, optimizer):
    args = SimpleNamespace(reserve_gb=32,model=module.MODEL,vision=module.VISION,
        max_length=4096,recipe=recipe,seed=42,epochs=1,effective_batch_size=64,
        eval_every=10000,checkpoint_format='legacy',alignment_checkpoint=None,
        max_steps=None,max_records=None,stage='SFT',optimizer_backend=optimizer,
        text_loss_weight=.15,vision_preprocessing_profile=None)
    module.validate_budget_args(args)
    args.reserve_gb=31
    with pytest.raises(ValueError):module.validate_budget_args(args)
    args.reserve_gb=32
    args.optimizer_backend='cpu_grad_bnb_paged_adamw8' if recipe!='4f' else 'torch_adamw32'
    with pytest.raises(ValueError, match='optimizer'):module.validate_budget_args(args)


def test_small_model_restore_math_is_literal_frozen_copy():
    def node(path, name):
        tree=ast.parse(Path(path).read_text())
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('require_complete_P','require_learned_state','patch_module'):
        assert ast.dump(node('src/utils/research_small_budget_v1.py',name)) == ast.dump(
            node('src/utils/research_small_budget_v3.py',name))


def test_resource_vision_preserves_exact_existing_export_and_cache_keys():
    for vision, profile in [('Qwen/Qwen2.5-VL-7B-Instruct','qwen_native_bounded_729')]:
        old=original_vision.resolve_derived_vision(vision,preprocessing_profile=profile)
        current=derived.resolve_derived_vision(vision,preprocessing_profile=profile)
        assert old==current
        assert derived.code_binding()==original_vision.code_binding()
        args=SimpleNamespace(model=small.SIG,vision=vision,vision_preprocessing_profile=profile,reserve_gb=32)
        args.model='Tiiny/SmallThinker-4BA0.6B-Instruct'
        composed().configure(args,current)
        assert args.derived_stock_execution['resource_execution']['disk_floor_GiB']==32
        args.reserve_gb=31
        with pytest.raises(ValueError):composed().configure(args,current)


def test_1c_A1_restores_all_learned_branches_without_requiring_frozen_LoRA():
    names=['projector.0.weight','llm.visual_experts.0.weight','llm.router_adapter.down.weight',
           'llm.modality_gate_logits','llm.q_proj.lora_A.weight']
    model=SimpleNamespace(named_parameters=lambda:[(n,SimpleNamespace(requires_grad=True)) for n in names])
    result=lfm.required_initial_names(model,{'stage':'A1','recipe':'1c'})
    assert result==set(names[:-1])


def test_cache_retirement_protects_current_data_and_refuses_stat_drift():
    from scripts.retire_research_budget18_cache_v1 import validate_candidate,plan
    # Keep this read-only guard fixture available after the old candidates retire.
    row=json.loads(next((plan.OUT/'protected_features.jsonl').open()))
    row['classification']='candidate_recomputable_old_feature'
    with pytest.raises(ValueError,match='outside-budget'):
        validate_candidate(row,{row['metadata']['image_hash']})
    changed=copy.deepcopy(row);changed['stat']['st_ino']+=1
    with pytest.raises(ValueError,match='identity'):
        validate_candidate(changed,set())
    result=validate_candidate(row,set())
    assert result['tensor_checksum_verified'] and len(result['whole_file_sha256'])==64
