"""Pure data-capacity conformance; no source downloads, Torch or real pools."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('_primary_capacity_test',Path(__file__).resolve().parents[1]/'src/dataset/research_primary_protocol_v3.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def counts(k=14000):return dict(vqav2=100000,gqa=100000,llava_instruct=10000,textvqa=30000,docvqa=16000,infographic_vqa=10000,chartqa=25000,ai2d=9000,aokvqa=k-3000,scienceqa=3000)

@pytest.mark.parametrize('units,expected',[(334,1000),(14000,42000),(17324,51000),(24999,74000),(25000,75000),(1000000,75000)])
def test_exact_fixed_floor3x_rule_and75000_ceiling(units,expected):assert p.knowledge_exposures(units)==expected

def test_ALL500K_and88236text_preserved_with_largest_remainder_and_cap4():
 c=counts();result=p.primary_quotas(c)
 assert {g:r['exposures'] for g,r in result.items()}==dict(general_vqa=208000,ocr_documents=150000,charts_diagrams=100000,knowledge_reasoning=42000)
 assert sum(r['exposures'] for r in result.values())==500000
 assert {s for r in result.values() for s in r['sources']}==set(c)
 assert all(0<quota<=4*c[s] for r in result.values() for s,quota in r['sources'].items())
 assert p.policy()['sft_text_train_exposures']==88236

@pytest.mark.parametrize('mutation',[{'chartqa':1,'ai2d':1},{'textvqa':1,'docvqa':1,'infographic_vqa':1}])
def test_insufficient_chart_or_OCR_never_silently_redistributed(mutation):
 c=counts();c.update(mutation)
 with pytest.raises(ValueError,match='Insufficient'):p.primary_quotas(c)

def test_every_source_required_positive_and_bool_is_not_count():
 c=counts();c.pop('scienceqa')
 with pytest.raises(ValueError):p.primary_quotas(c)
 c=counts();c['scienceqa']=False
 with pytest.raises(ValueError):p.primary_quotas(c)
 c=counts();c['scienceqa']=0
 with pytest.raises(ValueError):p.primary_quotas(c)

def test_policy_does_not_claim_actual_pools_or_new_authorization():
 policy=p.policy()
 assert policy['primary_ready'] is False and policy['actual_final_group_counts'] is None and policy['actual_common_audited_pool_counts'] is None
 assert policy['multimodal_repeat_cap_per_unit_per_epoch']==4 and policy['data_seed']==42 and policy['internal_validation_component_basis_points']==200
 assert policy['all_target_context_dedup_document_graph_language_context_and_heldout_guards_unchanged'] is True
