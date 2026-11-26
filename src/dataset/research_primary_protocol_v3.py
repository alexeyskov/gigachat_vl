"""Data-only capacity amendment; source normalization and audits stay frozen.

This calculator does not verify supplied counts or authorize training. A real
preparer must verify the common immutable audit pool/graph and fixed2% split.
No PRIMARY metric is accepted as an input to this policy.
"""
import copy,hashlib,importlib.util,re
from pathlib import Path
V2_PATH=Path(__file__).with_name('research_primary_protocol_v2.py')
V2_SHA='a0ab330a1c7b29889fcc5819abdc5db0af129b1486fc884b8d098663472c03c7'
if hashlib.sha256(V2_PATH.read_bytes()).hexdigest()!=V2_SHA:raise ValueError('Frozen primary target-context protocol v2 changed')
spec=importlib.util.spec_from_file_location('_primary_capacity_v3_frozen_target_context',V2_PATH)
v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
v1=v2.v1
PROFILE='primary_train_en_v3_common_audited_capacity'
RULE='knowledge_min75000_floor_3x_common_audited_TRAIN_units_to1000_v1'
GROUP_SOURCES={name:tuple(value['sources']) for name,value in v1.GROUPS.items()}

def knowledge_exposures(eligible_common_train_units):
 if type(eligible_common_train_units) is not int or eligible_common_train_units<1:raise ValueError('Positive exact common audited TRAIN unit count required')
 value=min(75000,(3*eligible_common_train_units//1000)*1000)
 if value<1:raise ValueError('Knowledge common pool cannot cover all declared sources at the frozen1000-exposure granularity')
 return value

def primary_quotas(pool_counts):
 """Pure calculator; actual count/audit verification is the caller's duty."""
 expected={s for values in GROUP_SOURCES.values() for s in values}
 if set(pool_counts)!=expected or any(type(n) is not int or n<1 for n in pool_counts.values()):raise ValueError('All10 positive common audited TRAIN source-unit counts required')
 knowledge=knowledge_exposures(sum(pool_counts[s] for s in GROUP_SOURCES['knowledge_reasoning']))
 budgets={'ocr_documents':150000,'charts_diagrams':100000,'knowledge_reasoning':knowledge,'general_vqa':250000-knowledge}
 result={}
 for group,sources in GROUP_SOURCES.items():
  target=budgets[group];quotas=v1.proportional_quotas({s:pool_counts[s] for s in sources},target,repeat_cap=4)
  result[group]={'exposures':target,'fraction_basis_points':target//50,'sources':quotas}
 if sum(r['exposures'] for r in result.values())!=500000:raise AssertionError('The full500K budget is mandatory')
 return result

def policy():
 result=copy.deepcopy(v1.policy())
 result.update(profile=PROFILE,groups={k:{'sources':list(v)} for k,v in GROUP_SOURCES.items()},
  group_budget_rule={'rule':RULE,'unit_count_scope':'ONE common source pool AFTER all frozen audits AND seed42 joint-component2% TRAIN split','knowledge_reasoning':'min(75000, floor(3 * actual common audited TRAIN knowledge units / 1000) * 1000)','ocr_documents':150000,'charts_diagrams':100000,'general_vqa':'250000 - knowledge_reasoning','final_counts_frozen_before_ANY_PRIMARY_scores':True,'same_final_manifests_all_backbones_donors_recipes':True,'insufficient_group_or_source_capacity':'fail closed; never silently raise cap4, relabel synthetic descriptions as original QA or shrink500K'},
  target_identity_rule=v2.TARGET_RULE,target_owner_rule=v2.OWNER_RULE,
  source_policy_revision_reason='OfficialAOK TRAIN overlaps5950QA on5741COCO image IDs with all40504heldoutVQAv2validationIDs. RemainingAOK11106+optimisticoriginalScience6218=17324 units cap4=69296<75000 before2%split andotherfilters.',
  previous_frozen_target_context_protocol=str(V2_PATH),previous_frozen_target_context_protocol_sha256=V2_SHA,
  actual_common_audited_pool_counts=None,actual_final_group_counts=None,final_primary_manifests_frozen=False,
  text_repeat_cap=1,sft_mm_train_exposures=500000,sft_text_train_exposures=88236,
  actual_eligible_pool_counts_known=False,actual_primary_source_audits_executed=False,
  primary_manifest_exists=False,primary_ready=False,GPU_training_authorized=False,
  metadata_calculator_verifies_external_audit_files=False,common_pool_audit_files_must_be_verified_by_actual_preparer=True,
  all_target_context_dedup_document_graph_language_context_and_heldout_guards_unchanged=True)
 return result
