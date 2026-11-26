#!/usr/bin/env python3
"""NEW metadata-only v3 registry; preserve all sealed v1/v2 source artifacts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'src/dataset/research_primary_protocol_v2.py'
spec=importlib.util.spec_from_file_location('primary_protocol_v2_registry',MODULE)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def immutable(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()
    if path.is_symlink():raise ValueError('Unknown output symlink preserved')
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Versioned metadata already differs; choose a new version')
    else:
        with path.open('xb') as handle:handle.write(raw)


def main():
    old_path=ROOT/'research_results/primary_source_registry_v2.json'
    old_config=ROOT/'configs/research/primary_train_en_v1.json'
    before={str(path.relative_to(ROOT)):sha(path) for path in (old_path,old_config,
        ROOT/'src/dataset/research_primary_protocol.py',ROOT/'tests/test_research_primary_protocol.py',
        ROOT/'configs/research/primary_source_protocol_ready.json')}
    base=read(old_path);config=read(old_config)
    for name,digest in base['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Previous sealed source differs')
    policy=p.policy();policy.update(sources=config['sources'],data_sampling_requires_source_registry_sha256=True,
        source_registry_path='research_results/primary_source_registry_v3.json',
        eligibility_receipt_requires_actual_external_audit_file_verification=True,
        source_registry_and_joint_content_receipt_required=True)
    for key in ('groups','A0_train_exposures','A0_repeat_cap','A1_train_exposures','sft_mm_train_exposures',
                'sft_text_train_exposures','multimodal_repeat_cap_per_unit_per_epoch','text_repeat_cap'):
        if p.v1.canonical(policy[key])!=p.v1.canonical(config[key]):raise ValueError('Preregistered numerical budget changed')
    config_path=ROOT/'configs/research/primary_train_en_v2.json'
    immutable(config_path,policy)
    registry=copy.deepcopy(base);registry.update(registry_revision=3,profile=p.PROFILE,
        previous_registry={'path':str(old_path.relative_to(ROOT)),'sha256':sha(old_path)},
        independent_previous_review={'path':'research_results/primary_protocol_independent_review_v1.json',
            'sha256':sha(ROOT/'research_results/primary_protocol_independent_review_v1.json')},
        configuration={'path':str(config_path.relative_to(ROOT)),'sha256':sha(config_path)},
        sampling_guard_revision='per-original-turn actual-context target ownership; recomputed exact content admission; source/audit verification still pending',
        target_deduplication={'protocol':p.DEDUP_PROTOCOL,'normalization_rule':p.TARGET_RULE,'owner_rule':p.OWNER_RULE,
            'scope':'global full multimodal candidate pool before per-source quotas; never infer actual source audits',
            'identity':'canonical RGB plus complete ordered emitted user/assistant history and current target; exclude representation kind and source',
            'document_policy':'metadata stays content-bound and leakage-unioned; same RGB cannot evade dedup via document aliases; distinct pages are not merged solely by document ID',
            'retained_dialogue_history':'full original preceding turns retained even if an earlier supervised target loses ownership',
            'eligible_turn_mapping':'original source-turn indices; zero-owned units remain in the full leakage graph',
            'unit_mass':'one uniform dialogue unit mass, cycle remaining eligible targets; no extra mass for longer dialogues',
            'immutable_receipts':'MM target pool plus complete supplied A0/MM/text raw-content graph and registry SHA plus external source/audit descriptors; actual audit files remain unverified',
            'actual_global_target_ownership_executed':False,'actual_source_audits_executed':False},
        eligibility_known_gaps=['actual post-audit source pools and complete pinned cross-stage candidate graph not materialized',
            'original QA IDs and original TRAIN correspondence require actual source joins, not packaging row indices',
            'LLaVA original ordered dialogue correspondence is not established by its source alias',
            'DocVQA/InfographicVQA original document/page component correspondence requires external mapping',
            'source sample metadata may have null nested image.path/source and no original ID fields; never infer verified linkage',
            'all official heldout coverage, language, context, near-image/document and text-contamination audits incomplete',
            'actual immutable external source/audit receipt files have not been verified',
            'full primary feature-cache/checkpoint disk admission remains pending'],
        primary_500K_manifest_exists=False,primary_train_contamination_clearance=False,
        primary_training_authorized=False,actual_eligible_source_pools_known=False)
    registry['source_sha256'].update({name:sha(ROOT/name) for name in (
        'src/dataset/research_primary_protocol_v2.py','scripts/prepare_research_primary_registry_v3.py',
        'tests/test_research_primary_protocol_v2.py')})
    if any(s['primary_eligible_pool_count'] is not None for s in registry['sources']):raise ValueError('No real eligible pool count may be invented')
    immutable(ROOT/'research_results/primary_source_registry_v3.json',registry)
    if before!={name:sha(ROOT/name) for name in before}:raise ValueError('Previous sealed release changed')
    print(json.dumps({'status':'source_metadata_and_sampling_protocol_only','profile':p.PROFILE,
        'registry_revision':3,'primary_training_authorized':False,'previous_release_unchanged':True}))


if __name__=='__main__':main()
