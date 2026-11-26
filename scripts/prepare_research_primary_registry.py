#!/usr/bin/env python3
"""Write prospective source inventory/config only; never prepare train shards."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('primary_registry_policy',ROOT/'src/dataset/research_primary_protocol.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def immutable(path,value):
    path=Path(path);raw=(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()
    if path.exists():
        if path.is_symlink() or path.read_bytes()!=raw:raise ValueError('Existing primary registry differs; use a versioned output')
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
def cards(evidence):
    import yaml
    result={}
    for item in evidence['readmes']:
        path=ROOT/item['retained_path'];assert sha(path)==item['sha256']
        text=path.read_text();front=yaml.safe_load(text.split('---',2)[1]) if text.startswith('---') else {}
        configs=front.get('dataset_info',[]);configs=[configs] if isinstance(configs,dict) else configs
        result[item['repo']]={c.get('config_name','default'):c for c in configs}
    return result


def main():
    evidence_path=ROOT/'research_results/primary_source_inventory/evidence_v1/evidence.json'
    evidence=read(evidence_path);c=cards(evidence)
    paths={'finevision':'datasets/research/FineVision.files.json','gqa':'datasets/research/eval-metadata/lmms-lab--GQA.json',
        'smoltalk':'datasets/research/smoltalk.files.json','pixmo_repack':'datasets/research/pixmo-cap-images.files.json',
        'cauldron':'research_results/primary_source_inventory/cauldron_metadata.json'}
    meta={key:read(ROOT/path) for key,path in paths.items()}
    pins={'finevision':'3c380a731a3429c1d04693d6ec16d7e683def84c','gqa':'a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8',
        'smoltalk':'5feaf2fd3ffca7c237fc38d1861bc30365d48ffa','pixmo_repack':'cad0be87ded859f1016349845d336c96651ce65a',
        'cauldron':'847a98a779b1652d65111daf20c972dfcd333605'}
    assert all(meta[key]['revision']==value for key,value in pins.items())
    entries=[]
    def entry(source_id,key,config,prefix,unit,scorer=None):
        m=meta[key];files=[f for f in m['files'] if f['path'].startswith(prefix) and f['path'].endswith('.parquet')]
        if not files or any('/train-' not in f['path'] for f in files):raise ValueError('Registry must contain only explicit TRAIN shards')
        card=c.get(m['repo'],{}).get(config,{})
        splits=[s for s in card.get('splits',[]) if s['name']=='train']
        values={'source_id':source_id,'repo':m['repo'],'revision':m['revision'],'config':config,'split':'train',
            'namespace':f"{m['repo']}@{m['revision']}:{config}:train",'files':files,
            'whole_declared_source_train_shards':len(files),'whole_declared_source_train_bytes':sum(f['size'] for f in files),
            'candidate_parent_rows_from_pinned_card':splits[0]['num_examples'] if len(splits)==1 else None,
            'counts_are_prefilter_not_eligible_unique_units':True,'sampling_unit':unit,'heldout_scorer_family':scorer,
            'licensing_references':[f"https://huggingface.co/datasets/{m['repo']}/blob/{m['revision']}/README.md"],
            'license_and_constituent_rights_verified':False,'primary_eligible_pool_count':None,
            'source_inventory_path':paths[key],'source_inventory_sha256':sha(ROOT/paths[key])}
        if source_id in p.FINEVISION_CONFIGS or source_id=='gqa':values['task_group']=p.group_for(source_id)
        entries.append(values);return values
    scorers={'vqav2':'vqa_accuracy/vqav2_official_legacy','gqa':'exact_match/gqa_official_normalization_sensitivity',
        'textvqa':'vqa_accuracy/textvqa_evalai','docvqa':'anls','infographic_vqa':'anls',
        'chartqa':'chart_relaxed','ai2d':'mcq_accuracy','aokvqa':'mcq_accuracy','scienceqa':'mcq_accuracy'}
    for source_id,config in p.FINEVISION_CONFIGS.items():
        entry(source_id,'finevision',config,config+'/',
            'one verified original ordered dialogue; one target turn per exposure' if source_id=='llava_instruct' else
            'one independent QA within an image-grouped source row',scorers.get(source_id))
    gqa=entry('gqa','gqa','train_balanced_instructions','train_balanced_instructions/','one balanced original TRAIN question-ID joined to original TRAIN image-ID',scorers['gqa'])
    images=[f for f in meta['gqa']['files'] if f['path'].startswith('train_balanced_images/')]
    gqa['image_config']='train_balanced_images';gqa['image_files']=images
    gqa['image_shard_bytes']=sum(f['size'] for f in images);gqa['image_shards']=len(images)
    gqa['join']={'instruction_key':'imageId','image_key':'id','instruction_identity':'id','requires_isBalanced':True,
        'train_test_mix_allowed':False,'row_position_join_allowed':False}
    for key in ('finevision','cauldron'):
        named=[f['path'] for f in meta[key]['files'] if 'gqa' in f['path'].split('/')[0].lower()]
        if named:raise ValueError('Named-GQA inventory assumption changed; new protocol review required')
    entry('pixmo_repack','pixmo_repack','default','data/train-',
        'one canonical RGB image-caption pair; original PixMo URL/ID/caption correspondence must be verified')
    entry('smoltalk','smoltalk','all','data/all/train-',
        'one original EN conversation unit, one deterministic supervised target turn per exposure; original role/history preserved')
    heldout=read(ROOT/'research_results/full_eval_image_exclusion_audit.json')
    policy=p.policy();policy['sources']=[e['source_id'] for e in entries]
    policy['data_sampling_requires_source_registry_sha256']=True
    registry={'schema_version':1,'registry_revision':2,'profile':p.PROFILE,'status':'source_metadata_and_sampling_protocol_only',
        'previous_registry_preserved':'research_results/primary_source_inventory/registry_v1_review_archive/preservation.json',
        'sampling_guard_revision':'reject duplicate normalized keys; fixed seed42 and2% split; numerical policy unchanged',
        'sources':entries,'source_metadata_sha256':{name:sha(ROOT/path) for name,path in paths.items()},
        'bounded_evidence':{'path':str(evidence_path.relative_to(ROOT)),'sha256':sha(evidence_path),
            'conservative_download_content_bytes':evidence['conservative_content_bytes_including_previous_gzip_attempt'],
            'phase_cap_bytes':20*2**20,'footer_counts':evidence['footer_observations']},
        'scienceqa_equal_source_quota_counterexample':evidence['scienceqa_capacity_counterexample'],
        'named_GQA_configuration_inventory':{'FineVision_absent':True,'Cauldron_absent':True,
            'scope':'named GQA configurations/shards absent from these pinned inventories; GQA-derived mixed-source content is not certified absent'},
        'text_heldout_exclusion':{'path':'datasets/research/primary-text-en-v1/released_test_exclusion.jsonl',
            'sha256':sha(ROOT/'datasets/research/primary-text-en-v1/released_test_exclusion.jsonl'),'released_rows':15902},
        'visual_heldout_exclusion':{'audit_path':'research_results/full_eval_image_exclusion_audit.json',
            'audit_sha256':sha(ROOT/'research_results/full_eval_image_exclusion_audit.json'),
            'unique_RGB_images':heldout['unique_heldout_images'],'coverage':'all DECLARED heldout splits only',
            'all_official_coverage_complete':False,'remaining_coverage_gaps':heldout['remaining_heldout_groups']},
        'source_sha256':{name:sha(ROOT/name) for name in ('src/dataset/research_primary_protocol.py',
            'scripts/prepare_research_primary_registry.py','tests/test_research_primary_protocol.py',
            'research_results/collect_primary_source_evidence.py')},
        'primary_500K_manifest_exists':False,'primary_train_contamination_clearance':False,
        'A0_download_source_choice':'dnth/pixmo-cap-images third-party packing; original allenai PixMo correspondence remains mandatory and not yet certified',
        'storage':{'reserve_GiB':120,'large_downloads_authorized_by_this_registry':False,
            'forecast_before_download_required':True,'feature_cache_floor_A0_200K_SigLIP729_BF16_bytes':200000*729*1152*2,
            'retained_all_models_checkpoints_and_current_pilot_caches_must_be_subtracted_first':True}}
    immutable(ROOT/'configs/research/primary_train_en_v1.json',policy)
    immutable(ROOT/'research_results/primary_source_registry_v2.json',registry)
    print(json.dumps({'status':registry['status'],'sources':len(entries),'MM_sources':10,
        'registry_sha256':sha(ROOT/'research_results/primary_source_registry_v2.json'),
        'policy_sha256':sha(ROOT/'configs/research/primary_train_en_v1.json')}))


if __name__=='__main__':main()
