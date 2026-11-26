#!/usr/bin/env python3
"""Header/processor-math census of future donor cache sizes; no model forward.

Image identity uses existing pinned canonical-RGB hashes. Current image headers
and EXIF orientation are read, without re-decoding or rehashing every pixel.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import copy
import argparse
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def emit(event,**values):print(json.dumps({'event':event,'time':time.time(),**values}),flush=True)
def jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def distribution(values):
    values=sorted(values)
    return {'count':len(values),'min':values[0],'max':values[-1],'mean':sum(values)/len(values),
        'p50':values[math.ceil(.5*len(values))-1],'p90':values[math.ceil(.9*len(values))-1],
        'p95':values[math.ceil(.95*len(values))-1],'p99':values[math.ceil(.99*len(values))-1],
        'histogram':{str(k):v for k,v in sorted(Counter(values).items())}}


def serialize_overhead(torch,n,dimension,metadata):
    feature=torch.zeros((n,dimension),dtype=torch.bfloat16)
    payload={'schema_version':1,'fingerprint':'f'*64,'image_hash':'a'*64,'features':feature,
             'checksum':'b'*64,'shape':[n,dimension],'storage_dtype':str(feature.dtype),
             'valid_token_count':n,'metadata':metadata}
    output=io.BytesIO();torch.save(payload,output)
    return output.tell()-feature.numel()*feature.element_size()



def storage_forecast(report,root=ROOT):
    """Capacity lower bounds from retained actual SmallThinker tensor inventories.

    Completion-model tensors are counted at current FP32 master dtype; old BF16
    file sizes must not size future full-training checkpoints. No deletion,
    lossy conversion, compression gain or remote archive is assumed.
    """
    full_recipes=('4f','2a','2b','2c','3a','3b','3c')
    quant_recipes=('4q','1a','1b','1c');full={};quant={};evidence={}
    for recipe in full_recipes:
        run=root/'runs/research'/f'smallthinker4__siglip2__{recipe}__SFT_smoke_s42'
        receipt=json.loads((run/'checkpoint_roundtrip.json').read_text())
        elements=receipt['checkpoints'][0]['saved_parameter_elements']
        full[recipe]=elements*4
        evidence[recipe]={'saved_parameter_elements':elements,'future_storage_bytes_per_element':4,
                          'future_model_tensor_bytes':full[recipe],'receipt_path':str(run/'checkpoint_roundtrip.json')}
    from huggingface_hub import hf_hub_download
    binding=report['tokenizer_binding']['model_ref']
    config_path=Path(hf_hub_download(binding['repo'],'config.json',revision=binding['revision'],local_files_only=True))
    hidden=json.loads(config_path.read_text())['hidden_size']
    projector_by_donor={name:(dimension+1)*hidden+(hidden+1)*hidden for name,dimension in [('siglip',1152),('qwen',3584),('gemma',2560)]}
    projector_elements=projector_by_donor['siglip']
    for recipe in quant_recipes:
        run=root/'runs/research'/f'smallthinker4__siglip2__{recipe}__SFT_smoke_s42'
        ready=next(row for row in jsonl(run/'metrics.jsonl') if row.get('event')=='ready')
        quant[recipe]=(ready['trainable']+projector_elements)*4
        evidence[recipe]={'trainable_elements':ready['trainable'],'frozen_saved_projector_elements':projector_elements,
                          'future_storage_bytes_per_element':4,'future_model_tensor_bytes':quant[recipe],
                          'ready_metrics_path':str(run/'metrics.jsonl')}
    full_one=sum(full.values());quant_one=sum(quant.values());one_set=full_one+quant_one
    cache_root=root/'cache/vision';namespaces={}
    for namespace in cache_root.iterdir():
        if namespace.is_dir():namespaces[namespace.name]=sum(path.stat().st_size for path in namespace.rglob('*.pt'))
    current_siglip=max(namespaces.values(),default=0)
    # Current free already accounts for the cache on disk. Only remaining growth
    # is subtracted, while preserving this namespace for every later backbone.
    siglip_total=report['profiles']['siglip2_fixed_729']['conservative_cache_total_bytes']
    siglip_growth=max(0,siglip_total-current_siglip)
    free=shutil.disk_usage(root).free;reserve=report['disk']['reserve_bytes']
    allocation=report['unique_image_count']*4096+256*4096
    largest_full=max(full.values())
    peak_scenarios={'model_plus_2_fp32_Adam_moments_staging':largest_full*3,
                    'model_plus_2_8bit_Adam_moments_staging_lower_bound':math.ceil(largest_full*1.5)}
    scenarios=[]
    for qname in ('qwen_native_bounded_2048','qwen_native_bounded_1024','qwen_native_bounded_729'):
        qwen_total=report['profiles'][qname]['conservative_cache_total_bytes']
        gemma_total=report['profiles']['gemma4_native_280']['conservative_cache_total_bytes']
        extra_caches=siglip_growth+qwen_total+gemma_total+3*allocation
        for donor_sets,label in ((1,'queued_all11_SigLIP_finals_plus_three_persistent_caches'),
                                 (3,'all11_SmallThinker_finals_for_each_of_three_donors_plus_three_caches')):
            for copies,retention in ((1,'best_last_share_one_object_lower_bound'),(2,'best_last_distinct_retained_objects')):
                for peak_name,peak in peak_scenarios.items():
                    projector_delta=0 if donor_sets==1 else (projector_by_donor['qwen']+projector_by_donor['gemma']-2*projector_elements)*4*11
                    weights=(one_set*donor_sets+projector_delta)*copies
                    required=reserve+extra_caches+weights+peak
                    missing=max(0,required-free)
                    scenarios.append({'qwen_profile':qname,'scope':label,'retention':retention,'active_peak_assumption':peak_name,
                        'current_free_bytes':free,'reserve_bytes':reserve,'remaining_siglip_cache_growth_bytes':siglip_growth,
                        'new_qwen_cache_bytes':qwen_total,'new_gemma_cache_bytes':gemma_total,
                        'cache_filesystem_allocation_allowance_bytes':3*allocation,
                        'pending_completed_model_tensor_bytes':weights,'active_checkpoint_staging_peak_bytes':peak,
                        'required_free_bytes':required,'remaining_free_above_reserve_after_scenario_bytes':free-required,
                        'additional_external_capacity_needed_bytes':missing,'additional_external_capacity_needed_gib':missing/2**30,
                        'fits_current_local_disk':not bool(missing)})
    report['campaign_storage_forecast']={'evidence':evidence,'seven_full_single_version_model_tensor_bytes':full_one,
        'four_quant_single_version_model_tensor_bytes':quant_one,'all11_single_version_model_tensor_bytes':one_set,
        'mlp_projection_contract':{'language_hidden_size':hidden,'architecture':'Linear(D,H),GELU,Linear(H,H)','parameter_elements_by_donor':projector_by_donor},
        'largest_existing_siglip_namespace_bytes':current_siglip,'all_existing_vision_cache_namespaces':namespaces,
        'scenarios':scenarios,'any_declared_scenario_fits_current_local_disk':any(s['fits_current_local_disk'] for s in scenarios),
        'current_free_bytes_at_forecast':free,
        'limitations':['Final tensors are a lower bound: excludes retained optimizers, serialization metadata, source archives and future backbone downloads.',
                      'All11 model-size evidence comes from actual SmallThinker technical tensor inventories, projected to current FP32 storage. Other backbones are unsized.',
                      'A single shared BEST/LAST is an optimistic bound, not guaranteed after training. Two distinct retained versions must be budgeted.',
                      'Active staging includes a new full model and moments while existing retained objects remain. Allocator-aware optimizer serialization can differ.',
                      'No scientific checkpoints or existing SigLIP cache are deleted or converted by this forecast.']}
    report['recommendation']={'fixed_profile_for_all_11_qwen_experiments_after_capacity_expansion':'qwen_native_bounded_729',
        'feasible_for_full_queued_retention_on_current_disk':False,
        'basis':'729 is the lowest declared native Qwen cap, but even it requires external capacity when persistent SigLIP/Qwen/Gemma caches, queued final weights, active checkpoint peak and120GiB reserve are retained.',
        'external_capacity_required':not report['campaign_storage_forecast']['any_declared_scenario_fits_current_local_disk'],
        'no_profile_change_applied':True,'no_deletion_assumed':True,'profile_switch_within_11_experiments_allowed':False,
        'quality_or_GPU_throughput_preference_proven':False,
        'native_scope_note':'729 is an explicit optional bounded native-resize profile, not an unchanged donor default or an equal-compute SigLIP comparison.'}
    return report


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data-dir',default='datasets/research/pilot-en-scienceqa-context-v2')
    p.add_argument('--eval-manifest',default='datasets/research/eval-heldout-v3/evaluation.jsonl')
    p.add_argument('--output',default='research_results/vision_pilot_cache_footprint.json')
    p.add_argument('--reserve-gb',type=float,default=120)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise RuntimeError('Hide GPUs explicitly for this CPU-only audit')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from PIL import Image
    import transformers
    from transformers import AutoProcessor,AutoConfig
    from scripts.run_research_stage import resolve_model
    from scripts.filter_research_context import make_native_tokenizer
    from src.model.research_vision import apply_preprocessing_profile,FEATURE_POINTS
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size
    from src.utils.research_storage import _atomic_json
    torch.set_num_threads(8)
    start=time.monotonic();data=Path(args.data_dir);audit=json.loads((data/'audit.json').read_text())
    manifests={f'{stage}_{part}.jsonl':data/f'{stage}_{part}.jsonl' for stage in ('alignment','sft','text') for part in ('train','validation')}
    manifests['official_evaluation']=Path(args.eval_manifest)
    records={name:jsonl(path) for name,path in manifests.items()}
    images={};memberships=defaultdict(set)
    for name,rows in records.items():
        for row in rows:
            if len(row.get('image_hashes',[]))!=len(row.get('image_paths',[])):raise ValueError('Image/hash ordering count mismatch')
            for image_hash,path in zip(row.get('image_hashes',[]),row.get('image_paths',[])):
                memberships[image_hash].add(name)
                images.setdefault(image_hash,{'path':path})
    orientations=Counter();sizes=[]
    for index,(image_hash,item) in enumerate(sorted(images.items())):
        with Image.open(item['path']) as image:
            w,h=image.size;orientation=image.getexif().get(274,1);orientations[str(orientation)]+=1
            if orientation in (5,6,7,8):w,h=h,w
        if w<=0 or h<=0:raise ValueError('Nonpositive image header size')
        item.update(height=h,width=w);sizes.append((h,w))
        if index%5000==0:emit('image_header_scan',completed=index,total=len(images))
    keys=sorted(images);sizes=[(images[k]['height'],images[k]['width']) for k in keys]
    tokenize,binding=make_native_tokenizer()
    prompt_rows={}
    for name,rows in records.items():
        measured=[]
        for row in rows:
            # Evaluation never supplies a gold target to this tokenizer audit.
            public={'record_id':row['record_id'],'question':row['question'],'image_paths':row.get('image_paths',[]),'answer':'x'} if name=='official_evaluation' else row
            pre,post,answer=tokenize(public)
            measured.append({'record_id':row['record_id'],'prompt_text_tokens':len(pre)+len(post),
                'answer_tokens':0 if name=='official_evaluation' else len(answer),'image_hashes':row.get('image_hashes',[])})
        prompt_rows[name]=measured
    report={'schema_version':1,'status':'running','scope':'Full declared pilot/eval image-header census + installed native processor resize/token math; no donor forward',
        'data_manifest_id':audit['manifest_id'],'source_manifest_sha256':{name:sha(path) for name,path in manifests.items()},
        'image_identity_policy':'existing immutable canonical EXIF-RGB hashes; headers re-read, pixel hashes not recomputed',
        'unique_image_count':len(images),'source_record_counts':{k:len(v) for k,v in records.items()},
        'unique_images_by_manifest':{name:sum(name in memberships[k] for k in keys) for name in manifests},
        'header_orientation_counts':dict(orientations),'max_height':max(h for h,w in sizes),'max_width':max(w for h,w in sizes),
        'gpu_used':False,'weights_loaded':False,'encoder_forwards':0,'downloads':0,'transformers_version':transformers.__version__,
        'tokenizer_binding':binding,'profiles':{},'native_processor_checks':[],
        'count_source_sha256':{'qwen_smart_resize':hashlib.sha256(inspect.getsource(smart_resize).encode()).hexdigest(),
                              'gemma_resize':hashlib.sha256(inspect.getsource(get_aspect_ratio_preserving_size).encode()).hexdigest()}}
    _atomic_json(Path(args.output),report)
    processors={}
    for repo in ('Qwen/Qwen2.5-VL-7B-Instruct','google/gemma-4-E4B-it'):
        path,ref=resolve_model(repo);processor=AutoProcessor.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        config=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        processors[repo]=(processor,config,ref)
    qwen,qcfg,qref=processors['Qwen/Qwen2.5-VL-7B-Instruct']
    gemma,gcfg,gref=processors['google/gemma-4-E4B-it']
    if qcfg.vision_config.out_hidden_size!=3584 or gcfg.text_config.hidden_size!=2560:raise ValueError('Unexpected native donor feature width')
    profiles=[]
    qwen_native=qwen
    for budget in (2048,1024,729):
        qwen=copy.deepcopy(qwen_native)
        name=f'qwen_native_bounded_{budget}'
        contract=apply_preprocessing_profile(SimpleNamespace(backend='qwen2_5_vl',processor=qwen),name)
        counts=qwen._get_num_multimodal_tokens(image_sizes=sizes).num_image_tokens
        if any(n<=0 or n>budget for n in counts):raise ValueError('Native Qwen image count outside declared budget')
        grids=[]
        for h,w in sizes:
            rh,rw=smart_resize(h,w,factor=28,min_pixels=contract['min_pixels'],max_pixels=contract['max_pixels'])
            grids.append((rh//14,rw//14))
        if [a*b//4 for a,b in grids]!=counts:raise ValueError('Qwen native processor count differs from smart_resize grids')
        profiles.append((name,3584,counts,grids,'qwen2_5_vl',qref,contract))
    gc=gemma.image_processor
    if not(gc.do_resize and gc.patch_size==16 and gc.pooling_kernel_size==3 and gc.max_soft_tokens==280):
        raise ValueError('Gemma native size contract changed; do not infer unsupported counts')
    gcounts=gemma._get_num_multimodal_tokens(image_sizes=sizes).num_image_tokens
    ggrids=[]
    for h,w in sizes:
        rh,rw=get_aspect_ratio_preserving_size(h,w,patch_size=16,max_patches=280*9,pooling_kernel_size=3)
        ggrids.append((rh//16,rw//16))
    if [a*b//9 for a,b in ggrids]!=gcounts or max(gcounts)>280:raise ValueError('Gemma native count/patch geometry mismatch')
    profiles.extend([('gemma4_native_280',2560,gcounts,ggrids,'gemma4',gref,{'max_soft_tokens':280,'patch_size':16,'pooling_kernel_size':3,'native_processor':gc.to_dict()}),
                     ('siglip2_fixed_729',1152,[729]*len(keys),[(27,27)]*len(keys),'siglip',{'repo':'google/siglip2-so400m-patch14-384','revision':'e8e487298228002f3d8a82e0cd5c8ea9c567f57f'},{'fixed_token_count':729})])
    # Check native count formulas against bounded actual preprocessing on the same
    # real image files at min/max count and landscape/portrait extremes (no model).
    for processor,counts,backend in [(qwen,profiles[2][2],'qwen2_5_vl'),(gemma,gcounts,'gemma4')]:
        indices={counts.index(min(counts)),counts.index(max(counts)),max(range(len(sizes)),key=lambda i:sizes[i][0]/sizes[i][1]),max(range(len(sizes)),key=lambda i:sizes[i][1]/sizes[i][0])}
        for index in sorted(indices):
            from PIL import ImageOps
            with Image.open(images[keys[index]]['path']) as image:rgb=ImageOps.exif_transpose(image).convert('RGB')
            batch=processor.image_processor(images=[rgb],return_tensors='pt');rgb.close()
            actual=int(batch['image_grid_thw'][0].prod())//4 if backend=='qwen2_5_vl' else int(batch['num_soft_tokens_per_image'][0])
            if actual!=counts[index]:raise ValueError('Real-image native preprocessing disagrees with exact header resize count')
            report['native_processor_checks'].append({'backend':backend,'profile':'qwen_native_bounded_729' if backend=='qwen2_5_vl' else 'gemma4_native_280',
                'image_hash':keys[index],'original_height_width':list(sizes[index]),'expected_and_actual_tokens':actual})
            del batch
    for name,dimension,counts,grids,backend,ref,contract in profiles:
        by_hash=dict(zip(keys,counts));raw_bytes=sum(counts)*dimension*2
        examples={counts.index(min(counts)),counts.index(max(counts)),len(keys)//2};overheads=[]
        for index in sorted(examples):
            n=counts[index];gh,gw=grids[index];h,w=sizes[index]
            metadata={'original_size':[w,h],'valid_token_count':n,'backend':backend,'feature_point':FEATURE_POINTS[backend]}
            if backend=='qwen2_5_vl':metadata['image_grid_thw']=[1,gh,gw]
            elif backend=='gemma4':
                positions=[[x,y] for y in range(gh) for x in range(gw)]
                positions += [[-1,-1]]*(280*9-len(positions))
                metadata.update(image_token_count=n,image_position_ids=positions)
            overheads.append({'tokens':n,'overhead_bytes':serialize_overhead(torch,n,dimension,metadata)})
        bound=64*1024 if backend=='gemma4' else 8192
        if max(x['overhead_bytes'] for x in overheads)>bound:raise ValueError('Cache metadata overhead exceeds declared conservative allowance')
        contexts={}
        for source,rows in prompt_rows.items():
            worst=None;truncations=0;overflows=0
            for row in rows:
                visual=sum(by_hash[h] for h in row['image_hashes']);expanded=visual+row['prompt_text_tokens'];sequence=expanded+row['answer_tokens']
                if worst is None or expanded>worst['expanded_prompt_tokens']:
                    worst={'record_id':row['record_id'],'visual_tokens':visual,'prompt_text_tokens':row['prompt_text_tokens'],
                           'expanded_prompt_tokens':expanded,'remaining_context_tokens':4096-expanded}
                overflows+=int(expanded>=4096);truncations+=int(sequence>4096)
            contexts[source]={'max_expanded_prompt':worst,'expanded_prompt_overflows':overflows,'target_truncation_records_at_4096':truncations,
                'max_full_training_sequence_tokens':max((sum(by_hash[h] for h in r['image_hashes'])+r['prompt_text_tokens']+r['answer_tokens'] for r in rows),default=0),
                'evaluation_generation_reserve_checked_tokens':64 if source=='official_evaluation' else None}
            if source=='official_evaluation':contexts[source]['generation_context_overflows_at_64']=sum(sum(by_hash[h] for h in r['image_hashes'])+r['prompt_text_tokens']+64>4096 for r in rows)
        entry={'repo_revision':ref,'feature_dimension':dimension,'storage_dtype':'bfloat16','token_distribution':distribution(counts),
               'exact_feature_payload_bytes':raw_bytes,'exact_feature_payload_gib':raw_bytes/2**30,
               'cache_entry_overhead_samples':overheads,'conservative_overhead_bytes_per_entry':bound,
               'conservative_cache_total_bytes':raw_bytes+bound*len(keys),'conservative_cache_total_gib':(raw_bytes+bound*len(keys))/2**30,
               'processor_contract':contract,'contexts':contexts,
               'context_counts_are_processor_inference_only':True,'all_dataset_features_actually_encoded':False}
        report['profiles'][name]=entry
        emit('profile_measured',profile=name,unique_images=len(keys),payload_gib=entry['exact_feature_payload_gib'],conservative_total_gib=entry['conservative_cache_total_gib'],max_tokens=max(counts))
    free=shutil.disk_usage(ROOT).free;reserve=int(args.reserve_gb*2**30)
    report['disk']={'free_bytes_at_census':free,'reserve_bytes':reserve,'free_above_reserve_bytes':free-reserve,
                    'existing_siglip_cache_retained':True,'future_checkpoint_sizes_unknown':True,
                    'required_checkpoint_headroom_assumption_bytes':None}
    for name,entry in report['profiles'].items():
        remaining=free-reserve-entry['conservative_cache_total_bytes']
        entry['disk_above_reserve_after_additional_full_cache_bytes']=remaining
    report.update(status='passed',elapsed_seconds=time.monotonic()-start,
        limits=['Resize/token-count inference over every declared image is not encoder numerical validation. Only prior paired two-image probes executed all real donor weights.',
                'Header scan trusts prior immutable canonical RGB identities; resized/reencoded semantic near-duplicate identity is not asserted.',
                'Byte totals cover BF16 feature payload and conservative per-entry pickle/ZIP metadata allowance; filesystem allocation overhead is reported separately below.',
                'All profiles require separate cache fingerprints. No active runner/profile/readiness source was changed.',
                'Future full-training optimizer/checkpoint and activation memory are not measured by this CPU census.'])
    report['filesystem_allocation_allowance_bytes']=len(keys)*4096+256*4096
    report=storage_forecast(report)
    report['source_script_sha256']=sha(__file__)
    _atomic_json(Path(args.output),report)
    emit('footprint_census_passed',report=args.output,unique_images=len(keys),free_gib=free/2**30)

if __name__=='__main__':main()
