"""Fresh low-memory sequential RGB replay; publish captions before SFT proof."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import importlib.util,json,os,sys,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_budget18_frozen_selection_producer',ROOT/'scripts/prepare_research_budget18.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
a=b.a;OUT=b.OUT;E=b.E
a.RSS_LIMIT=4*2**30
def hashes(records):return {h for r in records for h in r['image_hashes']}
def image_stat(path):
 s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def worker_limits():resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30))
def verify_RGB_worker(item):
 h,path,source_SHA=item;before=image_stat(path)
 if a.sha(__file__)!=source_SHA or b.cache.image_content_hash(path)!=h or image_stat(path)!=before:raise ValueError('Immutable current RGB/source verification failed')
 return {'image_hash':h,'path':path,'stat':before,'verifier_source_sha256':source_SHA,'actual_current_canonical_RGB_verified':True,'worker_resources':a.resources()}
def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise RuntimeError('CPU replay only')
 a.resources();names=('alignment_train','alignment_a1_train','alignment_validation','sft_multimodal_train','text_train','sft_train','sft_validation','heldout_evaluation','internal_selection_116')
 rows={n:list(a.rows(OUT/(n+'.jsonl'))) for n in names};expected=dict(alignment_train=4096,alignment_a1_train=256,alignment_validation=372,sft_multimodal_train=3000,text_train=529,sft_train=3529,sft_validation=226,heldout_evaluation=512,internal_selection_116=116)
 if {n:len(r) for n,r in rows.items()}!=expected:raise ValueError('Exact final18-scope counts required')
 parents={n:list(a.rows(b.P/(n+'.jsonl'))) for n in ('alignment_train','alignment_validation','sft_train','text_train','sft_validation','text_validation')};parent_audit=a.read(b.P/'audit.json')
 for n in parents:
  if a.sha(b.P/(n+'.jsonl'))!=parent_audit[n+'_sha256']:raise ValueError('Immutable parent source changed')
 for source,target in ((b.P/'alignment_validation.jsonl',OUT/'alignment_validation.jsonl'),(b.M/'sft_validation.jsonl',OUT/'sft_validation.jsonl'),(b.HELDOUT,OUT/'heldout_evaluation.jsonl')):
  if source.read_bytes()!=target.read_bytes():raise ValueError('Fixed validation bytes changed')
 native,Clock,english,code=a.native_functions();geo={r['image_hash']:r for r in a.rows(E/'all_parent_native_image_geometry.jsonl')};original=parents['alignment_train']+parents['sft_train']+parents['text_train'];source_by_id={r['record_id']:r for r in original};review_path=ROOT/'datasets/research/eval-full-overlap-v1/near_only_training_review_candidates.jsonl';review_ids={r['record_id'] for r in a.rows(review_path)};risk_images={h for r in original if r['record_id'] in review_ids for h in r['image_hashes']};official=ROOT/'datasets/research/eval-full-overlap-v1/full_heldout_image_index.jsonl';protected={r['image_hash'] for r in a.rows(official)}|hashes(rows['heldout_evaluation']);validation_images=hashes(rows['alignment_validation']+rows['sft_validation']);validation_ids={r['record_id'] for r in rows['alignment_validation']+rows['sft_validation']}
 bad_text=set();text_context={};selected_context={}
 for model in b.MODELS:
  alltext=list(a.rows(E/('all_text_context_'+model+'.jsonl')));bad_text.update(r['record_id'] for r in alltext if r['zero_supervised_position']);text_context[model]={'records':len(alltext),'zero_supervised_IDs':[r['record_id'] for r in alltext if r['zero_supervised_position']]}
  selected=list(a.rows(E/('selected_context_'+model+'.jsonl')))
  if len(selected)!=sum(expected[n] for n in ('alignment_train','alignment_a1_train','sft_train','sft_validation','alignment_validation')) or any(r['zero_supervised_position'] for r in selected):raise ValueError('All3model fixedmanifest context must be complete and pass')
  selected_context[model]=selected
 excluded=[];eligible=[]
 for r in original:
  reasons=[];hs=set(r['image_hashes'])
  if r['source']['split']!='train' or r['partition']!='train' or r['language']!='en':raise ValueError('Parent source TRAIN/EN provenance differs')
  if hs&protected:reasons.append('known_declared_official_heldout_exact_RGB')
  if hs&validation_images or r['record_id'] in validation_ids:reasons.append('fixed_internal_or_caption_validation_overlap')
  if hs&risk_images:reasons.append('conservative_existing_near_review_image_risk_exclusion_not_identity_claim')
  if any(not geo[h]['native_Qwen_geometry_valid'] for h in hs):reasons.append('native_Qwen_bounded729_geometry_rejected')
  if english({'question':r['question'],'answer':r['answer']}):reasons.append('existing_frozen_English_heuristic')
  if reasons:excluded.append({'record_id':r['record_id'],'reasons':reasons})
  else:eligible.append(r)
 eligible_ids={r['record_id'] for r in eligible};cap={}
 for r in sorted([r for r in parents['alignment_train'] if r['record_id'] in eligible_ids],key=lambda r:b.rank(r,'caption_duplicate_RGB_owner')):cap.setdefault(tuple(r['image_hashes']),r)
 A0=b.choose(list(cap.values()),4096,'A0');A1=b.choose([r for r in cap.values() if not hashes(A0).intersection(r['image_hashes'])],256,'A1')
 if rows['alignment_train']!=A0 or rows['alignment_a1_train']!=A1:raise ValueError('Caption seed42 source-only hash selection fails independent replay')
 mm=[]
 for group,n in b.GROUPS.items():mm+=b.choose([r for r in parents['sft_train'] if r['record_id'] in eligible_ids and r['task_group']==group],n,'SFT:'+group)
 text=b.choose([r for r in parents['text_train'] if r['record_id'] in eligible_ids and r['record_id'] not in bad_text],529,'SmolTalk')
 if rows['sft_multimodal_train']!=mm or rows['text_train']!=text or rows['sft_train']!=sorted(mm+text,key=lambda r:b.rank(r,'SFT_final_order')):raise ValueError('Unique MM/text group selection or final deterministic order differs')
 selection,selection_source=b.internal_selection(rows['sft_validation'])
 if rows['internal_selection_116']!=selection['records'] or a.read(OUT/'internal_selection_protocol.json')!=selection['protocol']:raise ValueError('Existing internal116 native selection changed')
 train=A0+A1+rows['sft_train']
 if hashes(A0)&hashes(A1) or hashes(train)&validation_images or hashes(train)&protected or hashes(train)&risk_images or len({r['record_id'] for r in rows['sft_train']})!=3529:raise ValueError('Train/val/official/risk/unique invariant failed')
 paths={h:path for r in train+rows['alignment_validation']+rows['sft_validation'] for h,path in zip(r['image_hashes'],r['image_paths'])};caption_hashes=hashes(A0+A1+rows['alignment_validation']);order=sorted(caption_hashes)+sorted(set(paths)-caption_hashes);verified=[];caption_receipt=None;started=time.time()
 journal=E/'current_RGB_journal_v3';journal.mkdir(exist_ok=True);source_SHA=a.sha(__file__);committed={};pending=[]
 for h in order:
  receipt=journal/(h+'.json')
  if receipt.exists():
   v=a.read(receipt)
   if v['image_hash']!=h or v['path']!=paths[h] or v['stat']!=image_stat(paths[h]) or v['verifier_source_sha256']!=source_SHA or v['actual_current_canonical_RGB_verified'] is not True:raise ValueError('Prior RGB journal binding differs')
   committed[h]=v
  else:pending.append((h,paths[h],source_SHA))
 pool=ProcessPoolExecutor(max_workers=4,max_tasks_per_child=16,initializer=worker_limits);fresh=iter(pool.map(verify_RGB_worker,pending))
 for i,h in enumerate(order):
  if h not in committed:
   proof=next(fresh)
   if proof['image_hash']!=h:raise ValueError('RGB worker order mismatch')
   a.save(journal/(h+'.json'),proof);committed[h]=proof
  path=paths[h]
  verified.append({'image_hash':h,'path':path,'actual_canonical_RGB_redecoded':True,'encoded_bytes':Path(path).stat().st_size,'native_Qwen_tokens':geo[h]['Qwen_bounded729_tokens'],'SigLIP_tokens':729,'feature_dimensions':[1152,3584]})
  a.resources()
  if i%250==0:print(a.canonical({'phase':'fresh_sequential_caption_then_SFT_RGB_replay','images':i+1,'total':len(order),'caption_total':len(caption_hashes),'resources':a.resources()}),flush=True)
  if i+1==len(caption_hashes):
   a.save_rows(E/'caption_current_RGB_verification.jsonl',verified)
   caption_ids={r['record_id'] for r in A0+A1+rows['alignment_validation']};caption_context={}
   for model,stats in selected_context.items():
    selected=[r for r in stats if r['record_id'] in caption_ids];caption_context[model]={'records':len(selected),'zero_supervised_positions':sum(r['zero_supervised_position'] for r in selected),'full_target_truncations':sum(r['full_target_truncated_by_existing_residual4096_policy'] for r in selected),'max_expanded_prompt_tokens':max(r['expanded_prompt_tokens_upper_bound'] for r in selected)}
   if caption_context['smallthinker4']['full_target_truncations']:raise ValueError('Requested Small full-target caption audit failed')
   caption_receipt={'schema_version':1,'status':'passed','profile':'budget18-en-v1','scope':'caption4096A0_256A1_372fixedvalidation_currentRGB_source_exclusions_and_native3model_context','A0_records':4096,'A1_records':256,'validation_records':372,'A0_A1_RGB_overlap':0,'caption_train_validation_RGB_overlap':0,'known_official_exact_or_existing_near_review_caption_RGB_overlap':0,'all_current_caption_RGB_assets_redecoded_verified':len(caption_hashes),'Small_complete_native_targets_fit4096':True,'all3_native_context':caption_context,'context_vision_profiles':{'SigLIP2':{'max_tokens':729,'feature_dim':1152},'qwen_native_bounded_729':{'max_tokens':729,'feature_dim':3584}},'files':{n:a.binding(OUT/n) for n in ('alignment_train.jsonl','alignment_a1_train.jsonl','alignment_validation.jsonl')},'RGB_proof':a.binding(E/'caption_current_RGB_verification.jsonl'),'producer_source':a.binding(ROOT/'scripts/prepare_research_budget18.py'),'validator_source':a.binding(__file__),'no_downloads_modelweights_GPU_or_parent_changes':True,'created_at':datetime.now(timezone.utc).isoformat()};a.save(OUT/'caption_readiness.json',caption_receipt);print(a.canonical({'phase':'CAPTION_READY_ROOT_MAY_START_SMALL_A0','receipt':a.binding(OUT/'caption_readiness.json')}),flush=True)
 pool.shutdown();a.save_rows(E/'selected_current_RGB_asset_verification.jsonl',verified);a.save_rows(E/'parent_exclusion_decisions.jsonl',excluded)
 context={m:{'records':len(v),'prompt_overflows':sum(x['zero_supervised_position'] for x in v),'target_truncation_records':sum(x['full_target_truncated_by_existing_residual4096_policy'] for x in v),'max_expanded_prompt_tokens_upper_bound':max(x['expanded_prompt_tokens_upper_bound'] for x in v)} for m,v in selected_context.items()}
 protocol={'schema_version':1,'profile':'budget18-en-v1','seed':42,'models':b.MODELS,'recipes':['1b','4f','3c'],'encoders':{'SigLIP2':{'feature_dim':1152,'tokens':729},'qwen_native_bounded_729':{'feature_dim':3584,'token_upper_bound':729}},'A0':4096,'A1':256,'SFT_MM':3000,'SFT_group_quotas':b.GROUPS,'SFT_text':529,'SFT_total':3529,'text_record_fraction':529/3529,'train_repeats':0,'selection_rule':'SHA256 seed42 source-identity ranking; exact original rows; caption RGB duplicate-owner and A0/A1 disjoint; group-stratified unique MM; no target-answer or score-based ranking','context_policy':{'max_length':4096,'prompt_schema_version':2,'native_targets_truncate_only_at_existing_residual_context':True,'arbitrary_extra_target_cap':None,'all3_native_context':context,'tokenizer_assets':b.assets(),'native_AST_source':code},'source':{'parent_audit':a.binding(b.P/'audit.json'),'parent_mix_audit':a.binding(b.M/'mix_audit.json'),'parent_manifests':{n:a.binding(b.P/(n+'.jsonl')) for n in parents},'known_full_heldout_RGB_index':a.binding(official),'known_existing_near_review_IDs':a.binding(review_path),'near_review_flags_used_for_conservative_exclusion_not_identity_or_manual_clearance':True},'source_sha256':{str(p.relative_to(ROOT)):a.sha(p) for p in [ROOT/'scripts/prepare_research_budget18.py',Path(__file__).resolve(),ROOT/'src/dataset/vision_cache.py',ROOT/'src/eval/research_selection.py']},'context_phase_separate_from_purePIL_current_RGB_phase_due2GiB_memory_guard':True,'all_current_selected_RGB_redecoded':len(verified),'all_primary396_or_private_unselected_fuzzy_text_clearance_claimed':False,'no_new_downloads_parent_or_training_source_edits':True}
 a.save(OUT/'protocol.json',protocol);audit={'schema_version':1,'status':'passed','profile':'budget18-en-v1','counts':expected,'actual_MM_group_counts':dict(Counter(r['task_group'] for r in mm)),'all3models_both_encoders_context':context,'all_common_text_prompt_overflow_exclusions':sorted(bad_text),'known_official_exact_RGB_overlap':0,'known_near_review_image_overlap':0,'train_fixed_validation_RGB_overlap':0,'A0_A1_RGB_overlap':0,'unique_SFT_record_count':3529,'all_source_fields_and_original_text_answers_unchanged':True,'fixed372_226_116_512_preserved':True,'selected_current_RGB_verified':len(verified),'all_input_assets_already_local_no_downloads':True,'language_model_or_vision_forwards':0,'Torch_imported':'torch' in sys.modules,'full_PRIMARY_or_396_completion_claim':False,'resources':a.resources(),'wall_seconds':time.time()-started,'caption_readiness':a.binding(OUT/'caption_readiness.json')};a.save(OUT/'audit.json',audit)
 release={'schema_version':1,'status':'passed','profile':'budget18-en-v1','seed':42,'counts':expected,'files':{p.name:a.binding(p) for p in OUT.iterdir() if p.is_file()},'evidence_files':{p.name:a.binding(p) for p in E.iterdir() if p.is_file()},'protocol':a.binding(OUT/'protocol.json'),'audit':a.binding(OUT/'audit.json'),'no_old_manifest_or_budget9_overwrite':True};a.save(OUT/'release.json',release);print(a.canonical({'phase':'BUDGET18_DATA_FULL_RELEASE_COMPLETE','release':a.binding(OUT/'release.json')}),flush=True)

if __name__=='__main__':main()
