"""Finish SFT RGB checks and freeze the already-selected common budget18 data."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import importlib.util,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_budget18_existing_selection',ROOT/'scripts/prepare_research_budget18.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
a=b.a;a.RSS_LIMIT=16*2**30;OUT=b.OUT;E=b.E
def hashes(rows):return {h for r in rows for h in r['image_hashes']}
def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
 names={'alignment_train':4096,'alignment_a1_train':256,'alignment_validation':372,'sft_multimodal_train':3000,'text_train':529,'sft_train':3529,'sft_validation':226,'heldout_evaluation':512,'internal_selection_116':116}
 rows={n:list(a.rows(OUT/(n+'.jsonl'))) for n in names}
 if {n:len(v) for n,v in rows.items()}!=names:raise ValueError('Exactfinal18budgetscope required')
 caption=a.read(OUT/'caption_readiness.json')
 if caption['status']!='passed':raise ValueError('Captionproof required')
 for ref in caption['files'].values():
  if a.sha(ref['path'])!=ref['sha256']:raise ValueError('Releasedcaptionmanifest changed')
 parents={n:list(a.rows(b.P/(n+'.jsonl'))) for n in ('alignment_train','alignment_validation','sft_train','sft_validation','text_train','text_validation')};source={r['record_id']:r for values in parents.values() for r in values};parent_audit=a.read(b.P/'audit.json')
 for n in parents:
  if a.sha(b.P/(n+'.jsonl'))!=parent_audit[n+'_sha256']:raise ValueError('Frozenparent changed')
 for n in ('alignment_train','alignment_a1_train','sft_multimodal_train','text_train'):
  if any(source.get(r['record_id'])!=r for r in rows[n]):raise ValueError('Budgetrow changedoriginaltext/answers/source IDs')
 if Counter(r['task_group'] for r in rows['sft_multimodal_train'])!=Counter(b.GROUPS):raise ValueError('Exact1050/900/600/450quota required')
 if len({r['record_id'] for r in rows['sft_train']})!=3529 or {r['record_id'] for r in rows['sft_train']}!={r['record_id'] for r in rows['sft_multimodal_train']+rows['text_train']}:raise ValueError('No trainingrepeats; exact529text/3000MM required')
 for original,target in ((b.P/'alignment_validation.jsonl',OUT/'alignment_validation.jsonl'),(b.M/'sft_validation.jsonl',OUT/'sft_validation.jsonl'),(b.HELDOUT,OUT/'heldout_evaluation.jsonl')):
  if original.read_bytes()!=target.read_bytes():raise ValueError('Existingvalidationbytes mustremainunchanged')
 selection,selection_source=b.internal_selection(rows['sft_validation'])
 if selection['records']!=rows['internal_selection_116'] or selection['protocol']!=a.read(OUT/'internal_selection_protocol.json'):raise ValueError('Samefrozeninternal116 required')
 alltrain=rows['alignment_train']+rows['alignment_a1_train']+rows['sft_train'];validation=rows['alignment_validation']+rows['sft_validation'];train_h=hashes(alltrain);val_h=hashes(validation)
 protected_index=ROOT/'datasets/research/eval-full-overlap-v1/full_heldout_image_index.jsonl';protected={r['image_hash'] for r in a.rows(protected_index)}|hashes(rows['heldout_evaluation']);review_index=ROOT/'datasets/research/eval-full-overlap-v1/near_only_training_review_candidates.jsonl';review_ids={r['record_id'] for r in a.rows(review_index)};review_images={h for r in source.values() if r['record_id'] in review_ids for h in r['image_hashes']}
 if train_h&val_h or train_h&protected or train_h&review_images or hashes(rows['alignment_train'])&hashes(rows['alignment_a1_train']):raise ValueError('KnownRGB leakage/disjoint guard failed')
 geo={r['image_hash']:r for r in a.rows(E/'all_parent_native_image_geometry.jsonl')};needed=hashes(rows['sft_train']+rows['sft_validation'])-hashes(rows['alignment_train']+rows['alignment_a1_train']+rows['alignment_validation']);paths={h:path for r in rows['sft_train']+rows['sft_validation'] for h,path in zip(r['image_hashes'],r['image_paths'])}
 def verify(h):
  if b.cache.image_content_hash(paths[h])!=h:raise ValueError('CurrentSFT RGBmismatch')
  return {'image_hash':h,'image_path':paths[h],'actual_current_canonical_RGB_verified':True,'native_Qwen_bounded729_tokens':geo[h]['Qwen_bounded729_tokens'],'SigLIP_tokens':729}
 with ThreadPoolExecutor(max_workers=8) as pool:verified=list(pool.map(verify,sorted(needed)))
 a.resources();a.save_rows(E/'SFT_remaining_current_RGB_verification.jsonl',verified)
 context={}
 for model in b.MODELS:
  records=list(a.rows(E/('selected_context_'+model+'.jsonl')));expected_rows=len(alltrain)+len(validation)
  if len(records)!=expected_rows or any(r['zero_supervised_position'] or r['full_target_truncated_by_existing_residual4096_policy'] for r in records):raise ValueError('All3native complete-target contexts mustfit4096')
  context[model]={'records':len(records),'expanded_prompt_overflows':0,'full_target_truncations':0,'max_full_prompt_visual_target_tokens':max(r['expanded_prompt_tokens_upper_bound']+r['complete_target_tokens'] for r in records),'source':a.binding(E/('selected_context_'+model+'.jsonl'))}
 if any(not geo[h]['native_Qwen_geometry_valid'] or not 0<geo[h]['Qwen_bounded729_tokens']<=729 for h in train_h|val_h):raise ValueError('Bothencoder729context and nativegeometry required')
 protocol={'schema_version':1,'profile':'budget18-en-v1','seed':42,'models':['SmallThinker-4B-A0.6B','LFM2.5-8B-A1B','GigaChat3.1-10B-A1.8B'],'recipes':['1b','4f','3c'],'encoders':{'SigLIP2':{'feature_dim':1152,'visual_tokens':729},'Qwen2.5VL7Bvision':{'profile':'qwen_native_bounded_729','feature_dim':3584,'visual_token_upper_bound':729}},'selection':'source-identity SHA256seed42; captionRGBdedup then4096A0/256remainingA1; unique4group MM/uniqueSmol; no target/score ranking','counts':names,'MM_group_quotas':b.GROUPS,'text_fraction':529/3529,'training_repeats':0,'context':{'max_length':4096,'prompt_schema_version':2,'arbitrary_extra_target_cap':None,'all_selected_complete_targets_fit_all3models':True,'all3_native_proof':context,'tokenizer_assets':b.assets()},'parent_audit':a.binding(b.P/'audit.json'),'parent_mix_audit':a.binding(b.M/'mix_audit.json'),'parent_manifests':{n:a.binding(b.P/(n+'.jsonl')) for n in parents},'known_pinned_protected_RGB':a.binding(protected_index),'known_near_review_IDs':a.binding(review_index),'near_review_used_as_conservative_exclusion_not_manual_identity_clearance':True,'fixed_validation_bytes_and_internal116_preserved':True,'all_selected_current_RGB_evidence':{'caption':a.binding(OUT/'caption_readiness.json'),'remaining_SFT':a.binding(E/'SFT_remaining_current_RGB_verification.jsonl')},'source_sha256':{str(p.relative_to(ROOT)):a.sha(p) for p in (ROOT/'scripts/prepare_research_budget18.py',Path(__file__).resolve(),ROOT/'scripts/finalize_research_budget18_caption_fast.py',ROOT/'src/dataset/vision_cache.py',ROOT/'src/eval/research_selection.py')},'no_new_downloads_models_or_GPU':True,'prior500K_396_or_full_official_private_split_decontamination_claim':False}
 a.save(OUT/'protocol.json',protocol);audit={'schema_version':1,'status':'passed','profile':'budget18-en-v1','counts':{'A0':4096,'A1':256,'MM':3000,'text':529,'SFT_total':3529,'caption_validation':372,'SFT_validation':226,'internal116':116,'heldout512':512},'group_counts':dict(Counter(r['task_group'] for r in rows['sft_multimodal_train'])),'training_repeated_record_ids':0,'A0_A1_RGB_overlap':0,'train_validation_RGB_overlap':0,'train_known_official_exact_RGB_overlap':0,'train_existing_near_review_image_overlap':0,'all3_models_both_encoder_complete_native_context':context,'all_current_RGB_evidence':protocol['all_selected_current_RGB_evidence'],'unique_caption_current_RGB_checks':4724,'additional_SFT_current_RGB_checks':len(verified),'source_original_text_answer_source_IDs_preserved':True,'fixed372_226_116_512_unchanged':True,'new_download_bytes':0,'Torch_imported':'torch' in sys.modules,'language_model_or_vision_forward_calls':0,'resources':a.resources(),'protocol':a.binding(OUT/'protocol.json'),'created_at':datetime.now(timezone.utc).isoformat()};a.save(OUT/'audit.json',audit)
 release={'schema_version':1,'status':'passed','profile':'budget18-en-v1','seed':42,'counts':audit['counts'],'protocol':a.binding(OUT/'protocol.json'),'audit':a.binding(OUT/'audit.json'),'files':{p.name:a.binding(p) for p in OUT.iterdir() if p.is_file()},'evidence_files':{p.name:a.binding(p) for p in E.iterdir() if p.is_file()},'old_manifests_or_budget9_outputs_overwritten':False};a.save(OUT/'release.json',release);print(a.canonical({'status':'passed','release':a.binding(OUT/'release.json'),'audit':audit}),flush=True)
if __name__=='__main__':main()
