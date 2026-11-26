"""Frozen seed42 budget18 manifests from existing pilot assets only."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import ast,copy,gc,hashlib,importlib.util,inspect,json,math,os,random,re,sys,unicodedata
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/research/budget18-en-v1'
E=ROOT/'research_results/budget18_data_v1'
P=ROOT/'datasets/research/pilot-en-scienceqa-context-v2'
M=ROOT/'datasets/research/pilot-en-scienceqa-context-v2-mixture'
HELDOUT=ROOT/'datasets/research/eval-heldout-v3/evaluation.jsonl'
GROUPS={'general_vqa':1050,'ocr_documents':900,'charts_diagrams':600,'knowledge_reasoning':450}
MODELS=('smallthinker4','lfm25','giga31')
spec=importlib.util.spec_from_file_location('_budget_native_context_AST',ROOT/'research_results/audit_direct_Doc_context_v14.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.OUT=E
spec=importlib.util.spec_from_file_location('_budget_immutable_RGB_hash',ROOT/'src/dataset/vision_cache.py');cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)
def rank(record,salt):
 source=record['source'];identity=[source['repo'],source['revision'],source['config'],source['row_id'],record.get('turn_index',0),record.get('image_hashes',[])]
 return (a.value_sha([42,'budget18-en-v1',salt,identity]),record['record_id'])
def choose(pool,n,salt):
 if len({r['record_id'] for r in pool})!=len(pool) or len(pool)<n:raise ValueError('Unique no-repeat source pool insufficient')
 return sorted(pool,key=lambda r:rank(r,salt))[:n]
def assets():
 index=a.read(ROOT/'datasets/research/text-eval-inputs/assets.json');result={}
 for model in MODELS:
  selected=[x for x in index['files'] if x['kind']=='tokenizer' and x['id']==model]
  if not {'config.json','tokenizer.json','tokenizer_config.json'}<={x['filename'] for x in selected}:raise ValueError('All3pinnedtokenizers required')
  for ref in selected:
   if a.sha(ref['path'])!=ref['sha256']:raise ValueError('Tokenizer bytes changed')
  result[model]=selected
 return result
def tokenizer(model,refs,Clock):
 from transformers import AutoTokenizer,PreTrainedConfig
 config=a.read(next(r['path'] for r in refs if r['filename']=='config.json'));path=Path(next(r['path'] for r in refs if r['filename']=='config.json')).parent
 kw={}
 if model=='giga31':generic=PreTrainedConfig();generic.model_type=config['model_type'];kw['config']=generic
 token=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False,**kw)
 return SimpleNamespace(tokenizer=Clock(token),base=SimpleNamespace(config=SimpleNamespace(model_type=config['model_type'])),args=SimpleNamespace(prompt_schema_version=2))
def measure(dummy,record,native):
 pre,post,target=native['tokenize'](dummy,record);prompt=len(pre)+len(post)+729*len(record['image_paths']);available=4096-prompt
 return {'expanded_prompt_tokens_upper_bound':prompt,'complete_target_tokens':len(target),'available_target_tokens':available,'zero_supervised_position':available<1 or not target,'full_target_truncated_by_existing_residual4096_policy':len(target)>max(0,available),'prompt_plus64_fits':prompt+64<=4096}
def image_shapes(records):
 from PIL import Image,ImageOps
 from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
 seen={};bad=set();observations=[]
 for record in records:
  for path,h in zip(record['image_paths'],record['image_hashes']):
   if h in seen:continue
   path=Path(path)
   if not path.is_file() or path.is_symlink() or h not in path.name:raise ValueError('Known local immutable pilot RGB reference required')
   with Image.open(path) as im:
    width,height=im.size
    if im.getexif().get(274,1) in (5,6,7,8):width,height=height,width
   try:
    rh,rw=smart_resize(height,width,factor=28,min_pixels=4*28*28,max_pixels=729*28*28);tokens=rh*rw//(28*28)
    if not 0<tokens<=729:raise ValueError('Native Qwen bounded729 guarantee failed')
    observation={'image_hash':h,'image_path':str(path),'EXIF_RGB_size':[width,height],'SigLIP2_tokens':729,'SigLIP2_feature_dim':1152,'Qwen_bounded729_tokens':tokens,'Qwen_feature_dim':3584,'native_Qwen_geometry_valid':True}
   except Exception as error:
    bad.add(h);observation={'image_hash':h,'image_path':str(path),'EXIF_RGB_size':[width,height],'native_Qwen_geometry_valid':False,'error_type':type(error).__name__,'error':str(error)}
   seen[h]=observation;observations.append(observation)
 return seen,bad,observations,{'source':a.binding(inspect.getsourcefile(smart_resize)),'profile':'qwen_native_bounded_729','patch_size':14,'merge_size':2,'min_pixels':3136,'max_pixels':571536,'variable_tokens_upper_bound':729,'two_feature_dims':[1152,3584],'feature_dimensions_do_not_change_context_length':True}
def internal_selection(records):
 path=ROOT/'src/eval/research_selection.py';names=['_train_derived_source','_teacher_reference','_explicit_mcq_letter','_convert_record','build_internal_validation'];tree=ast.parse(path.read_bytes());ns={'copy':copy,'hashlib':hashlib,'json':json,'math':math,'random':random,'re':re}
 selected=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('GROUPS','PROTOCOL_NAME') for t in n.targets)]
 selected += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(path),'exec'),ns)
 unique={r['record_id']:r for r in records};result=ns['build_internal_validation'](list(unique.values()),per_group=32,seed=42)
 if result['protocol']['records']!=116:raise ValueError('Existing frozen visual internal116 must stay exact')
 return result,a.binding(path)
def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise RuntimeError('Budget preparation is CPU-only')
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TORCH='0',TOKENIZERS_PARALLELISM='false')
 OUT.mkdir(exist_ok=True);E.mkdir(exist_ok=True);a.resources()
 if (OUT/'release.json').exists():
  release=a.read(OUT/'release.json')
  for ref in release['files'].values():
   if a.sha(ref['path'])!=ref['sha256']:raise ValueError('Existing budget release changed')
  print(a.canonical(release),flush=True);return
 parent_audit=a.read(P/'audit.json');mix=a.read(M/'mix_audit.json');parent={}
 for stage in ('alignment','sft','text'):
  for split in ('train','validation'):
   name=stage+'_'+split+'.jsonl';path=P/name
   if a.sha(path)!=parent_audit[stage+'_'+split+'_sha256']:raise ValueError('Parent immutable manifest changed')
   parent[name]=list(a.rows(path))
 if a.sha(M/'sft_validation.jsonl')!=mix['splits']['validation']['sha256']:raise ValueError('Exact fixed226validation changed')
 validation=list(a.rows(M/'sft_validation.jsonl'));heldout=list(a.rows(HELDOUT))
 if len(validation)!=226 or len(parent['alignment_validation.jsonl'])!=372 or len(heldout)!=512:raise ValueError('Fixed372/226/512validation scope required')
 all_train=parent['alignment_train.jsonl']+parent['sft_train.jsonl']+parent['text_train.jsonl'];by_id={r['record_id']:r for r in all_train}
 full_heldout=ROOT/'datasets/research/eval-full-overlap-v1/full_heldout_image_index.jsonl';protected_images={r['image_hash'] for r in a.rows(full_heldout)}|{h for r in heldout for h in r['image_hashes']}
 val_images={h for r in validation+parent['alignment_validation.jsonl'] for h in r['image_hashes']};val_ids={r['record_id'] for r in validation+parent['alignment_validation.jsonl']}
 review_path=ROOT/'datasets/research/eval-full-overlap-v1/near_only_training_review_candidates.jsonl';review_ids={r['record_id'] for r in a.rows(review_path)};review_images={h for r in all_train if r['record_id'] in review_ids for h in r['image_hashes']}
 native,Clock,english,code=a.native_functions();refs=assets();shapes,bad_geometry,observations,vision=image_shapes(all_train+validation+parent['alignment_validation.jsonl'])
 a.save_rows(E/'all_parent_native_image_geometry.jsonl',observations)
 excluded=[]
 def clean(row):
  reasons=[];hs=set(row['image_hashes'])
  if row['source']['split']!='train' or row['partition']!='train' or row['language']!='en':raise ValueError('Only original parent EN TRAIN pools allowed')
  if hs&protected_images:reasons.append('known_declared_official_heldout_exact_RGB')
  if hs&val_images or row['record_id'] in val_ids:reasons.append('fixed_internal_or_caption_validation_overlap')
  if hs&review_images:reasons.append('conservative_existing_near_review_image_risk_exclusion_not_identity_claim')
  if hs&bad_geometry:reasons.append('native_Qwen_bounded729_geometry_rejected')
  if english({'question':row['question'],'answer':row['answer']}):reasons.append('existing_frozen_English_heuristic')
  if reasons:excluded.append({'record_id':row['record_id'],'reasons':reasons})
  return not reasons
 eligible=[r for r in all_train if clean(r)];eligible_ids={r['record_id'] for r in eligible}
 text_pool=[r for r in parent['text_train.jsonl'] if r['record_id'] in eligible_ids];bad_text=set();text_context={}
 for model in MODELS:
  dummy=tokenizer(model,refs[model],Clock);report=[]
  for row in text_pool:
   stat=measure(dummy,row,native)
   if stat['zero_supervised_position']:bad_text.add(row['record_id'])
   report.append({'record_id':row['record_id'],**stat})
  text_context[model]={'records':len(report),'zero_supervised_position_records':[r['record_id'] for r in report if r['zero_supervised_position']]};a.save_rows(E/('all_text_context_'+model+'.jsonl'),report);del dummy;gc.collect()
 text=choose([r for r in text_pool if r['record_id'] not in bad_text],529,'SmolTalk')
 captions=[r for r in parent['alignment_train.jsonl'] if r['record_id'] in eligible_ids];unique_images={}
 for row in sorted(captions,key=lambda r:rank(r,'caption_duplicate_RGB_owner')):unique_images.setdefault(tuple(row['image_hashes']),row)
 cap_pool=list(unique_images.values());A0=choose(cap_pool,4096,'A0');used={h for r in A0 for h in r['image_hashes']};A1=choose([r for r in cap_pool if not used.intersection(r['image_hashes'])],256,'A1')
 mm=[]
 for group,n in GROUPS.items():mm.extend(choose([r for r in parent['sft_train.jsonl'] if r['record_id'] in eligible_ids and r['task_group']==group],n,'SFT:'+group))
 sft=sorted(mm+text,key=lambda r:rank(r,'SFT_final_order'))
 manifests={'alignment_train.jsonl':A0,'alignment_a1_train.jsonl':A1,'sft_multimodal_train.jsonl':mm,'text_train.jsonl':text,'sft_train.jsonl':sft}
 for name,values in manifests.items():a.save_rows(OUT/name,values)
 for source,name in ((P/'alignment_validation.jsonl','alignment_validation.jsonl'),(M/'sft_validation.jsonl','sft_validation.jsonl'),(HELDOUT,'heldout_evaluation.jsonl')):a.immutable(OUT/name,source.read_bytes())
 selection,selection_source=internal_selection(validation);a.save_rows(OUT/'internal_selection_116.jsonl',selection['records']);a.save(OUT/'internal_selection_protocol.json',selection['protocol'])
 selected=A0+A1+sft+validation+parent['alignment_validation.jsonl'];context={}
 for model in MODELS:
  dummy=tokenizer(model,refs[model],Clock);stats=[{'record_id':r['record_id'],**measure(dummy,r,native)} for r in selected];a.save_rows(E/('selected_context_'+model+'.jsonl'),stats)
  failures=[r for r in stats if r['zero_supervised_position']]
  context[model]={'records':len(stats),'prompt_overflows':len(failures),'overflow_IDs':[r['record_id'] for r in failures],'target_truncation_records':sum(r['full_target_truncated_by_existing_residual4096_policy'] for r in stats),'max_expanded_prompt_tokens_upper_bound':max(r['expanded_prompt_tokens_upper_bound'] for r in stats),'same_729_token_upper_bound_for_both_encoders':True};del dummy;gc.collect()
  if failures:raise ValueError('Fixed selected or unchanged validation prompt overflow for'+model+'; artifacts preserved, release refused')
 image_paths={h:path for r in selected for h,path in zip(r['image_hashes'],r['image_paths'])};verified=[]
 def verify_image(item):
  h,path=item;actual=cache.image_content_hash(path)
  if actual!=h:raise ValueError('Actual parent image RGB bytes changed')
  return {'canonical_RGB_sha256':h,'image_path':path,'encoded_bytes':Path(path).stat().st_size,'actual_current_RGB_redecoded':True,**shapes[h]}
 with ThreadPoolExecutor(max_workers=2) as pool:
  for i,row in enumerate(pool.map(verify_image,sorted(image_paths.items()))):
   verified.append(row)
   if i%500==0:print(a.canonical({'phase':'budget18_current_selected_RGB_verification','images':i,'total':len(image_paths),'resources':a.resources()}),flush=True)
 a.save_rows(E/'selected_current_RGB_asset_verification.jsonl',verified);a.save_rows(E/'parent_exclusion_decisions.jsonl',excluded)
 a0_h={h for r in A0 for h in r['image_hashes']};a1_h={h for r in A1 for h in r['image_hashes']};train_h={h for r in A0+A1+sft for h in r['image_hashes']}
 if a0_h&a1_h or train_h&val_images or train_h&protected_images or train_h&review_images or len({r['record_id'] for r in sft})!=3529:raise ValueError('Final leakage/unique/disjoint invariant failed')
 protocol={'schema_version':1,'profile':'budget18-en-v1','owner':'frozen_budget18_common_data_seed42','models':MODELS,'recipes':['1b','4f','3c'],'vision_profiles':{'SigLIP2':{'max_tokens':729,'feature_dim':1152},'Qwen2.5VL7Bvision':{'profile':'qwen_native_bounded_729','max_tokens':729,'feature_dim':3584}},'seed':42,'selection':'SHA256 hash ranking of original source repo/revision/config/rowID/turnIndex/imageIDs; no answer/quality-score ranking','no_training_repeats':True,'caption_RGB_duplicate_owner':'fixed hashseed42 owner before sampling','caption_A0':4096,'caption_A1':256,'A0_A1_image_disjoint':True,'SFT_MM':3000,'SFT_group_quotas':GROUPS,'SFT_text':529,'SFT_total':3529,'SFT_text_fraction':529/3529,'fixed_validation':{'caption372':a.binding(OUT/'alignment_validation.jsonl'),'SFT226':a.binding(OUT/'sft_validation.jsonl'),'internal116':a.binding(OUT/'internal_selection_116.jsonl'),'official_subset512':a.binding(HELDOUT)},'parent':{'audit':a.binding(P/'audit.json'),'mixture_audit':a.binding(M/'mix_audit.json'),'manifests':{name:a.binding(P/name) for name in parent}},'context':{'max_length':4096,'prompt_schema_version':2,'residual_target_truncation_policy_unchanged':True,'arbitrary_new_target_cap':None,'tokenizer_assets':refs,'native_code':code,'native_Qwen':vision},'known_heldout':{'full_RGB_index':a.binding(full_heldout),'existing_near_review_IDs':a.binding(review_path),'existing_near_flags_treated_as_conservative_exclusions_not_verified_identity':True,'all_official_private_unselected_and_fuzzy_text_coverage_claimed':False},'source_sha256':{str(Path(__file__).resolve().relative_to(ROOT)):a.sha(__file__),'src/dataset/vision_cache.py':a.sha(ROOT/'src/dataset/vision_cache.py'),'src/eval/research_selection.py':selection_source['sha256']},'new_download_bytes':0,'source_assets_or_parent_manifests_modified':False,'models_vision_weights_or_GPU_used':False,'full_primary_396_scope_satisfied':False}
 a.save(OUT/'protocol.json',protocol);audit={'schema_version':1,'status':'passed','profile':'budget18-en-v1','created_at':datetime.now(timezone.utc).isoformat(),'counts':{'A0':len(A0),'A1':len(A1),'MM':len(mm),'text':len(text),'SFT_total':len(sft),'caption_validation':372,'SFT_validation':226,'internal_selection':116,'official_heldout_subset':512},'actual_group_quotas':dict(Counter(r['task_group'] for r in mm)),'training_record_repeat_count':len(sft)-len({r['record_id'] for r in sft}),'A0_A1_RGB_overlap':len(a0_h&a1_h),'train_validation_RGB_overlap':len(train_h&val_images),'train_known_official_exact_RGB_overlap':len(train_h&protected_images),'train_existing_near_review_image_overlap':len(train_h&review_images),'actual_current_RGB_assets_verified':len(verified),'all3_models_both_encoder_context':context,'all_text_common_context_excluded_IDs':sorted(bad_text),'fixed_parent_validation_bytes_preserved':True,'new_download_bytes':0,'Torch_imported':'torch' in sys.modules,'language_model_or_encoder_forward_calls':0,'global_primary_decontamination_or_396_completion_claim':False,'resources':a.resources(),'protocol':a.binding(OUT/'protocol.json')};a.save(OUT/'audit.json',audit)
 release={'schema_version':1,'status':'passed','profile':'budget18-en-v1','seed':42,'audit':a.binding(OUT/'audit.json'),'protocol':a.binding(OUT/'protocol.json'),'files':{p.name:a.binding(p) for p in OUT.iterdir() if p.is_file()},'context_source':a.binding(__file__),'evidence_files':{p.name:a.binding(p) for p in E.iterdir() if p.is_file()},'prior_full_primary_or_budget9_outputs_overwritten':False};a.save(OUT/'release.json',release);print(a.canonical({'status':'passed','release':a.binding(OUT/'release.json'),'audit':audit}),flush=True)

if __name__=='__main__':main()
