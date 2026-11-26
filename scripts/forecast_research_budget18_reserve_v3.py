"""Read-only disk scenarios for1b/4f/1c plus prospective completedR2 writer."""
from __future__ import annotations
import hashlib,json,math,os,time
from pathlib import Path
from forecast_research_budget18_storage_v1 import inventory
ROOT=Path(__file__).resolve().parents[1];GIB=2**30;META=128*2**20
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def block(n):return ((n+4095)//4096)*4096
def margin(n):return max(16*2**20,math.ceil(.05*n))
def forecast():
 capacity_path=ROOT/'research_results/primary_capacity.json';capacity=json.loads(capacity_path.read_bytes())
 cache_path=ROOT/'research_results/budget18_cache_retirement_plan_v1/report.json';cache=json.loads(cache_path.read_bytes());unique=cache['protected_budget_plus_blank_unique_RGB_hashes'];warm=cache['totals']['protected_budget_or_blank_files'];missing=unique-warm
 if cache['active_Qwen_cache_fingerprint']!='c4dda758033d729407d3a323b91ee64a5482bc1939a5c425ae24abb47059b532' or unique!=7616 or warm!=7306:raise ValueError('Recompute scenarios if fixed planner scope differs')
 qwen_feature=729*3584*2+8192;sig_feature=729*1152*2+8192
 cache_growth=unique*block(qwen_feature)+missing*block(sig_feature)+2*257*4096+block(qwen_feature)
 old_blank=ROOT/'runs/research/smallthinker4__siglip2__4f__SFT_pilot_scienceqa_context_v2_balanced_s42/evaluation/grounding_v1/blank/ablation_images'
 blank_files=[p for p in old_blank.iterdir() if p.is_file() and not p.is_symlink()];blank_per_run=sum(p.stat().st_blocks*512 for p in blank_files);blank_growth=18*(blank_per_run+4096)
 inv=inventory();current_intermediate=sum(x['allocated_bytes'] for x in inv if x['current_series'] and x['stage'] in ('A0','A1'));current_SFT=sum(x['allocated_bytes'] for x in inv if x['current_series'] and x['stage']=='SFT')
 v=os.statvfs(ROOT);free=v.f_bavail*v.f_frsize;retained=0;phases=[];models=[]
 def phase(label,held,anticipated):phases.append({'phase':label,'future_retained_model_bound_bytes':retained,'held_intermediate_or_resume_bytes':held,'anticipated_new_write_bytes':anticipated,'manager_extra_margin_bytes':margin(anticipated),'total_new_series_phase_bound_bytes':retained+held+anticipated+margin(anticipated)})
 for mid in ('smallthinker4','lfm25','giga31'):
  model=next(x for x in capacity['models'] if x['id']==mid)
  for enc in ('siglip2','qwen25vision'):
   P=next(x for x in model['recipes'] if x['recipe']=='4f')['encoders'][enc]['frozen_projector_parameters'];a0=12*P+META;pair=mid+'/'+enc
   phase(pair+'/NF4_A0_final',0,a0);phase(pair+'/BF16_A0_final',a0,a0);a0held=2*a0
   previous_A1_waiting_parent=0
   for recipe in ('1b','4f','1c'):
    r=next(x for x in model['recipes'] if x['recipe']==recipe);n=r['llm_trainable_parameters'];raw=r['encoders'][enc]['minimum_raw_fp32_model_bytes'];model_bound=raw+META;prefix=pair+'/'+recipe;A1=0
    if recipe!='4f':
     a1n=sum(value for group,value in r['trainable_parameter_groups'].items() if group not in ('ordinary_lora','packed_expert_lora'));A1=12*a1n+4*P+META;phase(prefix+'/A1_final',a0held+previous_A1_waiting_parent,A1)
     if recipe=='1c':previous_A1_waiting_parent=0  # BothNF4A0 consumers now exist; parent then oldA1 may retire.
    resume=model_bound+(8*n if recipe!='4f' else 0)
    phase(prefix+'/SFT_final_write',a0held+A1+previous_A1_waiting_parent,resume)
    if recipe!='4f':phase(prefix+'/completed_SFT_model_compaction',a0held+A1+previous_A1_waiting_parent+resume,model_bound)
    retained+=model_bound;models.append({'model':mid,'encoder':enc,'recipe':recipe,'raw_model_tensor_bytes':raw,'model_payload_upper_metadata_allowance_bytes':META,'retained_final_upper_bound_bytes':model_bound})
    if recipe=='1b':previous_A1_waiting_parent=A1
 trim=cache['totals']['candidate_recomputable_old_feature_allocated_bytes'];other=2*GIB;additional=cache_growth+blank_growth+other
 scenarios=[]
 for cache_trim in (False,True):
  # Rebase only current-series objects explicitly superseded/pruned by this
  # same DAG, then count all its stage objects in the per-phase bound again.
  baseline=free+current_intermediate+current_SFT+(trim if cache_trim else 0)-additional
  worst=max(phases,key=lambda p:p['total_new_series_phase_bound_bytes']);minimum=baseline-worst['total_new_series_phase_bound_bytes'];final=baseline-retained
  for floor in (120,80,64,48,32):
   failures=[p['phase'] for p in phases if baseline-p['total_new_series_phase_bound_bytes']<floor*GIB]
   scenarios.append({'reserve_GiB':floor,'candidate_feature_retirement_performed_assumption':cache_trim,'old_unique_scientific_weight_retirement_bytes':0,'candidate_feature_credit_bytes':trim if cache_trim else 0,'minimum_actual_free_after_write_and_margin_bound_bytes':minimum,'final_free_upper_growth_bound_bytes':final,'fits_all_enumerated_bounds':not failures,'slack_above_floor_at_worst_bound_bytes':minimum-floor*GIB,'first_refusing_phase':failures[0] if failures else None,'worst_phase':worst['phase']})
 return {'schema_version':3,'owner':'budget18_prospective_modelonly_reserve_scenarios_v3','created_unix':time.time(),'status':'forecast_only_no_policy_changed_or_deletion','source_sha256':{'scripts/forecast_research_budget18_reserve_v3.py':sha(__file__),'research_results/primary_capacity.json':sha(capacity_path),'research_results/budget18_cache_retirement_plan_v1/report.json':sha(cache_path)},'snapshot':{'free_bytes':free,'current_series_owned_intermediate_bytes_rebased':current_intermediate,'current_series_owned_SFT_bytes_rebased':current_SFT,'old_unique_scientific_weights_kept':True},'prospective_policy':{'only_4f_SFT_completed_model_only_final_write':True,'R1_R3_native_resumable_final_then_existing_model_compaction':True,'all18_best_last_shared_final_models_retained':True,'host_headroom_GiB_unchanged':32,'disk_floor_not_changed_by_this_report':True},'feature_growth_upper_bound':{'protected_RGB_including_fixed_blank':unique,'existing_protected_Siglip_files':warm,'missing_Siglip_files':missing,'Qwen729_all_cold_files':unique,'Qwen_per_file_raw_storage_bytes':729*3584*2,'Qwen_per_file_native_cache_metadata_allowance_bytes':8192,'Qwen_total_logical_upper_bound_bytes':unique*qwen_feature,'filesystem_block_bytes':4096,'cache_parent_directory_allowance_bytes':2*257*4096,'one_atomic_feature_temp_allowance_bytes':block(qwen_feature),'all_missing_feature_allocated_upper_bound_bytes':cache_growth,'all18_blank_PNG_allocated_bound_from_same_native245assets_bytes':blank_growth,'other_logs_predictions_provenance_growth_allowance_bytes':other,'source_or_tensor_bodies_read':False,'actual_Qwen_token_counts_not_assumed_equal729':True},'all18_final_model_tensor_bytes':sum(m['raw_model_tensor_bytes'] for m in models),'all18_retained_model_payload_upper_bound_bytes':retained,'candidate_cache_recomputable_bytes':trim,'scenarios':scenarios,'models':models,'phases':phases,'conditions':['Prospective R2 policy must first be implemented/source-bound/tested and enabled explicitly. No disk floor changes here.','All intermediateA0/A1 and QLoRA-resume optimizer storages use all-initialized Adam32 upper bounds,128MiB metadata/object and manager5percent-or16MiB margin.','All completed current-series intermediate stages are rebased and recounted; only the exact approved DAG successful-dependent cleanup is assumed. Old unique scientific weights receive zero retirement credit.','Entire future maximum Qwen/Siglip feature growth is charged upfront for every phase, even when it actually occurs later.','Native cache file metadata allowance8192B follows frozen cache writer reservation; allocation is rounded to4096B and directory+atomic-temp costs are added.','Other2GiB is an explicit planning allowance, not a proven bound for unrelated future downloads/work. No unrelated growth/provisioning may be silently admitted.','Cache candidate credit is hypothetical until ROOT verifies scanner/closure and executes exact retirement. All images/models/checkpoints/proofs remain.','Theoretical sample compression receives no capacity credit.']}
def main():
 r=forecast();p=ROOT/'research_results/budget18_reserve_scenarios_v3.json';p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');lines=['# Prospective completedR2 storage scenarios','','All old unique scientific weights stay. No reserve or file was changed. Full future729-token feature growth, all temporary stages and2GiB other outputs are charged.','','| Reclaim only approved unused features | FloorGiB | Worst remaining freeGiB | SlackGiB | Fits |','|---|---:|---:|---:|---|']
 lines += [f"| {s['candidate_feature_retirement_performed_assumption']} | {s['reserve_GiB']} | {s['minimum_actual_free_after_write_and_margin_bound_bytes']/GIB:.3f} | {s['slack_above_floor_at_worst_bound_bytes']/GIB:.3f} | {s['fits_all_enumerated_bounds']} |" for s in r['scenarios']]
 lines += ['',f"Raw final model tensors: {r['all18_final_model_tensor_bytes']/GIB:.3f}GiB; payloads with128MiB/object bound: {r['all18_retained_model_payload_upper_bound_bytes']/GIB:.3f}GiB.",f"Missing feature upper allocation: {r['feature_growth_upper_bound']['all_missing_feature_allocated_upper_bound_bytes']/GIB:.3f}GiB. Recomputable unrelated Siglip candidates: {r['candidate_cache_recomputable_bytes']/GIB:.3f}GiB.",'','32GiB is the lowest requested scenario, not an automatic recommendation or an installed policy. A fitting row is conditional on the full lifecycle/cleanup/feature bounds and prior completedR2 implementation. Host32GiB remains independent and unchanged.']
 p.with_suffix('.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'path':str(p),'sha256':sha(p),'scenarios':r['scenarios']}))
if __name__=='__main__':main()
