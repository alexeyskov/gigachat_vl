"""Metadata/stat-only capacity forecast; never deletes or reads tensor bodies."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];GIB=2**30;RESERVE=120*GIB
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inventory():
 rows=[]
 for run in sorted((ROOT/'runs/research').iterdir()):
  p=run/'resolved_config.json';m=run/'.checkpoint_manager.json'
  if not p.is_file() or not m.is_file():continue
  cfg=json.loads(p.read_bytes());manager=json.loads(m.read_bytes());objects={}
  for role in ('best','last'):
   link=run/role
   if not link.is_symlink():continue
   obj=link.resolve(strict=True)
   if obj.parent!=(run/'.checkpoint_objects').resolve():raise ValueError('Role outside owned objects')
   marker=obj/'.research_checkpoint.json';mark=json.loads(marker.read_bytes())
   if mark.get('owner')!=manager['owner']:raise ValueError('Checkpoint owner differs')
   entry=objects.setdefault(str(obj),{'roles':[],'files':[]})
   entry['roles'].append(role)
   if not entry['files']:
    for f in sorted(obj.iterdir()):
     if f.is_symlink() or not f.is_file():raise ValueError('Unknown owned object member')
     s=f.stat();entry['files'].append({'path':str(f),'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns})
  if not objects:continue
  new='budget18-en-v1' in str(cfg);dependency=not new and (run.name=='smallthinker4__siglip2__4f__SFT_pilot_scienceqa_context_v2_balanced_s42' or run.name=='smallthinker4__siglip2__3b__SFT_pilot_scienceqa_context_v2_balanced_s42')
  rows.append({'run':str(run),'stage':cfg.get('stage'),'recipe':cfg.get('recipe'),'current_series':new,'objects':objects,'allocated_bytes':sum(f['allocated_bytes'] for o in objects.values() for f in o['files']),'retirement_class':'preserve_current_series' if new else 'preserve_old_archival_activation_calibration_until_replaced' if dependency else 'obsolete_cancelled_series_candidate_requires_explicit_root_retirement','keep_all_nonweight_metrics_config_provenance_receipts':True})
 return rows
def forecast():
 source=ROOT/'research_results/primary_capacity.json';capacity=json.loads(source.read_bytes());v=os.statvfs(ROOT);free=v.f_bavail*v.f_frsize;total=v.f_blocks*v.f_frsize
 inv=inventory();runs=[];retained=0
 for model in ('smallthinker4','lfm25','giga31'):
  census=next(x for x in capacity['models'] if x['id']==model)
  for encoder in ('siglip2','qwen25vision'):
   for recipe in ('1b','4f','1c'):
    r=next(x for x in census['recipes'] if x['recipe']==recipe);enc=r['encoders'][encoder];n=r['llm_trainable_parameters'];model_bytes=enc['minimum_raw_fp32_model_bytes'];moments=(8 if recipe in ('1b','1c') else 2)*n
    payload=model_bytes+moments+128*2**20
    # Lower-bound active peak: one final resumable object then one model-only
    # staging replacement. bnb small FP32 slots/scales and metadata can add more.
    guard=max(16*2**20,int(.05*model_bytes));peak=retained+payload+model_bytes+guard
    runs.append({'model':model,'encoder':encoder,'recipe':recipe,'model_parameter_census_bytes':model_bytes,'trainable_llm_parameters':n,'optimizer_main_state_lower_bound_bytes':moments,'final_save_payload_lower_bound_bytes':payload,'retained_previous_new_models_bytes':retained,'model_only_compaction_peak_lower_bound_bytes':peak,'minimum_free_from_snapshot_including120GiB_bytes':RESERVE+peak,'lower_bound_already_exceeds_snapshot_free':RESERVE+peak>free})
    retained+=model_bytes
 current=sum(x['allocated_bytes'] for x in inv if x['current_series']);old=sum(x['allocated_bytes'] for x in inv if not x['current_series']);candidate=sum(x['allocated_bytes'] for x in inv if x['retirement_class'].startswith('obsolete'))
 return {'schema_version':2,'created_unix':time.time(),'owner':'budget18_metadata_storage_forecast_v2','scope':{'models':3,'encoders':2,'recipes':['1b','4f','1c'],'SFT_runs':18,'full_model_runs':6,'QLoRA_runs':12,'A0_records':4096,'A1_records':256,'SFT_MM_records':3000,'SFT_text_records':529,'epochs':1,'seed':42,'one_final_best_last_shared_object':True,'recipe4f_skips_A1':True},'source_sha256':{'research_results/primary_capacity.json':sha(source),'scripts/forecast_research_budget18_storage_v1.py':sha(__file__),'src/utils/research_budget_storage_v1.py':sha(ROOT/'src/utils/research_budget_storage_v1.py')},'snapshot':{'total_bytes':total,'free_bytes':free,'reserve_bytes':RESERVE,'already_retained_owned_old_bytes':old,'already_retained_current_series_bytes':current},'all18_model_tensor_bytes_minimum':retained,'all18_model_only_additional_from_snapshot_bytes_minimum':retained,'raw_retention_gap_from_current_free_bytes_minimum':max(0,retained+RESERVE-free),'obsolete_candidate_bytes_no_calibration_dependencies':candidate,'after_all_old_weights_retired_still_raw_gap_bytes_minimum':max(0,retained+RESERVE-free-old),'after_only_noncalibration_candidates_retired_raw_gap_bytes_minimum':max(0,retained+RESERVE-free-candidate),'runs':runs,'owned_weight_inventory':inv,'limitations':['All forecasts use measured source parameter counts, not weight-body reads.','6 full FP32 final models plus12 QLoRA deltas; no unmeasured compression credit.','Per-run active estimates are LOWER bounds: native bnb small FP32 slots/scales/metadata and A0/A1 temporary stages can add more. Actual final payload storage counter and manager5%/16MiB+120GiB floor remain authoritative.','Candidate retirement is a plan only, no delete authorization or file mutations. All current-series weights and old calibration-dependent4f/3b sources are separately protected.','Existing Small trained lossless archive ratio is not proof for LFM/Giga or56-step models. A generic lossless writer+decode+full payload/GPU validation or additional external capacity is necessary before promising all18 retained.','All metrics/configs/predictions/provenance/reload receipts remain even if root later retires obsolete cancelled weights.','No rawdataset/featurecache/HF anchors or current model files are retirement candidates in this plan.'],'tensor_bodies_read':False,'files_deleted':False,'GPU_used':False}
def main():
 report=forecast();out=ROOT/'research_results/budget18_storage_forecast_v2.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 lines=['# Reduced18-run storage capacity (metadata only)','',f"Snapshot free: {report['snapshot']['free_bytes']/GIB:.3f}GiB; reserve120GiB.",f"All18 final FP32 model tensors alone: {report['all18_model_tensor_bytes_minimum']/GIB:.3f}GiB (6full+12delta).",f"Gap from current free, including reserve: {report['raw_retention_gap_from_current_free_bytes_minimum']/GIB:.3f}GiB.",f"Old retained owned weights: {report['snapshot']['already_retained_owned_old_bytes']/GIB:.3f}GiB; non-calibration obsolete candidates: {report['obsolete_candidate_bytes_no_calibration_dependencies']/GIB:.3f}GiB.",f"Even retiring ALL old weights leaves a raw final-retention gap of {report['after_all_old_weights_retired_still_raw_gap_bytes_minimum']/GIB:.3f}GiB, before active checkpoint staging, caches or other growth.",'','No files deleted. Native best/last,120GiB reserve and32GiB host floor remain enforced. No LFM/Giga lossless ratio has been measured.','','| Obsolete/protected run | Owned weightGiB | Classification |','|---|---:|---|']
 lines += [f"| `{Path(x['run']).name}` | {x['allocated_bytes']/GIB:.3f} | {x['retirement_class']} |" for x in sorted(report['owned_weight_inventory'],key=lambda x:-x['allocated_bytes'])]
 (out.with_suffix('.md')).write_text('\n'.join(lines)+'\n');print(json.dumps({'path':str(out),'sha256':sha(out),'raw_final_models_GiB':report['all18_model_tensor_bytes_minimum']/GIB,'current_gap_GiB':report['raw_retention_gap_from_current_free_bytes_minimum']/GIB,'gap_even_all_old_weights_retired_GiB':report['after_all_old_weights_retired_still_raw_gap_bytes_minimum']/GIB,'noncalibration_candidate_GiB':report['obsolete_candidate_bytes_no_calibration_dependencies']/GIB}))
if __name__=='__main__':main()
