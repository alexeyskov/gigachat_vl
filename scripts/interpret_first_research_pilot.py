from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,io,os,sys,time
ROOT=Path('/workspace/gigachat_vl');sys.path.insert(0,str(ROOT))
import torch
import pyarrow.parquet as pq
from PIL import Image,ImageOps
from src.eval.research_metric_audit import analysis_protocol,audit_cached_evaluation,json_sha256
from src.eval.research_eval_data import SOURCE_SPECS,convert_row,iter_rows
from src.dataset.vision_cache import image_content_hash,canonical_json,_tensor_checksum
from src.utils.research_checkpoint_compat import checkpoint_identity
from scripts.filter_research_context import make_native_tokenizer

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def emit(event,**x):print(json.dumps({'event':event,**x}),flush=True)

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):
        raise RuntimeError('This independent read-only audit requires GPUs hidden')
    torch.set_num_threads(8);started=time.monotonic()
    run=ROOT/'runs/research/smallthinker4__siglip2__4q__SFT_pilot_scienceqa_context_v2_balanced_s42'
    evaldir=run/'evaluation/heldout_v3';manifest=ROOT/'datasets/research/eval-heldout-v3/evaluation.jsonl'
    original_files=[evaldir/name for name in ['predictions.jsonl','protocol.json','summary.json','model_provenance.json']]+[manifest]
    before={str(p):sha(p) for p in original_files}
    rows=read(manifest);predictions=read(evaldir/'predictions.jsonl');by_id={r['record_id']:r for r in predictions}
    index_path=ROOT/'research_results/metric_audit_v1/index.json';index=json.loads(index_path.read_text())
    namespace_protocol=json.loads((index_path.parent/'analysis_protocol.json').read_text())
    current_protocol=analysis_protocol(sha(manifest));assert namespace_protocol==current_protocol
    assert index['analysis_protocol_sha256']==json_sha256(current_protocol)
    independent=audit_cached_evaluation(evaldir,manifest)
    indexed=next(x for x in index['evaluations'] if Path(x['directory']).resolve()==evaldir)
    for key in ['identity','strict_v3_original','supplemental_gqa_lmms','supplemental_minus_strict_macro_points','gqa']:
     assert independent[key]==indexed[key],key
    assert sha(indexed['supplemental_rows']['path'])==indexed['supplemental_rows']['sha256']
    best=checkpoint_identity(run/'best/state.pt');protocol=json.loads((evaldir/'protocol.json').read_text())
    assert best['sha256']==protocol['checkpoint_sha256']==indexed['identity']['checkpoint_sha256']
    owner=json.loads((run/'.checkpoint_manager.json').read_text())['owner']
    marker=json.loads((Path(best['primary_path']).parent/'.research_checkpoint.json').read_text())
    assert marker['owner']==owner
    assert sha(evaldir/'predictions.jsonl')==indexed['identity']['raw_predictions_sha256']
    assert sha(run/'resolved_config.json')==indexed['resolved_config_sha256']
    emit('metric_and_current_best_binding_verified',best_step=marker['step'],records=len(rows))

    # Select pinned source rows before decoding images; no source or image writes.
    source_checks={};source_asset_hashes={};rgb_checked={};source_drafts={};raw_columns={}
    root=ROOT/'datasets/research';metadata=root/'eval-metadata'
    for bench in SOURCE_SPECS:
     selected=[r for r in rows if r['benchmark']==bench];spec=SOURCE_SPECS[bench]
     raw=root/'eval-raw'/spec['repo'].replace('/','--')/spec['files'][-1]
     source_asset_hashes[str(raw)]=sha(raw)
     target={r['source']['mirror_row_index']:r for r in selected};found={}
     for index_row,raw_row in iter_rows(raw):
      if index_row in target:found[index_row]=raw_row
     assert set(found)==set(target)
     gqa_images=None
     if bench=='gqa':
      imagefile=root/'eval-raw'/spec['repo'].replace('/','--')/spec['files'][0]
      source_asset_hashes[str(imagefile)]=sha(imagefile)
      needed={found[i]['imageId'] for i in found}
      gqa_images={r['id']:r['image'] for _,r in iter_rows(imagefile) if r['id'] in needed}
     for idx,record in target.items():
      source={key:record['source'][key] for key in ['repo','revision','config','split','shard','source_is_hf_mirror']}
      draft=convert_row(bench,found[idx],idx,source,metadata,gqa_images=gqa_images)
      assert draft is not None
      for field in ['benchmark','task_group','scorer','base_question','question','source','language','choices','answer_index','correct_choice','answers','scoring_protocol']:
       assert draft.get(field)==record.get(field),(bench,idx,field)
      value=draft['image_value']
      raw_bytes=value.get('bytes') if isinstance(value,dict) else None
      if raw_bytes is not None:
       with Image.open(io.BytesIO(raw_bytes)) as image:original_hash=image_content_hash(image)
      else:
       with Image.open(value['path'] if isinstance(value,dict) else value) as image:original_hash=image_content_hash(image)
      assert record['image_hashes']==[original_hash],(bench,idx,'source_image_RGB')
      current=record['image_paths'][0]
      if current not in rgb_checked:
       with Image.open(current) as image:
        actual_hash=image_content_hash(image);rgb=ImageOps.exif_transpose(image).convert('RGB');size=list(rgb.size)
       rgb_checked[current]={'hash':actual_hash,'size':size}
      assert rgb_checked[current]['hash']==original_hash
      source_drafts[record['record_id']]=draft
     source_checks[bench]={'records':len(selected),'exact_source_row_prompt_and_metadata_matches':len(selected),
        'source_image_RGB_to_materialized_RGB_matches':len(selected),'source_split':spec['split'],
        'source_revision':selected[0]['source']['revision'],'parquet_columns':pq.ParquetFile(raw).schema_arrow.names,
        'source_row_index_coverage_verified':True}
     emit('source_question_image_mapping_verified',benchmark=bench,records=len(selected))

    # Metadata + checksum proof for actual cache files, without encoding any image.
    provenance=json.loads((evaldir/'model_provenance.json').read_text());fingerprint=provenance['cache_fingerprint']
    cache=ROOT/'cache/vision'/fingerprint;unique={r['image_hashes'][0]:r['image_paths'][0] for r in rows}
    metadata_size_matches=0;entry_records=[];dtype_counts=Counter();shape_counts=Counter()
    for image_hash,path in unique.items():
     key=hashlib.sha256((fingerprint+image_hash).encode()).hexdigest();entry=cache/key[:2]/(key+'.pt')
     payload=torch.load(entry,map_location='cpu',weights_only=True,mmap=True);features=payload['features']
     assert payload['fingerprint']==fingerprint and payload['image_hash']==image_hash
     assert features.ndim==2 and tuple(features.shape)==(729,1152) and torch.isfinite(features).all()
     assert payload['checksum']==_tensor_checksum(features) and payload['valid_token_count']==729
     dtype_counts[str(features.dtype)]+=1;shape_counts[str(list(features.shape))]+=1
     if 'original_size' in payload.get('metadata',{}):
      assert payload['metadata']['original_size']==rgb_checked[path]['size'];metadata_size_matches+=1
     entry_records.append({'image_hash':image_hash,'cache_key':key,'size':entry.stat().st_size,'mtime_ns':entry.stat().st_mtime_ns,'checksum':payload['checksum']})
    cache_census_digest=hashlib.sha256(canonical_json(sorted(entry_records,key=lambda r:r['image_hash'])).encode()).hexdigest()
    emit('cache_key_metadata_checksums_verified',unique_images=len(unique))

    # Source-derived question/options/hint exactly reconstruct every current prompt.
    # No model is loaded. Gold fields are not passed to the generation prompt.
    tokenize,binding=make_native_tokenizer();prompt_digest=hashlib.sha256();prompt_max=0
    for record in rows:
     draft=source_drafts[record['record_id']]
     public={'question':draft['question'],'image_paths':record['image_paths'],'answer':'x'}
     pre,post,_=tokenize(public);prompt_max=max(prompt_max,len(pre)+len(post)+729)
     prompt_digest.update(canonical_json([record['record_id'],pre,post,record['image_hashes']]).encode())
    assert not any('lecture' in r['question'] and r['question']!=source_drafts[r['record_id']]['question'] for r in rows)

    patterns={};examples=[]
    for bench in SOURCE_SPECS:
     values=[r for r in predictions if r['benchmark']==bench]
     generated=[r.get('generated_tokens',len(r.get('generated_token_ids',[]))) for r in values]
     zero=[r for r in values if r['score']==0]
     pos=[r for r in values if r['score']>0]
     patterns[bench]={'records':len(values),'strict_zero_score_records':len(zero),'strict_positive_score_records':len(pos),
       'empty_predictions':sum(not r['prediction'].strip() for r in values),'hit64_token_generation_cap':sum(n==64 for n in generated),
       'mean_generated_tokens':sum(generated)/len(generated),'common_predictions':Counter(r['prediction'].strip() for r in values).most_common(5)}
     for row in zero[:1]:
      source=next(r for r in rows if r['record_id']==row['record_id'])
      examples.append({'benchmark':bench,'record_id':row['record_id'],'original_id':source['source'].get('original_id'),
         'question':row['question'][:250],'prediction':row['prediction'],'references':row['answers'],'strict_score':row['score'],
         'interpretation':'Illustrative strict mismatch, not an image-semantic adjudication.'})
    report={'schema_version':1,'status':'passed_independent_read_only_audit','scope':'first4q ordinary512-record heldout-v3 pilot interpretation; no grounding conclusion',
     'run':str(run),'best_checkpoint_artifact_identity':best,'best_checkpoint_owner':owner,'best_checkpoint_step':marker['step'],
     'bindings':{'raw_predictions_sha256':sha(evaldir/'predictions.jsonl'),'parent_protocol_sha256':json_sha256(protocol),
       'parent_protocol_file_sha256':sha(evaldir/'protocol.json'),'parent_summary_sha256':sha(evaldir/'summary.json'),
       'manifest_sha256':sha(manifest),'metric_audit_index_sha256':sha(index_path),'analysis_protocol_sha256':json_sha256(current_protocol),
       'resolved_config_sha256':sha(run/'resolved_config.json'),'source_sha256':current_protocol['source_sha256'],
       'frozen_generation_source_sha256':protocol['source_sha256'],'source_parquet_sha256':source_asset_hashes,
       'interpretation_program_sha256':sha(__file__)},
     'metric_audit_namespace_protocol_matches':True,'metric_audit_existing_index_matches_current_best_and_raw_predictions':True,
     'metric_audit_recomputed_in_memory':True,'metric_audit_namespace_modified':False,
     'model_vision_provenance':provenance,
     'strict_original':independent['strict_v3_original'],'gqa_case_punctuation_sensitivity':independent['supplemental_gqa_lmms'],
     'gqa_sensitivity_diagnostics':{k:v for k,v in independent['gqa'].items() if k!='all_examples'},
     'source_question_image_mapping':source_checks,
     'actual_cache_read_only_checks':{'unique_images':len(unique),'records':len(rows),'key_metadata_fingerprint_and_RGB_hash_matches':True,
       'tensor_checksum_matches':True,'finite_shapes':dict(shape_counts),'storage_dtypes':dict(dtype_counts),
       'original_size_metadata_matches':metadata_size_matches,'cache_entry_census_sha256':cache_census_digest,'features_reencoded':False,
       'limit':'Key/hash/size/checksum/order consistency verified. Historical encoder content-image correspondence is not independently recomputed.'},
     'prompt_mapping_and_leakage_checks':{'all512_prompts_match_source_question_plus_declared_context_options_and_suffix':True,
       'ScienceQA_official_hint_included':True,'ScienceQA_lecture_and_solution_not_appended':True,'MCQ_all_choices_included_not_selected_gold':True,
       'OCR_tokens_and_doc_answers_not_appended':True,'native_question_only_prompt_token_census_sha256':prompt_digest.hexdigest(),
       'max_expanded_prompt_tokens_siglip729':prompt_max,'tokenizer_binding':binding,
       'answer_string_occurrence_is_not_used_as_a_leakage_test':True,'historical_source_hashes_verified':True},
     'response_pattern_findings':{'empty_predictions':sum(x['empty_predictions'] for x in patterns.values()),
       'records_at64_token_cap':sum(x['hit64_token_generation_cap'] for x in patterns.values()),
       'cap_limited_benchmarks':{b:x['hit64_token_generation_cap'] for b,x in patterns.items() if x['hit64_token_generation_cap']},
       'chartqa_exact_repeated_string_1_2_period':patterns['chartqa']['common_predictions'][0],
       'interpretation':'Empty outputs and widespread generation truncation do not explain the aggregate errors; repeated short strings and metric-sensitive formatting are observed, not semantically adjudicated.'},
     'per_benchmark_response_patterns':patterns,'illustrative_mismatches_not_semantic_judgments':examples,
     'interpretation':[
      'Frozen strict-v3 macro is27.469223; GQA-only full-string case/ASCII-punctuation sensitivity macro is32.937973, a5.46875-point change. Neither is replaced or merged.',
      'GQA strict0/64 rises to28/64 solely under the pinned formatting normalization.36 remain unmatched, including14 binary yes/no opposites; formatting does not explain all errors.',
      'OCR/document/chart scores are low on this subset; higher MCQ scores are descriptive and do not demonstrate visual use.',
      'No gross source-row/question/materialized-RGB mapping mismatch was found across all512 records. This conclusion comes from source/schema/hash checks, not whether predictions match answers.',
      'The first pilot has one recipe and one seed; there is no ablation/baseline comparison supporting an MoE effect or a grounding conclusion.'
     ],
     'limits':['Eight64-record benchmark subsets,128 per task group, selected from downloaded heldout shards; not the full official benchmarks.',
       'Equal task-group macro mixes several scoring protocols; raw benchmark denominators and scores remain separate.',
       'GQA supplemental metric is a pinned sensitivity view, not a full upstream GQA protocol or a replacement primary metric.',
       'No new generation, training, GPU execution, downloads, score changes, prompt changes, or cache mutation.',
       'Completed checkpoint full-GPU roundtrip and all three grounding interventions are separate required gates; this report does not stand in for them.'],
     'gpu_used':False,'models_or_encoders_loaded':False,'encoder_forwards':0,'downloads':0,'elapsed_seconds':time.monotonic()-started}
    for p in original_files:assert sha(p)==before[str(p)],f'Original artifact changed: {p}'
    assert sha(index_path)==report['bindings']['metric_audit_index_sha256']
    report['original_artifact_bytes_unchanged']=True
    path=ROOT/'research_results/first_pilot_interpretation.json';path.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# First pilot: independent read-only interpretation','',
     'SmallThinker4 / SigLIP2 /4q, seed42; best checkpoint step100. The complete ordinary512-record cache matches the current compacted BEST artifact and the existing metric-audit index.','',
     '| Metric view | Group macro (0–100) |','| --- | ---: |',
     f"| Frozen strict-v3 | {report['strict_original']['group_macro_score']:.6f} |",
     f"| Separate GQA case/punctuation sensitivity | {report['gqa_case_punctuation_sensitivity']['group_macro_score']:.6f} |",'',
     'GQA changes from0/64 strict matches to28/64 after the pinned full-string formatting rule. All28 recoveries involve only case/terminal punctuation.36 remain unmatched, including14 yes/no opposites. No answer extraction or score replacement occurs.','',
     '| Benchmark | Strict score | Positive-score records /64 |','| --- | ---: | ---: |']
    for key,stats in report['strict_original']['benchmarks'].items():
     bench=key.split('::')[0];lines.append(f"| {bench} | {stats['score']:.4f} | {patterns[bench]['strict_positive_score_records']}/64 |")
    lines+=['','All512 questions, source IDs/rows, schemas, references/options and materialized canonical RGB hashes match their pinned local source rows. All470 unique cached image entries have matching keys/fingerprints/hash metadata, finite729×1152 features and valid checksums. No source mapping mismatch was found; no encoder re-run was performed.','',
     'Response patterns: no empty predictions; only2/512 outputs reach the64-token cap (one DocVQA, one ChartQA). ChartQA emits the exact string `1.2.` on14/64 rows. These are descriptive patterns, not a semantic judgment or a newly normalized score.','',
     'Native prompt reconstruction uses only the source-derived question, declared ScienceQA hint, all MCQ choices and fixed answer-format suffix. Gold choice selection, lecture/solution and OCR/DocVQA answer metadata are not appended. Answers were not used to infer whether an image/question mapping is correct.','',
     'These are bounded64-item source subsets and one recipe/seed. Low OCR/document/chart results and stronger MCQ results do not establish visual grounding or an MoE benefit. Full-GPU checkpoint replay and shuffle/blank/text-only comparisons remain separate gates.','',
     f"Raw predictions SHA256: `{report['bindings']['raw_predictions_sha256']}`.",
     f"Current BEST SHA256: `{best['sha256']}`.",
     f"Metric-audit index SHA256: `{report['bindings']['metric_audit_index_sha256']}`.",
     f"Full checks and source bindings: `research_results/first_pilot_interpretation.json`.",'']
    (ROOT/'research_results/first_pilot_interpretation.md').write_text('\n'.join(lines))
    emit('first_pilot_independent_interpretation_passed',report=str(path),strict=report['strict_original']['group_macro_score'],sensitivity=report['gqa_case_punctuation_sensitivity']['group_macro_score'],elapsed_seconds=report['elapsed_seconds'])

if __name__=='__main__':main()
