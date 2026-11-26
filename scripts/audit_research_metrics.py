#!/usr/bin/env python3
"""CPU-only, read-only audit of cached 512-record strict-v3 predictions.

Writes a separate GQA case/punctuation sensitivity view. Does not launch model
generation, rewrite original scores, tune prompts, or affect checkpoint choice.
Rerun unchanged to include later completed recipes/grounding variants.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.eval.research_metric_audit import (
    GQA_VIEW,VARIANTS,analysis_protocol,audit_cached_evaluation,json_sha256,sha256_bytes,
)

RECIPES=('1a','1b','1c','2a','2b','2c','3a','3b','3c','4q','4f')


def atomic_write(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(text);os.replace(temporary,path)


def atomic_json(path,value):atomic_write(path,json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def run_directory(evaluation):
    for parent in Path(evaluation).parents:
        if (parent/'resolved_config.json').is_file() or '__SFT_' in parent.name:return parent
    raise ValueError('Cannot identify source SFT run directory')


def discover_evaluations(runs_root,pattern):
    result=[]
    for path in Path(runs_root).glob('*/evaluation/**/protocol.json'):
        run=run_directory(path.parent)
        if fnmatch.fnmatch(run.name,pattern):result.append(path.parent)
    return sorted(set(result))


def render_markdown(index):
    upstream='https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/c969af54173a5b5393a6dfe8304dd87423fc9388/'
    lines=['# Supplemental cached metric audit','',
        'Original strict-v3 scores, predictions, prompts, checkpoints and selection rules are preserved. '
        'This separate view changes only GQA scoring to the pinned lmms-eval case/punctuation rule. '
        'It is a sensitivity analysis on the same cached subset, not a full upstream task run.','',
        f"The [pinned GQA task]({upstream}lmms_eval/tasks/gqa/gqa.yaml) enables case and ASCII-punctuation ignoring. "
        f"The [metric implementation]({upstream}lmms_eval/api/metrics.py) lowercases with NumPy and deletes `string.punctuation`; "
        f"the [generate-until call site]({upstream}lmms_eval/api/task.py) first strips prediction outer whitespace. "
        'Full predictions are compared; no articles, number words, synonyms, sentence answers or explanations are extracted.','',
        '| Recipe | Variant | Original macro | GQA supplemental macro | GQA raw exact | GQA strict | GQA normalized |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in index['evaluations']:
        gqa=row['gqa'];n=gqa['records']
        lines.append(f"| {row['run_metadata'].get('recipe','?')} | {row['variant']} | "
            f"{row['strict_v3_original']['group_macro_score']:.6f} | {row['supplemental_gqa_lmms']['group_macro_score']:.6f} | "
            f"{gqa['raw_literal_correct']}/{n} | {gqa['strict_v3_correct']}/{n} | {gqa['lmms_case_punctuation_correct']}/{n} |")
    for row in index['evaluations']:
        gqa=row['gqa']
        lines+=['',f"`{row['run_metadata']['run_id']}` / `{row['variant']}`: "
            f"{gqa['recovered_from_strict']} GQA answers are recovered by full-string normalization; "
            f"{gqa['case_and_terminal_punctuation_only_recoveries']} involve only case and terminal ASCII punctuation. "
            f"{gqa['remaining_unmatched']} remain unmatched, including {gqa['remaining_binary_opposites']} explicit yes/no opposites. "
            'Other mismatches are not semantically adjudicated. '
            f"Raw predictions SHA: `{row['identity']['raw_predictions_sha256']}`. "
            f"Checkpoint SHA: `{row['identity']['checkpoint_sha256']}`."]
    lines+=['','The original VQAv2 legacy evaluator is unchanged: both reference and prediction '
        'normalization is conditional on non-unanimous raw references in the '
        '[pinned official evaluator](https://github.com/GT-Vision-Lab/VQA/blob/a013f0043c1e2cdc995922dfe257f7149aa9af06/PythonEvaluationTools/vqaEvaluation/vqaEval.py). '
        'This GQA-only view retains every original VQAv2 score.','',
        f"Completed caches analysed: {len(index['evaluations'])}. Pending cached variants: {len(index['pending'])}. "
        'The identical analysis protocol applies to all 11 recipes and none/shuffle/blank/text_only whenever '
        'their complete 512-record caches exist. Missing caches are not assigned scores.','',
        f"Analysis protocol SHA: `{index['analysis_protocol_sha256']}`. "
        'The JSON binds raw-prediction bytes, parent protocol JSON/file hashes, source/transformed manifest hashes, '
        'checkpoint identity inherited from the parent receipt, analysis source hashes and pinned upstream/license hashes. '
        'Original cached artifact bytes and every parent source hash are verified during the audit.','']
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root',default=str(ROOT/'runs/research'))
    parser.add_argument('--run-pattern',default='*__SFT_pilot_scienceqa_context_v2_balanced_s42')
    parser.add_argument('--evaluation',action='append',help='Explicit evaluation directory; repeat for multiple caches')
    parser.add_argument('--manifest',default=str(ROOT/'datasets/research/eval-heldout-v3/evaluation.jsonl'))
    parser.add_argument('--output',default=str(ROOT/'research_results/metric_audit_v1'))
    args=parser.parse_args(argv)
    manifest=Path(args.manifest);manifest_sha=sha256_bytes(manifest.read_bytes())
    protocol=analysis_protocol(manifest_sha);protocol_sha=json_sha256(protocol)
    evaluations=[Path(path) for path in args.evaluation] if args.evaluation else discover_evaluations(args.runs_root,args.run_pattern)
    output=Path(args.output).resolve()
    if output.is_relative_to(manifest.parent.resolve()) or any(output.is_relative_to(path.resolve()) for path in evaluations):
        raise ValueError('Supplemental output must be separate from original data/evaluation artifacts')
    protocol_path=output/'analysis_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Analysis code/upstream/manifest protocol changed; use a new supplemental output directory')
    completed=[];pending=[];coverage={recipe:{variant:[] for variant in VARIANTS} for recipe in RECIPES}
    for evaluation in evaluations:
        if not (evaluation/'summary.json').is_file():
            pending.append({'directory':str(evaluation),'status':'waiting_for_complete_parent_summary'})
            continue
        result=audit_cached_evaluation(evaluation,manifest)
        run=run_directory(evaluation)
        config_path=run/'resolved_config.json'
        config=json.loads(config_path.read_text()) if config_path.is_file() else {}
        recipe=config.get('recipe') or next((name for name in RECIPES if '__'+name+'__' in run.name),None)
        if recipe not in RECIPES:raise ValueError('Unknown research recipe in cached run')
        result['run_metadata']={'run_id':run.name,'recipe':recipe,**{name:config[name] for name in ('model','vision','seed','stage') if name in config}}
        if config_path.is_file():result['resolved_config_sha256']=sha256_bytes(config_path.read_bytes())
        rows=result.pop('rows')
        cache_id=hashlib.sha256((str(evaluation.resolve())+result['identity']['parent_protocol_sha256']).encode()).hexdigest()[:20]
        row_path=output/'rows'/(cache_id+'.jsonl')
        text=''.join(json.dumps(dict(row,analysis_protocol_sha256=protocol_sha),ensure_ascii=False)+'\n' for row in rows)
        result['supplemental_rows']={'path':str(row_path),'sha256':sha256_bytes(text.encode()),'records':len(rows)}
        result['analysis_protocol_sha256']=protocol_sha
        completed.append((result,row_path,text));coverage[recipe][result['variant']].append(str(evaluation))
    if analysis_protocol(sha256_bytes(manifest.read_bytes()))!=protocol:
        raise ValueError('Analysis source/dependencies/manifest changed during cached audit')
    index={'schema_version':1,'name':GQA_VIEW,'created_at':datetime.now(timezone.utc).isoformat(),
           'analysis_protocol_sha256':protocol_sha,'source_manifest_sha256':manifest_sha,
           'evaluations':[entry[0] for entry in completed],'pending':pending,'coverage':coverage,
           'scope':'supplemental cached metric sensitivity; not experiment completion or checkpoint selection',
           'original_artifacts_modified':False,'models_loaded':False,'gpu_used':False,'generation_launched':False}
    atomic_json(protocol_path,protocol)
    for result,path,text in completed:atomic_write(path,text)
    atomic_json(output/'index.json',index)
    atomic_write(output/'report.md',render_markdown(index))
    print(json.dumps({'output':str(output),'completed_caches':len(completed),'pending_caches':len(pending),
                      'analysis_protocol_sha256':protocol_sha,'gpu_used':False}))
    return index


if __name__=='__main__':main()
