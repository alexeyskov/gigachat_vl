"""Supplemental cached-prediction audit; frozen strict-v3 scores stay intact.

GQA normalization follows pinned lmms-eval c969af54173a5b5393a6dfe8304dd87423fc9388.
Adopted from lmms-eval https://github.com/EvolvingLMMs-Lab/lmms-eval;
the exact-match implementation is Apache-2.0, Copyright 2020 The HuggingFace
Datasets Authors and the current dataset script contributor. Unmodified source,
license notices and file hashes are in third_party/research_metric_audit/.

No answers are extracted from sentences/explanations. This is a separate metric
sensitivity view, not a full upstream task run or a replacement primary score.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import string
import sys

import numpy as np

from src.eval.research_metrics import aggregate_scores,score_prediction

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM_SHA='c969af54173a5b5393a6dfe8304dd87423fc9388'
UPSTREAM_REPOS={'EvolvingLMMs-Lab/lmms-eval':UPSTREAM_SHA,
                'GT-Vision-Lab/VQA':'a013f0043c1e2cdc995922dfe257f7149aa9af06'}
APACHE_SHA='cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'
GQA_VIEW='gqa_lmms_case_punctuation_sensitivity_v1'
PUNCTUATION=string.punctuation
_TRANSLATION=str.maketrans('','',PUNCTUATION)
SEMANTIC_FIELDS=('record_id','question','answers','scorer','task_group','source','benchmark','choices','scoring_protocol')
ROW_FIELDS=SEMANTIC_FIELDS+('image_hashes','image_ablation')
VARIANTS=('none','shuffle','blank','text_only')
EXPECTED_UPSTREAM={
 'lmms_eval/metrics.py':'49821add017a73ea89606d96d58a1f01edd3981528b4802b6e41ecd38c6fd553',
 'lmms_eval/task.py':'dba8d0e2fa113e5b74dc6734521d49d7d507f8329f1caf745c5f07980bd7d51d',
 'lmms_eval/gqa.yaml':'4530277d93b2644b121ce6741612ffb22bbeadc6bd2e5f5301860a238f36c542',
 'lmms_eval/LICENSE':'17358b5caadacb8108f14a6a319779cc8a20feed699f79c59e396e7e9dd5c33b',
 'vqa_official/vqaEval.py':'f08edfcad5be0112500993e245c706b6cb928eadebe203f89f838e5e0d04bec8',
 'vqa_official/LICENSE':'4bf86fe8b104b69fc72caaa095b2fff4bade0d3d9a7970c6a5fd01fb736cad04',
}


def sha256_bytes(value):return hashlib.sha256(value).hexdigest()
def json_sha256(value):return sha256_bytes(json.dumps(value,sort_keys=True).encode())
def read_rows(data):return [json.loads(line) for line in data.splitlines() if line.strip()]


def upstream_provenance(root=ROOT):
    directory=Path(root)/'third_party/research_metric_audit'
    for name,digest in EXPECTED_UPSTREAM.items():
        if sha256_bytes((directory/name).read_bytes())!=digest:
            raise ValueError(f'Pinned upstream source/license changed: {name}')
    metadata=json.loads((directory/'UPSTREAM.json').read_text())
    if {row['sha256'] for row in metadata['files']}!=set(EXPECTED_UPSTREAM.values()):
        raise ValueError('Upstream source manifest disagrees with pinned source/license hashes')
    for row in metadata['files']:
        if row['revision']!=UPSTREAM_REPOS.get(row['repo']):raise ValueError('Upstream revision is not the pinned commit')
        expected=f"https://raw.githubusercontent.com/{row['repo']}/{row['revision']}/{row['upstream_path']}"
        if row['url']!=expected:raise ValueError('Upstream source URL differs from pinned commit')
    if sha256_bytes((directory/'APACHE-2.0.txt').read_bytes())!=APACHE_SHA:
        raise ValueError('Apache-2.0 license text changed')
    return {'manifest_sha256':sha256_bytes((directory/'UPSTREAM.json').read_bytes()),
            'files':metadata['files'],'licenses':metadata['licenses'],
            'additional_license':metadata['additional_license']}


def normalize_gqa_lmms(text,*,prediction=False):
    """Exactly the scalar upstream NumPy transform, including Unicode behavior.

    generate_until strips prediction outer whitespace before the metric; the
    exact-match function lowercases and deletes ASCII punctuation on both sides.
    It does not strip the reference, fold Unicode case, collapse spaces, convert
    numbers/articles, accept synonyms, or extract an answer from an explanation.
    """
    if prediction:text='' if text is None else str(text).strip()
    elif not isinstance(text,str):raise ValueError('GQA reference must be a string')
    values=np.char.lower(np.asarray([text]))
    return str(np.char.translate(values,table=_TRANSLATION)[0])


def gqa_lmms_exact_match(prediction,references):
    if not isinstance(references,list) or len(references)!=1:
        raise ValueError('The pinned GQA task has exactly one string reference')
    return float(normalize_gqa_lmms(prediction,prediction=True)==normalize_gqa_lmms(references[0]))


def analysis_protocol(manifest_sha256,*,root=ROOT):
    root=Path(root);upstream=upstream_provenance(root)
    code_files=('src/eval/research_metric_audit.py','scripts/audit_research_metrics.py')
    strict_files=('src/eval/research_metrics.py','third_party/vqa_normalization/tables.json',
                  'third_party/vqa_normalization/provenance.json','third_party/vqa_normalization/LICENSE')
    return {'schema_version':1,'name':GQA_VIEW,'source_manifest_sha256':manifest_sha256,
            'software':{'numpy':np.__version__,'python':sys.version.split()[0]},
            'source_sha256':{name:sha256_bytes((root/name).read_bytes()) for name in code_files},
            'strict_dependency_sha256':{name:sha256_bytes((root/name).read_bytes()) for name in strict_files},
            'upstream':upstream,'rule':{'changed_benchmark':'gqa','references':1,
                'prediction_preprocessing':'strip outer whitespace, as upstream generate_until',
                'case':'numpy.char.lower','punctuation':'delete ASCII string.punctuation on prediction and reference',
                'ascii_punctuation':PUNCTUATION,'reference_whitespace':'unchanged',
                'other_benchmarks':'original frozen strict-v3 scores','answer_or_explanation_mining':False},
            'applies_uniformly_to_all_recipes_and_variants':True,
            'not_a_full_upstream_task_protocol':True,
            'primary_strict_v3_scores_replaced':False}


def _assert_metrics_equal(expected,actual,path='summary'):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or not expected.keys()<=actual.keys():raise ValueError(f'{path} keys differ')
        for key,value in expected.items():_assert_metrics_equal(value,actual[key],path+'.'+key)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not isinstance(actual,(int,float)) or not math.isclose(expected,actual,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError(f'{path} differs from recomputed frozen strict-v3 metric')
    elif expected!=actual:raise ValueError(f'{path} differs from recomputed frozen strict-v3 metric')


def _safe_parent_sources(protocol,root):
    source=protocol.get('source_sha256',{})
    if 'src/eval/research_metrics.py' not in source:raise ValueError('Parent frozen strict scorer hash is missing')
    for name,digest in source.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe parent source path')
        if sha256_bytes((root/path).read_bytes())!=digest:raise ValueError(f'Parent source differs from frozen generation protocol: {name}')
    return dict(source)


def gqa_diagnostics(rows):
    result=Counter(records=0,raw_literal_correct=0,strict_v3_correct=0,case_only_upstream_correct=0,
        lmms_case_punctuation_correct=0,recovered_from_strict=0,lost_from_strict=0,
        case_and_terminal_punctuation_only_recoveries=0,remaining_binary_opposites=0,remaining_other_mismatches=0)
    examples=[]
    for row in rows:
        if row['benchmark']!='gqa':continue
        result['records']+=1;p=row['prediction'];gold=row['answers'][0]
        normalized_p=normalize_gqa_lmms(p,prediction=True);normalized_gold=normalize_gqa_lmms(gold)
        strict=row['strict_v3_score'];changed=row['supplemental_score']
        result['raw_literal_correct']+=p==gold
        result['strict_v3_correct']+=int(strict==1.)
        result['case_only_upstream_correct']+=p.strip().lower()==gold.lower()
        result['lmms_case_punctuation_correct']+=int(changed==1.)
        recovered=strict==0. and changed==1.
        result['recovered_from_strict']+=recovered
        result['lost_from_strict']+=strict==1. and changed==0.
        terminal_only=recovered and p.strip().rstrip(PUNCTUATION).lower()==gold.lower()
        result['case_and_terminal_punctuation_only_recoveries']+=terminal_only
        if not changed:
            opposite=normalized_p in ('yes','no') and normalized_gold in ('yes','no') and normalized_p!=normalized_gold
            result['remaining_binary_opposites']+=opposite
            result['remaining_other_mismatches']+=not opposite
        examples.append({'record_id':row['record_id'],'prediction':p,'reference':gold,
                         'normalized_prediction':normalized_p,'normalized_reference':normalized_gold,
                         'strict_score':strict,'supplemental_score':changed,
                         'terminal_punctuation_only_recovery':terminal_only})
    output=dict(result)
    output['remaining_unmatched']=output['records']-output['lmms_case_punctuation_correct']
    output['interpretation']='Full-string normalization recoveries quantify metric formatting sensitivity; other mismatches are not semantically adjudicated. No explanation/substring answer extraction.'
    output['all_examples']=examples
    return output


def audit_cached_evaluation(directory,manifest_path,*,root=ROOT,require_records=512):
    """Read-only validation and rescoring of a complete generation cache."""
    root=Path(root);directory=Path(directory);manifest_path=Path(manifest_path)
    paths={'parent_protocol':directory/'protocol.json','parent_summary':directory/'summary.json',
           'raw_predictions':directory/'predictions.jsonl','source_manifest':manifest_path}
    contents={key:path.read_bytes() for key,path in paths.items()}
    protocol=json.loads(contents['parent_protocol']);summary=json.loads(contents['parent_summary'])
    records=read_rows(contents['source_manifest']);predictions=read_rows(contents['raw_predictions'])
    manifest_sha=sha256_bytes(contents['source_manifest']);parent_sha=json_sha256(protocol)
    if len(records)!=require_records or len(predictions)!=require_records:raise ValueError('Only complete fixed-manifest cached evaluations can be audited')
    ids=[row['record_id'] for row in records]
    if len(set(ids))!=len(ids) or len({row['record_id'] for row in predictions})!=len(predictions):raise ValueError('Duplicate manifest or prediction IDs')
    if (protocol.get('manifest_sha256')!=manifest_sha or protocol.get('manifest_records')!=len(records)
        or protocol.get('selected_record_ids_sha256')!=sha256_bytes(json.dumps(ids).encode())):
        raise ValueError('Parent protocol manifest identity differs')
    if (not summary.get('complete_manifest') or summary.get('records')!=len(records)
        or summary.get('manifest_records')!=len(records) or summary.get('protocol_sha256')!=parent_sha
        or summary.get('manifest_sha256')!=manifest_sha or summary.get('checkpoint_sha256')!=protocol.get('checkpoint_sha256')):
        raise ValueError('Completed summary/protocol/checkpoint identity differs')
    parent_sources=_safe_parent_sources(protocol,root)
    ablation=protocol.get('image_ablation',{})
    variant=ablation.get('variant')
    if variant not in VARIANTS or ablation.get('source_manifest_sha256')!=manifest_sha:
        raise ValueError('Unsupported/mismatched parent image-ablation identity')
    if summary.get('image_ablation')!=ablation:raise ValueError('Summary ablation identity differs')
    effective=records
    if variant!='none':
        transformed_path=directory/'transformed_manifest.jsonl'
        transformed_bytes=transformed_path.read_bytes()
        if sha256_bytes(transformed_bytes)!=ablation.get('transformed_manifest_sha256'):
            raise ValueError('Transformed manifest identity differs')
        effective=read_rows(transformed_bytes)
        if [row['record_id'] for row in effective]!=ids:raise ValueError('Transformed manifest IDs/order differ')
        for original,changed in zip(records,effective):
            if any(original.get(field)!=changed.get(field) for field in SEMANTIC_FIELDS):
                raise ValueError('Ablation changed semantic fields/reference answers')
        paths['transformed_manifest']=transformed_path;contents['transformed_manifest']=transformed_bytes
    elif ablation.get('transformed_manifest_sha256')!=manifest_sha:
        raise ValueError('None ablation transformed identity differs')
    by_id={row['record_id']:row for row in predictions}
    if set(by_id)!=set(ids):raise ValueError('Cached prediction IDs differ from manifest')
    strict_rows=[];supplemental_rows=[];details=[]
    for record in effective:
        row=by_id[record['record_id']]
        if row.get('protocol_sha256')!=parent_sha:raise ValueError('Mixed raw-prediction parent provenance')
        if any(row.get(field)!=record.get(field) for field in ROW_FIELDS):raise ValueError('Cached row fields/answers differ from manifest')
        prediction=row.get('prediction')
        if not isinstance(prediction,str):raise ValueError('Cached prediction is not a string')
        strict=score_prediction(record,prediction)
        if not isinstance(row.get('score'),(int,float)) or not math.isclose(row['score'],strict,rel_tol=0,abs_tol=1e-12):
            raise ValueError('Cached score differs from frozen strict-v3 recomputation')
        benchmark=record.get('benchmark') or record.get('source',{}).get('benchmark')
        if benchmark=='gqa' and record['scorer']!='exact_match':raise ValueError('Unexpected GQA baseline scorer')
        changed=gqa_lmms_exact_match(prediction,record['answers']) if benchmark=='gqa' else strict
        base=dict(record,score=strict);strict_rows.append(base)
        supplemental_rows.append(dict(base,score=changed,scorer='gqa_lmms_case_punct_exact_match' if benchmark=='gqa' else record['scorer']))
        details.append({'record_id':record['record_id'],'benchmark':benchmark,'task_group':record['task_group'],
                        'answers':record['answers'],'prediction':prediction,'strict_v3_score':strict,
                        'supplemental_score':changed,'changed':strict!=changed})
    strict=aggregate_scores(strict_rows);supplemental=aggregate_scores(supplemental_rows)
    _assert_metrics_equal(strict,summary)
    for key,path in paths.items():
        if path.read_bytes()!=contents[key]:raise ValueError(f'Cached artifact changed during audit: {key}')
    return {'directory':str(directory),'variant':variant,'records':len(records),
            'identity':{'checkpoint_sha256':protocol['checkpoint_sha256'],'manifest_sha256':manifest_sha,
                        'parent_protocol_sha256':parent_sha,'parent_protocol_file_sha256':sha256_bytes(contents['parent_protocol']),
                        'parent_summary_sha256':sha256_bytes(contents['parent_summary']),
                        'raw_predictions_sha256':sha256_bytes(contents['raw_predictions']),
                        'transformed_manifest_sha256':ablation['transformed_manifest_sha256'],
                        'parent_source_sha256':parent_sources,'checkpoint_rehashed_by_supplement':False},
            'strict_v3_original':strict,'supplemental_gqa_lmms':supplemental,
            'supplemental_minus_strict_macro_points':supplemental['group_macro_score']-strict['group_macro_score'],
            'gqa':gqa_diagnostics(details),'rows':details,
            'limits':['Supplemental cached-score sensitivity, not replacement of frozen strict-v3 or an official full benchmark score.',
                      'Checkpoint identity is inherited from the validated parent generation receipt; no model is loaded or rehashed here.',
                      'No prompts, model weights, predictions, references, or original reports are modified.']}
