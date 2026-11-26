"""Strict CPU scoring for the immutable primary_text_en_v1 input suite.

This module never generates model text/logits and never imports model packages.
Official IFEval is injected by an independently source-verified adapter.
"""
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re

OWNER='research_primary_text_scoring_v1'
SUITE_PROTOCOL_SHA256='6115e9e2e6e4f22efc5b758a3236e58b2ea6b6e313802e06f77c8244560e373b'
CHOICES=('A','B','C','D')
TOKEN_IDS={'smallthinker4':dict(zip(CHOICES,(32,33,34,35))),
    'lfm25':dict(zip(CHOICES,(41,42,43,44))),
    'giga31':dict(zip(CHOICES,(35,36,37,38))),
    'gptoss20':dict(zip(CHOICES,(32,33,34,35)))}
TASK_COUNTS={'gsm8k':512,'mmlu':912,'ifeval':256}


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def sha(raw): return hashlib.sha256(raw if isinstance(raw,bytes) else raw.encode()).hexdigest()
def object_sha(value): return sha(canonical(value))
def hex64(value):return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None
def read_jsonl(path):
    # str.splitlines() splits valid Unicode separators inside JSON strings.
    with Path(path).open(encoding='utf-8',newline='\n') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gsm_final_decimal(response):
    """One final anchored #### line, finite Decimal; no number fallback."""
    if not isinstance(response,str): raise ValueError('Raw response must be text')
    lines=response.split('\n'); found=[]
    for index,line in enumerate(lines):
        match=re.fullmatch(r'####[ \t]*([^\r\n]+?)[ \t]*\r?',line)
        if match: found.append((index,match.group(1)))
    if len(found)!=1 or any(line.strip() for line in lines[found[0][0]+1:]):
        return None
    try: result=Decimal(found[0][1].strip().replace(',',''))
    except InvalidOperation: return None
    return result if result.is_finite() else None


def score_gsm(record,response):
    prediction=gsm_final_decimal(response)
    try: reference=Decimal(record['reference']['numeric_answer'])
    except (InvalidOperation,KeyError,TypeError) as error: raise ValueError('Invalid bound GSM8K reference') from error
    if not reference.is_finite(): raise ValueError('GSM8K reference must be finite')
    return {'record_id':record['record_id'],'task':'gsm8k',
        'parsed_final_decimal':str(prediction) if prediction is not None else None,
        'valid_final_line':prediction is not None,'correct':prediction is not None and prediction==reference}


def score_mmlu(record,supplied,model_id,prompt_census):
    """Consume log likelihoods only, with explicit prompt/choice boundaries."""
    ll=supplied.get('candidate_log_likelihoods')
    if not isinstance(ll,dict) or set(ll)!=set(CHOICES):
        raise ValueError('Exactly A/B/C/D supplied log likelihoods are required')
    if any(type(ll[c]) not in (int,float) or not math.isfinite(ll[c]) or ll[c]>0 for c in CHOICES):
        raise ValueError('Candidate log likelihoods must be finite, nonpositive real numbers')
    if model_id not in TOKEN_IDS: raise ValueError('Unknown pinned model tokenizer')
    expected={c:[TOKEN_IDS[model_id][c]] for c in CHOICES}
    if supplied.get('candidate_token_ids')!=expected:
        raise ValueError('Candidate tokens differ from pinned native A/B/C/D boundaries')
    lengths=prompt_census.get('candidate_continuation_token_lengths',{})
    if lengths!=dict.fromkeys(CHOICES,1): raise ValueError('Frozen census does not prove one-token stable choices')
    prefixes=supplied.get('candidate_prefix_token_ids_sha256')
    if prefixes!=dict.fromkeys(CHOICES,prompt_census['prompt_token_ids_sha256']):
        raise ValueError('Choice likelihood prefix differs from the actual native prompt boundary')
    if supplied.get('length_normalization') is not False:
        raise ValueError('Preregistered MMLU uses summed conditional log likelihood, no length normalization')
    winner=max(CHOICES,key=lambda c:ll[c])  # stable A,B,C,D tie rule
    reference=record['reference']['answer_letter']
    if reference not in CHOICES: raise ValueError('Invalid MMLU reference')
    return {'record_id':record['record_id'],'task':'mmlu','subject':record['subject'],
        'selected_choice':winner,'correct':winner==reference,'candidate_log_likelihoods':dict(ll),
        'tie_choices':[c for c in CHOICES if ll[c]==ll[winner]]}


def aggregate_gsm(rows):
    if not rows: raise ValueError('GSM8K denominator may not be empty')
    return {'records':len(rows),'correct':sum(r['correct'] for r in rows),
        'accuracy':sum(r['correct'] for r in rows)/len(rows),
        'invalid_final_line':sum(not r['valid_final_line'] for r in rows)}


def aggregate_mmlu(rows,*,require_full_suite=True):
    subjects=defaultdict(list)
    for row in rows: subjects[row['subject']].append(row)
    if not rows: raise ValueError('MMLU denominator may not be empty')
    if require_full_suite and (len(subjects)!=57 or len(rows)!=912 or any(len(v)!=16 for v in subjects.values())):
        raise ValueError('MMLU full suite requires all57 subjects with16 rows each')
    per={s:{'records':len(v),'correct':sum(r['correct'] for r in v),
        'accuracy':sum(r['correct'] for r in v)/len(v)} for s,v in sorted(subjects.items())}
    return {'records':len(rows),'subjects':len(subjects),'per_subject':per,
        'subject_macro_accuracy':sum(v['accuracy'] for v in per.values())/len(per),
        'all_record_accuracy':sum(r['correct'] for r in rows)/len(rows),
        'tie_break':'first maximum in A,B,C,D','length_normalization':False}


def aggregate_ifeval(rows):
    if not rows: raise ValueError('IFEval denominator may not be empty')
    result={'records':len(rows)}
    for mode in ('strict','loose'):
        if any(len(row[mode]['follow_instruction_list'])!=len(row['instruction_ids']) for row in rows):
            raise ValueError('IFEval may not skip instruction denominators')
        flags=[flag for row in rows for flag in row[mode]['follow_instruction_list']]
        if not flags or any(type(flag) is not bool for flag in flags):
            raise ValueError('IFEval must return every instruction as a Boolean')
        if any(row[mode]['follow_all_instructions']!=all(row[mode]['follow_instruction_list']) for row in rows):
            raise ValueError('IFEval prompt/instruction flags are inconsistent')
        result[mode]={'prompt_accuracy':sum(row[mode]['follow_all_instructions'] for row in rows)/len(rows),
            'instruction_accuracy':sum(flags)/len(flags),'instructions':len(flags),
            'instructions_correct':sum(flags),'prompts_correct':sum(row[mode]['follow_all_instructions'] for row in rows)}
    return result


def load_suite(path):
    path=Path(path); protocol=json.loads((path/'protocol.json').read_text())
    if (protocol.get('name')!='primary_text_en_v1' or protocol.get('schema_version')!=1 or
            sha((path/'protocol.json').read_bytes())!=SUITE_PROTOCOL_SHA256):
        raise ValueError('Expected the fixed primary_text_en_v1 suite')
    for name,expected in protocol['files_sha256'].items():
        target=path/name
        if Path(name).name!=name or target.is_symlink() or sha(target.read_bytes())!=expected:
            raise ValueError('Immutable suite member hash differs: '+name)
    records=read_jsonl(path/'records.jsonl')
    if Counter(r['task'] for r in records)!=Counter(TASK_COUNTS) or len({r['record_id'] for r in records})!=1680:
        raise ValueError('Primary text suite row coverage differs')
    census=json.loads((path/'tokenizer_census.json').read_text())
    expected_ids=[r['record_id'] for r in records]
    if (len(census['models'])!=4 or {m['model_id'] for m in census['models']}!=set(TOKEN_IDS) or
            any([r['record_id'] for r in model['records']]!=expected_ids for model in census['models'])):
        raise ValueError('Pinned four-backbone tokenizer census coverage differs')
    root=Path(__file__).resolve().parents[2]
    for name,expected in protocol['source_sha256'].items():
        source=(root/name).resolve(strict=True)
        if not source.is_relative_to(root) or sha(source.read_bytes())!=expected:
            raise ValueError('Original frozen suite preparation source changed')
    return {'path':str(path.resolve()),'protocol':protocol,'records':records,
        'protocol_sha256':sha((path/'protocol.json').read_bytes()),
        'manifest_sha256':sha((path/'records.jsonl').read_bytes()),
        'census':{m['model_id']:m for m in census['models']}}


def validate_checkpoint_descriptor(identity,stage,model,*,owner=None,step=None):
    """Validate supplied provenance structure, without reading model weights.

    The future model producer remains responsible for actually loading the
    identified bytes. Scoring supplied log likelihoods cannot prove that load.
    """
    if not isinstance(identity,dict):raise ValueError('Checkpoint descriptor is missing')
    if stage=='baseline_before_vision':
        if (identity.get('format')!='pinned_baseline' or identity.get('repo')!=model['repo'] or
                identity.get('revision')!=model['revision'] or not hex64(identity.get('weight_inventory_sha256'))):
            raise ValueError('Baseline must bind pinned repo/revision and weight inventory SHA256')
        return
    if not isinstance(owner,str) or not owner or type(step) is not int or step<=0:
        raise ValueError('Learned checkpoint requires managed owner and positive training step')
    kind=identity.get('format');members=identity.get('members')
    if (kind not in ('legacy_single_file','bundle_v1') or identity.get('components_fully_hashed') is not True or
            not hex64(identity.get('sha256')) or not isinstance(members,list) or not members):
        raise ValueError('Learned checkpoint requires a fully identified managed compound artifact')
    for row in members:
        if (not isinstance(row,dict) or not isinstance(row.get('path'),str) or not Path(row['path']).is_absolute() or
                type(row.get('bytes')) is not int or row['bytes']<=0 or
                type(row.get('mtime_ns')) is not int or row['mtime_ns']<=0 or not hex64(row.get('sha256')) or
                not isinstance(row.get('relative_path'),str) or Path(row['relative_path']).name!=row['relative_path'] or
                not isinstance(row.get('role'),str)):
            raise ValueError('Checkpoint component path/size/stat/content identity differs')
    if (len({r['path'] for r in members})!=len(members) or
            len({r['relative_path'] for r in members})!=len(members) or
            identity.get('primary_path')!=members[0]['path'] or
            identity.get('primary_bytes')!=members[0]['bytes'] or
            identity.get('primary_mtime_ns')!=members[0]['mtime_ns'] or
            identity.get('total_bytes')!=sum(r['bytes'] for r in members)):
        raise ValueError('Compound checkpoint member coverage/stat totals differ')
    if kind=='legacy_single_file':
        if len(members)!=1 or members[0]['role']!='state' or identity['sha256']!=members[0]['sha256']:
            raise ValueError('Legacy checkpoint content/member descriptor differs')
    else:
        if [r['role'] for r in members] not in (['manifest','model'],['manifest','model','optimizer']):
            raise ValueError('Bundle component roles/coverage differ')
        stable={'format':kind,'members':[{k:r[k] for k in ('role','relative_path','bytes','sha256')} for r in members]}
        expected=sha(b'RESEARCH_CHECKPOINT_ARTIFACT_V1\0'+json.dumps(stable,sort_keys=True,separators=(',',':')).encode())
        if identity['sha256']!=expected:raise ValueError('Compound checkpoint digest does not bind all member bytes')


def validate_response_rows(suite,responses,producer,*,tasks):
    if not tasks or set(tasks)-set(TASK_COUNTS): raise ValueError('Unknown/empty scoring task scope')
    if (producer.get('schema_version')!=1 or producer.get('owner')!='research_primary_text_responses_v1' or
            producer.get('suite_protocol_sha256')!=suite['protocol_sha256'] or
            producer.get('manifest_sha256')!=suite['manifest_sha256'] or
            producer.get('tasks')!=list(tasks)):
        raise ValueError('Raw response producer/source/manifest/task scope differs')
    model=producer.get('model_id')
    if model not in suite['census'] or producer.get('model_revision')!=suite['census'][model]['revision']:
        raise ValueError('Raw responses use another pinned backbone/tokenizer revision')
    if producer.get('stage') not in ('baseline_before_vision','A0','A1','SFT'):
        raise ValueError('Baseline/A0/A1/SFT identity must be explicit')
    if producer.get('evidence_scope') not in ('model_responses','synthetic_scorer_validation'):
        raise ValueError('Raw producer must distinguish actual model responses from synthetic scorer validation')
    if not isinstance(producer.get('numerical_profile'),dict) or not producer['numerical_profile'].get('weights'):
        raise ValueError('BF16/NF4 numerical profile must be explicit')
    if producer['numerical_profile']['weights'] not in ('bf16','nf4','fp32'):
        raise ValueError('Unknown explicit weight precision profile')
    if not isinstance(producer.get('checkpoint_identity'),dict) or not producer['checkpoint_identity']:
        raise ValueError('Pinned baseline/managed learned checkpoint identity is required')
    if producer['evidence_scope']=='model_responses':
        validate_checkpoint_descriptor(producer['checkpoint_identity'],producer['stage'],suite['census'][model],
            owner=producer.get('checkpoint_owner'),step=producer.get('training_step'))
    if producer.get('generation')!=suite['protocol']['evaluation']['generation']:
        raise ValueError('Raw producer generation budgets/greedy policy differ')
    expected=[r for r in suite['records'] if r['task'] in tasks]
    if len(responses)!=len(expected) or [r.get('record_id') for r in responses]!=[r['record_id'] for r in expected]:
        raise ValueError('Raw responses must cover exactly the selected IDs in original order')
    census={r['record_id']:r for r in suite['census'][model]['records']}
    contract_sha=object_sha(producer)
    for response,record in zip(responses,expected):
        current=census[record['record_id']]
        fields={'task':record['task'],'source':record['source'],'input_text_sha256':sha(record['input_text']),
            'response_contract_sha256':contract_sha,'prompt_sha256':current['prompt_sha256'],
            'prompt_token_ids_sha256':current['prompt_token_ids_sha256'],'prompt_tokens':current['prompt_tokens']}
        if any(response.get(k)!=v for k,v in fields.items()):
            raise ValueError('Raw response source/input/native prompt/contract identity differs')
        if set(response)&{'reference','gold','answer','answers','kwargs','instruction_id_list','numeric_answer','answer_index'}:
            raise ValueError('Evaluator gold/reference fields must not enter raw model response rows')
        if record['task']!='mmlu' and not isinstance(response.get('response'),str):
            raise ValueError('Raw generated response text missing')
    return expected,census,contract_sha


def score_rows(suite,responses,producer,*,tasks,ifeval=None):
    expected,census,contract_sha=validate_response_rows(suite,responses,producer,tasks=tasks)
    if 'ifeval' in tasks and ifeval is None:
        raise RuntimeError('Official IFEval dependency/resource adapter is unavailable; no instruction may be skipped')
    scored=[]
    for record,raw in zip(expected,responses):
        if record['task']=='gsm8k': score=score_gsm(record,raw['response'])
        elif record['task']=='mmlu': score=score_mmlu(record,raw,producer['model_id'],census[record['record_id']])
        else: score=ifeval.score(record,raw['response'])
        score.update(raw_response_sha256=object_sha(raw),source=record['source'],response_contract_sha256=contract_sha)
        scored.append(score)
    metrics={}
    for task in tasks:
        selected=[r for r in scored if r['task']==task]
        metrics[task]={'gsm8k':aggregate_gsm,'mmlu':aggregate_mmlu,'ifeval':aggregate_ifeval}[task](selected)
    return scored,{'schema_version':1,'owner':OWNER,'status':'scored_supplied_responses_only',
        'suite_protocol_sha256':suite['protocol_sha256'],'manifest_sha256':suite['manifest_sha256'],
        'response_contract_sha256':contract_sha,'response_producer':producer,
        'records':len(scored),'tasks':metrics,
        'model_generation_performed':False,'model_logits_invented':False,
        'checkpoint_validation_scope':'supplied descriptor/source metadata only; scoring does not reload or prove model execution',
        'primary_completion_claim':False,'cross_task_composite':None}
