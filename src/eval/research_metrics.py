"""Auditable local VQA/document/chart/MCQ scorers, returning fractions in [0,1].

VQA normalization tables derive from Aishwarya Agrawal's BSD-licensed official
VQA evaluator; full notice/provenance is in third_party/vqa_normalization/.
ANLS and chart rules follow google-research/pix2struct/metrics.py (Apache-2.0).
No answers are mined from explanations and no external judge is invoked.
"""
from collections import defaultdict
from pathlib import Path
import json
import math
import re

_ROOT=Path(__file__).resolve().parents[2]
_TABLES=json.loads((_ROOT/'third_party/vqa_normalization/tables.json').read_text())
_PERIOD=re.compile(r'(?!<=\d)(\.)(?!\d)')
_COMMA=re.compile(r'(\d)(,)(\d)')


def normalize_vqa(answer):
    original=str(answer).replace('\n',' ').replace('\t',' ').strip()
    result=original
    numeric_comma=_COMMA.search(original) is not None
    for punctuation in _TABLES['punct']:
        replacement='' if numeric_comma or punctuation+' ' in original or ' '+punctuation in original else ' '
        result=result.replace(punctuation,replacement)
    result=_PERIOD.sub('',result)
    words=[]
    for word in result.lower().split():
        word=_TABLES['manualMap'].get(word,word)
        if word not in _TABLES['articles']:
            words.append(_TABLES['contractions'].get(word,word))
    return ' '.join(words)


def vqa_accuracy(prediction,answers,*,protocol='textvqa_evalai'):
    """Official leave-one-annotator-out consensus, not min(count/3,1).

    Exactly ten independently supplied references are required. The legacy
    VQAv2 script normalizes punctuation/digits only for non-unanimous answers;
    TextVQA's EvalAI protocol normalizes every answer. The distinction is explicit.
    """
    if len(answers)!=10:raise ValueError('VQA consensus requires all ten human answers')
    references=[str(a).replace('\n',' ').replace('\t',' ').strip() for a in answers]
    predicted=str(prediction).replace('\n',' ').replace('\t',' ').strip()
    if protocol not in {'textvqa_evalai','vqav2_official_legacy'}:raise ValueError('Unknown VQA normalization protocol')
    if protocol=='textvqa_evalai' or len(set(references))>1:
        references=[normalize_vqa(a) for a in references];predicted=normalize_vqa(predicted)
    count=sum(a==predicted for a in references)
    return sum(min(1.,(count-int(answer==predicted))/3.) for answer in references)/10.


def levenshtein(left,right):
    if len(left)<len(right):left,right=right,left
    previous=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        current=[i]
        for j,b in enumerate(right,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+int(a!=b)))
        previous=current
    return previous[-1]


def anls(prediction,answers):
    predicted=str(prediction).strip().lower()
    scores=[]
    for reference in answers:
        reference=str(reference).strip().lower()
        length=max(len(reference),len(predicted))
        if not length:
            scores.append(1.);continue
        distance=levenshtein(reference,predicted)/length
        scores.append(1.-distance if distance<.5 else 0.)
    return max(scores,default=0.)


def chart_relaxed(prediction,answers,tolerance=.05):
    def number(text):
        try:
            value=float(text[:-1])/100. if text.endswith('%') else float(text)
            return value if math.isfinite(value) else None
        except ValueError:return None
    prediction=str(prediction).strip()
    p=number(prediction)
    for reference in answers:
        reference=str(reference).strip();r=number(reference)
        # This retains pix2struct's exact-text fallback for a zero target.
        if r is not None and r!=0 and p is not None:
            if abs(p-r)/abs(r)<=tolerance:return 1.
        elif prediction.lower()==reference.lower():return 1.
    return 0.


def mcq_answer(prediction,choices):
    value=str(prediction).strip()
    match=re.fullmatch(r'(?:the\s+answer\s+is\s+|answer\s*:\s*)?[\(\[]?([A-Z])[\)\].:]?(?:\s*[^A-Za-z0-9]*)',value,re.I)
    if match:
        index=ord(match.group(1).upper())-ord('A')
        if 0<=index<len(choices):return chr(ord('A')+index)
    matches=[i for i,c in enumerate(choices) if str(c).strip().casefold()==value.casefold()]
    return chr(ord('A')+matches[0]) if len(matches)==1 else None


def score_prediction(record,prediction):
    references=record.get('answers')
    if not isinstance(references,list) or not references:raise ValueError('Evaluation references missing')
    scorer=record['scorer']
    if scorer=='vqa_accuracy':
        return vqa_accuracy(prediction,references,protocol=record.get('scoring_protocol','textvqa_evalai'))
    if scorer=='anls':return anls(prediction,references)
    if scorer=='chart_relaxed':return chart_relaxed(prediction,references)
    if scorer=='exact_match':return float(any(str(prediction).strip().casefold()==str(a).strip().casefold() for a in references))
    if scorer=='mcq_accuracy':
        choices=record.get('choices',[])
        if not choices:raise ValueError('MCQ evaluation requires choices')
        answer=mcq_answer(prediction,choices)
        return float(answer is not None and answer in [str(a).upper().strip() for a in references])
    raise ValueError(f'Unsupported scorer: {scorer}')


def aggregate_scores(rows,*,require_four_groups=True):
    """Equal weight per task group; retain per-benchmark/metric denominators."""
    groups=defaultdict(list);benchmarks=defaultdict(list)
    ids=set()
    for row in rows:
        if row['record_id'] in ids:raise ValueError('Duplicate scored evaluation record')
        ids.add(row['record_id'])
        value=float(row['score'])
        if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid evaluation score')
        groups[row['task_group']].append(value)
        source=row.get('source',{})
        name=row.get('benchmark') or source.get('benchmark') or source.get('repo','unknown')
        benchmarks[(name,row['scorer'])].append(value)
    expected={'general_vqa','ocr_documents','charts_diagrams','knowledge_reasoning'}
    complete=set(groups)==expected
    if require_four_groups and not complete:raise ValueError('All four task groups are required for selection')
    by_group={key:{'score':100*sum(values)/len(values),'records':len(values)} for key,values in groups.items()}
    return {'records':len(rows),'complete_four_group_coverage':complete,
            'group_macro_score':sum(v['score'] for v in by_group.values())/len(by_group) if by_group else None,
            'groups':by_group,'benchmarks':{name+'::'+metric:{'score':100*sum(v)/len(v),'records':len(v)} for (name,metric),v in benchmarks.items()},
            'scale':'0-100','aggregation':'equal_task_group_macro'}
