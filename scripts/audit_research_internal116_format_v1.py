#!/usr/bin/env python3
"""Post-hoc format sensitivity on unchanged internal116 model predictions.

Does not change checkpoint selection, labels, questions or training. VQA single
references receive the existing VQA normalizer; explicit ScienceQA choice-only
answers accept a bare letter or anchored 'Answer: X'. No explanation mining.
"""
from __future__ import annotations
import hashlib,json,os,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
from src.eval.research_metrics import aggregate_scores,normalize_vqa,score_prediction
from src.eval.vlm_benchmark_runtime_v1 import atomic,sha
DATA=ROOT/'datasets/research/budget18-en-v1'
OUT=ROOT/'research_results/budget18_internal_format_audit_v1'
SELECTION_SHA='325a06fa729330994e816e5692611b7c5f03c9463bcc1d8304a229a23e4a6b8a'


def explicit_science_choices(record):
    if record['source'].get('config')!='scienceqa':return None
    parts=record['question'].rsplit('\nChoices:\n',1)
    if len(parts)!=2 or not parts[1].rstrip().endswith('Answer with the letter.'):
        raise ValueError('ScienceQA format audit requires explicit choice-only prompt')
    body=parts[1].rstrip().removesuffix('Answer with the letter.').rstrip()
    options=re.findall(r'^([A-Z])\.\s+(.+)$',body,re.M)
    if not 2<=len(options)<=26 or [x[0] for x in options]!=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(options)]):
        raise ValueError('Ambiguous/noncontiguous prompt choices')
    return {letter for letter,_ in options}


def anchored_letter(text,choices):
    match=re.fullmatch(r'\s*(?:Answer\s*:\s*)?([A-Z])\s*\.?\s*',str(text),re.I)
    letter=match.group(1).upper() if match else None
    return letter if letter in choices else None


def canonical_score(record,prediction):
    if record['task_group']=='general_vqa' and record['scorer']=='exact_match':
        if len(record['answers'])!=1:raise ValueError('Only single teacher references supported')
        return float(normalize_vqa(prediction)==normalize_vqa(record['answers'][0])),'single_reference_vqa_normalized'
    if record['task_group']=='knowledge_reasoning' and record['scorer']=='exact_match':
        choices=explicit_science_choices(record)
        if choices is not None:
            gold=anchored_letter(record['answers'][0],choices)
            if gold is None:raise ValueError('Unambiguous explicit reference letter required')
            return float(anchored_letter(prediction,choices)==gold),'explicit_prompt_mc_letter_accuracy'
    return score_prediction(record,prediction),record['scorer']


def rescore(records,rows):
    if len(rows)!=116 or len(records)!=116:raise ValueError('Exactly116 unchanged predictions required')
    output=[];strict=[]
    for record,row in zip(records,rows):
        if any(row.get(k)!=record[k] for k in ('record_id','task_group','scorer','answers','source')):
            raise ValueError('Prediction selection metadata/order differs')
        old=score_prediction(record,row['prediction'])
        if abs(old-row['score'])>1e-12:raise ValueError('Original score cannot be reproduced')
        new,method=canonical_score(record,row['prediction'])
        strict.append(dict(row,score=old))
        output.append(dict(row,score=new,scorer=method,strict_scorer=record['scorer'],strict_score=old,
                           format_score_changed=new!=old))
    return dict(strict=aggregate_scores(strict),canonical=aggregate_scores(output),
        changed_records=sum(row['format_score_changed'] for row in output)),output


def main():
    records=[json.loads(x) for x in (DATA/'internal_selection_116.jsonl').read_text().splitlines()]
    digest=hashlib.sha256(json.dumps(records,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if digest!=SELECTION_SHA:raise ValueError('Frozen selection changed')
    plan=json.loads((ROOT/'configs/research/budget18_plan_v1.json').read_bytes());sources=[]
    for job in plan['jobs']:
        if job['stage']!='SFT':continue
        run=ROOT/'runs/research'/job['id'];path=run/'selection/best.predictions.jsonl';protocol=run/'selection/best.protocol.json'
        if not (run/'completed.json').exists() or not path.exists():continue
        value=json.loads(protocol.read_bytes())
        if value['selection_manifest_sha256']!=SELECTION_SHA:raise ValueError('Research selection protocol differs')
        sources.append(dict(id=job['id'],path=path,protocol=protocol,kind='research_checkpoint',
                            config_sha256=sha(run/'resolved_config.json')))
    baseline=ROOT/'research_results/qwen25_native_int4_internal116_v1'
    if (baseline/'summary.json').exists():
        summary=json.loads((baseline/'summary.json').read_bytes());path=baseline/'predictions.jsonl'
        if (summary.get('status')!='completed' or summary['records']!=116 or sha(path)!=summary['predictions_sha256']
            or summary['selection_protocol']['selection_manifest_sha256']!=SELECTION_SHA):raise ValueError('Baseline completion proof changed')
        sources.append(dict(id='Qwen2.5-VL-7B-Instruct__native_NF4_INT4',path=path,protocol=baseline/'protocol.json',kind='original_native_VLM'))
    report=dict(status='passed_posthoc_format_sensitivity',created_unix=time.time(),official_benchmark_score=False,
        original_checkpoint_selection_unchanged=True,selection_manifest_semantic_sha256=SELECTION_SHA,
        scope='same116 single-teacher records; general VQA normalization and explicitly prompted ScienceQA letter format only; no generations changed',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/eval/research_metrics.py',DATA/'internal_selection_116.jsonl']},rows=[])
    OUT.mkdir(parents=True,exist_ok=True)
    for item in sources:
        rows=[json.loads(x) for x in item['path'].read_text().splitlines()];metrics,output=rescore(records,rows)
        target=OUT/(item['id']+'.predictions.jsonl')
        temporary=target.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output));os.replace(temporary,target)
        report['rows'].append(dict(id=item['id'],kind=item['kind'],source_predictions=dict(path=str(item['path']),sha256=sha(item['path'])),
            source_protocol=dict(path=str(item['protocol']),sha256=sha(item['protocol'])),rescored_predictions=dict(path=str(target),sha256=sha(target)),**metrics))
    atomic(OUT/'report.json',report)
    lines=['# Внутренняя выборка116: чувствительность к формату ответа\n\n',
        'Дополнительный анализ после просмотра ответов Qwen. Исторические checkpoint selection и scores сохранены. '
        'Один teacher reference на вопрос; это не официальные benchmark scores. '
        'Нормализация VQA применяется только к general_vqa; ScienceQA допускает только целую строку `C` / `Answer: C` '
        'с явным вариантом в исходном вопросе. OCR ANLS и ChartQA не изменены.\n\n',
        '| Модель / эксперимент | Строгая116 | Формат канонизирован | Ответов с изменённым score |\n',
        '|---|---:|---:|---:|\n']
    for row in report['rows']:
        lines.append(f"| {row['id']} | {row['strict']['group_macro_score']:.2f} | {row['canonical']['group_macro_score']:.2f} | {row['changed_records']} |\n")
    lines.append('\nПодробные группы, SHA исходных ответов и отдельные пересчитанные журналы: [report.json](report.json). '
                 'Этот анализ оценивает чувствительность к формату; выборка мала, результаты одного seed.\n')
    (OUT/'report.md').write_text(''.join(lines))
    print(json.dumps(dict(status=report['status'],models=len(report['rows']),output=str(OUT)),ensure_ascii=False))


if __name__=='__main__':main()
