#!/usr/bin/env python3
"""Decode already generated prefixes up to Giga's native assistant turn end."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ['CUDA_VISIBLE_DEVICES']=''
from src.eval.research_giga_eot_v1 import terminated_prefix,EOT_ID,REVISION
from src.model.research_giga_loading_v2 import tokenizer_proxy
from src.eval.research_metrics import aggregate_scores,score_prediction
from src.eval.vlm_benchmark_runtime_v1 import atomic,sha
from scripts.audit_research_internal116_format_v1 import canonical_score


def main():
    from huggingface_hub import snapshot_download
    snapshot=snapshot_download('ai-sage/GigaChat3.1-10B-A1.8B-bf16',revision=REVISION,local_files_only=True)
    tokenizer=tokenizer_proxy().from_pretrained(snapshot,local_files_only=True)
    selected=[json.loads(x) for x in (ROOT/'datasets/research/budget18-en-v1/internal_selection_116.jsonl').read_text().splitlines()]
    output=ROOT/'research_results/budget18_giga_native_eot_audit_v1';output.mkdir(exist_ok=True)
    report=dict(status='completed_CPU_native_turn_end_prefix_audit',created_unix=time.time(),official_benchmark_score=False,
        source_sha256={p:sha(ROOT/p) for p in ['src/eval/research_giga_eot_v1.py','scripts/audit_research_giga_eot_v1.py']},
        scope='same previously generated token IDs; truncate only at native assistant turn token128001; no new GPU generation; original selection preserved',rows=[])
    for run in sorted((ROOT/'runs/research').glob('giga31__*__SFT__budget18_s42')):
        path=run/'selection/best.predictions.jsonl'
        if not (run/'completed.json').exists() or not path.exists():continue
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        if len(rows)!=116:raise ValueError('Exact116 prediction population required')
        native=[];canonical=[];trimmed=0
        for record,row in zip(selected,rows):
            if record['record_id']!=row['record_id']:raise ValueError('Selection order differs')
            ids=row['generation']['generated_token_ids'];decoded=tokenizer.decode(ids,skip_special_tokens=True).strip()
            if decoded!=row['prediction']:raise ValueError('Raw generated IDs do not reproduce saved prediction')
            prefix=terminated_prefix(ids,tokenizer);trimmed+=int(prefix['native_turn_end_found'])
            value=dict(row,prediction=prefix['prediction'],score=score_prediction(record,prefix['prediction']))
            native.append(value);score,method=canonical_score(record,value['prediction']);canonical.append(dict(value,score=score,scorer=method))
        result=dict(id=run.name,source_predictions_sha256=sha(path),original=aggregate_scores(rows),
            native_turn_end=aggregate_scores(native),native_turn_end_and_format=aggregate_scores(canonical),
            records_with_native_turn_end=trimmed,total_records=116)
        (output/(run.name+'.predictions.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in native))
        report['rows'].append(result)
    atomic(output/'report.json',report)
    lines=['# GigaChat: проверка конца реплики\n\n','Дополнительный анализ сохранённых токенов; исходные результаты не изменены. '
        'Native chat template заканчивает ответ токеном128001, generation_config задаёт толькоEOS2. '
        'Обрезание производится исключительно на native границе реплики.\n\n',
        '| Запуск | Исходная116 | Native конец реплики | Native конец + формат | Реплик с границей |\n','|---|---:|---:|---:|---:|\n']
    for r in report['rows']:lines.append(f"| {r['id']} | {r['original']['group_macro_score']:.2f} | {r['native_turn_end']['group_macro_score']:.2f} | {r['native_turn_end_and_format']['group_macro_score']:.2f} | {r['records_with_native_turn_end']}/116 |\n")
    (output/'report.md').write_text(''.join(lines));print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
