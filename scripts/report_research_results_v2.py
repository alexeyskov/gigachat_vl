#!/usr/bin/env python3
"""Canonical current results; keep the superseded 512-question study separate."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
NAMES={'smallthinker4':'SmallThinker','lfm25':'LFM2.5','giga31':'GigaChat3.1'}
VISIONS={'siglip2':'SigLIP2','qwen25':'Qwen2.5-VL'}


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def metric(value):
    if value is None:return '—'
    if not math.isfinite(value) or not 0<=value<=100:raise ValueError('Invalid metric on0..100 scale')
    return f'{value:.2f}'


def render(final,common,state,baseline):
    if len(final['rows'])!=18 or len({row['id'] for row in final['rows']})!=18:
        raise ValueError('Canonical final table requires all18 unique configurations')
    now=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())
    lines=['# Результаты исследований LLM → VLM','',f'Обновлено: {now}. Один training seed: 42.','',
        '## Завершённая финальная серия R1/R2/R3','',
        f"Независимую проверку прошли **{final['completed']}/18** конфигураций. Здесь результаты финального SFT с примесью детекции, а не старой отменённой серии.",
        '', 'VQA116 — внутренняя macro-метрика по четырём группам с одним teacher reference. Detection — macro F1@IoU0.5 по положительным запросам test64. Шкала 0–100; это не официальная benchmark accuracy и не COCO AP.', '',
        '| Модель | Энкодер | Метод | VQA116 до | VQA116 после | Detection test F1 | Проверка |',
        '| --- | --- | --- | ---: | ---: | ---: | --- |']
    order={'smallthinker4':0,'lfm25':1,'giga31':2}
    for row in sorted(final['rows'],key=lambda r:(order[r['model']],r['vision']!='siglip2',r['method'])):
        before=(row.get('before_internal116') or {}).get('group_macro_score')
        after=(row.get('after_internal116') or {}).get('group_macro_score')
        test=(row.get('heldout_OD_test') or {}).get('primary_score')
        passed=row['native_proof_passed']
        if passed and any(v is None for v in (before,after,test)):
            raise ValueError('Passed final row has missing published metrics: '+row['id'])
        lines.append(f"| {NAMES[row['model']]} | {VISIONS[row['vision']]} | {row['method']} | {metric(before)} | {metric(after)} | {metric(test)} | {'passed' if passed else row['status']} |")
    lines += ['', 'R1: QLoRA + четыре фиксированных визуальных Transformer-блока на MoE-слой. R2: full-train языковой базы без новых блоков. R3: QLoRA + четыре блока с обучаемыми text/visual gates. Vision encoder и caption-aligned MLP заморожены.', '',
        '«До» и «после» относятся к финальному проходу с детекцией. Независимый proof подтверждает точное восстановление learned tensors, воспроизведение 8 VQA и 8 detection ответов; отдельно оценены все 64 detection test запроса. Полный независимый replay всех 116 VQA этим не заявляется.', '']
    if baseline and baseline.get('status')=='completed':
        lines += [f"Исходная **Qwen2.5-VL-7B NF4 INT4** на том же internal116: **{metric(baseline['group_macro_score'])}**. Это отдельный reference, а не одна из 18 обученных конфигураций. [Исходные результаты](research_results/qwen25_native_int4_internal116_v1/summary.json).",'']
    lines += ['У всех 18 вариантов accuracy отрицательных detection-запросов равна 0/8. Ранее на первых 14 вариантах обнаружены грубые рамки почти на всё изображение; ненулевой F1 не подтверждает полезную локализацию. [Диагностика](research_results/detection_mix_v1/geometry_collapse_diagnostic.md).','',
        'Данные этой таблицы: [JSON](research_results/detection_mix_v1/current_metrics.json), [итоговый отчёт](research_results/detection_mix_v1/FINAL_RESULTS.md).','',
        '## Новые эксперименты R4/R5','']
    if common:
        complete=common.get('complete',0)
        lines += [f"Финально проверено **{complete}/6** конфигураций; очередь `{state.get('status','pending')}`. Во всех случаях frozen Qwen encoder, восемь новых нативных FFN-экспертов на MoE-слой и общий top-k=4, без modality masks.", '',
            'R4 обучает новый router полностью; R5 сохраняет исходный router и обучает LoRA-расширение. Языковая база в обоих случаях NF4 + QLoRA. SFT1/SFT2 — промежуточные этапы, final — после detection SFT.','',
            '| Модель | Метод | Статус | VQA116 SFT1 | VQA116 SFT2 | VQA116 final | Detection test F1 |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: |']
        for row in common['rows']:
            stage_status='завершён' if row['native_proof_DET'] else 'в очереди'
            active=state.get('active_job','')
            if not row['native_proof_DET'] and active.startswith(row['arm']+'__'):
                current=state.get('jobs',{}).get(active,{})
                stage_status=active.rsplit('__',1)[-1]+' / '+current.get('status','pending')
            def score(key,passed):
                value=row.get(key)
                return metric(value)+('*' if value is not None and not passed else '')
            values=[score('VQA116_'+phase,row['native_proof_'+phase]) for phase in ('SFT1','SFT2','DET')]
            values.append(score('detection_test_F1_IoU50',row['native_proof_DET']))
            lines.append(f"| {NAMES[row['model']]} | {row['method']} | {stage_status} | "+' | '.join(values)+' |')
        lines += ['', '«—» означает, что оценка нового варианта ещё не получена; «*» — предварительное число до passed native proof. Эти пропуски не относятся к завершённым 18 вариантам первой таблицы.', '',
                  '[Текущий статус](RESEARCH_COMMON_POOL_STATUS.md) · [JSON](research_results/common_pool_v1/current_metrics.json) · [CSV](research_results/common_pool_v1/comparison.csv).','']
    else:lines += ['Новые результаты ещё не опубликованы.','']
    lines += ['## Историческая диагностика на 512 вопросах','',
        'Прежняя первая таблица относилась к отдельному screening-протоколу image/shuffle/blank/text-only. В нём полностью подтверждена только одна строка; остальные ветви были заменены последующими экспериментами. Её пропуски не означали, что текущие 18 моделей не оценены.', '',
        '[Архив прежнего RESEARCH_RESULTS.md](RESEARCH_RESULTS_LEGACY_512.md). Числа VQA116 не подставляются в столбцы старой 512-вопросной диагностики. Условия, модели и метрики разных серий сохраняются отдельно.','']
    return '\n'.join(lines)


def main():
    from scripts.report_research_detection_mix_v1 import main as refresh_final
    from scripts.report_research_common_pool_v1 import main as refresh_common
    refresh_final();refresh_common()
    final=read(ROOT/'research_results/detection_mix_v1/current_metrics.json')
    if not final:return
    common=read(ROOT/'research_results/common_pool_v1/current_metrics.json')
    state=read(ROOT/'runs/research/common_pool_v1/queue_state.json',{})
    baseline=read(ROOT/'research_results/qwen25_native_int4_internal116_v1/summary.json')
    value=render(final,common,state,baseline)
    target=ROOT/'RESEARCH_RESULTS.md';archive=ROOT/'RESEARCH_RESULTS_LEGACY_512.md'
    if target.exists() and not archive.exists() and target.read_text().startswith('# Результаты серии18'):
        with archive.open('xb') as handle:handle.write(target.read_bytes())
    temporary=target.with_suffix('.tmp');temporary.write_text(value);temporary.replace(target)
    print(json.dumps(dict(report='RESEARCH_RESULTS.md',final_rows=18,final_proofs=final['completed'],new_complete=(common or {}).get('complete',0))))


if __name__=='__main__':main()
