#!/usr/bin/env python3
"""CPU-only table of the six new common-pool configurations."""
import csv
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def main():
    plan=read(ROOT/'configs/research/common_pool_plan_v1.json')
    if not plan:return
    state=read(ROOT/'runs/research/common_pool_v1/queue_state.json')
    table=[]
    for job in plan['jobs']:
        if job['phase']!='DET':continue
        arm=Path(job['output']).parent
        row=dict(model=job['model_key'],method=job['method'],arm=job['arm'])
        for phase in ('SFT1','SFT2','DET'):
            run=arm/phase
            metrics=read(run/'evaluation/after_internal116.metrics.json')
            proof=read(run/'native_reload_proof.json')
            row['VQA116_'+phase]=metrics.get('group_macro_score')
            row['native_proof_'+phase]=proof.get('status')=='passed'
        test=read(arm/'DET/evaluation/heldout_detection_test.metrics.json')
        row['detection_test_F1_IoU50']=test.get('primary_score')
        row['negative_query_accuracy']=test.get('negative_query_accuracy')
        row['status']='completed' if row['native_proof_DET'] else state.get('jobs',{}).get(job['id'],{}).get('status','pending')
        table.append(row)
    output=ROOT/'research_results/common_pool_v1';output.mkdir(parents=True,exist_ok=True)
    report=dict(updated_UTC=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),rows=table,
                complete=sum(row['native_proof_DET'] for row in table),total=6,
                scope='VQA116 internal teacher-reference macro; detection positive-query macro F1@IoU0.5, not official benchmark/AP',
                provisional_policy='Scores without the corresponding passed native proof are provisional')
    temporary=output/'current_metrics.json.tmp';temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(output/'current_metrics.json')
    with (output/'comparison.csv.tmp').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (output/'comparison.csv.tmp').replace(output/'comparison.csv')
    def score(value,passed):return '—' if value is None else f'{value:.2f}'+('' if passed else '*')
    lines=['# R4/R5: общий пул E+8', '', report['updated_UTC'], '',
           f"Независимо проверено {report['complete']}/6 финальных конфигураций. Qwen encoder, native k=4, без modality masks.", '',
           '| Модель | Метод | VQA116 SFT1 | VQA116 SFT2 | VQA116 final | Detection test F1 |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in table:
        lines.append(f"| {row['model']} | {row['method']} | "+' | '.join(
            score(row['VQA116_'+phase],row['native_proof_'+phase]) for phase in ('SFT1','SFT2','DET'))+
            ' | '+score(row['detection_test_F1_IoU50'],row['native_proof_DET'])+' |')
    lines+=['','* — предварительная оценка: независимый native proof ещё не пройден.','',report['scope'], '']
    temporary=output/'current_metrics.md.tmp';temporary.write_text('\n'.join(lines));temporary.replace(output/'current_metrics.md')


if __name__=='__main__':main()
