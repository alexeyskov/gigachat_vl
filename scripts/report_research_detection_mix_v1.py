#!/usr/bin/env python3
"""Small CPU-only status/report, never loads weights or changes scores."""
import json,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_results/detection_mix_v1'

def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError):return default

def main():
 plan=read(ROOT/'configs/research/detection_mix_plan_v1.json')
 if not plan:return
 state=read(ROOT/'runs/research/detection_mix_v1/queue_state.json',{})
 values=[]
 for job in plan['jobs']:
  run=Path(job['output']);proof=read(run/'native_reload_proof.json',{})
  values.append(dict(id=job['id'],model=job['model_key'],vision=job['vision_key'],method=job['method'],
   status=state.get('jobs',{}).get(job['id'],{}).get('status','waiting'),
   before_OD=read(run/'evaluation/before_detection_validation.metrics.json'),
   after_OD=read(run/'evaluation/after_detection_validation.metrics.json'),
   heldout_OD_test=read(run/'evaluation/heldout_detection_test.metrics.json'),
   before_internal116=read(run/'evaluation/before_internal116.metrics.json'),
   after_internal116=read(run/'evaluation/after_internal116.metrics.json'),
   native_proof_passed=proof.get('status')=='passed',source_retirement=read(run/'source_retired.json')))
 OUT.mkdir(parents=True,exist_ok=True)
 report=dict(updated_UTC=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),status=state.get('status','waiting_for_previous_matrix'),
  completed=sum(r['native_proof_passed'] for r in values),total=18,
  coordinate_system='xyxy_normalized_1000',metric='positive-only macro per-query F1@IoU0.5; not official COCO AP',rows=values)
 tmp=OUT/'current_metrics.json.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');tmp.replace(OUT/'current_metrics.json')
 def score(value,key='primary_score'):return '—' if not value else f"{value[key]:.2f}"
 lines=['# Дополнительный финальный SFT с примесью детекции','',report['updated_UTC'],'',
  f"Подтверждено: {report['completed']}/18. Состояние: {report['status']}.",'',
  '450 детекционных +2550 прежних мультимодальных +529 текстовых примеров; один проход,56 updates. Изображения/проектор frozen.','',
  '| Вариант | OD val до | OD val после | OD test | VQA116 до | VQA116 после | GPU proof |',
  '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
 for r in values:lines.append(f"| {r['id']} | {score(r['before_OD'])} | {score(r['after_OD'])} | {score(r['heldout_OD_test'])} | {score(r['before_internal116'],'group_macro_score')} | {score(r['after_internal116'],'group_macro_score')} | {'passed' if r['native_proof_passed'] else r['status']} |")
 lines+=['','OD: F1 по положительным запросам при IoU≥0.5,0–100. JSON validity, P/R, F1@0.75 и negative-query accuracy доступны в JSON. Это проверка поднабора, не официальный COCO mAP. VQA116 остаётся внутренней teacher-reference macro,0–100.','',
  'Новый source checkpoint сохраняется до оценки. Затем независимая fresh GPU загрузка точно восстанавливает все обученные тензоры, воспроизводит8 detection и8 VQA ответов, оценивает64 новых detection test. Исходные результаты предыдущих18 конфигураций не переписываются.','']
 diagnostic=read(OUT/'geometry_collapse_diagnostic.json')
 if diagnostic:
  lines += ['','## Проверка геометрии ответов','',
   f"Отдельная диагностика от {diagnostic['updated_UTC']} охватывает {diagnostic['completed_variants']} завершённых вариантов; это снимок на указанное время.",
   'Обнаружена склонность выдавать рамки почти на всё изображение. Сохранённые ответы повторно оценены с прежними предсказанными классами/числом объектов, но с заменой координат на[0,0,1000,1000]. Эта проверка не является отдельной моделью или независимым baseline.',
   'Подробности: [geometry_collapse_diagnostic.md](geometry_collapse_diagnostic.md). Высокая валидность JSON и ненулевой F1 сами по себе не подтверждают полезную локализацию. Обучение, ответы и выбор checkpoint эта диагностика не меняет.','']
 tmp=OUT/'current_metrics.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(OUT/'current_metrics.md')
 print(json.dumps(dict(completed=report['completed'],total=18,status=report['status'])))

if __name__=='__main__':main()
