#!/usr/bin/env python3
"""Read-only aggregation of verified capacity runs and retained full controls."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_results/capacity_eval_v5'
MODELS={'smallthinker4':'SmallThinker','lfm25':'LFM2.5','giga31':'GigaChat3.1'}
VISIONS={'siglip2':'SigLIP2','qwen25':'Qwen2.5-VL encoder'}

def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError):return default

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 state=read(ROOT/'runs/research/additive_visual_v1/queue_state.json',{})
 path=Path(state['active_plan_path']);plan=read(path)
 if path.name!='fixed_capacity_eval_plan_v5.json' or sha(path)!=state['plan_sha256']:raise ValueError('Exact active V5 plan required')
 records=[]
 for job in plan['jobs']:
  if job['stage']!='SFT':continue
  run=Path(job['output']);proof=read(run/'native_reload_proof.json');done=read(run/'completed.json')
  if not proof or proof.get('status')!='passed':continue
  config=read(run/'resolved_config.json')
  if (proof['records']!=116 or proof['mismatches'] or done['steps']!=56 or done['records']!=3529
      or proof['architecture']!=config['additive_execution'] or config['additive_execution']['branch']!=job['branch']
      or not (run/'last/state.pt').is_file()):raise ValueError('Complete bound native-replay proof required')
  for source,digest in config['capacity_evaluation_v5']['source_sha256'].items():
   if sha(ROOT/source)!=digest:raise ValueError('V5 source binding changed')
  records.append(dict(id=job['id'],arm=job['arm'],model=job['backend'],encoder=job['vision'],
   blocks=job['branch']['blocks'],learned_masks=job['learned_masks'],training_pass=job['training_pass'],
   score=proof['score'],checkpoint=str(run/'last/state.pt'),proof=str(run/'native_reload_proof.json'),
   checkpoint_sha256=proof['checkpoint_sha256'],MM_heldout_ce=(read(run/'multimodal_validation_ce.json') or {}).get('ce')))
 arms=[]
 for arm in plan['arms']:
  candidates=sorted((r for r in records if r['arm']==arm['id']),key=lambda r:r['training_pass'])
  if not candidates:continue
  best=max(candidates,key=lambda r:(r['score'],-r['training_pass']))
  arms.append(dict(**arm,passes=candidates,best=best,last=candidates[-1],
   second_minus_first=candidates[1]['score']-candidates[0]['score'] if len(candidates)==2 else None))
 old_eot=read(ROOT/'research_results/budget18_giga_native_eot_audit_v1/report.json',{})
 new_eot=read(OUT/'full_controls_native_eot/report.json',{})
 corrections={r['id']:r for report in (old_eot,new_eot) for r in report.get('rows',[])}
 controls=[]
 for control in plan['controls']:
  run=Path(control['run']);metrics=read(run/'selection/last.metrics.json')
  row=dict(id=control['id'],run=str(run),training_completed=(run/'completed.json').is_file(),
   historical_score=(metrics or {}).get('group_macro_score'),native_turn_prefix_score=None)
  correction=corrections.get(control['id'])
  if correction and sha(run/'selection/best.predictions.jsonl')==correction['source_predictions_sha256']:
   row['native_turn_prefix_score']=correction['native_turn_end']['group_macro_score']
   row['correction_scope']='CPU rescore of saved generated prefix up to native assistant EOT; original score preserved'
  controls.append(row)
 best_by_model={model:max((a['best'] for a in arms if a['backend']==model),key=lambda r:r['score']) for model in MODELS if any(a['backend']==model for a in arms)}
 changes=[a['second_minus_first'] for a in arms if a['second_minus_first'] is not None]
 legacy=read(ROOT/'runs/research/budget18_queue_state_v1.json',{})
 value=dict(updated_UTC=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),status=state['status'],
  scope='four fixed visual blocks per MoE layer; NF4 QLoRA; one seed42; internal116 teacher-reference macro,0-100; not an official benchmark',
  confirmed_SFT_stages=len(records),expected_SFT_stages=25,confirmed_four_block_arms_with_two_passes=sum(len(a['passes'])==2 for a in arms),
  active_revision_failures=sum(len(r.get('failures',[])) for r in state['jobs'].values()),
  second_pass_improved=sum(x>1e-9 for x in changes),second_pass_declined=sum(x< -1e-9 for x in changes),
  best_four_block_by_model=best_by_model,arms=arms,diagnostic_two_block_control=[r for r in records if r['blocks']==2],
  full_controls=controls,full_controls_trained=sum(r['training_completed'] for r in controls),
  active_full_control_job=legacy.get('active_job') if state['status']=='additive_training_complete_controls_pending' else None)
 OUT.mkdir(parents=True,exist_ok=True)
 for suffix in ('json','md'):
  source=OUT/f'current_metrics.{suffix}';first=OUT/f'first_confirmed_metrics.{suffix}'
  if source.exists() and not first.exists():first.write_bytes(source.read_bytes())
 atomic(OUT/'current_metrics.json',value)
 lines=['# Подтверждённые результаты четырёхблочной серии','',value['updated_UTC'],'',
  f"SFT-проверки: {len(records)}/25; B4 конфигурации с двумя проходами: {value['confirmed_four_block_arms_with_two_passes']}/12; full-train контролей обучено: {value['full_controls_trained']}/6.",
  f"Новых сбоев в v5: {value['active_revision_failures']}. Каждый принятый SFT имеет независимый native GPU replay116 без несовпадений.",'',
  'Основная метрика: внутренняя teacher-reference macro116,0–100. Один seed42. Это не официальный benchmark.','',
  '| Модель | Encoder | Метод | Проход1 | Проход2 | Лучший проход |','| --- | --- | --- | ---: | ---: | ---: |']
 for arm in arms:
  scores={r['training_pass']:r['score'] for r in arm['passes']}
  lines.append(f"| {MODELS[arm['backend']]} | {VISIONS[arm['vision']]} | {'R3, masks' if arm['learned_masks'] else 'R1, без masks'} | {scores.get(1,float('nan')):.4f} | {scores.get(2,float('nan')):.4f} | {arm['best']['training_pass']} |")
 lines+=['',f"Второй проход улучшил primary metric у {value['second_pass_improved']}/12 вариантов, ухудшил у {value['second_pass_declined']}/12. Лучшие и последние веса сохранены раздельно, когда они относятся к разным проходам.",'',
  '## Контроль двух блоков','', 'SmallThinker + Qwen encoder + R1: исходный B2/P1=21.5517; B2/P2=23.2759. Для B4 той же пары P1=22.4138, P2=18.9655. В этой паре добавление блоков и продолжение обучения не дали устойчивого преимущества.','',
  '## Full-train контроли','', '| Контроль | Обучен | Исторический score | CPU native EOT prefix score |','| --- | --- | ---: | ---: |']
 for r in controls:
  raw='—' if r['historical_score'] is None else f"{r['historical_score']:.4f}"
  corrected='—' if r['native_turn_prefix_score'] is None else f"{r['native_turn_prefix_score']:.4f}"
  lines.append(f"| {r['id']} | {'да' if r['training_completed'] else 'нет'} | {raw} | {corrected} |")
 lines+=['','Giga full-train historical generation использовала EOS2 и могла продолжать текст после native assistant end128001. CPU prefix audit сохраняет исходный score и пересчитывает только уже сгенерированный префикс до128001. Его следует явно отделять от исходного протокола. В новых B4 этот marker применяется во время генерации. R2 имеют один SFT-проход, поэтому их нельзя выдавать за одинаковый двухпроходный бюджет.','',
  f"Текущий full-control job: {value['active_full_control_job'] or 'см. ADDITIVE_STATUS.md'}.",
  'Старый сбой v4:33 неудачных попытки,8.9 GPU-часа training compute; подробности в `incident.json`. Все опубликованные выше SFT относятся к исправленному v5 и проверены независимо.','']
 tmp=OUT/'current_metrics.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(OUT/'current_metrics.md')
 print(json.dumps(dict(confirmed_SFT_stages=len(records),B4_arms_with_two_passes=value['confirmed_four_block_arms_with_two_passes'],new_failures=value['active_revision_failures'],second_pass_improved=value['second_pass_improved'],controls_trained=value['full_controls_trained'])))

if __name__=='__main__':main()
