#!/usr/bin/env python3
"""CPU-only live status for the additive series, separate from legacy reports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.research_storage import _atomic_json
BASE=ROOT/'runs/research/additive_visual_v1'
OUT=ROOT/'research_results/additive_visual_v1'

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def last(path):
    try:
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-32768));rows=f.read().splitlines()
        for line in reversed(rows):
            try:return json.loads(line)
            except ValueError:pass
    except OSError:pass
    return None

def collect_jobs(plan):
    jobs=[]
    for declaration in plan.get('jobs',[]):
        run=Path(declaration['output']);config=read(run/'resolved_config.json',{})
        if 'additive_execution' not in config:continue
        jobs.append(dict(run=run.name,output=str(run),arm=declaration.get('arm'),
            training_pass=config.get('training_pass',1),blocks=config['additive_execution']['branch']['blocks'],
            stage=config['stage'],last=last(run/'metrics.jsonl'),completed=read(run/'completed.json'),
            multimodal_ce=read(run/'multimodal_validation_ce.json'),reload_proof=read(run/'native_reload_proof.json'),
            reload_progress=read(run/'reload_progress.json')))
    return jobs

def active_phase(queue,legacy,detection=None):
    if detection:
        name=detection.get('active_job')
        root=ROOT/'runs/research/detection_mix_v1'/str(name)
        row=detection.get('jobs',{}).get(name,{})
        if detection.get('status')=='completed':return dict(phase='detection_completed',job=None,progress=None)
        progress=read(root/('test_progress.json' if row.get('status')=='native_validation_and_test' else 'evaluation_progress.json'))
        metrics=last(root/'metrics.jsonl')
        if metrics and row.get('status')=='train' and metrics.get('event')=='train':progress=metrics
        return dict(phase='detection_mix',job=name,worker=row.get('status'),progress=progress,status=detection.get('status'))
    if queue.get('status')=='additive_training_complete_controls_pending':
        name=legacy.get('active_job')
        row=legacy.get('jobs',{}).get(name,{})
        if row.get('validate_process',{}).get('status')=='running':
            return dict(phase='full_control_validation',job=name,progress=dict(event='native_GPU_checkpoint_replay',pid=row['validate_process'].get('pid')))
        if row.get('status')=='failed_retry_pending':
            return dict(phase='full_control_retry_wait',job=name,progress=dict(event='retry_pending',retry_after_unix=row.get('retry_after_unix'),error=row.get('error')))
        progress=last(ROOT/'runs/research'/str(name)/'metrics.jsonl') if name else None
        return dict(phase='full_control_tail',job=name,progress=progress)
    if queue.get('status')=='completed':return dict(phase='completed',job=None,progress=None)
    return dict(phase='additive',job=queue.get('active_job'),progress=queue.get('last_progress'))

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    last_summary=0
    while True:
        queue=read(BASE/'queue_state.json',{})
        plan=read(Path(queue.get('active_plan_path',ROOT/'configs/research/additive_visual_plan_v1.json')), {})
        total=plan.get('SFT_stages_total',plan.get('new_scientific_configurations',24))
        jobs=collect_jobs(plan)
        gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
        services=subprocess.run(['supervisorctl','status'],capture_output=True,text=True,timeout=10)
        filtered=[line for line in services.stdout.splitlines() if line.startswith(('research-additive-','research-fixed-','research-capacity-','research-detection-','research-control-tail-memory-'))]
        legacy=read(ROOT/'runs/research/budget18_queue_state_v1.json',{})
        oldjob=legacy.get('active_job');oldprogress=last(ROOT/'runs/research'/str(oldjob)/'metrics.jsonl') if oldjob else None
        detection_plan=read(ROOT/'configs/research/detection_mix_plan_v1.json')
        detection=read(ROOT/'runs/research/detection_mix_v1/queue_state.json',{})
        active=active_phase(queue,legacy,detection)
        detection_report=read(ROOT/'research_results/detection_mix_v1/current_metrics.json',{})
        controls=[dict(run=c['id'],completed=bool(read(Path(c['run'])/'completed.json'))) for c in plan.get('controls',[])]
        trained=sum(j['stage']=='SFT' and bool(j['completed']) for j in jobs)
        replayed=sum(j['stage']=='SFT' and (j['reload_proof'] or {}).get('status')=='passed' for j in jobs)
        failures=[dict(job=name,count=len(row['failures']),last=row['failures'][-1],status=row.get('status'))
            for name,row in queue.get('jobs',{}).items() if row.get('failures')]
        pending_evaluation=[j['run'] for j in jobs if (Path(j['output'])/'recovery/model.pt').is_file() and not j['completed']]
        arms_complete=sum(all(any(j['run']==name and (j['reload_proof'] or {}).get('status')=='passed'
            for j in jobs) for name in arm['stages'] if name.endswith('__SFT')) for arm in plan.get('arms',[]))
        value=dict(updated_UTC=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
            new_SFT_trained=trained,new_SFT_replayed=replayed,new_SFT_total=total,
            four_block_arms_completed=arms_complete,four_block_arms_total=len(plan.get('arms',[])),
            failures=failures,trained_pending_evaluation=pending_evaluation,
            active=active,full_controls=controls,full_controls_trained=sum(c['completed'] for c in controls),
            queue=queue,jobs=jobs,gpu=gpu.stdout.strip(),services=filtered,
            legacy_current_job=oldjob,legacy_progress=oldprogress,
            detection_mix=dict(queue=detection,report=detection_report,scheduled=bool(detection_plan)))
        _atomic_json(OUT/'live_status.json',value)
        lines=['# Новая серия: дополнительные визуальные блоки','',value['updated_UTC'],'',
            f'Этапов SFT обучено: {trained}/{total}; независимо восстановлено и проверено: {replayed}/{total}.',
            'Плюс 6 native full-finetune контролей. Старые common-pool результаты учитываются отдельно.','',
            'GPU: '+gpu.stdout.strip(),'', '```',*filtered,'```','']
        if active['phase']=='full_control_tail':
            progress=active['progress'] or {}
            lines[8:8]=[f"Сейчас выполняется последний full-train контроль: `{active['job']}`; событие {progress.get('event','—')}, шаг {progress.get('step','—')}/{progress.get('total_steps','—')}.",
                f"Full-train контролей обучено: {sum(c['completed'] for c in controls)}/6.",'']
        elif active['phase']=='full_control_validation':
            lines[8:8]=[f"Обучение full-train завершено. Сейчас GPU проверяет чекпоинт `{active['job']}`, PID {(active['progress'] or {}).get('pid','—')}.",'']
        elif active['phase']=='full_control_retry_wait':
            lines[8:8]=[f"Обучение завершено, но проверка `{active['job']}` упала; очередь ожидает повтор. GPU может простаивать. Причина: {(active['progress'] or {}).get('error','—')}.",'']
        if detection_plan:
            progress=active.get('progress') or {}
            lines += ['', '## Финальное дообучение с детекцией', '',
                '450 detection +2550 MM replay +529 text; один дополнительный проход для каждой из18 конфигураций.',
                f"Независимо проверено: {detection_report.get('completed',0)}/18; состояние: {detection.get('status','ожидание естественного завершения прежней матрицы')}."]
            if active['phase']=='detection_mix':
                lines += [f"Активный запуск: `{active['job']}`; процесс {active.get('worker','—')}; событие {progress.get('event','—')}; шаг {progress.get('step','—')}/{progress.get('total_steps','—')}; оценка {progress.get('completed',progress.get('completed_records','—'))}/{progress.get('total',progress.get('total_records','—'))}."]
            lines += ['Сводка: `research_results/detection_mix_v1/current_metrics.md`; протокол: `DETECTION_PIPELINE.md`.', '']
        lines += ['| Предыдущие запуски | Стадия | Последнее событие | Прогресс |', '| --- | --- | --- | --- |']
        for job in jobs:
            row=(job['reload_progress'] if job['completed'] and not job['reload_proof'] and job['reload_progress'] else job['last']) or {}
            step=str(row.get('step','—'))+'/'+str(row.get('total_steps','—'))
            if row.get('event')=='internal_selection_record':step=f"{row.get('completed_records')}/{row.get('total_records')} eval"
            lines.append(f"| {job['run']} | {job['stage']} | {row.get('event','—')} | {step} |")
        if failures:
            lines += ['',f'Ошибок в активной очереди: {sum(r["count"] for r in failures)}. Повтор не является завершённым результатом.']
            for failure in failures:lines.append(f'- {failure["job"]}: {failure["count"]} сбоев; {failure["status"]}; {failure["last"]["error"]}')
        if pending_evaluation:
            lines += ['', 'Обученные веса уже сохранены, ожидается оценка: '+', '.join(pending_evaluation)+'.']
        if plan.get('arms'):
            lines += ['',f'Четырёхблочных конфигураций с двумя проверенными проходами: {arms_complete}/12.',
                'Всего 25 SFT-этапов: 12 конфигураций × 2 прохода и один дополнительный проход двухблочного контроля.',
                '', '| Запуск | Macro116 | MM heldout CE |','| --- | ---: | ---: |']
            for job in jobs:
                proof=job['reload_proof'] or {}
                if proof.get('status')=='passed':
                    ce=job['multimodal_ce'] or {}
                    lines.append(f"| {job['run']} | {proof['score']:.4f} | {ce.get('ce','—')} |")
        lines += ['', 'Подробности: `research_results/additive_visual_v1/live_status.json`; протокол: `RESEARCH_PIPELINE.md`.',
                  'Активны только фиксированные блоки, с масками и без. Отдельный роутер исключён. Caption loss не является точностью на VQA.','']
        target=ROOT/'ADDITIVE_STATUS.md';temp=target.with_suffix('.tmp');temp.write_text('\n'.join(lines));temp.replace(target)
        if time.monotonic()-last_summary>120:
            try:
                scripts=[]
                if replayed==total==25:scripts+=['scripts/refresh_research_full_control_eot_v5.py','scripts/summarize_research_capacity_v5.py']
                if detection_plan:scripts+=['scripts/report_research_detection_mix_v1.py']
                for script in scripts:
                    subprocess.run([sys.executable,script],cwd=ROOT,check=True,timeout=45,capture_output=True,text=True)
            except (subprocess.SubprocessError,OSError) as error:
                print(json.dumps(dict(event='summary_refresh_error',error=str(error))),flush=True)
            last_summary=time.monotonic()
        print(json.dumps(dict(event='additive_status',time=value['updated_UTC'],SFT_trained=trained,SFT_replayed=replayed,gpu=value['gpu'])),flush=True)
        time.sleep(30)

if __name__=='__main__':run()
