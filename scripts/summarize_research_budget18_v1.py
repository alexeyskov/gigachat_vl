#!/usr/bin/env python3
"""Publish the18-cell results table from fully verified queue artifacts only."""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
from src.utils.research_storage import _atomic_json
PLAN=ROOT/'configs/research/budget18_plan_v1.json'
STATE=ROOT/'runs/research/budget18_queue_state_v1.json'
PLAN_SHA='ec5333b8db9ec234d37e283d2299df0de3daaa4d0a1f4da0ad2a6601b9adb9cc'


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path):return {'path':str(path),'sha256':sha(path),'bytes':Path(path).stat().st_size}
def argument(argv,key):return argv[argv.index(key)+1]


def live_phase(record):
    for key,phase in [('train_process','training'),('validate_process','validating'),('eval_process','evaluating')]:
        process=record.get(key,{})
        if process.get('status')!='running':continue
        try:argv=(Path('/proc')/str(process['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except (OSError,KeyError,UnicodeError):continue
        if argv==process.get('command'):return phase
    return None


def collect():
    if sha(PLAN)!=PLAN_SHA:raise ValueError('Exact18-cell intent required')
    plan=read(PLAN);state=read(STATE)
    digest=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if state.get('spec_sha256')!=digest:raise ValueError('Queue and intent differ')
    jobs=[j for j in plan['jobs'] if j['stage']=='SFT']
    if len(jobs)!=18:raise ValueError('Full18-cell matrix required')
    results=[]
    for job in jobs:
        record=state['jobs'].get(job['id'],{});run=ROOT/argument(job['argv'],'--output')
        status=record.get('status','pending')
        if status!='completed' and state.get('active_job')==job['id']:
            status=live_phase(record) or status
        result={'id':job['id'],'model':job['backend'],'encoder':job['encoder'],
                'recipe':job['recipe'],'status':status,
                'training_completed':bool(record.get('training_completed'))}
        if result['status']=='completed':
            config=read(run/'resolved_config.json');completed=read(run/'completed.json')
            if (completed.get('steps')!=56 or completed.get('records')!=3529 or config.get('seed')!=42
                    or config.get('recipe')!=job['recipe'] or config.get('epochs')!=1):
                raise ValueError('Completed science stage count/source differs')
            validation=record['checkpoint_validation'];replay=read(validation['report'])
            if (validation.get('status')!='completed' or replay.get('status')!='passed'
                    or replay.get('mode')!='full_gpu_replay' or sha(validation['report'])!=validation['report_sha256']):
                raise ValueError('Completed row lacks current full checkpoint replay')
            paired=record['paired_summary'];summary=read(paired['path'])
            if sha(paired['path'])!=paired['sha256'] or summary.get('status')!='completed':
                raise ValueError('Paired intervention proof changed')
            checkpoint_sha=record['evaluation']['checkpoint_sha256'];scores={};bindings={}
            for mode,key in [('none','evaluation'),('shuffle','grounding_shuffle'),
                             ('blank','grounding_blank'),('text_only','grounding_text_only')]:
                proof=record[key];out=Path(proof['output']);protocol=read(out/'protocol.json');metrics=read(out/'summary.json')
                if (proof.get('status')!='completed' or metrics.get('records')!=512
                        or metrics.get('complete_manifest') is not True
                        or metrics.get('checkpoint_sha256')!=checkpoint_sha
                        or protocol.get('checkpoint_sha256')!=checkpoint_sha
                        or protocol.get('image_ablation',{}).get('variant','none')!=mode):
                    raise ValueError('All four full512 views on one checkpoint required')
                with (out/'predictions.jsonl').open() as f:rows=[json.loads(line) for line in f]
                if len(rows)!=512 or len({r['record_id'] for r in rows})!=512:
                    raise ValueError('Unique complete raw predictions required')
                scores[mode]=metrics['group_macro_score']
                bindings[mode]={n:bind(out/n) for n in ('protocol.json','summary.json','predictions.jsonl')}
            # The queue already fully hashed/model-replayed this exact object;
            # tables check its current stat without warming GB of model pages.
            for entry in replay['checkpoints']:
                p=Path(entry['checkpoint_path']);st=p.stat()
                if st.st_size!=entry['checkpoint_size_bytes'] or st.st_mtime_ns!=entry['checkpoint_mtime_ns']:
                    raise ValueError('Validated checkpoint physical identity changed')
            events=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            text=next(e for e in reversed(events) if e.get('event')=='text_validation')
            training=[e for e in events if e.get('event')=='train']
            result.update(scores=scores,text34_ce=text['text_only_heldout_ce'],
                stage_seconds=completed['elapsed_seconds'],peak_train_VRAM_GiB=max(e['peak_vram_gb'] for e in training),
                checkpoint_sha256=checkpoint_sha,checkpoint_replay=bind(validation['report']),
                paired_summary=paired,paired_intervals=summary['paired_image_group_bootstrap'],
                evaluation_artifacts=bindings,config=bind(run/'resolved_config.json'),disk_reserve_gib=config['reserve_gb'])
            audit=ROOT/'research_results/budget18_parameter_updates_v1'/job['id']/'report.json'
            if audit.exists():
                report=read(audit)
                if report.get('status')!='passed' or Path(report['checkpoint']) not in {Path(x['checkpoint_path']) for x in replay['checkpoints']}:
                    raise ValueError('Parameter evidence does not refer to selected checkpoint')
                result['parameter_update_evidence']=bind(audit)
                result['parameter_update_groups']=report['common_tensor_groups']
        results.append(result)
    return {'updated_unix':time.time(),'scope':'budget18_only_seed42','matrix_cells':18,
            'fully_verified':sum(r['status']=='completed' for r in results),'trained':sum(r['training_completed'] for r in results),
            'plan':bind(PLAN),'rows':results,'official_full_benchmark_or_seed_variance_claimed':False,
            'checkpoint_body_hashes_recomputed_by_this_table':False}


def native_qwen_reference():
    root=ROOT/'research_results/qwen25_native_int4_internal116_v1'
    result={'path':str(root),'separate_reference_not_one_of_18':True,'records':0,'status':'waiting_for_compatible_training_window'}
    if (root/'progress.json').exists():
        progress=read(root/'progress.json');result.update(records=progress['completed_records'],status='in_progress',progress=progress)
    if (root/'summary.json').exists():
        summary=read(root/'summary.json')
        if summary.get('status')!='completed' or summary.get('records')!=116 or sha(root/'predictions.jsonl')!=summary['predictions_sha256']:
            raise ValueError('Native Qwen reference completion proof differs')
        result.update(status='completed',records=116,summary=summary)
        audit=ROOT/'research_results/budget18_internal_format_audit_v1/report.json'
        if audit.exists():
            rows=read(audit)['rows'];item=next((r for r in rows if r['id']=='Qwen2.5-VL-7B-Instruct__native_NF4_INT4'),None)
            if item and item['source_predictions']['sha256']==summary['predictions_sha256']:
                result['posthoc_format_score']=item['canonical']['group_macro_score']
                result['posthoc_format_report']=str(audit)
    return result


def publish():
    if (ROOT/'research_results/qwen25_native_int4_internal116_v1/summary.json').exists():
        from scripts.audit_research_internal116_format_v1 import main as audit_internal_format
        audit_internal_format()
    result=collect();supplemental_audit(result);gate_censuses(result);resource_census(result);_atomic_json(ROOT/'research_results/budget18_results.json',result)
    names={'smallthinker4':'SmallThinker','lfm25':'LFM2.5','giga31':'GigaChat3.1'}
    recipes={'1b':'R1 QLoRA+experts','4f':'R2 full','1c':'R3 QLoRA+experts+masks'}
    lines=['# Результаты серии18\n\n',f"Полностью проверено: **{result['fully_verified']}/18**. Обучено: **{result['trained']}/18**. Один seed42.\n\n",
        'Здесь только новая сокращённая серия. Скоры — group macro на512 диагностических вопросах (8×64), не полные официальные benchmark scores. Сохранены строгие исторические метрики; нормализованный GQA-аудит рассматривается отдельно.\n\n',
        '| Модель | Энкодер | Подход | Статус | Image | Shuffle | Blank | Text-only | Text CE34 |\n',
        '|---|---|---|---|---:|---:|---:|---:|---:|\n']
    for row in result['rows']:
        scores=row.get('scores',{});values=[f"{scores[k]:.2f}" if k in scores else '—' for k in ('none','shuffle','blank','text_only')]
        ce=f"{row['text34_ce']:.3f}" if 'text34_ce' in row else '—'
        lines.append('| '+' | '.join([names[row['model']], 'SigLIP2' if row['encoder']=='siglip2' else 'Qwen729', recipes[row['recipe']],row['status'],*values,ce])+' |\n')
    lines+=['\nДля завершённых строк все четыре512-оценки, уникальные predictions, source bindings, checkpoint replay и paired bootstrap проверены очередью. Таблица повторно сверяет опубликованные файлы и stat чекпоинта, не перечитывая GB tensor bodies.\n',
            '\nЭто короткий screening:4096caption alignment,256caption warmup R1/R3 и3529SFT records, одна эпоха. Насыщение обучения, устойчивость по seeds и превосходство метода этим не доказаны.\n']
    interim=ROOT/'research_results/budget18_interim_grounding_v1/image_vs_shuffle.json'
    if interim.exists():
        lines.append('\nПредварительный парный image-vs-shuffle контроль всех трёх SmallThinker/SigLIP подходов: [таблица и95%CI](research_results/budget18_interim_grounding_v1/image_vs_shuffle.md). У R3 правильные картинки дают15,91 против12,50, разность+3,41 [0,94;6,11]. Это отдельный неполный анализ: blank/text-only R3 ещё ожидаются, поправки на множество конфигураций и вариация по training seeds не оценены. Основной статус R3 остаётся evaluation-pending.\n')
    lines+=['\n## Внутренняя проверка обучения\n\n',
        'Результаты финальной внутренней проверки SFT. Это selection116 и отдельные text34; их нельзя напрямую сравнивать с оценками512 в основной таблице. Внешние проверки выполняются после прохода всех обучений.\n\n',
        '| Модель | Энкодер | Подход | Internal116 | Text CE34 |\n',
        '|---|---|---|---:|---:|\n']
    for row in result['rows']:
        resource=row.get('resources',{})
        if not resource.get('stage_completed') or 'internal116_group_macro_score' not in resource:continue
        ce=f"{resource['text34_heldout_ce']:.3f}" if 'text34_heldout_ce' in resource else '—'
        lines.append('| '+' | '.join([names[row['model']],row['encoder'],recipes[row['recipe']],
            f"{resource['internal116_group_macro_score']:.2f}",ce])+' |\n')
    reference=native_qwen_reference()
    if reference['status']=='completed':
        value=reference['summary']['group_macro_score']
        lines.append(f'\nОтдельный контроль исходной **Qwen2.5-VL-7B-Instruct NF4 INT4** на том же internal116: **{value:.2f}**. Native BF16 vision, Qwen729, greedy B1/64; без нового исследовательского MLP. Это внутренняя macro-метрика, не официальный benchmark score и не одна из18 обучаемых конфигураций. [Полный результат](research_results/qwen25_native_int4_internal116_v1/summary.json).\n')
        if 'posthoc_format_score' in reference:
            lines.append(f"\nПосле дополнительной канонизации формата Qwen получает **{reference['posthoc_format_score']:.2f}**: старое сравнение штрафует `C` против `Answer: C` и `no` против `No.`. На тех же ответах всех завершённых моделей выполнен общий [format audit](research_results/budget18_internal_format_audit_v1/report.md). Это post-hoc чувствительность; исходный отбор и scores сохранены.\n")
    else:lines.append(f"\nОтдельный контроль исходной Qwen2.5-VL-7B NF4: {reference['records']}/116, статус `{reference['status']}`. [Протокол и команды стандартных бенчмарков](BENCHMARKS.md).\n")
    lines+=['\n## Обучение и ресурсы\n\n',
        '| Модель | Энкодер | Подход | SFT, мин | Torch VRAM, ГиБ | RAM cgroup, ГиБ | Минимум диска, ГиБ | Замеров |\n',
        '|---|---|---|---:|---:|---:|---:|---:|\n']
    for row in result['rows']:
        resource=row.get('resources',{})
        if 'native_torch_peak_allocated_GiB' not in resource:continue
        def formatted(key):return f"{resource[key]:.2f}" if key in resource else '—'
        minutes=f"{resource['stage_elapsed_seconds']/60:.2f}" if resource.get('stage_completed') else 'идёт'
        ram=formatted('sampled_cgroup_max_GiB')
        if resource.get('early_training_sample_available') is False:ram+='*'
        lines.append('| '+' | '.join([names[row['model']],row['encoder'],recipes[row['recipe']],minutes,
            formatted('native_torch_peak_allocated_GiB'),ram,
            formatted('sampled_min_free_disk_GiB'),str(resource.get('training_worker_samples',0))])+' |\n')
    lines.append('\nVRAM — нативный peak allocated Torch. RAM и диск — максимум/минимум среди30секундных замеров verified training worker; cgroup включает другие процессы и file cache. Это не непрерывный peak RAM. Звёздочка означает отсутствие ранних замеров: первый получен более60с после первого update. Текущие runs показаны по snapshot. SFT-время включает финальную internal/text проверку и сохранение, исключает загрузку модели. Подробные source-prefix bindings: [resource census](research_results/budget18_resource_census_v1/report.json).\n')
    lines+=['\n## Маски и округление\n\n',
        'Условный пересчёт экспортированных FP32 sigmoid-гейтов в BF16. Он применим к пути, где гейт приводится к BF16 до умножения; это не новый GPU-прогон и не проверка всех runtime dtypes. Предпочтения параметров не доказывают причинной специализации.\n\n',
        '| Модель | Энкодер | Эксперты | Слой×эксперт | Visual>text FP32 | Visual>text BF16 | Text>visual BF16 | Равны BF16 |\n',
        '|---|---|---|---:|---:|---:|---:|---:|\n']
    for row in result['rows']:
        for key,group in row.get('conditional_BF16_gate_groups',{}).items():
            if not key.startswith('SFT_'):continue
            lines.append('| '+' | '.join([names[row['model']],row['encoder'],key.removeprefix('SFT_'),
                *[str(group[k]) for k in ('expert_block_pairs','FP32_visual_greater','BF16_visual_greater','BF16_text_greater','BF16_equal')]])+' |\n')
    for row in result['rows']:
        if 'conditional_BF16_gate_census' in row:
            lines.append(f"\n[{names[row['model']]} / {row['encoder']}: исходные CSV/гейты]({row['gate_census']['path']}) · [условный BF16 census]({row['conditional_BF16_gate_census']['path']}).\n\n")
    lines+=['\n## Дополнительная нормализация GQA\n\n',
        'Те же ответы, с игнорированием регистра и ASCII-пунктуации только для GQA по пинованному lmms-eval. Исходная таблица выше сохранена. Это чувствительность метрики на данной512-подвыборке; генерация и checkpoint selection не меняются.\n\n',
        '| Модель | Энкодер | Подход | Image | Shuffle | Blank | Text-only |\n',
        '|---|---|---|---:|---:|---:|---:|\n']
    for row in result['rows']:
        if 'supplemental_gqa_scores' not in row:continue
        values=[f"{row['supplemental_gqa_scores'][k]:.2f}" for k in ('none','shuffle','blank','text_only')]
        lines.append('| '+' | '.join([names[row['model']],row['encoder'],recipes[row['recipe']],*values])+' |\n')
    lines.append('\nВсе исходные scores повторно пересчитаны CPU-аудитом. Provenance и дополнительные pairedCI: [metric audit](research_results/budget18_metric_audit_v1/index.json).\n')
    for row in result['rows']:
        if row['status']!='completed':continue
        lines.append(f"\n## {names[row['model']]} / {row['encoder']} / {recipes[row['recipe']]}\n\n")
        for mode,entry in row['paired_intervals'].items():
            lo,hi=entry['paired_95_percentile_ci'];delta=entry['original_minus_variant_group_macro_points']
            lines.append(f"- Image−{mode}: {delta:.3f} пункта; paired95% CI [{lo:.3f}; {hi:.3f}].\n")
        lines.append('\nCI отражает вариацию данной подвыборки по image clusters; это не cross-seed uncertainty. Пересечение нуля не подтверждает устойчивый вклад изображения.\n')
        lines.append(f"\nАртефакты: [{row['id']}]({row['paired_summary']['path']}).\n")
        if 'parameter_update_evidence' in row:
            lines.append(f"\nПроверка фактических обновлений и неизменности проектора: [tensor evidence]({row['parameter_update_evidence']['path']}).\n")
        if 'supplemental_gqa_paired' in row:
            lines.append(f"\nPairedCI при дополнительной GQA-нормализации: [supplement]({row['supplemental_gqa_paired']['path']}).\n")
        if 'gate_census' in row:
            lines.append(f"\nИзучаемые маски всех экспертов: [CSV/JSON]({row['gate_census']['path']}) · [heatmap]({row['gate_heatmap']['path']}). Это census параметров, не доказательство причинной специализации.\n")
    # The retired 512-question screening must not replace the current results
    # table after the later detection/common-pool studies have been introduced.
    destination=ROOT/('RESEARCH_RESULTS_LEGACY_512.md' if (ROOT/'configs/research/detection_mix_plan_v1.json').exists() else 'RESEARCH_RESULTS.md')
    temporary=destination.with_suffix('.tmp');temporary.write_text(''.join(lines));os.replace(temporary,destination)
    publish_status(result)
    print(json.dumps({'verified':result['fully_verified'],'trained':result['trained'],'matrix':18}),flush=True)


def supplemental_audit(result):
    from scripts.audit_research_metrics import main as audit
    from scripts.research_grounding_queue import paired_image_bootstrap
    out=ROOT/'research_results/budget18_metric_audit_v1';manifest=ROOT/'datasets/research/budget18-en-v1/heldout_evaluation.jsonl'
    index=audit(['--run-pattern','*__SFT__budget18_s42','--manifest',str(manifest),'--output',str(out)])
    variants={}
    for entry in index['evaluations']:variants.setdefault(entry['run_metadata']['run_id'],{})[entry['variant']]=entry
    records=[json.loads(line) for line in manifest.read_text().splitlines()]
    for row in result['rows']:
        if row['status']!='completed':continue
        views=variants.get(row['id'],{})
        if set(views)!={'none','shuffle','blank','text_only'}:raise ValueError('Verified row lacks four supplemental views')
        bindings={mode:entry['supplemental_rows']['sha256'] for mode,entry in views.items()}
        sources={'analysis_protocol_sha256':index['analysis_protocol_sha256'],'supplemental_row_hashes':bindings,
                 'bootstrap_source_sha256':sha(ROOT/'scripts/research_grounding_queue.py')}
        paired=out/'paired'/(row['id']+'.json')
        paired.parent.mkdir(parents=True,exist_ok=True)
        if not paired.exists() or read(paired).get('source_bindings')!=sources:
            predictions={}
            for mode,entry in views.items():
                p=Path(entry['supplemental_rows']['path'])
                if sha(p)!=bindings[mode]:raise ValueError('Supplemental raw rows changed')
                predictions[mode]=[{'record_id':d['record_id'],'score':d['supplemental_score']} for d in (json.loads(line) for line in p.read_text().splitlines())]
            controls={mode:predictions[mode] for mode in ('shuffle','blank','text_only')}
            intervals=paired_image_bootstrap(predictions['none'],controls,records,replicates=1000,seed=42)
            for mode,entry in intervals.items():
                expected=views['none']['supplemental_gqa_lmms']['group_macro_score']-views[mode]['supplemental_gqa_lmms']['group_macro_score']
                if abs(entry['original_minus_variant_group_macro_points']-expected)>1e-10:raise ValueError('Supplemental paired point differs')
            _atomic_json(paired,{'status':'completed','run':row['id'],'source_bindings':sources,'paired_image_group_bootstrap':intervals,
                'scope':'supplemental GQA case/punctuation sensitivity; original strict-v3 and checkpoint selection unchanged'})
        row['supplemental_gqa_scores']={mode:entry['supplemental_gqa_lmms']['group_macro_score'] for mode,entry in views.items()}
        row['supplemental_gqa_paired']=bind(paired)
    result['supplemental_metric_audit']=bind(out/'index.json')


def gate_censuses(result):
    script=ROOT/'scripts/report_research_budget18_gates_v1.py'
    for row in result['rows']:
        if row['recipe']!='1c' or not row['training_completed']:continue
        out=ROOT/'research_results/budget18_gate_census_v1'/row['id'];receipt=out/'artifacts.json'
        names=('report.json','gates.csv','gates.png','gates.svg','gates.pdf')
        try:
            ready=read(receipt) if receipt.exists() else {}
            valid=(ready.get('source_sha256')==sha(script) and all((out/n).exists() and ready.get('files',{}).get(n)==sha(out/n) for n in names))
            if not valid:
                completed=subprocess.run(['/tmp/research_plot_venv/bin/python',str(script),'--run',str(ROOT/'runs/research'/row['id']),'--output',str(out),'--plot'],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60,check=True)
                _atomic_json(receipt,{'status':'completed_CPU_gate_export','source_sha256':sha(script),'files':{n:sha(out/n) for n in names},'GPU_used':False})
            row['gate_census']=bind(out/'report.json');row['gate_heatmap']=bind(out/'gates.png');row['gate_artifacts']=bind(receipt)
            from scripts.report_research_budget18_effective_gates_v1 import collect as effective_collect
            effective=ROOT/'research_results/budget18_effective_gates_v1'/row['id']/'report.json'
            effective_script=ROOT/'scripts/report_research_budget18_effective_gates_v1.py'
            saved=read(effective) if effective.exists() else {}
            if saved.get('source_sha256')!=sha(effective_script) or saved.get('source',{}).get('sha256')!=sha(out/'gates.csv'):
                effective.parent.mkdir(parents=True,exist_ok=True)
                _atomic_json(effective,effective_collect(out/'gates.csv',row['id']))
            row['conditional_BF16_gate_census']=bind(effective)
            row['conditional_BF16_gate_groups']=read(effective)['groups']
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            row['gate_census_error']=repr(error)
            print(json.dumps({'auxiliary_gate_export_error':row['id'],'error':repr(error)}),flush=True)


def resource_census(result):
    try:
        from scripts.report_research_budget18_resources_v1 import collect as resource_collect
        report=resource_collect();destination=ROOT/'research_results/budget18_resource_census_v1/report.json'
        destination.parent.mkdir(parents=True,exist_ok=True);_atomic_json(destination,report)
        by_id={row['run_id']:row for row in report['rows']}
        for row in result['rows']:row['resources']=by_id[row['id']]
        result['resource_census']=bind(destination)
        timing_rollup(result,report)
    except (OSError,ValueError,KeyError) as error:
        result['resource_census_error']=repr(error)
        print(json.dumps({'auxiliary_resource_export_error':repr(error)}),flush=True)


def timing_rollup(result,resources):
    completed=[r for r in resources['stage_rows'] if r.get('stage_completed')]
    measured={r['backend'] for r in completed if r['stage']=='SFT'}
    path=ROOT/'research_results/budget18_timing_rollup_v1/report.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'measured_terminal_stage_timing','created_unix':time.time(),
        'completed_stage_rows':completed,'resource_census':result['resource_census'],
        'trained_SFT_cells':result['trained'],'fully_verified_SFT_cells':result['fully_verified'],
        'unmeasured_SFT_backends':sorted({'smallthinker4','lfm25','giga31'}-measured),
        'scope':'native stage elapsed includes end evaluation/save and excludes model startup; not isolated kernel performance',
        'limitations':['Native completion timing is useful before full scientific replay/evaluation completes.',
            'Encoder, microbatch, cache warmth and optimizer placement differ; no causal speedup claim.',
            'The48–72hour planning range remains provisional until full LFM/Giga and external generation are measured.']}
    _atomic_json(path,report);result['timing_rollup']=bind(path)
    destination=ROOT/'research_results/budget18_timing_estimate.json';estimate=read(destination)
    estimate.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        completed_new_SFT=result['fully_verified'],trained_new_SFT=result['trained'],
        measured_timing_basis=str(path.relative_to(ROOT)),measured_timing_basis_binding=bind(path),
        unmeasured_SFT_backends=report['unmeasured_SFT_backends'])
    _atomic_json(destination,estimate)


def publish_status(result):
    state=read(STATE);plan=read(PLAN);now=datetime.datetime.now(datetime.timezone.utc)
    latest=state.get('controller_revisions',[{}])[-1]
    revision=latest.get('owner','').rsplit('_v',1)[-1]
    controller='research-budget18-v'+revision if revision in ('2','3','4','5','6','7') else 'research-budget18-v2'
    resource_ready=f'configs/research/budget18_resource_v{revision}_ready.json' if revision in ('6','7') else 'configs/research/budget18_resource_v3_ready.json'
    active=next((j for j in plan['jobs'] if j['id']==state.get('active_job')),None)
    status_path=ROOT/'research_results/budget18_status.json';status=read(status_path)
    status.update(updated_utc=now.isoformat(),completed_SFT_experiments=result['fully_verified'],
        completed_SFT_configurations=result['fully_verified'],trained_SFT_configurations=result['trained'])
    monitor=read(ROOT/'logs/research/budget18_resources_latest.json')
    current={}
    if active:
        row=state['jobs'].get(active['id'],{})
        current={'run':active['id'],'model':active['backend'],'encoder':active['encoder'],
                 'recipe':active['recipe'],'stage':active['stage'],'status':live_phase(row) or row.get('status','training'),
                 'supervisor':controller}
        for activity in monitor.get('activities',[]):
            if activity.get('run_id')==active['id'] and activity.get('activity_verified') and time.time()-monitor['time']<90:
                current.update({k:activity[k] for k in ('phase','step','total_steps','completed_records','total_records','cache','peak_vram_gb') if k in activity})
    status['current_stage']=status['current']=current
    status['scheduling_phase']=state.get('scheduling_phase','interleaved_training_evaluation')
    status['queue']['supervisor']=controller
    status['queue']['controller_readiness']='configs/research/budget18_queue_v'+controller.rsplit('-v',1)[-1]+'_ready.json'
    status['queue'].pop('pid',None)
    status['disk_policy']['actual_free_GiB']=monitor['disk']['free_gib']
    reference=native_qwen_reference();status['native_Qwen_INT4_internal116_reference']=reference
    _atomic_json(status_path,status)
    destination=ROOT/'RESEARCH_STATUS.md';old=destination.read_text();marker='Ниже сохранена история предыдущего плана.'
    history=old[old.index(marker):] if marker in old else ''
    active_text=f"`{current.get('run','—')}`; состояние `{current.get('status',state['status'])}`."
    head=f'''# Текущий статус: сокращённая серия из18 запусков

Обновлено {now:%d.%m.%Y %H:%M} UTC. **Обучено {result['trained']}/18; полностью проверено {result['fully_verified']}/18.** Текущая очередь: `{state['status']}`. Архивная таблица с четырьмя512-оценками, textCE и pairedCI: [RESEARCH_RESULTS_LEGACY_512.md](RESEARCH_RESULTS_LEGACY_512.md). Актуальные результаты следующих серий: [RESEARCH_RESULTS.md](RESEARCH_RESULTS.md).

Активная стадия: {active_text} Порядок очереди: `{state.get('scheduling_phase','interleaved_training_evaluation')}`. Сначала выполняются все обучения и проверки восстановления; длинные четыре512-оценки проходят отдельным завершающим проходом. Свежие шаги/records/GPU/RAM/cache/disk каждые30с: `logs/research/budget18_resources_latest.json`. Этот Markdown — snapshot смены стадии, а не live-счётчик каждого шага.

План: **SmallThinker → LFM2.5 → GigaChat3.1**, SigLIP2 и Qwen2.5-VL-7B vision bounded729; R1 frozenNF4/LoRA/новые experts, R2 full без новых experts, R3 **frozenNF4/LoRA/новые experts/изучаемые text-visual gates**. Один seed42,18ячеек/42trainingjobs. GPT-OSS и Gemma исключены. Полный протокол — [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md).

Данные `budget18-en-v1`:4096 A0,256 A1 R1/R3,3000MM+529SmolTalk SFT, одна эпоха; internal116/text34, heldout512. Source release SHA `910448c7fd680ec949d6521e32716aee2832a99fc534d185f0a06b226c0ec1c0`. Общие frozenfeatures переиспользуются; Qwen prefill3506images/600с/13,215ГиБ завершён, остаток заполняется при cache misses. Отдельный tensor-auditor фиксирует hashes A0/A1 до удаления промежуточных weights и проверяет реальные обновления; отчёты `research_results/budget18_parameter_updates_v1/`.

Отдельный контроль исходной Qwen2.5-VL-7B NF4 INT4: **{reference['records']}/116**, `{reference['status']}`. Результат и предсказания — `research_results/qwen25_native_int4_internal116_v1/`; он не входит в18 конфигураций. Поддержка16 стандартных бенчмарков и команды: [BENCHMARKS.md](BENCHMARKS.md).

Первый R1 SmallThinker/SigLIP2 полностью проверен: image/shuffle/blank/text-only **11,5820/11,3086/10,6771/9,1797**; все paired95%CI включают0. GPU replay6724тензоров passed; P побитово неизменён, updates новых experts/router подтверждены. Это короткий screening, устойчивый вклад изображения и насыщение обучения не установлены.

Supervisor `{controller}`; state `runs/research/budget18_queue_state_v1.json`; автоотчёт `research-budget18-results`; read-only auditor `research-budget18-parameter-audit`. Финальные веса сохраняются толькоbest/last. Resource32/V3: disk floor32ГиБ сверх всей ожидаемой записи/margin, host/cgroup floor32ГиБ. Первые три ручные stages сохраняют старый reserve120 в истории. R2 завершённый checkpoint пишется model-only без optimizer с сохранением FP32model/P/config/scheduler/RNG; первая фактическая запись SmallThinker прошла. Сейчас свободно{monitor['disk']['free_gib']:.1f}ГиБ по последнему monitor snapshot.

Удалено17981неиспользуемых SigLIPcacheфайлов (28,191ГиБ) после полного checksum/stat/source/open-file audit; исходные RGB/HFanchors/все научные weights сохранены. Консервативный capacity прогноз оставляет~10ГиБ сверх32ГиБ floor. Resource readiness `{resource_ready}`. Для SmallThinker/Qwen текущая очередь дополнительно фиксирует исправление нулевой суммы sigmoid в `small_qwen_router_numerics_v1`; все прежние SigLIP результаты сохраняют исходное исполнение.

Плановый ориентир **48–72GPU-часа с резервомдо96** остаётся предварительным и уточняется по реальным замерам моделей. Автоматическая сводка завершённых стадий: `research_results/budget18_timing_rollup_v1/report.json`; исторические первые SmallThinker замеры: `research_results/budget18_measured_timing_basis_v2/report.md`.

'''
    temporary=destination.with_suffix('.tmp');temporary.write_text(head+history);os.replace(temporary,destination)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');args=p.parse_args()
    running=True
    def stop(_n,_f):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    last=None
    while running:
        state=read(STATE);audits=tuple(sorted(str(p) for p in (ROOT/'research_results/budget18_parameter_updates_v1').glob('*/report.json')))
        reference_files=[ROOT/'research_results/qwen25_native_int4_internal116_v1'/n for n in ('progress.json','summary.json')]
        reference_stamp=tuple(p.stat().st_mtime_ns if p.exists() else None for p in reference_files)
        stamp=(state.get('active_job'),tuple((k,r.get('status'),bool(r.get('training_completed')),live_phase(r)) for k,r in state['jobs'].items()),audits,state.get('controller_revisions',[{}])[-1].get('source_sha256'),reference_stamp)
        if stamp!=last:publish();last=stamp
        if not args.watch:return
        for _ in range(30):
            if not running:break
            time.sleep(1)


if __name__=='__main__':main()
