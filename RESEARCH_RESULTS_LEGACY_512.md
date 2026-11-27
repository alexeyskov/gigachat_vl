# Результаты серии18

Полностью проверено: **1/18**. Обучено: **16/18**. Один seed42.

Здесь только новая сокращённая серия. Скоры — group macro на512 диагностических вопросах (8×64), не полные официальные benchmark scores. Сохранены строгие исторические метрики; нормализованный GQA-аудит рассматривается отдельно.

| Модель | Энкодер | Подход | Статус | Image | Shuffle | Blank | Text-only | Text CE34 |
|---|---|---|---|---:|---:|---:|---:|---:|
| SmallThinker | SigLIP2 | R1 QLoRA+experts | superseded_weights_retired | — | — | — | — | — |
| SmallThinker | SigLIP2 | R2 full | completed | 25.33 | 26.10 | 25.90 | 25.37 | 0.767 |
| SmallThinker | SigLIP2 | R3 QLoRA+experts+masks | superseded_weights_retired | — | — | — | — | — |
| SmallThinker | Qwen729 | R1 QLoRA+experts | superseded_weights_retired | — | — | — | — | — |
| SmallThinker | Qwen729 | R2 full | trained_pending_evaluation | — | — | — | — | — |
| SmallThinker | Qwen729 | R3 QLoRA+experts+masks | superseded_weights_retired | — | — | — | — | — |
| LFM2.5 | SigLIP2 | R1 QLoRA+experts | superseded_weights_retired | — | — | — | — | — |
| LFM2.5 | SigLIP2 | R2 full | trained_pending_evaluation | — | — | — | — | — |
| LFM2.5 | SigLIP2 | R3 QLoRA+experts+masks | superseded_weights_retired | — | — | — | — | — |
| LFM2.5 | Qwen729 | R1 QLoRA+experts | superseded_weights_retired | — | — | — | — | — |
| LFM2.5 | Qwen729 | R2 full | trained_pending_evaluation | — | — | — | — | — |
| LFM2.5 | Qwen729 | R3 QLoRA+experts+masks | superseded_weights_retired | — | — | — | — | — |
| GigaChat3.1 | SigLIP2 | R1 QLoRA+experts | superseded_weights_retired | — | — | — | — | — |
| GigaChat3.1 | SigLIP2 | R2 full | trained_pending_evaluation | — | — | — | — | — |
| GigaChat3.1 | SigLIP2 | R3 QLoRA+experts+masks | superseded_weights_retired | — | — | — | — | — |
| GigaChat3.1 | Qwen729 | R1 QLoRA+experts | superseded_weights_retired | — | — | — | — | — |
| GigaChat3.1 | Qwen729 | R2 full | trained_pending_evaluation | — | — | — | — | — |
| GigaChat3.1 | Qwen729 | R3 QLoRA+experts+masks | pending | — | — | — | — | — |

Для завершённых строк все четыре512-оценки, уникальные predictions, source bindings, checkpoint replay и paired bootstrap проверены очередью. Таблица повторно сверяет опубликованные файлы и stat чекпоинта, не перечитывая GB tensor bodies.

Это короткий screening:4096caption alignment,256caption warmup R1/R3 и3529SFT records, одна эпоха. Насыщение обучения, устойчивость по seeds и превосходство метода этим не доказаны.

Предварительный парный image-vs-shuffle контроль всех трёх SmallThinker/SigLIP подходов: [таблица и95%CI](research_results/budget18_interim_grounding_v1/image_vs_shuffle.md). У R3 правильные картинки дают15,91 против12,50, разность+3,41 [0,94;6,11]. Это отдельный неполный анализ: blank/text-only R3 ещё ожидаются, поправки на множество конфигураций и вариация по training seeds не оценены. Основной статус R3 остаётся evaluation-pending.

## Внутренняя проверка обучения

Результаты финальной внутренней проверки SFT. Это selection116 и отдельные text34; их нельзя напрямую сравнивать с оценками512 в основной таблице. Внешние проверки выполняются после прохода всех обучений.

| Модель | Энкодер | Подход | Internal116 | Text CE34 |
|---|---|---|---:|---:|
| SmallThinker | siglip2 | R1 QLoRA+experts | 18.97 | 6.730 |
| SmallThinker | siglip2 | R2 full | 19.46 | 0.767 |
| SmallThinker | siglip2 | R3 QLoRA+experts+masks | 15.52 | 5.818 |
| SmallThinker | qwen25 | R1 QLoRA+experts | 17.24 | 6.623 |
| SmallThinker | qwen25 | R2 full | 24.34 | 0.771 |
| SmallThinker | qwen25 | R3 QLoRA+experts+masks | 13.79 | 5.493 |
| LFM2.5 | siglip2 | R1 QLoRA+experts | 22.04 | 1.036 |
| LFM2.5 | siglip2 | R2 full | 24.14 | 0.685 |
| LFM2.5 | siglip2 | R3 QLoRA+experts+masks | 16.98 | 1.044 |
| LFM2.5 | qwen25 | R1 QLoRA+experts | 21.77 | 1.048 |
| LFM2.5 | qwen25 | R2 full | 22.93 | 0.675 |
| LFM2.5 | qwen25 | R3 QLoRA+experts+masks | 20.31 | 1.022 |
| GigaChat3.1 | siglip2 | R1 QLoRA+experts | 27.81 | 0.667 |
| GigaChat3.1 | siglip2 | R2 full | 0.00 | 0.621 |
| GigaChat3.1 | siglip2 | R3 QLoRA+experts+masks | 2.59 | 0.690 |
| GigaChat3.1 | qwen25 | R1 QLoRA+experts | 14.66 | 0.668 |
| GigaChat3.1 | qwen25 | R2 full | 12.93 | 0.619 |

Отдельный контроль исходной **Qwen2.5-VL-7B-Instruct NF4 INT4** на том же internal116: **36.00**. Native BF16 vision, Qwen729, greedy B1/64; без нового исследовательского MLP. Это внутренняя macro-метрика, не официальный benchmark score и не одна из18 обучаемых конфигураций. [Полный результат](research_results/qwen25_native_int4_internal116_v1/summary.json).

После дополнительной канонизации формата Qwen получает **73.93**: старое сравнение штрафует `C` против `Answer: C` и `no` против `No.`. На тех же ответах всех завершённых моделей выполнен общий [format audit](research_results/budget18_internal_format_audit_v1/report.md). Это post-hoc чувствительность; исходный отбор и scores сохранены.

## Обучение и ресурсы

| Модель | Энкодер | Подход | SFT, мин | Torch VRAM, ГиБ | RAM cgroup, ГиБ | Минимум диска, ГиБ | Замеров |
|---|---|---|---:|---:|---:|---:|---:|
| SmallThinker | siglip2 | R1 QLoRA+experts | 20.75 | 21.00 | 36.97* | 257.57 | 16 |
| SmallThinker | siglip2 | R2 full | 20.16 | 43.97 | 86.55 | 262.99 | 40 |
| SmallThinker | siglip2 | R3 QLoRA+experts+masks | 22.44 | 21.58 | 38.75 | 256.05 | 38 |
| SmallThinker | qwen25 | R1 QLoRA+experts | 15.95 | 34.54 | 54.29 | 241.02 | 26 |
| SmallThinker | qwen25 | R2 full | 17.41 | 71.36 | 97.00 | 224.05 | 34 |
| SmallThinker | qwen25 | R3 QLoRA+experts+masks | 15.20 | 35.34 | 54.23 | 224.68 | 25 |
| LFM2.5 | siglip2 | R1 QLoRA+experts | 9.79 | 45.98 | 80.17 | 212.41 | 17 |
| LFM2.5 | siglip2 | R2 full | 51.44 | 90.92 | 128.77 | 186.74 | 103 |
| LFM2.5 | siglip2 | R3 QLoRA+experts+masks | 10.98 | 47.26 | 88.06 | 170.22 | 19 |
| LFM2.5 | qwen25 | R1 QLoRA+experts | 9.00 | 44.23 | 70.40 | 183.25 | 16 |
| LFM2.5 | qwen25 | R2 full | 50.42 | 91.40 | 118.33 | 159.53 | 101 |
| LFM2.5 | qwen25 | R3 QLoRA+experts+masks | 9.74 | 45.26 | 72.84 | 149.48 | 17 |
| GigaChat3.1 | siglip2 | R1 QLoRA+experts | 60.96 | 70.72 | 68.59 | 149.30 | 78 |
| GigaChat3.1 | siglip2 | R2 full | 115.53 | 94.10 | 115.55 | 122.33 | 209 |
| GigaChat3.1 | siglip2 | R3 QLoRA+experts+masks | 64.30 | 74.00 | 53.75 | 107.71 | 98 |
| GigaChat3.1 | qwen25 | R1 QLoRA+experts | 58.56 | 69.90 | 70.60 | 104.97 | 92 |
| GigaChat3.1 | qwen25 | R2 full | 111.82 | 94.04 | — | — | 0 |

VRAM — нативный peak allocated Torch. RAM и диск — максимум/минимум среди30секундных замеров verified training worker; cgroup включает другие процессы и file cache. Это не непрерывный peak RAM. Звёздочка означает отсутствие ранних замеров: первый получен более60с после первого update. Текущие runs показаны по snapshot. SFT-время включает финальную internal/text проверку и сохранение, исключает загрузку модели. Подробные source-prefix bindings: [resource census](research_results/budget18_resource_census_v1/report.json).

## Маски и округление

Условный пересчёт экспортированных FP32 sigmoid-гейтов в BF16. Он применим к пути, где гейт приводится к BF16 до умножения; это не новый GPU-прогон и не проверка всех runtime dtypes. Предпочтения параметров не доказывают причинной специализации.

| Модель | Энкодер | Эксперты | Слой×эксперт | Visual>text FP32 | Visual>text BF16 | Text>visual BF16 | Равны BF16 |
|---|---|---|---:|---:|---:|---:|---:|
| SmallThinker | siglip2 | new | 64 | 63 | 5 | 0 | 59 |
| SmallThinker | siglip2 | old | 1024 | 411 | 66 | 362 | 596 |
| SmallThinker | qwen25 | new | 64 | 61 | 3 | 1 | 60 |
| SmallThinker | qwen25 | old | 1024 | 434 | 49 | 360 | 615 |
| LFM2.5 | siglip2 | new | 44 | 44 | 10 | 0 | 34 |
| LFM2.5 | siglip2 | old | 704 | 33 | 8 | 588 | 108 |
| LFM2.5 | qwen25 | new | 44 | 44 | 2 | 0 | 42 |
| LFM2.5 | qwen25 | old | 704 | 25 | 1 | 611 | 92 |

[SmallThinker / siglip2: исходные CSV/гейты](/workspace/gigachat_vl/research_results/budget18_gate_census_v1/smallthinker4__siglip2__1c__SFT__budget18_s42/report.json) · [условный BF16 census](/workspace/gigachat_vl/research_results/budget18_effective_gates_v1/smallthinker4__siglip2__1c__SFT__budget18_s42/report.json).


[SmallThinker / qwen25: исходные CSV/гейты](/workspace/gigachat_vl/research_results/budget18_gate_census_v1/smallthinker4__qwen25__1c__SFT__budget18_s42/report.json) · [условный BF16 census](/workspace/gigachat_vl/research_results/budget18_effective_gates_v1/smallthinker4__qwen25__1c__SFT__budget18_s42/report.json).


[LFM2.5 / siglip2: исходные CSV/гейты](/workspace/gigachat_vl/research_results/budget18_gate_census_v1/lfm25__siglip2__1c__SFT__budget18_s42/report.json) · [условный BF16 census](/workspace/gigachat_vl/research_results/budget18_effective_gates_v1/lfm25__siglip2__1c__SFT__budget18_s42/report.json).


[LFM2.5 / qwen25: исходные CSV/гейты](/workspace/gigachat_vl/research_results/budget18_gate_census_v1/lfm25__qwen25__1c__SFT__budget18_s42/report.json) · [условный BF16 census](/workspace/gigachat_vl/research_results/budget18_effective_gates_v1/lfm25__qwen25__1c__SFT__budget18_s42/report.json).


## Дополнительная нормализация GQA

Те же ответы, с игнорированием регистра и ASCII-пунктуации только для GQA по пинованному lmms-eval. Исходная таблица выше сохранена. Это чувствительность метрики на данной512-подвыборке; генерация и checkpoint selection не меняются.

| Модель | Энкодер | Подход | Image | Shuffle | Blank | Text-only |
|---|---|---|---:|---:|---:|---:|
| SmallThinker | siglip2 | R2 full | 30.02 | 30.98 | 30.20 | 29.67 |

Все исходные scores повторно пересчитаны CPU-аудитом. Provenance и дополнительные pairedCI: [metric audit](research_results/budget18_metric_audit_v1/index.json).

## SmallThinker / siglip2 / R2 full

- Image−blank: -0.571 пункта; paired95% CI [-1.945; 0.833].
- Image−shuffle: -0.766 пункта; paired95% CI [-2.464; 0.770].
- Image−text_only: -0.039 пункта; paired95% CI [-2.297; 2.036].

CI отражает вариацию данной подвыборки по image clusters; это не cross-seed uncertainty. Пересечение нуля не подтверждает устойчивый вклад изображения.

Артефакты: [smallthinker4__siglip2__4f__SFT__budget18_s42](/workspace/gigachat_vl/runs/research/smallthinker4__siglip2__4f__SFT__budget18_s42/evaluation/budget18_native_v1/paired_summary.json).

Проверка фактических обновлений и неизменности проектора: [tensor evidence](/workspace/gigachat_vl/research_results/budget18_parameter_updates_v1/smallthinker4__siglip2__4f__SFT__budget18_s42/report.json).

PairedCI при дополнительной GQA-нормализации: [supplement](/workspace/gigachat_vl/research_results/budget18_metric_audit_v1/paired/smallthinker4__siglip2__4f__SFT__budget18_s42.json).
