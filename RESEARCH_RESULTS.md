# Результаты исследований LLM → VLM

Обновлено: 2026-11-27 02:09:34 UTC. Один training seed: 42.

## Завершённая финальная серия R1/R2/R3

Независимую проверку прошли **18/18** конфигураций. Здесь результаты финального SFT с примесью детекции, а не старой отменённой серии.

VQA116 — внутренняя macro-метрика по четырём группам с одним teacher reference. Detection — macro F1@IoU0.5 по положительным запросам test64. Шкала 0–100; это не официальная benchmark accuracy и не COCO AP.

| Модель | Энкодер | Метод | VQA116 до | VQA116 после | Detection test F1 | Проверка |
| --- | --- | --- | ---: | ---: | ---: | --- |
| SmallThinker | SigLIP2 | R1 | 21.55 | 25.00 | 6.25 | passed |
| SmallThinker | SigLIP2 | R2 | 19.46 | 22.27 | 7.95 | passed |
| SmallThinker | SigLIP2 | R3 | 25.57 | 23.99 | 4.17 | passed |
| SmallThinker | Qwen2.5-VL | R1 | 22.41 | 18.97 | 4.02 | passed |
| SmallThinker | Qwen2.5-VL | R2 | 24.34 | 20.40 | 16.37 | passed |
| SmallThinker | Qwen2.5-VL | R3 | 22.41 | 22.13 | 4.17 | passed |
| LFM2.5 | SigLIP2 | R1 | 12.93 | 12.93 | 0.00 | passed |
| LFM2.5 | SigLIP2 | R2 | 24.14 | 21.55 | 17.38 | passed |
| LFM2.5 | SigLIP2 | R3 | 19.83 | 15.52 | 0.00 | passed |
| LFM2.5 | Qwen2.5-VL | R1 | 11.21 | 14.66 | 0.00 | passed |
| LFM2.5 | Qwen2.5-VL | R2 | 22.93 | 25.00 | 17.38 | passed |
| LFM2.5 | Qwen2.5-VL | R3 | 13.79 | 14.02 | 0.00 | passed |
| GigaChat3.1 | SigLIP2 | R1 | 27.59 | 28.06 | 14.40 | passed |
| GigaChat3.1 | SigLIP2 | R2 | 26.72 | 24.83 | 17.56 | passed |
| GigaChat3.1 | SigLIP2 | R3 | 26.72 | 28.45 | 12.20 | passed |
| GigaChat3.1 | Qwen2.5-VL | R1 | 27.59 | 22.41 | 1.79 | passed |
| GigaChat3.1 | Qwen2.5-VL | R2 | 24.14 | 30.17 | 15.60 | passed |
| GigaChat3.1 | Qwen2.5-VL | R3 | 26.72 | 25.00 | 3.57 | passed |

R1: QLoRA + четыре фиксированных визуальных Transformer-блока на MoE-слой. R2: full-train языковой базы без новых блоков. R3: QLoRA + четыре блока с обучаемыми text/visual gates. Vision encoder и caption-aligned MLP заморожены.

«До» и «после» относятся к финальному проходу с детекцией. Независимый proof подтверждает точное восстановление learned tensors, воспроизведение 8 VQA и 8 detection ответов; отдельно оценены все 64 detection test запроса. Полный независимый replay всех 116 VQA этим не заявляется.

Исходная **Qwen2.5-VL-7B NF4 INT4** на том же internal116: **36.00**. Это отдельный reference, а не одна из 18 обученных конфигураций. [Исходные результаты](research_results/qwen25_native_int4_internal116_v1/summary.json).

У всех 18 вариантов accuracy отрицательных detection-запросов равна 0/8. Ранее на первых 14 вариантах обнаружены грубые рамки почти на всё изображение; ненулевой F1 не подтверждает полезную локализацию. [Диагностика](research_results/detection_mix_v1/geometry_collapse_diagnostic.md).

Данные этой таблицы: [JSON](research_results/detection_mix_v1/current_metrics.json), [итоговый отчёт](research_results/detection_mix_v1/FINAL_RESULTS.md).

## Новые эксперименты R4/R5

Финально проверено **1/6** конфигураций; очередь `running`. Во всех случаях frozen Qwen encoder, восемь новых нативных FFN-экспертов на MoE-слой и общий top-k=4, без modality masks.

R4 обучает новый router полностью; R5 сохраняет исходный router и обучает LoRA-расширение. Языковая база в обоих случаях NF4 + QLoRA. SFT1/SFT2 — промежуточные этапы, final — после detection SFT.

| Модель | Метод | Статус | VQA116 SFT1 | VQA116 SFT2 | VQA116 final | Detection test F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SmallThinker | R4 | завершён | 20.69 | 13.79 | 16.38 | 0.00 |
| SmallThinker | R5 | DET / train | 18.10 | 17.24 | — | — |
| LFM2.5 | R4 | в очереди | — | — | — | — |
| LFM2.5 | R5 | в очереди | — | — | — | — |
| GigaChat3.1 | R4 | в очереди | — | — | — | — |
| GigaChat3.1 | R5 | в очереди | — | — | — | — |

«—» означает, что оценка нового варианта ещё не получена; «*» — предварительное число до passed native proof. Эти пропуски не относятся к завершённым 18 вариантам первой таблицы.

[Текущий статус](RESEARCH_COMMON_POOL_STATUS.md) · [JSON](research_results/common_pool_v1/current_metrics.json) · [CSV](research_results/common_pool_v1/comparison.csv).

## Историческая диагностика на 512 вопросах

Прежняя первая таблица относилась к отдельному screening-протоколу image/shuffle/blank/text-only. В нём полностью подтверждена только одна строка; остальные ветви были заменены последующими экспериментами. Её пропуски не означали, что текущие 18 моделей не оценены.

[Архив прежнего RESEARCH_RESULTS.md](RESEARCH_RESULTS_LEGACY_512.md). Числа VQA116 не подставляются в столбцы старой 512-вопросной диагностики. Условия, модели и метрики разных серий сохраняются отдельно.
