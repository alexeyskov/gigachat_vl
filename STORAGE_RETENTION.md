# Хранение текущей серии

Уточнения пользователя: удалять устаревшие ненужные веса, держать best/last, резервировать место под checkpoint. Подтверждение для разрешённой очистки не требуется. Текущие SFT — `runs/research/fixed_capacity_eval_v5`, обязательные A1 — `runs/research/fixed_capacity_v4`, четыре фиксированных блока, не более двух SFT-проходов.

Сохраняются:

- B4 checkpoint после P1/P2: максимум два физических SFT объекта на конфигурацию. На каждом этапе best/last — ссылки на один финальный объект. Между проходами aggregate best/last выбирается в `research_results/capacity_eval_v5/best_last_by_arm.json`; P1 нужен также для сравнения длительности.
- Завершённые B2 результаты в `runs/research/additive_visual_v1` для сравнения ёмкости. SmallThinker/Qwen/R1 P1 дополнительно является источником B2 continuation; его удалять нельзя.
- Шесть full-finetune R2-контролей и необходимые незавершённому R2 alignment checkpoints.
- Шесть маленьких caption-P артефактов `runs/research/additive_visual_v1/projectors/`. Их большие прежние источники уже удалены после потребления старыми12 A1; новая серия читает P-артефакты напрямую с проверкой hashes/refs.
- Frozen исходные LLM, vision encoder/merger, используемые датасеты и общий BF16 feature cache.
- Компактные metrics/configs/protocols, source snapshots и receipts удаления.

Новый B4 A1 удаляется только после passed независимого GPU reload/replay SFT/P1. SFT/P2 инициализируется P1, а не A1. Final delta checkpoints содержат LoRA/блоки/gates/P, без копии frozen base и без Adam states. P2 запускает новый AdamW, поэтому optimizer resume не заявляется. Epoch/update checkpoints сверх best/last не создаются.

Перед сохранениями и этапами проверяется минимум32GiB свободного места сверх staging. Удвоение проходов не удваивает downloads/cache. При нехватке места сначала удаляются разрешённые obsolete/A1 объекты; действующие зависимости и лучшие/последние веса сохраняются.

Перед удалением проверяются ownership, границы путей, hashes, зависимости, открытые mmap/files. GPU-worker не останавливается; краткая lease при необходимости касается только CPU admission. Каталог `datasets/research/pilot-v1/images` содержит используемые training images и не удаляется по историческому имени.

Уже удалены:28 старых pilot/smoke/common-pool checkpoint,113.708GiB; routed-веса новой отменённой ветви,5.2648GiB; шесть потреблённых больших источников проекторов,10.94555GiB. Compact routed metrics: `research_results/fixed_only_v2/archived_routed_metrics.json`. Журналы и планы: `research_results/storage_retirement_v1/`. Незапущенные старые B2 A1 после перехода не нужны B4; их возможная очистка требует той же проверки открытых файлов и зависимостей.

Операционные инструменты: `scripts/retire_superseded_research_v1.py`, `scripts/watch_research_storage_retention_v1.py`; штатное удаление новых A1 реализовано контроллером v4. Завершённый старый watcher потреблённых источников повторно запускать для B4 не требуется. Предыдущая политика сохранена в `research_results/capacity_v4/previous_STORAGE_RETENTION.md`.

После естественного перехода на B4 удалены ещё четыре неиспользуемых B2 Giga A1, **2.53165GiB**. Проверены отсутствие зависимостей активного плана, ownership и открытые fd/mmap; GPU-worker сигналов не получал. Журнал: `research_results/capacity_v4/obsolete_A1_retirement/completed.json`. Завершённые B2 SFT и источник B2 continuation сохранены.

V5 дополнительно сохраняет одну временную `recovery/model.pt` после последнего optimizer update и до evaluation. Она не содержит optimizer. Если evaluation падает, используется только оценка сохранённых весов. Recovery удаляется после passed final GPU replay, одновременно с потреблённым A1 после P1. Нельзя удалять root v4 целиком: там находятся все12 действующих B4 A1, ещё нужных SFT v5. Неудачные v4 SFT не имеют завершённых обученных checkpoint; их error/metrics history сохраняется для аудита33 неудачных попыток. Суммарно best/last политика по двум завершённым проходам не меняется.

После нового финального SFT с detection (`DETECTION_PIPELINE.md`) потреблённые старые full-train R2 веса могут удаляться только после независимого proof нового checkpoint, проверки исходного SHA и отсутствия fd/mmap. Предыдущие metrics/predictions/configs остаются. Старый controller autostart отключается после естественного завершения, новая очередь использует общий GPU admission lock. Pending last сохраняется до evaluation, scored best/last используют один физический файл через hardlink. Резерв на следующий checkpoint проверяется перед обучением, не только при сохранении.

Для копирования результатов экспериментов на свой компьютер используется `sync.sh`, запускаемый локально с обязательным каталогом назначения:

```bash
scp -P 23303 -i ~/.ssh/yacloud root@162.43.170.108:/workspace/gigachat_vl/sync.sh ./sync.sh
bash ./sync.sh ./experiment_results
```

Повтор команды докачивает файлы; удаления на сервере не удаляют локальные копии. В указанном каталоге сохраняется структура `runs/`, `research_results/`, `configs/`, `logs/` и исследовательские Markdown-отчёты/протоколы. Обученные checkpoint входят в копию. Исходники рабочего репозитория, `.git`, датасеты, vision cache, внешние базовые модели/HF cache и Python environments не скачиваются; код переносится через Git. Служебные Python caches и `.env` исключены и внутри результатов. Текущий объём артефактов около225ГиБ, главным образом веса. Только просмотр: `bash ./sync.sh --dry-run ./experiment_results`.

Внутренние ссылки best/last и hardlinks сохраняются; внешние/абсолютные ссылки на небольшие metadata-файлы материализуются через `--copy-unsafe-links`. Глобальное разыменование ссылок не применяется, чтобы не создавать лишние копии весов. Исторические абсолютные пути внутри JSON/provenance остаются серверными: скачивание не переписывает экспериментальные manifests и их SHA.
