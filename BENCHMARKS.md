# Оценка VLM на стандартных бенчмарках

Дополнительная оценка сохраняется отдельно от внутреннего отбора checkpoint на
116 примерах и от существующей проверки grounding на 512 примерах. Стандартные
оценки не участвуют в выборе best checkpoint и не меняют очередь 18 экспериментов.

Набор выбран по таблицам экспериментов в
[Qwen2.5-VL](https://arxiv.org/html/2502.13923v1),
[InternVL3](https://arxiv.org/html/2504.10479v1) и
[Qwen3-VL](https://arxiv.org/html/2511.21631v1). Конкретные пресеты, split,
метрики, необходимость судьи и ссылки на статьи записаны в
`configs/eval/vlm_benchmarks_v1.json`.

| Область | Поддерживаемые пресеты VLMEvalKit |
|---|---|
| Общее понимание | MMBench_DEV_EN_V11, MMStar, MME, RealWorldQA, MMVet |
| Знания и рассуждение | MMMU_DEV_VAL, MMMU_Pro_10c, MathVista_MINI |
| OCR, документы и графики | OCRBench, TextVQA_VAL, DocVQA_VAL, InfoVQA_VAL, ChartQA_TEST, AI2D_TEST |
| Галлюцинации | POPE, HallusionBench |

Используются оригинальные классы датасетов, построители промптов и оценщики
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit), закреплённые на commit
`ed3e918edf4a62e0bf42a2768655e02d45312fef`. Исходники harness не изменены.
Новый вход — `scripts/run_vlm_benchmarks.py`; прежний `src/eval/run_eval.py`
продолжает работать как общий вход VLMEvalKit.

## Окружение и предварительная проверка

Обучение использует `/venv/main`. Зависимости оценивания устанавливаются в
отдельное окружение с доступом к установленному Torch; основные библиотеки
обучения не обновляются. Точные дополнительные версии сохранены в
`configs/eval/vlmeval_extra_requirements_v1.txt`.

```bash
/venv/main/bin/python scripts/setup_vlm_benchmarks.py
CUDA_VISIBLE_DEVICES='' /tmp/research_benchmark_venv/bin/python scripts/run_vlm_benchmarks.py --check-harness
/venv/main/bin/python scripts/run_vlm_benchmarks.py --list
```

`--check-harness` проверяет commit, отсутствие изменений исходников и наличие
всех 16 пресетов без загрузки модели или датасетов на GPU. Полный стандартный
прогон требует свободной GPU: запуск с `--execute` отклоняется, если GPU занята,
сохраняя приоритет действующего обучения. `--mode score` работает на CPU и
использует уже сохранённые предсказания, без загрузки весов модели.

## Запуск исследовательского checkpoint

Команда без `--execute` печатает план. Данные скачиваются только при явном
`--allow-data-download`. Перед загрузкой и в ходе оценки проверяются лимит
каталога данных и свободное место; остаётся минимум 32 GiB. Большие TSV требуют
дополнительного запаса на извлечение изображений. Если лимита не хватает, запуск
завершается с ошибкой, а записанные предсказания остаются для продолжения.

```bash
/tmp/research_benchmark_venv/bin/python scripts/run_vlm_benchmarks.py \
  --run runs/research/smallthinker4__siglip2__1c__SFT__budget18_s42 \
  --suite core --max-context 8192 \
  --output research_results/benchmarks/smallthinker_siglip_r3 \
  --allow-data-download --data-budget-gib 12 --execute
```

Поддерживаются SmallThinker, LFM2.5 и GigaChat с обеими vision-конфигурациями и
всеми тремя методами. Восстановление выполняют их исходные версионные загрузчики,
включая исправленный normalized-sigmoid router SmallThinker с Qwen encoder.
Проверяются привязки к исходным моделям и сохранённые обученные тензоры. Адаптер
оценки замораживает загруженную модель, сохраняет порядок нескольких изображений
и строит соответствующую маску модальностей. Для одного изображения входные
embedding и маска совпадают с существующим исследовательским генератором.

Для короткой проверки можно указать, например, `--datasets MMStar --limit 64`.
Подвыборка определяется seed и записывается в протокол; её оценка помечается
диагностической. `--limit` запрещён для MME и HallusionBench, чтобы не разрушать
пары и группы, от которых зависит метрика. Для официального полного результата
следует убрать `--limit`.

Изображения исследовательских запусков сравниваются по decoded RGB SHA с
изображениями alignment/A1/SFT train. Число совпавших изображений и записей
сохраняется в `local_training_image_overlap`; записи не исключаются из
стандартного датасета. Такая проверка не устанавливает отсутствие пересечений
с неизвестными данными предобучения исходных моделей.

## Исходная Hugging Face VLM

Для native HF модели доступны BF16 и runtime bitsandbytes NF4 INT4 с двойным
квантованием. Это не маркируется как официальный AWQ/GPTQ checkpoint. Визуальная
часть остаётся BF16. Удалённая модель требует точный commit, загрузка новых весов
— отдельный `--allow-model-download` и проверку дискового бюджета.

```bash
/tmp/research_benchmark_venv/bin/python scripts/run_vlm_benchmarks.py \
  --hf-model Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --quantization nf4 --datasets MMStar AI2D_TEST RealWorldQA \
  --output research_results/benchmarks/qwen25_7b_nf4 \
  --allow-model-download --allow-data-download --execute
```

Можно передать локальный каталог вместо repo ID. Для него протокол сохраняет
SHA файлов весов, tokenizer и конфигурации. HF адаптер использует
`AutoModelForImageTextToText` и native processor/chat template без произвольного
remote code. Проверенный baseline — Qwen2.5-VL; другие архитектуры требуют
совместимости с этим AutoClass. Ограничение 729 visual tokens проверяется по
native `image_grid_thw` Qwen. Для иной архитектуры требуется отдельная проверка
её processor и бюджета изображений.

## Судьи и сопоставимость

MMVet и стандартный оценщик MathVista требуют LLM-судью. Можно сначала получить
предсказания для всех задач без внешних обращений:

```bash
/tmp/research_benchmark_venv/bin/python scripts/run_vlm_benchmarks.py \
  --run runs/research/smallthinker4__siglip2__1c__SFT__budget18_s42 \
  --suite all --mode infer --max-context 8192 \
  --output research_results/benchmarks/smallthinker_all \
  --allow-data-download --execute
```

Для оценки judge-задач следует использовать тот же `--output`, `--mode score`,
точные `--datasets`, соответствующий исходный `--run`/`--hf-model`, а также
`--allow-judge --judge-model ИМЯ_ЗАКРЕПЛЁННОЙ_ВЕРСИИ_СУДЬИ`. Параметры исходной
генерации берутся из сохранённого протокола. Установленный провайдеру ключ
читается из окружения и не записывается в результаты. Требуемый judge-score
не подменяется exact match; отказ внешнего судьи считается ошибкой оценки.

Для MCQ и yes/no по умолчанию используется исходный режим VLMEvalKit
`exact_matching`; при явном выборе судьи его имя записывается отдельно.
В частности, MMBench с таким извлечением ответов нельзя автоматически
приравнивать к таблице статьи с другим судьёй. Также выбран MMBench **dev v1.1**,
DocVQA/InfoVQA **validation** с открытыми ответами; скрытые test и отправки на
сервер leaderboard не выполняются. MathVista_MINI — стандартный testmini.

Сохраняются исходные шкалы: OCRBench total из 1000 и нормированное значение,
MME category/accuracy-plus, VQA consensus, ANLS, ChartQA numeric tolerance,
MCQ accuracy и POPE F1. Общая «точность по всем бенчмаркам» из разных шкал не
вычисляется. По умолчанию генерация greedy, B1, максимум 64 новых токена,
контекст 4096 и не более 729 visual tokens на изображение Qwen. Бюджеты можно
изменить явно; они записываются в протокол. При переполнении контекста нет
тихого обрезания текста или удаления изображений. Результат с этими бюджетами,
квантованием и split не является воспроизведением таблицы статьи.

Для каждого датасета сохраняются `protocol.json`, `predictions.jsonl`, таблица
для исходного оценщика, `evaluation_protocol.json`, `progress.json` и
`summary.json`. Журнал синхронизируется после каждого примера; повтор той же
команды продолжает его после проверки источника, порядка индексов и протокола.

## Контрольные проверки реализации

```bash
CUDA_VISIBLE_DEVICES='' BENCHMARK_HARNESS_TESTS=1 \
  /tmp/research_benchmark_venv/bin/python -m pytest -q \
  tests/test_vlm_benchmarks_v1.py tests/test_vlm_benchmarks_official_v1.py \
  tests/test_qwen25_native_baseline_storage_v1.py
```

Проверяются восстановление журнала, порядок изображений и маска, совпадение
одноизображенческого входа с замороженным генератором, восстановление CPU lease
при гибели координатора, а также реальные оценщики VLMEvalKit на синтетических
контрольных ответах: MCQ, OCRBench, document ANLS и ChartQA. Такие fixtures
помечены как синтетические и не являются результатами моделей на бенчмарках.

## Qwen INT4 на внутренней выборке 116

Отдельный запуск: `scripts/run_qwen25_native_baseline_parallel_v1.py` и
`scripts/eval_qwen25_native_internal116_v1.py`. Он использует оригинальную
Qwen2.5-VL-7B-Instruct указанного commit, NF4 INT4 для LLM, BF16 vision,
тот же список 116 записей (29 в каждой из четырёх групп), вопросы без изменения,
те же строгие внутренние scorers, greedy B1/64 и максимум 729 visual tokens.
Исходные 226 validation-строк дедуплицируются той же функцией, что в обучении;
результат полностью по структуре сравнивается с `internal_selection_116.jsonl`.

Протокол выбора имеет SHA
`325a06fa729330994e816e5692611b7c5f03c9463bcc1d8304a229a23e4a6b8a`.
Оценка запускается рядом с совместимым действующим trainer при достаточном
запасе GPU/RAM. Ограничение PyTorch allocator baseline — 12% GPU. Ни один
training worker не получает сигналов остановки; временный lease CPU-владельца
очереди защищает только границу запуска следующего GPU worker. Независимый
guardian восстанавливает эту границу при аварийной гибели координатора.
Сохраняются шаги trainer до и после параллельной оценки.

Предсказания, точный протокол, SHA исходных файлов, прогресс и итог находятся в
`research_results/qwen25_native_int4_internal116_v1/`. `summary.json` появляется
после всех 116 ответов. Это внутренняя macro-метрика с единственным teacher
reference, а не официальная accuracy VQAv2/MMMU и не оценка из статьи.
Временная отдельная копия исходных весов Qwen удаляется после завершения оценки
и выгрузки GPU-процесса; опубликованные ответы и привязки к источникам остаются.

Завершённый контроль Qwen: **35,9981** по исходному строгому internal116.
Разбор ответов выявил зависимость от teacher-формата: `C` при эталоне `Answer: C`
и `no` при `No.` дают ноль, хотя вопрос ScienceQA явно требует букву.
Дополнительный post-hoc пересчёт `scripts/audit_research_internal116_format_v1.py`
даёт **73,9291**. В нём general VQA использует существующую VQA-нормализацию,
а ScienceQA принимает только целую строку с буквой либо `Answer: X`, проверенной
по явным вариантам исходного вопроса. Объяснения не разбираются; OCR ANLS и
ChartQA не изменяются. Это один teacher reference, не human-consensus accuracy.

Тот же пересчёт применяется ко всем завершённым исследовательским моделям:
[таблица](research_results/budget18_internal_format_audit_v1/report.md).
Исторические результаты выбора checkpoint и вопросы сохранены; новые ответы
не генерировались. Автоотчёт расширяет таблицу по мере завершения остальных
конфигураций. Оба варианта оценки должны называться явно при сравнении.
