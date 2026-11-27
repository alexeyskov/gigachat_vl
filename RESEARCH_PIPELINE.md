# LLM → VLM: четыре фиксированных визуальных блока

Действующий протокол от 24 ноября 2026 года: четыре фиксированных визуальных Transformer-блока на каждом MoE-слое, максимум два SFT-прохода по малой замороженной выборке. Отдельный роутер новых экспертов не используется. По согласованию пользователя исследуем увеличение ёмкости и короткое продолжение обучения; улучшение метрик заранее не предполагается.

План: `configs/research/fixed_capacity_eval_plan_v5.json`, SHA-256 `aafd990e6325498ec9c34c5561cade44fe88be6829cfe1fff032a5ddda6eea06`. Контроллер: `scripts/research_capacity_queue_v5.py`; GPU worker: `scripts/run_research_capacity_eval_v5.py`. Все12 B4 A1 завершены в v4 и переиспользуются; новые SFT пишутся в отдельную root v5. Старые планы и numerical implementations не переписываются. Предыдущий текст протокола сохранён в `research_results/capacity_v4/previous_RESEARCH_PIPELINE.md`.

## 1. Матрица и сравнения

| Языковая база | Репозиторий |
| --- | --- |
| SmallThinker-4B-A0.6B | `Tiiny/SmallThinker-4BA0.6B-Instruct` |
| LFM2.5-8B-A1B | `LiquidAI/LFM2.5-8B-A1B` |
| GigaChat 3.1 10B-A1.8B | `ai-sage/GigaChat3.1-10B-A1.8B-bf16` |

Два frozen encoder: SigLIP2 `google/siglip2-so400m-patch14-384` и vision encoder + native merger из `Qwen/Qwen2.5-VL-7B-Instruct`. Для Qwen используется локальный vision-only экспорт, профиль `qwen_native_bounded_729`, максимум 729 визуальных токенов. Языковая часть донора не загружается. Ревизии фиксированы model inventory/метаданными экспорта. GPT-OSS, Gemma и дополнительные seeds исключены; seed эксперимента42.

| Метод | Обучение языковой базы | Новая ветвь |
| --- | --- | --- |
| R1-F4 | Frozen NF4 + LoRA | Четыре фиксированных Transformer-блока на MoE-слой |
| R3-F4 | Frozen NF4 + LoRA | Четыре блока с индивидуальными text/visual masks |
| R2, контроль | Full fine-tuning исходной LLM | Без новых блоков |

**3 базы × 2 энкодера × 3 метода = 18 основных конфигураций:** 12 четырёхблочных R1/R3 и шесть full-finetune R2. У каждой четырёхблочной конфигурации оцениваются один и два SFT-прохода. R2 остаются прежними одно-проходными контролями: их нельзя выдавать за одинаковый двухпроходный бюджет. Пять готовых R2 используются повторно; недостающий Giga/Qwen выполняется в хвосте очереди.

Дополнительный контроль длительности: уже проверенный **SmallThinker + Qwen encoder + R1 с двумя блоками** получает ровно один дополнительный SFT-проход с теми же настройками продолжения. Сравниваем B2/P1, B2/P2, B4/P1, B4/P2, разделяя эффект числа блоков и длительности обучения. Это один диагностический контроль сверх 18 основных конфигураций. Старые завершённые B2 результаты сохраняются отдельно и не объявляются результатами B4.

Всего новая очередь содержит **37 этапов: 12 A1 + 24 четырёхблочных SFT + один двухблочный SFT-продолжение**. Число SFT-этапов25 отличается от числа основных конфигураций18.

## 2. Архитектура

На каждом MoE-слое:

`y = NativeMoE(x; original_router, original_experts, original_k) + FixedVisualBranch(x, modality, valid)`.

Исходный MoE вызывается непосредственно; его экспертный набор, top-k и нормализация не расширяются. Shared expert GigaChat сохраняется. Новые блоки не занимают слоты исходных экспертов. Следующие слои получают состояния с добавленным вкладом ветви.

В новой ветви нет роутера. Каждый активный токен проходит **все четыре параллельных блока**, их выходы усредняются; это не четыре последовательных дополнительных слоя LLM. Визуальные токены всегда проходят добавочную ветвь. Текст после первого изображения тоже активен, чтобы вопрос/ответ использовал визуальный контекст. Текст до изображения, padding и полностью текстовые запросы обходят ветвь.

На каждом MoE-слое общие проекции H→256 и 256→H, четыре attention heads в каждом блоке, causal attention, LayerNorm, FFN256→1024→256 с GELU, синусоидальные позиции. Выходная проекция инициализируется нулями, сохраняя исходную функцию до первого update.

R3-F4 использует по два sigmoid-gates на блок: text и visual, начальное значение0.9. Source gate масштабирует value в attention, query gate — выход блока. Gates не меняют исходный routing; после обучения фиксируются. R1/R3 оба используют QLoRA: NF4 double quantization, BF16 compute; LoRA r16/alpha32/dropout0.05.

Проектор во всех конфигурациях: `Linear(Dvision,H) → GELU → Linear(H,H)`, FP32 masters; frozen после caption alignment. Четыре блока и gates обучаются отдельно от frozen vision encoder. Native sigmoid routing SmallThinker сохраняет проверенный эквивалентный log-sigmoid fallback только при нулевом знаменателе.

Увеличение с двух до четырёх блоков добавляет:

| База | Дополнительные параметры | FP32 веса + градиенты + Adam moments |
| --- | ---: | ---: |
| SmallThinker,32 MoE-слоя | 50 479 104 | 770.25 MiB |
| LFM,22 MoE-слоя | 34 704 384 | 529.55 MiB |
| GigaChat,25 MoE-слоёв | 39 436 800 | 601.76 MiB |

Это оценка параметров и optimizer storage, без дополнительных активаций. Реальные peak VRAM и скорость измеряются по GPU-журналам; прежние B2 пики45–56GiB не гарантируют такой же пик B4.

## 3. Данные и бюджет обучения

Неизменный release: `datasets/research/budget18-en-v1/release.json`, SHA-256 `910448c7fd680ec949d6521e32716aee2832a99fc534d185f0a06b226c0ec1c0`. Train и heldout разделены; подвыборки уже случайно выбраны с seed42 и общие между конфигурациями. Новый download не требуется.

| Этап | Данные | Обучается | Объём |
| --- | --- | --- | ---: |
| A0, ранее выполнен | Caption TRAIN PixMo | Только MLP-проектор | 4096 captions,64 updates |
| Новый A1 B4 | Другие caption TRAIN | Четыре свежих блока, masks в R3 | 256 captions,4 updates |
| SFT/P1 B4 | Мультимодальные инструкции + text replay | LoRA + блоки/masks | 3000MM + 529text,56 updates |
| SFT/P2 B4 или B2-контроль | Те же TRAIN-записи, другой порядок | Те же параметры | Ещё 3529 записей,56 updates |
| R2 | Та же исходная смесь | Все параметры языковой базы | 3529 записей,56 updates |

Alignment: `alignment_a1_train.jsonl`, `alignment_validation.jsonl`. SFT: `sft_multimodal_train.jsonl`, `sft_validation.jsonl`, `text_train.jsonl`. Смесь включает VQA, OCR/документы, графики/схемы и знания/рассуждение. Валидатор сверяет точные пути и SHA файлов с release.

Общий caption-проектор переиспользуется по паре LLM/encoder из сохранённого `alignment_projector_reuse_v1`: проверяются все четыре тензора, hashes, исходные refs и feature-cache fingerprint. Его извлечение не объявляется новым A0. Большие старые источники P уже удалены после завершения прежних 12 A1; новый runner читает сохранённый маленький P-артефакт. LoRA и визуальные блоки из старого B2 SFT в B4 не переносятся. Новый A1 сохраняет стартовые LoRA-факторы для точной инициализации SFT/P1.

Loss=`0.85 MM CE + 0.15 text CE`, с отдельными глобальными token-denominators эффективного batch. Отсутствующий поток даёт ноль без перенормировки. Последние девять записей включены в финальный update. Effective batch64, начальный microbatch64, max length4096. OOM откатывает всю accumulation group, градиенты и RNG, затем уменьшает microbatch.

Каждый SFT-этап — один проход. **P2 является явным warm restart из проверенных весов P1: новый AdamW, новый короткий scheduler, половина начального LR**, поскольку финальный P1 не содержит optimizer moments. Это не точный optimizer resume и не непрерывное двухэпоховое обучение. LR для LoRA/блоков: P1=1e-4, P2=5e-5; для mask gates: P1=1e-3, P2=5e-4, со штатным warmup/cosine decay внутри каждого этапа. Порядок P1 использует shuffle seed42, P2 —43 (`data_epoch_offset=1`); основной seed остаётся42. P2 не допускает источника с уже завершённым P2; третьего прохода нет.

Текущие данные составляют малый бюджет исследования. Для контекста официальный [LLaVA README](https://github.com/haotian-liu/LLaVA/blob/main/README.md) описывает558K alignment и около665K instruction examples. Поэтому наши3K MM нельзя считать гарантированно достаточным насыщением обучения; два коротких прохода проверяют полезность продолжения, не заменяя масштабный SFT.

## 4. Кэш и оценки

BF16 features кэшируются **до нашего MLP**. Encoder/merger frozen/eval, запускается только на cache miss. Fingerprint включает image hash, pinned encoder/merger, preprocessing, feature point и dtype. Кэш переиспользуется между LLM, B2/B4 и обоими проходами при совпадении fingerprint. Обучаемые блоки между примерами не кэшируются. Generation cache ветви принадлежит запросу и очищается до/после генерации, включая исключения. Causal attention запрещает доступ к будущему ответу.

После каждого SFT/P1 и P2:

1. Та же внутренняя выборка116:29 записей в каждой из четырёх групп, native greedy batch1, максимум64 new tokens. Основная метрика — macro score,0–100. Сохраняются ответы, scores, generated token IDs и prompt SHA. Это validation с одной train-derived teacher reference, не официальный benchmark. Canonical format rescoring учитывается отдельно.
2. Text-only heldout CE по прежней выборке.
3. Новый **multimodal heldout CE** на этих116 записях с `teacher_reference` только в teacher-forced loss. Он не попадает в prompt генерации. CE взвешивается числом supervised tokens; файл `multimodal_validation_ce.json`. Если train CE падает, а heldout CE/генерация ухудшаются, продолжение не считается улучшением.
4. Отдельный GPU-процесс заново создаёт модель, проверяет точное восстановление тензоров и повторяет116 генераций. Требуется passed native reload/replay.

Для GigaChat используется native assistant end marker128001. Сравнения требуют одинакового stopping rule. Diagnostics: в snapshot каждого слоя `visual_slots = 4 × valid_visual_tokens` у B4 и `2 × valid_visual_tokens` у B2; полный census эпохи не заявляется.

Лучший проход выбирается по primary macro116; при равенстве остаётся более ранний. Last — последний законченный проход. Сводка: `research_results/capacity_eval_v5/best_last_by_arm.json`. Выбор лучшего checkpoint по validation не превращает эту выборку в независимый test. На116 примерах и одном seed небольшие различия не доказывают устойчивое преимущество.

Официальные VLM-бенчмарки поддерживаются `scripts/run_vlm_benchmarks.py`, каталог/scorers в `BENCHMARKS.md`. Их результаты требуют отдельного запуска и не подменяются внутренними116.

## 5. Очередь, переход и наблюдение

Supervisor `research-capacity-handoff-v4` приостанавливает только CPU admission прежней очереди. Активный B2 GPU-worker заканчивает естественно. Затем новый контроллер `research-additive-v1` при необходимости независимо проверяет его законченный checkpoint и начинает B4. GPU-worker сигналов остановки не получает. Handoff receipt: `research_results/capacity_v4/handoff/completed.json`; прежний queue state архивируется.

В исправленной очереди первым выполняется LFM/SigLIP2/R3/P1 для проверки полного цикла. Затем SmallThinker/Qwen/R1 B4 P1 → B2 контроль P2 → B4 P2, остальные SmallThinker, LFM, GigaChat. Все12 A1 переиспользуются; R1 и R3 исследуются для обоих энкодеров. Ранее завершённые B2 результаты не переобучаются. Будущие незапущенные B2 задания вытеснены новой B4 матрицей.

Источники наблюдения:

- `ADDITIVE_STATUS.md` — актуальная очередь:25 SFT-этапов и12 B4 конфигураций.
- `runs/research/additive_visual_v1/queue_state.json` — master state, active plan/output/PID/progress.
- `runs/research/fixed_capacity_eval_v5/` — исправленные SFT/P1/P2 и B2-continuation; `runs/research/fixed_capacity_v4/` сохраняет12 обязательных A1 и историю неудачных SFT.
- `logs/research/capacity_eval_queue_v5.log` — события/ошибки/ограниченные повторы контроллера каждые30 секунд во время worker.
- `logs/research/additive_visual_v1/*__train.log` и `*__reload.log` — GPU worker logs.
- `research_results/capacity_eval_v5/best_last_by_arm.json` — лучшие/последние проверенные проходы.

Перед этапами и сохранениями проверяется запас минимум32GiB с учётом staging. Политика в `STORAGE_RETENTION.md`: один физический best/last на этап, максимум два SFT checkpoint на конфигурацию, A1 удаляется после passed P1 replay. Нет checkpoint на каждый update. Данные/кэш не дублируются; новая очередь использует offline model cache.

Оценка оставшегося времени должна строиться по реально измеренным длительностям B4 на текущем GPU, отдельно учитывая обучение,116 генераций, fresh reload и R2 full-finetune. До первого B4 измерения линейная экстраполяция B2 — лишь предварительная оценка; резерв времени не повод увеличивать число эпох.

## 6. Приёмка и исправления

- План фиксирует12 B4 конфигураций,6 R2, один B2 continuation и37 этапов; никакого нового роутера или третьего прохода.
- Все четыре блока получают визуальные позиции, маски имеют градиенты, causal checks проходят; original expert routing сохраняется.
- Каждый worker проверяет frozen data, refs/cache, source binding, pass и budgets. P2 принимает только same-architecture completed P1/56updates/3529records с passed GPU proof и тем же SHA checkpoint; dtype строго совпадает.
- Packed LoRA BF16 A1→FP32 SFT для LFM/Giga использует отдельный lossless validation bridge `src/utils/research_additive_init_dtype_v1.py`. Исходные checkpoint tensors не переписываются; другие dtype/shape/ref нарушения отклоняются.
- CPU-регрессии проверяют dispatch4, causality, budgets, pass2 guards, реальный half-LR AdamW без глобального Torch patch, heldout token-weighted CE и мониторинг новой output root.
- Старые численные источники и completed scores заморожены. Новый execution receipt отдельно фиксирует block count, pass, shuffle offset, optimizer restart и source hashes.

## Исправление потери private evaluation binding, v5

После запуска v4 обнаружены33 неудачных SFT-попытки: каждая доходила до итоговой оценки, но генерация использовала исходный оценщик вместо private additive evaluator. Причина — повторная компиляция `main` для изменения seed без повторного AST-преобразования import dependencies. В batch1 prefill оставалась маска последнего training batch64, guard выдавал `Stale additive-branch modality mask`. Final save располагался после оценки, поэтому завершённые веса не сохранялись. Подтверждённых SFT-результатов v4 нет. По журналам потеряно8.90 GPU-часа только вычисления training steps; это не включает loading/прочие накладные расходы. Все12 A1 корректны и сохранены. Полный аудит: `research_results/capacity_eval_v5/incident.json`.

V5 применяет seed offset и private import binding совместно к pinned исходнику. Архитектура, функции потерь, исходные данные и оптимизация не меняются. Первые наблюдаемые шаги нового LFM совпали с v4 точно по train loss и token counts; это проверка наблюдаемого префикса, не утверждение о всех весах.

До первого optimizer update каждого SFT выполняются native prefill+два decode токена и teacher-forced CE на одном heldout примере. Smoke использует только constructor weights, его ответ/CE не являются результатами обученной модели. Context восстанавливает RNG и trainability; проверяется файл `pretraining_evaluation_smoke.json`.

После56-го optimizer update, **до генерации**, атомарно сохраняется `recovery/model.pt`: модель без optimizer states, точные refs/config, SHA receipt, статус `trained_pending_evaluation`. При ошибке оценки очередь запускает `scripts/recover_research_capacity_eval_v5.py`, который повторяет только оценку этих весов, без training updates. Подтверждённый результат всё ещё требует обычных116 генераций, MM/text CE, official best/last и отдельного native GPU reload/replay. Recovery-копия удаляется после passed proof. Это временная защита последней обученной версии, не архив эпох.

Не более двух полных автоматических попыток, если ошибка случилась до recovery save; затем дальнейшее дорогое переобучение блокируется с `attention_required.json`. Оценка сохранённых final weights не переводится в повторное обучение. Первая исправленная конфигурация — LFM/SigLIP2/R3/P1 для проверки полного цикла, затем исходный порядок малых моделей. Handoff v5 дождался естественного завершения текущего GPU worker. Нельзя выдавать retry status за успешный результат; монитор публикует failures отдельно.

## Дополнение от25 ноября: небольшая примесь object detection

Проверка опубликованного loss Qwen3.5/3.6/3.8 и SFT Qwen3-VL описана в разделе «Сверка с опубликованным обучением Qwen» в [DETECTION_PIPELINE.md](DETECTION_PIPELINE.md). Сохранён исходный CE-протокол: отдельная подтверждённая формула IoU/GIoU loss для генеративных bbox в изученных источниках не найдена. Машинный отчёт CPU-проверки supervision: `research_results/qwen_detection_strategy_v1/report.json`.

По последнему уточнению пользователя ко всем18 конфигурациям добавляется один финальный SFT-проход на450 detection +2550 MM replay +529 text. OCR сохраняется; первые alignment стадии не повторяются. Координаты bbox_2d — целые0..1000, JSON label/bbox_2d. Текущий full-train заканчивает естественно. Новые результаты и веса отделены от предыдущей матрицы. Точная спецификация данных, objective, источников, валидации и хранения: [DETECTION_PIPELINE.md](DETECTION_PIPELINE.md).

## Дополнение от 26 ноября: R4/R5, общий пул из E+8 экспертов

Пользователь добавил шесть конфигураций: SmallThinker-4B-A0.6B, LFM2.5-8B-A1B и GigaChat3.1-10B-A1.8B, каждая с R4 и R5. Используется только frozen визуальный энкодер и merger донора Qwen2.5-VL-7B-Instruct, preprocessing `qwen_native_bounded_729` и тот же BF16 feature cache. GPT-OSS и дополнительные seed не запускаются; seed=42.

В каждом существующем MoE-слое добавляются **8 новых FFN-экспертов нативного размера**. Это расширение пула routed FFN, в отличие от компактных дополнительных causal Transformer-блоков прежних R1/R3. Поэтому число trainable parameters обязательно показывается в сравнении; одинаковая стоимость методов не предполагается.

| LLM | Исходных routed экспертов на MoE-слой | После расширения | Выбирается на токен |
| --- | ---: | ---: | ---: |
| SmallThinker | 32 | 40 | 4 |
| LFM2.5 | 32 | 40 | 4 |
| GigaChat3.1 | 64 | 72 | 4 |

Dense-слои и shared experts не расширяются. Новый пул использует один общий top-k с исходным `k`: новые эксперты конкурируют со старыми за четыре места. Выбор детерминированный, по правилам backbone; stochastic sampling не вводится. Все валидные текстовые и визуальные токены могут выбирать любые эксперты, в том числе в text-only запросах. Modality masks/gates и отдельный visual top-k отсутствуют; исключается только padding. Обязательное добавочное выполнение новых экспертов поверх старого top-k здесь не применяется.

- **R4:** frozen NF4 языковая база + LoRA r16/alpha32/dropout0.05; восемь новых FFN обучаются полностью. Исходный router заменяется новым trainable FP32 `Linear(H,E+8)` со случайной инициализацией. Корректирующие routing bias исходного router обнуляются. Обучаются новые FFN, новый router и языковые LoRA; исходные языковые веса заморожены.
- **R5:** та же NF4 база, LoRA и новые FFN. Исходные router weights и нативные correction bias заморожены. Вычисляются `concat(native_logits, zeros(8)) + (32/16) B(Ah)`, где `A: H→16`, `B: 16→E+8` — trainable LoRA router, B первоначально нулевой. Router scoring выполняется в FP32; повторно квантовать этот небольшой router не требуется. Сохраняются исходные logits старых экспертов, но итоговый top-k/нормированные веса могут измениться уже из-за появления восьми новых кандидатов.

Нативные правила weighting/scaling сохранены: normalized sigmoid SmallThinker с проверенным fallback только при нулевом знаменателе, sigmoid/correction bias/normalization LFM и GigaChat. Общий gradient clipping=1.0, finite-loss/gradient guards. Новый router R4 и LoRA router R5 не являются LoRA-адаптерами MLP-проектора: это разные части модели.

Начало каждого нового эксперимента — исходная pinned LLM и свежие FFN/LoRA. Из старых экспериментов переносятся **только** четыре тензора caption-aligned MLP-проектора, с проверкой SHA, base/vision revisions и feature-cache fingerprint. Сам проектор заморожен на всех новых этапах. Готовые SFT-веса R1/R2/R3 не используются для инициализации R4/R5.

| Этап каждого варианта | Данные и budget | Trainable |
| --- | --- | --- |
| A1 | 256 caption TRAIN, 4 updates | Новые FFN и router R4 / LoRA router R5; языковые LoRA frozen |
| SFT1 | 3000 MM + 529 text TRAIN, 56 updates | Новые FFN, соответствующий router, языковые LoRA |
| SFT2 | Те же 3529 TRAIN-записей, shuffle seed43, 56 updates | Те же параметры, LR вдвое меньше |
| DET | 450 detection + 2550 MM replay + 529 text, 56 updates | Те же параметры, LR=5e-5 |

Это **6 экспериментов, 24 этапа, 1032 optimizer updates**. Для A1/SFT1 LR=1e-4; AdamW FP32 masters/moments, `foreach=False`, weight_decay=.01, fresh optimizer на каждом этапе. Effective batch=64, начальный microbatch=64, max_length=4096. Activation OOM откатывает всю accumulation group и RNG, затем уменьшает microbatch. Дополнительные эпохи/seed не добавляются.

Только в A1 применяется exploration: для каждого валидного токена один новый кандидат выбирается циклически среди восьми. Это предотвращает пропуск части новых экспертов при одинаковых нулевых R5 logits. Значения logits для расчёта routing weights не меняются. На SFT1, SFT2, DET и во всей оценке действует чистый общий top-k, без принудительного выбора. Отдельный auxiliary routing loss не добавляется. Это eligibility всех экспертов, а не гарантия выбора визуальных экспертов после warmup; dispatch text/visual публикуется в `routing_diagnostics.json` с явной областью «последний training forward/recompute», не как полный census эпохи.

Objective A1 — caption supervised-token CE. SFT1/2 — `0.85 MM CE + 0.15 text CE` с раздельными token denominators effective batch. DET сохраняет исходный трёхпоточный `0.15 detection + 0.70 MM replay + 0.15 text` CE и integer bbox JSON0..1000. Никакой новой функции IoU-loss или изменения тестовых наборов нет.

После каждого SFT-этапа — прежняя внутренняя VQA116; после DET дополнительно detection validation64. Отдельный процесс заново загружает все learned tensors с точным сравнением и воспроизводит 8 VQA генераций (в A1 — 8 caption probes). Для DET он дополнительно воспроизводит 8 detection validation генераций и оценивает полный detection test64. VQA116 остаётся внутренним single-teacher-reference macro, detection F1@IoU0.5 не является COCO AP. Это не новые официальные benchmark результаты.

Код: `src/model/research_common_pool_v1.py`, `src/utils/research_common_pool_training_v1.py`. Очередь: `scripts/research_common_pool_queue_v1.py`; pinned plan: `configs/research/common_pool_plan_v1.json`. Запуски отделены в `runs/research/common_pool_v1/`, логи — `logs/research/common_pool_v1/`. Текущий статус обновляется каждые 30 секунд в `RESEARCH_COMMON_POOL_STATUS.md` и `research_results/common_pool_v1/status.json`; индивидуальный loss пишется после каждого update.

На каждом этапе один model-only final checkpoint записывается **до генерации**. Best/last после оценки ссылаются на один физический файл; optimizer states не сохраняются. При ошибке оценки повторяется оценка сохранённых весов. Веса потреблённого A1/SFT1/SFT2 удаляются только после passed независимого proof преемника, совпадения SHA и проверки отсутствия открытых fd/mmap. Финальные DET-веса шести вариантов сохраняются. Остальные 18 готовых checkpoint и `exports/paper_v1` не изменяются и не удаляются во время скачивания. Минимальный резерв на диске — 32 ГиБ сверх следующего checkpoint.
