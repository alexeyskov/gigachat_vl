# Финальное дообучение с небольшой примесью object detection

По уточнению пользователя от25 ноября2026: подмешать немного object detection в финальную смесь и ещё раз дообучить все модели. Прежняя матрица18 конфигураций сохраняется как сравнение до нового этапа. Текущий full-train GigaChat/Qwen заканчивает естественно; дополнительная очередь начнёт работу после завершения и проверки прежней матрицы.

## Состав и порядок стадий

Уже выполнены caption alignment MLP на4096 PixMo captions, короткий caption A1 визуальных блоков на256 примерах и мультимодальный SFT. Отдельной detection-разметки в прежних3000 MM не было:1050VQAv2,900TextVQA,600ChartQA,450ScienceQA. Второй проход повторял ту же смесь.

Новый одинарный SFT-проход имеет тот же объём3529 записей,56 optimizer updates при effective batch64:

| Поток | Примеры | Веса objective |
| --- | ---: | ---: |
| Prompted object detection | 450 | 0.15 |
| MM replay:892VQAv2+765TextVQA+510ChartQA+383ScienceQA | 2550 | 0.70 |
| Тот же английский text replay | 529 | 0.15 |

Детекция составляет15% MM примеров,12.75% всех записей. Длинные JSON ответы не получают непропорциональный вес: каждый из трёх потоков нормируется собственным числом supervised tokens во всём effective batch. Отсутствующий поток даёт нулевой вклад, веса не перенормируются. Loss=`.15CE_OD+.70CE_MM+.15CE_text`.

Повторяются все18 основных конфигураций:3 LLM ×2 frozen encoder ×3 метода. R1/R3 сохраняют QLoRA и четыре фиксированных визуальных блока, в R3 обучаются прежние modality gates. R2 сохраняет полное обучение языковой базы без новых блоков. Vision encoder/merger и MLP-проектор остаются frozen. Отдельный роутер не добавляется. Диагностический B2-контроль прежней серии не создаёт19-ю основную конфигурацию.

Начальные веса R1/R3 — лучший прежний SFT по внутренней116 из двух проходов, R2 — соответствующий завершённый full-train checkpoint. Это новый этап с новым optimizer, не точный resume. Seed42, один проход, LR LoRA/визуальных блоков5e-5, gates5e-4, full LLM5e-6; прежние warmup/cosine, clip1 и weight decay.01/zero для bias/norm/gates. Начальный microbatch64 у QLoRA и32 у full, с откатом всей группы/градиентов/RNG и делением microbatch при activation OOM. Никакого третьего прохода по новой смеси.

## Разметка и формат

Источник — [COCO2017 instances](https://cocodataset.org/#download), исходная схема `[x,y,width,height]` описана в [официальном формате COCO](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/format-data.htm). Загружены архив аннотаций и небольшой набор изображений, не полный image dataset. Source URL, hashes, annotation IDs, исходные размеры и лицензии изображений сохраняются в `datasets/research/detection-v1/`.

450 detection-запросов:270 «найди все объекты указанного класса»,67 пространственных (`leftmost/rightmost/topmost/bottommost`),68 «найди все объекты из80 классов»,45 запросов отсутствующего класса. Все80 классов представлены в обучении. Crowd regions и изображения с вырожденными после нормализации boxes исключены. На запрос приходится не больше четырёх полных целевых boxes; запросы с большим числом нужных объектов исключаются целиком, GT не обрезается. Это COCO-классы и простые пространственные описания; полноценное произвольное referring-expression grounding или OCR word-box обучение не заявляется.

Пример ответа:

```json
[{"bbox_2d":[120,85,640,910],"label":"person"}]
```

Формат: обычные JSON/digit токены, без расширения словаря. Порядок `xmin,ymin,xmax,ymax`, начало координат слева сверху, целые числа **0..1000 включительно**. Из COCO: сначала xywh→xyxy и clipping по изображению, затем `floor(coord/dimension*1000+.5)`. Положительная площадь должна сохраняться. Обратное преобразование: x·W/1000, y·H/1000. Пустой результат — `[]`.

Это явно выбранный пользовательский normalized1000 contract с JSON `bbox_2d/label`, а не утверждение о точном повторении обучения Qwen. В частности, [разработчики Qwen различают](https://github.com/QwenLM/Qwen3-VL/issues/866) normalized координаты Qwen2-VL и absolute resized-image координаты Qwen2.5-VL. Наши encoder donors не определяют формат числового ответа: ему обучается языковая часть по новой разметке.

Аннотации относятся к полному исходному изображению. COCO width/height и отсутствие EXIF-поворота проверены. SigLIP2 resize без crop и Qwen smart resize сохраняют осевую normalized геометрию; padded/cropped изображений в новой разметке нет.

## Разделение данных и проверка

Новый image pool:384 COCO train2017 для обучения,96+96 COCO val2017 для validation/test. Из него выпускаются450 TRAIN-запросов и по64 query в validation/test. Изображения разделены до построения запросов. Новые train-изображения не совпадают с известным старым heldout; новые heldout не совпадают с известными старыми train/validation и новым train. Проверены RGB SHA256 и pHash Hamming≤4. Это проверка известных локальных источников, не обещание отсутствия COCO в исходном pretraining публичных моделей.

`datasets/research/detection-v1/release.json` фиксирует manifest hashes, producer/source hashes и counts; `audit.json` — классы, exclusions и split checks. CPU preflight проверяет байты всех576 скачанных изображений и полный input+target context всех трёх токенизаторов. В текущем release самый длинный detection target≤115 токенов; generation cap256. Все исходные targets умещаются в4096, молчаливого truncation не допускается.

До SFT выполняется generation на64 OD validation запросах. После него — те же64 OD validation и исходная VQA116. Исходная VQA116 до этапа пересчитывается из сохранённых ответов проверенного predecessor; для Giga применяется native assistant EOT prefix. Ground truth/detection metadata никогда не передаются генератору, только public question/image fields.

OD метрики: строгая JSON+coordinate validity, precision/recall/F1 при IoU≥.5 и.75, macro F1 по положительным запросам, accuracy пустых запросов. Matching максимизирует число совпадений класса/IoU, затем сумму IoU; повторный bbox не получает повторного TP. Основная OD метрика — positive-query macro F1@IoU.5,0–100. Непарсируемый положительный ответ получает0. Пустой ответ не даёт положительного detection credit. Это не официальный COCO AP/mAP; у JSON без confidence scores не заявляется полноценный ranking-based COCO AP.

Отдельный native GPU процесс точно восстанавливает все learned tensors, повторяет8 validation detection и8 VQA генераций с совпадением token IDs/prompt SHA и оценивает64 новых test-запроса. Scope replay именно16 примеров, а не все116; VQA116 полностью измеряется после обучения основным процессом. Следим за регрессией старых задач, а не только за новым форматом JSON.

## Запуски и хранение

План зафиксирован с SHA256 `223fdd1f0ed2a45730e116fb893e9804cb04ac6bd92ca501b41d766e3d00bb67`. CPU проверки: 28 тестов и проверка полного контекста трёх токенизаторов; все 576 image assets проверены по исходным байтам. Сервис `research-detection-mix-v1` уже запущен и ожидает завершения текущей матрицы, не занимая GPU. Квитанция постановки и архив исходников: `research_results/detection_mix_v1/ready.json`, `frozen_detection_sources_v1.zip` в том же каталоге.

Предварительная оценка дополнительного этапа для всех 18 вариантов — 15–30 GPU-часов сверх оставшейся прежней серии. Это ещё не измеренная скорость нового SFT: уточнять её по первому запуску с учётом generation/evaluation и full-train. Новые detection-метрики появятся после фактической оценки; сама готовность данных не означает, что модели уже научились локализации.

- План: `configs/research/detection_mix_plan_v1.json`.
- Очередь: `scripts/research_detection_mix_queue_v1.py`.
- Trainer: `scripts/run_research_detection_mix_v1.py`.
- Проверка: `scripts/validate_research_detection_mix_v1.py`.
- CPU preflight: `scripts/preflight_research_detection_mix_v1.py`.
- Новые outputs: `runs/research/detection_mix_v1/`.
- Сводка: `research_results/detection_mix_v1/current_metrics.md`.

Последний model-only checkpoint записывается **до evaluation**, без optimizer. Пока оценка не закончилась, публикуется только `last`, без фиктивного best-score. После оценки best/last публикуются как один физический файл через hardlink, без второй копии40GiB. При ошибке evaluation повторяется только evaluation сохранённых весов. Ошибка до сохранения финальных весов останавливает очередь с `requires_repair`, без бесконечного полного переобучения.

Минимум32GiB свободного диска и место под следующий checkpoint+cache проверяются до расходования GPU-времени. Full trainer освобождает завершённые optimizer states перед CPU snapshot. Запись крупного ZIP использует64MiB fsync/FADV окна. RAM-cache pressure вызывает read-only advice неиспользуемых checkpoint pages и предупреждение, а не убийство активного тренера.

После независимого proof новый R2 full checkpoint заменяет потреблённый исходник прежней стадии в политике хранения: старые крупные веса могут удаляться с ownership/hash/fd/mmap проверками. Прежние metrics/predictions/configs и receipt сохраняются. Старый Supervisor owner отключает autostart только после естественного завершения всей прежней матрицы, чтобы после удаления потреблённых источников она не пыталась запуститься заново. Маленькие QLoRA-источники сохраняются для сравнения. Best/last нового этапа остаются; epoch-архивы не создаются.

Использование обученного варианта:

```bash
/venv/main/bin/python scripts/predict_research_detections_v1.py \
  --run runs/research/detection_mix_v1/MODEL_VARIANT \
  --image /path/to/image.jpg \
  --prompt 'Locate every dog in the image.'
```

CLI возвращает исходный текст, проверенные normalized boxes и координаты исходного изображения в пикселях. Невалидный ответ явно отмечается; правдоподобные boxes не подставляются искусственно.

## Сверка с опубликованным обучением Qwen — 25 ноября 2026

Проверены официальный репозиторий семейства [Qwen3.5/3.6/3.8](https://github.com/QwenLM/Qwen3.8), карточки соответствующих моделей и открытый SFT-пайплайн Qwen3-VL. Публичный/exported SFT-код не раскрывает полностью производственный pretraining или RL-рецепт.

В [официальном fine-tune коде](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-finetune/qwenvl/train/train_qwen.py) используется стандартный Trainer и авторегрессионная генерация ответа с координатами. [Разметка labels](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-finetune/qwenvl/data/data_processor.py) маскирует остальные позиции значением −100. Exported [Qwen3.5 ForConditionalGeneration](https://github.com/huggingface/transformers/blob/b62646bbda836d467060b7e3d64028202079b77d/src/transformers/models/qwen3_5/modeling_qwen3_5.py) вызывает стандартную loss_function; в [её диспетчере](https://github.com/huggingface/transformers/blob/b62646bbda836d467060b7e3d64028202079b77d/src/transformers/loss/loss_utils.py) это ForCausalLMLoss с cross-entropy. Отдельной IoU/GIoU/L1 bbox-цели в проверенном SFT-маршруте не найдено. Это не доказательство отсутствия дополнительных целей во всех закрытых стадиях Qwen.

[Отчёт Qwen3-VL](https://arxiv.org/html/2511.21631v2) описывает square-root reweighting/per-token normalization для баланса текстовых и мультимодальных данных. Точная формула знаменателя и коэффициенты не опубликованы; это не раскрытый специальный геометрический loss. В grounding-корпус входят COCO, Objects365, OpenImages и RefCOCO/+/g, дополненные синтетической разметкой и фильтрацией. В архитектуре также используются MRoPE и DeepStack. Перенос одного visual encoder/merger не воспроизводит весь этот рецепт.

Решение для действующей серии: сохраняется supervised token CE по assistant-ответу, включая все coordinate tokens. Внутри каждого effective batch остаются явные веса потоков .15 OD / .70 MM replay / .15 text и отдельные token denominators. Эта нормировка записывается как наш вариант, а не как точное воспроизведение неопубликованного square-root алгоритма Qwen. Подмена loss в середине текущих18 запусков нарушила бы общий протокол; оснований добавлять «особый Qwen IoU-loss» по найденным первичным источникам нет.

Добавлен CPU-аудит `scripts/audit_research_qwen_grounding_strategy_v1.py`; отчёт и snapshots первоисточников с SHA находятся в `research_results/qwen_detection_strategy_v1/`. Проверены450 detection targets для каждого из трёх native tokenizers: все цифры координат покрыты supervised tokens, target IDs совпадают с native tokenize, expanded context не обрезает ответы. Доля target tokens, пересекающих coordinate digits: SmallThinker40.56%, LFM20.54%, Giga40.83%. Это счётчики токенов, не измеренные доли CE или градиентов.

Отдельный CPU-тест приватной копии frozen `_forward_loss` с заменой только устройства проверяет masked autoregressive CE, causal shift и все градиенты относительно независимого эталона на синтетических embeddings. Он не заменяет GPU-проверки обученных LLM. Текущие trainer, checkpoint, dataset и численный objective не изменены, GPU worker не останавливался. Низкий heldout F1 остаётся фактом; правильный формат JSON не подтверждает пространственную точность.

## Промежуточная диагностика геометрии — 26 ноября 2026

После14 независимых проверок сохранённые test-ответы проверены на вырождение координат: `research_results/detection_mix_v1/geometry_collapse_diagnostic.{json,md}`. У обоих LFM full-train вариантов100% предсказанных рамок покрывают минимум90% изображения; у SmallThinker/Qwen full-train —94.19%. При сохранении предсказанных моделью классов/числа объектов и замене геометрии рамкой всего изображения F1@IoU.5 не ниже исходного у всех14 проверенных вариантов. Это диагностическая подстановка в сохранённые ответы, не независимо сгенерированный baseline и не новая модель. Она не использует GT для построения замен; GT применяется только при оценке.

Вывод на этом срезе: ненулевой F1 пока не свидетельствует о полезной локализации. Нулевая accuracy запросов отсутствующего класса у завершённых вариантов также указывает на проблему отказа от ложных детекций. Исходные метрики/ответы сохранены, обучение и выбор checkpoint не меняются. Вывод по оставшимся четырём вариантам требует их завершения.
