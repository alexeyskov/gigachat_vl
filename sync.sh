#!/usr/bin/env bash
# Run on the LOCAL computer. Pull a curated article bundle, not the whole workspace.
set -euo pipefail

usage() {
    cat <<'HELP'
Использование: bash sync.sh [--dry-run] [--metrics-only | --all-final] ЛОКАЛЬНЫЙ_КАТАЛОГ

Каталог назначения обязателен; можно указать относительный или абсолютный путь.
Пример: bash sync.sh ./paper_results
Источник: root@162.43.170.108:23303, /workspace/gigachat_vl/exports/paper_v1/.
Ключ: ~/.ssh/yacloud. Скрипт запускается на вашем компьютере.

По умолчанию: метрики/предсказания всех 18 вариантов и ровно один финальный
чекпоинт каждого варианта. Веса сжимаются без потерь при передаче средствами
rsync, а на компьютере сохраняются в исходном формате (~185 ГиБ всего).
Добавлены только оценочные изображения и их признаки; обучающие данные,
промежуточные веса и полный cache не скачиваются.
Объём пакета выводится перед скачиванием. Исходники переносятся через Git.

  --dry-run   Показать изменения без скачивания содержимого файлов.
              Пакет на сервере уже должен быть подготовлен.
  --metrics-only  Только метрики, предсказания и конфигурации, без весов/картинок.
  --all-final     Явно выбрать режим по умолчанию: все 18 финальных чекпоинтов.
  -h, --help  Эта справка.

Повторный запуск докачивает прерванные передачи и обновляет изменённые файлы.
Готовые файлы с совпадающими размером и временем изменения пропускаются.
Недокачанные части сохраняются в .rsync-partial и используются повторно.
После обрыва связи: до 5 попыток с паузой 10 секунд. Ctrl+C останавливает скрипт;
для продолжения запустите ту же команду с ТЕМ ЖЕ каталогом назначения.
Локальные файлы не удаляются; файлы прежней полной копии автоматически не очищаются.
При переходе с полной копии можно выбрать новый каталог. Для докачки используйте прежний.
Без --metrics-only сохраняются веса для повторного запуска каждой строки таблицы.
Базовые HF модели не включены: их repo/revision сохранены в конфигурациях.
Для распаковки tar.zst нужны zstd и tar; команды есть в README.md пакета.

Другой SSH-ключ/адрес/порт можно задать переменными:
  SYNC_SSH_KEY=/path/to/key SYNC_SSH_PORT=23303 bash sync.sh ./experiment_results
Также поддерживаются SYNC_SSH_HOST и SYNC_SSH_USER.
Настройки повторов: SYNC_MAX_ATTEMPTS=5, SYNC_RETRY_DELAY=10 (секунды),
SYNC_IO_TIMEOUT=120 (секунды без обмена данными).
Для работы нужны bash, ssh и rsync на локальном компьютере.
HELP
}

fail() { printf 'Ошибка: %s\n' "$*" >&2; exit 1; }

destination=''
destination_set=0
dry_run=0
profile=all-final
profile_set=0
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --metrics-only|--all-final)
            (( profile_set == 0 )) || fail 'Выберите только один режим: --metrics-only или --all-final.'
            profile_set=1
            if [[ "$1" == --metrics-only ]]; then profile=metrics; else profile=all-final; fi
            ;;
        -h|--help) usage; exit 0 ;;
        --)
            shift
            (( $# <= 1 )) || fail 'Нужен один локальный каталог.'
            if (( $# )); then
                (( destination_set == 0 )) || fail 'Каталог указан дважды.'
                destination=$1
                destination_set=1
            fi
            break
            ;;
        -*) fail "Неизвестный параметр: $1" ;;
        *)
            (( destination_set == 0 )) || fail 'Нужен один локальный каталог.'
            destination=$1
            destination_set=1
            ;;
    esac
    shift
done

(( destination_set == 1 )) || fail 'Укажите каталог назначения: bash sync.sh ./paper_results'
[[ -n "$destination" ]] || fail 'Пустой локальный каталог.'
for program in ssh rsync; do
    command -v "$program" >/dev/null 2>&1 || fail "Не найдена команда $program."
done

ssh_host=${SYNC_SSH_HOST:-162.43.170.108}
ssh_user=${SYNC_SSH_USER:-root}
ssh_port=${SYNC_SSH_PORT:-23303}
ssh_key=${SYNC_SSH_KEY:-${HOME}/.ssh/yacloud}
case "$ssh_key" in '~/'*) ssh_key="${HOME}/${ssh_key#\~/}" ;; esac
[[ "$ssh_host" =~ ^[a-zA-Z0-9_.-]+$ && "$ssh_host" != -* ]] || fail 'Некорректный SSH host (IPv4 или DNS).'
[[ "$ssh_user" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*$ ]] || fail 'Некорректный SSH user.'
[[ "$ssh_port" =~ ^[0-9]{1,5}$ ]] || fail 'Некорректный SSH port.'
(( 10#$ssh_port >= 1 && 10#$ssh_port <= 65535 )) || fail 'SSH port должен быть от 1 до 65535.'
[[ -f "$ssh_key" && -r "$ssh_key" ]] || fail "Недоступен SSH-ключ: $ssh_key"

max_attempts=${SYNC_MAX_ATTEMPTS:-5}
retry_delay=${SYNC_RETRY_DELAY:-10}
io_timeout=${SYNC_IO_TIMEOUT:-120}
[[ "$max_attempts" =~ ^[0-9]{1,3}$ ]] || fail 'SYNC_MAX_ATTEMPTS должен быть числом от 1 до 100.'
[[ "$retry_delay" =~ ^[0-9]{1,4}$ ]] || fail 'SYNC_RETRY_DELAY должен быть числом от 0 до 3600.'
[[ "$io_timeout" =~ ^[0-9]{1,5}$ ]] || fail 'SYNC_IO_TIMEOUT должен быть числом от 1 до 86400.'
max_attempts=$((10#$max_attempts))
retry_delay=$((10#$retry_delay))
io_timeout=$((10#$io_timeout))
(( max_attempts >= 1 && max_attempts <= 100 )) || fail 'SYNC_MAX_ATTEMPTS должен быть от 1 до 100.'
(( retry_delay <= 3600 )) || fail 'SYNC_RETRY_DELAY должен быть от 0 до 3600.'
(( io_timeout >= 1 && io_timeout <= 86400 )) || fail 'SYNC_IO_TIMEOUT должен быть от 1 до 86400.'

mkdir -p -- "$destination"
destination=$(cd -- "$destination" && pwd -P)
[[ "$destination" != / ]] || fail 'Корень файловой системы нельзя использовать как каталог копии.'

ssh_args=(-p "$ssh_port" -i "$ssh_key" -o IdentitiesOnly=yes
          -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
remote="${ssh_user}@${ssh_host}"

# rsync -e has its own quoting parser: quote each argument, doubling embedded
# double quotes. Backslash/%q shell escaping is not understood by that parser.
ssh_transport='ssh'
for arg in "${ssh_args[@]}"; do
    arg=${arg//\"/\"\"}
    ssh_transport+=" \"${arg}\""
done

rsync_args=(-aH --no-owner --no-group --copy-unsafe-links
            --partial --partial-dir=.rsync-partial --no-whole-file
            "--timeout=$io_timeout" --progress --stats --human-readable)
if (( dry_run )); then rsync_args+=(--dry-run --itemize-changes); fi
# Compress in transit; keep the original checkpoint bytes on the local computer.
rsync_args+=(-z)

remote_path="/workspace/gigachat_vl/exports/paper_v1/${profile}"
printf 'Источник: %s:%s\nПрофиль: %s\nКаталог копии: %s\n' "$remote" "$ssh_port" "$profile" "$destination"
if (( dry_run )); then
    ssh "${ssh_args[@]}" "$remote" "test -s ${remote_path}/manifest.json" || \
        fail 'Пакет ещё не подготовлен. Запустите без --dry-run для его подготовки и скачивания.'
else
    printf '\nПроверка готовности пакета на сервере (готовые архивы повторно не создаются):\n'
    ssh "${ssh_args[@]}" "$remote" \
        "/venv/main/bin/python -u /workspace/gigachat_vl/scripts/prepare_paper_export_v1.py --profile ${profile}"
fi
printf '\nРазмер данных на сервере:\n'
ssh "${ssh_args[@]}" "$remote" "du -sh ${remote_path}"
printf '\nСвободное место на локальном диске:\n'
df -h "$destination"

printf '\nСкачивание выбранного пакета результатов\n'
printf 'Готовые файлы пропускаются; недокачанные части используются повторно.\n'
transfer_bundle() {
    rsync "${rsync_args[@]}" -e "$ssh_transport" \
        --exclude='.git/' --exclude='.codex/' \
        --exclude='.env' --exclude='.env.*' --exclude='.hf_home/' \
        --exclude='.venv/' --exclude='venv/' --exclude='__pycache__/' \
        --exclude='.pytest_cache/' --exclude='.mypy_cache/' --exclude='.ruff_cache/' \
        --exclude='.rsync-partial/' --exclude='*.partial' --exclude='*.tmp' \
        --exclude='.export_owner.json' \
        "${remote}:${remote_path}/" "$destination/"
}

interrupted() {
    printf '\nПередача остановлена. Для докачки повторите команду с тем же каталогом: %s\n' "$destination" >&2
    exit "$1"
}
trap 'interrupted 130' INT
trap 'interrupted 143' TERM

attempt=1
while :; do
    if transfer_bundle; then
        break
    else
        transfer_status=$?
    fi
    # Retry transport failures only. Disk/permission errors and explicit
    # interrupts need attention or a later manual restart, not a retry loop.
    retryable=0
    case "$transfer_status" in 10|12|30|35|255) retryable=1 ;; esac
    if (( dry_run || ! retryable || attempt >= max_attempts )); then
        printf '\nПередача не завершена (код %s). Полученные части сохранены.\n' "$transfer_status" >&2
        printf 'Для докачки повторите ту же команду с каталогом: %s\n' "$destination" >&2
        exit "$transfer_status"
    fi
    attempt=$((attempt + 1))
    printf '\nОбрыв связи (код %s). Докачка: попытка %s/%s через %s секунд.\n' \
        "$transfer_status" "$attempt" "$max_attempts" "$retry_delay" >&2
    sleep "$retry_delay"
done
trap - INT TERM

if (( dry_run )); then
    printf '\nПредварительный просмотр завершён. Уберите --dry-run для скачивания.\n'
else
    printf '\nГотово: %s\nПовторите эту же команду для следующей синхронизации.\n' "$destination"
fi
