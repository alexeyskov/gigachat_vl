#!/usr/bin/env bash
# Run on the LOCAL computer. Pull experiment artifacts into the required directory.
set -euo pipefail

usage() {
    cat <<'HELP'
Использование: bash sync.sh [--dry-run] ЛОКАЛЬНЫЙ_КАТАЛОГ

Каталог назначения обязателен; можно указать относительный или абсолютный путь.
Пример: bash sync.sh ./experiment_results
Источник: root@162.43.170.108:23303, /workspace/gigachat_vl/.
Ключ: ~/.ssh/yacloud. Скрипт запускается на вашем компьютере.

В каталог назначения копируются:
  runs/              обученные веса, best/last, метрики и результаты запусков;
  research_results/  сравнительные таблицы, проверки и исследовательские артефакты;
  configs/           конфигурации экспериментов;
  logs/              журналы запусков;
  RESEARCH_*.md и отдельные протоколы/статусы исследования.

Исходники рабочего репозитория, .git, датасеты, vision cache и HF model cache
не скачиваются. Базовые модели не входят в эту копию; их repo/revision сохранены
в артефактах запусков. Общий объём сейчас примерно 225 ГиБ, в основном checkpoint.

  --dry-run   Показать изменения без скачивания содержимого файлов.
  -h, --help  Эта справка.

Повторный запуск докачивает прерванные передачи и обновляет изменённые файлы.
Локальные файлы не удаляются. Веса best/last сохраняются как ссылки без дубликатов.
.env и служебные Python caches исключены и внутри каталогов результатов.

Другой SSH-ключ/адрес/порт можно задать переменными:
  SYNC_SSH_KEY=/path/to/key SYNC_SSH_PORT=23303 bash sync.sh ./experiment_results
Также поддерживаются SYNC_SSH_HOST и SYNC_SSH_USER.
Для работы нужны bash, ssh и rsync на локальном компьютере.
HELP
}

fail() { printf 'Ошибка: %s\n' "$*" >&2; exit 1; }

destination=''
destination_set=0
dry_run=0
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1 ;;
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

(( destination_set == 1 )) || fail 'Укажите каталог назначения: bash sync.sh ./experiment_results'
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
(( 10#$ssh_port >= 1 && 10#$ssh_port <= 65535 )) || fail 'SSH port должен быть от1 до65535.'
[[ -f "$ssh_key" && -r "$ssh_key" ]] || fail "Недоступен SSH-ключ: $ssh_key"

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
            --partial --partial-dir=.rsync-partial --progress --stats --human-readable)
if (( dry_run )); then rsync_args+=(--dry-run --itemize-changes); fi

printf 'Источник: %s:%s\nКаталог копии: %s\n' "$remote" "$ssh_port" "$destination"
printf '\nРазмер данных на сервере:\n'
ssh "${ssh_args[@]}" "$remote" \
    'du -sh /workspace/gigachat_vl/runs /workspace/gigachat_vl/research_results /workspace/gigachat_vl/configs /workspace/gigachat_vl/logs'
printf '\nСвободное место на локальном диске:\n'
df -h "$destination"

printf '\nРезультаты, обученные веса, конфигурации и логи экспериментов\n'
rsync "${rsync_args[@]}" -e "$ssh_transport" \
    --exclude='.git/' --exclude='.codex/' \
    --exclude='.env' --exclude='.env.*' --exclude='.hf_home/' \
    --exclude='.venv/' --exclude='venv/' --exclude='__pycache__/' \
    --exclude='.pytest_cache/' --exclude='.mypy_cache/' --exclude='.ruff_cache/' \
    --exclude='.rsync-partial/' \
    --include='/runs/***' --include='/research_results/***' \
    --include='/configs/***' --include='/logs/***' \
    --include='/RESEARCH_*.md' --include='/ADDITIVE_STATUS.md' \
    --include='/BENCHMARKS.md' --include='/DETECTION_PIPELINE.md' \
    --include='/STORAGE_RETENTION.md' --exclude='*' \
    "${remote}:/workspace/gigachat_vl/" "$destination/"

if (( dry_run )); then
    printf '\nПредварительный просмотр завершён. Уберите --dry-run для скачивания.\n'
else
    printf '\nГотово: %s\nПовторите эту же команду для следующей синхронизации.\n' "$destination"
fi
