"""Exercise the real rsync client against a local, argument-checking SSH shim."""
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'sync.sh'
RSYNC = shutil.which('rsync')
pytestmark = pytest.mark.skipif(RSYNC is None, reason='rsync required')


@pytest.fixture
def remote(tmp_path):
    root = tmp_path / 'remote'
    for profile in ('best', 'metrics', 'all-final'):
        directory = root / profile
        (directory / 'reports').mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps(dict(status='ready')))
        (directory / 'reports/table.csv').write_text('model,score\nexample,25\n')
        (directory / '.env').write_text('PRIVATE=not-for-export')
        (directory / '.export_owner.json').write_text('{}')
        (directory / 'unfinished.partial').write_text('unfinished')
        if profile == 'best':
            (directory / 'weights').mkdir()
            (directory / 'weights/winner.tar.zst').write_bytes(b'archive fixture'*4096)
            run = directory / 'runs/example'
            run.mkdir(parents=True)
            # The unselected configuration's role links remain dangling.
            (run / 'last').symlink_to('.checkpoint_objects/unselected')
        if profile == 'all-final':
            (directory / 'weights').mkdir()
            (directory / 'weights/state.pt').write_bytes(b'native checkpoint'*4096)
            os.link(directory / 'weights/state.pt', directory / 'weights/hardlink.pt')
            run = directory / 'runs/example'
            run.mkdir(parents=True)
            (run / 'last').symlink_to('../../weights')
    key = tmp_path / 'key with spaces \' " \\ $(touch DO_NOT_CREATE)'
    key.touch()
    executable = tmp_path / 'bin/ssh'
    executable.parent.mkdir()
    executable.write_text('''#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['SHIM_LOG'], 'a') as log:
    log.write(json.dumps(args) + '\\n')
while args and args[0].startswith('-'):
    option = args.pop(0)
    if option not in ('-p', '-i', '-o', '-l'):
        raise SystemExit('unexpected SSH option: ' + option)
    value = args.pop(0)
    if option == '-i' and value != os.environ['SYNC_SSH_KEY']:
        raise SystemExit('SSH key quoting corrupted')
host = args.pop(0)
assert host in ('root@fixture.invalid', 'fixture.invalid'), host
if args[0] == 'rsync':
    mapped = [arg.replace('/workspace/gigachat_vl/exports/paper_v1', os.environ['SHIM_ROOT']) for arg in args[1:]]
    if os.environ.get('SHIM_FAIL_ALWAYS'):
        raise SystemExit(int(os.environ['SHIM_FAIL_ALWAYS']))
    marker = os.environ.get('SHIM_CUT_STREAM_ONCE')
    if marker and not Path(marker).exists():
        Path(marker).touch()
        child = subprocess.Popen([os.environ['SHIM_RSYNC']] + mapped, stdout=subprocess.PIPE)
        remaining = 512 * 1024
        try:
            while remaining:
                chunk = child.stdout.read1(min(65536, remaining))
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                remaining -= len(chunk)
        finally:
            child.terminate()
            child.stdout.close()
            child.wait(timeout=5)
        raise SystemExit(255)
    os.execv(os.environ['SHIM_RSYNC'], ['rsync'] + mapped)
command = args[0]
if command.startswith('/venv/main/bin/python -u /workspace/gigachat_vl/scripts/prepare_paper_export_v1.py --profile '):
    profile = command.split()[-1]
    assert (Path(os.environ['SHIM_ROOT']) / profile / 'manifest.json').exists()
    print('export_already_ready: ' + profile)
elif command.startswith('test -s '):
    target = command[len('test -s '):].replace('/workspace/gigachat_vl/exports/paper_v1', os.environ['SHIM_ROOT'])
    raise SystemExit(0 if Path(target).is_file() else 1)
elif command.startswith('du -sh '):
    print('fixture size')
else:
    raise SystemExit('unexpected remote command: ' + command)
''')
    executable.chmod(0o755)
    env = dict(os.environ, PATH=str(executable.parent)+os.pathsep+os.environ['PATH'],
               SYNC_SSH_KEY=str(key), SYNC_SSH_HOST='fixture.invalid', SYNC_SSH_PORT='23303',
               SHIM_ROOT=str(root), SHIM_RSYNC=RSYNC, SHIM_LOG=str(tmp_path / 'ssh.jsonl'),
               SYNC_RETRY_DELAY='0', SYNC_IO_TIMEOUT='5', LC_ALL='C')
    destination = tmp_path / 'local result with spaces and \'" quotes'
    return root, destination, env


def invoke(remote, *flags, check=True):
    _, destination, env = remote
    return subprocess.run(['bash', str(SCRIPT), *flags, str(destination)],
                          env=env, text=True, capture_output=True, check=check, timeout=30)


def statistic_bytes(output, label):
    match = re.search(re.escape(label)+r':\s+([\d.,]+)\s*([kKMGT]?)', output)
    assert match, output
    return float(match.group(1).replace(',', '')) * 1000**('KMGT'.find(match.group(2).upper())+1 if match.group(2) else 0)


def test_default_all_final_and_repeat_preserves_local_files(remote):
    root, destination, env = remote
    invoke(remote)
    assert (destination / 'weights/state.pt').read_bytes() == (root / 'all-final/weights/state.pt').read_bytes()
    assert not (destination / 'weights/winner.tar.zst').exists()
    assert (destination / 'runs/example/last').is_symlink()
    assert not (destination / '.env').exists()
    assert not (destination / '.export_owner.json').exists()
    assert not (destination / 'unfinished.partial').exists()
    (destination / 'keep-local.txt').write_text('keep')
    (destination / 'weights/state.pt').unlink()
    (destination / 'weights/hardlink.pt').unlink()
    partial = destination / 'weights/.rsync-partial/state.pt'
    partial.parent.mkdir()
    partial.write_bytes((root / 'all-final/weights/state.pt').read_bytes()[:4096])
    invoke(remote)
    assert (destination / 'keep-local.txt').read_text() == 'keep'
    assert (destination / 'weights/state.pt').read_bytes() == (root / 'all-final/weights/state.pt').read_bytes()
    calls = [json.loads(line) for line in Path(env['SHIM_LOG']).read_text().splitlines()]
    assert sum(any('prepare_paper_export_v1.py' in arg for arg in call) for call in calls) == 2


def test_metrics_profile_has_no_weights(remote):
    invoke(remote, '--metrics-only')
    assert (remote[1] / 'reports/table.csv').exists()
    assert not (remote[1] / 'weights').exists()


def test_all_final_keeps_hardlinks_and_native_bytes(remote):
    root, destination, _ = remote
    invoke(remote, '--all-final')
    assert (destination / 'weights/state.pt').read_bytes() == (root / 'all-final/weights/state.pt').read_bytes()
    assert os.path.samefile(destination / 'weights/state.pt', destination / 'weights/hardlink.pt')


def test_completed_files_are_skipped_without_replacement(remote):
    invoke(remote)
    paths = [remote[1] / 'weights/state.pt', remote[1] / 'reports/table.csv']
    before = [(p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in paths]
    result = invoke(remote)
    assert 'Number of regular files transferred: 0' in result.stdout
    assert statistic_bytes(result.stdout, 'Literal data') == 0
    assert before == [(p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in paths]


def test_existing_partial_file_reuses_received_bytes(remote):
    root, destination, _ = remote
    invoke(remote)
    payload = os.urandom(8 * 1024 * 1024)
    (root / 'all-final/weights/large.pt').write_bytes(payload)
    partial = destination / 'weights/.rsync-partial/large.pt'
    partial.parent.mkdir(exist_ok=True)
    partial.write_bytes(payload[:4*1024*1024])
    result = invoke(remote)
    assert (destination / 'weights/large.pt').read_bytes() == payload
    assert statistic_bytes(result.stdout, 'Matched data') > 3*1024*1024
    assert statistic_bytes(result.stdout, 'Literal data') < 5*1024*1024


def test_connection_cut_mid_file_automatically_retries_and_resumes(remote):
    root, destination, env = remote
    payload = os.urandom(8 * 1024 * 1024)
    (root / 'all-final/weights/large.pt').write_bytes(payload)
    env['SHIM_CUT_STREAM_ONCE'] = str(root / 'network-cut.marker')
    result = invoke(remote)
    assert (destination / 'weights/large.pt').read_bytes() == payload
    assert '2/5' in result.stderr
    assert statistic_bytes(result.stdout, 'Matched data') > 128*1024
    calls = [json.loads(line) for line in Path(env['SHIM_LOG']).read_text().splitlines()]
    assert sum('rsync' in call for call in calls) == 2


def test_network_retries_are_bounded(remote):
    remote[2].update(SHIM_FAIL_ALWAYS='255', SYNC_MAX_ATTEMPTS='2')
    result = invoke(remote, check=False)
    assert result.returncode != 0
    calls = [json.loads(line) for line in Path(remote[2]['SHIM_LOG']).read_text().splitlines()]
    assert sum('rsync' in call for call in calls) == 2


def test_non_transport_error_is_not_retried(remote):
    remote[2]['SHIM_FAIL_ALWAYS'] = '23'
    result = invoke(remote, check=False)
    assert result.returncode != 0
    calls = [json.loads(line) for line in Path(remote[2]['SHIM_LOG']).read_text().splitlines()]
    assert sum('rsync' in call for call in calls) == 1


def test_dry_run_does_not_prepare_or_transfer(remote):
    invoke(remote, '--dry-run')
    _, destination, env = remote
    assert not list(destination.iterdir())
    calls = Path(env['SHIM_LOG']).read_text()
    assert 'prepare_paper_export_v1.py' not in calls


def test_conflicting_profiles_rejected_before_ssh(remote):
    result = invoke(remote, '--metrics-only', '--all-final', check=False)
    assert result.returncode != 0
    assert not Path(remote[2]['SHIM_LOG']).exists()
