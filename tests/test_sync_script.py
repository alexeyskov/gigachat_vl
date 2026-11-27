"""Exercise the real rsync client against a local, argument-checking SSH shim."""
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

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
    subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-f',str(key)],check=True)
    executable = tmp_path / 'bin/ssh'
    executable.parent.mkdir()
    executable.write_text('''#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
assert 'BatchMode=yes' in args
assert 'IdentityAgent='+os.environ['SSH_AUTH_SOCK'] in args
assert Path(os.environ['SSH_AUTH_SOCK']).is_socket()
assert subprocess.run(['ssh-add','-l'],capture_output=True).returncode == 0
with open(os.environ['SHIM_AGENT_LOG'], 'a') as log:
    log.write(json.dumps(dict(socket=os.environ['SSH_AUTH_SOCK'],pid=int(os.environ['SSH_AGENT_PID']))) + '\\n')
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
               SYNC_RETRY_DELAY='0', SYNC_IO_TIMEOUT='5', LC_ALL='C',
               SHIM_AGENT_LOG=str(tmp_path/'agents.jsonl'))
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
    result = invoke(remote,check=False)
    assert result.returncode == 0, result.stderr
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


def assert_agent_stopped(info):
    assert not Path(info['socket']).exists()
    assert not Path(info['socket']).parent.exists()
    with pytest.raises(ProcessLookupError):os.kill(info['pid'],0)


def test_encrypted_key_unlocked_once_and_reused_after_connection_cut(remote):
    root,destination,env=remote
    key=Path(env['SYNC_SSH_KEY']);key.unlink();key.with_suffix('.pub').unlink()
    phrase='throwaway test key passphrase'
    subprocess.run(['ssh-keygen','-q','-t','ed25519','-N',phrase,'-f',str(key)],check=True)
    askpass=root/'askpass'
    askpass.write_text("#!/usr/bin/env python3\nimport os\nwith open(os.environ['ASKPASS_COUNT'],'a') as f:f.write('asked\\n')\nprint('throwaway test key passphrase')\n")
    askpass.chmod(0o700)
    env.update(SSH_ASKPASS=str(askpass),SSH_ASKPASS_REQUIRE='force',DISPLAY=':test',
               ASKPASS_COUNT=str(root/'askpass_count'),SHIM_CUT_STREAM_ONCE=str(root/'cut.marker'))
    (root/'all-final/weights/large.pt').write_bytes(os.urandom(2*1024*1024))
    result=invoke(remote)
    assert (root/'askpass_count').read_text().splitlines()==['asked']
    assert '2/5' in result.stderr
    agents=[json.loads(line) for line in Path(env['SHIM_AGENT_LOG']).read_text().splitlines()]
    assert len({item['socket'] for item in agents})==len({item['pid'] for item in agents})==1
    assert_agent_stopped(agents[0])


def test_private_agent_is_removed_after_transfer_failure(remote):
    remote[2].update(SHIM_FAIL_ALWAYS='23')
    result=invoke(remote,check=False)
    assert result.returncode!=0
    agent=json.loads(Path(remote[2]['SHIM_AGENT_LOG']).read_text().splitlines()[0])
    assert_agent_stopped(agent)


def test_existing_external_agent_is_not_modified_or_stopped(remote):
    with tempfile.TemporaryDirectory(prefix='sync-test-',dir='/tmp') as directory:
        socket=Path(directory)/'outer.sock'
        outer=subprocess.Popen(['ssh-agent','-D','-a',str(socket)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                if socket.is_socket():break
                time.sleep(.01)
            assert socket.is_socket()
            env=remote[2];env.update(SSH_AUTH_SOCK=str(socket),SSH_AGENT_PID=str(outer.pid))
            invoke(remote)
            assert outer.poll() is None and socket.is_socket()
            assert subprocess.run(['ssh-add','-l'],env=env,capture_output=True).returncode==1
            agents=[json.loads(line) for line in Path(env['SHIM_AGENT_LOG']).read_text().splitlines()]
            assert all(item['pid']!=outer.pid and item['socket']!=str(socket) for item in agents)
            assert_agent_stopped(agents[0])
        finally:
            outer.terminate();outer.wait(timeout=5)
