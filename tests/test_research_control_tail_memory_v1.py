import json

from scripts import guard_research_control_tail_memory_v1 as guard
from src.utils.research_storage import BestLastCheckpointManager, DiskBudget


def test_last_full_control_becomes_eligible_only_after_completion(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HOME', str(tmp_path / 'empty_hf'))
    run = tmp_path / 'giga31__qwen25__4f__SFT__budget18_s42'
    manager = BestLastCheckpointManager(run, DiskBudget(tmp_path, 0), mode='max')
    manager.save(lambda p: (p / 'state.pt').write_bytes(b'weights'),
                 step=56, metric=1, anticipated_bytes=7)
    plan = dict(jobs=[], controls=[dict(id=run.name, run=str(run))])
    assert list(guard.candidates(plan)) == []
    (run / 'completed.json').write_text(json.dumps(dict(steps=56, records=3529)))
    candidates = list(guard.candidates(plan))
    assert len(candidates) == 1  # best/last share one owned physical object.
    assert candidates[0][0] == (run / 'last/state.pt').resolve()


def test_reclaim_skips_mapped_checkpoint_without_advice(tmp_path, monkeypatch):
    path = tmp_path / 'state.pt'
    path.write_bytes(b'live weights')
    monkeypatch.setattr(guard, 'OUT', tmp_path / 'reports')
    monkeypatch.setattr(guard, 'ROOT', tmp_path)
    monkeypatch.setattr(guard, 'memory', lambda: dict(available_GiB=76))
    monkeypatch.setattr(guard, 'candidates', lambda plan: [(path, path.stat())])
    def refuse(identities):
        raise RuntimeError('mapped by an active model')
    monkeypatch.setattr(guard, 'scan_anchor_users', refuse)
    monkeypatch.setattr(guard.os, 'posix_fadvise',
                        lambda *args: (_ for _ in ()).throw(AssertionError('live file advised')))
    result = guard.reclaim({})
    assert result['advised'] == []
    assert len(result['skipped']) == 1
    assert path.read_bytes() == b'live weights'


def test_feature_cache_advice_batches_scanner_and_keeps_all_files(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, 'ROOT', tmp_path)
    monkeypatch.setattr(guard, 'OUT', tmp_path / 'reports')
    monkeypatch.setattr(guard, 'memory', lambda: dict(available_GiB=76))
    monkeypatch.setattr(guard, 'candidates', lambda plan: [])
    directory = tmp_path / 'cache/vision' / ('f' * 64) / '00'
    directory.mkdir(parents=True)
    for i in range(300):
        (directory / f'{i:064x}.pt').write_bytes(b'cached features')
    scans = []
    monkeypatch.setattr(guard, 'scan_anchor_users', lambda ids: scans.append(len(ids)))
    advice = []
    monkeypatch.setattr(guard.os, 'posix_fadvise', lambda *args: advice.append(args[0]))
    result = guard.reclaim({})
    assert scans == [256, 44]
    assert len(advice) == 300
    assert result['advised'][0]['files'] == 300
    assert len(list(directory.glob('*.pt'))) == 300
    assert all(p.read_bytes() == b'cached features' for p in directory.iterdir())
