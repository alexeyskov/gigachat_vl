"""CPU fixtures/mocked ownership lifecycle; no real GPU or process signals."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scripts import probe_research_parallel_b1 as p


def identity(pid, *, parent=1, birth=None, argv=None):
    argv = argv or ["python", "owned-probe", str(pid)]
    return {"pid": pid, "start_time": birth or pid * 100, "ppid": parent,
            "argv": argv, "command_sha256": p.digest(argv), "state": "S"}


def manifest():
    rows = []
    for benchmark, group in p.BENCHMARKS.items():
        for index in range(64):
            rows.append({"record_id": f"{benchmark}-{index}", "benchmark": benchmark,
                         "task_group": group, "question": "Visible question", "answers": ["answer"],
                         "scorer": "exact_match", "image_paths": ["image.png"], "image_hashes": ["a" * 64]})
    return rows


def generated(record):
    return dict(record_id=record["record_id"], generated_token_ids=[7, 3], generated_tokens=2,
                prediction="answer", prompt_sha256="b" * 64, prefill_tokens=31,
                visual_tokens=4, logits_to_keep=1, score=1.)


def test_selection_is64_balanced_original_order_and_gold_independent():
    rows = manifest(); selected = p.selected_indices(rows)
    assert len(selected) == 64 and selected == sorted(selected)
    assert {name: sum(rows[i]["benchmark"] == name for i in selected) for name in p.BENCHMARKS} == {name: 8 for name in p.BENCHMARKS}
    assert selected == [i for i in range(512) if i % 64 < 8]
    changed = copy.deepcopy(rows)
    for row in changed:
        row["answers"] = ["secret other reference"]
    assert p.selected_indices(changed) == selected
    assert rows[0]["answers"] == ["answer"]


@pytest.mark.parametrize("change", ["duplicate", "benchmark", "group", "short"])
def test_invalid_source_scope_refused(change):
    rows = manifest()
    if change == "duplicate": rows[1]["record_id"] = rows[0]["record_id"]
    if change == "benchmark": rows[0]["benchmark"] = "other"
    if change == "group": rows[0]["task_group"] = "other"
    if change == "short": rows.pop()
    with pytest.raises(ValueError): p.selected_indices(rows)


def test_generator_inputs_never_include_gold_or_cached_tokens():
    row = dict(manifest()[0], choices=["gold choice"], generated_token_ids=[9], score=1.)
    assert set(p.public_record(row)) == {"record_id", "question", "image_paths", "image_hashes"}
    assert p.public_record(row)["question"] == row["question"]


@pytest.mark.parametrize("change", ["token", "prompt", "prefill", "prediction", "bool_token", "length", "count"])
def test_every_native_token_and_prompt_metadata_must_match(change):
    row = manifest()[0]; old = generated(row); new = copy.deepcopy(old)
    if change == "token": new["generated_token_ids"][-1] = 4
    if change == "prompt": new["prompt_sha256"] = "c" * 64
    if change == "prefill": new["prefill_tokens"] += 1
    if change == "prediction": new["prediction"] = "different"
    if change == "bool_token": new["generated_token_ids"] = [True]
    if change == "length": new["generated_token_ids"] = [1] * 65
    if change == "count": new["generated_tokens"] = 1
    with pytest.raises(ValueError): p.token_parity(row, new, old)


def test_gpu_initial_exclusive_and_unknown_process_rules():
    snap = {"total_bytes": 96 * p.GIB, "free_bytes": 90 * p.GIB, "processes": []}
    p.validate_gpu(snap, exclusive=True)
    snap["processes"] = [{"pid": 999, "bytes": p.GIB}]
    with pytest.raises(RuntimeError, match="exclusive"): p.validate_gpu(snap, exclusive=True)
    with pytest.raises(RuntimeError, match="Unknown"): p.validate_gpu(snap)
    with pytest.raises(RuntimeError, match="Initial GPU headroom"):
        p.validate_gpu(dict(snap, processes=[], free_bytes=30 * p.GIB), exclusive=True)


def test_gpu_uuid_is_parsed_and_visibility_drift_is_refused(monkeypatch):
    responses = iter([SimpleNamespace(returncode=0, stdout='GPU-fixture, 98304, 92160\n'),
                      SimpleNamespace(returncode=0, stdout='')])
    monkeypatch.setattr(p.subprocess, 'run', lambda *a, **k: next(responses))
    snapshot = p.gpu_snapshot()
    assert snapshot['gpu_uuid'] == 'GPU-fixture' and snapshot['free_bytes'] == 90 * p.GIB
    p.validate_gpu(snapshot, exclusive=True, expected_uuid='GPU-fixture')
    with pytest.raises(RuntimeError, match='UUID'):
        p.validate_gpu(snapshot, exclusive=True, expected_uuid='GPU-other')


def test_gpu_caps_and_reused_pid_refused_even_if_allowed_number():
    worker = identity(20)
    snap = {"total_bytes": 96 * p.GIB, "free_bytes": 50 * p.GIB,
            "processes": [{"pid": 20, "bytes": 5 * p.GIB}]}
    p.validate_gpu(snap, [worker], lookup=lambda _: worker)
    with pytest.raises(RuntimeError, match="identity"):
        p.validate_gpu(snap, [worker], lookup=lambda _: dict(worker, start_time=999))
    with pytest.raises(RuntimeError, match="12GiB"):
        p.validate_gpu(dict(snap, processes=[{"pid": 20, "bytes": p.PROCESS_CAP + 1}]), [worker], lookup=lambda _: worker)
    with pytest.raises(RuntimeError, match="24GiB"):
        p.validate_gpu(dict(snap, free_bytes=p.MINIMUM_FREE - 1), [worker], lookup=lambda _: worker)


def test_stale_nvml_allowed_only_after_confirmed_owned_exit_never_pid_reuse():
    worker = identity(20)
    snap = {"total_bytes": 96 * p.GIB, "free_bytes": 90 * p.GIB,
            "processes": [{"pid": 20, "bytes": p.GIB}]}
    with pytest.raises(RuntimeError, match="identity"):
        p.validate_gpu(snap, [worker], lookup=lambda _: None)
    p.validate_gpu(snap, [worker], lookup=lambda _: None, exited_pids={20})
    with pytest.raises(RuntimeError, match="identity"):
        p.validate_gpu(snap, [worker], lookup=lambda _: dict(worker, start_time=999), exited_pids={20})


@pytest.mark.parametrize('death_state', ['absent', 'zombie', 'dying_empty_argv'])
def test_worker_exit_during_nvml_query_is_confirmed_fresh_through_owned_popen(death_state):
    worker = identity(20, parent=10)
    snapshot = {'total_bytes':96*p.GIB,'free_bytes':90*p.GIB,'processes':[{'pid':20,'bytes':5*p.GIB}]}
    live = None if death_state == 'absent' else dict(worker,state='Z' if death_state=='zombie' else 'R',argv=[])
    # The caller's previous Popen.poll was None, so exited_pids is still empty.
    confirmation = Mock(return_value=True)
    p.validate_gpu(snapshot,[worker],lookup=lambda _:live,exited_pids=(),confirm_owned_exit=confirmation)
    confirmation.assert_called_once_with(20)
    with pytest.raises(RuntimeError,match='identity'):
        p.validate_gpu(snapshot,[worker],lookup=lambda _:live,confirm_owned_exit=lambda _:False)


def test_exit_confirmation_cannot_excuse_reused_birth_or_foreign_nonempty_command():
    worker = identity(20,parent=10)
    snapshot={'total_bytes':96*p.GIB,'free_bytes':90*p.GIB,'processes':[{'pid':20,'bytes':5*p.GIB}]}
    for live in [dict(worker,start_time=999,state='Z',argv=[]),dict(worker,argv=['foreign']),dict(worker,ppid=99,state='Z',argv=[])]:
        confirmation=Mock(return_value=True)
        with pytest.raises(RuntimeError,match='identity'):
            p.validate_gpu(snapshot,[worker],lookup=lambda _:live,confirm_owned_exit=confirmation)
        confirmation.assert_not_called()


def barrier_fixture():
    parent = identity(10)
    workers = [identity(20, parent=10), identity(21, parent=10)]
    barrier = {"owner": p.OWNER, "status": "released", "nonce": "n" * 64,
               "parent": parent, "workers": workers, "batch_size": 1, "max_new_tokens": 64,
               "deadline_monotonic": 10 ** 20}
    return barrier, {10: parent, 20: workers[0], 21: workers[1]}


def test_barrier_binds_parent_and_both_worker_birth_argv_before_allocation():
    barrier, infos = barrier_fixture()
    assert p.barrier_worker(barrier, 0, 10, 1000, "n" * 64, infos[20], lookup=infos.get) is barrier


@pytest.mark.parametrize("change", ["preparing", "nonce", "parent_birth", "parent_argv", "peer_birth", "peer_argv", "worker_parent", "same_pid", "batch4", "expired"])
def test_barrier_rejects_unreleased_stale_reused_or_changed_commands(change):
    barrier, infos = barrier_fixture()
    if change == "preparing": barrier["status"] = "preparing"
    if change == "nonce": barrier["nonce"] = "other"
    if change == "parent_birth": infos[10] = dict(infos[10], start_time=999)
    if change == "parent_argv": infos[10] = dict(infos[10], argv=["foreign"])
    if change == "peer_birth": infos[21] = dict(infos[21], start_time=999)
    if change == "peer_argv": infos[21] = dict(infos[21], argv=["foreign"])
    if change == "worker_parent": infos[20] = dict(infos[20], ppid=1)
    if change == "same_pid": barrier["workers"][1] = barrier["workers"][0]
    if change == "batch4": barrier["batch_size"] = 4
    if change == "expired": barrier["deadline_monotonic"] = 0
    with pytest.raises((ValueError, TimeoutError)):
        p.barrier_worker(barrier, 0, 10, 1000, "n" * 64, infos[20], lookup=infos.get)


def test_owned_cleanup_does_not_signal_foreign_or_reused_pid():
    worker = identity(20)
    child = Mock(); child.poll.return_value = None
    assert not p.stop_owned(child, worker, lookup=lambda _: dict(worker, start_time=999))
    child.terminate.assert_not_called(); child.kill.assert_not_called()
    assert p.stop_owned(child, worker, lookup=lambda _: worker)
    child.terminate.assert_called_once(); child.wait.assert_called_once_with(timeout=3)


def test_owned_cleanup_escalates_only_same_child():
    worker = identity(20)
    child = Mock(); child.poll.return_value = None
    child.wait.side_effect = [subprocess.TimeoutExpired("own", 3), 0]
    assert p.stop_owned(child, worker, lookup=lambda _: worker)
    child.kill.assert_called_once(); assert child.wait.call_args.kwargs == {"timeout": 2}


def test_parent_death_guard_is_child_only_sigkill_and_race_checked():
    libc = Mock(); libc.prctl.return_value = 0
    with patch.object(p.ctypes, "CDLL", return_value=libc), patch.object(p.os, "getppid", return_value=10), patch.object(p.os, "_exit") as leave:
        p.arm_parent_death(10)
    libc.prctl.assert_called_once_with(1, signal.SIGKILL, 0, 0, 0); leave.assert_not_called()
    with patch.object(p.ctypes, "CDLL", return_value=libc), patch.object(p.os, "getppid", return_value=11), patch.object(p.os, "_exit") as leave:
        p.arm_parent_death(10)
    leave.assert_called_once_with(128 + signal.SIGKILL)


def test_waiting_worker_fails_before_any_torch_or_cuda_import(tmp_path):
    root = tmp_path; output = root / "research_results/new"; output.mkdir(parents=True)
    path = output / "allocation_barrier.json"
    path.write_text(json.dumps({"owner": p.OWNER, "status": "preparing", "nonce": "n"}))
    args = SimpleNamespace(parent_pid=10, parent_start_time=1000, barrier=str(path), worker=0, nonce="n")
    original = __import__
    def reject_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("src.model"):
            raise AssertionError("Training stack imported before released barrier")
        return original(name, *args, **kwargs)
    with patch.object(p, "ROOT", root), patch.object(p, "process_info", return_value=identity(10)), \
            patch.object(p, "arm_parent_death"), patch.object(p, "_check_deadline", side_effect=TimeoutError("waiting")), \
            patch("builtins.__import__", side_effect=reject_torch):
        with pytest.raises(TimeoutError, match="waiting"): p.worker_main(args)


def result_fixture():
    from src.eval.research_metrics import aggregate_scores
    rows = [manifest()[i] for i in p.selected_indices(manifest())]
    expected = [generated(row) for row in rows]
    context = {"record_ids": [r["record_id"] for r in rows], "model_provenance": {"fixture": "tiny_native"}}
    barrier, _ = barrier_fixture()
    barrier.update(generation_barrier={"owner": p.OWNER, "status": "released", "nonce": barrier["nonce"],
        "released_monotonic": 99., "allocation_barrier_sha256": "d" * 64}, observed_workers_done_monotonic=122., deadline_monotonic=200.)
    results = []
    for index in (0, 1):
        predictions = [dict(row, **generated(row)) for row in rows]
        start = 100. + index; end = 120. + index
        results.append({"owner": p.OWNER, "status": "passed", "worker": index,
                        "identity": barrier["workers"][index], "context_sha256": p.digest(context),
                        "allocation_barrier_sha256": "d" * 64, "model_provenance": context["model_provenance"],
                        "batch_size": 1, "records": 64, "record_ids": context["record_ids"],
                        "native_b1_exact_token_parity": True, "generation_started_monotonic": start,
                        "generation_ended_monotonic": end, "generation_seconds": end - start,
                        "records_per_second": 64 / (end - start), "pytorch_cap_bytes": p.PYTORCH_CAP,
                        "peak_pytorch_allocated_bytes": 5 * p.GIB,
                        "peak_pytorch_reserved_bytes": 6 * p.GIB,
                        "predictions": predictions, "subset_scores": aggregate_scores(predictions)})
    return context, barrier, results, rows, expected


def test_results_recompute_all128_tokens_scores_and_overlap_without_campaign_claim():
    context, barrier, results, rows, expected = result_fixture()
    with patch.object(p, "_read_selected", return_value=(rows, expected)):
        summary = p.validate_results(context, barrier, results)
    assert summary["distinct_original_records"] == 64
    assert summary["aggregate_duplicate_completions"] == 128
    assert summary["generation_window_seconds"] == 21
    assert summary["overlap_seconds"] == 19
    assert summary["aggregate_records_per_second"] == 128 / 21


@pytest.mark.parametrize("change", ["one_token", "score", "batch4", "order", "identity", "no_overlap", "metric", "cap",
                                    "question", "image_hashes", "source", "benchmark", "task_group", "model_provenance",
                                    "before_release", "after_observed", "generation_nonce", "barrier_hash"])
def test_results_refuse_mismatch_or_false_measurement(change):
    context, barrier, results, rows, expected = result_fixture()
    if change == "one_token": results[1]["predictions"][-1]["generated_token_ids"] = [7, 4]
    if change == "score": results[1]["predictions"][-1]["score"] = 0.
    if change == "batch4": results[1]["batch_size"] = 4
    if change == "order": results[1]["predictions"].reverse()
    if change == "identity": results[1]["identity"] = dict(results[1]["identity"], start_time=999)
    if change == "no_overlap":
        for row, start in zip(results, (100., 110.)):
            row.update(generation_started_monotonic=start, generation_ended_monotonic=start+5.,
                       generation_seconds=5., records_per_second=64/5.)
    if change == "metric": results[1]["subset_scores"]["group_macro_score"] = 0.
    if change == "cap": results[1]["peak_pytorch_allocated_bytes"] = p.PYTORCH_CAP + 1
    if change in ("question", "image_hashes", "source", "benchmark", "task_group"):
        results[1]["predictions"][-1][change] = "corrupted provenance"
    if change == "model_provenance": results[1]["model_provenance"] = {"fixture": "foreign"}
    if change == "before_release":
        results[1]["generation_started_monotonic"] -= 50
        results[1]["generation_ended_monotonic"] -= 50
    if change == "after_observed":
        results[1]["generation_started_monotonic"] += 10
        results[1]["generation_ended_monotonic"] += 10
    if change == "generation_nonce": barrier["generation_barrier"]["nonce"] = "other"
    if change == "barrier_hash": results[1]["allocation_barrier_sha256"] = "e" * 64
    with patch.object(p, "_read_selected", return_value=(rows, expected)):
        with pytest.raises(ValueError): p.validate_results(context, barrier, results)


class FakeChild:
    def __init__(self, pid, output, infos, argv):
        self.pid = pid; self.output = output; self.infos = infos; self.returncode = None
        self.index = pid - 20; self.terminated = False; self.killed = False
        infos[pid] = identity(pid, parent=os.getpid(), argv=argv)
        p.atomic_json(output / f"worker{self.index}.progress.json", {"phase": "model_ready"})
    def poll(self):
        if self.returncode is None and (self.output / "generation_barrier.json").exists():
            self.returncode = 0
            p.atomic_json(self.output / f"worker{self.index}.result.json", {
                "allocation_barrier_sha256": p.file_sha(self.output / "allocation_barrier.json")})
        return self.returncode
    def terminate(self): self.terminated = True; self.returncode = -15
    def kill(self): self.killed = True; self.returncode = -9
    def wait(self, timeout=None): return self.returncode


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    root = tmp_path; (root / "research_results").mkdir()
    output = root / "research_results/new_probe"
    parent = identity(os.getpid()); infos = {parent["pid"]: parent}; children = []
    def popen(argv, **kwargs):
        child = FakeChild(20 + len(children), output, infos, argv)
        child.environment = kwargs['env']; children.append(child); return child
    monkeypatch.setattr(p, "ROOT", root)
    monkeypatch.setattr(p, "process_info", lambda pid: infos.get(pid))
    monkeypatch.setattr(p, "prepare_context", lambda: {"a": "bound-context"})
    monkeypatch.setattr(p, "verify_context", lambda *a: None)
    monkeypatch.setattr(p, "gpu_snapshot", lambda: {"gpu_uuid": "GPU-fixture", "total_bytes": 96 * p.GIB, "free_bytes": 90 * p.GIB, "processes": []})
    monkeypatch.setattr(p.subprocess, "Popen", popen)
    monkeypatch.setattr(p, "validate_results", lambda *a: {"aggregate_duplicate_completions": 128})
    from src.utils.research_storage import DiskBudget
    monkeypatch.setattr(DiskBudget, "ensure_space", lambda *a: None)
    return SimpleNamespace(root=root, output=output, infos=infos, children=children)


def test_full_mocked_lifecycle_releases_two_bound_barriers_and_preserves_existing_cache(lifecycle):
    original = lifecycle.root / "untouched_B1_cache"; original.write_bytes(b"original frozen B1")
    result = p.run_probe(lifecycle.output, allow_gpu=True)
    assert result["status"] == "passed" and len(lifecycle.children) == 2
    barrier = p.read_json(lifecycle.output / "allocation_barrier.json")
    assert barrier["status"] == "released" and len(barrier["workers"]) == 2
    assert p.read_json(lifecycle.output / "generation_barrier.json")["allocation_barrier_sha256"] == p.file_sha(lifecycle.output / "allocation_barrier.json")
    assert original.read_bytes() == b"original frozen B1"
    assert all(not child.terminated for child in lifecycle.children)
    assert all(child.environment['CUDA_VISIBLE_DEVICES'] == 'GPU-fixture' for child in lifecycle.children)
    assert result["whole_probe_limit_seconds"] == 300
    assert result["execution_limit_seconds"] < 300


def test_unknown_gpu_before_release_stops_only_two_owned_children(lifecycle, monkeypatch):
    idle = {"gpu_uuid": "GPU-fixture", "total_bytes": 96 * p.GIB, "free_bytes": 90 * p.GIB, "processes": []}
    occupied = dict(idle, processes=[{"pid": 999, "bytes": p.GIB}])
    calls = iter([idle, occupied])
    monkeypatch.setattr(p, "gpu_snapshot", lambda: next(calls))
    with pytest.raises(RuntimeError, match="exclusive"):
        p.run_probe(lifecycle.output, allow_gpu=True)
    assert len(lifecycle.children) == 2
    assert all(child.terminated for child in lifecycle.children)
    assert p.read_json(lifecycle.output / "allocation_barrier.json")["status"] == "preparing"
    assert p.read_json(lifecycle.output / "report.json")["status"] == "failed"


def test_interrupt_cleans_children_and_restores_signal_handlers(lifecycle, monkeypatch):
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    snapshots = iter([{"gpu_uuid": "GPU-fixture", "total_bytes": 96 * p.GIB, "free_bytes": 90 * p.GIB, "processes": []}] * 2)
    calls = [0]
    def interrupt():
        calls[0] += 1
        if calls[0] == 3: raise InterruptedError("synthetic SIGTERM")
        return next(snapshots)
    monkeypatch.setattr(p, "gpu_snapshot", interrupt)
    with pytest.raises(InterruptedError): p.run_probe(lifecycle.output, allow_gpu=True)
    assert all(child.terminated for child in lifecycle.children)
    assert all(signal.getsignal(sig) is handler for sig, handler in original.items())


def test_parent_monitor_worker_exit_between_poll_and_nvml_does_not_false_fail(lifecycle, monkeypatch):
    # Real observed race: both children finish during the query, after the
    # parent's cached exited set was computed, while NVML still lists them.
    monkeypatch.setattr(FakeChild, 'poll', lambda child: child.returncode)
    calls=[0]
    def snapshot():
        calls[0]+=1
        value={'gpu_uuid':'GPU-fixture','total_bytes':96*p.GIB,'free_bytes':90*p.GIB,'processes':[]}
        if calls[0]>=4:
            for index,child in enumerate(lifecycle.children):
                child.returncode=0
                lifecycle.infos[child.pid]=None
                p.atomic_json(lifecycle.output/f'worker{index}.result.json',{
                    'allocation_barrier_sha256':p.file_sha(lifecycle.output/'allocation_barrier.json')})
                value['processes'].append({'pid':child.pid,'bytes':5*p.GIB})
        return value
    monkeypatch.setattr(p,'gpu_snapshot',snapshot)
    monkeypatch.setattr(p.time,'sleep',lambda *a:None)
    result=p.run_probe(lifecycle.output,allow_gpu=True)
    assert result['status']=='passed' and calls[0]>=4
    assert all(not child.terminated for child in lifecycle.children)


def test_existing_output_and_missing_authorization_never_spawn(lifecycle):
    with pytest.raises(ValueError, match="allow-gpu"):
        p.run_probe(lifecycle.output)
    lifecycle.output.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        p.run_probe(lifecycle.output, allow_gpu=True)
    assert lifecycle.children == []


def test_module_import_is_stdlib_only_and_never_initializes_cuda():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", "import sys; import scripts.probe_research_parallel_b1; assert 'torch' not in sys.modules"],
                            cwd=root, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_frozen_image_hash_helper_avoids_training_package_initializer(tmp_path):
    code = """
import sys
from pathlib import Path
from PIL import Image
from scripts.probe_research_parallel_b1 import ROOT,frozen_image_hasher
image=Image.new('RGB',(3,5),(12,23,34))
hasher=frozen_image_hasher(ROOT)
value=hasher(image)
import hashlib
assert value==hashlib.sha256(b'RGB:3:5:'+image.tobytes()).hexdigest()
assert 'torch' not in sys.modules
assert 'src.dataset' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_real_tiny_cpu_native_b1_two_independent_processes_match_preserved_tokens(tmp_path):
    # Three fresh interpreters: one serial reference and two simultaneous B1
    # processes with independent real native TinySmallThinker models. Timing is
    # not interpreted as GPU/campaign evidence.
    code = r'''
import json,sys,torch
from pathlib import Path
from tests.test_research_batch_inference import model,records
from src.eval.research_inference import generate_prediction
from scripts.probe_research_parallel_b1 import generate_checked_record,public_record
torch.set_num_threads(1)
output=[]
for policy in ('a','c'):
    research=model(policy)
    # Keep the production fixed64 generation argument, but cap the tiny fixture
    # to four native generated steps inside its test-only delegate.
    def native(m,r,*,max_new_tokens):
        assert max_new_tokens==64
        assert not {'answers','answer','score','generated_token_ids'} & set(r)
        return generate_prediction(m,r,max_new_tokens=4)
    for row in records():
        cached=native(research,public_record(row),max_new_tokens=64)
        cached.update(record_id=row['record_id'],score=1.)
        checked=generate_checked_record(research,row,cached,generator=native,scorer=lambda r,s:1.)
        output.append({k:checked[k] for k in ('generated_token_ids','prompt_sha256','prefill_tokens','visual_tokens')})
assert not torch.cuda.is_initialized()
Path(sys.argv[1]).write_text(json.dumps(output,sort_keys=True))
'''
    root = Path(__file__).resolve().parents[1]
    baseline = tmp_path / "reference.json"
    serial = subprocess.run([sys.executable, "-c", code, str(baseline)], cwd=root,
                            capture_output=True, text=True, timeout=60)
    assert serial.returncode == 0, serial.stderr[-2000:]
    children = []
    try:
        for index in (0, 1):
            path = tmp_path / f"child{index}.json"
            child = subprocess.Popen([sys.executable, "-c", code, str(path)], cwd=root,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            children.append((child, path))
        for child, path in children:
            _, error = child.communicate(timeout=60)
            assert child.returncode == 0, error[-2000:]
            assert path.read_bytes() == baseline.read_bytes()
        rows = json.loads(baseline.read_text())
        assert len(rows) == 8
        assert any(row['visual_tokens'] > 0 for row in rows)
        assert any(row['visual_tokens'] == 0 for row in rows)
        assert len({tuple(row['generated_token_ids']) for row in rows}) > 1
    finally:
        for child, _ in children:
            if child.poll() is None:
                child.kill(); child.wait(timeout=5)
