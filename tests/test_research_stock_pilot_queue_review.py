"""Independent tiny CPU/mock review; never launches a trainer, CUDA or signals."""
import copy
import fcntl
import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("stock_queue_independent_review",ROOT/"scripts/research_stock_pilot_queue.py")
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


def bare_queue(tmp_path,monkeypatch,jobs):
    monkeypatch.setattr(controller,"verify_inputs",lambda value:None)
    cls=controller.queue_class();queue=object.__new__(cls)
    queue.spec={"jobs":jobs,"queue_id":"synthetic_review"}
    queue.state={"queue_id":"synthetic_review","jobs":{}}
    queue.stop_before_job=None;queue.interrupted_by=None;queue.retry_failed=False;queue.process=None
    queue.budget=type("Budget",(),{"report":lambda self:{"free_gib":999},"ensure_space":lambda *args:None})()
    queue.log_dir=tmp_path
    queue.prune_finished_intermediates=lambda:None
    queue.lock_handle=(tmp_path/"review.lock").open("w+")
    fcntl.flock(queue.lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    queue.saved=[]
    def persist():
        assert not queue.lock_handle.closed
        queue.saved.append(copy.deepcopy(queue.state))
    queue.persist=persist
    # The run method belongs to the private native-base module.
    native=queue.run.__func__.__globals__
    monkeypatch.setitem(native,"emit",lambda *args,**kwargs:None)
    return queue,native


def test_maintenance_boundary_closes_lock_and_never_starts_job(tmp_path,monkeypatch):
    job={"id":"new_3c_A1","argv":["--output",str(tmp_path/"not_created")],"requires":[]}
    queue,native=bare_queue(tmp_path,monkeypatch,[job]);queue.stop_before_job=job["id"]
    class RefuseProcess:
        def __init__(self,*args,**kwargs):raise AssertionError("No child at maintenance boundary")
    monkeypatch.setattr(native["subprocess"],"Popen",RefuseProcess)
    with pytest.raises(controller.MaintenanceBoundary):queue.run()
    assert queue.lock_handle.closed
    assert queue.saved[-1]["status"]=="awaiting_scheduled_maintenance"
    assert queue.saved[-1]["next_job"]==job["id"] and queue.saved[-1]["jobs"]=={}
    assert not (tmp_path/"not_created").exists()


def test_actual_private_loop_records_rewritten_command_before_fake_spawn(tmp_path,monkeypatch):
    job={"id":"new_3a_A1","argv":["--output",str(tmp_path/"run"),"--stage","A1","--recipe","3a","--allow-legacy-a0-initialization"],"requires":[]}
    original=copy.deepcopy(job)
    queue,native=bare_queue(tmp_path,monkeypatch,[job])
    monkeypatch.setitem(native,"artifacts_for",lambda *args,**kwargs:{"synthetic_complete":True})
    seen=[]
    class CompletedProcess:
        def __init__(self,command,**kwargs):
            seen.append(list(command));self.pid=123456;self.returncode=0
            # This is durable in the queue before Popen receives it.
            assert queue.saved[-1]["jobs"][job["id"]]["command"]==command
        def poll(self):return 0
    monkeypatch.setattr(native["subprocess"],"Popen",CompletedProcess)
    assert queue.run()==0
    assert len(seen)==1 and Path(seen[0][2]).name=="run_research_stock_deterministic.py"
    assert seen[0][3:6]==["--numerical-profile",controller.stock.PROFILE,"--allow-gpu"]
    assert queue.state["jobs"][job["id"]]["training_completed"] is True
    assert job==original and queue.lock_handle.closed


def test_unknown_command_is_refused_without_rewrite():
    with pytest.raises(ValueError,match="Unexpected executable"):
        controller.command_profile(["python","-u","foreign.py"])
