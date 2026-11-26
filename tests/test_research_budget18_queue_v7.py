from types import SimpleNamespace
import pytest
from scripts import research_budget18_queue_v7 as q


class Fake(q.Queue):
    def __init__(self,jobs,rows):
        self.spec={'jobs':jobs};self.state={'jobs':rows};self.stopped=False;self.child=None;self.child_info=None
        self.handle=SimpleNamespace(close=lambda:None);self.started=[];self.finalized=[];self.evaluated=[]
    def persist(self):pass
    def prune(self):pass
    def adopt_or_train(self,job,row):
        self.started.append(job['id']);row['training_completed']=True
        if job.get('stop_after'):self.stopped=True
    def finalize_training(self,job,row):
        self.finalized.append(job['id'])
        if job.get('fail'):raise RuntimeError('deterministic validator failure')
        row.update(replayed=True,status='trained_pending_evaluation')
    def postprocess(self,job,row):self.evaluated.append(job['id'])


@pytest.fixture(autouse=True)
def native_guards(monkeypatch):
    monkeypatch.setattr(q.v4,'verify_plan',lambda _:None)
    monkeypatch.setattr(q.v4,'resources',lambda **_:None)
    monkeypatch.setattr(q.v4,'validate_completed',lambda _:dict(status='checked'))
    monkeypatch.setattr(q.v5,'verified_replay',lambda _j,r:bool(r.get('replayed')))


def test_failed_validation_continues_independent_training():
    jobs=[dict(id='bad',stage='SFT',requires=[],fail=True),dict(id='independent',stage='A0',requires=[],stop_after=True)]
    queue=Fake(jobs,{'bad':{'training_completed':True},'independent':{}})
    queue.run()
    assert queue.started==['independent'] and queue.state['jobs']['bad']['status']=='failed_retry_pending'
    assert queue.state['jobs']['bad']['retry_after_unix']>q.time.time()


def test_completed_training_reuses_validation_without_training_again():
    queue=Fake([dict(id='done',stage='SFT',requires=[])],{'done':{'training_completed':True}})
    assert queue.run()==0
    assert queue.started==[] and queue.finalized==['done'] and queue.evaluated==['done']
    assert queue.state['status']=='completed'


def test_global_provenance_failure_is_fatal_not_bypassed(monkeypatch):
    def fail(_):raise ValueError('source changed')
    monkeypatch.setattr(q.v4,'verify_plan',fail)
    queue=Fake([dict(id='new',stage='A0',requires=[])],{'new':{}})
    with pytest.raises(ValueError,match='source changed'):queue.run()
    assert queue.started==[] and queue.state['status']=='failed'


def test_retry_backoff_is_bounded():
    assert [q.retry_delay(n) for n in [1,2,3,4,5,20]]==[60,120,240,480,600,600]
