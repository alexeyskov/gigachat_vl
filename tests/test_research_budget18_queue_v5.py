"""Training priority does not masquerade as full experiment completion."""
from types import SimpleNamespace
import pytest
import scripts.research_budget18_queue_v5 as v


def scheduler(monkeypatch):
    q=object.__new__(v.Queue);q.state={'jobs':{}};q.stopped=False;q.child=None;q.child_info=None;q.handle=SimpleNamespace(close=lambda:None);q.persist=lambda:None
    q.spec={'jobs':[{'id':'A','stage':'A0','requires':[]},{'id':'S','stage':'SFT','requires':['A']},
                    {'id':'B','stage':'A0','requires':[]},{'id':'T','stage':'SFT','requires':['B']}]}
    log=[]
    monkeypatch.setattr(v.v4,'verify_plan',lambda spec:spec);monkeypatch.setattr(v.v4,'resources',lambda **kw:None)
    monkeypatch.setattr(v.v4,'validate_completed',lambda j:{'metadata':'verified'})
    def train(j,r):log.append(('train',j['id']));r['training_completed']=True
    def finish(j,r):log.append(('replay',j['id']));r['status']='trained_pending_evaluation'
    def evaluate(j,r):
        assert all(q.state['jobs'][x['id']]['training_completed'] for x in q.spec['jobs'])
        assert r['status']=='trained_pending_evaluation';log.append(('evaluate',j['id']))
    q.adopt_or_train=train;q.finalize_training=finish;q.postprocess=evaluate;q.prune=lambda:None
    return q,log


def test_all_training_and_replay_precede_external512_evaluation(monkeypatch):
    q,log=scheduler(monkeypatch);assert q.run()==0
    assert log==[('train','A'),('train','S'),('replay','S'),('train','B'),('train','T'),('replay','T'),('evaluate','S'),('evaluate','T')]
    assert q.state['jobs']['S']['status']==q.state['jobs']['T']['status']=='completed'


def test_existing_training_is_adopted_without_new_updates(monkeypatch):
    q,log=scheduler(monkeypatch);q.state['jobs']['S']={'training_completed':True,'status':'failed'}
    q.run();assert ('train','S') not in log and ('replay','S') in log and ('evaluate','S') in log


def test_evaluation_failure_keeps_training_durable_and_experiment_incomplete(monkeypatch):
    q,log=scheduler(monkeypatch)
    q.postprocess=lambda j,r:(_ for _ in ()).throw(RuntimeError('owned evaluator error'))
    with pytest.raises(RuntimeError,match='evaluator'):q.run()
    assert all(q.state['jobs'][x['id']]['training_completed'] for x in q.spec['jobs'])
    assert q.state['jobs']['S']['status']=='failed' and q.state['jobs']['T']['status']=='trained_pending_evaluation'


def test_training_failure_never_starts_external_evaluation(monkeypatch):
    q,log=scheduler(monkeypatch);original=q.adopt_or_train
    def fail(j,r):
        if j['id']=='B':raise RuntimeError('owned trainer error')
        original(j,r)
    q.adopt_or_train=fail
    with pytest.raises(RuntimeError,match='trainer'):q.run()
    assert not any(k=='evaluate' for k,_ in log)


def test_pending_SFT_consumer_can_prune_A1_only_after_full_replay(monkeypatch):
    import scripts.research_queue as low
    q=object.__new__(v.Queue);a={'id':'A','stage':'A0','requires':[],'discard_checkpoint_after_dependents':True};b={'id':'B','stage':'A1','requires':['A'],'discard_checkpoint_after_dependents':True};s={'id':'S','stage':'SFT','requires':['B'],'discard_checkpoint_after_dependents':False}
    q.spec={'jobs':[a,b,s]};q.state={'jobs':{'A':{'training_completed':True,'checkpoint_pruned':True},'B':{'training_completed':True},'S':{'training_completed':True,'status':'trained_pending_evaluation'}}};q.persist=lambda:None
    calls=[];monkeypatch.setattr(low,'prune_checkpoint_run',lambda *a:calls.append(a) or {'status':'pruned'});monkeypatch.setattr(v.old,'directory',lambda j:j['id'])
    monkeypatch.setattr(v,'verified_replay',lambda j,r:False);q.prune();assert not calls
    monkeypatch.setattr(v,'verified_replay',lambda j,r:True);q.prune();assert calls==[('B',['S'])] and q.state['jobs']['B']['checkpoint_pruned']


def test_source_A1_retained_until_shared_A0_pruned(monkeypatch):
    q=object.__new__(v.Queue);q.spec={'jobs':[{'id':'B','stage':'A1','requires':['A'],'discard_checkpoint_after_dependents':True}]};q.state={'jobs':{'A':{'training_completed':True},'B':{'training_completed':True}}};q.persist=lambda:None
    q.prune();assert not q.state['jobs']['B'].get('checkpoint_pruned')


def test_immutable_matrix_and_exact_GPU_worker_preserved():
    jobs=v.old.jobs();assert len(jobs)==42
    assert len({(j['backend'],j['encoder'],j['recipe']) for j in jobs if j['stage']=='SFT'})==18
    assert v.old.sha(v.ROOT/'scripts/research_budget18_queue_v4.py')==v.V4_SHA
