"""Mock-only opt-in queue lifecycle; no GPU, services or actual publications."""
import ast
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from scripts import research_archival_parallel_queue_v1 as q
from src.utils.research_checkpoint_archival_queue import ProspectiveSmallArchivalMixin


def job(tmp_path,recipe='3a',stage='SFT'):
    return {'id':f'smallthinker4__siglip2__{recipe}__{stage}_fixture','argv':['--stage',stage,'--recipe',recipe,'--output',str(tmp_path/'run')],
        'compact_completed':True,'evaluation':{'manifest':'fixed512','output':'fixed-output'}}


@pytest.fixture
def setup(tmp_path,monkeypatch):
    queue=q.ArchivalParallelResearchQueue.__new__(q.ArchivalParallelResearchQueue)
    queue.spec={'jobs':[],'reserve_gb':120};queue.state={'jobs':{},'extra':{}}
    queue.activation_bound={'path':str(q.archival.ACTIVATION),'sha256':'a'*64}
    queue._guard=lambda:None;queue.snapshots=[]
    queue.persist=lambda:queue.snapshots.append(copy.deepcopy(queue.state))
    calls=[];kind={'value':'success'};current=job(tmp_path);run=Path(current['argv'][-1]);run.mkdir()
    target=run/'.checkpoint_objects/object-native';target.mkdir(parents=True)
    (target/'state.pt').write_bytes(b'PK\x03\x04synthetic-native-fixture')
    roles={'best':{'object':'object-native','state':str(target/'state.pt'),'marker':{}},
        'last':{'object':'object-native','state':str(target/'state.pt'),'marker':{}}}
    plan={'roles':roles,'metadata_sha256':{'completed.json':'d'*64}}
    journal={'phase':'no_benefit','activation':queue.activation_bound,
        'source_roles':{'best':'object-native','last':'object-native'},'source_object':'object-native',
        'source_state_sha256':q.archival.sha(target/'state.pt')}
    (run/q.archival.JOURNAL).write_text(json.dumps(journal))
    (run/'checkpoint_compaction.json').write_text(json.dumps({'status':'completed','training_completed':True}))
    def compact(self,entry,record):
        calls.append(('raw_compactor',copy.deepcopy(entry)))
        record['compaction']={'status':'completed','training_completed':True}
    monkeypatch.setattr(q.ResearchQueue,'compact_completed_job',compact)
    monkeypatch.setattr(q,'artifacts_for',lambda entry:{'completion':True})
    import scripts.research_queue as native_base
    monkeypatch.setattr(native_base,'artifacts_for',lambda entry,**kwargs:{'completion':True})
    monkeypatch.setattr(q.archival,'eligibility',lambda run:plan)
    published={'status':'completed','protocol':q.archival.OWNER}
    def publish(run,**kwargs):
        calls.append(('publish',kwargs));assert kwargs['activation'] is None
        if kind['value']=='nobenefit':raise q.archival.NoBenefit('exact measured non-saving fixture')
        if kind['value']=='error':raise ValueError('source corruption fixture')
        return published
    monkeypatch.setattr(q.archival,'publish_completed_checkpoint',publish)
    monkeypatch.setattr(q.archival,'validate_completed_archival',lambda run:published)
    return SimpleNamespace(queue=queue,job=current,run=run,plan=plan,journal=journal,calls=calls,kind=kind)


@pytest.mark.parametrize('recipe',q.ELIGIBLE)
def test_transient_job_copy_and_default_fixed_activation_without_SPEC_mutation(setup,recipe):
    setup.job['argv'][3]=recipe
    (setup.run/q.archival.JOURNAL).unlink()
    setup.queue.spec['jobs']=[setup.job];before=copy.deepcopy(setup.queue.spec)
    record={'training_completed':True};setup.queue.compact_completed_job(setup.job,record)
    assert setup.queue.spec==before and 'archive_completed_small' not in setup.job
    compact=next(call for call in setup.calls if call[0]=='raw_compactor')
    assert compact[1]['archive_completed_small'] is True
    assert record['checkpoint_archival']['status']=='completed'
    assert record['training_completed'] is True


@pytest.mark.parametrize('recipe,stage',[('4f','SFT'),('3b','SFT'),('1c','SFT'),('3a','A1'),('4f','A0')])
def test_current_raw_and_warmup_recipes_never_get_archival_flag(setup,recipe,stage):
    setup.job['argv'][1]=stage;setup.job['argv'][3]=recipe
    setup.queue.compact_completed_job(setup.job,{})
    assert [call[0] for call in setup.calls]==['raw_compactor']
    assert 'archive_completed_small' not in setup.calls[0][1]


def test_mixed_partial_transaction_bypasses_old_raw_compactor_before_retry(setup):
    setup.plan['roles']['best']['marker']['checkpoint_archival']={'protocol':q.archival.OWNER}
    record={'training_completed':True};setup.queue.compact_completed_job(setup.job,record)
    assert [call[0] for call in setup.calls]==['publish']
    assert record['checkpoint_archival']['status']=='completed'


def test_only_NoBenefit_all_native_raw_is_explicit_successful_skip(setup):
    setup.kind['value']='nobenefit';record={'training_completed':True}
    setup.queue.compact_completed_job(setup.job,record)
    assert record['checkpoint_archival_skip']['status']=='skipped_no_benefit_native_raw'
    assert record['failure_phase'] is None and record['status']=='training_completed'
    assert 'error' not in record and 'error_type' not in record
    before=len(setup.calls);setup.queue.compact_completed_job(setup.job,record)
    assert len(setup.calls)==before


def test_NoBenefit_mixed_roles_stop_and_preserve_failure(setup):
    setup.kind['value']='nobenefit';setup.plan['roles']['best']['marker']['checkpoint_archival']={'protocol':q.archival.OWNER}
    record={}
    with pytest.raises(ValueError,match='mixed'):setup.queue.compact_completed_job(setup.job,record)
    assert record['training_completed'] is True and record['status']=='training_completed_compaction_failed'
    assert setup.run.joinpath(q.archival.JOURNAL).exists()


def test_every_other_error_keeps_completed_training_and_explicit_retry_status(setup):
    setup.kind['value']='error';record={}
    with pytest.raises(ValueError,match='source corruption'):setup.queue.compact_completed_job(setup.job,record)
    assert record['training_completed'] is True and record['failure_phase']=='checkpoint_compaction'
    assert record['status']=='training_completed_compaction_failed'


def test_already_started_native_evaluation_cannot_be_relabelled(setup):
    with pytest.raises(ValueError,match='already-started'):
        setup.queue.compact_completed_job(setup.job,{'evaluation':{'status':'completed'}})
    assert not setup.calls


def test_closed_activation_precedes_all_parent_initialization(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Parent would create state locks before activation')
    monkeypatch.setattr(q.parent.ParallelGroundingResearchQueue,'__init__',forbidden)
    with pytest.raises(q.archival.ArchivalGateClosed):
        q.ArchivalParallelResearchQueue(q.ROOT/q.SPEC,tmp_path/'state.json',proof_run='unused',activation_sha=None)
    assert not list(tmp_path.iterdir())


def test_activation_fixed_path_and_SHA_rechecked(monkeypatch):
    monkeypatch.setattr(q.archival,'activation_evidence',lambda:{'path':str(q.archival.ACTIVATION.resolve()),'sha256':'b'*64})
    with pytest.raises(q.archival.ArchivalGateClosed):q.activation('a'*64)


def test_private_run_adds_exactly_one_nonnumeric_job_entry_hook():
    original=ast.parse(textwrap.dedent(inspect.getsource(q.ResearchQueue.run)))
    expected=copy.deepcopy(original)
    loops=[node for node in ast.walk(expected) if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='job']
    assert len(loops)==1
    loops[0].body.insert(0,ast.parse('self._archival_job_boundary(job)').body[0])
    native_code=q.ResearchQueue.run.__code__
    private=q._private_native_run()
    assert private.__globals__ is not q.ResearchQueue.run.__globals__
    assert q.ResearchQueue.run.__code__ is native_code
    assert private.__code__.co_filename=='<archival_queue_v1_private_native_run>'
    assert 'self._archival_job_boundary(job)' in ast.unparse(expected)


def test_frozen_run_completed_training_path_never_launches_optimizer_or_trainer(setup,monkeypatch,tmp_path):
    queue=setup.queue;entry=setup.job;entry.pop('evaluation')
    queue.spec={'jobs':[entry],'stop_on_failure':True};queue.state={'queue_id':'synthetic','jobs':{entry['id']:{'training_completed':True,'status':'training_completed_compaction_failed'}}}
    queue.retry_failed=True;queue.interrupted_by=None;queue.lock_handle=(tmp_path/'lock').open('a')
    queue.budget=SimpleNamespace(report=lambda:{'free_gib':200});queue.prune_finished_intermediates=lambda:None
    boundaries=[];queue._archival_job_boundary=lambda job:boundaries.append(job['id'])
    queue.compact_completed_job=lambda job,record:None;queue.evaluate_completed_job=lambda job,record:True
    private=q._private_native_run();private.__globals__['artifacts_for']=lambda job,**kwargs:{'complete':True}
    def forbidden(*args,**kwargs):raise AssertionError('Completed training must never launch subprocess')
    monkeypatch.setattr(private.__globals__['subprocess'],'Popen',forbidden)
    assert private(queue)==0
    assert boundaries==[entry['id']] and queue.state['jobs'][entry['id']]['status']=='completed'


def test_default_CLI_is_closed_without_jobs_or_signals():
    result=subprocess.run([sys.executable,q.SCRIPT],cwd=q.ROOT,capture_output=True,text=True)
    assert result.returncode==2
    value=json.loads(result.stdout)
    assert value['status']=='closed_preflight' and value['jobs_started'] is False and value['signals_sent'] is False


def test_constructor_puts_profile_only_in_extra_and_never_changes_SPEC(tmp_path,monkeypatch):
    spec=q.spec_snapshot(q.ROOT/q.SPEC);before=copy.deepcopy(spec)
    evidence={'path':str(q.archival.ACTIVATION),'sha256':'a'*64};ready={'sha256':'b'*64}
    monkeypatch.setattr(q,'activation',lambda expected:evidence)
    monkeypatch.setattr(q,'cpu_readiness',lambda:ready)
    def parent_init(self,spec_path,state_path,retry_failed,**kwargs):
        self.spec_path=Path(spec_path);self.spec=copy.deepcopy(spec);self.spec_hash=q.SPEC_CANONICAL_SHA256
        self.state={'jobs':{'old':{'status':'completed'}},'extra':{}}
        self.lock_handle=(tmp_path/'lock').open('a')
    monkeypatch.setattr(q.parent.ParallelGroundingResearchQueue,'__init__',parent_init)
    obj=q.ArchivalParallelResearchQueue(q.ROOT/q.SPEC,tmp_path/'state',proof_run='actual-proof',activation_sha='a'*64)
    try:
        assert obj.spec==before and q.digest(obj.spec)==q.SPEC_CANONICAL_SHA256
        assert obj.state['jobs']=={'old':{'status':'completed'}}
        assert obj.state['extra'][q.OWNER]['activation']==evidence
        assert 'trained_small_archival_activation' not in obj.spec
    finally:obj.lock_handle.close()
