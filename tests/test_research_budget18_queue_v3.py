"""Regression: run the exact native validator parser with controller arguments."""
import argparse,ast,copy,json,math,sys
from pathlib import Path
import pytest
import scripts.research_budget18_queue_v3 as v


def native_validator_main():
    tree=ast.parse((v.ROOT/'scripts/validate_research_checkpoint.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    calls=[]
    def validate(*args,**kwargs):calls.append((args,kwargs));return {'unit_parser_only':True}
    scope={'argparse':argparse,'math':math,'json':json,'__doc__':'native CLI regression','validate_run':validate}
    exec(compile(ast.Module(body=[main],type_ignores=[]),'<exact-native-main>','exec'),scope)
    return scope['main'],calls


@pytest.mark.parametrize('family',['smallthinker4','lfm25','giga31'])
@pytest.mark.parametrize('encoder',['siglip2','qwen25'])
@pytest.mark.parametrize('existing_reserve',[False,True])
def test_future_native_replay_parser_accepts_actual_dispatch(family,encoder,existing_reserve,monkeypatch):
    job=next(j for j in v.old.jobs() if j['backend']==family and j['encoder']==encoder and j['stage']=='SFT' and j['recipe']=='4f')
    before=copy.deepcopy(job);argv=['--run','owned-output','--device','cuda']
    if existing_reserve:argv+=['--reserve-gb','32']
    actual=v.actual_argv(job,'validate',argv)
    assert '--reserve-gb' not in actual and job==before
    # All three frozen backend entrypoints consume numerical-profile themselves.
    at=actual.index('--numerical-profile');del actual[at:at+2]
    native,calls=native_validator_main();monkeypatch.setattr(sys,'argv',['validate',*actual]);native()
    assert calls==[(('owned-output',),{'cpu_structural_only':False,'device':'cuda','native_full_only':False,'atol':.002,'rtol':.0002})]


def test_previous_controller_reproduces_real_validator_error(monkeypatch):
    import scripts.research_budget18_queue_v2 as previous
    job=v.old.jobs()[4];argv=previous.actual_argv(job,'validate',['--run','owned-output','--device','cuda'])
    at=argv.index('--numerical-profile');del argv[at:at+2]
    native,calls=native_validator_main();monkeypatch.setattr(sys,'argv',['validate',*argv])
    with pytest.raises(SystemExit) as exc:native()
    assert exc.value.code==2 and calls==[]


@pytest.mark.parametrize('kind',['train','eval','validate'])
def test_manual_dispatch_keeps_its_original_arguments(kind):
    job=v.old.jobs()[2];argv=['--run','manual-output','--device','cuda']
    assert v.actual_argv(job,kind,argv)==argv


@pytest.mark.parametrize('kind',['train','eval'])
def test_training_and_evaluation_are_identical_to_previous_controller(kind):
    import scripts.research_budget18_queue_v2 as previous
    for job in v.old.jobs():
        argv=list(job['argv']) if kind=='train' else ['--run','owned-output','--reserve-gb','120']
        assert v.actual_argv(job,kind,argv)==previous.actual_argv(job,kind,argv)


def test_worker_uses_own_versioned_source_and_original_state_owner():
    assert v.SCOPE['__file__'].endswith('research_budget18_queue_v3.py')
    assert v.SCOPE['OWNER']==v.old.OWNER
    assert v.old.sha(v.ROOT/v.BASE)==v.BASE_SHA


def test_retry_completed_stage_adopts_without_training(monkeypatch):
    import src.utils.research_completed_r2_v1 as policy
    q=object.__new__(v.Queue);q.persist=lambda:None;calls=[]
    monkeypatch.setattr(policy,'recover_if_published',lambda run:True)
    monkeypatch.setattr(v,'_adopt',lambda self,job,row:calls.append(row.copy()))
    row={};q.adopt_or_train(v.old.jobs()[4],row)
    assert row['training_completed'] and row['model_only_publication_recovered_without_optimizer_updates'] and calls
