"""Performance-only batching policy preserves completed/resumable configurations."""
import copy
import pytest
import scripts.research_budget18_queue_v4 as v


@pytest.mark.parametrize('family',['smallthinker4','lfm25','giga31'])
@pytest.mark.parametrize('stage',['A0','A1','SFT'])
def test_new_training_uses_effective64_initial_microbatch64(family,stage,tmp_path,monkeypatch):
    job=next(j for j in v.old.jobs() if j['backend']==family and j['stage']==stage and j['id'] not in v.MANUAL)
    monkeypatch.setattr(v.old,'directory',lambda j:tmp_path)
    before=copy.deepcopy(job);actual=v.actual_argv(job,'train',job['argv'])
    assert v.old.value(actual,'--batch-size')=='64'
    assert v.old.value(actual,'--effective-batch-size')=='64'
    assert job==before


@pytest.mark.parametrize('size',[1,16,32,64])
def test_existing_initial_batch_config_kept_for_resume(size,tmp_path,monkeypatch):
    job=v.old.jobs()[6];monkeypatch.setattr(v.old,'directory',lambda j:tmp_path)
    v.old.atomic(tmp_path/'resolved_config.json',{'batch_size':size})
    monkeypatch.setattr(v,'PRESERVED_BATCH_CONFIGS',{job['id']:v.old.sha(tmp_path/'resolved_config.json')})
    assert v.old.value(v.actual_argv(job,'train',job['argv']),'--batch-size')==str(size)


@pytest.mark.parametrize('kind',['eval','validate'])
def test_all_postprocessing_args_identical_to_CLI_fixed_V3(kind):
    import scripts.research_budget18_queue_v3 as old
    for job in v.old.jobs():
        args=['--run','owned-output','--device','cuda']
        assert v.actual_argv(job,kind,args)==old.actual_argv(job,kind,args)


def test_three_historical_manual_training_args_preserved():
    for job in v.old.jobs():
        if job['id'] in v.MANUAL:assert v.actual_argv(job,'train',job['argv'])==job['argv']


def test_new_worker_source_is_own_version():
    assert v.SCOPE['__file__'].endswith('research_budget18_queue_v4.py')
    assert v.old.sha(v.ROOT/v.BASE)==v.BASE_SHA


def test_unknown_old_microbatch_rejected(tmp_path,monkeypatch):
    job=v.old.jobs()[6];monkeypatch.setattr(v.old,'directory',lambda j:tmp_path)
    v.old.atomic(tmp_path/'resolved_config.json',{'batch_size':32})
    with pytest.raises(ValueError,match='preserved config'):v.initial_microbatch(job)


def test_completion_checks_actual_initial_batch_and_keeps_immutable_intent(tmp_path,monkeypatch):
    job=next(j for j in v.old.jobs() if j['backend']=='smallthinker4' and j['encoder']=='qwen25' and j['stage']=='A1' and j['recipe']=='1b');original=copy.deepcopy(job)
    monkeypatch.setattr(v.old,'directory',lambda j:tmp_path);v.old.atomic(tmp_path/'resolved_config.json',{'batch_size':64,'reserve_gb':32});calls=[]
    monkeypatch.setattr(v,'_future_completed',lambda j:calls.append(j) or {'proof':True})
    assert v.validate_completed(job)=={'proof':True}
    assert v.old.value(calls[0]['argv'],'--batch-size')=='64' and job==original
