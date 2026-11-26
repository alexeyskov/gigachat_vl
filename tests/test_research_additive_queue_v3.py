from scripts import research_additive_queue_v3 as q


def test_bridge_changes_only_training_entry_for8_packed_SFT_jobs():
    old=q.previous.build_plan();new=q.build_plan()
    assert len(new['jobs'])==24 and new['controls']==old['controls']
    affected=[]
    for before,after in zip(old['jobs'],new['jobs'],strict=True):
        saved=dict(after);entry=saved.pop('train_entry',None)
        assert saved==before  # exact manifests/budgets/checkpoint paths/branch math
        if entry:
            assert before['backend'] in ('lfm25','giga31') and before['stage']=='SFT'
            assert before['mode']=='fixed' and entry==q.ENTRY
            affected.append(before['id'])
        cmd=['scripts/run_research_additive_v1.py',*before['argv']]
        actual=q.actual_command(after,cmd,'train')
        assert actual[1:]==cmd[1:]
        assert actual[0]==(entry or cmd[0])
    assert len(affected)==8


def test_gpu_checkpoint_replay_entry_is_unchanged():
    for job in q.build_plan()['jobs']:
        cmd=['scripts/validate_research_additive_v1.py','--run',job['output']]
        assert q.actual_command(job,cmd,'reload')==cmd
