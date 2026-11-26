import copy

from scripts import research_additive_queue_v2 as fixed


def test_fixed_only_matrix_has12_variants_and6_unchanged_controls():
    old=fixed.native.build_plan();new=fixed.build_plan()
    assert new['fixed_only'] is True
    assert new['new_scientific_configurations']==12
    assert new['total_primary_configurations']==18
    assert len(new['jobs'])==24 and len(new['controls'])==6
    assert new['controls']==old['controls']
    assert new['jobs']==[j for j in old['jobs'] if j['mode']=='fixed']
    assert all(j['branch']['blocks']==2 and j['branch']['mode']=='fixed' for j in new['jobs'])
    assert all(j['argv'][j['argv'].index('--visual-branch')+1]=='fixed' for j in new['jobs'])
    assert {j['learned_masks'] for j in new['jobs']}=={False,True}


def test_migration_reuses_completed_fixed_stages_without_resuming_routed_jobs():
    plan=fixed.build_plan();one=plan['jobs'][0]['id'];two=plan['jobs'][1]['id']
    routed=one.replace('__fixed__','__routed__')
    old={'plan_sha256':fixed.PREVIOUS_SHA,'active_job':routed,'jobs':{
        one:dict(status='completed',completed_unix=123),
        two:dict(status='retry_pending',retry_after_unix=999999,error='prior GPU occupancy'),
        routed:dict(status='reload',pid=1234)}}
    untouched=copy.deepcopy(old)
    state=fixed.migrated_state(plan,old,'new-bound-plan')
    assert old==untouched
    assert state['jobs'][one]==untouched['jobs'][one]
    assert routed not in state['jobs'] and state['active_job'] is None
    assert state['jobs'][two]['status']=='pending' and 'retry_after_unix' not in state['jobs'][two]
    assert state['plan_sha256']=='new-bound-plan'
    assert state['routed_experiments_excluded'] is True


def test_retention_waits_for12_fixed_alignments_and_never_waits_for_excluded_routing():
    from scripts.watch_research_storage_retention_v1 import all_alignment_consumed
    plan=fixed.build_plan()
    alignments=[j for j in plan['jobs'] if j['stage']=='A1']
    state={'jobs':{j['id']:{'status':'completed'} for j in alignments}}
    assert len(alignments)==12 and all_alignment_consumed(plan,state)
    state['jobs'][alignments[-1]['id']]['status']='train'
    assert not all_alignment_consumed(plan,state)
