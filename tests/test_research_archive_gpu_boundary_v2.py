"""Frozen v1 lifecycle replay plus explicit4f-source/3b-target readonly advice."""
import copy
import importlib.util
import inspect
import json
import mmap
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('archive_boundary_v2_tests',ROOT/'scripts/probe_research_archive_gpu_at_boundary_v2.py')
v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
p=v2.core
_tests_spec=importlib.util.spec_from_file_location('frozen_archive_v1_lifecycle_tests',ROOT/'tests/test_research_archive_gpu_boundary.py')
_tests=importlib.util.module_from_spec(_tests_spec);_tests_spec.loader.exec_module(_tests)
_tests.p=p
_fixture_phase=_tests.fixture_phase
def fixture_phase(root,parent):
    result=_fixture_phase(root,parent);result[0]['job_order']=[p.TARGET];return result
_tests.fixture_phase=fixture_phase
for _name,_value in vars(_tests).items():
    if _name.startswith('test_') or _name in ('lease','info','put','Process'):
        globals()[_name]=_value
# The v2 file is a facade over a private core. Point the disposable child
# fixture's idle stub at that core too; production CLI paths are untouched.
_guardian_fixture=inspect.getsource(_tests.test_real_guardian_survives_coordinator_sigkill_cleans_aux_then_continues_own_parent)
_guardian_fixture=_guardian_fixture.replace('p.gpu_idle=lambda:True;p.guardian','p.core.gpu_idle=lambda:True;p.guardian')
_fixture_globals=dict(vars(_tests))
exec(compile(_guardian_fixture,__file__,'exec'),_fixture_globals)
test_real_guardian_survives_coordinator_sigkill_cleans_aux_then_continues_own_parent=_fixture_globals['test_real_guardian_survives_coordinator_sigkill_cleans_aux_then_continues_own_parent']


def test_frozen_v1_source_and_owner_unchanged():
    assert v2.file_sha(ROOT/v2.V1_SCRIPT)==v2.V1_SHA256
    assert v2.OWNER!='actual4f_archive_gpu_textonly_boundary_v1'
    assert p.TARGET==v2.TARGET and v2.SOURCE!=v2.TARGET
    assert p.HELPER_READY.endswith('_v2.json') and v2.HELPER_READY_SHA256=='6d6e017f6d4ad662a67decf20fb2a169c427f828a1c255b151175eb7bc47fc96'


@pytest.mark.parametrize('active',['earlier','target_training','later'])
def test_v2_waits_earlier_or_target_training_without_binding_any_child(tmp_path,active):
    bound,state,path,child,output=fixture_phase(tmp_path,info());bound['job_order']=['earlier',v2.TARGET,'later']
    name=v2.TARGET if active=='target_training' else active
    state['jobs']={name:{'status':'running'}};put(path,state)
    assert p.phase(tmp_path,bound)==(('missed' if active=='later' else 'waiting'),None)


@pytest.fixture
def advice(tmp_path,monkeypatch):
    from src.utils import research_trained_delta_probe as probe
    source=tmp_path/'runs/research'/v2.SOURCE;target=tmp_path/'runs/research'/v2.TARGET
    for run,names in ((source,('shared','shared')),(target,('best','last'))):
        for name in set(names):
            path=run/'.checkpoint_objects'/('object-'+name)/'state.pt';path.parent.mkdir(parents=True);path.write_bytes(b'owned complete raw bits')
    def lookup(run,allow_archived=False):
        assert allow_archived is False
        run=Path(run);names=('shared','shared') if run==source else ('best','last');roles={}
        owner='source_owner' if run==source else 'target_owner'
        for role,name in zip(('best','last'),names):
            path=run/'.checkpoint_objects'/('object-'+name)/'state.pt'
            roles[role]={'state':str(path),'stat':probe.stat(path),'marker':{'owner':owner,'step':170 if run==source else 100 if role=='best' else 170,
                'training_completed':True,'resumable_optimizer':False}}
        return {'run':str(run),'manager':{'owner':owner},'config':{'recipe':'4f' if run==source else '3b','stage':'SFT'},'roles':roles}
    original=lookup(source)
    saved={r:{'source':x['state'],'stat':x['stat'],'marker':x['marker']} for r,x in original['roles'].items()}
    bound={'source_run':str(source),'target_run':str(target),'plan':{'gate':{'roles':saved}}}
    evidence={'child':{'synthetic':'no real native/GPU process'}}
    monkeypatch.setattr(p,'zero_exit',lambda *a:True);monkeypatch.setattr(p,'gpu_idle',lambda:True)
    host=lambda:{'MemAvailable_bytes':100*2**30,'memory.max':170*2**30,'memory.current':60*2**30}
    return bound,evidence,lookup,host


def test_scoped_advice_unique_source_and_target_state_files_unchanged(advice):
    bound,evidence,lookup,host=advice
    paths={Path(x['state']) for run in (bound['source_run'],bound['target_run']) for x in lookup(run)['roles'].values()}
    before={str(path):path.read_bytes() for path in paths}
    result=v2.postvalidation_cache_advice(bound,evidence,lookup=lookup,host=host)
    assert result['unique_owned_files']==3 and result['source_run']!=result['target_run']
    assert result['reclaimable_cache_credit_assumed_bytes']==0 and result['native_helper_memory_admission_still_required']
    assert result['actual_available_after_bytes']==result['actual_available_before_bytes']
    assert {str(path):path.read_bytes() for path in paths}==before


@pytest.mark.parametrize('reason',['child_alive','gpu_busy','foreign_source_owner','source_stat_changed','unknown_target_path','own_mmap'])
def test_advice_refuses_unverified_or_mapped_files_without_advising(advice,monkeypatch,reason,tmp_path):
    bound,evidence,lookup,host=advice;called=[];mapping=None
    if reason=='child_alive':monkeypatch.setattr(p,'zero_exit',lambda *a:False)
    if reason=='gpu_busy':monkeypatch.setattr(p,'gpu_idle',lambda:False)
    real=lookup
    def changed(run,**kwargs):
        result=real(run,**kwargs)
        if reason=='foreign_source_owner' and str(run)==bound['source_run']:result['manager']['owner']='foreign'
        if reason=='unknown_target_path' and str(run)==bound['target_run']:
            path=tmp_path/'user.file';path.write_bytes(b'never advise');result['roles']['best']['state']=str(path)
        return result
    if reason=='source_stat_changed':bound['plan']['gate']['roles']['best']['stat']['st_size']+=1
    if reason=='own_mmap':
        path=Path(real(bound['source_run'])['roles']['best']['state']);handle=path.open('rb');mapping=mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ)
    try:
        with pytest.raises(ValueError):v2.postvalidation_cache_advice(bound,evidence,lookup=changed,advisor=lambda *a:called.append(a),host=host)
        assert not called
    finally:
        if mapping is not None:mapping.close();handle.close()


def test_advice_role_or_file_mutation_is_detected_after_readonly_operation(advice):
    bound,evidence,lookup,host=advice;changed=[]
    def bad(path,stat):
        if not changed:path.write_bytes(path.read_bytes()+b'mutation');changed.append(str(path))
        return {'synthetic_fault':True}
    with pytest.raises(ValueError,match='changed or raced'):
        v2.postvalidation_cache_advice(bound,evidence,lookup=lookup,advisor=bad,host=host)


def test_unchanged_host_headroom_does_not_create_assumed_credit(advice):
    bound,evidence,lookup,host=advice
    low=lambda:{'MemAvailable_bytes':33*2**30,'memory.max':170*2**30,'memory.current':140*2**30}
    result=v2.postvalidation_cache_advice(bound,evidence,lookup=lookup,advisor=lambda *a:{'readonly':True},host=low)
    assert result['actual_available_after_bytes']==30*2**30
    assert result['reclaimable_cache_credit_assumed_bytes']==0 and result['native_helper_memory_admission_still_required']


def test_gpu_result_validation_uses_source4f_not_target3b_and_labels_both(tmp_path,monkeypatch):
    bound={'run':'/target3b','source_run':'/source4f','target_run':'/target3b'};seen=[]
    def checked(source,*args,**kwargs):seen.append(source);return {'status':'passed','source_authoritative':True}
    monkeypatch.setattr(v2,'_validate_result',checked)
    result=v2.validate_result(bound,tmp_path,'a'*64,worker={},coordinator={},deadline=1)
    assert seen[0]['run']=='/source4f' and bound['run']=='/target3b'
    assert result['source_job']==v2.SOURCE and result['target_job']==v2.TARGET
    assert result['GPU_parity_model_is_source4f_not_target3b']


@pytest.mark.parametrize('change',['report_path','report_sha'])
def test_v2_rejects_any_floating_or_different_source_cpu_report_before_arm(tmp_path,change):
    path=tmp_path/v2.CPU_REPORT;digest=v2.CPU_REPORT_SHA256
    if change=='report_path':path=tmp_path/'other.json'
    else:digest='a'*64
    with pytest.raises(ValueError,match='fixed actual4f'):v2.arm(tmp_path,path,digest,98979)
