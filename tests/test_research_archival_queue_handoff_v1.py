"""Fixed-target metadata/lease tests; no Supervisor service or GPU launch."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts import handoff_research_archival_queue_v1 as h


def info(pid=12,ppid=11,argv=None,state='S'):
    return {'pid':pid,'ppid':ppid,'start_time':123,'argv':argv or ['python'],'cwd':str(h.ROOT),'state':state}


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'ROOT',tmp_path)
    old=SimpleNamespace(process_info=lambda pid:None)
    lease=SimpleNamespace(old=old,stable_identity=lambda row:{key:row[key] for key in ('pid','ppid','start_time','argv','cwd')})
    parent=info(11,1)
    target={'id':h.TARGET,'argv':['--stage','A1','--recipe','3a','--output',str(tmp_path/'run')]}
    spec={'jobs':[{'id':'earlier','argv':[]},target,{'id':'later','argv':[]}]}
    command=[sys.executable,'-u',str(tmp_path/'scripts/run_research_stage.py'),*target['argv']]
    trainer=info(12,11,command)
    old.process_info=lambda pid:trainer if pid==12 else parent
    state={'spec_sha256':h.digest(spec),'status':'running','jobs':{'earlier':{'status':'completed'},h.TARGET:{'status':'running','pid':12,'command':command}}}
    proof={'job_id':h.TARGET,'steps':16,'records':1000,'roles':{'last':{'state_sha256':'a'*64}}}
    monkeypatch.setattr(h,'completed_A1',lambda *args,**kwargs:copy.deepcopy(proof))
    return SimpleNamespace(root=tmp_path,lease=lease,parent=parent,trainer=trainer,target=target,spec=spec,state=state,proof=proof)


def test_fixed_phase_waits_earlier_then_binds_exact_target_and_refuses_late(fixture):
    f=fixture;state=copy.deepcopy(f.state);state['jobs'].pop(h.TARGET)
    assert h.phase(f.lease,f.spec,state,f.parent)==('waiting',None)
    kind,value=h.phase(f.lease,f.spec,f.state,f.parent)
    assert kind=='ready' and value['trainer']['pid']==12 and value['pre_state']==f.state
    state['jobs']['later']={'status':'running'}
    assert h.phase(f.lease,f.spec,state,f.parent)==('missed',None)
    state=copy.deepcopy(f.state);state['jobs'][h.TARGET]['status']='completed'
    assert h.phase(f.lease,f.spec,state,f.parent)==('missed',None)


def test_binding_while_training_waits_for_completed16_without_retarget(fixture,monkeypatch):
    f=fixture
    def missing(*args,**kwargs):raise FileNotFoundError('not completed yet')
    monkeypatch.setattr(h,'completed_A1',missing)
    kind,bound=h.phase(f.lease,f.spec,f.state,f.parent)
    assert kind=='waiting' and bound['start_time']==123
    f.trainer['start_time']=124
    with pytest.raises(ValueError,match='replaced'):h.phase(f.lease,f.spec,f.state,f.parent,bound)


@pytest.mark.parametrize('mutation',['ppid','argv','recipe'])
def test_foreign_trainer_metadata_refuses(fixture,mutation):
    f=fixture
    if mutation=='ppid':f.trainer['ppid']=99
    elif mutation=='argv':f.trainer['argv']=['foreign']
    else:f.state['jobs'][h.TARGET]['command']=[*f.trainer['argv'],'--unexpected']
    with pytest.raises(ValueError):h.phase(f.lease,f.spec,f.state,f.parent)


def test_actual_child_natural_zombie_code0_is_read_without_signals():
    child=subprocess.Popen([sys.executable,'-c','pass'])
    try:
        raw=Path(f'/proc/{child.pid}/stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        expected={'pid':child.pid,'ppid':os.getpid(),'start_time':int(fields[19])}
        deadline=time.monotonic()+10;proof=None
        while proof is None and time.monotonic()<deadline:
            proof=h.zombie_exit(expected)
            if proof is None:time.sleep(.02)
        assert proof['exit_code']==0 and proof['kernel_state']=='Z' and proof['trainer_signaled'] is False
    finally:child.wait(timeout=10)


def test_nonzero_zombie_exit_is_not_normal_completion():
    child=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)'])
    try:
        raw=Path(f'/proc/{child.pid}/stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        expected={'pid':child.pid,'ppid':os.getpid(),'start_time':int(fields[19])}
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            try:
                if h.zombie_exit(expected) is None:time.sleep(.02);continue
            except ValueError:break
            raise AssertionError('nonzero child falsely accepted')
        else:raise AssertionError('no zombie observed')
    finally:child.wait(timeout=10)


def test_only_fixed_A1_progress_transition_and_idempotent_receipt(fixture,monkeypatch):
    f=fixture;path=(f.root/h.SPEC).with_suffix('.state.json');path.parent.mkdir(parents=True)
    pre=copy.deepcopy(f.state);after=copy.deepcopy(pre);after['jobs'][h.TARGET].update(status='interrupted',returncode=0,finished='now')
    path.write_text(json.dumps(after))
    def atomic(p,value):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(value))
    f.lease._atomic_json=atomic
    import scripts.research_queue as base
    monkeypatch.setattr(base,'artifacts_for',lambda job:{'completed':True})
    record={'pre_state':pre,'identity':{'canonical_spec_sha256':h.digest(f.spec)},'trainer':f.trainer,
        'trainer_exit':{'exit_code':0},'completion_hashed':f.proof}
    result=h.recover_completed_training(f.lease,f.spec,record)
    updated=h.read(path)
    assert updated['jobs']['earlier']==pre['jobs']['earlier']
    assert updated['jobs'][h.TARGET]['training_completed'] is True and updated['jobs'][h.TARGET]['status']=='completed'
    assert updated['extra'][h.OWNER]['sha256']==result['sha256']
    assert h.recover_completed_training(f.lease,f.spec,record)==result


def test_unrelated_progress_is_never_migrated(fixture):
    f=fixture;path=(f.root/h.SPEC).with_suffix('.state.json');path.parent.mkdir(parents=True)
    state=copy.deepcopy(f.state);state['jobs']['earlier']['status']='changed';path.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='Unrelated'):
        h.recover_completed_training(f.lease,f.spec,{'pre_state':f.state,'identity':{'canonical_spec_sha256':h.digest(f.spec)}})


def test_private_proven_lease_code_and_late_STOP_are_unchanged_and_fail_closed(monkeypatch,tmp_path):
    lease=h._private_lease()
    assert lease.__name__ not in sys.modules
    nonce='n';path=tmp_path/'lease.json'
    record={'owner':h.OWNER,'lease_nonce':nonce,'lease_revoked':True,'parent_pause_outstanding':True,'pause_deadline_monotonic':time.monotonic()+10}
    path.write_text(json.dumps(record));calls=[]
    monkeypatch.setattr(os,'kill',lambda *args:calls.append(args))
    with pytest.raises(TimeoutError):lease.begin_owned_pause(path,nonce,info())
    assert calls==[]


def test_guardian_can_only_CONT_exact_owned_controller(monkeypatch,tmp_path):
    lease=h._private_lease();calls=[];parent=info(22,1,state='T')
    monkeypatch.setattr(h,'old_controller_info',lambda owner,value:value is parent)
    monkeypatch.setattr(lease.old,'process_info',lambda pid:parent)
    monkeypatch.setattr(os,'kill',lambda *args:calls.append(args))
    record={'owner':h.OWNER,'parent_pause_outstanding':True,'stop_committed':True,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'controller':lease.stable_identity(parent)}
    lease.resume_owned_parent(record)
    assert calls==[(22,signal.SIGCONT)]
    parent['start_time']+=1;lease.resume_owned_parent(record)
    assert calls==[(22,signal.SIGCONT)]


def test_default_CLI_never_operates_or_signals():
    result=subprocess.run([sys.executable,h.SCRIPT],cwd=h.ROOT,capture_output=True,text=True)
    assert result.returncode==2 and json.loads(result.stdout)['signals_sent'] is False
