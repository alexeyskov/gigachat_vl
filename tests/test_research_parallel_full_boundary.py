"""Controller/lease lifecycle only: no real signals, subprocesses or GPU calls."""
import json
import copy
from pathlib import Path
import signal
import sys
import types
import ast
import os
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from scripts import eval_research_parallel_b1_at_boundary as c
REAL_START_WATCHDOG=c.start_watchdog
REAL_CONTEXTS=c.contexts


@pytest.fixture
def window(tmp_path,monkeypatch):
    spec=json.loads((Path(__file__).parents[1]/c.SPEC).read_text())
    source=next(j for j in spec['jobs'] if j['id']==c.SOURCE)
    target=next(j for j in spec['jobs'] if j['id']==c.TARGET)
    def write(path,value):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    write(c.SPEC,spec)
    target_run=c.at(tmp_path,c.old.argv_value(target['argv'],'--output'))
    output=target_run/'evaluation/grounding_v1/text_only'
    command=['/venv/main/bin/python','-u',str(tmp_path/'scripts/eval_research_checkpoint.py'),
        '--run',str(target_run),'--manifest',str(c.at(tmp_path,target['evaluation']['manifest'])),
        '--output',str(output),'--checkpoint','best','--max-new-tokens','64',
        '--image-ablation','text_only','--ablation-seed','42']
    parent={'pid':c.CONTROLLER,'start_time':123,'state':'S','ppid':1,'cwd':str(tmp_path),
        'argv':['python','-u',c.old.CONTROLLER_SCRIPT,'--spec',c.SPEC,'--retry-failed']}
    child={'pid':c.TARGET_PID,'start_time':456,'state':'S','ppid':c.CONTROLLER,'cwd':str(tmp_path),'argv':command}
    coordinator={'pid':c.os.getpid(),'start_time':789,'state':'S','cwd':str(tmp_path),'argv':['python',c.SCRIPT,'--workers','4']}
    infos={c.CONTROLLER:parent,c.TARGET_PID:child,c.os.getpid():coordinator}
    _,snapshot=c.spec_snapshot(tmp_path)
    state={'status':'running','spec_sha256':snapshot['canonical_spec_sha256'],'jobs':{
        c.SOURCE:{'status':'completed'},c.TARGET:{'status':'grounding_evaluating','training_completed':True,
        'grounding_status':'running','active_grounding_variant':'text_only',
        'grounding_evaluations':{'text_only':{'status':'running','pid':c.TARGET_PID,'command':command,'output':str(output)}}}}}
    state_path=Path(c.SPEC).with_suffix('.state.json');write(state_path,state)
    bound={'source4q':{'run':'fixed4q','manifest':'fixed512','native_cache':'original4qnative','checkpoint_identity':{'sha256':'source'}},
           'target':{'run':'fixed1b'},'output':str(tmp_path/'parallel_none')}
    proof={'own':'passed','full512':'passed'};signals=[]
    def kill(pid,sig):
        signals.append((pid,sig))
        if pid==c.CONTROLLER and sig==signal.SIGSTOP:parent['state']='T'
        if pid==c.CONTROLLER and sig==signal.SIGCONT:parent['state']='S'
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'boot_id',lambda:'boot')
    monkeypatch.setattr(c,'process_info',lambda pid:infos.get(pid))
    monkeypatch.setattr(c.boundary,'process_info',lambda pid:infos.get(pid))
    monkeypatch.setattr(c.boundary,'wrapper',lambda root,pid:infos.get(pid))
    monkeypatch.setattr(c,'readiness',lambda root:proof)
    monkeypatch.setattr(c,'contexts',lambda *a:bound)
    monkeypatch.setattr(c.boundary,'verify_completed_variant',lambda *a:None)
    monkeypatch.setattr(c.old,'wait_idle',lambda *a:None)
    monkeypatch.setattr(c,'host_available_bytes',lambda:100*2**30)
    monkeypatch.setattr(c.DiskBudget,'ensure_space',lambda *a:None)
    monkeypatch.setattr(c.os,'kill',kill)
    monkeypatch.setattr(c,'start_watchdog',lambda *a:SimpleNamespace(wait=lambda **k:None))
    def complete():
        values={'summary.json':{'complete_manifest':True,'records':512,'manifest_records':512},
            'progress.json':{'event':'completed','completed_records':512},'protocol.json':{},
            'predictions.jsonl':{'mock':'512 rows are validated by separate full verifier'},'transformed_manifest.jsonl':{}}
        for n,v in values.items():write(output/n,v)
    return SimpleNamespace(root=tmp_path,spec=spec,source=source,target=target,parent=parent,child=child,
        coordinator=coordinator,infos=infos,state=state,state_path=state_path,write=write,complete=complete,
        bound=bound,proof=proof,signals=signals)


def test_completed512_pauses_only_controller_then_cont_and_keeps_queue_bytes(window,monkeypatch):
    w=window;w.complete();before=(w.root/w.state_path).read_bytes()
    def execute(root,bound,workers,path,record,deadline):
        assert c.read_json(path)['parent_pause_outstanding'] is True
        record['attempted']=True
        return {'status':'completed','records':512,'worker_count':4}
    monkeypatch.setattr(c,'execute',execute)
    result=c.run(w.root,4)
    assert result['status']=='completed' and result['attempted']
    assert w.signals==[(c.CONTROLLER,signal.SIGSTOP),(c.CONTROLLER,signal.SIGCONT)]
    assert (w.root/w.state_path).read_bytes()==before
    assert c.read_json(c.receipt_path(w.root,4))['parent_pause_outstanding'] is False


def test_exact_current_child_missed_or_pid_reuse_never_signals(window):
    w=window;w.complete()
    status,e=c.phase(w.root,w.spec,w.state,w.target,w.parent)
    assert status=='ready'
    w.child['start_time']=999
    assert c.phase(w.root,w.spec,w.state,w.target,w.parent,e)[0]=='missed'
    w.state['jobs'][c.TARGET]['active_grounding_variant']='blank';w.write(w.state_path,w.state)
    assert c.run(w.root,4)['status']=='skipped' and not w.signals


def test_no_ready_window_advance_skips_without_pause(window,monkeypatch):
    w=window;w.complete();monkeypatch.setattr(c,'readiness',lambda root:None)
    def advance(seconds):
        w.state['jobs'][c.TARGET]['status']='completed';w.write(w.state_path,w.state)
    monkeypatch.setattr(c.time,'sleep',advance)
    result=c.run(w.root,4)
    assert result['reason']=='explicit_current_text_only_window_missed' and not w.signals


def test_phase_race_after_stop_continues_without_worker(window,monkeypatch):
    w=window;w.complete();original=c.os.kill
    def shifted(pid,sig):
        original(pid,sig)
        if sig==signal.SIGSTOP:
            w.state['jobs'][c.TARGET]['status']='completed';w.write(w.state_path,w.state)
    monkeypatch.setattr(c.os,'kill',shifted)
    monkeypatch.setattr(c,'execute',lambda *a:pytest.fail('No worker after missed boundary'))
    result=c.run(w.root,4)
    assert result['reason']=='phase_advanced_at_pause'
    assert w.signals==[(c.CONTROLLER,signal.SIGSTOP),(c.CONTROLLER,signal.SIGCONT)]


@pytest.mark.parametrize('error',[RuntimeError('worker failed'),InterruptedError('signal'),TimeoutError('timeout')])
def test_all_worker_failures_resume_owned_parent(window,monkeypatch,error):
    w=window;w.complete()
    monkeypatch.setattr(c,'execute',lambda *a:(_ for _ in ()).throw(error))
    with pytest.raises(type(error)):c.run(w.root,4)
    assert w.signals[-1]==(c.CONTROLLER,signal.SIGCONT)
    assert c.read_json(c.receipt_path(w.root,4))['parent_pause_outstanding'] is False


def lease(w):
    return {'owner':c.OWNER,'boot_id':'boot','lease_nonce':'nonce','parent_pause_outstanding':True,
        'controller_pid':c.CONTROLLER,'controller_start_time':123,'controller_command_sha256':c.command_sha(w.parent['argv']),
        'coordinator_pid':w.coordinator['pid'],'coordinator_start_time':789,
        'coordinator_command_sha256':c.command_sha(w.coordinator['argv']),'pause_deadline_epoch':c.time.time()+1000,
        'pause_deadline_monotonic':c.time.monotonic()+1000}


def test_crash_recovery_runs_before_readiness_and_does_not_relaunch(window,monkeypatch):
    w=window;w.parent['state']='T';record=lease(w);record['attempted']=True
    w.write(c.receipt_path(w.root,4),record)
    with pytest.raises(RuntimeError,match='one-shot'):c.run(w.root,4)
    assert w.signals==[(c.CONTROLLER,signal.SIGCONT)]


def test_watchdog_coordinator_death_and_deadline_resume_only_receipt_parent(window):
    w=window;w.parent['state']='T';record=lease(w);path=c.receipt_path(w.root,4)
    w.write(path,record);w.infos[w.coordinator['pid']]=None
    # Watchdog's ownprocess identity is required; simulate a different live PID.
    w.infos[c.os.getpid()]=dict(w.coordinator)
    record['coordinator_pid']=999999;w.write(path,record)
    c.watchdog(w.root,path,'nonce')
    assert w.signals==[(c.CONTROLLER,signal.SIGCONT)]
    assert not c.read_json(path)['parent_pause_outstanding']
    w.signals.clear();w.parent['state']='T';record=lease(w);record['pause_deadline_monotonic']=0;w.write(path,record)
    c.watchdog(w.root,path,'nonce')
    assert w.signals==[(c.CONTROLLER,signal.SIGCONT)]


def test_foreign_nonce_parent_reuse_and_boot_change_are_never_signalled(window):
    w=window;w.parent['state']='T';record=lease(w)
    for changes in [{'owner':'foreign'},{'controller_start_time':999},{'boot_id':'other'}]:
        assert c.resume_parent(w.root,dict(record,**changes))!='resumed'
    path=c.receipt_path(w.root,4);w.write(path,record);c.watchdog(w.root,path,'foreign_nonce')
    assert not w.signals


def test_guardian_start_failure_does_not_stop_parent(window,monkeypatch):
    w=window;w.complete()
    monkeypatch.setattr(c,'start_watchdog',lambda *a:(_ for _ in ()).throw(RuntimeError('guardian not ready')))
    with pytest.raises(RuntimeError,match='guardian'):c.run(w.root,4)
    assert not w.signals


def test_worker_cli_is_fixed4q_none_native512_workers4_and800seconds(window):
    command=c.worker_command(window.root,window.bound,4,800)
    assert command[command.index('--run')+1]=='fixed4q'
    assert command[command.index('--native-cache')+1]=='original4qnative'
    assert command[command.index('--workers')+1]=='4'
    assert command[command.index('--timeout-seconds')+1]=='800'
    assert command[command.index('--image-ablation')+1]=='none'
    assert '--resume' not in command


def test_foreign_receipt_not_overwritten(window):
    w=window;p=w.write(c.receipt_path(w.root,4),{'owner':'foreign'});before=p.read_bytes()
    with pytest.raises(ValueError,match='Foreign receipt'):c.run(w.root,4)
    assert p.read_bytes()==before and not w.signals


def test_corrupt_ready_is_not_old32_success(tmp_path):
    path=tmp_path/c.READY;path.parent.mkdir(parents=True);path.write_text(json.dumps({'status':'passed','tests_passed':32,'checks':{}}))
    with pytest.raises(ValueError,match='readiness'):c.readiness(tmp_path)


def future_target(w,target=c.TARGET1A):
    job=next(j for j in w.spec['jobs'] if j['id']==target)
    w.state['jobs'][c.TARGET]['status']='completed'
    row=copy.deepcopy(w.state['jobs'][c.TARGET]);row['status']='running'
    w.state['jobs'][target]=row
    return job,row


def future_shuffle(w,job,row):
    run=c.at(w.root,c.old.argv_value(job['argv'],'--output'));output=run/'evaluation/grounding_v1/shuffle'
    command=list(w.child['argv'])
    command[command.index('--run')+1]=str(run);command[command.index('--output')+1]=str(output)
    command[command.index('--image-ablation')+1]='shuffle'
    child=dict(w.child,pid=999991,start_time=888,argv=command)
    w.infos[child['pid']]=child
    row.update(status='grounding_evaluating',active_grounding_variant='shuffle',grounding_status='running',
        grounding_evaluations={'shuffle':{'status':'running','pid':child['pid'],'command':command,'output':str(output)}})
    return child,output


def test_explicit_future1a_waits_training_ordinary_and_gpu_replay_without_pause(window):
    w=window;job,row=future_target(w)
    for phase in ['running','evaluating','checkpoint_validating']:
        row['status']=phase
        assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('waiting',None)
    assert not w.signals
    bad=dict(w.state,spec_sha256='wrong')
    with pytest.raises(ValueError,match='State/spec'):c.phase(w.root,w.spec,bad,job,w.parent,variant='shuffle',required_pid=None)


def test_future1a_binds_exact_shuffle_pid_once_and_never_retargets(window):
    w=window;job,row=future_target(w);child,_=future_shuffle(w,job,row)
    status,e=c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)
    assert status=='waiting' and e['evaluation_pid']==child['pid'] and e['evaluation_start_time']==888
    child['start_time']=999
    assert c.phase(w.root,w.spec,w.state,job,w.parent,e,variant='shuffle',required_pid=999991)[0]=='missed'
    assert not w.signals


def test_future1a_waits_post_replay_before_first_variant_assignment(window):
    w=window;job,row=future_target(w)
    row.update(status='grounding_evaluating',grounding_status='running',training_completed=True,
        checkpoint_validation={'status':'completed'},grounding_evaluations={})
    row.pop('active_grounding_variant',None)
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('waiting',None)
    assert not w.signals


def test_future1a_exact_shuffle_receipt_before_pid_waits_then_binds_once(window):
    w=window;job,row=future_target(w);child,_=future_shuffle(w,job,row)
    receipt=row['grounding_evaluations']['shuffle'];pid=receipt.pop('pid')
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('waiting',None)
    receipt['pid']=pid
    status,evidence=c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)
    assert status=='waiting' and evidence['evaluation_pid']==pid and evidence['evaluation_start_time']==888
    receipt.pop('pid')
    with pytest.raises(ValueError,match='identity missing'):
        c.phase(w.root,w.spec,w.state,job,w.parent,evidence,variant='shuffle',required_pid=pid)
    assert not w.signals


@pytest.mark.parametrize('bad_pid',[None,True,'999991',-1])
def test_future1a_malformed_present_pid_is_not_prebinding_wait(window,bad_pid):
    w=window;job,row=future_target(w);future_shuffle(w,job,row)
    row['grounding_evaluations']['shuffle']['pid']=bad_pid
    if bad_pid == -1:
        assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('missed',None)
    else:
        with pytest.raises(ValueError,match='identity missing'):
            c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)
    assert not w.signals


@pytest.mark.parametrize('flag,value',[('--image-ablation','blank'),('--checkpoint','last'),('--max-new-tokens','32'),('--manifest','foreign.jsonl')])
def test_future1a_pre_spawn_receipt_validates_exact_command(window,flag,value):
    w=window;job,row=future_target(w);future_shuffle(w,job,row)
    receipt=row['grounding_evaluations']['shuffle'];receipt.pop('pid')
    receipt['command'][receipt['command'].index(flag)+1]=value
    with pytest.raises(ValueError,match='Pre-spawn shuffle command/output'):
        c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)
    assert not w.signals


@pytest.mark.parametrize('variant',['blank','text_only'])
def test_future1a_later_variant_before_binding_still_misses(window,variant):
    w=window;job,row=future_target(w);future_shuffle(w,job,row)
    row['active_grounding_variant']=variant
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('missed',None)
    assert not w.signals


def test_future1a_missing_variant_with_previous_grounding_receipt_is_not_fresh_prefix(window):
    w=window;job,row=future_target(w);future_shuffle(w,job,row)
    row.pop('active_grounding_variant')
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('missed',None)
    assert not w.signals


def test_default1b_stays_skipped_while_explicit1a_is_running(window):
    w=window;future_target(w);w.write(w.state_path,w.state)
    result=c.run(w.root,4)
    assert result['status']=='skipped' and not w.signals
    assert result['target_job']==c.TARGET


def test_future1a_requires_explicit_matching_variant_and_separate_receipt(window):
    w=window
    with pytest.raises(ValueError,match='explicit'):c.run(w.root,4,target_id=c.TARGET1A,variant='none')
    assert c.receipt_path(w.root,4,c.TARGET1A,'shuffle')!=c.receipt_path(w.root,4)
    with pytest.raises(ValueError,match='target'):c.exact_jobs(w.root,w.spec,'arbitrary_future_job')
    assert not w.signals


def test_future1c_explicit_recipe_and_distinct_shuffle_receipt(window):
    w=window;source,job=c.exact_jobs(w.root,w.spec,c.TARGET1C)
    assert source['id']==c.SOURCE and c.old.argv_value(job['argv'],'--recipe')=='1c'
    assert c.TARGET_RECIPES=={c.TARGET:'1b',c.TARGET1A:'1a',c.TARGET1C:'1c'}
    assert c.TARGET_VARIANTS[c.TARGET1C]==('shuffle',)
    own=c.receipt_path(w.root,4,c.TARGET1C,'shuffle')
    assert own.name=='parallel_b1_full_boundary_1c_shuffle_w4.json'
    assert own not in [c.receipt_path(w.root,4,c.TARGET1A,v) for v in c.TARGET_VARIANTS[c.TARGET1A]]
    for variant in ('blank','text_only','none'):
        with pytest.raises(ValueError,match='explicit'):c.receipt_path(w.root,4,c.TARGET1C,variant)
        with pytest.raises(ValueError,match='explicit'):c.phase(w.root,w.spec,w.state,job,w.parent,variant=variant,required_pid=None)
    changed=copy.deepcopy(w.spec);wrong=next(j for j in changed['jobs'] if j['id']==c.TARGET1C)
    wrong['argv'][wrong['argv'].index('--recipe')+1]='1a'
    with pytest.raises(ValueError,match='recipe'):c.exact_jobs(w.root,changed,c.TARGET1C)
    assert not w.signals


def test_future1c_waits_earlier_jobs_training_ordinary_and_gpu_replay(window):
    w=window;job,row=future_target(w,c.TARGET1C)
    row['status']='pending';w.state['jobs'][c.TARGET1A]={'status':'grounding_evaluating'}
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('waiting',None)
    w.state['jobs'][c.TARGET1A]['status']='completed'
    for status in ('running','evaluating','checkpoint_validating'):
        row['status']=status
        assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('waiting',None)
    row.update(status='grounding_evaluating',grounding_status='running',training_completed=True,
        checkpoint_validation={'status':'completed'},grounding_evaluations={})
    row.pop('active_grounding_variant',None)
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('waiting',None)
    assert not w.signals


def test_future1c_waits_known_between_job_gap_but_not_failed_or_later_history(window):
    w=window;job,row=future_target(w,c.TARGET1C);row.clear()
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('waiting',None)
    w.state['jobs'][c.TARGET1A]={'status':'failed'}
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('missed',None)
    w.state['jobs'][c.TARGET1A]={'status':'completed'}
    later=w.spec['jobs'][[entry['id'] for entry in w.spec['jobs']].index(c.TARGET1C)+1]['id']
    w.state['jobs'][later]={'status':'completed'}
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('missed',None)
    assert not w.signals


def test_future1c_prespawn_then_exact_shuffle_pid_binds_once(window):
    w=window;job,row=future_target(w,c.TARGET1C);child,_=future_shuffle(w,job,row)
    receipt=row['grounding_evaluations']['shuffle'];pid=receipt.pop('pid')
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('waiting',None)
    receipt['pid']=pid
    status,evidence=c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)
    assert status=='waiting' and evidence['safe_job']==c.TARGET1C and evidence['evaluation_pid']==pid
    assert evidence['evaluation_start_time']==child['start_time']
    foreign=dict(evidence,safe_job=c.TARGET1A)
    assert c.phase(w.root,w.spec,w.state,job,w.parent,foreign,variant='shuffle',required_pid=pid)==('missed',None)
    child['start_time']+=1
    assert c.phase(w.root,w.spec,w.state,job,w.parent,evidence,variant='shuffle',required_pid=pid)==('missed',None)
    assert not w.signals


@pytest.mark.parametrize('bad_pid',[None,True,'999991',-1])
def test_future1c_malformed_present_pid_never_waits_or_retargets(window,bad_pid):
    w=window;job,row=future_target(w,c.TARGET1C);future_shuffle(w,job,row)
    row['grounding_evaluations']['shuffle']['pid']=bad_pid
    if bad_pid==-1:
        assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('missed',None)
    else:
        with pytest.raises(ValueError,match='identity missing'):c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)
    assert not w.signals


@pytest.mark.parametrize('late',['blank','text_only','completed'])
def test_future1c_late_window_skips_without_following_next_variant(window,late):
    w=window;job,row=future_target(w,c.TARGET1C);future_shuffle(w,job,row)
    if late=='completed':row['status']='completed'
    else:row['active_grounding_variant']=late
    w.write(w.state_path,w.state)
    result=c.run(w.root,4,target_id=c.TARGET1C,variant='shuffle')
    assert result['status']=='skipped' and result['reason']=='explicit_1c_shuffle_window_missed'
    assert result['target_job']==c.TARGET1C and result['variant']=='shuffle' and result['attempted'] is False
    assert not w.signals


def test_future1c_context_binds_recipe1c_not_legacy1a(window,monkeypatch):
    w=window;job,row=future_target(w,c.TARGET1C);recipes=[]
    def context(root,spec,selected,state):
        run=c.at(root,c.old.argv_value(selected['argv'],'--output'))
        return {'run':str(run),'output':'unused'}
    monkeypatch.setattr(c.boundary,'context',context)
    def metadata(root,value,recipe):recipes.append((Path(value['run']).name,recipe));return {'recipe':recipe}
    monkeypatch.setattr(c.old,'run_metadata',metadata)
    value=REAL_CONTEXTS(w.root,w.spec,w.state,w.source,job,4)
    assert recipes==[(c.SOURCE,'4q'),(c.TARGET1C,'1c')]
    assert value['target']['run_metadata']['recipe']=='1c'
    assert Path(value['output']).is_relative_to(Path(value['source4q']['run']))
    assert value['source4q']['run']!=value['target']['run']


def test_watchdog_ready_handshake_binds_birth_argv_and_nonce(window,monkeypatch):
    w=window;record=lease(w);path=c.receipt_path(w.root,4);fake=SimpleNamespace(pid=999992,poll=lambda:None)
    def launch(command,**kwargs):
        info={'pid':fake.pid,'start_time':990,'argv':command};w.infos[fake.pid]=info
        w.write(Path(str(path)+'.nonce.watchdog.json'),{'owner':c.OWNER,'nonce':'nonce','pid':fake.pid,
            'start_time':990,'command_sha256':c.command_sha(command)})
        return fake
    monkeypatch.setattr(c.subprocess,'Popen',launch)
    assert REAL_START_WATCHDOG(w.root,path,record) is fake
    assert not w.signals


def test_watchdog_changed_birth_cannot_acknowledge_pause(window,monkeypatch):
    w=window;record=lease(w);path=c.receipt_path(w.root,4);fake=SimpleNamespace(pid=999992,poll=lambda:None)
    def launch(command,**kwargs):
        w.infos[fake.pid]={'pid':fake.pid,'start_time':990,'argv':command}
        w.write(Path(str(path)+'.nonce.watchdog.json'),{'owner':c.OWNER,'nonce':'nonce','pid':fake.pid,
            'start_time':989,'command_sha256':c.command_sha(command)})
        return fake
    monkeypatch.setattr(c.subprocess,'Popen',launch)
    with pytest.raises(ValueError,match='watchdog readiness'):REAL_START_WATCHDOG(w.root,path,record)
    assert not w.signals


def test_result_requires_actual_full512_tokens_and_each_model64_guard(window,monkeypatch):
    module=types.ModuleType('scripts.eval_research_checkpoint_parallel_b1')
    module.OWNER='producer'
    producer={'owner':module.OWNER,'status':'completed','records':512,'worker_count':4,
        'full512_previous_native_cache_comparison':True,'per_model_parallel_native64_guard_verified':True,
        'protocol_sha256':'protocol','summary_sha256':'summary'}
    window.write(Path(window.bound['output'])/'execution_summary.json',producer)
    result={'status':'passed','records':512,'worker_count':4,
        'full512_exact_previous_native_comparison':True,'per_model_native64_guard_verified':True,
        'checkpoint_identity':window.bound['source4q']['checkpoint_identity'],
        'protocol_sha256':'protocol','summary_sha256':'summary'}
    module.validate_completed_output=lambda *a:result
    monkeypatch.setitem(sys.modules,module.__name__,module)
    import scripts
    monkeypatch.setattr(scripts,'eval_research_checkpoint_parallel_b1',module,raising=False)
    assert c.validate_result(window.bound,4)==result
    result['full512_exact_previous_native_comparison']=False
    with pytest.raises(ValueError,match='Raw full512'):c.validate_result(window.bound,4)
    result['full512_exact_previous_native_comparison']=True;result['per_model_native64_guard_verified']=False
    with pytest.raises(ValueError,match='Raw full512'):c.validate_result(window.bound,4)


def test_recover_never_signals_target_child_or_reused_worker(window):
    w=window;record=lease(w)
    record.update(worker_pid=c.TARGET_PID,worker_start_time=456,
        worker_command=w.child['argv'],worker_command_sha256=c.command_sha(w.child['argv']))
    c.recover_worker(w.root,record)
    assert not w.signals


def test_real_ready_schema_rejects_old32_or_changed_tested_source(tmp_path,monkeypatch):
    own_names=(c.SCRIPT,c.TEST,'scripts/eval_research_batch4_at_boundary.py',
        'scripts/probe_parallel_b1_at_boundary.py','scripts/research_queue.py')
    full_names=(c.FULL_SCRIPT,'src/eval/research_parallel_protocol.py',
        'tests/test_research_parallel_evaluator.py','tests/test_research_parallel_protocol.py')
    def proof(names,checks):
        sources={}
        for name in names:
            p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture '+name)
            sources[name]=c.digest(p.read_bytes())
        return {'status':'passed','tests_passed':1,'test_command':'fixture schema test',
                'source_sha256':sources,'checks':{k:True for k in checks}}
    own=proof(own_names,c.CHECKS)
    full=proof(full_names,['full512_raw_token_score_provenance_revalidation'])
    full.update(owner='fullowner',worker_counts=[2,4],full_manifest_records=512,max_new_tokens=64,actual_gpu_validated=False)
    for name,value in [(c.READY,own),(c.FULL_READY,full)]:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    own['upstream_full_readiness_sha256']=c.digest((tmp_path/c.FULL_READY).read_bytes())
    (tmp_path/c.READY).write_text(json.dumps(own))
    module=types.ModuleType('scripts.eval_research_checkpoint_parallel_b1');module.OWNER='fullowner'
    module.validate_completed_output=lambda *a:None
    monkeypatch.setitem(sys.modules,module.__name__,module)
    import scripts
    monkeypatch.setattr(scripts,'eval_research_checkpoint_parallel_b1',module,raising=False)
    assert set(c.readiness(tmp_path))=={c.READY,c.FULL_READY}
    full['full_manifest_records']=32;(tmp_path/c.FULL_READY).write_text(json.dumps(full))
    with pytest.raises(ValueError,match='full512'):c.readiness(tmp_path)
    full['full_manifest_records']=512;(tmp_path/c.FULL_READY).write_text(json.dumps(full))
    (tmp_path/c.FULL_SCRIPT).write_text('changed')
    with pytest.raises(ValueError):c.readiness(tmp_path)


def test_ready_rejects_changed_upstream_binding_even_if_its_checks_still_pass(tmp_path,monkeypatch):
    own_names=(c.SCRIPT,c.TEST,'scripts/eval_research_batch4_at_boundary.py','scripts/probe_parallel_b1_at_boundary.py','scripts/research_queue.py')
    full_names=(c.FULL_SCRIPT,'src/eval/research_parallel_protocol.py','tests/test_research_parallel_evaluator.py','tests/test_research_parallel_protocol.py')
    def proof(names,checks):
        sources={}
        for name in names:
            path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture '+name);sources[name]=c.digest(path.read_bytes())
        return {'status':'passed','tests_passed':3,'test_command':'CPU fixture','source_sha256':sources,'checks':{check:True for check in checks}}
    full=proof(full_names,['full512_raw_token_score_provenance_revalidation'])
    full.update(owner='fullowner',worker_counts=[2,4],full_manifest_records=512,max_new_tokens=64,actual_gpu_validated=False)
    own=proof(own_names,c.CHECKS)
    path=tmp_path/c.FULL_READY;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(full))
    own['upstream_full_readiness_sha256']=c.digest(path.read_bytes());(tmp_path/c.READY).write_text(json.dumps(own))
    module=types.ModuleType('scripts.eval_research_checkpoint_parallel_b1');module.OWNER='fullowner';module.validate_completed_output=lambda *args:None
    monkeypatch.setitem(sys.modules,module.__name__,module)
    import scripts
    monkeypatch.setattr(scripts,'eval_research_checkpoint_parallel_b1',module,raising=False)
    assert c.readiness(tmp_path)
    full['tests_passed']=4;path.write_text(json.dumps(full))
    with pytest.raises(ValueError,match='tested coordinator binding'):c.readiness(tmp_path)


def test_no_python_preexec_or_fork_callback_exists_after_torch_context_import():
    tree=ast.parse((Path(__file__).parents[1]/c.SCRIPT).read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
    assert calls and all(all(k.arg!='preexec_fn' for k in n.keywords) for n in calls)


def gateway_record(w):
    record=lease(w)
    command=c.worker_command(w.root,w.bound,4,800)
    record.update(contexts=w.bound,workers=4,target_job=c.TARGET,variant='text_only',worker_timeout_seconds=800,
        worker_command=command,worker_command_sha256=c.command_sha(command))
    return record


@pytest.mark.parametrize('death',['none','before','after'])
def test_fresh_gateway_parent_birth_checks_arm_before_exec(window,monkeypatch,death):
    w=window;record=gateway_record(w);path=c.receipt_path(w.root,4);w.write(path,record)
    own_pid=999993;w.infos[own_pid]={'pid':own_pid,'start_time':991,'argv':['python',c.SCRIPT,'--exec-worker']}
    monkeypatch.setattr(c.os,'getpid',lambda:own_pid)
    monkeypatch.setattr(c.os,'getppid',lambda:w.coordinator['pid'])
    events=[]
    def arm(pid):
        events.append('arm')
        if death=='after':w.infos[w.coordinator['pid']]=None
    class ExecReached(Exception):pass
    def execv(executable,command):
        events.append('exec');assert command==record['worker_command'];raise ExecReached()
    monkeypatch.setattr(c,'arm_parent_death',arm);monkeypatch.setattr(c.os,'execv',execv)
    if death=='before':w.infos[w.coordinator['pid']]=None
    if death=='none':
        with pytest.raises(ExecReached):c.exec_worker(w.root,path,'nonce')
        assert events==['arm','exec']
        ready=c.read_json(Path(str(path)+'.nonce.worker.json'));assert ready['pid']==own_pid and ready['start_time']==991
    else:
        with pytest.raises(RuntimeError,match='Coordinator died'):c.exec_worker(w.root,path,'nonce')
        assert events==([] if death=='before' else ['arm'])
    assert not w.signals


def test_watchdog_recovers_self_registered_worker_if_parent_died_before_pid_persist(window,monkeypatch):
    w=window;record=gateway_record(w);path=c.receipt_path(w.root,4)
    pid=999994;w.infos[pid]={'pid':pid,'start_time':992,'argv':record['worker_command'],'ppid':1}
    w.write(Path(str(path)+'.nonce.worker.json'),{'owner':c.OWNER,'nonce':'nonce','pid':pid,'start_time':992,
        'destination_command_sha256':record['worker_command_sha256']})
    original=c.os.kill
    def kill(pid,sig):original(pid,sig);w.infos[pid]=None
    monkeypatch.setattr(c.os,'kill',kill)
    c.recover_worker(w.root,record)
    assert w.signals==[(pid,signal.SIGTERM)]


def test_watchdog_self_registered_pid_reuse_is_never_signalled(window):
    w=window;record=gateway_record(w);path=c.receipt_path(w.root,4)
    pid=999994;w.infos[pid]={'pid':pid,'start_time':993,'argv':record['worker_command'],'ppid':1}
    w.write(Path(str(path)+'.nonce.worker.json'),{'owner':c.OWNER,'nonce':'nonce','pid':pid,'start_time':992,
        'destination_command_sha256':record['worker_command_sha256']})
    c.recover_worker(w.root,record)
    assert not w.signals


def test_monotonic_lease_expires_even_if_wallclock_moves_backwards(window,monkeypatch):
    w=window;record=lease(w);record['pause_deadline_monotonic']=0;record['pause_deadline_epoch']=10**12
    w.parent['state']='T';path=c.receipt_path(w.root,4);w.write(path,record)
    monkeypatch.setattr(c.time,'time',lambda:-10**12)
    monkeypatch.setattr(c.time,'sleep',lambda *a:pytest.fail('Backwards wallclock must not prolong lease'))
    c.watchdog(w.root,path,'nonce')
    assert w.signals==[(c.CONTROLLER,signal.SIGCONT)]


def test_execute_spawns_fresh_interpreter_without_preexec_and_checks_initial_ppid(window,monkeypatch):
    w=window;record=gateway_record(w);path=c.receipt_path(w.root,4);fake=SimpleNamespace(pid=999995,poll=lambda:None)
    def popen(command,**kwargs):
        assert 'preexec_fn' not in kwargs and '--exec-worker' in command
        w.infos[fake.pid]={'pid':fake.pid,'start_time':995,'ppid':999999,'argv':command};return fake
    stopped=[];monkeypatch.setattr(c.subprocess,'Popen',popen);monkeypatch.setattr(c,'stop_probe',lambda p:stopped.append(p.pid))
    with pytest.raises(RuntimeError,match='spawn identity'):c.execute(w.root,w.bound,4,path,record,c.time.monotonic()+1000)
    assert stopped==[fake.pid] and not w.signals


def test_execute_live_argv_change_stops_only_own_popen(window,monkeypatch):
    w=window;record=gateway_record(w);path=c.receipt_path(w.root,4);fake=SimpleNamespace(pid=999995,poll=lambda:None)
    def popen(command,**kwargs):
        w.infos[fake.pid]={'pid':fake.pid,'start_time':995,'ppid':c.os.getpid(),'argv':command};return fake
    original=c.process_info;count=[0]
    def info(pid):
        value=original(pid)
        if pid==fake.pid:
            count[0]+=1
            if count[0]>1:return dict(value,argv=['foreign'])
        return value
    stopped=[];monkeypatch.setattr(c.subprocess,'Popen',popen);monkeypatch.setattr(c,'process_info',info)
    monkeypatch.setattr(c,'stop_probe',lambda p:stopped.append(p.pid))
    with pytest.raises(RuntimeError,match='live birth'):c.execute(w.root,w.bound,4,path,record,c.time.monotonic()+1000)
    assert stopped==[fake.pid] and not w.signals


def future_blank(w,job,row):
    child,old_output=future_shuffle(w,job,row);output=old_output.parent/'blank'
    command=child['argv'];command[command.index('--image-ablation')+1]='blank';command[command.index('--output')+1]=str(output)
    row['active_grounding_variant']='blank'
    receipt=row['grounding_evaluations'].pop('shuffle');receipt['output']=str(output)
    row['grounding_evaluations']['blank']=receipt
    return child,output


def test_blank_is_separate_root_explicit_target_never_automatic_retarget(window):
    w=window;job,row=future_target(w);future_shuffle(w,job,row)
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='blank',required_pid=None)==('waiting',None)
    child,_=future_blank(w,job,row)
    status,evidence=c.phase(w.root,w.spec,w.state,job,w.parent,variant='blank',required_pid=None)
    assert status=='waiting' and evidence['evaluation_pid']==child['pid']
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='shuffle',required_pid=None)==('missed',None)
    assert c.receipt_path(w.root,4,c.TARGET1A,'blank')!=c.receipt_path(w.root,4,c.TARGET1A,'shuffle')
    row['active_grounding_variant']='text_only'
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='blank',required_pid=None)==('missed',None)
    assert not w.signals


def test_blank_exact_prespawn_receipt_without_pid_waits(window):
    w=window;job,row=future_target(w);future_blank(w,job,row)
    row['grounding_evaluations']['blank'].pop('pid')
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='blank',required_pid=None)==('waiting',None)


def future_text_only(w,job,row):
    child,old_output=future_blank(w,job,row);output=old_output.parent/'text_only'
    command=child['argv'];command[command.index('--image-ablation')+1]='text_only'
    command[command.index('--output')+1]=str(output)
    row['active_grounding_variant']='text_only'
    receipt=row['grounding_evaluations'].pop('blank');receipt['output']=str(output)
    row['grounding_evaluations']['text_only']=receipt
    return child,output


def test_text_only_root_explicit_target_separate_from_failed_earlier_attempts(window):
    w=window;job,row=future_target(w);future_blank(w,job,row)
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='text_only',required_pid=None)==('waiting',None)
    child,_=future_text_only(w,job,row)
    status,evidence=c.phase(w.root,w.spec,w.state,job,w.parent,variant='text_only',required_pid=None)
    assert status=='waiting' and evidence['evaluation_pid']==child['pid']
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='blank',required_pid=None)==('missed',None)
    assert len({c.receipt_path(w.root,4,c.TARGET1A,v) for v in ('shuffle','blank','text_only')})==3
    assert c.phase(w.root,w.spec,w.state,job,w.parent,evidence,variant='text_only',required_pid=child['pid']+1)==('missed',None)
    assert not w.signals


def test_text_only_exact_prespawn_waits_but_malformed_pid_refused(window):
    w=window;job,row=future_target(w);future_text_only(w,job,row)
    receipt=row['grounding_evaluations']['text_only'];receipt.pop('pid')
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='text_only',required_pid=None)==('waiting',None)
    receipt['pid']=None
    with pytest.raises(ValueError):c.phase(w.root,w.spec,w.state,job,w.parent,variant='text_only',required_pid=None)


def test_completed_text_only_never_retargets_a_later_training_job(window):
    w=window;job,row=future_target(w);future_text_only(w,job,row)
    row.update(status='completed',grounding_status='completed')
    assert c.phase(w.root,w.spec,w.state,job,w.parent,variant='text_only',required_pid=None)==('missed',None)
    assert not w.signals


def test_startup_empty_argv_is_bounded_and_keeps_own_birth_diagnostics(window,monkeypatch):
    w=window;record=gateway_record(w);path=c.receipt_path(w.root,4);fake=SimpleNamespace(pid=999995,poll=lambda:None)
    w.infos[fake.pid]={'pid':fake.pid,'start_time':995,'ppid':c.os.getpid(),'argv':[]}
    counter=[0.]
    monkeypatch.setattr(c.time,'monotonic',lambda:counter[0])
    monkeypatch.setattr(c.time,'sleep',lambda seconds:counter.__setitem__(0,counter[0]+seconds))
    with pytest.raises(TimeoutError,match='startup handshake'):c.wait_worker_startup(fake,path,record,['launcher'],['final'],1000)
    assert record['worker_pid']==fake.pid and record['worker_start_time']==995
    assert 1<=len(record['worker_startup_observations'])<=64
    assert record['worker_last_observation']['argv_state']=='empty' and counter[0]<=15.1
    assert not w.signals


def test_startup_birth_change_refused_even_when_argv_empty(window):
    w=window;record=gateway_record(w);record['worker_start_time']=994
    fake=SimpleNamespace(pid=999995,poll=lambda:None)
    w.infos[fake.pid]={'pid':fake.pid,'start_time':995,'ppid':c.os.getpid(),'argv':[]}
    with pytest.raises(RuntimeError,match='birth/PPID'):c.observe_worker(fake,record,['launcher'],['final'],startup=True)
    assert not w.signals


def _real_cpu_gateway_files(tmp_path, sleep_seconds):
    """Test-only child shim substitutes a CPU destination; production has no override."""
    repo=Path(__file__).resolve().parents[1]
    shim=tmp_path/c.SCRIPT;shim.parent.mkdir(parents=True)
    shim.write_text('import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(repo))+')\n'
        'from scripts import eval_research_parallel_b1_at_boundary as c\n'
        "path=Path(sys.argv[sys.argv.index('--receipt')+1]);nonce=sys.argv[sys.argv.index('--nonce')+1]\n"
        "c.worker_command=lambda *args:c.read_json(path)['worker_command']\n"
        'c.exec_worker(Path.cwd(),path,nonce)\n')
    destination=tmp_path/'cpu_final.py';marker=tmp_path/'cpu_final.json'
    destination.write_text('import ctypes,json,os,signal,time\nfrom pathlib import Path\n'
        'value=ctypes.c_int();lib=ctypes.CDLL(None);assert lib.prctl(2,ctypes.byref(value),0,0,0)==0\n'
        'Path('+repr(str(marker))+').write_text(json.dumps({"pid":os.getpid(),"ppid":os.getppid(),"pdeath":value.value}))\n'
        'time.sleep('+repr(sleep_seconds)+')\n')
    command=[sys.executable,'-u',str(destination)]
    return repo,marker,command


def test_real_fresh_gateway_after_torch_import_with_transient_proc_reads(tmp_path,monkeypatch):
    import torch
    assert not torch.cuda.is_initialized()
    _repo,marker,command=_real_cpu_gateway_files(tmp_path,.4)
    parent=c.process_info(os.getpid());path=tmp_path/'receipt.json'
    record={'owner':c.OWNER,'lease_nonce':'real_cpu_nonce','boot_id':c.boot_id(),'parent_pause_outstanding':True,
        'coordinator_pid':parent['pid'],'coordinator_start_time':parent['start_time'],
        'coordinator_command_sha256':c.command_sha(parent['argv']),'contexts':{},'workers':4}
    monkeypatch.setattr(c,'worker_command',lambda *args:command)
    original=c.process_info;reads=[0]
    def info(pid):
        value=original(pid)
        if pid!=os.getpid() and value is not None:
            reads[0]+=1
            if reads[0]==1:return None
            if reads[0]==2:return dict(value,argv=[])
        return value
    monkeypatch.setattr(c,'process_info',info)
    def result(*args):
        final=json.loads(marker.read_text())
        assert final['pid']==record['worker_pid'] and final['ppid']==os.getpid() and final['pdeath']==signal.SIGTERM
        return {'status':'cpu_integration_only_no_gpu'}
    monkeypatch.setattr(c,'validate_result',result)
    value=c.execute(tmp_path,{},4,path,record,time.monotonic()+1000)
    assert value['status']=='cpu_integration_only_no_gpu' and record['worker_startup_verified']
    assert {row['argv_state'] for row in record['worker_startup_observations']} >= {'unavailable','empty','final'}
    assert c.worker_handshake(path,record) and not torch.cuda.is_initialized()


def test_real_kernel_parent_death_survives_gateway_exec(tmp_path):
    repo,marker,command=_real_cpu_gateway_files(tmp_path,60)
    controller=tmp_path/'cpu_controller.py';receipt=tmp_path/'receipt.json'
    controller.write_text('import sys,time,os\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(repo))+')\n'
        'from scripts import eval_research_parallel_b1_at_boundary as c\n'
        'parent=c.process_info(os.getpid())\n'
        'record={"owner":c.OWNER,"lease_nonce":"cpu_death_nonce","boot_id":c.boot_id(),"parent_pause_outstanding":True,'
        '"coordinator_pid":parent["pid"],"coordinator_start_time":parent["start_time"],'
        '"coordinator_command_sha256":c.command_sha(parent["argv"]),"contexts":{},"workers":4}\n'
        'c.worker_command=lambda *args:'+repr(command)+'\n'
        'c.execute(Path('+repr(str(tmp_path))+'),{},4,Path('+repr(str(receipt))+'),record,time.monotonic()+1000)\n')
    process=subprocess.Popen([sys.executable,'-u',str(controller)],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    child=None
    try:
        deadline=time.monotonic()+8
        while not marker.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.025)
        assert marker.is_file(),process.communicate(timeout=1)
        final=json.loads(marker.read_text());child=c.process_info(final['pid'])
        assert child is not None and final['pdeath']==signal.SIGTERM and child['ppid']==process.pid
        process.terminate();process.wait(timeout=3)
        deadline=time.monotonic()+3
        while c.process_info(child['pid']) is not None and time.monotonic()<deadline:time.sleep(.025)
        assert c.process_info(child['pid']) is None
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=3)
        live=c.process_info(child['pid']) if child else None
        if live is not None and live['start_time']==child['start_time'] and live['argv']==command:
            os.kill(live['pid'],signal.SIGKILL)
