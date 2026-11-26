"""One-shot controller-only B4 window; all process/GPU actions are mocked."""
import copy
import json
from pathlib import Path
import signal
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import eval_research_batch4_at_boundary as c


@pytest.fixture
def window(tmp_path,monkeypatch):
    spec=json.loads((Path(__file__).resolve().parents[1]/c.SPEC).read_text())
    job=next(j for j in spec['jobs'] if j['id']==c.JOB)
    def write(path,value):
        path=tmp_path/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    write(c.SPEC,spec)
    run=c.at(tmp_path,c.argv_value(job['argv'],'--output'))
    output=run/'evaluation/grounding_v1/shuffle'
    command=['/venv/main/bin/python','-u',str(tmp_path/'scripts/eval_research_checkpoint.py'),
        '--run',str(run),'--manifest',str(c.at(tmp_path,job['evaluation']['manifest'])),
        '--output',str(output),'--checkpoint','best','--max-new-tokens','64','--image-ablation','shuffle','--ablation-seed','42']
    parent={'pid':44472,'start_time':123,'state':'S','ppid':1,'cwd':str(tmp_path),
        'argv':['/venv/main/bin/python','-u',c.CONTROLLER_SCRIPT,'--spec',c.SPEC,'--retry-failed']}
    child={'pid':44541,'start_time':456,'state':'S','ppid':44472,'cwd':str(tmp_path),'argv':command}
    infos={44472:parent,44541:child}
    _,snapshot=c.spec_snapshot(tmp_path)
    state={'status':'running','spec_sha256':snapshot['canonical_spec_sha256'],'jobs':{c.JOB:{
        'status':'grounding_evaluating','training_completed':True,'grounding_status':'running','active_grounding_variant':'shuffle',
        'grounding_evaluations':{'shuffle':{'status':'running','pid':44541,'command':command,'output':str(output)}}}}}
    state_path=Path(c.SPEC).with_suffix('.state.json');write(state_path,state)
    proof={'readiness_sha256':'ready','profile_sha256':'profile','tested_sources':{}}
    source={'run':str(run),'manifest':'fixed512','manifest_sha256':'manifest','output':str(run/'evaluation/batch4_v1/none'),
        'native_cache':str(run/'evaluation/heldout_v3'),'native_cache_files':{}}
    signals=[]
    def kill(pid,sig):
        signals.append((pid,sig))
        if pid==44472 and sig==signal.SIGSTOP:parent['state']='T'
        if pid==44472 and sig==signal.SIGCONT:parent['state']='S'
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'process_info',lambda pid:infos.get(pid))
    monkeypatch.setattr(c,'service_status',lambda _: 'research-pilot RUNNING pid 44472, uptime 1:00:00')
    monkeypatch.setattr(c,'readiness',lambda _:proof)
    monkeypatch.setattr(c,'context',lambda *args:source)
    monkeypatch.setattr(c,'verify_completed_variant',lambda *args:None)
    monkeypatch.setattr(c,'host_available_bytes',lambda:100*2**30)
    monkeypatch.setattr(c.DiskBudget,'ensure_space',lambda *args:None)
    monkeypatch.setattr(c.os,'kill',kill)
    monkeypatch.setattr(c,'gpu_pids',lambda: (_ for _ in ()).throw(AssertionError('No real NVML calls')))
    def complete():
        for name,value in {'summary.json':{'complete_manifest':True,'records':512,'manifest_records':512},
            'progress.json':{'event':'completed','records':512},'protocol.json':{'mode':'shuffle'},
            'predictions.jsonl':{'controlled':'512 rows verified by mocked validator'},
            'transformed_manifest.jsonl':{'fixture':True}}.items():write(output/name,value)
    return SimpleNamespace(root=tmp_path,spec=spec,job=job,run=run,output=output,parent=parent,child=child,
        infos=infos,state=state,state_path=state_path,write=write,complete=complete,signals=signals,source=source,proof=proof)


def test_completed_shuffle_boundary_pauses_only_wrapper_and_resumes_on_success(window,monkeypatch):
    w=window;w.complete();before=(w.root/w.state_path).read_bytes()
    monkeypatch.setattr(c,'wait_idle',lambda _:None)
    def execute(source,receipt,record,timeout):
        record['attempted']=True
        return {'status':'passed','compared_records':512,'exact_generated_token_ids':True}
    monkeypatch.setattr(c,'execute',execute)
    result=c.run(w.root)
    assert result['status']=='completed'
    assert w.signals==[(44472,signal.SIGSTOP),(44472,signal.SIGCONT)]
    assert (w.root/w.state_path).read_bytes()==before
    assert not c.read_json(w.root/c.RECEIPT)['parent_pause_outstanding']


def test_next_variant_is_skip_without_signalling_any_child(window):
    w=window;w.complete();w.state['jobs'][c.JOB]['active_grounding_variant']='blank';w.write(w.state_path,w.state)
    result=c.run(w.root)
    assert result['reason']=='target_boundary_missed' and not w.signals


def test_phase_race_after_pause_immediately_continues_without_worker(window,monkeypatch):
    w=window;w.complete();original=c.os.kill
    def shifted(pid,sig):
        original(pid,sig)
        if sig==signal.SIGSTOP:
            w.state['jobs'][c.JOB]['active_grounding_variant']='blank';w.write(w.state_path,w.state)
    monkeypatch.setattr(c.os,'kill',shifted)
    monkeypatch.setattr(c,'execute',lambda *args:(_ for _ in ()).throw(AssertionError('No worker')))
    result=c.run(w.root)
    assert result['reason']=='phase_advanced_at_barrier'
    assert w.signals==[(44472,signal.SIGSTOP),(44472,signal.SIGCONT)]


def test_worker_failure_or_parity_mismatch_always_continues_parent(window,monkeypatch):
    w=window;w.complete();monkeypatch.setattr(c,'wait_idle',lambda _:None)
    monkeypatch.setattr(c,'execute',lambda *args:(_ for _ in ()).throw(ValueError('four-mode guard or512 parity failed')))
    with pytest.raises(ValueError,match='parity'):c.run(w.root)
    assert w.signals==[(44472,signal.SIGSTOP),(44472,signal.SIGCONT)]
    assert c.read_json(w.root/c.RECEIPT)['status']=='failed'


def test_pid_reuse_or_new_child_refuses_boundary(window):
    w=window;w.complete()
    status,evidence=c.phase(w.root,w.spec,w.state,w.job,w.parent,'shuffle')
    assert status=='ready'
    w.child['start_time']=999
    assert c.phase(w.root,w.spec,w.state,w.job,w.parent,'shuffle',evidence)[0]=='missed'
    assert not w.signals


def test_natural_child_exit_and_stale_gpu_pid_wait_do_not_signal_evaluator(window,monkeypatch):
    w=window;w.complete();_,evidence=c.phase(w.root,w.spec,w.state,w.job,w.parent,'shuffle')
    w.infos[44541]=None
    with patch.object(c,'gpu_pids',side_effect=[[44541],[]]),patch.object(c.time,'sleep'):
        c.wait_idle(evidence)
    assert not w.signals
    monkeypatch.setattr(c,'gpu_pids',lambda:[999])
    with pytest.raises(RuntimeError,match='Other GPU'):c.wait_idle(evidence)
    assert not w.signals


def test_recovery_unpauses_owned_parent_before_any_bad_readiness_or_failed_result(window,monkeypatch):
    w=window;w.parent['state']='T'
    w.write(c.RECEIPT,{'owner':c.OWNER,'parent_pause_outstanding':True,'controller_pid':44472,
        'controller_start_time':123,'controller_command_sha256':c.command_sha(w.parent['argv']),'attempted':True})
    monkeypatch.setattr(c,'readiness',lambda _:(_ for _ in ()).throw(ValueError('changed READY')))
    with pytest.raises(ValueError,match='READY'):c.run(w.root)
    assert w.signals==[(44472,signal.SIGCONT)]


def test_foreign_or_reused_parent_receipt_never_signalled(window):
    w=window;w.parent['state']='T'
    c.resume_parent({'owner':'foreign','parent_pause_outstanding':True,'controller_pid':44472})
    c.resume_parent({'owner':c.OWNER,'parent_pause_outstanding':True,'controller_pid':44472,'controller_start_time':999,
        'controller_command_sha256':c.command_sha(w.parent['argv'])})
    assert not w.signals


def test_fixed_worker_cli_compares_full_native_cache_and_never_uses_old_failed_b8_report(window):
    command=c.worker_command(window.source)
    assert command[command.index('--image-ablation')+1]=='none'
    assert command[command.index('--compare-native-cache')+1]==window.source['native_cache']
    assert command[command.index('--checkpoint')+1]=='best'
    assert 'batch_inference_4q_actual.json' not in ' '.join(command)


def test_new_ready_flags_and_source_bindings_are_mandatory(tmp_path,monkeypatch):
    # Exercise real readiness validator rather than the lifecycle fixture mock.
    path=tmp_path/c.READY;path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status':'passed','tests_passed':48,'test_command':'pytest','checks':{}}))
    with pytest.raises(ValueError,match='readiness checks'):c.readiness(tmp_path)


def test_owned_worker_timeout_cleans_only_owned_worker(window,monkeypatch):
    actions=[];fake=SimpleNamespace(pid=999,returncode=None,poll=lambda:None,
        terminate=lambda:actions.append('terminate owned worker'),wait=lambda **kwargs:actions.append('wait owned worker'),
        kill=lambda:actions.append('kill owned worker'))
    monkeypatch.setattr(c.subprocess,'Popen',lambda *args,**kwargs:fake)
    with patch.object(c.time,'monotonic',side_effect=[0,1]):
        with pytest.raises(TimeoutError,match='1200s'):c.execute(window.source,window.root/c.RECEIPT,{},.5)
    assert actions==['terminate owned worker','wait owned worker']
    assert not window.signals

