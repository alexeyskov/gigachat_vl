"""Mocked CPU handoff lifecycle; no process/service/GPU action in these tests."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from scripts import handoff_research_parallel_grounding as h


@pytest.fixture
def window(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'ROOT',tmp_path)
    repo=Path(__file__).resolve().parents[1]
    spec=json.loads((repo/h.SPEC).read_text());job=next(j for j in spec['jobs'] if '--stage' in j['argv'] and j['argv'][j['argv'].index('--stage')+1]=='SFT')
    def write(path,value):
        path=tmp_path/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    write(Path(h.SPEC),spec);_,snapshot=h.old.spec_snapshot(tmp_path)
    identity=dict(snapshot,handoff_ready_sha256='a'*64,actual_full512_proof_sha256='b'*64)
    monkeypatch.setattr(h,'readiness',lambda root:(spec,identity))
    cfg=tmp_path/'supervisor/research-pilot.conf';cfg.parent.mkdir()
    cfg.write_text('[program:research-pilot]\nautorestart=false\n'+h.OLD_COMMAND+'\n')
    monkeypatch.setattr(h,'CONFIG',cfg)
    directory=h.at(tmp_path,h.argv_value(job['argv'],'--output'))
    write(directory/'resolved_config.json',{'stage':'SFT','output':str(directory)})
    write(directory/'completed.json',{'steps':170})
    write(directory/'checkpoint_compaction.json',{'status':'completed','training_completed':True,'resumable_optimizer':False})
    write(directory/'.checkpoint_manager.json',{'owner':'owned'})
    obj=directory/'.checkpoint_objects/object-owned';obj.mkdir(parents=True)
    (obj/'state.pt').write_bytes(b'owned completed fixture')
    write(obj/'.research_checkpoint.json',{'owner':'owned','step':170,'training_completed':True,'resumable_optimizer':False})
    for label in ('best','last'):(directory/label).symlink_to(obj)
    manifest=h.at(tmp_path,job['evaluation']['manifest']);manifest.parent.mkdir(parents=True,exist_ok=True);manifest.write_text('{"record_id":"fixture"}\n')
    output=h.at(tmp_path,job['evaluation']['output'])
    write(output/'protocol.json',{'image_ablation':{'variant':'none'},'manifest_sha256':h.sha(manifest)})
    parent={'pid':1001,'ppid':1,'start_time':11,'state':'S','cwd':str(tmp_path),
        'argv':['/venv/main/bin/python','-u',h.OLD_SCRIPT,'--spec',h.SPEC,'--retry-failed']}
    child={'pid':2002,'ppid':1001,'start_time':22,'state':'S','cwd':str(tmp_path),
        'argv':['/venv/main/bin/python','-u',str(tmp_path/'scripts/eval_research_checkpoint.py'),
            '--run',str(directory),'--manifest',str(manifest),'--output',str(output),
            '--checkpoint','best','--max-new-tokens','64','--reserve-gb','120']}
    replacement={'pid':3003,'ppid':1,'start_time':33,'state':'S','cwd':str(tmp_path),
        'argv':['/venv/main/bin/python','-u',h.NEW_SCRIPT,'--spec',h.SPEC,'--proof-run',h.PROOF_RUN,
                '--workers','4','--timeout-seconds','1200','--retry-failed']}
    handoff={'pid':os.getpid(),'ppid':os.getppid(),'start_time':99,'state':'S','cwd':str(tmp_path),'argv':['python','owned-handoff']}
    infos={1001:parent,2002:child,3003:replacement,os.getpid():handoff}
    monkeypatch.setattr(h.old,'process_info',lambda pid:infos.get(pid))
    state={'status':'running','spec_sha256':snapshot['canonical_spec_sha256'],'jobs':{
        job['id']:{'status':'evaluating','training_completed':True,
                   'evaluation':{'status':'running','pid':2002,'output':str(output)}}}}
    state_path=write(Path(h.SPEC).with_suffix('.state.json'),state)
    (tmp_path/'research_results').mkdir(exist_ok=True)
    signals=[]
    def kill(pid,sig):
        signals.append((pid,sig))
        if sig==signal.SIGSTOP:infos[pid]['state']='T'
        if sig==signal.SIGCONT:infos[pid]['state']='S'
    monkeypatch.setattr(h.os,'kill',kill)
    calls=[]
    monkeypatch.setattr(h.subprocess,'run',lambda cmd,**kw:calls.append(cmd))
    monkeypatch.setattr(h.old,'wait_gpu_idle',lambda **kw:calls.append('gpu_idle'))
    monkeypatch.setattr(h,'assert_no_workers',lambda spec:calls.append('no_workers'))
    monkeypatch.setattr(h.old,'service_status',lambda name:'research-pilot EXITED')
    monkeypatch.setattr(h.old,'wait_for_running',lambda name:'research-pilot RUNNING pid 3003, uptime 1s')
    monkeypatch.setattr(h,'start_watchdog',lambda *args:SimpleNamespace(wait=lambda **kw:0))
    return SimpleNamespace(root=tmp_path,spec=spec,job=job,identity=identity,config=cfg,write=write,
        directory=directory,manifest=manifest,output=output,parent=parent,child=child,replacement=replacement,
        infos=infos,state=state,state_path=state_path,signals=signals,calls=calls,handoff=handoff)


def boundary(w):return h.safe_boundary(w.root,w.spec,w.state,w.parent)


@pytest.mark.parametrize('variant',['none','shuffle','blank','text_only'])
def test_only_completed_compacted_sft_live_native_evaluator_is_safe(window,variant):
    w=window
    if variant!='none':
        out=w.directory/'evaluation/grounding_v1'/variant
        w.write(out/'protocol.json',{'image_ablation':{'variant':variant},'manifest_sha256':h.sha(w.manifest)})
        w.child['argv'][w.child['argv'].index('--output')+1]=str(out)
        w.child['argv']+=['--image-ablation',variant,'--ablation-seed','42']
        w.state['jobs'][w.job['id']].update(status='grounding_evaluating',active_grounding_variant=variant,
            grounding_evaluations={variant:{'status':'running','pid':2002,'output':str(out)}})
    result=boundary(w)
    assert result['variant']==variant and result['evaluator']['pid']==2002
    assert 'state' not in result['evaluator'] and not w.signals


@pytest.mark.parametrize('status',['running','checkpoint_validating','parallel_evaluating'])
def test_trainers_and_gpu_replay_are_never_bound_or_signaled(window,status):
    w=window;w.state['jobs'][w.job['id']]['status']=status
    assert boundary(w) is None and w.signals==[]


@pytest.mark.parametrize('change',['child_parent','child_script','missing_compact','not_completed','stale_spec'])
def test_invalid_phase_or_identity_is_refused(window,change):
    w=window
    if change=='child_parent':w.child['ppid']=999
    if change=='child_script':w.child['argv'][2]=str(w.root/'scripts/run_research_stage.py')
    if change=='missing_compact':(w.directory/'checkpoint_compaction.json').unlink()
    if change=='not_completed':w.state['jobs'][w.job['id']]['training_completed']=False
    if change=='stale_spec':
        w.state['spec_sha256']='stale'
        with pytest.raises(ValueError,match='specification'):boundary(w)
    else:assert boundary(w) is None
    assert w.signals==[]


def test_controller_only_barrier_revalidates_then_interrupts_and_always_continues(window):
    w=window;point=boundary(w);before=w.state_path.read_bytes();original=w.config.read_bytes()
    assert h.pause_and_interrupt(w.root,w.spec,w.identity,point)
    assert w.signals==[(1001,signal.SIGSTOP),(1001,signal.SIGINT),(1001,signal.SIGCONT)]
    assert w.state_path.read_bytes()==before and w.config.read_bytes()==original
    record=h.read(w.root/h.RECEIPT)
    assert record['status']=='stopping' and record['parent_pause_outstanding'] is False


def test_child_birth_or_phase_shift_after_pause_only_unpauses(window):
    w=window;point=boundary(w);w.child['start_time']+=1
    assert not h.pause_and_interrupt(w.root,w.spec,w.identity,point)
    assert w.signals==[(1001,signal.SIGSTOP),(1001,signal.SIGCONT)]


def test_source_proof_failure_at_pause_finally_unpauses(window,monkeypatch):
    w=window;point=boundary(w)
    monkeypatch.setattr(h,'readiness',lambda root:(_ for _ in ()).throw(ValueError('source proof changed')))
    with pytest.raises(ValueError,match='source proof'):h.pause_and_interrupt(w.root,w.spec,w.identity,point)
    assert w.signals==[(1001,signal.SIGSTOP),(1001,signal.SIGCONT)]


def test_crash_recovery_unpauses_only_recorded_birth_even_without_readiness(window,monkeypatch):
    w=window;point=boundary(w);record={'owner':h.OWNER,'status':'pause_intent','parent_pause_outstanding':True,
        'stop_committed':True,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),**point}
    w.write(Path(h.RECEIPT),record);w.parent['state']='T'
    monkeypatch.setattr(h,'readiness',lambda root:(_ for _ in ()).throw(ValueError('proof vanished')))
    with pytest.raises(ValueError):h.run(w.root,1001)
    assert w.signals==[(1001,signal.SIGCONT)]
    w.signals.clear();w.parent.update(start_time=99,state='T')
    with pytest.raises(ValueError):h.run(w.root,1001)
    assert w.signals==[]


def test_missing_actual_proof_refuses_before_new_stop_or_service_action(window,monkeypatch):
    w=window
    monkeypatch.setattr(h,'readiness',lambda root:(_ for _ in ()).throw(FileNotFoundError('actualfull512 missing')))
    with pytest.raises(FileNotFoundError):h.run(w.root,1001)
    assert not w.signals and not w.calls


def test_replacement_changes_only_named_owned_config_after_exit_idle_and_guards(window):
    w=window;point=boundary(w);h.pause_and_interrupt(w.root,w.spec,w.identity,point)
    record=h.read(w.root/h.RECEIPT);before=w.state_path.read_bytes()
    h.launch_replacement(w.root,w.spec,w.identity,record)
    assert h.NEW_COMMAND in w.config.read_text().splitlines()
    assert w.calls==['gpu_idle','no_workers',['supervisorctl','reread'],['supervisorctl','update','research-pilot'],['supervisorctl','start','research-pilot']]
    assert w.state_path.read_bytes()==before
    assert h.read(w.root/h.RECEIPT)['replacement_controller']['start_time']==33


def test_no_replacement_while_service_is_live(window,monkeypatch):
    w=window;point=boundary(w);h.pause_and_interrupt(w.root,w.spec,w.identity,point)
    record=h.read(w.root/h.RECEIPT)
    monkeypatch.setattr(h.old,'service_status',lambda name:'research-pilot RUNNING pid 1001')
    with pytest.raises(RuntimeError,match='must be EXITED'):h.launch_replacement(w.root,w.spec,w.identity,record)
    assert w.calls==[] and h.OLD_COMMAND in w.config.read_text().splitlines()


def test_starting_recovery_never_adopts_unbound_running_replacement_pid(window,monkeypatch):
    w=window;point=boundary(w);h.pause_and_interrupt(w.root,w.spec,w.identity,point)
    record=h.read(w.root/h.RECEIPT);record['status']='starting';w.write(Path(h.RECEIPT),record)
    monkeypatch.setattr(h.old,'service_status',lambda name:'research-pilot RUNNING pid 3003')
    with pytest.raises(RuntimeError,match='manual reconciliation'):h.run(w.root,1001)
    assert w.calls==[]


@pytest.mark.parametrize('extra',['[program:caddy]','command=unrelated'])
def test_unrelated_or_ambiguous_supervisor_config_is_never_rewritten(window,extra):
    w=window;original=w.config.read_text()+extra+'\n'
    with pytest.raises(ValueError,match='Supervisor configuration'):h.replacement_config(original)


def test_module_import_is_torch_free():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-c',"import sys;import scripts.handoff_research_parallel_grounding;assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_sigterm_during_owned_stop_unwinds_finally_and_restores_handlers(window,monkeypatch):
    w=window;point=boundary(w);monkeypatch.setattr(h,'ROOT',w.root)
    monkeypatch.setattr(h,'run',lambda root,pid:h.pause_and_interrupt(root,w.spec,w.identity,point))
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def stop_and_term(pid,sig):
        w.signals.append((pid,sig))
        if sig==signal.SIGSTOP:
            w.infos[pid]['state']='T'
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        if sig==signal.SIGCONT:w.infos[pid]['state']='S'
    monkeypatch.setattr(h.os,'kill',stop_and_term)
    assert h.main(['--controller-pid','1001'])==128+signal.SIGTERM
    assert w.signals==[(1001,signal.SIGSTOP),(1001,signal.SIGCONT)]
    assert all(signal.getsignal(sig)==handler for sig,handler in previous.items())
    assert h.read(w.root/h.RECEIPT)['parent_pause_outstanding'] is False


def test_watchdog_revocation_prevents_late_interrupt(window,monkeypatch):
    w=window;point=boundary(w)
    def revoke_during_checks(root):
        path=root/h.RECEIPT;record=h.read(path)
        h.revoke_pause(path,record['lease_nonce'],'simulated_expiry')
        return w.spec,w.identity
    monkeypatch.setattr(h,'readiness',revoke_during_checks)
    with pytest.raises(TimeoutError,match='revoked/expired'):
        h.pause_and_interrupt(w.root,w.spec,w.identity,point)
    assert w.signals==[(1001,signal.SIGSTOP),(1001,signal.SIGCONT)]
    assert h.read(w.root/h.RECEIPT)['lease_revoked'] is True


def test_independent_watchdog_resumes_only_recorded_parent_when_handoff_dies(window,monkeypatch):
    w=window;point=boundary(w);w.parent['state']='T';nonce='guardian-nonce'
    record={'owner':h.OWNER,'status':'pause_intent','parent_pause_outstanding':True,'lease_nonce':nonce,
        'stop_committed':True,
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'handoff_process':h.stable_identity(w.handoff),
        'pause_deadline_monotonic':time.monotonic()+120,**point}
    path=w.write(Path(h.RECEIPT),record)
    guardian={'pid':9009,'ppid':w.handoff['pid'],'start_time':90,'state':'S','cwd':str(w.root),'argv':['owned-guardian']}
    w.infos[9009]=guardian;w.infos[w.handoff['pid']]=None
    monkeypatch.setattr(h.os,'getpid',lambda:9009)
    h.watchdog(w.root,path,nonce)
    assert w.signals==[(1001,signal.SIGCONT)]
    assert h.read(path)['parent_pause_outstanding'] is False and h.read(path)['lease_revoked'] is True


def test_even_matching_pid_birth_cannot_resume_an_unrelated_program(window):
    w=window;w.parent.update(state='T',argv=['/venv/main/bin/python','unrelated_service.py'])
    record={'owner':h.OWNER,'parent_pause_outstanding':True,'controller':h.stable_identity(w.parent),
            'stop_committed':True,
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    h.resume_owned_parent(record)
    assert w.signals==[]


def test_actual_independent_watchdog_continues_owned_cpu_parent_after_sigkill(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    native=tmp_path/h.OLD_SCRIPT;native.parent.mkdir(parents=True)
    native.write_text('import time\ntime.sleep(60)\n')
    guardian=tmp_path/h.SCRIPT
    guardian.write_text('import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(repo))+')\n'
        'from scripts import handoff_research_parallel_grounding as h\nh.ROOT=Path.cwd()\nh.main()\n')
    parent=subprocess.Popen([sys.executable,'-u',str(native),'--spec',h.SPEC,'--retry-failed'],cwd=tmp_path)
    record_path=tmp_path/h.RECEIPT;record_path.parent.mkdir(parents=True)
    paused=tmp_path/'paused.json';coordinator=tmp_path/'coordinator.py'
    coordinator.write_text('import json,os,signal,sys,time\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(repo))+')\n'
        'from scripts import handoff_research_parallel_grounding as h\n'
        'root=Path('+repr(str(tmp_path))+');path=root/h.RECEIPT\n'
        'parent=h.old.process_info('+str(parent.pid)+');own=h.old.process_info(os.getpid())\n'
        'record={"owner":h.OWNER,"status":"pause_intent","lease_nonce":"real-guardian-nonce",'
        '"parent_pause_outstanding":True,"boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),'
        '"controller":h.stable_identity(parent),"handoff_process":h.stable_identity(own),"pause_deadline_monotonic":time.monotonic()+30}\n'
        'h._atomic_json(path,record)\nh.start_watchdog(root,path,record)\n'
        'h.begin_owned_pause(path,record["lease_nonce"],h.stable_identity(parent))\n'
        'Path('+repr(str(paused))+').write_text("paused")\ntime.sleep(60)\n')
    process=subprocess.Popen([sys.executable,'-u',str(coordinator)],cwd=tmp_path,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    own_parent=None
    try:
        deadline=time.monotonic()+8
        while not paused.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.025)
        assert paused.is_file(),process.communicate(timeout=1)
        own_parent=h.old.process_info(parent.pid)
        assert own_parent is not None and own_parent['state'] in ('T','t')
        process.kill();process.wait(timeout=3)
        deadline=time.monotonic()+4
        while time.monotonic()<deadline:
            current=h.old.process_info(parent.pid)
            if current is not None and current['state'] not in ('T','t'):break
            time.sleep(.025)
        else:pytest.fail('independent guardian did not release the owned CPU fixture pause')
        record=h.read(record_path)
        assert record['lease_revoked'] and record['parent_pause_outstanding'] is False
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=3)
        current=h.old.process_info(parent.pid)
        if current is not None and (own_parent is None or current['start_time']==own_parent['start_time']):
            if current['state'] in ('T','t'):os.kill(parent.pid,signal.SIGCONT)
            parent.terminate()
        parent.wait(timeout=3)


def test_explicit_standby_waits_without_signals_then_calls_strict_proof(window,monkeypatch):
    w=window;monkeypatch.setattr(h,'cpu_readiness',lambda root:(w.spec,w.identity,str(w.manifest)))
    monkeypatch.setattr(h.old,'service_status',lambda name:'research-pilot RUNNING pid 1001')
    output=w.root/h.PROOF_RUN/'evaluation/parallel_b1_v1/workers4/none'
    sleeps=[];audits=[]
    def sleep(seconds):
        assert w.signals==[] and w.calls==[];sleeps.append(seconds);output.mkdir(parents=True,exist_ok=True)
        for name in ('protocol.json','predictions.jsonl','summary.json','execution_summary.json','parallel_guard_passed.json'):
            (output/name).write_text('fixture: filenames alone are not proof')
    monkeypatch.setattr(h.time,'sleep',sleep)
    def audit(root):audits.append(True);return w.spec,w.identity
    monkeypatch.setattr(h,'readiness',audit)
    assert h.wait_for_actual_proof(w.root,1001)==(w.spec,w.identity)
    assert sleeps==[5] and audits==[True] and w.signals==[] and w.calls==[]


def test_standby_never_accepts_complete_filenames_with_failed_raw_proof(window,monkeypatch):
    w=window;monkeypatch.setattr(h,'cpu_readiness',lambda root:(w.spec,w.identity,str(w.manifest)))
    monkeypatch.setattr(h.old,'service_status',lambda name:'research-pilot RUNNING pid 1001')
    output=w.root/h.PROOF_RUN/'evaluation/parallel_b1_v1/workers4/none';output.mkdir(parents=True)
    for name in ('protocol.json','predictions.jsonl','summary.json','execution_summary.json','parallel_guard_passed.json'):
        (output/name).write_text('{}')
    monkeypatch.setattr(h,'readiness',lambda root:(_ for _ in ()).throw(ValueError('raw512 tokens differ')))
    with pytest.raises(ValueError,match='tokens differ'):h.wait_for_actual_proof(w.root,1001)
    assert not w.signals and not w.calls


def test_standby_refuses_reused_controller_or_changed_source(window,monkeypatch):
    w=window;monkeypatch.setattr(h,'cpu_readiness',lambda root:(w.spec,w.identity,str(w.manifest)))
    monkeypatch.setattr(h.old,'service_status',lambda name:'research-pilot RUNNING pid 1001')
    def sleep(_seconds):w.parent['start_time']+=1
    monkeypatch.setattr(h.time,'sleep',sleep)
    with pytest.raises(RuntimeError,match='identity changed'):h.wait_for_actual_proof(w.root,1001)
    assert not w.signals and not w.calls


def test_guardian_revocation_before_stop_cannot_be_followed_by_late_stop(window,monkeypatch):
    w=window;point=boundary(w)
    def revoked_before_stop(root,path,record):
        h.revoke_pause(path,record['lease_nonce'],'expired_before_stop')
        return SimpleNamespace(wait=lambda **kw:0)
    monkeypatch.setattr(h,'start_watchdog',revoked_before_stop)
    with pytest.raises(TimeoutError,match='before STOP'):h.pause_and_interrupt(w.root,w.spec,w.identity,point)
    assert w.signals==[] and w.parent['state']=='S'


def test_external_parent_pause_during_watchdog_startup_is_never_continued(window,monkeypatch):
    w=window;point=boundary(w)
    def external_pause(root,path,record):
        w.parent['state']='T'
        return SimpleNamespace(wait=lambda **kw:0)
    monkeypatch.setattr(h,'start_watchdog',external_pause)
    assert not h.pause_and_interrupt(w.root,w.spec,w.identity,point)
    assert w.signals==[] and w.parent['state']=='T'
    assert h.read(w.root/h.RECEIPT)['parent_pause_outstanding'] is False
