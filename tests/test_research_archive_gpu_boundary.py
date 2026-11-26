"""Owned CPU processes and mocked metadata only; no service/GPU action."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('archive_boundary_test',ROOT/'scripts/probe_research_archive_gpu_at_boundary.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def info(pid=700,*,state='R',argv=None,ppid=42,start=100):
    argv=argv or ['python','owned.py']
    return {'pid':pid,'start_time':start,'ppid':ppid,'argv':argv,'command_sha256':p.digest(argv),'state':state}
def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def lease(tmp_path,monkeypatch):
    parent=info();own=info(701,ppid=os.getpid());path=tmp_path/'receipt.json';now=time.monotonic()
    value={'owner':p.OWNER,'boot_id':p.boot(),'nonce':'a'*64,'status':'pause_intent','coordinator':p.stable(own),
        'bound':{'parent':p.stable(parent)},'parent_pause_outstanding':True,'stop_committed':False,
        'lease_deadline_monotonic':now+1200,'worker_deadline_monotonic':now+1170,'pause_started_monotonic':now}
    put(path,value);signals=[]
    monkeypatch.setattr(p.os,'kill',lambda pid,sig:signals.append((pid,sig)))
    monkeypatch.setattr(p.proc,'process_info',lambda pid:parent if pid==700 else own if pid==701 else None)
    return path,value,parent,signals


def test_default_import_no_torch_gpu_or_output(tmp_path):
    code="import sys,importlib.util; s=importlib.util.spec_from_file_location('boundary',sys.argv[1]); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules"
    r=subprocess.run([sys.executable,'-c',code,str(ROOT/p.SCRIPT)],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr


def test_stop_requires_guarded_lease_and_unpaused_exact_parent(lease):
    path,value,parent,signals=lease
    assert p.begin_pause(path,value['nonce'])
    assert signals==[(700,signal.SIGSTOP)] and p.read(path)['stop_committed']


@pytest.mark.parametrize('mutation',['revoked','expired','foreign_birth','foreign_argv','external_T','boot'])
def test_no_late_stop_or_external_pause_signal(lease,mutation):
    path,value,parent,signals=lease
    if mutation=='revoked':value['lease_revoked']=True
    if mutation=='expired':value['lease_deadline_monotonic']=time.monotonic()-1
    if mutation=='foreign_birth':parent['start_time']+=1
    if mutation=='foreign_argv':parent['argv']=['python','foreign.py']
    if mutation=='external_T':parent['state']='T'
    if mutation=='boot':value['boot_id']='different-boot'
    put(path,value)
    try:assert p.begin_pause(path,value['nonce']) is False
    except TimeoutError:pass
    assert not signals


def test_no_auxiliary_means_guardian_never_interrupts_native_evaluator(lease,monkeypatch):
    path,value,parent,signals=lease;parent['state']='T';value['stop_committed']=True;put(path,value)
    monkeypatch.setattr(p,'gpu_idle',lambda:False)
    p.finish_pause(path,value['nonce'],'test_no_auxiliary')
    assert signals==[(700,signal.SIGCONT)]
    assert not p.read(path).get('controller_exit_requested_before_CONT')


def test_finally_cont_cancels_committed_stop_before_T_ack(lease):
    path,value,parent,signals=lease;parent['state']='R';value['stop_committed']=True;put(path,value)
    p.finish_pause(path,value['nonce'],'stop_ack_failed')
    assert signals==[(700,signal.SIGCONT)] and not p.read(path)['parent_pause_outstanding']


@pytest.mark.parametrize('gone,idle,expected',[(True,True,[signal.SIGCONT]),(False,False,[signal.SIGINT,signal.SIGCONT]),(True,False,[signal.SIGINT,signal.SIGCONT])])
def test_aux_cleanup_before_cont_and_uncertain_cleanup_exits_only_owned_parent(lease,monkeypatch,gone,idle,expected):
    path,value,parent,signals=lease;parent['state']='T'
    value.update(stop_committed=True,worker_launch_intent=True,lease_deadline_monotonic=time.monotonic()+.01);put(path,value)
    order=[]
    monkeypatch.setattr(p,'cleanup_worker',lambda *a:order.append('cleanup') or gone)
    monkeypatch.setattr(p,'gpu_idle',lambda:idle)
    p.finish_pause(path,value['nonce'],'test')
    assert order==['cleanup'] and [sig for _,sig in signals]==expected
    assert all(pid==700 for pid,_ in signals)
    assert p.read(path)['parent_pause_outstanding'] is False


def test_finish_never_signals_reused_or_external_parent(lease,monkeypatch):
    path,value,parent,signals=lease;value['stop_committed']=True;put(path,value)
    parent['start_time']+=1;parent['state']='T'
    monkeypatch.setattr(p,'cleanup_worker',lambda *a:True)
    p.finish_pause(path,value['nonce'],'test')
    assert not signals


def test_missing_actual_cpu_report_refuses_before_controller_probe(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'controller',lambda *a:pytest.fail('No controller/signal before actual CPU proof'))
    with pytest.raises((OSError,ValueError)):p.arm(tmp_path,tmp_path/'missing.json','a'*64,700)


def fixture_phase(tmp_path,parent):
    run=tmp_path/'runs/research'/p.TARGET;output=run/'evaluation/grounding_v1/text_only';output.mkdir(parents=True)
    manifest=tmp_path/'manifest.jsonl';manifest.write_text('')
    command=[sys.executable,'-u',str(tmp_path/'scripts/eval_research_checkpoint.py'),'--run',str(run),
        '--manifest',str(manifest),'--checkpoint','best','--max-new-tokens','64','--output',str(output),
        '--image-ablation','text_only','--ablation-seed','42']
    child=info(900,argv=command,ppid=parent['pid']);job={'id':p.TARGET,'evaluation':{'manifest':str(manifest)}}
    bound={'spec_sha256':'s','parent':p.stable(parent),'run':str(run),'job':job}
    holder={'status':'running','pid':900,'command':command,'output':str(output)}
    row={'status':'grounding_evaluating','training_completed':True,'grounding_status':'running',
        'active_grounding_variant':'text_only','grounding_evaluations':{'text_only':holder}}
    state={'status':'running','spec_sha256':'s','jobs':{p.TARGET:row}}
    path=tmp_path/(p.SPEC.removesuffix('.json')+'.state.json');put(path,state)
    return bound,state,path,child,output


def test_phase_binds_live_exact_child_once_then_completed512(tmp_path,monkeypatch):
    parent=info();bound,state,path,child,output=fixture_phase(tmp_path,parent)
    monkeypatch.setattr(p.proc,'process_info',lambda pid:child)
    status,evidence=p.phase(tmp_path,bound);assert status=='waiting' and evidence['child']['pid']==900
    put(output/'summary.json',{'complete_manifest':True,'records':512,'manifest_records':512})
    put(output/'progress.json',{'event':'completed','completed_records':512})
    assert p.phase(tmp_path,bound,evidence)==('ready',evidence)


@pytest.mark.parametrize('lookup',[None,'empty','foreign'])
def test_initial_native_spawn_gap_waits_without_binding_foreign_or_completed_child(tmp_path,monkeypatch,lookup):
    bound,state,path,child,output=fixture_phase(tmp_path,info())
    current=None if lookup is None else dict(child,argv=[] if lookup=='empty' else ['python','foreign.py'])
    monkeypatch.setattr(p.proc,'process_info',lambda pid:current)
    assert p.phase(tmp_path,bound)==(('missed' if lookup=='foreign' else 'waiting'),None)
    if lookup is None:
        put(output/'summary.json',{'complete_manifest':True,'records':512,'manifest_records':512})
        put(output/'progress.json',{'event':'completed','completed_records':512})
        assert p.phase(tmp_path,bound)==('missed',None)


@pytest.mark.parametrize('mutation',['foreign_argv','reused_birth','foreign_ppid','other_child','other_active_job','already_done','trainer','replay','parallel'])
def test_phase_never_targets_other_worker_or_advanced_window(tmp_path,monkeypatch,mutation):
    parent=info();bound,state,path,child,output=fixture_phase(tmp_path,parent)
    monkeypatch.setattr(p.proc,'process_info',lambda pid:child)
    _,evidence=p.phase(tmp_path,bound)
    row=state['jobs'][p.TARGET]
    if mutation=='foreign_argv':child['argv']=['python','trainer.py']
    if mutation=='reused_birth':child['start_time']+=1
    if mutation=='foreign_ppid':child['ppid']+=1
    if mutation=='other_child':row['grounding_evaluations']['text_only']['pid']=901
    if mutation=='other_active_job':state['jobs']['later']={'status':'running'}
    if mutation=='already_done':row['status']='completed'
    if mutation=='trainer':row['status']='running'
    if mutation=='replay':row['status']='checkpoint_validating'
    if mutation=='parallel':row['status']='parallel_evaluating'
    put(path,state)
    assert p.phase(tmp_path,bound,evidence)[0]=='missed'


@pytest.mark.parametrize('earlier',['evaluating','checkpoint_validating','shuffle','blank'])
def test_earlier_native_phases_wait_only_before_child_binding(tmp_path,monkeypatch,earlier):
    bound,state,path,child,output=fixture_phase(tmp_path,info());row=state['jobs'][p.TARGET]
    if earlier in ('shuffle','blank'):row['active_grounding_variant']=earlier
    else:row['status']=earlier
    put(path,state)
    assert p.phase(tmp_path,bound)==('waiting',None)


class Process:
    pid=800
    def poll(self):return None


def test_worker_startup_waits_missing_empty_then_exact_final_with_nonce(lease,monkeypatch):
    path,value,parent,signals=lease;command=['python','helper.py','--nonce',value['nonce']];launcher=['python','gateway.py']
    value['worker_command']=command;put(path,value)
    child=info(800,argv=command,ppid=value['coordinator']['pid']);raw={k:child[k] for k in ('pid','ppid','start_time','state')}
    monkeypatch.setattr(p,'raw_birth',lambda pid:raw)
    observations=iter([None,dict(child,argv=[]),child])
    monkeypatch.setattr(p.proc,'process_info',lambda pid:next(observations))
    monkeypatch.setattr(p,'worker_registration',lambda *a:{'identity':p.stable(child)})
    p.await_worker_startup(Process(),path,value,launcher,time.monotonic()+1)
    assert p.read(path)['worker']['argv']==command


def test_worker_startup_foreign_nonempty_argv_gets_no_grace(lease,monkeypatch):
    path,value,parent,signals=lease;value['worker_command']=['python','helper.py'];put(path,value)
    child=info(800,argv=['python','foreign.py'],ppid=value['coordinator']['pid'])
    monkeypatch.setattr(p,'raw_birth',lambda pid:{k:child[k] for k in ('pid','ppid','start_time','state')})
    monkeypatch.setattr(p.proc,'process_info',lambda pid:child)
    with pytest.raises(ValueError,match='Foreign nonempty'):p.await_worker_startup(Process(),path,value,['python','gateway'],time.monotonic()+1)


def test_stale_worker_status_update_cannot_resurrect_revoked_pause(lease):
    path,value,parent,signals=lease;value['lease_revoked']=True;value['parent_pause_outstanding']=False;put(path,value)
    with pytest.raises(TimeoutError):p.update_lease(path,value['nonce'],status='running_archive_parity')
    assert not p.read(path)['parent_pause_outstanding']


def test_worker_command_binds_parent_nonce_exact_cpu_proof_and_shared_work_deadline(lease,tmp_path):
    path,value,parent,signals=lease
    value['bound']['cpu_report']='/owned/exact/verified.json'
    command=p.helper_command(tmp_path,value);args=p.options(command,tmp_path,p.HELPER)
    assert args['--cpu-report']=='/owned/exact/verified.json' and args['--nonce']==value['nonce']
    assert args['--parent-pid']==str(value['coordinator']['pid'])
    assert args['--parent-start']==str(value['coordinator']['start_time'])
    assert args['--parent-command-sha256']==value['coordinator']['command_sha256']
    assert float(args['--deadline-monotonic'])==value['worker_deadline_monotonic']
    assert int(args['--timeout-seconds'])<=1170


def test_real_zombie_native_zero_exit_has_exact_birth_and_parent():
    child=subprocess.Popen([sys.executable,'-c','pass'])
    expected=None
    try:
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            current=p.proc.process_info(child.pid)
            if current is not None and current.get('argv'):expected=p.stable(current);break
        assert expected is not None
        while time.monotonic()<deadline:
            current=p.proc.process_info(child.pid)
            if current and current['state']=='Z':break
            time.sleep(.01)
        assert p.zero_exit(expected)
    finally:child.wait(timeout=2)


def test_real_nonzero_native_exit_refused():
    child=subprocess.Popen([sys.executable,'-c','import sys;sys.exit(3)']);expected=None
    try:
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            current=p.proc.process_info(child.pid)
            if current is not None and current.get('argv'):expected=p.stable(current);break
        assert expected is not None
        while time.monotonic()<deadline:
            current=p.proc.process_info(child.pid)
            if current and current['state']=='Z':break
            time.sleep(.01)
        with pytest.raises(ValueError,match='code0'):p.zero_exit(expected)
    finally:child.wait(timeout=2)


def test_real_guardian_survives_coordinator_sigkill_cleans_aux_then_continues_own_parent(tmp_path):
    """Only disposable CPU children; fake idle callback never queries a GPU."""
    parent=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    receipt=tmp_path/'receipt.json';ready=tmp_path/'paused.json';nonce='a'*64
    coordinator_code=r'''
import importlib.util,json,os,pathlib,subprocess,sys,time
module,parent_pid,path,ready,nonce=sys.argv[1:]
s=importlib.util.spec_from_file_location('cpu_archive_guardian_fixture',module);p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
parent=p.proc.process_info(int(parent_pid));own=p.proc.process_info(os.getpid());now=time.monotonic()
worker_code='import sys,time;from scripts.probe_research_parallel_b1 import arm_parent_death;arm_parent_death(int(sys.argv[1]));time.sleep(30)'
command=[sys.executable,'-c',worker_code,str(os.getpid()),'--nonce',nonce]
value={'owner':p.OWNER,'boot_id':p.boot(),'nonce':nonce,'status':'pause_intent','coordinator':p.stable(own),
 'bound':{'parent':p.stable(parent)},'parent_pause_outstanding':True,'stop_committed':False,
 'lease_deadline_monotonic':now+15,'worker_deadline_monotonic':now+12,'pause_started_monotonic':now,
 'worker_launch_intent':True,'worker_command':command}
p.persist(pathlib.Path(path),value)
worker=subprocess.Popen(command,cwd=p.ROOT,start_new_session=True)
child=None
for _ in range(100):
 child=p.proc.process_info(worker.pid)
 if child and child.get('argv')==command:break
 time.sleep(.01)
assert child and child.get('argv')==command
p.persist(pathlib.Path(path+'.'+nonce+'.worker.json'),{'owner':p.OWNER,'nonce':nonce,'boot_id':p.boot(),
 'identity':p.stable(child),'launcher':command,'command':command,'coordinator':p.stable(own)})
guardian_code="import importlib.util,pathlib,sys;s=importlib.util.spec_from_file_location('cpu_guardian',sys.argv[1]);p=importlib.util.module_from_spec(s);s.loader.exec_module(p);p.gpu_idle=lambda:True;p.guardian(pathlib.Path(sys.argv[2]),sys.argv[3])"
guardian=subprocess.Popen([sys.executable,'-c',guardian_code,module,path,nonce],cwd=p.ROOT,start_new_session=True)
for _ in range(200):
 if pathlib.Path(path+'.'+nonce+'.guardian.json').exists():break
 time.sleep(.01)
assert pathlib.Path(path+'.'+nonce+'.guardian.json').exists()
assert p.begin_pause(pathlib.Path(path),nonce)
for _ in range(200):
 if p.proc.process_info(parent['pid'])['state']=='T':break
 time.sleep(.01)
pathlib.Path(ready).write_text(json.dumps({'worker':p.stable(child),'guardian_pid':guardian.pid}))
time.sleep(30)
'''
    coord=subprocess.Popen([sys.executable,'-c',coordinator_code,str(ROOT/p.SCRIPT),str(parent.pid),str(receipt),str(ready),nonce],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    identities=None
    try:
        deadline=time.monotonic()+8
        while not ready.exists() and time.monotonic()<deadline:
            if coord.poll() is not None:
                out,err=coord.communicate();raise AssertionError(out+err)
            time.sleep(.02)
        assert ready.exists();identities=json.loads(ready.read_text())
        assert p.proc.process_info(parent.pid)['state']=='T'
        coord.kill();coord.wait(timeout=3)
        while time.monotonic()<deadline:
            value=p.read(receipt);worker=p.proc.process_info(identities['worker']['pid'])
            current=p.proc.process_info(parent.pid)
            if not value.get('parent_pause_outstanding') and current and current['state'] not in ('T','t') and (worker is None or worker['state'] in ('Z','X')):break
            time.sleep(.02)
        value=p.read(receipt)
        assert value['worker_cleanup_proven'] and value['gpu_idle_before_CONT'] and not value['parent_pause_outstanding']
        assert not value.get('controller_exit_requested_before_CONT')
        assert p.proc.process_info(parent.pid)['state'] not in ('T','t')
    finally:
        if coord.poll() is None:coord.kill();coord.wait(timeout=3)
        if parent.poll() is None:
            os.kill(parent.pid,signal.SIGCONT);parent.terminate();parent.wait(timeout=3)
        if identities:
            expected=identities['worker'];current=p.proc.process_info(expected['pid'])
            if current and current.get('start_time')==expected['start_time'] and current['state'] not in ('Z','X'):
                os.killpg(expected['pid'],signal.SIGKILL)


def test_real_fresh_gateway_exec_startup_missing_empty_observations_then_nonce_ack(tmp_path,monkeypatch):
    parent=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
    path=tmp_path/'gateway_receipt.json';nonce='b'*64;helper=tmp_path/'cpu_helper.py'
    helper.write_text('import time;time.sleep(1)\n')
    code="import importlib.util,pathlib,sys;s=importlib.util.spec_from_file_location('gateway_fixture',sys.argv[1]);p=importlib.util.module_from_spec(s);s.loader.exec_module(p);p.exec_worker(p.ROOT,pathlib.Path(sys.argv[2]),sys.argv[3])"
    launcher=[sys.executable,'-c',code,str(ROOT/p.SCRIPT),str(path),nonce]
    command=[sys.executable,str(helper),'--nonce',nonce]
    process=None
    try:
        identity=p.proc.process_info(parent.pid);own=p.proc.process_info(os.getpid());now=time.monotonic()
        value={'owner':p.OWNER,'boot_id':p.boot(),'nonce':nonce,'coordinator':p.stable(own),'bound':{'parent':p.stable(identity)},
            'parent_pause_outstanding':True,'stop_committed':True,'lease_deadline_monotonic':now+15,
            'worker_deadline_monotonic':now+12,'worker_command':command,'worker_launcher':launcher,'worker_launch_intent':True}
        put(path,value);os.kill(parent.pid,signal.SIGSTOP)
        deadline=time.monotonic()+2
        while p.proc.process_info(parent.pid)['state']!='T' and time.monotonic()<deadline:time.sleep(.01)
        process=subprocess.Popen(launcher,cwd=ROOT,start_new_session=True)
        actual_lookup=p.proc.process_info;reads=[]
        def lookup(pid):
            result=actual_lookup(pid)
            if pid==process.pid:
                reads.append(1)
                if len(reads)==1:return None
                if len(reads)==2:return dict(result,argv=[])
            return result
        monkeypatch.setattr(p.proc,'process_info',lookup)
        p.await_worker_startup(process,path,value,launcher,time.monotonic()+5)
        ack=p.worker_registration(path,p.read(path))
        assert len(reads)>=3 and ack['identity']['pid']==process.pid and ack['identity']['ppid']==os.getpid()
        assert ack['launcher']==launcher and ack['command']==command and ack['nonce']==nonce
        assert p.read(path)['worker']['argv']==command
        assert process.wait(timeout=3)==0
    finally:
        if process is not None and process.poll() is None:process.kill();process.wait(timeout=3)
        if parent.poll() is None:os.kill(parent.pid,signal.SIGCONT);parent.terminate();parent.wait(timeout=3)


def test_actual_unregistered_raw_bound_launcher_is_cleaned_while_caller_remains_live(tmp_path):
    command=[sys.executable,'-c','import time;time.sleep(20)']
    process=subprocess.Popen(command,start_new_session=True)
    try:
        current=p.proc.process_info(process.pid)
        value={'owner':p.OWNER,'nonce':'c'*64,'boot_id':p.boot(),'worker_launch_intent':True,
            'worker_raw_birth':{k:current[k] for k in ('pid','ppid','start_time','state')},'worker_launcher':command,
            'worker_command':['python','helper-not-started.py'],'coordinator':p.stable(p.proc.process_info(os.getpid()))}
        assert p.cleanup_worker(tmp_path/'missing_registration.json',value,time.monotonic()+2)
        process.wait(timeout=2)
        assert process.returncode!=0
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=2)
