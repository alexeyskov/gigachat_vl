"""CPU-only admission/barrier/journal lifecycle for the parallel B1 proposal."""
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

from scripts import eval_research_checkpoint_parallel_b1 as e
from tests.test_research_parallel_protocol import data, generated, native_cache


def identity(pid, parent=1, birth=None, argv=None):
    command=argv or ['python','owned-parallel',str(pid)]
    return dict(pid=pid,ppid=parent,start_time=birth or pid*100,argv=command,
                command_sha256=e.proc.digest(command),state='S')


@pytest.mark.parametrize('recipe,count',[('4q',118226944),('1a',351155264),('1b',351155264),('1c',351157440)])
@pytest.mark.parametrize('workers',[2,4])
def test_readonly_qlora_resource_admission_is_explicit_bounded_not_measured_proof(recipe,count,workers):
    profile=e.resource_profile(recipe,count,workers)
    assert profile['estimated_torch_bytes']<=8*e.GIB
    assert profile['initial_gpu_free_bytes']==(24+12*workers)*e.GIB
    assert profile['initial_host_available_bytes']==(32+8*workers)*e.GIB
    assert profile['estimate_is_measured_fit_proof'] is False
    assert all(p['batch_size']==1 and p['execution_script']==e.SCRIPT for p in e.profiles(profile,workers))


@pytest.mark.parametrize('recipe',['4f','2a','2b','2c','3a','3b','3c'])
def test_full_training_profiles_are_not_silently_admitted(recipe):
    with pytest.raises(ValueError,match='resource-blocked'): e.resource_profile(recipe,4035085824,2)


def test_unproved_huge_qlora_count_or_eight_workers_refused():
    with pytest.raises(ValueError,match='estimate'):e.resource_profile('1a',600000000,2)
    with pytest.raises(ValueError):e.resource_profile('4q',118226944,8)


def test_gpu_cohort_worst_case_headroom_unknown_pid_and_uuid_are_checked():
    profile=e.resource_profile('4q',118226944,4)
    snapshot=dict(gpu_uuid='GPU-fixture',total_bytes=96*e.GIB,free_bytes=80*e.GIB,processes=[])
    e.admit_gpu(snapshot,[],profile,initial=True)
    with pytest.raises(RuntimeError,match='entire cohort'):
        e.admit_gpu(dict(snapshot,free_bytes=70*e.GIB),[],profile,initial=True)
    with pytest.raises(RuntimeError,match='exclusive'):
        e.admit_gpu(dict(snapshot,processes=[dict(pid=999,bytes=e.GIB)]),[],profile,initial=True)
    with pytest.raises(RuntimeError,match='UUID'):
        e.admit_gpu(snapshot,[],dict(profile,gpu_uuid='GPU-other'))


def barrier(world=4,active=(0,1,2,3)):
    parent=identity(10)
    cohort=[dict(logical_worker=i,identity=identity(20+i,10)) for i in active]
    value=dict(owner=e.OWNER,status='released',nonce='nonce',parent=parent,cohort=cohort,
               world_size=world,deadline_monotonic=10**20)
    infos={10:parent,**{r['identity']['pid']:r['identity'] for r in cohort}}
    return value,infos


@pytest.mark.parametrize('world,active',[(1,(0,)),(2,(0,1)),(4,(0,1,2,3)),(4,(0,2,3)),(4,(3,))])
def test_fresh_barrier_covers_full_or_resumed_logical_cohort(world,active):
    value,infos=barrier(world,active)
    e.barrier_validate(value,infos[20+active[0]],active[0],'nonce',lookup=infos.get)


@pytest.mark.parametrize('change',['parent_birth','peer_birth','peer_argv','duplicate_pid','duplicate_index','outside_world','preparing','nonce'])
def test_barrier_refuses_pid_reuse_foreign_command_or_invalid_assignment(change):
    value,infos=barrier()
    if change=='parent_birth':infos[10]=dict(infos[10],start_time=999)
    if change=='peer_birth':infos[21]=dict(infos[21],start_time=999)
    if change=='peer_argv':infos[21]=dict(infos[21],argv=['foreign'])
    if change=='duplicate_pid':value['cohort'][1]['identity']=value['cohort'][0]['identity']
    if change=='duplicate_index':value['cohort'][1]['logical_worker']=0
    if change=='outside_world':value['cohort'][1]['logical_worker']=8
    if change=='preparing':value['status']='preparing'
    with pytest.raises(ValueError):e.barrier_validate(value,infos[20],0,'wrong' if change=='nonce' else 'nonce',lookup=infos.get)


def test_two_and_four_worker_outputs_are_separate_and_old_namespaces_refused(tmp_path):
    two=e.output_path(tmp_path,'none',workers=2);four=e.output_path(tmp_path,'none',workers=4)
    assert two!=four and two==tmp_path/'evaluation/parallel_b1_v1/none'
    assert four==tmp_path/'evaluation/parallel_b1_v1/workers4/none'
    for path in (tmp_path/'evaluation/heldout_v3',tmp_path/'evaluation/grounding_v1/none'):
        with pytest.raises(ValueError):e.output_path(tmp_path,'none',str(path))


class FakeChild:
    def __init__(self,command,infos):
        self.pid=100+int(command[command.index('--logical-worker')+1]);self.command=command
        self.index=self.pid-100;self.path=Path(command[command.index('--barrier')+1]);self.attempt=self.path.parent
        self.info=identity(self.pid,os.getpid(),argv=command);infos[self.pid]=self.info
        self.returncode=None;self.terminated=False
        value=e.proc.read_json(self.path);context=value['context'];protocol=e.proc.read_json(value['protocol_path'])
        e.proc.atomic_json(self.attempt/f'worker{self.index}.startup.json',dict(owner=e.OWNER,status='ready_before_allocation',
            nonce=value['nonce'],logical_worker=self.index,identity=self.info,parent=value['parent'],
            initial_allocation_barrier_sha256=e.proc.file_sha(self.path),protocol_sha256=value['protocol_sha256']))
        if value['guard_required']:
            rows=[dict(index=i,**generated(context['records'][i])) for i in e.module().select_guard_indices(context['records'])]
            self.guard_rows=rows
            phase='guard_completed'
        else:phase='model_ready'
        e.proc.atomic_json(self.attempt/f'worker{self.index}.progress.json',dict(phase=phase))
    def poll(self):
        gate=self.attempt/'generation_barrier.json'
        if self.returncode is None and gate.exists():
            value=e.proc.read_json(self.path);context=value['context'];protocol=e.proc.read_json(value['protocol_path'])
            output=Path(value['output']);committed=e.module().read_committed(output,context['records'],protocol)
            pending=e.module().pending_indices(context['records'],protocol,committed,self.index)
            for i in pending:e.module().commit_record(output,i,context['records'][i],generated(context['records'][i]),protocol,worker_index=self.index)
            start=e.proc.read_json(gate)['released_monotonic']+.001
            e.proc.atomic_json(self.attempt/f'worker{self.index}.result.json',dict(owner=e.OWNER,status='completed',
                identity=self.info,logical_worker=self.index,model_provenance=context['model_provenance'],
                protocol_sha256=e.module().protocol_sha256(protocol),allocation_barrier_sha256=e.proc.file_sha(self.path),
                generation_started_monotonic=start,generation_ended_monotonic=start+.001,
                generated_this_execution=len(pending),global_indices=pending,pytorch_cap_bytes=e.PT_CAP,
                journal_records_generated=len(pending),diagnostic_guard_records_generated=64 if value['guard_required'] else 0,
                peak_pytorch_allocated_bytes=4*e.GIB,peak_pytorch_reserved_bytes=5*e.GIB,batch_size=1))
            self.returncode=0
        elif self.returncode is None and hasattr(self,'guard_rows'):
            value=e.proc.read_json(self.path)
            if value['status']=='released':
                e.proc.atomic_json(self.attempt/f'worker{self.index}.guard.json',dict(owner=e.OWNER,status='passed',
                    identity=self.info,model_provenance=value['context']['model_provenance'],
                    protocol_sha256=value['protocol_sha256'],allocation_barrier_sha256=e.proc.file_sha(self.path),
                    guard_rows=self.guard_rows))
        return self.returncode
    def terminate(self):self.terminated=True;self.returncode=-15
    def wait(self,timeout=None):return self.returncode


@pytest.fixture
def cohort(data,monkeypatch):
    root=data['root'];infos={os.getpid():identity(os.getpid())};children=[]
    monkeypatch.setattr(e,'ROOT',root)
    monkeypatch.setattr(e.proc,'process_info',lambda pid:infos.get(pid))
    monkeypatch.setattr(e,'verify_context',lambda context:None)
    monkeypatch.setattr(e,'host_available',lambda:200*e.GIB)
    def gpu():
        live=[]
        for child in children:
            if e.proc.read_json(child.path).get('status')=='released':live.append(dict(pid=child.pid,bytes=5*e.GIB))
        return dict(gpu_uuid='GPU-fixture',total_bytes=96*e.GIB,free_bytes=80*e.GIB,processes=live)
    monkeypatch.setattr(e.proc,'gpu_snapshot',gpu)
    def popen(command,**kwargs):
        assert 'preexec_fn' not in kwargs
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='GPU-fixture'
        value=FakeChild(command,infos);children.append(value);return value
    monkeypatch.setattr(e.subprocess,'Popen',popen)
    monkeypatch.setattr(e.time,'sleep',lambda *a:None)
    return dict(data=data,root=root,infos=infos,children=children)


def context_for(data,workers):
    return dict(owner=e.OWNER,records=data['records'],source_sha256=data['sources'],
        model_provenance=data['provenance'],profile=e.resource_profile('4q',118226944,workers))


@pytest.mark.parametrize('workers',[2,4])
def test_complete_mocked_cohort_checks_each_model64_then_disjoint512_and_resume_no_rerun(cohort,workers):
    d=cohort['data'];context=context_for(d,workers);output=d['root']/f'out{workers}'
    protocol=d['protocol'](workers=workers);e.module().init_journal(output,protocol)
    reference={str(i):dict(record_id=d['records'][i]['record_id'],**generated(d['records'][i])) for i in e.module().select_guard_indices(d['records'])}
    result=e.run_cohort(context,output,protocol,list(range(workers)),reference_rows=reference,guard_required=True,timeout=120)
    assert result['status']=='completed' and len(cohort['children'])==workers
    assert all(not c.terminated for c in cohort['children'])
    assert e.validated_parallel_guard(output,context,protocol,reference)['status']=='passed'
    committed=e.module().read_committed(output,d['records'],protocol)
    assert len(committed)==512
    assert all(e.module().pending_indices(d['records'],protocol,committed,i)==[] for i in range(workers))
    rows,summary=e.module().merge_completed(output,d['records'],protocol)
    assert [row['record_id'] for row in rows]==[row['record_id'] for row in d['records']]
    assert summary['records']==512 and summary['complete_manifest']


def test_unknown_gpu_after_launch_cleanup_only_owned_children(cohort,monkeypatch):
    d=cohort['data'];context=context_for(d,2);output=d['root']/'out';protocol=d['protocol']();e.module().init_journal(output,protocol)
    count=[0]
    def gpu():
        count[0]+=1
        return dict(gpu_uuid='GPU-fixture',total_bytes=96*e.GIB,free_bytes=90*e.GIB,
                    processes=[dict(pid=999,bytes=e.GIB)] if count[0]==3 else [])
    monkeypatch.setattr(e.proc,'gpu_snapshot',gpu)
    reference={str(i):dict(record_id=d['records'][i]['record_id'],**generated(d['records'][i])) for i in e.module().select_guard_indices(d['records'])}
    with pytest.raises(RuntimeError,match='Unknown'):e.run_cohort(context,output,protocol,[0,1],reference_rows=reference,guard_required=True,timeout=120)
    assert all(child.terminated for child in cohort['children'])


def test_invalid_cli_never_launches_gpu(monkeypatch):
    gpu=Mock(side_effect=AssertionError('No GPU query for an invalid CLI'))
    monkeypatch.setattr(e.proc,'gpu_snapshot',gpu)
    with pytest.raises(SystemExit):e.main(['--workers','8'])
    gpu.assert_not_called()


def _cpu_startup_shim(root, *, stall_worker=None):
    script=root/e.SCRIPT;script.parent.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[1]
    script.write_text('''import os,sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,'''+repr(str(repo))+''')
from scripts import eval_research_checkpoint_parallel_b1 as e
from tests.test_research_parallel_protocol import generated
assert 'torch' not in sys.modules
def value(flag):return sys.argv[sys.argv.index(flag)+1]
args=SimpleNamespace(parent_pid=int(value('--parent-pid')),parent_start=int(value('--parent-start')),
    barrier=value('--barrier'),logical_worker=int(value('--logical-worker')),nonce=value('--nonce'))
if args.logical_worker=='''+repr(stall_worker)+''':
    e.proc.arm_parent_death(args.parent_pid)
    while True:time.sleep(.05)
path=e.worker_startup_registration(args)
while e.proc.read_json(path)['status']!='released':time.sleep(.01)
b=e.proc.read_json(path);own=e.proc.process_info(os.getpid());i=args.logical_worker
e.barrier_validate(b,own,i,args.nonce)
protocol=e.proc.read_json(b['protocol_path']);context=b['context'];attempt=path.parent
sha=e.proc.file_sha(path)
if b['guard_required']:
    rows=[dict(index=j,**generated(context['records'][j])) for j in e.module().select_guard_indices(context['records'])]
    e.proc.atomic_json(attempt/f'worker{i}.guard.json',dict(owner=e.OWNER,status='passed',identity=own,
        model_provenance=context['model_provenance'],protocol_sha256=b['protocol_sha256'],allocation_barrier_sha256=sha,guard_rows=rows))
    phase='guard_completed'
else:phase='model_ready'
e.proc.atomic_json(attempt/f'worker{i}.progress.json',dict(phase=phase))
gate=attempt/'generation_barrier.json'
while not gate.exists():time.sleep(.01)
started=time.monotonic();output=Path(b['output'])
committed=e.module().read_committed(output,context['records'],protocol)
pending=e.module().pending_indices(context['records'],protocol,committed,i)
for j in pending:e.module().commit_record(output,j,context['records'][j],generated(context['records'][j]),protocol,worker_index=i)
ended=time.monotonic()
e.proc.atomic_json(attempt/f'worker{i}.result.json',dict(owner=e.OWNER,status='completed',identity=own,logical_worker=i,
    model_provenance=context['model_provenance'],protocol_sha256=b['protocol_sha256'],allocation_barrier_sha256=sha,
    generation_started_monotonic=started,generation_ended_monotonic=ended,generated_this_execution=len(pending),
    global_indices=pending,journal_records_generated=len(pending),diagnostic_guard_records_generated=64 if b['guard_required'] else 0,
    pytorch_cap_bytes=e.PT_CAP,peak_pytorch_allocated_bytes=0,peak_pytorch_reserved_bytes=0,batch_size=1))
assert 'torch' not in sys.modules
''')


@pytest.mark.parametrize('stall',[False,True])
def test_actual_cpu_subprocesses_run_cohort_launch_nonce_barrier_and_partial_cleanup(data,monkeypatch,stall):
    root=data['root'];_cpu_startup_shim(root,stall_worker=1 if stall else None)
    monkeypatch.setattr(e,'ROOT',root);monkeypatch.setattr(e,'verify_context',lambda context:None)
    monkeypatch.setattr(e,'host_available',lambda:200*e.GIB)
    original_popen=e.subprocess.Popen;original_info=e.proc.process_info;children=[];reads={}
    def popen(command,**kwargs):
        assert 'preexec_fn' not in kwargs
        child=original_popen(command,**kwargs);children.append(child);return child
    def info(pid):
        actual=original_info(pid)
        if pid!=os.getpid() and actual is not None:
            reads[pid]=reads.get(pid,0)+1
            if reads[pid]==1:return None
            if reads[pid]==2:return dict(actual,argv=[])
        return actual
    monkeypatch.setattr(e.subprocess,'Popen',popen);monkeypatch.setattr(e.proc,'process_info',info)
    def gpu():
        live=[]
        for child in children:
            if child.poll() is None:
                paths=list((root/'actual'/ 'execution').glob('*/allocation_barrier.json'))
                if paths and e.proc.read_json(paths[0])['status']=='released':live.append(dict(pid=child.pid,bytes=5*e.GIB))
        return dict(gpu_uuid='GPU-CPU-FIXTURE',total_bytes=96*e.GIB,free_bytes=80*e.GIB,processes=live)
    monkeypatch.setattr(e.proc,'gpu_snapshot',gpu)
    if stall:
        wait=e.wait_worker_startup
        def short_wait(children,commands,path,parent,nonce,deadline,receipt,identities=None):
            receipt['deadline_monotonic']=min(time.monotonic()+1,deadline)
            return wait(children,commands,path,parent,nonce,deadline,receipt,identities)
        monkeypatch.setattr(e,'wait_worker_startup',short_wait)
    context=context_for(data,2);output=root/'actual';protocol=data['protocol']();e.module().init_journal(output,protocol)
    reference={str(i):dict(record_id=data['records'][i]['record_id'],**generated(data['records'][i])) for i in e.module().select_guard_indices(data['records'])}
    try:
        if stall:
            with pytest.raises(TimeoutError,match='startup handshake'):
                e.run_cohort(context,output,protocol,[0,1],reference_rows=reference,guard_required=True,timeout=120)
            attempt=next((output/'execution').iterdir())
            assert e.proc.read_json(attempt/'allocation_barrier.json')['status']=='preparing'
            assert e.module().read_committed(output,data['records'],protocol)=={}
        else:
            result=e.run_cohort(context,output,protocol,[0,1],reference_rows=reference,guard_required=True,timeout=120)
            assert result['status']=='completed'
            assert len(e.module().read_committed(output,data['records'],protocol))==512
        assert len(children)==2 and all(child.poll() is not None for child in children)
        assert all(value>=2 for value in reads.values())
    finally:
        for child in children:
            if child.poll() is None:child.kill();child.wait(timeout=3)


@pytest.mark.parametrize('mutation',['nonce','file_sha','missing','timing','birth','argv'])
def test_posthoc_startup_proof_corruption_refused(cohort,mutation):
    d=cohort['data'];context=context_for(d,2);output=d['root']/'raw';protocol=d['protocol']();e.module().init_journal(output,protocol)
    reference={str(i):dict(record_id=d['records'][i]['record_id'],**generated(d['records'][i])) for i in e.module().select_guard_indices(d['records'])}
    e.run_cohort(context,output,protocol,[0,1],reference_rows=reference,guard_required=True,timeout=120)
    attempt=next((output/'execution').iterdir());barrier=e.proc.read_json(attempt/'allocation_barrier.json');receipt=e.proc.read_json(attempt/'receipt.json')
    startup=attempt/'worker_startup.json';report=e.proc.read_json(startup)
    if mutation=='missing':(attempt/'worker0.startup.json').unlink()
    elif mutation in ('nonce','birth','argv'):
        path=attempt/'worker0.startup.json';ready=e.proc.read_json(path)
        if mutation=='nonce':ready['nonce']='foreign'
        elif mutation=='birth':ready['identity']['start_time']+=1
        else:ready['identity']['argv']=['foreign']
        e.proc.atomic_json(path,ready);report['startup_files_sha256']['0']=e.proc.file_sha(path)
    elif mutation=='timing':report['deadline_monotonic']+=16
    else:report['startup_files_sha256']['0']='0'*64
    if mutation!='missing':
        e.proc.atomic_json(startup,report);barrier['worker_startup_sha256']=receipt['worker_startup_sha256']=e.proc.file_sha(startup)
    with pytest.raises((ValueError,FileNotFoundError)):
        e.validate_startup_proof(attempt,barrier,receipt)


def test_startup_absent_and_empty_argv_have_one_shared15s_deadline(tmp_path,monkeypatch):
    clock=[0.];parent=identity(10);own=identity(11,10);child=SimpleNamespace(pid=11,poll=lambda:None)
    path=tmp_path/'allocation_barrier.json';e.proc.atomic_json(path,{'protocol_sha256':'a'*64})
    receipt={'owner':e.OWNER,'status':'starting','started_monotonic':0.,'deadline_monotonic':15.,
        'workers':{'0':{'pid':11,'start_time':None,'observations':[]}}}
    monkeypatch.setattr(e.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(e.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    reads=[0]
    def info(pid):
        reads[0]+=1;return None if reads[0]%2 else dict(own,argv=[])
    monkeypatch.setattr(e.proc,'process_info',info)
    monkeypatch.setattr(e,'worker_birth',lambda pid:dict(own,argv=None))
    with pytest.raises(TimeoutError,match='exceeded15s'):
        e.wait_worker_startup({0:child},{0:own['argv']},path,parent,'nonce',1000,receipt)
    assert 15<=clock[0]<15.1 and receipt['workers']['0']['start_time']==own['start_time']
    assert len(receipt['workers']['0']['observations'])==64


@pytest.mark.parametrize('mutation',['foreign','reused_empty','foreign_parent'])
def test_startup_foreign_nonempty_or_reused_identity_fails_immediately(monkeypatch,mutation):
    parent=identity(10);own=identity(11,10);live=copy.deepcopy(own)
    if mutation=='foreign':live['argv']=['foreign-command']
    if mutation=='reused_empty':live.update(argv=[],start_time=own['start_time']+1)
    if mutation=='foreign_parent':live['ppid']=999
    receipt={'workers':{'0':{'pid':11,'start_time':own['start_time'],'observations':[]}}}
    monkeypatch.setattr(e.proc,'process_info',lambda pid:live)
    with pytest.raises(RuntimeError,match='foreign nonempty|birth/PPID'):
        e.observe_starting_worker(SimpleNamespace(pid=11,poll=lambda:None),own['argv'],parent,receipt,0)


@pytest.mark.parametrize('mutation',['birth','argv','ppid'])
def test_pre_nonce_cleanup_never_signals_reused_or_foreign_child(monkeypatch,mutation):
    own=identity(11,10);live=copy.deepcopy(own)
    if mutation=='birth':live['start_time']+=1
    if mutation=='argv':live['argv']=['foreign']
    if mutation=='ppid':live['ppid']=999
    monkeypatch.setattr(e.proc,'process_info',lambda pid:live)
    child=SimpleNamespace(pid=11,poll=lambda:None,terminate=Mock(side_effect=AssertionError('foreign signal')))
    assert e.stop_starting_worker(child,{'start_time':own['start_time']},own['argv'],identity(10)) is False
    child.terminate.assert_not_called()


@pytest.mark.parametrize('state',['missing','empty','zombie'])
def test_gpu_exit_gap_waits_for_original_popen_zero_only(monkeypatch,state):
    expected=identity(22,10);clock=[0.];statuses=iter([None,None,0]);polls=[]
    def poll():
        value=next(statuses,0);polls.append(value);return value
    child=SimpleNamespace(pid=22,poll=poll)
    live=None if state=='missing' else dict(expected,argv=[],state='Z' if state=='zombie' else 'R')
    monkeypatch.setattr(e.proc,'process_info',lambda pid:live)
    monkeypatch.setattr(e,'worker_birth',lambda pid:dict(expected,argv=None))
    monkeypatch.setattr(e.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(e.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    checks=[];confirmed=[];snapshot=dict(gpu_uuid='GPU-exit',total_bytes=96*e.GIB,free_bytes=80*e.GIB,
                                      processes=[dict(pid=22,bytes=5*e.GIB)])
    callback=lambda pid:e.confirm_owned_worker_exit(pid,{0:child},{0:expected},checks=checks,confirmations=confirmed)
    e.admit_gpu(snapshot,[expected],dict(gpu_uuid='GPU-exit'),confirm_owned_exit=callback)
    assert polls==[None,None,0] and checks[0]['status']=='confirmed' and confirmed[0]['returncode']==0
    assert 0<confirmed[0]['wait_seconds']<=1.


@pytest.mark.parametrize('returncode',[None,7,-15])
def test_gpu_exit_gap_still_live_or_nonzero_handle_is_never_exempt(monkeypatch,returncode):
    expected=identity(22,10);clock=[0.]
    monkeypatch.setattr(e.proc,'process_info',lambda pid:None)
    monkeypatch.setattr(e,'worker_birth',lambda pid:dict(expected,argv=None))
    monkeypatch.setattr(e.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(e.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    checks=[]
    assert not e.confirm_owned_worker_exit(22,{0:SimpleNamespace(pid=22,poll=lambda:returncode)},{0:expected},checks=checks)
    assert clock[0]<=1.01 and checks[0]['status']=='refused'


@pytest.mark.parametrize('mutation',['birth','ppid','foreign_argv'])
def test_gpu_exit_gap_foreign_identity_rejects_without_wait_or_popen_poll(monkeypatch,mutation):
    expected=identity(22,10);live=copy.deepcopy(expected)
    if mutation=='birth':live.update(start_time=999,argv=[])
    if mutation=='ppid':live.update(ppid=999,argv=[])
    if mutation=='foreign_argv':live['argv']=['foreign']
    monkeypatch.setattr(e.proc,'process_info',lambda pid:live)
    child=SimpleNamespace(pid=22,poll=Mock(side_effect=AssertionError('foreign process must not poll-exempt')))
    assert not e.confirm_owned_worker_exit(22,{0:child},{0:expected})
    child.poll.assert_not_called()


def test_gpu_exit_gap_rechecks_birth_after_zero_status(monkeypatch):
    expected=identity(22,10);state={'reused':False}
    monkeypatch.setattr(e.proc,'process_info',lambda pid:None)
    monkeypatch.setattr(e,'worker_birth',lambda pid:dict(expected,argv=None,start_time=999 if state['reused'] else expected['start_time']))
    def poll():state['reused']=True;return 0
    assert not e.confirm_owned_worker_exit(22,{0:SimpleNamespace(pid=22,poll=poll)},{0:expected})


@pytest.mark.parametrize('code',[0,3])
def test_real_cpu_shutdown_edge_requires_fresh_original_popen_status(tmp_path,monkeypatch,code):
    child=subprocess.Popen([sys.executable,'-c','import time,sys;time.sleep(.15);sys.exit('+str(code)+')'])
    original=e.proc.process_info;deadline=time.monotonic()+3;expected=None
    while expected is None and time.monotonic()<deadline:expected=original(child.pid)
    assert expected is not None
    polls=[0]
    def delayed_poll():
        polls[0]+=1
        if polls[0]<=2:return None
        return child.poll()
    monkeypatch.setattr(e.proc,'process_info',lambda pid:None)
    checks=[]
    try:
        result=e.confirm_owned_worker_exit(child.pid,{0:SimpleNamespace(pid=child.pid,poll=delayed_poll)},
            {0:expected},checks=checks)
        assert result is (code==0) and child.poll()==code
        assert checks[0]['status']==('confirmed' if code==0 else 'refused') and polls[0]>=3
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=3)


def test_full_parent_survives_own_mm_exit_gap_during_nvml_and_records_evidence(cohort,monkeypatch):
    d=cohort['data'];original_popen=e.subprocess.Popen;original_polls={};remaining={};state={'exiting':False}
    def popen(command,**kwargs):
        child=original_popen(command,**kwargs);original_polls[child.pid]=child.poll
        def poll():
            status=original_polls[child.pid]()
            if status==0:
                if not state['exiting']:return None
                if remaining.get(child.pid,0)>0:remaining[child.pid]-=1;return None
            return status
        child.poll=poll;return child
    monkeypatch.setattr(e.subprocess,'Popen',popen)
    def gpu():
        live=[];children=cohort['children']
        if children and (children[0].attempt/'generation_barrier.json').exists() and not state['exiting']:
            for child in children:
                original_polls[child.pid]();cohort['infos'][child.pid]=None;remaining[child.pid]=2
            state['exiting']=True
        for child in children:
            if e.proc.read_json(child.path)['status']=='released':live.append(dict(pid=child.pid,bytes=5*e.GIB))
        return dict(gpu_uuid='GPU-fixture',total_bytes=96*e.GIB,free_bytes=80*e.GIB,processes=live)
    monkeypatch.setattr(e.proc,'gpu_snapshot',gpu)
    monkeypatch.setattr(e,'worker_birth',lambda pid:next((dict(child.info,argv=None) for child in cohort['children'] if child.pid==pid),None))
    context=context_for(d,2);output=d['root']/'exit';protocol=d['protocol']();e.module().init_journal(output,protocol)
    reference={str(i):dict(record_id=d['records'][i]['record_id'],**generated(d['records'][i])) for i in e.module().select_guard_indices(d['records'])}
    result=e.run_cohort(context,output,protocol,[0,1],reference_rows=reference,guard_required=True,timeout=120)
    assert result['status']=='completed' and state['exiting']
    assert len(result['owned_exit_checks'])==len(result['owned_exit_confirmations'])==2
    assert all(check['status']=='confirmed' and check['failed_lookup']['live'] is None for check in result['owned_exit_checks'])
    assert len(e.module().read_committed(output,d['records'],protocol))==512


def test_exit_confirmation_never_waits_beyond_remaining_cohort_deadline(monkeypatch):
    expected=identity(22,10);clock=[0.]
    monkeypatch.setattr(e.proc,'process_info',lambda pid:None)
    monkeypatch.setattr(e,'worker_birth',lambda pid:dict(expected,argv=None))
    monkeypatch.setattr(e.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(e.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    checks=[]
    assert not e.confirm_owned_worker_exit(22,{0:SimpleNamespace(pid=22,poll=lambda:None)},{0:expected},
        checks=checks,cohort_deadline=.025)
    assert .025<=clock[0]<.026 and checks[0]['deadline_monotonic']==.025 and checks[0]['status']=='refused'


def test_original_zombie_foreign_nonempty_argv_never_qualifies_after_it_disappears(monkeypatch):
    expected=identity(22,10);monkeypatch.setattr(e.proc,'process_info',lambda pid:None)
    monkeypatch.setattr(e,'worker_birth',lambda pid:None)
    child=SimpleNamespace(pid=22,poll=Mock(side_effect=AssertionError('initial foreign argv must reject before polling')))
    checks=[];failed={'pid':22,'live':dict(expected,state='Z',argv=['foreign-nonempty'])}
    assert not e.confirm_owned_worker_exit(22,{0:child},{0:expected},checks=checks,failed_lookup=failed)
    child.poll.assert_not_called()
    assert checks[0]['status']=='refused' and checks[0]['refusal_reason'].startswith('initial_failed_lookup')


@pytest.fixture
def complete_output(cohort,monkeypatch):
    d=cohort['data'];run=d['root']/'run';run.mkdir()
    raw=b''.join(e.module().json_bytes(row)+b'\n' for row in d['records'])
    manifest=d['root']/'full.jsonl';manifest.write_bytes(raw)
    context=context_for(d,2)
    context.update(run=str(run),checkpoint_identity=d['identity'],manifest=str(manifest),
        effective_manifest=str(manifest),manifest_sha256=d['manifest_sha'],transformed_manifest_sha256=d['manifest_sha'],
        variant='none',environment=d['protocol']()['environment'],eos_token_ids=[3])
    old=d['root']/'native';preliminary=e.protocol_for(context,2)
    native_cache(old,d,d['records'],preliminary)
    original=e.module().validate_native_cache
    def validate(directory,records,protocol,**kwargs):
        return original(directory,records,protocol,root=d['root'],**kwargs)
    monkeypatch.setattr(e.module(),'validate_native_cache',validate)
    monkeypatch.setattr(e,'prepare_context',lambda *args,**kwargs:copy.deepcopy(context))
    from src.utils.research_storage import DiskBudget
    monkeypatch.setattr(DiskBudget,'ensure_space',lambda *a:None)
    args=SimpleNamespace(run=str(run),manifest=str(manifest),image_ablation='none',workers=2,output=None,
        resume=False,allow_gpu=True,preflight_only=False,native_cache=str(old),reserve_gb=120,timeout_seconds=120)
    result=e.evaluate(args)
    return dict(run=run,manifest=manifest,old=old,context=context,output=e.output_path(run,'none'),
                result=result,args=args,cohort=cohort)


def test_actual_public_cpu_verifier_revalidates_all512_native_reference_rows_and_execution(complete_output):
    case=complete_output
    result=e.validate_completed_output(case['run'],case['manifest'],workers=2,native_cache=case['old'])
    assert result['status']=='passed' and result['records']==512
    assert result['full512_exact_previous_native_comparison'] is True
    assert result['per_model_native64_guard_verified'] is True
    assert result['group_macro_score']==100.


@pytest.mark.parametrize('corruption',['raw_prediction','worker_model','worker_cap','worker_time','guard_token','source_zip','execution_summary'])
def test_public_full512_verifier_refuses_raw_runtime_or_score_corruption(complete_output,corruption):
    case=complete_output;output=case['output'];attempt=next((output/'execution').iterdir())
    if corruption=='raw_prediction':
        (output/'predictions.jsonl').write_bytes((output/'predictions.jsonl').read_bytes()+b'corrupt\n')
    elif corruption=='source_zip':
        archive=attempt/'execution_sources.zip';archive.write_bytes(archive.read_bytes()+b'corrupt')
    else:
        path=attempt/'worker0.result.json'
        if corruption=='guard_token':path=attempt/'worker0.guard.json'
        if corruption=='execution_summary':path=output/'execution_summary.json'
        value=e.proc.read_json(path)
        if corruption=='worker_model':value['model_provenance']={'foreign':True}
        if corruption=='worker_cap':value['peak_pytorch_reserved_bytes']=e.PT_CAP+1
        if corruption=='worker_time':value['generation_started_monotonic']=0
        if corruption=='guard_token':value['guard_rows'][0]['generated_token_ids']=[8,3]
        if corruption=='execution_summary':value['full512_previous_native_cache_comparison']=False
        e.proc.atomic_json(path,value)
    with pytest.raises((ValueError,json.JSONDecodeError)):
        e.validate_completed_output(case['run'],case['manifest'],workers=2,native_cache=case['old'])


def test_complete_resume_does_not_spawn_or_rerun_any_committed_row(complete_output,monkeypatch):
    case=complete_output
    before={p.name:p.read_bytes() for p in (case['output']/'prediction_records').iterdir()}
    popen=Mock(side_effect=AssertionError('All512 rows were already committed'))
    monkeypatch.setattr(e.subprocess,'Popen',popen)
    args=copy.copy(case['args']);args.resume=True
    result=e.evaluate(args)
    assert result['status']=='completed';popen.assert_not_called()
    assert before=={p.name:p.read_bytes() for p in (case['output']/'prediction_records').iterdir()}


@pytest.mark.parametrize('field',['pending_duplicate','pending_foreign_worker','timeout'])
def test_raw_cohort_rejects_self_consistent_pending_or_timeout_corruption(complete_output,field):
    case=complete_output;attempt=next((case['output']/'execution').iterdir())
    path=attempt/'allocation_barrier.json';value=e.proc.read_json(path)
    if field=='pending_duplicate':value['initial_pending']['0'].append(value['initial_pending']['0'][0])
    if field=='pending_foreign_worker':value['initial_pending']['0'][0]=1
    if field=='timeout':value['deadline_monotonic']+=100
    e.proc.atomic_json(path,value);digest=e.proc.file_sha(path)
    for name in ('receipt.json','generation_barrier.json'):
        file=attempt/name;data=e.proc.read_json(file);data['allocation_barrier_sha256']=digest;e.proc.atomic_json(file,data)
    protocol=e.proc.read_json(case['output']/'protocol.json')
    with pytest.raises(ValueError,match='Pending|deadline|Raw startup'):
        e.validate_cohort_attempt(attempt,case['context'],protocol)


def test_partial_resume_guards_each_new_model_but_never_recommits_prior_journal_rows(complete_output,monkeypatch):
    case=complete_output;cohort=case['cohort'];new_run=case['run'].parent/'interrupted_run';new_run.mkdir()
    context=copy.deepcopy(case['context']);context['run']=str(new_run)
    monkeypatch.setattr(e,'prepare_context',lambda *a,**k:copy.deepcopy(context))
    args=copy.copy(case['args']);args.run=str(new_run);args.resume=False
    output=e.output_path(new_run,'none');cohort['children'].clear()
    original_poll=FakeChild.poll;stopped=[False]
    def interrupted(child):
        gate=child.attempt/'generation_barrier.json'
        if gate.exists() and child.returncode is None:
            if not stopped[0] and child.index==0:
                barrier=e.proc.read_json(child.path);protocol=e.proc.read_json(barrier['protocol_path'])
                for i in barrier['initial_pending']['0'][:10]:
                    e.module().commit_record(output,i,context['records'][i],generated(context['records'][i]),protocol,worker_index=0)
                stopped[0]=True;child.returncode=1
                raise RuntimeError('injected interrupted partial generation')
            if stopped[0]:return None
        return original_poll(child)
    monkeypatch.setattr(FakeChild,'poll',interrupted)
    with pytest.raises(RuntimeError,match='interrupted partial'):e.evaluate(args)
    protocol=e.proc.read_json(output/'protocol.json')
    committed=e.module().read_committed(output,context['records'],protocol)
    assert len(committed)==10
    initial_guard=(output/'parallel_guard_passed.json').read_bytes()
    before={i:(output/'prediction_records'/f'row-{i:04d}.json').read_bytes() for i in committed}
    cohort['children'].clear();monkeypatch.setattr(FakeChild,'poll',original_poll)
    original_commit=e.module().commit_record;written=[]
    def commit(directory,index,*a,**kw):
        written.append(index);return original_commit(directory,index,*a,**kw)
    monkeypatch.setattr(e.module(),'commit_record',commit)
    args.resume=True
    assert e.evaluate(args)['status']=='completed'
    assert len(written)==502 and not set(written)&set(committed)
    assert (output/'parallel_guard_passed.json').read_bytes()==initial_guard
    for i,raw in before.items():assert (output/'prediction_records'/f'row-{i:04d}.json').read_bytes()==raw
    attempts=list((output/'execution').iterdir())
    assert len(attempts)==2 and all((a/'guard_passed.json').is_file() for a in attempts)
    assert e.validate_completed_output(new_run,args.manifest,native_cache=case['old'])['status']=='passed'


def test_four_worker_main_and_public512_validator_use_separate_namespace(complete_output,monkeypatch):
    case=complete_output;context=copy.deepcopy(case['context'])
    context['profile']=e.resource_profile('4q',118226944,4)
    monkeypatch.setattr(e,'prepare_context',lambda *a,**k:copy.deepcopy(context))
    case['cohort']['children'].clear()
    args=copy.copy(case['args']);args.workers=4
    result=e.evaluate(args)
    assert result['status']=='completed' and result['worker_count']==4
    output=e.output_path(case['run'],'none',workers=4)
    assert output!=case['output']
    assert e.validate_completed_output(case['run'],case['manifest'],workers=4,native_cache=case['old'])['status']=='passed'
