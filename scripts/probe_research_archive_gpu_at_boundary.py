#!/usr/bin/env python3
"""ROOT-only fixed4f native text-only boundary; no GPU/service work at import.

Existing surviving actual CPU proof and a pinned reviewed helper release are
mandatory before arming. Only this exact controller can be paused. The native
evaluator finishes naturally; an independent owner/nonce/birth-bound guardian
cleans our own auxiliary process before resuming the controller.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import probe_research_parallel_b1 as proc

OWNER='actual4f_archive_gpu_textonly_boundary_v1'
SCRIPT='scripts/probe_research_archive_gpu_at_boundary.py'
TEST='tests/test_research_archive_gpu_boundary.py'
READY='configs/research/archive_gpu_boundary_ready.json'
RECEIPT='research_results/archive_gpu_boundary_4f_text_only.json'
CONTROLLER_SCRIPT='scripts/research_parallel_grounding_queue.py'
CONTROLLER_READY='configs/research/parallel_grounding_queue_ready.json'
CONTROLLER_PID=98979
SPEC='configs/research/smallthinker_pilot_scienceqa_context_v2_balanced_queue.json'
PROOF_RUN='runs/research/smallthinker4__siglip2__4q__SFT_pilot_scienceqa_context_v2_balanced_s42'
TARGET='smallthinker4__siglip2__4f__SFT_pilot_scienceqa_context_v2_balanced_s42'
HELPER='scripts/probe_research_small_archive_gpu_parity.py'
HELPER_READY='research_results/trained_small_archive_gpu_parity_ready_v2.json'
OUTPUT='research_results/trained_small_archive_gpu_parity_actual4f_text_boundary_v1'
LEASE_SECONDS=1200
CLEANUP_SECONDS=30
CHECKS=('actual_cpu_proof_and_pinned_helper_before_arm','fixed4f_native_text_only_never_retarget',
    'exact_live_child_bound_before_completed512','completed_compacted_source_roles_and_raw512_replay',
    'serialized_guardian_lease_stop_int_cont','natural_child_zero_exit_and_idle_gpu_before_helper',
    'fresh_exec_pdeath_nonce_parent_deadline_worker','bounded_missing_empty_argv_startup_only',
    'guardian_own_worker_cleanup_before_controller_cont','uncertain_gpu_cleanup_interrupts_only_owned_paused_controller',
    'term_unwind_and_sigkill_independent_guardian','no_spec_pointer_service_or_training_changes')


def sha(path):return proc.file_sha(Path(path))
def digest(value):return proc.digest(value)
def read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>16*2**20:raise ValueError('Bounded local owned metadata required')
    value=json.loads(path.read_bytes())
    if not isinstance(value,dict):raise ValueError('Expected metadata object')
    return value
def persist(path,value):proc.atomic_json(Path(path),value)
def boot():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def stable(info):return {key:info[key] for key in ('pid','start_time','ppid','argv','command_sha256')}
def same(expected,actual,*,ppid=True):
    keys=('pid','start_time','argv','command_sha256')+(('ppid',) if ppid else ())
    return isinstance(expected,dict) and isinstance(actual,dict) and all(actual.get(k)==expected.get(k) for k in keys)
def live(info):return isinstance(info,dict) and info.get('state') not in ('Z','X')
def absolute(root,value):
    p=Path(value);return (p if p.is_absolute() else root/p).resolve()
def emit(event,**fields):print(json.dumps({'event':event,'time':time.time(),**fields}),flush=True)


def options(argv,root,script):
    if not isinstance(argv,list) or len(argv)<3:return None
    matches=[i for i,a in enumerate(argv) if absolute(root,a)==(root/script).resolve()]
    if len(matches)!=1:return None
    args=argv[matches[0]+1:];result={};i=0
    while i<len(args):
        flag=args[i]
        if not flag.startswith('--') or flag in result:return None
        if i+1<len(args) and not args[i+1].startswith('--'):result[flag]=args[i+1];i+=2
        else:result[flag]=True;i+=1
    return result


def controller(root,pid):
    info=proc.process_info(pid);args=options(info.get('argv') if info else None,root,CONTROLLER_SCRIPT)
    expected={'--spec','--proof-run','--workers','--timeout-seconds','--retry-failed'}
    if (not live(info) or info.get('state') in ('T','t') or args is None or set(args)!=expected or
        absolute(root,args['--spec'])!=(root/SPEC).resolve() or
        absolute(root,args['--proof-run'])!=(root/PROOF_RUN).resolve() or
        args['--workers']!='4' or args['--timeout-seconds']!='1200' or args['--retry-failed'] is not True):
        raise ValueError('Only the exact unpaused owned parallel research controller is accepted')
    status=subprocess.run(['supervisorctl','status','research-pilot'],capture_output=True,text=True,timeout=10,check=True).stdout
    if 'RUNNING' not in status or f'pid {pid},' not in status:raise ValueError('Exact research-pilot Supervisor PID is not live')
    return stable(info)


def readiness(root):
    proof=read(root/READY)
    if (proof.get('owner')!=OWNER or proof.get('status')!='passed' or type(proof.get('tests_passed')) is not int or
        proof['tests_passed']<1 or not proof.get('test_command') or any(proof.get('checks',{}).get(k) is not True for k in CHECKS)):
        raise ValueError('Archive boundary CPU release is absent/incomplete')
    required={SCRIPT,TEST,CONTROLLER_SCRIPT,'scripts/probe_research_parallel_b1.py',
        'scripts/eval_research_batch4_at_boundary.py','scripts/research_grounding_queue.py',HELPER,
        'src/utils/research_archive_gpu_parity.py'}
    if not required<=set(proof.get('source_sha256',{})):raise ValueError('Boundary source release omits required sources')
    for name,h in proof['source_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Archive boundary source changed: '+name)
    if proof.get('helper_readiness_sha256')!=sha(root/HELPER_READY):raise ValueError('Pinned final helper release changed')
    if proof.get('controller_readiness_sha256')!=sha(root/CONTROLLER_READY):raise ValueError('Pinned active controller release changed')
    active=read(root/CONTROLLER_READY)
    if active.get('status')!='passed' or CONTROLLER_SCRIPT not in active.get('source_sha256',{}):
        raise ValueError('Actual active controller CPU source release missing')
    for name,h in active['source_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Armed controller source changed: '+name)
    return {'path':str(root/READY),'sha256':sha(root/READY),'helper_ready_sha256':sha(root/HELPER_READY),
        'controller_ready_sha256':sha(root/CONTROLLER_READY)}


def arm(root,cpu_report,cpu_sha,controller_pid):
    read(cpu_report)  # Bound local metadata before hashing any caller path.
    if not isinstance(cpu_sha,str) or len(cpu_sha)!=64 or sha(cpu_report)!=cpu_sha:raise ValueError('Explicit surviving actual CPU report SHA required')
    release=readiness(root)
    from src.utils import research_archive_gpu_parity as parity
    plan=parity.prepare_plan(cpu_report)
    if Path(plan['run'])!=(root/'runs/research'/TARGET).resolve():raise ValueError('Only the explicit actual4f run is accepted')
    parent=controller(root,controller_pid);spec=read(root/SPEC);state=read(root/(SPEC.removesuffix('.json')+'.state.json'))
    if len(spec.get('jobs',[]))!=24 or state.get('spec_sha256')!=digest(spec):raise ValueError('Immutable24-job state/spec differs')
    jobs=[j for j in spec['jobs'] if j.get('id')==TARGET]
    if len(jobs)!=1:raise ValueError('Exact fixed4f job absent or duplicated')
    job=jobs[0]
    from scripts.research_queue import argv_value
    if (argv_value(job['argv'],'--stage')!='SFT' or argv_value(job['argv'],'--recipe')!='4f' or
        absolute(root,argv_value(job['argv'],'--output'))!=Path(plan['run']) or
        not job.get('compact_completed') or job.get('evaluation',{}).get('checkpoint','best')!='best'):
        raise ValueError('Target training/evaluation profile differs')
    return {'owner':OWNER,'boot_id':boot(),'parent':parent,'spec_sha256':digest(spec),'spec_file_sha256':sha(root/SPEC),
        'arm_state_sha256':sha(root/(SPEC.removesuffix('.json')+'.state.json')),'release':release,
        'cpu_report':str(Path(cpu_report).resolve()),'cpu_report_sha256':cpu_sha,'plan':plan,'job':job,
        'run':plan['run'],'target_job':TARGET,'variant':'text_only'}


def assert_arm(root,bound):
    if bound.get('owner')!=OWNER or bound['boot_id']!=boot() or sha(root/SPEC)!=bound['spec_file_sha256']:
        raise ValueError('Armed owner/boot/immutable specification changed')
    if readiness(root)!=bound['release'] or sha(bound['cpu_report'])!=bound['cpu_report_sha256']:
        raise ValueError('Pinned CPU/helper/controller release changed')
    from src.utils import research_archive_gpu_parity as parity
    parity.assert_plan(bound['plan'])


def phase(root,bound,expected=None):
    state_path=root/(SPEC.removesuffix('.json')+'.state.json');state=read(state_path)
    if state.get('spec_sha256')!=bound['spec_sha256'] or state.get('status')!='running':return 'missed',None
    rows=state.get('jobs')
    if not isinstance(rows,dict) or any(not isinstance(r,dict) for r in rows.values()):raise ValueError('Malformed queue state')
    row=rows.get(TARGET,{})
    active=[name for name,r in rows.items() if r.get('status') in ('running','evaluating','checkpoint_validating','grounding_evaluating','parallel_evaluating')]
    if active!=[TARGET]:return 'missed',None
    if row.get('status')!='grounding_evaluating' or row.get('active_grounding_variant')!='text_only':
        earlier=(row.get('status') in ('evaluating','checkpoint_validating') or
            row.get('status')=='grounding_evaluating' and row.get('active_grounding_variant') in ('shuffle','blank'))
        return ('waiting' if earlier and expected is None else 'missed'),None
    if row.get('training_completed') is not True or row.get('grounding_status')!='running':raise ValueError('Completed full SFT grounding phase required')
    holder=row.get('grounding_evaluations',{}).get('text_only',{});pid=holder.get('pid');command=holder.get('command')
    if holder.get('status')!='running' or type(pid) is not int or not isinstance(command,list):return ('waiting' if expected is None else 'missed'),None
    output=Path(bound['run'])/'evaluation/grounding_v1/text_only';args=options(command,root,'scripts/eval_research_checkpoint.py')
    if (args is None or args.get('--image-ablation')!='text_only' or args.get('--ablation-seed')!='42' or
        args.get('--checkpoint')!='best' or args.get('--max-new-tokens')!='64' or
        absolute(root,args.get('--run',''))!=Path(bound['run']) or absolute(root,args.get('--output',''))!=output or
        absolute(root,args.get('--manifest',''))!=absolute(root,bound['job']['evaluation']['manifest']) or
        absolute(root,holder.get('output',''))!=output):raise ValueError('Only exact native4f text-only child is accepted')
    child=proc.process_info(pid)
    if expected is None:
        if child is None or live(child) and not child.get('argv'):
            completed=False
            if (output/'progress.json').is_file() and (output/'summary.json').is_file():
                progress=read(output/'progress.json');summary=read(output/'summary.json')
                completed=progress.get('event')=='completed' and summary.get('complete_manifest') is True
            return ('missed' if completed else 'waiting'),None
        if not live(child) or child['ppid']!=bound['parent']['pid'] or child['argv']!=command:return 'missed',None
        evidence={'child':stable(child),'output':str(output),'state_at_binding_sha256':sha(state_path)}
    else:
        evidence=expected
        if pid!=expected['child']['pid'] or command!=expected['child']['argv']:return 'missed',None
        # The controller has not reaped a zombie while we own its STOP. Missing
        # or replaced identities are a missed window, never a stale-PID excuse.
        if child is None or child['start_time']!=expected['child']['start_time'] or child['ppid']!=expected['child']['ppid']:
            return 'missed',None
        if child.get('argv') not in ([],command):return 'missed',None
    if not (output/'progress.json').is_file() or not (output/'summary.json').is_file():return 'waiting',evidence
    progress=read(output/'progress.json');summary=read(output/'summary.json')
    ready=(progress.get('event')=='completed' and progress.get('completed_records',progress.get('records'))==512 and
        summary.get('complete_manifest') is True and summary.get('records')==summary.get('manifest_records')==512)
    return ('ready' if ready else 'waiting'),evidence


@contextmanager
def lease_lock(path):
    lock=Path(str(path)+'.lease.lock')
    if lock.is_symlink():raise ValueError('Foreign lease-lock symlink preserved')
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def lease(path,nonce):
    value=read(path)
    if (value.get('owner')!=OWNER or value.get('nonce')!=nonce or value.get('boot_id')!=boot() or
        value.get('lease_revoked') or not value.get('parent_pause_outstanding') or
        time.monotonic()>=value.get('lease_deadline_monotonic',0)):
        raise TimeoutError('Owned pause lease expired/revoked')
    return value


def update_lease(path,nonce,**fields):
    with lease_lock(path):
        value=lease(path,nonce);value.update(fields);persist(path,value);return value


def begin_pause(path,nonce):
    with lease_lock(path):
        value=lease(path,nonce);current=proc.process_info(value['bound']['parent']['pid'])
        if not same(value['bound']['parent'],current) or not live(current) or current.get('state') in ('T','t'):
            value.update(parent_pause_outstanding=False,status='external_pause_or_changed_parent_declined');persist(path,value);return False
        value['stop_committed']=True;persist(path,value)
        if time.monotonic()>=value['lease_deadline_monotonic']:raise TimeoutError('STOP intent deadline reached')
        os.kill(current['pid'],signal.SIGSTOP);return True


def worker_registration(path,value):
    record=Path(str(path)+'.'+value['nonce']+'.worker.json')
    if not record.exists():return None
    ready=read(record)
    if (ready.get('owner')!=OWNER or ready.get('nonce')!=value['nonce'] or ready.get('boot_id')!=value['boot_id'] or
        ready.get('command')!=value.get('worker_command') or ready.get('coordinator')!=value.get('coordinator')):
        raise ValueError('Owned worker registration differs')
    info=ready.get('identity')
    if not isinstance(info,dict) or info.get('ppid')!=value['coordinator']['pid']:raise ValueError('Owned worker original parent differs')
    return ready


class AmbiguousOwnedProcess(ValueError):pass


def owned_worker(path,value):
    registration=worker_registration(path,value)
    if registration is None:
        expected=value.get('worker_raw_birth');launcher=value.get('worker_launcher')
        if expected is None or not isinstance(launcher,list):return None,None
        current=proc.process_info(expected['pid'])
        if current is None:
            raw=raw_birth(expected['pid'])
            if raw is None:return expected,None
            if any(raw.get(k)!=expected.get(k) for k in ('pid','start_time','ppid')):raise ValueError('Raw launcher PID reused')
            if raw['state'] in ('Z','X'):return expected,dict(raw,argv=[])
            raise AmbiguousOwnedProcess('Owned raw launcher lookup is temporarily unavailable')
        if (current.get('start_time')!=expected['start_time'] or current.get('argv') not in (launcher,[]) or
            current.get('ppid')!=expected['ppid']):
            raise ValueError('Unregistered raw-bound launcher identity changed; never signalled')
        if current.get('argv')==[] and current.get('state') not in ('Z','X'):
            raise AmbiguousOwnedProcess('Owned launcher exit/exec argv temporarily empty; never signalled')
        return expected,current
    expected=registration['identity'];current=proc.process_info(expected['pid'])
    if current is None:
        raw=raw_birth(expected['pid'])
        if raw is None:return expected,None
        if raw['start_time']!=expected['start_time']:raise ValueError('Raw registered helper PID reused')
        if raw['state'] in ('Z','X'):return expected,dict(raw,argv=[])
        raise AmbiguousOwnedProcess('Owned registered helper lookup temporarily unavailable')
    if current.get('start_time')!=expected['start_time'] or current.get('argv') not in (registration['launcher'],registration['command'],[]):
        raise ValueError('Auxiliary PID reused/foreign nonempty argv; never signalled')
    if current.get('argv')==[] and current.get('state') not in ('Z','X'):
        raise AmbiguousOwnedProcess('Owned registered helper exit/exec argv temporarily empty; never signalled')
    return expected,current


def cleanup_worker(path,value,deadline):
    """Only the registered own process group. Foreign/reused PIDs are refused."""
    def observed():
        ambiguous_deadline=min(deadline,time.monotonic()+1)
        while True:
            try:return owned_worker(path,value)
            except AmbiguousOwnedProcess:
                if time.monotonic()>=ambiguous_deadline:raise
                time.sleep(.025)
    try:expected,current=observed()
    except ValueError:return False
    if expected is None:return not value.get('worker_launch_intent',False)
    if current is not None and current.get('state') not in ('Z','X'):
        try:
            if os.getpgid(current['pid'])!=current['pid']:return False
            os.killpg(current['pid'],signal.SIGTERM)
        except ProcessLookupError:
            try:_,after=observed()
            except ValueError:return False
            return after is None or after.get('state') in ('Z','X')
    kill_at=min(deadline,time.monotonic()+5)
    while time.monotonic()<deadline:
        try:_,current=observed()
        except ValueError:return False
        if current is None or current.get('state') in ('Z','X'):return True
        if time.monotonic()>=kill_at:
            try:
                if os.getpgid(current['pid'])!=current['pid']:return False
                os.killpg(current['pid'],signal.SIGKILL);kill_at=deadline+1
            except ProcessLookupError:
                try:_,after=observed()
                except ValueError:return False
                return after is None or after.get('state') in ('Z','X')
        time.sleep(.05)
    return False


def gpu_idle():
    gpu=proc.gpu_snapshot()
    return not gpu['processes'] and gpu['free_bytes']>=48*2**30


def finish_pause(path,nonce,reason):
    lock=Path(str(path)+'.cleanup.lock')
    if lock.is_symlink():raise ValueError('Foreign cleanup-lock symlink preserved')
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        return _finish_pause_serial(path,nonce,reason)


def _finish_pause_serial(path,nonce,reason):
    # Revoke before potentially slow worker cleanup; the caller can no longer
    # issue a late STOP or launch an auxiliary under this lease.
    with lease_lock(path):
        value=read(path)
        if (value.get('owner')!=OWNER or value.get('nonce')!=nonce or value.get('boot_id')!=boot() or
            not value.get('parent_pause_outstanding')):return
        value.update(lease_revoked=True,lease_revocation_reason=reason);persist(path,value)
        if not value.get('stop_committed'):
            value['parent_pause_outstanding']=False;persist(path,value);return
    cleanup_deadline=min(value['lease_deadline_monotonic'],time.monotonic()+CLEANUP_SECONDS)
    try:gone=cleanup_worker(path,value,cleanup_deadline)
    except BaseException as error:
        gone=False
        with lease_lock(path):
            saved=read(path);saved['cleanup_error']=type(error).__name__+': '+str(error);persist(path,saved)
    idle=False
    auxiliary_attempted=value.get('worker_launch_intent',False)
    while auxiliary_attempted and gone and time.monotonic()<cleanup_deadline:
        try:idle=gpu_idle()
        except Exception:idle=False
        if idle:break
        time.sleep(.1)
    with lease_lock(path):
        value=read(path)
        if value.get('owner')!=OWNER or value.get('nonce')!=nonce or not value.get('parent_pause_outstanding'):return
        current=proc.process_info(value['bound']['parent']['pid'])
        if value.get('stop_committed') and same(value['bound']['parent'],current) and live(current):
            try:
                if current.get('state') in ('T','t') and auxiliary_attempted and (not gone or not idle):
                    value['controller_exit_requested_before_CONT']=True;persist(path,value)
                    os.kill(current['pid'],signal.SIGINT)
                # CONT also cancels a committed STOP that has not reached T
                # yet; otherwise a failed immediate recheck can strand a late T.
                os.kill(current['pid'],signal.SIGCONT)
            except ProcessLookupError:value['parent_exited_before_CONT']=True
        value.update(parent_pause_outstanding=False,worker_cleanup_proven=gone,gpu_idle_before_CONT=idle)
        if auxiliary_attempted and (not gone or not idle):value['manual_reconciliation_required']=True
        persist(path,value)


def guardian(path,nonce):
    value=read(path)
    if value.get('owner')!=OWNER or value.get('nonce')!=nonce or value.get('boot_id')!=boot():return
    own=proc.process_info(os.getpid())
    persist(Path(str(path)+'.'+nonce+'.guardian.json'),{'owner':OWNER,'nonce':nonce,'identity':stable(own),'boot_id':boot()})
    while True:
        value=read(path)
        if value.get('owner')!=OWNER or value.get('nonce')!=nonce or not value.get('parent_pause_outstanding'):return
        if value.get('boot_id')!=boot():return
        if not same(value['coordinator'],proc.process_info(value['coordinator']['pid'])) or time.monotonic()>=value['worker_deadline_monotonic']:
            finish_pause(path,nonce,'coordinator_dead_or_lease_expired');return
        time.sleep(.1)


def start_guardian(root,path,value):
    command=[sys.executable,'-u',str(root/SCRIPT),'--guardian',str(path),'--nonce',value['nonce']]
    process=subprocess.Popen(command,cwd=root,start_new_session=True)
    ready=Path(str(path)+'.'+value['nonce']+'.guardian.json');deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('Guardian exited before readiness; controller never stopped')
        if ready.exists():
            proof=read(ready);current=proc.process_info(process.pid)
            if proof.get('owner')!=OWNER or proof.get('nonce')!=value['nonce'] or proof.get('boot_id')!=boot() or not same(proof.get('identity'),current) or current['argv']!=command:
                raise ValueError('Guardian readiness/birth/argv differs')
            return process
        time.sleep(.025)
    raise TimeoutError('Independent guardian readiness missing; controller never stopped')


def zero_exit(expected):
    """A paused parent cannot reap its own zombie; inspect exact raw exit code."""
    current=proc.process_info(expected['pid'])
    if current is None:return False
    if current.get('start_time')!=expected['start_time'] or current.get('ppid')!=expected['ppid'] or current.get('argv') not in ([],expected['argv']):
        raise ValueError('Native evaluator PID reused/foreign command; no helper launch')
    if current.get('state') not in ('Z','X'):return False
    raw=(Path('/proc')/str(expected['pid'])/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
    if int(parts[19])!=expected['start_time'] or int(parts[1])!=expected['ppid']:raise ValueError('Native exit identity changed')
    if len(parts)<=49 or int(parts[49])!=0:raise ValueError('Native text evaluator did not exit naturally with code0')
    return True


def wait_native_exit(root,path,nonce,evidence):
    while True:
        value=lease(path,nonce);assert_arm(root,value['bound'])
        if time.monotonic()>=min(value['worker_deadline_monotonic'],value['pause_started_monotonic']+90):
            raise TimeoutError('Bound native evaluator did not naturally finish/idle within90s')
        if zero_exit(evidence['child']) and gpu_idle():return
        time.sleep(.1)


def wait_parent_stopped(path,nonce):
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        value=lease(path,nonce);current=proc.process_info(value['bound']['parent']['pid'])
        if not same(value['bound']['parent'],current):raise ValueError('Controller changed before STOP acknowledgement')
        if current['state'] in ('T','t'):return
        time.sleep(.01)
    raise TimeoutError('Controller STOP acknowledgement missing; finally cancels committed STOP')


def verify_native_boundary(root,bound,evidence):
    """Read-only frozen native validators; no old coordinator owner/actions."""
    from scripts import eval_research_batch4_at_boundary as native_checks
    from scripts.research_grounding_queue import GroundingResearchQueue
    state=read(root/(SPEC.removesuffix('.json')+'.state.json'));row=state['jobs'][TARGET]
    source=native_checks.context(root,read(root/SPEC),bound['job'],row)
    if source['run']!=bound['run']:raise ValueError('Native source context moved')
    native_checks.verify_completed_variant(root,bound['job'],source,{'variant':'text_only','variant_output':evidence['output']})
    verifier=GroundingResearchQueue.__new__(GroundingResearchQueue)
    from scripts.validate_research_checkpoint import managed_checkpoints
    from src.utils.research_checkpoint_compat import checkpoint_identity
    run=Path(bound['run']);config=read(run/'resolved_config.json');entries=managed_checkpoints(run)
    for entry in entries:entry['artifact_identity']=checkpoint_identity(entry['path'])
    from scripts.research_grounding_queue import checkpoint_selection_binding
    checked=verifier._validate_checkpoint_report({'run':run,'config':config,'entries':entries,
        'config_sha256':sha(run/'resolved_config.json'),'selection_binding':checkpoint_selection_binding(config)})
    if checked['report_sha256']!=source['roundtrip_sha256']:raise ValueError('Both retained replay receipt changed')
    output=Path(evidence['output'])
    return {'native_source_context':source,'retained_replay':checked,
        'text_only_files_sha256':{name:sha(output/name) for name in ('summary.json','protocol.json','predictions.jsonl','transformed_manifest.jsonl','progress.json')}}


def helper_command(root,value):
    own=value['coordinator'];remaining=int(value['worker_deadline_monotonic']-time.monotonic())
    if remaining<120:raise TimeoutError('Insufficient bounded work time after native validation')
    return [sys.executable,'-u',str(root/HELPER),'--cpu-report',value['bound']['cpu_report'],'--output',str(root/OUTPUT),
        '--allow-gpu','--nonce',value['nonce'],'--parent-pid',str(own['pid']),'--parent-start',str(own['start_time']),
        '--parent-command-sha256',own['command_sha256'],'--deadline-monotonic',str(value['worker_deadline_monotonic']),
        '--timeout-seconds',str(min(remaining,1170)),'--reserve-gb','120']


def exec_worker(root,path,nonce):
    value=lease(path,nonce);parent=value['coordinator'];actual=proc.process_info(os.getppid())
    if not same(parent,actual) or os.getppid()!=parent['pid']:raise ValueError('Fresh worker gateway parent differs')
    proc.arm_parent_death(parent['pid'])
    if not same(parent,proc.process_info(parent['pid'])):raise ValueError('Parent changed before helper exec')
    own=proc.process_info(os.getpid());command=value['worker_command']
    with lease_lock(path):
        value=lease(path,nonce)
        current_parent=proc.process_info(value['bound']['parent']['pid'])
        if not same(value['bound']['parent'],current_parent) or current_parent.get('state') not in ('T','t'):
            raise ValueError('Helper gateway has no exact owned paused controller')
        if time.monotonic()>=value['worker_deadline_monotonic']:raise TimeoutError('Work lease expired before exec')
        persist(Path(str(path)+'.'+nonce+'.worker.json'),{'owner':OWNER,'nonce':nonce,'boot_id':boot(),
            'identity':stable(own),'launcher':own['argv'],'command':command,'coordinator':parent})
    os.execv(command[0],command)


def raw_birth(pid):
    try:
        raw=(Path('/proc')/str(pid)/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
        return {'pid':pid,'start_time':int(parts[19]),'ppid':int(parts[1]),'state':parts[0]}
    except (OSError,ValueError,IndexError):return None


def await_worker_startup(process,path,value,launcher,deadline):
    expected=raw_birth(process.pid)
    if expected is None or expected['ppid']!=value['coordinator']['pid']:raise ValueError('Cannot immediately bind auxiliary raw birth/PPID')
    update_lease(path,value['nonce'],worker_raw_birth=expected)
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('Owned gateway/helper exited before startup handshake')
        raw=raw_birth(process.pid)
        if raw is None or any(raw[k]!=expected[k] for k in ('pid','start_time','ppid')):
            raise ValueError('Auxiliary PID birth/PPID changed at startup')
        current=proc.process_info(process.pid)
        if current is not None and current.get('argv') not in ([],launcher,value['worker_command']):
            raise ValueError('Foreign nonempty auxiliary argv; never granted grace')
        registration=worker_registration(path,value)
        if registration and current is not None and current.get('argv')==value['worker_command']:
            if registration['identity']['pid']!=process.pid or registration['identity']['start_time']!=expected['start_time']:
                raise ValueError('Gateway registration birth differs')
            updated=update_lease(path,value['nonce'],worker=stable(current),status='running_archive_parity')
            value.update(updated);return
        time.sleep(.025)
    raise TimeoutError('Auxiliary15s missing/empty/fresh-exec handshake deadline expired')


def validate_result(bound,output,nonce,*,worker,coordinator,deadline):
    from src.utils import research_checkpoint_archival as publication
    from src.utils import research_archive_gpu_parity as parity
    report=read(Path(output)/'report.json');cpu=read(bound['cpu_report']);plan=bound['plan']
    required={'schema_version':1,'protocol':publication.GPU_PROTOCOL,'status':'passed','producer':'ROOT','device':'cuda',
        'run':bound['run'],'nonce':nonce,'source_state_sha256':cpu['source_sha256'],'archive_sha256':cpu['archive_sha256'],
        'source_payload_sha256':cpu['source_payload_sha256'],'restored_payload_sha256':cpu['restored_payload_sha256'],
        'all_model_tensors_exact':True,'generated_token_ids_exact':True,'generation_records':64,
        'selection_replay_exact':True,'selection_raw_source_archive_exact':True,'selection_records':116,
        'projector_frozen_unchanged':True,'source_authoritative':True,'publication_performed':False,
        'code_sha256':publication.code_binding(),'helper_source_sha256':parity.source_binding()}
    if any(report.get(k)!=v for k,v in required.items()):raise ValueError('Actual ROOT GPU source/archive parity proof differs')
    expected_parent={k:coordinator[k] for k in ('pid','start_time','command_sha256')}
    if (not same(coordinator,report.get('parent')) or not same(worker,report.get('process')) or
        report.get('expected_parent')!=expected_parent or report.get('coordinator_deadline_monotonic')!=deadline or
        type(report.get('deadline_monotonic')) not in (float,int) or report['deadline_monotonic']>deadline):
        raise ValueError('GPU parity parent/worker birth/argv/nonce/shared work deadline differs')
    parity.assert_plan(plan,rehash=True)
    phases=report['phases'];parity.assert_equal_census(phases['source']['model_tensors'],phases['archive']['model_tensors'])
    parity.compare_generations(phases['source']['native64'],phases['archive']['native64'])
    if phases['source']['internal116']!=phases['archive']['internal116']:raise ValueError('Raw internal selection parity differs')
    return {'status':'passed','report':str(Path(output)/'report.json'),'report_sha256':sha(Path(output)/'report.json'),
        'source_authoritative':True,'publication_performed':False}


def execute_helper(root,path,value):
    with lease_lock(path):
        current=lease(path,value['nonce'])
        if not gpu_idle():raise RuntimeError('GPU no longer exclusive before helper launch')
        if os.path.lexists(root/OUTPUT):raise ValueError('Existing auxiliary output preserved; no implicit retry/reuse')
        launcher=[sys.executable,'-u',str(root/SCRIPT),'--exec-worker',str(path),'--nonce',value['nonce']]
        current.update(worker_command=helper_command(root,current),worker_launcher=launcher,
            worker_launch_intent=True,status='launching_archive_parity')
        persist(path,current);value=current
        env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        log=root/'logs/research/archive_gpu_boundary_4f_worker.log';log.parent.mkdir(parents=True,exist_ok=True)
        started=time.monotonic()
        with log.open('a',buffering=1) as handle:
            process=subprocess.Popen(launcher,cwd=root,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    await_worker_startup(process,path,value,launcher,min(started+15,value['worker_deadline_monotonic']))
    heartbeat=time.monotonic()
    while process.poll() is None:
        current=lease(path,value['nonce'])
        if time.monotonic()>=current['worker_deadline_monotonic']:raise TimeoutError('Auxiliary work deadline reached')
        live_info=proc.process_info(process.pid)
        if live_info is not None and live_info.get('argv') not in ([],value['worker_command']):raise ValueError('Auxiliary command changed during execution')
        if live_info is not None and live_info.get('start_time')!=value['worker']['start_time']:raise ValueError('Auxiliary PID reused')
        if time.monotonic()-heartbeat>=30:emit('archive_gpu_helper_running',pid=process.pid,remaining_seconds=current['worker_deadline_monotonic']-time.monotonic());heartbeat=time.monotonic()
        time.sleep(.1)
    if process.returncode!=0:raise RuntimeError('Owned GPU helper returned '+str(process.returncode))
    result=validate_result(value['bound'],root/OUTPUT,value['nonce'],worker=value['worker'],
        coordinator=value['coordinator'],deadline=value['worker_deadline_monotonic'])
    update_lease(path,value['nonce'],result=result,worker_returncode=0,status='completed')
    return result


def run(root,cpu_report,cpu_sha,*,controller_pid=CONTROLLER_PID):
    root=Path(root).resolve();path=root/RECEIPT
    lock=path.with_suffix('.lock')
    if lock.is_symlink():raise ValueError('Foreign coordinator-lock symlink preserved')
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if os.path.lexists(path):
            previous=read(path)
            if previous.get('owner')!=OWNER:raise ValueError('Foreign boundary receipt preserved')
            if previous.get('parent_pause_outstanding'):finish_pause(path,previous['nonce'],'explicit_startup_recovery')
            raise ValueError('Previous one-shot receipt preserved; ROOT must explicitly review, never auto-retry')
        bound=arm(root,cpu_report,cpu_sha,controller_pid)
        value={'owner':OWNER,'status':'waiting','boot_id':boot(),'nonce':uuid.uuid4().hex+uuid.uuid4().hex,
            'bound':bound,'coordinator':stable(proc.process_info(os.getpid())),'parent_pause_outstanding':False}
        persist(path,value);evidence=None
        while True:
            assert_arm(root,bound)
            if not same(bound['parent'],proc.process_info(controller_pid)):raise RuntimeError('Armed controller identity changed')
            status,observed=phase(root,bound,evidence)
            if observed is not None and evidence is None:evidence=observed;value['native_child']=evidence;persist(path,value)
            if status=='missed':value.update(status='skipped',reason='explicit4f_text_only_window_missed');persist(path,value);return value
            if status=='ready':break
            time.sleep(.1)
        started=time.monotonic();value.update(status='pause_intent',parent_pause_outstanding=True,stop_committed=False,
            pause_started_monotonic=started,lease_deadline_monotonic=started+LEASE_SECONDS,
            worker_deadline_monotonic=started+LEASE_SECONDS-CLEANUP_SECONDS)
        persist(path,value);guard=None
        try:
            guard=start_guardian(root,path,value)
            if not begin_pause(path,value['nonce']):return read(path)
            wait_parent_stopped(path,value['nonce'])
            status,fresh=phase(root,bound,evidence)
            if status!='ready' or fresh!=evidence:raise RuntimeError('Exact completed native boundary changed after controller STOP')
            wait_native_exit(root,path,value['nonce'],evidence)
            proof=verify_native_boundary(root,bound,evidence)
            with lease_lock(path):
                current=lease(path,value['nonce']);current['native_completed_proof']=proof;persist(path,current);value=current
            execute_helper(root,path,value)
        except BaseException as error:
            with lease_lock(path):
                current=read(path)
                if current.get('owner')==OWNER and current.get('nonce')==value['nonce']:
                    current.update(status='failed',error_type=type(error).__name__,error=str(error));persist(path,current)
            raise
        finally:
            finish_pause(path,value['nonce'],'coordinator_finally')
            if guard is not None:
                try:guard.wait(timeout=2)
                except subprocess.TimeoutExpired:pass
        return read(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-report',type=Path);parser.add_argument('--cpu-report-sha256')
    parser.add_argument('--controller-pid',type=int,default=CONTROLLER_PID)
    parser.add_argument('--guardian',type=Path);parser.add_argument('--exec-worker',type=Path);parser.add_argument('--nonce')
    args=parser.parse_args(argv)
    private=args.guardian or args.exec_worker
    if private is not None:
        if private.resolve()!=(ROOT/RECEIPT).resolve() or args.nonce is None:parser.error('Only exact own receipt/nonce private modes')
        return guardian(private,args.nonce) if args.guardian else exec_worker(ROOT,private,args.nonce)
    if args.cpu_report is None or args.cpu_report_sha256 is None:parser.error('Actual surviving CPU report and explicit SHA required before ARM')
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)};seen=[]
    def interrupted(signum,_):
        if not seen:seen.append(signum);raise InterruptedError('Own boundary coordinator interrupted')
    try:
        for s in handlers:signal.signal(s,interrupted)
        print(json.dumps(run(ROOT,args.cpu_report,args.cpu_report_sha256,controller_pid=args.controller_pid),indent=2))
    finally:
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':main()
