#!/usr/bin/env python3
"""Root-only proposal: evaluator-only handoff after actual full512 proof.

Never signal a trainer or replay process. The only controller interrupted is
the exact Supervisor-owned native wrapper at a completed compact SFT's live
native evaluator boundary. Existing specification/state/outputs stay in place.
No service action occurs on import; only root may explicitly invoke main().
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import handoff_research_grounding as old
from scripts.research_queue import argv_value
from src.utils.research_storage import _atomic_json,_fsync_directory,DiskBudget

OWNER='parallel_grounding_controller_handoff_v1'
SCRIPT='scripts/handoff_research_parallel_grounding.py'
TEST='tests/test_research_parallel_grounding_handoff.py'
READY='configs/research/parallel_grounding_handoff_ready.json'
RECEIPT='research_results/parallel_grounding_handoff.json'
SPEC=old.SPEC
CONFIG=Path('/etc/supervisor/conf.d/research-pilot.conf')
OLD_SCRIPT='scripts/research_grounding_queue.py'
NEW_SCRIPT='scripts/research_parallel_grounding_queue.py'
PROOF_RUN='runs/research/smallthinker4__siglip2__4q__SFT_pilot_scienceqa_context_v2_balanced_s42'
OLD_COMMAND='command=/venv/main/bin/python -u '+OLD_SCRIPT+' --spec '+SPEC+' --retry-failed'
NEW_COMMAND=('command=/venv/main/bin/python -u '+NEW_SCRIPT+' --spec '+SPEC+
             ' --proof-run '+PROOF_RUN+' --workers 4 --timeout-seconds 1200 --retry-failed')
CHECKS=('actual_full512_proof_before_any_stop','same24job_spec_and_completed_compact_sft',
        'exact_live_native_child_never_trainer_or_replay','controller_only_stop_revalidate_interrupt_finally_cont',
        'durable_pause_intent_known_birth_recovery','graceful_service_exit_and_gpu_idle_before_replacement',
        'only_research_pilot_config_changes','completed_native_and_partial_outputs_preserved',
        'starting_recovery_refuses_unbound_running_pid')
PAUSE_LEASE_SECONDS=120

read=old.read_json
at=old.at
sha=lambda path:old.digest(Path(path).read_bytes())


def same_process(info,expected):
    return (info is not None and info.get('pid')==expected.get('pid') and
            info.get('start_time')==expected.get('start_time') and
            info.get('argv')==expected.get('argv') and info.get('cwd')==expected.get('cwd'))


def stable_identity(info):
    return {key:info[key] for key in ('pid','ppid','start_time','argv','cwd')}


def controller(root,status,script):
    if 'RUNNING' not in status:return None
    match=old.re.search(r'\bpid\s+(\d+)\b',status)
    info=old.process_info(int(match[1])) if match else None
    args=old.options(info,root,script)
    expected={'--spec','--retry-failed'} if script==OLD_SCRIPT else {
        '--spec','--proof-run','--workers','--timeout-seconds','--retry-failed'}
    if args is None or set(args)!=expected or args.get('--retry-failed') is not True or at(root,args['--spec'])!=(root/SPEC).resolve():return None
    if script==NEW_SCRIPT and (at(root,args['--proof-run'])!=(root/PROOF_RUN).resolve() or
                               args.get('--workers')!='4' or args.get('--timeout-seconds')!='1200'):return None
    return info


def cpu_readiness(root):
    proof=read(root/READY)
    if (proof.get('owner')!=OWNER or proof.get('status')!='passed' or
        type(proof.get('tests_passed')) is not int or proof['tests_passed']<1 or not proof.get('test_command') or
        any(proof.get('checks',{}).get(name) is not True for name in CHECKS)):
        raise ValueError('Parallel handoff CPU readiness is not passed')
    old.verify_sources(root,proof.get('source_sha256'),(SCRIPT,TEST,NEW_SCRIPT,'scripts/research_queue.py',
        'scripts/eval_research_checkpoint_parallel_b1.py'))
    spec,snapshot=old.spec_snapshot(root)
    manifests={str(at(root,j['evaluation']['manifest'])) for j in spec['jobs'] if j.get('evaluation')}
    if len(manifests)!=1:raise ValueError('Expected the same fixed512 manifest across the immutable queue')
    weighted=read(root/'configs/research/weighted_sft_ready.json')
    if weighted.get('status')!='passed':raise ValueError('Weighted trainer proof is not passed')
    old.verify_sources(root,weighted.get('source_sha256'),('scripts/run_research_stage.py',))
    return spec,snapshot,next(iter(manifests))


def readiness(root):
    spec,snapshot,manifest=cpu_readiness(root)
    # Lazy import: the handoff module itself stays stdlib-only. This strict
    # public reader constructs no model and must pass BEFORE any signal/write.
    from scripts import research_parallel_grounding_queue as proposal
    actual=proposal.adoption_evidence(root/PROOF_RUN,manifest,4)
    return spec,dict(snapshot,handoff_ready_sha256=sha(root/READY),
        proposal_ready_sha256=sha(root/proposal.READY),evaluator_ready_sha256=sha(root/proposal.EVALUATOR_READY),
        weighted_ready_sha256=sha(root/'configs/research/weighted_sft_ready.json'),
        actual_full512_proof_sha256=old.digest(json.dumps(actual,sort_keys=True).encode()))


def wait_for_actual_proof(root,controller_pid):
    """Explicit root standby; missing proof never permits signals or writes."""
    baseline=cpu_readiness(root)
    anchor=controller(root,old.service_status('research-pilot'),OLD_SCRIPT)
    if anchor is None or anchor['pid']!=controller_pid:raise RuntimeError('Exact native controller is required while waiting for actual proof')
    anchor=stable_identity(anchor)
    output=root/PROOF_RUN/'evaluation/parallel_b1_v1/workers4/none'
    required=('protocol.json','predictions.jsonl','summary.json','execution_summary.json','parallel_guard_passed.json')
    last=-float('inf')
    while True:
        if cpu_readiness(root)!=baseline:raise ValueError('Tested source/spec changed while waiting for actual proof')
        if not same_process(controller(root,old.service_status('research-pilot'),OLD_SCRIPT),anchor):
            raise RuntimeError('Native controller identity changed; no automatic standby retarget')
        if all((output/name).is_file() for name in required):
            return readiness(root)  # Strict raw512 verifier; filenames are never proof.
        if time.monotonic()-last>=30:
            print(json.dumps({'event':'parallel_handoff_waiting_actual_full512','controller_pid':controller_pid}),flush=True)
            last=time.monotonic()
        time.sleep(5)


def safe_boundary(root,spec,state,parent):
    canonical=old.digest(json.dumps(spec,sort_keys=True,separators=(',',':')).encode())
    if state.get('spec_sha256')!=canonical:raise ValueError('Native state/specification SHA differs')
    rows=state.get('jobs')
    if not isinstance(rows,dict) or any(not isinstance(v,dict) for v in rows.values()):raise ValueError('Malformed native queue state')
    if state.get('status')!='running' or parent is None:return None
    active=[(name,row) for name,row in rows.items() if row.get('status') in
            ('running','evaluating','checkpoint_validating','grounding_evaluating','parallel_evaluating')]
    if len(active)!=1:return None
    name,row=active[0]
    if row.get('status') not in ('evaluating','grounding_evaluating') or row.get('training_completed') is not True:return None
    job=next((j for j in spec['jobs'] if j['id']==name),None)
    compact=old.compacted_run(root,job) if job else None
    if compact is None:return None
    variant='none' if row['status']=='evaluating' else row.get('active_grounding_variant')
    if variant not in ('none','shuffle','blank','text_only'):return None
    receipt=row.get('evaluation') if variant=='none' else row.get('grounding_evaluations',{}).get(variant)
    if not isinstance(receipt,dict) or receipt.get('status')!='running':return None
    child=old.process_info(receipt.get('pid'));args=old.options(child,root,'scripts/eval_research_checkpoint.py')
    if child is None or child.get('ppid')!=parent['pid'] or args is None:return None
    run=at(root,argv_value(job['argv'],'--output'));settings=job['evaluation']
    output=at(root,settings['output']) if variant=='none' else run/'evaluation/grounding_v1'/variant
    if (at(root,receipt.get('output'))!=output or at(root,args.get('--output'))!=output or
        at(root,args.get('--run'))!=run or at(root,args.get('--manifest'))!=at(root,settings['manifest']) or
        args.get('--checkpoint')!='best' or args.get('--max-new-tokens')!='64' or
        args.get('--image-ablation','none')!=variant or (variant!='none' and args.get('--ablation-seed')!='42')):return None
    protocol=output/'protocol.json'
    if not protocol.is_file() or protocol.is_symlink():return None
    body=read(protocol)
    if (body.get('manifest_sha256')!=sha(at(root,settings['manifest'])) or
        body.get('image_ablation',{}).get('variant','none')!=variant):return None
    return {'safe_job':name,'variant':variant,'controller':stable_identity(parent),'evaluator':stable_identity(child),
        'native_output':str(output),'native_protocol_sha256':sha(protocol),'compacted':compact}


def replacement_config(original):
    lines=original.splitlines()
    if ([line for line in lines if line.startswith('[program:')]!=['[program:research-pilot]'] or
        sum(line.startswith('command=') for line in lines)!=1 or 'autorestart=false' not in lines or
        not any(line in (OLD_COMMAND,NEW_COMMAND) for line in lines)):
        raise ValueError('Unexpected research-pilot Supervisor configuration; unrelated services stay untouched')
    return '\n'.join(NEW_COMMAND if line==OLD_COMMAND else line for line in lines)+'\n'


def atomic_config(value):
    if CONFIG.is_symlink():raise ValueError('Research Supervisor config cannot be a symlink')
    temporary=CONFIG.with_name('.'+CONFIG.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as stream:stream.write(value);stream.flush();os.fsync(stream.fileno())
        os.chmod(temporary,CONFIG.stat().st_mode&0o777);os.replace(temporary,CONFIG);_fsync_directory(CONFIG.parent)
    finally:temporary.unlink(missing_ok=True)


def resume_owned_parent(record):
    if record.get('owner')!=OWNER or not record.get('parent_pause_outstanding') or not record.get('stop_committed'):return
    if record.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():return
    info=old.process_info(record.get('controller',{}).get('pid'))
    args=old.options(info,ROOT,OLD_SCRIPT)
    owned=(args is not None and set(args)=={'--spec','--retry-failed'} and
           args.get('--retry-failed') is True and at(ROOT,args['--spec'])==(ROOT/SPEC).resolve())
    if owned and same_process(info,record['controller']) and info['state'] in ('T','t'):
        os.kill(info['pid'],signal.SIGCONT)


@contextmanager
def lease_lock(path):
    with Path(str(path)+'.lease.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def revoke_pause(path,nonce,reason):
    """The watchdog and INT sender serialize ownership through this lock."""
    with lease_lock(path):
        record=read(path)
        if record.get('owner')!=OWNER or record.get('lease_nonce')!=nonce or not record.get('parent_pause_outstanding'):return
        record.update(lease_revoked=True,lease_revocation_reason=reason)
        if record.get('status')!='stopping':record['status']='pause_revoked'
        _atomic_json(path,record)  # Revoke BEFORE CONT; a late caller cannot INT.
        resume_owned_parent(record)
        record['parent_pause_outstanding']=False;_atomic_json(path,record)


def watchdog(root,path,nonce):
    record=read(path)
    if (record.get('owner')!=OWNER or record.get('lease_nonce')!=nonce or
        record.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip()):return
    own=old.process_info(os.getpid())
    if own is None:raise RuntimeError('Watchdog own birth unavailable')
    _atomic_json(Path(str(path)+'.'+nonce+'.watchdog.json'),
                 {'owner':OWNER,'nonce':nonce,'identity':stable_identity(own)})
    while True:
        record=read(path)
        if record.get('owner')!=OWNER or record.get('lease_nonce')!=nonce or not record.get('parent_pause_outstanding'):return
        if record.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():return
        if not same_process(old.process_info(record['handoff_process']['pid']),record['handoff_process']):
            revoke_pause(path,nonce,'handoff_process_died_or_changed');return
        if time.monotonic()>=record['pause_deadline_monotonic']:
            revoke_pause(path,nonce,'bounded_pause_lease_expired');return
        time.sleep(.1)


def start_watchdog(root,path,record):
    command=[sys.executable,'-u',str(root/SCRIPT),'--watchdog',str(path),'--nonce',record['lease_nonce']]
    process=subprocess.Popen(command,cwd=root,start_new_session=True)
    ready=Path(str(path)+'.'+record['lease_nonce']+'.watchdog.json');deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('Independent pause watchdog exited before readiness')
        if ready.is_file():
            proof=read(ready);info=old.process_info(process.pid)
            if (proof.get('owner')!=OWNER or proof.get('nonce')!=record['lease_nonce'] or
                    info is None or info.get('argv')!=command or not same_process(info,proof.get('identity',{}))):
                raise ValueError('Independent pause watchdog birth/argv/nonce readiness differs')
            return process
        time.sleep(.025)
    raise TimeoutError('Independent pause watchdog did not become ready; controller was never stopped')


def finish_owned_pause(path,nonce):
    with lease_lock(path):
        record=read(path)
        if record.get('owner')!=OWNER or record.get('lease_nonce')!=nonce:return
        # A revoked lease no longer owns any later STOP state of this PID.
        if record.get('parent_pause_outstanding') and not record.get('lease_revoked'):
            resume_owned_parent(record)
        record['parent_pause_outstanding']=False;_atomic_json(path,record)


def begin_owned_pause(path,nonce,parent):
    """No STOP can be sent after a guardian revoked the same serialized lease."""
    with lease_lock(path):
        record=read(path)
        if (record.get('owner')!=OWNER or record.get('lease_nonce')!=nonce or record.get('lease_revoked') or
            not record.get('parent_pause_outstanding') or time.monotonic()>=record['pause_deadline_monotonic']):
            raise TimeoutError('Pause lease expired/revoked before STOP; no late signal')
        current=old.process_info(parent['pid'])
        if not same_process(current,parent) or current.get('state') in ('T','t'):
            record.update(parent_pause_outstanding=False,status='external_pause_or_changed_parent_declined')
            _atomic_json(path,record);return False
        # Durable intent precedes the syscall so SIGKILL immediately after STOP
        # cannot strand a pause invisible to the independent guardian.
        record['stop_committed']=True;_atomic_json(path,record)
        if time.monotonic()>=record['pause_deadline_monotonic']:
            record.update(lease_revoked=True,parent_pause_outstanding=False,status='pause_revoked')
            _atomic_json(path,record);raise TimeoutError('Pause lease expired during durable STOP intent')
        os.kill(parent['pid'],signal.SIGSTOP)
        return True


def pause_and_interrupt(root,spec,identity,boundary):
    path=root/RECEIPT;parent=boundary['controller'];current=old.process_info(parent['pid'])
    if not same_process(current,parent) or current.get('state') in ('T','t'):return False
    original=CONFIG.read_text();replacement=replacement_config(original)
    if OLD_COMMAND not in original.splitlines():raise ValueError('Native configuration changed before pause')
    record={'owner':OWNER,'status':'pause_intent','boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'parent_pause_outstanding':True,'stop_committed':False,'identity':identity,**boundary,
        'lease_nonce':uuid.uuid4().hex,'pause_deadline_monotonic':time.monotonic()+PAUSE_LEASE_SECONDS,
        'handoff_process':stable_identity(old.process_info(os.getpid())),
        'source_supervisor_sha256':old.digest(original.encode()),'replacement_supervisor_sha256':old.digest(replacement.encode())}
    _atomic_json(path,record)
    guardian=None
    try:
        guardian=start_watchdog(root,path,record)
        if not begin_owned_pause(path,record['lease_nonce'],parent):return False
        deadline=time.monotonic()+5
        while True:
            current=old.process_info(parent['pid'])
            if not same_process(current,parent):raise RuntimeError('Native controller identity changed at pause')
            if current['state'] in ('T','t'):break
            if time.monotonic()>=deadline:raise TimeoutError('Native controller pause did not become observable')
            time.sleep(.025)
        fresh_spec,fresh_identity=readiness(root)
        if fresh_spec!=spec or fresh_identity!=identity:raise ValueError('Immutable source/spec/full512 proof changed at pause')
        fresh=safe_boundary(root,spec,read((root/SPEC).with_suffix('.state.json')),current)
        # Compare stable parent identity, allowing only its STOP state to differ.
        if fresh is not None:fresh['controller']=parent
        with lease_lock(path):
            current_record=read(path)
            if (current_record.get('owner')!=OWNER or current_record.get('lease_nonce')!=record['lease_nonce'] or
                    current_record.get('lease_revoked') or not current_record.get('parent_pause_outstanding') or
                    time.monotonic()>=current_record['pause_deadline_monotonic']):
                raise TimeoutError('Pause lease was revoked/expired; no late controller interrupt')
            current=old.process_info(parent['pid'])
            if not same_process(current,parent) or current.get('state') not in ('T','t'):
                raise RuntimeError('Native controller left the owned pause before interrupt')
            if fresh!=boundary:
                current_record['status']='window_missed';_atomic_json(path,current_record);return False
            current_record['status']='stopping';_atomic_json(path,current_record)
            os.kill(parent['pid'],signal.SIGINT)
            return True
    finally:
        finish_owned_pause(path,record['lease_nonce'])
        if guardian is not None:
            try:guardian.wait(timeout=2)
            except subprocess.TimeoutExpired:pass  # Durable flagfalse ends the independent guardian; never signal an unbound PID.


def stopped(status):return any(value in status for value in ('STOPPED','EXITED'))


def verify_recovery(root,spec,identity,record):
    if record.get('owner')!=OWNER or record.get('identity')!=identity:raise ValueError('Handoff recovery source/proof ownership differs')
    job=next((job for job in spec['jobs'] if job['id']==record.get('safe_job')),None)
    if job is None or old.compacted_run(root,job)!=record.get('compacted'):raise ValueError('Completed compacted SFT identity changed')
    state=read((root/SPEC).with_suffix('.state.json'))
    if state.get('spec_sha256')!=identity['canonical_spec_sha256']:raise ValueError('Recovery state/spec changed')
    row=state.get('jobs',{}).get(record['safe_job'],{})
    if row.get('training_completed') is not True:raise ValueError('Recovery lost durable completed training')
    if sha(Path(record['native_output'])/'protocol.json')!=record['native_protocol_sha256']:raise ValueError('Partial native protocol changed')
    value=CONFIG.read_text();replacement_config(value)
    if old.digest(value.encode()) not in (record['source_supervisor_sha256'],record['replacement_supervisor_sha256']):
        raise ValueError('Supervisor configuration changed outside this handoff')


def assert_no_workers(spec):
    from scripts import research_parallel_grounding_queue as proposal
    proposal.assert_no_live_workers(spec)


def launch_replacement(root,spec,identity,record):
    verify_recovery(root,spec,identity,record)
    if not stopped(old.service_status('research-pilot')):raise RuntimeError('Research pilot must be EXITED/STOPPED; no running process is stopped here')
    old.wait_gpu_idle(timeout=30)
    if readiness(root)!=(spec,identity):raise ValueError('Source/spec/actual proof changed before replacement')
    assert_no_workers(spec)
    current=CONFIG.read_text();replacement=replacement_config(current)
    backup=root/'research_results/parallel_grounding_source_supervisor.conf'
    if backup.exists() and old.digest(backup.read_bytes())!=record['source_supervisor_sha256']:raise ValueError('Foreign Supervisor backup preserved')
    if not backup.exists():
        if old.digest(current.encode())!=record['source_supervisor_sha256']:
            raise ValueError('Original Supervisor backup disappeared; no guessed replacement backup')
        with backup.open('x') as stream:stream.write(current);stream.flush();os.fsync(stream.fileno())
    record.update(status='starting',parent_pause_outstanding=False);_atomic_json(root/RECEIPT,record)
    if current!=replacement:atomic_config(replacement)
    subprocess.run(['supervisorctl','reread'],check=True)
    subprocess.run(['supervisorctl','update','research-pilot'],check=True)
    if stopped(old.service_status('research-pilot')):subprocess.run(['supervisorctl','start','research-pilot'],check=True)
    info=controller(root,old.wait_for_running('research-pilot'),NEW_SCRIPT)
    if info is None:raise RuntimeError('Replacement Supervisor PID/argv does not match the explicit proposal')
    record.update(replacement_controller=info,status='completed',training_spec_unchanged=True,outputs_preserved=True)
    _atomic_json(root/RECEIPT,record)


def run(root,controller_pid):
    path=root/RECEIPT;previous=read(path) if path.exists() else {}
    if previous and previous.get('owner')!=OWNER:raise ValueError('Foreign handoff receipt preserved')
    # Recovery of our own STOP always precedes readiness, even if proof vanished.
    if previous.get('parent_pause_outstanding') and previous.get('lease_nonce'):
        revoke_pause(path,previous['lease_nonce'],'explicit_restart_recovery')
        previous=read(path)
    else:resume_owned_parent(previous)
    spec,identity=readiness(root)  # Actual full512 proof is mandatory before any new stop.
    status=old.service_status('research-pilot')
    if previous.get('status')=='completed':
        verify_recovery(root,spec,identity,previous)
        if not same_process(controller(root,status,NEW_SCRIPT),previous.get('replacement_controller',{})):
            raise RuntimeError('Completed replacement PID/birth no longer matches; no service action')
        return previous
    if previous.get('status') in ('stopping','starting'):
        verify_recovery(root,spec,identity,previous)
        replacement=controller(root,status,NEW_SCRIPT)
        if replacement is not None:
            if not same_process(replacement,previous.get('replacement_controller',{})):
                raise RuntimeError('Running replacement birth was not durably recorded; explicit manual reconciliation required')
            previous.update(status='completed');_atomic_json(path,previous);return previous
        if stopped(status):launch_replacement(root,spec,identity,previous);return read(path)
    parent=controller(root,status,OLD_SCRIPT)
    if parent is None or parent['pid']!=controller_pid:raise RuntimeError('Exact current native Supervisor controller is required')
    while True:
        DiskBudget(root,120).ensure_space(64*2**20)
        current=old.process_info(parent['pid'])
        if not same_process(current,parent):raise RuntimeError('Native controller PID/birth/argv changed; no automatic retarget')
        boundary=safe_boundary(root,spec,read((root/SPEC).with_suffix('.state.json')),current)
        if boundary is not None and pause_and_interrupt(root,spec,identity,boundary):break
        time.sleep(1)
    record=read(path);deadline=time.monotonic()+180
    while not stopped(old.service_status('research-pilot')):
        info=old.process_info(parent['pid'])
        if info is not None and not same_process(info,parent):raise RuntimeError('Controller PID reused during graceful shutdown')
        if time.monotonic()>=deadline:raise TimeoutError('Native evaluator/controller did not exit gracefully')
        time.sleep(1)
    launch_replacement(root,spec,identity,record)
    return read(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--controller-pid',type=int)
    parser.add_argument('--wait-for-proof',action='store_true',help='Root standby: verify tested sources while waiting; no stop before strict actual full512 proof')
    parser.add_argument('--watchdog');parser.add_argument('--nonce')
    args=parser.parse_args(argv)
    if os.geteuid()!=0:raise PermissionError('Only root may explicitly operate the owned research-pilot service')
    if args.watchdog:
        if not args.nonce or Path(args.watchdog).resolve()!=(ROOT/RECEIPT).resolve():raise ValueError('Only the owned nonce-bound watchdog receipt is supported')
        watchdog(ROOT,Path(args.watchdog),args.nonce);return 0
    if args.controller_pid is None:parser.error('Explicit --controller-pid is required')
    path=ROOT/RECEIPT;path.parent.mkdir(parents=True,exist_ok=True)
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)};interrupted=[]
    def stop(sig,_frame):
        if not interrupted:interrupted.append(sig);raise InterruptedError('Parallel handoff interrupted')
    try:
        for sig in previous:signal.signal(sig,stop)
        with path.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.wait_for_proof:
                existing=read(path) if path.exists() else {}
                if existing.get('owner') not in (None,OWNER):raise ValueError('Foreign handoff receipt preserved')
                if existing.get('parent_pause_outstanding') and existing.get('lease_nonce'):
                    revoke_pause(path,existing['lease_nonce'],'explicit_restart_before_proof_wait')
                else:resume_owned_parent(existing)
                wait_for_actual_proof(ROOT,args.controller_pid)
            result=run(ROOT,args.controller_pid)
    except InterruptedError:
        if not interrupted:raise
        return 128+interrupted[0]
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps(result,sort_keys=True),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
