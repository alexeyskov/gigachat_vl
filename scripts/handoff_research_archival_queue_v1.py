#!/usr/bin/env python3
"""Root-only fixed3a/A1 natural-exit handoff, closed unless actual activation.

Default is CPU preflight. No trainer/replay is signaled. A private byte-pinned
copy of the proven controller-only pause lease supplies independent SIGKILL
recovery; all source/spec/state transitions have separate new receipts.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import ModuleType

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OWNER='archival_queue_fixed3a_A1_handoff_v1'
SCRIPT='scripts/handoff_research_archival_queue_v1.py'
TEST='tests/test_research_archival_queue_handoff_v1.py'
READY='configs/research/archival_queue_handoff_v1_ready.json'
RECEIPT='research_results/archival_queue_fixed3a_A1_handoff_v1.json'
RECOVERY='research_results/archival_queue_fixed3a_A1_completion_v1.json'
LEASE_SOURCE='scripts/handoff_research_parallel_grounding.py'
LEASE_SHA256='122c31e748c02649fdccb823c4bf2c67c8062b2cb93b0cb5e6be56bdc22e0c23'
SPEC='configs/research/smallthinker_pilot_scienceqa_context_v2_balanced_queue.json'
TARGET='smallthinker4__siglip2__3a__A1_pilot_scienceqa_disjoint_v1_s42'
OLD_SCRIPT='scripts/research_parallel_grounding_queue.py'
NEW_SCRIPT='scripts/research_archival_parallel_queue_v1.py'
PROOF_RUN='runs/research/smallthinker4__siglip2__4q__SFT_pilot_scienceqa_context_v2_balanced_s42'
CONFIG=Path('/etc/supervisor/conf.d/research-pilot.conf')
OLD_COMMAND='command=/venv/main/bin/python -u '+OLD_SCRIPT+' --spec '+SPEC+' --proof-run '+PROOF_RUN+' --workers 4 --timeout-seconds 1200 --retry-failed'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def same(info,expected):
    return info is not None and all(info.get(key)==expected.get(key) for key in ('pid','ppid','start_time','argv','cwd'))


def _private_lease():
    raw=(ROOT/LEASE_SOURCE).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=LEASE_SHA256:raise ValueError('Frozen pause-lease source changed')
    module=ModuleType('scripts._archival_handoff_private_lease_v1');module.__file__=str(ROOT/LEASE_SOURCE);module.__package__='scripts'
    exec(compile(raw,module.__file__,'exec'),module.__dict__)
    module.OWNER=OWNER;module.SCRIPT=SCRIPT;module.TEST=TEST;module.RECEIPT=RECEIPT
    module.CONFIG=CONFIG
    module.OLD_SCRIPT=OLD_SCRIPT;module.NEW_SCRIPT=NEW_SCRIPT;module.same_process=same
    def resume(record):
        if (record.get('owner')!=OWNER or not record.get('parent_pause_outstanding') or not record.get('stop_committed') or
            record.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip()):return
        info=module.old.process_info(record.get('controller',{}).get('pid'))
        if same(info,record['controller']) and old_controller_info(module,info) and info.get('state') in ('T','t'):
            os.kill(info['pid'],signal.SIGCONT)
    module.resume_owned_parent=resume
    return module


def old_controller_info(lease,info):
    args=lease.old.options(info,ROOT,OLD_SCRIPT)
    return (args is not None and set(args)=={'--spec','--proof-run','--workers','--timeout-seconds','--retry-failed'} and
        args.get('--retry-failed') is True and (ROOT/args['--spec']).resolve()==(ROOT/SPEC).resolve() and
        (ROOT/args['--proof-run']).resolve()==(ROOT/PROOF_RUN).resolve() and args['--workers']=='4' and args['--timeout-seconds']=='1200')


def readiness(activation_sha):
    from scripts import research_archival_parallel_queue_v1 as queue
    proof=read(ROOT/READY)
    if proof.get('owner')!=OWNER or proof.get('status')!='passed' or proof.get('tests_passed',0)<1:raise ValueError('Archival handoff CPU proof is not passed')
    for name,expected in proof['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Archival handoff source changed: '+name)
    evidence=queue.activation(activation_sha);queue_ready=queue.cpu_readiness();spec=queue.spec_snapshot(ROOT/SPEC)
    manifests={str(queue.parent.resolve_path(job['evaluation']['manifest'])) for job in spec['jobs'] if job.get('evaluation')}
    if len(manifests)!=1:raise ValueError('Unchanged fixed512 manifest required')
    adoption=queue.parent.adoption_evidence(ROOT/PROOF_RUN,next(iter(manifests)),4)
    return spec,{'spec_file_sha256':queue.SPEC_SHA256,'canonical_spec_sha256':queue.SPEC_CANONICAL_SHA256,
        'activation':evidence,'queue_ready':queue_ready,'handoff_ready_sha256':sha(ROOT/READY),'actual_parallel_adoption_sha256':digest(adoption)}


def completed_A1(spec,*,hash_payloads=False):
    from scripts.research_queue import argv_value,output_dir
    job=next(job for job in spec['jobs'] if job['id']==TARGET);run=output_dir(job['argv'])
    config=read(run/'resolved_config.json');done=read(run/'completed.json');manager=read(run/'.checkpoint_manager.json')
    if (config.get('stage')!='A1' or config.get('recipe')!='3a' or config.get('model')!='Tiiny/SmallThinker-4BA0.6B-Instruct' or
        config.get('manifest_sha256')!=sha((ROOT/config['manifest']).resolve()) or done.get('steps')!=16 or done.get('records')!=1000 or
        manager.get('owner') is None or not (run/'.checkpoint.lock').is_file()):raise ValueError('Fixed3a A1 completed16/1000 source/manager proof differs')
    files={name:{'path':str(run/name),'sha256':sha(run/name)} for name in ('resolved_config.json','completed.json','.checkpoint_manager.json')}
    roles={}
    for name in ('best','last'):
        pointer=run/name;target=pointer.resolve(strict=True);marker=read(target/'.research_checkpoint.json');state=target/'state.pt'
        if (not pointer.is_symlink() or target.parent!=run/'.checkpoint_objects' or marker.get('owner')!=manager['owner'] or
            type(marker.get('step')) is not int or not 0<marker['step']<=16 or (name=='last' and marker['step']!=16) or
            state.is_symlink() or not state.is_file()):raise ValueError('Fixed3a A1 role/terminalLAST identity differs')
        s=state.stat();roles[name]={'pointer':os.readlink(pointer),'object':str(target),'marker_sha256':sha(target/'.research_checkpoint.json'),
            'marker':marker,'state':str(state),'state_stat':[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]}
        if hash_payloads:roles[name]['state_sha256']=sha(state)
    return {'job_id':TARGET,'run':str(run),'argv':job['argv'],'manager_owner':manager['owner'],'files':files,'roles':roles,'steps':16,'records':1000}


def phase(lease,spec,state,parent,expected=None):
    if state.get('spec_sha256')!=digest(spec) or state.get('status')!='running':return 'missed',None
    order=[job['id'] for job in spec['jobs']];index=order.index(TARGET);rows=state.get('jobs',{})
    if any(rows.get(name,{}).get('status') not in (None,'pending','blocked_dependency') for name in order[index+1:]):return 'missed',None
    row=rows.get(TARGET)
    if row is None:return 'waiting',None
    if row.get('status')!='running':return 'missed',None
    trainer=lease.old.process_info(row.get('pid'))
    if trainer is None:return 'missed',None
    job=spec['jobs'][index];command=row.get('command',[])
    if (trainer.get('ppid')!=parent['pid'] or trainer.get('argv')!=command or
        len(command)<4 or Path(command[2]).resolve()!=(ROOT/'scripts/run_research_stage.py').resolve() or
        command[3:] not in (job['argv'],[*job['argv'],'--resume'])):raise ValueError('Fixed3a A1 trainer exact birth/PPID/argv differs')
    stable=lease.stable_identity(trainer)
    if expected is not None and stable!=expected:raise ValueError('Fixed3a A1 trainer was replaced; no retarget')
    try:proof=completed_A1(spec)
    except FileNotFoundError:return 'waiting',stable
    return 'ready',{'trainer':stable,'completion':proof,'pre_state':copy.deepcopy(state)}


def zombie_exit(expected):
    try:
        raw=Path(f"/proc/{expected['pid']}/stat").read_text();fields=raw[raw.rfind(')')+2:].split()
    except OSError:return None
    if int(fields[1])!=expected['ppid'] or int(fields[19])!=expected['start_time']:raise ValueError('A1 trainer PID birth/parent changed during natural exit')
    if fields[0]!='Z':return None
    code=int(fields[49])
    if not os.WIFEXITED(code) or os.WEXITSTATUS(code)!=0:raise ValueError('A1 trainer did not exit naturally with code0')
    return {'pid':expected['pid'],'ppid':expected['ppid'],'start_time':expected['start_time'],'kernel_state':'Z','wait_status':code,'exit_code':0,'trainer_signaled':False}


def recover_completed_training(lease,spec,record):
    """Only one receipt-proven normal progress transition; no schema migration."""
    from scripts.research_queue import artifacts_for
    path=(ROOT/SPEC).with_suffix('.state.json');state=read(path);before=record['pre_state']
    destination=ROOT/RECOVERY
    if destination.exists():
        receipt=read(destination);bound=state.get('extra',{}).get(OWNER,{})
        if (receipt.get('owner')!=OWNER or receipt.get('identity')!=record['identity'] or
            receipt.get('completed_source')!=record['completion_hashed'] or receipt.get('trainer_exit')!=record['trainer_exit'] or
            bound!={'completion_receipt':str(destination),'sha256':sha(destination),'optimizer_updates_added':0} or
            digest(state.get('jobs',{}).get(TARGET))!=receipt.get('new_target_sha256')):
            raise ValueError('Existing fixed A1 completion transition differs')
        return {'path':str(destination),'sha256':sha(destination)}
    if state.get('spec_sha256')!=record['identity']['canonical_spec_sha256']:raise ValueError('Completion recovery SPEC differs')
    if {k:v for k,v in state['jobs'].items() if k!=TARGET}!={k:v for k,v in before['jobs'].items() if k!=TARGET}:
        raise ValueError('Unrelated job records changed during fixed handoff')
    row=state['jobs'][TARGET];old_row=before['jobs'][TARGET]
    ignore={'status','returncode','finished'}
    if ({k:v for k,v in row.items() if k not in ignore}!={k:v for k,v in old_row.items() if k not in ignore} or
        row.get('status')!='interrupted' or row.get('returncode')!=0 or record.get('trainer_exit',{}).get('exit_code')!=0):
        raise ValueError('Only exact naturally completed interrupted3a A1 can transition')
    if completed_A1(spec,hash_payloads=True)!=record['completion_hashed']:raise ValueError('RetainedA1 source changed before durability transition')
    job=next(job for job in spec['jobs'] if job['id']==TARGET)
    updated=copy.deepcopy(row);updated.update(training_completed=True,status='completed',artifacts=artifacts_for(job))
    receipt={'owner':OWNER,'schema_version':1,'target_job':TARGET,'identity':record['identity'],
        'trainer':record['trainer'],'trainer_exit':record['trainer_exit'],'completed_source':record['completion_hashed'],
        'pre_handoff_state_sha256':digest(before),'old_target_sha256':digest(row),'new_target_sha256':digest(updated),
        'old_status':'interrupted','new_status':'completed','optimizer_updates_added':0}
    if destination.exists() and read(destination)!=receipt:raise ValueError('Existing completion receipt differs')
    lease._atomic_json(destination,receipt)
    state['jobs'][TARGET]=updated
    state.setdefault('extra',{})[OWNER]={'completion_receipt':str(destination),'sha256':sha(destination),'optimizer_updates_added':0}
    lease._atomic_json(path,state)
    return {'path':str(destination),'sha256':sha(destination)}


def replacement_command(activation_sha):
    return 'command=/venv/main/bin/python -u '+NEW_SCRIPT+' --spec '+SPEC+' --proof-run '+PROOF_RUN+' --workers 4 --timeout-seconds 1200 --activation-sha '+activation_sha+' --run-queue --retry-failed'


def replacement_controller(lease,status,activation_sha):
    match=lease.old.re.search(r'\bpid\s+(\d+)\b',status)
    info=lease.old.process_info(int(match[1])) if match else None
    args=lease.old.options(info,ROOT,NEW_SCRIPT)
    expected={'--spec','--proof-run','--workers','--timeout-seconds','--activation-sha','--run-queue','--retry-failed'}
    if (args is None or set(args)!=expected or args.get('--activation-sha')!=activation_sha or
        args.get('--run-queue') is not True or args.get('--retry-failed') is not True or
        (ROOT/args['--spec']).resolve()!=(ROOT/SPEC).resolve() or (ROOT/args['--proof-run']).resolve()!=(ROOT/PROOF_RUN).resolve() or
        args['--workers']!='4' or args['--timeout-seconds']!='1200'):return None
    return info


def launch_replacement(lease,spec,identity,record,activation_sha):
    if record.get('identity')!=identity or record.get('trainer_exit',{}).get('exit_code')!=0:
        raise ValueError('Durable activation/natural-exit proof differs')
    if not lease.stopped(lease.old.service_status('research-pilot')):raise ValueError('Only EXITED/STOPPED old controller may be replaced')
    if completed_A1(spec,hash_payloads=True)!=record['completion_hashed']:raise ValueError('RetainedA1 changed during handoff recovery')
    lease.old.wait_gpu_idle(timeout=30)
    record['completion_recovery']=recover_completed_training(lease,spec,record)
    if readiness(activation_sha)!=(spec,identity):raise ValueError('Activation/source changed before replacement')
    from scripts.research_parallel_grounding_queue import assert_no_live_workers
    assert_no_live_workers(spec)
    original=CONFIG.read_text();replacement='\n'.join(replacement_command(activation_sha) if line==OLD_COMMAND else line for line in original.splitlines())+'\n'
    if hashlib.sha256(original.encode()).hexdigest() not in (record['source_supervisor_sha256'],record['replacement_supervisor_sha256']):
        raise ValueError('Supervisor bytes changed outside this one-shot handoff')
    backup=ROOT/'research_results/archival_queue_source_supervisor_v1.conf'
    if backup.exists() and sha(backup)!=record['source_supervisor_sha256']:raise ValueError('Foreign Supervisor backup preserved')
    if not backup.exists():
        if hashlib.sha256(original.encode()).hexdigest()!=record['source_supervisor_sha256']:raise ValueError('Original Supervisor backup missing')
        with backup.open('x') as stream:stream.write(original);stream.flush();os.fsync(stream.fileno())
    record.update(status='starting',parent_pause_outstanding=False);lease._atomic_json(ROOT/RECEIPT,record)
    if original!=replacement:lease.atomic_config(replacement)
    subprocess.run(['supervisorctl','reread'],check=True);subprocess.run(['supervisorctl','update','research-pilot'],check=True)
    if lease.stopped(lease.old.service_status('research-pilot')):subprocess.run(['supervisorctl','start','research-pilot'],check=True)
    info=replacement_controller(lease,lease.old.wait_for_running('research-pilot'),activation_sha)
    if info is None:raise ValueError('Replacement exact Supervisor PID/birth/argv was not confirmed')
    record.update(status='completed',replacement_controller=lease.stable_identity(info),training_spec_unchanged=True,
        optimizer_updates_added=0,source_checkpoints_preserved=True)
    lease._atomic_json(ROOT/RECEIPT,record);return record


def run(controller_pid,activation_sha):
    lease=_private_lease();path=ROOT/RECEIPT;previous=read(path) if path.exists() else {}
    if previous and previous.get('owner')!=OWNER:raise ValueError('Foreign archival handoff receipt preserved')
    if previous.get('parent_pause_outstanding'):lease.revoke_pause(path,previous['lease_nonce'],'explicit_restart_recovery')
    spec,identity=readiness(activation_sha)
    if previous:
        if previous.get('status')=='completed':
            info=replacement_controller(lease,lease.old.service_status('research-pilot'),activation_sha)
            if previous.get('identity')!=identity or not same(info,previous.get('replacement_controller',{})):
                raise ValueError('Completed replacement identity changed; no service action')
            return previous
        if previous.get('status') in ('stopping','starting'):
            status=lease.old.service_status('research-pilot')
            if not lease.stopped(status):raise ValueError('Unbound running replacement needs explicit reconciliation; no signals')
            return launch_replacement(lease,spec,identity,previous,activation_sha)
        raise ValueError('Existing skipped/revoked one-shot is preserved; no automatic retarget')
    parent=lease.old.process_info(controller_pid)
    status=lease.old.service_status('research-pilot')
    if f'pid {controller_pid},' not in status or not old_controller_info(lease,parent):raise ValueError('Exact Supervisor-owned parallel controller required')
    parent=lease.stable_identity(parent);expected=None
    while True:
        current=lease.old.process_info(controller_pid)
        if not same(current,parent):raise ValueError('Controller birth/argv changed; no retarget')
        state=read((ROOT/SPEC).with_suffix('.state.json'));kind,bound=phase(lease,spec,state,current,expected)
        if kind=='missed':
            result={'owner':OWNER,'status':'skipped_window_missed','target_job':TARGET,'signals_sent':False};lease._atomic_json(path,result);return result
        if kind=='ready':break
        if bound is not None:expected=bound
        time.sleep(.1)
    original=CONFIG.read_text();new_line=replacement_command(activation_sha)
    if (original.splitlines().count(OLD_COMMAND)!=1 or [line for line in original.splitlines() if line.startswith('[program:')]!=['[program:research-pilot]'] or
        len([line for line in original.splitlines() if line.startswith('command=')])!=1 or 'autorestart=false' not in original):
        raise ValueError('Only exact current research-pilot configuration is supported')
    replacement='\n'.join(new_line if line==OLD_COMMAND else line for line in original.splitlines())+'\n'
    record={'owner':OWNER,'status':'pause_intent','identity':identity,'controller':parent,**bound,
        'target_job':TARGET,'parent_pause_outstanding':True,'stop_committed':False,'lease_nonce':os.urandom(16).hex(),
        'pause_deadline_monotonic':time.monotonic()+lease.PAUSE_LEASE_SECONDS,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'handoff_process':lease.stable_identity(lease.old.process_info(os.getpid())),
        'source_supervisor_sha256':hashlib.sha256(original.encode()).hexdigest(),'replacement_supervisor_sha256':hashlib.sha256(replacement.encode()).hexdigest()}
    lease._atomic_json(path,record);watchdog=None
    try:
        watchdog=lease.start_watchdog(ROOT,path,record)
        if not lease.begin_owned_pause(path,record['lease_nonce'],parent):return read(path)
        deadline=time.monotonic()+10
        while True:
            observed=lease.old.process_info(controller_pid)
            if not same(observed,parent):raise ValueError('Controller changed while observing STOP')
            if observed.get('state') in ('T','t'):break
            if time.monotonic()>deadline:raise TimeoutError('Controller STOP not observed')
            time.sleep(.025)
        deadline=record['pause_deadline_monotonic']-20
        while True:
            exit_proof=zombie_exit(record['trainer'])
            if exit_proof is not None:break
            if time.monotonic()>deadline:raise TimeoutError('A1 natural exit missed bounded lease')
            time.sleep(.025)
        record.update(trainer_exit=exit_proof,completion_hashed=completed_A1(spec,hash_payloads=True))
        if readiness(activation_sha)!=(spec,identity):raise ValueError('Actual activation/source/SPEC changed during pause')
        with lease.lease_lock(path):
            durable=read(path)
            if durable.get('lease_revoked') or not durable.get('parent_pause_outstanding') or time.monotonic()>=durable['pause_deadline_monotonic']:
                raise TimeoutError('Late interrupt after lease revocation refused')
            if not same(lease.old.process_info(controller_pid),parent):raise ValueError('Controller changed before INT')
            record['status']='stopping';lease._atomic_json(path,record)
            os.kill(controller_pid,signal.SIGINT)
    finally:
        lease.finish_owned_pause(path,record['lease_nonce'])
        if watchdog is not None:
            try:watchdog.wait(timeout=2)
            except subprocess.TimeoutExpired:pass
    deadline=time.monotonic()+120
    while not lease.stopped(lease.old.service_status('research-pilot')):
        if time.monotonic()>deadline:raise TimeoutError('Old controller failed graceful shutdown')
        time.sleep(.1)
    return launch_replacement(lease,spec,identity,read(path),activation_sha)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--controller-pid',type=int)
    parser.add_argument('--activation-sha');parser.add_argument('--operate',action='store_true')
    parser.add_argument('--watchdog');parser.add_argument('--nonce');args=parser.parse_args(argv)
    if args.watchdog:
        if Path(args.watchdog).resolve()!=(ROOT/RECEIPT).resolve() or not args.nonce:raise ValueError('Owned watchdog path/nonce required')
        _private_lease().watchdog(ROOT,Path(args.watchdog),args.nonce);return 0
    if not args.operate:
        try:readiness(args.activation_sha)
        except (OSError,ValueError) as error:print(json.dumps({'owner':OWNER,'status':'closed_preflight','reason':str(error),'signals_sent':False}));return 2
        print(json.dumps({'owner':OWNER,'status':'metadata_ready_only','signals_sent':False}));return 0
    if os.geteuid()!=0 or args.controller_pid is None:raise ValueError('Explicit root operate/controller PID required')
    path=ROOT/RECEIPT;path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=run(args.controller_pid,args.activation_sha)
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
