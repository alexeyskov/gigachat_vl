#!/usr/bin/env python3
"""One isolated4q batch probe while a completed1b trainer's controller is paused.

No Supervisor configuration/state is edited. Only the exact native controller
gets SIGSTOP/SIGCONT. Training children are never signalled. The owned benchmark
has a600s deadline and Linux parent-death SIGTERM, preventing an orphan GPU job
if this coordinator crashes before recording its child PID.
"""
from pathlib import Path
import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.handoff_research_grounding import (
    SPEC,OLD_SCRIPT,at,compacted_run,digest,options,process_info,read_json,
    service_status,spec_snapshot,verify_sources,
)
from scripts.research_queue import argv_value
from src.utils.research_checkpoint_compat import checkpoint_identity,same_checkpoint_identity
from src.utils.research_storage import DiskBudget,_atomic_json,_fsync_directory

OWNER='research_batch_boundary_probe_v1'
TARGET_JOB='smallthinker4__siglip2__1b__SFT_pilot_scienceqa_context_v2_balanced_s42'
SOURCE_JOB='smallthinker4__siglip2__4q__SFT_pilot_scienceqa_context_v2_balanced_s42'
REPORT='research_results/batch_inference_4q_actual.json'
RECEIPT='research_results/batch_inference_4q_boundary.json'
LOG='logs/research/batch_inference_4q_actual.log'
CONTROLLER_PID=33707
PROBE_MODULE='src.eval.research_batch_inference'
PROBE_SOURCES=('src/eval/research_batch_inference.py','src/eval/research_inference.py',
               'src/model/research_moe.py','src/model/smallthinker.py')


def emit(event,**values):print(json.dumps(dict(event=event,time=time.time(),**values)),flush=True)
def command_sha(argv):return digest(json.dumps(argv,separators=(',',':')).encode())


def persist(path,record):
    path.parent.mkdir(parents=True,exist_ok=True);_atomic_json(path,record);_fsync_directory(path.parent)


def native_controller(root,pid):
    status=service_status('research-pilot')
    if 'RUNNING' not in status or ('pid '+str(pid)+',') not in status:return None
    info=process_info(pid);args=options(info,root,OLD_SCRIPT)
    if args is None or set(args)!={'--spec'} or at(root,args['--spec'])!=(root/SPEC).resolve():return None
    return info


def gpu_pids():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                          check=True,capture_output=True,text=True)
    rows=[r.strip() for r in result.stdout.splitlines() if r.strip()]
    if any(not row.isdigit() for row in rows):raise RuntimeError('Unrecognized GPU process listing')
    return [int(row) for row in rows]


def host_available_bytes():
    available=None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):available=int(line.split()[1])*1024;break
    if available is None:raise RuntimeError('Host available-memory counter is unavailable')
    # MemAvailable can describe the host even when the container has a smaller
    # cgroup limit. Respect both; no cgroup setting is changed.
    for maximum,current in (('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')):
        try:
            raw=Path(maximum).read_text().strip()
            if raw!='max':available=min(available,max(0,int(raw)-int(Path(current).read_text())))
        except (OSError,ValueError):pass
    return available


def completed_current(root,job):
    directory=at(root,argv_value(job['argv'],'--output'))
    done_path=directory/'completed.json'
    if not done_path.is_file():return None
    if done_path.is_symlink():raise ValueError('Completion marker must be a local file')
    done=read_json(done_path);config=read_json(directory/'resolved_config.json')
    owner=read_json(directory/'.checkpoint_manager.json').get('owner')
    if (not isinstance(done.get('steps'),int) or isinstance(done['steps'],bool) or done['steps']<1 or
        done.get('status') in ('failed','interrupted','incomplete') or config.get('stage')!='SFT' or
        config.get('recipe')!=argv_value(job['argv'],'--recipe') or at(root,config.get('output'))!=directory or not owner):
        raise ValueError('Target does not have successful completed SFT evidence')
    objects=directory/'.checkpoint_objects'
    if objects.is_symlink() or not objects.is_dir():raise ValueError('Target object store is not local')
    # Never acquire this manager's lock: the paused controller may hold its
    # compaction lock. Atomic pointers and immutable ownership metadata suffice.
    for role in ('best','last'):
        pointer=directory/role
        if not pointer.is_symlink():raise ValueError('Completed target lacks an owned checkpoint pointer')
        target=pointer.resolve();marker=target/'.research_checkpoint.json';state=target/'state.pt'
        if (target.parent!=objects.resolve() or marker.is_symlink() or not marker.is_file() or
            state.is_symlink() or not state.is_file()):raise ValueError('Target checkpoint ownership/path mismatch')
        meta=read_json(marker)
        if meta.get('owner')!=owner or role=='last' and meta.get('step')!=done['steps']:
            raise ValueError('Completed target checkpoint step/owner mismatch')
    return {'completed_steps':done['steps'],'completed_sha256':digest(done_path.read_bytes())}


def phase(root,spec,state,job,controller,expected_child=None):
    canonical=digest(json.dumps(spec,sort_keys=True,separators=(',',':')).encode())
    if state.get('spec_sha256')!=canonical:raise ValueError('Native queue state/specification mismatch')
    records=state.get('jobs')
    if not isinstance(records,dict) or any(not isinstance(v,dict) for v in records.values()):
        raise ValueError('Malformed native job state')
    row=records.get(job['id'])
    if row is None:return 'waiting',None
    if row.get('status')!='running' or state.get('status')!='running':return 'missed',None
    active=[name for name,value in records.items() if value.get('status') in
            ('running','evaluating','checkpoint_validating','grounding_evaluating')]
    if active!=[job['id']]:return 'missed',None
    pid=row.get('pid')
    if not isinstance(pid,int) or isinstance(pid,bool):raise ValueError('Target training PID is malformed')
    command=row.get('command')
    if not isinstance(command,list) or any(not isinstance(v,str) for v in command):
        raise ValueError('Target trainer command is missing')
    declared=options({'argv':command,'cwd':str(root)},root,'scripts/run_research_stage.py')
    if (declared is None or declared.get('--stage')!='SFT' or declared.get('--recipe')!=argv_value(job['argv'],'--recipe') or
            at(root,declared.get('--output'))!=at(root,argv_value(job['argv'],'--output')) or
            at(root,declared.get('--manifest'))!=at(root,argv_value(job['argv'],'--manifest'))):
        raise ValueError('Persisted trainer command differs from the explicit target')
    child=process_info(pid)
    if expected_child is not None and (pid!=expected_child['training_pid'] or command_sha(command)!=expected_child['training_command_sha256']):
        return 'missed',None
    if child is not None:
        args=options(child,root,'scripts/run_research_stage.py')
        if (child['ppid']!=controller['pid'] or child['argv']!=command or args is None or
            args.get('--stage')!='SFT' or args.get('--recipe')!=argv_value(job['argv'],'--recipe') or
            at(root,args.get('--output'))!=at(root,argv_value(job['argv'],'--output'))):
            return 'missed',None
        if expected_child is not None and child['start_time']!=expected_child.get('training_start_time'):
            return 'missed',None
    start=child['start_time'] if child is not None else (expected_child or {}).get('training_start_time')
    evidence={'training_pid':pid,'training_start_time':start,'training_command_sha256':command_sha(command)}
    completion=completed_current(root,job)
    return ('ready',dict(evidence,**completion)) if completion is not None else ('waiting',evidence)


def source_context(root,spec,target_job):
    source=next((job for job in spec['jobs'] if job['id']==SOURCE_JOB),None)
    if source is None or compacted_run(root,source) is None:raise ValueError('4q probe source is not completed and compacted')
    if source['evaluation']['manifest']!=target_job['evaluation']['manifest']:
        raise ValueError('Probe source/target evaluation manifests differ')
    checkpoint=at(root,argv_value(source['argv'],'--output'))/'best/state.pt'
    identity=checkpoint_identity(checkpoint)
    manifest=at(root,source['evaluation']['manifest']);raw=manifest.read_bytes()
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    ids=[row['record_id'] for row in rows[:32]]
    if len(ids)!=32 or len(set(ids))!=32:raise ValueError('Probe requires32 unique fixed record IDs')
    weighted=read_json(root/'configs/research/weighted_sft_ready.json')
    if weighted.get('status')!='passed':raise ValueError('Weighted training proof is not passed')
    verify_sources(root,weighted.get('source_sha256'),('scripts/run_research_stage.py',))
    return {'source_checkpoint':str(checkpoint.resolve()),'checkpoint_identity':identity,'manifest':str(manifest),
        'manifest_sha256':digest(raw),'record_ids':ids,
        'source_sha256':{name:digest((root/name).read_bytes()) for name in PROBE_SOURCES},
        'weighted_readiness_sha256':digest((root/'configs/research/weighted_sft_ready.json').read_bytes())}


def validate_report(path,context):
    report=read_json(path)
    if report.get('status')!='passed':raise ValueError('Existing probe report did not pass; no automatic duplicate probe')
    if (report.get('batch_sizes')!=[1,4,8] or report.get('max_new_tokens')!=64 or
        report.get('manifest_sha256')!=context['manifest_sha256'] or report.get('record_ids')!=context['record_ids'] or
        report.get('source_sha256')!=context['source_sha256'] or
        not same_checkpoint_identity(report.get('checkpoint_identity',{}),context['checkpoint_identity']) or
        report.get('observed_gpu_process_checks_exclusive') is not True):raise ValueError('Probe report binding differs')
    profiles=report.get('profiles')
    if not isinstance(profiles,list) or len(profiles)!=3:raise ValueError('Probe report lacks all three batch profiles')
    baseline=None
    for size,row in zip((1,4,8),profiles):
        tokens=row.get('generated_token_ids')
        seconds=row.get('seconds')
        if (row.get('batch_size')!=size or row.get('records')!=32 or row.get('exact_token_parity_with_native_b1') is not True or
            not isinstance(seconds,(int,float)) or isinstance(seconds,bool) or not math.isfinite(seconds) or seconds<=0 or
            not isinstance(tokens,list) or len(tokens)!=32 or
            any(not isinstance(ids,list) or len(ids)>64 or any(not isinstance(i,int) or isinstance(i,bool) or i<0 for i in ids) for ids in tokens)):
            raise ValueError('Probe profiles are incomplete or parity failed')
        if baseline is None:baseline=tokens
        if tokens!=baseline:raise ValueError('Probe token IDs differ across batch sizes')
        if not math.isclose(row.get('records_per_second',float('nan')),32/seconds,rel_tol=1e-6):
            raise ValueError('Probe timing rates do not match measured records/time')
    return report


def resume_owned_parent(record):
    if record.get('owner')!=OWNER or record.get('parent_pause_outstanding') is not True:return False
    if record.get('queue_spec_path')!=SPEC or any(not isinstance(record.get(key),str) or len(record[key])!=64
            for key in ('queue_spec_sha256','canonical_spec_sha256')):return False
    info=process_info(record.get('controller_pid'))
    args=options(info,ROOT,OLD_SCRIPT)
    if (info is None or info['start_time']!=record.get('controller_start_time') or
        command_sha(info['argv'])!=record.get('controller_command_sha256') or args is None or
        set(args)!={'--spec'} or at(ROOT,args['--spec'])!=(ROOT/SPEC).resolve()):return False
    if info['state'] in ('T','t'):os.kill(info['pid'],signal.SIGCONT)
    return True


def arm_parent_death(parent_pid):
    # Only our new benchmark child executes this before exec; never the trainer.
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG failed')
    if os.getppid()!=parent_pid:os._exit(128+signal.SIGTERM)


def stop_probe(process):
    if process is None or process.poll() is not None:return
    process.terminate()
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)


def recover_orphan_probe(record):
    info=process_info(record.get('benchmark_pid'))
    if info is None:return
    expected=record.get('benchmark_command',[])
    if (info['start_time']!=record.get('benchmark_start_time') or info['argv']!=expected or
        command_sha(expected)!=record.get('benchmark_command_sha256') or
        '-m' not in expected or PROBE_MODULE not in expected):return
    os.kill(info['pid'],signal.SIGTERM)
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        current=process_info(info['pid'])
        if current is None or current['start_time']!=info['start_time']:return
        time.sleep(.1)
    current=process_info(info['pid'])
    if current is not None and current['start_time']==info['start_time'] and current['argv']==expected:
        os.kill(info['pid'],signal.SIGKILL)


def wait_training_and_gpu(evidence,timeout=90):
    deadline=time.monotonic()+timeout;last=0.
    while True:
        child=process_info(evidence['training_pid'])
        if child is not None and evidence.get('training_start_time') is not None and child['start_time']!=evidence['training_start_time']:
            raise RuntimeError('Training PID was reused; no probe launched')
        pids=gpu_pids()
        others=[pid for pid in pids if pid not in (evidence['training_pid'],os.getpid())]
        if others:raise RuntimeError('Other GPU process present; no shared benchmark launched')
        if child is None and not pids:return
        if time.monotonic()>=deadline:raise RuntimeError('Training child/GPU cleanup exceeded90s; no probe launched')
        if time.monotonic()-last>=30:emit('waiting_training_cleanup',training_pid=evidence['training_pid'],gpu_pids=pids);last=time.monotonic()
        time.sleep(1)


def benchmark_command(root,context,report):
    return [sys.executable,'-u','-m',PROBE_MODULE,'--checkpoint',context['source_checkpoint'],
        '--manifest',context['manifest'],'--output',str(report),'--records','32','--batch-sizes','1','4','8',
        '--max-new-tokens','64','--minimum-free-gib','12','--reserve-gib','120','--allow-gpu']


def execute_probe(root,context,report,receipt,record,timeout=600):
    command=benchmark_command(root,context,report);log=root/LOG;log.parent.mkdir(parents=True,exist_ok=True)
    record.update(status='probe_starting',attempted=True,benchmark_command=command,benchmark_command_sha256=command_sha(command))
    record['parent_death_guard']='Linux PR_SET_PDEATHSIG SIGTERM on owned benchmark only'
    persist(receipt,record);process=None
    try:
        with log.open('a') as handle:
            handle.write(json.dumps({'time':time.time(),'command':command})+'\n');handle.flush()
            parent_pid=os.getpid()
            process=subprocess.Popen(command,cwd=root,stdout=handle,stderr=subprocess.STDOUT,
                start_new_session=True,preexec_fn=lambda:arm_parent_death(parent_pid))
            info=process_info(process.pid)
            record.update(status='probing',benchmark_pid=process.pid,
                          benchmark_start_time=None if info is None else info['start_time'])
            persist(receipt,record);started=time.monotonic();heartbeat=started
            while process.poll() is None:
                now=time.monotonic()
                if now-started>=timeout:raise TimeoutError('Owned benchmark exceeded600s cap')
                if now-heartbeat>=30:
                    DiskBudget(root,120).ensure_space(0);emit('batch_probe_running',pid=process.pid,elapsed_seconds=now-started,log=str(log));heartbeat=now
                time.sleep(1)
            if process.returncode!=0:raise RuntimeError('Owned batch probe failed; inspect '+str(log))
        return validate_report(report,context)
    finally:stop_probe(process)


def run(root,target_job=TARGET_JOB,controller_pid=CONTROLLER_PID,receipt=None,report=None,timeout=600):
    receipt=root/RECEIPT if receipt is None else receipt;report=root/REPORT if report is None else report
    previous=read_json(receipt) if receipt.is_file() else {}
    # Restore only our recorded parent, before checking even a failed report.
    if previous.get('owner')==OWNER and previous.get('parent_pause_outstanding') is True:
        try:recover_orphan_probe(previous)
        finally:
            resume_owned_parent(previous)
            previous['parent_pause_outstanding']=False;previous['recovered_after_crash']=True
            persist(receipt,previous)
    spec,snapshot=spec_snapshot(root)
    job=next((job for job in spec['jobs'] if job['id']==target_job),None)
    if job is None or argv_value(job['argv'],'--stage')!='SFT':raise ValueError('Explicit target must be an SFT job in the same specification')
    context=source_context(root,spec,job)
    if report.exists():
        validate_report(report,context)
        result=dict(previous,owner=OWNER,status='completed',report_reused=True,parent_pause_outstanding=False,report_sha256=digest(report.read_bytes()))
        persist(receipt,result);return result
    if previous.get('attempted'):
        result=dict(previous,status='failed',error='A prior attempt has no verified report; no duplicate probe launched')
        persist(receipt,result);return result
    controller=native_controller(root,controller_pid)
    if controller is None:return {'status':'skipped','reason':'named_native_controller_not_running'}
    record=dict(owner=OWNER,status='waiting',target_job=target_job,controller_pid=controller_pid,
        controller_start_time=controller['start_time'],controller_command_sha256=command_sha(controller['argv']),
        controller_command=controller['argv'],
        queue_spec_path=SPEC,**snapshot,**context,parent_pause_outstanding=False,attempted=False)
    persist(receipt,record);child=None;last=0.
    while True:
        fresh_spec,fresh_snapshot=spec_snapshot(root)
        if fresh_snapshot!=snapshot:raise ValueError('Immutable specification changed before probe')
        current=native_controller(root,controller_pid)
        if current is None or current['start_time']!=record['controller_start_time']:
            record.update(status='skipped',reason='native_controller_changed');persist(receipt,record);return record
        state=read_json((root/SPEC).with_suffix('.state.json'))
        status,evidence=phase(root,fresh_spec,state,job,current,child)
        if evidence is not None:child=evidence
        if status=='missed':
            record.update(status='skipped',reason='target_window_missed_no_process_stopped');persist(receipt,record);return record
        if status=='ready':break
        if time.monotonic()-last>=30:emit('waiting_completed_target_boundary',target=target_job);last=time.monotonic()
        time.sleep(1)
    if host_available_bytes()<32*1024**3:
        record.update(status='skipped',reason='less_than32GiB_host_headroom');persist(receipt,record);return record
    DiskBudget(root,120).ensure_space(128*1024**2)
    current=process_info(controller_pid)
    if (current is None or current['start_time']!=record['controller_start_time'] or
            command_sha(current['argv'])!=record['controller_command_sha256']):
        record.update(status='skipped',reason='controller_changed_before_pause');persist(receipt,record);return record
    record.update(status='pause_intent',parent_pause_outstanding=True,**child);persist(receipt,record)
    try:
        os.kill(controller_pid,signal.SIGSTOP)
        deadline=time.monotonic()+5
        while True:
            current=process_info(controller_pid)
            if current is None or current['start_time']!=record['controller_start_time']:raise RuntimeError('Controller changed during pause')
            if current['state'] in ('T','t'):break
            if time.monotonic()>=deadline:raise RuntimeError('Controller did not enter pause barrier')
            time.sleep(.05)
        fresh_spec,fresh_snapshot=spec_snapshot(root)
        if fresh_snapshot!=snapshot:raise ValueError('Immutable specification changed at pause barrier')
        status,fresh=phase(root,fresh_spec,read_json((root/SPEC).with_suffix('.state.json')),job,current,child)
        if status!='ready' or fresh!=child:
            record.update(status='skipped',reason='phase_advanced_at_barrier');return record
        current_context=source_context(root,fresh_spec,job)
        if current_context!=context:raise ValueError('Immutable probe source changed at pause barrier')
        record.update(status='paused');persist(receipt,record)
        wait_training_and_gpu(child)
        result=execute_probe(root,context,report,receipt,record,timeout)
        record.update(status='completed',report_sha256=digest(report.read_bytes()),
            measured_profiles=[{k:r[k] for k in ('batch_size','seconds','records_per_second','exact_token_parity_with_native_b1')} for r in result['profiles']])
        return record
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        # Unpause first: receipt I/O failure must never leave training parked.
        resume_owned_parent(record)
        record['parent_pause_outstanding']=False;record['resumed_time']=time.time()
        persist(receipt,record)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',default=TARGET_JOB)
    p.add_argument('--controller-pid',type=int,default=CONTROLLER_PID);p.add_argument('--timeout',type=int,default=600)
    args=p.parse_args(argv)
    if not 0<args.timeout<=600:raise ValueError('Probe deadline must be1..600s')
    lock=ROOT/(RECEIPT+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    def interrupted(signum,_frame):raise InterruptedError('Boundary probe interrupted by signal '+str(signum))
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=run(ROOT,args.job,args.controller_pid,timeout=args.timeout)
    emit('batch_boundary_probe_finished',**result)
    return 0 if result['status'] in ('completed','skipped') else 1


if __name__=='__main__':raise SystemExit(main())

