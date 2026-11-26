#!/usr/bin/env python3
"""Explicit1b,1a or future1c/shuffle boundary, full512 fixed4q parallel B1.

Only ROOT launches this coordinator. No trainer/current evaluator is stopped.
Only the exact controller receives STOP/CONT after completed512; an independent
lease watchdog resumes it on coordinator death/deadline. No automatic retarget.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT))
from scripts import eval_research_batch4_at_boundary as boundary
from scripts import probe_parallel_b1_at_boundary as old
from scripts.handoff_research_grounding import SPEC,at,digest,options,process_info,read_json,spec_snapshot,verify_sources
from scripts.probe_research_batch_at_boundary import persist,command_sha,arm_parent_death,stop_probe,host_available_bytes
from src.utils.research_storage import DiskBudget
from src.utils.research_checkpoint_compat import same_checkpoint_identity

OWNER = 'full512_parallel_b1_current1b_text_boundary_v1'
SCRIPT = 'scripts/eval_research_parallel_b1_at_boundary.py'
TEST = 'tests/test_research_parallel_full_boundary.py'
READY = 'configs/research/parallel_b1_full_boundary_ready.json'
FULL_READY = 'configs/research/parallel_b1_full_evaluator_ready.json'
FULL_SCRIPT = 'scripts/eval_research_checkpoint_parallel_b1.py'
CONTROLLER = 44472
TARGET_PID = 56812
TARGET = old.TARGET_JOB
TARGET1A = 'smallthinker4__siglip2__1a__SFT_pilot_scienceqa_context_v2_balanced_s42'
TARGET1C = 'smallthinker4__siglip2__1c__SFT_pilot_scienceqa_context_v2_balanced_s42'
TARGET_RECIPES = {TARGET:'1b',TARGET1A:'1a',TARGET1C:'1c'}
TARGET_VARIANTS = {TARGET:('text_only',),TARGET1A:('shuffle','blank','text_only'),TARGET1C:('shuffle',)}
FUTURE_TARGETS = (TARGET1A,TARGET1C)
SOURCE = old.SOURCE_JOB
VARIANT = 'text_only'
MAX_PAUSE = 1000
CHECKS = ('exact_current_child_no_retarget','source_target_contexts_separately_frozen',
          'completed512_before_controller_stop','pid_birth_argv_and_pause_intent_bound',
          'natural_child_exit_idle_gpu_only','watchdog_crash_deadline_owned_cont',
          'finally_cont_on_all_failures','upstream_full512_ready_not_old32',
          'raw512_cpu_result_revalidation','no_implicit_worker_retry',
          'fresh_exec_gateway_no_fork_callback','parent_death_and_pid_persist_gap_guarded',
          'monotonic_watchdog_deadline','initial_and_live_worker_identity_bound',
          'explicit_future1a_shuffle_after_gpu_replay_only','bounded_exec_startup_empty_proc_only',
          'real_cpu_gateway_exec_and_parent_death_tested','explicit1a_blank_separate_receipt',
          'explicit1a_text_only_separate_receipt','explicit_future1c_shuffle_after_gpu_replay_only',
          'target_recipe_explicit_mapping_no_auto_retarget','exact_upstream_CLI3_ready_sha_bound')


def emit(event,**fields):print(json.dumps(dict(event=event,time=time.time(),**fields)),flush=True)
def receipt_path(root,workers,target=TARGET,variant=VARIANT):
    if variant not in TARGET_VARIANTS.get(target,()):raise ValueError('Only explicit1b/text_only,1a/shuffle/blank/text_only or1c/shuffle targets')
    recipe=TARGET_RECIPES[target]
    return root/f'research_results/parallel_b1_full_boundary_{recipe}_{variant}_w{workers}.json'
def boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def same(info,record,prefix='controller'):
    return (info is not None and info['pid']==record.get(prefix+'_pid') and
            info['start_time']==record.get(prefix+'_start_time') and
            command_sha(info['argv'])==record.get(prefix+'_command_sha256'))


def readiness(root):
    result={};full_proof=None;coordinator_proof=None
    for name,required,checks in ((READY,(SCRIPT,TEST,'scripts/eval_research_batch4_at_boundary.py',
            'scripts/probe_parallel_b1_at_boundary.py','scripts/research_queue.py'),CHECKS),
            (FULL_READY,(FULL_SCRIPT,'src/eval/research_parallel_protocol.py',
                'tests/test_research_parallel_evaluator.py','tests/test_research_parallel_protocol.py'),None)):
        path=root/name
        if not path.is_file():return None
        raw=path.read_bytes();proof=json.loads(raw)
        if (proof.get('status')!='passed' or type(proof.get('tests_passed')) is not int or proof['tests_passed']<1 or
            not proof.get('test_command') or not isinstance(proof.get('checks'),dict) or not proof['checks'] or
            any(v is not True for v in proof['checks'].values()) or
            (checks and any(proof['checks'].get(c) is not True for c in checks))):
            raise ValueError('Full512 coordinator/evaluator readiness has not passed')
        verify_sources(root,proof.get('source_sha256'),required)
        result[name]={'sha256':digest(raw),'sources':proof['source_sha256']}
        if name==FULL_READY:full_proof=proof
        else:coordinator_proof=proof
    if coordinator_proof.get('upstream_full_readiness_sha256')!=result[FULL_READY]['sha256']:
        raise ValueError('Pinned upstream full512 READY differs from the tested coordinator binding')
    from scripts import eval_research_checkpoint_parallel_b1 as cli
    if (not callable(getattr(cli,'validate_completed_output',None)) or full_proof.get('owner')!=cli.OWNER or
        full_proof.get('worker_counts')!=[2,4] or full_proof.get('full_manifest_records')!=512 or
        full_proof.get('max_new_tokens')!=64 or full_proof.get('actual_gpu_validated') is not False or
        full_proof['checks'].get('full512_raw_token_score_provenance_revalidation') is not True):
        raise ValueError('Released full512 raw verifier missing')
    return result


def contexts(root,spec,state,source_job,target_job,workers):
    if state['jobs'].get(SOURCE,{}).get('status')!='completed':raise ValueError('Fixed4q source not fully completed')
    result={}
    target_recipe=TARGET_RECIPES[target_job['id']]
    for name,job,recipe in (('source4q',source_job,'4q'),('target',target_job,target_recipe)):
        value=boundary.context(root,spec,job,state['jobs'][job['id']])
        value['run_metadata']=old.run_metadata(root,value,recipe);value.pop('output',None);result[name]=value
    if result['source4q']['run']==result['target']['run']:raise ValueError('Source and target must remain separate')
    source=result['source4q'];run=Path(source['run'])
    output=run/'evaluation/parallel_b1_v1'
    if workers==4:output=output/'workers4'
    result['output']=str(output/'none')
    return result


def exact_jobs(root,spec,target_id):
    if target_id not in TARGET_VARIANTS:raise ValueError('Explicit target not supported; never auto-retarget')
    by_id={j['id']:j for j in spec['jobs']};source=by_id.get(SOURCE);target=by_id.get(target_id)
    if source is None or target is None:raise ValueError('Source/target absent from immutable specification')
    for job,recipe in ((source,'4q'),(target,TARGET_RECIPES[target_id])):
        run=at(root,old.argv_value(job['argv'],'--output'))
        if (old.argv_value(job['argv'],'--stage')!='SFT' or old.argv_value(job['argv'],'--recipe')!=recipe or
            not job.get('evaluation') or run.name!=job['id']):
            raise ValueError('Explicit source/target recipe or SFT scope differs')
    if at(root,old.argv_value(source['argv'],'--output'))!=(root/'runs/research'/SOURCE).resolve():
        raise ValueError('Fixed4q source run namespace differs')
    if at(root,source['evaluation']['manifest'])!=at(root,target['evaluation']['manifest']):raise ValueError('Original512 manifest must match')
    return source,target


def phase(root,spec,state,job,controller,expected=None,variant=VARIANT,required_pid=TARGET_PID):
    if variant not in TARGET_VARIANTS.get(job['id'],()):raise ValueError('Target/variant is not explicitly authorized; never auto-retarget')
    if state.get('spec_sha256')!=digest(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()):
        raise ValueError('State/spec SHA mismatch')
    if expected is not None and (expected.get('safe_job')!=job['id'] or expected.get('variant')!=variant):return 'missed',None
    if job['id'] in FUTURE_TARGETS and expected is None:
        ids=[j['id'] for j in spec['jobs']];rows=state.get('jobs',{})
        if not isinstance(rows,dict) or any(not isinstance(row,dict) for row in rows.values()):
            raise ValueError('Malformed wrapper state')
        active=[name for name,row in rows.items() if row.get('status') in ('running','evaluating','checkpoint_validating','grounding_evaluating')]
        if state.get('status')=='running' and len(active)==1 and active[0] in ids and ids.index(active[0])<ids.index(job['id']):
            return 'waiting',None
        row=rows.get(job['id'],{})
        # Between prior-job completion and the next spawn, the durable queue
        # can briefly have no active row. An entirely unstarted future target
        # still waits, provided no failed prior job or started later job exists.
        if (state.get('status')=='running' and not active and row.get('status') in (None,'pending') and
            not row.get('training_completed') and not row.get('grounding_evaluations') and
            all(rows.get(name,{}).get('status') in (None,'pending','completed') for name in ids[:ids.index(job['id'])]) and
            all(rows.get(name,{}).get('status') in (None,'pending') for name in ids[ids.index(job['id'])+1:])):
            return 'waiting',None
        if (variant in TARGET_VARIANTS[job['id']] and state.get('status')=='running' and active==[job['id']] and
            row.get('status')=='grounding_evaluating' and row.get('grounding_status')=='running'):
            if row.get('training_completed') is not True:raise ValueError('Training is not completed')
            current=row.get('active_grounding_variant')
            evaluations=row.get('grounding_evaluations',{})
            if not isinstance(evaluations,dict):raise ValueError('Malformed grounding evaluations')
            # The wrapper persists its grounding-running phase before assigning
            # the first variant. Only this unbound, never-started prefix waits.
            if current is None and not evaluations:return 'waiting',None
            evaluation=evaluations.get(variant,{})
            if not isinstance(evaluation,dict):raise ValueError('Malformed target receipt')
            # It also persists an exact command receipt before Popen/pid. A
            # present-but-invalid pid is corruption, never a pre-spawn state.
            if current==variant and evaluation.get('status')=='running' and 'pid' not in evaluation:
                command=evaluation.get('command')
                declared=options({'argv':command,'cwd':str(root)},root,'scripts/eval_research_checkpoint.py') if isinstance(command,list) else None
                output=at(root,old.argv_value(job['argv'],'--output'))/'evaluation/grounding_v1'/variant
                if (declared is None or declared.get('--image-ablation')!=variant or declared.get('--ablation-seed')!='42' or
                    declared.get('--checkpoint')!='best' or declared.get('--max-new-tokens')!='64' or
                    at(root,declared.get('--run'))!=at(root,old.argv_value(job['argv'],'--output')) or
                    at(root,declared.get('--manifest'))!=at(root,job['evaluation']['manifest']) or
                    at(root,declared.get('--output'))!=output or at(root,evaluation.get('output'))!=output):
                    raise ValueError('Pre-spawn '+variant+' command/output differs from exact target')
                return 'waiting',None
        status,evidence=old.phase(root,spec,state,job,controller,variant,expected)
        if status=='waiting' and evidence is None:return status,evidence
    else:status,evidence=boundary.phase(root,spec,state,job,controller,variant,expected)
    if evidence is not None and required_pid is not None and evidence['evaluation_pid']!=required_pid:return 'missed',None
    if expected is None and (evidence is None or evidence.get('evaluation_start_time') is None):return 'missed',None
    return status,evidence


def resume_parent(root,record):
    if record.get('owner')!=OWNER or record.get('parent_pause_outstanding') is not True:return 'no_owned_pause'
    if record.get('boot_id')!=boot_id():return 'boot_changed'
    info=process_info(record.get('controller_pid'));args=options(info,root,old.CONTROLLER_SCRIPT)
    if not same(info,record) or args is None or '--spec' not in args or at(root,args['--spec'])!=(root/SPEC).resolve():return 'identity_changed'
    if info['state'] in ('T','t'):
        os.kill(info['pid'],signal.SIGCONT);return 'resumed'
    return 'already_running'


def worker_command(root,bound,workers,timeout=900):
    source=bound['source4q']
    return [sys.executable,'-u',str(root/FULL_SCRIPT),'--run',source['run'],
        '--manifest',source['manifest'],'--image-ablation','none','--workers',str(workers),
        '--native-cache',source['native_cache'],'--allow-gpu','--timeout-seconds',str(timeout)]


def recover_worker(root,record):
    if record.get('owner')!=OWNER or record.get('boot_id')!=boot_id():return
    if ((not record.get('worker_pid') or record.get('worker_start_time') is None) and record.get('lease_nonce') and record.get('worker_command') and
        record.get('workers') in (2,4) and record.get('target_job') in TARGET_VARIANTS):
        ready_path=Path(str(receipt_path(root,record['workers'],record['target_job'],record['variant']))+'.'+record['lease_nonce']+'.worker.json')
        if ready_path.is_file():
            ready=read_json(ready_path)
            if (ready.get('owner')==OWNER and ready.get('nonce')==record['lease_nonce'] and
                (not record.get('worker_pid') or ready.get('pid')==record['worker_pid']) and
                ready.get('destination_command_sha256')==record.get('worker_command_sha256')):
                record.update(worker_pid=ready['pid'],worker_start_time=ready['start_time'])
    info=process_info(record.get('worker_pid'));command=record.get('worker_command',[])
    if not same(info,record,'worker') or info['argv']!=command or str(root/FULL_SCRIPT) not in command:return
    os.kill(info['pid'],signal.SIGTERM);deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        if not same(process_info(info['pid']),record,'worker'):return
        time.sleep(.1)
    if same(process_info(info['pid']),record,'worker'):os.kill(info['pid'],signal.SIGKILL)


def recover(root,path,record):
    if record.get('owner')!=OWNER or not record.get('parent_pause_outstanding'):return record
    try:recover_worker(root,record)
    finally:
        record['resume_status']=resume_parent(root,record)
        record.update(parent_pause_outstanding=False,recovered_after_crash=True)
        persist(path,record)
    return record


def watchdog(root,path,nonce):
    record=read_json(path)
    if record.get('owner')!=OWNER or record.get('lease_nonce')!=nonce:return
    own=process_info(os.getpid())
    if own is None:raise RuntimeError('Watchdog birth identity missing')
    persist(Path(str(path)+'.'+nonce+'.watchdog.json'),{'owner':OWNER,'nonce':nonce,'pid':own['pid'],
        'start_time':own['start_time'],'command_sha256':command_sha(own['argv'])})
    while True:
        record=read_json(path)
        if record.get('owner')!=OWNER or record.get('lease_nonce')!=nonce or not record.get('parent_pause_outstanding'):return
        coordinator=process_info(record.get('coordinator_pid'))
        expired=time.monotonic()>=record.get('pause_deadline_monotonic',0)-15
        if expired or not same(coordinator,record,'coordinator'):
            recover(root,path,record);return
        time.sleep(.5)


def start_watchdog(root,path,record):
    command=[sys.executable,'-u',str(root/SCRIPT),'--watchdog','--receipt',str(path),'--nonce',record['lease_nonce']]
    process=subprocess.Popen(command,cwd=root,start_new_session=True)
    ready_path=Path(str(path)+'.'+record['lease_nonce']+'.watchdog.json');deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('Independent pause watchdog exited before readiness')
        if ready_path.is_file():
            proof=read_json(ready_path);info=process_info(process.pid)
            if (proof.get('owner')==OWNER and proof.get('nonce')==record['lease_nonce'] and
                info is not None and proof.get('pid')==info['pid']==process.pid and
                proof.get('start_time')==info['start_time'] and info['argv']==command and
                proof.get('command_sha256')==command_sha(command)):
                return process
            raise ValueError('Pause watchdog readiness identity differs')
        time.sleep(.1)
    raise TimeoutError('Pause watchdog did not become ready; controller was not stopped')


def validate_result(bound,workers):
    from scripts import eval_research_checkpoint_parallel_b1 as cli
    source=bound['source4q']
    result=cli.validate_completed_output(source['run'],source['manifest'],'none',workers,source['native_cache'])
    producer=read_json(Path(bound['output'])/'execution_summary.json')
    if (result.get('status')!='passed' or result.get('records')!=512 or result.get('worker_count')!=workers or
        result.get('full512_exact_previous_native_comparison') is not True or
        result.get('per_model_native64_guard_verified') is not True or
        producer.get('owner')!=cli.OWNER or producer.get('status')!='completed' or
        producer.get('records')!=512 or producer.get('worker_count')!=workers or
        producer.get('full512_previous_native_cache_comparison') is not True or
        producer.get('per_model_parallel_native64_guard_verified') is not True or
        result.get('protocol_sha256')!=producer.get('protocol_sha256') or
        result.get('summary_sha256')!=producer.get('summary_sha256') or
        not same_checkpoint_identity(result.get('checkpoint_identity',{}),source['checkpoint_identity'])):
        raise ValueError('Raw full512 parallel/native token+score verification did not pass')
    return result


def exec_worker(root,path,nonce):
    """Fresh interpreter only: arm parent-death before replacing self with CLI.

    No Python preexec callback runs in a fork of the Torch-importing coordinator.
    Death before arming fails parent/PPID validation; death after arming kills
    this same PID, including after exec and before parent persists its birth.
    """
    record=read_json(path)
    if (record.get('owner')!=OWNER or record.get('lease_nonce')!=nonce or
        record.get('boot_id')!=boot_id() or record.get('parent_pause_outstanding') is not True):
        raise ValueError('Owned fresh worker lease differs')
    parent=record['coordinator_pid'];live=process_info(parent)
    if not same(live,record,'coordinator') or os.getppid()!=parent:
        raise RuntimeError('Coordinator died/changed before parent-death arm')
    command=worker_command(root,record['contexts'],record['workers'],record['worker_timeout_seconds'])
    if command!=record.get('worker_command') or command_sha(command)!=record.get('worker_command_sha256'):
        raise ValueError('Fresh worker destination differs from durable intent')
    arm_parent_death(parent)
    if not same(process_info(parent),record,'coordinator') or os.getppid()!=parent:
        raise RuntimeError('Coordinator died/changed after parent-death arm')
    own=process_info(os.getpid())
    if own is None:raise RuntimeError('Fresh worker own birth unavailable')
    persist(Path(str(path)+'.'+nonce+'.worker.json'),{'owner':OWNER,'nonce':nonce,
        'pid':own['pid'],'start_time':own['start_time'],'destination_command_sha256':command_sha(command)})
    os.execv(command[0],command)
    raise RuntimeError('Fresh worker exec unexpectedly returned')


def worker_handshake(path,record):
    ready_path=Path(str(path)+'.'+record['lease_nonce']+'.worker.json')
    if not ready_path.is_file():return False
    ready=read_json(ready_path)
    if (ready.get('owner')!=OWNER or ready.get('nonce')!=record['lease_nonce'] or
        ready.get('pid')!=record.get('worker_pid') or ready.get('start_time')!=record.get('worker_start_time') or
        ready.get('destination_command_sha256')!=record['worker_command_sha256']):
        raise ValueError('Fresh worker ready identity differs')
    return True


def _worker_birth(pid):
    """Read birth/PPID even when cmdline or cwd is briefly unavailable at exec."""
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        if fields[0] in ('Z','X'):return None
        return {'pid':pid,'state':fields[0],'ppid':int(fields[1]),'start_time':int(fields[19]),'argv':None}
    except (OSError,ValueError,IndexError):return None


def observe_worker(process,record,launcher,command,*,startup):
    """A bounded startup may see unavailable/empty cmdline, never foreign argv."""
    live=process_info(process.pid)
    if live is None and process.poll() is None:live=_worker_birth(process.pid)
    argv=None if live is None else live.get('argv')
    kind=('unavailable' if argv is None else 'empty' if argv==[] else
          'launcher' if argv==launcher else 'final' if argv==command else 'foreign')
    observation={'time':time.time(),'pid':process.pid,'ppid':None if live is None else live.get('ppid'),
        'start_time':None if live is None else live.get('start_time'),'argv_state':kind,
        'argv_sha256':None if argv is None else command_sha(argv),'argv_count':None if argv is None else len(argv)}
    observations=record.setdefault('worker_startup_observations',[])
    if startup and len(observations)<64:observations.append(observation)
    record['worker_last_observation']=observation
    if live is not None:
        if (live.get('pid')!=process.pid or live.get('ppid')!=os.getpid() or
            (record.get('worker_start_time') is not None and live.get('start_time')!=record['worker_start_time'])):
            raise RuntimeError('Owned worker spawn identity/live birth/PPID differs: '+json.dumps(observation))
        if record.get('worker_start_time') is None:record['worker_start_time']=live['start_time']
    if kind=='foreign':raise RuntimeError('Owned worker live birth/PPID/argv changed: '+json.dumps(observation))
    if not startup and (live is None or kind in ('empty','unavailable')):
        if process.poll() is not None:return None
        raise RuntimeError('Owned worker live birth/PPID/argv unavailable after startup: '+json.dumps(observation))
    return kind


def wait_worker_startup(process,path,record,launcher,command,deadline):
    """Require nonce registration AND observed final exec before normal polling."""
    startup_deadline=min(time.monotonic()+15,deadline-820)
    record.update(worker_pid=process.pid,worker_start_time=None)
    persist(path,record)  # Keep the owned Popen PID even before its first /proc sample.
    while time.monotonic()<startup_deadline:
        if process.poll() is not None:raise RuntimeError('Owned worker exited before final exec/nonce handshake')
        kind=observe_worker(process,record,launcher,command,startup=True)
        if record.get('worker_start_time') is not None and kind=='final' and worker_handshake(path,record):
            record.update(status='running_full512',worker_startup_verified=True)
            persist(path,record);return
        time.sleep(.025)
    raise TimeoutError('Owned worker final exec/nonce startup handshake did not arrive: '+json.dumps(record.get('worker_last_observation')))


def execute(root,bound,workers,path,record,deadline):
    if deadline-time.monotonic()<850:raise TimeoutError('Insufficient lease for800s worker plus startup/cleanup')
    timeout=800
    command=worker_command(root,bound,workers,timeout)
    launcher=[sys.executable,'-u',str(root/SCRIPT),'--exec-worker','--receipt',str(path),'--nonce',record['lease_nonce']]
    log=root/f'logs/research/parallel_b1_full_boundary_w{workers}.log';log.parent.mkdir(parents=True,exist_ok=True)
    record.update(status='worker_starting',attempted=True,worker_command=command,worker_command_sha256=command_sha(command),
        worker_launch_command=launcher,worker_timeout_seconds=timeout)
    persist(path,record);process=None
    try:
        with log.open('a') as handle:
            handle.write(json.dumps({'time':time.time(),'command':command})+'\n');handle.flush()
            process=subprocess.Popen(launcher,cwd=root,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            wait_worker_startup(process,path,record,launcher,command,deadline)
            last=time.monotonic();handshake=True
            while process.poll() is None:
                kind=observe_worker(process,record,launcher,command,startup=False)
                if kind is None and process.poll() is not None:break
                if kind!='final':raise RuntimeError('Owned worker returned to launcher after final exec')
                if time.monotonic()>=deadline-20:raise TimeoutError('Owned worker exceeded bounded pause lease')
                if time.monotonic()-last>=30:
                    DiskBudget(root,120).ensure_space(0);emit('full512_worker_running',pid=process.pid);last=time.monotonic()
                time.sleep(.5)
            if process.returncode!=0:raise RuntimeError('Parallel full512 CLI failed; inspect '+str(log))
            handshake=worker_handshake(path,record) or handshake
            if not handshake:raise RuntimeError('Full512 worker exited without verified fresh-exec handshake')
        return validate_result(bound,workers)
    finally:stop_probe(process)


def run(root,workers=2,timeout=MAX_PAUSE,target_id=TARGET,variant=VARIANT,controller_pid=CONTROLLER):
    if workers not in (2,4) or not 0<timeout<=MAX_PAUSE:raise ValueError('Only workers2/4 and pause<=1000s')
    path=receipt_path(root,workers,target_id,variant);previous=read_json(path) if path.is_file() else {}
    if previous and previous.get('owner')!=OWNER:raise ValueError('Foreign receipt preserved')
    if previous.get('owner')==OWNER and previous.get('parent_pause_outstanding'):recover(root,path,previous)
    if previous.get('attempted'):
        if previous.get('contexts') and (Path(previous['contexts']['output'])/'execution_summary.json').is_file():
            if readiness(root)!=previous.get('proof'):raise ValueError('Recovered tested source proof changed')
            result=validate_result(previous['contexts'],workers)
            previous.update(status='completed',result=result,reused_verified=True);persist(path,previous);return previous
        raise RuntimeError('Prior one-shot worker attempt exists; no automatic retry/retarget')
    spec,snapshot=spec_snapshot(root);source_job,target_job=exact_jobs(root,spec,target_id)
    controller=boundary.wrapper(root,controller_pid)
    if controller is None or controller.get('state') in ('T','t'):
        result={'owner':OWNER,'status':'skipped','reason':'exact_controller_not_running_unpaused'};persist(path,result);return result
    own=process_info(os.getpid())
    record=dict(owner=OWNER,status='waiting',boot_id=boot_id(),lease_nonce=uuid.uuid4().hex,
        controller_pid=controller_pid,controller_start_time=controller['start_time'],controller_command_sha256=command_sha(controller['argv']),
        coordinator_pid=os.getpid(),coordinator_start_time=own['start_time'],coordinator_command_sha256=command_sha(own['argv']),
        source_job=SOURCE,target_job=target_id,target_pid=TARGET_PID if target_id==TARGET else None,variant=variant,workers=workers,
        parent_pause_outstanding=False,attempted=False,**snapshot)
    expected=None;bound=None;proof=None;last=0.
    while True:
        fresh_spec,fresh_snapshot=spec_snapshot(root)
        if fresh_snapshot!=snapshot:raise ValueError('Frozen specification changed')
        live=boundary.wrapper(root,controller_pid)
        if not same(live,record):record.update(status='skipped',reason='controller_changed');persist(path,record);return record
        state=read_json((root/SPEC).with_suffix('.state.json'))
        status,evidence=phase(root,fresh_spec,state,target_job,live,expected,variant,record['target_pid'])
        if status=='missed':
            reason='explicit_1c_shuffle_window_missed' if target_id==TARGET1C else 'explicit_current_text_only_window_missed'
            record.update(status='skipped',reason=reason);persist(path,record);return record
        if evidence is not None:
            expected=evidence;record['target_pid']=evidence['evaluation_pid']
        current=readiness(root)
        if current is not None:
            if proof is not None and current!=proof:raise ValueError('Tested full512 readiness changed')
            proof=current
            if bound is None and evidence is not None:
                bound=contexts(root,fresh_spec,state,source_job,target_job,workers)
                if Path(bound['output']).exists():raise ValueError('Fresh full512 namespace required; no implicit retry')
                record.update(contexts=bound,proof=proof)
        if status=='ready' and proof is not None and bound is not None:break
        if time.monotonic()-last>=30:
            persist(path,record);emit('waiting_explicit_grounding512',job=target_id,variant=variant,pid=record['target_pid'],ready=proof is not None);last=time.monotonic()
        time.sleep(.25)
    if host_available_bytes()<32*2**30:raise RuntimeError('Insufficient host/cgroup headroom')
    DiskBudget(root,120).ensure_space(2*2**30)
    live=process_info(controller_pid)
    if not same(live,record) or live['state'] in ('T','t'):record.update(status='skipped',reason='controller_changed_before_pause');persist(path,record);return record
    start=time.monotonic();deadline=start+timeout;guardian=None
    record.update(status='pause_intent',parent_pause_outstanding=True,pause_started_epoch=time.time(),
        pause_deadline_epoch=time.time()+timeout,pause_deadline_monotonic=deadline,**expected)
    persist(path,record)
    try:
        # Separate session survives a coordinator SIGKILL and releases only this
        # receipt-bound controller. It never starts a GPU process.
        guardian=start_watchdog(root,path,record)
        os.kill(controller_pid,signal.SIGSTOP)
        while True:
            live=process_info(controller_pid)
            if not same(live,record):raise RuntimeError('Controller identity changed at pause')
            if live['state'] in ('T','t'):break
            if time.monotonic()-start>5:raise TimeoutError('Controller pause barrier exceeded5s')
            time.sleep(.05)
        fresh_spec,fresh_snapshot=spec_snapshot(root)
        if fresh_snapshot!=snapshot or readiness(root)!=proof:raise ValueError('Source/readiness changed at pause')
        state=read_json((root/SPEC).with_suffix('.state.json'))
        status,fresh=phase(root,fresh_spec,state,target_job,live,expected,variant,record['target_pid'])
        if status!='ready' or fresh!=expected:record.update(status='skipped',reason='phase_advanced_at_pause');return record
        if contexts(root,fresh_spec,state,source_job,target_job,workers)!=bound:raise ValueError('Frozen source/target identity changed')
        boundary.verify_completed_variant(root,target_job,bound['target'],fresh)
        if deadline-time.monotonic()<880:raise TimeoutError('Insufficient lease for idle/startup/800s worker/cleanup')
        old.wait_idle(fresh,min(deadline-40,time.monotonic()+30))
        record.update(status='paused_idle');persist(path,record)
        result=execute(root,bound,workers,path,record,deadline)
        if spec_snapshot(root)[1]!=snapshot or readiness(root)!=proof or contexts(root,fresh_spec,state,source_job,target_job,workers)!=bound:
            raise ValueError('Frozen context/spec/readiness changed during measurement')
        if boundary.files_identity(Path(fresh['variant_output']),fresh['completed_variant_files'])!=fresh['completed_variant_files']:
            raise ValueError('Completed target cache bytes changed')
        if time.monotonic()>=deadline:raise TimeoutError('Total pause lease expired before verified completion')
        record.update(status='completed',result=result);return record
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        record['resume_status']=resume_parent(root,record)
        record.update(parent_pause_outstanding=False,paused_seconds=time.monotonic()-start)
        persist(path,record)
        if guardian is not None:
            try:guardian.wait(timeout=2)
            except subprocess.TimeoutExpired:pass  # Durable leasefalse makes it exit; never signal a foreign PID.


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,choices=[2,4],default=2);p.add_argument('--timeout',type=int,default=MAX_PAUSE)
    p.add_argument('--controller-pid',type=int,default=CONTROLLER)
    p.add_argument('--target-job',choices=list(TARGET_VARIANTS),default=TARGET)
    p.add_argument('--target-variant',choices=['text_only','shuffle','blank'],default=VARIANT)
    p.add_argument('--watchdog',action='store_true');p.add_argument('--receipt');p.add_argument('--nonce')
    p.add_argument('--exec-worker',action='store_true')
    args=p.parse_args(argv)
    if args.watchdog or args.exec_worker:
        if not args.receipt or not args.nonce:raise ValueError('Owned watchdog receipt+nonce required')
        path=Path(args.receipt).resolve()
        if path not in [receipt_path(ROOT,w,t,v).resolve() for w in (2,4) for t,variants in TARGET_VARIANTS.items() for v in variants]:raise ValueError('Foreign watchdog receipt path')
        if args.exec_worker:exec_worker(ROOT,path,args.nonce)
        else:watchdog(ROOT,path,args.nonce)
        return 0
    def interrupted(sig,_):raise InterruptedError('Boundary interrupted '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    path=receipt_path(ROOT,args.workers,args.target_job,args.target_variant);path.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(path)+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=run(ROOT,args.workers,args.timeout,args.target_job,args.target_variant,args.controller_pid)
    emit('full512_parallel_boundary_finished',**result)
    return 0 if result['status'] in ('completed','skipped') else 1


if __name__=='__main__':raise SystemExit(main())
