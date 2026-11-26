#!/usr/bin/env python3
"""Migrate only our controller at a completed-SFT ordinary evaluation boundary.

Pause the controller alone, revalidate its exact evaluator child, queue SIGINT,
and always resume it. No trainer or process group is signalled by this tool.
Durable stopping/starting receipts recover a stopped controller without waiting
for another evaluator. The training specification/state are never rewritten.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.research_storage import DiskBudget,_atomic_json,_fsync_directory
from scripts.research_queue import validate_spec,argv_value
from scripts.handoff_research_sft_balance import service_status,wait_for_running

SPEC='configs/research/smallthinker_pilot_scienceqa_context_v2_balanced_queue.json'
READY='configs/research/grounding_queue_ready.json'
RECEIPT='research_results/grounding_queue_handoff.json'
CONFIG=Path('/etc/supervisor/conf.d/research-pilot.conf')
CHECKS=('standard_none_unchanged','raw_transformed_and_checkpoint_binding',
        'interrupted_eval_resume_no_retrain','distinct_variant_outputs','disk_and_signal_guards',
        'checkpoint_full_gpu_replay_required','checkpoint_roundtrip_reuse_binds_both_best_last',
        'checkpoint_validation_failure_resume_no_retrain')
REQUIRED_SOURCES=('scripts/research_grounding_queue.py','scripts/research_queue.py',
                  'scripts/eval_research_checkpoint.py','src/eval/research_ablation.py')
OLD_SCRIPT='scripts/research_queue.py'
NEW_SCRIPT='scripts/research_grounding_queue.py'
OLD_COMMAND='command=/venv/main/bin/python -u '+OLD_SCRIPT+' --spec '+SPEC
NEW_COMMAND='command=/venv/main/bin/python -u '+NEW_SCRIPT+' --spec '+SPEC


def emit(event,**fields):print(json.dumps(dict(time=time.time(),event=event,**fields)),flush=True)
def digest(raw):return hashlib.sha256(raw).hexdigest()


def read_json(path):
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict):raise ValueError('Expected JSON object: '+str(path))
    return value


def at(root,value):
    if not isinstance(value,str) or not value:raise ValueError('Missing explicit owned path')
    p=Path(value)
    return (root/p).resolve() if not p.is_absolute() else p.resolve()


def spec_snapshot(root):
    raw=(root/SPEC).read_bytes();spec=validate_spec(json.loads(raw))
    if len(spec['jobs'])!=24:raise ValueError('Migration requires the unchanged24-job specification')
    return spec,{'queue_spec_sha256':digest(raw),
        'canonical_spec_sha256':digest(json.dumps(spec,sort_keys=True,separators=(',',':')).encode())}


def verify_sources(root,sources,required):
    if not isinstance(sources,dict) or any(name not in sources for name in required):
        raise ValueError('Readiness lacks required tested source binding')
    for name,expected in sources.items():
        source=at(root,name)
        if not source.is_relative_to(root.resolve()) or not source.is_file() or digest(source.read_bytes())!=expected:
            raise ValueError('Validated source changed after tests: '+str(name))


def ready_proof(root):
    path=root/READY
    if not path.is_file():return None
    proof_raw=path.read_bytes();proof=json.loads(proof_raw)
    if not isinstance(proof,dict):raise ValueError('Malformed grounding readiness proof')
    if (proof.get('status')!='passed' or not isinstance(proof.get('test_command'),str) or not proof['test_command'] or
        not isinstance(proof.get('tests_passed'),int) or isinstance(proof['tests_passed'],bool) or proof['tests_passed']<1 or
        not isinstance(proof.get('checks'),dict) or any(proof['checks'].get(name) is not True for name in CHECKS)):
        raise ValueError('Grounding queue lacks its required regression checks')
    verify_sources(root,proof.get('source_sha256'),REQUIRED_SOURCES)
    weighted_path=root/'configs/research/weighted_sft_ready.json'
    weighted_raw=weighted_path.read_bytes();weighted=json.loads(weighted_raw)
    if not isinstance(weighted,dict):raise ValueError('Malformed weighted readiness proof')
    if weighted.get('status')!='passed':raise ValueError('Weighted training readiness is not passed')
    verify_sources(root,weighted.get('source_sha256'),('scripts/run_research_stage.py',))
    return {'readiness_sha256':digest(proof_raw),'weighted_readiness_sha256':digest(weighted_raw)}


def process_info(pid):
    if not isinstance(pid,int) or isinstance(pid,bool) or pid<1:return None
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        if fields[0] in ('Z','X'):return None
        return {'pid':pid,'state':fields[0],'ppid':int(fields[1]),'start_time':int(fields[19]),
            'argv':[os.fsdecode(x) for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x],
            'cwd':str(Path(f'/proc/{pid}/cwd').resolve(strict=True))}
    except (OSError,ValueError,IndexError):return None


def options(info,root,script):
    if info is None:return None
    argv=info.get('argv',[])
    if not argv or not Path(argv[0]).name.startswith('python'):return None
    i=1
    while i<len(argv) and argv[i] in ('-u','-B','-E','-s'):i+=1
    if i>=len(argv):return None
    p=Path(argv[i]);cwd=Path(info.get('cwd','/'))
    if ((cwd/p).resolve() if not p.is_absolute() else p.resolve())!=(root/script).resolve():return None
    result={};i+=1
    while i<len(argv):
        name=argv[i]
        if not name.startswith('--') or name in result:return None
        if i+1<len(argv) and not argv[i+1].startswith('--'):result[name]=argv[i+1];i+=2
        else:result[name]=True;i+=1
    return result


def controller_info(root,status,script):
    if 'RUNNING' not in status:return None
    match=re.search(r'\bpid\s+(\d+)\b',status)
    info=process_info(int(match[1])) if match else None
    args=options(info,root,script)
    if args is None or set(args)!={'--spec'} or at(root,args['--spec'])!=(root/SPEC).resolve():return None
    return info


def compacted_run(root,job):
    if argv_value(job['argv'],'--stage')!='SFT':return None
    directory=at(root,argv_value(job['argv'],'--output'))
    files=('completed.json','checkpoint_compaction.json','resolved_config.json','.checkpoint_manager.json')
    if any(not (directory/name).is_file() or (directory/name).is_symlink() for name in files):return None
    done=read_json(directory/'completed.json');compact=read_json(directory/'checkpoint_compaction.json')
    cfg=read_json(directory/'resolved_config.json');owner=read_json(directory/'.checkpoint_manager.json').get('owner')
    if (cfg.get('stage')!='SFT' or at(root,cfg.get('output'))!=directory or compact.get('status')!='completed' or
        compact.get('training_completed') is not True or compact.get('resumable_optimizer') is not False or
        not isinstance(done.get('steps'),int) or isinstance(done['steps'],bool) or done['steps']<1 or
        done.get('status') in ('failed','interrupted','incomplete') or not owner):return None
    objects=directory/'.checkpoint_objects'
    if objects.is_symlink() or not objects.is_dir():return None
    retained={}
    for name in ('best','last'):
        pointer=directory/name
        if not pointer.is_symlink():return None
        target=pointer.resolve();marker_path=target/'.research_checkpoint.json';state=target/'state.pt'
        if target.parent!=objects.resolve() or marker_path.is_symlink() or not marker_path.is_file() or state.is_symlink() or not state.is_file():return None
        marker=read_json(marker_path)
        if marker.get('owner')!=owner or marker.get('resumable_optimizer') is not False or marker.get('training_completed') is not True:return None
        if name=='last' and marker.get('step')!=done['steps']:return None
        retained[name]={'object':target.name,'step':marker.get('step'),'owner':owner}
    return {'completed_steps':done['steps'],'retained_checkpoints':retained}


def safe_evaluation_boundary(root,spec,state,controller):
    expected=digest(json.dumps(spec,sort_keys=True,separators=(',',':')).encode())
    if state.get('spec_sha256')!=expected:raise ValueError('Controller state/specification mismatch')
    records=state.get('jobs')
    if not isinstance(records,dict) or any(not isinstance(v,dict) for v in records.values()):
        raise ValueError('Malformed controller job state')
    if controller is None or state.get('status')!='running':return None
    active=[(name,row) for name,row in records.items() if row.get('status') in ('running','evaluating','grounding_evaluating')]
    if len(active)!=1 or active[0][1].get('status')!='evaluating':return None
    name,row=active[0];job=next((j for j in spec['jobs'] if j['id']==name),None)
    if job is None or row.get('training_completed') is not True:return None
    checkpoint=compacted_run(root,job)
    if checkpoint is None:return None
    configured=job.get('evaluation');evaluation=row.get('evaluation')
    if not isinstance(configured,dict) or not isinstance(evaluation,dict) or evaluation.get('status')!='running':return None
    child=process_info(evaluation.get('pid'));args=options(child,root,'scripts/eval_research_checkpoint.py')
    if child is None or child['ppid']!=controller['pid'] or args is None:return None
    output=at(root,configured.get('output'))
    if (at(root,evaluation.get('output'))!=output or at(root,args.get('--output'))!=output or
        at(root,args.get('--run'))!=at(root,argv_value(job['argv'],'--output')) or
        at(root,args.get('--manifest'))!=at(root,configured.get('manifest')) or
        args.get('--image-ablation','none')!='none' or args.get('--checkpoint')!=configured.get('checkpoint','best') or
        args.get('--max-new-tokens')!=str(configured.get('max_new_tokens',64))):return None
    protocol_path=output/'protocol.json'
    if not protocol_path.is_file() or protocol_path.is_symlink():return None
    protocol=read_json(protocol_path);ablation=protocol.get('image_ablation',{})
    if not isinstance(ablation,dict) or ablation.get('variant','none')!='none':return None
    if protocol.get('manifest_sha256')!=digest(at(root,configured['manifest']).read_bytes()):return None
    return {'safe_job':name,'evaluation_pid':child['pid'],'evaluation_start_time':child['start_time'],
        'controller_pid':controller['pid'],'controller_start_time':controller['start_time'],
        'standard_protocol_sha256':digest(protocol_path.read_bytes()),'training_was_completed':True,**checkpoint}


def same_identity(expected,current):
    for name in ('queue_spec_sha256','canonical_spec_sha256','readiness_sha256','weighted_readiness_sha256'):
        if expected.get(name)!=current.get(name):raise ValueError('Grounding handoff identity changed: '+name)


def configuration(original):
    lines=original.splitlines()
    if sum(line.startswith('command=') for line in lines)!=1 or 'autorestart=false' not in lines:
        raise ValueError('Unexpected owned Supervisor configuration')
    if OLD_COMMAND not in lines and NEW_COMMAND not in lines:raise ValueError('Pilot controller command changed unexpectedly')
    return '\n'.join(NEW_COMMAND if line==OLD_COMMAND else line for line in lines)+'\n'


def atomic_configuration(value):
    tmp=CONFIG.with_name('.'+CONFIG.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('x') as h:h.write(value);h.flush();os.fsync(h.fileno())
        os.chmod(tmp,CONFIG.stat().st_mode & 0o777)
        os.replace(tmp,CONFIG);_fsync_directory(CONFIG.parent)
    finally:tmp.unlink(missing_ok=True)


def pause_and_interrupt(root,spec,identity,boundary):
    """Controller-only barrier; same live evaluator or no interruption."""
    pid=boundary['controller_pid'];paused=False
    try:
        current=process_info(pid)
        if current is None or current['start_time']!=boundary['controller_start_time']:return False
        args=options(current,root,OLD_SCRIPT)
        if args is None or set(args)!={'--spec'} or at(root,args['--spec'])!=(root/SPEC).resolve():return False
        os.kill(pid,signal.SIGSTOP);paused=True
        deadline=time.monotonic()+5
        while True:
            current=process_info(pid)
            if current is None or current['start_time']!=boundary['controller_start_time']:return False
            if current['state'] in ('T','t'):break
            if time.monotonic()>=deadline:raise RuntimeError('Controller did not enter pause barrier')
            time.sleep(.05)
        fresh_spec,snapshot=spec_snapshot(root);proof=ready_proof(root)
        if proof is None:raise ValueError('Grounding readiness disappeared at pause barrier')
        same_identity(identity,dict(snapshot,**proof))
        fresh=safe_evaluation_boundary(root,fresh_spec,read_json((root/SPEC).with_suffix('.state.json')),current)
        if fresh!=boundary:return False
        original=CONFIG.read_text();replacement=configuration(original)
        if OLD_COMMAND not in original.splitlines():raise ValueError('Native controller configuration changed before barrier')
        (root/RECEIPT).parent.mkdir(parents=True,exist_ok=True)
        _atomic_json(root/RECEIPT,dict(status='stopping',time=time.time(),**identity,**boundary,
            source_supervisor_sha256=digest(original.encode()),replacement_supervisor_sha256=digest(replacement.encode())))
        os.kill(pid,signal.SIGINT)
        return True
    finally:
        if paused:
            current=process_info(pid)
            if current is not None and current['start_time']==boundary['controller_start_time']:
                try:os.kill(pid,signal.SIGCONT)
                except ProcessLookupError:pass


def wait_gpu_idle(timeout=30):
    deadline=time.monotonic()+timeout
    while True:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if not result.stdout.strip():return
        if any(not r.strip().isdigit() for r in result.stdout.splitlines() if r.strip()):raise RuntimeError('Unrecognized GPU process listing')
        if time.monotonic()>=deadline:raise RuntimeError('CUDA process remains; no replacement controller launched')
        time.sleep(1)


def stopped(status):return any(word in status for word in ('STOPPED','EXITED','FATAL'))


def wait_controller_stopped(boundary,timeout=180):
    deadline=time.monotonic()+timeout
    while True:
        if stopped(service_status('research-pilot')):return
        info=process_info(boundary['controller_pid'])
        if info is not None and info['start_time']!=boundary['controller_start_time']:raise RuntimeError('Controller PID reused during shutdown')
        if time.monotonic()>=deadline:raise RuntimeError('Controller did not finish graceful evaluation interruption')
        time.sleep(1)


def verify_stopped_recovery(root,spec,previous):
    job=next((j for j in spec['jobs'] if j['id']==previous.get('safe_job')),None)
    checkpoint=compacted_run(root,job) if job else None
    if checkpoint is None or any(previous.get(k)!=checkpoint[k] for k in checkpoint):
        raise ValueError('Stopped handoff lost its completed compacted checkpoint binding')
    state=read_json((root/SPEC).with_suffix('.state.json'))
    canonical=digest(json.dumps(spec,sort_keys=True,separators=(',',':')).encode())
    if state.get('spec_sha256')!=canonical or not isinstance(state.get('jobs'),dict):
        raise ValueError('Stopped controller state/specification mismatch')
    row=state['jobs'].get(previous['safe_job'],{})
    if (not isinstance(row,dict) or row.get('training_completed') is not True or
            row.get('status') not in ('evaluating','training_completed_eval_interrupted','training_completed_eval_failed','completed')):
        raise ValueError('Stopped controller no longer records completed training')
    protocol=at(root,job['evaluation']['output'])/'protocol.json'
    if not protocol.is_file() or digest(protocol.read_bytes())!=previous.get('standard_protocol_sha256'):
        raise ValueError('Ordinary evaluation protocol changed during handoff')
    content=CONFIG.read_text();configuration(content)
    if digest(content.encode()) not in {previous.get('source_supervisor_sha256'),previous.get('replacement_supervisor_sha256')}:
        raise ValueError('Supervisor configuration changed during recovery')


def launch_replacement(root,spec,identity,previous):
    verify_stopped_recovery(root,spec,previous);wait_gpu_idle()
    _,snapshot=spec_snapshot(root);proof=ready_proof(root)
    if proof is None:raise ValueError('Grounding readiness disappeared before launch')
    same_identity(identity,dict(snapshot,**proof))
    if not stopped(service_status('research-pilot')):
        raise RuntimeError('Pilot became active before replacement; no running task stopped')
    current=CONFIG.read_text();replacement=configuration(current)
    backup=root/'research_results/grounding_source_supervisor.conf'
    if not backup.exists():backup.write_text(current)
    if current!=replacement:atomic_configuration(replacement)
    starting=dict(previous,status='starting',time=time.time());_atomic_json(root/RECEIPT,starting)
    subprocess.run(['supervisorctl','reread'],check=True)
    subprocess.run(['supervisorctl','update','research-pilot'],check=True)
    if stopped(service_status('research-pilot')):subprocess.run(['supervisorctl','start','research-pilot'],check=True)
    status=wait_for_running('research-pilot')
    if controller_info(root,status,NEW_SCRIPT) is None:raise RuntimeError('Service does not own the grounding wrapper')
    _atomic_json(root/RECEIPT,dict(starting,status='completed',time=time.time(),
        ordinary_evaluation_resumes=True,training_config_unchanged=True))
    emit('grounding_handoff_completed')


def _main():
    root=ROOT;receipt=root/RECEIPT;previous=read_json(receipt) if receipt.exists() else {}
    if previous.get('status')=='completed':return
    # Restore a controller left paused by a killed handoff, even if the new
    # readiness proof subsequently disappeared. No new interruption is issued.
    if previous.get('status')=='stopping':
        info=process_info(previous.get('controller_pid'))
        args=options(info,root,OLD_SCRIPT)
        if (info is not None and info['state'] in ('T','t') and info['start_time']==previous.get('controller_start_time') and
                args is not None and set(args)=={'--spec'} and at(root,args['--spec'])==(root/SPEC).resolve()):
            os.kill(info['pid'],signal.SIGCONT)
    _,baseline=spec_snapshot(root)
    while True:
        DiskBudget(root,120).ensure_space(64*2**20)
        spec,snapshot=spec_snapshot(root)
        if snapshot!=baseline:raise ValueError('Immutable queue specification changed during handoff')
        proof=ready_proof(root);status=service_status('research-pilot')
        identity=dict(snapshot,**proof) if proof is not None else None
        if previous.get('status') in ('stopping','starting') and identity is not None:
            same_identity(previous,identity)
            replacement=controller_info(root,status,NEW_SCRIPT)
            if replacement is not None:
                if previous['status']!='starting':raise ValueError('Unexpected wrapper before starting receipt')
                if NEW_COMMAND not in CONFIG.read_text().splitlines():raise ValueError('Replacement configuration differs')
                _atomic_json(receipt,dict(previous,status='completed',recovered_after_start=True));return
            if stopped(status):launch_replacement(root,spec,identity,previous);return
        native=controller_info(root,status,OLD_SCRIPT);boundary=None
        if proof is not None and native is not None:
            boundary=safe_evaluation_boundary(root,spec,read_json((root/SPEC).with_suffix('.state.json')),native)
            if boundary is None and previous.get('status')=='stopping' and native['state'] in ('T','t'):
                if (native['pid'],native['start_time'])==(previous.get('controller_pid'),previous.get('controller_start_time')):
                    os.kill(native['pid'],signal.SIGCONT)
        if boundary is not None and pause_and_interrupt(root,spec,identity,boundary):
            previous=read_json(receipt);wait_controller_stopped(boundary)
            launch_replacement(root,spec,identity,previous);return
        emit('grounding_handoff_waiting',reason='code_readiness' if proof is None else 'completed_compacted_SFT_standard_evaluation_boundary')
        time.sleep(15)


def main():
    lock_path=(ROOT/RECEIPT).with_suffix('.lock');lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another grounding handoff already owns this controller migration')
        return _main()


if __name__=='__main__':main()

