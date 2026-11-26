#!/usr/bin/env python3
"""One isolated B4-none/full512 comparison after an owned grounding child ends.

Only the exact wrapper controller is paused. No trainer/evaluator is interrupted.
Its current child exits naturally before the guarded B4 worker starts. Durable
pause intent, PID birth/argv binding, parent-death guard and finally-SIGCONT
protect the running campaign. Production state/configs and B1 caches are read-only.
"""
from pathlib import Path
import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.handoff_research_grounding import SPEC,at,compacted_run,digest,options,process_info,read_json,service_status,spec_snapshot,verify_sources
from scripts.probe_research_batch_at_boundary import gpu_pids,host_available_bytes,persist,arm_parent_death,stop_probe,command_sha
from scripts.research_queue import argv_value
from src.utils.research_checkpoint_compat import checkpoint_identity,same_checkpoint_identity
from src.utils.research_storage import DiskBudget

OWNER='batch4_grounding_boundary_v1'
CONTROLLER_PID=44472
CONTROLLER_SCRIPT='scripts/research_grounding_queue.py'
JOB='smallthinker4__siglip2__4q__SFT_pilot_scienceqa_context_v2_balanced_s42'
READY='configs/research/batch4_evaluator_ready_proposal.json'
RECEIPT='research_results/batch4_none_boundary.json'
LOG='logs/research/batch4_none_boundary_worker.log'
SCRIPT='scripts/eval_research_checkpoint_batch4.py'
CHECKS=('fixed_batch4_original_order_atomic_shard_resume','per_checkpoint_all8_benchmark_32_all4_mode_native_token_guard',
 'raw_score_and_four_group_provenance_recomputed','completed_cache_corruption_refused',
 'new_namespace_old_outputs_untouched','no_automatic_batch_fallback','native_full512_reference_comparison_supported',
 'native_tiny_cpu_eos_padding_test','frozen_15_sources_and_batch_helper_unchanged')


def emit(event,**fields):print(json.dumps(dict(event=event,time=time.time(),**fields)),flush=True)
def files_identity(directory,names):
    return {name:digest((directory/name).read_bytes()) for name in names}


def readiness(root):
    path=root/READY
    if not path.is_file():return None
    raw=path.read_bytes();proof=json.loads(raw)
    if (not isinstance(proof,dict) or proof.get('status')!='passed' or
        not isinstance(proof.get('tests_passed'),int) or isinstance(proof['tests_passed'],bool) or proof['tests_passed']<1 or
        not proof.get('test_command') or not isinstance(proof.get('checks'),dict) or
        any(proof['checks'].get(name) is not True for name in CHECKS)):
        raise ValueError('B4 evaluator readiness checks have not passed')
    verify_sources(root,proof.get('source_sha256'),(SCRIPT,'src/eval/research_batch_inference.py','scripts/eval_research_checkpoint.py'))
    from scripts import eval_research_checkpoint_batch4 as cli
    if proof.get('profile')!=cli.PROFILE or proof.get('profile_sha256')!=cli.digest(cli.PROFILE):
        raise ValueError('Actual B4 profile differs from tested readiness')
    # deployment_authorized:false describes this proposal's deployment status;
    # root separately authorizes this isolated measurement, not a queue migration.
    return {'readiness_sha256':digest(raw),'profile_sha256':proof['profile_sha256'],
            'tested_sources':proof['source_sha256']}


def wrapper(root,pid):
    status=service_status('research-pilot')
    if 'RUNNING' not in status or 'pid '+str(pid)+',' not in status:return None
    info=process_info(pid);args=options(info,root,CONTROLLER_SCRIPT)
    if (args is None or '--spec' not in args or set(args)-{'--spec','--retry-failed'} or
        args.get('--retry-failed',True) is not True or at(root,args['--spec'])!=(root/SPEC).resolve()):return None
    return info


def context(root,spec,job,row):
    from scripts import eval_research_checkpoint_batch4 as cli
    if compacted_run(root,job) is None:raise ValueError('Source SFT is not completed/compacted')
    run=at(root,argv_value(job['argv'],'--output'));manifest=at(root,job['evaluation']['manifest'])
    raw=manifest.read_bytes();records=cli.read_rows(manifest);cli.validate_manifest(records)
    artifacts={}
    for role in ('best','last'):
        path=(run/role/'state.pt').resolve()
        artifacts.setdefault(str(path),checkpoint_identity(path))
    selected=artifacts[str((run/'best/state.pt').resolve())]
    validation=row.get('checkpoint_validation',{})
    if (validation.get('status')!='completed' or validation.get('mode')!='full_gpu_replay' or
        validation.get('selection_metric_replayed') is not True):raise ValueError('Both retained checkpoints have not passed the full GPU replay gate')
    stored=validation.get('checkpoint_artifacts',{})
    if set(stored)!=set(artifacts) or any(not same_checkpoint_identity(stored[name],value) for name,value in artifacts.items()):
        raise ValueError('Retained checkpoint identities differ from validated ones')
    roundtrip=run/'checkpoint_roundtrip.json'
    if digest(roundtrip.read_bytes())!=validation.get('report_sha256') or read_json(roundtrip).get('status')!='passed':
        raise ValueError('Full checkpoint replay receipt changed')
    native=at(root,job['evaluation']['output'])
    original=cli.read_rows(native/'predictions.jsonl')
    protocol_stub={'checkpoint_sha256':selected['sha256'],'manifest_sha256':digest(raw)}
    checked=cli.compare_native_cache(native,records,protocol_stub,original)
    if checked['status']!='passed':raise ValueError('Native none cache is invalid')
    return {'run':str(run),'manifest':str(manifest),'manifest_sha256':digest(raw),
        'checkpoint_identity':selected,'retained_checkpoint_artifacts':artifacts,
        'roundtrip_sha256':digest(roundtrip.read_bytes()),'native_cache':str(native),
        'native_cache_files':files_identity(native,('summary.json','protocol.json','predictions.jsonl')),
        'output':str(run/'evaluation/batch4_v1/none')}


def phase(root,spec,state,job,controller,variant,expected=None):
    canonical=digest(json.dumps(spec,sort_keys=True,separators=(',',':')).encode())
    if state.get('spec_sha256')!=canonical:raise ValueError('State/spec SHA mismatch')
    rows=state.get('jobs')
    if not isinstance(rows,dict) or any(not isinstance(r,dict) for r in rows.values()):raise ValueError('Malformed wrapper state')
    row=rows.get(job['id'],{})
    active=[name for name,value in rows.items() if value.get('status') in ('running','evaluating','checkpoint_validating','grounding_evaluating')]
    if (state.get('status')!='running' or active!=[job['id']] or row.get('status')!='grounding_evaluating' or
        row.get('grounding_status')!='running' or row.get('active_grounding_variant')!=variant):return 'missed',None
    if row.get('training_completed') is not True:raise ValueError('Training is not completed')
    evaluation=row.get('grounding_evaluations',{}).get(variant,{})
    if evaluation.get('status') not in ('running','completed'):return 'missed',None
    pid=evaluation.get('pid');command=evaluation.get('command')
    if not isinstance(pid,int) or isinstance(pid,bool) or not isinstance(command,list):raise ValueError('Grounding child identity missing')
    declared=options({'argv':command,'cwd':str(root)},root,'scripts/eval_research_checkpoint.py')
    output=at(root,argv_value(job['argv'],'--output'))/'evaluation/grounding_v1'/variant
    if (declared is None or declared.get('--image-ablation')!=variant or declared.get('--ablation-seed')!='42' or
        declared.get('--checkpoint')!='best' or declared.get('--max-new-tokens')!='64' or
        at(root,declared.get('--run'))!=at(root,argv_value(job['argv'],'--output')) or
        at(root,declared.get('--manifest'))!=at(root,job['evaluation']['manifest']) or
        at(root,declared.get('--output'))!=output or at(root,evaluation.get('output'))!=output):
        raise ValueError('Grounding child command/output differs from exact target')
    child=process_info(pid)
    if child is not None and (child['ppid']!=controller['pid'] or child['argv']!=command):return 'missed',None
    start=child['start_time'] if child is not None else (expected or {}).get('evaluation_start_time')
    evidence={'safe_job':job['id'],'variant':variant,'evaluation_pid':pid,'evaluation_start_time':start,
        'evaluation_command':command,'evaluation_command_sha256':command_sha(command),'variant_output':str(output)}
    if expected and any(evidence[k]!=expected[k] for k in ('evaluation_pid','evaluation_start_time','evaluation_command_sha256')):
        return 'missed',None
    if not (output/'summary.json').is_file() or not (output/'progress.json').is_file():return 'waiting',evidence
    summary=read_json(output/'summary.json');progress=read_json(output/'progress.json')
    if (summary.get('complete_manifest') is not True or summary.get('records')!=512 or summary.get('manifest_records')!=512 or
        progress.get('event')!='completed' or progress.get('completed_records',progress.get('records'))!=512):
        return 'waiting',evidence
    evidence['completed_variant_files']=files_identity(output,('summary.json','protocol.json','predictions.jsonl','transformed_manifest.jsonl','progress.json'))
    return 'ready',evidence


def verify_completed_variant(root,job,source,evidence):
    from scripts.research_grounding_queue import GroundingResearchQueue
    from src.eval.research_ablation import transform_image_ablation
    from scripts import eval_research_checkpoint_batch4 as cli
    records=cli.read_rows(source['manifest'])
    transformed,descriptor=transform_image_ablation(records,variant=evidence['variant'],seed=42,
        output_dir=Path(evidence['variant_output']),source_manifest_sha256=source['manifest_sha256'],reserve_bytes=120*2**30)
    identity={'records':records,'transformed_records':transformed,'image_ablation':descriptor,
        'output':Path(evidence['variant_output']),'manifest_sha256':source['manifest_sha256'],
        'checkpoint_sha256':source['checkpoint_identity']['sha256'],'checkpoint_artifact_identity':source['checkpoint_identity']}
    # No constructor: it would contend for the live controller's state lock.
    verifier=GroundingResearchQueue.__new__(GroundingResearchQueue)
    verifier._validate_grounding(job,identity,evidence['variant'])


def validate_result(source):
    from scripts import eval_research_checkpoint_batch4 as cli
    from src.eval.research_ablation import transform_image_ablation
    output=Path(source['output']);records=cli.read_rows(source['manifest'])
    transformed={};descriptors={}
    for mode in cli.MODES:
        transformed[mode],descriptors[mode]=transform_image_ablation(records,variant=mode,seed=42,
            output_dir=output.parent/mode,source_manifest_sha256=source['manifest_sha256'],reserve_bytes=120*2**30)
    binding=cli.guard_binding(source['checkpoint_identity'],source['manifest_sha256'],records,descriptors,
        cli.rgb_census(records),cli.source_identity(),cli.package_identity())
    guard=cli.validate_guard(read_json(output.parent/'parity_guard.json'),binding)
    protocol=read_json(output/'protocol.json')
    expected=cli.make_protocol(source['checkpoint_identity'],source['manifest_sha256'],transformed['none'],descriptors['none'],
        binding['rgb_census'],guard,cli.source_identity(),cli.package_identity())
    if protocol!=expected:raise ValueError('Completed B4 protocol binding differs')
    rows,shards=cli.read_committed_shards(output,transformed['none'],protocol)
    if len(rows)!=512 or len(shards)!=128 or cli.read_rows(output/'predictions.jsonl')!=rows:
        raise ValueError('B4 journal/raw view does not cover512 exact fixed-batch rows')
    summary=read_json(output/'summary.json');metrics=cli.aggregate_scores(rows,require_four_groups=True)
    if (summary.get('complete_manifest') is not True or summary.get('manifest_records')!=512 or
        summary.get('protocol_sha256')!=cli.protocol_digest(protocol) or summary.get('profile')!=cli.PROFILE or
        summary.get('raw_predictions_sha256')!=cli.sha(output/'predictions.jsonl') or
        summary.get('journal_shards')!={str(k):v for k,v in shards.items()} or any(summary.get(k)!=v for k,v in metrics.items())):
        raise ValueError('B4 raw scores/summary/shards differ')
    comparison=cli.compare_native_cache(source['native_cache'],records,protocol,rows)
    if comparison.get('status')!='passed' or comparison.get('exact_generated_token_ids') is not True or comparison.get('compared_records')!=512:
        raise ValueError('Full512 B1/B4 comparison did not pass')
    if read_json(output/'native_b1_comparison.json')!=comparison:raise ValueError('Stored full512 comparison differs')
    if files_identity(Path(source['native_cache']),source['native_cache_files'])!=source['native_cache_files']:
        raise ValueError('Frozen native none cache changed')
    return {'status':'passed','compared_records':512,'exact_generated_token_ids':True,
        'guard_sha256':digest((output.parent/'parity_guard.json').read_bytes()),
        'summary_sha256':digest((output/'summary.json').read_bytes()),'comparison_sha256':digest((output/'native_b1_comparison.json').read_bytes()),
        'all11_deployment_authorized':False}


def resume_parent(record):
    if record.get('owner')!=OWNER or record.get('parent_pause_outstanding') is not True:return
    info=process_info(record.get('controller_pid'));args=options(info,ROOT,CONTROLLER_SCRIPT)
    if (info is not None and info['start_time']==record.get('controller_start_time') and
        command_sha(info['argv'])==record.get('controller_command_sha256') and args is not None and
        '--spec' in args and at(ROOT,args['--spec'])==(ROOT/SPEC).resolve() and info['state'] in ('T','t')):
        os.kill(info['pid'],signal.SIGCONT)


def recover_worker(record):
    info=process_info(record.get('worker_pid'));command=record.get('worker_command',[])
    if (info is None or info['start_time']!=record.get('worker_start_time') or info['argv']!=command or
        command_sha(command)!=record.get('worker_command_sha256') or str(ROOT/SCRIPT) not in command):return
    os.kill(info['pid'],signal.SIGTERM)
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        new=process_info(info['pid'])
        if new is None or new['start_time']!=info['start_time']:return
        time.sleep(.1)
    new=process_info(info['pid'])
    if new is not None and new['start_time']==info['start_time'] and new['argv']==command:os.kill(info['pid'],signal.SIGKILL)


def wait_idle(evidence):
    deadline=time.monotonic()+90;last=0.
    while True:
        child=process_info(evidence['evaluation_pid'])
        if child is not None and (child['start_time']!=evidence['evaluation_start_time'] or child['argv']!=evidence['evaluation_command']):
            raise RuntimeError('Completed evaluator PID identity changed')
        pids=gpu_pids()
        if any(pid not in (evidence['evaluation_pid'],os.getpid()) for pid in pids):raise RuntimeError('Other GPU process present; no shared worker launched')
        if child is None and not pids:return
        if time.monotonic()>=deadline:raise TimeoutError('Completed evaluator/GPU cleanup exceeded90s')
        if time.monotonic()-last>=30:emit('waiting_natural_evaluator_exit',pid=evidence['evaluation_pid']);last=time.monotonic()
        time.sleep(1)


def worker_command(source):
    return [sys.executable,'-u',str(ROOT/SCRIPT),'--run',source['run'],'--manifest',source['manifest'],
        '--checkpoint','best','--image-ablation','none','--max-new-tokens','64','--ablation-seed','42',
        '--reserve-gb','120','--compare-native-cache',source['native_cache']]


def execute(source,receipt,record,timeout):
    command=worker_command(source)
    if (Path(source['output'])/'protocol.json').is_file():command.append('--resume')
    log=ROOT/LOG;log.parent.mkdir(parents=True,exist_ok=True)
    record.update(status='worker_starting',attempted=True,worker_command=command,worker_command_sha256=command_sha(command))
    persist(receipt,record);process=None
    try:
        with log.open('a') as handle:
            handle.write(json.dumps({'time':time.time(),'command':command})+'\n');handle.flush()
            parent=os.getpid()
            process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,
                preexec_fn=lambda:arm_parent_death(parent))
            info=process_info(process.pid)
            record.update(status='running_b4',worker_pid=process.pid,worker_start_time=None if info is None else info['start_time'])
            persist(receipt,record);start=time.monotonic();last=start
            while process.poll() is None:
                now=time.monotonic()
                if now-start>=timeout:raise TimeoutError('Owned B4 worker exceeded1200s cap')
                if now-last>=30:
                    DiskBudget(ROOT,120).ensure_space(0);emit('batch4_worker_running',pid=process.pid,elapsed_seconds=now-start,log=str(log));last=now
                time.sleep(1)
            if process.returncode!=0:raise RuntimeError('B4 evaluator/parity failed; inspect '+str(log))
        return validate_result(source)
    finally:stop_probe(process)


def run(root,controller_pid=CONTROLLER_PID,job_id=JOB,variant='shuffle',timeout=1200):
    receipt=root/RECEIPT;previous=read_json(receipt) if receipt.is_file() else {}
    if previous.get('owner')==OWNER and previous.get('parent_pause_outstanding') is True:
        try:recover_worker(previous)
        finally:
            resume_parent(previous);previous.update(parent_pause_outstanding=False,recovered_after_crash=True);persist(receipt,previous)
    spec,snapshot=spec_snapshot(root);job=next((j for j in spec['jobs'] if j['id']==job_id),None)
    if job is None:raise ValueError('Explicit target job not in immutable specification')
    controller=wrapper(root,controller_pid)
    if controller is None:return {'status':'skipped','reason':'named_wrapper_not_running'}
    anchor={'controller_pid':controller_pid,'controller_start_time':controller['start_time'],'controller_command_sha256':command_sha(controller['argv'])}
    expected=None;record=dict(owner=OWNER,status='waiting',**anchor,**snapshot,job_id=job_id,variant=variant,
        parent_pause_outstanding=False,attempted=False);source=None;proof=None;last=0.
    while True:
        fresh_spec,current_spec=spec_snapshot(root)
        if current_spec!=snapshot:raise ValueError('Immutable specification changed')
        live=wrapper(root,controller_pid)
        if live is None or live['start_time']!=anchor['controller_start_time']:return dict(record,status='skipped',reason='wrapper_changed')
        state=read_json((root/SPEC).with_suffix('.state.json'))
        status,evidence=phase(root,fresh_spec,state,job,live,variant,expected)
        if status=='missed':record.update(status='skipped',reason='target_boundary_missed');persist(receipt,record);return record
        if evidence is not None:expected=evidence
        current_proof=readiness(root)
        if current_proof is not None:
            if proof is not None and proof!=current_proof:raise ValueError('Tested B4 readiness changed during wait')
            proof=current_proof
            if source is None:
                source=context(root,fresh_spec,job,state['jobs'][job_id]);record.update(source=source,proof=proof);persist(receipt,record)
                if (Path(source['output'])/'summary.json').is_file():
                    result=validate_result(source);record.update(status='completed',result=result,reused=True);persist(receipt,record);return record
                if previous.get('attempted'):raise RuntimeError('Prior one-shot attempt has no verified result; no duplicate launch')
        if status=='ready' and proof is not None and source is not None:break
        if time.monotonic()-last>=30:emit('waiting_completed_grounding_boundary',job=job_id,variant=variant,ready_proof=proof is not None);last=time.monotonic()
        time.sleep(.25)
    if host_available_bytes()<32*2**30:raise RuntimeError('Insufficient cgroup/host headroom for isolated worker')
    DiskBudget(root,120).ensure_space(128*2**20)
    live=process_info(controller_pid)
    if live is None or live['start_time']!=anchor['controller_start_time'] or command_sha(live['argv'])!=anchor['controller_command_sha256']:
        return dict(record,status='skipped',reason='wrapper_changed_before_pause')
    record.update(status='pause_intent',parent_pause_outstanding=True,**expected);persist(receipt,record)
    try:
        os.kill(controller_pid,signal.SIGSTOP);deadline=time.monotonic()+5
        while True:
            live=process_info(controller_pid)
            if live is None or live['start_time']!=anchor['controller_start_time']:raise RuntimeError('Wrapper identity changed at barrier')
            if live['state'] in ('T','t'):break
            if time.monotonic()>=deadline:raise TimeoutError('Wrapper pause barrier timed out')
            time.sleep(.05)
        fresh_spec,current_spec=spec_snapshot(root)
        if current_spec!=snapshot or readiness(root)!=proof:raise ValueError('Spec/tested readiness changed at barrier')
        state=read_json((root/SPEC).with_suffix('.state.json'))
        status,fresh=phase(root,fresh_spec,state,job,live,variant,expected)
        if status!='ready' or fresh!=expected:record.update(status='skipped',reason='phase_advanced_at_barrier');return record
        if context(root,fresh_spec,job,state['jobs'][job_id])!=source:raise ValueError('Immutable none/checkpoint source changed at barrier')
        verify_completed_variant(root,job,source,fresh)
        if files_identity(Path(fresh['variant_output']),fresh['completed_variant_files'])!=fresh['completed_variant_files']:
            raise ValueError('Completed grounding cache changed during verification')
        wait_idle(fresh);record['status']='paused';persist(receipt,record)
        result=execute(source,receipt,record,timeout)
        if context(root,fresh_spec,job,state['jobs'][job_id])!=source:raise ValueError('Frozen source cache/checkpoint changed during B4 run')
        record.update(status='completed',result=result);return record
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        resume_parent(record);record.update(parent_pause_outstanding=False,resumed_time=time.time());persist(receipt,record)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--controller-pid',type=int,default=CONTROLLER_PID)
    p.add_argument('--job',default=JOB);p.add_argument('--variant',choices=['shuffle','blank','text_only'],default='shuffle')
    p.add_argument('--timeout',type=int,default=1200);args=p.parse_args(argv)
    if not 0<args.timeout<=1200:raise ValueError('Worker timeout must be1..1200s')
    def interrupted(signum,_):raise InterruptedError('Boundary worker interrupted by signal '+str(signum))
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    lock=root_lock=ROOT/(RECEIPT+'.lock');root_lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=run(ROOT,args.controller_pid,args.job,args.variant,args.timeout)
    emit('batch4_boundary_finished',**result)
    return 0 if result['status'] in ('completed','skipped') else 1


if __name__=='__main__':raise SystemExit(main())

