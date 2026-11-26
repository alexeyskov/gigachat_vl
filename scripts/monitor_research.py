#!/usr/bin/env python3
"""Append GPU/disk/progress snapshots without importing the training stack."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.research_storage import DiskBudget, _atomic_json

PROBE_OWNER = 'batch4_grounding_boundary_v1'
PROBE_RECEIPT = 'research_results/batch4_none_boundary.json'
BATCH4_SCRIPT = 'eval_research_checkpoint_batch4.py'
PARALLEL_SCRIPT = 'eval_research_checkpoint_parallel_b1.py'
PARALLEL_OWNER = 'parallel_native_b1_evaluation_proposal_v1'


def _serial_reference_progress(root,run,output,attempt,barrier,receipt,protocol,parent,options,workers,variant):
    """A live one-model reference64 is diagnostic, never scientific512 progress."""
    context=barrier['context'];nonce=barrier.get('nonce');selected=protocol.get('selected_global_indices')
    if (protocol.get('owner')!='parallel_native_b1_evaluator_v1' or protocol.get('namespace')!='parallel_b1_v1' or
            protocol.get('workers')!=1 or protocol.get('evaluation_records')!=64 or
            protocol.get('assignment')!='named_serial_guard64' or protocol.get('seed')!=42 or
            protocol.get('variant')!=variant or context.get('variant')!=variant or
            protocol.get('manifest_sha256')!=context.get('manifest_sha256') or
            protocol.get('transformed_manifest_sha256')!=context.get('transformed_manifest_sha256') or
            Path(options.get('--manifest','')).resolve()!=Path(context['manifest']).resolve() or
            barrier.get('world_size')!=1 or barrier.get('guard_required') is not False or
            not isinstance(nonce,str) or len(nonce)!=64 or any(c not in '0123456789abcdef' for c in nonce) or
            not isinstance(selected,list) or len(selected)!=64 or len(set(selected))!=64 or
            any(type(i) is not int or not 0<=i<512 for i in selected) or sorted(selected)!=selected):
        return None
    pending=barrier['initial_pending'];committed=barrier.get('initial_committed')
    if (set(pending)!={'0'} or not isinstance(committed,list) or
            any(type(i) is not int for i in pending['0']+committed) or
            len(set(pending['0']+committed))!=64 or sorted(pending['0']+committed)!=selected):return None
    cohort=barrier.get('cohort')
    if (not isinstance(cohort,list) or len(cohort)!=1 or cohort[0].get('logical_worker')!=0 or
            receipt.get('cohort')!=cohort):return None
    identity=cohort[0]['identity'];actual=process_identity(identity['pid'])
    if (actual is None or identity.get('ppid')!=parent['pid'] or actual.get('ppid')!=parent['pid'] or
            any(actual.get(k)!=identity.get(k) for k in ('pid','start_time','argv'))):return None
    argv=identity['argv'];command_hash=hashlib.sha256(json.dumps(argv,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    child_options=_command_options(argv);child_script=command_script(argv)
    if (identity.get('command_sha256')!=command_hash or child_options is None or child_script is None or
            (root/child_script).resolve()!=root/'scripts'/PARALLEL_SCRIPT or
            child_options!={'--private-worker':True,'--barrier':str(attempt/'allocation_barrier.json'),
                '--logical-worker':'0','--parent-pid':str(parent['pid']),
                '--parent-start':str(parent['start_time']),'--nonce':nonce}):return None
    live_parent=process_identity(parent['pid'])
    if live_parent is None or any(live_parent.get(k)!=parent.get(k) for k in ('pid','ppid','start_time','argv')):return None
    progress=attempt/'worker0.progress.json'
    if progress.is_symlink():return None
    item=json.loads(progress.read_text()) if progress.exists() else {'phase':'model_loading','logical_worker':0}
    if item.get('logical_worker')!=0 or item.get('phase') not in ('model_loading','model_ready','native_b1_generation','completed'):
        return None
    result={'run_id':run.name,'stage':'SFT','phase':'parallel_native_b1_serial_reference64',
        'activity_verified':True,'activity_source':'verified_parallel_native_b1_serial_reference',
        'diagnostic':True,'scientific_full512_progress':False,'reference_records':64,
        'source':str(progress),'file_age_seconds':round(max(0,time.time()-progress.stat().st_mtime),1) if progress.exists() else None,
        'worker_pid':identity['pid'],'parent_pid':parent['pid'],'workers':1,'configured_parallel_workers':workers,
        'active_models':1,'variant':variant,'completed_records':len(committed),'total_records':64,
        'event':'serial_reference_'+item['phase']}
    gate=attempt/'generation_barrier.json'
    if item['phase'] in ('native_b1_generation','completed'):
        if (not gate.is_file() or gate.is_symlink() or type(item.get('completed_records')) is not int or
                not 0<=item['completed_records']<=len(pending['0']) or item.get('total_records')!=len(pending['0'])):return None
        generation=json.loads(gate.read_text())
        if (generation.get('owner')!=PARALLEL_OWNER or generation.get('status')!='released' or
                generation.get('allocation_barrier_sha256')!=receipt['allocation_barrier_sha256'] or
                generation.get('nonce')!=nonce):return None
        elapsed=time.monotonic()-generation['released_monotonic']
        if not math.isfinite(elapsed) or elapsed<0:return None
        count=item['completed_records'];done=len(committed)+count
        result.update(completed_records=done,elapsed_seconds=elapsed)
        if count:result['eta_seconds']=elapsed/count*(64-done)
    return result


def parallel_b1_progress(root):
    """Follow a live, birth/command-bound cohort without importing its model."""
    root=Path(root).resolve()
    paths=(root/'runs/research').glob('*/evaluation/parallel_b1_v1/**/execution/*/allocation_barrier.json')
    for path in sorted(paths,key=lambda p:p.stat().st_mtime,reverse=True):
        try:
            if any(p.is_symlink() for p in (path,*path.parents)):continue
            barrier=json.loads(path.read_text());attempt=path.parent
            receipt=json.loads((attempt/'receipt.json').read_text())
            if (barrier.get('owner')!=PARALLEL_OWNER or barrier.get('status')!='released' or
                    receipt.get('owner')!=PARALLEL_OWNER or receipt.get('status') not in ('running','completed') or
                    receipt.get('allocation_barrier_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest()):continue
            parent=barrier['parent'];live=process_identity(parent['pid'])
            if live is None or any(live.get(k)!=parent.get(k) for k in ('pid','start_time','argv')):continue
            script=command_script(live['argv']);options=_command_options(live['argv'])
            if script is None or (root/script).resolve()!=root/'scripts'/PARALLEL_SCRIPT or options is None:continue
            context=barrier['context'];run=Path(context['run']).resolve();output=Path(barrier['output']).resolve()
            workers=int(options.get('--workers',2));variant=options.get('--image-ablation','none')
            expected=run/'evaluation/parallel_b1_v1'
            if workers==4:expected=expected/'workers4'
            serial=expected/'native_guard'/variant
            expected=expected/variant
            if (workers not in (2,4) or variant not in ('none','shuffle','blank','text_only') or
                    run.parent!=root/'runs/research' or Path(options.get('--run','')).resolve()!=run or
                    output not in (expected,serial) or path.parent.parent.parent!=output or
                    Path(barrier['protocol_path'])!=output/'protocol.json'):continue
            if output==serial and any(p.is_symlink() for p in (output/'protocol.json',attempt/'receipt.json')):continue
            protocol=json.loads(Path(barrier['protocol_path']).read_text())
            if (protocol.get('manifest_records')!=512 or
                    protocol.get('checkpoint_sha256')!=context['checkpoint_identity']['sha256']):continue
            bound=hashlib.sha256(json.dumps(protocol,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            if bound!=barrier.get('protocol_sha256') or bound!=receipt.get('protocol_sha256'):continue
            if protocol.get('scope')=='guard64_serial_reference':
                if output!=serial:continue
                value=_serial_reference_progress(root,run,output,attempt,barrier,receipt,protocol,parent,options,workers,variant)
                if value is not None:return value
                continue
            if output!=expected or protocol.get('scope')!='full512_parallel' or protocol.get('workers')!=workers:continue
            pending=barrier['initial_pending'];indices=[i for group in pending.values() for i in group]
            if len(indices)!=len(set(indices)) or any(type(i) is not int or not 0<=i<512 for i in indices):continue
            phases=[]
            for member in barrier['cohort']:
                index=member['logical_worker'];progress=attempt/f'worker{index}.progress.json'
                item=json.loads(progress.read_text()) if progress.exists() else {}
                phases.append((index,item))
            result={'run_id':run.name,'stage':'SFT','phase':'parallel_native_b1_guard',
                'activity_verified':True,'activity_source':'verified_parallel_native_b1_cohort',
                'source':str(attempt/'receipt.json'),'file_age_seconds':round(max(0,time.time()-(attempt/'receipt.json').stat().st_mtime),1),
                'worker_pid':parent['pid'],'workers':workers,'active_models':len(phases),'variant':variant,
                'guard_unique_records':64,'guard_model_requests_total':64*len(phases),
                'guard_models_completed':sum(item.get('phase')=='guard_completed' for _,item in phases)}
            guarded=0
            for _,item in phases:
                count=(64 if item.get('phase')=='guard_completed' else
                       item.get('completed_records',0) if item.get('phase')=='native_guard64_generation' else 0)
                if type(count) is not int or not 0<=count<=64:raise ValueError('Invalid guard count')
                guarded+=count
            result['guard_model_requests_completed']=guarded
            gate=attempt/'generation_barrier.json'
            if gate.exists():
                generation=json.loads(gate.read_text())
                if generation.get('allocation_barrier_sha256')!=receipt['allocation_barrier_sha256'] or generation.get('nonce')!=barrier['nonce']:continue
                done=512-len(indices)
                for index,item in phases:
                    count=item.get('completed_records',0) if item.get('phase') in ('native_b1_generation','completed') else 0
                    if type(count) is not int or not 0<=count<=len(pending[str(index)]):raise ValueError('Invalid cohort count')
                    done+=count
                elapsed=time.monotonic()-generation['released_monotonic']
                if not math.isfinite(elapsed) or elapsed<0:continue
                result.update(phase='parallel_native_b1_generation',completed_records=done,total_records=512,
                    guard_models_completed=len(phases),guard_model_requests_completed=64*len(phases))
                # Resume's existing rows did not consume this execution's time.
                fresh=done-(512-len(indices))
                if fresh>0:
                    result.update(elapsed_seconds=elapsed,eta_seconds=elapsed/fresh*(512-done))
                if done==512:result['phase']='parallel_native_b1_verifying'
            return result
        except (OSError,ValueError,TypeError,KeyError,IndexError):continue
    return None


def process_identity(pid, proc_root='/proc'):
    """Minimal PID-birth/argv verification; no Torch or training imports."""
    if not isinstance(pid,int) or isinstance(pid,bool) or pid<1:return None
    directory=Path(proc_root)/str(pid)
    try:
        stat=directory.joinpath('stat').read_text()
        fields=stat[stat.rfind(')')+2:].split()
        if fields[0] in ('Z','X'):return None
        argv=[os.fsdecode(part) for part in directory.joinpath('cmdline').read_bytes().split(b'\0') if part]
        return {'pid':pid,'ppid':int(fields[1]),'start_time':int(fields[19]),'argv':argv}
    except (OSError,ValueError,IndexError):return None


def command_script(argv):
    if not argv or not Path(argv[0]).name.startswith('python'):return None
    index=1
    while index<len(argv) and argv[index] in ('-u','-B','-E','-s'):index+=1
    return None if index>=len(argv) else argv[index]


def _command_options(argv):
    script=command_script(argv)
    if script is None:return None
    index=argv.index(script)+1;result={}
    while index<len(argv):
        key=argv[index]
        if not key.startswith('--') or key in result:return None
        if index+1<len(argv) and not argv[index+1].startswith('--'):
            result[key]=argv[index+1];index+=2
        else:result[key]=True;index+=1
    return result


def live_batch4_probe(root):
    """Trust only this coordinator's live, exact-birth, exact-command worker."""
    root=Path(root).resolve()
    try:
        record=json.loads((root/PROBE_RECEIPT).read_text())
        if (not isinstance(record,dict) or record.get('owner')!=PROBE_OWNER or
                record.get('status')!='running_b4'):return None
        pid=record.get('worker_pid');birth=record.get('worker_start_time');argv=record.get('worker_command')
        if (not isinstance(birth,int) or isinstance(birth,bool) or birth<1 or
                not isinstance(argv,list) or not argv or any(not isinstance(arg,str) for arg in argv)):return None
        checksum=hashlib.sha256(json.dumps(argv,separators=(',',':')).encode()).hexdigest()
        if record.get('worker_command_sha256')!=checksum:return None
        if command_script(argv)!=str(root/'scripts'/BATCH4_SCRIPT):return None
        process=process_identity(pid)
        if process is None or process['start_time']!=birth or process['argv']!=argv:return None
        source=record.get('source',{})
        if not isinstance(source,dict):return None
        run=Path(source['run']).resolve();output=Path(source['output']).resolve()
        expected=run/'evaluation/batch4_v1/none'
        if (run.parent!=root/'runs/research' or run.name!=record.get('job_id') or output!=expected or
            any(path.is_symlink() for path in (run/'evaluation',run/'evaluation/batch4_v1',run/'evaluation/batch4_v1/none'))):return None
        args=_command_options(argv)
        if (args is None or Path(args.get('--run','')).resolve()!=run or
                Path(args.get('--manifest','')).resolve()!=Path(source['manifest']).resolve() or
                args.get('--checkpoint')!='best' or args.get('--image-ablation')!='none' or
                args.get('--max-new-tokens')!='64' or args.get('--ablation-seed')!='42' or
                Path(args.get('--compare-native-cache','')).resolve()!=Path(source['native_cache']).resolve() or
                ('--output' in args and Path(args['--output']).resolve()!=expected)):
            return None
        return {'run_id':run.name,'output':str(output),'worker_pid':pid,'activity_source':'verified_batch4_probe'}
    except (OSError,ValueError,TypeError,KeyError):return None


def batch4_progress(active):
    """Guard counters are32; full evaluation counters512; never borrow train ETA."""
    path=Path(active['output'])/'progress.json'
    base={'run_id':active['run_id'],'stage':'SFT','phase':'batch4_loading_model',
          'activity_verified':True,'activity_source':active.get('activity_source','queue'),
          'source':str(path),'file_age_seconds':None}
    if active.get('worker_pid') is not None:base['worker_pid']=active['worker_pid']
    try:
        item=json.loads(path.read_text());modified=path.stat().st_mtime
        if not isinstance(item,dict):raise ValueError('Progress is not an object')
    except (OSError,ValueError,TypeError):
        return dict(base,event='batch4_loading_before_first_progress')
    # Explicit allowlist prevents answers/predictions or nested private payloads
    # from leaking through an otherwise verified telemetry file.
    public={key:value for key,value in item.items() if key in
            ('event','time','profile','image_ablation','variant','batch_index','elapsed_seconds') and
            (isinstance(value,str) or isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value))}
    public.update(base,file_age_seconds=round(max(0,time.time()-modified),1))
    event=item.get('event');variant=item.get('variant')
    total=None
    if item.get('profile')!='batch4_v1':
        public.update(phase='batch4_preparing',event='batch4_unrecognized_progress_profile')
    elif event in ('guard_native_b1','guard_batch4') and variant in ('none','shuffle','blank','text_only'):
        public['phase']='batch4_guard_'+variant;total=32
    elif event in ('generated_batch','completed'):
        public['phase']='batch4_generation_completed' if event=='completed' else 'batch4_generation';total=512
    elif event=='loading_model':public['phase']='batch4_loading_model'
    else:public['phase']='batch4_preparing'
    completed=item.get('completed_records',item.get('records') if event=='completed' else None)
    if total is not None:
        public['total_records']=total
        if isinstance(completed,int) and not isinstance(completed,bool) and 0<=completed<=total:
            public['completed_records']=completed
        else:
            public.pop('elapsed_seconds',None);public['progress_error']='invalid_record_count'
    # Only an actual elapsed counter can produce ETA. Wall timestamps and stale
    # B1/training duration are not evidence for this worker's current phase.
    return with_progress_eta(public)


def gpu_snapshot():
    fields = ["index", "name", "memory.total", "memory.used", "utilization.gpu", "power.draw", "temperature.gpu"]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        gpus = []
        for values in csv.reader(result.stdout.splitlines()):
            row = {}
            for key, value in zip(fields, values):
                value = value.strip()
                try:
                    row[key] = float(value) if "." in value else int(value)
                except ValueError:
                    row[key] = value
            gpus.append(row)
        return gpus
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": type(error).__name__}


def ram_snapshot(meminfo_path="/proc/meminfo", cgroup_root="/sys/fs/cgroup"):
    """Keep host counters distinct from the container's actual hard limit.

    Unused hard-limit bytes are conservative: charged file cache may be
    reclaimable. Neither counter promises that a model/optimizer will fit.
    """
    values = {}
    try:
        for line in Path(meminfo_path).read_text().splitlines():
            name, raw = line.split(":", 1)
            if name in {"MemTotal", "MemAvailable"}:
                values[name] = int(raw.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    cgroup_root = Path(cgroup_root)
    for version, maximum, current, stat in (
        (2, "memory.max", "memory.current", "memory.stat"),
        (1, "memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes", "memory/memory.stat"),
    ):
        try:
            raw = (cgroup_root / maximum).read_text().strip()
            limit = None if raw == "max" else int(raw)
            used = int((cgroup_root / current).read_text())
            if used < 0 or limit is not None and limit <= 0:
                continue
        except (OSError, ValueError):
            continue
        group = {"version": version, "limit_bytes": limit, "current_bytes": used,
                 "unused_hard_limit_bytes": None if limit is None else max(0, limit - used)}
        try:
            counters = dict(line.split() for line in (cgroup_root / stat).read_text().splitlines())
            group["inactive_file_bytes"] = int(counters.get("total_inactive_file", counters.get("inactive_file", "0")))
        except (OSError, ValueError):
            pass
        values["cgroup"] = group
        break
    if "MemTotal" in values:
        limit = values.get("cgroup", {}).get("limit_bytes")
        values["effective_capacity_bytes"] = min(values["MemTotal"], limit) if limit is not None else values["MemTotal"]
    if "MemAvailable" in values:
        unused = values.get("cgroup", {}).get("unused_hard_limit_bytes")
        values["conservative_available_bytes"] = min(values["MemAvailable"], unused) if unused is not None else values["MemAvailable"]
    return values


def live_queue_job(path):
    """Queue metadata is considered active only with a matching live process."""
    if path is None:return None
    try:
        state=json.loads(Path(path).read_text())
        for run_id,record in state.get('jobs',{}).items():
            evaluation=record.get('status')=='evaluating'
            validation=record.get('status')=='checkpoint_validating'
            grounding=record.get('status')=='grounding_evaluating'
            if grounding:
                variant=record.get('active_grounding_variant')
                diagnostic=record.get('grounding_evaluations',{}).get(variant,{})
                if diagnostic.get('status')!='running':continue
                record=dict(record,evaluation=diagnostic,grounding_variant=variant)
                evaluation=True
            if not evaluation and not validation and record.get('status')!='running':continue
            pid=(record.get('checkpoint_validation',{}).get('pid') if validation else
                 record.get('evaluation',{}).get('pid') if evaluation else record.get('pid'))
            if not isinstance(pid,int):continue
            command=Path(f'/proc/{pid}/cmdline').read_bytes()
            argv=[os.fsdecode(part) for part in command.split(b'\0') if part]
            script=command_script(argv);name=Path(script).name if script is not None else None
            expected=({'validate_research_checkpoint.py','validate_research_stock_deterministic.py'} if validation else
                      {'eval_research_checkpoint.py','eval_research_stock_deterministic.py',
                       'eval_research_archived_native_v1.py',BATCH4_SCRIPT} if evaluation else
                      {'run_research_stage.py','run_research_stock_deterministic.py',
                       'run_research_derived_stock_v1.py'})
            if name in expected:
                return dict(record,run_id=run_id,evaluation_active=evaluation,validation_active=validation,
                            batch4_active=evaluation and name==BATCH4_SCRIPT)
    except (OSError,ValueError,TypeError):pass
    return None


def live_archived_evaluation(root):
    """Recognize the owned standalone evaluator while queue recovery is idle."""
    root = Path(root).resolve()
    script_path = root/'scripts/eval_research_archived_native_v1.py'
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        process = process_identity(int(directory.name))
        if process is None or command_script(process['argv']) != str(script_path):
            continue
        options = _command_options(process['argv'])
        if options is None or options.get('--allow-gpu') is not True:
            continue
        try:
            run = Path(options['--run']).resolve()
            variant = options.get('--image-ablation', 'none')
            role = options.get('--checkpoint', 'best')
            output = Path(options['--output']).resolve()
            expected = run/'evaluation/archived_stock_single_decode_v1'/variant/role
            if (run.parent != root/'runs/research' or output != expected or
                    variant not in ('none', 'shuffle', 'blank', 'text_only') or
                    role not in ('best', 'last') or
                    options.get('--numerical-profile') != 'stock_deterministic_small_v1'):
                continue
            record = dict(run_id=run.name, evaluation_active=True, validation_active=False,
                          evaluation={'output':str(output), 'pid':process['pid']})
            if variant != 'none':
                record['grounding_variant'] = variant
            found.append(record)
        except (KeyError, OSError, ValueError):
            continue
    return found[0] if len(found) == 1 else None


def read_progress(path: Path, *, root=ROOT, queue_state=None):
    parallel=parallel_b1_progress(root)
    if parallel is not None:return parallel
    probe=live_batch4_probe(root)
    if probe is not None:return batch4_progress(probe)
    try:
        raw = json.loads(path.read_text())
        raw["file_age_seconds"] = round(max(0, time.time() - path.stat().st_mtime), 1)
    except (OSError, ValueError, TypeError) as error:
        return {"error": type(error).__name__, "path": str(path)}
    root=Path(root);active=live_queue_job(queue_state) or live_archived_evaluation(root)
    raw['activity_verified']=active is not None
    raw['phase']='train'
    if active is not None and active.get('validation_active'):
        return {'run_id':active['run_id'],'stage':'SFT','phase':'checkpoint_roundtrip',
                'activity_verified':True,'event':'reconstructing_and_replaying_best_last',
                'validation':active.get('checkpoint_validation',{}),'file_age_seconds':None}
    if active is not None and active.get('evaluation_active'):
        if active.get('batch4_active'):
            return batch4_progress({'run_id':active['run_id'],'output':active['evaluation']['output'],
                                    'activity_source':'verified_live_queue'})
        source=Path(active['evaluation']['output'])/'progress.json'
        try:
            item=json.loads(source.read_text());modified=source.stat().st_mtime
        except (OSError,ValueError,TypeError):
            return {'run_id':active['run_id'],'phase':'loading_model','activity_verified':True,
                    'event':'evaluation_loading_before_first_progress','file_age_seconds':None}
        phase='grounding_'+active['grounding_variant'] if active.get('grounding_variant') else 'benchmark_evaluation'
        current={key:value for key,value in item.items() if key not in ('prediction','answers','answer','score')}
        current.update(run_id=active['run_id'],stage='SFT',phase=phase,activity_verified=True,
                       source=str(source),file_age_seconds=round(max(0,time.time()-modified),1),
                       total_records=item.get('selected_records',item.get('manifest_records')))
        if item.get('event') in ('completed','partial_completed'):
            current['phase']=phase+'_completed';current.setdefault('completed_records',item.get('records'))
        return with_progress_eta(current)
    if active is not None and active['run_id']!=raw.get('run_id'):
        return {'run_id':active['run_id'],'phase':'loading_model','activity_verified':True,
                'event':'model_loading_before_first_progress','file_age_seconds':None}
    run_id=raw.get('run_id','')
    if isinstance(run_id,str) and run_id and Path(run_id).name==run_id:
        run=root/'runs/research'/run_id
        candidates=[(run/'selection_progress.json','internal_validation'),
                    (run/'evaluation/heldout_v3/progress.json','benchmark_evaluation')]
        candidates.extend((candidate,'grounding_'+candidate.parent.name)
                          for candidate in (run/'evaluation/grounding_v1').glob('*/progress.json'))
        if active is not None and active.get('evaluation_active'):
            phase='grounding_'+active['grounding_variant'] if active.get('grounding_variant') else 'benchmark_evaluation'
            candidates.append((Path(active['evaluation']['output'])/'progress.json',phase))
        newer=[]
        for candidate,phase in candidates:
            try:
                if candidate.stat().st_mtime>path.stat().st_mtime:
                    item=json.loads(candidate.read_text())
                    newer.append((candidate.stat().st_mtime,candidate,phase,item))
            except (OSError,ValueError,TypeError):pass
        if newer:
            modified,source,phase,item=max(newer,key=lambda entry:entry[0])
            training={'step':raw.get('step'),'total_steps':raw.get('total_steps'),
                      'completed_records':raw.get('completed_records'),'total_records':raw.get('total_records'),
                      'stage':raw.get('stage')}
            # Phase telemetry contains no teacher references or predictions.
            raw={key:value for key,value in item.items() if key not in ('prediction','answers','answer','score')}
            raw.update(run_id=run_id,phase=phase,training_progress=training,
                       stage=training['stage'],
                       activity_verified=active is not None,source=str(source),
                       file_age_seconds=round(max(0,time.time()-modified),1))
            if phase=='internal_validation':
                raw['elapsed_seconds']=max(0,modified-path.stat().st_mtime)
                if raw.get('completed_records')==raw.get('total_records'):
                    raw['phase']='checkpointing_after_validation'
            else:
                raw['total_records']=raw.get('selected_records',raw.get('manifest_records'))
                if raw.get('event') in ('completed','partial_completed'):
                    raw['phase']=phase+'_completed'
                    raw.setdefault('completed_records',raw.get('records'))
    return with_progress_eta(raw)


def with_progress_eta(raw):
    completed, total, elapsed = (raw.get(k) for k in ("completed_records", "total_records", "elapsed_seconds"))
    if (all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (completed,total,elapsed))
            and total>=completed>0 and elapsed>0):
        rate = completed / elapsed
        raw["average_records_per_second"] = rate
        raw["eta_current_stage_hours"] = max(0, total - completed) / rate / 3600
        raw["estimated_total_current_stage_hours"] = total / rate / 3600
    return raw


def snapshot(root: str | Path, reserve_gb: float, progress_path: str | Path, queue_state=None):
    disk = DiskBudget(root, reserve_gb).report()
    return {
        "time": datetime.now(timezone.utc).isoformat(), "unix_time": time.time(),
        "disk": disk, "disk_reserve_intact": disk["free_bytes"] >= disk["reserve_bytes"],
        "ram": ram_snapshot(), "gpus": gpu_snapshot(), "progress": read_progress(Path(progress_path),root=root,queue_state=queue_state),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output", default=str(ROOT / "logs/research/resources.jsonl"))
    parser.add_argument("--progress", default=str(ROOT / "logs/research/progress.json"))
    parser.add_argument("--queue-state",help='Optional queue state; live PIDs identify model loading/evaluation phases')
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--reserve-gb", type=float, default=120)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    running = True

    def stop(_signal, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while running:
        start = time.monotonic()
        current = snapshot(args.root, args.reserve_gb, args.progress,args.queue_state)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(current, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        _atomic_json(output.with_name("resources_latest.json"), current)
        progress = current["progress"]
        print(json.dumps({"time": current["time"], "stage": progress.get("stage"),
                          "phase":progress.get('phase'),"activity_verified":progress.get('activity_verified'),
                          "run_id": progress.get("run_id"), "free_gib": current["disk"]["free_gib"],
                          "reserve_intact": current["disk_reserve_intact"],
                          "completed_records": progress.get("completed_records"),
                          "total_records": progress.get("total_records"),
                          "eta_stage_hours": progress.get("eta_current_stage_hours"),
                          "gpus": current["gpus"]}, ensure_ascii=False), flush=True)
        if args.once:
            break
        while running and time.monotonic() - start < args.interval:
            time.sleep(min(1, args.interval - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
