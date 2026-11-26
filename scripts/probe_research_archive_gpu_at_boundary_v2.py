#!/usr/bin/env python3
"""ROOT-only fixed SOURCE4f calibration at fixed TARGET3b/text_only boundary.

This is a new one-shot owner. The unused4f-target v1 release stays byte-frozen.
Its tested process/lease machinery is loaded into a private module instance;
only explicit source/target admission, target waiting, proof labelling and the
approved readonly post-validation state-file cache advice are replaced.
"""
from __future__ import annotations
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
V1_SCRIPT='scripts/probe_research_archive_gpu_at_boundary.py'
V1_SHA256='96689e3690f704bb3beeed9157e6c90ab88412a31af58f6d47e97f24ac3c4462'
OWNER='source4f_target3b_archive_gpu_textonly_boundary_v2'
SCRIPT='scripts/probe_research_archive_gpu_at_boundary_v2.py'
TEST='tests/test_research_archive_gpu_boundary_v2.py'
READY='configs/research/archive_gpu_boundary_3b_ready_v2.json'
RECEIPT='research_results/archive_gpu_boundary_source4f_target3b_text_only_v2.json'
SOURCE='smallthinker4__siglip2__4f__SFT_pilot_scienceqa_context_v2_balanced_s42'
TARGET='smallthinker4__siglip2__3b__SFT_pilot_scienceqa_context_v2_balanced_s42'
CPU_REPORT='research_results/trained_small_delta_actual4f_best_v2/verified.json'
CPU_REPORT_SHA256='2393256f5648b7aee67301b7f00665eeeda509a2c168104394f606a1aa204c80'
HELPER_READY='research_results/trained_small_archive_gpu_parity_ready_v2.json'
HELPER_READY_SHA256='6d6e017f6d4ad662a67decf20fb2a169c427f828a1c255b151175eb7bc47fc96'
OUTPUT='research_results/trained_small_archive_gpu_parity_source4f_target3b_v2'


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if file_sha(ROOT/V1_SCRIPT)!=V1_SHA256:raise ValueError('Frozen boundary v1 source differs')
_spec=importlib.util.spec_from_file_location('_private_archive_boundary_v2_core',ROOT/V1_SCRIPT)
core=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(core)
if file_sha(ROOT/V1_SCRIPT)!=V1_SHA256:raise ValueError('Frozen boundary v1 changed while loading')
_readiness=core.readiness
_phase=core.phase
_verify_native=core.verify_native_boundary
_validate_result=core.validate_result
for _key in ('OWNER','SCRIPT','TEST','READY','RECEIPT','TARGET','HELPER_READY','OUTPUT'):
    setattr(core,_key,globals()[_key])
core.CHECKS=(*(x.replace('fixed4f_native_text_only','fixed3b_native_text_only').replace('completed_compacted_source_roles','completed_compacted_target_roles')
    for x in core.CHECKS),'fixed_source4f_distinct_target3b','target_earlier_training_wait_no_stop',
    'source_target_native_proofs_separately_bound','approved_scoped_readonly_postvalidation_cache_advice',
    'advice_unique_owned_completed_files_unmapped_stat_roles_unchanged','remeasure_no_assumed_reclaimable_credit')


def readiness(root):
    if core.sha(root/HELPER_READY)!=HELPER_READY_SHA256:raise ValueError('Only the explicit reviewed helper v2 is accepted')
    result=_readiness(root)
    if core.sha(root/V1_SCRIPT)!=V1_SHA256:raise ValueError('Frozen boundary v1 source changed')
    proof=core.read(root/READY)
    if proof.get('source_run')!=SOURCE or proof.get('target_run')!=TARGET or proof.get('cpu_report_sha256')!=CPU_REPORT_SHA256:
        raise ValueError('Boundary v2 source/target/CPU proof binding differs')
    return result


def arm(root,cpu_report,cpu_sha,controller_pid):
    if Path(cpu_report).resolve()!=(root/CPU_REPORT).resolve() or cpu_sha!=CPU_REPORT_SHA256:
        raise ValueError('Only the fixed actual4f surviving CPU proof is accepted')
    core.read(cpu_report)
    if core.sha(cpu_report)!=CPU_REPORT_SHA256:raise ValueError('Actual4f CPU proof bytes changed')
    release=readiness(root)
    from src.utils import research_archive_gpu_parity as parity
    plan=parity.prepare_plan(cpu_report)
    source_run=(root/'runs/research'/SOURCE).resolve();target_run=(root/'runs/research'/TARGET).resolve()
    if Path(plan['run'])!=source_run or source_run==target_run:raise ValueError('GPU helper source must be4f, never the3b timing target')
    parent=core.controller(root,controller_pid);spec=core.read(root/core.SPEC)
    state=core.read(root/(core.SPEC.removesuffix('.json')+'.state.json'))
    if len(spec.get('jobs',[]))!=24 or state.get('spec_sha256')!=core.digest(spec):raise ValueError('Immutable24-job state/spec differs')
    from scripts.research_queue import argv_value
    jobs=[j for j in spec['jobs'] if j.get('id')==TARGET]
    if len(jobs)!=1:raise ValueError('Explicit3b SFT timing target absent/duplicated')
    job=jobs[0]
    if (argv_value(job['argv'],'--stage')!='SFT' or argv_value(job['argv'],'--recipe')!='3b' or
        core.absolute(root,argv_value(job['argv'],'--output'))!=target_run or
        not job.get('compact_completed') or job.get('evaluation',{}).get('checkpoint','best')!='best'):
        raise ValueError('Only fixed3b full SFT native timing target accepted')
    return {'owner':OWNER,'boot_id':core.boot(),'parent':parent,'spec_sha256':core.digest(spec),
        'spec_file_sha256':core.sha(root/core.SPEC),'arm_state_sha256':core.sha(root/(core.SPEC.removesuffix('.json')+'.state.json')),
        'release':release,'cpu_report':str(Path(cpu_report).resolve()),'cpu_report_sha256':CPU_REPORT_SHA256,'plan':plan,
        'job':job,'run':str(target_run),'source_run':str(source_run),'target_run':str(target_run),
        'source_job':SOURCE,'target_job':TARGET,'job_order':[j['id'] for j in spec['jobs']],'variant':'text_only'}


def assert_arm(root,bound):
    if (bound.get('owner')!=OWNER or bound.get('boot_id')!=core.boot() or
        bound.get('source_job')!=SOURCE or bound.get('target_job')!=TARGET or
        Path(bound['source_run'])!=(root/'runs/research'/SOURCE).resolve() or
        Path(bound['target_run'])!=(root/'runs/research'/TARGET).resolve() or
        bound['run']!=bound['target_run'] or bound['plan']['run']!=bound['source_run']):
        raise ValueError('Fixed distinct source4f/target3b binding changed')
    if core.sha(root/core.SPEC)!=bound['spec_file_sha256'] or readiness(root)!=bound['release']:
        raise ValueError('Bound source/spec/helper/controller release changed')
    if bound.get('job_order')!=[j['id'] for j in core.read(root/core.SPEC)['jobs']]:raise ValueError('Immutable target job ordering changed')
    if core.sha(bound['cpu_report'])!=CPU_REPORT_SHA256:raise ValueError('Pinned actual4f CPU proof changed')
    from src.utils import research_archive_gpu_parity as parity
    parity.assert_plan(bound['plan'])


def phase(root,bound,expected=None):
    if expected is None:
        state=core.read(root/(core.SPEC.removesuffix('.json')+'.state.json'))
        if state.get('status')!='running' or state.get('spec_sha256')!=bound['spec_sha256']:return 'missed',None
        rows=state.get('jobs')
        if not isinstance(rows,dict) or any(not isinstance(r,dict) for r in rows.values()):raise ValueError('Malformed queue state')
        order=bound['job_order']
        active=[name for name,r in rows.items() if r.get('status') in ('running','evaluating','checkpoint_validating','grounding_evaluating','parallel_evaluating')]
        if len(active)!=1 or active[0] not in order or TARGET not in order:return 'missed',None
        if order.index(active[0])<order.index(TARGET):return 'waiting',None
        if order.index(active[0])>order.index(TARGET):return 'missed',None
        row=rows[TARGET]
        if row.get('status')=='running':return 'waiting',None
    return _phase(root,bound,expected)


def _available(snapshot):
    value=snapshot['MemAvailable_bytes']
    if snapshot.get('memory.max') is not None:value=min(value,max(0,snapshot['memory.max']-snapshot['memory.current']))
    return value


def postvalidation_cache_advice(bound,evidence,*,lookup=None,advisor=None,host=None):
    """Only original completed manager-owned source/target state files.

    These injectable callbacks exist solely for synthetic CPU fixtures. Actual
    execution uses the frozen ownership/mapping/stat guard and readonly fadvise.
    """
    from src.utils import research_checkpoint_archival as archival
    from src.utils import research_trained_delta_probe as probe
    from src.utils import research_archive_gpu_parity as parity
    lookup=lookup or archival.eligibility;advisor=advisor or probe.advise_source_cache;host=host or parity.host_snapshot
    if not core.zero_exit(evidence['child']) or not core.gpu_idle():raise ValueError('Native3b child must be naturally dead and GPU idle before advice')
    runs=(Path(bound['source_run']),Path(bound['target_run']))
    if runs[0]==runs[1]:raise ValueError('Advice requires distinct explicit4f source and3b target')
    before={str(run):lookup(run,allow_archived=False) for run in runs}
    if (before[str(runs[0])]['config'].get('recipe')!='4f' or before[str(runs[1])]['config'].get('recipe')!='3b' or
        any(value['config'].get('stage')!='SFT' for value in before.values())):
        raise ValueError('Only completed full4f source and3b target cache files may be advised')
    source_roles=before[str(runs[0])]['roles'];saved=bound['plan']['gate']['roles']
    for role,item in source_roles.items():
        if Path(item['state'])!=Path(saved[role]['source']) or item['stat']!=saved[role]['stat'] or item['marker']!=saved[role]['marker']:
            raise ValueError('Authoritative actual4f roles/stat/owner changed before readonly advice')
    unique={}
    for run,value in before.items():
        owner=value['manager']['owner']
        for role,item in value['roles'].items():
            if item['marker'].get('owner')!=owner:raise ValueError('Advice candidate foreign manager owner')
            path=Path(item['state'])
            if path.name!='state.pt' or path.is_symlink() or path.parent.parent!=Path(run)/'.checkpoint_objects':
                raise ValueError('Advice cannot target unknown/non-managed files')
            probe._no_source_mappings(path)
            if path in unique and unique[path]['stat']!=item['stat']:raise ValueError('Ambiguous shared checkpoint stat')
            unique.setdefault(path,{'stat':item['stat'],'roles':[]})['roles'].append({'run':run,'role':role,'owner':owner})
    initial=host();events=[]
    for path,item in unique.items():
        events.append({'path':str(path),'roles':item['roles'],'result':advisor(path,item['stat'])})
    after={str(run):lookup(run,allow_archived=False) for run in runs}
    if before!=after:raise ValueError('Readonly cache advice changed or raced manager ownership/roles/stat/metadata')
    final=host()
    return {'operation':'scoped_POSIX_FADV_DONTNEED_after_strict_validation','source_run':str(runs[0]),
        'target_run':str(runs[1]),'unique_owned_files':len(unique),'events':events,
        'source_target_metadata_ownership_roles_and_stat_unchanged':True,'own_source_mmaps_absent':True,
        'host_before':initial,'host_after':final,'actual_available_before_bytes':_available(initial),
        'actual_available_after_bytes':_available(final),'reclaimable_cache_credit_assumed_bytes':0,
        'native_helper_memory_admission_still_required':True,'checkpoint_bytes_or_pointers_changed':False}


def verify_native_boundary(root,bound,evidence):
    proof=_verify_native(root,bound,evidence)
    assert_arm(root,bound)
    proof.update(source_run=bound['source_run'],target_run=bound['target_run'],source_cpu_report_sha256=CPU_REPORT_SHA256,
        source_role_snapshot=bound['plan']['gate']['roles'])
    proof['postvalidation_cache_advice']=postvalidation_cache_advice(bound,evidence)
    assert_arm(root,bound)
    return proof


def validate_result(bound,output,nonce,*,worker,coordinator,deadline):
    source_bound=dict(bound,run=bound['source_run'])
    result=_validate_result(source_bound,output,nonce,worker=worker,coordinator=coordinator,deadline=deadline)
    result.update(source_run=bound['source_run'],target_run=bound['target_run'],source_job=SOURCE,target_job=TARGET,
        timing_boundary_variant='text_only',GPU_parity_model_is_source4f_not_target3b=True)
    return result


for _name in ('readiness','arm','assert_arm','phase','verify_native_boundary','validate_result'):
    setattr(core,_name,globals()[_name])
# Re-export the unchanged private process/lease API for CPU fixture reuse.
for _name,_value in vars(core).items():
    if not _name.startswith('_') and _name not in globals():globals()[_name]=_value


if __name__=='__main__':core.main()
