#!/usr/bin/env python3
"""Closed-by-default archival execution profile for the unchanged24-job queue.

No service or signal occurs at import/default preflight. Root's actual4f CPU+
GPU activation is mandatory. Only future full Small3a/3c/2abc job COPIES receive
the existing frozen archival mixin flag; SPEC/argv/numerical policies stay intact.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_parallel_grounding_queue as parent
from scripts.research_queue import ResearchQueue,argv_value,output_dir,artifacts_for,timestamp
from src.utils.research_checkpoint_archival_queue import ProspectiveSmallArchivalMixin
from src.utils import research_checkpoint_archival as archival

OWNER='prospective_small_archival_parallel_queue_v1'
SCRIPT='scripts/research_archival_parallel_queue_v1.py'
TEST='tests/test_research_archival_parallel_queue_v1.py'
READY='configs/research/archival_parallel_queue_v1_ready.json'
SPEC='configs/research/smallthinker_pilot_scienceqa_context_v2_balanced_queue.json'
SPEC_SHA256='c4c8181b032538f7d382c10e17af5d826141fae0df18a420dd891d55b75a7442'
SPEC_CANONICAL_SHA256='708a1fab40cab99eeac6f4cdb97d83a86c8c8c783bbecae3a945eac30b1c4f20'
ELIGIBLE=('3a','3c','2a','2b','2c')
PINNED={'scripts/research_queue.py':'446d928181aba5f77d7e02692279741eb10d8fc3d3409af985e07d9239263bda',
 'scripts/research_parallel_grounding_queue.py':'71c5220d305229c3192b14765e56bd99042921c454f32b3f4f86e6314dfb9d6b',
 'src/utils/research_checkpoint_archival_queue.py':'f16d1620975a23bc30315811f98daf5d1bf060a4591e9dd41ba3fd8ffc563ef1',
 'src/utils/research_checkpoint_archival.py':'cd507acf15f4c96abb9c0ca232a9c198f590906345ee6080d38325083e2777b3'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def verify_sources():
    for name,expected in PINNED.items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen archival queue dependency changed: '+name)


def activation(expected_sha):
    if not isinstance(expected_sha,str) or len(expected_sha)!=64 or any(x not in '0123456789abcdef' for x in expected_sha):
        raise archival.ArchivalGateClosed('Explicit ROOT actual activation SHA256 is required; no boolean bypass')
    # The frozen mixin passes None, which resolves to this exact fixed path.
    evidence=archival.activation_evidence()
    if evidence['path']!=str(archival.ACTIVATION.resolve()) or evidence['sha256']!=expected_sha:
        raise archival.ArchivalGateClosed('Fixed actual activation path/SHA changed')
    return evidence


def cpu_readiness():
    verify_sources();value=json.loads((ROOT/READY).read_bytes())
    if value.get('owner')!=OWNER or value.get('status')!='passed' or value.get('tests_passed',0)<1:
        raise ValueError('New archival queue CPU readiness is not passed')
    for name,expected in value['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Archival execution source/proof changed: '+name)
    return {'path':READY,'sha256':sha(ROOT/READY),'source_sha256':value['source_sha256']}


def spec_snapshot(path):
    path=Path(path).resolve(strict=True)
    if path!=(ROOT/SPEC).resolve() or sha(path)!=SPEC_SHA256:raise ValueError('Only the same immutable24-job SPEC is permitted')
    value=json.loads(path.read_bytes())
    if len(value['jobs'])!=24 or digest(value)!=SPEC_CANONICAL_SHA256:raise ValueError('Immutable queue canonical digest differs')
    if 'trained_small_archival_activation' in value or any('archive_completed_small' in job for job in value['jobs']):
        raise ValueError('Archival execution metadata must never be inserted in SPEC')
    return value


def _private_native_run():
    """One nonnumeric loop-entry hook, all other frozen base run AST unchanged."""
    verify_sources()
    tree=ast.parse(textwrap.dedent(inspect.getsource(ResearchQueue.run)))
    loops=[node for node in ast.walk(tree) if isinstance(node,ast.For) and isinstance(node.target,ast.Name)
        and node.target.id=='job' and ast.unparse(node.iter)=="self.spec['jobs']"]
    if len(loops)!=1:raise ValueError('Frozen queue24-job loop shape changed')
    loops[0].body.insert(0,ast.parse('self._archival_job_boundary(job)').body[0])
    ast.fix_missing_locations(tree)
    namespace=dict(ResearchQueue.run.__globals__)
    exec(compile(tree,'<archival_queue_v1_private_native_run>','exec'),namespace)
    return namespace['run']


def _native_raw_roles(run):
    plan=archival.eligibility(run)
    for item in plan['roles'].values():
        if item['marker'].get('checkpoint_archival'):raise ValueError('NoBenefit fallback refuses archived/mixed roles')
        with Path(item['state']).open('rb') as handle:
            if handle.read(4)!=b'PK\x03\x04':raise ValueError('NoBenefit fallback needs native compacted raw ZIP roles')
    return plan


class ArchivalParallelResearchQueue(ProspectiveSmallArchivalMixin,parent.ParallelGroundingResearchQueue):
    def __init__(self,spec_path,state_path=None,retry_failed=False,*,proof_run,activation_sha,workers=4,timeout_seconds=1200):
        # All closed gates run BEFORE parent creates a state lock/log directory.
        evidence=activation(activation_sha);ready=cpu_readiness();original=spec_snapshot(spec_path)
        self.activation_sha=activation_sha;self.activation_bound=evidence;self.archival_ready=ready
        super().__init__(spec_path,state_path,retry_failed,proof_run=proof_run,workers=workers,timeout_seconds=timeout_seconds)
        try:
            if self.spec!=original or self.spec_hash!=SPEC_CANONICAL_SHA256:raise ValueError('Parent altered immutable SPEC')
            descriptor={'owner':OWNER,'schema_version':1,'spec_sha256':self.spec_hash,'spec_file_sha256':SPEC_SHA256,
                'activation':evidence,'execution_ready':ready,'eligible_full_recipes':list(ELIGIBLE),
                'raw_authoritative_recipes':['4f','3b'],'transient_job_copy_only':True,
                'base_run_hook':'one activation/SPEC check at each frozen loop entry; no trainer argv changes'}
            extra=self.state.setdefault('extra',{})
            if not isinstance(extra,dict):raise ValueError('Queue extra metadata must be an object')
            previous=extra.get(OWNER)
            if previous is not None and previous!=descriptor:raise ValueError('Archival execution profile changed; no implicit state migration')
            extra[OWNER]=descriptor
        except BaseException:
            self.lock_handle.close();raise

    def _guard(self):
        verify_sources()
        if activation(self.activation_sha)!=self.activation_bound or cpu_readiness()!=self.archival_ready:
            raise ValueError('Actual archival activation or execution proof changed')
        if spec_snapshot(self.spec_path)!=self.spec or digest(self.spec)!=self.spec_hash:
            raise ValueError('Immutable SPEC/argv was modified during archival execution')

    def _archival_job_boundary(self,job):
        self._guard()
        original=next((value for value in self.spec['jobs'] if value['id']==job['id']),None)
        if original!=job:raise ValueError('Frozen run received a modified job instead of original SPEC entry')

    def run(self):
        self._guard()
        return _private_native_run()(self)

    def _archival_job(self,job):
        if argv_value(job['argv'],'--stage')!='SFT' or argv_value(job['argv'],'--recipe') not in ELIGIBLE:return False
        if not job['id'].startswith('smallthinker4__siglip2__'):raise ValueError('Archival scope requires exact Small/SigLIP2 pilot jobs')
        if not job.get('compact_completed') or not job.get('evaluation'):raise ValueError('Future archival SFT requires compacted pre-evaluation lifecycle')
        return True

    def _validate_skip(self,job,receipt):
        run=output_dir(job['argv']);plan=_native_raw_roles(run)
        if (receipt.get('owner')!=OWNER or receipt.get('status')!='skipped_no_benefit_native_raw' or
            receipt.get('activation')!=self.activation_bound or receipt.get('metadata_sha256')!=plan['metadata_sha256']):
            raise ValueError('Measured NoBenefit raw-source receipt changed')
        journal=archival.read(run/archival.JOURNAL)
        if (journal.get('phase')!='no_benefit' or journal.get('activation')!=self.activation_bound or
            journal.get('source_roles')!=receipt['source_roles'] or journal.get('source_state_sha256')!=receipt['source_state_sha256']):
            raise ValueError('Measured NoBenefit journal/source differs')
        if {role:item['object'] for role,item in plan['roles'].items()}!=receipt['source_roles']:
            raise ValueError('NoBenefit original compacted raw role identity changed')
        source=run/'.checkpoint_objects'/journal['source_object']/'state.pt'
        if archival.sha(source)!=journal['source_state_sha256']:raise ValueError('NoBenefit measured native source bytes changed')
        return plan

    def compact_completed_job(self,job,record):
        self._guard()
        effective=copy.deepcopy(job)
        if not self._archival_job(job):return super().compact_completed_job(effective,record)
        effective['archive_completed_small']=True
        if record.get('checkpoint_archival_skip'):
            self._validate_skip(job,record['checkpoint_archival_skip']);return
        if (record.get('evaluation') or record.get('checkpoint_validation') or record.get('grounding_evaluations')) and not record.get('checkpoint_archival'):
            raise ValueError('Prospective archive cannot relabel an already-started ordinary evaluation')
        try:
            return super().compact_completed_job(effective,record)
        except archival.NoBenefit as error:
            run=output_dir(job['argv']);plan=_native_raw_roles(run);journal=archival.read(run/archival.JOURNAL)
            receipt={'owner':OWNER,'status':'skipped_no_benefit_native_raw','reason':str(error),
                'activation':self.activation_bound,'source_roles':journal.get('source_roles'),
                'source_state_sha256':journal.get('source_state_sha256'),'metadata_sha256':plan['metadata_sha256']}
            self._validate_skip(job,receipt)
            record.update(training_completed=True,status='training_completed',failure_phase=None,
                checkpoint_archival_skip=receipt,checkpoint_postprocess_kind='measured_no_benefit_native_raw',artifacts=artifacts_for(job))
            for key in ('error','error_type'):record.pop(key,None)
            self.persist()

    def evaluate_completed_job(self,job,record):
        self._guard()
        if self._archival_job(job):
            if record.get('checkpoint_archival_skip'):self._validate_skip(job,record['checkpoint_archival_skip'])
            elif archival.validate_completed_archival(output_dir(job['argv']))!=record.get('checkpoint_archival'):
                raise ValueError('Evaluation requires exact completed archival or measured raw NoBenefit receipt')
        return super().evaluate_completed_job(job,record)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',default=SPEC);parser.add_argument('--state');parser.add_argument('--proof-run')
    parser.add_argument('--activation-sha');parser.add_argument('--workers',type=int,choices=(2,4),default=4)
    parser.add_argument('--timeout-seconds',type=int,default=1200);parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--run-queue',action='store_true')
    args=parser.parse_args(argv)
    if not args.run_queue:
        try:evidence=activation(args.activation_sha);ready=cpu_readiness();spec_snapshot(args.spec)
        except (OSError,ValueError) as error:
            print(json.dumps({'owner':OWNER,'status':'closed_preflight','reason':str(error),'jobs_started':False,'signals_sent':False}));return 2
        print(json.dumps({'owner':OWNER,'status':'eligible_execution_metadata_only','activation':evidence,'ready':ready,'jobs_started':False,'signals_sent':False}));return 0
    if not args.proof_run:parser.error('--run-queue needs the unchanged actual --proof-run')
    queue=ArchivalParallelResearchQueue(args.spec,args.state,args.retry_failed,proof_run=args.proof_run,
        activation_sha=args.activation_sha,workers=args.workers,timeout_seconds=args.timeout_seconds)
    signal.signal(signal.SIGINT,queue.signal);signal.signal(signal.SIGTERM,queue.signal)
    return queue.run()


if __name__=='__main__':raise SystemExit(main())
