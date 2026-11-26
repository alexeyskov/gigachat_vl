#!/usr/bin/env python3
"""Opt-in ordinary evaluation followed by mandatory three-way grounding checks.

Uses the unchanged ResearchQueue trainer/compaction/ordinary-none hook. Each
intervention gets its own evaluator output and strict receipt. No service is
started by importing or creating this wrapper.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];CODE_ROOT=ROOT
sys.path.insert(0,str(ROOT))
from scripts.research_queue import ResearchQueue,argv_value,output_dir,resolve_path,timestamp,emit
from src.utils.research_storage import _atomic_json
from src.utils.research_checkpoint_compat import same_checkpoint_identity
from src.eval.research_ablation import transform_image_ablation,write_transformed_manifest
from src.eval.research_metrics import score_prediction,aggregate_scores

MODES=('shuffle','blank','text_only')
REQUIRED_RECORDS=512
GROUPS={'general_vqa','ocr_documents','charts_diagrams','knowledge_reasoning'}
ROW_FIELDS=('record_id','question','image_hashes','answers','scorer','task_group','source','benchmark','choices','scoring_protocol','image_ablation')
REQUIRED_SOURCE_FILES=('scripts/eval_research_checkpoint.py','src/eval/research_ablation.py','src/eval/research_metrics.py','src/eval/research_inference.py','scripts/run_research_stage.py')


def read_rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_selection_binding(config):
    """Rebuild the actual sampled selection, separately from its source file.

    resolved_config.selection_manifest_sha256 hashes all source JSONL bytes;
    the native replay receipt hashes the converted, sampled record list. These
    identities have different scopes even when every source row is selected.
    """
    from scripts.run_research_stage import prepare_group_macro_selection
    selection=prepare_group_macro_selection(config)
    return {'manifest_sha256':selection['protocol']['selection_manifest_sha256'],
            'records':len(selection['records'])}


def paired_image_bootstrap(original,variants,records,*,replicates=1000,seed=42):
    """Paired original-image-cluster bootstrap, stratified by task group.

    All questions of a sampled original canonical image move together. Clusters
    are sampled with replacement within each group; each replicate recomputes
    question-weighted group means and then the equal four-group macro delta.
    """
    base={r['record_id']:r for r in original};rng=random.Random(seed);result={}
    for variant,rows in variants.items():
        changed={r['record_id']:r for r in rows};clusters=defaultdict(lambda:defaultdict(list))
        for row in records:
            identity=tuple(row['image_hashes'])
            clusters[row['task_group']][identity].append(float(base[row['record_id']]['score'])-float(changed[row['record_id']]['score']))
        draws=[];point=[]
        for group in sorted(GROUPS):
            values=[v for cluster in clusters[group].values() for v in cluster]
            point.append(sum(values)/len(values))
        for _ in range(replicates):
            means=[]
            for group in sorted(GROUPS):
                values=list(clusters[group].values());selected=rng.choices(values,k=len(values))
                means.append(sum(sum(x) for x in selected)/sum(len(x) for x in selected))
            draws.append(100*sum(means)/len(means))
        draws.sort()
        result[variant]={'original_minus_variant_group_macro_points':100*sum(point)/len(point),
            'paired_95_percentile_ci':[draws[math.floor(.025*(len(draws)-1))],draws[math.ceil(.975*(len(draws)-1))]],
            'replicates':replicates,'seed':seed,'original_image_clusters_by_group':{g:len(v) for g,v in clusters.items()},
            'method':'paired original canonical-image cluster bootstrap stratified by task group; question-weighted group means, equal-group macro',
            'full_benchmark_or_cross_seed_inference':False}
    return result


class GroundingResearchQueue(ResearchQueue):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        try:
            for job in self.spec['jobs']:
                if argv_value(job['argv'],'--stage')!='SFT':continue
                evaluation=job.get('evaluation')
                if not evaluation or evaluation.get('checkpoint','best')!='best' or evaluation.get('max_new_tokens',64)!=64:
                    raise ValueError('Grounding SFT requires ordinary evaluation on best with64 new tokens')
                if argv_value(job['argv'],'--selection-metric')!='group_macro':
                    raise ValueError('Grounding checkpoint gate requires group_macro SFT selection')
                rows=read_rows(resolve_path(evaluation['manifest']))
                if (len(rows)!=REQUIRED_RECORDS or len({r['record_id'] for r in rows})!=REQUIRED_RECORDS or
                    {r['task_group'] for r in rows}!=GROUPS or any(len(r.get('image_paths',[]))!=1 or len(r.get('image_hashes',[]))!=1 for r in rows)):
                    raise ValueError('Grounding requires512 distinct evaluation records covering all four groups, each with one image')
            self.state['grounding_controller']={'schema_version':1,'variants':list(MODES),'records':REQUIRED_RECORDS,
                'checkpoint':'best','max_new_tokens':64,'seed':42,'controller_source_sha256':sha(__file__),
                'parent_queue_source_sha256':sha(CODE_ROOT/'scripts/research_queue.py')}
        except Exception:
            self.lock_handle.close();raise

    def persist(self):
        # Parent run() preserves training_completed but names all post-train
        # failures eval_failed. Keep our distinct phase while retaining its
        # lifecycle, dependency handling and trainer-never-relaunch branch.
        for row in self.state.get('jobs',{}).values():
            if row.get('failure_phase')=='checkpoint_validation' and row.get('checkpoint_validation',{}).get('status') in ('failed','interrupted'):
                row['status']='training_completed_checkpoint_validation_'+row['checkpoint_validation']['status']
            if row.get('failure_phase')=='grounding' and row.get('grounding_status') in ('failed','interrupted'):
                row['status']='training_completed_grounding_'+row['grounding_status']
        super().persist()

    def _checkpoint_validation_identity(self,job):
        from scripts.validate_research_checkpoint import managed_checkpoints,check_manifest_hashes
        from src.utils.research_checkpoint_compat import checkpoint_identity
        run=output_dir(job['argv']);config=json.loads((run/'resolved_config.json').read_text())
        if config.get('stage')!='SFT' or config.get('selection_metric')!='group_macro':
            raise ValueError('Full checkpoint gate requires SFT group_macro selection')
        check_manifest_hashes(config)
        last=[time.monotonic()]
        def progress(done,total):
            if self.interrupted_by:raise InterruptedError('Interrupted while identifying retained validation checkpoints')
            if time.monotonic()-last[0]>=30:
                self.budget.ensure_space(0);emit('checkpoint_validation_hashing',job=job['id'],bytes_completed=done,bytes_total=total)
                last[0]=time.monotonic()
        entries=managed_checkpoints(run)
        for entry in entries:entry['artifact_identity']=checkpoint_identity(entry['path'],progress=progress)
        return {'run':run,'config':config,'entries':entries,'config_sha256':sha(run/'resolved_config.json'),
                'selection_binding':checkpoint_selection_binding(config)}

    def _validate_checkpoint_report(self,identity):
        from src.eval.research_selection import PROTOCOL_NAME
        path=identity['run']/'checkpoint_roundtrip.json';report=json.loads(path.read_text())
        if report.get('schema_version')!=1 or report.get('status')!='passed' or report.get('mode')!='full_gpu_replay' or report.get('run')!=str(identity['run']):
            raise ValueError('Current full_gpu_replay checkpoint roundtrip report is required')
        rows=report.get('checkpoints',[]);by_path={row.get('checkpoint_path'):row for row in rows}
        if len(rows)!=len(by_path) or set(by_path)!={str(entry['path']) for entry in identity['entries']}:
            raise ValueError('Checkpoint roundtrip must cover every current unique best/last object exactly')
        for entry in identity['entries']:
            row=by_path[str(entry['path'])];stats=entry['path'].stat()
            expected={'schema_version':1,'status':'passed','checkpoint_owner':entry['owner'],'checkpoint_step':entry['marker']['step'],
                'checkpoint_size_bytes':stats.st_size,'checkpoint_mtime_ns':stats.st_mtime_ns,'all_tensors_finite':True,
                'expected_keys_match':True,'restored_tensor_max_abs_diff':0.,'projector_frozen':True,
                'projector_unchanged_during_validation':True,'selection_metric_replayed':True,'selection_protocol':PROTOCOL_NAME,
                'heldout_ce_replayed':False,'official_benchmark_score':False,
                'selection_manifest_sha256':identity['selection_binding']['manifest_sha256'],
                'validation_records':identity['selection_binding']['records']}
            mismatches=[key for key,value in expected.items() if row.get(key)!=value]
            if mismatches:raise ValueError('Checkpoint roundtrip ownership/stat/reconstruction/selection proof differs: '+', '.join(mismatches))
            if sorted(row.get('checkpoint_labels',[]))!=sorted(entry['labels']):raise ValueError('Checkpoint best/last labels differ')
            count=row.get('validated_tensor_count')
            if not isinstance(count,int) or isinstance(count,bool) or count<=0:raise ValueError('Checkpoint report has no validated tensors')
            if not isinstance(row.get('validation_records'),int) or isinstance(row['validation_records'],bool) or row['validation_records']<=0:
                raise ValueError('Checkpoint report replayed no internal selection records')
            if entry['artifact_identity']['format']=='bundle_v1' and not same_checkpoint_identity(entry['artifact_identity'],row.get('checkpoint_bundle_identity',{})):
                raise ValueError('Checkpoint roundtrip bundle components changed')
            saved=float(row['group_macro_score_saved']);actual=float(row['group_macro_score_replayed'])
            difference=float(row['group_macro_score_abs_diff']);tolerance=2e-3+2e-4*abs(saved)
            if (not all(math.isfinite(x) for x in (saved,actual,difference)) or not 0<=saved<=100 or not 0<=actual<=100 or saved!=entry['marker'].get('metric') or
                not math.isclose(difference,abs(actual-saved),abs_tol=1e-12) or difference>tolerance or
                not math.isclose(float(row['group_macro_score_tolerance']),tolerance,abs_tol=1e-12)):
                raise ValueError('Checkpoint group macro replay does not match actual managed selection metric')
        if report.get('validated_tensor_count')!=sum(r['validated_tensor_count'] for r in rows):
            raise ValueError('Checkpoint report tensor total differs')
        return {'status':'completed','mode':'full_gpu_replay','report':str(path),'report_sha256':sha(path),
            'resolved_config_sha256':identity['config_sha256'],'validated_tensor_count':sum(r['validated_tensor_count'] for r in rows),
            'selection_metric_replayed':True,'selection_protocol':PROTOCOL_NAME,
            'selection_source_file_sha256':identity['config']['selection_manifest_sha256'],
            'selected_manifest_sha256':identity['selection_binding']['manifest_sha256'],
            'retained_checkpoint_objects':len(rows),'internal_selection_records_replayed':sum(r['validation_records'] for r in rows),
            'checkpoint_artifacts':{str(e['path']):e['artifact_identity'] for e in identity['entries']},
            'validator_source_sha256':sha(CODE_ROOT/'scripts/validate_research_checkpoint.py'),'verified':timestamp()}

    def _validate_completed_checkpoints(self,job,record):
        record.update(status='checkpoint_validating',failure_phase='checkpoint_validation')
        previous=copy.deepcopy(record.get('checkpoint_validation',{}))
        record['checkpoint_validation']={'status':'preparing','mode':'full_gpu_replay','report':str(output_dir(job['argv'])/'checkpoint_roundtrip.json')}
        self.persist()
        try:
            self.budget.ensure_space(64*2**20);identity=self._checkpoint_validation_identity(job)
            if self.interrupted_by:raise InterruptedError('Interrupted before checkpoint validator launch')
            selected=next(e for e in identity['entries'] if 'best' in e['labels'])
            if selected['artifact_identity']['sha256']!=record['evaluation']['checkpoint_sha256']:
                raise ValueError('Checkpoint gate selected BEST differs from verified ordinary evaluation')
            reuse=True
            if previous.get('status')=='completed':
                previous_artifacts=previous.get('checkpoint_artifacts',{})
                reuse=(previous.get('resolved_config_sha256')==identity['config_sha256'] and
                    set(previous_artifacts)=={str(e['path']) for e in identity['entries']} and
                    all(same_checkpoint_identity(previous_artifacts[str(e['path'])],e['artifact_identity']) for e in identity['entries']))
            pid=previous.get('pid')
            if isinstance(pid,int) and Path(f'/proc/{pid}/cmdline').is_file():
                commandline=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if 'validate_research_checkpoint.py' in commandline and str(identity['run']) in commandline:
                    raise RuntimeError(f'Checkpoint validator PID {pid} is still live; refusing duplicate worker')
            if reuse:
                try:verified=self._validate_checkpoint_report(identity)
                except (OSError,ValueError,KeyError,TypeError):pass
                else:
                    record['checkpoint_validation']={**previous,**verified};record['failure_phase']=None;self.persist();return True
            command=[sys.executable,'-u',str(ROOT/'scripts/validate_research_checkpoint.py'),'--run',str(identity['run']),'--device','cuda']
            validation={'status':'running','mode':'full_gpu_replay','command':command,
                'log':str(self.log_dir/f"{job['id']}.checkpoint-validation.log"),'started':timestamp(),'report':str(identity['run']/'checkpoint_roundtrip.json')}
            record['checkpoint_validation']=validation;self.persist();emit('checkpoint_validation_started',job=job['id'],command=command)
            with Path(validation['log']).open('a',buffering=1) as handle:
                handle.write(json.dumps({'time':timestamp(),'command':command})+'\n')
                if self.interrupted_by:raise InterruptedError('Interrupted before checkpoint validator spawn')
                started=time.monotonic();self.process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                if self.interrupted_by:
                    try:os.killpg(self.process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                validation['pid']=self.process.pid;self.persist();heartbeat=time.monotonic();interrupted_at=None
                while self.process.poll() is None:
                    if self.interrupted_by:
                        interrupted_at=interrupted_at or time.monotonic()
                        if time.monotonic()-interrupted_at>30:os.killpg(self.process.pid,signal.SIGKILL)
                    if time.monotonic()-heartbeat>=30:
                        self.budget.ensure_space(0);self.persist();emit('checkpoint_validation_running',job=job['id'],pid=self.process.pid,free_gib=self.budget.report()['free_gib'])
                        heartbeat=time.monotonic()
                    time.sleep(1)
                validation.update(returncode=self.process.returncode,finished=timestamp(),wall_seconds=time.monotonic()-started);self.process=None
            if self.interrupted_by:raise InterruptedError('Interrupted during checkpoint roundtrip validation')
            if validation['returncode']!=0:raise RuntimeError(f"Checkpoint validator returned {validation['returncode']}; inspect {validation['log']}")
            final=self._checkpoint_validation_identity(job)
            if (final['config_sha256']!=identity['config_sha256'] or
                [str(e['path']) for e in final['entries']]!=[str(e['path']) for e in identity['entries']] or
                any(not same_checkpoint_identity(a['artifact_identity'],b['artifact_identity']) for a,b in zip(final['entries'],identity['entries']))):
                raise ValueError('Retained checkpoint or configuration changed during full validation')
            validation.update(self._validate_checkpoint_report(final));record['failure_phase']=None;self.persist();return True
        except Exception as error:
            self._stop_owned_child();interrupted=bool(self.interrupted_by)
            record['failure_phase']='checkpoint_validation'
            record.setdefault('checkpoint_validation',{}).update(status='interrupted' if interrupted else 'failed',error_type=type(error).__name__,error=str(error),finished=timestamp())
            self.persist();emit('checkpoint_validation_failed',job=job['id'],status=record['status'],error=str(error))
            if interrupted:return False
            raise

    def _grounding_identity(self,job,variant):
        identity=self._evaluation_identity(job)
        identity['output']=output_dir(job['argv'])/'evaluation/grounding_v1'/variant
        transformed,descriptor=transform_image_ablation(identity['records'],variant=variant,seed=42,
            output_dir=identity['output'],source_manifest_sha256=identity['manifest_sha256'],
            reserve_bytes=int(self.spec.get('reserve_gb',120)*2**30))
        write_transformed_manifest(transformed,identity['output']/'transformed_manifest.jsonl',
            reserve_bytes=int(self.spec.get('reserve_gb',120)*2**30))
        identity.update(transformed_records=transformed,image_ablation=descriptor)
        return identity

    def _validate_grounding(self,job,identity,variant):
        destination=identity['output'];summary=json.loads((destination/'summary.json').read_text())
        protocol=json.loads((destination/'protocol.json').read_text());descriptor=identity['image_ablation']
        if len(identity['records'])!=REQUIRED_RECORDS:raise ValueError('Grounding requires the complete512-record original manifest')
        for report in (summary,protocol):
            if report.get('image_ablation')!=descriptor:raise ValueError('Grounding variant/seed/transformation provenance differs')
            for field in ('checkpoint_sha256','manifest_sha256'):
                if report.get(field)!=identity[field]:raise ValueError('Grounding checkpoint/original manifest identity differs')
            if identity['checkpoint_artifact_identity']['format']=='bundle_v1' and not same_checkpoint_identity(identity['checkpoint_artifact_identity'],report.get('checkpoint_bundle_identity',{})):
                raise ValueError('Grounding does not bind every checkpoint bundle component')
        if descriptor['variant']!=variant or descriptor['seed']!=42 or descriptor['source_manifest_sha256']!=identity['manifest_sha256']:
            raise ValueError('Grounding fixed variant protocol differs')
        if sha(destination/'transformed_manifest.jsonl')!=descriptor['transformed_manifest_sha256']:
            raise ValueError('Grounding transformed manifest was corrupted')
        if read_rows(destination/'transformed_manifest.jsonl')!=identity['transformed_records']:
            raise ValueError('Grounding transformed records differ from actual image intervention')
        generation=protocol.get('generation',{})
        if (generation.get('max_new_tokens')!=64 or generation.get('do_sample') is not False or
            generation.get('use_cache') is not True or generation.get('num_beams')!=1):
            raise ValueError('Grounding generation must be greedy64-token cached generation')
        expected_ids=hashlib.sha256(json.dumps([r['record_id'] for r in identity['transformed_records']]).encode()).hexdigest()
        if protocol.get('manifest_records')!=REQUIRED_RECORDS or protocol.get('selected_record_ids_sha256')!=expected_ids:
            raise ValueError('Grounding protocol selects an incomplete or different record sequence')
        source_hashes=protocol.get('source_sha256',{})
        if not all(path in source_hashes for path in REQUIRED_SOURCE_FILES):raise ValueError('Grounding protocol has incomplete source provenance')
        for relative,digest in source_hashes.items():
            path=(CODE_ROOT/relative).resolve()
            if not path.is_relative_to(CODE_ROOT) or not path.is_file() or sha(path)!=digest:
                raise ValueError('Grounding evaluator source code changed')
        protocol_sha=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        if summary.get('protocol_sha256')!=protocol_sha:raise ValueError('Grounding protocol digest mismatch')
        rows=read_rows(destination/'predictions.jsonl');expected={r['record_id']:r for r in identity['transformed_records']}
        if len(rows)!=REQUIRED_RECORDS or len({r['record_id'] for r in rows})!=REQUIRED_RECORDS or {r['record_id'] for r in rows}!=set(expected):
            raise ValueError('Grounding raw predictions do not cover exactly all512 original records')
        sentinel=object()
        for row in rows:
            record=expected[row['record_id']]
            if row.get('protocol_sha256')!=protocol_sha:raise ValueError('Grounding prediction protocol differs')
            if any(row.get(field,sentinel)!=record.get(field,sentinel) for field in ROW_FIELDS):
                raise ValueError('Grounding prediction metadata/reference/source/image provenance differs')
            tokens=row.get('generated_token_ids')
            if (not isinstance(row.get('prediction'),str) or not isinstance(tokens,list) or len(tokens)>64 or
                any(not isinstance(x,int) or isinstance(x,bool) for x in tokens) or row.get('generated_tokens')!=len(tokens)):
                raise ValueError('Grounding generated token metadata differs')
            actual=score_prediction(record,row['prediction'])
            if not math.isfinite(float(row['score'])) or not math.isclose(float(row['score']),actual,abs_tol=1e-12):
                raise ValueError('Grounding raw score differs from actual prediction/references')
        computed=aggregate_scores(rows,require_four_groups=True)
        if (summary.get('complete_manifest') is not True or summary.get('manifest_records')!=REQUIRED_RECORDS or
            summary.get('records')!=REQUIRED_RECORDS or summary.get('complete_four_group_coverage') is not True or
            any(summary.get(field)!=computed[field] for field in ('groups','benchmarks','group_macro_score','scale','aggregation'))):
            raise ValueError('Grounding summary differs from recomputed four-group raw metrics')
        return {'status':'completed','output':str(destination),'summary':str(destination/'summary.json'),
            'records':REQUIRED_RECORDS,'checkpoint_sha256':identity['checkpoint_sha256'],
            'checkpoint_artifact_identity':identity['checkpoint_artifact_identity'],
            'manifest_sha256':identity['manifest_sha256'],'transformed_manifest_sha256':descriptor['transformed_manifest_sha256'],
            'protocol_sha256':protocol_sha,'variant':variant,'seed':42,'group_macro_score':computed['group_macro_score'],
            'controller_source_sha256':sha(__file__),'evaluator_source_sha256':source_hashes,'verified':timestamp()}

    def _stop_owned_child(self):
        if self.process is None:return
        if self.process.poll() is None:
            try:os.killpg(self.process.pid,signal.SIGTERM);self.process.wait(timeout=10)
            except (ProcessLookupError,subprocess.TimeoutExpired):
                if self.process.poll() is None:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=10)
        self.process=None

    def _update_grounding_eta(self,record):
        receipts=record.get('grounding_evaluations',{})
        pending=sum(receipts.get(mode,{}).get('status')!='completed' for mode in MODES)
        measured=[row['wall_seconds'] for row in receipts.values() if row.get('status')=='completed' and row.get('wall_seconds',0)>0]
        if not measured:
            ordinary=record.get('evaluation',{})
            if ordinary.get('started') and ordinary.get('finished'):
                measured=[max(0,(datetime.fromisoformat(ordinary['finished'])-datetime.fromisoformat(ordinary['started'])).total_seconds())]
        record['grounding_remaining_evaluation_passes']=pending
        record['grounding_eta_seconds']=pending*sum(measured)/len(measured) if measured else None
        record['grounding_eta_basis']='remaining complete512-record passes times observed evaluation walltime; model-loading/intervention costs can differ'

    def _evaluate_variant(self,job,record,variant):
        record['active_grounding_variant']=variant
        self.budget.ensure_space(64*2**20)
        identity=self._grounding_identity(job,variant)
        if self.interrupted_by:raise InterruptedError('Queue interrupted while preparing grounding inputs')
        if identity['checkpoint_sha256']!=record['evaluation']['checkpoint_sha256']:
            raise ValueError('Grounding must use the same checkpoint as verified ordinary evaluation')
        receipt=record.setdefault('grounding_evaluations',{}).get(variant,{})
        if receipt.get('status')=='completed':
            receipt.update(self._validate_grounding(job,identity,variant));self._update_grounding_eta(record);self.persist();return
        # A controller crash may leave its worker alive. Never launch a second
        # GPU evaluation over an already-owned live output.
        pid=receipt.get('pid')
        if isinstance(pid,int) and Path(f'/proc/{pid}/cmdline').is_file():
            commandline=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if 'eval_research_checkpoint.py' in commandline and str(identity['output']) in commandline:
                raise RuntimeError(f'Grounding evaluator PID {pid} is still live; no duplicate worker launched')
        command=[sys.executable,'-u',str(ROOT/'scripts/eval_research_checkpoint.py'),'--run',str(output_dir(job['argv'])),
            '--manifest',str(identity['manifest']),'--output',str(identity['output']),'--checkpoint','best',
            '--max-new-tokens','64','--image-ablation',variant,'--ablation-seed','42',
            '--reserve-gb',str(self.spec.get('reserve_gb',120))]
        if (identity['output']/'protocol.json').is_file():command.append('--resume')
        receipt={'status':'running','started':timestamp(),'command':command,'output':str(identity['output']),
            'log':str(self.log_dir/f"{job['id']}.grounding-{variant}.log")}
        record['grounding_evaluations'][variant]=receipt;record['active_grounding_variant']=variant
        self._update_grounding_eta(record);self.persist()
        emit('grounding_started',job=job['id'],variant=variant,log=receipt['log'])
        with Path(receipt['log']).open('a',buffering=1) as handle:
            handle.write(json.dumps({'time':timestamp(),'command':command})+'\n')
            if self.interrupted_by:raise InterruptedError('Queue interrupted before grounding worker launch')
            wall_start=time.monotonic()
            self.process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            if self.interrupted_by:
                try:os.killpg(self.process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            receipt['pid']=self.process.pid;self.persist();heartbeat=time.monotonic();interrupted_at=None
            while self.process.poll() is None:
                if self.interrupted_by:
                    interrupted_at=interrupted_at or time.monotonic()
                    if time.monotonic()-interrupted_at>30:os.killpg(self.process.pid,signal.SIGKILL)
                if time.monotonic()-heartbeat>=30:
                    self.budget.ensure_space(0);self.persist()
                    emit('grounding_running',job=job['id'],variant=variant,pid=self.process.pid,free_gib=self.budget.report()['free_gib'])
                    heartbeat=time.monotonic()
                time.sleep(1)
            receipt.update(returncode=self.process.returncode,finished=timestamp(),wall_seconds=time.monotonic()-wall_start);self.process=None
        if self.interrupted_by:raise InterruptedError('Queue interrupted during grounding evaluation')
        if receipt['returncode']!=0:raise RuntimeError(f"Grounding evaluator returned {receipt['returncode']}; inspect {receipt['log']}; no implicit retry")
        final=self._grounding_identity(job,variant)
        if (final['manifest_sha256']!=identity['manifest_sha256'] or
            not same_checkpoint_identity(final['checkpoint_artifact_identity'],identity['checkpoint_artifact_identity'])):
            raise ValueError('Grounding checkpoint or original manifest changed during evaluation')
        receipt.update(self._validate_grounding(job,final,variant));self._update_grounding_eta(record);self.persist()
        emit('grounding_completed',job=job['id'],variant=variant,receipt=receipt)

    def evaluate_completed_job(self,job,record):
        is_sft=argv_value(job['argv'],'--stage')=='SFT'
        if is_sft and record.get('checkpoint_validation',{}).get('status')=='failed' and not self.retry_failed:
            raise RuntimeError('Checkpoint validation previously failed; inspect logs and use --retry-failed')
        if is_sft and record.get('grounding_status')=='failed' and not self.retry_failed:
            raise RuntimeError('Grounding previously failed; inspect logs and use--retry-failed after fixing it')
        record['failure_phase']='standard_evaluation'
        # The parent's unmodified none guard is always authoritative first.
        if not super().evaluate_completed_job(job,record):return False
        if not is_sft:return True
        if not self._validate_completed_checkpoints(job,record):return False
        record.update(failure_phase='grounding',grounding_status='running',status='grounding_evaluating')
        self.persist()
        try:
            for variant in MODES:
                if self.interrupted_by:raise InterruptedError('Queue interrupted before grounding variant')
                self._evaluate_variant(job,record,variant)
            original=read_rows(resolve_path(job['evaluation']['manifest']))
            baseline=read_rows(resolve_path(job['evaluation']['output'])/'predictions.jsonl')
            variants={mode:read_rows(output_dir(job['argv'])/'evaluation/grounding_v1'/mode/'predictions.jsonl') for mode in MODES}
            summary={'schema_version':1,'status':'completed','scope':'paired held-out subset grounding diagnostics; ordinary scores remain separate',
                'ordinary_evaluation':record['evaluation'],'checkpoint_validation':record['checkpoint_validation'],
                'grounding_evaluations':record['grounding_evaluations'],
                'paired_image_group_bootstrap':paired_image_bootstrap(baseline,variants,original),
                'additional_evaluation_passes_per_SFT':3,'controller_source_sha256':sha(__file__)}
            destination=output_dir(job['argv'])/'evaluation/grounding_v1/paired_summary.json';_atomic_json(destination,summary)
            record.update(grounding_status='completed',grounding_summary=str(destination),failure_phase=None)
            self.persist();return True
        except Exception as error:
            self._stop_owned_child();interrupted=bool(self.interrupted_by)
            record.update(grounding_status='interrupted' if interrupted else 'failed',failure_phase='grounding',
                error_type=type(error).__name__,error=str(error),finished=timestamp())
            mode=record.get('active_grounding_variant')
            if mode:record.setdefault('grounding_evaluations',{}).setdefault(mode,{}).update(status=record['grounding_status'],error=str(error))
            self.persist();emit('grounding_failed',job=job['id'],status=record['status'],error=str(error))
            if interrupted:return False
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--spec',required=True)
    parser.add_argument('--state');parser.add_argument('--retry-failed',action='store_true');args=parser.parse_args(argv)
    queue=GroundingResearchQueue(args.spec,args.state,args.retry_failed)
    signal.signal(signal.SIGTERM,queue.signal);signal.signal(signal.SIGINT,queue.signal)
    return queue.run()

if __name__=='__main__':raise SystemExit(main())
