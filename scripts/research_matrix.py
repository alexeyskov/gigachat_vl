#!/usr/bin/env python3
"""Read-only evidence audit for all 396 required primary research trajectories.

Writes only its own matrix.json/csv/md reports. It neither schedules runs nor
changes configs. Smoke/pilot artifacts never count as primary scientific results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
MODELS = [
    ('smallthinker4', 'Tiiny/SmallThinker-4BA0.6B-Instruct', 4_000_000_000),
    ('lfm25', 'LiquidAI/LFM2.5-8B-A1B', 8_000_000_000),
    ('giga31', 'ai-sage/GigaChat3.1-10B-A1.8B-bf16', 10_000_000_000),
    ('gptoss20', 'openai/gpt-oss-20b', 20_000_000_000),
]
ENCODERS = [
    ('siglip2', 'google/siglip2-so400m-patch14-384'),
    ('qwen25vision', 'Qwen/Qwen2.5-VL-7B-Instruct'),
    ('gemma4vision', 'google/gemma-4-E4B-it'),
]
RECIPES = ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '3c', '4q', '4f']
SEEDS = [42, 123, 2026]
TASK_GROUPS = {'general_vqa', 'ocr_documents', 'charts_diagrams', 'knowledge_reasoning'}
REQUIREMENTS = {'alignment_pairs': 200_000, 'a1_pairs': 10_000, 'sft_multimodal_records': 500_000, 'text_record_fraction': .15}
GIB = 1024 ** 3


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def path_at(root, value):
    path = Path(str(value)).expanduser()
    return (root / path).absolute() if not path.is_absolute() else path.absolute()


def tail_events(path, byte_limit=65536):
    try:
        with Path(path).open('rb') as handle:
            size = handle.seek(0, 2)
            handle.seek(max(0, size-byte_limit))
            lines = handle.read().decode('utf-8', errors='replace').splitlines()
        events = []
        for line in lines:
            try:
                row = json.loads(line)
                if isinstance(row, dict): events.append(row)
            except ValueError: pass
        return events
    except OSError:
        return []


def manifest_statistics(path, cache):
    path = Path(path)
    key = str(path)
    if key in cache: return cache[key]
    digest = hashlib.sha256()
    total = multimodal = malformed = 0
    try:
        with path.open('rb') as handle:
            for raw in handle:
                digest.update(raw)
                if not raw.strip(): continue
                try:
                    row = json.loads(raw)
                    total += 1
                    multimodal += int(bool(row.get('image_paths') or row.get('images') or row.get('image')))
                except (ValueError, AttributeError): malformed += 1
        value = {'records':total, 'multimodal_records':multimodal, 'text_records':total-multimodal,
                 'sha256':digest.hexdigest(), 'malformed_records':malformed, 'path':key}
    except OSError:
        value = {'records':0, 'multimodal_records':0, 'text_records':0, 'sha256':None,
                 'malformed_records':0, 'path':key, 'missing':True}
    cache[key] = value
    return value


def parse_job_argv(argv):
    config = {'model': MODELS[0][1], 'vision': ENCODERS[0][1], 'recipe':'4f', 'seed':42,
              'stage':None, 'epochs':1, 'text_fraction':.15}
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith('--'):
            i += 1; continue
        if '=' in token:
            name, value = token[2:].split('=', 1)
        elif i+1 < len(argv) and not argv[i+1].startswith('--'):
            name, value = token[2:], argv[i+1]; i += 1
        else:
            name, value = token[2:], True
        if isinstance(value, str):
            try: value = int(value)
            except ValueError:
                try: value = float(value)
                except ValueError: pass
        config[name.replace('-', '_')] = value
        i += 1
    return config


def scope_of(config, directory):
    text = ' '.join(str(config.get(k,'')) for k in ('scope','profile','data_profile','manifest')) + ' ' + directory.name
    text = text.lower()
    if 'smoke' in text: return 'smoke'
    if 'pilot' in text: return 'pilot'
    if any(word in text for word in ('fastbench','batch32','benchmark_only')): return 'technical'
    if 'primary' in text: return 'primary'
    return 'unclassified'


def technical_weights_retired(directory, config):
    """Recognize explicit disposable-smoke cleanup, without claiming retained weights."""
    if scope_of(config,directory) != 'smoke':return False
    scientific_scope=' '.join(str(config.get(key,'')) for key in ('scope','profile','data_profile','manifest'))+' '+str(directory)
    if any(word in scientific_scope.lower() for word in ('pilot','primary')):return False
    for field,limit in (('max_records',2000),('max_steps',100)):
        value=config.get(field)
        if not isinstance(value,int) or isinstance(value,bool) or not 0<value<=limit:return False
    receipt=read_json(directory/'technical_smoke_pruned.json',{})
    manager=read_json(directory/'.checkpoint_manager.json',{})
    if (receipt.get('status')!='pruned' or receipt.get('owner')!=manager.get('owner') or
            receipt.get('run_dir')!=str(directory) or receipt.get('scope')!='disposable_technical_smoke_weights_only'):
        return False
    for filename,key in (('resolved_config.json','config_sha256'),('checkpoint_roundtrip.json','validation_receipt_sha256')):
        try:digest=hashlib.sha256((directory/filename).read_bytes()).hexdigest()
        except OSError:return False
        if receipt.get(key)!=digest:return False
    return True


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0: return False
    try:
        os.kill(pid, 0)
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
        return b'run_research_stage.py' in cmdline
    except (OSError, ValueError): return False


def checkpoint_evidence(directory):
    """Validate local ownership, pointers and current object metadata, no torch load."""
    manager = read_json(directory/'.checkpoint_manager.json', {})
    result = {'retained':False, 'checkpoints':{}, 'optimizer_resumable':None}
    if not manager.get('owner'): return result
    objects = directory/'.checkpoint_objects'
    if objects.is_symlink() or not objects.is_dir(): return result
    for name in ('best','last'):
        pointer = directory/name
        if not pointer.is_symlink(): continue
        target = pointer.resolve()
        metadata = read_json(target/'.research_checkpoint.json', {})
        payload = target/'state.pt'
        if target.parent != objects or metadata.get('owner') != manager['owner'] or not payload.is_file() or payload.is_symlink(): continue
        result['checkpoints'][name] = {'path':str(payload), 'step':metadata.get('step'),
                                      'owner':manager['owner'], 'metric':metadata.get('metric'),
                                      'bytes':payload.stat().st_size, 'mtime_ns':payload.stat().st_mtime_ns,
                                      'optimizer_resumable':metadata.get('resumable_optimizer', True)}
        from src.utils.research_checkpoint_compat import checkpoint_kind,checkpoint_identity
        try:
            if checkpoint_kind(payload)=='bundle_v1':
                identity=checkpoint_identity(payload,verify_components=False)
                from src.utils.research_checkpoint_bundle import inspect_checkpoint_bundle
                resumable=inspect_checkpoint_bundle(payload,verify_hashes=False)['fully_resumable']
                result['checkpoints'][name].update(bundle_identity=identity,sha256=identity['sha256'],artifact_total_bytes=identity['total_bytes'],optimizer_resumable=resumable)
        except (ValueError,OSError) as error:
            # A broken artifact is evidence of failure, rather than a reason to
            # abort the entire 396-trajectory status report.
            result['checkpoints'].pop(name,None)
            result.setdefault('errors',[]).append({'role':name,'error':str(error)})
    result['retained'] = 'last' in result['checkpoints']
    if result['retained']: result['optimizer_resumable'] = result['checkpoints']['last']['optimizer_resumable']
    result['unique_retained_bytes'] = sum(row.get('artifact_total_bytes',row['bytes']) for row in {v['path']:v for v in result['checkpoints'].values()}.values())
    return result


def roundtrip_passed(directory, checkpoint):
    report = read_json(directory/'checkpoint_roundtrip.json', {})
    if report.get('status') != 'passed': return False
    for row in report.get('checkpoints', [report]):
        if (isinstance(row, dict) and row.get('status') == 'passed' and row.get('checkpoint_path') == checkpoint['path']
            and row.get('checkpoint_owner') == checkpoint['owner'] and row.get('checkpoint_step') == checkpoint['step']
            and row.get('checkpoint_size_bytes') == checkpoint['bytes'] and row.get('checkpoint_mtime_ns') == checkpoint['mtime_ns']
            and row.get('all_tensors_finite') is True and row.get('expected_keys_match') is True
            and isinstance(row.get('validated_tensor_count'), int) and row['validated_tensor_count'] > 0):
            if checkpoint.get('bundle_identity'):
                from src.utils.research_checkpoint_compat import same_checkpoint_identity
                if row.get('checkpoint_sha256')!=checkpoint['sha256'] or not same_checkpoint_identity(checkpoint['bundle_identity'],row.get('checkpoint_bundle_identity',{})):
                    continue
            return True
    return False


def benchmark_evidence(directory, checkpoint, manifest_sha):
    """Accept only checkpoint-bound official benchmark evidence, never heldout CE."""
    for relative in ('benchmark_results.json', 'evaluation/benchmark_results.json', 'metrics/benchmark_results.json'):
        path = directory/relative
        report = read_json(path, {})
        ablation = report.get('image_ablation', {})
        if not isinstance(ablation, dict) or ablation.get('variant', 'none') != 'none':
            continue
        if (report.get('status') != 'completed' or report.get('evaluation_kind') != 'official_benchmark'
            or report.get('checkpoint_path') != checkpoint['path'] or report.get('manifest_sha256') != manifest_sha
            or report.get('checkpoint_size_bytes') != checkpoint['bytes'] or report.get('checkpoint_mtime_ns') != checkpoint['mtime_ns']
            or not report.get('evaluation_manifest_sha256') or not report.get('harness_revision')
            or report.get('train_eval_overlap_checked') is not True
            or not TASK_GROUPS.issubset(set(report.get('task_groups', [])))):
            continue
        if checkpoint.get('bundle_identity'):
            from src.utils.research_checkpoint_compat import same_checkpoint_identity
            if report.get('checkpoint_sha256')!=checkpoint['sha256'] or not same_checkpoint_identity(checkpoint['bundle_identity'],report.get('checkpoint_bundle_identity',{})):
                continue
        metrics = report.get('metrics')
        if not isinstance(metrics, dict) or not metrics or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in metrics.values()): continue
        group_scores = report.get('task_group_scores', {})
        if not isinstance(group_scores,dict) or not TASK_GROUPS.issubset(group_scores) or not all(isinstance(group_scores[group],(int,float)) and not isinstance(group_scores[group],bool) and math.isfinite(group_scores[group]) for group in TASK_GROUPS): continue
        return {'valid':True, 'path':str(path), 'metrics':metrics, 'evaluation_manifest_sha256':report['evaluation_manifest_sha256'],
                'harness_revision':report['harness_revision']}
    return {'valid':False}


def config_ancestor(root, checkpoint_value):
    if not checkpoint_value: return None
    path = path_at(root, checkpoint_value)
    for candidate in [path, *list(path.parents)[:6]]:
        if (candidate/'resolved_config.json').is_file(): return candidate
    resolved = path.resolve()
    for candidate in [resolved, *list(resolved.parents)[:6]]:
        if (candidate/'resolved_config.json').is_file(): return candidate
    return None


def run_revisions(directory, config, inventory):
    provenance = read_json(directory/'provenance.json', {})
    model_ref = provenance.get('model_ref', {})
    vision_ref = provenance.get('vision_ref', {})
    attested_model = config.get('model_revision') or provenance.get('model_revision') or model_ref.get('revision')
    attested_vision = config.get('encoder_revision') or config.get('vision_revision') or provenance.get('encoder_revision') or vision_ref.get('revision')
    return {'model_revision':attested_model or inventory.get(config.get('model'),{}).get('sha'),
            'encoder_revision':attested_vision or inventory.get(config.get('vision'),{}).get('sha'),
            'model_revision_source':'run_attestation' if attested_model else 'current_inventory_not_run_attestation',
            'encoder_revision_source':'run_attestation' if attested_vision else 'current_inventory_not_run_attestation'}


def primary_budget_evidence(root, directory, config, cache):
    reasons = []
    lineage = []
    current = directory
    for stage in ('SFT','A1','A0'):
        if current is None:
            reasons.append(f'missing_{stage}_lineage'); break
        cfg = read_json(current/'resolved_config.json', {})
        if cfg.get('stage') != stage:
            reasons.append(f'lineage_stage_expected_{stage}'); break
        if any(cfg.get(key) != config.get(key) for key in ('model','vision','seed')):
            reasons.append(f'lineage_model_vision_seed_mismatch_{stage}')
        if any(run_revisions(current,cfg,{})[key] != run_revisions(directory,config,{})[key] for key in ('model_revision','encoder_revision')):
            reasons.append(f'lineage_revision_mismatch_{stage}')
        done = read_json(current/'completed.json', {})
        stats = manifest_statistics(path_at(root,cfg.get('manifest','missing')), cache)
        if cfg.get('manifest_sha256') != stats['sha256']:
            reasons.append(f'manifest_hash_mismatch_{stage}')
        effective = min(stats['records'], cfg.get('max_records') or stats['records'])
        if stage == 'SFT':
            if cfg.get('selection_metric') != 'group_macro':
                reasons.append('SFT_checkpoint_selection_not_internal_group_macro')
            if read_json(current/'.checkpoint_manager.json',{}).get('mode') != 'max':
                reasons.append('SFT_checkpoint_selection_direction_not_max')
            selection_input = path_at(root,cfg.get('selection_manifest') or cfg.get('validation','missing'))
            selection_stats = manifest_statistics(selection_input,cache)
            if (not selection_stats.get('sha256') or
                    cfg.get('selection_manifest_sha256') != selection_stats['sha256']):
                reasons.append('SFT_selection_manifest_not_bound')
            protocol = read_json(current/'selection/best.protocol.json',{})
            selected_metrics = read_json(current/'selection/best.metrics.json',{})
            selected_checkpoint = checkpoint_evidence(current)['checkpoints'].get('best',{})
            if (protocol.get('name') != 'internal_teacher_reference_group_macro' or
                    protocol.get('official_benchmark_score') is not False or
                    protocol.get('selection_direction') != 'max' or protocol.get('seed') != cfg.get('seed') or
                    not protocol.get('selection_manifest_sha256') or
                    protocol.get('validation_step') != selected_checkpoint.get('step') or
                    selected_metrics.get('validation_step') != selected_checkpoint.get('step') or
                    not TASK_GROUPS.issubset(selected_metrics.get('groups',{}))):
                reasons.append('SFT_internal_selection_artifacts_missing_or_stale')
            # A random max-records cap cannot prove preservation of the 500K MM
            # budget, even when an uncapped source manifest was larger.
            if cfg.get('max_records') and cfg['max_records'] < stats['records']:
                reasons.append('SFT_manifest_capped')
            if stats['multimodal_records'] < REQUIREMENTS['sft_multimodal_records']:
                reasons.append('SFT_multimodal_budget_below_500K')
            mm, text = stats['multimodal_records'], stats['text_records']
            if cfg.get('text_manifest'):
                replay = manifest_statistics(path_at(root,cfg['text_manifest']), cache)
                requested = round(effective*cfg.get('text_fraction',.15)/(1-cfg.get('text_fraction',.15)))
                if replay['text_records'] < requested: reasons.append('insufficient_text_replay')
                text += min(requested,replay['text_records'])
            fraction = text/max(1,mm+text)
            if abs(fraction-REQUIREMENTS['text_record_fraction']) > max(1e-5,1/max(1,mm+text)):
                reasons.append('text_replay_not_primary_15_percent')
            minimum = mm+text
        else:
            minimum = REQUIREMENTS['alignment_pairs' if stage == 'A0' else 'a1_pairs']
            if effective < minimum: reasons.append(f'{stage}_data_budget_below_{minimum}')
        if done.get('records',0) < minimum: reasons.append(f'{stage}_training_budget_not_completed')
        lineage.append({'stage':stage,'directory':str(current),'manifest_sha256':stats['sha256'],
                        'manifest_records':stats['records'],'effective_source_records':effective,'completed_records':done.get('records',0)})
        current = config_ancestor(root, cfg.get('initial_checkpoint') or cfg.get('alignment_checkpoint'))
    revisions = run_revisions(directory,config,{})
    for kind in ('model','encoder'):
        revision = revisions[kind+'_revision']
        if not isinstance(revision,str) or len(revision) not in (40,64) or any(c not in '0123456789abcdef' for c in revision.lower()):
            reasons.append('run_'+kind+'_revision_not_pinned')
    audit_path = path_at(root, config['scientific_data_audit']) if config.get('scientific_data_audit') else directory/'scientific_data_audit.json'
    audit = read_json(audit_path,{})
    if not (audit.get('status') == 'passed' and audit.get('train_eval_overlap_checked') is True
            and audit.get('manifest_sha256') == config.get('manifest_sha256') and audit.get('source_revisions_pinned') is True):
        reasons.append('scientific_data_audit_missing_or_pending')
    return {'valid':not reasons, 'reasons':reasons, 'lineage':lineage, 'data_audit':str(audit_path)}


def scan_runs(root, inventory):
    pending = {}
    for spec in sorted((root/'configs/research').glob('*.json')):
        if spec.name.endswith('.state.json'): continue
        document = read_json(spec,{})
        state = read_json(spec.with_suffix('.state.json'),{})
        for job in document.get('jobs',[]):
            cfg = parse_job_argv(job.get('argv',[]))
            if not cfg.get('output'): continue
            key = str(path_at(root,cfg['output']).resolve())
            pending[key] = {'config':cfg, 'queue':str(spec), 'queue_status':state.get('jobs',{}).get(job.get('id'),{})}
    for config_path in (root/'runs/research').glob('*/resolved_config.json'):
        key = str(config_path.parent.resolve())
        pending.setdefault(key,{})['config'] = read_json(config_path,{})
    cache = {}
    observed = []
    for directory_name, row in sorted(pending.items()):
        directory = Path(directory_name); cfg = row.get('config',{})
        complete = read_json(directory/'completed.json',{})
        cp = checkpoint_evidence(directory)
        qstate = row.get('queue_status',{})
        events = tail_events(directory/'metrics.jsonl')
        last_event = events[-1] if events else {}
        error = qstate.get('error') or (last_event.get('error') if last_event.get('event')=='failed' else None)
        kind = qstate.get('error_type') or last_event.get('error_type','')
        status, reason = 'planned', 'configured_not_started'
        if complete.get('steps',0)>0:
            if cp['retained'] and cp['checkpoints']['last']['step']==complete['steps']:
                status,reason = 'training_completed','completed_stage_and_retained_owned_checkpoint'
            elif cp['retained']:
                status,reason = 'failed','completion_checkpoint_step_mismatch'
            elif (read_json(directory/'checkpoint_pruned.json',{}).get('status')=='pruned' and cfg.get('stage') in {'A0','A1'}):
                status,reason = 'training_completed','completed_intermediate_weights_pruned_after_consumption'
            elif technical_weights_retired(directory,cfg):
                status,reason = 'training_completed','completed_disposable_technical_smoke_weights_retired_after_verified_roundtrip'
            else:
                status,reason = 'failed','completion_marker_without_retained_checkpoint'
        elif qstate.get('status')=='running' and process_alive(qstate.get('pid')):
            status,reason = 'running','live_training_process_in_queue_state'
        elif error or qstate.get('status')=='failed':
            resource = any(word in f'{kind} {error}'.lower() for word in ('outofmemory','out of memory','insufficientdisk','disk reserve','no space'))
            status,reason = ('resource_blocked' if resource else 'failed'), str(error or 'queue_failure')
        elif qstate.get('status')=='running':
            status,reason = 'failed','stale_running_state_without_live_trainer'
        elif last_event.get('event')=='train':
            status,reason = 'planned','partial_progress_without_verified_live_process'
        scope = scope_of(cfg,directory)
        model_id = next((m[0] for m in MODELS if cfg.get('model')==m[1]),None)
        encoder_id = next((v[0] for v in ENCODERS if cfg.get('vision')==v[1]),None)
        record = {'run_id':directory.name,'directory':str(directory),'scope':scope,'stage':cfg.get('stage'),
                  'model_id':model_id,'encoder_id':encoder_id,'model_repo':cfg.get('model'),'encoder_repo':cfg.get('vision'),
                  **run_revisions(directory,cfg,inventory),
                  'recipe':cfg.get('recipe'),'seed':cfg.get('seed'), 'status':status,'status_reason':reason,
                  'manifest':cfg.get('manifest'),'manifest_sha256':cfg.get('manifest_sha256'),
                  'validation_sha256':cfg.get('validation_sha256'),'text_manifest_sha256':cfg.get('text_manifest_sha256'),
                  'config':cfg,'completion':complete,'checkpoint_evidence':cp,'queue':row.get('queue')}
        if cfg.get('stage')=='SFT' and status=='training_completed' and cp['retained']:
            selected = cp['checkpoints'].get('best',cp['checkpoints']['last'])
            record['roundtrip_passed'] = roundtrip_passed(directory, selected)
            record['benchmark'] = benchmark_evidence(directory,selected,cfg.get('manifest_sha256'))
            if scope in {'primary','unclassified'}:
                record['primary_budget'] = primary_budget_evidence(root,directory,cfg,cache)
                if record['primary_budget']['valid']:
                    record['scope'] = 'primary'
                    if record['roundtrip_passed'] and record['benchmark']['valid']:
                        record['status'],record['status_reason'] = 'completed','full_primary_training_checkpoint_roundtrip_and_official_benchmarks_verified'
                    else:
                        record['status_reason'] = 'primary_training_completed_official_evaluation_or_roundtrip_pending'
                else:
                    if scope=='primary': record['status']='failed'
                    record['status_reason'] = 'primary_requirements_unmet: '+','.join(record['primary_budget']['reasons'])
        observed.append(record)
    return observed


def capacity_forecast(root, observed, reserve_gib=120, gpu_vram_gib=97887/1024):
    usage = shutil.disk_usage(root)
    models = {}
    for model_id,repo,nominal in MODELS:
        full_counts = []
        recipe_counts = {}
        for row in observed:
            if row['model_id'] != model_id or row['stage']!='SFT': continue
            params = read_json(Path(row['directory'])/'trainable_parameters.json',{})
            count = sum(v for v in params.values() if isinstance(v,int) and not isinstance(v,bool))
            if count:
                recipe_counts[row['recipe']] = count
                if row['recipe']=='4f': full_counts.append(count)
        base = max(full_counts) if full_counts else nominal
        profiles = {}
        for recipe in RECIPES:
            quantized = recipe.startswith('1') or recipe=='4q'
            if recipe in recipe_counts:
                trainable = recipe_counts[recipe]; source='observed_trainable_parameter_audit'
            else:
                trainable = int(base*((.015 if recipe=='4q' else .075) if quantized else (1 if recipe=='4f' else 1.06)))
                source='unmeasured_nominal_capacity_estimate'
            # 4-byte parameters + 4-byte gradients + two fp32 Adam moments.
            train_state = trainable*16 + (base*.8 if quantized else 0)
            resume_payload = trainable*12
            compact_payload = trainable*4
            worst_active_disk = resume_payload*3
            profiles[recipe] = {'trainable_parameters':trainable,'parameter_count_source':source,
                'fp32_training_state_lower_bound_gib':round(train_state/GIB,2),
                'resume_payload_estimated_gib':round(resume_payload/GIB,2),
                'compacted_model_payload_estimated_gib':round(compact_payload/GIB,2),
                'worst_best_last_plus_staging_gib':round(worst_active_disk/GIB,2),
                'required_free_with_reserve_gib':round(worst_active_disk/GIB+reserve_gib,2),
                'state_alone_exceeds_current_gpu':train_state/GIB>gpu_vram_gib,
                'worst_active_disk_exceeds_current_free':worst_active_disk+reserve_gib*GIB>usage.free,
                'estimate_excludes':['activations','vision_stack','CUDA_workspace','source_weights_download','images','feature_cache','serialization_metadata'],
                'optimizer_resumable_during_training':True,'optimizer_discard_allowed_only_after_completed_SFT':True}
        models[model_id] = {'repo':repo,'base_parameters':base,'base_count_source':'observed_4f_parameter_audit' if full_counts else 'nominal_model_name_estimate',
                            'profiles':profiles}
    retained = sum(row['checkpoint_evidence'].get('unique_retained_bytes',0) for row in observed)
    minimum_full_final = sum(models[mid]['profiles'][recipe]['compacted_model_payload_estimated_gib']
                             for mid,_,_ in MODELS for recipe in RECIPES)*len(ENCODERS)*len(SEEDS)
    return {'reserve_gib':reserve_gib,'gpu_vram_gib':round(gpu_vram_gib,3),
            'total_disk_gib':round(usage.total/GIB,3),'free_disk_gib':round(usage.free/GIB,3),
            'available_after_reserve_gib':round(max(0,usage.free/GIB-reserve_gib),3),
            'observed_retained_unique_checkpoint_gib':round(retained/GIB,3),
            'all_396_compacted_last_only_estimated_gib':round(minimum_full_final,2),
            'all_396_compacted_best_and_last_upper_estimated_gib':round(2*minimum_full_final,2),
            'all_final_weights_fit_current_disk':minimum_full_final+reserve_gib < usage.total/GIB,
            'caveats':['Forecast is capacity arithmetic, not a measured throughput/fit result.',
                       'Full profiles use fp32 parameters and Adam moments with bf16 compute.',
                       'QLoRA unmeasured 1.5% LoRA / 6% new expert fractions and 0.8 bytes/base-param storage are estimates.',
                       'Packed expert QLoRA/offload support must be validated separately.',
                       'Source checkpoint file bytes are not interpreted as parameter counts for quantized GPT-OSS.'],
            'models':models}


def build_report(root=ROOT, reserve_gib=120, gpu_vram_gib=97887/1024):
    root = Path(root).resolve()
    inventory_rows = read_json(root/'research_results/model_inventory.json',[])
    inventory = {row['repo']:row for row in inventory_rows if isinstance(row,dict) and row.get('repo')}
    observed = scan_runs(root,inventory)
    capacity = capacity_forecast(root,observed,reserve_gib,gpu_vram_gib)
    primary_candidates = defaultdict(list)
    for row in observed:
        if row['stage']=='SFT' and row['scope']=='primary':
            primary_candidates[(row['model_id'],row['encoder_id'],row['recipe'],row['seed'])].append(row)
    primary = []
    rank = {'completed':6,'training_completed':5,'running':4,'resource_blocked':3,'failed':2,'planned':1}
    for (model_id,repo,_),(encoder_id,vision),recipe,seed in itertools.product(MODELS,ENCODERS,RECIPES,SEEDS):
        profile = capacity['models'][model_id]['profiles'][recipe]
        row = {'trajectory_id':f'{model_id}__{encoder_id}__{recipe}__s{seed}__primary-en-v1',
               'model_id':model_id,'model_repo':repo,'model_revision':inventory.get(repo,{}).get('sha'),
               'encoder_id':encoder_id,'encoder_repo':vision,'encoder_revision':inventory.get(vision,{}).get('sha'),
               'recipe':recipe,'seed':seed,'scope':'primary','status':'planned','execution_status':'not_started',
               'status_reason':'no_primary_execution_evidence','manifest_sha256':None,'run_directory':None,
               'roundtrip_passed':False,'official_benchmark_passed':False}
        evidence = primary_candidates.get((model_id,encoder_id,recipe,seed),[])
        if evidence:
            selected = max(evidence,key=lambda e:rank.get(e['status'],0))
            row.update(status=selected['status'],status_reason=selected['status_reason'],execution_status=selected['status'],
                       manifest_sha256=selected['manifest_sha256'],run_directory=selected['directory'],
                       roundtrip_passed=selected.get('roundtrip_passed',False),official_benchmark_passed=selected.get('benchmark',{}).get('valid',False))
        elif profile['state_alone_exceeds_current_gpu']:
            row.update(status='resource_blocked',status_reason='forecast_fp32_training_state_exceeds_current_single_gpu_without_validated_offload; not_attempted')
        row['resource_forecast_profile'] = f'{model_id}/{recipe}'
        primary.append(row)
    return {'schema_version':1,'generated_at':datetime.now(timezone.utc).isoformat(),'root':str(root),
            'plan_file':str(root/'RESEARCH_PIPELINE.md'),'requirements':REQUIREMENTS,
            'required_trajectories':396,'required_per_seed':132,'model_execution_order':[m[0] for m in MODELS],
            'encoders':[v[0] for v in ENCODERS],'recipes':RECIPES,'seeds':SEEDS,
            'primary_status_counts':dict(Counter(row['status'] for row in primary)),
            'observed_stage_status_counts':{scope:dict(Counter(row['status'] for row in observed if row['scope']==scope))
                                            for scope in sorted({row['scope'] for row in observed})},
            'primary':primary,'observed_stages':observed,'capacity':capacity,
            'completion_contract':{'training_completed_is_scientific_completed':False,
                'required':['primary A0/A1/SFT budgets and immutable lineage','retained owned SFT checkpoint',
                            'current exact-object full roundtrip receipt','checkpoint-bound official benchmark report for four task groups',
                            'pinned source revisions and train/eval overlap audit'],
                'benchmark_report_schema':'benchmark_results.json: status=completed,evaluation_kind=official_benchmark,checkpoint_path/size/mtime,manifest_sha256,evaluation_manifest_sha256,harness_revision,train_eval_overlap_checked=true,task_groups,finite metrics,finite task_group_scores for all four groups'}}


def markdown_report(report):
    counts = report['primary_status_counts']; cap=report['capacity']
    lines = ['# Research matrix status', '', f"Generated: {report['generated_at']}", '',
             '**Required primary scope: 396 trajectories = 4 models × 3 encoders × 11 recipes × 3 seeds.**',
             'Smoke/pilot training and internal held-out CE never count as completed primary experiments.', '',
             '| Primary status | Trajectories |','| --- | ---: |']
    for status,count in sorted(counts.items()): lines.append(f'| {status} | {count} |')
    lines += ['', '## Coverage by model and encoder', '', '| Model | Encoder | Scientifically completed | Training completed, evaluation pending | Running | Planned | Resource blocked | Failed |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model,_,_ in MODELS:
        for vision,_ in ENCODERS:
            sub=Counter(row['status'] for row in report['primary'] if row['model_id']==model and row['encoder_id']==vision)
            lines.append(f"| {model} | {vision} | {sub['completed']} | {sub['training_completed']} | {sub['running']} | {sub['planned']} | {sub['resource_blocked']} | {sub['failed']} |")
    lines += ['', '## Observed stages outside primary', '', '| Scope | Status counts |', '| --- | --- |']
    for scope,values in report['observed_stage_status_counts'].items(): lines.append(f'| {scope} | {json.dumps(values,sort_keys=True)} |')
    lines += ['', 'Only SFT can complete a trajectory; an A0 or A1 completion is an intermediate stage.', '',
              '## Current capacity forecast', '',
              f"Disk: {cap['free_disk_gib']:.1f} GiB free, {cap['reserve_gib']:.0f} GiB reserved; GPU: {cap['gpu_vram_gib']:.1f} GiB.",
              f"All 396 compacted last-only model checkpoints: approximately {cap['all_396_compacted_last_only_estimated_gib']:.0f} GiB, before source weights/images/feature caches.",
              'Completed SFT compaction removes optimizer state only. During training, best/last remain resumable.', '',
              '| Model | Recipe | FP32 training state minimum, GiB | Resume payload, GiB | Compacted model, GiB | Worst active disk + reserve, GiB |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for model,_,_ in MODELS:
        for recipe in ('1b','4q','3b','4f'):
            p=cap['models'][model]['profiles'][recipe]
            lines.append(f"| {model} | {recipe} | {p['fp32_training_state_lower_bound_gib']} | {p['resume_payload_estimated_gib']} | {p['compacted_model_payload_estimated_gib']} | {p['required_free_with_reserve_gib']} |")
    lines += ['', 'Forecasts exclude activations, donor vision tensors, CUDA workspace, images, cache and downloads. Unmeasured model counts use nominal sizes; QLoRA fractions are estimates. Resource-blocked forecast rows were not executed and can be unblocked by validated offload or more memory.', '',
              '## Recent evidence', '', '| Run | Scope/stage | Status | Reason |', '| --- | --- | --- | --- |']
    for row in report['observed_stages'][-30:]:
        reason=row['status_reason'].replace('|','/')[:180]
        lines.append(f"| {row['run_id']} | {row['scope']}/{row['stage']} | {row['status']} | {reason} |")
    lines += ['', 'Detailed manifest hashes, checkpoint ownership, model/encoder revisions, status reasons and capacity assumptions are in matrix.json; matrix.csv contains every required primary row.', '']
    return '\n'.join(lines)


def write_reports(report, output_dir):
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    (output/'matrix.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (output/'matrix.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(report['primary'][0]))
        writer.writeheader();writer.writerows(report['primary'])
    (output/'matrix.md').write_text(markdown_report(report))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(ROOT));parser.add_argument('--output-dir')
    parser.add_argument('--reserve-gib',type=float,default=120);parser.add_argument('--gpu-vram-gib',type=float,default=97887/1024)
    args=parser.parse_args();root=Path(args.root).resolve()
    report=build_report(root,args.reserve_gib,args.gpu_vram_gib)
    write_reports(report,args.output_dir or root/'research_results')
    print(json.dumps({'required_primary':396,'primary_status_counts':report['primary_status_counts'],
                      'observed_stage_status_counts':report['observed_stage_status_counts'],'free_disk_gib':report['capacity']['free_disk_gib']},ensure_ascii=False))


if __name__=='__main__': main()
