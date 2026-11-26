"""Isolated serial P07 owner. No service, training or GPU starts at import.

The frozen gate registry remains a separate partial release. This owner adds a
new journal, mandatory same-checkpoint native512 reference and fresh learned64
guard for each loaded process. Original evaluation files are read-only inputs.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
import uuid

from src.eval import research_gate_controls as controls

ROOT = Path(__file__).resolve().parents[2]
OWNER = 'research_small_gate_control_serial_p07_v1'
NAMESPACE = 'gate_controls_p07_v1'
PARTIAL_READY = 'configs/research/gate_controls_context_ready.json'
PARTIAL_READY_SHA256 = '1799eeb17ae1401cca4813586e3d4c1df8f7a1c5405b1504108e9e27b39be58e'
REFERENCE_HELPER = 'src/eval/research_parallel_protocol.py'
REFERENCE_HELPER_SHA256 = 'c4b6b693dc1e7f8592290361573344dbf2760c3bcbe8dc870358c35062e2fa30'
MAX_FILE_BYTES = 16 * 2**20
MAX_ROW_BYTES = 256 * 2**10
MAX_OUTPUT_BYTES = 2**30
MAX_ATTEMPTS = 16
HEX = re.compile(r'^[0-9a-f]{64}$')
canonical = controls.canonical
digest = controls.digest
file_sha = controls.file_sha


def clone(value): return json.loads(canonical(value))


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('P07 requires a bounded ordinary JSON file')
    return json.loads(path.read_bytes())


def stat(path):
    value = Path(path).stat()
    return {name: getattr(value, name) for name in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def descriptor(path):
    path = Path(path).resolve(strict=True)
    before = stat(path)
    result = {'path': str(path), 'stat': before, 'sha256': file_sha(path)}
    if stat(path) != before: raise ValueError('P07 descriptor source changed')
    return result


def verify_partial_release():
    if file_sha(ROOT / PARTIAL_READY) != PARTIAL_READY_SHA256:
        raise ValueError('P07 partial registry release changed')
    ready = read_json(ROOT / PARTIAL_READY)
    if ready.get('status') != 'context_registry_tests_passed' or ready.get('tests_passed') != 49:
        raise ValueError('P07 registry release proof differs')
    for name, expected in ready['source_sha256'].items():
        if file_sha(ROOT / name) != expected: raise ValueError('P07 registry source changed: ' + name)
    if file_sha(ROOT / REFERENCE_HELPER) != REFERENCE_HELPER_SHA256:
        raise ValueError('P07 native reference validator source changed')
    controls.verify_frozen_sources()
    return descriptor(ROOT / PARTIAL_READY)


def native_environment():
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'bitsandbytes', 'tokenizers', 'huggingface-hub'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return {'python': sys.version, 'packages': packages}


def generation_asset(provenance):
    """Resolve only a small already-cached native JSON asset; no download."""
    from huggingface_hub import try_to_load_from_cache
    ref = provenance['model_ref']
    path = try_to_load_from_cache(ref['repo'], 'generation_config.json', revision=ref['revision'])
    if not isinstance(path, str): raise FileNotFoundError('Pinned native generation_config.json is not cached')
    path = Path(path).resolve(strict=True)
    value = read_json(path)
    eos = value.get('eos_token_id')
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    if any(type(token) is not int or token < 0 for token in eos): raise ValueError('Invalid native EOS configuration')
    return descriptor(path), sorted(set(eos))


def validate_reference(context, native_cache, *, verify_pixels=True):
    """Audit all original512 raw tokens, unchanged scorers and source provenance.

    No originals are generated here. A missing/incompatible reference is a hard
    closed gate, including a BEST reference supplied for a different LAST.
    """
    from src.eval import research_parallel_protocol as reference
    from src.eval.research_caption_generation import _rows
    verify_partial_release()
    static = context['static']; checkpoint = static['checkpoint']
    directory = Path(native_cache).resolve(strict=True)
    paths = {name: directory / name for name in ('protocol.json', 'predictions.jsonl', 'summary.json', 'model_provenance.json')}
    files = {name: descriptor(path) for name, path in paths.items()}
    parent = read_json(paths['protocol.json']); summary = read_json(paths['summary.json'])
    provenance = read_json(paths['model_provenance.json'])
    records = _rows(static['manifest']['path']); reference.validate_records(records)
    if [controls.public_record(row) for row in records] != context['records']:
        raise ValueError('P07 full scorer records differ from public manifest')
    ids = [record['record_id'] for record in records]
    expected_selected = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    ablation = parent.get('image_ablation', {})
    if (parent.get('schema_version') != 1 or parent.get('checkpoint_sha256') != checkpoint['artifact']['sha256'] or
            parent.get('manifest_sha256') != controls.MANIFEST_SHA256 or parent.get('manifest_records') != 512 or
            parent.get('selected_record_ids_sha256') != expected_selected or parent.get('profile') is not None or
            parent.get('generation') != reference.NATIVE_GENERATION or ablation.get('variant') != 'none' or
            ablation.get('seed') != 42 or ablation.get('records') != 512 or
            ablation.get('source_manifest_sha256') != controls.MANIFEST_SHA256 or
            ablation.get('transformed_manifest_sha256') != controls.MANIFEST_SHA256):
        raise ValueError('P07 needs same-checkpoint complete original native512 B1/max64 reference')
    parent_sources = parent.get('source_sha256', {})
    if not parent_sources or not all(name in parent_sources for name in (
            'src/eval/research_inference.py', 'src/eval/research_metrics.py', 'scripts/eval_research_checkpoint.py')):
        raise ValueError('Original native source proof missing')
    for name, expected in parent_sources.items():
        path = (ROOT / name).resolve(strict=True)
        if Path(name).is_absolute() or not path.is_relative_to(ROOT) or not HEX.fullmatch(str(expected)) or file_sha(path) != expected:
            raise ValueError('Original native source changed: ' + name)
    if parent.get('environment') != native_environment(): raise ValueError('Original native runtime versions differ')
    config = checkpoint['config']; marker = checkpoint['roles'][checkpoint['role']]['marker']
    if (provenance.get('recipe') != config['recipe'] or provenance.get('stage') != 'SFT' or
            provenance.get('training_step') != marker['step'] or provenance.get('model_ref', {}).get('repo') != config['model'] or
            provenance.get('vision_ref', {}).get('repo') != config['vision']):
        raise ValueError('Original native loaded checkpoint role/step/backbone differs')
    for name in ('model_ref', 'vision_ref'):
        if not re.fullmatch('[0-9a-f]{40}', str(provenance.get(name, {}).get('revision', ''))):
            raise ValueError('Original native model/vision revision is not pinned')
    if not HEX.fullmatch(str(provenance.get('cache_fingerprint', ''))): raise ValueError('Original native cache fingerprint missing')
    asset, eos = generation_asset(provenance)
    rows = _rows(paths['predictions.jsonl'])
    if len(rows) != 512 or [row.get('record_id') for row in rows] != ids:
        raise ValueError('Original native reference is incomplete, duplicate or reordered')
    parent_sha = hashlib.sha256(json.dumps(parent, sort_keys=True).encode()).hexdigest()
    images = {}
    for record, row in zip(records, rows):
        if row.get('protocol_sha256') != parent_sha or any(row.get(key) != record.get(key) for key in reference.ROW_FIELDS):
            raise ValueError('Original native row source/gold/question provenance differs')
        reference.validate_generated(record, row, eos_token_ids=eos)
        if type(row.get('score')) not in (int, float) or row['score'] != reference.score_prediction(record, row['prediction']):
            raise ValueError('Original native raw score differs from frozen scorer')
        if verify_pixels:
            for path, image_hash in zip(record['image_paths'], record['image_hashes']):
                if path not in images:
                    before = stat(path); actual = reference._rgb_hash(path)
                    if stat(path) != before or actual != image_hash:
                        raise ValueError('P07 original RGB pixels changed')
                    images[path] = actual
                if images[path] != image_hash: raise ValueError('P07 RGB identity conflict')
    metrics = reference.aggregate_scores(rows, require_four_groups=True)
    if (any(summary.get(key) != value for key, value in metrics.items()) or
            summary.get('complete_manifest') is not True or summary.get('manifest_records') != 512 or
            summary.get('protocol_sha256') != parent_sha or summary.get('checkpoint_sha256') != checkpoint['artifact']['sha256'] or
            summary.get('manifest_sha256') != controls.MANIFEST_SHA256 or summary.get('image_ablation') != ablation):
        raise ValueError('Original native summary is not the complete recomputed512 score')
    for item in files.values():
        if stat(item['path']) != item['stat'] or file_sha(item['path']) != item['sha256']:
            raise ValueError('Original native cache changed while auditing')
    return {'binding': {'directory': str(directory), 'files': files, 'native_protocol_sha256': parent_sha,
        'native_model_provenance': provenance, 'native_generation_asset': asset, 'eos_token_ids': eos,
        'records': 512, 'record_order_sha256': digest(ids), 'raw_rows_sha256': digest(rows), 'scorer_records_sha256': digest(records),
        'metrics': metrics, 'canonical_RGB_paths_verified': len(images), 'original_cache_modified': False},
        'rows': rows, 'records': records, 'guard_indices': reference.select_guard_indices(records)}


def prepare_context(run, *, checkpoint='best', condition='learned', manifest=None, native_cache=None, cache_dir='cache/vision'):
    partial = verify_partial_release()
    base = controls.prepare_context(run, checkpoint=checkpoint, condition=condition, manifest=manifest)
    directory = Path(native_cache) if native_cache else Path(base['static']['checkpoint']['run']) / 'evaluation/heldout_v3'
    original = validate_reference(base, directory)
    names = ('src/eval/research_gate_evaluation.py', 'scripts/eval_research_gate_checkpoint.py',
        'tests/test_research_gate_evaluation.py', REFERENCE_HELPER, 'scripts/research_grounding_queue.py',
        'third_party/vqa_normalization/tables.json')
    files = {name: descriptor(ROOT / name) for name in names}
    static = {'schema_version': 1, 'owner': OWNER, 'partial_registry_release': partial,
        'base': base['static'], 'original_native512': original['binding'], 'sources': files,
        'environment': native_environment(), 'cache_dir': str(Path(cache_dir).resolve()),
        'guard': {'policy': 'fresh learned identity on all64 stratified records for every loaded process before any controlled row',
                  'indices': original['guard_indices'], 'allow_original_generation_fallback': False},
        'output_scope': 'secondary P07 mechanistic pilot; no primary or P06 completion claim',
        'resources': {'execution': 'exclusive serial native B1; no parallel workers', 'max_new_tokens': 64,
            'preserve_checkpoint_master_dtypes': True, 'context': 'frozen native builder; no truncation',
            'disk_reserve_bytes': 120 * 2**30, 'maximum_output_bytes': MAX_OUTPUT_BYTES,
            'GPU_fit_measured': False},
        'paired_statistics': {'reference': 'all512 same-checkpoint native original predictions, never synthesized',
            'replicates': 1000, 'seed': 42, 'unit': 'original canonical image clusters within each task group',
            'sign': 'original minus controlled score in percentage points'}}
    return {'static': static, 'base': base, 'records': base['records'], 'original': original}


def assert_context(context, *, rehash=False):
    verify_partial_release(); controls.assert_context(context['base'], rehash=rehash)
    static = context['static']; original = static['original_native512']
    files = [static['partial_registry_release'], *static['sources'].values(),
             *original['files'].values(), original['native_generation_asset']]
    for item in files:
        if stat(item['path']) != item['stat'] or (rehash and file_sha(item['path']) != item['sha256']):
            raise ValueError('P07 owner/native reference/source changed: ' + item['path'])
    if native_environment() != static['environment']: raise ValueError('P07 runtime environment changed')
    if digest(context['original']['rows']) != original['raw_rows_sha256']:
        raise ValueError('P07 original reference rows changed in memory')
    if digest(context['original']['records']) != original['scorer_records_sha256']:
        raise ValueError('P07 original scorer/gold records changed in memory')
    from src.eval.research_parallel_protocol import select_guard_indices
    indices = select_guard_indices(context['original']['records'])
    if indices != static['guard']['indices'] or indices != context['original']['guard_indices']:
        raise ValueError('P07 learned64 selected records changed')


def output_path(context, requested=None):
    base = context['static']['base']; checkpoint = base['checkpoint']
    expected = Path(checkpoint['run']) / 'evaluation' / NAMESPACE / base['condition'] / checkpoint['role']
    if requested is not None and Path(requested).resolve() != expected.resolve():
        raise ValueError('P07 requires its own condition/role secondary namespace')
    for parent in (expected, *expected.parents):
        if parent.is_symlink(): raise ValueError('P07 output cannot traverse symlinks')
    return expected


def output_bytes(output):
    total = 0
    for path in Path(output).rglob('*'):
        if path.is_symlink(): raise ValueError('P07 output contains a symlink')
        if path.is_file(): total += path.stat().st_size
    return total


def disk_guard(output, reserve_gb, anticipated=0):
    if not math.isfinite(reserve_gb) or reserve_gb < 120: raise ValueError('P07 preserves at least120GiB disk reserve')
    if anticipated < 0 or output_bytes(output) + anticipated > MAX_OUTPUT_BYTES:
        raise OSError('P07 output staging bound exceeded')
    if shutil.disk_usage(output).free < reserve_gb * 2**30 + anticipated:
        raise OSError('P07 disk reserve would be consumed')


def write(output, path, value, reserve_gb, *, immutable=True, raw=False):
    path = Path(path); blob = value if raw else canonical(value) + b'\n'
    if len(blob) > MAX_FILE_BYTES: raise ValueError('P07 metadata file bound exceeded')
    if path.is_symlink() or not path.resolve().is_relative_to(Path(output).resolve()):
        raise ValueError('P07 writes must stay inside owned output')
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError('P07 write path traverses a symlink')
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != blob: raise ValueError('P07 immutable commit differs: ' + path.name)
        return
    disk_guard(output, reserve_gb, len(blob))
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as handle: handle.write(blob); handle.flush(); os.fsync(handle.fileno())
        if immutable:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def preserve_staging(output, directory, allowed, protocol_sha, reserve_gb, *, resume):
    """A killed atomic write is not a committed row; preserve its bytes first."""
    directory = Path(directory)
    if directory.is_symlink(): raise ValueError('P07 staging directory is a symlink')
    if not directory.exists(): return
    for path in directory.iterdir():
        match = re.fullmatch(r'\.(.+)\.([0-9a-f]{32})\.tmp', path.name)
        if match and match[1] in allowed:
            if not resume or path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError('P07 owned staging recovery needs explicit resume')
            before = stat(path); raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
            if stat(path) != before: raise ValueError('P07 staging changed during preservation')
            saved = Path(output) / 'recovery' / (path.name + '.' + sha + '.uncommitted')
            write(output, saved, raw, reserve_gb, raw=True)
            write(output, saved.with_name(saved.name + '.json'), {'owner': OWNER, 'source_name': path.name,
                'bytes': len(raw), 'sha256': sha, 'committed_record': False, 'protocol_sha256': protocol_sha}, reserve_gb)
            if stat(path) != before: raise ValueError('P07 staging changed before removal')
            path.unlink()
            fd = os.open(directory, os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)


def gate_bits(entries):
    return [{'module': name, 'gate_logit_tensor': controls.tensor_identity(parameter)} for name, block, parameter in entries]


def receipt_bits(receipt):
    return [{key: item[key] for key in ('module', 'gate_logit_tensor')} for item in receipt['gates']]


def applied_sha(receipt):
    return digest([{'module': item['module'], 'applied_gate_tensor': item['applied_gate_tensor']} for item in receipt['gates']])


def parameter_census(model):
    return [(name, id(value), value.data_ptr(), value._version, str(value.dtype), str(value.device),
             list(value.shape), value.requires_grad) for name, value in model.named_parameters()]


def parameter_bits(model):
    """All unique named parameter bytes, streamed in at most8MiB CPU chunks.

    Hashing compressed Params4bit storage does not dequantize it. This detects
    even .data writes which do not increment PyTorch's version counter. The
    source buffers/temporary KV and grouped-weight caches are not parameters.
    """
    import torch
    result = []; total = 0
    with torch.no_grad():
        for name, value in model.named_parameters():
            if value.device.type == 'meta' or not value.is_contiguous():
                raise ValueError('P07 parameter raw-bit census requires actual contiguous Small parameters')
            raw = value.detach().reshape(-1).view(torch.uint8)
            sha = hashlib.sha256()
            for chunk in raw.split(8 * 2**20):
                sha.update(chunk.cpu().numpy().tobytes())
            result.append({'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
                'bytes': raw.numel(), 'sha256': sha.hexdigest()})
            total += raw.numel()
    return {'parameters': len(result), 'bytes': total, 'raw_parameter_census_sha256': digest(result)}


def validate_generated(record, generated, original, eos, condition):
    from src.eval import research_parallel_protocol as reference
    expected_keys = set(reference.GEN_FIELDS) | {'prefix_token_ids', 'suffix_token_ids', 'prompt_token_ids_sha256'}
    if set(generated) != expected_keys: raise ValueError('P07 generator returned unexpected/missing fields, possibly gold')
    reference.validate_generated(record, generated, eos_token_ids=eos)
    prefix, suffix = generated['prefix_token_ids'], generated['suffix_token_ids']
    if any(not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids) for ids in (prefix, suffix)):
        raise ValueError('P07 actual prompt IDs must be nonnegative token lists')
    captured = {key: generated[key] for key in ('prefix_token_ids', 'suffix_token_ids', 'prompt_sha256', 'visual_tokens', 'prefill_tokens')}
    if (generated['prefill_tokens'] != len(prefix) + len(suffix) + generated['visual_tokens'] or
            generated['prompt_token_ids_sha256'] != digest(captured)):
        raise ValueError('P07 actual prompt-token hash/order differs')
    for key in ('prompt_sha256', 'visual_tokens', 'prefill_tokens', 'logits_to_keep'):
        if generated[key] != original[key]: raise ValueError('P07 controlled prompt differs from original native reference')
    if condition == 'learned' and any(generated[key] != original[key] for key in reference.PARITY_FIELDS):
        raise ValueError('P07 learned identity differs from original native tokens/metadata')
    return clone(generated)


class NativeRuntime(controls.NativeGateRuntime):
    def describe(self, model, provenance):
        from src.eval.research_caption_generation import NativeRuntime as CaptionRuntime
        result = CaptionRuntime().describe(model, provenance)
        eos = getattr(model.base.generation_config, 'eos_token_id', None)
        result['eos_token_ids'] = [] if eos is None else sorted(set(eos if isinstance(eos, list) else [eos]))
        return result


def assert_runtime(runtime, *, rehash=False):
    from src.eval.research_caption_generation import assert_model_files
    assert_model_files({'model_provenance': runtime}, rehash=rehash)


def gpu_idle(context):
    value = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=15)
    if value.stdout.strip(): raise RuntimeError('P07 requires an exclusive root-scheduled idle GPU')
    value = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=15)
    free = [int(item.strip()) * 2**20 for item in value.stdout.splitlines() if item.strip()]
    if len(free) != 1 or free[0] < 64 * 2**30:
        raise RuntimeError('P07 serial proposal requires one idle GPU with at least64GiB free; fit remains unmeasured')
    host = {key: int(value.split()[0]) * 1024 for key, value in
        (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    available = host['MemAvailable']; maximum = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if maximum != 'max': available = min(available, int(maximum) - int(Path('/sys/fs/cgroup/memory.current').read_text()))
    source = context['static']['base']['checkpoint']['artifact']['total_bytes']
    if available < 32 * 2**30 + max(12 * 2**30, 2 * source):
        raise RuntimeError('P07 host/cgroup admission does not retain32GiB headroom plus source/load allowance')


def guard_record(context, model, runtime, loaded, entries, output, reserve_gb, progress):
    """Always fresh learned64, including resumes. Failed parity blocks namespace."""
    import torch
    directory = Path(output) / 'guards'; directory.mkdir(exist_ok=True)
    attempts = [path for path in directory.iterdir() if path.suffix == '.json' and not path.name.startswith('.')]
    if len(attempts) >= MAX_ATTEMPTS: raise ValueError('P07 guard attempt bound reached')
    for path in attempts:
        if read_json(path).get('status') == 'failed_parity': raise ValueError('Prior learned64 parity failure blocks this namespace')
    attempt = uuid.uuid4().hex; path = directory / (attempt + '.json')
    before = gate_bits(entries); before_sha = digest(before); census = parameter_census(model)
    report = {'owner': OWNER, 'attempt': attempt, 'status': 'running', 'admission_sha256': digest(context['static']),
        'runtime_sha256': digest(loaded), 'indices': context['original']['guard_indices'], 'records': [],
        'original_gate_pre_sha256': before_sha, 'original_gate_post_sha256': None}
    write(output, path, report, reserve_gb, immutable=False)
    try:
        with torch.no_grad(), controls.gate_control(model, 'learned', recipe=context['static']['base']['checkpoint']['config']['recipe']) as receipt:
            report['learned_control'] = receipt
            if receipt_bits(receipt) != before: raise ValueError('P07 learned64 initial gate bits differ')
            for index in report['indices']:
                assert_context(context); assert_runtime(loaded)
                generated = runtime.generate(model, context['records'][index])
                validate_generated(context['records'][index], generated, context['original']['rows'][index],
                    context['static']['original_native512']['eos_token_ids'], 'learned')
                report['records'].append({'index': index, 'record_id': context['records'][index]['record_id'],
                    'generated': generated, 'native_reference_row_sha256': digest(context['original']['rows'][index])})
                write(output, path, report, reserve_gb, immutable=False)
                if progress: progress({'status': 'learned_identity_guard', 'guard_records': len(report['records']), 'guard_total': 64})
        after = gate_bits(entries)
        if after != before or parameter_census(model) != census: raise ValueError('P07 learned64 changed original parameters')
        report.update(status='passed', original_gate_post_sha256=digest(after), parameter_identity_preserved=True,
            exact_native64_token_prompt_parity=True, scoped_overrides_restored=True)
        write(output, path, report, reserve_gb, immutable=False)
        return descriptor(path)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'failed_parity',
            error_type=type(error).__name__, error=str(error))
        write(output, path, report, reserve_gb, immutable=False)
        raise


def validate_guard(item, context, runtime_sha):
    if stat(item['path']) != item['stat'] or file_sha(item['path']) != item['sha256']:
        raise ValueError('P07 learned64 guard receipt changed')
    report = read_json(item['path']); indices = context['original']['guard_indices']
    if (report.get('owner') != OWNER or report.get('status') != 'passed' or
            report.get('admission_sha256') != digest(context['static']) or report.get('runtime_sha256') != runtime_sha or
            report.get('indices') != indices or len(report.get('records', [])) != 64 or
            report.get('original_gate_pre_sha256') != report.get('original_gate_post_sha256') or
            report.get('exact_native64_token_prompt_parity') is not True or report.get('parameter_identity_preserved') is not True or
            report.get('scoped_overrides_restored') is not True):
        raise ValueError('P07 learned64 guard is incomplete/incompatible')
    learned = report.get('learned_control', {})
    if (learned.get('owner') != controls.OWNER or learned.get('condition') != 'learned' or
            learned.get('recipe') != context['static']['base']['checkpoint']['config']['recipe'] or
            learned.get('parameter_values_written') is not False or digest(learned.get('gates')) != learned.get('gates_sha256') or
            digest(receipt_bits(learned)) != report.get('original_gate_pre_sha256')):
        raise ValueError('P07 learned64 original gate receipt differs')
    validate_control_against_guard(learned, learned)
    for index, item in zip(indices, report['records']):
        if item.get('index') != index or item.get('record_id') != context['records'][index]['record_id'] or item.get('native_reference_row_sha256') != digest(context['original']['rows'][index]):
            raise ValueError('P07 learned64 guard record binding differs')
        validate_generated(context['records'][index], item['generated'], context['original']['rows'][index],
            context['static']['original_native512']['eos_token_ids'], 'learned')
    return report


def validate_control_against_guard(control, learned):
    """Pure CPU reconstruction of endpoint bits from observed learned gates."""
    condition = control.get('condition')
    if condition not in controls.CONDITIONS or control.get('recipe') != learned.get('recipe'):
        raise ValueError('P07 condition/family differs from learned64 gate receipt')
    if digest(control.get('gates')) != control.get('gates_sha256'):
        raise ValueError('P07 controlled gate receipt checksum differs')
    before = learned.get('gates'); after = control.get('gates')
    if not isinstance(before, list) or not before or not isinstance(after, list) or len(before) != len(after):
        raise ValueError('P07 gate layer census differs')
    if len({item.get('module') for item in before}) != len(before): raise ValueError('P07 gate module names duplicate')
    for original, changed in zip(before, after):
        for key in ('module', 'old_experts', 'added_experts', 'gate_logit_tensor', 'learned_gate_tensor', 'columns'):
            if original.get(key) != changed.get(key): raise ValueError('P07 original gate metadata changed')
        values = original.get('applied_values')
        total = original.get('old_experts', 0) + original.get('added_experts', 0)
        if not isinstance(values, list) or len(values) != total or original.get('columns') != ['text', 'visual']:
            raise ValueError('P07 learned gate shape/column census differs')
        expected = []
        for pair in values:
            if (not isinstance(pair, list) or len(pair) != 2 or
                    any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in pair)):
                raise ValueError('P07 learned probabilities are malformed/nonfinite')
            text, visual = pair
            if condition == 'all_open': text = visual = 1.
            elif condition == 'visual_closed': visual = 0.
            elif condition == 'text_closed': text = 0.
            elif condition == 'all_closed': text = visual = 0.
            expected.append([text, visual])
        if changed.get('applied_values') != expected: raise ValueError('P07 applied endpoints differ from requested intervention')
        learned_bytes = struct.pack('<' + 'f' * (2 * total), *(value for pair in values for value in pair))
        expected_bytes = struct.pack('<' + 'f' * (2 * total), *(value for pair in expected for value in pair))
        for tensor, raw in ((original.get('learned_gate_tensor', {}), learned_bytes),
                            (original.get('applied_gate_tensor', {}), learned_bytes),
                            (changed.get('applied_gate_tensor', {}), expected_bytes)):
            if tensor != {'dtype': 'torch.float32', 'shape': [total, 2], 'sha256': hashlib.sha256(raw).hexdigest()}:
                raise ValueError('P07 gate probability raw-bit SHA differs')
    return True


def make_protocol(context, receipt, runtime, *, synthetic=False):
    original = context['static']['original_native512']['native_model_provenance']
    if runtime.get('loaded') != original or runtime.get('eos_token_ids') != context['static']['original_native512']['eos_token_ids']:
        raise ValueError('P07 reloaded native checkpoint/EOS differs from all512 original reference')
    inner = controls.control_protocol(context['base'], receipt, runtime['loaded'])
    if int(runtime.get('context_limit', 0)) < 128: raise ValueError('P07 native context metadata missing')
    return {'schema_version': 1, 'owner': OWNER, 'admission': context['static'], 'control': inner,
        'runtime': runtime, 'namespace': f"evaluation/{NAMESPACE}/{context['static']['base']['condition']}/{context['static']['base']['checkpoint']['role']}",
        'evidence_scope': 'synthetic_CPU_lifecycle_fixture' if synthetic else 'actual_native_model_predictions',
        'legacy_cache_compatible': False, 'primary_completion_claim': False, 'P06_complete': False}


def validate_protocol(context, protocol):
    if protocol.get('evidence_scope') not in ('synthetic_CPU_lifecycle_fixture', 'actual_native_model_predictions'):
        raise ValueError('P07 evidence scope missing/unknown')
    rebuilt = make_protocol(context, protocol['control']['observed_gate_control'], protocol['runtime'],
        synthetic=protocol['evidence_scope'] == 'synthetic_CPU_lifecycle_fixture')
    if rebuilt != protocol: raise ValueError('P07 reconstructed control/gates/runtime protocol differs')
    return protocol


def open_attempt(output, context, protocol, guard, start, reserve_gb):
    attempt = uuid.uuid4().hex
    path = Path(output) / 'attempts' / attempt / 'start.json'
    value = {'owner': OWNER, 'attempt': attempt, 'protocol_sha256': digest(protocol),
        'admission_sha256': digest(context['static']), 'runtime_sha256': digest(protocol['runtime']),
        'start_index': start, 'guard': guard,
        'original_parameter_bits': protocol['runtime']['original_parameter_bits'],
        'original_gate_sha256': digest(receipt_bits(protocol['control']['observed_gate_control']))}
    write(output, path, value, reserve_gb)
    return descriptor(path)


def attempt_start(output, item, protocol):
    path = Path(item.get('path', ''))
    if (path.name != 'start.json' or not re.fullmatch('[0-9a-f]{32}', path.parent.name) or
            path.parent.parent != (Path(output) / 'attempts').resolve() or path.is_symlink() or
            stat(path) != item.get('stat') or file_sha(path) != item.get('sha256')):
        raise ValueError('P07 model-lifetime attempt header changed/escaped')
    value = read_json(path)
    if (value.get('owner') != OWNER or value.get('attempt') != path.parent.name or
            value.get('protocol_sha256') != digest(protocol) or value.get('admission_sha256') != digest(protocol['admission']) or
            value.get('runtime_sha256') != digest(protocol['runtime']) or type(value.get('start_index')) is not int or
            not 0 <= value['start_index'] <= 512 or value.get('original_parameter_bits') != protocol['runtime']['original_parameter_bits'] or
            value.get('original_gate_sha256') != digest(receipt_bits(protocol['control']['observed_gate_control']))):
        raise ValueError('P07 attempt original model/protocol/range differs')
    return value


def attempt_rows(output, item, header):
    """Read durable rows, including a commit interrupted before list.append."""
    result = []
    for index in range(header['start_index'], 512):
        path = Path(output) / 'records' / f'{index:06d}.json'
        if not path.exists():
            break
        row = read_json(path)
        if row.get('attempt') != item:
            break
        if row.get('index') != index: raise ValueError('P07 attempt row index differs')
        result.append(row)
    return result


def close_attempt(output, context, protocol, item, model, entries, census, reserve_gb, termination, graceful):
    """Only the ORIGINAL live model may certify its accepted row range.

    Runs on graceful exceptions as well as completion. An unclosed SIGKILL range
    remains forensic and is rejected before any future model is loaded.
    """
    header = attempt_start(output, item, protocol)
    rows = attempt_rows(output, item, header)
    path = Path(item['path']).with_name('closed.json')
    value = {'owner': OWNER, 'attempt': header['attempt'], 'attempt_header': item,
        'protocol_sha256': digest(protocol), 'start_index': header['start_index'],
        'end_index_exclusive': header['start_index'] + len(rows), 'records': len(rows),
        'raw_rows_sha256': digest(rows), 'termination': termination,
        'termination_verified_graceful': graceful,
        'original_parameter_bits_pre': header['original_parameter_bits'],
        'original_parameter_bits_post': None, 'status': 'unverified',
        'original_gate_pre_sha256': header['original_gate_sha256'], 'original_gate_post_sha256': None}
    try:
        value['original_parameter_bits_post'] = parameter_bits(model)
        value['original_gate_post_sha256'] = digest(gate_bits(entries))
        stable = (value['original_parameter_bits_post'] == value['original_parameter_bits_pre'] and
            value['original_gate_post_sha256'] == value['original_gate_pre_sha256'] and parameter_census(model) == census and
            all('sigmoid' not in parameter.__dict__ for _, _, parameter in entries))
        allowed_scope = graceful and termination in ('completed', 'InterruptedError', 'KeyboardInterrupt')
        if not stable or not allowed_scope:
            value['status'] = 'failed_integrity' if not stable else 'failed_scope_integrity'
            write(output, path, value, reserve_gb)
            write(output, Path(output) / 'failed_integrity.json', {'owner': OWNER, 'status': value['status'],
                'attempt': item, 'before': value['original_parameter_bits_pre'], 'after': value['original_parameter_bits_post'],
                'admission_sha256': digest(context['static']), 'scientific_completion_claim': False}, reserve_gb)
            write(output, Path(output) / 'progress.json', {'owner': OWNER, 'status': 'failed_integrity',
                'completed_records': value['end_index_exclusive'], 'scientific_completion_claim': False}, reserve_gb, immutable=False)
            if not stable:
                raise ValueError('P07 original parameter raw bits changed; committed rows cannot be published or resumed')
            raise ValueError('P07 unexpected control-scope failure leaves original model lifetime untrusted; no publication or resume')
        assert_context(context, rehash=True); assert_runtime(protocol['runtime'], rehash=True)
        value.update(status='closed_unchanged', parameter_identity_preserved=True,
            scoped_sigmoid_overrides_restored=True, source_files_rehashed_unchanged=True)
        write(output, path, value, reserve_gb)
        return value
    except BaseException as error:
        if not path.exists():
            value.update(status='unverified', error_type=type(error).__name__, error=str(error))
            write(output, path, value, reserve_gb)
        raise


def validate_attempt(output, item, context, protocol):
    header = attempt_start(output, item, protocol)
    path = Path(item['path']).with_name('closed.json')
    if not path.is_file() or path.is_symlink():
        raise ValueError('P07 unclosed model-lifetime attempt rows are forensic only; fresh model cannot certify them')
    closed = read_json(path); rows = attempt_rows(output, item, header)
    if (closed.get('owner') != OWNER or closed.get('status') != 'closed_unchanged' or closed.get('attempt_header') != item or
            closed.get('attempt') != header['attempt'] or closed.get('protocol_sha256') != digest(protocol) or
            closed.get('start_index') != header['start_index'] or closed.get('end_index_exclusive') != header['start_index'] + len(rows) or
            closed.get('records') != len(rows) or closed.get('raw_rows_sha256') != digest(rows) or
            closed.get('original_parameter_bits_pre') != protocol['runtime']['original_parameter_bits'] or
            closed.get('original_parameter_bits_post') != protocol['runtime']['original_parameter_bits'] or
            closed.get('original_gate_pre_sha256') != header['original_gate_sha256'] or
            closed.get('original_gate_post_sha256') != header['original_gate_sha256'] or
            closed.get('parameter_identity_preserved') is not True or closed.get('scoped_sigmoid_overrides_restored') is not True or
            closed.get('source_files_rehashed_unchanged') is not True or closed.get('termination_verified_graceful') is not True or
            closed.get('termination') not in ('completed', 'InterruptedError', 'KeyboardInterrupt')):
        raise ValueError('P07 original model-lifetime row-range closure proof differs/untrusted')
    return closed


def closure_census(output, rows):
    result = []; seen = set()
    for row in rows:
        item = row['attempt']; key = digest(item)
        if key not in seen:
            seen.add(key); result.append(descriptor(Path(item['path']).with_name('closed.json')))
    return result


def make_row(context, protocol, index, generated, pre, post, guard, elapsed, attempt):
    record = context['records'][index]; condition = context['static']['base']['condition']
    generated = validate_generated(record, generated, context['original']['rows'][index],
        context['static']['original_native512']['eos_token_ids'], condition)
    receipt = protocol['control']['observed_gate_control']; original_sha = digest(receipt_bits(receipt))
    if pre != original_sha or post != original_sha: raise ValueError('P07 controlled generation changed original gate bits')
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0: raise ValueError('P07 invalid elapsed time')
    if generated['prefill_tokens'] + 64 > protocol['runtime']['context_limit']: raise ValueError('P07 generated beyond native context')
    row = {'owner': OWNER, 'index': index, 'record_id': record['record_id'], 'public_record_sha256': digest(record),
        'condition': condition, 'protocol_sha256': digest(protocol), 'generated': generated,
        'generated_token_ids_sha256': digest(generated['generated_token_ids']),
        'native_reference_row_sha256': digest(context['original']['rows'][index]),
        'original_gate_pre_sha256': pre, 'original_gate_post_sha256': post,
        'applied_gate_mask_sha256': applied_sha(receipt), 'guard': guard, 'generation_seconds': elapsed, 'attempt': attempt}
    if len(canonical(row)) > MAX_ROW_BYTES: raise ValueError('P07 journal row bound exceeded')
    return row


def committed_rows(output, context, protocol, reserve_gb, *, resume):
    directory = Path(output) / 'records'; directory.mkdir(exist_ok=True)
    expected = {f'{index:06d}.json' for index in range(512)}
    preserve_staging(output, directory, expected, digest(protocol), reserve_gb, resume=resume)
    if any(path.is_symlink() or not path.is_file() or path.name not in expected for path in directory.iterdir()):
        raise ValueError('P07 journal contains a foreign/non-owned entry')
    rows = []; guards = {}; attempts = {}
    for index in range(512):
        path = directory / f'{index:06d}.json'
        if not path.exists():
            if any((directory / f'{later:06d}.json').exists() for later in range(index + 1, 512)):
                raise ValueError('P07 journal is not a contiguous manifest prefix')
            break
        saved = read_json(path)
        guard = saved.get('guard', {})
        key = digest(guard)
        if key not in guards:
            guard_path = Path(guard.get('path', ''))
            if guard_path.parent != (Path(output) / 'guards').resolve() or not re.fullmatch(r'[0-9a-f]{32}\.json', guard_path.name):
                raise ValueError('P07 row guard escaped owned guard directory')
            guards[key] = validate_guard(guard, context, digest(protocol['runtime']))
            validate_control_against_guard(protocol['control']['observed_gate_control'], guards[key]['learned_control'])
        attempt = saved.get('attempt', {}); attempt_key = digest(attempt)
        if attempt_key not in attempts:
            attempts[attempt_key] = validate_attempt(output, attempt, context, protocol)
        closure = attempts[attempt_key]
        header = read_json(attempt['path'])
        if not closure['start_index'] <= index < closure['end_index_exclusive'] or header['guard'] != guard:
            raise ValueError('P07 row falls outside its certified original model lifetime')
        rebuilt = make_row(context, protocol, index, saved['generated'], saved['original_gate_pre_sha256'],
            saved['original_gate_post_sha256'], guard, saved['generation_seconds'], attempt)
        if saved != rebuilt: raise ValueError('P07 journal row prompt/token/gate/source hash differs')
        rows.append(saved)
    return rows


def reconcile(output, rows, protocol_sha, reserve_gb, *, resume):
    path = Path(output) / 'predictions.jsonl'; wanted = b''.join(canonical(row) + b'\n' for row in rows)
    if not path.exists():
        if rows: write(output, path, wanted, reserve_gb, raw=True, immutable=False)
        return
    if path.is_symlink() or path.stat().st_size > MAX_FILE_BYTES: raise ValueError('P07 raw projection bound/path differs')
    raw = path.read_bytes()
    if raw == wanted: return
    if not resume or not wanted.startswith(raw): raise ValueError('P07 raw projection diverges from committed journal')
    sha = hashlib.sha256(raw).hexdigest(); saved = Path(output) / 'recovery' / (sha + '.jsonl.fragment')
    write(output, saved, raw, reserve_gb, raw=True)
    write(output, saved.with_name(saved.name + '.json'), {'owner': OWNER, 'old_sha256': sha,
        'new_sha256': hashlib.sha256(wanted).hexdigest(), 'protocol_sha256': protocol_sha,
        'reason': 'exact durable-row prefix projection restored; uncommitted raw suffix is never trusted'}, reserve_gb)
    write(output, path, wanted, reserve_gb, raw=True, immutable=False)


def summarize(output, context, protocol, rows, reserve_gb):
    from src.eval.research_metrics import score_prediction, aggregate_scores
    from scripts.research_grounding_queue import paired_image_bootstrap
    if len(rows) != 512: raise ValueError('P07 incomplete control rows cannot be scored/paired as a full result')
    scored = []
    for record, row in zip(context['original']['records'], rows):
        generated = row['generated']; score = score_prediction(record, generated['prediction'])
        scored.append({key: record[key] for key in ('record_id', 'task_group', 'source', 'benchmark', 'scorer')})
        scored[-1].update(prediction=generated['prediction'], score=score, raw_record_sha256=digest(row))
    metrics = aggregate_scores(scored, require_four_groups=True)
    condition = context['static']['base']['condition']
    paired = paired_image_bootstrap(context['original']['rows'], {condition: scored}, context['original']['records'], replicates=1000, seed=42)[condition]
    summary = {**metrics, 'owner': OWNER, 'schema_version': 1, 'status': 'completed', 'complete_manifest': True,
        'manifest_records': 512, 'protocol_sha256': digest(protocol), 'condition': condition,
        'checkpoint_sha256': context['static']['base']['checkpoint']['artifact']['sha256'],
        'manifest_sha256': controls.MANIFEST_SHA256, 'original_native512_binding_sha256': digest(context['static']['original_native512']),
        'raw_predictions_sha256': file_sha(Path(output) / 'predictions.jsonl'), 'paired_original_minus_control': paired,
        'evidence_scope': protocol['evidence_scope'], 'P07_mechanistic_scope_only': True,
        'primary_completion_claim': False, 'P06_complete': False, 'P07_complete': False, 'checkpoint_source_modified': False}
    write(output, Path(output) / 'scored_predictions.jsonl', b''.join(canonical(row) + b'\n' for row in scored), reserve_gb, raw=True)
    write(output, Path(output) / 'summary.json', summary, reserve_gb)
    return summary


def evaluate(context, *, output=None, resume=False, allow_gpu=False, cache_dir=None, reserve_gb=120, runtime=None, progress=None):
    """Production CLI cannot inject runtime. Tests label injected rows synthetic."""
    output = output_path(context, output); output.mkdir(parents=True, exist_ok=True)
    lock = output / '.gate_control.lock'
    if lock.is_symlink(): raise ValueError('P07 lock is a symlink')
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        disk_guard(output, reserve_gb); assert_context(context, rehash=True)
        admission_path = output / 'admission.json'; expected = context['static']
        existing = admission_path.exists()
        if existing:
            if not resume or read_json(admission_path) != expected: raise ValueError('P07 resume admission/checkpoint/source differs')
        elif any(path.name != lock.name for path in output.iterdir()):
            raise ValueError('P07 unowned partial output has no admission protocol')
        write(output, admission_path, expected, reserve_gb)
        root_names = {'admission.json', 'protocol.json', 'progress.json', 'generation_complete.json', 'failed_integrity.json',
            'scored_predictions.jsonl', 'summary.json', 'predictions.jsonl'}
        preserve_staging(output, output, root_names, digest(expected), reserve_gb, resume=resume)
        allowed = root_names | {lock.name, 'records', 'guards', 'recovery', 'attempts'}
        if any(path.name not in allowed or path.is_symlink() for path in output.iterdir()):
            raise ValueError('P07 output contains a foreign/non-owned artifact')
        if (output / 'failed_integrity.json').exists():
            raise ValueError('P07 prior full-parameter integrity failure blocks this namespace')
        if cache_dir is not None and str(Path(cache_dir).resolve()) != expected['cache_dir']:
            raise ValueError('P07 cache directory differs from bound admission protocol')
        guard_directory = output / 'guards'
        if guard_directory.exists():
            names = set()
            for path in guard_directory.iterdir():
                match = re.fullmatch(r'\.([0-9a-f]{32}\.json)\.[0-9a-f]{32}\.tmp', path.name)
                if match: names.add(match[1])
                elif not re.fullmatch(r'[0-9a-f]{32}\.json', path.name) or not path.is_file() or path.is_symlink():
                    raise ValueError('P07 guard directory contains a foreign artifact')
            preserve_staging(output, guard_directory, names, digest(expected), reserve_gb, resume=resume)
        attempt_directory = output / 'attempts'
        if attempt_directory.exists():
            for directory in attempt_directory.iterdir():
                if directory.is_symlink() or not directory.is_dir() or not re.fullmatch('[0-9a-f]{32}', directory.name):
                    raise ValueError('P07 attempt directory contains a foreign artifact')
                preserve_staging(output, directory, {'start.json', 'closed.json'}, digest(expected), reserve_gb, resume=resume)
                if any(path.is_symlink() or not path.is_file() or path.name not in ('start.json', 'closed.json') for path in directory.iterdir()):
                    raise ValueError('P07 model-lifetime attempt contains a foreign artifact')
        protocol_path = output / 'protocol.json'; protocol = read_json(protocol_path) if protocol_path.exists() else None
        if protocol is not None:
            if protocol.get('owner') != OWNER or protocol.get('admission') != expected:
                raise ValueError('P07 saved protocol belongs to another condition/source/checkpoint')
            validate_protocol(context, protocol)
            assert_runtime(protocol['runtime'])
            rows = committed_rows(output, context, protocol, reserve_gb, resume=resume)
            reconcile(output, rows, digest(protocol), reserve_gb, resume=resume)
        else:
            if (output / 'records').exists() and any((output / 'records').iterdir()):
                raise ValueError('P07 records without observed gate protocol are refused')
            rows = []
        completion_path = output / 'generation_complete.json'
        if len(rows) < 512 or not completion_path.exists():
            if not allow_gpu: raise RuntimeError('P07 pending generation/restore proof requires root-scheduled --allow-gpu')
            if runtime is None: gpu_idle(context)
            engine = runtime or NativeRuntime()
            model, loaded = engine.load(expected['base']['checkpoint']['checkpoint_path'], cache_dir or expected['cache_dir'], reserve_gb)
            actual_runtime = engine.describe(model, loaded)
            if progress: progress({'status': 'hashing_original_runtime_parameters', 'chunk_bytes': 8 * 2**20})
            actual_runtime['original_parameter_bits'] = parameter_bits(model)
            if actual_runtime.get('loaded') != expected['original_native512']['native_model_provenance']:
                raise ValueError('P07 actual loaded checkpoint differs from original native512')
            entries = controls.gate_registry(model, recipe=expected['base']['checkpoint']['config']['recipe'])
            guard = guard_record(context, model, engine, actual_runtime, entries, output, reserve_gb, progress)
            guard_proof = validate_guard(guard, context, digest(actual_runtime))
            before = gate_bits(entries); census = parameter_census(model); attempt = None; closed = None
            import torch
            try:
                with torch.no_grad(), controls.gate_control(model, expected['base']['condition'], recipe=expected['base']['checkpoint']['config']['recipe']) as receipt:
                    validate_control_against_guard(receipt, guard_proof['learned_control'])
                    observed = make_protocol(context, receipt, actual_runtime, synthetic=runtime is not None)
                    if protocol is not None and observed != protocol: raise ValueError('P07 reloaded gates/runtime/source differs on resume')
                    protocol = observed; write(output, protocol_path, protocol, reserve_gb)
                    attempt = open_attempt(output, context, protocol, guard, len(rows), reserve_gb)
                    for index in range(len(rows), 512):
                        assert_context(context); assert_runtime(actual_runtime); disk_guard(output, reserve_gb, MAX_ROW_BYTES)
                        pre = digest(gate_bits(entries)); started = time.monotonic()
                        generated = engine.generate(model, context['records'][index])
                        elapsed = time.monotonic() - started; post = digest(gate_bits(entries))
                        if parameter_census(model) != census: raise ValueError('P07 controlled forward changed original parameter identity/version')
                        row = make_row(context, protocol, index, generated, pre, post, guard, elapsed, attempt)
                        assert_context(context); assert_runtime(actual_runtime)
                        write(output, output / 'records' / f'{index:06d}.json', row, reserve_gb)
                        with (output / 'predictions.jsonl').open('ab') as sink:
                            sink.write(canonical(row) + b'\n'); sink.flush(); os.fsync(sink.fileno())
                        rows.append(row)
                        event = {'owner': OWNER, 'status': 'generating', 'condition': expected['base']['condition'],
                            'completed_records': len(rows), 'records': 512, 'original_gate_bits_preserved': True}
                        write(output, output / 'progress.json', event, reserve_gb, immutable=False)
                        if progress: progress(event)
            finally:
                if attempt is not None:
                    active_error = sys.exc_info()[1]
                    graceful = active_error is None or isinstance(active_error, (InterruptedError, KeyboardInterrupt))
                    termination = ('completed' if active_error is None else 'InterruptedError' if isinstance(active_error, InterruptedError)
                        else 'KeyboardInterrupt' if isinstance(active_error, KeyboardInterrupt) else type(active_error).__name__)
                    closed = close_attempt(output, context, protocol, attempt, model, entries, census, reserve_gb,
                        termination, graceful)
            after = gate_bits(entries)
            if after != before or parameter_census(model) != census: raise ValueError('P07 scope restoration changed original parameters')
            after_all = closed['original_parameter_bits_post']
            assert_context(context, rehash=True); assert_runtime(actual_runtime, rehash=True)
            rows = committed_rows(output, context, protocol, reserve_gb, resume=resume)
            reconcile(output, rows, digest(protocol), reserve_gb, resume=resume)
            proof = {'owner': OWNER, 'status': 'generation_complete', 'records': len(rows), 'protocol_sha256': digest(protocol),
                'original_gate_pre_sha256': digest(before), 'original_gate_post_sha256': digest(after),
                'parameter_identity_preserved': True, 'scoped_sigmoid_overrides_restored': True,
                'all_named_parameter_raw_bits_preserved': True, 'original_parameter_bits': after_all,
                'attempt_closures': closure_census(output, rows),
                'checkpoint_file_rehashed_unchanged': True, 'model_snapshot_files_rehashed_unchanged': True,
                'evidence_scope': protocol['evidence_scope'], 'raw_predictions_sha256': file_sha(output / 'predictions.jsonl')}
            write(output, completion_path, proof, reserve_gb)
            del model
        rows = committed_rows(output, context, protocol, reserve_gb, resume=resume)
        reconcile(output, rows, digest(protocol), reserve_gb, resume=resume)
        complete = read_json(completion_path)
        if (complete.get('owner') != OWNER or complete.get('records') != 512 or complete.get('protocol_sha256') != digest(protocol) or
                complete.get('raw_predictions_sha256') != file_sha(output / 'predictions.jsonl') or
                complete.get('scoped_sigmoid_overrides_restored') is not True or complete.get('parameter_identity_preserved') is not True or
                complete.get('all_named_parameter_raw_bits_preserved') is not True or
                complete.get('original_parameter_bits') != protocol['runtime']['original_parameter_bits'] or
                complete.get('attempt_closures') != closure_census(output, rows)):
            raise ValueError('P07 generation completion/restore proof differs')
        assert_context(context, rehash=True); assert_runtime(protocol['runtime'], rehash=True)
        return summarize(output, context, protocol, rows, reserve_gb)
