"""Separate A0/A1 caption metadata profile around sealed v1/v2 generation.

Only eligibility/provenance/namespace hooks differ. The native generator,
atomic journal, deferred references and full-corpus CIDEr implementation are
executed from byte-pinned private modules; canonical v1/v2 globals never change.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
OWNER = 'research_caption_completed_stages_native_b1_v1'
SOURCE = 'src/eval/research_caption_stages.py'
CLI = 'scripts/eval_research_stage_caption.py'
TEST = 'tests/test_research_caption_stages.py'
V1_SOURCE = 'src/eval/research_caption_generation.py'
V1_SHA256 = '754c261266ab3544d81560abd90447d6abc2ad6b74b987321d007815407abefa'
V2_SOURCE = 'src/eval/research_caption_generation_v2.py'
V2_SHA256 = 'a27f5bc5074352f1941bc4e34e519b9bac107667cdf7221a3a5b446d384cdb5e'
TEXT_SOURCE = 'src/eval/research_text_generation.py'
TEXT_SHA256 = 'b311a58969a28888169ed965c5d919b9dee2ddf4743787655e54af9d33182e0e'
DATA_PROTOCOL_SHA256 = 'af605a57cdfa89deb99a98aa071f02bc4ddd3975fb196ff16f083287bbdb1aa6'
DEFAULT_BUNDLE = ROOT / 'datasets/research/caption-eval-pilot-en-v2'
NAMESPACE = 'caption_stages_pilot_en_v2'
RECORDS = 372
HEX = re.compile('[0-9a-f]{64}')


def _sha(raw): return hashlib.sha256(raw).hexdigest()


def _available_host_bytes():
    rows = {key: int(value.split()[0]) * 1024 for key, value in
        (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    available = rows['MemAvailable']
    maximum = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if maximum != 'max':
        available = min(available, int(maximum) - int(Path('/sys/fs/cgroup/memory.current').read_text()))
    return max(0, available)


def _memory_guard():
    available = _available_host_bytes()
    if available < 32 * 2**30:
        raise MemoryError('Stage-caption metadata/hash inspection preserves32GiB actual host/cgroup headroom; no reclaimable-cache credit')
    return available


def _private_v2():
    for name, expected in ((V1_SOURCE, V1_SHA256), (V2_SOURCE, V2_SHA256), (TEXT_SOURCE, TEXT_SHA256)):
        if _sha((ROOT / name).read_bytes()) != expected:
            raise ValueError('Stage-caption reused sealed source differs: ' + name)
    raw = (ROOT / V2_SOURCE).read_bytes()
    module = ModuleType('src.eval._stage_caption_private_frozen_v2')
    module.__file__ = str(ROOT / V2_SOURCE); module.__package__ = 'src.eval'
    exec(compile(raw, module.__file__, 'exec'), module.__dict__)
    if (ROOT / V2_SOURCE).read_bytes() != raw: raise ValueError('Sealed caption v2 source changed during private loading')
    module.DATA_PROTOCOL_SHA256 = DATA_PROTOCOL_SHA256; module.RECORDS = RECORDS
    return module


def _artifact_identity(path):
    """Existing whole-artifact checksum policy, streamed with real headroom checks."""
    from src.utils.research_checkpoint_compat import checkpoint_identity
    _memory_guard()
    last = [time.monotonic()]
    def progress(done, total):
        _memory_guard()
        if time.monotonic() - last[0] >= 15:
            print(json.dumps({'event': 'stage_caption_hashing_checkpoint', 'bytes_completed': done, 'bytes_total': total}), flush=True)
            last[0] = time.monotonic()
    result = checkpoint_identity(path, progress=progress)
    _memory_guard()
    return result


def _checkpoint_context(owner, run, role):
    """Completed raw A0/A1 eligibility; terminal LAST proves completion."""
    run = Path(run).resolve(strict=True)
    if role not in ('best', 'last'): raise ValueError('Stage-caption retained BEST/LAST role required')
    lock = run / '.checkpoint.lock'
    if not lock.is_file() or lock.is_symlink(): raise ValueError('Existing completed-stage manager lock required')
    with lock.open('rb') as handle:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        config = owner.read_json(run / 'resolved_config.json'); completion = owner.read_json(run / 'completed.json')
        manager = owner.read_json(run / '.checkpoint_manager.json')
        declared = Path(config.get('output', '')); declared = (declared if declared.is_absolute() else ROOT / declared).resolve()
        if (declared != run or config.get('stage') not in ('A0', 'A1') or
                type(completion.get('steps')) is not int or completion['steps'] <= 0 or
                completion.get('status') in ('failed', 'incomplete', 'interrupted') or
                not isinstance(manager.get('owner'), str) or not manager['owner']):
            raise ValueError('Actual completed owned A0/A1 run required; SFT uses its separate sealed owner')
        if config.get('recipe') not in ('4f', '4q', '1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '3c'):
            raise ValueError('Unsupported stage-caption recipe')
        roles = {}
        for name in ('best', 'last'):
            pointer = run / name
            if not pointer.is_symlink(): raise ValueError('Retained completed-stage BEST/LAST roles required; retired roles are never recreated')
            target = pointer.resolve(strict=True)
            if target.parent != run / '.checkpoint_objects' or not target.name.startswith('object-'):
                raise ValueError('Stage-caption role escaped owned checkpoint objects')
            marker = owner.read_json(target / '.research_checkpoint.json')
            if (marker.get('owner') != manager['owner'] or type(marker.get('step')) is not int or
                    not 0 < marker['step'] <= completion['steps'] or
                    (name == 'last' and marker['step'] != completion['steps']) or
                    type(marker.get('created_unix')) not in (float, int) or not math.isfinite(marker['created_unix']) or
                    marker['created_unix'] > (run / 'completed.json').stat().st_mtime + 1):
                raise ValueError('Stage-caption role owner/step/completion chronology differs')
            roles[name] = {'pointer': os.readlink(pointer), 'object': str(target), 'marker': marker,
                'marker_file': owner._file(target / '.research_checkpoint.json')}
        checkpoint = Path(roles[role]['object']) / 'state.pt'
        if checkpoint.is_symlink() or not checkpoint.is_file(): raise ValueError('Retained stage payload missing; no checkpoint recreation')
        # The current raw checkpoints can include45GiB of model+Adam storage.
        # Native ZIP mmap is the reviewed path; compressed/bundle decoding needs
        # its own memory profile rather than pretending source bytes equal RAM.
        with checkpoint.open('rb') as payload:
            if payload.read(4) != b'PK\x03\x04': raise ValueError('Stage-caption currently requires a native Torch ZIP mmap payload; no implicit full decode profile')
        files = {name: owner._file(run / name) for name in
            ('resolved_config.json', 'completed.json', '.checkpoint_manager.json', 'provenance_latest.json')}
        if (run / 'provenance.json').is_file(): files['provenance.json'] = owner._file(run / 'provenance.json')
        provenance = owner.read_json(run / 'provenance_latest.json')
        expected_config = _sha(json.dumps(config, sort_keys=True, allow_nan=False).encode())
        if provenance.get('resolved_config_sha256') != expected_config:
            raise ValueError('Completed-stage source receipt is not bound to the resolved training config')
        for field, name in (('model_ref', 'model'), ('vision_ref', 'vision')):
            ref = provenance.get(field, {})
            if ref.get('repo') != config.get(name) or not re.fullmatch('[0-9a-f]{40,64}', str(ref.get('revision', ''))):
                raise ValueError('Completed-stage model/vision source revision missing or different')
        if not HEX.fullmatch(str(provenance.get('cache_fingerprint', ''))): raise ValueError('Completed-stage cache source binding missing')
        if not HEX.fullmatch(str(provenance.get('code_sha256', ''))) or not HEX.fullmatch(str(provenance.get('source_archive_sha256', ''))):
            raise ValueError('Completed-stage durable source archive commitments missing')
        archive = Path(provenance.get('source_archive', '')); archive = (archive if archive.is_absolute() else ROOT / archive).resolve(strict=True)
        if archive.stat().st_size > 32 * 2**20: raise ValueError('Stage training source archive exceeds metadata bound')
        archive_file = owner._file(archive, provenance.get('source_archive_sha256'))
        files['training_source_archive'] = archive_file
        identity = owner._checkpoint_identity(checkpoint)
    return {'run': str(run), 'role': role, 'roles': roles, 'checkpoint_path': str(checkpoint), 'artifact': identity,
        'config': config, 'files': files, 'member_stats': {item['path']: owner.stat(item['path']) for item in identity['members']},
        'stage_training_provenance': provenance, 'completed_stage': config['stage'],
        'eligibility': 'completed.json + same-owner historical BEST and exact terminal LAST; no SFT compaction precondition',
        'native_restore': 'frozen load_research_checkpoint validates P and every constructor-required learned key before strict-false alias-aware restore'}


def _model_provenance(owner, value, context):
    loaded = value.get('loaded', {}); model = context['static']['checkpoint']; config = model['config']
    marker = model['roles'][model['role']]['marker']; declared = model['stage_training_provenance']
    if (loaded.get('stage') != config['stage'] or loaded.get('recipe') != config['recipe'] or
            loaded.get('training_step') != marker['step'] or type(value.get('context_limit')) is not int or value['context_limit'] <= 256):
        raise ValueError('Stage-caption loaded stage/recipe/step/context differs')
    for field in ('model_ref', 'vision_ref'):
        if loaded.get(field) != declared[field]: raise ValueError('Stage-caption actual pinned source differs from completed-stage receipt')
    if loaded.get('cache_fingerprint') != declared['cache_fingerprint']:
        raise ValueError('Stage-caption actual cache fingerprint differs from training source')
    if set(value.get('pinned_snapshots', {})) != {'model', 'vision'} or any(not item.get('files') for item in value['pinned_snapshots'].values()):
        raise ValueError('Stage-caption actual native snapshot bindings missing')


def _private_owner():
    v2 = _private_v2(); owner = v2._private_owner()
    owner.OWNER = OWNER
    owner.SOURCES = tuple(dict.fromkeys((*owner.SOURCES, SOURCE, CLI, TEST, TEXT_SOURCE)))
    owner.checkpoint_context = lambda run, role: _checkpoint_context(owner, run, role)
    owner._model_provenance = lambda value, context: _model_provenance(owner, value, context)
    owner._checkpoint_identity = _artifact_identity
    return owner


def _binding():
    return {'owner': OWNER, 'reused_frozen_v1_source': V1_SOURCE, 'reused_frozen_v1_sha256': V1_SHA256,
        'reused_frozen_v2_source': V2_SOURCE, 'reused_frozen_v2_sha256': V2_SHA256,
        'completed_stage_eligibility_reference': TEXT_SOURCE, 'completed_stage_eligibility_source_sha256': TEXT_SHA256,
        'data_protocol_sha256': DATA_PROTOCOL_SHA256, 'preparation_profile': 'caption_pilot_en_v2_signal_safe',
        'metadata_hooks_only': ['checkpoint_context', '_model_provenance', 'streaming_checkpoint_identity_headroom_guard'],
        'generation_and_gold_policy': 'verbatim private sealed v1/v2; native B1/max256; references first opened after full372 durable predictions',
        'namespace': NAMESPACE, 'actual_primary_or_GPU_completion_claim': False}


def _verify_context(owner, context):
    static = context['static']
    if static.get('owner') != OWNER or static.get('stage_caption_profile') != _binding() or static.get('data_protocol_sha256') != DATA_PROTOCOL_SHA256:
        raise ValueError('Stage-caption context belongs to a different stage/data/owner profile')
    for name in (V1_SOURCE, V2_SOURCE, SOURCE, CLI, TEST, TEXT_SOURCE):
        if static.get('source_files', {}).get(name, {}).get('sha256') != owner.file_sha(ROOT / name):
            raise ValueError('Stage-caption source binding omitted/changed: ' + name)
    if static['checkpoint']['config'].get('stage') not in ('A0', 'A1'):
        raise ValueError('Stage-caption cannot reuse SFT or baseline-native results')


def prepare_context(run, bundle=DEFAULT_BUNDLE, *, checkpoint='best', cache_dir='cache/vision', assets=None):
    _memory_guard(); v2 = _private_v2(); owner = _private_owner(); v2._verify_bundle(owner, bundle)
    context = owner.prepare_context(run, bundle, checkpoint=checkpoint, cache_dir=cache_dir, assets=assets)
    context['static']['stage_caption_profile'] = _binding()
    context['static']['scope'] = 'internal pilot captioning at completed ' + context['static']['checkpoint']['config']['stage'] + '; separate stage/role profile, no primary completion claim'
    _verify_context(owner, context)
    return context


def output_path(context, requested=None):
    model = context['static']['checkpoint']
    expected = Path(model['run']) / 'evaluation' / NAMESPACE / model['config']['stage'] / model['role']
    if requested is not None and Path(requested).resolve() != expected.resolve(): raise ValueError('Stage-caption requires its own exact stage/role namespace; no v1/v2/SFT retag')
    if any(path.is_symlink() for path in (expected, *expected.parents)): raise ValueError('Stage-caption output cannot traverse symlinks')
    return expected


def evaluate(context, output=None, **kwargs):
    owner = _private_owner(); _verify_context(owner, context)
    _private_v2()._verify_bundle(owner, context['data']['bundle'])
    _memory_guard()
    return owner.evaluate(context, output_path(context, output), **kwargs)


def protocol_digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode())
