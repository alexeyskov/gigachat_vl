"""Pure-CPU provenance and immutable per-record journals for native B1 workers.

This module never loads models, initializes CUDA, generates predictions or
changes the frozen scorer. A separate owner must validate actual worker process
identities, runtime resources, model provenance and serial/parallel token guards.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from src.eval.research_metrics import aggregate_scores, score_prediction

ROOT = Path(__file__).resolve().parents[2]
OWNER = 'parallel_native_b1_evaluator_v1'
NAMESPACE = 'parallel_b1_v1'
VARIANTS = ('none', 'shuffle', 'blank', 'text_only')
SCOPES = ('full512_parallel', 'guard64_serial_reference')
BENCHMARKS = {'vqav2': 'general_vqa', 'gqa': 'general_vqa',
    'textvqa': 'ocr_documents', 'docvqa': 'ocr_documents',
    'chartqa': 'charts_diagrams', 'ai2d': 'charts_diagrams',
    'scienceqa': 'knowledge_reasoning', 'aokvqa': 'knowledge_reasoning'}
SCORERS = {'vqav2': 'vqa_accuracy', 'gqa': 'exact_match', 'textvqa': 'vqa_accuracy',
    'docvqa': 'anls', 'chartqa': 'chart_relaxed', 'ai2d': 'mcq_accuracy',
    'scienceqa': 'mcq_accuracy', 'aokvqa': 'mcq_accuracy'}
ROW_FIELDS = ('record_id', 'question', 'image_hashes', 'answers', 'scorer',
    'task_group', 'source', 'benchmark', 'choices', 'scoring_protocol', 'image_ablation')
GEN_FIELDS = ('prediction', 'generated_token_ids', 'generated_tokens', 'prompt_sha256',
              'prefill_tokens', 'visual_tokens', 'logits_to_keep')
PARITY_FIELDS = GEN_FIELDS
NATIVE_GENERATION = {'do_sample': False, 'num_beams': 1, 'use_cache': True,
    'max_new_tokens': 64, 'autocast': 'cuda_bf16_preserving_master_dtypes',
    'prompt': 'native_user_only_image_marker_no_gold', 'logits_to_keep': '1_when_supported'}


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode()


def protocol_sha256(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def _hex(value, length=64):
    return isinstance(value, str) and len(value) == length and all(c in '0123456789abcdef' for c in value)


def _clone(value):
    return json.loads(json_bytes(value))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def public_record(row):
    """The native generator receives no answers, choices, sources or cache tokens."""
    return _clone({key: row[key] for key in ('record_id', 'question', 'image_paths', 'image_hashes') if key in row})


def validate_records(records, *, original=True):
    if not isinstance(records, list) or len(records) != 512:
        raise ValueError('Exactly512 original-order records are required')
    ids = [row.get('record_id') for row in records]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != 512:
        raise ValueError('All512 record IDs must be unique nonempty strings')
    if Counter(row.get('benchmark') for row in records) != Counter({key: 64 for key in BENCHMARKS}):
        raise ValueError('Exactly64 records from each of the eight benchmarks are required')
    for row in records:
        bench = row['benchmark']
        if row.get('task_group') != BENCHMARKS[bench] or row.get('scorer') != SCORERS[bench]:
            raise ValueError('Benchmark/group/scorer mapping differs')
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError('Question is empty')
        answers = row.get('answers')
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) for a in answers):
            raise ValueError('String references are required')
        if row['scorer'] == 'vqa_accuracy':
            expected = 'vqav2_official_legacy' if bench == 'vqav2' else 'textvqa_evalai'
            if len(answers) != 10 or row.get('scoring_protocol') != expected:
                raise ValueError('VQA requires ten actual references and the declared normalization protocol')
        if bench == 'gqa' and len(answers) != 1:
            raise ValueError('GQA requires its single reference')
        if row['scorer'] == 'mcq_accuracy' and (not isinstance(row.get('choices'), list) or
                not row['choices'] or any(not isinstance(x, str) for x in row['choices'])):
            raise ValueError('MCQ choices are missing')
        source = row.get('source')
        if (not isinstance(source, dict) or source.get('benchmark') != bench or
                not source.get('repo') or not _hex(source.get('revision'), 40) or
                not source.get('split') or not source.get('shard')):
            raise ValueError('Pinned source/split/shard provenance is missing')
        paths = row.get('image_paths'); hashes = row.get('image_hashes')
        if (not isinstance(paths, list) or not isinstance(hashes, list) or len(paths) != len(hashes) or
                len(paths) > 1 or (original and len(paths) != 1) or
                any(not isinstance(p, str) or not p for p in paths) or any(not _hex(h) for h in hashes)):
            raise ValueError('Expected one original image or explicitly transformed text-only image fields')
    json_bytes(records)


def select_guard_indices(records):
    validate_records(records, original=False)
    seen = Counter(); selected = []
    for index, row in enumerate(records):
        bench = row['benchmark']
        if seen[bench] < 8:
            selected.append(index); seen[bench] += 1
    return selected


def assigned_indices(worker_index, workers=2, total=512):
    if type(workers) is not int or workers not in (2, 4) or type(worker_index) is not int or not 0 <= worker_index < workers:
        raise ValueError('Only two or four disjoint nativeB1 workers are supported')
    if type(total) is not int or total != 512:
        raise ValueError('Disjoint assignment is defined on the complete512 manifest')
    return list(range(worker_index, total, workers))


def _checkpoint(identity):
    if (not isinstance(identity, dict) or not _hex(identity.get('sha256')) or
            identity.get('components_fully_hashed') is not True or
            identity.get('format') not in ('legacy_single_file', 'bundle_v1') or
            not isinstance(identity.get('members'), list) or not identity['members'] or
            not Path(identity.get('primary_path', '')).is_absolute()):
        raise ValueError('Fully hashed compound checkpoint identity is required')
    for member in identity['members']:
        if (not isinstance(member, dict) or not _hex(member.get('sha256')) or
                not Path(member.get('path', '')).is_absolute() or
                not isinstance(member.get('role'), str) or not member['role'] or
                not isinstance(member.get('relative_path'), str) or not member['relative_path'] or
                Path(member['relative_path']).is_absolute() or '..' in Path(member['relative_path']).parts or
                type(member.get('bytes')) is not int or member['bytes'] <= 0 or
                type(member.get('mtime_ns')) is not int or member['mtime_ns'] <= 0):
            raise ValueError('Checkpoint member byte/stat identity is malformed')
    members = identity['members']
    if (len({m['path'] for m in members}) != len(members) or members[0]['path'] != identity['primary_path'] or
            identity.get('primary_bytes') != members[0]['bytes'] or
            identity.get('primary_mtime_ns') != members[0]['mtime_ns'] or
            identity.get('total_bytes') != sum(m['bytes'] for m in members)):
        raise ValueError('Compound checkpoint primary/member totals disagree')
    if identity['format'] == 'legacy_single_file':
        if len(members) != 1 or identity['sha256'] != members[0]['sha256']:
            raise ValueError('Legacy checkpoint content digest differs from its actual member')
    else:
        stable = {'format': 'bundle_v1', 'members': [{k: m[k] for k in
            ('role', 'relative_path', 'bytes', 'sha256')} for m in members]}
        expected = hashlib.sha256(b'RESEARCH_CHECKPOINT_ARTIFACT_V1\0' +
            json.dumps(stable, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if identity['sha256'] != expected:
            raise ValueError('Bundle identity must cover all component bytes, not just its manifest')


def _make_protocol(records, *, checkpoint_identity, manifest_sha256, transformed_manifest_sha256,
        source_sha256, model_provenance, worker_profiles, variant, seed=42, environment=None,
        eos_token_ids=(), scope='full512_parallel', reference_binding=None):
    validate_records(records, original=variant == 'none')
    _checkpoint(checkpoint_identity)
    if variant not in VARIANTS or type(seed) is not int or seed != 42 or scope not in SCOPES:
        raise ValueError('Fixed variants,seed42 and explicit evaluation scope are required')
    if any((not row['image_paths']) != (variant == 'text_only') for row in records):
        raise ValueError('Only text_only may remove the image, and it must remove every image')
    if variant != 'none' and any(row.get('image_ablation', {}).get('variant') != variant or
            row.get('image_ablation', {}).get('seed') != 42 for row in records):
        raise ValueError('Every effective row must declare the same image intervention and seed')
    if not _hex(manifest_sha256) or not _hex(transformed_manifest_sha256):
        raise ValueError('Source and transformed manifest byte hashes are required')
    if variant == 'none' and transformed_manifest_sha256 != manifest_sha256:
        raise ValueError('None must retain the source manifest bytes')
    required_sources = ('src/eval/research_parallel_protocol.py', 'src/eval/research_metrics.py',
                        'src/eval/research_inference.py', 'scripts/eval_research_checkpoint.py')
    if not isinstance(source_sha256, dict) or any(name not in source_sha256 for name in required_sources):
        raise ValueError('Tested source SHA bindings are required')
    for name, value in source_sha256.items():
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts or not _hex(value):
            raise ValueError('Source SHA binding is unsafe/malformed')
    if (not isinstance(model_provenance, dict) or not model_provenance.get('recipe') or
            model_provenance.get('stage') != 'SFT' or not model_provenance.get('model_ref') or
            not model_provenance.get('vision_ref') or type(model_provenance.get('training_step')) is not int or
            model_provenance['training_step'] < 1 or
            not _hex(model_provenance.get('cache_fingerprint')) or
            any(not isinstance(model_provenance.get(key), dict) or
                not model_provenance[key].get('repo') or not _hex(model_provenance[key].get('revision'), 40)
                for key in ('model_ref', 'vision_ref'))):
        raise ValueError('Explicit loaded-model provenance is required')
    allowed = (2, 4) if scope == 'full512_parallel' else (1,)
    if not isinstance(worker_profiles, list) or len(worker_profiles) not in allowed:
        raise ValueError('Scope requires two/four parallel workers or one named serial reference worker')
    for index, profile in enumerate(worker_profiles):
        if (not isinstance(profile, dict) or profile.get('worker_index') != index or
                type(profile.get('worker_index')) is not int or profile.get('batch_size') != 1 or
                type(profile.get('batch_size')) is not int):
            raise ValueError('Each worker profile must declare its index and native batch1')
    if (not isinstance(eos_token_ids, (tuple, list)) or
            any(type(value) is not int or value < 0 for value in eos_token_ids)):
        raise ValueError('Native EOS policy must contain nonnegative token IDs')
    indices = list(range(512)) if scope == 'full512_parallel' else select_guard_indices(records)
    generation = dict(NATIVE_GENERATION, batch_size=1, eos_token_ids=sorted(set(eos_token_ids)),
        padding='native_single_record_no_batch_padding', no_batch_fallback=True)
    return _clone({'schema_version': 1, 'owner': OWNER, 'namespace': NAMESPACE, 'scope': scope,
        'checkpoint_sha256': checkpoint_identity['sha256'], 'checkpoint_artifact_identity': checkpoint_identity,
        'manifest_sha256': manifest_sha256, 'transformed_manifest_sha256': transformed_manifest_sha256,
        'manifest_records': 512, 'evaluation_records': len(indices), 'variant': variant, 'seed': seed,
        'source_sha256': source_sha256, 'model_provenance': model_provenance, 'environment': environment or {},
        'worker_profiles': worker_profiles, 'workers': len(worker_profiles),
        'assignment': 'global_index_mod_workers' if scope == 'full512_parallel' else 'named_serial_guard64',
        'selected_global_indices': indices, 'selected_record_ids': [records[i]['record_id'] for i in indices],
        'record_sha256': [protocol_sha256(row) for row in records], 'generation': generation,
        'reference_binding': reference_binding})


def make_protocol(records, *, checkpoint_identity, manifest_sha256, transformed_manifest_sha256,
        source_sha256, model_provenance, worker_profiles, variant, seed=42, environment=None,
        eos_token_ids=(), transformed_manifest_bytes=None, scope='full512_parallel', reference_binding=None):
    if transformed_manifest_bytes is not None:
        if (not isinstance(transformed_manifest_bytes, bytes) or
                hashlib.sha256(transformed_manifest_bytes).hexdigest() != transformed_manifest_sha256 or
                [json.loads(x) for x in transformed_manifest_bytes.splitlines() if x.strip()] != records):
            raise ValueError('Actual transformed manifest bytes/hash/records differ')
    elif variant != 'none':
        raise ValueError('Non-none variants require the actual full512 transformed manifest bytes')
    return _make_protocol(records, checkpoint_identity=checkpoint_identity,
        manifest_sha256=manifest_sha256, transformed_manifest_sha256=transformed_manifest_sha256,
        source_sha256=source_sha256, model_provenance=model_provenance, worker_profiles=worker_profiles,
        variant=variant, seed=seed, environment=environment, eos_token_ids=eos_token_ids,
        scope=scope, reference_binding=reference_binding)


def _validate_protocol_records(records, protocol):
    validate_records(records, original=protocol.get('variant') == 'none')
    if (protocol.get('owner') != OWNER or protocol.get('namespace') != NAMESPACE or
            protocol.get('scope') not in SCOPES or protocol.get('record_sha256') != [protocol_sha256(r) for r in records]):
        raise ValueError('Protocol owner/scope/effective record identities differ')
    # Reconstruct every declarative field rather than trusting selected indices.
    rebuilt = _make_protocol(records, checkpoint_identity=protocol['checkpoint_artifact_identity'],
        manifest_sha256=protocol['manifest_sha256'], transformed_manifest_sha256=protocol['transformed_manifest_sha256'],
        source_sha256=protocol['source_sha256'], model_provenance=protocol['model_provenance'],
        worker_profiles=protocol['worker_profiles'], variant=protocol['variant'], seed=protocol['seed'],
        environment=protocol['environment'], eos_token_ids=protocol['generation']['eos_token_ids'],
        scope=protocol['scope'], reference_binding=protocol.get('reference_binding'))
    if rebuilt != protocol:
        raise ValueError('Reconstructed nativeB1 protocol differs')


def _safe_path(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('Journal paths may not follow symlinks')
    return path


def _commit_bytes(path, value):
    """Hard-link a fully fsynced temp file; never replace an existing commit."""
    path = _safe_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def init_journal(output, protocol, *, resume=False):
    output = _safe_path(output); output.mkdir(parents=True, exist_ok=True)
    path = output / 'protocol.json'
    if path.exists():
        if not resume or json.loads(path.read_bytes()) != protocol:
            raise ValueError('Existing journal needs explicit resume of the exact immutable protocol')
    else:
        if any(output.iterdir()):
            raise ValueError('Orphan output without its protocol is refused')
        _commit_bytes(path, json_bytes(protocol) + b'\n')
    return protocol_sha256(protocol)


def _journal_protocol(output, protocol):
    path = _safe_path(Path(output) / 'protocol.json')
    if not path.is_file() or json.loads(path.read_bytes()) != protocol:
        raise ValueError('Journal source/checkpoint/manifest/variant/worker/generation protocol differs')


def validate_generated(record, generated, *, eos_token_ids=()):
    ids = generated.get('generated_token_ids')
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 64 or
            any(type(value) is not int or value < 0 for value in ids) or
            generated.get('generated_tokens') != len(ids) or
            type(generated.get('generated_tokens')) is not int or
            not isinstance(generated.get('prediction'), str) or not _hex(generated.get('prompt_sha256')) or
            type(generated.get('prefill_tokens')) is not int or generated['prefill_tokens'] < 1 or
            type(generated.get('visual_tokens')) is not int or generated['visual_tokens'] < 0 or
            generated['visual_tokens'] > generated['prefill_tokens'] or
            generated.get('logits_to_keep') not in (None, 1) or isinstance(generated.get('logits_to_keep'), bool)):
        raise ValueError('Generated nativeB1 tokens/prompt metadata is malformed')
    if any(value in eos_token_ids for value in ids[:-1]):
        raise ValueError('Native output contains post-EOS tokens')
    if bool(record.get('image_paths')) != (generated['visual_tokens'] > 0):
        raise ValueError('Native visual-token count disagrees with the effective images')


def _worker_for(index, protocol):
    return index % protocol['workers'] if protocol['scope'] == 'full512_parallel' else 0


def _validate_row(row, record, index, protocol):
    sentinel = object()
    if any(row.get(key, sentinel) != record.get(key, sentinel) for key in ROW_FIELDS):
        raise ValueError('Committed row semantic/scorer/source/image fields differ')
    if (row.get('protocol_sha256') != protocol_sha256(protocol) or row.get('parallel_owner') != OWNER or
            row.get('global_index') != index or type(row.get('global_index')) is not int or
            row.get('worker_index') != _worker_for(index, protocol) or type(row.get('worker_index')) is not int):
        raise ValueError('Committed owner/protocol/global index/disjoint worker assignment differs')
    validate_generated(record, row, eos_token_ids=protocol['generation']['eos_token_ids'])
    actual = score_prediction(record, row['prediction'])
    if type(row.get('score')) not in (int, float) or not math.isfinite(row['score']) or row['score'] != actual:
        raise ValueError('Committed raw score does not reproduce with the frozen scorer')


def commit_record(output, index, record, generated, protocol, *, worker_index):
    _journal_protocol(output, protocol)
    if (type(index) is not int or index not in protocol['selected_global_indices'] or
            type(worker_index) is not int or worker_index != _worker_for(index, protocol) or
            protocol_sha256(record) != protocol['record_sha256'][index]):
        raise ValueError('Row index/effective record/worker is outside the declared disjoint assignment')
    sentinel = object()
    if any(key in generated and generated[key] != record.get(key, sentinel) for key in ROW_FIELDS):
        raise ValueError('Generator attempted to replace semantic/reference fields')
    row = {key: record[key] for key in ROW_FIELDS if key in record}
    row.update({key: generated[key] for key in GEN_FIELDS if key in generated})
    row.update(score=score_prediction(record, row.get('prediction')), protocol_sha256=protocol_sha256(protocol),
               parallel_owner=OWNER, global_index=index, worker_index=worker_index)
    _validate_row(row, record, index, protocol)
    envelope = {'schema_version': 1, 'owner': OWNER, 'protocol_sha256': protocol_sha256(protocol),
        'global_index': index, 'record_id': record['record_id'], 'worker_index': worker_index,
        'payload_sha256': protocol_sha256(row), 'row': row}
    _commit_bytes(Path(output) / 'prediction_records' / f'row-{index:04d}.json', json_bytes(envelope) + b'\n')
    return row


def read_committed(output, records, protocol):
    _validate_protocol_records(records, protocol); _journal_protocol(output, protocol)
    directory = _safe_path(Path(output) / 'prediction_records'); result = {}
    if not directory.exists(): return result
    expected = {f'row-{index:04d}.json': index for index in protocol['selected_global_indices']}
    for path in sorted(directory.iterdir()):
        if path.name.startswith('.') and path.name.endswith('.tmp'):
            continue  # An interrupted, never-committed temporary write is not a prediction.
        if path.name not in expected or not _safe_path(path).is_file():
            raise ValueError('Unexpected or unsafe journal shard')
        index = expected[path.name]; envelope = json.loads(path.read_bytes()); row = envelope.get('row')
        if (envelope.get('schema_version') != 1 or envelope.get('owner') != OWNER or
                envelope.get('protocol_sha256') != protocol_sha256(protocol) or
                envelope.get('global_index') != index or type(envelope.get('global_index')) is not int or
                envelope.get('record_id') != records[index]['record_id'] or
                envelope.get('worker_index') != _worker_for(index, protocol) or
                type(envelope.get('worker_index')) is not int or
                not isinstance(row, dict) or envelope.get('payload_sha256') != protocol_sha256(row)):
            raise ValueError('Immutable committed envelope/checksum/index/record differs')
        _validate_row(row, records[index], index, protocol); result[index] = row
    return result


def pending_indices(records, protocol, committed, worker_index):
    _validate_protocol_records(records, protocol)
    if type(worker_index) is not int or not 0 <= worker_index < protocol['workers']:
        raise ValueError('Unknown worker index')
    if not isinstance(committed, dict) or any(type(i) is not int or i not in protocol['selected_global_indices'] for i in committed):
        raise ValueError('Committed indices are outside the fixed manifest selection')
    for index, row in committed.items(): _validate_row(row, records[index], index, protocol)
    return [i for i in protocol['selected_global_indices'] if _worker_for(i, protocol) == worker_index and i not in committed]


def compare_reference(rows, reference):
    reference = reference['rows'] if isinstance(reference, dict) else reference
    if not isinstance(reference, list) or len({r['record_id'] for r in reference}) != len(reference):
        raise ValueError('Reference cache has duplicate/malformed records')
    by_id = {r['record_id']: r for r in reference}; mismatches = []
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Parallel comparison contains duplicate rows')
    for row in rows:
        previous = by_id.get(row['record_id'])
        if previous is None or any(previous.get(k) != row.get(k) for k in (*ROW_FIELDS, *PARITY_FIELDS)):
            mismatches.append(row['record_id'])
    return {'status': 'passed' if not mismatches else 'failed', 'compared_records': len(rows),
        'exact_generated_token_ids_and_prompt_metadata': not mismatches, 'mismatched_record_ids': mismatches}


def _write_or_verify(path, raw):
    if path.exists():
        if _safe_path(path).read_bytes() != raw:
            raise ValueError('Completed merged view/summary is immutable and differs from its verified journal')
    else: _commit_bytes(path, raw)


def merge_completed(output, records, protocol, *, reference=None):
    committed = read_committed(output, records, protocol)
    indices = protocol['selected_global_indices']
    if set(committed) != set(indices):
        raise ValueError('Only all declared records exactly once can be merged/scored')
    ordered = [committed[i] for i in indices]
    comparison = compare_reference(ordered, reference) if reference is not None else None
    if comparison is not None and comparison['status'] != 'passed':
        raise ValueError('Native reference token/prompt/semantic parity failed; no completed summary')
    metrics = aggregate_scores(ordered, require_four_groups=True)
    raw = b''.join(json_bytes(row) + b'\n' for row in ordered)
    directory = Path(output) / 'prediction_records'
    summary = dict(metrics, schema_version=1, owner=OWNER, scope=protocol['scope'],
        complete_manifest=protocol['scope'] == 'full512_parallel', complete_selected_records=True,
        manifest_records=512, selected_records=len(indices), protocol_sha256=protocol_sha256(protocol),
        checkpoint_sha256=protocol['checkpoint_sha256'], checkpoint_artifact_identity=protocol['checkpoint_artifact_identity'],
        manifest_sha256=protocol['manifest_sha256'], transformed_manifest_sha256=protocol['transformed_manifest_sha256'],
        variant=protocol['variant'], seed=42, model_provenance=protocol['model_provenance'],
        raw_predictions_sha256=hashlib.sha256(raw).hexdigest(),
        journal_shards={str(i): file_sha256(directory / f'row-{i:04d}.json') for i in indices},
        native_reference_comparison=comparison, original_outputs_modified=False)
    _write_or_verify(Path(output) / 'predictions.jsonl', raw)
    _write_or_verify(Path(output) / 'summary.json', json_bytes(summary) + b'\n')
    return ordered, summary


def _rgb_hash(path):
    # Same canonical RGB policy as frozen src/dataset/vision_cache.py, without
    # importing the dataset package (which would import Torch in a parent).
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert('RGB')
        value = hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode())
        value.update(rgb.tobytes()); return value.hexdigest()


def validate_native_cache(directory, records, protocol, *, scope='full512', root=ROOT,
                          verify_images=True):
    """Validate a preserved full native cache or our named serial64 reference.

    A native blank cache may live under another directory: its own transformed
    bytes are verified, while effective semantic/RGB identities match rather
    than requiring its absolute image paths to equal the new namespace paths.
    """
    _validate_protocol_records(records, protocol)
    directory = Path(directory); root = Path(root)
    for name, expected_hash in protocol['source_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path) != expected_hash:
            raise ValueError('Current reference-validation source binding changed')
    paths = {name: directory / name for name in ('protocol.json', 'summary.json', 'predictions.jsonl')}
    before = {name: file_sha256(path) for name, path in paths.items()}
    parent = json.loads(paths['protocol.json'].read_bytes()); summary = json.loads(paths['summary.json'].read_bytes())
    raw = read_rows(paths['predictions.jsonl'])
    if scope == 'guard64':
        if parent.get('scope') != 'guard64_serial_reference':
            raise ValueError('Guard64 references must use the named own serial-cache scope')
        for key in ('owner', 'checkpoint_artifact_identity', 'manifest_sha256', 'source_sha256',
                    'model_provenance', 'environment', 'variant', 'seed', 'generation'):
            if parent.get(key) != protocol.get(key):
                raise ValueError('Serial guard reference provenance differs')
        committed, rebuilt = merge_completed(directory, records, parent)
        if rebuilt != summary or raw != committed:
            raise ValueError('Serial reference raw summary/journal differs')
        selected = select_guard_indices(records); parent_sha = protocol_sha256(parent)
        effective = records
    elif scope == 'full512':
        selected = list(range(512)); parent_sha = hashlib.sha256(json.dumps(parent, sort_keys=True).encode()).hexdigest()
        ablation = parent.get('image_ablation', {})
        generation = parent.get('generation', {})
        if (parent.get('checkpoint_sha256') != protocol['checkpoint_sha256'] or
                parent.get('manifest_sha256') != protocol['manifest_sha256'] or parent.get('manifest_records') != 512 or
                parent.get('selected_record_ids_sha256') != hashlib.sha256(json.dumps([r['record_id'] for r in records]).encode()).hexdigest() or
                ablation.get('source_manifest_sha256') != protocol['manifest_sha256'] or
                ablation.get('variant') != protocol['variant'] or ablation.get('seed') != 42 or
                generation != NATIVE_GENERATION or parent.get('profile') is not None):
            raise ValueError('Preserved native full512 checkpoint/manifest/variant/generation protocol differs')
        source = parent.get('source_sha256', {})
        if not source or 'src/eval/research_metrics.py' not in source:
            raise ValueError('Native parent source binding is missing')
        for name, expected in source.items():
            path = (root / name).resolve()
            if (Path(name).is_absolute() or not path.is_relative_to(root.resolve()) or
                    protocol['source_sha256'].get(name) != expected or file_sha256(path) != expected):
                raise ValueError('Frozen native reference source changed')
        native_environment = parent.get('environment', {})
        if (not isinstance(native_environment, dict) or set(native_environment) != {'python', 'packages'} or
                any(protocol['environment'].get(k) != v for k, v in native_environment.items())):
            raise ValueError('Native reference Python/package environment differs')
        provenance_path = directory / 'model_provenance.json'; paths['model_provenance.json'] = provenance_path
        before['model_provenance.json'] = file_sha256(provenance_path)
        if json.loads(provenance_path.read_bytes()) != protocol['model_provenance']:
            raise ValueError('Preserved loaded-model provenance differs')
        effective = records
        if protocol['variant'] != 'none':
            path = directory / 'transformed_manifest.jsonl'; paths['transformed_manifest.jsonl'] = path
            before['transformed_manifest.jsonl'] = file_sha256(path)
            if before['transformed_manifest.jsonl'] != ablation.get('transformed_manifest_sha256'):
                raise ValueError('Native reference own transformed manifest bytes differ')
            effective = read_rows(path); validate_records(effective, original=False)
            if any(any(a.get(k) != b.get(k) for k in ROW_FIELDS) for a, b in zip(records, effective)):
                raise ValueError('Native transformed semantic/RGB identities differ from the requested full transformation')
        elif ablation.get('transformed_manifest_sha256') != protocol['manifest_sha256']:
            raise ValueError('Native none transformed bytes differ')
    else:
        raise ValueError('Reference scope must be explicitly full512 or guard64')
    expected = [effective[i] for i in selected]
    if len(raw) != len(selected) or [r.get('record_id') for r in raw] != [r['record_id'] for r in expected]:
        raise ValueError('Native cache coverage/order/unique record IDs differ')
    image_checks = {}
    for record, generated in zip(expected, raw):
        if (generated.get('protocol_sha256') != parent_sha or
                any(generated.get(k) != record.get(k) for k in ROW_FIELDS)):
            raise ValueError('Native cached row source/question/answers/image/score provenance differs')
        validate_generated(record, generated, eos_token_ids=protocol['generation']['eos_token_ids'])
        if type(generated.get('score')) not in (int, float) or generated['score'] != score_prediction(record, generated['prediction']):
            raise ValueError('Native cached score differs from frozen raw scoring')
        if verify_images:
            for path, expected_hash in zip(record['image_paths'], record['image_hashes']):
                if path not in image_checks: image_checks[path] = _rgb_hash(path)
                if image_checks[path] != expected_hash: raise ValueError('Native reference canonical RGB identity changed')
    metrics = aggregate_scores(raw, require_four_groups=True)
    if (summary.get('protocol_sha256') != parent_sha or any(summary.get(k) != v for k, v in metrics.items()) or
            summary.get('checkpoint_sha256') != protocol['checkpoint_sha256'] or
            summary.get('manifest_sha256') != protocol['manifest_sha256'] or
            (scope == 'full512' and (summary.get('complete_manifest') is not True or summary.get('manifest_records') != 512 or
                                   summary.get('image_ablation') != parent['image_ablation']))):
        raise ValueError('Native summary is not the complete recomputed raw score view')
    if any(file_sha256(paths[name]) != value for name, value in before.items()):
        raise ValueError('Native reference changed during read-only validation')
    binding = {'scope': scope, 'directory': str(directory.resolve()), 'files_sha256': before,
        'checkpoint_artifact_identity': protocol['checkpoint_artifact_identity'],
        'manifest_sha256': protocol['manifest_sha256'], 'variant': protocol['variant'], 'seed': 42,
        'parent_transformed_manifest_sha256': parent.get('transformed_manifest_sha256',
            parent.get('image_ablation', {}).get('transformed_manifest_sha256')),
        'requested_transformed_manifest_sha256': protocol['transformed_manifest_sha256'],
        'record_ids': [row['record_id'] for row in raw], 'raw_rows_canonical_sha256': protocol_sha256(raw)}
    return {'status': 'passed', 'scope': scope, 'records': len(raw), 'rows': raw,
            'files_sha256': before, 'binding': binding, 'model_provenance': protocol['model_provenance'],
            'canonical_RGB_paths_verified': len(image_checks), 'original_cache_modified': False}
