"""No-model CPU tests for fixed nativeB1 scope, immutable journals and references."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image

from src.eval import research_parallel_protocol as p
from src.eval.research_metrics import aggregate_scores, score_prediction


@pytest.fixture
def data(tmp_path):
    image = tmp_path / 'image.png'; Image.new('RGB', (3, 2), 'red').save(image)
    image_hash = p._rgb_hash(image); records = []
    for benchmark, group in p.BENCHMARKS.items():
        for index in range(64):
            row = dict(record_id=f'{benchmark}-{index}', question='What is visible? A. yes B. no',
                benchmark=benchmark, task_group=group, scorer=p.SCORERS[benchmark],
                image_paths=[str(image)], image_hashes=[image_hash], answers=['yes'],
                source=dict(repo='fixture/heldout', revision='c' * 40, split='test',
                            shard='test.parquet', benchmark=benchmark, mirror_row_index=index))
            if benchmark in ('vqav2', 'textvqa'):
                row['answers'] = ['yes'] * 10
                row['scoring_protocol'] = 'vqav2_official_legacy' if benchmark == 'vqav2' else 'textvqa_evalai'
            if p.SCORERS[benchmark] == 'mcq_accuracy':
                row['choices'] = ['yes', 'no']; row['answers'] = ['A']
            records.append(row)
    source_names = ['src/eval/research_parallel_protocol.py', 'src/eval/research_metrics.py',
                    'src/eval/research_inference.py', 'scripts/eval_research_checkpoint.py']
    sources = {}
    for name in source_names:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('source fixture ' + name); sources[name] = p.file_sha256(path)
    checkpoint = tmp_path / 'state.pt'; checkpoint.write_bytes(b'weights fixture')
    stat = checkpoint.stat(); sha = p.file_sha256(checkpoint)
    identity = dict(format='legacy_single_file', sha256=sha, primary_path=str(checkpoint),
        primary_bytes=stat.st_size, primary_mtime_ns=stat.st_mtime_ns, total_bytes=stat.st_size,
        components_fully_hashed=True, members=[dict(role='state', relative_path='state.pt',
            path=str(checkpoint), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha)])
    provenance = dict(recipe='4q', stage='SFT', training_step=100,
        model_ref=dict(repo='fixture/model', revision='a' * 40),
        vision_ref=dict(repo='fixture/vision', revision='b' * 40), cache_fingerprint='e' * 64)
    raw = b''.join(p.json_bytes(r) + b'\n' for r in records)
    manifest_sha = hashlib.sha256(raw).hexdigest()
    def protocol(*, rows=None, workers=2, variant='none', scope='full512_parallel', **extra):
        rows = records if rows is None else rows
        transformed = b''.join(p.json_bytes(r) + b'\n' for r in rows)
        return p.make_protocol(rows, checkpoint_identity=identity, manifest_sha256=manifest_sha,
            transformed_manifest_sha256=hashlib.sha256(transformed).hexdigest(), source_sha256=sources,
            model_provenance=provenance, worker_profiles=[dict(worker_index=i, batch_size=1,
                pytorch_cap_bytes=8 * 2**30, sampled_nvml_cap_bytes=12 * 2**30) for i in range(workers)],
            variant=variant, scope=scope, eos_token_ids=[3], transformed_manifest_bytes=transformed,
            environment={'python': 'fixture Python', 'packages': {'transformers': 'fixture'}}, **extra)
    return dict(root=tmp_path, records=records, identity=identity, sources=sources,
                provenance=provenance, protocol=protocol, manifest_sha=manifest_sha)


def generated(record):
    return dict(prediction='yes', generated_token_ids=[7, 3], generated_tokens=2,
                prompt_sha256='d' * 64, prefill_tokens=10,
                visual_tokens=4 if record['image_paths'] else 0, logits_to_keep=1)


def commit_all(output, records, protocol):
    for index in reversed(protocol['selected_global_indices']):
        p.commit_record(output, index, records[index], generated(records[index]), protocol,
                        worker_index=index % protocol['workers'] if protocol['scope'] == 'full512_parallel' else 0)


def native_cache(directory, data, records, protocol):
    directory.mkdir(parents=True)
    native_sources = {key: value for key, value in data['sources'].items()
                      if key != 'src/eval/research_parallel_protocol.py'}
    transformed = b''.join(p.json_bytes(r) + b'\n' for r in records)
    native = dict(schema_version=1, checkpoint_sha256=protocol['checkpoint_sha256'],
        manifest_sha256=protocol['manifest_sha256'], manifest_records=512,
        selected_record_ids_sha256=hashlib.sha256(json.dumps([r['record_id'] for r in records]).encode()).hexdigest(),
        generation=p.NATIVE_GENERATION, source_sha256=native_sources,
        environment=protocol['environment'], image_ablation=dict(variant=protocol['variant'], seed=42,
            source_manifest_sha256=protocol['manifest_sha256'],
            transformed_manifest_sha256=hashlib.sha256(transformed).hexdigest()))
    parent_sha = hashlib.sha256(json.dumps(native, sort_keys=True).encode()).hexdigest()
    rows = []
    for record in records:
        row = {k: record[k] for k in p.ROW_FIELDS if k in record}
        row.update(generated(record), score=score_prediction(record, 'yes'), protocol_sha256=parent_sha)
        rows.append(row)
    summary = dict(aggregate_scores(rows), complete_manifest=True, manifest_records=512,
                   protocol_sha256=parent_sha, checkpoint_sha256=protocol['checkpoint_sha256'],
                   manifest_sha256=protocol['manifest_sha256'], image_ablation=native['image_ablation'])
    for name, value in [('protocol.json', native), ('summary.json', summary),
                        ('model_provenance.json', protocol['model_provenance'])]:
        (directory / name).write_text(json.dumps(value))
    (directory / 'predictions.jsonl').write_bytes(b''.join(p.json_bytes(r) + b'\n' for r in rows))
    if protocol['variant'] != 'none':
        (directory / 'transformed_manifest.jsonl').write_bytes(transformed)
    return rows


def test_full512_scope_guard64_order_and_gold_filtered_inputs(data):
    records = data['records']; p.validate_records(records)
    indices = p.select_guard_indices(records)
    assert indices == [i for i in range(512) if i % 64 < 8]
    changed = copy.deepcopy(records)
    for row in changed: row['answers'] = ['secret'] * len(row['answers'])
    assert p.select_guard_indices(changed) == indices
    public = p.public_record(dict(records[0], prediction='gold', generated_token_ids=[1]))
    assert set(public) == {'record_id', 'question', 'image_paths', 'image_hashes'}
    public['image_hashes'][0] = 'tamper'
    assert records[0]['image_hashes'][0] != 'tamper'


@pytest.mark.parametrize('kind', ['short', 'duplicate', 'group', 'benchmark', 'scorer',
                                 'vqa_refs', 'vqa_protocol', 'source_revision', 'image', 'question'])
def test_malformed_official_scope_and_reference_mapping_rejected(data, kind):
    rows = copy.deepcopy(data['records'])
    if kind == 'short': rows.pop()
    if kind == 'duplicate': rows[1]['record_id'] = rows[0]['record_id']
    if kind == 'group': rows[0]['task_group'] = 'other'
    if kind == 'benchmark': rows[0]['benchmark'] = 'other'
    if kind == 'scorer': rows[0]['scorer'] = 'exact_match'
    if kind == 'vqa_refs': rows[0]['answers'].pop()
    if kind == 'vqa_protocol': rows[0]['scoring_protocol'] = 'textvqa_evalai'
    if kind == 'source_revision': rows[0]['source']['revision'] = 'main'
    if kind == 'image': rows[0]['image_hashes'] = []
    if kind == 'question': rows[0]['question'] = ''
    with pytest.raises(ValueError): p.validate_records(rows)


@pytest.mark.parametrize('workers', [2, 4])
def test_exact_disjoint_assignments_cover512_once(workers):
    parts = [p.assigned_indices(i, workers) for i in range(workers)]
    assert sorted(sum(parts, [])) == list(range(512))
    assert all(not (set(a) & set(b)) for i, a in enumerate(parts) for b in parts[i + 1:])


@pytest.mark.parametrize('workers', [0, 1, 3, 8, True])
def test_no_implicit_worker_count_fallback(workers):
    with pytest.raises(ValueError): p.assigned_indices(0, workers)


@pytest.mark.parametrize('workers', [2, 4])
def test_unordered_partial_commits_resume_only_pending_and_merge_original_order(data, workers):
    records = data['records']; protocol = data['protocol'](workers=workers); output = data['root'] / 'out'
    p.init_journal(output, protocol)
    for index in [9, 2, 7, 0]:
        p.commit_record(output, index, records[index], generated(records[index]), protocol, worker_index=index % workers)
    before = {path.name: path.read_bytes() for path in (output / 'prediction_records').iterdir()}
    p.init_journal(output, protocol, resume=True)
    committed = p.read_committed(output, records, protocol)
    pending = [p.pending_indices(records, protocol, committed, i) for i in range(workers)]
    assert sorted(sum(pending, [])) == [i for i in range(512) if i not in committed]
    for worker_index, indices in enumerate(pending):
        for index in indices:
            p.commit_record(output, index, records[index], generated(records[index]), protocol, worker_index=worker_index)
    rows, summary = p.merge_completed(output, records, protocol)
    assert [r['record_id'] for r in rows] == [r['record_id'] for r in records]
    assert summary['records'] == 512 and summary['complete_manifest'] is True
    assert summary['group_macro_score'] == 100
    assert all((output / 'prediction_records' / name).read_bytes() == value for name, value in before.items())
    assert p.merge_completed(output, records, protocol) == (rows, summary)


def test_atomic_row_race_commits_exactly_once_and_never_overwrites(data):
    rows = data['records']; protocol = data['protocol'](); output = data['root'] / 'race'
    p.init_journal(output, protocol)
    def commit(_):
        try: p.commit_record(output, 0, rows[0], generated(rows[0]), protocol, worker_index=0); return True
        except FileExistsError: return False
    with ThreadPoolExecutor(max_workers=8) as pool: result = list(pool.map(commit, range(8)))
    assert sum(result) == 1
    assert set(p.read_committed(output, rows, protocol)) == {0}
    assert not list((output / 'prediction_records').glob('.*.tmp'))


@pytest.mark.parametrize('change', ['owner', 'source', 'checkpoint', 'manifest', 'generation', 'indices', 'workers'])
def test_resume_rejects_every_mixed_protocol_identity(data, change):
    protocol = data['protocol'](); output = data['root'] / 'resume'; p.init_journal(output, protocol)
    changed = copy.deepcopy(protocol)
    if change == 'owner': changed['owner'] = 'foreign'
    if change == 'source': changed['source_sha256']['src/eval/research_metrics.py'] = 'a' * 64
    if change == 'checkpoint': changed['checkpoint_artifact_identity']['members'][0]['sha256'] = 'a' * 64
    if change == 'manifest': changed['manifest_sha256'] = 'a' * 64
    if change == 'generation': changed['generation']['max_new_tokens'] = 63
    if change == 'indices': changed['selected_global_indices'].reverse()
    if change == 'workers': changed['workers'] = 4
    with pytest.raises(ValueError, match='exact immutable protocol'): p.init_journal(output, changed, resume=True)


@pytest.mark.parametrize('field', ['question', 'source', 'answers', 'image_hashes', 'worker_index', 'global_index', 'score'])
def test_valid_checksum_cannot_hide_row_semantic_assignment_or_score_corruption(data, field):
    records = data['records']; protocol = data['protocol'](); output = data['root'] / 'corrupt'; p.init_journal(output, protocol)
    p.commit_record(output, 0, records[0], generated(records[0]), protocol, worker_index=0)
    path = output / 'prediction_records/row-0000.json'; envelope = json.loads(path.read_bytes())
    value = {'question': 'other question', 'source': {}, 'answers': ['other'], 'image_hashes': ['a' * 64],
             'worker_index': 1, 'global_index': 2, 'score': 0.}[field]
    envelope['row'][field] = value; envelope['payload_sha256'] = p.protocol_sha256(envelope['row'])
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError): p.read_committed(output, records, protocol)


def test_orphans_duplicates_missing_and_symlinks_never_receive_completed_scores(data):
    records = data['records']; protocol = data['protocol'](); output = data['root'] / 'missing'; p.init_journal(output, protocol)
    p.commit_record(output, 0, records[0], generated(records[0]), protocol, worker_index=0)
    with pytest.raises(ValueError, match='Only all declared'): p.merge_completed(output, records, protocol)
    assert not (output / 'summary.json').exists()
    path = output / 'prediction_records/row-0000.json'
    extra = path.with_name('row-0512.json'); extra.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='Unexpected'): p.read_committed(output, records, protocol)
    extra.unlink(); original = path.read_bytes(); path.unlink(); target = data['root'] / 'target'; target.write_bytes(original); path.symlink_to(target)
    with pytest.raises(ValueError, match='symlink'): p.read_committed(output, records, protocol)
    orphan = data['root'] / 'orphan'; orphan.mkdir(); (orphan / 'predictions.jsonl').write_text('{}')
    with pytest.raises(ValueError, match='Orphan'): p.init_journal(orphan, protocol)


@pytest.mark.parametrize('field,value', [('generated_token_ids', [True]), ('generated_tokens', True),
    ('prompt_sha256', 'bad'), ('visual_tokens', 0), ('prefill_tokens', 0),
    ('logits_to_keep', True), ('generated_token_ids', [3, 7])])
def test_native_generation_token_prompt_eos_policy_is_strict(data, field, value):
    row = generated(data['records'][0]); row[field] = value
    with pytest.raises(ValueError): p.validate_generated(data['records'][0], row, eos_token_ids=[3])


def test_full_native_reference_validated_without_mutation_and_token_parity_is_exact(data):
    records = data['records']; protocol = data['protocol'](); directory = data['root'] / 'native'
    raw = native_cache(directory, data, records, protocol)
    before = {f.name: f.read_bytes() for f in directory.iterdir()}
    result = p.validate_native_cache(directory, records, protocol, root=data['root'])
    assert result['status'] == 'passed' and result['records'] == 512
    assert result['canonical_RGB_paths_verified'] == 1
    assert result['rows'] == raw
    assert p.compare_reference(raw, result)['status'] == 'passed'
    changed = copy.deepcopy(raw); changed[13]['generated_token_ids'][0] = 8
    assert p.compare_reference(changed, result)['mismatched_record_ids'] == [raw[13]['record_id']]
    assert all((directory / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize('kind', ['question', 'reference', 'score', 'order', 'model', 'source', 'summary', 'environment'])
def test_native_reference_corruption_is_refused(data, kind):
    records = data['records']; protocol = data['protocol'](); directory = data['root'] / 'nativebad'
    raw = native_cache(directory, data, records, protocol)
    if kind in ('question', 'reference', 'score', 'order'):
        if kind == 'question': raw[0]['question'] = 'other'
        if kind == 'reference': raw[0]['answers'][0] = 'other'
        if kind == 'score': raw[0]['score'] = 0.
        if kind == 'order': raw.reverse()
        (directory / 'predictions.jsonl').write_bytes(b''.join(p.json_bytes(r) + b'\n' for r in raw))
    elif kind == 'model':
        path = directory / 'model_provenance.json'; value = json.loads(path.read_text()); value['recipe'] = '1b'; path.write_text(json.dumps(value))
    elif kind == 'source':
        (data['root'] / 'src/eval/research_metrics.py').write_text('changed')
    elif kind == 'environment':
        protocol['environment']['packages']['transformers'] = 'changed'
    else:
        path = directory / 'summary.json'; value = json.loads(path.read_text()); value['complete_manifest'] = False; path.write_text(json.dumps(value))
    with pytest.raises(ValueError): p.validate_native_cache(directory, records, protocol, root=data['root'])


def test_guard64_serial_cache_is_named_separate_scope_and_never_full512(data):
    records = data['records']; guard = data['protocol'](workers=1, scope='guard64_serial_reference')
    full = data['protocol'](); directory = data['root'] / 'native_guard'; p.init_journal(directory, guard)
    commit_all(directory, records, guard)
    raw, summary = p.merge_completed(directory, records, guard)
    assert len(raw) == 64 and summary['complete_manifest'] is False and summary['complete_selected_records'] is True
    assert {r['benchmark']: sum(row['benchmark'] == r['benchmark'] for row in raw) for r in raw} == {b: 8 for b in p.BENCHMARKS}
    result = p.validate_native_cache(directory, records, full, scope='guard64', root=data['root'])
    assert result['records'] == 64 and result['binding']['scope'] == 'guard64'
    with pytest.raises(ValueError): p.validate_native_cache(directory, records, full, scope='full512')


def test_blank_reference_own_paths_and_transformed_hash_can_differ_with_same_rgb_identity(data):
    def transformed(name):
        image = data['root'] / name / 'black.png'; image.parent.mkdir(); Image.new('RGB', (3, 2)).save(image)
        records = copy.deepcopy(data['records'])
        for row in records:
            row['image_paths'] = [str(image)]; row['image_hashes'] = [p._rgb_hash(image)]
            row['image_ablation'] = dict(variant='blank', seed=42, actual_rgb_dimensions=[[3, 2]])
        return records
    requested = transformed('new'); previous = transformed('old')
    protocol = data['protocol'](rows=requested, variant='blank')
    directory = data['root'] / 'old_native'; native_cache(directory, data, previous, protocol)
    result = p.validate_native_cache(directory, requested, protocol, root=data['root'])
    assert result['status'] == 'passed'
    assert result['binding']['parent_transformed_manifest_sha256'] != result['binding']['requested_transformed_manifest_sha256']


def test_actual_transformed_bytes_and_changed_generation_cannot_be_hidden_on_resume(data):
    records = copy.deepcopy(data['records'])
    for row in records:
        row['image_paths'] = []; row['image_hashes'] = []
        row['image_ablation'] = dict(variant='text_only', seed=42)
    protocol = data['protocol'](rows=records, variant='text_only'); output = data['root'] / 'text'; p.init_journal(output, protocol)
    p.commit_record(output, 0, records[0], generated(records[0]), protocol, worker_index=0)
    changed = copy.deepcopy(protocol); changed['generation']['do_sample'] = True
    (output / 'protocol.json').write_bytes(p.json_bytes(changed))
    with pytest.raises(ValueError, match='Reconstructed'): p.read_committed(output, records, changed)
    with pytest.raises(ValueError, match='Non-none'):
        p.make_protocol(records, checkpoint_identity=data['identity'], manifest_sha256=data['manifest_sha'],
            transformed_manifest_sha256='a' * 64, source_sha256=data['sources'], model_provenance=data['provenance'],
            worker_profiles=[dict(worker_index=i, batch_size=1) for i in (0, 1)], variant='text_only')


def test_completed_full512_view_and_summary_corruption_refused_without_repair(data):
    records = data['records']; protocol = data['protocol'](); output = data['root'] / 'finished'
    p.init_journal(output, protocol); commit_all(output, records, protocol)
    rows, _ = p.merge_completed(output, records, protocol)
    raw_path = output / 'predictions.jsonl'; old_raw = raw_path.read_bytes()
    raw_path.write_bytes(old_raw + b'\n')
    with pytest.raises(ValueError, match='immutable'): p.merge_completed(output, records, protocol)
    assert raw_path.read_bytes() == old_raw + b'\n'
    raw_path.write_bytes(old_raw)
    summary = output / 'summary.json'; value = json.loads(summary.read_bytes()); value['group_macro_score'] = 0.
    summary.write_bytes(p.json_bytes(value) + b'\n')
    with pytest.raises(ValueError, match='immutable'): p.merge_completed(output, records, protocol)
    assert json.loads(summary.read_bytes())['group_macro_score'] == 0.


def test_failed_full512_native_token_reference_cannot_receive_a_completed_summary(data):
    records = data['records']; protocol = data['protocol'](); output = data['root'] / 'reference'
    p.init_journal(output, protocol); commit_all(output, records, protocol)
    reference = native_cache(data['root'] / 'preserved', data, records, protocol)
    reference[42]['generated_token_ids'][0] = 99
    with pytest.raises(ValueError, match='parity failed'):
        p.merge_completed(output, records, protocol, reference={'rows': reference})
    assert not (output / 'summary.json').exists() and not (output / 'predictions.jsonl').exists()


@pytest.mark.parametrize('field', ['components_fully_hashed', 'sha256', 'total_bytes', 'primary_mtime_ns'])
def test_compound_checkpoint_cannot_be_replaced_by_partial_or_inconsistent_identity(data, field):
    changed = copy.deepcopy(data['identity'])
    changed[field] = {'components_fully_hashed': False, 'sha256': 'f' * 64,
                      'total_bytes': 999, 'primary_mtime_ns': 1}[field]
    with pytest.raises(ValueError):
        p.make_protocol(data['records'], checkpoint_identity=changed,
            manifest_sha256=data['manifest_sha'], transformed_manifest_sha256=data['manifest_sha'],
            source_sha256=data['sources'], model_provenance=data['provenance'],
            worker_profiles=[dict(worker_index=i, batch_size=1) for i in (0, 1)], variant='none')


def test_module_imports_in_fresh_cpu_parent_without_torch():
    result = subprocess.run([sys.executable, '-c',
        "import sys;from src.eval import research_parallel_protocol;assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1], env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
