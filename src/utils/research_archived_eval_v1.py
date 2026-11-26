"""Read-only admission and one guarded CPU decode for owned Small archives.

Only completed stock-v2 publisher objects are accepted.  Metadata inspection
does not read checkpoint bodies; explicit admission validates physical hashes.
The frozen native inference body, constructor, restoration and generator are
reused on a private module. This is for trusted local Torch payloads only.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType

from src.utils import research_archive_stock_v2 as archive
from src.utils import research_stock_determinism as stock
from src.utils import research_stream_delta as stream
from src.utils import research_trained_delta_probe as probe

ROOT = Path(__file__).resolve().parents[2]
OWNER = 'owned_small_archived_evaluation_v1'
ACTIVATION_SHA256 = 'b2af334341c0b9acde2d50ec13592c4fad137ce402baced2e7a7db44e27e0841'
GPU_PROOF_SHA256 = '1dbd3160ef5d5c97d9bdea35810eb5ec044747b35ffafec40dad2dc54ebbaa65'
OWN = ('src/utils/research_archived_eval_v1.py', 'tests/test_research_archived_eval_v1.py')
GIB = 2**30
CONSTRUCTOR_ALLOWANCE = 16 * GIB
METADATA_LIMIT = 32 * 2**20
READY = ROOT / 'configs/research/archived_eval_v1_ready.json'


def _publisher(): return archive.publisher_v2()
def sha(path): return probe.file_sha(path)
def stat(path): return probe.stat(path)


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > METADATA_LIMIT:
        raise ValueError('Bounded ordinary archival metadata required')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value: raise ValueError('Duplicate archival metadata key')
            value[key] = item
        return value
    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def descriptor(path):
    path = Path(path).resolve(strict=True); before = stat(path)
    result = {'path': str(path), 'stat': before, 'sha256': sha(path)}
    if stat(path) != before: raise ValueError('Archival metadata changed while identifying')
    return result


def source_binding():
    return {**_publisher().code_binding(), **stock.definition()['source_sha256'],
        **{name: sha(ROOT / name) for name in OWN}}


def execution_profile():
    return {'owner': OWNER, 'schema_version': 1, 'stock_numerical_profile': stock.definition(),
        'source_sha256': source_binding(), 'activation_sha256': ACTIVATION_SHA256,
        'actual_GPU_parity_sha256': GPU_PROOF_SHA256, 'CPU_payload_decodes_per_load': 1,
        'map_location': 'cpu', 'format': 'owned_guarded_native_legacy',
        'constructor_allowance_bytes': CONSTRUCTOR_ALLOWANCE, 'headroom_bytes': 32 * GIB,
        'archive_file_pages_charged': True, 'reclaimable_cache_credit_bytes': 0,
        'numerical_math_or_checkpoint_values_changed': False, 'fallback': False}


def _activation(publisher):
    if sha(archive.ACTIVATION) != ACTIVATION_SHA256:
        raise ValueError('Exact ROOT-approved stock archival activation required')
    evidence = publisher.activation_evidence()
    if evidence.get('sha256') != ACTIVATION_SHA256 or evidence.get('gpu_report_sha256') != GPU_PROOF_SHA256:
        raise ValueError('Actual CPU/GPU archival evidence differs')
    return evidence


@contextmanager
def _shared(run):
    run = Path(run).resolve(strict=True); lock = run / '.checkpoint.lock'
    if lock.is_symlink() or not lock.is_file(): raise ValueError('Existing owned manager lock required')
    with lock.open('rb') as handle:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield run


def _metadata_unlocked(run, role):
    if role not in ('best', 'last'): raise ValueError('Retained best/last role required')
    publisher = _publisher(); plan = publisher.eligibility(run, allow_archived=True)
    stock.validate_run_context(run)
    evidence = _activation(publisher); report = read(run / publisher.REPORT); journal = read(run / publisher.JOURNAL)
    if (report.get('status') != 'completed' or report.get('protocol') != publisher.OWNER or report.get('run') != str(run) or
            report.get('manager_owner') != plan['manager']['owner'] or report.get('activation') != evidence or
            report.get('roles') != publisher._role_names(run, plan['manager']['owner']) or
            journal.get('phase') != 'completed' or not any(item == journal for item in report.get('transactions', []))):
        raise ValueError('Completed archival report/journal/roles differ')
    publisher._journal_valid(journal, plan, evidence)
    compaction = read(run / 'checkpoint_compaction.json'); transactions = {}
    for name, item in plan['roles'].items():
        marker = item['marker']; lineage = marker.get('checkpoint_archival', {})
        if type(marker.get('step')) is not int or not 0 < marker['step'] <= plan['completion']['steps']:
            raise ValueError('Archived BEST/LAST chronology differs')
        matches = [value for value in report['transactions'] if value.get('destination') == item['object']]
        if len(matches) != 1: raise ValueError('Archive lacks one exact source transaction')
        transaction = matches[0]; derived = transaction.get('derived', {})
        expected = dict(transaction.get('source_marker', {})); expected['checkpoint_archival'] = transaction.get('target_lineage')
        if (lineage.get('protocol') != publisher.OWNER or marker != expected or transaction.get('phase') != 'completed' or
                transaction.get('activation') != evidence or transaction.get('metadata_sha256') != plan['metadata_sha256'] or
                transaction.get('manager_owner') != plan['manager']['owner'] or transaction.get('run') != str(run) or
                not any(value.get('new_object') == transaction.get('source_object') for value in compaction.get('checkpoints', [])) or
                derived.get('all_payload_fields_and_backing_bits_exact') is not True or
                derived.get('payload_sha256') != lineage.get('payload_sha256') or
                derived.get('archive_sha256') != lineage.get('archive_sha256') or
                derived.get('archive_bytes') != item['stat']['st_size'] or lineage.get('archive_bytes') != item['stat']['st_size'] or
                lineage.get('activation_sha256') != evidence['sha256'] or lineage.get('decode_code_sha256') != evidence['code_sha256']):
            raise ValueError('Archive complete source marker/compaction/payload lineage differs')
        header = derived.get('reader_header', {})
        required = stream.memory_requirement(header)  # Header/anchor stat checks only; no Tensor values.
        if (header.get('decoded_model_bytes') != derived.get('decoded_original_bytes') or
                header.get('normal_storage_bytes') != derived.get('normal_storage_bytes')):
            raise ValueError('Archived decoder header/census differs')
        transactions[name] = {'transaction': transaction, 'reader_header': header,
            'decoder_extra_bytes': required, 'physical_sha256': lineage['archive_sha256'],
            'archive_file_bytes': item['stat']['st_size'],
            'load_extra_bytes': required + item['stat']['st_size'] + CONSTRUCTOR_ALLOWANCE}
    files = {name: descriptor(run / name) for name in (*plan['metadata_sha256'], publisher.REPORT, publisher.JOURNAL)}
    for name in ('provenance.json', 'provenance_latest.json'):
        if (run / name).is_file(): files[name] = descriptor(run / name)
    for name in ('receipt.json', 'source.zip'):
        files['stock_execution/' + name] = descriptor(run / 'stock_execution_v1' / name)
    files['archival_activation'] = descriptor(archive.ACTIVATION)
    for field in ('trained_probe_report', 'root_gpu_parity_report'):
        files[field] = descriptor(read(archive.ACTIVATION)[field])
    return {'owner': OWNER, 'schema_version': 1, 'run': str(run), 'role': role, 'publisher_plan': plan,
        'report': report, 'journal': journal, 'activation': evidence, 'files': files, 'transactions': transactions,
        'execution_profile': execution_profile(), 'checkpoint_bodies_read': False,
        'physical_archives_verified': False, 'payload_decoded': False}


def inspect(run, role='best'):
    """Metadata/anchor stat only. Claimed physical hashes are not recertified."""
    with _shared(run) as resolved: return _context(_metadata_unlocked(resolved, role))


def _assert_metadata(plan):
    if plan.get('owner') != OWNER or plan.get('execution_profile') != execution_profile():
        raise ValueError('Archived evaluation source/profile changed')
    for item in plan['files'].values():
        if stat(item['path']) != item['stat'] or sha(item['path']) != item['sha256']:
            raise ValueError('Archived evaluation metadata/activation changed')
    run = Path(plan['run'])
    for role, item in plan['publisher_plan']['roles'].items():
        pointer = run / role
        if (not pointer.is_symlink() or pointer.resolve() != Path(item['state']).parent or
                stat(item['state']) != item['stat'] or read(pointer / '.research_checkpoint.json') != item['marker']):
            raise ValueError('Archived evaluation role/marker/stat changed')
    # Budgets, reader headers and signature expectations are projections of
    # the durable publisher transaction, never caller-controlled allowances.
    fresh = _metadata_unlocked(run, plan['role'])
    flags = ('physical_archives_verified', 'checkpoint_bodies_read', 'payload_decoded')
    if {k: v for k, v in plan.items() if k not in flags} != {k: v for k, v in fresh.items() if k not in flags}:
        raise ValueError('Archived admission transaction/header/budget projection changed')


def _full_unlocked(plan):
    _assert_metadata(plan)
    from src.utils import research_lossless_delta as codec
    unique = {item['state']: item for item in plan['publisher_plan']['roles'].values()}
    codec.memory_guard(sum(item['stat']['st_size'] for item in unique.values()) + METADATA_LIMIT)
    publisher = _publisher()
    publisher._completed_roles_verified(plan['publisher_plan'], plan['report'], plan['journal'], plan['activation'])
    _assert_metadata(plan)
    result = dict(plan, physical_archives_verified=True, checkpoint_bodies_read=True)
    return result


def _artifact(plan):
    item = plan['publisher_plan']['roles'][plan['role']]; value = plan['transactions'][plan['role']]
    member = {'role': 'state', 'relative_path': 'state.pt', 'path': item['state'], 'bytes': item['stat']['st_size'],
        'mtime_ns': item['stat']['st_mtime_ns'], 'sha256': value['physical_sha256']}
    return {'format': 'legacy_single_file', 'sha256': member['sha256'], 'primary_path': item['state'],
        'primary_bytes': member['bytes'], 'primary_mtime_ns': member['mtime_ns'], 'total_bytes': member['bytes'],
        'components_fully_hashed': plan['physical_archives_verified'], 'members': [member]}


def _context(plan):
    run = Path(plan['run']); roles = {}
    for role, item in plan['publisher_plan']['roles'].items():
        roles[role] = {'pointer': os.readlink(run / role), 'object': str(Path(item['state']).parent),
            'marker': item['marker'], 'marker_file': descriptor(Path(item['state']).parent / '.research_checkpoint.json')}
    artifact = _artifact(plan)
    return {'run': str(run), 'role': plan['role'], 'roles': roles,
        'checkpoint_path': plan['publisher_plan']['roles'][plan['role']]['state'], 'artifact': artifact,
        'config': plan['publisher_plan']['config'], 'files': plan['files'],
        'member_stats': {item['path']: stat(item['path']) for item in artifact['members']},
        'metadata_only': not plan['physical_archives_verified'], 'archive_admission': plan}


def checkpoint_context(run, role='best'):
    """Explicit full physical publication validation; no decode or model yet."""
    with _shared(run) as resolved: return _context(_full_unlocked(_metadata_unlocked(resolved, role)))


def assert_context(context):
    with _shared(context['run']):
        _assert_metadata(context['archive_admission'])
        if _context(context['archive_admission']) != context:
            raise ValueError('Archived checkpoint context projection differs')


def _payload(payload, plan):
    stock.validate_config(payload.get('config', {}))
    if payload['config'] != plan['publisher_plan']['config']:
        raise ValueError('Decoded payload config differs from original completed run')
    transaction = plan['transactions'][plan['role']]['transaction']
    snapshot = Path(transaction['derived']['reader_header']['anchor_files'][0]['snapshot'])
    _publisher()._census(payload, plan['publisher_plan'], snapshot)
    signature = probe.payload_signature(payload)
    if signature['sha256'] != transaction['derived']['payload_sha256']:
        raise ValueError('Decoded complete payload/backing/metadata signature differs')
    return signature


def _decode_admitted(checkpoint, plan):
    """Internal: the native bridge holds the fresh full-hash SH lease."""
    if not plan.get('physical_archives_verified'): raise ValueError('Actual physical archive admission required before decode')
    expected = Path(plan['publisher_plan']['roles'][plan['role']]['state'])
    if Path(checkpoint).resolve(strict=True) != expected: raise ValueError('Wrong admitted archive requested')
    _assert_metadata(plan)
    from src.utils import research_lossless_delta as codec
    from src.utils.research_checkpoint_compat import load_research_payload
    extra = plan['transactions'][plan['role']]['load_extra_bytes']
    codec.memory_guard(extra)
    payload = load_research_payload(expected, map_location='cpu', mmap=True)
    # Frozen stream prefix/per-tensor guards remain active during the decode.
    # After decode, compressed pickle memo/scratch are gone; protect census and
    # constructor using actual remaining memory without credit for file pages.
    codec.memory_guard(CONSTRUCTOR_ALLOWANCE + METADATA_LIMIT)
    _payload(payload, plan); _assert_metadata(plan)
    codec.memory_guard(CONSTRUCTOR_ALLOWANCE + METADATA_LIMIT)
    return payload


def load_payload(checkpoint, plan):
    """Standalone CPU API also obtains its own fresh physical proof and SH."""
    with _shared(plan['run']):
        return _decode_admitted(checkpoint, _full_unlocked(plan))


def inference(full_context=None):
    """Private unchanged native inference, with one guarded payload import."""
    captured = {'plan': None, 'loads': 0}
    compatible = ModuleType('_owned_archived_payload_loader')
    def guarded(path, *, map_location='cpu', mmap=True):
        if map_location != 'cpu' or mmap is not True or captured['plan'] is None:
            raise ValueError('Archived native load requires explicit admitted CPU/mmap request')
        captured['loads'] += 1
        if captured['loads'] != 1: raise ValueError('Archive model construction must decode exactly once')
        return _decode_admitted(path, captured['plan'])
    compatible.load_research_payload = guarded
    private_runner = stock.runner(); native_model = private_runner.ResearchModel
    class GuardedResearchModel(native_model):
        def __init__(self, args):
            if captured['plan'] is None: raise ValueError('Archive constructor is outside its admitted SH lease')
            from src.utils import research_lossless_delta as codec
            codec.memory_guard(CONSTRUCTOR_ALLOWANCE + METADATA_LIMIT)
            super().__init__(args)
            codec.memory_guard(METADATA_LIMIT)
    private_runner.ResearchModel = GuardedResearchModel
    module = stock._clone('src/eval/research_inference.py', {
        'scripts.run_research_stage': private_runner, 'src.utils.research_checkpoint_compat': compatible})
    native_load = module.load_research_checkpoint
    def load(path, **kwargs):
        path = Path(path).resolve(strict=True)
        if full_context is None:
            run = path.parent.parent.parent
            roles = [name for name in ('best', 'last') if (run / name).resolve(strict=True) / 'state.pt' == path]
            if not roles: raise ValueError('Archive is not a retained owned role')
            role = roles[0]; supplied = None
        else:
            run = Path(full_context['run']); role = full_context['role']; supplied = full_context['archive_admission']
            if full_context['checkpoint_path'] != str(path): raise ValueError('Archive context/checkpoint differs')
        with _shared(run) as resolved:
            if supplied is None: plan = _full_unlocked(_metadata_unlocked(resolved, role))
            else:
                _assert_metadata(supplied)
                if _context(supplied) != full_context: raise ValueError('Archived checkpoint context projection differs')
                if not supplied.get('physical_archives_verified'): raise ValueError('Archive context physical proof missing')
                # The physical read and decode share this same live SH lease.
                # A supplied dictionary cannot turn a historical boolean into
                # fresh proof of the current bytes.
                plan = _full_unlocked(supplied)
            captured.update(plan=plan, loads=0)
            try:
                stock.activate(); model, provenance = native_load(path, **kwargs); stock.assert_active()
                if captured['loads'] != 1: raise ValueError('Native archived loader did not perform exactly one guarded decode')
                _assert_metadata(plan)
                from src.utils import research_lossless_delta as codec
                codec.memory_guard(METADATA_LIMIT)
                provenance.update(numerical_profile=stock.definition(), archived_evaluation=execution_profile(),
                    archived_publication_sha256=plan['files'][_publisher().REPORT]['sha256'],
                    archived_physical_sha256=plan['transactions'][role]['physical_sha256'])
                return model, provenance
            finally: captured['plan'] = None
    module.load_research_checkpoint = load
    return module


def verify_release():
    value = read(READY)
    if (value.get('owner') != OWNER or value.get('status') != 'cpu_tests_passed_actual_consumer_GPU_not_run' or
            value.get('source_sha256') != source_binding() or value.get('execution_profile') != execution_profile()):
        raise ValueError('Exact shared archived evaluation CPU release required')
    return {'path': str(READY), 'sha256': sha(READY)}
