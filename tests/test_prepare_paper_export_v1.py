"""Byte identity matters: archive verified model weights without rewriting them."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'paper_export', Path(__file__).resolve().parents[1] / 'scripts/prepare_paper_export_v1.py')
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def test_selection_uses_internal_vqa_only_and_preserves_all_rows():
    rows = []
    for family in ('smallthinker4', 'lfm25', 'giga31'):
        rows.extend([
            dict(id=family+'-high-test', model=family, VQA_after=10,
                 checkpoint_bytes=10, detection_test_F1_IoU50=99),
            dict(id=family+'-large', model=family, VQA_after=20,
                 checkpoint_bytes=20, detection_test_F1_IoU50=1),
            dict(id=family+'-small', model=family, VQA_after=20,
                 checkpoint_bytes=5, detection_test_F1_IoU50=0)])
    assert [r['id'] for r in export.choose_weights(rows, 'best')] == [
        family+'-small' for family in ('smallthinker4', 'lfm25', 'giga31')]
    assert export.choose_weights(rows, 'all-final') == rows
    assert export.choose_weights(rows, 'metrics') == []


def test_archive_roundtrip_native_digest_and_corruption_detection(tmp_path):
    source = tmp_path / 'state.pt'
    source.write_bytes(bytes(range(256)) * 8192)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    name = 'runs/example/.checkpoint_objects/final/state.pt'
    output = tmp_path / 'weights.tar.zst'
    receipt = export.archive_files([(name, source)], output, {name: digest}, threads=1)
    assert receipt['status'] == 'byte_exact_verified'
    assert receipt['members'][name] == dict(bytes=source.stat().st_size, sha256=digest)
    assert receipt['archive_sha256'] == export.sha(output)
    export.verify_archive(output, receipt['members'])
    bad_expected = {name: dict(bytes=source.stat().st_size, sha256='0'*64)}
    with pytest.raises(ValueError, match='original bytes'):
        export.verify_archive(output, bad_expected)
    # A file can only be published when its SHA matches the native GPU proof.
    unpublished = tmp_path / 'rejected.tar.zst'
    with pytest.raises(ValueError, match='native GPU proof'):
        export.archive_files([(name, source)], unpublished, {name: '0'*64}, threads=1)
    assert not unpublished.exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_metadata_never_copies_weight_payload_and_keeps_role_links(tmp_path):
    source = tmp_path / 'run'
    physical = source / '.checkpoint_objects/final'
    physical.mkdir(parents=True)
    (physical / 'state.pt').write_bytes(b'checkpoint')
    (physical / '.research_checkpoint.json').write_text('{}')
    (source / 'completed.json').write_text('{}')
    (source / 'last').symlink_to('.checkpoint_objects/final')
    (source / 'best').symlink_to('.checkpoint_objects/final')
    target = tmp_path / 'export'
    export.copy_metadata(source, target)
    assert (target / 'completed.json').exists()
    assert (target / '.checkpoint_objects/final/.research_checkpoint.json').exists()
    assert not (target / '.checkpoint_objects/final/state.pt').exists()
    assert (target / 'best').readlink() == Path('.checkpoint_objects/final')
