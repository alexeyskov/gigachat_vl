"""Source receipts preserve original bytes and keep secret stores outside ZIPs."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.utils.research_provenance import capture_source_snapshot,write_run_provenance


class TestResearchProvenance(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root/'src').mkdir()
        (self.root/'src/model.py').write_text('VERSION = 1\n')
        (self.root/'requirements.txt').write_text('torch==1\n')

    def tearDown(self):self.temporary.cleanup()

    def test_snapshots_preserve_changed_source_and_exclude_secret_store(self):
        (self.root/'.hf_home').mkdir()
        (self.root/'.hf_home/token').write_text('sentinel-private-credential')
        first = capture_source_snapshot(self.root,reserve_gb=0)
        repeated = capture_source_snapshot(self.root,reserve_gb=0)
        self.assertEqual(first,repeated)
        (self.root/'src/model.py').write_text('VERSION = 2\n')
        second = capture_source_snapshot(self.root,reserve_gb=0)
        self.assertNotEqual(first['code_sha256'],second['code_sha256'])
        with zipfile.ZipFile(self.root/first['source_archive']) as saved:
            self.assertEqual(saved.read('src/model.py'),b'VERSION = 1\n')
            self.assertEqual(set(saved.namelist()),{'src/model.py','requirements.txt'})
            self.assertNotIn(b'sentinel-private-credential',b''.join(saved.read(n) for n in saved.namelist()))

    def test_resume_has_separate_attempt_without_rewriting_original(self):
        run = self.root/'runs/example'
        first = write_run_provenance(self.root,run,{'resume':False},reserve_gb=0)
        (self.root/'src/model.py').write_text('VERSION = 2\n')
        second = write_run_provenance(self.root,run,{'resume':True},reserve_gb=0)
        self.assertEqual(json.loads((run/'provenance.json').read_text()),first)
        self.assertEqual(json.loads((run/'provenance_latest.json').read_text()),second)
        self.assertEqual(len(list((run/'provenance_attempts').glob('*.json'))),2)
        self.assertNotEqual(first['code_sha256'],second['code_sha256'])

    def test_altered_archive_and_symlink_sources_refused(self):
        captured = capture_source_snapshot(self.root,reserve_gb=0)
        with zipfile.ZipFile(self.root/captured['source_archive'],'w') as saved:
            saved.writestr('src/model.py','changed')
        with self.assertRaisesRegex(ValueError,'content address'):
            capture_source_snapshot(self.root,reserve_gb=0)
        (self.root/'src/model.py').unlink()
        (self.root/'src/model.py').symlink_to(self.root/'requirements.txt')
        with self.assertRaisesRegex(ValueError,'regular local'):
            capture_source_snapshot(self.root,reserve_gb=0)
