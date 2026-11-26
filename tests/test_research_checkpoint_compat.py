import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.utils.research_checkpoint_bundle import save_checkpoint_bundle,retire_best_optimizer
from src.utils.research_checkpoint_compat import (
    checkpoint_identity,checkpoint_kind,load_research_payload,same_checkpoint_identity,
)
from src.utils.research_storage import BestLastCheckpointManager,DiskBudget


class TestCheckpointCompat(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.manager=BestLastCheckpointManager(self.root/'run',DiskBudget(self.root,0))
    def tearDown(self):self.temp.cleanup()
    def payload(self,step=1):
        return {'step':step,'model':{'weight':torch.arange(100,dtype=torch.float32)+step},
                'optimizer':{'state':{0:{'moment':torch.ones(100)}},'param_groups':[{'params':[0]}]},
                'rng_torch':torch.get_rng_state()}
    def save_bundle(self,step=1,metric=.5):
        self.manager.save(lambda d:save_checkpoint_bundle(self.payload(step),d),step=step,metric=metric,anticipated_bytes=1024**2)

    def test_legacy_sha_is_exact_file_sha_and_payload_unchanged(self):
        self.manager.save_torch(self.payload(),step=1,metric=.5,anticipated_bytes=1024**2)
        path=self.manager.root/'last/state.pt';identity=checkpoint_identity(path)
        self.assertEqual(identity['format'],'legacy_single_file')
        self.assertEqual(identity['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        restored=load_research_payload(path)
        self.assertTrue(torch.equal(restored['model']['weight'],self.payload()['model']['weight']))

    def test_bundle_identity_binds_every_component_not_only_manifest(self):
        self.save_bundle();path=self.manager.root/'last/state.pt'
        identity=checkpoint_identity(path)
        self.assertEqual(identity['format'],'bundle_v1')
        self.assertEqual({row['role'] for row in identity['members']},{'manifest','model','optimizer'})
        self.assertGreater(identity['total_bytes'],identity['primary_bytes'])
        self.assertNotEqual(identity['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        metadata=checkpoint_identity(path,verify_components=False)
        self.assertFalse(metadata['components_fully_hashed'])
        self.assertTrue(same_checkpoint_identity(identity,metadata))
        component=path.resolve().parent/'model.state.pt'
        with component.open('r+b') as handle:
            handle.seek(-8,2);data=handle.read(1);handle.seek(-1,1);handle.write(bytes([data[0]^1]))
        with self.assertRaisesRegex(ValueError,'SHA256'):checkpoint_identity(path)

    def test_best_retirement_changes_identity_but_preserves_model_component(self):
        self.save_bundle(1,.3);self.save_bundle(2,.5)
        before=checkpoint_identity(self.manager.root/'best')
        before_model=next(row['sha256'] for row in before['members'] if row['role']=='model')
        retire_best_optimizer(self.manager.root)
        after=checkpoint_identity(self.manager.root/'best')
        self.assertFalse(same_checkpoint_identity(before,after))
        self.assertNotEqual(before['sha256'],after['sha256'])
        self.assertEqual(before_model,next(row['sha256'] for row in after['members'] if row['role']=='model'))
        self.assertEqual({row['role'] for row in after['members']},{'manifest','model'})
        self.assertFalse(load_research_payload(self.manager.root/'best')['resumable_optimizer'])
        self.assertTrue(load_research_payload(self.manager.root/'last')['resumable_optimizer'])

    def test_progress_counts_all_bundle_bytes(self):
        self.save_bundle();progress=[]
        identity=checkpoint_identity(self.manager.root/'last',progress=lambda n,total:progress.append((n,total)))
        self.assertEqual(progress[-1],(identity['total_bytes'],identity['total_bytes']))
        self.assertTrue(all(n<=total for n,total in progress))

    def test_identity_inside_exclusive_manager_lock_avoids_nested_reader_lock(self):
        self.save_bundle();expected=checkpoint_identity(self.manager.root/'last')
        with self.manager._lock():
            with patch('src.utils.research_checkpoint_compat._reader_lock',side_effect=AssertionError('nested lock')):
                actual=checkpoint_identity(self.manager.root/'last',manager_lock_held=True)
        self.assertTrue(same_checkpoint_identity(expected,actual))


if __name__=='__main__':unittest.main()
