"""Split bundles preserve scientific weights and retain one exact optimizer resume."""
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import src.utils.research_checkpoint_bundle as bundle
from src.utils.research_checkpoint_io import load_checkpoint,save_checkpoint
from src.utils.research_storage import BestLastCheckpointManager,DiskBudget


class TestResearchCheckpointBundle(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.manager=BestLastCheckpointManager(self.root/'run',DiskBudget(self.root,0))
    def tearDown(self):self.temp.cleanup()
    def payload(self,step=1):
        base=torch.arange(16,dtype=torch.float32).reshape(4,4)+step
        return {'model':{'weight':base,'tied_weight':base,'view':base[:,1:3],
                         'projector':torch.linspace(-1,1,12,dtype=torch.bfloat16)},
                'optimizer':{'state':{0:{'step':torch.tensor(float(step)),
                                        'exp_avg':torch.ones(4,4)*.1,'exp_avg_sq':torch.ones(4,4)*.01}},
                             'param_groups':[{'params':[0],'lr':.001}]},
                'scheduler':{'last_epoch':step},'rng_torch':torch.get_rng_state(),'rng_python':random.getstate(),
                'rng_numpy':np.random.get_state(),'step':step,'seen':step*64,
                'model_ref':{'repo':'test','revision':'a'*40},'vision_ref':{'repo':'vision','revision':'b'*40},
                'manifest_sha256':'manifest-data-hash','cache_fingerprint':'frozen-vision-hash',
                'config':{'projector':'mlp','all_model_tensors_saved':True}}
    def save(self,step=1,metric=.5,compression='none',payload=None):
        payload=self.payload(step) if payload is None else payload
        return self.manager.save(lambda directory:bundle.save_checkpoint_bundle(payload,directory,compression=compression),
                                 step=step,metric=metric,anticipated_bytes=1024**2)
    def equal(self,a,b):
        if isinstance(a,torch.Tensor):
            self.assertEqual(a.dtype,b.dtype);self.assertEqual(a.shape,b.shape);self.assertEqual(a.stride(),b.stride())
            self.assertTrue(torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8)))
        elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
        elif isinstance(a,dict):
            self.assertEqual(set(a),set(b))
            for key in a:self.equal(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(type(a),type(b));self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.equal(x,y)
        else:self.assertEqual(a,b)

    def test_lossless_model_optimizer_rng_and_tied_storage_native_and_gzip(self):
        for codec in ('none','gzip'):
            with self.subTest(codec=codec):
                manager=BestLastCheckpointManager(self.root/codec,DiskBudget(self.root,0))
                payload=self.payload()
                manager.save(lambda d:bundle.save_checkpoint_bundle(payload,d,compression=codec),step=1,metric=.5,anticipated_bytes=1024**2)
                restored=bundle.load_checkpoint_bundle(manager.root/'last',mmap=True)
                self.assertTrue(restored.pop('resumable_optimizer'));self.equal(payload,restored)
                self.assertEqual(restored['model']['weight'].untyped_storage().data_ptr(),restored['model']['tied_weight'].untyped_storage().data_ptr())
                self.assertEqual(restored['model']['weight'].untyped_storage().data_ptr(),restored['model']['view'].untyped_storage().data_ptr())
                info=bundle.inspect_checkpoint_bundle(manager.root/'last')
                self.assertTrue(info['fully_resumable'])
                self.assertEqual(info['manifest']['model']['compression'],codec)

    def test_role_transitions_retire_only_distinct_best_and_preserve_all_model_bytes(self):
        self.save(1,.3)
        self.assertEqual(bundle.retire_best_optimizer(self.manager.root)['status'],'skipped_shared_best_last')
        best=(self.manager.root/'best').resolve()
        original=(best/bundle.MODEL_FILE).read_bytes()
        self.save(2,.5)
        result=bundle.retire_best_optimizer(self.manager.root)
        self.assertEqual(result['status'],'retired');self.assertTrue(result['last_fully_resumable'])
        self.assertEqual(original,(best/bundle.MODEL_FILE).read_bytes())
        self.assertFalse((best/bundle.OPTIMIZER_FILE).exists())
        best_payload=bundle.load_checkpoint_bundle(self.manager.root/'best')
        last_payload=bundle.load_checkpoint_bundle(self.manager.root/'last')
        self.assertFalse(best_payload['resumable_optimizer']);self.assertNotIn('optimizer',best_payload)
        self.assertTrue(last_payload['resumable_optimizer']);self.assertIn('optimizer',last_payload)
        self.equal(self.payload(1)['model'],best_payload['model'])
        self.assertEqual(bundle.retire_best_optimizer(self.manager.root)['status'],'already_retired')
        self.save(3,.1)
        self.assertEqual((self.manager.root/'best').resolve(),(self.manager.root/'last').resolve())
        self.assertTrue(bundle.load_checkpoint_bundle(self.manager.root/'last')['resumable_optimizer'])
        self.assertEqual(len(list(self.manager.objects.iterdir())),1)

    def test_component_corruption_or_missing_last_optimizer_prevents_retirement(self):
        self.save(1,.3);self.save(2,.5)
        best=(self.manager.root/'best').resolve();last=(self.manager.root/'last').resolve()
        path=last/bundle.OPTIMIZER_FILE
        with path.open('r+b') as handle:
            handle.seek(-10,2);value=handle.read(1);handle.seek(-1,1);handle.write(bytes([value[0]^1]))
        with self.assertRaisesRegex(bundle.CheckpointBundleError,'SHA256'):
            bundle.retire_best_optimizer(self.manager.root)
        self.assertTrue((best/bundle.OPTIMIZER_FILE).exists())
        with self.assertRaisesRegex(bundle.CheckpointBundleError,'SHA256'):
            bundle.load_checkpoint_bundle(self.manager.root/'last')

    def test_manifest_escape_symlink_and_owner_mismatch_are_refused(self):
        self.save()
        target=(self.manager.root/'last').resolve();path=target/bundle.MANIFEST_FILE
        original=bundle._read_manifest(path)
        changed=dict(original);changed['model']=dict(original['model'],path='../outside.pt')
        bundle._write_manifest(path,changed)
        with self.assertRaisesRegex(bundle.CheckpointBundleError,'exact owned relative'):
            bundle.load_checkpoint_bundle(path)
        bundle._write_manifest(path,original)
        component=target/bundle.MODEL_FILE;outside=self.root/'outside.pt';component.rename(outside);component.symlink_to(outside)
        with self.assertRaisesRegex(bundle.CheckpointBundleError,'escapes'):
            bundle.load_checkpoint_bundle(path)
        component.unlink();outside.rename(component)
        changed=dict(original);changed['checkpoint_owner']='wrong-owner';bundle._write_manifest(path,changed)
        with self.assertRaisesRegex(bundle.CheckpointBundleError,'owner/step'):
            bundle.load_checkpoint_bundle(path)

    def test_component_provenance_checked_even_with_updated_transport_digest(self):
        self.save();target=(self.manager.root/'last').resolve();path=target/bundle.MODEL_FILE
        payload=load_checkpoint(path);payload['bundle_id']='wrong-bundle';save_checkpoint(payload,path)
        manifest=bundle._read_manifest(target/bundle.MANIFEST_FILE)
        manifest['model'].update(sha256=bundle._sha256(path),bytes=path.stat().st_size)
        bundle._write_manifest(target/bundle.MANIFEST_FILE,manifest)
        with self.assertRaisesRegex(bundle.CheckpointBundleError,'provenance'):
            bundle.load_checkpoint_bundle(target)

    def test_retirement_checks_last_optimizer_provenance_even_when_digest_matches(self):
        self.save(1,.3);self.save(2,.5)
        last=(self.manager.root/'last').resolve();path=last/bundle.OPTIMIZER_FILE
        payload=load_checkpoint(path);payload['bundle_id']='wrong-last-bundle';save_checkpoint(payload,path)
        manifest=bundle._read_manifest(last/bundle.MANIFEST_FILE)
        manifest['optimizer'].update(sha256=bundle._sha256(path),bytes=path.stat().st_size)
        bundle._write_manifest(last/bundle.MANIFEST_FILE,manifest)
        with self.assertRaisesRegex(bundle.CheckpointBundleError,'Optimizer component bundle provenance'):
            bundle.retire_best_optimizer(self.manager.root)
        self.assertTrue((self.manager.root/'best'/bundle.OPTIMIZER_FILE).exists())

    def test_gzip_best_retirement_structure_check_uses_fake_storages(self):
        self.save(1,.3,compression='gzip');self.save(2,.5,compression='gzip')
        result=bundle.retire_best_optimizer(self.manager.root)
        self.assertEqual(result['status'],'retired')
        self.assertTrue(bundle.load_checkpoint_bundle(self.manager.root/'last')['resumable_optimizer'])
        self.assertFalse(bundle.load_checkpoint_bundle(self.manager.root/'best')['resumable_optimizer'])

    def test_failed_optimizer_write_preserves_previous_best_last(self):
        self.save();before=(self.manager.root/'last').resolve();original=bundle.save_checkpoint
        def failure(value,path,**kwargs):
            if Path(path).name==bundle.OPTIMIZER_FILE:raise OSError('simulated optimizer serialization crash')
            return original(value,path,**kwargs)
        with patch.object(bundle,'save_checkpoint',side_effect=failure):
            with self.assertRaisesRegex(OSError,'simulated'):self.save(2,.4)
        self.assertEqual(before,(self.manager.root/'last').resolve())
        self.assertTrue(bundle.load_checkpoint_bundle(before)['resumable_optimizer'])
        self.assertEqual(len(list(self.manager.objects.iterdir())),1)

    def test_crash_after_model_only_manifest_commit_keeps_best_loadable_and_retryable(self):
        self.save(1,.3);self.save(2,.5)
        best=(self.manager.root/'best').resolve();optimizer=best/bundle.OPTIMIZER_FILE
        model_bytes=(best/bundle.MODEL_FILE).read_bytes();old_manifest_hash=bundle._sha256(best/bundle.MANIFEST_FILE)
        unlink=Path.unlink
        def crash(path,*args,**kwargs):
            if path==optimizer:raise OSError('simulated crash before optimizer unlink')
            return unlink(path,*args,**kwargs)
        with patch.object(Path,'unlink',crash):
            with self.assertRaisesRegex(OSError,'simulated'):bundle.retire_best_optimizer(self.manager.root)
        loaded=bundle.load_checkpoint_bundle(best)
        self.assertFalse(loaded['resumable_optimizer']);self.assertNotIn('optimizer',loaded)
        self.assertTrue(optimizer.exists());self.assertEqual(model_bytes,(best/bundle.MODEL_FILE).read_bytes())
        self.assertNotEqual(old_manifest_hash,bundle._sha256(best/bundle.MANIFEST_FILE))
        result=bundle.retire_best_optimizer(self.manager.root)
        self.assertTrue(result['invalidates_prior_roundtrip_receipt']);self.assertFalse(optimizer.exists())
        self.assertTrue(bundle.load_checkpoint_bundle(self.manager.root/'last')['resumable_optimizer'])

    def test_legacy_loader_and_retirement_leave_single_file_checkpoints_untouched(self):
        first=self.payload(1);second=self.payload(2)
        self.manager.save_torch(first,step=1,metric=.3,anticipated_bytes=1024**2)
        self.manager.save_torch(second,step=2,metric=.5,anticipated_bytes=1024**2)
        path=self.manager.root/'best/state.pt';before=path.read_bytes()
        self.equal(second,bundle.load_checkpoint_bundle(self.manager.root/'last'))
        self.assertEqual(bundle.retire_best_optimizer(self.manager.root)['status'],'skipped_legacy_single_file')
        self.assertEqual(before,path.read_bytes())

    def test_cross_component_alias_is_explicitly_refused(self):
        payload=self.payload();payload['optimizer']['state'][0]['exp_avg']=payload['model']['weight']
        with self.assertRaisesRegex(bundle.CheckpointBundleError,'Cross-component shared'):
            self.save(payload=payload)
        self.assertFalse((self.manager.root/'last').exists())

    def test_nonresumable_last_cannot_authorize_retiring_best(self):
        self.save(1,.3);payload=self.payload(2);del payload['optimizer'];payload['resumable_optimizer']=False
        self.save(2,.5,payload=payload)
        with self.assertRaisesRegex(bundle.CheckpointBundleError,'LAST is not fully resumable'):
            bundle.retire_best_optimizer(self.manager.root)
        self.assertTrue((self.manager.root/'best'/bundle.OPTIMIZER_FILE).exists())

    def test_actual_adam_continuation_matches_after_bundle_restore(self):
        torch.manual_seed(11)
        model=nn.Linear(4,3);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        x=torch.randn(5,4);target=torch.randn(5,3)
        def step(m,o):o.zero_grad();loss=(m(x)-target).square().sum();loss.backward();o.step()
        step(model,optimizer)
        payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':1,'rng_torch':torch.get_rng_state()}
        self.save(payload=payload)
        restored=bundle.load_checkpoint_bundle(self.manager.root/'last')
        other=nn.Linear(4,3);other.load_state_dict(restored['model']);other_optimizer=torch.optim.AdamW(other.parameters(),lr=.01)
        other_optimizer.load_state_dict(restored['optimizer'])
        step(model,optimizer);step(other,other_optimizer)
        for name,value in model.state_dict().items():self.assertTrue(torch.equal(value,other.state_dict()[name]))
        self.equal(optimizer.state_dict(),other_optimizer.state_dict())

    def test_allocator_aware_envelope_and_pending_gradients_roundtrip_verbatim(self):
        for codec in ('none','gzip'):
            with self.subTest(codec=codec):
                payload=self.payload()
                payload['optimizer']={
                    'research_optimizer':{'version':1,'backend':'cpu_gradient_paged_adamw',
                        'optim_bits':8,'optimizer_steps':3,'evict_paged_states':True,'parameter_shapes':{0:[4,4]}},
                    'native_optimizer':payload['optimizer'],
                    'cpu_gradients':{0:torch.linspace(-1,1,16).reshape(4,4)}}
                manager=BestLastCheckpointManager(self.root/f'envelope-{codec}',DiskBudget(self.root,0))
                manager.save(lambda d:bundle.save_checkpoint_bundle(payload,d,compression=codec),
                             step=1,metric=.3,anticipated_bytes=1024**2)
                inspected=bundle.inspect_checkpoint_bundle(manager.root/'last')
                self.assertEqual(inspected['manifest']['optimizer_schema'],'cpu_gradient_paged_adamw_v1')
                self.equal(payload['optimizer'],bundle.load_checkpoint_bundle(manager.root/'last')['optimizer'])
                bundle._verify_component_structure_without_materializing(
                    (manager.root/'last').resolve(),inspected['manifest'])

    def test_allocator_aware_envelope_rejects_lost_shape_or_pending_gradient(self):
        payload=self.payload()
        payload['optimizer']={'research_optimizer':{'version':1,'backend':'cpu_gradient_paged_adamw',
            'optim_bits':8,'parameter_shapes':{0:[4,4]}},'native_optimizer':payload['optimizer'],
            'cpu_gradients':{0:torch.ones(2)}}
        with self.assertRaisesRegex(bundle.CheckpointBundleError,'pending CPU gradients'):
            self.save(payload=payload)
        self.assertFalse((self.manager.root/'last').exists())

    def test_completed_bundle_compaction_preserves_roles_models_and_is_idempotent(self):
        self.manager.save_torch(self.payload(1),checkpoint_format='bundle',step=1,metric=.3,anticipated_bytes=1024**2)
        self.manager.save_torch(self.payload(2),checkpoint_format='bundle',step=2,metric=.5,anticipated_bytes=1024**2)
        targets={role:(self.manager.root/role).resolve() for role in ('best','last')}
        model_bytes={role:(target/bundle.MODEL_FILE).read_bytes() for role,target in targets.items()}
        before={role:(target/bundle.MANIFEST_FILE).read_bytes() for role,target in targets.items()}
        (self.manager.root/'resolved_config.json').write_text(json.dumps({'stage':'SFT'}))
        (self.manager.root/'completed.json').write_text(json.dumps({'steps':2}))
        dry=bundle.compact_completed_bundle(self.manager.root,reserve_gb=0,dry_run=True)
        self.assertEqual(dry['status'],'validated_dry_run')
        for role,target in targets.items():self.assertEqual(before[role],(target/bundle.MANIFEST_FILE).read_bytes())
        result=bundle.compact_completed_bundle(self.manager.root,reserve_gb=0)
        self.assertGreater(result['logical_bytes_reclaimed'],0)
        for role,target in targets.items():
            self.assertEqual(target,(self.manager.root/role).resolve())
            self.assertEqual(model_bytes[role],(target/bundle.MODEL_FILE).read_bytes())
            restored=bundle.load_checkpoint_bundle(target)
            self.assertFalse(restored['resumable_optimizer']);self.assertTrue(restored['training_completed'])
            self.assertNotIn('optimizer',restored)
            self.equal(self.payload(1 if role=='best' else 2)['model'],restored['model'])
        manifests={role:(target/bundle.MANIFEST_FILE).read_bytes() for role,target in targets.items()}
        again=bundle.compact_completed_bundle(self.manager.root,reserve_gb=0)
        self.assertEqual(again['logical_bytes_reclaimed'],0)
        for role,target in targets.items():self.assertEqual(manifests[role],(target/bundle.MANIFEST_FILE).read_bytes())

    def test_completed_bundle_compaction_prevalidates_all_objects_before_changes(self):
        self.save(1,.3);self.save(2,.5)
        (self.manager.root/'resolved_config.json').write_text(json.dumps({'stage':'SFT'}))
        (self.manager.root/'completed.json').write_text(json.dumps({'steps':2}))
        best=(self.manager.root/'best').resolve();last=(self.manager.root/'last').resolve()
        before=(best/bundle.MANIFEST_FILE).read_bytes()
        with (last/bundle.OPTIMIZER_FILE).open('r+b') as handle:
            handle.seek(-10,2);value=handle.read(1);handle.seek(-1,1);handle.write(bytes([value[0]^1]))
        with self.assertRaisesRegex(bundle.CheckpointBundleError,'SHA256'):
            bundle.compact_completed_bundle(self.manager.root,reserve_gb=0)
        self.assertEqual(before,(best/bundle.MANIFEST_FILE).read_bytes())
        self.assertTrue((best/bundle.OPTIMIZER_FILE).exists())

    def test_capacity_math_counts_two_resume_payloads_plus_one_best_model(self):
        values=bundle.checkpoint_bundle_capacity(4*1024**3,8*1024**3)
        self.assertEqual(values['legacy_three_resume_payloads_bytes'],36*1024**3)
        self.assertEqual(values['bundle_best_model_plus_last_and_staging_bytes'],28*1024**3+values['manifest_and_owner_allowance_bytes'])
        self.assertTrue(values['does_not_claim_current_disk_fit'])


if __name__=='__main__':unittest.main()
