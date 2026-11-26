"""Completed optimizer compaction must preserve model state and checkpoint roles."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.utils.research_storage import BestLastCheckpointManager, DiskBudget, InsufficientDiskSpace

spec = importlib.util.spec_from_file_location("compact_research_checkpoints", Path(__file__).parents[1] / "scripts/compact_research_checkpoints.py")
compactor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compactor)


class TestCompletedCheckpointCompaction(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def make_run(self, two=False, stage="SFT", large_model=False):
        directory = self.root / "run"
        manager = BestLastCheckpointManager(directory, DiskBudget(self.root, 0))
        def payload(step):
            return {"model": {"llm.weight": torch.arange(1024*1024,dtype=torch.float32)+step if large_model else torch.arange(16, dtype=torch.bfloat16).reshape(4, 4) + step,
                              "projector.weight": torch.linspace(-1, 1, 12).reshape(3, 4)},
                    "optimizer": {"state": {0: {"exp_avg": torch.ones(100_000), "exp_avg_sq": torch.ones(100_000)}}},
                    "scheduler": {"last_epoch": step}, "step": step,
                    "config": {"stage": stage, "anything": [1, "kept"]},
                    "rng_torch": torch.get_rng_state(), "rng_python": random.getstate(),
                    "rng_cuda": [], "manifest_sha256": "manifest-hash", "model_ref": {"revision": "pinned"}}
        manager.save_torch(payload(1), step=1, metric=0.4, anticipated_bytes=2_000_000)
        if two:
            manager.save_torch(payload(2), step=2, metric=0.5, anticipated_bytes=2_000_000)
        (directory / "resolved_config.json").write_text(json.dumps({"output": str(directory), "stage": stage}))
        (directory / "completed.json").write_text(json.dumps({"steps": 2 if two else 1}))
        return directory, manager
    def load(self, directory, name):
        return torch.load(directory / name / "state.pt", map_location="cpu", weights_only=False)

    def test_preserves_every_model_tensor_dtype_rng_and_best_last_association(self):
        directory, manager = self.make_run(two=True)
        before = {name: self.load(directory, name) for name in ("best", "last")}
        original_markers = {name: json.loads((directory / name / ".research_checkpoint.json").read_text()) for name in ("best", "last")}
        report = compactor.compact_completed_checkpoint(directory, reserve_gb=0)
        self.assertEqual(report["status"], "completed")
        self.assertGreater(report["logical_bytes_reclaimed"], 1_000_000)
        for name in ("best", "last"):
            after = self.load(directory, name)
            self.assertNotIn("optimizer", after)
            self.assertTrue(after["training_completed"])
            self.assertFalse(after["resumable_optimizer"])
            self.assertEqual(set(after["model"]), set(before[name]["model"]))
            for key, tensor in before[name]["model"].items():
                self.assertEqual(tensor.dtype, after["model"][key].dtype)
                torch.testing.assert_close(tensor, after["model"][key], rtol=0, atol=0)
            for key in ("scheduler", "step", "config", "rng_python", "rng_cuda", "manifest_sha256", "model_ref"):
                self.assertEqual(before[name][key], after[key])
            torch.testing.assert_close(before[name]["rng_torch"], after["rng_torch"], rtol=0, atol=0)
            marker = json.loads((directory / name / ".research_checkpoint.json").read_text())
            self.assertEqual(marker["step"], original_markers[name]["step"])
            self.assertEqual(marker["metric"], original_markers[name]["metric"])
        self.assertEqual(len(list(manager.objects.iterdir())), 2)
        self.assertNotEqual((directory / "best").resolve(), (directory / "last").resolve())

    def test_shared_best_last_stays_shared_and_repeated_compaction_is_idempotent(self):
        directory, manager = self.make_run()
        compactor.compact_completed_checkpoint(directory, reserve_gb=0)
        target = (directory / "best").resolve()
        self.assertEqual(target, (directory / "last").resolve())
        report = compactor.compact_completed_checkpoint(directory, reserve_gb=0)
        self.assertTrue(report["checkpoints"][0]["already_compacted"])
        self.assertEqual(target, (directory / "best").resolve())
        self.assertEqual(len(list(manager.objects.iterdir())), 1)

    def test_serialization_failure_preserves_prior_resumable_checkpoints(self):
        directory, manager = self.make_run(two=True)
        pointers = {name: (directory / name).resolve() for name in ("best", "last")}
        with patch.object(torch, "save", side_effect=RuntimeError("simulated write failure")):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                compactor.compact_completed_checkpoint(directory, reserve_gb=0)
        self.assertEqual(pointers, {name: (directory / name).resolve() for name in pointers})
        self.assertIn("optimizer", self.load(directory, "last"))
        self.assertEqual(len(list(manager.objects.iterdir())), 2)

    def test_reserve_failure_does_not_replace_or_delete_anything(self):
        directory, manager = self.make_run()
        original = (directory / "last").resolve()
        with patch.object(DiskBudget, "ensure_space", side_effect=InsufficientDiskSpace("reserved")):
            with self.assertRaises(InsufficientDiskSpace):
                compactor.compact_completed_checkpoint(directory, reserve_gb=0)
        self.assertEqual(original, (directory / "last").resolve())
        self.assertEqual(len(list(manager.objects.iterdir())), 1)

    def test_interrupted_pointer_update_can_be_recovered_without_weight_loss(self):
        directory, manager = self.make_run(two=True)
        original_set = BestLastCheckpointManager._set_pointer
        def interrupted_set(instance, name, target):
            if name == "last":
                raise RuntimeError("interrupted before last pointer")
            original_set(instance, name, target)
        with patch.object(BestLastCheckpointManager, "_set_pointer", interrupted_set):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                compactor.compact_completed_checkpoint(directory, reserve_gb=0)
        self.assertEqual(self.load(directory, "best")["step"], 1)
        self.assertEqual(self.load(directory, "last")["step"], 2)
        compactor.compact_completed_checkpoint(directory, reserve_gb=0)
        self.assertNotIn("optimizer", self.load(directory, "best"))
        self.assertNotIn("optimizer", self.load(directory, "last"))
        self.assertEqual(len(list(manager.objects.iterdir())), 2)

    def test_incomplete_or_non_sft_stages_are_refused(self):
        directory, _ = self.make_run(stage="A1")
        with self.assertRaisesRegex(ValueError, "SFT"):
            compactor.compact_completed_checkpoint(directory, reserve_gb=0)
        (directory / "resolved_config.json").write_text(json.dumps({"output": str(directory), "stage": "SFT"}))
        (directory / "completed.json").unlink()
        with self.assertRaisesRegex(ValueError, "completed.json"):
            compactor.compact_completed_checkpoint(directory, reserve_gb=0)

    def test_dry_run_and_unknown_files_are_preserved(self):
        directory, manager = self.make_run()
        unknown = manager.objects / "user-data"
        unknown.mkdir()
        (unknown / "keep").write_text("important")
        original = (directory / "last").resolve()
        report = compactor.compact_completed_checkpoint(directory, reserve_gb=0, dry_run=True)
        self.assertEqual(report["status"], "validated_dry_run")
        self.assertEqual(original, (directory / "last").resolve())
        compactor.compact_completed_checkpoint(directory, reserve_gb=0)
        self.assertEqual((unknown / "keep").read_text(), "important")

    def test_distinct_objects_fit_one_model_staging_headroom_and_reclaim_before_next(self):
        directory,manager=self.make_run(two=True,large_model=True)
        old_best=(directory/'best').resolve()
        baseline=compactor._tree_size(manager.objects)
        reserve=1024**3;headroom=22*1024**2
        checks=[];original_ensure=DiskBudget.ensure_space
        def report(budget):
            current=compactor._tree_size(manager.objects)
            free=reserve+headroom-(current-baseline)
            return {'root':str(budget.root),'total_bytes':reserve+headroom+baseline,
                    'used_bytes':current,'free_bytes':free,'reserve_bytes':budget.reserve_bytes,
                    'available_after_reserve_bytes':max(0,free-budget.reserve_bytes),
                    'free_gib':free/1024**3,'reserve_gib':budget.reserve_bytes/1024**3}
        def ensure(budget,anticipated_bytes=0):
            if anticipated_bytes:
                checks.append({'anticipated':anticipated_bytes,'old_best_exists':old_best.exists(),
                               'objects':len(list(manager.objects.iterdir()))})
            return original_ensure(budget,anticipated_bytes)
        before={name:self.load(directory,name)['model'] for name in ('best','last')}
        with patch.object(DiskBudget,'report',report),patch.object(DiskBudget,'ensure_space',ensure):
            result=compactor.compact_completed_checkpoint(directory,reserve_gb=1)
        self.assertEqual(len(checks),2)
        self.assertTrue(checks[0]['old_best_exists'])
        self.assertFalse(checks[1]['old_best_exists'])
        self.assertEqual(checks[1]['objects'],2)
        # Holding the first staged model before the second would consume more
        # than the available estimate margin, so the former all-at-once scheme
        # fails this actual file-size-based disk simulation.
        first_model_size=(directory/'best/state.pt').stat().st_size
        self.assertGreater(checks[1]['anticipated'],headroom-first_model_size)
        self.assertEqual(result['status'],'completed')
        for name in before:
            for key,value in before[name].items():self.assertTrue(torch.equal(value,self.load(directory,name)['model'][key]))

    def test_crash_between_replacements_keeps_first_committed_and_retry_writes_only_second(self):
        directory,manager=self.make_run(two=True)
        before={name:self.load(directory,name) for name in ('best','last')}
        old_best=(directory/'best').resolve();old_last=(directory/'last').resolve()
        original_save=torch.save;calls=0
        def save(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('crash staging second replacement')
            return original_save(*args,**kwargs)
        with patch.object(torch,'save',side_effect=save):
            with self.assertRaisesRegex(RuntimeError,'second replacement'):
                compactor.compact_completed_checkpoint(directory,reserve_gb=0)
        first=(directory/'best').resolve()
        self.assertNotEqual(first,old_best);self.assertFalse(old_best.exists())
        self.assertEqual((directory/'last').resolve(),old_last)
        self.assertNotIn('optimizer',self.load(directory,'best'))
        self.assertIn('optimizer',self.load(directory,'last'))
        with patch.object(torch,'save',wraps=original_save) as retry_save:
            compactor.compact_completed_checkpoint(directory,reserve_gb=0)
        self.assertEqual(retry_save.call_count,1)
        self.assertEqual(first,(directory/'best').resolve())
        self.assertEqual(len(list(manager.objects.iterdir())),2)
        for name in ('best','last'):
            after=self.load(directory,name)
            self.assertEqual(after['step'],before[name]['step'])
            for key,value in before[name]['model'].items():self.assertTrue(torch.equal(value,after['model'][key]))
            self.assertEqual(after['scheduler'],before[name]['scheduler'])
            self.assertTrue(torch.equal(after['rng_torch'],before[name]['rng_torch']))

    def test_shared_pointer_publish_crash_reuses_sha_bound_replacement_without_second_write(self):
        directory,manager=self.make_run()
        before=self.load(directory,'last');original_set=BestLastCheckpointManager._set_pointer
        def fail_last(instance,name,target):
            if name=='last':raise RuntimeError('crash publishing shared last')
            return original_set(instance,name,target)
        with patch.object(BestLastCheckpointManager,'_set_pointer',fail_last):
            with self.assertRaisesRegex(RuntimeError,'shared last'):
                compactor.compact_completed_checkpoint(directory,reserve_gb=0)
        replacement=(directory/'best').resolve()
        self.assertNotEqual(replacement,(directory/'last').resolve())
        with patch.object(torch,'save',side_effect=AssertionError('retry must reuse existing replacement')):
            result=compactor.compact_completed_checkpoint(directory,reserve_gb=0)
        self.assertEqual(replacement,(directory/'last').resolve())
        self.assertEqual(len(list(manager.objects.iterdir())),1)
        self.assertTrue(any(row.get('reuses_published_object') for row in result['checkpoints']))
        for key,value in before['model'].items():self.assertTrue(torch.equal(value,self.load(directory,'last')['model'][key]))

    def test_shared_recovery_refuses_source_modified_after_first_role_publication(self):
        directory,_=self.make_run();original_set=BestLastCheckpointManager._set_pointer
        def fail_last(instance,name,target):
            if name=='last':raise RuntimeError('partial shared publication')
            return original_set(instance,name,target)
        with patch.object(BestLastCheckpointManager,'_set_pointer',fail_last):
            with self.assertRaises(RuntimeError):compactor.compact_completed_checkpoint(directory,reserve_gb=0)
        replacement=(directory/'best').resolve()
        source=self.load(directory,'last');source['model']['llm.weight'][0,0]+=1
        torch.save(source,directory/'last/state.pt')
        with self.assertRaisesRegex(ValueError,'differs from its published'):
            compactor.compact_completed_checkpoint(directory,reserve_gb=0)
        self.assertEqual(replacement,(directory/'best').resolve())
        self.assertIn('optimizer',self.load(directory,'last'))


if __name__ == "__main__":
    unittest.main()
