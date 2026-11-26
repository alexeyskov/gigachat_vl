"""Guards preventing disposable-smoke cleanup from deleting research weights."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.utils.research_storage import BestLastCheckpointManager, DiskBudget

spec = importlib.util.spec_from_file_location("prune_technical_smoke", Path(__file__).parents[1] / "scripts/prune_technical_smoke.py")
pruner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pruner)


class TestTechnicalSmokeCleanup(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()

    def make_run(self, name="smallthinker__smoke_4f_sft", two_checkpoints=False):
        directory = self.root / name
        manager = BestLastCheckpointManager(directory, DiskBudget(self.root, 0))
        manager.save(lambda d: (d / "state.pt").write_bytes(b"checkpoint-one"), step=1, metric=0.4, anticipated_bytes=100)
        if two_checkpoints:
            manager.save(lambda d: (d / "state.pt").write_bytes(b"checkpoint-two"), step=2, metric=0.5, anticipated_bytes=100)
        (directory / "resolved_config.json").write_text(json.dumps({"output": str(directory), "stage": "SFT", "max_records": 128, "max_steps": 2}))
        (directory / "completed.json").write_text(json.dumps({"steps": 2 if two_checkpoints else 1}))
        (directory / "metrics.jsonl").write_text("important results")
        self.write_validation(directory)
        return directory

    def write_validation(self, directory):
        owner = json.loads((directory / ".checkpoint_manager.json").read_text())["owner"]
        targets = {(directory / name / "state.pt").resolve() for name in ("best", "last")}
        receipts = []
        for target in targets:
            metadata = json.loads((target.parent / ".research_checkpoint.json").read_text())
            receipts.append({"schema_version": 1, "status": "passed", "checkpoint_owner": owner,
                             "checkpoint_path": str(target), "checkpoint_step": metadata["step"],
                             "validated_tensor_count": 10, "all_tensors_finite": True,
                             "expected_keys_match": True, "checkpoint_size_bytes": target.stat().st_size,
                             "checkpoint_mtime_ns": target.stat().st_mtime_ns})
        (directory / "checkpoint_roundtrip.json").write_text(json.dumps({"schema_version": 1, "status": "passed", "checkpoints": receipts}))

    def rewrite(self, directory, file, **changes):
        path = directory / file
        data = json.loads(path.read_text())
        data.update(changes)
        path.write_text(json.dumps(data))

    def test_valid_smoke_weights_removed_metrics_and_unknown_files_retained(self):
        directory = self.make_run(two_checkpoints=True)
        unknown = directory / ".checkpoint_objects/user-data"
        unknown.mkdir()
        (unknown / "keep").write_bytes(b"keep")
        preview = pruner.prune_technical_smoke(directory, dry_run=True)
        self.assertEqual(preview["status"], "validated_dry_run")
        self.assertTrue((directory / "last/state.pt").exists())
        result = pruner.prune_technical_smoke(directory)
        self.assertEqual(result["status"], "pruned")
        self.assertFalse((directory / "last").exists())
        self.assertFalse((directory / "best").exists())
        self.assertEqual((directory / "metrics.jsonl").read_text(), "important results")
        self.assertTrue((directory / "resolved_config.json").exists())
        self.assertTrue((directory / "checkpoint_roundtrip.json").exists())
        self.assertEqual((unknown / "keep").read_bytes(), b"keep")
        self.assertTrue(pruner.prune_technical_smoke(directory)["already_pruned"])

    def test_scientific_pilot_primary_or_non_smoke_names_refused(self):
        for name in ("smallthinker-pilot", "smallthinker__smoke_primary", "smallthinker__smoke_pilot", "ordinary-run"):
            with self.subTest(name=name):
                directory = self.make_run(name)
                with self.assertRaises(ValueError):
                    pruner.prune_technical_smoke(directory)
                self.assertTrue((directory / "last/state.pt").exists())

    def test_authorized_full_smoke_basename_requires_exact_research_directory(self):
        name = "smallthinker4__siglip2__3c__SFT_smoke_s42"
        directory = self.make_run("runs/research/" + name)
        with patch.object(pruner, "ROOT", self.root):
            result = pruner.prune_technical_smoke(directory, dry_run=True)
        self.assertEqual(result["status"], "validated_dry_run")
        self.assertTrue((directory / "last/state.pt").exists())
        elsewhere = self.make_run(name)
        with patch.object(pruner, "ROOT", self.root), self.assertRaisesRegex(ValueError, "explicitly authorized"):
            pruner.prune_technical_smoke(elsewhere)
        self.assertTrue((elsewhere / "last/state.pt").exists())

    def test_full_smoke_exception_does_not_accept_lookalike_or_other_recipe(self):
        for name in ("smallthinker4__siglip2__3c__SFT_smoke_s43",
                     "smallthinker4__siglip2__1a__SFT_smoke_s42",
                     "smallthinker4__siglip2__3c__SFT_smoke_s42_copy"):
            with self.subTest(name=name):
                directory = self.make_run("runs/research/" + name)
                with patch.object(pruner, "ROOT", self.root), self.assertRaisesRegex(ValueError, "explicitly authorized"):
                    pruner.prune_technical_smoke(directory)
                self.assertTrue((directory / "last/state.pt").exists())

    def test_pilot_primary_manifest_profile_and_scope_are_refused(self):
        for field, value in (("manifest", "datasets/research/pilot-v1/train.jsonl"),
                             ("validation_manifest", "datasets/PRIMARY/validation.jsonl"),
                             ("vision_preprocessing_profile", "primary_native"),
                             ("experiment_scope", {"tier": "pilot"})):
            with self.subTest(field=field):
                directory = self.make_run("test__smoke_" + field)
                self.rewrite(directory, "resolved_config.json", **{field: value})
                with self.assertRaisesRegex(ValueError, "manifest, profile or scope"):
                    pruner.prune_technical_smoke(directory)
                self.assertTrue((directory / "last/state.pt").exists())

    def test_explicit_small_max_records_and_steps_required(self):
        for field, value in (("max_records", None), ("max_records", 2001), ("max_records", True), ("max_steps", None), ("max_steps", 101)):
            with self.subTest(field=field, value=value):
                directory = self.make_run("test__smoke_" + field + str(value))
                self.rewrite(directory, "resolved_config.json", **{field: value})
                with self.assertRaises(ValueError):
                    pruner.prune_technical_smoke(directory)
                self.assertTrue((directory / "last/state.pt").exists())

    def test_structural_only_validation_is_insufficient(self):
        directory = self.make_run()
        self.rewrite(directory, "checkpoint_roundtrip.json", status="structural_only")
        with self.assertRaisesRegex(ValueError, "roundtrip"):
            pruner.prune_technical_smoke(directory)
        self.assertTrue((directory / "last/state.pt").exists())

    def test_changed_checkpoint_or_wrong_owner_receipt_refused(self):
        directory = self.make_run()
        (directory / "last/state.pt").write_bytes(b"changed-payload")
        with self.assertRaisesRegex(ValueError, "changed since"):
            pruner.prune_technical_smoke(directory)
        self.write_validation(directory)
        report_path = directory / "checkpoint_roundtrip.json"
        report = json.loads(report_path.read_text())
        report["checkpoints"][0]["checkpoint_owner"] = "wrong-owner"
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "owner mismatch"):
            pruner.prune_technical_smoke(directory)

    def test_both_distinct_best_and_last_need_validation(self):
        directory = self.make_run(two_checkpoints=True)
        path = directory / "checkpoint_roundtrip.json"
        report = json.loads(path.read_text())
        report["checkpoints"] = report["checkpoints"][:1]
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "No full roundtrip"):
            pruner.prune_technical_smoke(directory)
        self.assertTrue((directory / "best/state.pt").exists())
        self.assertTrue((directory / "last/state.pt").exists())

    def test_bundle_requires_and_rechecks_every_component_identity(self):
        import torch
        from src.utils.research_checkpoint_bundle import save_checkpoint_bundle
        from src.utils.research_checkpoint_compat import checkpoint_identity
        directory=self.make_run()
        target=(directory/'last/state.pt').resolve()
        target.unlink()
        save_checkpoint_bundle({'step':1,'model':{'weight':torch.arange(4.)},'resumable_optimizer':False},target.parent)
        self.write_validation(directory)
        with self.assertRaisesRegex(ValueError,'component-bound'):
            pruner.prune_technical_smoke(directory,dry_run=True)
        receipt_path=directory/'checkpoint_roundtrip.json'
        receipt=json.loads(receipt_path.read_text())
        receipt['checkpoints'][0]['checkpoint_bundle_identity']=checkpoint_identity(target)
        receipt_path.write_text(json.dumps(receipt))
        self.assertEqual(pruner.prune_technical_smoke(directory,dry_run=True)['status'],'validated_dry_run')
        component=target.parent/'model.state.pt';before=component.stat()
        with component.open('r+b') as handle:
            handle.seek(-1,2);value=handle.read(1);handle.seek(-1,1);handle.write(bytes([value[0]^1]))
        os.utime(component,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'SHA256'):
            pruner.prune_technical_smoke(directory,dry_run=True)
        self.assertTrue(component.is_file())

    def test_missing_completion_or_validation_marker_refused(self):
        directory = self.make_run()
        (directory / "completed.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            pruner.prune_technical_smoke(directory)
        self.assertTrue((directory / "last/state.pt").exists())

    def test_checkpoint_objects_symlink_refused(self):
        directory = self.make_run()
        real = directory / ".checkpoint_objects"
        moved = self.root / "moved-objects"
        real.rename(moved)
        real.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "local directory"):
            pruner.prune_technical_smoke(directory)
        self.assertTrue((moved / next(moved.iterdir()).name / "state.pt").exists())


if __name__ == "__main__":
    unittest.main()
