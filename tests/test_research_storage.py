"""Meaningful storage failure/recovery tests, without requiring CUDA or torch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.utils.research_storage import (
    BestLastCheckpointManager, DiskBudget, GIB, InsufficientDiskSpace,
)


class TestResearchStorage(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.budget = DiskBudget(self.root, reserve_gb=0)
        self.manager = BestLastCheckpointManager(self.root / "run", self.budget)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, step, metric=None, **kwargs):
        return self.manager.save(
            lambda directory: (directory / "state.json").write_text(json.dumps({"step": step})),
            step=step, metric=metric, anticipated_bytes=1024, **kwargs,
        )

    def test_only_best_and_last_retain_immutable_payloads(self):
        self.save(1, 0.5)
        self.assertEqual((self.manager.root / "best").resolve(), (self.manager.root / "last").resolve())
        self.save(2, 0.9)
        self.assertNotEqual((self.manager.root / "best").resolve(), (self.manager.root / "last").resolve())
        self.save(3, 0.4)
        self.save(4, 0.7)
        self.save(5, 0.8)
        self.assertEqual(len(list(self.manager.objects.iterdir())), 2)
        self.assertEqual(json.loads((self.manager.root / "best/state.json").read_text()), {"step": 3})
        self.assertEqual(json.loads((self.manager.root / "last/state.json").read_text()), {"step": 5})
        restored = BestLastCheckpointManager(self.manager.root, self.budget)
        self.assertEqual(restored.describe()["best"]["step"], 3)

    def test_writer_crash_preserves_both_previous_checkpoints(self):
        self.save(1, 0.5)
        self.save(2, 0.7)
        before = {name: (self.manager.root / name).resolve() for name in ("best", "last")}

        def crash(directory):
            (directory / "partial.bin").write_bytes(b"partial")
            raise RuntimeError("simulated killed writer")

        with self.assertRaisesRegex(RuntimeError, "simulated"):
            self.manager.save(crash, step=3, metric=0.1, anticipated_bytes=1024)
        self.assertEqual(before, {name: (self.manager.root / name).resolve() for name in before})
        self.assertEqual(len(list(self.manager.objects.iterdir())), 2)

    def test_peak_write_budget_checked_before_serialization(self):
        fake_usage = type("Usage", (), {"total": 2 * GIB, "used": GIB, "free": GIB})()
        budget = DiskBudget(self.root, reserve_gb=0.9)
        manager = BestLastCheckpointManager(self.root / "limited", budget)
        invoked = []
        with patch("src.utils.research_storage.shutil.disk_usage", return_value=fake_usage):
            with self.assertRaises(InsufficientDiskSpace):
                manager.save(lambda _: invoked.append(True), step=0, anticipated_bytes=GIB // 2)
        self.assertFalse(invoked)
        self.assertFalse((manager.root / "last").exists())

    def test_first_write_requires_size_estimate(self):
        with self.assertRaisesRegex(ValueError, "anticipated_bytes"):
            self.manager.save(lambda _: None, step=0)

    def test_unknown_directories_survive_cleanup(self):
        unknown = self.manager.objects / "user-data"
        unknown.mkdir()
        (unknown / "important").write_text("keep")
        self.save(1, 0.5)
        self.save(2, 0.4)
        self.assertEqual((unknown / "important").read_text(), "keep")

    def test_unmanaged_pointer_not_replaced(self):
        (self.manager.root / "best").mkdir()
        with self.assertRaisesRegex(ValueError, "unmanaged"):
            self.save(1, 0.5)
        self.assertTrue((self.manager.root / "best").is_dir())

    def test_owned_crash_orphans_reclaimed_on_next_save(self):
        self.save(1, 0.5)
        orphan = self.manager.objects / ".stage-interrupted"
        orphan.mkdir()
        (orphan / ".research_checkpoint.json").write_text(json.dumps({"owner": self.manager.owner}))
        (orphan / "partial").write_bytes(b"partial")
        self.save(2, 0.6)
        self.assertFalse(orphan.exists())

    def test_metric_direction_and_nan(self):
        manager = BestLastCheckpointManager(self.root / "maximize", self.budget, mode="max")
        for step, metric in enumerate([0.5, 0.8, 0.7]):
            manager.save(lambda d: (d / "state").write_text("state"), step=step,
                         metric=metric, anticipated_bytes=10)
        self.assertEqual(manager.describe()["best"]["metric"], 0.8)
        with self.assertRaises(ValueError):
            self.save(4, float("nan"))

    def test_payload_symlinks_cannot_point_outside_checkpoint(self):
        outside = self.root / "important.bin"
        outside.write_bytes(b"keep")
        def writer(directory):
            (directory / "external.bin").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.manager.save(writer, step=0, anticipated_bytes=1024)
        self.assertEqual(outside.read_bytes(), b"keep")
        self.assertEqual(list(self.manager.objects.iterdir()), [])

    def test_monitor_estimate_uses_measured_progress(self):
        spec = importlib.util.spec_from_file_location("monitor_research", Path(__file__).parents[1] / "scripts/monitor_research.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        progress = self.root / "progress.json"
        progress.write_text(json.dumps({"completed_records": 100, "total_records": 1000, "elapsed_seconds": 20}))
        result = module.read_progress(progress)
        self.assertEqual(result["average_records_per_second"], 5)
        self.assertAlmostEqual(result["eta_current_stage_hours"], 0.05)


if __name__ == "__main__":
    unittest.main()
