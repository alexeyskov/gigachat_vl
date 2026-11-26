import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from src.utils.research_storage import BestLastCheckpointManager, DiskBudget
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("research_queue", Path(__file__).parents[1] / "scripts/research_queue.py")
queue_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue_module)


class TestResearchQueue(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(queue_module, "ROOT", self.root)
        self.root_patch.start()
        self.calls = []

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def specification(self, jobs):
        path = self.root / "queue.json"
        path.write_text(json.dumps({"queue_id": "test-smoke", "reserve_gb": 0, "jobs": jobs}))
        return path

    def job(self, name, requires=None):
        return {"id": name, "requires": requires or [], "argv": ["--stage", "A0", "--recipe", "4f", "--output", str(self.root / name)]}

    def successful_process(self, command, **_kwargs):
        self.calls.append(command)
        output = Path(queue_module.argv_value(command, "--output"))
        if Path(command[2]).name=="eval_research_checkpoint.py":
            from src.eval.research_metrics import aggregate_scores
            manifest = Path(queue_module.argv_value(command,"--manifest"))
            checkpoint = Path(queue_module.argv_value(command,"--run"))/queue_module.argv_value(command,"--checkpoint")/"state.pt"
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            protocol = {"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "generation":{"max_new_tokens":int(queue_module.argv_value(command,"--max-new-tokens")),"do_sample":False,"use_cache":True}}
            protocol_sha = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
            rows = [dict(record,prediction="yes",score=1.,protocol_sha256=protocol_sha) for record in records]
            summary = aggregate_scores(rows)
            summary.update(complete_manifest=True,manifest_records=len(rows),protocol_sha256=protocol_sha,
                           checkpoint_sha256=protocol["checkpoint_sha256"],manifest_sha256=protocol["manifest_sha256"])
            output.mkdir(parents=True,exist_ok=True)
            (output/"protocol.json").write_text(json.dumps(protocol))
            (output/"summary.json").write_text(json.dumps(summary))
            (output/"predictions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            return type("Process",(),{"pid":1234,"returncode":0,"poll":lambda _self:0})()
        (output / "last").mkdir(parents=True, exist_ok=True)
        (output / "last/state.pt").write_bytes(b"fake-test-payload")
        (output / "best").mkdir(exist_ok=True)
        (output / "best/state.pt").write_bytes(b"fake-test-payload")
        (output / "completed.json").write_text(json.dumps({"steps": 1, "records": 1}))
        return type("Process", (), {"pid": 1234, "returncode": 0, "poll": lambda _self: 0})()

    def test_runs_dependencies_once_and_restores_completed_state(self):
        path = self.specification([self.job("alignment"), self.job("sft", ["alignment"])])
        with patch.object(queue_module.subprocess, "Popen", side_effect=self.successful_process):
            self.assertEqual(queue_module.ResearchQueue(path).run(), 0)
            self.assertEqual(queue_module.ResearchQueue(path).run(), 0)
        self.assertEqual(len(self.calls), 2)
        state = json.loads(path.with_suffix(".state.json").read_text())
        self.assertEqual(state["status"], "completed")

    def test_failures_are_persisted_and_not_retried_implicitly(self):
        path = self.specification([self.job("alignment"), self.job("sft", ["alignment"])])
        fake = type("Process", (), {"pid": 1234, "returncode": 7, "poll": lambda _self: 7})()
        with patch.object(queue_module.subprocess, "Popen", return_value=fake) as popen:
            self.assertEqual(queue_module.ResearchQueue(path).run(), 1)
            self.assertEqual(queue_module.ResearchQueue(path).run(), 1)
            self.assertEqual(popen.call_count, 1)
        state = json.loads(path.with_suffix(".state.json").read_text())
        self.assertEqual(state["jobs"]["alignment"]["status"], "failed")
        self.assertNotIn("sft", state["jobs"])

    def test_interrupted_job_uses_last_checkpoint_to_resume(self):
        path = self.specification([self.job("alignment")])
        (self.root / "alignment/last").mkdir(parents=True)
        (self.root / "alignment/last/state.pt").write_bytes(b"resume-test-payload")
        with patch.object(queue_module.subprocess, "Popen", side_effect=self.successful_process):
            self.assertEqual(queue_module.ResearchQueue(path).run(), 0)
        self.assertIn("--resume", self.calls[0])

    def test_return_zero_without_artifacts_is_a_failure(self):
        path = self.specification([self.job("alignment")])
        fake = type("Process", (), {"pid": 1234, "returncode": 0, "poll": lambda _self: 0})()
        with patch.object(queue_module.subprocess, "Popen", return_value=fake):
            self.assertEqual(queue_module.ResearchQueue(path).run(), 1)
        state = json.loads(path.with_suffix(".state.json").read_text())
        self.assertEqual(state["jobs"]["alignment"]["status"], "failed")

    def test_changed_spec_requires_new_state(self):
        path = self.specification([self.job("alignment")])
        with patch.object(queue_module.subprocess, "Popen", side_effect=self.successful_process):
            queue_module.ResearchQueue(path).run()
        spec_data = json.loads(path.read_text())
        spec_data["queue_id"] = "changed"
        path.write_text(json.dumps(spec_data))
        with self.assertRaisesRegex(ValueError, "spec changed"):
            queue_module.ResearchQueue(path)

    def test_duplicate_outputs_and_forward_dependencies_rejected(self):
        a, b = self.job("a"), self.job("b")
        b["argv"] = a["argv"]
        with self.assertRaisesRegex(ValueError, "Output reused"):
            queue_module.validate_spec({"jobs": [a, b]})
        with self.assertRaisesRegex(ValueError, "later dependency"):
            queue_module.validate_spec({"jobs": [self.job("a", ["b"]), self.job("b")]})

    def test_optional_completed_compaction_hook_runs_before_success(self):
        job = self.job("sft")
        job["argv"][1] = "SFT"
        job["compact_completed"] = True
        path = self.specification([job])
        with patch.object(queue_module.subprocess, "Popen", side_effect=self.successful_process), \
             patch("scripts.compact_research_checkpoints.compact_completed_checkpoint", return_value={"status": "completed", "resumable_optimizer": False}) as compact:
            self.assertEqual(queue_module.ResearchQueue(path).run(), 0)
        compact.assert_called_once()
        state = json.loads(path.with_suffix(".state.json").read_text())
        self.assertFalse(state["jobs"]["sft"]["compaction"]["resumable_optimizer"])

    def test_compaction_flag_refuses_active_alignment_stages(self):
        job = self.job("alignment")
        job["compact_completed"] = True
        with self.assertRaisesRegex(ValueError, "SFT"):
            queue_module.validate_spec({"jobs": [job]})

    def test_compaction_failure_retries_postprocessing_without_trainer(self):
        job = self.evaluated_job()
        job["compact_completed"] = True
        path = self.specification([job])
        attempts = []
        def compact(directory, **_):
            state = json.loads(path.with_suffix(".state.json").read_text())
            self.assertTrue(state["jobs"]["sft"]["training_completed"])
            attempts.append(directory)
            # Simulate optimizer retirement before a later publication failure.
            for name in ("best", "last"):
                (directory/name/"state.pt").write_bytes(b"optimizer-retired-model")
            if len(attempts) == 1:
                raise RuntimeError("publication failed after optimizer retirement")
            return {"status":"completed","resumable_optimizer":False}
        with patch.object(queue_module.subprocess,"Popen",side_effect=self.successful_process), \
             patch("scripts.compact_research_checkpoints.compact_completed_checkpoint",side_effect=compact):
            self.assertEqual(queue_module.ResearchQueue(path).run(),1)
            state = json.loads(path.with_suffix(".state.json").read_text())
            self.assertEqual(state["jobs"]["sft"]["status"],"training_completed_compaction_failed")
            self.assertEqual(queue_module.ResearchQueue(path).run(),1)
            self.assertEqual(len(attempts),1)
            self.assertEqual(queue_module.ResearchQueue(path,retry_failed=True).run(),0)
        self.assertEqual(len(attempts),2)
        self.assertEqual([Path(c[2]).name for c in self.calls],["run_research_stage.py","eval_research_checkpoint.py"])

    def test_crash_after_optimizer_retirement_recovers_stage_without_evaluation(self):
        job = self.job("sft")
        job["argv"][1] = "SFT"
        job["compact_completed"] = True
        path = self.specification([job])
        attempts = []
        def compact(directory, **_):
            attempts.append(directory)
            if len(attempts) == 1:
                raise SystemExit("simulated abrupt controller exit")
            return {"status":"completed","resumable_optimizer":False}
        with patch.object(queue_module.subprocess,"Popen",side_effect=self.successful_process), \
             patch("scripts.compact_research_checkpoints.compact_completed_checkpoint",side_effect=compact):
            with self.assertRaises(SystemExit):
                queue_module.ResearchQueue(path).run()
            state = json.loads(path.with_suffix(".state.json").read_text())
            self.assertTrue(state["jobs"]["sft"]["training_completed"])
            self.assertNotIn("compaction",state["jobs"]["sft"])
            self.assertEqual(queue_module.ResearchQueue(path).run(),0)
        self.assertEqual(len(attempts),2)
        self.assertEqual(len(self.calls),1)

    def evaluated_job(self,name="sft"):
        job = self.job(name)
        job["argv"][1] = "SFT"
        manifest = self.root/"evaluation.jsonl"
        groups = ["general_vqa","ocr_documents","charts_diagrams","knowledge_reasoning"]
        records = [{"record_id":str(i),"question":"question","answers":["yes"],
                    "scorer":"exact_match","task_group":group,"source":{"repo":"test"}}
                   for i,group in enumerate(groups)]
        manifest.write_text("".join(json.dumps(row)+"\n" for row in records))
        job["evaluation"] = {"manifest":str(manifest),"output":str(self.root/name/"evaluation/heldout"),
                             "checkpoint":"best","max_new_tokens":64}
        return job

    def test_evaluation_runs_after_compaction_and_verified_completion(self):
        job = self.evaluated_job()
        job["compact_completed"] = True
        path = self.specification([job])
        events = []
        def compact(directory,**_):
            events.append("compaction")
            for name in ("best","last"):
                (directory/name/"state.pt").write_bytes(b"compacted-weights")
            return {"status":"completed"}
        def process(command,**kwargs):
            events.append(Path(command[2]).name)
            return self.successful_process(command,**kwargs)
        with patch.object(queue_module.subprocess,"Popen",side_effect=process), \
             patch("scripts.compact_research_checkpoints.compact_completed_checkpoint",side_effect=compact):
            self.assertEqual(queue_module.ResearchQueue(path).run(),0)
            self.assertEqual(queue_module.ResearchQueue(path).run(),0)
        self.assertEqual(events,["run_research_stage.py","compaction","eval_research_checkpoint.py"])
        state = json.loads(path.with_suffix(".state.json").read_text())
        record = state["jobs"]["sft"]
        self.assertEqual(record["status"],"completed")
        self.assertEqual(record["evaluation"]["records"],4)
        self.assertEqual(record["evaluation"]["checkpoint_sha256"],hashlib.sha256(b"compacted-weights").hexdigest())

    def test_completed_legacy_state_without_evaluation_receipt_is_evaluated(self):
        job = self.evaluated_job()
        path = self.specification([job])
        with patch.object(queue_module.subprocess,"Popen",side_effect=self.successful_process):
            self.assertEqual(queue_module.ResearchQueue(path).run(),0)
            state_path = path.with_suffix(".state.json")
            state = json.loads(state_path.read_text())
            state["jobs"]["sft"].pop("evaluation")
            state_path.write_text(json.dumps(state))
            self.assertEqual(queue_module.ResearchQueue(path).run(),0)
        self.assertEqual(len(self.calls),3)
        self.assertEqual(Path(self.calls[-1][2]).name,"eval_research_checkpoint.py")
        self.assertIn("--resume",self.calls[-1])

    def test_evaluation_failure_is_distinct_and_explicit_retry_does_not_retrain(self):
        path = self.specification([self.evaluated_job()])
        def fail_evaluation(command,**kwargs):
            if Path(command[2]).name=="eval_research_checkpoint.py":
                self.calls.append(command)
                return type("Process",(),{"pid":1234,"returncode":9,"poll":lambda _self:9})()
            return self.successful_process(command,**kwargs)
        with patch.object(queue_module.subprocess,"Popen",side_effect=fail_evaluation):
            self.assertEqual(queue_module.ResearchQueue(path).run(),1)
            self.assertEqual(queue_module.ResearchQueue(path).run(),1)
        state = json.loads(path.with_suffix(".state.json").read_text())
        self.assertEqual(state["jobs"]["sft"]["status"],"training_completed_eval_failed")
        self.assertEqual(len(self.calls),2)
        with patch.object(queue_module.subprocess,"Popen",side_effect=self.successful_process):
            self.assertEqual(queue_module.ResearchQueue(path,retry_failed=True).run(),0)
        self.assertEqual(len(self.calls),3)
        self.assertEqual(Path(self.calls[-1][2]).name,"eval_research_checkpoint.py")

    def test_stale_checkpoint_hash_or_partial_summary_never_marks_completed(self):
        for fault in ("checkpoint_sha256","complete_manifest","image_ablation"):
            with self.subTest(fault=fault):
                name = "sft-"+fault
                job = self.evaluated_job(name)
                path = self.root/(name+".queue.json")
                path.write_text(json.dumps({"reserve_gb":0,"jobs":[job]}))
                def corrupt(command,**kwargs):
                    result = self.successful_process(command,**kwargs)
                    if Path(command[2]).name=="eval_research_checkpoint.py":
                        summary_path = Path(job["evaluation"]["output"])/"summary.json"
                        summary = json.loads(summary_path.read_text())
                        summary[fault] = ({"variant":"shuffle"} if fault=="image_ablation"
                                          else "wrong" if fault.endswith("sha256") else False)
                        summary_path.write_text(json.dumps(summary))
                    return result
                with patch.object(queue_module.subprocess,"Popen",side_effect=corrupt):
                    self.assertEqual(queue_module.ResearchQueue(path).run(),1)
                state = json.loads(path.with_suffix(".state.json").read_text())
                self.assertEqual(state["jobs"][name]["status"],"training_completed_eval_failed")

    def test_evaluation_validation_and_resume_spec_mismatch(self):
        job = self.evaluated_job()
        job["evaluation"]["max_new_tokens"] = 0
        with self.assertRaisesRegex(ValueError,"max_new_tokens"):
            queue_module.validate_spec({"jobs":[job]})
        job = self.evaluated_job()
        path = self.specification([job])
        with patch.object(queue_module.subprocess,"Popen",side_effect=self.successful_process):
            self.assertEqual(queue_module.ResearchQueue(path).run(),0)
        data = json.loads(path.read_text())
        data["jobs"][0]["evaluation"]["max_new_tokens"] = 32
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"spec changed"):
            queue_module.ResearchQueue(path)

    def test_evaluation_wait_emits_heartbeat_and_rechecks_disk(self):
        path = self.specification([self.evaluated_job()])
        elapsed = [0]
        def monotonic():
            elapsed[0]+=31
            return elapsed[0]
        def process(command,**kwargs):
            result = self.successful_process(command,**kwargs)
            if Path(command[2]).name=="eval_research_checkpoint.py":
                polls = iter([None,0])
                result.poll = lambda:next(polls,0)
            return result
        events = []
        with patch.object(queue_module.subprocess,"Popen",side_effect=process), \
             patch.object(queue_module,"time",SimpleNamespace(monotonic=monotonic,sleep=lambda _:None)), \
             patch.object(queue_module,"emit",side_effect=lambda event,**_:events.append(event)):
            queue = queue_module.ResearchQueue(path)
            with patch.object(queue.budget,"ensure_space",wraps=queue.budget.ensure_space) as guard:
                self.assertEqual(queue.run(),0)
                self.assertGreaterEqual(guard.call_count,3)
        self.assertIn("job_evaluation_running",events)


class TestIntermediateCheckpointPruning(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def managed_run(self, name, stage, source=None):
        directory = self.root / name
        manager = BestLastCheckpointManager(directory, DiskBudget(self.root, 0))
        manager.save(lambda d: (d / "state.pt").write_bytes(b"checkpoint"), step=1, metric=1., anticipated_bytes=100)
        (directory / "resolved_config.json").write_text(json.dumps({"stage": stage, "initial_checkpoint": str(source / "last/state.pt") if source else None}))
        (directory / "completed.json").write_text(json.dumps({"steps": 1}))
        (directory / "metrics.jsonl").write_text("metrics must survive")
        return directory

    def test_prunes_owned_intermediate_and_preserves_metrics_and_consumer(self):
        source = self.managed_run("a1", "A1")
        consumer = self.managed_run("sft", "SFT", source)
        unknown = source / ".checkpoint_objects/user-data"
        unknown.mkdir()
        (unknown / "keep").write_text("unrelated")
        result = queue_module.prune_checkpoint_run(source, [consumer])
        self.assertEqual(result["status"], "pruned")
        self.assertFalse((source / "last").exists())
        self.assertFalse((source / "best").exists())
        self.assertEqual((source / "metrics.jsonl").read_text(), "metrics must survive")
        self.assertTrue((source / "completed.json").exists())
        self.assertTrue((consumer / "last/state.pt").is_file())
        self.assertEqual((unknown / "keep").read_text(), "unrelated")
        # Idempotent standalone call and queue restart accept the owned receipt.
        self.assertEqual(queue_module.prune_checkpoint_run(source, [consumer])["status"], "pruned")
        artifact = queue_module.artifacts_for({"id": "a1", "argv": ["--output", str(source)]}, allow_pruned=True)
        self.assertTrue(artifact["checkpoint_pruned"])
        self.assertIsNone(artifact["last"])

    def test_queue_restart_accepts_guarded_external_pruning_receipt_without_spec_change(self):
        source = self.managed_run("a1", "A1")
        consumer = self.managed_run("sft", "SFT", source)
        specification = self.root / "external-prune-queue.json"
        specification.write_text(json.dumps({"reserve_gb": 0, "jobs": [
            {"id": "a1", "argv": ["--stage", "A1", "--output", str(source)]},
            {"id": "sft", "argv": ["--stage", "SFT", "--output", str(consumer)], "requires": ["a1"]}]}))
        queue = queue_module.ResearchQueue(specification)
        queue.state["jobs"] = {"a1": {"status": "completed"}, "sft": {"status": "completed"}}
        queue.persist()
        queue_module.prune_checkpoint_run(source, [consumer])
        with patch.object(queue_module.subprocess, "Popen") as popen:
            self.assertEqual(queue.run(), 0)
            popen.assert_not_called()
        self.assertTrue(queue.state["jobs"]["a1"]["checkpoint_pruned"])

    def test_final_sft_and_unrelated_consumers_cannot_be_pruned(self):
        source = self.managed_run("source", "SFT")
        consumer = self.managed_run("consumer", "SFT", source)
        with self.assertRaisesRegex(ValueError, "final SFT"):
            queue_module.prune_checkpoint_run(source, [consumer])
        source = self.managed_run("a1", "A1")
        unrelated = self.managed_run("unrelated", "SFT")
        with self.assertRaisesRegex(ValueError, "does not reference"):
            queue_module.prune_checkpoint_run(source, [unrelated])
        self.assertTrue((source / "last/state.pt").is_file())

    def test_consumer_must_have_a_valid_retained_checkpoint(self):
        source = self.managed_run("a1", "A1")
        consumer = self.managed_run("sft", "SFT", source)
        (consumer / "last").unlink()
        with self.assertRaisesRegex(ValueError, "retained last"):
            queue_module.prune_checkpoint_run(source, [consumer])
        self.assertTrue((source / "last/state.pt").is_file())

    def test_final_sft_discard_flag_rejected_in_spec(self):
        with self.assertRaisesRegex(ValueError, "final SFT"):
            queue_module.validate_spec({"jobs": [{"id": "sft", "argv": ["--stage", "SFT", "--output", str(self.root / "sft")], "discard_checkpoint_after_dependents": True}]})


if __name__ == "__main__":
    unittest.main()
