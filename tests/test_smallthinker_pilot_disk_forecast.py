"""Phase accounting tests; no torch, CUDA, checkpoints or queue mutations."""
import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/forecast_smallthinker_pilot_disk.py"
SPEC = importlib.util.spec_from_file_location("disk_forecast", PATH)
forecast = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(forecast)


def snapshot(completed=False, a1_existing=0):
    return {"disk": {"free": 100 * forecast.GIB}, "reserve_bytes": 0, "projector_parameters": 10,
            "jobs": [
                {"id": "a1", "stage": "A1", "recipe": "4f", "trainable_parameters": 100,
                 "objects": [{"allocated_bytes": a1_existing}] if a1_existing else [],
                 "pruned": False, "discard_after_dependents": True},
                {"id": "sft", "stage": "SFT", "recipe": "4f", "trainable_parameters": 100,
                 "completed_steps": 170 if completed else None,
                 "checkpoint_steps": [100, 170], "requires": ["a1"]}]}


def test_guard_preserves_runner_multiplier_and_manager_five_percent():
    import math
    s = forecast.checkpoint_sizes(100, 10)
    assert s["model_tensor_bytes"] == 440
    assert s["resumable_bytes"] == 1240
    assert s["save_guard_anticipated_bytes"] == math.ceil((440 * 4 + 128 * forecast.MIB) * 1.05)


def test_distinct_two_resumables_plus_one_compact_not_three_resumables():
    s = snapshot()
    r = forecast.simulate(s, {"4f"})
    saves = [e for e in r["events"] if e["phase"].startswith("SFT_save")]
    assert len(saves) == 2
    compact = [e for e in r["events"] if e["phase"].startswith("compact_unique")]
    assert len(compact) == 2
    # A1+two resumable SFT+one compact SFT; each source freed sequentially.
    assert s["disk"]["free"] - r["minimum_physical_free_bytes"] == 3 * 1240 + 440
    assert r["counterfactual_final_free_bytes"] == s["disk"]["free"] - 2 * 440


def test_shared_pointers_cleanup_old_before_single_compaction():
    s = snapshot()
    r = forecast.simulate(s, set())
    compact = [e for e in r["events"] if e["phase"].startswith("compact_unique")]
    assert len(compact) == 1
    assert r["counterfactual_final_free_bytes"] == s["disk"]["free"] - 440
    phases = [e["phase"] for e in r["events"]]
    assert phases.index("publish_shared_best_last170_cleanup100") < phases.index("compact_unique_object_1_of_1")


def test_existing_a1_not_double_counted_retained_until_consumer_eval_complete():
    s = snapshot(a1_existing=1234)
    r = forecast.simulate(s, set())
    assert all(e["phase"] != "A1_save16" for e in r["events"])
    assert r["counterfactual_final_free_bytes"] == s["disk"]["free"] - 440 + 1234
    assert r["events"][-2]["phase"] == "post_training_evaluation"
    assert r["events"][-1]["phase"] == "consumer_completed_then_prune_A1"


def test_completed_compact_sft_not_rewritten_and_still_releases_a1():
    s = snapshot(completed=True, a1_existing=1234)
    r = forecast.simulate(s, {"4f"})
    assert [e["phase"] for e in r["events"]] == ["post_training_evaluation", "consumer_completed_then_prune_A1"]
    assert r["counterfactual_final_free_bytes"] == s["disk"]["free"] + 1234


def test_first_failure_kept_separate_from_counterfactual_later_peak():
    s = snapshot()
    s["disk"]["free"] = 100
    r = forecast.simulate(s, {"4f"})
    assert r["first_refusal"]["phase"] == "A1_save16"
    assert r["minimum_physical_free_bytes"] < 0
    assert r["required_additional_bytes_to_complete"] > 0


def test_alternative_a1_compaction_retains_model_until_dependent_finishes():
    s = snapshot(a1_existing=1240)
    r = forecast.simulate(s, set(), compact_a1=True)
    assert r["events"][0]["phase"] == "alternative_compact_completed_A1"
    assert r["events"][0]["released_bytes"] == 1240
    assert r["events"][-1]["released_bytes"] == 440
    assert r["counterfactual_final_free_bytes"] == s["disk"]["free"] + 1240 - 440
