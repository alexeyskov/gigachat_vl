"""CPU-only profile wiring; actual CUDA support is a separate root-owned proof."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from src.utils import research_stock_determinism as stock


@pytest.fixture(autouse=True)
def restore_torch_and_environment():
    old = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cuda.matmul.allow_tf32, torch.get_num_threads(), os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    yield
    torch.use_deterministic_algorithms(old[0], warn_only=old[1])
    torch.backends.cuda.matmul.allow_tf32 = old[2]; torch.set_num_threads(old[3])
    if old[4] is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = old[4]


def config(**extra):
    return dict(model=stock.SMALL, numerical_profile=stock.definition(), stage="SFT", recipe="3b", **extra)


def test_three_metadata_facades_do_not_import_torch_or_create_outputs():
    for name in ("run", "eval", "validate"):
        code = ("import runpy,sys;sys.argv=['x','--metadata'];"
                f"runpy.run_path('scripts/{name}_research_stock_deterministic.py',run_name='__main__');"
                "assert 'torch' not in sys.modules")
        result = subprocess.run([sys.executable,"-c",code], capture_output=True,text=True,check=True)
        assert json.loads(result.stdout)["status"] == "future_profile_metadata_only"


def test_explicit_environment_conflict_refused_before_torch():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    with pytest.raises(ValueError,match="Conflicting"):
        stock.bootstrap_environment()


def test_profile_binds_frozen_and_facade_sources_and_rejects_mutation(monkeypatch):
    value = stock.definition()
    assert value["deterministic_algorithms"] is True and value["warn_only"] is False
    assert set(stock.OWN) <= value["source_sha256"].keys()
    assert value["runtime_versions"]["__version__"]==str(torch.__version__)
    assert value["runtime_versions"]["cuda"]==torch.version.cuda
    pins = dict(stock.PINS);pins["src/model/smallthinker.py"] = "0"*64
    monkeypatch.setattr(stock,"PINS",pins)
    with pytest.raises(ValueError,match="Frozen"):
        stock.definition()


@pytest.mark.parametrize("value", [{}, {"name":"legacy"}])
def test_legacy_or_changed_resume_profile_refused(value):
    module = stock.runner()
    current = config()
    old = dict(current,numerical_profile=value)
    with pytest.raises(ValueError,match="numerical profile"):
        module.validate_resume_config(old,current)
    module.validate_resume_config(current,copy.deepcopy(current))


def test_constructor_checks_profile_before_any_cuda_model_construction():
    stock.activate();module = stock.runner()
    with pytest.raises(ValueError,match="numerical profile"):
        module.ResearchModel(SimpleNamespace(model=stock.SMALL))
    assert not torch.cuda.is_initialized()


def test_legacy_a0_initialization_requires_explicit_stage_and_owner_model():
    stock.activate()
    module = stock.runner()
    model = SimpleNamespace(model_ref={},vision_ref={},cache=SimpleNamespace(fingerprint="f"))
    previous = {"config":{"model":stock.SMALL,"stage":"A0","recipe":"4f"}}
    args = SimpleNamespace(model=stock.SMALL,numerical_profile=stock.definition(),stage="A1",recipe="3a",allow_legacy_a0_initialization=False)
    with pytest.raises(ValueError,match="Legacy initialization"):
        module.validate_initial_checkpoint(previous,model,args)
    args.allow_legacy_a0_initialization=True
    assert module.validate_initial_checkpoint(previous,model,args)
    for key,value in (("stage","A1"),("model","foreign/model")):
        bad = copy.deepcopy(previous);bad["config"][key]=value
        with pytest.raises(ValueError,match="Legacy initialization"):
            module.validate_initial_checkpoint(bad,model,args)
    args.stage="SFT"
    with pytest.raises(ValueError,match="Legacy initialization"):
        module.validate_initial_checkpoint(previous,model,args)


def test_private_inference_reuses_native_generation_without_canonical_runner_patch():
    import src.eval.research_inference as native
    canonical = sys.modules.get("scripts.run_research_stage")
    private = stock.inference()
    assert private.generate_prediction.__code__.co_code == native.generate_prediction.__code__.co_code
    assert private.build_generation_inputs.__code__.co_code == native.build_generation_inputs.__code__.co_code
    assert sys.modules.get("scripts.run_research_stage") is canonical


def test_new_loader_refuses_legacy_before_model_construction(tmp_path):
    path=tmp_path/"legacy.pt"
    torch.save({"config":{"model":stock.SMALL},"model":{}},path)
    with pytest.raises(ValueError,match="numerical profile"):
        stock.inference().load_research_checkpoint(path)
    assert not torch.cuda.is_initialized()


def test_eval_protocol_binding_and_imports_are_isolated():
    private = stock.inference()
    cli = stock._clone("scripts/eval_research_checkpoint.py",{"src.eval.research_inference":private},bind_eval_protocol=True)
    assert cli.load_research_checkpoint is private.load_research_checkpoint
    old = {"generation":{"max_new_tokens":64}}
    result=cli._bind_stock_protocol(old)
    assert result["numerical_profile"]==stock.definition()
    assert "numerical_profile" not in old


def test_own_source_sidecar_is_exact_idempotent_and_mutation_refuses(tmp_path):
    import zipfile
    first=stock.execution_sources(tmp_path,create=True,reserve_gb=0)
    assert first==stock.execution_sources(tmp_path,create=True,reserve_gb=0)
    with zipfile.ZipFile(tmp_path/"stock_execution_v1/source.zip") as archive:
        assert set(archive.namelist())==set(stock.OWN)
        assert all(archive.read(name)==(stock.ROOT/name).read_bytes() for name in stock.OWN)
    (tmp_path/"resolved_config.json").write_text(json.dumps(config()))
    assert stock.validate_run_context(tmp_path)["model"]==stock.SMALL
    (tmp_path/"stock_execution_v1/source.zip").write_bytes(b"changed")
    with pytest.raises(ValueError,match="sidecar changed"):
        stock.validate_run_context(tmp_path)


def test_profile_source_sidecar_is_required_before_evaluation(tmp_path):
    (tmp_path/"resolved_config.json").write_text(json.dumps(config()))
    with pytest.raises(FileNotFoundError,match="sidecar"):
        stock.validate_run_context(tmp_path)


def test_private_validator_redirects_all_runner_imports():
    private=stock.runner()
    validator=stock._clone("scripts/validate_research_checkpoint.py",{"scripts.run_research_stage":private})
    names=set()
    import ast
    tree=ast.parse((stock.ROOT/"scripts/validate_research_checkpoint.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module=="scripts.run_research_stage":
            names.update(alias.name for alias in node.names)
    assert names=={"ResearchModel","load_records","validate_model_binding","resolve_model","prepare_group_macro_selection","validate_resume_config"}
    assert all(getattr(private,name) is not None for name in names)
    assert validator._stock_dependency_0 is private


def test_actual_training_argument_parser_and_new_run_archive_without_constructor(tmp_path,monkeypatch):
    module=stock.runner()
    def metadata_only_main():
        args=module.parse_args();module.validate_args(args)
        assert args.stage=="A1" and args.recipe=="3a" and args.allow_legacy_a0_initialization
        stock.validate_config(vars(args));stock.assert_active()
        return vars(args)
    module.main=metadata_only_main
    monkeypatch.setattr(stock,"runner",lambda:module)
    result=stock.train_main(["--numerical-profile",stock.PROFILE,"--allow-gpu","--allow-legacy-a0-initialization",
        "--stage","A1","--recipe","3a","--manifest","train.jsonl","--validation","val.jsonl",
        "--initial-checkpoint","legacy_a0.pt","--output",str(tmp_path),"--reserve-gb","0"])
    assert result["numerical_profile"]==stock.definition()
    assert (tmp_path/"stock_execution_v1/source.zip").is_file()
    assert not torch.cuda.is_initialized()


def test_drift_and_implicit_execution_refused():
    stock.activate();stock.assert_active()
    torch.use_deterministic_algorithms(False)
    with pytest.raises(RuntimeError,match="drifted"):
        stock.assert_active()
    with pytest.raises(ValueError,match="Explicit"):
        stock.train_main([])
    with pytest.raises(ValueError,match="allow-gpu"):
        stock.train_main(["--numerical-profile",stock.PROFILE])
    with pytest.raises(ValueError,match="allow-gpu"):
        stock.validate_main(["--numerical-profile",stock.PROFILE,"--run","cpu"])


def test_actual_tiny_native_stock_cpu_forward_and_gradients_repeat():
    from src.model.smallthinker import SmallThinkerConfig,SmallThinkerMoeBlock
    stock.activate()
    cfg=SmallThinkerConfig(vocab_size=32,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,
        num_key_value_heads=1,head_dim=4,moe_ffn_hidden_size=12,moe_num_primary_experts=6,moe_num_active_primary_experts=4)
    torch.manual_seed(42)
    model=SmallThinkerMoeBlock(cfg)
    values=torch.randn(1,9,8)
    outputs=[]
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        x=values.clone().requires_grad_(True)
        y=model(x,x)[0];y.square().sum().backward()
        outputs.append([y.detach(),x.grad,*[p.grad for p in model.parameters() if p.grad is not None]])
    assert all(torch.equal(a,b) for row in outputs[1:] for a,b in zip(outputs[0],row))
    assert not torch.cuda.is_initialized()
