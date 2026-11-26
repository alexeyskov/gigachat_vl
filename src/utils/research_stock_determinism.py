"""Opt-in, versioned stock Torch numerical profile; frozen math stays unchanged.

Private copies redirect only runner/inference imports and add provenance hooks.
No canonical modules, kernels, checkpoint values or historical scores are patched.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import types
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "stock_deterministic_small_v1"
WORKSPACE = ":4096:8"
SMALL = "Tiiny/SmallThinker-4BA0.6B-Instruct"
PINS = {
    "scripts/run_research_stage.py": "f7fd22d1c94fb337af2be60d3324415b2b2c8cd3d62e9799f2da4c05ace87cfc",
    "scripts/eval_research_checkpoint.py": "cf06fcf7ba5c2be4860d58781478de7b51a101c21a4bc8cd95ebb333220baff2",
    "scripts/validate_research_checkpoint.py": "5fb59747dad1badb376e5cc95aa4f9551e35556a24067d1bf182c89140ee9626",
    "src/eval/research_inference.py": "47576ba901c5e6ba3dc95b8c8777872d36c3ea73f2a4e0c6d74a0955df75744d",
    "src/eval/research_selection.py": "91c05924df717ce46ef22787dfa5b4b0e45e35533096769373be62ceda3cade2",
    "src/model/smallthinker.py": "7d5aaa551b98e7958f1a69930c797f7906d7c770084b156ba36cc5d577cbdd26",
    "src/model/research_moe.py": "f8cbfd6a8fdb7044d25d031f0a028055cb114a7ebd71a9b4f873dcd752674c6c",
}
OWN = ("src/utils/research_stock_determinism.py", "scripts/run_research_stock_deterministic.py",
       "scripts/eval_research_stock_deterministic.py", "scripts/validate_research_stock_deterministic.py")


def bootstrap_environment():
    """Call before importing Torch in every new facade."""
    old = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if old not in (None, WORKSPACE):
        raise ValueError("Conflicting CUBLAS_WORKSPACE_CONFIG for explicit stock profile")
    torch = sys.modules.get("torch")
    if old is None and torch is not None and torch.cuda.is_initialized():
        raise RuntimeError("Cannot establish cuBLAS environment after CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = WORKSPACE


def definition():
    hashes = {}
    for name in (*PINS, *OWN):
        digest = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if name in PINS and digest != PINS[name]:
            raise ValueError("Frozen stock-profile source changed: " + name)
        hashes[name] = digest
    packages = {}
    for package in ("torch","transformers","peft","bitsandbytes","tokenizers","huggingface-hub"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    version_path = Path(importlib.metadata.distribution("torch").locate_file("torch/version.py"))
    versions = {}
    for node in ast.parse(version_path.read_bytes()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id in ("__version__","cuda","git_version","hip"):
            versions[node.target.id] = ast.literal_eval(node.value)
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ("__version__","cuda","git_version","hip"):
                    versions[target.id] = ast.literal_eval(node.value)
    return {"name": PROFILE, "schema_version": 1, "model": SMALL,
            "deterministic_algorithms": True, "warn_only": False,
            "cublas_workspace_config": WORKSPACE, "matmul_allow_tf32": True,
            "torch_threads": 8, "custom_kernels": False, "source_sha256": hashes,
            "runtime_versions":versions, "package_versions":packages,
            "torch_version_source_sha256":hashlib.sha256(version_path.read_bytes()).hexdigest()}


def execution_sources(run, *, create=False, reserve_gb=120):
    """Small immutable own-source sidecar; native provenance additionally archives all Python."""
    run = Path(run)
    profile = definition()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(OWN):
            raw = (ROOT/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != profile["source_sha256"][name]:
                raise ValueError("Facade source changed while archiving")
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.external_attr=0o100644<<16
            archive.writestr(info,raw)
    raw_zip=buffer.getvalue()
    receipt={"schema_version":1,"numerical_profile":profile,
             "source_zip_sha256":hashlib.sha256(raw_zip).hexdigest(),"source_zip_bytes":len(raw_zip),
             "scope":"Four new execution-profile source files; full filesystem code captured by native write_run_provenance"}
    directory=run/"stock_execution_v1"
    if directory.exists():
        if (json.loads((directory/"receipt.json").read_text())!=receipt or
            (directory/"source.zip").read_bytes()!=raw_zip):
            raise ValueError("Stock execution source sidecar changed")
        return receipt
    if not create:
        raise FileNotFoundError("Future run lacks its mandatory stock execution source sidecar")
    import shutil
    run.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(run).free < int(reserve_gb*2**30)+len(raw_zip)+2**20:
        raise OSError("Stock source sidecar would violate disk reserve")
    staging=run/(".stock_execution_v1."+uuid.uuid4().hex+".tmp")
    staging.mkdir()
    for name,data in (("source.zip",raw_zip),("receipt.json",(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())):
        with (staging/name).open("xb") as handle:
            handle.write(data);handle.flush();os.fsync(handle.fileno())
    fd=os.open(staging,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    os.rename(staging,directory)
    fd=os.open(run,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return receipt


def validate_run_context(run):
    config=json.loads((Path(run)/"resolved_config.json").read_text())
    validate_config(config)
    execution_sources(run)
    return config


def validate_config(config):
    if config.get("model") != SMALL:
        raise ValueError("Stock deterministic execution profile is verified for SmallThinker only")
    if config.get("numerical_profile") != definition():
        raise ValueError("Missing, legacy or changed numerical profile; implicit mixed resume is refused")


def activate():
    bootstrap_environment()
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(8)
    assert_active()


def assert_active():
    import torch
    if (os.environ.get("CUBLAS_WORKSPACE_CONFIG") != WORKSPACE or
        not torch.are_deterministic_algorithms_enabled() or
        torch.is_deterministic_algorithms_warn_only_enabled() or
        not torch.backends.cuda.matmul.allow_tf32 or torch.get_num_threads() != 8):
        raise RuntimeError("Stock deterministic execution profile drifted")
    versions=definition()["runtime_versions"]
    if (str(torch.__version__)!=versions["__version__"] or torch.version.cuda!=versions["cuda"] or
        torch.version.git_version!=versions["git_version"]):
        raise RuntimeError("Actual Torch/CUDA runtime differs from recorded profile")


class _Imports(ast.NodeTransformer):
    def __init__(self, dependencies, bind_eval_protocol=False):
        self.dependencies = dependencies
        self.bind_eval_protocol = bind_eval_protocol
        self.protocol_rewrites = 0

    def visit_Import(self, node):
        if len(node.names) == 1 and node.names[0].name in self.dependencies:
            alias = node.names[0]
            if not alias.asname:
                raise ValueError("Expected explicit private runner import alias")
            return ast.copy_location(ast.Assign([ast.Name(alias.asname, ast.Store())],
                ast.Name(self.dependencies[alias.name], ast.Load())), node)
        return node

    def visit_ImportFrom(self, node):
        if node.module in self.dependencies:
            return [ast.copy_location(ast.Assign([ast.Name(item.asname or item.name, ast.Store())],
                ast.Attribute(ast.Name(self.dependencies[node.module], ast.Load()), item.name, ast.Load())), node)
                for item in node.names]
        return node

    def visit_Assign(self, node):
        if self.bind_eval_protocol and any(isinstance(t, ast.Name) and t.id == "protocol" for t in node.targets) and isinstance(node.value, ast.Dict):
            node.value = ast.Call(ast.Name("_bind_stock_protocol", ast.Load()), [node.value], [])
            self.protocol_rewrites += 1
        return self.generic_visit(node)


def _clone(relative, dependencies=None, bind_eval_protocol=False):
    definition()  # Check exact source bytes before executing a private copy.
    path = ROOT/relative
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != PINS[relative]:
        raise ValueError("Native source changed during private loading")
    module = types.ModuleType("_research_stock_private_" + uuid.uuid4().hex)
    module.__file__ = str(path)
    globals_map = {"_bind_stock_protocol": lambda value: dict(value, numerical_profile=definition())}
    mapped = {}
    for i, (original, dependency) in enumerate((dependencies or {}).items()):
        key = "_stock_dependency_"+str(i)
        mapped[original] = key; globals_map[key] = dependency
    module.__dict__.update(globals_map)
    transformer = _Imports(mapped, bind_eval_protocol)
    tree = transformer.visit(ast.parse(source, filename=str(path)))
    if bind_eval_protocol and transformer.protocol_rewrites != 1:
        raise ValueError("Native evaluator protocol hook structure changed")
    ast.fix_missing_locations(tree)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def runner():
    bootstrap_environment()
    module = _clone("scripts/run_research_stage.py")
    original_model = module.ResearchModel
    class ProfileResearchModel(original_model):
        def __init__(self, args):
            validate_config(vars(args)); assert_active()
            super().__init__(args)
            assert_active()
    module.ResearchModel = ProfileResearchModel
    native_resume = module.validate_resume_config
    def resume(saved, current):
        validate_config(saved); validate_config(current)
        return native_resume(saved, current)
    module.validate_resume_config = resume
    native_initial = module.validate_initial_checkpoint
    def initial(payload, model, args):
        validate_config(vars(args)); assert_active()
        previous = payload.get("config", {})
        if previous.get("numerical_profile") is None:
            if not (getattr(args,"allow_legacy_a0_initialization",False) and previous.get("model")==SMALL and previous.get("stage")=="A0" and args.stage=="A1"):
                raise ValueError("Legacy initialization is allowed only for explicit A0 -> A1, never resume")
        else:
            validate_config(previous)
        return native_initial(payload, model, args)
    module.validate_initial_checkpoint = initial
    return module


def inference(private_runner=None):
    private_runner = private_runner or runner()
    module = _clone("src/eval/research_inference.py", {"scripts.run_research_stage":private_runner})
    native = module.load_research_checkpoint
    def load(checkpoint, **kwargs):
        from src.utils.research_checkpoint_compat import load_research_payload
        saved = load_research_payload(Path(checkpoint), map_location="cpu", mmap=True)
        validate_config(saved["config"]); del saved
        activate()
        model, provenance = native(checkpoint, **kwargs)
        assert_active()
        return model, dict(provenance, numerical_profile=definition())
    module.load_research_checkpoint = load
    return module


def train_main(argv=None):
    opts, remaining = _options(argv, legacy_initialization=True)
    if opts.metadata:
        return _print_metadata()
    if not opts.allow_gpu:
        raise ValueError("Future training requires explicit --allow-gpu")
    activate()
    module = runner(); native_parse = module.parse_args
    def parse():
        previous = sys.argv
        try:
            sys.argv = [previous[0], *remaining]
            args = native_parse()
        finally:
            sys.argv = previous
        args.numerical_profile = definition()
        args.allow_legacy_a0_initialization = opts.allow_legacy_a0_initialization
        if args.model != SMALL:
            raise ValueError("Stock deterministic profile is SmallThinker only")
        path=Path(args.output)/"resolved_config.json"
        if path.exists():validate_config(json.loads(path.read_text()))
        execution_sources(args.output,create=True,reserve_gb=args.reserve_gb)
        return args
    module.parse_args = parse
    return module.main()


def eval_main(argv=None):
    opts, remaining = _options(argv)
    if opts.metadata:
        return _print_metadata()
    if not opts.allow_gpu:
        raise ValueError("Future generation requires explicit --allow-gpu")
    activate()
    module = _clone("scripts/eval_research_checkpoint.py", {"src.eval.research_inference":inference()}, bind_eval_protocol=True)
    args=module.parse_args(remaining)
    validate_run_context(args.run)
    from src.utils.research_checkpoint_compat import load_research_payload
    payload=load_research_payload(Path(args.run)/args.checkpoint/"state.pt",map_location="cpu",mmap=True)
    validate_config(payload["config"]);del payload
    return module.main(remaining)


def validate_main(argv=None):
    opts, remaining = _options(argv)
    if opts.metadata:
        return _print_metadata()
    device_parser = argparse.ArgumentParser(add_help=False)
    device_parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    device_parser.add_argument("--cpu-structural-only",action="store_true")
    device_options, _ = device_parser.parse_known_args(remaining)
    if not opts.allow_gpu and not device_options.cpu_structural_only and device_options.device != "cpu":
        raise ValueError("Future GPU replay requires explicit --allow-gpu")
    activate()
    private_runner = runner()
    module = _clone("scripts/validate_research_checkpoint.py", {"scripts.run_research_stage":private_runner})
    native = module.validate_run
    def validate(run, **kwargs):
        validate_run_context(run); assert_active()
        return native(run, **kwargs)
    module.validate_run = validate
    native_atomic = module.atomic_json
    def atomic(path, payload):
        payload["numerical_profile"] = definition()
        return native_atomic(path, payload)
    module.atomic_json = atomic
    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        return module.main()
    finally:
        sys.argv = previous


def _options(argv, legacy_initialization=False):
    bootstrap_environment()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--numerical-profile", choices=[PROFILE])
    parser.add_argument("--allow-gpu", action="store_true")
    if legacy_initialization:
        parser.add_argument("--allow-legacy-a0-initialization", action="store_true")
    opts, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not opts.metadata and opts.numerical_profile != PROFILE:
        raise ValueError("Explicit --numerical-profile " + PROFILE + " is required")
    return opts, remaining


def _print_metadata():
    print(json.dumps({"status":"future_profile_metadata_only", "gpu_used":False,
                      "numerical_profile":definition()}, indent=2))
    return 0
