import argparse
import json
import platform
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


REQUIRED_FILES = (
    "vlm_meta.json",
    "projector.pt",
)

OPTIONAL_FILES = (
    "vl_experts.pt",
    "vision_encoder.pt",
    "vision_projector.pt",
)

OPTIONAL_DIRS = (
    "tokenizer",
    "vision_processor",
    "vision_lora",
    "vision_llm_lora",
    "connector_llm_lora",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize a trained GigaChatVL full-LLM checkpoint to bitsandbytes int4. "
            "The input checkpoint must contain the VLM artifacts saved by "
            "SaveVLMArtifactsCallback: llm/, projector.pt, vlm_meta.json, etc."
        )
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Path to a checkpoint-* directory containing llm/ and VLM sidecar files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Where to write the quantized inference checkpoint.",
    )
    parser.add_argument(
        "--quant-type",
        choices=("nf4", "fp4"),
        default="nf4",
        help="bitsandbytes 4-bit quantization type.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Compute dtype stored in the quantization config.",
    )
    parser.add_argument(
        "--no-double-quant",
        action="store_true",
        help="Disable nested/double quantization.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "Device used while quantizing. On Apple Silicon, bitsandbytes should be "
            "attempted through its CPU backend, not MPS."
        ),
    )
    parser.add_argument(
        "--max-shard-size",
        default="4GB",
        help="Shard size passed to save_pretrained for the quantized LLM.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def check_input_checkpoint(checkpoint_dir: Path) -> Dict[str, Any]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"checkpoint_dir must be a directory: {checkpoint_dir}")

    llm_dir = checkpoint_dir / "llm"
    if not llm_dir.is_dir():
        raise FileNotFoundError(
            "Missing `llm/` inside checkpoint_dir. This script expects a checkpoint "
            "that already contains artifacts from SaveVLMArtifactsCallback. "
            "A raw Trainer checkpoint with only optimizer/model state is not enough "
            "for this converter."
        )

    for filename in REQUIRED_FILES:
        path = checkpoint_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required VLM sidecar file: {path}")

    meta_path = checkpoint_dir / "vlm_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("use_llm_lora", False):
        raise ValueError(
            "This converter is intended for full-LLM checkpoints with use_llm_lora=False. "
            "For LoRA checkpoints, keep the base model separate and quantize/load it "
            "with the adapter."
        )

    if bool(meta.get("enable_vl_experts", False)) and not (checkpoint_dir / "vl_experts.pt").exists():
        raise FileNotFoundError(
            "Checkpoint metadata says VL experts are enabled, but `vl_experts.pt` "
            "is missing. Loading such a checkpoint would silently create fresh experts."
        )

    return meta


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output_dir is not empty: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def copy_sidecars(checkpoint_dir: Path, output_dir: Path, meta: Dict[str, Any]) -> None:
    copied = []
    for filename in REQUIRED_FILES + OPTIONAL_FILES:
        src = checkpoint_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
            copied.append(filename)

    for dirname in OPTIONAL_DIRS:
        src = checkpoint_dir / dirname
        if src.exists():
            dst = output_dir / dirname
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied.append(f"{dirname}/")

    meta_out = dict(meta)
    meta_out["llm_quantization"] = {
        "backend": "bitsandbytes",
        "load_in_4bit": True,
        "note": "The quantized LLM is saved in output_dir/llm.",
    }
    (output_dir / "vlm_meta.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Copied VLM sidecars: {', '.join(copied)}")


def resolve_device_map(device: str):
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return "auto"

    if device == "cpu":
        return {"": "cpu"}

    if torch.cuda.is_available():
        return "auto"

    return {"": "cpu"}


def print_backend_warning(device_map: Any) -> None:
    system = platform.system()
    machine = platform.machine()
    bnb_version = package_version("bitsandbytes")
    transformers_version = package_version("transformers")
    print(f"bitsandbytes={bnb_version}, transformers={transformers_version}")

    if device_map == {"": "cpu"}:
        print(
            "Warning: quantizing with bitsandbytes on CPU. This can be slow, and "
            "4-bit CPU/macOS support depends on your installed bitsandbytes build."
        )
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        print(
            "Warning: Apple Silicon does not use the CUDA bitsandbytes path. "
            "If 4-bit loading/saving fails here, run this converter on the NVIDIA "
            "server or use a Mac-native format such as MLX/GGUF for local inference."
        )


def quantize_llm(checkpoint_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    llm_dir = checkpoint_dir / "llm"
    output_llm_dir = output_dir / "llm"
    compute_dtype = dtype_from_name(args.compute_dtype)
    device_map = resolve_device_map(args.device)
    print_backend_warning(device_map)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_use_double_quant=not args.no_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print(f"Loading and quantizing LLM from: {llm_dir}")
    print(f"Device map: {device_map}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(llm_dir),
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load the LLM with bitsandbytes 4-bit quantization. "
            "On Apple Silicon this usually means the installed bitsandbytes build "
            "does not support the required 4-bit CPU/macOS path. Try updating "
            "`transformers`, `accelerate`, and `bitsandbytes`, or run conversion "
            "on the NVIDIA CUDA server."
        ) from e

    print(f"Saving quantized LLM to: {output_llm_dir}")
    try:
        model.save_pretrained(
            str(output_llm_dir),
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
    except Exception as e:
        raise RuntimeError(
            "The model was loaded in 4-bit, but saving the quantized weights failed. "
            "Use recent versions of transformers and bitsandbytes, or run the "
            "conversion on the CUDA server where bitsandbytes 4-bit is fully supported."
        ) from e

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_quantization_info(
    output_dir: Path,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> None:
    payload = {
        "source_checkpoint": str(checkpoint_dir),
        "quantized_llm_dir": "llm",
        "backend": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": args.quant_type,
        "bnb_4bit_use_double_quant": not args.no_double_quant,
        "bnb_4bit_compute_dtype": args.compute_dtype,
        "bitsandbytes_version": package_version("bitsandbytes"),
        "transformers_version": package_version("transformers"),
    }
    (output_dir / "quantization_info.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    meta = check_input_checkpoint(checkpoint_dir)
    output_dir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)

    quantize_llm(checkpoint_dir, output_dir, args)
    copy_sidecars(checkpoint_dir, output_dir, meta)
    write_quantization_info(output_dir, checkpoint_dir, args)

    print("Done.")
    print(f"Quantized inference checkpoint: {output_dir}")
    print(
        "Load it with GigaChatVLForInference(checkpoint_dir=..., "
        "vision_name=/path/to/Qwen2.5-VL-7B-Instruct, use_4bit_llm=False)."
    )


if __name__ == "__main__":
    main()
