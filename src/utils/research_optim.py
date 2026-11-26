"""Explicit memory profiles for full-parameter research training on one GPU.

The streamed backend keeps FP32 model parameters on CUDA, moves accumulated
gradients to CPU after each leaf accumulation, and stages one parameter gradient
at a time for a native bitsandbytes paged AdamW update. It changes optimizer-state
precision when ``optim_bits=8``; callers must record that profile rather than
present it as an identical FP32-AdamW experiment.
"""

from __future__ import annotations

from collections import defaultdict
import ctypes
import math
from typing import Iterable

import torch


GIB = 2**30
_CUDA_DRIVER = None


def _prefetch_paged_to_cpu(tensor):
    """Bypass bnb's invalid cudaDeviceGetAttribute(-1) CPU-prefetch path.

    CUDA's driver retains this legacy entrypoint with a CUdevice destination,
    including CU_DEVICE_CPU=-1. Unlike bnb's C++ wrapper it returns an error code
    instead of aborting the process. The caller synchronizes after enqueueing.
    """
    global _CUDA_DRIVER
    if _CUDA_DRIVER is None:
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuMemPrefetchAsync.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        driver.cuMemPrefetchAsync.restype = ctypes.c_int
        driver.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        driver.cuGetErrorString.restype = ctypes.c_int
        _CUDA_DRIVER = driver
    code = _CUDA_DRIVER.cuMemPrefetchAsync(tensor.data_ptr(), tensor.numel()*tensor.element_size(), -1, None)
    if code:
        message = ctypes.c_char_p()
        _CUDA_DRIVER.cuGetErrorString(code, ctypes.byref(message))
        detail = message.value.decode() if message.value else "unknown CUDA driver error"
        raise RuntimeError(f"CPU prefetch of paged optimizer state failed: {code} ({detail})")


def plan_full_training_memory(parameters: Iterable[torch.Tensor], *,
                              gpu_available_bytes: int,
                              cpu_available_bytes: int,
                              activation_reserve_gib: float = 8,
                              cuda_overhead_gib: float = 2,
                              paged_working_set_gib: float = 2):
    """Return estimates and a proposed profile; never change a running optimizer.

    All estimates assume FP32 trainable master weights and include every supplied
    trainable parameter once. Paged states need host memory and supported CUDA UVM.
    Predictions do not replace an actual full-model CUDA memory smoke test.
    """
    unique = {id(p): p for p in parameters if p.requires_grad}
    count = sum(p.numel() for p in unique.values())
    largest = max((p.numel() for p in unique.values()), default=0)
    weights = 4*count
    margin = int((activation_reserve_gib+cuda_overhead_gib)*GIB)
    working = int(paged_working_set_gib*GIB)
    candidates = [
        {"backend": "torch_adamw32", "gpu_bytes": 16*count+margin,
         "host_bytes": 0, "optimizer_state_bytes": 8*count,
         "optimizer_state_bits": 32},
        {"backend": "bnb_paged_adamw32", "gpu_bytes": 8*count+margin+working,
         "host_bytes": 8*count, "optimizer_state_bytes": 8*count,
         "optimizer_state_bits": 32},
        {"backend": "cpu_grad_bnb_paged_adamw8", "gpu_bytes": weights+4*largest+margin+working,
         "host_bytes": 6*count, "optimizer_state_bytes": 2*count,
         "optimizer_state_bits": 8},
    ]
    for row in candidates:
        row["fits_estimate"] = row["gpu_bytes"] <= gpu_available_bytes and row["host_bytes"] <= cpu_available_bytes
        row["checkpoint_single_bytes"] = weights+row["optimizer_state_bytes"]
        row["best_last_plus_atomic_bytes"] = 3*row["checkpoint_single_bytes"]
    selected = next((row["backend"] for row in candidates if row["fits_estimate"]), None)
    return {
        "trainable_parameters": count, "largest_parameter_elements": largest,
        "fp32_weight_bytes": weights, "fp32_gradient_bytes": weights,
        "gpu_available_bytes": int(gpu_available_bytes),
        "cpu_available_bytes": int(cpu_available_bytes),
        "proposed_backend": selected, "candidates": candidates,
        "estimate_only": True,
        "caveats": [
            "Paged CUDA states require UVM support and consume host RAM.",
            "Transient backward gradients, checkpoint recomputation and casts may raise the actual peak.",
            "CPU gradient transfers reduce throughput and do not freeze any parameter.",
            "8-bit moment estimates exclude small FP32 fallback states and quantization metadata.",
        ],
    }


def make_paged_adamw32(params, **kwargs):
    import bitsandbytes as bnb
    return bnb.optim.PagedAdamW32bit(params, **kwargs)


class CPUGradientOffloadOptimizer(torch.optim.Optimizer):
    """Synchronous CPU gradient accumulation + streamed native paged AdamW.

    Normalize each microbatch loss by the *whole optimizer-step* supervised-token
    denominator exactly as usual. This class never rescales the loss. Call its
    ``clip_grad_norm_`` method, since ``parameter.grad`` is intentionally cleared
    after every backward accumulation. Update the scheduler once per complete
    ``step()``. No optimizer update occurs during backward.

    Native bnb's state initialization/update entrypoints are used with original
    group/index values. Visiting each parameter once avoids calling the complete
    native optimizer loop O(number_of_parameters) times.
    """

    STATE_VERSION = 1

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, *, optim_bits=8, min_8bit_size=4096,
                 pin_memory=False, evict_paged_states=True,
                 error_if_nonfinite=True):
        import bitsandbytes as bnb
        if optim_bits not in (8, 32):
            raise ValueError("optim_bits must be 8 or 32.")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        cls = bnb.optim.PagedAdamW8bit if optim_bits == 8 else bnb.optim.PagedAdamW32bit
        self.inner = cls(self.param_groups, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, min_8bit_size=min_8bit_size)
        required = ("check_overrides", "to_gpu", "init_state", "prefetch_state", "update_step", "get_state_buffer")
        if not all(callable(getattr(self.inner, key, None)) for key in required):
            raise RuntimeError("Installed bitsandbytes does not expose the validated streamed-update API.")
        self.param_groups = self.inner.param_groups
        self.defaults = self.inner.defaults
        self.state = self.inner.state
        self.optim_bits = optim_bits
        self.pin_memory = bool(pin_memory)
        self.evict_paged_states = bool(evict_paged_states)
        self.error_if_nonfinite = bool(error_if_nonfinite)
        self.optimizer_steps = 0
        self._generation = 0
        self._validated_generation = -1
        self._cpu_grads = {}
        self._handles = []
        self._failed_step = False
        self._closed = False
        seen = set()
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) in seen:
                    raise ValueError("A parameter occurs more than once in the optimizer.")
                seen.add(id(p))
                if p.device.type != "cuda" or p.dtype != torch.float32:
                    raise ValueError("Streamed full training requires CUDA FP32 master parameters.")
                if p.grad is not None:
                    raise ValueError("Install gradient offload before the first backward, with cleared gradients.")
        # Validation above is complete before attaching any hooks.
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self._handles.append(p.register_post_accumulate_grad_hook(self._capture_gradient))

    @torch.no_grad()
    def _capture_gradient(self, parameter):
        if self._closed:
            raise RuntimeError("Gradient-offload optimizer has been closed.")
        gradient = parameter.grad
        if gradient is None:
            return
        if gradient.is_sparse:
            raise NotImplementedError("Sparse gradient offload is not supported.")
        cpu = self._cpu_grads.get(parameter)
        if cpu is None:
            cpu = torch.empty(parameter.shape, device="cpu", dtype=torch.float32,
                              pin_memory=self.pin_memory)
            cpu.copy_(gradient.detach(), non_blocking=False)
            self._cpu_grads[parameter] = cpu
        else:
            cpu.add_(gradient.detach().to(device="cpu", dtype=torch.float32, non_blocking=False))
        parameter.grad = None
        self._generation += 1

    @property
    def pending_gradient_bytes(self):
        return sum(g.numel()*g.element_size() for g in self._cpu_grads.values())

    @property
    def pending_parameter_count(self):
        return len(self._cpu_grads)

    @torch.no_grad()
    def zero_grad(self, set_to_none=True):
        if set_to_none:
            self._cpu_grads.clear()
        else:
            for gradient in self._cpu_grads.values():
                gradient.zero_()
        self.inner.zero_grad(set_to_none=True)
        self._generation += 1

    @torch.no_grad()
    def scale_gradients_(self, factor):
        if not math.isfinite(float(factor)):
            raise ValueError("Gradient scale must be finite.")
        for gradient in self._cpu_grads.values():
            gradient.mul_(factor)
        self._generation += 1
        return self

    @torch.no_grad()
    def grad_norm(self, norm_type=2.0):
        norm_type = float(norm_type)
        if norm_type != math.inf and norm_type <= 0:
            raise ValueError("norm_type must be positive.")
        values = []
        for gradient in self._cpu_grads.values():
            if norm_type == math.inf:
                values.append(float(gradient.abs().max()))
            else:
                value = torch.linalg.vector_norm(gradient.reshape(-1), ord=norm_type, dtype=torch.float64)
                values.append(float(value)**norm_type)
        total = max(values, default=0.0) if norm_type == math.inf else sum(values)**(1/norm_type)
        if self.error_if_nonfinite and not math.isfinite(total):
            raise FloatingPointError("Nonfinite CPU-accumulated gradient norm; no parameters were updated.")
        self._validated_generation = self._generation
        return torch.tensor(total, dtype=torch.float64)

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm, norm_type=2.0):
        if not math.isfinite(float(max_norm)) or max_norm < 0:
            raise ValueError("max_norm must be finite and nonnegative.")
        total = self.grad_norm(norm_type)
        coefficient = min(1.0, float(max_norm)/(float(total)+1e-6))
        if coefficient < 1:
            self.scale_gradients_(coefficient)
            self._validated_generation = self._generation
        return total

    def _evict_state(self, parameter):
        if self.evict_paged_states:
            for key in ("state1", "state2"):
                value = self.state[parameter].get(key)
                if value is not None and getattr(value, "is_paged", False):
                    _prefetch_paged_to_cpu(value)

    @torch.no_grad()
    def step(self, closure=None):
        if self._closed:
            raise RuntimeError("Gradient-offload optimizer has been closed.")
        if self._failed_step:
            raise RuntimeError("A previous streamed update failed part-way; restore a complete checkpoint.")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if any(p.grad is not None for group in self.param_groups for p in group["params"]):
            raise RuntimeError("Found uncaptured CUDA gradients; do not manually assign parameter.grad with this backend.")
        if self.error_if_nonfinite and self._validated_generation != self._generation:
            self.grad_norm()
        if not self.inner.initialized:
            self.inner.check_overrides()
            self.inner.to_gpu()
            self.inner.initialized = True
        touched_devices = set()
        try:
            for gindex, group in enumerate(self.param_groups):
                for pindex, parameter in enumerate(group["params"]):
                    gradient = self._cpu_grads.get(parameter)
                    if gradient is None:
                        continue
                    parameter.grad = gradient.to(parameter.device, dtype=parameter.dtype, non_blocking=False)
                    touched_devices.add(parameter.device)
                    try:
                        if not self.state[parameter]:
                            self.inner.init_state(group, parameter, gindex, pindex)
                        self.inner.prefetch_state(parameter)
                        self.inner.update_step(group, parameter, gindex, pindex)
                        torch.cuda.synchronize(parameter.device)
                        self._evict_state(parameter)
                        torch.cuda.synchronize(parameter.device)
                    finally:
                        parameter.grad = None
                    del self._cpu_grads[parameter]
            for device in touched_devices:
                torch.cuda.synchronize(device)
        except Exception:
            self._failed_step = True
            raise
        self.optimizer_steps += 1
        self._generation += 1
        return loss

    def state_dict(self):
        native = self.inner.state_dict()
        # Recent bnb wraps non-castable tensors even outside FSDP. This backend is
        # single-GPU, so store their ordinary names for streamed reconstruction.
        wrapped_key = getattr(self.inner, "_FSDP_WRAPPED_QUANT_STATE_KEY", "__bnb_optimizer_quant_state__")
        for values in native["state"].values():
            if wrapped_key in values:
                values.update(values.pop(wrapped_key))
        indices = {p: index for group, serialized in zip(self.param_groups, native["param_groups"])
                   for p, index in zip(group["params"], serialized["params"])}
        return {
            "research_optimizer": {
                "version": self.STATE_VERSION, "backend": "cpu_gradient_paged_adamw",
                "optim_bits": self.optim_bits, "optimizer_steps": self.optimizer_steps,
                "evict_paged_states": self.evict_paged_states,
                "parameter_shapes": {indices[p]: list(p.shape) for p in indices},
            },
            "native_optimizer": native,
            # Standard state_dict semantics: references, not a second 84GB clone.
            "cpu_gradients": {indices[p]: gradient.detach() for p, gradient in self._cpu_grads.items()},
        }

    def _release_paged_buffers(self):
        ids = {id(value) for values in self.state.values() for value in values.values()
               if isinstance(value, torch.Tensor) and getattr(value, "is_paged", False)}
        if ids:
            manager = self.inner.page_mng
            manager.paged_tensors[:] = [value for value in manager.paged_tensors if id(value) not in ids]

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """Restore one parameter state at a time, avoiding native full deepcopy.

        Large moment arrays are recreated as paged UVM buffers instead of loading
        every state eagerly into device VRAM. Callers should release the loaded
        checkpoint dictionary afterwards to reclaim its CPU source tensors.
        """
        metadata = state_dict.get("research_optimizer", {})
        if metadata.get("version") != self.STATE_VERSION or metadata.get("optim_bits") != self.optim_bits:
            raise ValueError("Incompatible streamed optimizer checkpoint format/profile.")
        native = state_dict["native_optimizer"]
        old_groups = self.param_groups
        saved_groups = native["param_groups"]
        if len(old_groups) != len(saved_groups) or any(len(a["params"]) != len(b["params"]) for a,b in zip(old_groups, saved_groups)):
            raise ValueError("Optimizer checkpoint parameter groups differ.")
        indices = {index:p for actual, saved in zip(old_groups, saved_groups)
                   for p,index in zip(actual["params"], saved["params"])}
        if any(list(p.shape) != metadata["parameter_shapes"].get(index) for index,p in indices.items()):
            raise ValueError("Optimizer checkpoint parameter shapes differ.")
        self._release_paged_buffers()
        self.inner.state = defaultdict(dict)
        self.state = self.inner.state
        self.inner.param_groups = [dict(saved, params=actual["params"])
                                   for actual,saved in zip(old_groups, saved_groups)]
        self.param_groups = self.inner.param_groups
        for index, values in native["state"].items():
            if index not in indices:
                raise ValueError("Unexpected optimizer state index.")
            parameter = indices[index]
            values = dict(values)
            wrapped_key = getattr(self.inner, "_FSDP_WRAPPED_QUANT_STATE_KEY", "__bnb_optimizer_quant_state__")
            if wrapped_key in values:
                values.update(values.pop(wrapped_key))
            for key,value in values.items():
                if isinstance(value, torch.Tensor):
                    if key in ("state1", "state2"):
                        buffer = self.inner.get_state_buffer(parameter, dtype=value.dtype)
                        buffer.copy_(value, non_blocking=False)
                        self.state[parameter][key] = buffer
                    else:
                        self.state[parameter][key] = value.to(device=parameter.device).clone()
                else:
                    self.state[parameter][key] = value
            self._evict_state(parameter)
            torch.cuda.synchronize(parameter.device)
        self._cpu_grads.clear()
        for index,gradient in state_dict.get("cpu_gradients", {}).items():
            if index not in indices or tuple(gradient.shape) != tuple(indices[index].shape):
                raise ValueError("Pending CPU gradient shape/index mismatch.")
            target = torch.empty(gradient.shape, dtype=torch.float32, device="cpu", pin_memory=self.pin_memory)
            target.copy_(gradient)
            self._cpu_grads[indices[index]] = target
        self.optimizer_steps = int(metadata["optimizer_steps"])
        self.inner.initialized = False
        self._failed_step = False
        self._generation += 1
        self._validated_generation = -1

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._cpu_grads.clear()
        self._closed = True


def make_full_training_optimizer(params, *, backend, **kwargs):
    """Explicit factory: choice is logged by the caller, never inferred silently."""
    if backend == "torch_adamw32":
        return torch.optim.AdamW(params, **kwargs)
    if backend == "bnb_paged_adamw32":
        return make_paged_adamw32(params, **kwargs)
    if backend == "cpu_grad_bnb_paged_adamw8":
        return CPUGradientOffloadOptimizer(params, optim_bits=8, **kwargs)
    if backend == "cpu_grad_bnb_paged_adamw32":
        return CPUGradientOffloadOptimizer(params, optim_bits=32, **kwargs)
    raise ValueError(f"Unknown full-training optimizer backend: {backend}")
