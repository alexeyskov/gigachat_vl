"""Memory planning and real CUDA streamed-gradient optimizer correctness."""

import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.utils.research_optim import (
    GIB, CPUGradientOffloadOptimizer, make_full_training_optimizer,
    plan_full_training_memory,
)


def meta_bank(count):
    block = 500_000_000
    return [nn.Parameter(torch.empty(min(block, count-offset), device="meta"))
            for offset in range(0, count, block)]


@pytest.mark.parametrize("count,expected", [
    (4_000_000_000, "torch_adamw32"),
    (8_000_000_000, "bnb_paged_adamw32"),
    (21_000_000_000, "cpu_grad_bnb_paged_adamw8"),
])
def test_memory_plan_includes_all_parameters_and_explicit_profile(count, expected):
    plan = plan_full_training_memory(meta_bank(count), gpu_available_bytes=96*GIB,
                                    cpu_available_bytes=225*GIB)
    assert plan["trainable_parameters"] == count
    assert plan["proposed_backend"] == expected
    assert plan["estimate_only"]
    assert all(row["best_last_plus_atomic_bytes"] == 3*row["checkpoint_single_bytes"] for row in plan["candidates"])


def test_memory_plan_reports_no_fit_and_deduplicates_tied_weights():
    parameters = meta_bank(21_000_000_000)
    plan = plan_full_training_memory(parameters+[parameters[0]], gpu_available_bytes=8*GIB,
                                    cpu_available_bytes=8*GIB)
    assert plan["proposed_backend"] is None
    assert plan["trainable_parameters"] == 21_000_000_000


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA paged optimizers required")


def batches():
    generator = torch.Generator(device="cuda").manual_seed(136)
    return [(torch.randn(3, 128, generator=generator, device="cuda"),
             torch.randn(3, 64, generator=generator, device="cuda")) for _ in range(3)]


def accumulate(weight, inputs, denominator=9):
    for x,y in inputs:
        (F.linear(x, weight)-y).square().sum().div(denominator).backward()


@cuda
@pytest.mark.parametrize("bits", [8, 32])
def test_streamed_accumulation_clipping_and_native_optimizer_parity(bits):
    import bitsandbytes as bnb
    torch.manual_seed(44)
    parameter = nn.Parameter(torch.randn(64, 128, device="cuda")*.03)
    reference = nn.Parameter(parameter.detach().clone())
    streamed = CPUGradientOffloadOptimizer([parameter], lr=.001, weight_decay=.02,
                                          optim_bits=bits, min_8bit_size=4096)
    cls = bnb.optim.PagedAdamW8bit if bits == 8 else bnb.optim.PagedAdamW32bit
    native = cls([reference], lr=.001, weight_decay=.02, min_8bit_size=4096)
    for _ in range(3):
        streamed.zero_grad(); native.zero_grad(set_to_none=True)
        data = batches()
        accumulate(parameter, data); accumulate(reference, data)
        assert parameter.grad is None
        assert streamed.pending_parameter_count == 1
        torch.testing.assert_close(streamed._cpu_grads[parameter], reference.grad.cpu())
        norm = streamed.clip_grad_norm_(.7)
        expected_norm = torch.nn.utils.clip_grad_norm_([reference], .7)
        torch.testing.assert_close(norm.float(), expected_norm.cpu(), atol=2e-5, rtol=2e-6)
        streamed.step(); native.step()
        torch.testing.assert_close(parameter, reference, atol=2e-6, rtol=2e-5)
        assert streamed.pending_gradient_bytes == 0
        assert parameter.grad is None
    assert streamed.state[parameter]["step"] == 3
    assert streamed.optimizer_steps == 3
    assert streamed.state[parameter]["state1"].dtype == (torch.uint8 if bits == 8 else torch.float32)
    streamed.close()


@cuda
def test_streamed_fp32_matches_torch_adamw_update():
    p = nn.Parameter(torch.randn(64, 128, device="cuda")*.03)
    q = nn.Parameter(p.detach().clone())
    streamed = CPUGradientOffloadOptimizer([p], lr=.0003, weight_decay=.01, optim_bits=32)
    native = torch.optim.AdamW([q], lr=.0003, weight_decay=.01, foreach=False)
    for _ in range(2):
        streamed.zero_grad();native.zero_grad(set_to_none=True)
        accumulate(p, batches());accumulate(q, batches())
        streamed.step();native.step()
        torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5)
    streamed.close()


@cuda
@pytest.mark.parametrize("pending", [False, True])
def test_optimizer_checkpoint_resume_native_moments_and_pending_cpu_gradients(tmp_path, pending):
    p = nn.Parameter(torch.randn(64, 128, device="cuda")*.03)
    original = CPUGradientOffloadOptimizer([p], lr=.001, optim_bits=8)
    data = batches()
    accumulate(p, data);original.step()
    if pending:
        accumulate(p,data[:1])
    torch.save({"parameter":p.detach().cpu(),"optimizer":original.state_dict()}, tmp_path/"state.pt")
    saved = torch.load(tmp_path/"state.pt", map_location="cpu", weights_only=False)
    restored_p = nn.Parameter(saved["parameter"].to("cuda"))
    restored = CPUGradientOffloadOptimizer([restored_p], lr=.9, optim_bits=8)
    restored.load_state_dict(saved["optimizer"])
    assert restored.param_groups[0]["lr"] == .001
    assert restored.optimizer_steps == original.optimizer_steps
    assert restored.pending_parameter_count == int(pending)
    remaining = data[1:] if pending else data
    accumulate(p,remaining);accumulate(restored_p,remaining)
    original.clip_grad_norm_(.5);restored.clip_grad_norm_(.5)
    original.step();restored.step()
    torch.testing.assert_close(p,restored_p,atol=1e-7,rtol=1e-6)
    assert restored.state[restored_p]["step"] == original.state[p]["step"]
    original.close();restored.close()


@cuda
def test_actual_uvm_moments_restore_as_paged_buffers(tmp_path):
    p = nn.Parameter(torch.randn(100352, device="cuda")*.01)
    optimizer = CPUGradientOffloadOptimizer([p], lr=.001, optim_bits=8)
    p.square().sum().backward();optimizer.step()
    assert optimizer.state[p]["state1"].is_paged
    torch.save(optimizer.state_dict(), tmp_path/"moments.pt")
    saved = torch.load(tmp_path/"moments.pt", map_location="cpu", weights_only=False)
    clone_p = nn.Parameter(p.detach().clone())
    restored = CPUGradientOffloadOptimizer([clone_p], lr=.001, optim_bits=8)
    restored.load_state_dict(saved)
    assert restored.state[clone_p]["state1"].is_paged
    p.square().sum().backward();clone_p.square().sum().backward()
    optimizer.step();restored.step()
    torch.testing.assert_close(p,clone_p,atol=1e-7,rtol=1e-6)
    optimizer.close();restored.close()


@cuda
def test_missing_gradients_parameter_groups_scheduler_and_nonfinite_fail_before_update():
    p = nn.Parameter(torch.tensor([1.,2.],device="cuda"))
    q = nn.Parameter(torch.tensor([3.,4.],device="cuda"))
    optimizer = CPUGradientOffloadOptimizer([
        {"params":[p],"lr":.001,"weight_decay":0},
        {"params":[q],"lr":.002,"weight_decay":.01},
    ], optim_bits=32)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.5)
    before = q.detach().clone()
    p.square().sum().backward();optimizer.step();scheduler.step()
    assert q not in optimizer.state
    torch.testing.assert_close(q,before)
    assert optimizer.param_groups[1]["lr"] == .001
    checkpoint = p.detach().clone()
    (p.sum()*float('inf')).backward()
    with pytest.raises(FloatingPointError):
        optimizer.step()
    torch.testing.assert_close(p,checkpoint)
    optimizer.zero_grad()
    assert optimizer.pending_parameter_count == 0
    optimizer.close()


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        make_full_training_optimizer([],backend="silently_freeze_the_model")
