"""Tiny actual Small MoE math/RNG/gradient and transactional routing tests."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.model.smallthinker import SmallThinkerConfig, SmallThinkerMoeBlock
from src.model.research_moe import ResearchSmallThinkerMoE
from src.utils import research_routing_aggregate_v1 as routing


class Tiny(nn.Module):
    def __init__(self, recipe='3c', *, k=2, dropout=False, recompute=False):
        super().__init__(); self.recompute = recompute
        config = SmallThinkerConfig(vocab_size=8, hidden_size=4, num_hidden_layers=1, num_attention_heads=1,
            num_key_value_heads=1, head_dim=4, moe_ffn_hidden_size=5, moe_num_primary_experts=3,
            moe_num_active_primary_experts=k, max_position_embeddings=128)
        native = SmallThinkerMoeBlock(config)
        self.moe = native if recipe == '4f' else ResearchSmallThinkerMoE(native, int(recipe[0]), recipe[1], added_experts=2)
        if dropout:
            experts = list(self.moe.experts) + list(getattr(self.moe, 'visual_experts', []))
            for expert in experts: expert.up = nn.Sequential(nn.Dropout(.2), expert.up)
    def forward(self, value):
        if self.recompute: return checkpoint(self.moe, value, value, use_reentrant=False)[0]
        return self.moe(value, value)[0]


def rows():
    return [routing.TaggedRow(([1], [2], [3], [torch.zeros(2, 3)]), {'record_id': 'mm', 'task_group': 'vqa'}),
        routing.TaggedRow(([], [1], [2], []), {'record_id': 'text', 'task_group': 'raw_text'})]


def masks(model):
    visual = torch.tensor([[False, True, True, False, False], [False] * 5])
    valid = torch.tensor([[True] * 5, [True, True, False, False, False]])
    if hasattr(model.moe, 'set_modality_mask'): model.moe.set_modality_mask(visual, valid)
    return visual, valid


def forward_backward(model, value, collector=None):
    batch = rows(); masks(model)
    if collector:
        collector.begin_window([batch]); collector.begin_attempt()
        with collector.forward_scope(batch): output = model(value)
    else: output = model(value)
    output.square().sum().backward()
    if collector: collector.finish_attempt()
    return output


@pytest.mark.parametrize('recipe', ['4f'] + [str(family) + policy for family in (1, 2, 3) for policy in 'abc'])
def test_actual_native_forward_gradients_and_rng_are_bit_exact(recipe):
    torch.manual_seed(917); original = Tiny(recipe, dropout=True); observed = copy.deepcopy(original)
    value = torch.randn(2, 5, 4, requires_grad=True); second = value.detach().clone().requires_grad_()
    before = torch.get_rng_state(); expected = forward_backward(original, value); after = torch.get_rng_state()
    torch.set_rng_state(before)
    collector = routing.RoutingAggregate(observed, ['vqa', 'raw_text'])
    actual = forward_backward(observed, second, collector)
    assert torch.equal(expected, actual) and torch.equal(value.grad, second.grad) and torch.equal(after, torch.get_rng_state())
    for (_, left), (_, right) in zip(original.named_parameters(), observed.named_parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None: assert torch.equal(left.grad, right.grad)
    optimizer = torch.optim.AdamW(observed.parameters(), lr=1e-3); schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    optimizer.step(); schedule.step(); collector.commit(1, 2)
    state = collector.export_state(step=1, seen=2); counters = state['layers']['moe']['counters']
    assert counters['tokens'].tolist() == [[3, 2], [2, 0]]
    assert torch.equal(counters['dispatch_counts'], counters['norm_dispatch_counts'])
    assert int(counters['dispatch_counts'].sum()) == 14
    assert (counters['output_l2_sum'] >= 0).all() and state['P06_complete'] is False
    assert state['shared_expert_contribution']['present'] is False
    collector.close(); assert 'forward' not in observed.moe.__dict__ and '_routing_aggregate_observer' not in observed.moe.__dict__


def test_hard_mask_unfilled_slots_are_not_misreported_as_overflow_or_drops():
    torch.manual_seed(18); model = Tiny('2a', k=3); collector = routing.RoutingAggregate(model, ['vqa', 'raw_text'])
    forward_backward(model, torch.randn(2, 5, 4, requires_grad=True), collector)
    torch.optim.SGD(model.parameters(), lr=.01).step(); collector.commit(1, 2)
    state = collector.export_state(step=1, seen=2); counts = state['layers']['moe']['counters']
    assert counts['unfilled_topk_slots'].tolist() == [[0, 2], [0, 0]]
    assert int(counts['dispatch_counts'][0, 1, :3].sum()) == 0
    assert int(counts['dispatch_counts'][:, 0, 3:].sum()) == 0
    assert state['capacity_overflow_and_dropped_tokens'] == {'mechanism_present': False, 'structural_count': 0}
    collector.close()


def test_R3_conditional_old_distribution_survives_saturated_actual_alpha():
    torch.manual_seed(99); model = Tiny('3c')
    with torch.no_grad(): model.moe.router_mass.down.bias.fill_(100.)
    collector = routing.RoutingAggregate(model, ['vqa', 'raw_text'])
    forward_backward(model, torch.randn(2, 5, 4, requires_grad=True), collector)
    torch.optim.SGD(model.parameters(), lr=.01).step(); collector.commit(1, 2)
    counts = collector.export_state(step=1, seen=2)['layers']['moe']['counters']
    assert torch.equal(counts['alpha_sum'], counts['tokens'].double())
    assert counts['pre_mask_mass_sum'][..., :3].sum() == 0
    torch.testing.assert_close(counts['conditional_old_current_mass_sum'][..., :3].sum(-1), counts['tokens'].double())
    collector.close()


def test_native_BF16_autocast_probabilities_keep_actual_activation_rounding():
    torch.manual_seed(42); model = Tiny('4f'); collector = routing.RoutingAggregate(model, ['vqa', 'raw_text'])
    value = torch.randn(2, 5, 4, requires_grad=True); batch = rows(); collector.begin_window([batch]); collector.begin_attempt()
    with collector.forward_scope(batch), torch.autocast('cpu', dtype=torch.bfloat16): output = model(value)
    output.sum().backward(); collector.finish_attempt()
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16): logits = model.moe.primary_router(value.reshape(-1, 4))
    represented = logits.sigmoid().float(); expected = represented / represented.sum(-1, keepdim=True)
    _, valid = masks(model)
    observed = collector.layers['moe']['stage']['pre_mask_mass_sum'].sum((0, 1))
    torch.testing.assert_close(observed, expected[valid.reshape(-1)].double().sum(0))
    collector.abort_window(); collector.close()


def test_gradient_checkpoint_recomputation_and_internal_eval_do_not_add_counts():
    torch.manual_seed(7); original = Tiny('3c', recompute=True); observed = copy.deepcopy(original)
    value = torch.randn(2, 5, 4, requires_grad=True); second = value.detach().clone().requires_grad_()
    expected = forward_backward(original, value)
    collector = routing.RoutingAggregate(observed, ['vqa', 'raw_text'])
    actual = forward_backward(observed, second, collector)
    assert torch.equal(expected, actual) and torch.equal(value.grad, second.grad)
    # The pending update is not a forward scope: even full internal CE passes
    # cannot change staged counts after backward has finished.
    before = {k: v.clone() for k, v in collector.layers['moe']['stage'].items()}
    observed.eval()
    with torch.no_grad(): observed(second.detach())
    assert all(torch.equal(value, collector.layers['moe']['stage'][key]) for key, value in before.items())
    observed.train(); torch.optim.SGD(observed.parameters(), lr=.01).step(); collector.commit(1, 2)
    assert collector.export_state(step=1, seen=2)['layers']['moe']['counters']['tokens'].sum() == 7
    collector.close()


def test_OOM_partial_attempt_is_discarded_and_only_final_window_commits():
    torch.manual_seed(33); model = Tiny('3c'); collector = routing.RoutingAggregate(model, ['vqa', 'raw_text'])
    batch = rows(); value = torch.randn(2, 5, 4, requires_grad=True); masks(model)
    collector.begin_window([batch]); collector.begin_attempt()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        with collector.forward_scope(batch):
            model(value)
            raise torch.cuda.OutOfMemoryError('synthetic after routing observations, before backward/update')
    collector.abort_attempt(); collector.begin_attempt()
    with collector.forward_scope(batch): output = model(value)
    output.sum().backward(); collector.finish_attempt()
    assert collector.step == 0
    with pytest.raises(ValueError): collector.export_state(step=1, seen=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01); scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    optimizer.step(); scheduler.step(); collector.commit(1, 2)
    state = collector.export_state(step=1, seen=2)
    assert state['discarded_attempts'] == 1 and state['layers']['moe']['counters']['tokens'].sum() == 7
    collector.close()


def test_resume_state_is_native_CPU_serializable_and_exactly_bound_to_step_profile_and_layers():
    torch.manual_seed(2); model = Tiny('3c'); other = copy.deepcopy(model)
    collector = routing.RoutingAggregate(model, ['vqa', 'raw_text'])
    forward_backward(model, torch.randn(2, 5, 4, requires_grad=True), collector)
    torch.optim.SGD(model.parameters(), lr=.01).step(); collector.commit(1, 2)
    state = collector.export_state(step=1, seen=2); blob = io.BytesIO(); torch.save(state, blob)
    loaded = torch.load(io.BytesIO(blob.getvalue()), weights_only=False, map_location='cpu')
    restored = routing.RoutingAggregate(other, ['vqa', 'raw_text']); restored.restore_state(loaded, step=1, seen=2)
    repeated = restored.export_state(step=1, seen=2)
    assert state['coverage_sha256'] == repeated['coverage_sha256']
    for key in state['layers']['moe']['counters']:
        assert torch.equal(state['layers']['moe']['counters'][key], repeated['layers']['moe']['counters'][key])
    collector.close(); restored.close()
    wrong = routing.RoutingAggregate(other, ['vqa', 'raw_text'])
    with pytest.raises(ValueError): wrong.restore_state(loaded, step=2, seen=2)
    wrong.close()


def test_untagged_unknown_group_duplicate_scope_and_modified_masks_fail_closed():
    model = Tiny('2c'); collector = routing.RoutingAggregate(model, ['vqa', 'raw_text'])
    with pytest.raises(ValueError): collector.begin_window([[tuple(rows()[0])]])
    batch = rows(); collector.begin_window([batch]); collector.begin_attempt(); masks(model)
    with pytest.raises(ValueError, match='Nested'):
        with collector.forward_scope(batch):
            with collector.forward_scope(batch): pass
    collector.abort_attempt(); collector.begin_attempt()
    model.moe.set_modality_mask(torch.zeros(2, 5, dtype=torch.bool), torch.ones(2, 5, dtype=torch.bool))
    with collector.forward_scope(batch): output = model(torch.randn(2, 5, 4, requires_grad=True))
    output.sum().backward(); collector.finish_attempt(); torch.optim.SGD(model.parameters(), lr=.01).step(); collector.commit(1, 2)
    with pytest.raises(ValueError, match='mask/score/norm'): collector.export_state(step=1, seen=2)
    collector.close()


def test_metadata_source_profile_imports_no_torch_in_fresh_process():
    source = "import sys,json;from src.utils import research_routing_aggregate_v1 as a;p=a.profile(['captioning']);assert 'torch' not in sys.modules;print(json.dumps(p))"
    result = subprocess.run([sys.executable, '-c', source], cwd=routing.ROOT, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['actual_GPU_parity_and_overhead_validated'] is False
