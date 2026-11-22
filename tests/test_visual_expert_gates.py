"""CPU checks against the actual native Transformers 5.15 DeepSeek-V2 FFNs."""

import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2MLP, DeepseekV2Moe

from src.model.visual_expert_gates import NativeDeepseekVisualGates


class NativeDeepseekVisualGatesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(173)
        self.config = DeepseekV2Config(
            hidden_size=8,
            intermediate_size=12,
            moe_intermediate_size=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=4,
            n_shared_experts=2,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            topk_method="greedy",
            routed_scaling_factor=1.0,
            mlp_bias=False,
        )
        self.config._experts_implementation = "eager"
        self.hidden = torch.randn(2, 3, 8)
        self.mask = torch.tensor([[False, True, True], [True, False, False]])

    def make_base(self, dense=False):
        base = DeepseekV2MLP(self.config) if dense else DeepseekV2Moe(self.config)
        for parameter in base.parameters():
            nn.init.normal_(parameter, std=0.2)
        return base

    @staticmethod
    def set_logits(gates, value):
        with torch.no_grad():
            for parameter in gates.parameters():
                parameter.fill_(value)

    def test_fully_open_matches_native_moe_and_dense(self):
        for dense in (False, True):
            with self.subTest(dense=dense):
                base = self.make_base(dense=dense)
                gates = NativeDeepseekVisualGates(base, self.config).eval()
                self.set_logits(gates, 20.0)
                torch.testing.assert_close(gates(base, self.hidden, self.mask), base(self.hidden))

    def test_closure_removes_visual_ffn_output_and_preserves_text(self):
        for dense in (False, True):
            with self.subTest(dense=dense):
                base = self.make_base(dense=dense)
                gates = NativeDeepseekVisualGates(base, self.config).eval()
                self.set_logits(gates, -20.0)
                actual = gates(base, self.hidden, self.mask)
                expected = base(self.hidden)
                torch.testing.assert_close(actual[self.mask], torch.zeros_like(actual[self.mask]), rtol=0, atol=0)
                torch.testing.assert_close(actual[~self.mask], expected[~self.mask])

    def test_no_visual_input_is_exact_native_path(self):
        base = self.make_base()
        gates = NativeDeepseekVisualGates(base, self.config).train()
        for mask in (None, torch.zeros_like(self.mask)):
            with self.subTest(mask=mask):
                state_before = torch.random.get_rng_state()
                actual = gates(base, self.hidden, mask)
                self.assertTrue(torch.equal(state_before, torch.random.get_rng_state()))
                torch.testing.assert_close(actual, base(self.hidden), rtol=0, atol=0)

    def test_router_receives_original_input_once(self):
        base = self.make_base()
        gates = NativeDeepseekVisualGates(base, self.config).eval()
        self.set_logits(gates, -20.0)
        with patch.object(base.gate, "forward", wraps=base.gate.forward) as router:
            gates(base, self.hidden, self.mask)
            router.assert_called_once()
            self.assertIs(router.call_args.args[0], self.hidden)

    def test_shared_experts_have_independent_input_masks(self):
        base = self.make_base()
        gates = NativeDeepseekVisualGates(base, self.config).eval()
        shared = base.shared_experts
        for shared_idx in range(2):
            with self.subTest(shared_idx=shared_idx):
                self.set_logits(gates, -20.0)
                with torch.no_grad():
                    gates.shared_logits[shared_idx] = 20.0
                actual = gates(base, self.hidden, torch.ones_like(self.mask))
                start, end = shared_idx * 4, (shared_idx + 1) * 4
                gate = F.linear(self.hidden, shared.gate_proj.weight[start:end])
                up = F.linear(self.hidden, shared.up_proj.weight[start:end])
                expected = F.linear(shared.act_fn(gate) * up, shared.down_proj.weight[:, start:end])
                torch.testing.assert_close(actual, expected)

    def test_gate_and_base_gradients_and_expected_l0(self):
        for dense in (False, True):
            with self.subTest(dense=dense):
                base = self.make_base(dense=dense)
                gates = NativeDeepseekVisualGates(base, self.config).eval()
                hidden = self.hidden.clone().requires_grad_()
                task_loss = gates(base, hidden, self.mask).square().sum()
                task_loss.backward()
                self.assertTrue(torch.isfinite(hidden.grad).all())
                self.assertTrue(any(parameter.grad is not None for parameter in base.parameters()))
                self.assertTrue(any(torch.count_nonzero(parameter.grad) > 0 for parameter in gates.parameters()))
                gates.zero_grad()
                penalty = gates.regularization_loss()
                self.assertGreater(penalty.item(), 0.0)
                self.assertLess(penalty.item(), 1.0)
                penalty.backward()
                for parameter in gates.parameters():
                    self.assertTrue(torch.all(parameter.grad > 0))

    def test_bfloat16_activations_keep_fp32_gate_parameters_and_finite_gradients(self):
        for dense in (False, True):
            with self.subTest(dense=dense):
                base = self.make_base(dense=dense).to(dtype=torch.bfloat16)
                gates = NativeDeepseekVisualGates(base, self.config, init_open=0.5).train()
                hidden = self.hidden.to(dtype=torch.bfloat16).requires_grad_()
                output = gates(base, hidden, self.mask)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(output).all())
                (output.float().square().sum() + gates.regularization_loss()).backward()
                self.assertEqual(hidden.grad.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(hidden.grad).all())
                for parameter in gates.parameters():
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_initialization_diagnostics_and_hard_concrete_sampling(self):
        base = self.make_base()
        gates = NativeDeepseekVisualGates(base, self.config, init_open=0.9)
        for value in gates.gate_values().values():
            torch.testing.assert_close(value, torch.full_like(value, 0.9))
            self.assertFalse(value.requires_grad)
        probability = 1 / (1 + math.exp(-(math.log(5.0) - (2 / 3) * math.log(1 / 11))))
        self.assertAlmostEqual(gates.regularization_loss().item(), probability, places=6)
        samples = gates._gate(torch.zeros(1000), sample=True)
        self.assertTrue(torch.any(samples == 0))
        self.assertTrue(torch.any(samples == 1))
        self.assertTrue(torch.all((samples >= 0) & (samples <= 1)))

    def test_module_owns_only_gate_parameters(self):
        base = self.make_base()
        gates = NativeDeepseekVisualGates(base, self.config)
        self.assertEqual(set(gates.state_dict()), {"routed_logits", "shared_logits"})
        self.assertEqual(sum(parameter.numel() for parameter in gates.parameters()), 6)

    def test_stochastic_gate_checkpoint_recomputation_preserves_gradients(self):
        base = self.make_base()
        gates = NativeDeepseekVisualGates(base, self.config, init_open=0.5).train()
        hidden = self.hidden.clone().requires_grad_()
        torch.manual_seed(42)
        gates(base, hidden, self.mask).square().sum().backward()
        reference_hidden_grad = hidden.grad.clone()
        reference_gate_grads = [parameter.grad.clone() for parameter in gates.parameters()]
        hidden.grad = None
        base.zero_grad()
        gates.zero_grad()
        torch.manual_seed(42)
        checkpoint(
            lambda value: gates(base, value, self.mask),
            hidden,
            use_reentrant=False,
            preserve_rng_state=True,
        ).square().sum().backward()
        torch.testing.assert_close(hidden.grad, reference_hidden_grad)
        for parameter, expected_grad in zip(gates.parameters(), reference_gate_grads):
            torch.testing.assert_close(parameter.grad, expected_grad)

    def test_rejects_adapter_projections_even_after_installation(self):
        class CustomLinear(nn.Linear):
            pass

        base = self.make_base()
        gates = NativeDeepseekVisualGates(base, self.config)
        base.shared_experts.gate_proj = CustomLinear(8, 8, bias=False)
        with self.assertRaisesRegex(TypeError, "ordinary nn.Linear"):
            gates(base, self.hidden, self.mask)
        with self.assertRaisesRegex(TypeError, "ordinary nn.Linear"):
            NativeDeepseekVisualGates(base, self.config)

    def test_rejects_invalid_mask_and_remote_code_router_signature(self):
        base = self.make_base()
        gates = NativeDeepseekVisualGates(base, self.config)
        with self.assertRaisesRegex(ValueError, "token dimensions"):
            gates(base, self.hidden, self.mask[:, :1])
        with self.assertRaisesRegex(TypeError, "boolean"):
            gates(base, self.hidden, self.mask.float())
        with patch.object(base.gate, "forward", return_value=(torch.ones(6, 2), torch.zeros(6, 2))):
            with self.assertRaisesRegex(TypeError, "router output"):
                gates(base, self.hidden, self.mask)


if __name__ == "__main__":
    unittest.main()
