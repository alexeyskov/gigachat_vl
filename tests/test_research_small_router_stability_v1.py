"""Numerical regression at the actual failing router logits, including backward."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import unittest
import torch
from src.utils.research_small_router_stability_v1 import normalize_sigmoid, private_smallthinker


class RouterStability(unittest.TestCase):
    def test_actual_failure_and_backward(self):
        logits = torch.tensor([[-97., -120., -124.5, -125.5]], requires_grad=True)
        self.assertEqual(float(logits.sigmoid().sum()), 0.)
        weights = normalize_sigmoid(logits)
        oracle = torch.sigmoid(logits.detach().double())
        oracle /= oracle.sum(-1, keepdim=True)
        torch.testing.assert_close(weights.double(), oracle, atol=1e-14, rtol=2e-6)
        (weights*torch.arange(4)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_ordinary_values_and_gradients_exact(self):
        for dtype in (torch.float32, torch.bfloat16):
            left = torch.tensor([[-2., 0., 1., 3.], [8., -8., .2, -.3]], dtype=dtype, requires_grad=True)
            right = left.detach().clone().requires_grad_(True)
            scores = left.sigmoid()
            expected = scores/scores.sum(-1, keepdim=True)
            actual = normalize_sigmoid(right)
            self.assertTrue(torch.equal(expected, actual))
            factors = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]], dtype=dtype)
            (expected*factors).sum().backward()
            (actual*factors).sum().backward()
            self.assertTrue(torch.equal(left.grad, right.grad))

    def test_masked_slots_and_empty_rows(self):
        logits = torch.tensor([[-97., -120., -124.5, -125.5], [-2., 3., 4., 5.]], requires_grad=True)
        mask = torch.tensor([[False, True, True, False], [False]*4])
        weights = normalize_sigmoid(logits, slot_valid=mask, clamp_tiny=True)
        self.assertEqual(float(weights[1].sum()), 0.)
        self.assertEqual(float(weights[0, ~mask[0]].abs().sum()), 0.)
        self.assertAlmostEqual(float(weights[0].sum()), 1., places=6)
        (weights*torch.arange(4)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask])))

    def test_nonfinite_is_not_silenced(self):
        logits = torch.tensor([[float('nan'), -97., -120., -124.]])
        self.assertFalse(torch.isfinite(normalize_sigmoid(logits)).all())

    def test_private_clone_preserves_canonical_class(self):
        from src.model.smallthinker import SmallThinkerMoeBlock
        native = SmallThinkerMoeBlock.forward
        private = private_smallthinker()
        self.assertIs(SmallThinkerMoeBlock.forward, native)
        self.assertIsNot(private.SmallThinkerMoeBlock, SmallThinkerMoeBlock)
        self.assertIn('_safe_normalize_sigmoid', private.SmallThinkerMoeBlock.forward.__globals__)

    def test_real_moe_forward_and_backward_at_failed_logits(self):
        from src.model.smallthinker import SmallThinkerMoeBlock, SmallThinkerConfig
        private = private_smallthinker()
        config = SmallThinkerConfig(hidden_size=4, num_hidden_layers=1, num_attention_heads=1,
            num_key_value_heads=1, head_dim=4, moe_ffn_hidden_size=8,
            moe_num_primary_experts=4, moe_num_active_primary_experts=4)
        native = SmallThinkerMoeBlock(config).bfloat16()
        corrected = private.SmallThinkerMoeBlock(config).bfloat16()
        with torch.no_grad():
            native.primary_router.weight.zero_()
            native.primary_router.weight[:, 0].copy_(torch.tensor([-97., -120., -124.5, -125.5]))
        corrected.load_state_dict(native.state_dict())
        native.use_grouped_mm = corrected.use_grouped_mm = False
        router = torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.bfloat16)
        hidden = torch.ones_like(router).requires_grad_(True)
        self.assertFalse(torch.isfinite(native(router, hidden)[0]).all())
        output = corrected(router, hidden)[0]
        self.assertTrue(torch.isfinite(output).all())
        output.float().sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in corrected.parameters()))


if __name__ == '__main__':
    unittest.main()
