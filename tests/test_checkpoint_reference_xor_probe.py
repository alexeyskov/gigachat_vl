import importlib.util
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location('checkpoint_reference_xor_probe',Path(__file__).parents[1]/'scripts/probe_checkpoint_reference_xor.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class TestCheckpointReferenceXOR(unittest.TestCase):
    def test_exact_fp32_bit_reconstruction_including_non_arithmetic_float_values(self):
        bits=torch.tensor([0,-2147483648,2139095040,-8388608,2143289345,1065353217],dtype=torch.int32)
        trained=bits.view(torch.float32).reshape(2,3)
        original=torch.tensor([[0.,0.,float('inf')],[float('-inf'),float('nan'),1.]],dtype=torch.bfloat16)
        encoded=probe.xor_shuffle(trained,original)
        self.assertEqual(len(encoded),trained.numel()*4)
        restored=probe.decode_xor_shuffle(encoded,original)
        self.assertEqual(restored.dtype,torch.float32)
        self.assertTrue(torch.equal(trained.view(torch.uint8),restored.view(torch.uint8)))

    def test_noncontiguous_tensors_are_reconstructed_without_reference_mutation(self):
        original=torch.randn(4,6,dtype=torch.bfloat16)[:,::2]
        before=original.clone()
        trained=(original.float()+torch.linspace(-1e-6,1e-6,12).reshape(4,3)).t()
        reference=original.t()
        encoded=probe.xor_shuffle(trained,reference)
        restored=probe.decode_xor_shuffle(encoded,reference)
        torch.testing.assert_close(restored,trained,rtol=0,atol=0)
        torch.testing.assert_close(original,before,rtol=0,atol=0)

    def test_shape_dtype_and_length_mismatches_fail_explicitly(self):
        with self.assertRaises(ValueError):probe.xor_shuffle(torch.zeros(3),torch.zeros(4,dtype=torch.bfloat16))
        with self.assertRaises(ValueError):probe.xor_shuffle(torch.zeros(3,dtype=torch.float16),torch.zeros(3,dtype=torch.bfloat16))
        with self.assertRaises(ValueError):probe.decode_xor_shuffle(bytes(3),torch.zeros(1,dtype=torch.bfloat16))


if __name__=='__main__':unittest.main()
