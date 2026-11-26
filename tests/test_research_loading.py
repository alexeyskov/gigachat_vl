"""GPT source-loader kwargs and tiny native CUDA quantized training contracts."""
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import Mxfp4Config
import src.model.research_loading as loading


class FakeLoadedGPT(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(model_type='gpt_oss')
        self.weight=nn.Parameter(torch.zeros(4,4,dtype=torch.bfloat16))


class TestGPTResearchLoaderContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        (self.path/'config.json').write_text(json.dumps({'model_type':'gpt_oss','quantization_config':{'quant_method':'mxfp4'}}))
    def tearDown(self):self.temp.cleanup()
    def bank(self):
        return SimpleNamespace(gate_up_proj=torch.zeros(4,4,8,dtype=torch.bfloat16),down_proj=torch.zeros(4,4,4,dtype=torch.bfloat16))

    def test_full_source_uses_native_mxfp4_dequantization_not_bnb_override(self):
        model=FakeLoadedGPT()
        with patch.object(loading.AutoModelForCausalLM,'from_pretrained',return_value=model) as source, \
             patch.object(loading,'_native_banks',return_value=[('experts',self.bank())]), \
             patch.object(loading,'_memory',return_value={'available':False}), \
             patch.object(loading,'quantize_packed_experts') as packed:
            returned,report=loading.load_gpt_oss_for_research(self.path,revision='a'*40,stage='SFT',recipe='4f')
        self.assertIs(returned,model);packed.assert_not_called()
        kwargs=source.call_args.kwargs
        self.assertIsInstance(kwargs['quantization_config'],Mxfp4Config)
        self.assertIs(kwargs['quantization_config'].dequantize,True)
        self.assertEqual(kwargs['dtype'],torch.bfloat16)
        self.assertEqual(kwargs['device_map'],{'':'cuda:0'})
        self.assertEqual(kwargs['attn_implementation'],'eager')
        self.assertTrue(kwargs['local_files_only']);self.assertFalse(kwargs['trust_remote_code'])
        self.assertEqual(report['loading_route'],['original_mxfp4','dequantized_bf16'])
        self.assertFalse(report['direct_bitsandbytes_source_override'])
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_qlora_rank_and_trainability_are_stage_recipe_specific(self):
        for stage,recipe,rank,enabled in [('A0','1c',0,False),('A1','1b',16,False),('A1','4q',16,True),('SFT','1a',16,True)]:
            with self.subTest(stage=stage,recipe=recipe):
                with patch.object(loading.AutoModelForCausalLM,'from_pretrained',return_value=FakeLoadedGPT()) as source, \
                     patch.object(loading,'_native_banks',return_value=[('experts',self.bank())]), \
                     patch.object(loading,'_memory',return_value={'available':False}), \
                     patch.object(loading,'quantize_packed_experts',return_value={'banks':1}) as packed, \
                     patch.object(loading,'quantize_gpt_oss_attention',return_value={'modules':['q_proj']}) as dense, \
                     patch.object(loading,'set_packed_lora_trainable') as trainable:
                    _,report=loading.load_gpt_oss_for_research(self.path,revision='a'*40,stage=stage,recipe=recipe)
                self.assertEqual(packed.call_args.kwargs['rank'],rank)
                self.assertEqual(dense.call_args.kwargs['rank'],rank)
                trainable.assert_called_once_with(source.return_value,enabled=enabled)
                self.assertIsInstance(source.call_args.kwargs['quantization_config'],Mxfp4Config)
                self.assertEqual(report['lora_enabled'],enabled)

    def test_remote_unpinned_or_wrong_source_configs_are_refused_before_loading(self):
        with patch.object(loading.AutoModelForCausalLM,'from_pretrained') as source:
            with self.assertRaisesRegex(ValueError,'existing local'):loading.load_gpt_oss_for_research('openai/gpt-oss-20b',revision='a'*40)
            with self.assertRaisesRegex(ValueError,'pinned'):loading.load_gpt_oss_for_research(self.path)
            (self.path/'config.json').write_text(json.dumps({'model_type':'gpt_oss','quantization_config':{'quant_method':'bitsandbytes'}}))
            with self.assertRaisesRegex(ValueError,'original GPT-OSS MXFP4'):loading.load_gpt_oss_for_research(self.path,revision='a'*40)
            source.assert_not_called()

    def test_cpu_device_and_unknown_stage_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'CUDA'):loading.load_gpt_oss_for_research(self.path,revision='a'*40,device='cpu')
        with self.assertRaisesRegex(ValueError,'stage or recipe'):loading.load_gpt_oss_for_research(self.path,revision='a'*40,stage='unknown')


@unittest.skipUnless(torch.cuda.is_available(),'Native NF4 integration requires CUDA')
class TestTinyGPTResearchNF4CUDA(unittest.TestCase):
    def test_rank_zero_alignment_keeps_all_weights_frozen_but_input_gradient_flows(self):
        from transformers import GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
        from src.model.research_quantization import NF4LoRALinear
        config=GptOssConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,num_local_experts=4,
            num_experts_per_tok=2,layer_types=['full_attention'],
            rope_parameters={'rope_type':'default','rope_theta':10000.})
        config._attn_implementation='eager'
        native=GptOssForCausalLM(config).to(device='cuda',dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            (path/'config.json').write_text(json.dumps({'model_type':'gpt_oss','quantization_config':{'quant_method':'mxfp4'}}))
            with patch.object(loading.AutoModelForCausalLM,'from_pretrained',return_value=native):
                model,report=loading.load_gpt_oss_for_research(path,revision='a'*40,stage='A0',recipe='1b')
            self.assertEqual(report['rank'],0);self.assertEqual(report['trainable_parameter_count'],0)
            modules=[module for module in model.modules() if isinstance(module,NF4LoRALinear)]
            self.assertEqual(len(modules),12);self.assertTrue(all(module.lora_A is None for module in modules))
            inputs=torch.randn(1,4,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            output=model(inputs_embeds=inputs,labels=torch.randint(0,64,(1,4),device='cuda'),use_cache=False)
            output.loss.backward()
            self.assertTrue(torch.isfinite(inputs.grad).all());self.assertGreater(inputs.grad.abs().sum(),0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        del native,model,modules,inputs,output
        gc.collect();torch.cuda.empty_cache()

    def test_native_bf16_gpt_quantization_lora_ek_forward_backward_under_100mib(self):
        from transformers import GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
        from src.model.research_quantization import NF4LoRALinear,set_packed_lora_trainable
        from src.model.research_moe import install_research_moe,set_modality_mask,configure_research_trainability
        gc.collect();torch.cuda.empty_cache();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        config=GptOssConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,num_local_experts=4,
            num_experts_per_tok=2,layer_types=['full_attention'],
            rope_parameters={'rope_type':'default','rope_theta':10000.},attention_dropout=0.)
        config._attn_implementation='eager'
        native=GptOssForCausalLM(config).to(device='cuda',dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            (path/'config.json').write_text(json.dumps({'model_type':'gpt_oss','quantization_config':{'quant_method':'mxfp4'}}))
            with patch.object(loading.AutoModelForCausalLM,'from_pretrained',return_value=native):
                model,report=loading.load_gpt_oss_for_research(path,revision='a'*40,stage='SFT',recipe='1c',lora_dropout=0.)
            modules=[module for module in model.modules() if isinstance(module,NF4LoRALinear)]
            # Four experts x gate_up/down, plus q/k/v/o attention projections.
            self.assertEqual(len(modules),12)
            self.assertEqual(report['attention_conversion']['modules'],[
                'model.layers.0.self_attn.q_proj','model.layers.0.self_attn.k_proj',
                'model.layers.0.self_attn.v_proj','model.layers.0.self_attn.o_proj'])
            self.assertTrue(all(module.base.weight.dtype==torch.uint8 and not module.base.weight.requires_grad for module in modules))
            self.assertTrue(all(not p.requires_grad for name,p in model.named_parameters() if 'lora_' not in name))
            self.assertTrue(all(module.lora_A.requires_grad and module.lora_B.requires_grad for module in modules))
            self.assertTrue(all(module.base.bias is None or not module.base.bias.requires_grad for module in modules))
            # The common helper must also toggle the ordinary attention adapters.
            set_packed_lora_trainable(model,False)
            self.assertTrue(all(not module.lora_A.requires_grad for module in modules))
            set_packed_lora_trainable(model,True)
            wrappers=install_research_moe(model,1,'c',added_experts=2)
            configure_research_trainability(model,stage='SFT')
            wrappers[0].warmup_force_new=True
            mask=torch.tensor([[True,False]*4],device='cuda')
            set_modality_mask(model,mask,torch.ones_like(mask))
            inputs=torch.randn(1,8,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            labels=torch.randint(0,128,(1,8),device='cuda')
            output=model(inputs_embeds=inputs,labels=labels,use_cache=False,return_dict=True)
            self.assertTrue(torch.isfinite(output.loss));output.loss.backward()
            self.assertIsNotNone(inputs.grad);self.assertTrue(torch.isfinite(inputs.grad).all());self.assertGreater(inputs.grad.abs().sum(),0)
            self.assertTrue(any(module.lora_B.grad is not None and module.lora_B.grad.abs().sum()>0 for module in modules))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in wrappers[0].visual_experts.parameters()))
            self.assertTrue(all(module.base.weight.grad is None for module in modules))
            self.assertLess(torch.cuda.max_memory_allocated()-baseline,100*1024**2)
        del output,model,native,modules,wrappers,inputs
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':unittest.main()
