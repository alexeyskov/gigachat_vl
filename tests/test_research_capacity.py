"""CPU/meta arithmetic checks; no language or vision checkpoint fixtures."""
import torch

from scripts.research_capacity import count_recipe,make_meta,ordinary_lora_targets,packed_lora_parameters,projector_parameters
from src.model.smallthinker import SmallThinkerConfig,register_smallthinker


def tiny_config():
    register_smallthinker()
    return SmallThinkerConfig(vocab_size=32,hidden_size=8,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=4,
        moe_ffn_hidden_size=12,moe_num_primary_experts=4,
        moe_num_active_primary_experts=2,pad_token_id=0,bos_token_id=1,eos_token_id=2,
        tie_word_embeddings=True)


def test_native_unique_count_deduplicates_tied_embeddings_and_uses_meta():
    model=make_meta(tiny_config())
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert all(parameter.device.type=='meta' for parameter in model.parameters())
    assert count_recipe(tiny_config(),'4f')['llm_trainable_parameters']==sum(p.numel() for p in model.parameters())


def test_actual_expansion_counts_equal_independent_ffn_router_and_gate_formulas():
    config=tiny_config();base=count_recipe(config,'4f')['llm_trainable_parameters']
    visual=2*3*8*12
    assert count_recipe(config,'2b')['llm_trainable_parameters']==base+visual+2*8
    def mlp(inputs,outputs):return inputs*128+128+128*outputs+outputs
    assert count_recipe(config,'3b')['llm_trainable_parameters']==base+visual+mlp(8+4,2)+mlp(8+4,1)
    for family in '123':
        b=count_recipe(config,family+'b')['llm_trainable_parameters']
        c=count_recipe(config,family+'c')['llm_trainable_parameters']
        assert c-b==2*(4+2)
        assert count_recipe(config,family+'a')['llm_trainable_parameters']==b


def test_ordinary_peft_and_residual_router_shapes_match_qlora_count():
    config=tiny_config();model=make_meta(config)
    targets={'q_proj','k_proj','v_proj','o_proj','up','gate','down'}
    lora=sum(16*(module.in_features+module.out_features) for name,module in model.named_modules()
             if isinstance(module,torch.nn.Linear) and name.rsplit('.',1)[-1] in targets)
    assert count_recipe(config,'4q')['llm_trainable_parameters']==lora
    residual_router=8*128+128+128*6+6
    assert count_recipe(config,'1b')['llm_trainable_parameters']==lora+2*3*8*12+residual_router


def test_packed_lora_uses_joint_gate_up_factorization_and_preserves_meta():
    from transformers import GptOssConfig
    config=GptOssConfig(vocab_size=32,hidden_size=8,intermediate_size=12,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=4,num_local_experts=4,
        num_experts_per_tok=2,layer_types=['full_attention'])
    model=make_meta(config);count,banks=packed_lora_parameters(model)
    assert count==4*16*(2*8+3*12)
    assert len(banks)==1
    assert all(parameter.device.type=='meta' for parameter in model.parameters())
    attention=sum(16*(module.in_features+module.out_features) for name,module in model.named_modules()
                  if isinstance(module,torch.nn.Linear) and '.self_attn.' in name)
    assert count_recipe(config,'4q')['llm_trainable_parameters']==count+attention


def test_projector_matches_both_bias_terms_for_all_donors():
    for dimension in (1152,3584,2560):
        assert projector_parameters(dimension,1536)==dimension*1536+1536+1536*1536+1536


def test_deepseek_count_only_scope_avoids_double_counting_native_packed_factors():
    from transformers import DeepseekV3Config
    config=DeepseekV3Config(vocab_size=32,hidden_size=8,intermediate_size=12,moe_intermediate_size=6,
        num_hidden_layers=1,first_k_dense_replace=0,num_attention_heads=2,num_key_value_heads=1,
        n_routed_experts=4,n_shared_experts=2,num_experts_per_tok=2,n_group=1,topk_group=1,
        routed_scaling_factor=1.0,q_lora_rank=None,kv_lora_rank=4,
        qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=4)
    model=make_meta(config);packed,_=packed_lora_parameters(model)
    targets=set(ordinary_lora_targets(model))
    ordinary=sum(16*(module.in_features+module.out_features) for name,module in model.named_modules()
                 if name in targets)
    details=count_recipe(config,'4q')
    assert details['llm_trainable_parameters']==ordinary+packed
    assert details['count_only_peft_scope_override'].startswith('direct ordinary PEFT Linear')
