"""Actual tiny native CPU parity; no production profile or GPU is changed."""
import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.eval.research_batch_inference import (
    build_batch_generation_inputs,generate_batch_predictions,trim_generated_ids,benchmark_main,require_idle_gpu,
)
from src.eval.research_inference import build_generation_inputs,freeze_for_generation,generate_prediction
from src.model.research_moe import install_research_moe,set_modality_mask
from tests.test_research_inference import TinyResearchModel


@pytest.fixture(autouse=True)
def cpu_threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def records():
    return [
        {'record_id':'a','question':'question red','image_paths':['first.png'],'image_hashes':['first'],'answers':['gold0']},
        {'record_id':'b','question':'blue question red blue red question','image_paths':[],'image_hashes':[],'answer':'gold1'},
        {'record_id':'c','question':'red','image_paths':['second.png'],'image_hashes':['second'],'answers':['gold2']},
        {'record_id':'d','question':'blue red blue red blue red blue red question','image_paths':[],'image_hashes':[]},
    ]


def model(policy=None):
    torch.manual_seed(144)
    result=TinyResearchModel()
    # Exercise contextual dependence rather than a tiny residual-dominated
    # fixture that always emits the same assistant token regardless of images.
    with torch.no_grad():
        for module in result.base.modules():
            if isinstance(module,nn.Linear):module.weight.mul_(8.)
    # An independent output head avoids the trivial tied-embedding self-repeat
    # solution, making token parity sensitive to contextual attention changes.
    result.base.lm_head=nn.Linear(result.base.config.hidden_size,result.base.config.vocab_size,bias=False)
    nn.init.normal_(result.base.lm_head.weight,std=.7)
    result.wrappers=install_research_moe(result.base,3,policy) if policy else []
    if policy=='c':
        with torch.no_grad():result.wrappers[0].modality_gate_logits.copy_(
            torch.linspace(-2,2,result.wrappers[0].modality_gate_logits.numel()).reshape_as(result.wrappers[0].modality_gate_logits))
    def features(paths,encode,**kwargs):
        result.cache.calls.append((list(paths),kwargs))
        return [torch.tensor([[1.,2.,3.],[4.,5.,6.]]) if path=='first.png'
                else torch.tensor([[-1.,-2.,1.],[2.,-.5,3.],[1.,1.,1.]]) for path in paths]
    result.cache.get_many=features
    return freeze_for_generation(result)


@pytest.mark.parametrize('policy',[None,'a','c'])
@pytest.mark.parametrize('batch_size',[1,2,4])
def test_actual_smallthinker_cpu_batch_matches_native_single_tokens(policy,batch_size):
    research=model(policy);rows=records()
    expected=[generate_prediction(research,row,max_new_tokens=6) for row in rows]
    assert len({tuple(row['generated_token_ids']) for row in expected})>1
    actual=[]
    for start in range(0,len(rows),batch_size):
        actual.extend(generate_batch_predictions(research,rows[start:start+batch_size],max_new_tokens=6))
    for a,b in zip(actual,expected):
        assert a['generated_token_ids']==b['generated_token_ids']
        for field in ('prediction','generated_tokens','prompt_sha256','prefill_tokens','visual_tokens','logits_to_keep'):
            assert a[field]==b[field]
        assert a['logits_to_keep']==1


def test_left_padding_positions_visual_order_cache_hashes_and_native_prefill_logits():
    research=model('a');rows=records();prepared=build_batch_generation_inputs(research,rows,max_new_tokens=5)
    assert len(set(row['prefill_tokens'] for row in prepared['rows']))>1
    for index,(record,row) in enumerate(zip(rows,prepared['rows'])):
        length=row['prefill_tokens'];offset=prepared['padded_prefill_tokens']-length
        assert not prepared['attention_mask'][index,:offset].any()
        assert not prepared['modality_mask'][index,:offset].any()
        assert prepared['position_ids'][index,offset:].tolist()==list(range(length))
        assert torch.equal(prepared['inputs_embeds'][index,offset:],row['inputs_embeds'][0])
    assert research.cache.calls==[(['first.png'],{'image_hashes':['first']}),(['second.png'],{'image_hashes':['second']})]
    with torch.no_grad():
        set_modality_mask(research.base,prepared['modality_mask'],prepared['attention_mask'])
        batched=research.base(inputs_embeds=prepared['inputs_embeds'],attention_mask=prepared['attention_mask'],
            position_ids=prepared['position_ids'],use_cache=False,return_dict=True).logits
        for index,row in enumerate(prepared['rows']):
            set_modality_mask(research.base,row['modality_mask'],row['attention_mask'])
            single=research.base(inputs_embeds=row['inputs_embeds'],attention_mask=row['attention_mask'],
                use_cache=False,return_dict=True).logits
            torch.testing.assert_close(batched[index,-row['prefill_tokens']:],single[0],atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('policy',['a','c'])
def test_actual_cached_batch_masks_prefill_padding_and_generated_text(policy):
    research=model(policy);prepared=build_batch_generation_inputs(research,records(),max_new_tokens=4)
    observed=[];wrapped=research.wrappers[0]
    def capture(module,args):
        hidden=args[1]
        observed.append((tuple(hidden.shape),module._resolve_one_mask(module._modality_mask,hidden,default=False).clone(),
                         module._resolve_one_mask(module._attention_mask,hidden,default=True).clone()))
    hook=wrapped.register_forward_pre_hook(capture)
    try:generate_batch_predictions(research,records(),max_new_tokens=4)
    finally:hook.remove()
    assert len(observed)==4
    assert observed[0][0][:2]==prepared['attention_mask'].shape
    assert torch.equal(observed[0][1],prepared['modality_mask'].flatten())
    assert torch.equal(observed[0][2],prepared['attention_mask'].flatten())
    for shape,visual,valid in observed[1:]:
        assert shape[:2]==(4,1)
        assert not visual.any()
        assert valid.all()


def test_batch_is_gold_independent_does_not_mutate_records_or_reorder_images():
    class Guard(dict):
        def get(self,key,*args):
            if key in ('answer','answers','label'):raise AssertionError('Gold read')
            return super().get(key,*args)
        def __getitem__(self,key):
            if key in ('answer','answers','label'):raise AssertionError('Gold read')
            return super().__getitem__(key)
    research=model('c');rows=records();before=copy.deepcopy(rows)
    first=generate_batch_predictions(research,[Guard(row) for row in rows],max_new_tokens=4)
    changed=copy.deepcopy(rows)
    for row in changed:row.update(answer='different',answers=['different'])
    second=generate_batch_predictions(research,changed,max_new_tokens=4)
    assert [row['generated_token_ids'] for row in first]==[row['generated_token_ids'] for row in second]
    assert rows==before
    assert [paths for paths,_ in research.cache.calls]==[['first.png'],['second.png'],['first.png'],['second.png']]


def test_per_row_eos_trimming_includes_eos_and_removes_only_following_pad():
    research=model();research.base.generation_config.eos_token_id=[2,9]
    def generate(**kwargs):
        assert kwargs['use_cache'] and kwargs['num_beams']==1 and kwargs['logits_to_keep']==1
        # Include dummy input IDs to check score-count slicing.
        return SimpleNamespace(sequences=torch.tensor([[7,7,2,0,0],[7,7,8,9,0],[7,7,0,8,7],[7,7,9,0,0]]),
            scores=tuple(torch.zeros(4,32) for _ in range(3)))
    with patch.object(research.base,'generate',side_effect=generate):
        actual=generate_batch_predictions(research,records(),max_new_tokens=3)
    assert [row['generated_token_ids'] for row in actual]==[[2],[8,9],[0,8,7],[9]]
    assert [row['generated_tokens'] for row in actual]==[1,2,3,1]
    assert trim_generated_ids([0,8,7],set())==[0,8,7]


@pytest.mark.parametrize('policy',[None,'a','c'])
@pytest.mark.parametrize('batch_size',[2,4])
def test_real_native_per_row_early_eos_lengths_match_separate_generation(policy,batch_size):
    research=model(policy);research.base.generation_config.eos_token_id=[12,2]
    expected=[generate_prediction(research,row,max_new_tokens=6) for row in records()]
    assert len({row['generated_tokens'] for row in expected})>1
    actual=[]
    for offset in range(0,4,batch_size):
        actual.extend(generate_batch_predictions(research,records()[offset:offset+batch_size],max_new_tokens=6))
    assert [row['generated_token_ids'] for row in actual]==[row['generated_token_ids'] for row in expected]
    assert [row['generated_tokens'] for row in actual]==[row['generated_tokens'] for row in expected]


def test_context_empty_batch_and_invalid_native_logits_fail():
    research=model()
    with pytest.raises(ValueError,match='nonempty'):build_batch_generation_inputs(research,[])
    research.args.max_length=3
    with pytest.raises(ValueError,match='context'):build_batch_generation_inputs(research,records(),max_new_tokens=4)
    research.args.max_length=64
    bad=SimpleNamespace(sequences=torch.ones(4,1,dtype=torch.long),scores=(torch.full((4,32),float('nan')),))
    with patch.object(research.base,'generate',return_value=bad):
        with pytest.raises(ValueError,match='invalid logits'):generate_batch_predictions(research,records(),max_new_tokens=1)


@pytest.mark.parametrize('batch_size',[1,2,4])
def test_real_native_gpt_oss_cpu_keeps_final_prefix_and_matches_single(batch_size):
    from transformers import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
    torch.manual_seed(733)
    research=TinyResearchModel()
    cfg=GptOssConfig(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,num_local_experts=4,
        num_experts_per_tok=2,layer_types=['full_attention'],pad_token_id=0,bos_token_id=1,eos_token_id=2,
        max_position_embeddings=128,rope_parameters={'rope_type':'default','rope_theta':10000.},attention_dropout=0.)
    cfg._attn_implementation='eager'
    research.base=GptOssForCausalLM(cfg);research.llm=research.base
    research.projector=nn.Linear(3,16,bias=False)
    research.base.generation_config.eos_token_id=None
    freeze_for_generation(research)
    prepared=build_batch_generation_inputs(research,records(),max_new_tokens=4)
    assert all(row['prompt'].endswith('<|channel|>final<|message|>') for row in prepared['rows'])
    expected=[generate_prediction(research,row,max_new_tokens=4) for row in records()]
    actual=[]
    for offset in range(0,4,batch_size):
        actual.extend(generate_batch_predictions(research,records()[offset:offset+batch_size],max_new_tokens=4))
    assert [row['generated_token_ids'] for row in actual]==[row['generated_token_ids'] for row in expected]
    assert all(row['logits_to_keep']==1 for row in actual)


def test_optional_gpu_probe_is_disabled_before_any_cuda_or_model_work(tmp_path):
    with patch.object(torch.cuda,'is_available',side_effect=AssertionError('No CUDA work allowed')):
        with pytest.raises(RuntimeError,match='probe disabled'):
            benchmark_main(['--checkpoint',str(tmp_path/'checkpoint'),'--manifest',str(tmp_path/'manifest'),
                            '--output',str(tmp_path/'probe.json')])
    assert not (tmp_path/'probe.json').exists()


def test_gpu_probe_refuses_shared_process_without_signalling_it():
    with patch('subprocess.run',return_value=SimpleNamespace(stdout='999999\n')),patch('os.kill') as kill:
        with pytest.raises(RuntimeError,match='coordinated idle GPU'):require_idle_gpu()
    kill.assert_not_called()

