"""F32 scope, untouched4-bit storage/metadata and native CPU no-autocast inputs."""
from types import SimpleNamespace
import copy

import pytest
import torch
from torch import nn

import scripts.countercheck_research_batch4_f32 as f32
import scripts.diagnose_research_batch4_logits as bf16
from tests.test_research_inference import TinyResearchModel
from src.eval.research_inference import freeze_for_generation


class FakeQuant(nn.Module):
    def __init__(self):
        super().__init__()
        # Floating quant-storage representation must also be excluded from casts.
        self.weight=nn.Parameter(torch.tensor([[1.],[2.]],dtype=torch.bfloat16),requires_grad=False)
        scale=torch.tensor([.25,.5],dtype=torch.bfloat16)
        self.weight.quant_state=SimpleNamespace(absmax=scale,code=torch.tensor([-1.,0.,1.]),dtype=torch.bfloat16,
                                                shape=(2,4),blocksize=64,quant_type='nf4',nested=False,state2=None)
        self.weight.quant_type='nf4';self.weight.bnb_quantized=True;self.compute_dtype=torch.bfloat16;self.compute_type_is_set=False
        self.register_buffer('quant_scale_alias',scale)


def fake_model():
    llm=nn.Module();llm.quant=FakeQuant();llm.linear=nn.Linear(4,4).to(torch.bfloat16)
    llm.register_buffer('ordinary_buffer',torch.ones(2,dtype=torch.float16))
    return SimpleNamespace(llm=llm,projector=nn.Linear(3,4).to(torch.float16),
                           requires_grad_=lambda _:None)


class Container(nn.Module):
    def __init__(self):
        super().__init__();self.llm=nn.Module();self.llm.quant=FakeQuant();self.llm.linear=nn.Linear(4,4).to(torch.bfloat16)
        self.llm.register_buffer('ordinary_buffer',torch.ones(2,dtype=torch.float16))
        self.projector=nn.Linear(3,4).to(torch.float16)


def test_only_ordinary_parameters_and_buffers_promoted_quant_bits_and_metadata_identical():
    model=Container();before,ids=f32.quantization_snapshot(model)
    weight=model.llm.quant.weight;metadata=weight.quant_state;scale=model.llm.quant.quant_scale_alias
    report=f32.promote_ordinary_f32(model,ids);after,_=f32.quantization_snapshot(model)
    assert before==after and before['quantized_parameter_count']==1
    assert model.llm.quant.weight is weight and weight.dtype==torch.bfloat16
    assert weight.quant_state is metadata and metadata.dtype==torch.bfloat16
    assert model.llm.quant.quant_scale_alias is scale and scale.dtype==torch.bfloat16
    assert model.llm.linear.weight.dtype==model.llm.ordinary_buffer.dtype==model.projector.weight.dtype==torch.float32
    assert model.llm.quant.compute_dtype==torch.float32 and model.llm.quant.compute_type_is_set
    assert report['ordinary_tensor_change_count']>0 and len(report['runtime_nf4_compute_changes'])==1
    assert f32.ordinary_floating_dtypes(model,ids)=={'torch.float32':sum(p.numel() for n,p in model.named_parameters() if not f32.quantized_parameter(p))+2}
    assert all(not p.requires_grad for p in model.parameters())


def test_quant_metadata_tampering_is_detected_and_unknown_type_blocks():
    model=Container();before,_=f32.quantization_snapshot(model)
    model.llm.quant.weight.quant_state.code[0]=42
    after,_=f32.quantization_snapshot(model);assert before!=after
    model.llm.quant.weight.quant_state.unsupported=object()
    with pytest.raises(f32.CountercheckBlocked,match='metadata type'):f32.quantization_snapshot(model)


def test_ordinary_tied_alias_preserved_and_float64_not_silently_demoted():
    model=Container();model.llm.other=nn.Linear(4,4,bias=False).to(torch.bfloat16)
    model.llm.other.weight=model.llm.linear.weight
    _,ids=f32.quantization_snapshot(model);f32.promote_ordinary_f32(model,ids)
    assert model.llm.other.weight is model.llm.linear.weight and model.llm.other.weight.dtype==torch.float32
    model.llm.extra=nn.Parameter(torch.ones(2,dtype=torch.float64),requires_grad=False)
    with pytest.raises(f32.CountercheckBlocked,match='ordinary parameter dtype'):f32.promote_ordinary_f32(model,ids)


def records():
    return [{'record_id':str(i),'question':'question red '+('blue '*i),'image_paths':['cached.png'],'image_hashes':['hash'],
             'answers':['GOLD_MUST_NOT_ENTER_THE_PROMPT']} for i in range(4)]


def test_actual_native_tiny_cpu_f32_no_autocast_same_positions_and_cache_identity():
    torch.set_num_threads(2);torch.manual_seed(29)
    model=freeze_for_generation(TinyResearchModel(wrapped=True));model.llm.to(dtype=torch.bfloat16);model.projector.to(dtype=torch.bfloat16)
    rows=records();old=bf16.capture_generation(model,rows,batched=True,max_new_tokens=4)
    fingerprint=model.cache.fingerprint;before,ids=f32.quantization_snapshot(model)
    assert before['quantized_parameter_count']==0 #This fixture tests native math, not actual NF4 CUDA.
    f32.promote_ordinary_f32(model,ids);audit=f32.DtypeAudit(model)
    old_autocast=bf16._autocast
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        result=f32.capture_f32(model,rows,batched=True,max_new_tokens=4)
        assert torch.is_autocast_enabled('cpu')
    observed=audit.finish()
    assert observed and not any(r['autocast_observed'] for r in observed.values())
    assert all(set(r['inputs']+r['outputs'])<= {'torch.float32'} for r in observed.values())
    assert result['raw_logits'].dtype==result['output_scores'].dtype==torch.float32
    assert result['rows']==old['rows'] and model.cache.fingerprint==fingerprint and bf16._autocast is old_autocast
    assert all(call[1]['read_only'] is True for call in model.cache.calls[-4:])
    assert all(len(r)<=4 for r in result['token_ids'])


def test_cache_miss_never_invokes_visual_encoder():
    model=freeze_for_generation(TinyResearchModel())
    class Missing:
        def get_many(self,paths,encode,**kwargs):
            assert kwargs['read_only'];raise KeyError('missing')
    model.cache=Missing()
    with pytest.raises(f32.CountercheckBlocked,match='no encoder fallback'):f32.f32_inputs(model,records()[0],4)


def test_dtype_audit_rejects_accidental_bf16_math():
    model=freeze_for_generation(TinyResearchModel());model.projector.to(torch.bfloat16);audit=f32.DtypeAudit(model)
    model.projector(torch.ones((2,3),dtype=torch.bfloat16))
    with pytest.raises(f32.CountercheckBlocked,match='not F32'):audit.finish()


def test_countercheck_is_opt_in_and_does_not_promise_token_parity(tmp_path):
    with pytest.raises(RuntimeError,match='--allow-gpu'):
        f32.main(['--bf16-report','missing','--manifest','missing','--output',str(tmp_path/'never.json')])
    #No guard-reproduction requirement is applied across different precisions;
    #only order/prompt/positions and preserved quantized base are fixed.
    assert not hasattr(f32,'require_reproduction')
