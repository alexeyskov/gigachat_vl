from pathlib import Path
from types import SimpleNamespace
import sys
import json
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.check_research_backbone_cpu import require_gpu_hidden,conv_state_snapshot,json_ready


def test_cpu_probe_requires_gpu_hidden_before_cuda_calls(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    with pytest.raises(RuntimeError,match='requires CUDA_VISIBLE_DEVICES'):
        require_gpu_hidden(SimpleNamespace())
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False,device_count=lambda:0,is_initialized=lambda:False))
    require_gpu_hidden(fake)
    fake.cuda.device_count=lambda:1
    with pytest.raises(RuntimeError,match='may not allocate'):
        require_gpu_hidden(fake)


def test_native_conv_cache_dict_is_copied_and_shape_checked():
    state=torch.ones(1,4,3)
    cache=SimpleNamespace(layers=[SimpleNamespace(conv_states={0:state}),SimpleNamespace()])
    config=SimpleNamespace(layer_types=['conv','full_attention'],conv_L_cache=3)
    snapshot=conv_state_snapshot(cache,config)
    state.zero_()
    assert snapshot[0].sum()==12
    cache.layers[0].conv_states[0]=torch.ones(1,4,2)
    with pytest.raises(ValueError,match='invalid shape'):
        conv_state_snapshot(cache,config)


def test_loading_info_sets_are_json_serializable():
    info={'missing_keys':set(),'unexpected_keys':{'wrong'},'nested':({'x'},)}
    assert json.loads(json.dumps(json_ready(info)))=={'missing_keys':[],'unexpected_keys':['wrong'],'nested':[['x']]}
