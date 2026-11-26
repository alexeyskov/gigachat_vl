"""Exercise the read-only audit on a real native conv/conv/attention LFM model."""
import json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import pytest
import torch
from safetensors.torch import save_file
from transformers import Lfm2MoeConfig,Lfm2MoeForCausalLM
from scripts.audit_research_budget18_updates_v1 import snapshot,compare,read


@pytest.mark.parametrize('changed',[True,False])
def test_full_audit_handles_native_attention_after_conv_layers(tmp_path,monkeypatch,changed):
    monkeypatch.setenv('HF_HOME',str(tmp_path/'hf'))
    config=Lfm2MoeConfig(vocab_size=32,hidden_size=8,intermediate_size=16,
        moe_intermediate_size=4,num_hidden_layers=3,num_dense_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=4,
        num_experts=4,num_experts_per_tok=2,
        layer_types=['conv','conv','full_attention'],pad_token_id=0)
    model=Lfm2MoeForCausalLM(config)
    original={k:v.detach().clone().contiguous() for k,v in model.state_dict().items()}
    attention=[k for k,v in original.items() if '.self_attn.' in k and v.ndim==2]
    assert len(attention)>=3 and all('.layers.2.' in k for k in attention)
    assert not any('.layers.0.self_attn.' in k for k in original)
    directory=tmp_path/'hf/hub/models--fixture--native-lfm/snapshots/pinned'
    directory.mkdir(parents=True)
    save_file(original,str(directory/'model.safetensors'))
    a0=tmp_path/'A0';sft=tmp_path/'SFT';output=tmp_path/'audit/report.json'
    for run in (a0,sft):
        (run/'best').mkdir(parents=True)
        (run/'resolved_config.json').write_text(json.dumps({'fixture':run.name}))
    projector={'projector.0.weight':torch.ones(8,4),'projector.0.bias':torch.zeros(8),
               'projector.2.weight':torch.eye(8),'projector.2.bias':torch.zeros(8)}
    torch.save({'model':projector,'step':64},a0/'best/state.pt')
    learned={'llm.'+k:v.clone() for k,v in original.items()}
    if changed:
        for key in attention[:3]:learned['llm.'+key].add_(.125)
    learned.update(projector)
    torch.save({'model':learned,'step':56,'seen':3529,
                'model_ref':{'repo':'fixture/native-lfm','revision':'pinned'}},sft/'best/state.pt')
    output.parent.mkdir()
    args={'initial_checkpoint':str(a0/'best/state.pt'),'recipe':'4f'}
    initial=output.parent/'initial_tensors.json'
    snapshot(sft,args,initial)
    if not changed:
        with pytest.raises(ValueError,match='Full-base sample updates not established'):
            compare(sft,args,read(initial),output)
        assert not output.exists()
        return
    compare(sft,args,read(initial),output)
    report=read(output)
    assert report['status']=='passed'
    assert report['common_tensor_groups']['projector']['changed_tensors']==0
    assert len(report['full_base_update_samples'])==3
    assert all('.layers.2.self_attn.' in r['name'] and r['changed_elements']==r['elements']
               for r in report['full_base_update_samples'])
