"""Behavioral checks for benchmark inference boundaries, resume and admission."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.eval.research_benchmark_model_v1 import prepare_interleaved
from src.eval.vlm_benchmark_runtime_v1 import choose,journal_rows,selected_rows,evaluate_official


def model():
    class Tokenizer:
        def apply_chat_template(self,messages,**_):return '<u>'+messages[0]['content']+'<a>'
        def encode(self,text,**_):return [ord(c)%31 for c in text]
    class Cache:
        def get_many(self,paths,encode):return [torch.tensor([[10.,10.],[11.,11.]]) if p=='a' else torch.tensor([[20.,20.]]) for p in paths]
    embedding=nn.Embedding(31,2);projector=nn.Linear(2,2,bias=False)
    with torch.no_grad():projector.weight.copy_(torch.eye(2))
    return SimpleNamespace(tokenizer=Tokenizer(),base=SimpleNamespace(get_input_embeddings=lambda:embedding,
        config=SimpleNamespace(max_position_embeddings=1024)),cache=Cache(),projector=projector,encode=None,
        args=SimpleNamespace(max_length=1024))


def test_interleaved_images_preserve_order_and_modality():
    value=prepare_interleaved(model(),[{'type':'text','value':'before'}, {'type':'image','value':'a'},
        {'type':'text','value':'between'},{'type':'image','value':'b'},{'type':'text','value':'after'}],64)
    visual=value['inputs_embeds'][0,value['modality_mask'][0]]
    assert visual.tolist()==[[10.,10.],[11.,11.],[20.,20.]]
    positions=value['modality_mask'][0].nonzero().flatten().tolist()
    assert positions[1]+1<positions[2]  # intervening text is not treated as visual
    assert value['visual_token_counts']==[2,1]
    assert value['attention_mask'].shape==value['modality_mask'].shape


def test_no_silent_image_drop_or_context_truncation():
    m=model();m.args.max_length=8
    with pytest.raises(ValueError,match='context exceeded'):
        prepare_interleaved(m,[{'type':'image','value':'a'},{'type':'text','value':'long question'}],64)
    with pytest.raises(ValueError,match='marker collision'):
        prepare_interleaved(model(),[{'type':'text','value':'[BENCHMARK_IMAGE_0]'}],1)


def test_single_image_matches_frozen_research_generation_inputs():
    from src.eval.research_inference import build_generation_inputs
    m=model();message=[{'type':'image','value':'a'},{'type':'text','value':'Question?'}]
    actual=prepare_interleaved(m,message,64)
    expected=build_generation_inputs(m,dict(question='Question?',image_paths=['a']),max_new_tokens=64)
    for key in ('inputs_embeds','attention_mask','modality_mask'):
        assert torch.equal(actual[key],expected[key])
    assert actual['prefill_tokens']==expected['prefill_tokens']


def test_resume_rejects_reordered_inputs_and_changed_protocol(tmp_path):
    path=tmp_path/'predictions.jsonl'
    path.write_text(json.dumps(dict(index='x',ordinal=2,protocol_sha256='pin',prediction='A'))+'\n')
    assert len(journal_rows(path,[2],['a','b','x'],'pin'))==1
    with pytest.raises(ValueError):journal_rows(path,[1],['a','b','x'],'pin')
    with pytest.raises(ValueError):journal_rows(path,[2],['a','b','x'],'new')
    path.write_text(path.read_text().rstrip())
    with pytest.raises(ValueError,match='Partial'):journal_rows(path,[2],['a','b','x'],'pin')


def test_judge_required_never_silently_exact_matches():
    called=[];dataset=SimpleNamespace(evaluate=lambda *_a,**_kw:called.append(True))
    with pytest.raises(ValueError,match='requires explicit'):
        evaluate_official(dataset,'none',choose(names=['MathVista_MINI'])[0])
    assert called==[]


def test_full_paired_benchmark_cannot_be_randomly_cut():
    # Paired MME/Hallusion metrics require complete groups.
    with pytest.raises(ValueError,match='Paired'):
        selected_rows(SimpleNamespace(data=__import__('pandas').DataFrame({'index':[1,2]})),1,42,{'paired':True})


def test_orphan_guardian_resumes_only_owned_cpu_controller(tmp_path):
    from scripts.run_qwen25_native_baseline_parallel_v1 import q
    controller=subprocess.Popen(['/usr/bin/sleep','60'],start_new_session=True)
    owner=subprocess.Popen(['/usr/bin/sleep','60'],start_new_session=True)
    guard=None;path=tmp_path/'lease.json'
    try:
        q.atomic(path,dict(active=True,controller=q.proc(controller.pid),coordinator=q.proc(owner.pid)))
        os.kill(controller.pid,signal.SIGSTOP)
        guard=subprocess.Popen([sys.executable,'scripts/run_qwen25_native_baseline_parallel_v1.py','--guardian',str(path)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
        owner.kill();owner.wait(timeout=5);guard.wait(timeout=12)
        assert q.proc(controller.pid)['state']!='T'
        lease=json.loads(path.read_bytes());assert not lease['active'] and lease['recovered_by_guardian']
    finally:
        for child in (guard,owner,controller):
            if child and child.poll() is None:child.kill();child.wait(timeout=5)


def test_completed_zombie_at_training_boundary_is_not_a_new_worker(monkeypatch):
    from scripts import run_qwen25_native_baseline_parallel_v1 as coordinator
    monkeypatch.setattr(coordinator.q,'supervisor_info',lambda _:dict(statename='RUNNING',pid=10))
    parent=dict(pid=10,birth=1,ppid=1,state='S',argv=['queue'])
    zombie=dict(pid=20,birth=2,ppid=10,state='Z',argv=[])
    monkeypatch.setattr(coordinator.q,'proc',lambda pid:parent if pid==10 else zombie)
    monkeypatch.setattr(coordinator.q,'read',lambda _:dict(active_job='old',jobs={'old':{'train_process':{'pid':20,'status':'running'}}}))
    info,reason=coordinator.target()
    assert info is None and reason=='waiting_for_owned_trainer'


def test_real_transformers_conversion_preserves_nested_native_vision():
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    from transformers.integrations.bitsandbytes import replace_with_bnb_linear
    from src.eval.research_hf_vlm_v1 import NATIVE_UNQUANTIZED_PATTERNS
    with torch.device('meta'):
        m=nn.Module();m.model=nn.Module()
        m.model.visual=nn.Sequential(nn.Linear(4,4,dtype=torch.bfloat16))
        m.model.language_model=nn.Sequential(nn.Linear(4,4,dtype=torch.bfloat16))
        m.lm_head=nn.Linear(4,8,dtype=torch.bfloat16)
    replace_with_bnb_linear(m,modules_to_not_convert=NATIVE_UNQUANTIZED_PATTERNS,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True))
    assert type(m.model.visual[0]) is nn.Linear
    assert type(m.lm_head) is nn.Linear
    assert isinstance(m.model.language_model[0],bnb.nn.Linear4bit)
