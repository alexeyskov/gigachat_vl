from types import SimpleNamespace
from src.eval.research_giga_eot_v1 import EOT_ID,EOT_TEXT,MODEL,REVISION,terminated_prefix,install_generation_stop


class Tokenizer:
    def encode(self,text,**kwargs):assert text==EOT_TEXT;return [EOT_ID]
    def apply_chat_template(self,*args,**kwargs):return 'assistant Answer.'+EOT_TEXT
    def decode(self,ids,**kwargs):return ''.join({10:'Answer: B',11:'assistant continued',EOT_ID:''}.get(i,'') for i in ids)


def test_trim_only_at_native_marker_without_answer_mining():
    result=terminated_prefix([10,EOT_ID,11],Tokenizer())
    assert result['prediction']=='Answer: B' and result['generated_token_ids']==[10]
    assert result['native_turn_end_found']
    assert terminated_prefix([10,11],Tokenizer())['prediction']=='Answer: Bassistant continued'


def test_generate_adds_native_turn_end_and_preserves_explicit_eos():
    calls=[];base=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2),
        generate=lambda **kwargs:calls.append(kwargs))
    model=SimpleNamespace(model_ref={'repo':MODEL,'revision':REVISION},tokenizer=Tokenizer(),base=base)
    install_generation_stop(model);base.generate(max_new_tokens=64)
    assert calls[-1]['eos_token_id']==[2,EOT_ID]
    base.generate(eos_token_id=[7,2],max_new_tokens=64)
    assert calls[-1]['eos_token_id']==[7,2,EOT_ID]
    assert base.generation_config.eos_token_id==2


def test_actual_transformers_generation_stops_at_native_marker_with_identical_prefix():
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel,LogitsProcessor,LogitsProcessorList
    class Forced(LogitsProcessor):
        def __call__(self,ids,scores):
            token=10 if ids.shape[1]==1 else EOT_ID if ids.shape[1]==2 else 11
            scores.fill_(-float('inf'));scores[:,token]=0;return scores
    torch.manual_seed(42)
    base=GPT2LMHeadModel(GPT2Config(vocab_size=128032,n_embd=8,n_layer=1,n_head=1,
        bos_token_id=1,eos_token_id=2,pad_token_id=2)).eval()
    inputs=torch.tensor([[1]]);kwargs=dict(max_new_tokens=4,do_sample=False,logits_processor=LogitsProcessorList([Forced()]))
    old=base.generate(inputs,**kwargs)
    install_generation_stop(SimpleNamespace(model_ref={'repo':MODEL,'revision':REVISION},tokenizer=Tokenizer(),base=base))
    new=base.generate(inputs,**kwargs)
    assert old.tolist()==[[1,10,EOT_ID,11,11]]
    assert new.tolist()==[[1,10,EOT_ID]] and torch.equal(old[:,:new.shape[1]],new)


def test_eval_clone_scope_binds_protocol_and_restores_original_factory(monkeypatch):
    from src.utils import research_stock_determinism as stock
    from src.utils.research_giga_eval_stop_v1 import corrected_evaluation_clones
    evaluator=SimpleNamespace(_bind_stock_protocol=lambda x:dict(x,numerical_profile='native'))
    evaluator.main=lambda:evaluator._bind_stock_protocol({'greedy':True})
    native=lambda *a,**k:evaluator
    monkeypatch.setattr(stock,'_clone',native)
    with corrected_evaluation_clones():
        private=stock._clone('scripts/eval_research_checkpoint.py');result=private.main()
        assert result['numerical_profile']=='native' and result['giga_native_turn_stop']['assistant_turn_end_id']==EOT_ID
    assert stock._clone is native
