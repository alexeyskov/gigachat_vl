"""Gold independence, actual tiny native generation and evaluation provenance."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from src.eval.research_inference import (
    _validate_learned_state, build_generation_inputs, freeze_for_generation,
    generate_prediction, load_research_checkpoint,
)
from src.model.research_moe import install_research_moe, set_modality_mask
from src.model.smallthinker import SmallThinkerConfig, SmallThinkerForCausalLM
import scripts.eval_research_checkpoint as cli


class TinyCache:
    fingerprint = "tiny"

    def __init__(self):
        self.calls = []

    def get_many(self, paths, encode, **kwargs):
        self.calls.append((list(paths),kwargs))
        return [torch.tensor([[1.,2.,3.],[4.,5.,6.]]) for _ in paths]


class TinyResearchModel(nn.Module):
    def __init__(self, wrapped=False):
        super().__init__()
        cfg = SmallThinkerConfig(vocab_size=32, hidden_size=8,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, moe_ffn_hidden_size=12, moe_num_primary_experts=4,
            moe_num_active_primary_experts=2, pad_token_id=0, bos_token_id=1,
            eos_token_id=2, max_position_embeddings=64)
        self.base = SmallThinkerForCausalLM(cfg)
        self.llm = self.base
        self.wrappers = install_research_moe(self.base,3,"c") if wrapped else []
        self.base.generation_config.eos_token_id = None
        vocabulary = {f"token{i}":i for i in range(32)}
        for index,word in {0:"<pad>",1:"<bos>",2:"<eos>",3:"<unk>",4:"user",5:"assistant",6:"question",7:"red",8:"blue"}.items():
            del vocabulary[f"token{index}"]
            vocabulary[word] = index
        tokenizer = Tokenizer(WordLevel(vocabulary,unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
            pad_token="<pad>",bos_token="<bos>",eos_token="<eos>",unk_token="<unk>",
            chat_template="{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}")
        self.projector = nn.Linear(3,8,bias=False)
        self.args = SimpleNamespace(max_length=64)
        self.cache = TinyCache()

    def encode(self, paths):
        raise AssertionError("The test cache is a hit; encoder should not be called.")


def record():
    return {"record_id":"one","question":"question red", "image_paths":["image.png"],
            "image_hashes":["hash"],"answers":["GOLD_NEVER_IN_PROMPT"],"answer":"ALSO_GOLD"}


def test_visual_feature_order_and_gold_independent_prompt():
    torch.manual_seed(35)
    model = freeze_for_generation(TinyResearchModel())
    first = build_generation_inputs(model,record(),max_new_tokens=4)
    changed = record();changed["answers"]=["different"];changed["answer"]="different"
    second = build_generation_inputs(model,changed,max_new_tokens=4)
    assert first["prompt"]==second["prompt"]
    assert "GOLD" not in first["prompt"]
    torch.testing.assert_close(first["inputs_embeds"],second["inputs_embeds"])
    begin = len(first["prefix_token_ids"])
    expected = model.projector(torch.tensor([[1.,2.,3.],[4.,5.,6.]]))
    torch.testing.assert_close(first["inputs_embeds"][0,begin:begin+2],expected)
    assert first["modality_mask"][0].nonzero().flatten().tolist()==[begin,begin+1]
    assert first["attention_mask"].all()
    assert model.cache.calls[-1]==(["image.png"],{"image_hashes":["hash"]})


def test_no_gold_fields_are_accessed_and_generation_is_deterministic():
    class Guard(dict):
        def __getitem__(self,key):
            if key in ("answer","answers","label"):
                raise AssertionError("Gold field accessed during generation")
            return super().__getitem__(key)
        def get(self,key,*args):
            if key in ("answer","answers","label"):
                raise AssertionError("Gold field accessed during generation")
            return super().get(key,*args)
    model = freeze_for_generation(TinyResearchModel())
    first = generate_prediction(model,Guard(record()),max_new_tokens=4)
    second = generate_prediction(model,Guard(record()),max_new_tokens=4)
    assert first["generated_token_ids"]==second["generated_token_ids"]
    assert first["generated_tokens"]==4
    assert first["logits_to_keep"]==1
    assert not model.base.training
    assert all(not p.requires_grad for p in model.parameters())
    assert next(model.base.parameters()).dtype==torch.float32


def test_actual_native_kv_generation_matches_full_prefill_reference_and_masks():
    model = freeze_for_generation(TinyResearchModel(wrapped=True))
    observed = []
    wrapped = model.wrappers[0]
    def capture(module,args):
        current = module._resolve_one_mask(module._modality_mask,args[1],default=False)
        observed.append((args[1].shape[1],int(current.sum())))
    hook = wrapped.register_forward_pre_hook(capture)
    generated = generate_prediction(model,record(),max_new_tokens=4)
    hook.remove()
    assert observed[0][1]==2
    assert len(observed)==4
    assert all(length==1 and visual==0 for length,visual in observed[1:])
    prepared = build_generation_inputs(model,record(),max_new_tokens=4)
    embeds = prepared["inputs_embeds"]
    modality = prepared["modality_mask"]
    expected = []
    with torch.no_grad():
        for _ in range(4):
            attention = torch.ones(embeds.shape[:2],dtype=torch.bool)
            set_modality_mask(model.base,modality,attention)
            result = model.base(inputs_embeds=embeds,attention_mask=attention,use_cache=False,
                                return_dict=True,logits_to_keep=1)
            token = result.logits[:,-1].argmax(-1)
            expected.append(int(token))
            embeds = torch.cat((embeds,model.base.get_input_embeddings()(token)[:,None,:]),dim=1)
            modality = torch.cat((modality,torch.zeros((1,1),dtype=torch.bool)),dim=1)
    assert generated["generated_token_ids"]==expected


def test_context_limit_hash_count_and_multiple_images_fail_explicitly():
    model = TinyResearchModel()
    model.args.max_length = 3
    with pytest.raises(ValueError,match="context"):
        build_generation_inputs(model,record(),max_new_tokens=4)
    changed = record();changed["image_hashes"]=[]
    with pytest.raises(ValueError,match="path/hash"):
        build_generation_inputs(model,changed)
    changed = record();changed["image_paths"]*=2
    with pytest.raises(NotImplementedError):
        build_generation_inputs(model,changed)


def test_state_validation_rejects_unknown_shapes_and_nonfinite_values():
    model = TinyResearchModel()
    model.base.requires_grad_(False)
    state = {"projector.weight":model.projector.weight.detach().clone()}
    _validate_learned_state(model,state)
    with pytest.raises(ValueError,match="Unrecognized"):
        _validate_learned_state(model,dict(state,fake=torch.ones(1)))
    with pytest.raises(ValueError,match="shape/type"):
        _validate_learned_state(model,{"projector.weight":torch.ones(4,4)})
    state["projector.weight"][0,0]=float('nan')
    with pytest.raises(ValueError,match="Nonfinite"):
        _validate_learned_state(model,state)
    model.base.requires_grad_(True)
    with pytest.raises(ValueError,match="omits learned"):
        _validate_learned_state(model,{"projector.weight":torch.zeros_like(model.projector.weight)})


def test_checkpoint_loader_pins_refs_restores_and_freezes(tmp_path,monkeypatch):
    import scripts.run_research_stage as runner
    import huggingface_hub
    original = TinyResearchModel()
    references = [{"repo":"example/model","revision":"a"*40},
                  {"repo":"example/vision","revision":"b"*40}]
    config = {"model":"example/model","vision":"example/vision","recipe":"4f","stage":"A0"}
    path = tmp_path/"state.pt"
    torch.save({"config":config,"model":original.state_dict(),"model_ref":references[0],
                "vision_ref":references[1],"step":7,"cache_fingerprint":"tiny"},path)
    requests = []
    def snapshot(repo,revision,local_files_only,**kwargs):
        assert local_files_only
        requests.append((repo,revision))
        return "/already-cached"
    monkeypatch.setattr(huggingface_hub,"snapshot_download",snapshot)
    def factory(args):
        runner.resolve_model(args.model);runner.resolve_model(args.vision)
        return TinyResearchModel()
    monkeypatch.setattr(runner,"ResearchModel",factory)
    restored,provenance = load_research_checkpoint(path)
    assert requests==[(r["repo"],r["revision"]) for r in references]
    assert provenance["training_step"]==7
    torch.testing.assert_close(restored.projector.weight,original.projector.weight)
    assert not restored.training
    assert all(not p.requires_grad for p in restored.parameters())


def test_occupied_gpu_refused(monkeypatch):
    monkeypatch.setattr(cli.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0,stdout="999999, 1024\n",stderr=""))
    with pytest.raises(RuntimeError,match="occupied"):
        cli.assert_gpu_idle()


def cli_fixture(tmp_path,monkeypatch):
    run = tmp_path/"run";best=run/"best";best.mkdir(parents=True)
    (best/"state.pt").write_bytes(b"controlled fake checkpoint for CLI tests")
    groups = ["general_vqa","ocr_documents","charts_diagrams","knowledge_reasoning"]
    records = [{"record_id":str(i),"question":"question","answers":["red"],
                "scorer":"exact_match","task_group":group,"source":{"repo":"test"}}
               for i,group in enumerate(groups)]
    manifest = tmp_path/"eval.jsonl"
    manifest.write_text("".join(json.dumps(r)+"\n" for r in records))
    monkeypatch.setattr(cli,"assert_gpu_idle",lambda *a:None)
    monkeypatch.setattr(cli,"load_research_checkpoint",lambda *a,**k:(None,{"tiny":True}))
    monkeypatch.setattr(cli,"generate_prediction",lambda *a,**k:{"prediction":"red","generated_token_ids":[7],"generated_tokens":1})
    # Other agents may edit source files during these unit tests; pin a synthetic
    # code fingerprint so this tests CLI resume semantics, not workspace timing.
    monkeypatch.setattr(cli,"sha256_file",lambda *a,**k:"a"*64)
    return ["--run",str(run),"--manifest",str(manifest),"--output",str(tmp_path/"output")]


def test_cli_complete_macro_resume_and_mixed_protocol_rejection(tmp_path,monkeypatch):
    argv = cli_fixture(tmp_path,monkeypatch)
    first = cli.main(argv)
    assert first["complete_manifest"]
    assert first["group_macro_score"]==100
    resumed = cli.main(argv+["--resume"])
    assert resumed==first
    with pytest.raises(ValueError,match="differs"):
        cli.main(argv+["--resume","--max-new-tokens","32"])


def test_partial_cli_never_claims_four_group_macro(tmp_path,monkeypatch):
    argv = cli_fixture(tmp_path,monkeypatch)
    result = cli.main(argv+["--max-records","1"])
    assert not result["complete_manifest"]
    assert result["group_macro_score"] is None
    assert not (tmp_path/"output/summary.json").exists()


def test_resume_can_repair_only_unfinished_final_json_append(tmp_path):
    path = tmp_path/"predictions.jsonl"
    path.write_bytes(b'{"record_id":"ok"}\n{"unfinish')
    assert cli._read_predictions(path)==[{"record_id":"ok"}]
    assert path.read_bytes()==b'{"record_id":"ok"}\n'
    path.write_bytes(b'{"record_id":"complete_no_newline"}')
    assert cli._read_predictions(path)==[{"record_id":"complete_no_newline"}]
    assert path.read_bytes().endswith(b"\n")
    path.write_bytes(b'{"invalid}\n')
    with pytest.raises(ValueError,match="Corrupt"):
        cli._read_predictions(path)
