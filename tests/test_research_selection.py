"""Train-derived sampling, honest single-reference metrics and state restoration."""

import copy
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.eval.research_selection import (
    GROUPS, PROTOCOL_NAME, build_internal_validation, evaluate_internal_group_macro,
)


def validation_records(counts=(3,3,3,3)):
    records = []
    answers = {"general_vqa":"red","ocr_documents":"hello",
               "charts_diagrams":"100","knowledge_reasoning":"B"}
    for group,count in zip(GROUPS,counts):
        for index in range(count):
            row = {"record_id":f"{group}-{index}","question":f"question {group}",
                   "answer":answers[group],"image_paths":[f"{group}-{index}.png"],
                   "image_hashes":[f"{group}-{index}-hash"],"task_group":group,
                   "partition":"validation","source":{"repo":"training-package","split":"train"}}
            if group=="knowledge_reasoning":
                row.update(choices=["red","blue"],question="Choose the color. A: red. B: blue.")
            records.append(row)
    return records


class SelectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3,3)
        self.llm = self.base
        self.dropout = nn.Dropout(.1)
        self.frozen = nn.Linear(3,3).requires_grad_(False)
        self.config = SimpleNamespace(use_cache=False,flags={"nested":True})
        self.generation_config = SimpleNamespace(use_cache=False,do_sample=True)
        self.base.config = self.config
        self.base.generation_config = self.generation_config
        self.base.gradient_checkpointing = True
        self.base._gradient_checkpointing_func = object()
        self.base._modality_mask = torch.tensor([[True,False]])
        self.extractor = SimpleNamespace(vision_tower=nn.Linear(3,3).requires_grad_(False).eval(),
                                         vision_projector=None)
        self.train()
        self.dropout.eval()
        self.frozen.eval()


def test_sampling_is_order_independent_balanced_and_does_not_modify_input():
    records = validation_records((7,5,4,3))
    original = copy.deepcopy(records)
    first = build_internal_validation(records,per_group=5,seed=17)
    random.Random(9).shuffle(records)
    second = build_internal_validation(records,per_group=5,seed=17)
    assert first==second
    assert first["protocol"]["selected_per_group"]==3
    assert first["protocol"]["name"]==PROTOCOL_NAME
    assert first["protocol"]["official_benchmark_score"] is False
    assert len(first["records"])==12
    assert sorted(records,key=lambda r:r["record_id"])==sorted(original,key=lambda r:r["record_id"])
    with pytest.raises(ValueError,match="Cannot supply"):
        build_internal_validation(records,per_group=5,require_per_group=True)


def test_missing_groups_rejected_and_text_only_kept_separate():
    records = validation_records((1,1,1,0))
    with pytest.raises(ValueError,match="All four"):
        build_internal_validation(records)
    records = validation_records((1,1,1,1))
    text = {"record_id":"text","question":"text question","answer":"text answer", "image_paths":[],
            "task_group":"text_only","partition":"validation","source":{"split":"train"}}
    records.append(text)
    selection = build_internal_validation(records)
    assert selection["text_records"]==[text]
    assert selection["protocol"]["text_only_records_excluded"]==1
    assert all(row["image_paths"] for row in selection["records"])


@pytest.mark.parametrize("split", ["validation","test","val","unknown",""])
def test_official_or_unproven_source_cannot_select_checkpoints(split):
    records = validation_records()
    records[0]["source"]["split"] = split
    with pytest.raises(ValueError,match="source.split must be train"):
        build_internal_validation(records)


def test_single_teacher_reference_never_fabricates_ten_vqa_answers():
    selection = build_internal_validation(validation_records(),per_group=1)
    mapping = {row["task_group"]:row for row in selection["records"]}
    assert mapping["general_vqa"]["scorer"]=="exact_match"
    assert mapping["general_vqa"]["answers"]==["red"]
    assert mapping["ocr_documents"]["scorer"]=="anls"
    assert mapping["charts_diagrams"]["scorer"]=="chart_relaxed"
    assert mapping["knowledge_reasoning"]["scorer"]=="mcq_accuracy"
    assert all(row["scorer"]!="vqa_accuracy" and len(row["answers"])==1 for row in selection["records"])


def test_repackaged_official_test_and_nonfinite_teacher_labels_rejected():
    records = validation_records()
    records[0]["source"]["original_split"]="test"
    with pytest.raises(ValueError,match="repackaged official"):
        build_internal_validation(records)
    records = validation_records()
    records[0]["answer"]=float("nan")
    with pytest.raises(ValueError,match="Nonfinite numeric"):
        build_internal_validation(records)


@pytest.mark.parametrize("answer,extra,expected", [
    ("B",{},"mcq_accuracy"),
    ("blue",{},"exact_match"),
    ("The correct answer is B because the object is blue.",{},"exact_match"),
    ("The correct answer is B because the object is blue.",{"reference_letter":"B"},"mcq_accuracy"),
    ("B",{"answer_letter":"A"},"exact_match"),
    ("Z",{},"exact_match"),
    ("B",{"choices":["red","RED"]},"exact_match"),
])
def test_reasoning_only_uses_unambiguous_explicit_letter_and_choices(answer,extra,expected):
    records = validation_records((1,1,1,1))
    reasoning = records[-1]
    reasoning["answer"] = answer
    reasoning["question"] += " The prompt mentions B, which must not be mined as a reference."
    reasoning.update(extra)
    selected = build_internal_validation(records)["records"][-1]
    assert selected["scorer"]==expected
    if expected=="exact_match":
        assert selected["answers"]==[answer]


def test_no_gold_passed_to_generator_and_per_scorer_group_macro_semantics():
    records = validation_records((1,1,1,1))
    model = SelectionModel()
    seen = []
    predictions = {"general_vqa":"red.","ocr_documents":"hell",
                   "charts_diagrams":"104","knowledge_reasoning":"blue"}
    def generate(current,public,max_new_tokens):
        assert current is model
        assert set(public)<=set(("record_id","question","image_paths","image_hashes"))
        assert "answer" not in public and "answers" not in public and "teacher_reference" not in public
        assert not torch.is_grad_enabled()
        assert not current.training and not current.base.training
        assert all(not p.requires_grad for p in current.parameters())
        assert current.config.use_cache
        assert not current.base.gradient_checkpointing
        seen.append(public["record_id"])
        group = public["record_id"].rsplit("-",1)[0]
        return {"prediction":predictions[group],"generated_tokens":1}
    before = {name:parameter.detach().clone() for name,parameter in model.named_parameters()}
    result = evaluate_internal_group_macro(model,records,per_group=1,generator=generate)
    assert len(seen)==4
    assert result["metrics"]["group_macro_score"]==pytest.approx(70.)
    assert result["metrics"]["protocol"]==PROTOCOL_NAME
    assert result["metrics"]["official_benchmark_score"] is False
    for name,parameter in model.named_parameters():
        torch.testing.assert_close(parameter,before[name])


@pytest.mark.parametrize("failure", [False,True])
def test_all_modes_requires_config_checkpoint_masks_and_rng_restore_even_on_error(failure):
    import numpy as np
    model = SelectionModel()
    modules = list(model.modules())+[model.extractor.vision_tower]
    modes = {id(module):module.training for module in modules}
    flags = {name:p.requires_grad for name,p in model.named_parameters()}
    old_mask = model.base._modality_mask
    checkpoint_function = model.base._gradient_checkpointing_func
    config_flags = model.config.flags
    random_before = random.getstate()
    torch_before = torch.get_rng_state()
    numpy_before = np.random.get_state()
    def generate(current,_record,**_):
        assert current.config is current.base.config
        current.config.use_cache=False
        current.config.flags["nested"]=False
        current.config.new_flag=True
        current.base._modality_mask=torch.zeros((1,5),dtype=torch.bool)
        current.base._frozen_grouped_cache=("temporary",torch.ones(4))
        current.generation_config.do_sample=False
        torch.rand(3);random.random();np.random.rand()
        if failure:
            raise RuntimeError("deliberate generation error")
        return "no"
    if failure:
        with pytest.raises(RuntimeError,match="deliberate"):
            evaluate_internal_group_macro(model,validation_records(),per_group=1,generator=generate)
    else:
        evaluate_internal_group_macro(model,validation_records(),per_group=1,generator=generate)
    assert {id(module):module.training for module in modules}==modes
    assert {name:p.requires_grad for name,p in model.named_parameters()}==flags
    assert model.base.gradient_checkpointing
    assert model.base._gradient_checkpointing_func is checkpoint_function
    assert model.base._modality_mask is old_mask
    assert not hasattr(model.base,"_frozen_grouped_cache")
    assert model.config.use_cache is False
    assert model.config.flags is config_flags and config_flags=={"nested":True}
    assert not hasattr(model.config,"new_flag")
    assert model.generation_config.do_sample is True
    assert random.getstate()==random_before
    assert torch.equal(torch.get_rng_state(),torch_before)
    after = np.random.get_state()
    assert after[0]==numpy_before[0] and np.array_equal(after[1],numpy_before[1]) and after[2:]==numpy_before[2:]
    assert torch.is_grad_enabled()


def test_mutated_built_manifest_and_fake_consensus_are_rejected():
    selected = build_internal_validation(validation_records(),per_group=1)
    selected["records"][0]["answers"]*=10
    selected["records"][0]["scorer"]="vqa_accuracy"
    with pytest.raises(ValueError,match="fabricated VQA consensus"):
        evaluate_internal_group_macro(SelectionModel(),selected,generator=lambda *a,**k:"red")
    selected = build_internal_validation(validation_records(),per_group=1)
    selected["records"][0]["answers"]=["tampered"]
    with pytest.raises(ValueError,match="manifest changed"):
        evaluate_internal_group_macro(SelectionModel(),selected,generator=lambda *a,**k:"red")


def test_actual_tiny_native_generation_does_not_break_training_checkpointing():
    # Reuse the independently tested native SmallThinker/tokenizer/cache fixture;
    # this test exercises the real generator rather than a fake predictor.
    from test_research_inference import TinyResearchModel
    model = TinyResearchModel(wrapped=True)
    model.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.train()
    before = {name:p.requires_grad for name,p in model.named_parameters()}
    records = validation_records((1,1,1,1))
    for row in records:
        row["question"]="question red"
    result = evaluate_internal_group_macro(model,records,per_group=1,max_new_tokens=2)
    assert result["metrics"]["records"]==4
    assert result["metrics"]["complete_four_group_coverage"]
    assert model.training and model.base.is_gradient_checkpointing
    assert {name:p.requires_grad for name,p in model.named_parameters()}==before
    tokens = torch.tensor([[1,6,7,8]])
    loss = model.base(input_ids=tokens,labels=tokens,use_cache=False).loss
    loss.backward()
    gradient = model.base.get_input_embeddings().weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
