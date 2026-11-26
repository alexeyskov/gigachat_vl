"""Internal train-derived, single-teacher-reference checkpoint selection.

This protocol is deliberately distinct from official benchmark scores. In
particular, a single VQA teacher answer is exact-matched; it is never repeated
ten times to manufacture human-consensus accuracy.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
import random
import re

import torch

from src.eval.research_inference import generate_prediction
from src.eval.research_metrics import aggregate_scores, score_prediction


PROTOCOL_NAME = "internal_teacher_reference_group_macro"
GROUPS = ("general_vqa","ocr_documents","charts_diagrams","knowledge_reasoning")
_MISSING = object()


def _train_derived_source(record):
    source = record.get("source")
    if not isinstance(source,dict):
        raise ValueError("Internal selection requires explicit train-derived source metadata.")
    split = str(source.get("split","")).strip().lower()
    if not re.fullmatch(r"train(?:\[[^\]]+\])?",split):
        raise ValueError("Official validation/test or unproven source splits cannot select checkpoints; source.split must be train.")
    for key in ("original_split","official_split","benchmark_split"):
        if key in source and not re.fullmatch(r"train(?:\[[^\]]+\])?",str(source[key]).strip().lower()):
            raise ValueError("A repackaged official validation/test split cannot select checkpoints.")
    partition = record.get("partition")
    if partition is not None and str(partition).lower() not in ("validation","val","heldout","internal_validation"):
        raise ValueError("Selection records must be an internal held-out partition, not training/test rows.")
    return source


def _teacher_reference(record):
    answer = record.get("answer")
    if not isinstance(answer,(str,int,float)) or isinstance(answer,bool):
        raise ValueError("A single explicit teacher answer is required for internal selection.")
    if isinstance(answer,float) and not math.isfinite(answer):
        raise ValueError("Nonfinite numeric teacher references cannot be scored.")
    answer = str(answer).strip()
    if not answer:
        raise ValueError("Empty teacher references cannot be scored.")
    return answer


def _explicit_mcq_letter(record,answer,choices):
    if not isinstance(choices,list) or not 2<=len(choices)<=26 or not all(isinstance(c,str) and c.strip() for c in choices):
        return None
    if len({choice.strip().casefold() for choice in choices})!=len(choices):
        return None
    candidates = []
    for raw,explicit in ((answer,False),(record.get("reference_letter"),True),(record.get("answer_letter"),True)):
        value = raw
        if value is None:
            continue
        value = str(value).strip().upper()
        if re.fullmatch(r"[A-Z]",value) and ord(value)-ord("A")<len(choices):
            candidates.append(value)
        elif explicit:
            # Invalid explicit labels cannot establish an unambiguous reference.
            return None
    return candidates[0] if candidates and len(set(candidates))==1 else None


def _convert_record(record):
    source = _train_derived_source(record)
    if not isinstance(record.get("record_id"),str) or not record["record_id"]:
        raise ValueError("Selection record requires a stable record_id.")
    if not isinstance(record.get("question"),str) or not record["question"].strip():
        raise ValueError("Selection record requires a nonempty question.")
    group = record.get("task_group")
    if group not in GROUPS:
        raise ValueError(f"Unknown multimodal task group: {group}")
    answer = _teacher_reference(record)
    scorer = {"general_vqa":"exact_match","ocr_documents":"anls",
              "charts_diagrams":"chart_relaxed","knowledge_reasoning":"exact_match"}[group]
    choices = record.get("choices")
    letter = _explicit_mcq_letter(record,answer,choices) if group=="knowledge_reasoning" else None
    if letter is not None:
        scorer = "mcq_accuracy"
    result = {"record_id":record["record_id"],"question":record["question"],
              "image_paths":list(record.get("image_paths",[])),
              "answers":[letter if letter is not None else answer],
              "teacher_reference":answer,"task_group":group,"scorer":scorer,
              "source":copy.deepcopy(source),"partition":"internal_validation",
              "reference_protocol":"single_train_derived_teacher_reference"}
    if "image_hashes" in record:
        if len(record["image_hashes"])!=len(result["image_paths"]):
            raise ValueError("Selection image paths/hashes differ in count.")
        result["image_hashes"] = list(record["image_hashes"])
    if letter is not None:
        result["choices"] = list(choices)
    if "benchmark" in record:
        result["benchmark"] = record["benchmark"]
    return result


def build_internal_validation(records, *, per_group=32, seed=42, require_per_group=False):
    """Deterministic balanced sampling, independent of input order and model.

    Text-only rows are returned separately and never improve the visual macro.
    ``require_per_group=False`` uses the same N=min(requested,all-group-counts)
    everywhere; True fails if any group cannot supply the requested count.
    """
    if isinstance(per_group,bool) or not isinstance(per_group,int) or per_group<=0:
        raise ValueError("per_group must be a positive integer.")
    pools = {group:[] for group in GROUPS}
    text = []
    seen = set()
    for original in records:
        _train_derived_source(original)
        record_id = original.get("record_id")
        if not isinstance(record_id,str) or not record_id or record_id in seen:
            raise ValueError("Internal selection needs unique stable record IDs.")
        seen.add(record_id)
        if not original.get("image_paths"):
            text.append(copy.deepcopy(original))
            continue
        converted = _convert_record(original)
        pools[converted["task_group"]].append(converted)
    missing = [group for group,rows in pools.items() if not rows]
    if missing:
        raise ValueError(f"All four multimodal groups are required; missing: {missing}")
    available = {group:len(rows) for group,rows in pools.items()}
    if require_per_group and min(available.values())<per_group:
        raise ValueError(f"Cannot supply requested {per_group} records per group: {available}")
    count = min(per_group,*available.values())
    selected = []
    for group in GROUPS:
        ordered = sorted(pools[group],key=lambda row:row["record_id"])
        generator = random.Random(f"{seed}:{group}")
        selected.extend(generator.sample(ordered,count))
    selected.sort(key=lambda row:(GROUPS.index(row["task_group"]),row["record_id"]))
    text.sort(key=lambda row:row["record_id"])
    digest = hashlib.sha256(json.dumps(selected,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
    protocol = {"name":PROTOCOL_NAME,"schema_version":1,"official_benchmark_score":False,
                "reference_type":"single_train_derived_teacher_reference",
                "selection_direction":"max","seed":seed,"requested_per_group":per_group,
                "selected_per_group":count,"available_per_group":available,
                "strict_requested_count":bool(require_per_group),"records":len(selected),
                "text_only_records_excluded":len(text),"selection_manifest_sha256":digest,
                "choice_prompt_policy":"source_question_unchanged_no_answer_mining",
                "scorers":{"general_vqa":"single_reference_exact_match","ocr_documents":"anls",
                           "charts_diagrams":"chart_relaxed","knowledge_reasoning":"explicit_mcq_letter_else_full_reference_exact_match"}}
    return {"records":selected,"text_records":text,"protocol":protocol}


def _roots_and_modules(model):
    roots = [model]
    extractor = getattr(model,"extractor",None)
    if extractor is not None:
        roots.extend(module for name in ("vision_tower","vision_projector")
                     if (module:=getattr(extractor,name,None)) is not None)
    modules = {id(module):module for root in roots for module in root.modules()}
    return roots,list(modules.values())


@contextmanager
def selection_generation_mode(model):
    """Temporarily evaluate without altering the subsequent training state."""
    roots,modules = _roots_and_modules(model)
    parameters = {id(parameter):parameter for module in modules for parameter in module.parameters(recurse=False)}
    requires_grad = {key:parameter.requires_grad for key,parameter in parameters.items()}
    training = {id(module):module.training for module in modules}
    runtime_names = ("gradient_checkpointing","_gradient_checkpointing_func",
                     "_modality_mask","_attention_mask","warmup_force_new",
                     "_diagnostic_counts","_diagnostic_mass","_diagnostic_alpha",
                     "last_indices","last_weights","_frozen_grouped_cache")
    runtime = {id(module):{name:module.__dict__.get(name,_MISSING) for name in runtime_names}
               for module in modules}
    configs = {}
    for module in modules:
        for name in ("config","generation_config"):
            config = getattr(module,name,None)
            if config is not None and hasattr(config,"__dict__") and id(config) not in configs:
                original = dict(config.__dict__)
                mutable = {name:copy.deepcopy(value) for name,value in original.items() if isinstance(value,(dict,list,set))}
                configs[id(config)] = (config,original,mutable)
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        import numpy as np
        numpy_rng = np.random.get_state()
    except ImportError:
        np = numpy_rng = None
    try:
        for root in roots:
            root.eval()
        for parameter in parameters.values():
            parameter.requires_grad_(False)
        for module in modules:
            if isinstance(getattr(module,"gradient_checkpointing",None),bool):
                module.gradient_checkpointing = False
        for config,_snapshot,_mutable in configs.values():
            if hasattr(config,"use_cache"):
                config.use_cache = True
        with torch.no_grad():
            yield
    finally:
        for config,snapshot,mutable in configs.values():
            # Keep shared config object/container identities. Replacing a
            # nested vision/text config with a deepcopy can break live aliases.
            for name,value in mutable.items():
                original = snapshot[name]
                if isinstance(original,dict):
                    original.clear();original.update(value)
                elif isinstance(original,list):
                    original[:] = value
                else:
                    original.clear();original.update(value)
            config.__dict__.clear();config.__dict__.update(snapshot)
        for module in modules:
            module.training = training[id(module)]
            for name,value in runtime[id(module)].items():
                if value is _MISSING:
                    module.__dict__.pop(name,None)
                else:
                    module.__dict__[name] = value
        for key,parameter in parameters.items():
            parameter.requires_grad_(requires_grad[key])
        random.setstate(python_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if np is not None:
            np.random.set_state(numpy_rng)


def _validate_built_selection(selection):
    protocol = selection.get("protocol",{})
    if protocol.get("name")!=PROTOCOL_NAME or protocol.get("official_benchmark_score") is not False:
        raise ValueError("Expected the internal teacher-reference selection protocol.")
    counts = {group:0 for group in GROUPS}
    ids = set()
    for row in selection["records"]:
        _train_derived_source(row)
        if row["record_id"] in ids or row.get("task_group") not in counts:
            raise ValueError("Invalid internal selection IDs/groups.")
        ids.add(row["record_id"])
        counts[row["task_group"]]+=1
        if row.get("scorer") not in {"exact_match","anls","chart_relaxed","mcq_accuracy"} or len(row.get("answers",[]))!=1:
            raise ValueError("Internal selection must use one teacher reference, never fabricated VQA consensus.")
    if not min(counts.values()) or len(set(counts.values()))!=1:
        raise ValueError("Internal selection must balance all four multimodal groups.")
    digest = hashlib.sha256(json.dumps(selection["records"],sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
    if digest!=protocol.get("selection_manifest_sha256"):
        raise ValueError("Internal selection manifest changed after sampling.")


def evaluate_internal_group_macro(model, records_or_selection, *, per_group=32, seed=42,
                                  require_per_group=False, max_new_tokens=64,
                                  generator=None, progress=None):
    selection = (records_or_selection if isinstance(records_or_selection,dict) else
                 build_internal_validation(records_or_selection,per_group=per_group,seed=seed,
                                           require_per_group=require_per_group))
    _validate_built_selection(selection)
    if max_new_tokens<=0:
        raise ValueError("max_new_tokens must be positive.")
    predict = generator or generate_prediction
    predictions = []
    with selection_generation_mode(model):
        for index,record in enumerate(selection["records"]):
            # Do not even pass a gold-bearing dict to generation. Explicit public
            # fields prevent accidental future prompt construction from answers.
            public = {key:record[key] for key in ("record_id","question","image_paths","image_hashes") if key in record}
            generated = predict(model,public,max_new_tokens=max_new_tokens)
            prediction = generated["prediction"] if isinstance(generated,dict) else str(generated)
            row = {key:copy.deepcopy(record[key]) for key in ("record_id","task_group","scorer","answers","source","choices","benchmark") if key in record}
            row.update(prediction=prediction,score=score_prediction(record,prediction))
            if isinstance(generated,dict):
                row["generation"] = generated
            predictions.append(row)
            if progress is not None:
                progress({"event":"internal_selection_record","completed_records":index+1,
                          "total_records":len(selection["records"]),"record_id":record["record_id"],
                          "task_group":record["task_group"],"prediction":prediction,"score":row["score"]})
    metrics = aggregate_scores(predictions,require_four_groups=True)
    metrics.update(protocol=PROTOCOL_NAME,official_benchmark_score=False,
                   reference_type="single_train_derived_teacher_reference",selection_direction="max",
                   text_only_records_excluded=len(selection["text_records"]))
    protocol = dict(selection["protocol"],max_new_tokens=max_new_tokens,generation="greedy_native_cached_no_gold")
    return {"metrics":metrics,"predictions":predictions,"protocol":protocol}
