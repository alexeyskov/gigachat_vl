#!/usr/bin/env python3
"""Exact meta-device parameter counts; lower bounds for the unchanged 396 runs.

Reads pinned, locally cached configuration JSON only. Never loads/downloads
language/vision weights, initializes CUDA, schedules jobs, or changes a recipe.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# PEFT's optional backends may probe CUDA during import. This standalone audit
# must not create a GPU context alongside training, including during imports.
if __name__=='__main__':os.environ['CUDA_VISIBLE_DEVICES']=''

import torch
from torch import nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from src.model.research_config import load_research_native_config
from src.model.research_loading import ATTENTION_PROJECTIONS
from src.model.research_moe import install_research_moe,configure_research_trainability

MODELS=[('smallthinker4','Tiiny/SmallThinker-4BA0.6B-Instruct'),
        ('lfm25','LiquidAI/LFM2.5-8B-A1B'),
        ('giga31','ai-sage/GigaChat3.1-10B-A1.8B-bf16'),
        ('gptoss20','openai/gpt-oss-20b')]
ENCODERS=[('siglip2','google/siglip2-so400m-patch14-384'),
          ('qwen25vision','Qwen/Qwen2.5-VL-7B-Instruct'),
          ('gemma4vision','google/gemma-4-E4B-it')]
RECIPES=['1a','1b','1c','2a','2b','2c','3a','3b','3c','4q','4f']
SEEDS=[42,123,2026]
GIB=2**30
TIB=2**40


def config_path(inventory,repo):
    row=next(item for item in inventory if item['repo']==repo)
    # local_files_only is deliberate: a missing config fails instead of fetching.
    return Path(hf_hub_download(repo,'config.json',revision=row['sha'],local_files_only=True)),row


def make_meta(config):
    with torch.device('meta'):
        model=AutoModelForCausalLM.from_config(copy.deepcopy(config),
            trust_remote_code=False,attn_implementation='eager')
    if not all(parameter.device.type=='meta' for parameter in model.parameters()):
        raise RuntimeError('Native construction unexpectedly materialized weights')
    return model


def projector_parameters(input_size,hidden_size):
    with torch.device('meta'):
        projector=nn.Sequential(nn.Linear(input_size,hidden_size),nn.GELU(),
                                nn.Linear(hidden_size,hidden_size))
    return sum(parameter.numel() for parameter in projector.parameters())


def packed_lora_parameters(model,rank=16):
    """Exact A/B shape counts for the actual per-expert NF4 wrapper contract.

    This is not quantization: NF4 bases cannot be constructed from meta weights.
    gate/up share one rank-r factorization in the packed backend.
    """
    total=0;banks=[]
    for name,module in model.named_modules():
        kind=type(module).__name__
        if kind not in ('Lfm2MoeExperts','DeepseekV3Experts','GptOssExperts'):continue
        e=module.num_experts
        h=module.hidden_size if kind=='GptOssExperts' else module.hidden_dim
        i=module.intermediate_size if kind=='GptOssExperts' else module.intermediate_dim
        count=e*rank*(2*h+3*i)
        total+=count
        banks.append({'name':name,'experts':e,'hidden_size':h,'intermediate_size':i,
                      'rank':rank,'lora_parameters':count,
                      'formula':'E*r*((H+2*I)+(I+H))'})
    return total,banks


def ordinary_lora_targets(model):
    kind=model.config.model_type
    if kind=='smallthinker':return ['q_proj','k_proj','v_proj','o_proj','up','gate','down']
    if kind=='gpt_oss':
        return [f'{parent_name}.{name}' for parent_name,parent in model.named_modules()
                if parent_name.endswith(('self_attn','attention'))
                for name,child in parent.named_children()
                if name in ATTENTION_PROJECTIONS and type(child) is nn.Linear]
    return [name for name,module in model.named_modules()
            if isinstance(module,nn.Linear) and name!='lm_head'
            and not name.endswith(('.gate','.router')) and '.experts.' not in name]


def parameter_groups(model):
    result={}
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:continue
        if 'lora_' in name:group='ordinary_lora'
        elif '.visual_experts.' in name:group='visual_experts'
        elif any(f'.{item}.' in name for item in ('router_adapter','router_new','router_mass')):group='router_adapters'
        elif 'modality_gate_logits' in name:group='modality_gates'
        elif '.primary_router.' in name:group='research_primary_routers'
        else:group='other_native_parameters'
        result[group]=result.get(group,0)+parameter.numel()
    return result


def count_recipe(config,recipe):
    model=make_meta(config)
    quantized=recipe.startswith('1') or recipe=='4q'
    packed_count=0;packed_banks=[];targets=[]
    if quantized:
        from peft import LoraConfig,get_peft_model
        model.requires_grad_(False)
        packed_count,packed_banks=packed_lora_parameters(model)
        targets=ordinary_lora_targets(model)
        # GPT's loader uses native attention A/B with the same shapes as PEFT.
        # Giga's auto conversion also remaps dense/shared targets into 3D expert
        # parameters. Direct PEFT layers count the declared ordinary scope only;
        # the unmodified high-level API failure is recorded separately below.
        options={'target_parameters':[]} if config.model_type=='gpt_oss' else {}
        lora_config=LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,
                               lora_dropout=.05,bias='none',target_modules=targets,**options)
        with torch.device('meta'):
            if config.model_type=='deepseek_v3':
                from peft.tuners.lora.layer import Linear as PeftLinear
                for name in targets:
                    parent_name,_,leaf=name.rpartition('.')
                    parent=model.get_submodule(parent_name)
                    child=getattr(parent,leaf)
                    setattr(parent,leaf,PeftLinear(child,'default',config=lora_config,r=16,lora_alpha=32))
            else:model=get_peft_model(model,lora_config)
    else:model.requires_grad_(True)
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    wrappers=[]
    if recipe[0] in '123':
        with torch.device('meta'):
            wrappers=install_research_moe(base,int(recipe[0]),recipe[1],added_experts=2)
        configure_research_trainability(base,stage='SFT')
    groups=parameter_groups(model)
    if packed_count:groups['packed_expert_lora']=packed_count
    count=sum(groups.values())
    if not all(parameter.device.type=='meta' for parameter in model.parameters()):
        raise RuntimeError('Recipe construction unexpectedly materialized weights')
    largest=max(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    for bank in packed_banks:
        largest=max(largest,bank['rank']*bank['hidden_size'],
                    2*bank['rank']*bank['intermediate_size'])
    return {'recipe':recipe,'training':'qlora' if quantized else 'full_fp32_masters_bf16_compute',
            'added_experts_per_moe_layer':2 if wrappers else 0,'moe_layers_expanded':len(wrappers),
            'llm_trainable_parameters':count,'trainable_parameter_groups':groups,
            'ordinary_lora_targets':targets,'packed_lora_banks':packed_banks,
            'count_only_peft_scope_override':('direct ordinary PEFT Linear layers, bypassing automatic Transformers-v4/v5 target conversion'
                if quantized and config.model_type=='deepseek_v3' else 'target_parameters=[]; ordinary attention shapes match native NF4 wrapper'
                if quantized and config.model_type=='gpt_oss' else None),
            'count_method':'native meta + actual PEFT adapters + actual research E+K wrappers; packed A/B exact shape formula',
            'largest_trainable_tensor_parameters':largest,
            'raw_adam32_moment_bytes':8*count,
            'raw_adam8_main_moment_bytes_estimate':2*count,
            'adam8_estimate_exclusions':'small parameters use FP32 states; block scales, metadata and alignment excluded',
            **({'memory_estimates':{
                'fp32_llm_weights_bytes':4*sum(parameter.numel() for parameter in model.parameters()),
                'all_fp32_gradients_bytes':4*count,
                'one_staged_fp32_gradient_bytes':4*largest,
                'excluded':'activations, frozen vision/projector, attention workspaces, allocator overhead, BF16 casts/autocast cache, optimizer residency and CUDA context',
                'capacity_validated':False}} if not quantized else {})}


def generate_report(inventory):
    if torch.cuda.is_initialized():raise RuntimeError('Capacity task must start without initialized CUDA')
    encoders=[]
    for encoder_id,repo in ENCODERS:
        path,row=config_path(inventory,repo);raw=json.loads(path.read_bytes())
        dim=(raw['vision_config']['hidden_size'] if encoder_id=='siglip2'
             else raw['vision_config']['out_hidden_size'] if encoder_id=='qwen25vision'
             else raw['text_config']['hidden_size'])
        encoders.append({'id':encoder_id,'repo':repo,'revision':row['sha'],
                         'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'projector_input_dimension':dim,
                         'feature_point':{'siglip2':'vision patch hidden states','qwen25vision':'after native spatial merger',
                                          'gemma4vision':'after frozen donor multimodal embedder'}[encoder_id]})
    models=[];full_bytes=qlora_bytes=projector_bytes=0
    for model_id,repo in MODELS:
        path,inventory_row=config_path(inventory,repo)
        config,config_receipt=load_research_native_config(path)
        native=make_meta(config)
        base_count=sum(parameter.numel() for parameter in native.parameters())
        row={'id':model_id,'repo':repo,'revision':inventory_row['sha'],'config':config_receipt,
             'base_unique_parameters':base_count,
             'base_named_tensor_count':len(list(native.parameters())),
             'input_output_embeddings_tied':native.get_input_embeddings().weight is native.get_output_embeddings().weight,
             'hidden_size':config.hidden_size,'recipes':[]}
        if config.model_type=='deepseek_v3':
            # Reproduce the current runner API gap, then count the intended
            # ordinary+separate-packed scope without silently enabling 3D PEFT.
            from peft import LoraConfig,get_peft_model
            probe=make_meta(config);probe.requires_grad_(False)
            try:
                with torch.device('meta'):
                    get_peft_model(probe,LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,
                        lora_dropout=.05,bias='none',target_modules=ordinary_lora_targets(probe)))
                row['original_peft_meta_probe']={'passed':True}
            except ValueError as error:
                if str(error)!='lora.ParamWrapper does not work with lora_dropout != 0.':raise
                row['original_peft_meta_probe']={'passed':False,'error_type':type(error).__name__,
                    'message':str(error),'count_only_repair':'direct ordinary PEFT Linear layers bypass automatic conversion; native NF4 packed A/B counted separately',
                    'target_parameters_empty_list_alone_is_insufficient':True,
                    'production_loader_changed':False}
            del probe
        del native
        for recipe in RECIPES:
            details=count_recipe(config,recipe)
            details['encoders']={}
            for encoder in encoders:
                projector=projector_parameters(encoder['projector_input_dimension'],config.hidden_size)
                model_bytes=4*(details['llm_trainable_parameters']+projector)
                details['encoders'][encoder['id']]={'frozen_projector_parameters':projector,
                    'sft_checkpoint_parameters':details['llm_trainable_parameters']+projector,
                    'minimum_raw_fp32_model_bytes':model_bytes}
                projector_bytes+=4*projector*len(SEEDS)
                if recipe.startswith('1') or recipe=='4q':qlora_bytes+=model_bytes*len(SEEDS)
                else:full_bytes+=model_bytes*len(SEEDS)
            row['recipes'].append(details)
        models.append(row)
    if torch.cuda.is_initialized():raise RuntimeError('Capacity task unexpectedly initialized CUDA')
    minimum=full_bytes+qlora_bytes
    usage=shutil.disk_usage(ROOT)
    largest=max((dict(model_id=model['id'],recipe=recipe['recipe'],encoder_id=encoder_id,
                       model_bytes=entry['minimum_raw_fp32_model_bytes'],
                       adam32_moment_bytes=recipe['raw_adam32_moment_bytes'],
                       adam8_main_moment_bytes_estimate=recipe['raw_adam8_main_moment_bytes_estimate'])
                 for model in models for recipe in model['recipes'] if recipe['training'].startswith('full')
                 for encoder_id,entry in recipe['encoders'].items()),key=lambda item:item['model_bytes'])
    sources=['scripts/research_capacity.py','src/model/research_config.py','src/model/research_moe.py',
             'src/model/research_quantization.py','src/model/research_loading.py','scripts/run_research_stage.py',
             'src/model/research_vision.py','src/model/smallthinker.py']
    return {'version':1,'created_at':datetime.now(timezone.utc).isoformat(),
        'scope':'capacity audit only; primary budgets, recipes, retention and completion criteria unchanged',
        'software':{'torch':torch.__version__,'transformers':version('transformers'),'peft':version('peft')},
        'proof':{'weights_loaded':False,'weight_download_bytes':0,'metadata_download_bytes':0,
                 'all_native_and_recipe_parameters_on_meta':True,'cuda_initialized':False,
                 'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}},
        'matrix':{'models':4,'encoders':3,'seeds':SEEDS,'recipes':RECIPES,'trajectories':396,
                  'full_recipes':['2a','2b','2c','3a','3b','3c','4f'],'full_trajectories':252,
                  'qlora_recipes':['1a','1b','1c','4q'],'qlora_trajectories':144},
        'encoders':encoders,'models':models,
        'retention':{'bytes_per_saved_learned_parameter':4,'projector_saved_per_sft_checkpoint':True,
            'frozen_qlora_bases_saved_per_sft_checkpoint':False,
            'minimum_distinct_versions_per_trajectory':1,
            'minimum_version_condition':'best and last identify the same state; if distinct, both are retained',
            'minimum_raw_model_only_bytes':minimum,'minimum_raw_full_model_bytes':full_bytes,
            'minimum_raw_qlora_model_bytes':qlora_bytes,'projector_component_bytes':projector_bytes,
            'minimum_tib':minimum/TIB,'two_distinct_best_last_raw_model_bytes':2*minimum,
            'two_distinct_best_last_tib':2*minimum/TIB,
            'excluded':'A0/A1 checkpoints, original bases, vision weights, image data, frozen-feature caches, optimizer/scheduler/RNG, serialization headers, manifests, evaluation outputs, provenance, transient staging and reserve',
            'no_destructive_policy_change':'This lower bound does not authorize deleting divergent best/last or scientific artifacts.'},
        'largest_active_full_sft':dict(largest,
            single_resumable_raw_adam32_bytes=largest['model_bytes']+largest['adam32_moment_bytes'],
            single_resumable_adam8_main_bytes_estimate=largest['model_bytes']+largest['adam8_main_moment_bytes_estimate'],
            three_objects_model_only_staging_bytes=3*largest['model_bytes'],
            three_objects_resumable_adam32_staging_bytes=3*(largest['model_bytes']+largest['adam32_moment_bytes']),
            three_objects_resumable_adam8_main_staging_bytes_estimate=3*(largest['model_bytes']+largest['adam8_main_moment_bytes_estimate']),
            adam8_estimate_exclusions='small-parameter FP32 states, block scales, metadata and alignment; not a measured checkpoint size',
            staging_scope='two distinct best/last objects plus one new atomic-save object; raw tensors only'),
        'disk_snapshot':{'total_bytes':usage.total,'used_bytes':usage.used,'free_bytes':usage.free,
                         'unchanged_reserve_bytes':120*GIB,
                         'minimum_archive_shortfall_vs_entire_local_disk_bytes':max(0,minimum-usage.total),
                         'minimum_archive_exceeds_entire_disk_factor':minimum/usage.total},
        'limits':['Native meta counts prove dimensions and parameter sharing, not weight-loading compatibility, CUDA kernels, training throughput or full-model memory fit.',
                  'QLoRA counts use actual ordinary PEFT A/B modules and exact packed-backend factor shapes; frozen quantized base/quantization-state storage is excluded.',
                  'Giga QLoRA capacity uses direct ordinary PEFT Linear construction because high-level PEFT misremaps dense/shared projection suffixes into packed expert factors and rejects dropout .05; target_parameters=[] alone is insufficient. The API gap is reproduced in original_peft_meta_probe; no production repair is wired.',
                  'GigaChat config num_nextn_predict_layers=1 is not an extra module in native HF DeepseekV3ForCausalLM; the count describes the actual causal LM instantiated by the runner, not unused donor MTP tensors.',
                  'Tied embeddings are counted once, following named_parameters and the training checkpoint saver; source repositories can store duplicated tied tensors.',
                  'These sizes are uncompressed raw tensor payloads. Serialization, compression or mathematical deduplication are separate storage designs and are not assumed.']}


def markdown(report):
    retention=report['retention'];lines=[
        '# Primary research capacity audit','',
        'The unchanged 396-trajectory matrix needs at least '
        f"**{retention['minimum_tib']:.3f} TiB** for one compact FP32 model payload per SFT trajectory. "
        'This already excludes optimizer states, original weights, data, feature caches and staging. '
        'Two distinct best/last states require '
        f"**{retention['two_distinct_best_last_tib']:.3f} TiB** of raw model tensors.",'',
        '| Backbone | Native unique parameters | E+2 / family 2b | E+2 / family 3b | 4q trainable LoRA |',
        '| --- | ---: | ---: | ---: | ---: |']
    for model in report['models']:
        recipes={row['recipe']:row for row in model['recipes']}
        lines.append(f"| {model['id']} | {model['base_unique_parameters']:,} | {recipes['2b']['llm_trainable_parameters']:,} | {recipes['3b']['llm_trainable_parameters']:,} | {recipes['4q']['llm_trainable_parameters']:,} |")
    lines+=['','All seven full and four QLoRA recipe counts, including the small per-expert policy-c gates, are in the JSON. '
             'The MLP projector is frozen at SFT and saved separately in every checkpoint.','',
             '| Backbone | SigLIP2 D=1152 projector | Qwen2.5-VL D=3584 projector | Gemma4 D=2560 projector |',
             '| --- | ---: | ---: | ---: |']
    for model in report['models']:
        enc=model['recipes'][0]['encoders']
        lines.append('| '+model['id']+' | '+' | '.join(f"{enc[item]['frozen_projector_parameters']:,}" for item,_ in ENCODERS)+' |')
    largest=report['largest_active_full_sft']
    lines+=['',f"Largest active full model: `{largest['model_id']}/{largest['recipe']}/{largest['encoder_id']}`. "
            f"Its model payload is {largest['model_bytes']/GIB:.3f} GiB; two FP32 Adam moments add {largest['adam32_moment_bytes']/GIB:.3f} GiB. "
            f"One resumable checkpoint therefore has at least {largest['single_resumable_raw_adam32_bytes']/GIB:.3f} GiB of raw tensors. "
            f"Two retained objects plus atomic-save staging can need {largest['three_objects_resumable_adam32_staging_bytes']/GIB:.3f} GiB during that job. "
            'Optimizer offload changes where working states reside; it does not remove their checkpoint payload. '
            f"For Adam8, main moments alone are approximately {largest['adam8_main_moment_bytes_estimate']/GIB:.3f} GiB; "
            f"three resumable objects are approximately {largest['three_objects_resumable_adam8_main_staging_bytes_estimate']/GIB:.3f} GiB "
            'before small-parameter FP32 states, block scales, metadata or the reserve. This is a tensor-size estimate, '
            'not a measured checkpoint size.','',
            f"Local capacity at audit time: {report['disk_snapshot']['total_bytes']/GIB:.0f} GiB. "
            f"The one-version SFT archive alone exceeds the entire disk by a factor of {report['disk_snapshot']['minimum_archive_exceeds_entire_disk_factor']:.2f}. "
            'The 120 GiB reserve stays unchanged. External archival capacity is required to retain the complete primary matrix; '
            'this report changes no completion or retention rules.','',
            'Counts come from pinned native `from_config` models on the meta device, actual E+2 wrappers, '
            'and PEFT adapter shapes. No weights were loaded or downloaded and CUDA was not initialized. '
            'FP32 weight/gradient figures are arithmetic lower bounds, not a 96 GiB GPU-fit claim.','',
            'The pinned GigaChat configuration declares Transformers 4.57.3; the installed version is '
            f"{report['models'][2]['config']['installed_transformers_version']}. Its integer `routed_scaling_factor: 1` "
            'fails strict native validation. The audit uses a separately recorded, in-memory, type-only conversion to `1.0`; '
            'original configuration bytes and SHA remain unchanged. No production loader is modified.','']
    lines+=['GigaChat QLoRA also needs an explicit future PEFT scope repair: the current runner API incorrectly remaps '
            'dense/shared FFN projection suffixes to 3D experts and rejects dropout 0.05. For counting, ordinary '
            'PEFT Linear layers are installed directly on the resolved 2D modules and our native packed factors '
            'are counted separately. An empty `target_parameters` list alone does not bypass the faulty conversion. The original failure is reproduced '
            'in the JSON; this does not claim that the unchanged production Giga loader is runnable.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',default=str(ROOT/'research_results/model_inventory.json'))
    parser.add_argument('--output',default=str(ROOT/'research_results/primary_capacity.json'))
    args=parser.parse_args()
    report=generate_report(json.loads(Path(args.inventory).read_text()))
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n');path.with_suffix('.md').write_text(markdown(report))
    print(json.dumps({'output':str(path),'minimum_tib':report['retention']['minimum_tib'],
                      'two_versions_tib':report['retention']['two_distinct_best_last_tib'],
                      'cuda_initialized':torch.cuda.is_initialized()}))


if __name__=='__main__':main()
