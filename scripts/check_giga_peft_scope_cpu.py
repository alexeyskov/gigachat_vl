#!/usr/bin/env python3
"""Pinned GigaChat ordinary-LoRA metadata proof; strictly CPU/meta, no downloads."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig,get_peft_model
from transformers import AutoModelForCausalLM

from src.model.research_config import load_research_native_config
from src.model.research_giga_peft import get_giga_ordinary_peft_model,resolve_giga_ordinary_targets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'research_results/giga_ordinary_peft_cpu.json'))
    args=parser.parse_args()
    if torch.cuda.is_initialized():raise RuntimeError('CPU proof initialized CUDA before model construction')
    inventory=json.loads((ROOT/'research_results/model_inventory.json').read_text())
    entry=next(row for row in inventory if row['repo']=='ai-sage/GigaChat3.1-10B-A1.8B-bf16')
    path=Path(hf_hub_download(entry['repo'],'config.json',revision=entry['sha'],local_files_only=True))
    config,config_receipt=load_research_native_config(path)
    with torch.device('meta'):
        model=AutoModelForCausalLM.from_config(config,trust_remote_code=False,attn_implementation='eager')
    targets=resolve_giga_ordinary_targets(model)
    native_count=sum(parameter.numel() for parameter in model.parameters())
    original_config=copy.deepcopy(model.config.to_dict())
    packed={name:module for name,module in model.named_modules() if type(module).__name__=='DeepseekV3Experts'}
    packed_parameter_ids={name:id(parameter) for name,parameter in model.named_parameters() if '.experts.' in name}
    routers={name:module for name,module in model.named_modules() if name.endswith('.gate')}
    head=model.lm_head
    original_error=None
    try:
        with torch.device('meta'):
            get_peft_model(copy.deepcopy(model),LoraConfig(task_type='CAUSAL_LM',r=16,lora_alpha=32,
                lora_dropout=.05,bias='none',target_modules=targets))
    except ValueError as error:
        if str(error)!='lora.ParamWrapper does not work with lora_dropout != 0.':raise
        original_error={'type':type(error).__name__,'message':str(error)}
    if original_error is None:raise RuntimeError('Expected original API gap was not reproduced; re-review compatibility helper')
    wrapped,receipt=get_giga_ordinary_peft_model(model,target_modules=targets)
    assert wrapped.get_base_model() is model
    assert all(parameter.device.type=='meta' for parameter in wrapped.parameters())
    assert all(model.get_submodule(name) is module for name,module in packed.items())
    assert all(model.get_submodule(name) is module for name,module in routers.items())
    assert {name:id(parameter) for name,parameter in model.named_parameters() if '.experts.' in name}==packed_parameter_ids
    assert model.lm_head is head and model.config.to_dict()==original_config
    assert hashlib.sha256(path.read_bytes()).hexdigest()==config_receipt['source_sha256']
    independent=sum(16*(row['in_features']+row['out_features']) for row in receipt['target_shapes'])
    assert independent==receipt['ordinary_lora_parameters']
    assert sum(parameter.numel() for parameter in wrapped.parameters() if parameter.requires_grad)==independent
    assert not torch.cuda.is_initialized()
    ready=json.loads((ROOT/'configs/research/weighted_sft_ready.json').read_text())
    unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
                  for name,digest in ready['source_sha256'].items())
    assert unchanged
    counts={'attention':0,'dense_ffn':0,'shared_ffn':0}
    for name in targets:
        counts['attention' if '.self_attn.' in name else 'shared_ffn' if '.shared_experts.' in name else 'dense_ffn']+=1
    sources=['src/model/research_giga_peft.py','tests/test_research_giga_peft.py',
             'scripts/check_giga_peft_scope_cpu.py','src/model/research_config.py']
    report={'version':1,'status':'passed','created_at':datetime.now(timezone.utc).isoformat(),
            'model_ref':{'repo':entry['repo'],'revision':entry['sha']},'config':config_receipt,
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
            'original_native_peft_error':original_error,'helper':receipt,
            'base_unique_parameters':native_count,'ordinary_target_module_count':len(targets),
            'ordinary_target_categories':counts,'packed_banks_unchanged':len(packed),
            'router_modules_unchanged':len(routers),'lm_head_unchanged':True,
            'ready_bound_sources_unchanged':unchanged,
            'proof':{'all_parameters_meta':True,'cuda_initialized':False,'weight_download_bytes':0,
                     'metadata_download_bytes':0,'weights_loaded':False,'configuration_source_bytes_unchanged':True},
            'limits':['Pinned metadata proof only; no actual GigaChat checkpoint was loaded.',
                      'Separate tiny real native CPU tests use eager native attention/experts and prove forward/backward/update/dropout/state contracts.',
                      'Actual ordinary NF4 CUDA parity is a separate tiny-fixture proof; this pinned metadata run does not certify full GigaChat checkpoint loading or memory fit.',
                      'Restore requires rebuilding via this helper and canonical native state keys; generic PEFT auto-converting reload is not validated.',
                      'Current runner, inference, selection and READY sources remain unchanged.']}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(out),'status':'passed','ordinary_targets':len(targets),
                      'ordinary_lora_parameters':independent,'cuda_initialized':False}))


if __name__=='__main__':main()
