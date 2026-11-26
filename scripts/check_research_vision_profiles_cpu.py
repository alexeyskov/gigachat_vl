#!/usr/bin/env python3
"""Test actual cached Qwen processor caps without vision/LLM weight execution."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from huggingface_hub import hf_hub_download
    from src.model.research_vision import apply_preprocessing_profile,QWEN_PREPROCESSING_PROFILES
    from scripts.check_research_backbone_cpu import require_gpu_hidden
    from src.utils.research_storage import _atomic_json
    require_gpu_hidden(torch);torch.set_num_threads(8)
    repo='Qwen/Qwen2.5-VL-7B-Instruct';sha='cc594898137f460bfe9f0759e9844b3ce807cfb5'
    snapshot=Path(hf_hub_download(repo,'config.json',revision=sha,local_files_only=True)).parent
    rows=[]
    for profile,budget in QWEN_PREPROCESSING_PROFILES.items():
        processor=AutoProcessor.from_pretrained(snapshot,trust_remote_code=False,local_files_only=True)
        native=processor.image_processor.to_dict()
        extractor=SimpleNamespace(backend='qwen2_5_vl',processor=processor)
        contract=apply_preprocessing_profile(extractor,profile)
        shapes=[]
        for size in [(2048,2048),(4096,512),(8192,128),(1792,896)]:
            image=Image.new('RGB',size,(30,80,120))
            batch=processor.image_processor(images=[image],return_tensors='pt')
            grid=batch['image_grid_thw'][0].tolist();tokens=grid[0]*grid[1]*grid[2]//4
            if tokens>budget:raise ValueError(f'Actual native resize exceeded cap: {tokens}>{budget}')
            shapes.append({'original_size':list(size),'image_grid_thw':grid,'post_merger_tokens':tokens})
            del batch
        rows.append({'profile':profile,'contract':contract,'actual_processor_cases':shapes,'native_processor_config_before_override':native})
    report={'schema_version':1,'status':'passed','repo':repo,'revision':sha,'profiles':rows,
            'actual_processor_tested':True,'vision_weights_executed':False,'language_model_loaded':False,'gpu_initialized':False,
            'current_siglip_profile_changed':False,'limits':'Processor/grid bounds validated on real large images; no high-resolution encoder/GPU training benchmark implied'}
    target=ROOT/'research_results/qwen_preprocessing_profile_validation.json';_atomic_json(target,report)
    print(json.dumps({'status':'passed','report':str(target),'profiles':[row['profile'] for row in rows]}),flush=True)

if __name__=='__main__':main()
