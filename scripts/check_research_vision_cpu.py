#!/usr/bin/env python3
"""Validate an actual strict vision-only Qwen donor on CPU using tiny RGB fixtures."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.research_storage import _atomic_json
from scripts.check_research_backbone_cpu import require_gpu_hidden,json_ready


def emit(event,**fields):print(json.dumps({'time':time.time(),'event':event,**fields}),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--repo',default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--output',default='research_results/qwen25_vision_cpu_checkpoint_validation.json')
    args=parser.parse_args();started=time.monotonic();output=Path(args.output)
    report={'schema_version':1,'status':'running','repo':args.repo,'scope':'actual_vision_only_cpu_fixture_validation',
            'gpu_training_validated':False,'llm_loaded':False,'current_siglip_profile_modified':False}
    try:
        import torch
        from PIL import Image,ImageDraw
        from huggingface_hub import hf_hub_download,try_to_load_from_cache
        from src.model.research_vision import build_frozen_vision,required_vision_weight_files
        from src.dataset.vision_cache import VisionFeatureCache,encoder_fingerprint
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        require_gpu_hidden(torch);torch.set_num_threads(8);torch.set_num_interop_threads(1)
        row=next(item for item in json.loads((ROOT/'research_results/model_inventory.json').read_text()) if item['repo']==args.repo)
        if args.repo!='Qwen/Qwen2.5-VL-7B-Instruct':raise ValueError('This actual probe is scoped to Qwen2.5 vision only')
        config_path=hf_hub_download(args.repo,'config.json',revision=row['sha'],local_files_only=True)
        snapshot=Path(config_path).parent
        required=required_vision_weight_files(snapshot,'qwen2_5_vl')
        weights=[f['path'] for f in row['files'] if f['path'].endswith('.safetensors')]
        excluded=[name for name in weights if name not in required]
        if any(isinstance(try_to_load_from_cache(args.repo,name,revision=row['sha']),str) for name in excluded):
            raise ValueError('Probe expected no unnecessary donor language-weight shards in the partial snapshot')
        fixture_dir=ROOT/'research_results/vision_probes/qwen25';fixture_dir.mkdir(parents=True,exist_ok=True)
        paths=[]
        for name,size,color in [('landscape',(112,84),(200,30,20)),('portrait',(84,112),(20,50,200))]:
            image=Image.new('RGB',size,color);ImageDraw.Draw(image).rectangle((0,0,size[0]//2,size[1]//2),fill=(220,220,220))
            path=fixture_dir/(name+'.png');image.save(path);paths.append(str(path))
        emit('loading_actual_qwen_vision_only',revision=row['sha'],required_shards=required,device='cpu',dtype='float32')
        vision=build_frozen_vision(str(snapshot),torch.device('cpu'),torch.float32)
        if vision.backend!='qwen2_5_vl' or vision.feature_dim!=3584:raise ValueError('Actual Qwen donor feature contract differs from planned3584D merger output')
        if any(p.requires_grad or p.device.type!='cpu' for p in vision.vision_tower.parameters()):raise ValueError('Vision donor is not entirely frozen on CPU')
        bad=[name for name,module in vision.vision_tower.named_modules() if any(word in name for word in ['language_model','embed_tokens','lm_head']) or 'ForCausalLM' in type(module).__name__]
        if bad:raise ValueError('A donor language module was instantiated during strict vision loading')
        first=vision.encode_paths(paths);second=vision.encode_paths(paths[::-1])
        errors=[]
        for index in range(2):
            actual=first[index]['features'];reversed_result=second[1-index]['features']
            torch.testing.assert_close(actual,reversed_result,atol=1e-5,rtol=1e-5)
            errors.append((actual-reversed_result).abs().max().item())
            grid=first[index]['metadata']['image_grid_thw']
            expected=grid[0]*grid[1]*grid[2]//4
            if actual.shape!=(expected,3584) or not torch.isfinite(actual).all():raise ValueError('Qwen spatial merger/grid metadata does not match actual output')
        if first[0]['metadata']['image_grid_thw']!=[1,6,8] or first[1]['metadata']['image_grid_thw']!=[1,8,6]:
            raise ValueError('Synthetic aspect ratio/image ordering metadata was not preserved')
        cache=VisionFeatureCache(ROOT/'cache/vision-cpu-probes',encoder_fingerprint(vision.fingerprint_config),reserve_bytes=120*1024**3,ram_max_bytes=0)
        cached=cache.get_many(paths,vision.encode_paths)
        reordered=cache.get_many(paths[::-1],lambda paths:(_ for _ in ()).throw(AssertionError('Unexpected re-encoding on read-only second pass')),read_only=True)
        for index in range(2):
            torch.testing.assert_close(cached[index],first[index]['features'],atol=0,rtol=0)
            torch.testing.assert_close(reordered[1-index],cached[index],atol=0,rtol=0)
        processor=vision.processor.image_processor
        processor_dict=processor.to_dict()
        minimum=getattr(processor,'min_pixels',None) or processor_dict.get('min_pixels')
        maximum=getattr(processor,'max_pixels',None) or processor_dict.get('max_pixels')
        # Different Transformers image-processor APIs may expose these limits as
        # size.shortest_edge/longest_edge; preserve exact actual values in report.
        size=getattr(processor,'size',{})
        if minimum is None:minimum=size.get('shortest_edge')
        if maximum is None:maximum=size.get('longest_edge')
        if minimum is None or maximum is None:raise ValueError('Could not establish actual native Qwen pixel budget')
        examples=[]
        for width,height in [(112,84),(1024,1024),(1536,1536),(2048,2048),(4096,4096)]:
            new_height,new_width=smart_resize(height,width,factor=28,min_pixels=minimum,max_pixels=maximum)
            tokens=(new_height//28)*(new_width//28)
            examples.append({'original_size':[width,height],'resized_size':[new_width,new_height],
                             'visual_tokens':tokens,'exceeds_4096_without_any_text':tokens>=4096})
        require_gpu_hidden(torch)
        report.update(status='passed',revision=row['sha'],snapshot=str(snapshot),backend=vision.backend,feature_dim=vision.feature_dim,
            vision_class=type(vision.vision_tower).__name__,vision_parameter_count=sum(p.numel() for p in vision.vision_tower.parameters()),
            all_vision_parameters_frozen=True,all_vision_parameters_cpu=True,no_language_model_instantiated=True,
            required_weight_files=required,unneeded_weight_shards_absent=excluded,
            feature_shapes=[list(row['features'].shape) for row in first],feature_metadata=[row['metadata'] for row in first],
            reversed_batch_max_abs_differences=errors,all_features_finite=True,read_only_cache_roundtrip_exact=True,
            cache_stats=cache.stats,fingerprint_config=vision.fingerprint_config,
            native_processor={'min_pixels':minimum,'max_pixels':maximum,'patch_size':14,'merge_size':2,'temporal_patch_size':2,'processor_class':type(processor).__name__},
            native_resolution_token_examples=examples,
            context_limit_note='Native max_pixels permits16384visualtokens, exceeding4k text+vision context. Future Qwen research profile must explicitly bound image/visual-token budget and reserve room for prompt/answer; current SigLIP profile was not modified.',
            elapsed_seconds=time.monotonic()-started,gpu_initialized=False)
        _atomic_json(output,json_ready(report));emit('actual_qwen_vision_cpu_validation_passed',output=str(output),feature_shapes=report['feature_shapes'],elapsed_seconds=report['elapsed_seconds'])
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        _atomic_json(output,json_ready(report));emit('actual_qwen_vision_cpu_validation_failed',error_type=type(error).__name__,error=str(error));raise

if __name__=='__main__':main()
