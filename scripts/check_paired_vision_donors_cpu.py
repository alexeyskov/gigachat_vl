#!/usr/bin/env python3
"""Actual CPU feature smoke for all three frozen donors on identical RGB fixtures."""
from pathlib import Path
import gc
import hashlib
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.research_storage import _atomic_json
from scripts.check_research_backbone_cpu import require_gpu_hidden,json_ready


def emit(event,**fields):print(json.dumps({'time':time.time(),'event':event,**fields}),flush=True)


def main():
    import torch
    from huggingface_hub import hf_hub_download
    from src.model.research_vision import build_frozen_vision
    from src.dataset.vision_cache import image_content_hash,encoder_fingerprint,VisionFeatureCache
    require_gpu_hidden(torch);torch.set_num_threads(8);torch.set_num_interop_threads(1)
    inventory=json.loads((ROOT/'research_results/model_inventory.json').read_text())
    repos=['google/siglip2-so400m-patch14-384','Qwen/Qwen2.5-VL-7B-Instruct','google/gemma-4-E4B-it']
    paths=[str(ROOT/'research_results/vision_probes/qwen25'/name) for name in ['landscape.png','portrait.png']]
    if not all(Path(path).exists() for path in paths):raise FileNotFoundError('Create the actual Qwen CPU fixtures before paired donor validation')
    target=ROOT/'research_results/paired_vision_donors_cpu_validation.json'
    report={'schema_version':1,'status':'running','scope':'actual native frozen vision-only CPU features on two identical synthetic images',
            'input_paths':paths,'input_image_hashes':[image_content_hash(path) for path in paths],
            'donors':[],'gpu_training_validated':False,'current_siglip_profile_modified':False}
    started=time.monotonic()
    try:
        for repo in repos:
            row=next(item for item in inventory if item['repo']==repo)
            snapshot=Path(hf_hub_download(repo,'config.json',revision=row['sha'],local_files_only=True)).parent
            emit('paired_donor_cpu_start',repo=repo,revision=row['sha'])
            vision=build_frozen_vision(str(snapshot),torch.device('cpu'),torch.float32)
            expected=1152 if repo.startswith('google/siglip') else 3584 if repo.startswith('Qwen/') else 2560
            if vision.feature_dim!=expected:raise ValueError(f'Unexpected actual feature width for {repo}')
            modules=[m for m in [vision.vision_tower,vision.vision_projector] if m is not None]
            if any(p.requires_grad or p.device.type!='cpu' for m in modules for p in m.parameters()):raise ValueError('Donor modules are not frozen CPU-only')
            bad=[name for module in modules for name,submodule in module.named_modules() if any(x in name for x in ['language_model','embed_tokens','lm_head']) or 'ForCausalLM' in type(submodule).__name__]
            if bad:raise ValueError(f'Donor loaded language modules: {bad}')
            first=vision.encode_paths(paths);reversed_batch=vision.encode_paths(paths[::-1])
            diffs=[];stats=[]
            for i,record in enumerate(first):
                tensor=record['features']
                if tensor.ndim!=2 or tensor.shape[1]!=expected or not torch.isfinite(tensor).all():raise ValueError('Invalid actual donor feature tensor')
                torch.testing.assert_close(tensor,reversed_batch[1-i]['features'],atol=1e-5,rtol=1e-5)
                diffs.append((tensor-reversed_batch[1-i]['features']).abs().max().item())
                stats.append({'shape':list(tensor.shape),'mean':tensor.float().mean().item(),'std':tensor.float().std().item(),
                              'sha256_float32':hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest(),
                              'metadata':record['metadata']})
            cache_fp=encoder_fingerprint(vision.fingerprint_config)
            cache=VisionFeatureCache(ROOT/'cache/vision-paired-cpu-probes',cache_fp,reserve_bytes=120*1024**3,ram_max_bytes=0)
            saved=cache.get_many(paths,vision.encode_paths)
            restored=cache.get_many(paths[::-1],lambda paths:(_ for _ in ()).throw(AssertionError('Unexpected second-pass encode')),read_only=True)
            for i in range(2):torch.testing.assert_close(saved[i],restored[1-i],atol=0,rtol=0)
            entry={'repo':repo,'revision':row['sha'],'status':'passed','backend':vision.backend,'feature_dim':vision.feature_dim,
                   'vision_class':type(vision.vision_tower).__name__,'donor_embedder_class':type(vision.vision_projector).__name__ if vision.vision_projector is not None else None,
                   'vision_parameter_count':sum(p.numel() for m in modules for p in m.parameters()),
                   'language_model_instantiated':False,'all_vision_parameters_frozen':True,'all_modules_cpu':True,
                   'feature_rows':stats,'reversed_image_order_max_abs_diffs':diffs,'cache_roundtrip_exact':True,
                   'cache_fingerprint':cache_fp,'cache_stats':cache.stats,'preprocessing_profile':None}
            report['donors'].append(entry);_atomic_json(target,json_ready(report))
            emit('paired_donor_cpu_passed',repo=repo,feature_shapes=[s['shape'] for s in stats],parameters=entry['vision_parameter_count'])
            del vision,first,reversed_batch,cache,saved,restored,modules
            gc.collect()
        if len({row['cache_fingerprint'] for row in report['donors']})!=len(repos):raise ValueError('Different donor feature spaces shared a cache fingerprint')
        require_gpu_hidden(torch)
        report.update(status='passed',elapsed_seconds=time.monotonic()-started,gpu_initialized=False,
                      limits='Same RGB inputs and frozen native feature boundaries validated. Different token counts/widths/resolutions are not a matched-compute comparison or a VLM accuracy benchmark.')
        _atomic_json(target,json_ready(report));emit('paired_all_donors_cpu_passed',report=str(target),elapsed_seconds=report['elapsed_seconds'])
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        _atomic_json(target,json_ready(report));emit('paired_donor_cpu_failed',error_type=type(error).__name__,error=str(error));raise

if __name__=='__main__':main()
