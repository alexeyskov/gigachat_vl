#!/usr/bin/env python3
"""Download one pinned research model, with per-file disk reservation and resume."""
from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.research_storage import DiskBudget,_atomic_json


def emit(event,**fields):
    print(json.dumps({'time':time.time(),'event':event,**fields},ensure_ascii=False),flush=True)


def native_assets(row):
    return [file for file in row['files'] if '/' not in file['path'] and
            (Path(file['path']).suffix in {'.json','.safetensors','.jinja','.model'} or file['path']=='merges.txt')]


def vision_scope_assets(row,metadata_dir):
    from src.model.research_vision import required_vision_weight_files
    config=json.loads((Path(metadata_dir)/'config.json').read_text())
    model_type=config['model_type']
    if model_type not in {'qwen2_5_vl','gemma4','siglip','siglip2','siglip_vision_model','siglip2_vision_model'}:
        raise ValueError('Vision-only download requested for a non-vision model')
    backend='siglip2' if model_type.startswith('siglip2') else 'siglip' if model_type.startswith('siglip') else model_type
    required=required_vision_weight_files(metadata_dir,backend)
    allowed=native_assets(row)
    if not set(required).issubset({file['path'] for file in allowed}):
        raise ValueError('Vision shard index references files absent from pinned inventory')
    selected=[file for file in allowed if not file['path'].endswith('.safetensors') or file['path'] in required]
    excluded=[file['path'] for file in allowed if file['path'].endswith('.safetensors') and file['path'] not in required]
    return selected,required,excluded


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--repo',required=True)
    parser.add_argument('--inventory',default='research_results/model_inventory.json')
    parser.add_argument('--reserve-gb',type=float,default=120)
    parser.add_argument('--vision-only',action='store_true')
    parser.add_argument('--max-new-gib',type=float,help='Additional-byte cap, including metadata downloaded in this invocation')
    args=parser.parse_args()
    from huggingface_hub import hf_hub_download,try_to_load_from_cache,constants
    inventory=json.loads(Path(args.inventory).read_text())
    rows=[row for row in inventory if row['repo']==args.repo]
    if len(rows)!=1: raise ValueError('Requested repository must occur exactly once in pinned inventory')
    row=rows[0]
    # Top-level native model assets only; never download original/duplicate formats.
    files=native_assets(row)
    if not files: raise ValueError('No allowed model assets in inventory')
    root=Path(constants.HF_HUB_CACHE);root.mkdir(parents=True,exist_ok=True)
    budget=DiskBudget(root,args.reserve_gb)
    def cached(file):
        path=try_to_load_from_cache(args.repo,file['path'],revision=row['sha'])
        return isinstance(path,str) and Path(path).is_file() and Path(path).stat().st_size==file['bytes']
    metadata_downloaded=0;required_weights=[];excluded_weights=[]
    if args.vision_only:
        metadata=[file for file in files if not file['path'].endswith('.safetensors')]
        if not any(file['path']=='config.json' for file in metadata): raise ValueError('Pinned donor inventory lacks config.json')
        missing_metadata=sum(file['bytes'] for file in metadata if not cached(file))
        if args.max_new_gib is not None and missing_metadata>args.max_new_gib*2**30: raise ValueError('Metadata exceeds extra download cap')
        budget.ensure_space(missing_metadata+64*2**20)
        for file in metadata:
            if not cached(file):
                budget.ensure_space(file['bytes']+16*2**20)
                emit('vision_metadata_download',repo=args.repo,path=file['path'],bytes=file['bytes'])
                path=hf_hub_download(args.repo,filename=file['path'],revision=row['sha'],force_download=False)
                if Path(path).stat().st_size!=file['bytes']: raise ValueError('Metadata size differs from pinned inventory')
                metadata_downloaded+=file['bytes']
        config_path=try_to_load_from_cache(args.repo,'config.json',revision=row['sha'])
        files,required_weights,excluded_weights=vision_scope_assets(row,Path(config_path).parent)
    missing=[file for file in files if not cached(file)]
    anticipated=sum(file['bytes'] for file in missing)
    if args.max_new_gib is not None and metadata_downloaded+anticipated>args.max_new_gib*2**30:
        raise ValueError('Selected model download exceeds explicit additional-byte cap')
    budget.ensure_space(math.ceil(anticipated*1.05)+64*2**20)
    emit('download_preflight',repo=args.repo,revision=row['sha'],missing_files=len(missing),anticipated_bytes=anticipated,disk=budget.report())
    started=time.monotonic();completed=[]
    for file in files:
        if cached(file):
            emit('download_cache_hit',repo=args.repo,path=file['path'],bytes=file['bytes'])
        else:
            budget.ensure_space(math.ceil(file['bytes']*1.05)+64*2**20)
            emit('download_file_start',repo=args.repo,path=file['path'],bytes=file['bytes'],disk=budget.report())
            path=hf_hub_download(args.repo,filename=file['path'],revision=row['sha'],force_download=False)
            if Path(path).stat().st_size!=file['bytes']:
                raise ValueError(f'Download size differs from pinned inventory: {file["path"]}')
            budget.ensure_space()
            emit('download_file_done',repo=args.repo,path=file['path'],bytes=file['bytes'],elapsed_seconds=time.monotonic()-started,disk=budget.report())
        completed.append(file['path'])
    report={'status':'completed','repo':args.repo,'revision':row['sha'],'files':completed,
            'partial_vision_scope':args.vision_only,'scope':'vision_only_subset' if args.vision_only else 'native_model',
            'selected_weight_files':required_weights,'excluded_weight_files':excluded_weights,
            'mixed_shards_may_include_unneeded_text_tensors':args.vision_only,
            'metadata_downloaded_bytes':metadata_downloaded,
            'elapsed_seconds':time.monotonic()-started,'disk':budget.report()}
    destination=ROOT/'research_results'/('download-'+args.repo.replace('/','--')+('-vision-only' if args.vision_only else '')+'.json')
    destination.parent.mkdir(parents=True,exist_ok=True)
    _atomic_json(destination,report)
    emit('download_completed',**report)

if __name__=='__main__':main()
