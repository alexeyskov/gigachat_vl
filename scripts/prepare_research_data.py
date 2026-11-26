#!/usr/bin/env python3
"""Download bounded, pinned train shards and materialize research manifests."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dataset.research_data import prepare_from_specs
from src.dataset.vision_cache import ensure_disk_space

DEFAULT_SOURCES = [
    ('caption','dnth/pixmo-cap-images','data','captioning','data/train-00000-of-00381.parquet'),
    ('finevision','HuggingFaceM4/FineVision','vqav2','general_vqa','vqav2/train-00000-of-00009.parquet'),
    ('finevision','HuggingFaceM4/FineVision','scienceqa','knowledge_reasoning','scienceqa/train-00000-of-00001.parquet'),
    ('finevision','HuggingFaceM4/FineVision','textvqa','ocr_documents','textvqa/train-00000-of-00005.parquet'),
    ('text','HuggingFaceTB/smoltalk','all','text_instruction','data/all/train-00000-of-00009.parquet'),
]

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',default='datasets/research')
    parser.add_argument('--output',default='datasets/research/smoke-v1')
    parser.add_argument('--caption-limit',type=int,default=1200)
    parser.add_argument('--caption-shards',type=int,default=1,help='Explicit upper bound on PixMo shard downloads; ~1535 records each')
    parser.add_argument('--include-chart',action='store_true',help='Add FineVision ChartQA group; ~403MB first shard')
    parser.add_argument('--sft-limit-per-source',type=int,default=200)
    parser.add_argument('--text-limit',type=int,default=120)
    parser.add_argument('--reserve-gib',type=float,default=120)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validation-fraction',type=float,default=.02)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    from huggingface_hub import HfApi,hf_hub_download
    root=Path(args.root); root.mkdir(parents=True,exist_ok=True)
    specs=[]; metadata={}
    sources=list(DEFAULT_SOURCES)
    if args.include_chart:
        sources.insert(-1,('finevision','HuggingFaceM4/FineVision','chartqa','charts_diagrams','chartqa/train-00000-of-00002.parquet'))
    for kind,repo,config,group,shard in sources:
        if repo not in metadata:
            metadata_path=root/(repo.split('/')[-1]+'.files.json')
            if metadata_path.exists(): meta=json.loads(metadata_path.read_text())
            else:
                info=HfApi().dataset_info(repo,files_metadata=True)
                meta={'repo':repo,'revision':info.sha,'files':[{'path':s.rfilename,'size':s.size} for s in info.siblings if s.rfilename.endswith('.parquet')]}
                metadata_path.write_text(json.dumps(meta))
            metadata[repo]=meta
        meta=metadata[repo]
        limit=args.caption_limit if kind=='caption' else args.text_limit if kind=='text' else args.sft_limit_per_source
        if not limit: continue
        shard_names = [shard]
        if kind=='caption':
            if args.caption_shards < 1: parser.error('--caption-shards must be positive')
            shard_names = [f['path'] for f in meta['files'] if f['path'].startswith('data/train-')][:args.caption_shards]
        paths=[]
        for selected in shard_names:
            path=root/'raw'/repo.replace('/','--')/selected
            if not path.exists():
                if not args.download: raise FileNotFoundError(f'{path}; run with --download')
                size=next(f['size'] for f in meta['files'] if f['path']==selected)
                ensure_disk_space(root,size,args.reserve_gib*1024**3)
                print(f'Downloading {repo}/{selected} ({size/1e6:.1f} MB)',flush=True)
                hf_hub_download(repo,filename=selected,repo_type='dataset',revision=meta['revision'],local_dir=str(root/'raw'/repo.replace('/','--')))
            paths.append(str(path.resolve()))
        specs.append({'kind':kind,'repo':repo,'revision':meta['revision'],'config':config,
                      'task_group':group,'files':paths,'limit':limit})
    result=prepare_from_specs(specs,args.output,reserve_bytes=int(args.reserve_gib*1024**3),seed=args.seed,validation_fraction=args.validation_fraction)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
