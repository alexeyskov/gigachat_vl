#!/usr/bin/env python3
"""Download the complete selected held-out split allowlist, then audit image overlap.

The extra-download budget is explicit. No GPU work, publication or modification
of current training profiles is performed. Image signatures are streamed/cached;
full benchmark image copies are not extracted.
"""
from pathlib import Path
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor,as_completed

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.eval.full_heldout_audit import selected_source_files,full_heldout_audit,emit
from src.dataset.vision_cache import ensure_disk_space


def download_full_sources(root,*,max_extra_bytes=9*1024**3,reserve_bytes=120*1024**3,workers=3):
    from huggingface_hub import hf_hub_download
    root=Path(root).resolve();sources=selected_source_files(root);missing=[]
    for source in sources.values():
        for files in source['splits'].values():
            for file in files:
                path=root/'eval-raw'/source['repo'].replace('/','--')/file['path']
                if path.exists():
                    if path.stat().st_size!=file['size']:raise ValueError(f'Unexpected existing raw file size: {path}')
                else:missing.append((source,file))
    additional=sum(file['size'] for _,file in missing)
    if additional>max_extra_bytes:raise ValueError(f'Additional download budget exceeded: {additional}>{max_extra_bytes}')
    ensure_disk_space(root,additional+128*1024**2,reserve_bytes)
    emit('full_audit_download_preflight',missing_files=len(missing),additional_bytes=additional,max_extra_bytes=max_extra_bytes)
    completed=0;started=time.monotonic()
    def fetch(item):
        source,file=item
        ensure_disk_space(root,file['size']+16*1024**2,reserve_bytes)
        emit('full_audit_download_start',repo=source['repo'],revision=source['revision'],file=file['path'],bytes=file['size'])
        path=hf_hub_download(source['repo'],file['path'],repo_type='dataset',revision=source['revision'],local_dir=str(root/'eval-raw'/source['repo'].replace('/','--')))
        if Path(path).stat().st_size!=file['size']:raise ValueError('Downloaded pinned shard size does not match metadata')
        ensure_disk_space(root,0,reserve_bytes)
        return source,file
    if workers<1:raise ValueError('Download worker count must be positive')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fetch,item) for item in missing]
        for future in as_completed(futures):
            source,file=future.result();completed+=file['size']
            emit('full_audit_download_done',repo=source['repo'],file=file['path'],completed_bytes=completed,total_bytes=additional,elapsed_seconds=time.monotonic()-started)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',default='datasets/research')
    parser.add_argument('--output',default='datasets/research/eval-full-overlap-v1')
    parser.add_argument('--train-root',default='datasets/research/pilot-en-v1')
    parser.add_argument('--max-extra-gib',type=float,default=9)
    parser.add_argument('--reserve-gib',type=float,default=120)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--download-workers',type=int,default=3)
    args=parser.parse_args()
    reserve=int(args.reserve_gib*1024**3)
    download_full_sources(args.root,max_extra_bytes=int(args.max_extra_gib*1024**3),reserve_bytes=reserve,workers=args.download_workers)
    manifests=[str(Path(args.train_root)/f'{stage}_{partition}.jsonl') for stage in ['alignment','sft'] for partition in ['train','validation']]
    result=full_heldout_audit(args.root,args.output,manifests,workers=args.workers,reserve_bytes=reserve)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
