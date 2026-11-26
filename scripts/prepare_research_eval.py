#!/usr/bin/env python3
"""Prepare a bounded held-out benchmark subset and audit training image overlap."""
from pathlib import Path
import argparse
import json
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.eval.research_eval_data import SOURCE_SPECS,prepare_evaluation,derive_scoring_protocols,full_scienceqa_overlap,derive_scienceqa_disjoint_training
from src.dataset.vision_cache import ensure_disk_space

ORIGINAL_METADATA={
'ScienceQA-original-problems.json':'https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/problems.json',
'ScienceQA-original-pid_splits.json':'https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/pid_splits.json',
'ChartQA-original-test_human.json':'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/test/test_human.json',
'ChartQA-original-test_augmented.json':'https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/test/test_augmented.json'}


def prepare_sources(root,*,download=False,reserve_bytes=120*1024**3,max_download_bytes=2_000_000_000):
    from huggingface_hub import HfApi,hf_hub_download
    root=Path(root);metadata=root/'eval-metadata';metadata.mkdir(parents=True,exist_ok=True)
    missing=[];downloaded=0
    for spec in SOURCE_SPECS.values():
        path=metadata/(spec['repo'].replace('/','--')+'.json')
        if not path.exists():
            if not download: raise FileNotFoundError(f'Missing metadata {path}; pass --download')
            info=HfApi().dataset_info(spec['repo'],files_metadata=True)
            path.write_text(json.dumps({'repo':spec['repo'],'revision':info.sha,'files':[{'path':file.rfilename,'size':file.size} for file in info.siblings if file.rfilename.endswith('.parquet')]}))
        meta=json.loads(path.read_text())
        for name in spec['files']:
            local=root/'eval-raw'/spec['repo'].replace('/','--')/name
            size=next(file['size'] for file in meta['files'] if file['path']==name)
            if not local.exists(): missing.append((spec,meta,name,size,local))
            elif local.stat().st_size!=size: raise ValueError(f'Source shard size mismatch: {local}')
    anticipated=sum(item[3] for item in missing)
    if anticipated>max_download_bytes: raise ValueError('Bounded evaluation download budget exceeded')
    ensure_disk_space(root,anticipated,reserve_bytes)
    for spec,meta,name,size,local in missing:
        if not download: raise FileNotFoundError(f'Missing {local}; pass --download')
        ensure_disk_space(root,size,reserve_bytes)
        print(json.dumps({'event':'eval_download','repo':spec['repo'],'file':name,'bytes':size}),flush=True)
        hf_hub_download(spec['repo'],name,repo_type='dataset',revision=meta['revision'],local_dir=str(root/'eval-raw'/spec['repo'].replace('/','--')))
        downloaded+=size
    for name,url in ORIGINAL_METADATA.items():
        path=metadata/name
        if not path.exists():
            if not download: raise FileNotFoundError(f'Missing original release metadata {path}; pass --download')
            data=urllib.request.urlopen(url,timeout=60).read()
            if downloaded+len(data)>max_download_bytes: raise ValueError('Original release metadata exceeds download budget')
            ensure_disk_space(root,len(data),reserve_bytes)
            path.write_bytes(data);downloaded+=len(data)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',default='datasets/research')
    parser.add_argument('--output',default='datasets/research/eval-heldout-v3')
    parser.add_argument('--protocol-parent',help='Create a protocol-only derivative, reusing source image paths')
    parser.add_argument('--full-scienceqa-overlap-only',action='store_true')
    parser.add_argument('--disjoint-training-parent')
    parser.add_argument('--scienceqa-audit-dir',default='datasets/research/eval-scienceqa-full-overlap-v1')
    parser.add_argument('--train-root',default='datasets/research/pilot-en-v1')
    parser.add_argument('--train-manifest',action='append')
    parser.add_argument('--per-benchmark',type=int,default=64)
    parser.add_argument('--candidate-multiplier',type=int,default=2)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--reserve-gib',type=float,default=120)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    reserve=int(args.reserve_gib*1024**3)
    if args.disjoint_training_parent:
        print(json.dumps(derive_scienceqa_disjoint_training(args.disjoint_training_parent,args.scienceqa_audit_dir,args.output),indent=2));return
    if args.protocol_parent:
        print(json.dumps(derive_scoring_protocols(args.protocol_parent,args.output),indent=2));return
    prepare_sources(args.root,download=args.download,reserve_bytes=reserve)
    manifests=args.train_manifest or [str(Path(args.train_root)/f'{stage}_{partition}.jsonl') for stage in ['alignment','sft'] for partition in ['train','validation']]
    if args.full_scienceqa_overlap_only:
        print(json.dumps(full_scienceqa_overlap(args.root,args.output,manifests,workers=args.workers,reserve_bytes=reserve),indent=2));return
    print(json.dumps(prepare_evaluation(args.root,args.output,manifests,per_benchmark=args.per_benchmark,
            candidate_multiplier=args.candidate_multiplier,seed=args.seed,workers=args.workers,reserve_bytes=reserve),indent=2))

if __name__=='__main__': main()
