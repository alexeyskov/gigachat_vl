#!/usr/bin/env python3
"""Build an immutable finite 35/30/20/15 MM mix with 15% text replay."""
from pathlib import Path
import argparse
import hashlib
import json
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dataset.vision_cache import ensure_disk_space

WEIGHTS={'general_vqa':.35,'ocr_documents':.30,'charts_diagrams':.20,'knowledge_reasoning':.15}

def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def sample_pool(pool,count,rng,max_repeat=2):
    if count>len(pool)*max_repeat:raise ValueError(f'Excessive oversampling: {count}/{len(pool)}')
    result=[]
    while len(result)<count:
        cycle=list(pool);rng.shuffle(cycle)
        result+=cycle[:count-len(result)]
    return result

def build(mm,text,seed=42,text_fraction=.15,max_repeat=2):
    if not 0<=text_fraction<1:raise ValueError('Invalid text fraction')
    rng=random.Random(seed)
    groups={key:[] for key in WEIGHTS}
    for record in mm:
        groups[record['task_group']].append(record)
    if any(not v for v in groups.values()):raise ValueError('All four task groups are required')
    total=len(mm);chosen=[]
    quotas={k:round(total*w) for k,w in WEIGHTS.items()}
    quotas['general_vqa']+=total-sum(quotas.values())
    for group,count in quotas.items():chosen+=sample_pool(groups[group],count,rng,max_repeat)
    text_count=round(total*text_fraction/(1-text_fraction))
    chosen+=sample_pool(text,text_count,rng,max_repeat)
    rng.shuffle(chosen)
    return chosen,{'multimodal_records':total,'text_records':text_count,'group_quotas':quotas,
                   'unique_records':len({r['record_id'] for r in chosen}),'total_records':len(chosen),
                   'max_source_repeat':max_repeat,'seed':seed,'text_fraction':text_fraction}

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--data-dir',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--reserve-gb',type=float,default=120);args=p.parse_args()
    data=Path(args.data_dir);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    ensure_disk_space(out,32*2**20,int(args.reserve_gb*2**30))
    reports={};source_hashes={}
    for split in ['train','validation']:
        mm_path=data/f'sft_{split}.jsonl';text_path=data/f'text_{split}.jsonl'
        for path in [mm_path,text_path]:source_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows,report=build(read(mm_path),read(text_path),args.seed)
        raw=''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows)
        target=out/f'sft_{split}.jsonl'
        if target.exists() and target.read_text()!=raw:raise ValueError('Refusing to overwrite an existing different mixture')
        target.write_text(raw);reports[split]=dict(report,sha256=hashlib.sha256(raw.encode()).hexdigest())
    (out/'mix_audit.json').write_text(json.dumps({'source_hashes':source_hashes,'splits':reports},indent=2))
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
