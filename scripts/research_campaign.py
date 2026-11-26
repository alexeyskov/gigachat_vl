#!/usr/bin/env python3
"""Managed SmallThinker pilot campaign after explicit technical smoke validation.

Does not declare the full132-cell matrix complete. No implicit failed-job retries.
"""
from pathlib import Path
import argparse
import hashlib
import zipfile
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.research_storage import DiskBudget

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--pilot-data',default='datasets/research/pilot-en-v1')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--alignment-data',help='Unchanged caption profile reused across SFT-only data revisions')
    parser.add_argument('--queue-tag',default='v1')
    args=parser.parse_args();os.chdir(ROOT)
    data=Path(args.pilot_data)
    smoke_state=Path('configs/research/smallthinker_smoke_queue.state.json')
    while True:
        status=json.loads(smoke_state.read_text()).get('status') if smoke_state.exists() else 'pending'
        if status=='completed':break
        if status=='failed':raise RuntimeError('Technical smoke failed; pilot not admitted')
        print(json.dumps({'event':'waiting_for_smoke_completion','status':status,'time':time.time()}),flush=True)
        time.sleep(30)
    while not (data/'audit.json').exists():
        DiskBudget(ROOT,120).ensure_space()
        print(json.dumps({'event':'waiting_for_pilot_data','path':str(data),'time':time.time()}),flush=True)
        time.sleep(30)
    mixture=Path('datasets/research/pilot-mixture-en-v1') if args.queue_tag=='v1' else data.parent/(data.name+'-mixture')
    alignment=Path(args.alignment_data) if args.alignment_data else data
    subprocess.run([sys.executable,'scripts/build_research_mix.py','--data-dir',str(data),'--output',str(mixture)],check=True)
    # Exact precision/data/optimizer recipe is fixed at pilot queue creation.
    jobs=[];root=Path('runs/research');seed=f's{args.seed}'
    def job(name,stage,recipe,init=None,requires=None):
        out=root/name
        if stage in ('A0','A1'):
            train=alignment/'alignment_train.jsonl';val=alignment/'alignment_validation.jsonl'
        else:train=mixture/'sft_train.jsonl';val=mixture/'sft_validation.jsonl'
        batch=64 if stage=='A0' or recipe.startswith('1') or recipe=='4q' else 16
        argv=['--stage',stage,'--recipe',recipe,'--manifest',str(train),'--validation',str(val),
              '--output',str(out),'--batch-size',str(batch),'--effective-batch-size','64','--seed',str(args.seed),
              '--eval-records','64','--eval-every','100']
        if stage=='A1':argv+=['--max-records','1000']
        if init:argv+=['--initial-checkpoint',str(root/init/'best/state.pt')]
        jobs.append({'id':name,'argv':argv,'requires':requires or [],
                     'discard_checkpoint_after_dependents':stage=='A1','compact_completed':stage=='SFT'})
        if stage=='SFT' and args.queue_tag!='v1':
            jobs[-1]['evaluation']={'manifest':'datasets/research/eval-heldout-v3/evaluation.jsonl',
                'output':str(out/'evaluation'/'heldout_v3'),'checkpoint':'best','max_new_tokens':64}
        return name
    bf=job(f'smallthinker4__siglip2__A0_bf16__pilot_{seed}','A0','4f')
    nf=job(f'smallthinker4__siglip2__A0_nf4__pilot_{seed}','A0','4q')
    # Compare actual QLoRA control first, then adapters/routers and full branches.
    tag='' if args.queue_tag=='v1' else args.queue_tag+'_'
    for recipe in ['4q','1b','1a','1c','4f','3b','3a','3c','2b','2a','2c']:
        base=nf if recipe.startswith('1') or recipe=='4q' else bf
        warm=job(f'smallthinker4__siglip2__{recipe}__A1_pilot_{tag}{seed}','A1',recipe,base,[base])
        job(f'smallthinker4__siglip2__{recipe}__SFT_pilot_{tag}{seed}','SFT',recipe,warm,[warm])
    path=Path('configs/research/smallthinker_pilot_queue.json') if args.queue_tag=='v1' else Path('configs/research')/('smallthinker_pilot_'+args.queue_tag+'_queue.json')
    spec={'queue_id':'smallthinker-siglip2-pilot-'+args.queue_tag,'reserve_gb':120,'stop_on_failure':True,'jobs':jobs}
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Refusing changed pilot queue')
    path.write_text(json.dumps(spec,indent=2))
    # Keep the actual source bytes used by this campaign, not just a dirty git SHA.
    sources=sorted(list(Path('src').rglob('*.py'))+list(Path('scripts').glob('*.py'))+list(Path('.').glob('requirements*.txt')))
    content={str(source):source.read_bytes() for source in sources}
    hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in content.items()}
    code_sha=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    archive=Path('research_results/source_snapshots')/(code_sha+'.zip');archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as handle:
            for name,raw in content.items():handle.writestr(name,raw)
    provenance={'queue_id':spec['queue_id'],'created_unix':time.time(),'source_sha256':hashes,
        'source_archive':str(archive),'code_sha256':code_sha,
        'queue_spec_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'data_profile':str(data),'alignment_profile':str(alignment),
        'sft_manifest_sha256':hashlib.sha256((mixture/'sft_train.jsonl').read_bytes()).hexdigest(),
        'scientific_scope':'pilot; not the completed primary matrix'}
    (path.with_suffix('.provenance.json')).write_text(json.dumps(provenance,indent=2))
    subprocess.run([sys.executable,'-u','scripts/research_queue.py','--spec',str(path)],check=True)
    print(json.dumps({'event':'pilot_completed','full_matrix_completed':False,'time':time.time()}),flush=True)

if __name__=='__main__':main()
