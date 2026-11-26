#!/usr/bin/env python3
"""One-time supervised, checkpoint-boundary migration to an audited SFT profile."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.research_storage import _atomic_json


def main():
    run=ROOT/'runs/research/smallthinker4__siglip2__A0_bf16__pilot_s42'
    request=ROOT/'configs/research/scienceqa_handoff.json'
    spec=json.loads(request.read_text())
    receipt=ROOT/'research_results/scienceqa_handoff.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('status')=='completed':return
    expected_owner=json.loads((run/'.checkpoint_manager.json').read_text())['owner']
    while True:
        pointer=run/'last'
        if pointer.is_symlink():
            marker=json.loads((pointer/'.research_checkpoint.json').read_text())
            if marker['owner']!=expected_owner:raise ValueError('Checkpoint ownership changed during handoff')
            progress_path=ROOT/'logs/research/progress.json'
            progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
            fresh=((progress.get('run_id')==run.name and progress.get('step',0)<=marker['step']+1) or (run/'completed.json').is_file())
            if fresh and marker['step']>=spec['checkpoint_min_step'] and all((ROOT/name).is_file() for name in spec['required_audit_files']):break
        status=subprocess.run(['supervisorctl','status','research-pilot'],text=True,capture_output=True,check=False)
        if 'RUNNING' not in status.stdout:raise RuntimeError('Pilot is no longer live; inspect its state before handoff')
        print(json.dumps({'event':'handoff_waiting_for_checkpoint','minimum_step':spec['checkpoint_min_step'],'time':time.time()}),flush=True)
        time.sleep(10)
    # All future SFT jobs get new immutable configs. A0 resumes identical data,
    # seed and optimizer from the checkpoint; no running Python code is patched.
    for filename in spec['required_audit_files']:
        if not (ROOT/filename).is_file():raise FileNotFoundError(filename)
    saved_step=marker['step']
    print(json.dumps({'event':'handoff_stopping_controller','saved_step':saved_step,'time':time.time()}),flush=True)
    subprocess.run(['supervisorctl','stop','research-pilot'],check=True)
    config=Path('/etc/supervisor/conf.d/research-pilot.conf')
    text=config.read_text()
    lines=[]
    for line in text.splitlines():
        if line.startswith('command='):
            line='command=/venv/main/bin/python -u scripts/research_campaign.py --pilot-data datasets/research/pilot-en-scienceqa-disjoint-v1 --alignment-data datasets/research/pilot-en-v1 --queue-tag scienceqa_disjoint_v1'
        lines.append(line)
    config.write_text('\n'.join(lines)+'\n')
    subprocess.run(['supervisorctl','reread'],check=True)
    subprocess.run(['supervisorctl','update','research-pilot'],check=True)
    status=subprocess.run(['supervisorctl','status','research-pilot'],text=True,capture_output=True,check=False)
    if 'STOPPED' in status.stdout or 'EXITED' in status.stdout:
        subprocess.run(['supervisorctl','start','research-pilot'],check=True)
    _atomic_json(receipt,{'status':'completed','time':time.time(),'source_owner':expected_owner,
        'resume_checkpoint_step':saved_step,'data_profile':'pilot-en-scienceqa-disjoint-v1',
        'caption_profile':'pilot-en-v1','reason':'Exclude full official ScienceQA validation/test image groups before SFT'})

if __name__=='__main__':main()
