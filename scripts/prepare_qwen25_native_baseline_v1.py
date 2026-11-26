#!/usr/bin/env python3
"""Download only the pinned native Qwen baseline into a separately owned cache."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
from src.utils.research_storage import _atomic_json,DiskBudget
REPO='Qwen/Qwen2.5-VL-7B-Instruct';REV='cc594898137f460bfe9f0759e9844b3ce807cfb5'
OUT=ROOT/'cache/baselines/qwen25_native_int4_v1/source'
RECEIPTS=ROOT/'research_results/qwen25_native_int4_internal116_v1'


def main():
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download
    inventory=json.loads((ROOT/'research_results/model_inventory.json').read_bytes())
    row=next(x for x in inventory if x['repo']==REPO)
    if row['sha']!=REV:raise ValueError('Pinned original Qwen revision required')
    RECEIPTS.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    marker=OUT/'.baseline_owner.json'
    owner={'owner':'qwen25_native_int4_internal116_v1','repo':REPO,'revision':REV}
    if marker.exists() and json.loads(marker.read_bytes())!=owner:raise ValueError('Foreign baseline directory')
    _atomic_json(marker,owner)
    DiskBudget(ROOT,32).ensure_space(row['bytes']+256*2**20)
    _atomic_json(RECEIPTS/'download_intent.json',dict(owner,source=str(OUT),temporary_source=True,
        expected_bytes=row['bytes'],shared_training_weights_modified=False,created_unix=time.time()))
    def download(entry):
        name=entry['path'];target=OUT/name
        try:cached=Path(hf_hub_download(REPO,name,revision=REV,local_files_only=True))
        except Exception:cached=None
        borrowed=cached is not None and cached.is_file() and cached.stat().st_size==entry['bytes']
        if borrowed:
            if not target.exists():target.symlink_to(cached.resolve())
            if target.resolve()!=cached.resolve():raise ValueError('Existing baseline source differs')
        else:
            target=Path(hf_hub_download(REPO,name,revision=REV,local_dir=OUT))
        if target.stat().st_size!=entry['bytes']:raise ValueError('Downloaded size differs: '+name)
        # New private source pages are not part of training's memory budget.
        if not borrowed:
            fd=os.open(target,os.O_RDONLY)
            try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            finally:os.close(fd)
        result={'name':name,'path':str(target),'bytes':entry['bytes'],'borrowed_shared_file':borrowed}
        print(json.dumps({'downloaded':result}),flush=True);return result
    with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(download,row['files']))
    _atomic_json(RECEIPTS/'source_ready.json',dict(owner,status='complete_pinned_source_ready',source=str(OUT),files=files,created_unix=time.time()))
    print(json.dumps({'status':'source_ready','files':len(files),'bytes':sum(x['bytes'] for x in files)}),flush=True)


if __name__=='__main__':main()
