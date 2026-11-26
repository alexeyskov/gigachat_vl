#!/usr/bin/env python3
"""Evict only clean inactive checkpoint file-cache; keep the32GiB guard intact.

No weights are removed, no cgroups/sysctls are changed, no GPU signals sent.
An all-UID fd/maps scan excludes files currently used by any process.
"""
import hashlib,json,os,time,sys,re
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.research_archival_stock_queue_v5 import scan_anchor_users
from src.utils.research_storage import _atomic_json
OUT=ROOT/'research_results/control_tail_memory_v1'
PLAN=ROOT/'configs/research/fixed_capacity_eval_plan_v5.json'
EXPECTED='aafd990e6325498ec9c34c5561cade44fe88be6829cfe1fff032a5ddda6eea06'
GiB=2**30
# Replay first scans the40GiB mmap checkpoint, then needs~44.8GiB extra
# host memory plus32GiB headroom for construction. Leave room for BOTH,
# including faulted checkpoint pages, rather than only the constructor.
RECLAIM_BELOW_GIB=128
RECLAIM_TARGET_GIB=136

def memory():
 current=int(Path('/sys/fs/cgroup/memory.current').read_text());limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
 stat={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
 return dict(current_GiB=current/GiB,available_GiB=(limit-current)/GiB,anonymous_GiB=stat['anon']/GiB,file_cache_GiB=stat['file']/GiB)

def candidates(plan):
 runs=[Path(j['output']) for j in plan['jobs'] if j['stage']=='SFT']
 # A completed marker and the all-UID fd/maps scan below protect the last
 # control too. Excluding it after completion stranded its40GiB page cache.
 runs += [Path(c['run']) for c in plan['controls']]
 seen=set()
 for run in runs:
  if not (run/'completed.json').is_file() or not (run/'last/state.pt').is_file():continue
  owner=json.loads((run/'.checkpoint_manager.json').read_text())['owner']
  for role in ('best','last'):
   pointer=run/role
   if not pointer.is_symlink():raise ValueError('Owned best/last symlink required')
   obj=pointer.resolve(strict=True)
   if obj.parent!=run/'.checkpoint_objects' or json.loads((obj/'.research_checkpoint.json').read_text())['owner']!=owner:raise ValueError('Checkpoint ownership mismatch')
   path=obj/'state.pt';st=path.stat();key=(st.st_dev,st.st_ino)
   if key not in seen:seen.add(key);yield path,st
 # Pinned, already downloaded HF model files can also hold unused pages.
 # Never download here or advise an open/mapped source. Validate exact
 # inventory sizes and keep every resolved file inside the configured hub.
 hub=Path(os.environ.get('HF_HOME','/workspace/.hf_home'))/'hub'
 wanted={'Tiiny/SmallThinker-4BA0.6B-Instruct','LiquidAI/LFM2.5-8B-A1B',
  'ai-sage/GigaChat3.1-10B-A1.8B-bf16','google/siglip2-so400m-patch14-384',
  'Qwen/Qwen2.5-VL-7B-Instruct'}
 for item in json.loads((ROOT/'research_results/model_inventory.json').read_text()):
  if item['repo'] not in wanted:continue
  snapshot=hub/('models--'+item['repo'].replace('/','--'))/'snapshots'/item['sha']
  for entry in item['files']:
   if not entry['path'].endswith('.safetensors'):continue
   candidate=snapshot/entry['path']
   if not candidate.exists():continue
   path=candidate.resolve(strict=True);path.relative_to(hub.resolve())
   st=path.stat()
   if not path.is_file() or st.st_size!=entry['bytes']:raise ValueError('Pinned HF source size changed')
   key=(st.st_dev,st.st_ino)
   if key not in seen:seen.add(key);yield path,st

def reclaim(plan):
 before=memory();actions=[];skipped=[]
 for path,original in candidates(plan):
  if memory()['available_GiB']>=RECLAIM_TARGET_GIB:break
  try:scan_anchor_users({(original.st_dev,original.st_ino)})
  except (RuntimeError,PermissionError) as error:
   skipped.append(dict(path=str(path),reason=str(error)));continue
  fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   actual=os.fstat(fd)
   if (actual.st_dev,actual.st_ino,actual.st_size,actual.st_mtime_ns)!=(original.st_dev,original.st_ino,original.st_size,original.st_mtime_ns):raise ValueError('Candidate changed')
   os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
  finally:os.close(fd)
  actions.append(dict(path=str(path),bytes=actual.st_size))
 # Native replay scans its mmap weights before constructing the model. Its
 # pages are then legitimately mapped and must stay alone. Clean feature
 # cache pages are the other large contributor and are safe to drop without
 # deleting features or running an encoder again. Use bounded inode batches
 # so the all-UID scanner never exceeds the OS argument-size limit.
 if memory()['available_GiB']<RECLAIM_TARGET_GIB:
  paths=[]
  for path in (ROOT/'cache/vision').rglob('*.pt'):
   if (path.is_symlink() or not re.fullmatch('[0-9a-f]{64}',path.stem)
    or path.parent.name!=path.stem[:2] or not re.fullmatch('[0-9a-f]{64}',path.parent.parent.name)):continue
   if path.resolve()!=path.absolute():raise ValueError('Feature cache escapes ordinary local path')
   paths.append((path,path.stat()))
  count=total=0
  for start in range(0,len(paths),256):
   if memory()['available_GiB']>=RECLAIM_TARGET_GIB:break
   batch=paths[start:start+256]
   try:scan_anchor_users({(s.st_dev,s.st_ino) for _,s in batch})
   except (RuntimeError,PermissionError) as error:
    skipped.append(dict(scope='feature_cache_batch',offset=start,reason=str(error)));continue
   for path,original in batch:
    if memory()['available_GiB']>=RECLAIM_TARGET_GIB:break
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
     actual=os.fstat(fd)
     if (actual.st_dev,actual.st_ino,actual.st_size,actual.st_mtime_ns)!=(original.st_dev,original.st_ino,original.st_size,original.st_mtime_ns):raise ValueError('Feature cache changed before advice')
     os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    finally:os.close(fd)
    count+=1;total+=actual.st_size
  if count:actions.append(dict(scope='clean_inactive_feature_cache',files=count,bytes=total,feature_files_deleted=0))
 after=memory()
 value=dict(event='inactive_checkpoint_cache_advice',before=before,after=after,advised=actions,skipped=skipped,
  measured_available_gain_GiB=after['available_GiB']-before['available_GiB'],checkpoint_bytes_deleted=0,GPU_worker_signals_sent=0,time_unix=time.time())
 OUT.mkdir(parents=True,exist_ok=True);_atomic_json(OUT/'status.json',value)
 with (OUT/'actions.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
 print(json.dumps({k:v for k,v in value.items() if k not in ('advised','skipped')}),flush=True)
 return value

def main():
 if hashlib.sha256(PLAN.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Exact current experiment plan required')
 plan=json.loads(PLAN.read_text())
 while True:
  state=json.loads((ROOT/'runs/research/additive_visual_v1/queue_state.json').read_text())
  if state.get('status')=='completed':return
  if memory()['available_GiB']<RECLAIM_BELOW_GIB:reclaim(plan)
  time.sleep(15)

if __name__=='__main__':main()
