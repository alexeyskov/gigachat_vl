#!/usr/bin/env python3
import hashlib,json,os,shutil,time,zipfile
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'datasets/research/detection-v1/raw'
URL='https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'

def main():
 import requests
 OUT.mkdir(parents=True,exist_ok=True)
 target=OUT/'annotations_trainval2017.zip';temporary=target.with_suffix('.zip.part')
 if not target.exists():
  if shutil.disk_usage(OUT).free<32*2**30+2*2**30:raise OSError('Reserve32GiB plus bounded2GiB dataset')
  with requests.get(URL,stream=True,timeout=(30,60)) as response:
   response.raise_for_status();total=int(response.headers['Content-Length']);written=0;start=time.monotonic();last=start
   if total>300*2**20:raise ValueError('Unexpected annotation download size')
   with temporary.open('wb') as f:
    for chunk in response.iter_content(2**20):
     f.write(chunk);written+=len(chunk)
     if time.monotonic()-last>20:print(json.dumps(dict(event='annotations_download',MiB=round(written/2**20,1),total_MiB=round(total/2**20,1))),flush=True);last=time.monotonic()
   if written!=total:raise ValueError('Short annotation download')
  temporary.replace(target)
 digest=hashlib.sha256(target.read_bytes()).hexdigest()
 with zipfile.ZipFile(target) as archive:
  for name in ('instances_train2017.json','instances_val2017.json'):
   out=OUT/name
   if not out.exists():
    tmp=out.with_suffix('.json.part')
    with archive.open('annotations/'+name) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,2**20)
    tmp.replace(out)
 (OUT/'source.json').write_text(json.dumps(dict(url=URL,official_catalog='https://cocodataset.org/#download',zip_sha256=digest,
  zip_bytes=target.stat().st_size,annotation_files={name:dict(bytes=(OUT/name).stat().st_size,sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest()) for name in ('instances_train2017.json','instances_val2017.json')}),indent=2)+'\n')
 print(json.dumps(dict(status='annotations_ready',zip_sha256=digest)),flush=True)

if __name__=='__main__':main()
