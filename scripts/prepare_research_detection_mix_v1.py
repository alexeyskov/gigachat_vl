#!/usr/bin/env python3
"""Prepare a small image-disjoint COCO prompted-detection replay mixture on CPU."""
from __future__ import annotations
import collections,gc,hashlib,json,os,shutil,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dataset.research_detection_v1 import normalize_box,canonical_answer,prompt,rank,COORDINATE_SYSTEM
from src.dataset.vision_cache import image_content_hash
BASE=ROOT/'datasets/research/detection-v1';RAW=BASE/'raw';OLD=ROOT/'datasets/research/budget18-en-v1'
MAX_BOXES=4


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def emit(event,**kwargs):print(json.dumps(dict(event=event,time=time.time(),**kwargs)),flush=True)
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def write_rows(path,values):
 tmp=path.with_suffix('.jsonl.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in values));tmp.replace(path)

def perceptual(path):
 from PIL import Image,ImageOps
 import numpy as np
 # Deterministic32x32 DCT pHash, avoiding new packages in the training env.
 k=np.arange(32)[:,None];n=np.arange(32)[None,:]
 transform=2*np.cos(np.pi*k*(2*n+1)/64)
 with Image.open(path) as im:array=np.asarray(ImageOps.exif_transpose(im).convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
 low=(transform@array@transform.T)[:8,:8];bits=(low>np.median(low)).reshape(-1)
 value=0
 for bit in bits:value=(value<<1)|int(bit)
 return value


def old_protection():
 train=[];protected=[]
 for name in ('alignment_train.jsonl','alignment_a1_train.jsonl','sft_multimodal_train.jsonl'):train+=rows(OLD/name)
 for name in ('alignment_validation.jsonl','sft_validation.jsonl','heldout_evaluation.jsonl'):
  if (OLD/name).exists():protected+=rows(OLD/name)
 def collect(values):
  result={}
  for record in values:
   for image_hash,path in zip(record.get('image_hashes',[]),record.get('image_paths',[]),strict=True):result[image_hash]=Path(path)
  return result
 a,b=collect(train),collect(protected);all_paths={**a,**b}
 with ThreadPoolExecutor(max_workers=4) as executor:
  phashes=dict(zip(all_paths,executor.map(perceptual,all_paths.values()),strict=True))
 emit('old_protected_images_indexed',train=len(a),heldout=len(b))
 return a,b,phashes


def index(split):
 source=RAW/f'instances_{split}.json';data=read(source);categories={c['id']:c['name'] for c in data['categories']}
 if len(categories)!=80:raise ValueError('Expected80 COCO object categories')
 images={im['id']:dict(im,objects=[]) for im in data['images']};bad=set()
 for ann in data['annotations']:
  im=images[ann['image_id']]
  if ann.get('iscrowd'):bad.add(im['id']);continue
  try:box=normalize_box(ann['bbox'],im['width'],im['height'])
  except ValueError:bad.add(im['id']);continue
  im['objects'].append(dict(bbox_2d=box,label=categories[ann['category_id']],annotation_id=ann['id']))
 result={i:im for i,im in images.items() if i not in bad and im['objects']}
 emit('COCO_index',split=split,eligible_images=len(result),excluded_crowd_or_degenerate_images=len(bad))
 return result,categories,data['licenses']


def download(im,split):
 import requests
 from PIL import Image
 path=BASE/'images'/split/im['file_name'];path.parent.mkdir(parents=True,exist_ok=True)
 url='https://s3.amazonaws.com/images.cocodataset.org/'+split+'/'+im['file_name']
 if not path.exists():
  if shutil.disk_usage(BASE).free<34*2**30:raise OSError('Keep32GiB reserve plus2GiB dataset budget')
  tmp=path.with_suffix('.jpg.part')
  for attempt in range(3):
   try:
    with requests.get(url,stream=True,timeout=(20,40)) as response:
     response.raise_for_status();written=0
     with tmp.open('wb') as f:
      for chunk in response.iter_content(128*1024):
       written+=len(chunk)
       if written>16*2**20:raise ValueError('Unexpectedly large COCO image')
       f.write(chunk)
    tmp.replace(path);break
   except (requests.RequestException,OSError):
    tmp.unlink(missing_ok=True)
    if attempt==2:raise
    time.sleep(2)
 with Image.open(path) as image:
  if image.size!=(im['width'],im['height']) or image.getexif().get(274,1)!=1:raise ValueError('COCO geometry/EXIF transform differs')
 return dict(im,path=str(path),rgb_sha256=image_content_hash(path),phash=perceptual(path),file_sha256=sha(path),url=url)


def select_images(indexed,categories,count,split,salt,deny_hashes,deny_phashes,used_ids=None):
 used_ids=used_ids or set();ids=[];seen=set();by_class=collections.defaultdict(list)
 for identity,im in indexed.items():
  if identity in used_ids:continue
  for label in {v['label'] for v in im['objects']}:by_class[label].append(identity)
 for label in sorted(categories.values()):
  for identity in sorted(by_class[label],key=lambda i:rank(i,salt+':'+label))[:4 if split=='train2017' else 1]:
   if identity not in seen:seen.add(identity);ids.append(identity)
 ids += sorted((i for i in indexed if i not in seen and i not in used_ids),key=lambda i:rank(i,salt))
 selected=[];rejected=collections.Counter();phashes=list(deny_phashes);blocked=set(deny_hashes)
 cursor=0
 with ThreadPoolExecutor(max_workers=6) as executor:
  while len(selected)<count:
   batch=ids[cursor:cursor+min(24,count-len(selected))];cursor+=len(batch)
   if not batch:raise ValueError('Insufficient uncontaminated COCO images')
   for value in executor.map(lambda i:download(indexed[i],split),batch):
    if value['rgb_sha256'] in blocked:rejected['RGB_overlap']+=1;continue
    if any((value['phash']^p).bit_count()<=4 for p in phashes):rejected['perceptual_overlap_hamming_le4']+=1;continue
    selected.append(value);blocked.add(value['rgb_sha256']);phashes.append(value['phash'])
   emit('detection_images_selected',partition=salt,selected=len(selected),target=count,rejected=dict(rejected))
 return selected,dict(rejected)


def requests_for(images,categories,partition,annotation_sha):
 labels=sorted(categories.values());pools=collections.defaultdict(list)
 for im in images:
  grouped=collections.defaultdict(list)
  for obj in im['objects']:grouped[obj['label']].append(obj)
  tasks=[]
  for label,objects in grouped.items():
   if len(objects)<=MAX_BOXES:tasks.append(('class_all',[label],None,objects))
   if len(objects)>1:
    for relation,axis,reverse in (('leftmost',0,False),('rightmost',0,True),('topmost',1,False),('bottommost',1,True)):
     ordered=sorted(objects,key=lambda o:o['bbox_2d'][axis]+o['bbox_2d'][axis+2],reverse=reverse)
     score=lambda o:o['bbox_2d'][axis]+o['bbox_2d'][axis+2]
     if score(ordered[0])!=score(ordered[1]):tasks.append(('spatial',[label],relation,ordered[:1]))
  if len(im['objects'])<=MAX_BOXES:tasks.append(('all80',labels,None,im['objects']))
  absent=sorted(set(labels)-set(grouped),key=lambda label:rank(label,f'negative:{im["id"]}'))[:2]
  tasks += [('negative',[label],None,[]) for label in absent]
  for kind,classes,relation,objects in tasks:
   question=prompt(kind,classes,relation=relation,all_classes=labels)
   identity=hashlib.sha256(f'coco2017:{im["id"]}:{question}'.encode()).hexdigest()
   row=dict(record_id=identity,question=question,answer=canonical_answer(objects),image_paths=[im['path']],
    image_hashes=[im['rgb_sha256']],task_group='object_detection',language='en',partition=partition,
    messages=[dict(role='user',content=question)],
    source=dict(repo='COCO2017',split='train2017' if partition=='train' else 'val2017',image_id=im['id'],annotation_sha256=annotation_sha,
     annotation_ids=[v['annotation_id'] for v in objects],image_url=im['url'],image_license_id=im['license']),
    detection=dict(query_kind=kind,query_classes=classes,relation=relation,objects=[dict(bbox_2d=v['bbox_2d'],label=v['label']) for v in objects],
     coordinate_system=COORDINATE_SYSTEM,original_width=im['width'],original_height=im['height']))
   pools[kind].append(row)
 return pools


def choose(pools,quotas,salt,balanced_classes=False):
 values=[]
 for kind,n in quotas.items():
  pool=sorted(pools[kind],key=lambda r:rank(r['record_id'],salt+kind))
  chosen=[]
  if kind=='class_all' and balanced_classes:
   groups=collections.defaultdict(list)
   for row in pool:groups[row['detection']['query_classes'][0]].append(row)
   if len(groups)!=80 or any(len(v)<3 for v in groups.values()):raise ValueError('At least3 eligible class queries per COCO class required')
   chosen=[r for label in sorted(groups) for r in groups[label][:3]]
  ids={r['record_id'] for r in chosen};chosen += [r for r in pool if r['record_id'] not in ids][:n-len(chosen)]
  if len(chosen)!=n:raise ValueError(f'Insufficient {kind}: available{len(pool)}, wanted{n}')
  values+=chosen
 if len({r['record_id'] for r in values})!=len(values):raise ValueError('Duplicate detection request')
 return sorted(values,key=lambda r:rank(r['record_id'],salt+'final'))


def main():
 BASE.mkdir(parents=True,exist_ok=True)
 if (BASE/'release.json').exists():
  release=read(BASE/'release.json')
  for name,binding in release['files'].items():
   if sha(BASE/name)!=binding['sha256']:raise ValueError('Immutable detection release changed')
  emit('detection_release_already_ready');return
 source=read(RAW/'source.json')
 for name,binding in source['annotation_files'].items():
  if sha(RAW/name)!=binding['sha256']:raise ValueError('Original COCO annotations changed')
 train_old,heldout_old,old_phashes=old_protection()
 train_index,categories,licenses=index('train2017')
 train,train_rejected=select_images(train_index,categories,384,'train2017','OD_train',heldout_old,
  [old_phashes[h] for h in heldout_old]);del train_index;gc.collect()
 val_index,categories,_=index('val2017')
 protected={**train_old,**heldout_old};deny_hashes=set(protected)|{im['rgb_sha256'] for im in train}
 deny_phashes=list(old_phashes.values())+[im['phash'] for im in train]
 validation,val_rejected=select_images(val_index,categories,96,'val2017','OD_validation',deny_hashes,deny_phashes)
 test,test_rejected=select_images(val_index,categories,96,'val2017','OD_test',deny_hashes|{im['rgb_sha256'] for im in validation},
  deny_phashes+[im['phash'] for im in validation],{im['id'] for im in validation});del val_index;gc.collect()
 od=choose(requests_for(train,categories,'train',source['annotation_files']['instances_train2017.json']['sha256']),
  dict(class_all=270,spatial=67,all80=68,negative=45),'train',balanced_classes=True)
 quotas=dict(class_all=40,spatial=8,all80=8,negative=8)
 val=choose(requests_for(validation,categories,'validation',source['annotation_files']['instances_val2017.json']['sha256']),quotas,'val')
 testing=choose(requests_for(test,categories,'test',source['annotation_files']['instances_val2017.json']['sha256']),quotas,'test')
 parent=rows(OLD/'sft_multimodal_train.jsonl');replay=[]
 for group,count in dict(general_vqa=892,ocr_documents=765,charts_diagrams=510,knowledge_reasoning=383).items():
  pool=sorted((r for r in parent if r['task_group']==group),key=lambda r:rank(r['record_id'],'replay:'+group));replay+=pool[:count]
 if len(replay)!=2550 or len(od)!=450:raise ValueError('Exact small detection mixture required')
 mm=sorted(replay+od,key=lambda r:rank(r['record_id'],'MM_mix'));text=rows(OLD/'text_train.jsonl')
 if len(text)!=529:raise ValueError('Same529 text replay required')
 files={'detection_train.jsonl':od,'sft_multimodal_train.jsonl':mm,'text_train.jsonl':text,
  'detection_validation.jsonl':val,'detection_test.jsonl':testing}
 for name,values in files.items():write_rows(BASE/name,values)
 all_images=train+validation+test
 atomic(BASE/'images.json',dict(images=[{k:im[k] for k in ('id','path','rgb_sha256','phash','file_sha256','url','width','height','license')} for im in all_images],licenses=licenses))
 stats=dict(status='passed',old_train_images=len(train_old),old_heldout_images=len(heldout_old),
  train_image_pool=len(train),validation_image_pool=len(validation),test_image_pool=len(test),
  train_protected_overlap=0,heldout_old_used_overlap=0,image_partition_overlap=0,
  near_duplicate_exclusion='pHash hamming<=4; excludes known original-training/validation images from new heldout, original heldout from new training; not a claim about unknown base-model pretraining',
  rejection_counts=dict(train=train_rejected,validation=val_rejected,test=test_rejected),
  task_counts=dict(collections.Counter(r['detection']['query_kind'] for r in od)),
  training_classes=sorted({o['label'] for r in od for o in r['detection']['objects']}),max_boxes=MAX_BOXES,
  no_crowd_regions=True,no_degenerate_quantized_boxes=True,deterministic_seed=42)
 if len(stats['training_classes'])!=80:raise ValueError('All80 COCO object classes must be trained')
 atomic(BASE/'audit.json',stats)
 release=dict(status='passed',profile='final_sft_prompted_detection_mix_v1',seed=42,
  counts=dict(detection=450,MM_replay=2550,MM_total=3000,text=529,total=3529,optimizer_updates=56,validation_detection=64,test_detection=64),
  coordinate_system=COORDINATE_SYSTEM,coordinate_range=[0,1000],answer_format='JSON array of {bbox_2d:[xmin,ymin,xmax,ymax],label:string}',
  source=source,legacy_release_sha256=sha(OLD/'release.json'),
  files={name:dict(path=str(BASE/name),sha256=sha(BASE/name),bytes=(BASE/name).stat().st_size) for name in files},
  images_sha256=sha(BASE/'images.json'),audit_sha256=sha(BASE/'audit.json'),
  source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__),'src/dataset/research_detection_v1.py':sha(ROOT/'src/dataset/research_detection_v1.py')})
 atomic(BASE/'release.json',release);emit('detection_mix_ready',counts=release['counts'],free_disk_GiB=round(shutil.disk_usage(BASE).free/2**30,2))

if __name__=='__main__':main()
