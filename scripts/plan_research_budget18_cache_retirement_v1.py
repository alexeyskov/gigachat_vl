#!/usr/bin/env python3
"""Metadata-only, no-delete retirement proposal for unused old vision features."""
from __future__ import annotations
import collections,hashlib,io,json,os,pickle,stat,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'research_results/budget18_cache_retirement_plan_v1'
DATA=ROOT/'datasets/research/budget18-en-v1'
RELEASE='910448c7fd680ec949d6521e32716aee2832a99fc534d185f0a06b226c0ec1c0'
SIG='72068dd258319f911109ed526e1f1ad4a721d2c3efa146ba784089a6712b304c'
INPUTS={}

def sha(raw):return hashlib.sha256(raw).hexdigest()
def stat_row(path):
 s=Path(path).stat(follow_symlinks=False)
 return {name:getattr(s,name) for name in ('st_dev','st_ino','st_mode','st_nlink','st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns','st_blocks')}
def bind(path):
 p=Path(path);raw=p.read_bytes();v={'path':str(p),'sha256':sha(raw),'bytes':len(raw)};INPUTS[str(p.relative_to(ROOT))]=v;return raw
def read(path):return json.loads(bind(path))
def read_rows(path):
 raw=bind(path)
 return [json.loads(line) for line in raw.split(b'\n') if line]
def guard():
 if 'torch' in sys.modules:raise RuntimeError('Torch imported into metadata-only planner')
 from src.utils.research_lossless_delta import memory_guard
 from src.utils.research_storage import DiskBudget
 memory_guard(0,headroom_bytes=32*2**30);DiskBudget(ROOT,120).ensure_space(0)

def rebuilt(storage,offset,shape,stride,requires_grad,hooks,*extra):
 if requires_grad or hooks or extra or type(offset) is not int:raise ValueError('Unexpected cache tensor metadata')
 return {'storage':storage,'offset':offset,'shape':list(shape),'stride':list(stride)}
class MetadataOnly(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name)==('torch._utils','_rebuild_tensor_v2'):return rebuilt
  if (module,name)==('torch','BFloat16Storage'):return ('storage_type','bfloat16')
  if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
  raise ValueError('Unallowlisted cache metadata global: '+module+'.'+name)
 def persistent_load(self,value):
  if not (isinstance(value,tuple) and len(value)==5 and value[0]=='storage' and value[1]==('storage_type','bfloat16') and value[3]=='cpu' and type(value[4]) is int and value[4]>=0):raise ValueError('Unexpected tensor storage descriptor')
  return {'key':value[2],'device':value[3],'elements':value[4],'dtype':'bfloat16'}

def header(path):
 before=stat_row(path)
 if not stat.S_ISREG(before['st_mode']) or before['st_nlink']!=1:raise ValueError('Nonordinary feature file preserved')
 with zipfile.ZipFile(path) as z:
  members=z.infolist();names=[m.filename for m in members]
  entries=[m for m in members if m.filename.endswith('/data.pkl')]
  if len(entries)!=1 or entries[0].file_size>65536:raise ValueError('Unexpected cache metadata envelope')
  raw=z.read(entries[0]);payload=MetadataOnly(io.BytesIO(raw)).load()
  if not isinstance(payload,dict):raise ValueError('Mapping cache payload required')
  image_hash=payload.get('image_hash');fingerprint=payload.get('fingerprint')
  if fingerprint!=SIG or not isinstance(image_hash,str) or len(image_hash)!=64 or any(c not in '0123456789abcdef' for c in image_hash):raise ValueError('Cache content identity mismatch')
  key=sha((fingerprint+image_hash).encode())
  if Path(path).name!=key+'.pt' or Path(path).parent.name!=key[:2]:raise ValueError('Content-addressed cache path mismatch')
  tensor=payload.get('features',{});storage=tensor.get('storage',{})
  if tensor.get('shape')!=[729,1152] or storage.get('elements')!=729*1152 or tensor.get('offset')!=0:raise ValueError('Unexpected legacy Siglip BF16 feature shape')
  data=[m for m in members if m.filename.endswith('/data/'+str(storage['key']))]
  if len(data)!=1 or data[0].file_size!=729*1152*2:raise ValueError('Feature storage size differs')
  checksum=payload.get('checksum')
  if not isinstance(checksum,str) or len(checksum)!=64:raise ValueError('Missing stored feature checksum')
  descriptor={'metadata_pickle_sha256':sha(raw),'metadata_pickle_bytes':len(raw),'fingerprint':fingerprint,'image_hash':image_hash,
    'stored_feature_tensor_checksum':checksum,'feature_shape':tensor['shape'],'feature_dtype':'bfloat16',
    'zip_members':[{'filename':m.filename,'bytes':m.file_size,'compressed_bytes':m.compress_size,'CRC32':m.CRC,'header_offset':m.header_offset} for m in members],
    'tensor_body_read_or_checksum_recomputed':False,'whole_feature_file_sha256':None}
 if stat_row(path)!=before:raise ValueError('Cache file changed during metadata-only scan')
 return before,descriptor

def collect(records,images,*,allow_nonimages=True):
 for record in records:
  paths,hashes=record.get('image_paths',[]),record.get('image_hashes',[])
  if len(paths)!=len(hashes):raise ValueError('Record image/hash schema mismatch')
  for path,h in zip(paths,hashes):
   if len(h)!=64:raise ValueError('Canonical hash missing')
   p=Path(path);p=p if p.is_absolute() else ROOT/p
   images.setdefault(h,set()).add(str(p))

def main():
 if OUT.exists():raise ValueError('Immutable output exists')
 guard();OUT.mkdir();start=time.monotonic()
 release=read(DATA/'release.json')
 if sha((DATA/'release.json').read_bytes())!=RELEASE or release.get('status')!='passed':raise ValueError('Exact passed budget18 release required')
 protected={};counts={}
 for name,item in release['files'].items():
  if not name.endswith('.jsonl'):continue
  p=Path(item['path']);raw=p.read_bytes()
  if sha(raw)!=item['sha256'] or len(raw)!=item['bytes']:raise ValueError('Protected budget source changed')
  records=read_rows(p);counts[name]=len(records);collect(records,protected)
 budget_only=set(protected)
 blank=ROOT/'runs/research/smallthinker4__siglip2__4f__SFT_pilot_scienceqa_context_v2_balanced_s42/evaluation/grounding_v1/blank'
 protocol=read(blank/'protocol.json')
 if protocol['manifest_sha256']!=release['files']['heldout_evaluation.jsonl']['sha256']:raise ValueError('Blank reference is not the exact current heldout512')
 records=read_rows(blank/'transformed_manifest.jsonl')
 if len(records)!=512 or any(r['image_ablation']['variant']!='blank' or r['image_ablation']['seed']!=42 for r in records):raise ValueError('Full declared blank intervention required')
 collect(records,protected)
 # Preserve the complete future derived namespace, not only currently warm entries.
 from src.utils.research_derived_vision_v1 import resolve_derived_vision
 qwen=resolve_derived_vision('Qwen/Qwen2.5-VL-7B-Instruct',preprocessing_profile='qwen_native_bounded_729')
 if 'torch' in sys.modules:raise RuntimeError('Derived metadata resolver unexpectedly imported Torch')
 known={h:set(p) for h,p in protected.items()}
 source_files=[]
 for directory in ('pilot-v1','smoke-v1','eval-heldout-v1','eval-heldout-v2','eval-heldout-v3'):
  base=ROOT/'datasets/research'/directory
  for p in sorted(base.glob('*.jsonl')):
   if p.name.startswith(('text_','excluded_','training_exclusion')):continue
   if p.stat().st_size>64*2**20:continue
   records=read_rows(p)
   if any('image_paths' in r for r in records):collect(records,known);source_files.append(str(p.relative_to(ROOT)))
 cache=ROOT/'cache/vision'/SIG
 paths=sorted(cache.rglob('*.pt'));totals=collections.Counter();unknown=[]
 with (OUT/'feature_entries.jsonl').open('x') as allfile,(OUT/'candidate_features.jsonl').open('x') as candidates,(OUT/'protected_features.jsonl').open('x') as kept:
  for index,path in enumerate(paths):
   if index%500==0:guard();print(json.dumps({'event':'cache_metadata_scan','done':index,'total':len(paths),'elapsed':time.monotonic()-start}),flush=True)
   try:
    st,meta=header(path);h=meta['image_hash'];sources=[]
    for name in sorted(known.get(h,set())):
     p=Path(name)
     if p.is_file() and not p.is_symlink():sources.append({'path':name,'stat':stat_row(p),'current_pixels_redecoded':False})
    classification='protected_budget_or_blank' if h in protected else 'candidate_recomputable_old_feature' if sources else 'preserve_source_unavailable_or_unmapped'
    row={'path':str(path),'stat':st,'metadata':meta,'classification':classification,'available_original_images':sources,'byte_content_verified_now':False}
    totals[classification+'_files']+=1;totals[classification+'_logical_bytes']+=st['st_size'];totals[classification+'_allocated_bytes']+=st['st_blocks']*512
    target=candidates if classification=='candidate_recomputable_old_feature' else kept
    target.write(json.dumps(row,sort_keys=True)+'\n');allfile.write(json.dumps(row,sort_keys=True)+'\n')
   except (ValueError,OSError,pickle.UnpicklingError,zipfile.BadZipFile) as error:
    unknown.append({'path':str(path),'reason':str(error),'stat':stat_row(path)});totals['preserve_unreadable_or_unknown_files']+=1
  for f in (allfile,candidates,kept):f.flush();os.fsync(f.fileno())
 protected_list=[]
 for h,ps in sorted(protected.items()):
  key=sha((SIG+h).encode());p=cache/key[:2]/(key+'.pt')
  protected_list.append({'image_hash':h,'required_by_budget_manifest':h in budget_only,'source_paths':sorted(ps),'siglip_feature_path':str(p),'currently_exists':p.is_file()})
 (OUT/'protected_image_closure.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in protected_list))
 (OUT/'preserved_unknown_entries.json').write_text(json.dumps(unknown,indent=2)+'\n')
 outputs={p.name:{'path':str(p),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.is_file()}
 report={'schema_version':1,'owner':'budget18_unused_feature_retirement_plan_v1','status':'metadata_only_retirement_proposal_no_deletion_authority',
  'created_unix':time.time(),'elapsed_seconds':time.monotonic()-start,'script_sha256':sha(Path(__file__).read_bytes()),'budget_data_release_sha256':RELEASE,
  'budget_manifest_records':counts,'budget_unique_RGB_hashes':len(budget_only),'protected_budget_plus_blank_unique_RGB_hashes':len(protected),
  'active_Siglip_cache_fingerprint':SIG,'active_Qwen_cache_fingerprint':qwen['cache_fingerprint'],
  'protected_entire_Qwen_namespace':str(ROOT/'cache/vision/derived_vision_v1'/qwen['cache_fingerprint']),
  'cache_feature_files_scanned':len(paths),'totals':dict(totals),'inputs':INPUTS,'outputs':outputs,'source_manifest_inputs':source_files,
  'no_original_images_models_checkpoints_or_sources_proposed_for_deletion':True,'GPU_or_network_used':False,'Torch_imported':False,
  'metadata_not_tensor_body_hash':True,'deletion_performed':False,
  'requires_ROOT_before_any_retirement':['Approve exact candidate file list after checking actual budget/current-controller source identity.','Freshly bind candidate stats/metadata hashes, current protected closure, UID-wide fd/maps and absence of old corpus producers.','Whole feature-file hashing was intentionally not performed; executor may compute full SHA before authorized retirement and preserve its receipt.','Recomputability evidence is extant original image paths/stat plus original RGB metadata; current unselected source RGB pixels were not redecoded. Images are preserved.','Old corpus training feature hits/timing will be lost and recomputation incurs encoder work. No old prediction/config/metric/cache source references may be retagged.','This proposal does not authorize deletion of any raw dataset/parquet, source pixels, HF anchors, selected features or derived namespace.']}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'event':'completed_metadata_plan','output':str(OUT),'report_sha256':sha((OUT/'report.json').read_bytes()),'totals':dict(totals)}),flush=True)
 if 'torch' in sys.modules:raise RuntimeError('Torch unexpectedly imported')

if __name__=='__main__':main()
