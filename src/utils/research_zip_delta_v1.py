"""Isolated lossless virtual-native-ZIP archive; no production integration.

Every native Torch ZIP byte is retained. Frames are bounded and independently
decoded; torch.load reads a seekable virtual native file without a raw disk copy.
Only explicit full-storage FP32 tensors use pinned original BF16 XOR anchors.
"""
from __future__ import annotations
import bisect,hashlib,io,json,math,os,struct,uuid,zipfile
from pathlib import Path
MAGIC=b'RZIPX1\0\0';END=b'RZIPEND1';OWNER='lossless_virtual_torch_zip_v1';CHUNK=8*2**20;META=64*2**20
ROOT=Path(__file__).resolve().parents[2]
MODELS={
 'Tiiny/SmallThinker-4BA0.6B-Instruct':('a24932bb880c85c04c794283202a9194b1b42709','smallthinker','40adb346175e86f3840b0b16224acce3f01d0002b2bba76ffd2818f3fce1e876'),
 'LiquidAI/LFM2.5-8B-A1B':('dd83f5cb02aa2938b66bba9a9998d38d0517c1f3','lfm2_moe',''),
 'ai-sage/GigaChat3.1-10B-A1.8B-bf16':('189fff27a1dee68473960c3d5bca53e0e07a3191','deepseek_v3','6a6b8260f08791c4968f70934903e9aa892a53ee3f2e2e05b61a68ea1ff33503')}
ANCHOR_METADATA=ROOT/'research_results/budget18_generic_anchor_metadata_v1.json'
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def stat(path):
 s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def guard(extra=8*CHUNK):
 from src.utils.research_lossless_delta import memory_guard
 return memory_guard(extra)
def file_sha(path):
 before=stat(path);h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(CHUNK),b''):guard();h.update(b)
 if stat(path)!=before:raise ValueError('File changed while hashing')
 return h.hexdigest()

class Anchors:
 def __init__(self,snapshot,profile,*,verify=True):
  self.snapshot=Path(snapshot);self.profile=profile;self.files={};self.keys={};self.verified=False
  for name,expected in sorted(profile['files'].items()):
   if Path(name).name!=name:raise ValueError('Anchor path escape')
   p=self.snapshot/name;before=stat(p)
   if before[2]!=expected['bytes']:raise ValueError('Anchor source size differs')
   with p.open('rb') as f:
    length=struct.unpack('<Q',f.read(8))[0]
    if length>META:raise ValueError('Anchor header too large')
    raw=f.read(length)
   if len(raw)!=length or stat(p)!=before:raise ValueError('Anchor header changed')
   entries=json.loads(raw);self.files[name]={'path':str(p),'stat':before,'expected_sha256':expected['sha256'],'header_sha256':sha(raw),'data_start':8+length}
   for key,spec in entries.items():
    if key=='__metadata__':continue
    if key in self.keys:raise ValueError('Duplicate original anchor name')
    a,b=spec['data_offsets'];elements=math.prod(spec['shape']);size={'BF16':2,'F32':4,'F16':2,'I64':8,'I32':4,'U8':1,'BOOL':1}.get(spec['dtype'])
    if size is None or a<0 or b-a!=elements*size or 8+length+b>before[2]:raise ValueError('Invalid anchor tensor bounds')
    self.keys[key]=dict(spec,file=name)
  if verify:self.verify()
 def verify(self):
  if self.verified:
   self.check();return
  for entry in self.files.values():
   if file_sha(entry['path'])!=entry['expected_sha256']:raise ValueError('Canonical full anchor SHA differs')
  self.verified=True;self.check()
 def check(self):
  for row in self.files.values():
   if tuple(stat(row['path']))!=tuple(row['stat']):raise ValueError('Anchor changed after admission')
 def descriptor(self,name,shape):
  name=name.removeprefix('llm.').replace('.original.','.')
  def piece(key,start,expected):
   row=self.keys.get(key)
   if row is None or row['dtype']!='BF16' or row['shape']!=list(expected):return None
   return {'start':start,'elements':math.prod(expected),'key':key,'file':row['file'],'bf16_offset':row['data_offsets'][0]}
  direct=piece(name,0,shape)
  if direct:return {'name':name,'shape':list(shape),'pieces':[direct]}
  family=self.profile['family'];gate='w1' if family=='lfm2_moe' else 'gate_proj';up='w3' if family=='lfm2_moe' else 'up_proj';down='w2' if family=='lfm2_moe' else 'down_proj'
  if family not in ('lfm2_moe','deepseek_v3') or len(shape)!=3:return None
  if name.endswith('.experts.gate_up_proj'):
   prefix=name.removesuffix('.gate_up_proj');experts,twice,hidden=shape
   if twice%2:return None
   intermediate=twice//2;pieces=[]
   for e in range(experts):
    for half,kind in enumerate((gate,up)):
     p=piece(f'{prefix}.{e}.{kind}.weight',(e*2+half)*intermediate*hidden,[intermediate,hidden])
     if p is None:return None
     pieces.append(p)
  elif name.endswith('.experts.down_proj'):
   prefix=name.removesuffix('.down_proj');experts,hidden,intermediate=shape;pieces=[]
   for e in range(experts):
    p=piece(f'{prefix}.{e}.{down}.weight',e*hidden*intermediate,[hidden,intermediate])
    if p is None:return None
    pieces.append(p)
  else:return None
  return {'name':name,'shape':list(shape),'pieces':pieces}
 def fragment(self,descriptor,start,count):
  import numpy as np
  self.check();out=np.empty(count,dtype='<u4');covered=0
  for part in descriptor['pieces']:
   a=max(start,part['start']);b=min(start+count,part['start']+part['elements'])
   if a>=b:continue
   f=self.files[part['file']];offset=f['data_start']+part['bf16_offset']+2*(a-part['start'])
   with Path(f['path']).open('rb') as handle:raw=os.pread(handle.fileno(),2*(b-a),offset)
   if len(raw)!=2*(b-a):raise ValueError('Short BF16 anchor read')
   out[a-start:b-start]=np.frombuffer(raw,dtype='<u2').astype('<u4')<<16;covered+=b-a
  if covered!=count:raise ValueError('Anchor fragment coverage differs')
  return out.tobytes()
 def binding(self):return {'profile':self.profile,'snapshot':str(self.snapshot),'files':self.files}

def canonical_anchors(repo,revision):
 if repo not in MODELS or revision!=MODELS[repo][0]:raise ValueError('Only exact three pinned native families supported')
 _,family,config_sha=MODELS[repo];snapshot=Path('/workspace/.hf_home/hub')/('models--'+repo.replace('/','--'))/'snapshots'/revision
 raw=(snapshot/'config.json').read_bytes()
 if not config_sha:
  capacity=json.loads((ROOT/'research_results/primary_capacity.json').read_bytes());config_sha=next(x for x in capacity['models'] if x['repo']==repo)['config']['source_sha256']
 if sha(raw)!=config_sha:raise ValueError('Canonical anchor config differs')
 if family=='smallthinker':
  from src.utils.research_lossless_delta import SHARD_SHA256
  files={n:{'sha256':s,'bytes':(snapshot/n).stat().st_size} for n,s in SHARD_SHA256.items()};metadata_sha='frozen Small canonical LFS hashes'
 else:
  metadata=json.loads(ANCHOR_METADATA.read_bytes());entry=metadata['models'][repo]
  if entry['revision']!=revision or metadata['status']!='canonical_HF_metadata_only':raise ValueError('Canonical HF source metadata differs')
  files=entry['files'];metadata_sha=sha(ANCHOR_METADATA.read_bytes())
 return Anchors(snapshot,{'repo':repo,'revision':revision,'family':family,'config_sha256':config_sha,'canonical_HF_metadata_sha256':metadata_sha,'files':files})

def _storage_plan(source,anchors):
 import torch
 from torch._subclasses.fake_tensor import FakeTensorMode
 if torch.cuda.is_initialized():raise ValueError('Archive planning is CPU only')
 before=stat(source)
 with FakeTensorMode():payload=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
 if payload.get('model_ref')!={'repo':anchors.profile['repo'],'revision':anchors.profile['revision']}:raise ValueError('Source model identity differs from anchor profile')
 if payload.get('optimizer') not in (None,{}):raise ValueError('Prototype requires completed model-only payload; optimizer cannot be discarded by codec')
 storages={}
 for name,t in payload['model'].items():
  if not torch.is_tensor(t):raise ValueError('Model tensor required')
  storage=t.untyped_storage();offset=getattr(storage,'_checkpoint_offset',None)
  if offset is None:raise ValueError('Installed FakeTensorMode lacks native checkpoint offset')
  if t.dtype==torch.float32 and t.is_contiguous() and t.storage_offset()==0 and t.numel()*4==storage.nbytes():
   desc=anchors.descriptor(name,list(t.shape));storages.setdefault(offset,{'name':name,'bytes':storage.nbytes(),'transform':'xor_shuffle' if desc else 'shuffle','anchor':desc})
 segments=[];cursor=0;seen=set()
 with zipfile.ZipFile(source) as archive:
  infos=archive.infolist()
  if len({i.filename for i in infos})!=len(infos):raise ValueError('Duplicate native ZIP members')
  for info in infos:
   if info.compress_type!=zipfile.ZIP_STORED or info.flag_bits&1:raise ValueError('Only unencrypted native ZIP_STORED source supported')
   with Path(source).open('rb') as f:
    f.seek(info.header_offset);local=f.read(30)
   if len(local)!=30 or local[:4]!=b'PK\x03\x04':raise ValueError('Malformed native ZIP local header')
   n,e=struct.unpack_from('<HH',local,26);offset=info.header_offset+30+n+e
   if offset in storages:
    row=storages[offset]
    if row['bytes']!=info.file_size or offset<cursor:raise ValueError('Native storage bounds differ')
    if offset>cursor:segments.append({'offset':cursor,'bytes':offset-cursor,'transform':'raw','anchor':None})
    segments.append(dict(row,offset=offset));cursor=offset+info.file_size;seen.add(offset)
 if seen!=set(storages):raise ValueError('Fake storage offsets not mapped to unique native ZIP records')
 if cursor<before[2]:segments.append({'offset':cursor,'bytes':before[2]-cursor,'transform':'raw','anchor':None})
 if sum(x['bytes'] for x in segments)!=before[2] or stat(source)!=before:raise ValueError('Source segment coverage changed')
 return {'source_stat':before,'source_bytes':before[2],'native_storage_bytes':sum(i.file_size for i in infos if '/data/' in i.filename),'segments':segments,'model_ref':payload['model_ref'],'model_tensor_names':len(payload['model']),'anchored_storage_bytes':sum(s['bytes'] for s in segments if s['transform']=='xor_shuffle'),'unanchored_FP32_storage_bytes':sum(s['bytes'] for s in segments if s['transform']=='shuffle')}

def _shuffle(raw,anchor=None,inverse=False):
 import numpy as np
 if len(raw)%4 or anchor is not None and len(raw)!=len(anchor):raise ValueError('Invalid aligned FP32 fragment')
 if inverse:values=np.frombuffer(raw,dtype='u1').reshape(4,-1).T.copy().reshape(-1).view('<u4')
 else:values=np.frombuffer(raw,dtype='<u4')
 if anchor is not None:values=np.bitwise_xor(values,np.frombuffer(anchor,dtype='<u4'))
 return values.astype('<u4',copy=False).tobytes() if inverse else values.astype('<u4',copy=False).view('u1').reshape(-1,4).T.copy().tobytes()

def encode(source,output,anchors,*,reserve_gb=120,chunk_bytes=CHUNK):
 from src.utils.research_storage import DiskBudget
 from src.utils.research_lossless_delta import _compress
 if reserve_gb<120 or not math.isfinite(reserve_gb) or chunk_bytes<=0 or chunk_bytes>CHUNK or chunk_bytes%4:raise ValueError('Preserve120GiB reserve and bounded aligned chunks')
 source=Path(source).resolve(strict=True);output=Path(output).absolute();output.parent.mkdir(parents=True,exist_ok=True)
 if os.path.lexists(output):raise FileExistsError('Archive destination must be new')
 guard();anchors.verify();plan=_storage_plan(source,anchors);budget=DiskBudget(output.parent,reserve_gb);budget.ensure_space(plan['source_bytes']+2*META)
 header={'owner':OWNER,'schema_version':1,'chunk_bytes':chunk_bytes,'source_bytes':plan['source_bytes'],'native_storage_bytes':plan['native_storage_bytes'],'source_stat':plan['source_stat'],'model_ref':plan['model_ref'],'anchors':anchors.binding(),'segments':plan['segments']};raw_header=canonical(header)
 if len(raw_header)>META:raise ValueError('Archive metadata bound exceeded')
 temp=output.with_name(output.name+'.partial-'+uuid.uuid4().hex);frames=[];source_sha=hashlib.sha256();archive_sha=hashlib.sha256();since_flush=0;total=0
 try:
  with source.open('rb') as src,temp.open('xb') as dst:
   def write(raw):
    nonlocal since_flush,total
    dst.write(raw);archive_sha.update(raw);since_flush+=len(raw);total+=len(raw)
    if since_flush>=64*2**20:
     dst.flush();os.fsync(dst.fileno());os.posix_fadvise(dst.fileno(),total-since_flush,since_flush,os.POSIX_FADV_DONTNEED);since_flush=0;budget.ensure_space(0);guard()
   write(MAGIC+struct.pack('<Q',len(raw_header))+raw_header)
   for index,segment in enumerate(plan['segments']):
    for relative in range(0,segment['bytes'],chunk_bytes):
     guard();length=min(chunk_bytes,segment['bytes']-relative);raw=src.read(length)
     if len(raw)!=length:raise ValueError('Short native source read')
     source_sha.update(raw);anchor=anchors.fragment(segment['anchor'],relative//4,length//4) if segment['transform']=='xor_shuffle' else None
     transformed=raw if segment['transform']=='raw' else _shuffle(raw,anchor);compressed=_compress(transformed)
     frames.append({'raw_offset':segment['offset']+relative,'raw_bytes':length,'archive_offset':total,'compressed_bytes':len(compressed),'raw_sha256':sha(raw),'compressed_sha256':sha(compressed),'segment':index,'relative':relative});write(compressed)
   if src.read(1) or stat(source)!=tuple(plan['source_stat']):raise ValueError('Native source changed while encoding')
   anchors.check();footer=canonical({'owner':OWNER,'header_sha256':sha(raw_header),'source_sha256':source_sha.hexdigest(),'source_bytes':plan['source_bytes'],'frames':frames})
   if len(footer)>META:raise ValueError('Frame index bound exceeded')
   write(footer+struct.pack('<Q',len(footer))+END);dst.flush();os.fsync(dst.fileno());os.posix_fadvise(dst.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  os.link(temp,output);temp.unlink()
  fd=os.open(output.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
  return {'status':'encoded_prototype_only','owner':OWNER,'source':str(source),'source_stat':plan['source_stat'],'source_sha256':source_sha.hexdigest(),'archive':str(output),'archive_bytes':total,'archive_sha256':archive_sha.hexdigest(),'ratio':total/plan['source_bytes'],'anchored_storage_bytes':plan['anchored_storage_bytes'],'unanchored_FP32_storage_bytes':plan['unanchored_FP32_storage_bytes'],'model_tensor_names':plan['model_tensor_names'],'frames':len(frames),'no_scientific_publication_or_pointer_changes':True}
 except BaseException:
  if temp.exists():temp.unlink()
  raise

class Reader(io.RawIOBase):
 def __init__(self,path,anchors,*,verify_archive_sha=None):
  super().__init__();self.path=Path(path);self.before=stat(path);self.handle=self.path.open('rb');self.anchors=anchors;self.position=0;self.cache_index=None;self.cache=None
  if verify_archive_sha and file_sha(path)!=verify_archive_sha:raise ValueError('Archive physical SHA differs')
  if self.handle.read(8)!=MAGIC:raise ValueError('Unknown archive format')
  size=struct.unpack('<Q',self.handle.read(8))[0]
  if size>META:raise ValueError('Header bound exceeded')
  raw=self.handle.read(size);self.header=json.loads(raw);self.handle.seek(-16,2);length=struct.unpack('<Q',self.handle.read(8))[0]
  if self.handle.read(8)!=END or length>META:raise ValueError('Footer bound or magic differs')
  footer_offset=self.before[2]-16-length
  if footer_offset<16+size:raise ValueError('Footer overlap')
  self.handle.seek(footer_offset);self.footer=json.loads(self.handle.read(length));self.frames=self.footer['frames']
  if self.header['owner']!=OWNER or self.footer['owner']!=OWNER or self.footer['header_sha256']!=sha(raw) or canonical(self.header['anchors'])!=canonical(anchors.binding()):raise ValueError('Archive/source binding differs')
  rawpos=0;archivepos=16+size
  for frame in self.frames:
   index=frame['segment']
   if not 0<=index<len(self.header['segments']):raise ValueError('Unknown frame segment')
   segment=self.header['segments'][index]
   if frame['raw_offset']!=rawpos or frame['archive_offset']!=archivepos or not 0<frame['raw_bytes']<=self.header['chunk_bytes']<=CHUNK or not 0<frame['compressed_bytes']<=CHUNK+2**20 or frame['relative']<0 or frame['relative']+frame['raw_bytes']>segment['bytes'] or segment['offset']+frame['relative']!=rawpos:raise ValueError('Invalid/overlapping frame coverage')
   if segment['transform'] not in ('raw','shuffle','xor_shuffle') or segment['transform']!='raw' and (frame['relative']%4 or frame['raw_bytes']%4):raise ValueError('Unknown transform/alignment')
   rawpos+=frame['raw_bytes'];archivepos+=frame['compressed_bytes']
  if rawpos!=self.header['source_bytes'] or rawpos!=self.footer['source_bytes'] or archivepos!=footer_offset:raise ValueError('Incomplete virtual native file coverage')
  self.offsets=[f['raw_offset'] for f in self.frames];anchors.verify();self.decoded_frames=set()
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.position
 def seek(self,offset,whence=0):
  target=offset if whence==0 else self.position+offset if whence==1 else self.header['source_bytes']+offset if whence==2 else -1
  if target<0:raise ValueError('Invalid native virtual seek')
  self.position=target;return target
 def _frame(self,index):
  from src.utils.research_lossless_delta import _decompress
  if self.cache_index==index:return self.cache
  guard();f=self.frames[index];self.handle.seek(f['archive_offset']);compressed=self.handle.read(f['compressed_bytes'])
  if sha(compressed)!=f['compressed_sha256']:raise ValueError('Compressed frame corrupt')
  raw=_decompress(compressed,f['raw_bytes']);segment=self.header['segments'][f['segment']]
  if segment['transform']!='raw':
   anchor=self.anchors.fragment(segment['anchor'],f['relative']//4,f['raw_bytes']//4) if segment['transform']=='xor_shuffle' else None;raw=_shuffle(raw,anchor,inverse=True)
  if sha(raw)!=f['raw_sha256'] or stat(self.path)!=self.before:raise ValueError('Decoded native bytes/physical source changed')
  self.cache_index=index;self.cache=raw;self.decoded_frames.add(index);return raw
 def readinto(self,buffer):
  view=memoryview(buffer).cast('B');wanted=min(len(view),max(0,self.header['source_bytes']-self.position));done=0
  while done<wanted:
   index=bisect.bisect_right(self.offsets,self.position)-1;frame=self.frames[index];data=self._frame(index);offset=self.position-frame['raw_offset'];n=min(wanted-done,len(data)-offset);view[done:done+n]=data[offset:offset+n];done+=n;self.position+=n
  return done
 def read(self,size=-1):
  if size<0:size=max(0,self.header['source_bytes']-self.position)
  size=min(size,max(0,self.header['source_bytes']-self.position))
  if size>META:raise MemoryError('Large native record must use readinto, not allocate another whole buffer')
  out=bytearray(size);n=self.readinto(out);return bytes(out[:n])
 def close(self):
  if not self.closed:self.handle.close();self.cache=None
  super().close()

def verify_native_bytes(path,anchors,expected_archive_sha=None):
 with Reader(path,anchors,verify_archive_sha=expected_archive_sha) as reader:
  h=hashlib.sha256();buffer=bytearray(CHUNK);count=0
  while (n:=reader.readinto(buffer)):
   h.update(memoryview(buffer)[:n]);count+=n
  if h.hexdigest()!=reader.footer['source_sha256']:raise ValueError('Whole native checkpoint byte SHA differs')
  return {'status':'passed','all_native_ZIP_bytes_exact':True,'decoded_source_bytes':count,'decoded_source_sha256':h.hexdigest(),'all_frames_verified':len(reader.decoded_frames)==len(reader.frames),'frames':len(reader.frames)}

def load(path,anchors,*,expected_archive_sha=None):
 import torch
 if torch.cuda.is_initialized():raise ValueError('Prototype loader is CPU only')
 with Reader(path,anchors,verify_archive_sha=expected_archive_sha) as reader:
  storage_bytes=reader.header['native_storage_bytes']
  if not 0<=storage_bytes<=reader.header['source_bytes']:raise ValueError('Native storage allocation bound differs')
  guard(storage_bytes+8*CHUNK+META)
  return torch.load(reader,map_location='cpu',mmap=False,weights_only=False)
