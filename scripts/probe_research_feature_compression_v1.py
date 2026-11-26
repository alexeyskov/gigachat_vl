#!/usr/bin/env python3
"""Bounded read-only study; never alters or adopts an active feature-cache format."""
from __future__ import annotations
import gzip,hashlib,io,json,os,random,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
LIMIT=64*2**20;COUNT=32;OWNER='read_only_feature_lossless_compression_probe_v1'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def transform(raw,shape,kind,inverse=False):
 import numpy as np
 n=len(raw)//2
 if kind=='byte_shuffle':
  return (np.frombuffer(raw,np.uint8).reshape(2,-1).T.copy().tobytes() if inverse else np.frombuffer(raw,np.uint8).reshape(-1,2).T.copy().tobytes())
 if kind=='bit_shuffle':
  elements=shape[0]*shape[1];width=(elements+7)//8
  if inverse:
   bits=np.unpackbits(np.frombuffer(raw,np.uint8).reshape(16,width),axis=1)[:,:elements].T.copy()
   return np.packbits(bits,axis=1).tobytes()
  bits=np.unpackbits(np.frombuffer(raw,np.uint8).reshape(-1,2),axis=1).T.copy()
  return np.packbits(bits,axis=1).tobytes()
 if kind=='xor_previous_token_byte_shuffle':
  if inverse:
   unshuffled=transform(raw,shape,'byte_shuffle',True);a=np.frombuffer(unshuffled,dtype='<u2').reshape(shape)
   return np.bitwise_xor.accumulate(a,axis=0).astype('<u2',copy=False).tobytes()
  a=np.frombuffer(raw,dtype='<u2').reshape(shape);delta=a.copy();delta[1:]=np.bitwise_xor(a[1:],a[:-1]);return transform(delta.tobytes(),shape,'byte_shuffle')
 if kind=='identity':return raw
 raise ValueError('Unknown lossless transform')

def probe(root=ROOT):
 from src.utils import research_lossless_delta as codec
 codec.memory_guard(512*2**20)
 import numpy as np
 import torch
 from src.dataset.vision_cache import canonical_json
 if torch.cuda.is_initialized():raise RuntimeError('Feature probe is CPU only')
 torch.set_num_threads(1)
 base=root/'cache/vision';files=sorted(base.glob('*/*/*.pt'));chosen=random.Random(42).sample(files,COUNT);total=sum(p.stat().st_size for p in chosen)
 if total>LIMIT:raise ValueError('Feature-read sample exceeds64MiB cap')
 baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;rows=[];start=time.monotonic()
 algorithms=[('whole_pt_gzip6',None,'gzip'),('whole_pt_zstd3',None,'zstd'),('raw_BF16_zstd3','identity','zstd'),('byte_shuffle_BF16_zstd3','byte_shuffle','zstd'),('bit_shuffle_BF16_zstd3','bit_shuffle','zstd'),('xor_previous_token_byte_shuffle_zstd3','xor_previous_token_byte_shuffle','zstd')]
 for index,path in enumerate(chosen):
  before=path.stat();rawfile=path.read_bytes();payload=torch.load(io.BytesIO(rawfile),map_location='cpu',weights_only=True);tensor=payload['features'];shape=list(tensor.shape)
  if tensor.dtype!=torch.bfloat16 or len(shape)!=2 or not tensor.is_contiguous() or not torch.isfinite(tensor).all():raise ValueError('Finite contiguous BF16 feature sample required')
  raw=tensor.view(torch.uint8).numpy().tobytes();metadata={k:v for k,v in payload.items() if k!='features'};meta=canonical_json(metadata).encode();fingerprint=path.parent.parent.name
  if payload['checksum']!=sha(raw) or payload['shape']!=shape or payload['storage_dtype']!=str(tensor.dtype) or payload['fingerprint']!=fingerprint or sha((fingerprint+payload['image_hash']).encode())!=path.stem:raise ValueError('Original feature checksum/key/metadata mismatch')
  result={}
  for name,kind,algorithm in algorithms:
   original=rawfile if kind is None else raw;begin=time.perf_counter();shuffled=original if kind is None else transform(original,shape,kind)
   encoded=gzip.compress(shuffled,compresslevel=6,mtime=0) if algorithm=='gzip' else codec._compress(shuffled);encode_seconds=time.perf_counter()-begin
   begin=time.perf_counter();decoded=gzip.decompress(encoded) if algorithm=='gzip' else codec._decompress(encoded,len(shuffled));restored=decoded if kind is None else transform(decoded,shape,kind,True);decode_seconds=time.perf_counter()-begin
   if restored!=original:raise ValueError('Lossless feature bit reconstruction failed')
   if kind is not None:
    restored_tensor=torch.frombuffer(bytearray(restored),dtype=torch.bfloat16).reshape(shape)
    if not torch.equal(restored_tensor.view(torch.uint8),tensor.view(torch.uint8)) or canonical_json(metadata).encode()!=meta:raise ValueError('Reconstructed dtype/shape/metadata orbits changed')
   # A prospective transformed envelope needs the entire unchanged metadata,
   # dimensions/dtype plus checksum. This is a measured canonical JSON size,
   # not an adopted cache schema nor a claim about filesystem block packing.
   envelope=len(encoded)+(0 if kind is None else len(meta)+256)
   result[name]={'encoded_bytes':len(encoded),'envelope_planning_bytes':envelope,'encode_seconds':encode_seconds,'decode_seconds':decode_seconds,'all_original_bytes_or_BF16bits_exact':True,'metadata_shape_dtype_unchanged':True}
  after=path.stat()
  if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Active feature source changed')
  rows.append({'path':str(path.relative_to(root)),'source_file_sha256':sha(rawfile),'source_stat':{k:getattr(before,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')},'fingerprint':fingerprint,'image_hash':payload['image_hash'],'feature_checksum':sha(raw),'raw_feature_bytes':len(raw),'shape':shape,'storage_dtype':str(tensor.dtype),'metadata_sha256':sha(meta),'algorithms':result})
  del payload,tensor,raw,rawfile,restored,restored_tensor,encoded,decoded,shuffled
 totals={}
 for name,_,_ in algorithms:
  encoded=sum(r['algorithms'][name]['encoded_bytes'] for r in rows);envelope=sum(r['algorithms'][name]['envelope_planning_bytes'] for r in rows);enc=sum(r['algorithms'][name]['encode_seconds'] for r in rows);dec=sum(r['algorithms'][name]['decode_seconds'] for r in rows);raw=sum(r['raw_feature_bytes'] for r in rows)
  totals[name]={'encoded_bytes':encoded,'envelope_planning_bytes':envelope,'ratio_to_sample_raw_feature_bytes':envelope/raw,'ratio_to_sample_original_pt_bytes':envelope/total,'encode_seconds':enc,'decode_seconds':dec,'encode_raw_MiB_per_second':raw/2**20/enc,'decode_raw_MiB_per_second':raw/2**20/dec,'all32_exact':True}
 output={'schema_version':1,'owner':OWNER,'status':'bounded_sample_exact_lossless_roundtrip_passed','sample_seed':42,'selection':'uniform32 from lexically sorted existing content-addressed cache paths; no labels/model outputs used','existing_cache_files_at_selection':len(files),'sample_file_bytes_read':total,'feature_sample_cap_bytes':LIMIT,'sample_raw_feature_bytes':sum(r['raw_feature_bytes'] for r in rows),'sample_records':len(rows),'source_code_sha256':{str(Path(__file__).relative_to(root)):sha(Path(__file__).read_bytes()),'src/dataset/vision_cache.py':sha((root/'src/dataset/vision_cache.py').read_bytes()),'src/utils/research_lossless_delta.py':sha((root/'src/utils/research_lossless_delta.py').read_bytes())},'environment':{'torch':torch.__version__,'numpy':np.__version__,'zstd':codec._zstd().ZSTD_versionString().decode(),'byteorder':sys.byteorder,'CPU_threads':1,'CUDA_initialized':torch.cuda.is_initialized()},'aggregate':totals,'rows':rows,'wall_seconds':time.monotonic()-start,'process_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'baseline_after_Torch_import_RSS_KiB':baseline,'temporary_disk_payload_bytes_created':0,'temporary_feature_transform_buffers_bounded_bytes':64*2**20,'active_cache_or_decoder_or_scientific_files_changed':False,'new_format_adopted':False,'representative_PRIMARY_ratio_or_admission_credit_proven':False}
 target=root/'research_results/feature_lossless_compression_sample_v1.json';target.write_text(json.dumps(output,sort_keys=True,indent=2)+'\n');print(json.dumps({'report':str(target),'sha256':sha(target.read_bytes()),'sample_file_bytes_read':total,'shape_set':sorted({tuple(r['shape']) for r in rows}),'sample_only_no_adoption':True,'aggregate':totals,'wall_seconds':output['wall_seconds'],'peak_RSS_KiB':output['process_peak_RSS_KiB']},indent=2))
if __name__=='__main__':
 os.environ['CUDA_VISIBLE_DEVICES']='';probe()
