"""Bounded CPU-only samples, never a checkpoint codec or full-size fit claim."""
from __future__ import annotations
import hashlib,json,os,struct,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
MODELS={'lfm':('LiquidAI/LFM2.5-8B-A1B','dd83f5cb02aa2938b66bba9a9998d38d0517c1f3'), 'giga':('ai-sage/GigaChat3.1-10B-A1.8B-bf16','189fff27a1dee68473960c3d5bca53e0e07a3191'),'small':('Tiiny/SmallThinker-4BA0.6B-Instruct','a24932bb880c85c04c794283202a9194b1b42709')}
PER_MODEL_FP32_BYTES=8*2**20;MAX_ACTUAL_TENSOR_BYTES=64*2**20
def digest(raw):return hashlib.sha256(raw).hexdigest()
def stat(path):
 s=Path(path).stat();return {'dev':s.st_dev,'ino':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def headers(snapshot):
 out={};proof=[]
 for path in sorted(snapshot.glob('*.safetensors')):
  before=stat(path)
  with path.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0]
   if n>16*2**20:raise ValueError('Unexpected large source header')
   raw=f.read(n)
  if len(raw)!=n or stat(path)!=before:raise ValueError('Source header changed')
  head=json.loads(raw);proof.append({'path':str(path),'resolved_path':str(path.resolve()),'header_sha256':digest(raw),'stat':before,'full_shard_SHA_reverified_by_this_bounded_probe':False})
  for name,spec in head.items():
   if name=='__metadata__':continue
   if name in out:raise ValueError('Duplicate original tensor key')
   out[name]=(path,8+n,spec,before)
 return out,proof
def read_bf16_prefix(entry,elements):
 path,base,spec,before=entry
 if spec['dtype']!='BF16':raise ValueError('Only actual original BF16 source samples')
 size=spec['data_offsets'][1]-spec['data_offsets'][0];count=min(elements,size//2)
 with path.open('rb') as f:raw=os.pread(f.fileno(),count*2,base+spec['data_offsets'][0])
 if len(raw)!=count*2 or stat(path)!=before:raise ValueError('Anchor sample changed or short read')
 return raw
def transform(raw,anchor=None):
 import numpy as np
 if len(raw)%4 or anchor is not None and len(raw)!=len(anchor):raise ValueError('Equal FP32 byte lengths required')
 x=np.frombuffer(raw,dtype='<u4')
 if anchor is not None:x=np.bitwise_xor(x,np.frombuffer(anchor,dtype='<u4'))
 return x.astype('<u4',copy=False).view('u1').reshape(-1,4).T.copy().tobytes()
def inverse(encoded,anchor=None):
 import numpy as np
 if len(encoded)%4 or anchor is not None and len(encoded)!=len(anchor):raise ValueError('Equal FP32 byte lengths required')
 x=np.frombuffer(encoded,dtype='u1').reshape(4,-1).T.copy().reshape(-1).view('<u4')
 if anchor is not None:x=np.bitwise_xor(x,np.frombuffer(anchor,dtype='<u4'))
 return x.astype('<u4',copy=False).tobytes()
def measurements(raw,anchor):
 from src.utils.research_lossless_delta import _compress,_decompress
 results=[]
 for name,encoded in [('raw_zstd3',raw),('byte_shuffle_zstd3',transform(raw)),('xor_anchor_byte_shuffle_zstd3',transform(raw,anchor))]:
  start=time.monotonic();compressed=_compress(encoded,3);encode_seconds=time.monotonic()-start
  start=time.monotonic();decoded=_decompress(compressed,len(encoded));restored=decoded if name=='raw_zstd3' else inverse(decoded,anchor if name.startswith('xor') else None);decode_seconds=time.monotonic()-start
  if restored!=raw:raise AssertionError('All sample bits must reconstruct exactly')
  results.append({'codec':name,'raw_FP32_bytes':len(raw),'compressed_bytes':len(compressed),'ratio':len(compressed)/len(raw),'encode_seconds':encode_seconds,'decode_and_transform_seconds':decode_seconds,'exact_all_bits':True})
 return results
def select(entries):
 # Every selected class is labelled; no all-parameter representativeness claim.
 patterns=[lambda k:'embed_tokens.weight' in k,lambda k:('norm.weight' in k and 'layers.0.' in k),lambda k:('self_attn.' in k and k.endswith('q_proj.weight')),lambda k:('conv.' in k and k.endswith('in_proj.weight')),lambda k:('.experts.0.' in k and (k.endswith('gate_proj.weight') or k.endswith('w1.weight'))),lambda k:('.experts.0.' in k and (k.endswith('down_proj.weight') or k.endswith('w2.weight'))),lambda k:('layers.0.' in k and (k.endswith('up_proj.weight') or k.endswith('w3.weight'))),lambda k:('layers.0.' in k and (k.endswith('down_proj.weight') or k.endswith('w2.weight')))]
 chosen=[]
 for predicate in patterns:
  key=next((k for k,v in sorted(entries.items()) if predicate(k) and v[2]['dtype']=='BF16' and k not in chosen),None)
  if key:chosen.append(key)
 return chosen
def probe():
 import torch
 from src.utils.research_storage import DiskBudget
 from src.utils.research_lossless_delta import memory_guard
 DiskBudget(ROOT,120).ensure_space(256*2**20);memory_guard(256*2**20);torch.set_num_threads(1)
 actual_read=0;results=[];sources={};started=time.monotonic()
 for name in ('lfm','giga','small'):
  repo,revision=MODELS[name];snapshot=Path('/workspace/.hf_home/hub')/('models--'+repo.replace('/','--'))/'snapshots'/revision
  entries,proof=headers(snapshot);sources[name]={'repo':repo,'revision':revision,'headers':proof,'config_sha256':digest((snapshot/'config.json').read_bytes())}
  checkpoint=None;source_state=None
  if name=='small':
   run=ROOT/'runs/research/smallthinker4__siglip2__4f__SFT_pilot_scienceqa_context_v2_balanced_s42';checkpoint=(run/'best/state.pt').resolve(strict=True);checkpoint_stat=stat(checkpoint)
   source_state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)['model'];sources[name]['actual_trained_checkpoint']={'path':str(checkpoint),'stat':checkpoint_stat,'whole_checkpoint_SHA_recomputed_by_this_bounded_probe':False}
  used=0;rows=[];chosen=select(entries);per=(PER_MODEL_FP32_BYTES//len(chosen)//4)
  for key in chosen:
   if source_state is not None and ('llm.'+key not in source_state or not source_state['llm.'+key].is_contiguous()):continue
   anchor_raw=read_bf16_prefix(entries[key],per);actual_read+=len(anchor_raw)
   original=torch.frombuffer(bytearray(anchor_raw),dtype=torch.bfloat16).float();anchor=original.view(torch.uint8).numpy().tobytes();n=original.numel()
   if source_state is not None:
    value=source_state['llm.'+key].detach().reshape(-1)[:n]
    if value.dtype!=torch.float32:raise ValueError('Actual trained fullFP32 sample required')
    raw=value.numpy().tobytes();actual_read+=len(raw);cases=[('ACTUAL_old_Small_trained_FP32',raw)]
   else:
    generator=torch.Generator(device='cpu').manual_seed(42+len(rows));noise=torch.randn(n,generator=generator)
    cases=[('untrained_exact_BF16_promoted_to_FP32',anchor),('SYNTHETIC_absolute_update_std1e-4',(original+noise*1e-4).view(torch.uint8).numpy().tobytes()),('SYNTHETIC_absolute_update_std1e-3',(original+noise*1e-3).view(torch.uint8).numpy().tobytes())]
   for label,raw in cases:rows.append({'case':label,'name':key,'source_shape':entries[key][2]['shape'],'sample_elements':n,'sample_raw_FP32_bytes':len(raw),'anchor_BF16_prefix_sha256':digest(anchor_raw),'anchor_promoted_FP32_prefix_sha256':digest(anchor),'sample_trained_or_synthetic_FP32_sha256':digest(raw),'measurements':measurements(raw,anchor)})
   used+=len(anchor)
   if actual_read>MAX_ACTUAL_TENSOR_BYTES:raise ValueError('Actual source tensor-read cap exceeded')
  if checkpoint is not None:
   if stat(checkpoint)!=checkpoint_stat:raise ValueError('Actual trained checkpoint changed during samples')
   del source_state
  summary=[]
  for case in dict.fromkeys(row['case'] for row in rows):
   selected=[x for x in rows if x['case']==case];raw_bytes=sum(x['sample_raw_FP32_bytes'] for x in selected)
   for codec in ('raw_zstd3','byte_shuffle_zstd3','xor_anchor_byte_shuffle_zstd3'):
    metrics=[next(m for m in x['measurements'] if m['codec']==codec) for x in selected];size=sum(m['compressed_bytes'] for m in metrics);enc=sum(m['encode_seconds'] for m in metrics);dec=sum(m['decode_and_transform_seconds'] for m in metrics)
    summary.append({'case':case,'codec':codec,'raw_FP32_sample_bytes':raw_bytes,'compressed_bytes':size,'ratio':size/raw_bytes,'encode_MiB_s':raw_bytes/2**20/enc,'decode_and_inverse_MiB_s':raw_bytes/2**20/dec,'exact_all_bits':all(m['exact_all_bits'] for m in metrics)})
  results.append({'model':name,'unique_FP32_sample_bytes':used,'selected_tensor_classes':len(chosen),'measurements':summary,'rows':rows})
 return {'schema_version':1,'status':'bounded_samples_passed','owner':'budget18_lossless_anchor_sample_v1','actual_source_tensor_bytes_read':actual_read,'actual_source_tensor_read_cap':MAX_ACTUAL_TENSOR_BYTES,'temporary_payload_disk_bytes':0,'elapsed_seconds':time.monotonic()-started,'source_sha256':{'scripts/probe_budget_anchor_compression_v1.py':digest(Path(__file__).read_bytes()),'src/utils/research_lossless_delta.py':digest((ROOT/'src/utils/research_lossless_delta.py').read_bytes())},'sources':sources,'results':results,'GPU_used':False,'active_formats_or_files_changed':False,'limitations':['Only selected small prefixes; tensor classes and unequal sample sizes are disclosed. No complete-model ratio or fit is established.','LFM/Giga values after updates are SYNTHETIC amplitude scenarios, not actual trained weights. Zero-delta initialization is not a training compression forecast.','Small samples are ACTUAL old4f-trained weights, not the new56step series and not LFM/Giga.','No optimizer compressibility measurement; final model-only archives would be after completed-stage optimizer compaction.','Only exact FP32 bits are reconstructed; no BF16 rounding, quantization or arithmetic delta approximation.','Pinned revision/config/header/sample hashes bind this bounded measurement; full-shard SHA validation remains mandatory for any future real codec.','Existing Small native-legacy codec remains untouched; this file is not a model loader or archive format.']}
def main():
 report=probe();path=ROOT/'research_results/budget18_anchor_compression_samples_v1.json';path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'path':str(path),'sha256':digest(path.read_bytes()),'actual_source_tensor_bytes_read':report['actual_source_tensor_bytes_read'],'elapsed_seconds':report['elapsed_seconds'],'summary':[{k:r[k] for k in ('model','measurements')} for r in report['results']]}))
if __name__=='__main__':main()
