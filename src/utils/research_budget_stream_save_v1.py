"""Bounded page-cache native Torch writer for a manager-owned new staging file."""
from __future__ import annotations
import os
from pathlib import Path
CHUNK=64*2**20

class _Writer:
 def __init__(self,handle,*,guard,budget,chunk=CHUNK):
  self.handle=handle;self.guard=guard;self.budget=budget;self.chunk=chunk;self.pending=0;self.total=0;self.start=0;self.commits=0;self.minimum=None
 def write(self,data):
  view=memoryview(data);length=len(view);offset=0
  while offset<length:
   count=min(self.chunk-self.pending,length-offset);written=self.handle.write(view[offset:offset+count])
   if written!=count:raise OSError('Incomplete native checkpoint staging write')
   self.pending+=written;self.total+=written;offset+=written
   if self.pending==self.chunk:self.commit()
  return length
 def flush(self):self.handle.flush()
 def commit(self):
  self.handle.flush();os.fsync(self.handle.fileno())
  if self.total>self.start:os.posix_fadvise(self.handle.fileno(),self.start,self.total-self.start,os.POSIX_FADV_DONTNEED)
  self.budget.ensure_space(0);snapshot=self.guard(self.chunk);available=snapshot['available_bytes'];self.minimum=available if self.minimum is None else min(self.minimum,available);self.start=self.total;self.pending=0;self.commits+=1


def install_manager(module,*,guard):
 """Install BEFORE the shared budget manager, retaining its final-step checks."""
 if module.__name__=='scripts.run_research_stage':raise ValueError('Canonical checkpoint writer patch forbidden')
 Native=module.BestLastCheckpointManager
 class StreamingManager(Native):
  def save_torch(self,payload,*,checkpoint_format='legacy',**kwargs):
   if checkpoint_format!='legacy':raise ValueError('Budget streaming writer supports native legacy-single-file ZIP only')
   import torch
   from src.utils.research_budget_storage_v1 import payload_storage_budget
   def writer(directory):
    path=Path(directory)/'state.pt'
    if path.exists() or path.is_symlink():raise ValueError('Only new manager-owned staging payload may be written')
    # Native Torch can pass one large backing to its Python sink. Charge its
    # possible transfer buffer before serialization, then guard actual memory
    # after every fsynced/advised64MiB window; no predicted reclaim is credited.
    seen=set();largest=0
    def visit(v):
     nonlocal largest
     if id(v) in seen:return
     seen.add(id(v))
     if torch.is_tensor(v):largest=max(largest,v.untyped_storage().nbytes())
     elif isinstance(v,dict):
      for x in v.values():visit(x)
     elif isinstance(v,(tuple,list)):
      for x in v:visit(x)
    visit(payload);guard(2*largest+CHUNK)
    with path.open('xb') as handle:
     sink=_Writer(handle,guard=guard,budget=self.budget);torch.save(payload,sink);sink.commit()
    self._budget_stream_write={'native_torch_serialization_unchanged':True,'source_or_old_objects_advised':False,'only_new_staging_output_advised':True,'assumed_cache_credit_bytes':0,'max_unflushed_output_bytes':CHUNK,'written_bytes':sink.total,'fsync_advice_commits':sink.commits,'minimum_measured_available_bytes':sink.minimum}
   result=self.save(writer,**kwargs)
   from src.utils.research_storage import _atomic_json
   _atomic_json(self.root/'checkpoint_stream_write.json',self._budget_stream_write)
   return result
 module.BestLastCheckpointManager=StreamingManager;return StreamingManager
