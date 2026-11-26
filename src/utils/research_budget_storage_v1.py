"""Lossless native final-checkpoint storage accounting; no serialization changes."""
from __future__ import annotations
import math
from pathlib import Path
OWNER='native_exact_storage_budget_v1';METADATA_ALLOWANCE=128*2**20

def payload_storage_budget(payload):
 """Charge every unique full strided backing, including aliases, plus metadata.

 The existing atomic manager independently adds its5%/16MiB staging margin and
 unchanged120GiB reserve. NumPy RNG arrays are conservatively charged per object
 because pickle may serialize separate views. No dtype/model/optimizer filtering.
 """
 import torch
 import numpy as np
 seen=set();storages=set();tensor_bytes=0;array_bytes=0;metadata=0;count=0
 def walk(value):
  nonlocal tensor_bytes,array_bytes,metadata,count
  metadata+=128
  if metadata>METADATA_ALLOWANCE:raise ValueError('Native checkpoint metadata exceeds128MiB allowance')
  if id(value) in seen:return
  seen.add(id(value))
  if torch.is_tensor(value):
   if value.layout!=torch.strided or value.device.type not in ('cpu','cuda'):raise ValueError('Native budget supports actual CPU/CUDA strided storage only')
   storage=value.untyped_storage();key=(str(value.device),storage._cdata)
   if key not in storages:storages.add(key);tensor_bytes+=storage.nbytes();count+=1
  elif isinstance(value,np.ndarray):array_bytes+=value.nbytes
  elif isinstance(value,np.generic):metadata+=value.nbytes
  elif isinstance(value,dict):
   for key,child in value.items():walk(key);walk(child)
  elif isinstance(value,(list,tuple)):
   for child in value:walk(child)
  elif isinstance(value,(str,bytes,bytearray)):metadata+=len(value.encode() if isinstance(value,str) else value)
  elif value is None or isinstance(value,(bool,int,float)):
   if isinstance(value,int) and value.bit_length()>1024:raise ValueError('Unexpected giant checkpoint integer')
  else:raise ValueError('Unexpected native checkpoint metadata type: '+type(value).__name__)
  if metadata>METADATA_ALLOWANCE:raise ValueError('Native checkpoint metadata exceeds128MiB allowance')
 walk(payload)
 anticipated=tensor_bytes+array_bytes+METADATA_ALLOWANCE
 return {'owner':OWNER,'unique_tensor_storages':count,'unique_tensor_storage_bytes':tensor_bytes,'numpy_rng_storage_bytes':array_bytes,'metadata_scan_allowance_bytes':metadata,'metadata_reserved_bytes':METADATA_ALLOWANCE,'anticipated_bytes':anticipated,'native_serialization_unchanged':True,'tensor_dtype_or_payload_fields_changed':False}

def install_manager(module,*,validate_final=None,before_save=None):
 """Wrap only the private native manager; every payload field passes unchanged."""
 if module.__name__=='scripts.run_research_stage':raise ValueError('Canonical checkpoint manager cannot be patched')
 Native=module.BestLastCheckpointManager
 class BudgetManager(Native):
  def save_torch(self,payload,*args,**kwargs):
   if validate_final is not None:validate_final(payload)
   report=payload_storage_budget(payload)
   if before_save is not None:before_save(payload,report)
   kwargs['anticipated_bytes']=report['anticipated_bytes']
   result=super().save_torch(payload,*args,**kwargs)
   from src.utils.research_storage import _atomic_json
   _atomic_json(Path(self.root)/'checkpoint_storage_budget.json',report)
   return result
 module.BestLastCheckpointManager=BudgetManager;return BudgetManager
