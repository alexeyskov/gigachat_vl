#!/usr/bin/env python3
"""Same CPU-only controller, adding retained-file advice after verified derivation."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
from types import ModuleType
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OWNER='post_derive_cache_archival_stock_queue_v4'
SOURCE='scripts/research_archival_stock_queue_v4.py';TEST='tests/test_research_archival_stock_queue_v4.py'
READY=ROOT/'configs/research/archival_stock_queue_v4_ready.json'
V3='scripts/research_archival_stock_queue_v3.py'
V3_SHA='e4a624bb3493aa23d7dcdbda2b1032d62be5492d16947f6df51f343ddce3230a'
V3_READY=ROOT/'configs/research/archival_stock_queue_v3_ready.json'
V3_READY_SHA='ef3ccbab9686b2469aa0daf8a718804fb54cd7da24c2b9a66db35fd3b683fbb2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_binding():
 if sha(ROOT/V3)!=V3_SHA or sha(V3_READY)!=V3_READY_SHA:raise ValueError('Frozen CPU controller v3 release changed')
 value=json.loads(V3_READY.read_bytes())['source_sha256']
 if any(sha(ROOT/n)!=h for n,h in value.items()):raise ValueError('Frozen CPU controller dependency changed')
 return {**value,str(V3_READY.relative_to(ROOT)):V3_READY_SHA,SOURCE:sha(ROOT/SOURCE),TEST:sha(ROOT/TEST)}
def readiness():
 value=json.loads(READY.read_bytes())
 if value.get('owner')!=OWNER or value.get('status')!='cpu_post_derive_advice_tests_passed' or value.get('source_sha256')!=source_binding():raise ValueError('Exact post-derive cache controller release required')
 return {'path':str(READY),'sha256':sha(READY)}
def install_advice(publisher,live,advice,emit):
 original=publisher._derive_unlocked
 def derive(source,stage,plan,*a,**k):
  result=original(source,stage,plan,*a,**k)
  if not isinstance(result,dict) or result.get('all_payload_fields_and_backing_bits_exact') is not True:
   raise ValueError('Post-derive advice requires original exact whole-payload verification')
  queue=live['queue'];receipt=advice(queue,plan['run'],selected_source=source)
  row={'source':str(source),'verified_stage':str(stage),'verified_archive_sha256':result.get('archive_sha256'),
   'derived_archive_advised':False,'original_derived_payload_exact':True,'receipt':receipt}
  history=queue.state.setdefault('archival_post_derive_cache_advice',[]);history.append(row);del history[:-16]
  queue.persist();emit('archival_post_derive_cache_advice',run=str(plan['run']),source=str(source),
   actual_available_after_bytes=receipt.get('actual_available_after_bytes'),reclaimable_cache_credit_assumed_bytes=0)
  return result
 publisher._derive_unlocked=derive
 return derive
def load_controller():
 source_binding();v3=ModuleType('_post_derive_private_v3');v3.__file__=str(ROOT/V3)
 exec(compile((ROOT/V3).read_bytes(),v3.__file__,'exec'),v3.__dict__)
 v3.assert_cpu();module=v3.load_controller();original=module.queue_class
 def queue_class():
  parent=original();namespace=parent.compact_completed_job.__globals__;publisher=namespace['archival'];live={}
  install_advice(publisher,live,namespace['advise_completed'],namespace['emit'])
  class PostDeriveQueue(parent):
   def __init__(self,*a,**k):
    super().__init__(*a,**k)
    try:
     descriptor={'owner':OWNER,'controller_source':SOURCE,'controller_sha256':sha(ROOT/SOURCE),'v3_source_sha256':V3_SHA,'v3_ready_sha256':V3_READY_SHA,
      'post_derive_policy':'same frozen advise_completed after exact original derivation, before final publication guards',
      'stage_advice':False,'guard_or_checkpoint_changes':False,'same_SPEC_state_activation_and_training_profile':True,'release':readiness()}
     extra=self.state.setdefault('extra',{});old=extra.get(OWNER)
     if old is not None and old!=descriptor:raise ValueError('Post-derive controller profile changed')
     previous=dict(self.state['numerical_execution']);history=self.state.setdefault('controller_execution_history',[])
     if previous not in history:history.append(previous)
     extra[OWNER]=descriptor;self.state['numerical_execution'].update(controller_source=SOURCE,controller_sha256=descriptor['controller_sha256'],post_derive_advice_owner=OWNER)
     live['queue']=self
    except BaseException:self.lock_handle.close();raise
   def _guard(self):readiness();return super()._guard()
  return PostDeriveQueue
 module.queue_class=queue_class;return module
def main(argv=None):
 if 'torch' in sys.modules:raise RuntimeError('CPU archival controller v4 requires a fresh process before Torch import')
 import os
 os.environ['CUDA_VISIBLE_DEVICES']=''
 return load_controller().main(argv)
if __name__=='__main__':raise SystemExit(main())
