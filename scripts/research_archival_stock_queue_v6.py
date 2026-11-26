#!/usr/bin/env python3
"""Immutable completed-publication revalidation around the frozen v5 queue."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OWNER='immutable_publication_archival_stock_queue_v6'
SOURCE='scripts/research_archival_stock_queue_v6.py';TEST='tests/test_research_archival_stock_queue_v6.py'
V5='scripts/research_archival_stock_queue_v5.py';V5_SHA='d112d49616139f2b517de7ab3dc57cf1b693dc8e333d87c87476500d865a99b0'
V5_READY=ROOT/'configs/research/archival_stock_queue_v5_ready.json';V5_READY_SHA='47a7cd96ad5c0d3323adb75b555b036ed60331939e6c39aef9873644feda0af1'
READY=ROOT/'configs/research/archival_stock_queue_v6_ready.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_binding():
 if sha(ROOT/V5)!=V5_SHA or sha(V5_READY)!=V5_READY_SHA:raise ValueError('Frozen v5 controller changed')
 value=json.loads(V5_READY.read_bytes())['source_sha256']
 if any(sha(ROOT/n)!=h for n,h in value.items()):raise ValueError('Frozen v5 dependency changed')
 return {**value,str(V5_READY.relative_to(ROOT)):V5_READY_SHA,SOURCE:sha(ROOT/SOURCE),TEST:sha(ROOT/TEST)}
def readiness():
 value=json.loads(READY.read_bytes())
 if value.get('owner')!=OWNER or value.get('status')!='immutable_completed_publication_CPU_tests_passed' or value.get('source_sha256')!=source_binding():raise ValueError('Exact immutable-publication controller release required')
 return {'path':str(READY),'sha256':sha(READY)}
def install_readonly(publisher):
 original=publisher.publish_completed_checkpoint
 def publish(run,*a,**k):
  run=Path(run).resolve();report=run/publisher.REPORT;journal=run/publisher.JOURNAL
  if os.path.lexists(report):
   value=publisher.read(report)
   if value.get('status') not in ('publishing','completed'):raise ValueError('Unknown/failed archival report preserved; no normalization')
   if value.get('status')=='completed' and os.path.lexists(journal):
    transaction=publisher.read(journal);plan=publisher.eligibility(run)
    if transaction.get('phase')=='completed' and all(item['marker'].get('checkpoint_archival') for item in plan['roles'].values()):
     before={p:(p.read_bytes(),p.stat()) for p in (report,journal)}
     result=publisher.validate_completed_archival(run)
     if any(p.read_bytes()!=raw or p.stat()!=stat for p,(raw,stat) in before.items()):raise ValueError('Completed archival metadata changed during readonly validation')
     if result!=value:raise ValueError('Completed archival read-only validation differs from exact durable report')
     return result
  return original(run,*a,**k)
 publisher.publish_completed_checkpoint=publish;return publish
def load_controller(gpu_proof_sha):
 source_binding();v5=types.ModuleType('_immutable_publication_private_v5');v5.__file__=str(ROOT/V5)
 exec(compile((ROOT/V5).read_bytes(),v5.__file__,'exec'),v5.__dict__)
 module=v5.load_controller(gpu_proof_sha);original=module.queue_class
 def queue_class():
  parent=original();publisher=parent.compact_completed_job.__globals__['archival'];install_readonly(publisher)
  class ImmutablePublicationQueue(parent):
   def __init__(self,*a,**k):
    super().__init__(*a,**k)
    try:
     descriptor={'owner':OWNER,'controller_source':SOURCE,'controller_sha256':sha(ROOT/SOURCE),'v5_source_sha256':V5_SHA,'v5_ready_sha256':V5_READY_SHA,
      'completed_publication_policy':'exact full readonly validate_completed_archival; no disk snapshot/report rewrite',
      'pending_publication_policy':'unchanged frozen publisher','proof_or_scientific_guards_relaxed':False,'release':readiness()}
     extra=self.state.setdefault('extra',{});existing=extra.get(OWNER)
     if existing is not None and existing!=descriptor:raise ValueError('Immutable publication controller profile changed')
     previous=dict(self.state['numerical_execution']);history=self.state.setdefault('controller_execution_history',[])
     if previous not in history:history.append(previous)
     extra[OWNER]=descriptor;self.state['numerical_execution'].update(controller_source=SOURCE,controller_sha256=descriptor['controller_sha256'],immutable_publication_owner=OWNER)
    except BaseException:self.lock_handle.close();raise
   def _guard(self):readiness();return super()._guard()
  return ImmutablePublicationQueue
 module.queue_class=queue_class;return module
def main(argv=None):
 if 'torch' in sys.modules:raise RuntimeError('CPU v6 controller requires a fresh process before Torch import')
 os.environ['CUDA_VISIBLE_DEVICES']='';parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--gpu-proof-sha');options,remaining=parser.parse_known_args(argv)
 try:module=load_controller(options.gpu_proof_sha)
 except (OSError,ValueError,KeyError) as error:
  if '--run-queue' in remaining:raise
  print(json.dumps({'owner':OWNER,'status':'closed_GPU_proof_required','reason':str(error),'jobs_started':False}));return 2
 return module.main(remaining)
if __name__=='__main__':raise SystemExit(main())
