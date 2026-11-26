"""Separate archive-only text consumer of the sealed public1680 producer."""
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path
from types import ModuleType
from src.utils import research_stock_determinism as stock
from src.utils import research_archived_eval_v1 as shared
ROOT=Path(__file__).resolve().parents[2]
OWNER='research_primary_text_stock_archived_v2'
NAMESPACE='text_stock_archived_v2'
RAW='src/eval/research_text_stock_generation.py'
RAW_SHA='d2ce7960029f2f492b55d9dddeb5c77c420019e3e9e0c90b360c8cc3e33cac8f'
RAW_READY=ROOT/'configs/research/primary_text_stock_producer_ready.json'
RAW_READY_SHA='4a504f239e7c208bb3df746a41c34bb01225da2d1f8f9b0fe570052392ea1e54'
SHARED_READY_SHA='5258d5b454d77c87c349ad2bb3819143d65abaff701669b2f271e534434c2738'
NEW=('src/eval/research_text_stock_archived_v2.py','scripts/eval_research_text_stock_archived_v2.py','tests/test_research_text_stock_archived_v2.py')
READY=ROOT/'configs/research/primary_text_stock_archived_v2_ready.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_parents():
 if sha(ROOT/RAW)!=RAW_SHA or sha(RAW_READY)!=RAW_READY_SHA or sha(shared.READY)!=SHARED_READY_SHA:raise ValueError('Pinned text/archive release changed')
 value=json.loads(RAW_READY.read_bytes())
 if any(sha(ROOT/n)!=h for n,h in value['source_sha256'].items()):raise ValueError('Pinned stock text dependency changed')
 shared.verify_release();return value
def source_files():return {**verify_parents()['source_sha256'],**shared.source_binding(),**{n:sha(ROOT/n) for n in NEW}}
def execution_profile():
 return {'owner':OWNER,'stock_numerical_definition':stock.definition(),'archive_loader':shared.execution_profile(),
  'helper_ready_sha256':SHARED_READY_SHA,'scope':'completed owned archived FULL Small SFT only; baselines and raw stages retain their previous owner',
  'weights':'preserved checkpoint dtypes','legacy_or_mixed_resume_allowed':False,'fallback':False}
def projection(context):
 """Compact exact metadata projection; expected SHA is not a physical proof."""
 value={key:copy.deepcopy(context[key]) for key in ('run','role','roles','checkpoint_path','config','files','member_stats','artifact')}
 value['artifact']['components_fully_hashed']=False
 value['retained_state_stats']={str(Path(role['object'])/'state.pt'):shared.stat(Path(role['object'])/'state.pt') for role in context['roles'].values()}
 return value
def output_path(plan,requested=None):
 expected=Path(plan['run'])/'evaluation'/NAMESPACE/plan['role']
 if requested is not None and Path(requested).resolve()!=expected.resolve():raise ValueError('Archived text requires its own exact run/role namespace')
 if any(p.is_symlink() for p in (expected,*expected.parents)):raise ValueError('Archived text output cannot traverse symlinks')
 return expected
def producer():
 verify_parents();raw=ModuleType('_private_archived_stock_text_v1');raw.__file__=str(ROOT/RAW);raw.__package__='src.eval'
 exec(compile((ROOT/RAW).read_bytes(),raw.__file__,'exec'),raw.__dict__)
 raw.OWNER=OWNER;raw.execution_profile=execution_profile;active={'plan':None}
 class Lazy(raw._LazyInference):
  def get(self):
   if self.value is None:self.value=shared.inference()
   return self.value
 raw._LazyInference=Lazy;module=raw.producer();module.source_files=source_files
 original_plan=module.learned_plan
 def learned(run,role,context):
  plan=original_plan(run,role,context);admission=shared.inspect(run,role)
  if plan['stage']!='SFT' or admission['config']['stage']!='SFT' or admission['checkpoint_path']!=plan['checkpoint']:
   raise ValueError('Archived text requires matching completed owned FULL SFT')
  plan['archive_metadata']=projection(admission);return plan
 module.learned_plan=learned
 def baseline(*a,**k):raise ValueError('Raw/bf16/nf4/full-FP32 baselines use the preserved stock text owner')
 module.baseline_plan=baseline;runtime=module.NativeTextRuntime
 class ArchivedRuntime(runtime):
  def load_baseline(self,*a,**k):return baseline()
  def load_learned(self,checkpoint,*a,**k):
   plan=active['plan']
   if plan is None or str(Path(checkpoint).resolve())!=plan['checkpoint']:raise ValueError('Bound archived text plan required before loading')
   full=shared.checkpoint_context(plan['run'],plan['role'])
   if projection(full)!=plan['archive_metadata'] or full.get('metadata_only') is not False:raise ValueError('Archive metadata/full physical admission changed')
   module._stock_native.value=shared.inference(full)
   model,loaded=super().load_learned(checkpoint,*a,**k)
   report=full['files'][shared._publisher().REPORT]['sha256']
   if loaded.get('archived_evaluation')!=shared.execution_profile() or loaded.get('archived_physical_sha256')!=full['artifact']['sha256'] or loaded.get('archived_publication_sha256')!=report:
    raise ValueError('Actual archive loader/profile/physical provenance differs')
   return model,loaded
 module.NativeTextRuntime=ArchivedRuntime
 original_assert=module.assert_static
 def assert_static(context,static):
  original_assert(context,static);metadata=static['plan'].get('archive_metadata')
  if not metadata or metadata['config']['stage']!='SFT':raise ValueError('Archived text protocol lacks admission metadata')
  for item in metadata['files'].values():
   if module.stat(item['path'])!=item['stat'] or module.sha(item['path'])!=item['sha256']:raise ValueError('Archive publication/activation metadata changed')
  for name,value in metadata['roles'].items():
   pointer=Path(metadata['run'])/name;marker=value['marker_file']
   if not pointer.is_symlink() or str(pointer.resolve())!=value['object'] or module.stat(marker['path'])!=marker['stat'] or module.sha(marker['path'])!=marker['sha256']:
    raise ValueError('Archive role/complete marker changed')
  for path,expected in metadata['retained_state_stats'].items():
   if module.stat(path)!=expected:raise ValueError('Retained archive state changed')
 module.assert_static=assert_static;evaluate=module.evaluate
 def run(context,plan,output=None,**kwargs):
  if plan.get('kind')!='learned' or plan.get('stage')!='SFT' or 'archive_metadata' not in plan:raise ValueError('Archived learned SFT plan required')
  if active['plan'] is not None:raise ValueError('Concurrent text evaluation in one owner is refused')
  active['plan']=plan
  try:return evaluate(context,plan,output_path(plan,output),**kwargs)
  finally:active['plan']=None
 module.evaluate=run;return module
def verify_release():
 value=json.loads(READY.read_bytes())
 if value.get('owner')!=OWNER or value.get('status')!='cpu_tests_passed_actual_gpu_not_run' or value.get('source_sha256')!=source_files() or value.get('execution_profile')!=execution_profile():raise ValueError('Exact archived text CPU release required')
 return {'path':str(READY),'sha256':sha(READY)}
