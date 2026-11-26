"""Opt-in stock SmallThinker captioning; sealed caption/scorer math is reused."""
from __future__ import annotations
import ast,hashlib,json,os
from pathlib import Path
from types import ModuleType
from src.utils import research_stock_determinism as stock
ROOT=Path(__file__).resolve().parents[2]
OWNER='research_caption_stock_native_b1_v1'
NAMESPACE='caption_stock_native_b1_v1'
READY=ROOT/'configs/research/caption_stock_evaluator_ready.json'
PARENTS={'configs/research/caption_evaluator_v2_ready.json':'9fef32e40ccdd34876bdf76e7a066b0cf6abd4bfe6c388ba12e4500d4363e335',
 'configs/research/caption_stage_evaluator_ready.json':'960a05a4d0ab3d9633e591071bc25ac9ffdb2123a003a8dde207521e8cedb4f4'}
V1='src/eval/research_caption_generation.py';V2='src/eval/research_caption_generation_v2.py'
STAGES='src/eval/research_caption_stages.py';TEXT='src/eval/research_text_generation.py'
NEW=('src/eval/research_caption_stock_generation.py','scripts/eval_research_caption_stock_checkpoint.py','tests/test_research_caption_stock_generation.py')
DEFAULT_BUNDLE=ROOT/'datasets/research/caption-eval-pilot-en-v2'
RECORDS=372

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_parents():
 sources={}
 for name,expected in PARENTS.items():
  if sha(ROOT/name)!=expected:raise ValueError('Sealed caption readiness changed')
  sources.update(json.loads((ROOT/name).read_bytes())['source_sha256'])
 if any(sha(ROOT/n)!=h for n,h in sources.items()):raise ValueError('Sealed caption dependency changed')
 return sources
def execution_profile():
 return {'owner':OWNER,'numerical_profile':stock.definition(),'checkpoint_format':'native_Torch_ZIP_only',
  'legacy_or_mixed_profile_allowed':False,'archived_SFT_supported':False,'native_batch_size':1,'max_new_tokens':256,'fallback':False}
def _clone(name,tree=None,extra=None):
 sources=verify_parents();raw=(ROOT/name).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=sources[name]:raise ValueError('Pinned caption source changed')
 m=ModuleType('_private_stock_caption_'+Path(name).stem);m.__file__=str(ROOT/name);m.__package__='src.eval';m.__dict__.update(extra or {})
 exec(compile(ast.fix_missing_locations(tree) if tree is not None else raw,m.__file__,'exec'),m.__dict__)
 return m
class _LazyNative:
 def __init__(self):self.value=None
 def get(self):
  if self.value is None:self.value=stock.inference()
  return self.value
 def __getattr__(self,name):return getattr(self.get(),name)
def _base():
 native=_LazyNative();raw=(ROOT/V1).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=verify_parents()[V1]:raise ValueError('Pinned caption source changed before import rewrite')
 tree=ast.parse(raw)
 class Redirect(ast.NodeTransformer):
  count=0
  def visit_ImportFrom(self,node):
   if node.module=='src.eval.research_inference':
    self.count+=1;return [ast.Assign([ast.Name(a.asname or a.name,ast.Store())],ast.Attribute(ast.Name('_stock_native',ast.Load()),a.name,ast.Load())) for a in node.names]
   if node.module=='src.eval' and len(node.names)==1 and node.names[0].name=='research_inference':
    self.count+=1;return ast.Assign([ast.Name(node.names[0].asname or 'research_inference',ast.Store())],ast.Call(ast.Attribute(ast.Name('_stock_native',ast.Load()),'get',ast.Load()),[],[]))
   return node
 rewrite=Redirect();tree=rewrite.visit(tree)
 if rewrite.count!=2:raise ValueError('Pinned caption native import structure changed')
 m=_clone(V1,tree,{'_stock_native':native});v2=_clone(V2)
 m.DATA_PROTOCOL_SHA256=v2.DATA_PROTOCOL_SHA256;m.RECORDS=RECORDS;m.OWNER=OWNER
 return m
def _configure(owner,stage_module):
 owner.OWNER=OWNER;owner.SOURCES=tuple(dict.fromkeys((*owner.SOURCES,V2,'scripts/eval_research_caption_checkpoint_v2.py',STAGES,TEXT,*NEW,*stock.definition()['source_sha256'])))
 checkpoint=owner.checkpoint_context
 def checked(run,role):
  config=owner.read_json(Path(run)/'resolved_config.json');stock.validate_config(config)
  for name in ('best','last'):
   marker=owner.read_json((Path(run)/name).resolve(strict=True)/'.research_checkpoint.json')
   if marker.get('checkpoint_archival'):raise ValueError('Archived caption checkpoints need a separate lineage/decoder-memory profile; raw sources only')
  result=checkpoint(run,role)
  with Path(result['checkpoint_path']).open('rb') as handle:
   if handle.read(4)!=b'PK\x03\x04':raise ValueError('Stock caption v1 requires native Torch ZIP mmap payload')
  return result
 owner.checkpoint_context=checked;owner._checkpoint_identity=stage_module._artifact_identity
 original=owner.NativeRuntime
 class StockRuntime(original):
  def load(self,*a,**k):
   stock.activate();model,loaded=super().load(*a,**k);stock.assert_active()
   if loaded.get('numerical_profile')!=stock.definition():raise ValueError('Loaded caption checkpoint lacks exact stock profile')
   return model,loaded
  def describe(self,*a,**k):
   stock.assert_active();value=super().describe(*a,**k)
   import torch
   value['stock_execution']=execution_profile();value['runtime'].update(torch_threads=torch.get_num_threads(),deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
   return value
  def generate(self,*a,**k):
   stock.assert_active();value=super().generate(*a,**k);stock.assert_active();return value
 owner.NativeRuntime=StockRuntime
 provenance=owner._model_provenance
 def verify(value,context):
  provenance(value,context)
  if value.get('loaded',{}).get('numerical_profile')!=stock.definition() or value.get('stock_execution')!=execution_profile():raise ValueError('Caption loaded/profile provenance differs')
 owner._model_provenance=verify
 return owner
def _module(stage):
 v2=_clone(V2);stages=_clone(STAGES);v2.OWNER=stages.OWNER=OWNER
 if stage in ('A0','A1'):
  v2._private_owner=_base;stages._private_v2=lambda:v2;original=stages._private_owner
  stages._private_owner=lambda:_configure(original(),stages)
  return stages
 if stage!='SFT':raise ValueError('Completed stock A0/A1/SFT required')
 v2._private_owner=lambda:_configure(_base(),stages)
 return v2
def metadata_plan(run,bundle=DEFAULT_BUNDLE,*,checkpoint='best'):
 """Bounded metadata only: no RGB audit, weight reads, Tensor load or gold."""
 text=_clone(TEXT);run=Path(run).resolve(strict=True);config=text.read(run/'resolved_config.json');stock.validate_config(config)
 context={'model':{'repo':stock.SMALL}};plans={role:text.learned_plan(run,role,context) for role in ('best','last')}
 for value in plans.values():
  marker=text.read(Path(value['checkpoint']).parent/'.research_checkpoint.json')
  if marker.get('checkpoint_archival'):raise ValueError('Archived caption checkpoints need a separate lineage/decoder-memory profile; raw sources only')
 if checkpoint not in plans:raise ValueError('Retained best/last required')
 module=_module(config['stage']);owner=module._private_owner();_clone(V2)._verify_bundle(owner,bundle)
 protocol=owner.read_json(Path(bundle)/'protocol.json');declared={row['path']:row['sha256'] for row in protocol['options']['source_manifests']}
 train=Path(config.get('manifest',''));train=(train if train.is_absolute() else ROOT/train).resolve()
 if declared.get(str(train))!=config.get('manifest_sha256') or config.get('text_manifest'):raise ValueError('Training data is outside the sealed caption exclusion review')
 return {'owner':OWNER,'stage':config['stage'],'role':checkpoint,'records':RECORDS,'checkpoint':plans[checkpoint],
  'execution_profile':execution_profile(),'data_protocol_sha256':module.DATA_PROTOCOL_SHA256,
  'checkpoint_content_hash_computed':False,'full_RGB_checkpoint_and_source_audit_pending':True}
def prepare_context(run,bundle=DEFAULT_BUNDLE,*,checkpoint='best',cache_dir='cache/vision',assets=None):
 plan=metadata_plan(run,bundle,checkpoint=checkpoint);module=_module(plan['stage'])
 context=module.prepare_context(run,bundle,checkpoint=checkpoint,cache_dir=cache_dir,assets=assets)
 context['static']['stock_caption_profile']=execution_profile()
 return context
def output_path(context,requested=None):
 model=context['static']['checkpoint'];expected=Path(model['run'])/'evaluation'/NAMESPACE/model['config']['stage']/model['role']
 if requested is not None and Path(requested).resolve()!=expected.resolve():raise ValueError('Stock caption requires its own exact stage/role namespace')
 if any(p.is_symlink() for p in (expected,*expected.parents)):raise ValueError('Stock caption output cannot traverse symlinks')
 return expected
def evaluate(context,output=None,**kwargs):
 stock.validate_config(context['static']['checkpoint']['config'])
 if context['static'].get('stock_caption_profile')!=execution_profile():raise ValueError('Missing/mixed stock caption context')
 module=_module(context['static']['checkpoint']['config']['stage']);owner=module._private_owner()
 module._verify_context(owner,context);_clone(V2)._verify_bundle(owner,context['data']['bundle'])
 return owner.evaluate(context,output_path(context,output),**kwargs)
def verify_release():
 value=json.loads(READY.read_bytes());expected={**verify_parents(),**stock.definition()['source_sha256'],**{n:sha(ROOT/n) for n in NEW}}
 if value.get('owner')!=OWNER or value.get('status')!='cpu_tests_passed_actual_gpu_not_run' or value.get('source_sha256')!=expected or value.get('execution_profile')!=execution_profile():raise ValueError('Exact stock caption CPU release required')
 return {'path':str(READY),'sha256':sha(READY)}
