"""Versioned caption consumer for raw stock stages and admitted stock archives."""
from __future__ import annotations
import copy,hashlib,importlib,json
from pathlib import Path
from types import ModuleType
ROOT=Path(__file__).resolve().parents[2]
OWNER='research_caption_stock_native_b1_archived_v2'
NAMESPACE='caption_stock_native_b1_archived_v2'
SOURCE='src/eval/research_caption_stock_generation.py'
SOURCE_SHA='27fd94c7ecccdac6cd5d4a563df2c4f539391e8442bfedeb4b4932323df303c2'
RAW_READY=ROOT/'configs/research/caption_stock_evaluator_ready.json'
RAW_READY_SHA='5095810791f9cd7757e4f7062e99fc1b377cc88f48997f0388fb9d14914bce4c'
HELPER_READY=ROOT/'configs/research/archived_eval_v1_ready.json'
HELPER_READY_SHA='5258d5b454d77c87c349ad2bb3819143d65abaff701669b2f271e534434c2738'
READY=ROOT/'configs/research/caption_stock_archived_v2_ready.json'
NEW=('src/eval/research_caption_stock_archived_v2.py','scripts/eval_research_caption_stock_archived_v2.py','tests/test_research_caption_stock_archived_v2.py')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _raw_release():
 if sha(RAW_READY)!=RAW_READY_SHA or sha(ROOT/SOURCE)!=SOURCE_SHA:raise ValueError('Frozen raw stock caption release changed')
 value=json.loads(RAW_READY.read_bytes())
 if any(sha(ROOT/n)!=h for n,h in value['source_sha256'].items()):raise ValueError('Frozen raw stock caption dependency changed')
 return value
def _shared_release():
 if not isinstance(HELPER_READY_SHA,str) or len(HELPER_READY_SHA)!=64:raise ValueError('Archive caption profile is closed until the reviewed shared helper is pinned')
 if sha(HELPER_READY)!=HELPER_READY_SHA:raise ValueError('Pinned shared archive admission readiness changed')
 value=json.loads(HELPER_READY.read_bytes());sources=value['source_sha256']
 if not sources or any(sha(ROOT/n)!=h for n,h in sources.items()):raise ValueError('Shared archive admission dependency changed')
 return {'path':str(HELPER_READY),'sha256':HELPER_READY_SHA,'source_sha256':sources}
def _archive():
 _shared_release();module=importlib.import_module('src.utils.research_archived_eval_v1');module.verify_release();return module
def source_files():
 return {**_raw_release()['source_sha256'],**_shared_release()['source_sha256'],**{n:sha(ROOT/n) for n in NEW}}
def factory(bound_context=None):
 _raw_release();shared=_shared_release();archive=_archive()
 raw=ModuleType('_private_caption_stock_raw_v1');raw.__file__=str(ROOT/SOURCE);raw.__package__='src.eval'
 exec(compile((ROOT/SOURCE).read_bytes(),raw.__file__,'exec'),raw.__dict__)
 original_profile=raw.execution_profile
 def profile():
  value=copy.deepcopy(original_profile());value.update(owner=OWNER,checkpoint_format='native raw ZIP or explicitly admitted completed stock lossless archive',archived_SFT_supported=True)
  value['archive_admission']={'helper_ready_sha256':shared['sha256'],'source_sha256':shared['source_sha256'],'execution_profile':archive.execution_profile(),
   'decode_policy':'one guarded CPU compatibility load; exact payload/config/census before the frozen native constructor; measured headroom and manager SH lock'}
  return value
 raw.OWNER=OWNER;raw.NAMESPACE=NAMESPACE;raw.execution_profile=profile;raw.NEW=tuple(dict.fromkeys((*raw.NEW,*NEW,*shared['source_sha256'])))
 original_configure=raw._configure
 def configured(owner,stage_module):
  owner=original_configure(owner,stage_module);original_checkpoint=owner.checkpoint_context
  def checkpoint(run,role):
   return archive.checkpoint_context(run,role) if _archived(owner,run) else original_checkpoint(run,role)
  owner.checkpoint_context=checkpoint;runtime=owner.NativeRuntime
  original_provenance=owner._model_provenance
  def provenance(value,context):
   original_provenance(value,context);model=context['static']['checkpoint']
   if model.get('archive_admission'):
    loaded=value.get('loaded',{})
    if model.get('metadata_only') is not False or loaded.get('archived_evaluation')!=archive.execution_profile() or loaded.get('archived_physical_sha256')!=model['artifact']['sha256']:
     raise ValueError('Actual archive caption loading/profile/physical provenance differs')
  owner._model_provenance=provenance
  class Runtime(runtime):
   def load(self,checkpoint,*a,**k):
    model=(bound_context or {}).get('static',{}).get('checkpoint',{})
    if model.get('archive_admission'):
     if Path(checkpoint).resolve()!=Path(model['checkpoint_path']).resolve():raise ValueError('Bound archive caption checkpoint changed')
     owner._stock_native.value=archive.inference(model)
    elif owner.read_json(Path(checkpoint).parent/'.research_checkpoint.json').get('checkpoint_archival'):
     raise ValueError('Archive caption inference requires its fully admitted bound context')
    return super().load(checkpoint,*a,**k)
  owner.NativeRuntime=Runtime
  return owner
 raw._configure=configured;original_metadata=raw.metadata_plan
 def metadata(run,bundle=raw.DEFAULT_BUNDLE,*,checkpoint='best'):
  reader=raw._clone(raw.TEXT);config=reader.read(Path(run)/'resolved_config.json');raw.stock.validate_config(config)
  if not _archived(reader,run,read=reader.read):return original_metadata(run,bundle,checkpoint=checkpoint)
  inspected=archive.inspect(run,checkpoint)
  if inspected['config']!=config or config.get('stage')!='SFT':raise ValueError('Archive caption supports matching completed SFT config only')
  module=raw._module('SFT');owner=module._private_owner();raw._clone(raw.V2)._verify_bundle(owner,bundle)
  protocol=owner.read_json(Path(bundle)/'protocol.json');declared={row['path']:row['sha256'] for row in protocol['options']['source_manifests']}
  train=Path(config.get('manifest',''));train=(train if train.is_absolute() else ROOT/train).resolve()
  if declared.get(str(train))!=config.get('manifest_sha256') or config.get('text_manifest'):raise ValueError('Archive training data is outside the sealed caption exclusion review')
  return {'owner':OWNER,'stage':'SFT','role':checkpoint,'records':raw.RECORDS,'checkpoint':inspected,'execution_profile':profile(),
   'data_protocol_sha256':module.DATA_PROTOCOL_SHA256,'checkpoint_content_hash_computed':False,'full_RGB_checkpoint_and_source_audit_pending':True}
 raw.metadata_plan=metadata
 return raw
def _archived(reader,run,*,read=None):
 read=read or reader.read_json
 return any(read((Path(run)/role).resolve(strict=True)/'.research_checkpoint.json').get('checkpoint_archival') for role in ('best','last'))
def metadata_plan(*a,**k):return factory().metadata_plan(*a,**k)
def prepare_context(*a,**k):return factory().prepare_context(*a,**k)
def evaluate(context,*a,**k):return factory(context).evaluate(context,*a,**k)
def verify_release():
 value=json.loads(READY.read_bytes())
 if value.get('owner')!=OWNER or value.get('status')!='cpu_tests_passed_actual_gpu_not_run' or value.get('source_sha256')!=source_files() or value.get('execution_profile')!=factory().execution_profile():raise ValueError('Exact archive-aware stock caption CPU release required')
 return {'path':str(READY),'sha256':sha(READY)}
