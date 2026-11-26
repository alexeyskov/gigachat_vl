"""Single-decode archive facade over the frozen native image evaluator."""
from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path
from src.utils import research_stock_determinism as stock
from src.utils import research_archived_eval_v2 as shared
from src.utils import research_trained_delta_probe as fileio
ROOT=Path(__file__).resolve().parents[2]
OWNER='archived_stock_native_single_decode_v1'
NAMESPACE='archived_stock_single_decode_v1'
NATIVE='scripts/eval_research_checkpoint.py'
NATIVE_SHA='cf06fcf7ba5c2be4860d58781478de7b51a101c21a4bc8cd95ebb333220baff2'
HELPER_READY_SHA='3a16684fd6a0dfd902e9645807b84355c5750d619345a26a5618f16fd00dd7a9'
OWN=('src/eval/research_archived_native_evaluation_v1.py','scripts/eval_research_archived_native_v1.py','tests/test_research_archived_native_evaluation_v1.py')
READY=ROOT/'configs/research/archived_native_single_decode_v1_ready.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_parents():
 if not isinstance(HELPER_READY_SHA,str) or len(HELPER_READY_SHA)!=64:raise ValueError('Single-decode evaluator is closed until phased helper release is pinned')
 if sha(ROOT/NATIVE)!=NATIVE_SHA or sha(shared.READY)!=HELPER_READY_SHA:raise ValueError('Frozen native/shared archive release changed')
 shared.verify_release()
def source_binding():
 verify_parents();return {**shared.source_binding(),NATIVE:NATIVE_SHA,str(shared.READY.relative_to(ROOT)):HELPER_READY_SHA,**{n:sha(ROOT/n) for n in OWN}}
def execution_profile():
 return {'owner':OWNER,'stock_numerical_profile':stock.definition(),'archive_loader':shared.execution_profile(),
  'CPU_payload_decodes_per_fresh_load':1,'stock_eval_main_used':False,'stock_inference_double_peek_used':False,
  'native_generation_ablation_scoring':'unchanged pinned native evaluator, greedy B1','namespace':NAMESPACE,'fallback':False}
def output_path(run,variant,role,requested=None):
 if variant not in ('none','shuffle','blank','text_only') or role not in ('best','last'):raise ValueError('Known native image variant and retained role required')
 expected=Path(run).resolve()/'evaluation'/NAMESPACE/variant/role
 if requested is not None and Path(requested).resolve()!=expected:raise ValueError('Single-decode evaluation requires its separate exact namespace')
 if any(p.is_symlink() for p in (expected,*expected.parents)):raise ValueError('Single-decode output cannot traverse symlinks')
 return expected
def metadata(run,role='best'):
 verify_parents();context=shared.inspect(run,role)
 return {'owner':OWNER,'status':'metadata_only_expected_SHA_no_payload_load','run':context['run'],'role':role,
  'expected_checkpoint_sha256':context['artifact']['sha256'],'physical_hash_verified':False,'execution_profile':execution_profile(),
  'Torch_imported':__import__('sys').modules.get('torch') is not None,'checkpoint_tensor_bodies_opened':False,'GPU_used':False}
def _sentinels(context):
 for item in context['files'].values():
  if fileio.stat(item['path'])!=item['stat'] or fileio.file_sha(item['path'])!=item['sha256']:raise ValueError('Archived evaluation metadata/source changed')
 for role,value in context['roles'].items():
  pointer=Path(context['run'])/role;marker=value['marker_file']
  if not pointer.is_symlink() or str(pointer.resolve())!=value['object'] or fileio.stat(marker['path'])!=marker['stat'] or fileio.file_sha(marker['path'])!=marker['sha256']:
   raise ValueError('Archived evaluation role/marker changed')
 for path,before in context['member_stats'].items():
  if fileio.stat(path)!=before:raise ValueError('Archived evaluation physical state changed')
def evaluator(context):
 """Actual runtime factory; caller establishes CUBLAS before Torch import."""
 verify_parents();stock.activate();native=shared.inference();module=stock._clone(NATIVE,{'src.eval.research_inference':native},bind_eval_protocol=True)
 loaded=module.load_research_checkpoint;generate=module.generate_prediction
 def load(checkpoint,**kwargs):
  if str(Path(checkpoint).resolve())!=context['checkpoint_path']:raise ValueError('Bound archive checkpoint changed before loading')
  _sentinels(context);stock.assert_active();model,provenance=loaded(checkpoint,**kwargs);stock.assert_active()
  if str(Path(checkpoint).resolve())!=context['checkpoint_path'] or provenance.get('archived_evaluation')!=shared.execution_profile() or provenance.get('archived_physical_sha256')!=context['artifact']['sha256'] or provenance.get('archived_publication_sha256')!=context['files']['checkpoint_archival.json']['sha256']:
   raise ValueError('Single-decode actual archive/model provenance differs')
  _sentinels(context);return model,provenance
 def predict(model,record,**kwargs):
  _sentinels(context);stock.assert_active();value=generate(model,record,**kwargs);stock.assert_active();_sentinels(context);return value
 module.load_research_checkpoint=load;module.generate_prediction=predict
 def bind(protocol):
  result=dict(protocol,numerical_profile=stock.definition(),single_decode_execution=execution_profile())
  result['source_sha256']={**protocol['source_sha256'],**source_binding()}
  result['single_decode_environment']={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','TOKENIZERS_PARALLELISM')}
  return result
 module.__dict__['_bind_stock_protocol']=bind
 return module
def evaluate(args):
 if args.gpu!=0 or not math.isfinite(args.reserve_gb) or args.reserve_gb<120:raise ValueError('Current archived Small profile requires GPU0 and120GiB disk reserve')
 verify_release();context=shared.inspect(args.run,args.checkpoint);output=output_path(args.run,args.image_ablation,args.checkpoint,args.output)
 module=evaluator(context)
 argv=['--run',str(args.run),'--manifest',str(args.manifest),'--output',str(output),'--checkpoint',args.checkpoint,
  '--max-new-tokens',str(args.max_new_tokens),'--gpu','0','--cache-dir',args.cache_dir,'--reserve-gb',str(args.reserve_gb),
  '--image-ablation',args.image_ablation,'--ablation-seed',str(args.ablation_seed)]
 if args.max_records is not None:argv.extend(['--max-records',str(args.max_records)])
 if args.resume:argv.append('--resume')
 _sentinels(context);result=module.main(argv);_sentinels(context);return result
def verify_release():
 value=json.loads(READY.read_bytes())
 if value.get('owner')!=OWNER or value.get('status')!='cpu_tests_passed_actual_gpu_not_run' or value.get('source_sha256')!=source_binding() or value.get('execution_profile')!=execution_profile():raise ValueError('Exact single-decode native evaluator CPU release required')
 return {'path':str(READY),'sha256':sha(READY)}
