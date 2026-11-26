"""Opt-in lossless archival proof/publication under a distinct stock numeric policy.

Frozen v1 codec, helper and publisher files are never edited or monkeypatched.
Historical stochastic metric replay is observational, never manufactured true.
"""
from __future__ import annotations
import hashlib,json,os
from collections import Counter
from pathlib import Path
from types import ModuleType
import zipfile
ROOT=Path(__file__).resolve().parents[2]
OWNER='actual_trained_small_archive_stock_gpu_v2'
PUBLICATION_OWNER='prospective_completed_small_archival_stock_v2'
ACTIVATION_PROTOCOL='actual_trained_small_archival_stock_activation_v2'
ACTIVATION=ROOT/'configs/research/trained_small_archival_activation_stock_v2.json'
READY=ROOT/'research_results/archive_stock_v2_ready.json'
STOCK_REPORT=ROOT/'research_results/selection_deterministic_3b_best170_v2/report.json'
STOCK_REPORT_SHA='8dbd08fbbcfa7806a8fc7c81ed1867ce5b979c9b0fcdf2a04f1b89567d59c1bc'
STOCK_READY_SHA='bb8c3ebf52d907b3605fb52b4ec29ce0af16aaf3b3cb495b89c4817dabb6a14a'
PINNED={'src/utils/research_archive_gpu_parity.py':'b7cc6d13cc4f504f37755dd0d3d2f43a5834a3f25bc0661a0e7a08de183fde05',
 'src/utils/research_checkpoint_archival.py':'cd507acf15f4c96abb9c0ca232a9c198f590906345ee6080d38325083e2777b3',
 'scripts/probe_research_selection_deterministic.py':'fc29685a8f460f02d348c3010220ff0d509cd9aaa69212a2f19bca28fd1954fd'}
NEW=('src/utils/research_archive_stock_v2.py','scripts/probe_research_archive_stock_v2.py','tests/test_research_archive_stock_v2.py')
GPU_PROFILE={'name':'source_archive_stock_deterministic_v2','CUBLAS_WORKSPACE_CONFIG':':4096:8',
 'torch_use_deterministic_algorithms':True,'warn_only':False,
 'native64':{'matmul_allow_tf32':False,'torch_threads':4},
 'internal116':{'matmul_allow_tf32':True,'torch_threads':8},'historical_saved_metric_required':False,
 'source_archive_raw_tokens_protocol_scores_required_exact':True,'legacy_validation_passed':False}

def read(path):
 p=Path(path)
 if p.is_symlink() or not p.is_file() or p.stat().st_size>32*2**20:raise ValueError('Bounded independent stock evidence required')
 return json.loads(p.read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stat(path):
 s=Path(path).stat();return {k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
def _pin():
 if any(sha(ROOT/n)!=h for n,h in PINNED.items()):raise ValueError('Frozen archival/stock dependency changed')
def _clone(name):
 _pin();p=ROOT/name;m=ModuleType('_private_stock_'+Path(name).stem);m.__file__=str(p);m.__package__='src.utils'
 exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m

def verify_stock_report(path=STOCK_REPORT):
 """Pure CPU metadata/raw audit. Running, partial or unproved348 remains closed."""
 from scripts import probe_research_selection_deterministic as stock
 from src.eval.research_metrics import aggregate_scores,score_prediction
 if Path(path).resolve()!=STOCK_REPORT.resolve() or sha(path)!=STOCK_REPORT_SHA:raise ValueError('Exact ROOT-approved actual348 report bytes required')
 d=stock.clone_native();proof=read(path);base=Path(path).parent
 if (sha(stock.READY)!=STOCK_READY_SHA or proof.get('owner')!=stock.OWNER or proof.get('status')!='completed_diagnostic_only' or
  proof.get('producer')!='ROOT' or proof.get('numerical_profile')!=stock.PROFILE or
  proof.get('source_inputs_unchanged') is not True or proof.get('actual116_completed') is not True or
  proof.get('legacy_validation_passed') is not False or proof.get('three_pass_token_mismatch_records')!=0 or
  proof.get('three_pass_prompt_mismatch_records')!=0 or proof.get('training_updates')!=0 or
  proof.get('release',{}).get('sha256')!=STOCK_READY_SHA):raise ValueError('Actual complete stable ROOT348 stock proof required')
 plan=proof['plan']
 if plan.get('source_sha256')!=d.source_binding() or plan.get('checkpoint_sha256')!=d.CHECKPOINT_SHA256:raise ValueError('Stock source pins differ')
 if stat(plan['checkpoint'])!=plan['checkpoint_stat'] or {k:str((Path(plan['run'])/k/'state.pt').resolve()) for k in ('best','last')}!=plan['roles']:
  raise ValueError('Stock reference weights/roles changed')
 restoration=proof['restoration']
 if (restoration.get('restored_tensor_values_exact_frozen_validator') is not True or
  not all(restoration.get('tied_embedding_alias',{}).get(k) is True for k in ('same_parameter_object','same_storage_layout')) or
  restoration.get('runtime_census_before_selection')!=proof.get('runtime_census_after_selection')):raise ValueError('Stock restoration/alias/census proof differs')
 env=proof['runtime_environment']
 if env.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8' or env.get('deterministic_algorithms') is not True or env.get('context',{}).get('matmul_allow_tf32') is not True or env.get('context',{}).get('torch_threads')!=8:
  raise ValueError('Actual stock numerical context absent')
 selections=read(base/'selection_records.json');records=selections['records'];first=read(base/'replayed_selection.json');repeats=proof['repeated_all116']
 if len(records)!=116 or len(first['predictions'])!=116 or len(repeats)!=2 or any(len(r)!=116 for r in repeats):raise ValueError('All348 rows required')
 if len({r['record_id'] for r in records})!=116 or set(Counter(r['task_group'] for r in records).values())!={29}:raise ValueError('Unique balanced116 stock scope required')
 if hashlib.sha256(d.canonical(records)).hexdigest()!=d.SELECTION_SHA256:raise ValueError('Stock canonical selection manifest differs')
 for index,record in enumerate(records):
  original=first['predictions'][index]
  if any(original.get(k)!=record.get(k) for k in d.FIELDS) or original['score']!=score_prediction(record,original['prediction']):raise ValueError('Stock scorer/source scope differs')
  for r in repeats:
   row=r[index]
   if (row['record_id']!=record['record_id'] or row['generation']!=original['generation'] or row['score']!=original['score']):raise ValueError('Stock same-model348 raw parity fails')
 for phase,rows in [('full116',first['predictions']),('full116_repeat_1',repeats[0]),('full116_repeat_2',repeats[1])]:
  for i,row in enumerate(rows):
   raw=read(base/'generated_rows'/f'{phase}_{i:03}.json');g=row['generation'];prompt=raw['prompt']
   if raw['record_id']!=records[i]['record_id'] or raw['generation']!=g or hashlib.sha256(prompt['prompt'].encode()).hexdigest()!=g['prompt_sha256']:
    raise ValueError('Stock raw row/prompt receipt differs')
 if first['metrics']['group_macro_score']!=aggregate_scores(first['predictions'],require_four_groups=True)['group_macro_score']:raise ValueError('Stock raw score aggregate differs')
 if sha(base/'source.zip')!=proof['source_archive']['sha256']:raise ValueError('Stock source archive digest differs')
 with zipfile.ZipFile(base/'source.zip') as z:
  if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(plan['source_sha256']):raise ValueError('Stock source archive members differ')
  for name,h in plan['source_sha256'].items():
   if hashlib.sha256(z.read(name)).hexdigest()!=h or sha(ROOT/name)!=h:raise ValueError('Stock source bytes differ')
 return {'path':str(Path(path).resolve()),'sha256':sha(path),'source_ready_sha256':STOCK_READY_SHA,
  'source_checkpoint_sha256':plan['checkpoint_sha256'],'profile':stock.PROFILE,'records':116,'passes':3,'raw_rows':348,'all_generation_fields_exact':True}

def publisher_v2():
 """Same owned publication/journal/lock/codec implementation; new activation only."""
 p=_clone('src/utils/research_checkpoint_archival.py');p.OWNER=PUBLICATION_OWNER;p.ACTIVATION_PROTOCOL=ACTIVATION_PROTOCOL;p.GPU_PROTOCOL=OWNER;p.ACTIVATION=ACTIVATION
 p.CODE_FILES=(*p.CODE_FILES,*NEW,'src/utils/research_archive_gpu_parity.py','scripts/probe_research_selection_deterministic.py','tests/test_research_selection_deterministic.py')
 p.activation_evidence=lambda path=None:activation_evidence(path,publisher=p)
 return p

def _historical(plan,selection):
 """References/protocol/order remain hard requirements; old metric is observation."""
 from src.eval.research_metrics import aggregate_scores,score_prediction
 files=plan['saved_selection_files'];old_metrics=read(files['metrics.json']['path']);old_protocol=read(files['protocol.json']['path'])
 if {k:v for k,v in old_protocol.items() if k!='validation_step'}!=selection['protocol']:raise ValueError('Historical data/scorer/generation protocol changed')
 if 'predictions.jsonl' not in files:raise ValueError('Historical raw116 reference binding required')
 with Path(files['predictions.jsonl']['path']).open() as f:old=[json.loads(line) for line in f if line.strip()]
 new=selection['predictions']
 if len(old)!=116 or len(new)!=116:raise ValueError('Historical116 count differs')
 for a,b in zip(old,new):
  for k in ('record_id','task_group','scorer','answers','source','choices','benchmark'):
   if a.get(k)!=b.get(k):raise ValueError('Historical reference/order/scorer differs')
  for row in (a,b):
   if row.get('score')!=score_prediction(row,row.get('prediction','')) or row.get('generation',{}).get('prediction')!=row.get('prediction'):
    raise ValueError('Historical/current raw score or prediction differs')
 if old_metrics['group_macro_score']!=aggregate_scores(old,require_four_groups=True)['group_macro_score'] or selection['metrics']['group_macro_score']!=aggregate_scores(new,require_four_groups=True)['group_macro_score']:
  raise ValueError('Historical/current aggregate is not backed by raw scores')
 return {'historical_comparison_observational':True,'historical_saved_metric_required':False,
  'saved_metric':old_metrics['group_macro_score'],'current_metric':selection['metrics']['group_macro_score'],
  'metric_difference':selection['metrics']['group_macro_score']-old_metrics['group_macro_score'],
  'historical_raw_mismatched_record_ids':[a['record_id'] for a,b in zip(old,new) if a['generation']!=b['generation']],
  'historical_raw_row_parity_claimed':False,'data_scorer_protocol_and_order_exact':True}

def gpu_helper_v2():
 """Small isolated control adaptation; frozen native numeric functions unchanged."""
 d=_clone('src/utils/research_archive_gpu_parity.py');d.verify_release() # verifies reviewed v1 before rebinding
 d.OWNER=OWNER;d.READY=READY;d.SOURCES=(*d.SOURCES,*NEW)
 def verify():
  value=read(READY)
  if value.get('owner')!=OWNER or value.get('status')!='cpu_tests_passed_actual_gpu_gate_closed' or value.get('source_sha256')!=d.source_binding() or value.get('profile')!=GPU_PROFILE:raise ValueError('Reviewed stock archive helper release required')
  return {'helper_ready':d.binding(READY),'publication_ready':d.binding(d.PUBLICATION_READY)}
 d.verify_release=verify
 original_prepare=d.prepare_plan
 def prepare(cpu_report,**kwargs):
  proof=verify_stock_report();plan=original_prepare(cpu_report,**kwargs);plan.update(stock348_proof=proof,numerical_profile=GPU_PROFILE)
  return plan
 d.prepare_plan=prepare
 original_assert=d.assert_plan
 def assert_plan(plan,**kwargs):
  original_assert(plan,**kwargs)
  if plan.get('stock348_proof')!=verify_stock_report():raise ValueError('Stock348 prerequisite changed')
 d.assert_plan=assert_plan
 d.check_saved_selection=_historical
 original_runtime=d.NativeRuntime
 capture={'output':None,'plan':None,'contexts':{}}
 def actual_context(kind):
  import torch
  result=dict(d.math_context(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
   warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
  expected=GPU_PROFILE[kind]
  if result['deterministic_algorithms'] is not True or result['warn_only'] is not False or result['CUBLAS_WORKSPACE_CONFIG']!=':4096:8' or any(result[k]!=v for k,v in expected.items()):
   raise ValueError('Actual stock generation policy differs from declared scope')
  return result
 class Runtime(original_runtime):
  def load(self,path,cache_dir,reserve_gb):
   import torch
   if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Deterministic workspace must precede Torch import')
   torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(4)
   self.phase='source' if str(path)==capture['plan']['source'] else 'archive';self.count=0
   return super().load(path,cache_dir,reserve_gb)
  def generate(self,model,record):
   context=actual_context('native64');capture['contexts'].setdefault(self.phase,{})['native64']=context
   result=super().generate(model,record)
   if actual_context('native64')!=context:raise ValueError('Native64 stock context changed')
   path=capture['output']/('raw_'+self.phase);path.mkdir(exist_ok=True)
   if self.count==0:d.write(path/'native_context.json',context)
   d.write(path/f'native64_{self.count:03}.json',dict(result,record_id=record['record_id']));self.count+=1
   return result
  def selection(self,model,config,check):
   context=actual_context('internal116');capture['contexts'].setdefault(self.phase,{})['internal116']=context
   from scripts.validate_research_checkpoint import select_validation_records
   from scripts.run_research_stage import load_records
   from src.eval.research_selection import evaluate_internal_group_macro
   from src.eval.research_inference import generate_prediction
   selected=select_validation_records(config,load_records)
   if len(selected['records'])!=116:raise ValueError('Exact116 selection required')
   path=capture['output']/('raw_'+self.phase);path.mkdir(exist_ok=True);count=0
   d.write(path/'selection_input.json',selected);d.write(path/'internal_context.json',context)
   def predict(m,r,**kwargs):
    nonlocal count
    check();result=generate_prediction(m,r,**kwargs);check()
    d.write(path/f'internal116_{count:03}.json',dict(result,record_id=r['record_id']));count+=1;return result
   result=evaluate_internal_group_macro(model,selected,max_new_tokens=64,generator=predict)
   if actual_context('internal116')!=context:raise ValueError('Internal116 stock context changed')
   d.write(path/'selection_complete.json',result);return result
 d.NativeRuntime=Runtime
 # Remove only the old historical metric rejection. Native generation/model math
 # and final source/archive equality remain unchanged in the pinned execute AST.
 import ast
 tree=ast.parse((ROOT/'src/utils/research_archive_gpu_parity.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
 class RemoveHistorical(ast.NodeTransformer):
  removed=0
  def visit_If(self,n):
   if (isinstance(n.body[0] if n.body else None,ast.Raise) and isinstance(n.body[0].exc,ast.Call) and n.body[0].exc.args and isinstance(n.body[0].exc.args[0],ast.Constant) and n.body[0].exc.args[0].value=='Source saved internal selection metric is not replayed exactly'):
    self.removed+=1;return None
   return self.generic_visit(n)
 fix=RemoveHistorical();node=fix.visit(node)
 if fix.removed!=1:raise ValueError('Exactly one pinned historical metric control branch required')
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ROOT/'src/utils/research_archive_gpu_parity.py'),'exec'),d.__dict__)
 original_execute=d.execute;original_write=d.write
 def write(path,value):
  path=Path(path)
  if isinstance(value,dict) and value.get('protocol')==OWNER and path.name=='report.json':
   value.update(actual_stock_phase_contexts=capture['contexts'],numerical_profile=GPU_PROFILE,stock348_proof=capture['plan']['stock348_proof'],historical_saved_metric_required=False,legacy_validation_passed=False)
   if value.get('status')=='passed':
    value.pop('selection_replay_exact',None);value['source_archive_selection_exact']=True
    value['legacy_codec_code_sha256']=value['code_sha256'];value['code_sha256']=publisher_v2().code_binding()
  original_write(path,value)
  fd=os.open(path.parent,os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
 d.write=write
 def execute(plan,output,**kwargs):
  capture.update(output=Path(output),plan=plan,contexts={})
  if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Stock workspace must precede Torch import')
  import torch
  torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(4)
  return original_execute(plan,output,**kwargs)
 d.execute=execute
 return d

def activation_evidence(path=None,*,publisher=None):
 """New schema cannot be mistaken for the frozen legacy activation."""
 p=publisher or publisher_v2();path=Path(path) if path is not None else ACTIVATION
 a=read(path);stock=verify_stock_report()
 if a.get('schema_version')!=2 or a.get('protocol')!=ACTIVATION_PROTOCOL or a.get('status')!='passed' or a.get('code_sha256')!=p.code_binding():raise p.ArchivalGateClosed('New actual stock archival activation required')
 cpu_path=Path(a['trained_probe_report']);gpu_path=Path(a['root_gpu_parity_report'])
 if sha(cpu_path)!=a['trained_probe_report_sha256'] or sha(gpu_path)!=a['root_gpu_parity_report_sha256']:raise p.ArchivalGateClosed('Actual stock CPU/GPU report bytes changed')
 cpu=read(cpu_path);gpu=read(gpu_path)
 from src.utils import research_trained_delta_probe as probe,research_trained_delta_probe_v2 as adapter
 required={'protocol':probe.OWNER,'status':'verified_actual_trained_derived_copy_only','decoded_model_bytes':probe.BASE_ELEMENTS*4,
  'all_payload_fields_and_backing_bits_exact':True,'scheduler_rng_config_compaction_metadata_preserved':True,'source_modified':False,'manager_roles_modified':False}
 if any(cpu.get(k)!=v for k,v in required.items()) or cpu.get('source_payload_sha256')!=cpu.get('restored_payload_sha256') or cpu.get('metadata_adapter',{}).get('source_sha256')!=adapter.source_binding():raise p.ArchivalGateClosed('Exact actual trained CPU proof required')
 config=read(Path(cpu['run'])/'resolved_config.json')
 if (config.get('model')!=probe.REPO or config.get('recipe')!='4f' or config.get('stage')!='SFT' or
  not 0<cpu.get('archive_bytes',0)<cpu.get('source_stat',{}).get('st_size',0) or cpu.get('archive_deleted') is not False):
  raise p.ArchivalGateClosed('Actual beneficial surviving full4f calibration required')
 required={'schema_version':1,'protocol':OWNER,'status':'passed','producer':'ROOT','device':'cuda','run':cpu['run'],
  'source_state_sha256':cpu['source_sha256'],'archive_sha256':cpu['archive_sha256'],'source_payload_sha256':cpu['source_payload_sha256'],
  'restored_payload_sha256':cpu['restored_payload_sha256'],'all_model_tensors_exact':True,'generated_token_ids_exact':True,
  'source_archive_selection_exact':True,'selection_records':116,'generation_records':64,'source_authoritative':True,'publication_performed':False,
  'historical_saved_metric_required':False,'legacy_validation_passed':False,'numerical_profile':GPU_PROFILE,'stock348_proof':stock,'code_sha256':p.code_binding()}
 if any(gpu.get(k)!=v for k,v in required.items()) or 'selection_replay_exact' in gpu:raise p.ArchivalGateClosed('Actual source/archive stock GPU equality required; legacy flag cannot substitute')
 for phase in ('source','archive'):
  for kind in ('native64','internal116'):
   observed=gpu.get('actual_stock_phase_contexts',{}).get(phase,{}).get(kind,{})
   expected=dict(GPU_PROFILE[kind],deterministic_algorithms=True,warn_only=False,CUBLAS_WORKSPACE_CONFIG=':4096:8')
   if any(observed.get(k)!=v for k,v in expected.items()):raise p.ArchivalGateClosed('Actual source/archive numerical scopes not attested')
 helper=gpu_helper_v2()
 if gpu.get('helper_source_sha256')!=helper.source_binding():raise p.ArchivalGateClosed('GPU helper/native source changed')
 source,archive=gpu['phases']['source'],gpu['phases']['archive']
 if source['model_tensors']!=archive['model_tensors'] or source['native64']!=archive['native64'] or source['internal116']!=archive['internal116'] or source['math_contexts']!=archive['math_contexts']:
  raise p.ArchivalGateClosed('Source/archive complete raw evidence differs')
 for phase in (source,archive):
  native=phase.get('native64',[]);selection=phase.get('internal116',{});predictions=selection.get('predictions',[])
  if len(native)!=64 or len({r.get('record_id') for r in native})!=64 or len(predictions)!=116 or len({r.get('record_id') for r in predictions})!=116:
   raise p.ArchivalGateClosed('Complete64/116 raw rows, not scalar claims, required')
  _historical(gpu['admission'],phase['internal116'])
  if not phase.get('model_tensors',{}).get('tensors') or phase.get('projector_frozen') is not True:
   raise p.ArchivalGateClosed('Complete GPU tensor census/frozen projector required')
 if not all(phase.get('all_parameter_and_buffer_bits_unchanged_after_forward') is True for phase in (source,archive)):raise p.ArchivalGateClosed('GPU state preservation absent')
 return {'path':str(path.resolve()),'sha256':sha(path),'cpu_report_sha256':sha(cpu_path),'gpu_report_sha256':sha(gpu_path),
  'calibration_source_sha256':cpu['source_sha256'],'calibration_archive_sha256':cpu['archive_sha256'],'code_sha256':p.code_binding(),
  'numerical_profile':GPU_PROFILE,'stock348_proof':stock,'historical_quality_validation_passed':False}
