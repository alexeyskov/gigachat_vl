import copy,json
from pathlib import Path
import pytest
from src.utils import research_archive_stock_v2 as v2


def test_frozen_helper_and_publisher_globals_unchanged():
 from src.utils import research_archive_gpu_parity as old,research_checkpoint_archival as oldp
 before=(old.OWNER,old.check_saved_selection,oldp.OWNER,oldp.activation_evidence)
 helper=v2.gpu_helper_v2();publisher=v2.publisher_v2()
 assert helper.OWNER==v2.OWNER and publisher.OWNER==v2.PUBLICATION_OWNER
 assert (old.OWNER,old.check_saved_selection,oldp.OWNER,oldp.activation_evidence)==before
 assert helper.NativeRuntime.__mro__[1].__module__!='src.utils.research_archive_gpu_parity'


def historical(tmp_path):
 from src.eval.research_metrics import aggregate_scores
 old=[]
 for i in range(116):
  group=('general_vqa','ocr_documents','charts_diagrams','knowledge_reasoning')[i//29]
  old.append({'record_id':str(i),'answers':['yes'],'source':{'split':'train'},'scorer':'exact_match','task_group':group,
   'generation':{'prediction':'no','generated_token_ids':[1]},'prediction':'no','score':0.})
 current=copy.deepcopy(old);current[0].update(prediction='yes',score=1.);current[0]['generation']={'prediction':'yes','generated_token_ids':[2]}
 protocol={'name':'internal','validation_step':170}
 (tmp_path/'metrics').write_text(json.dumps(dict(aggregate_scores(old,require_four_groups=True),validation_step=170)))
 (tmp_path/'protocol').write_text(json.dumps(protocol))
 with (tmp_path/'raw').open('w') as f:
  for r in old:f.write(json.dumps(r)+'\n')
 plan={'saved_selection_files':{k:{'path':str(tmp_path/n)} for k,n in [('metrics.json','metrics'),('protocol.json','protocol'),('predictions.jsonl','raw')]}}
 selection={'metrics':aggregate_scores(current,require_four_groups=True),'protocol':{'name':'internal'},'predictions':current}
 return plan,selection


def test_old_metric_difference_observed_but_same_reference_protocol_required(tmp_path):
 plan,selection=historical(tmp_path);result=v2._historical(plan,selection)
 assert result['saved_metric']==0. and result['current_metric']==pytest.approx(100/116) and result['metric_difference']==pytest.approx(100/116)
 assert result['historical_saved_metric_required'] is False and result['historical_comparison_observational']
 assert result['historical_raw_row_parity_claimed'] is False

@pytest.mark.parametrize('mutation',['count','reference','protocol','order'])
def test_historical_data_semantics_not_relaxed(tmp_path,mutation):
 plan,selection=historical(tmp_path)
 if mutation=='count':selection['predictions'].pop()
 elif mutation=='reference':selection['predictions'][0]['answers']=['wrong']
 elif mutation=='protocol':selection['protocol']['name']='wrong'
 else:selection['predictions'][0]['record_id']='wrong'
 with pytest.raises(ValueError):v2._historical(plan,selection)


def test_actual348_prerequisite_is_verified_without_torch_or_weights(monkeypatch):
 original=Path.open
 def guarded(path,*a,**k):
  if str(path).endswith('state.pt'):pytest.fail('Stock proof audit must not read weight bodies')
  return original(path,*a,**k)
 monkeypatch.setattr(Path,'open',guarded)
 result=v2.verify_stock_report()
 assert result['sha256']==v2.STOCK_REPORT_SHA and result['raw_rows']==348


def test_running_or_mutated_stock_report_cannot_substitute(tmp_path):
 p=tmp_path/'report.json';p.write_text(json.dumps({'status':'running'}))
 with pytest.raises(ValueError,match='Exact ROOT-approved'):v2.verify_stock_report(p)


def test_old_legacy_activation_stays_closed_and_new_schema_refuses_old(tmp_path):
 p=tmp_path/'activation.json';p.write_text(json.dumps({'schema_version':1,'protocol':'actual_trained_small_archival_activation_v1','status':'passed'}))
 with pytest.raises(ValueError):v2.activation_evidence(p)


def test_new_publisher_code_binding_includes_new_contract():
 p=v2.publisher_v2();bindings=p.code_binding()
 assert all(name in bindings for name in v2.NEW) and 'src/utils/research_checkpoint_archival.py' in bindings
 assert p.ACTIVATION!=__import__('src.utils.research_checkpoint_archival',fromlist=['ACTIVATION']).ACTIVATION


def test_new_publisher_default_is_closed_without_torch_or_files(tmp_path):
 import subprocess,sys
 command="import sys; from src.utils.research_archive_stock_v2 import publisher_v2; p=publisher_v2(); r=p.preflight("+repr(str(tmp_path/'absent'))+"); assert r['status']=='gate_closed'; assert 'torch' not in sys.modules"
 result=subprocess.run([sys.executable,'-c',command],capture_output=True,text=True,cwd=v2.ROOT)
 assert result.returncode==0,result.stderr
 assert not list(tmp_path.iterdir())


def test_new_publication_shared_roles_payload_and_crash_recovery(tmp_path,monkeypatch):
 # Run the proven codec/role tests through the NEW private owner/activation API.
 import test_research_trained_delta_probe as fixture_module
 import test_research_checkpoint_archival as oldtests
 p=v2.publisher_v2();monkeypatch.setattr(oldtests,'archival',p)
 fixture=fixture_module.fixture.__wrapped__(tmp_path,monkeypatch)
 prepared=oldtests.prepared.__wrapped__(fixture,monkeypatch)
 oldtests.test_shared_best_last_exact_payload_marker_and_idempotence(prepared)
 report=p.read(prepared['run']/p.REPORT)
 assert report['protocol']==v2.PUBLICATION_OWNER and p.read(prepared['run']/p.JOURNAL)['protocol']==v2.PUBLICATION_OWNER

@pytest.mark.parametrize('failure',['after_first_pointer','after_derived_journal'])
def test_new_publication_reuses_atomic_recovery_without_weight_or_role_loss(tmp_path,monkeypatch,failure):
 import test_research_trained_delta_probe as fixture_module
 import test_research_checkpoint_archival as oldtests
 p=v2.publisher_v2();monkeypatch.setattr(oldtests,'archival',p)
 fixture=fixture_module.fixture.__wrapped__(tmp_path,monkeypatch);prepared=oldtests.prepared.__wrapped__(fixture,monkeypatch)
 oldtests.test_crash_recovery_does_not_repeat_encoding_or_change_role_payload(prepared,monkeypatch,failure)


def activation_fixture(tmp_path):
 from src.utils import research_trained_delta_probe as probe,research_trained_delta_probe_v2 as adapter
 p=v2.publisher_v2();plan,selection=historical(tmp_path)
 run=tmp_path/'run';run.mkdir();(run/'resolved_config.json').write_text(json.dumps({'model':probe.REPO,'recipe':'4f','stage':'SFT'}))
 cpu={'protocol':probe.OWNER,'status':'verified_actual_trained_derived_copy_only','decoded_model_bytes':probe.BASE_ELEMENTS*4,
  'all_payload_fields_and_backing_bits_exact':True,'scheduler_rng_config_compaction_metadata_preserved':True,'source_modified':False,'manager_roles_modified':False,
  'source_payload_sha256':'c'*64,'restored_payload_sha256':'c'*64,'metadata_adapter':{'source_sha256':adapter.source_binding()},
  'run':str(run),'archive_bytes':1,'source_stat':{'st_size':2},'archive_deleted':False,'source_sha256':'a'*64,'archive_sha256':'b'*64}
 stock=v2.verify_stock_report();census={'tensors':{'model.param':{'sha256':'d'*64}}}
 phase={'model_tensors':census,'native64':[{'record_id':str(i),'prediction':'yes'} for i in range(64)],'internal116':selection,'math_contexts':{},'projector_frozen':True,'all_parameter_and_buffer_bits_unchanged_after_forward':True}
 gpu={'schema_version':1,'protocol':v2.OWNER,'status':'passed','producer':'ROOT','device':'cuda','run':cpu['run'],
  'source_state_sha256':cpu['source_sha256'],'archive_sha256':cpu['archive_sha256'],'source_payload_sha256':cpu['source_payload_sha256'],
  'restored_payload_sha256':cpu['restored_payload_sha256'],'all_model_tensors_exact':True,'generated_token_ids_exact':True,
  'source_archive_selection_exact':True,'selection_records':116,'generation_records':64,'source_authoritative':True,'publication_performed':False,
  'historical_saved_metric_required':False,'legacy_validation_passed':False,'numerical_profile':v2.GPU_PROFILE,'stock348_proof':stock,'code_sha256':p.code_binding(),
  'helper_source_sha256':v2.gpu_helper_v2().source_binding(),'phases':{'source':phase,'archive':copy.deepcopy(phase)},'admission':plan,
  'actual_stock_phase_contexts':{name:{kind:dict(v2.GPU_PROFILE[kind],deterministic_algorithms=True,warn_only=False,CUBLAS_WORKSPACE_CONFIG=':4096:8') for kind in ('native64','internal116')} for name in ('source','archive')}}
 cpu_path=tmp_path/'cpu.json';gpu_path=tmp_path/'gpu.json';cpu_path.write_text(json.dumps(cpu));gpu_path.write_text(json.dumps(gpu))
 activation={'schema_version':2,'protocol':v2.ACTIVATION_PROTOCOL,'status':'passed','code_sha256':p.code_binding(),
  'trained_probe_report':str(cpu_path),'trained_probe_report_sha256':v2.sha(cpu_path),'root_gpu_parity_report':str(gpu_path),'root_gpu_parity_report_sha256':v2.sha(gpu_path)}
 path=tmp_path/'activation.json';path.write_text(json.dumps(activation));return p,path,cpu_path,gpu_path,activation


def test_new_activation_accepts_source_archive_equality_without_historical_quality_claim(tmp_path):
 p,path,_,_,_=activation_fixture(tmp_path);proof=v2.activation_evidence(path,publisher=p)
 assert proof['historical_quality_validation_passed'] is False and proof['stock348_proof']['raw_rows']==348
 from src.utils import research_checkpoint_archival as old
 with pytest.raises(old.ArchivalGateClosed):old.activation_evidence(path)

@pytest.mark.parametrize('mutation',['old_flag','phase_token','empty64','scalar_only','wrong_context','cpu_bits','wrong_source'])
def test_resealed_gpu_or_cpu_metadata_cannot_bypass_new_hard_requirements(tmp_path,mutation):
 p,path,cpu_path,gpu_path,activation=activation_fixture(tmp_path);gpu=json.loads(gpu_path.read_text());cpu=json.loads(cpu_path.read_text())
 if mutation=='old_flag':gpu['selection_replay_exact']=True
 elif mutation=='phase_token':gpu['phases']['archive']['native64'][0]['prediction']='different'
 elif mutation=='empty64':
  for v in gpu['phases'].values():v['native64']=[]
 elif mutation=='scalar_only':
  for v in gpu['phases'].values():v['model_tensors']={}
 elif mutation=='wrong_context':gpu['actual_stock_phase_contexts']['source']['internal116']['deterministic_algorithms']=False
 elif mutation=='cpu_bits':cpu['all_payload_fields_and_backing_bits_exact']=False
 else:gpu['source_state_sha256']='0'*64
 gpu_path.write_text(json.dumps(gpu));cpu_path.write_text(json.dumps(cpu));activation.update(root_gpu_parity_report_sha256=v2.sha(gpu_path),trained_probe_report_sha256=v2.sha(cpu_path));path.write_text(json.dumps(activation))
 with pytest.raises(p.ArchivalGateClosed):v2.activation_evidence(path,publisher=p)


def test_partial_native_and_selection_raw_survive_late_failure(tmp_path,monkeypatch):
 import torch
 from types import SimpleNamespace
 from src.utils import research_lossless_delta as codec
 from src.eval import research_parallel_protocol as protocol,research_selection,research_inference
 from scripts import validate_research_checkpoint as validator
 helper=v2.gpu_helper_v2();base=helper.NativeRuntime.__mro__[1]
 source=tmp_path/'source';source.mkdir();weight=source/'original.pt';weight.write_bytes(b'authoritative')
 records=[{'record_id':str(i),'question':'question','image_paths':[]} for i in range(116)]
 model=SimpleNamespace(projector=SimpleNamespace(parameters=lambda:[]))
 monkeypatch.setattr(base,'load',lambda *a:(model,{'fixture':True}))
 generated={'prediction':'yes','generated_token_ids':[1],'generated_tokens':1,'prompt_sha256':'a'*64,'prefill_tokens':1,'visual_tokens':0,'logits_to_keep':1}
 monkeypatch.setattr(base,'generate',lambda *a:copy.deepcopy(generated))
 monkeypatch.setattr(helper,'tensor_census',lambda *a:{'fixture':True})
 monkeypatch.setattr(helper,'assert_plan',lambda *a,**k:None)
 monkeypatch.setattr(helper,'rows',lambda *a:records)
 monkeypatch.setattr(codec,'memory_guard',lambda *a:{'fixture':True})
 monkeypatch.setattr(protocol,'select_guard_indices',lambda rows:[0,1])
 monkeypatch.setattr(validator,'select_validation_records',lambda *a:{'records':records})
 monkeypatch.setattr(research_inference,'generate_prediction',lambda *a,**k:copy.deepcopy(generated))
 def late(model,selection,*,generator,**kwargs):
  generator(model,selection['records'][0],max_new_tokens=64)
  raise RuntimeError('late selection fixture failure')
 monkeypatch.setattr(research_selection,'evaluate_internal_group_macro',late)
 monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
 class Guard:
  phase='admission';samples=[]
  def check(self):pass
  def fast_check(self):pass
  def __enter__(self):return self
  def __exit__(self,*a):pass
 plan={'run':str(source),'role':'best','source':str(weight),'archive':'not_loaded','reserve_gb':120,'gate':{'roles':{},'config':{}},
  'source_stat':{'st_size':13},'archive_stat':{'st_size':1},'decoded_model_bytes':1,'stock348_proof':{'fixture':True},'manifest':{'path':'fixture'}}
 output=tmp_path/'output';old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled();threads=torch.get_num_threads();tf32=torch.backends.cuda.matmul.allow_tf32
 try:
  with pytest.raises(RuntimeError,match='late selection'):helper.execute(plan,output,nonce='f'*64,guard=Guard())
  assert (output/'raw_source/native64_000.json').is_file() and (output/'raw_source/native64_001.json').is_file()
  assert (output/'raw_source/internal116_000.json').is_file() and (output/'raw_source/selection_input.json').is_file()
  assert json.loads((output/'report.json').read_text())['status']=='failed'
  assert weight.read_bytes()==b'authoritative' and not (output/'raw_archive').exists()
 finally:
  torch.use_deterministic_algorithms(old,warn_only=warn);torch.set_num_threads(threads);torch.backends.cuda.matmul.allow_tf32=tf32
