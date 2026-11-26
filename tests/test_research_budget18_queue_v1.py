"""Budget DAG/lifecycle/adoption CPU fixtures; never launch model/GPU workers."""
import copy,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import scripts.research_budget18_queue_v1 as q


def fake_queue(spec=None):
 x=object.__new__(q.Queue);x.spec=spec or {'jobs':[]};x.state={'jobs':{}};x.stopped=False;x.child=None;x.child_info=None;x.persist=lambda:None;x.handle=SimpleNamespace(close=lambda:None);x.prune=lambda:None;return x

def test_exact42_DAG_18_cells_no_R2_A1_and_fullCPU8():
 jobs=q.jobs();assert len(jobs)==42 and len({j['id'] for j in jobs})==42
 assert {s:sum(j['stage']==s for j in jobs) for s in ('A0','A1','SFT')}=={'A0':12,'A1':12,'SFT':18}
 seen=set()
 for job in jobs:
  assert set(job['requires'])<=seen;seen.add(job['id']);assert not(job['stage']=='A1' and job['recipe']=='4f')
  assert q.value(job['argv'],'--eval-every')=='10000' and q.value(job['argv'],'--effective-batch-size')=='64'
  assert q.value(job['argv'],'--seed')=='42' and q.value(job['argv'],'--reserve-gb')=='120'
  if job['stage']=='SFT':
   assert q.value(job['argv'],'--text-manifest')==str(q.DATA/'text_train.jsonl')
   assert q.value(job['argv'],'--selection-manifest') is None and q.value(job['argv'],'--text-validation') is None
   assert q.value(job['argv'],'--optimizer-backend')==('cpu_grad_bnb_paged_adamw8' if job['recipe']=='4f' else 'torch_adamw32')


def test_manual_paths_and_batch_policy_match_actual_ROOT_jobs():
 jobs=q.jobs();assert [j['id'] for j in jobs[:3]]==list(q.MANUAL_SERVICES)
 assert q.value(jobs[0]['argv'],'--batch-size')=='16' and q.value(jobs[1]['argv'],'--batch-size')=='16' and q.value(jobs[2]['argv'],'--batch-size')=='32'
 for j in jobs[3:]:assert q.value(j['argv'],'--batch-size')==('64' if j['stage']=='A0' else '32' if j['backend']=='smallthinker4' else '16')


def test_both_vision_contracts_and_sharedBF16_consumers():
 jobs=q.jobs()
 for j in jobs:
  assert q.value(j['argv'],'--vision-preprocessing-profile')==('qwen_native_bounded_729' if j['encoder']=='qwen25' else None)
  if j['stage']=='A0' and j['recipe']=='4f':
   users=[x for x in jobs if j['id'] in x['requires']];assert {(x['stage'],x['recipe']) for x in users}=={('SFT','4f')}
  if j['stage']=='A0' and j['recipe']=='1b':
   users=[x for x in jobs if j['id'] in x['requires']];assert {(x['stage'],x['recipe']) for x in users}=={('A1','1b'),('A1','1c')}


def test_real_plan_all_backend_sourcebindings_validate_metadata_only():
 plan=q.make_plan();q.verify_plan(plan)
 bad=copy.deepcopy(plan);bad['jobs'][0]['argv']+=['--max-steps','1']
 with pytest.raises(ValueError,match='42-job'):q.verify_plan(bad)
 bad=copy.deepcopy(plan);bad['backend_readiness']['lfm25']['sha256']='0'*64
 with pytest.raises(ValueError,match='readiness'):q.verify_plan(bad)


def test_actual_existing_A0_A1_metadata_adoption_no_tensor_reads():
 for job in q.jobs()[:2]:
  proof=q.validate_completed(job);assert proof['completed']['steps']==job['expected_steps'];assert proof['roles']['best']==proof['roles']['last']
 bad=copy.deepcopy(q.jobs()[0]);bad['expected_steps']=63
 with pytest.raises(ValueError,match='step/record'):q.validate_completed(bad)


def test_completed_external_adopt_no_train(tmp_path,monkeypatch):
 job=q.jobs()[0];(tmp_path/'completed.json').write_text('{}');x=fake_queue();calls=[]
 monkeypatch.setattr(q,'directory',lambda j:tmp_path);monkeypatch.setattr(q,'live_external',lambda j:None);monkeypatch.setattr(q,'supervisor_info',lambda n:{'statename':'EXITED','exitstatus':0,'start':5})
 monkeypatch.setattr(q,'validate_completed',lambda j:{'proof':'verified'});x.execute=lambda *a:pytest.fail('completed trainer repeated')
 row={};x.adopt_or_train(job,row);assert row['training_completed'] and row['training_adopted_without_new_updates']


def test_external_Supervisor_nonzero_exit_refuses_even_marker_exists(tmp_path,monkeypatch):
 job=q.jobs()[0];(tmp_path/'completed.json').write_text('{}');x=fake_queue()
 monkeypatch.setattr(q,'directory',lambda j:tmp_path);monkeypatch.setattr(q,'live_external',lambda j:None);monkeypatch.setattr(q,'supervisor_info',lambda n:{'statename':'EXITED','exitstatus':1,'start':5})
 with pytest.raises(ValueError,match='naturally exit0'):x.adopt_or_train(job,{})


def test_live_manual_wait_birth_and_natural_exit_no_signals(tmp_path,monkeypatch):
 job=q.jobs()[2];(tmp_path/'completed.json').write_text('{}');x=fake_queue();live={'pid':123,'birth':45,'ppid':1,'state':'S','argv':['python','trainer']}
 monkeypatch.setattr(q,'directory',lambda j:tmp_path);monkeypatch.setattr(q,'live_external',lambda j:live);monkeypatch.setattr(q,'proc',lambda p:None);monkeypatch.setattr(q,'resources',lambda **kw:None)
 infos=iter([{'statename':'RUNNING','exitstatus':0,'start':5,'pid':123},{'statename':'EXITED','exitstatus':0,'start':5,'pid':0}]);monkeypatch.setattr(q,'supervisor_info',lambda n:next(infos));monkeypatch.setattr(q,'validate_completed',lambda j:{})
 row={};x.execute=lambda *a:pytest.fail('live manual trainer duplicated');x.adopt_or_train(job,row)
 assert row['external_trainer']==live and row['external_trainer_naturally_exited'] and row['training_completed']


def test_manual_supervisor_PID_mismatch_closes_gate(tmp_path,monkeypatch):
 job=q.jobs()[2];x=fake_queue();monkeypatch.setattr(q,'directory',lambda j:tmp_path);monkeypatch.setattr(q,'live_external',lambda j:{'pid':123});monkeypatch.setattr(q,'supervisor_info',lambda n:{'statename':'RUNNING','pid':456})
 with pytest.raises(RuntimeError,match='Supervisor PID'):x.adopt_or_train(job,{})


def test_unregistered_complete_output_never_silently_adopted(tmp_path,monkeypatch):
 job=q.jobs()[3];(tmp_path/'completed.json').write_text('{}');x=fake_queue();monkeypatch.setattr(q,'directory',lambda j:tmp_path);monkeypatch.setattr(q,'live_external',lambda j:None)
 with pytest.raises(ValueError,match='Unregistered'):x.adopt_or_train(job,{})


def test_durable_training_missing_artifact_never_retrained(tmp_path,monkeypatch):
 job=q.jobs()[3];x=fake_queue();monkeypatch.setattr(q,'directory',lambda j:tmp_path);monkeypatch.setattr(q,'live_external',lambda j:None)
 with pytest.raises(RuntimeError,match='no retraining'):x.adopt_or_train(job,{'training_completed':True})


def test_completed_SFT_revalidates_without_training(monkeypatch):
 job=q.jobs()[2];x=fake_queue({'jobs':[job]});x.state['jobs'][job['id']]={'status':'completed'};calls=[]
 monkeypatch.setattr(q,'validate_completed',lambda j:calls.append('artifacts'));x.postprocess=lambda j,r:calls.append('postprocess');x.adopt_or_train=lambda *a:pytest.fail('completed retrained')
 x.run();assert calls==['artifacts','postprocess'] and x.state['status']=='completed'


def test_pruned_complete_intermediate_receipt_ownership_required(tmp_path,monkeypatch):
 job=q.jobs()[0];monkeypatch.setattr(q,'directory',lambda j:tmp_path);(tmp_path/'resolved_config.json').write_text('{}');receipt={'status':'pruned','owner':'own','source_run':str(tmp_path)};q.atomic(tmp_path/'checkpoint_pruned.json',receipt)
 x=fake_queue({'jobs':[job]});x.state['jobs'][job['id']]={'status':'completed','checkpoint_pruned':True,'training_proof':{'manager_owner':'own','config_sha256':q.sha(tmp_path/'resolved_config.json')}};x.run();assert x.state['status']=='completed'
 receipt['owner']='foreign';q.atomic(tmp_path/'checkpoint_pruned.json',receipt)
 with pytest.raises(ValueError,match='pruning'):x.run()
 assert x.state['status']!='completed'


def test_failed_job_requires_explicit_retry(monkeypatch):
 job=q.jobs()[0];x=fake_queue({'jobs':[job]});x.retry=False;x.state['jobs'][job['id']]={'status':'failed'}
 with pytest.raises(RuntimeError,match='retry-failed'):x.run()


def test_four_evaluations_roundtrip_order_and_paired_bootstrap(tmp_path,monkeypatch):
 import scripts.research_grounding_queue as g
 job=q.jobs()[2];x=fake_queue();calls=[];row={'compaction':{'done':True}};identity={'checkpoint_sha256':'c'*64,'output':tmp_path/'none'}
 class Oracle:
  def _evaluation_identity(self,j):return dict(identity)
  def _grounding_identity(self,j,v):return dict(identity,output=tmp_path/v)
  def _validate_evaluation(self,j,i):return {'status':'completed','output':str(i['output'])}
  def _validate_grounding(self,j,i,v):return {'status':'completed','output':str(i['output'])}
  def _checkpoint_validation_identity(self,j):return {}
  def _validate_checkpoint_report(self,i):return {'status':'completed'}
 x.oracle=Oracle();monkeypatch.setattr(q,'directory',lambda j:tmp_path);monkeypatch.setattr(q,'expected_profile',lambda *f:{'name':'synthetic'});monkeypatch.setattr(g,'read_rows',lambda p:[]);monkeypatch.setattr(g,'paired_image_bootstrap',lambda *a:{'synthetic_fixture':True})
 def execute(job,kind,argv,row):
  calls.append((kind,q.value(argv,'--image-ablation') or 'none'))
  if kind=='eval':q.atomic(Path(q.value(argv,'--output'))/'protocol.json',{'numerical_profile':{'name':'synthetic'}})
 x.execute=execute;x.postprocess(job,row)
 assert calls==[('eval','none'),('validate','none'),('eval','shuffle'),('eval','blank'),('eval','text_only')]
 assert q.read(tmp_path/'evaluation/budget18_native_v1/paired_summary.json')['paired_image_group_bootstrap']=={'synthetic_fixture':True}


def test_real_fresh_worker_nonce_parent_birth_PDeath_handshake_without_Torch(tmp_path):
 request=tmp_path/'request.json';parent=q.proc(os.getpid());req={'owner':q.OWNER,'kind':'train','backend':'lfm25','argv':[],'parent':parent,'nonce':'fixed-unit-nonce','controller_source_sha256':q.sha(q.__file__),'data_release_sha256':q.RELEASE_SHA};q.atomic(request,req)
 code="import sys,types,os;sys.argv.append('--worker-request');import scripts.research_budget18_queue_v1 as q;q.ARGS=types.SimpleNamespace(request_sha256=sys.argv[2]);q.resources=lambda **kw:None;q.BACKEND_MODULES['lfm25']='_fixture';sys.modules['_fixture']=types.SimpleNamespace(train_main=lambda a:print('synthetic_cpu_dispatch'));q.worker(sys.argv[1]);assert 'torch' not in sys.modules"
 result=subprocess.run([sys.executable,'-c',code,str(request),q.sha(request)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),capture_output=True,text=True,timeout=10,check=True)
 ready=q.read(tmp_path/'worker_ready.json');assert ready['nonce']==req['nonce'] and ready['worker']['ppid']==os.getpid() and ready['worker']['birth']>0 and 'synthetic_cpu_dispatch' in result.stdout
