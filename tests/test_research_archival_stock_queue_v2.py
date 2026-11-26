"""Bounded mock-only controller lifecycle; no training/GPU/actual publication."""
import copy,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scripts import research_archival_stock_queue_v2 as q


def make_job(tmp_path,recipe='3c',stage='SFT'):
 return {'id':f'smallthinker4__siglip2__{recipe}__{stage}_fixture__stockdet_v1','argv':['--stage',stage,'--recipe',recipe,'--output',str(tmp_path/'run')],
  'compact_completed':True,'evaluation':{'manifest':'fixed512','output':'preserved'}}

@pytest.fixture
def setup(tmp_path,monkeypatch):
 p=q.contract.publisher_v2();calls=[];mode={'value':'success'};entry=make_job(tmp_path);run=Path(entry['argv'][-1]);run.mkdir()
 target=run/'.checkpoint_objects/object-native';target.mkdir(parents=True);(target/'state.pt').write_bytes(b'PK\x03\x04synthetic-raw')
 plan={'roles':{r:{'object':target.name,'state':str(target/'state.pt'),'marker':{}} for r in ('best','last')},'metadata_sha256':{'completed.json':'d'*64}}
 bound={'path':str(p.ACTIVATION),'sha256':'a'*64};journal={'protocol':p.OWNER,'phase':'no_benefit','activation':bound,
  'source_roles':{'best':target.name,'last':target.name},'source_object':target.name,'source_state_sha256':p.sha(target/'state.pt')}
 spec={'jobs':[entry],'reserve_gb':120};state_path=tmp_path/'stock.state.json'
 original={'numerical_execution':{'profile':q.stock.stock.definition(),'controller_sha256':q.PINS['scripts/research_stock_pilot_queue.py']},
  'jobs':{'completed3a':{'status':'completed','evaluation':{'output':'original3aordinary'}}},'extra':{}}
 state_path.write_text(json.dumps(original))
 class Base:
  def __init__(self,spec_path,state_path,retry_failed):
   self.spec_path=Path(spec_path);self.state_path=Path(state_path);self.spec=copy.deepcopy(spec);self.spec_hash=q.digest(spec)
   self.state=copy.deepcopy(original);self.state['numerical_execution']={'profile':q.stock.stock.definition(),'controller_sha256':q.PINS['scripts/research_stock_pilot_queue.py']}
   self.lock_handle=SimpleNamespace(close=Mock());self.stop_before_job=None
  def persist(self):calls.append(('persist',copy.deepcopy(self.state)))
  def _stock_boundary(self,job):calls.append(('stock_boundary',copy.deepcopy(job)))
  def compact_completed_job(self,job,record):
   calls.append(('raw_compact',copy.deepcopy(job)));record['compaction']={'status':'completed'}
   (run/'checkpoint_compaction.json').write_text(json.dumps(record['compaction']))
  def evaluate_completed_job(self,job,record):calls.append(('evaluate',copy.deepcopy(job)))
 monkeypatch.setattr(q,'advise_completed',lambda *a,**k:{'fixture_readonly_advice':True});monkeypatch.setattr(q.contract,'publisher_v2',lambda:p);monkeypatch.setattr(q.stock,'queue_class',lambda:Base)
 monkeypatch.setattr(q,'STATE',state_path);monkeypatch.setattr(q,'activation',lambda expected,publisher:bound)
 monkeypatch.setattr(q,'readiness',lambda:{'fixture':'ready'});monkeypatch.setattr(q,'spec_snapshot',lambda path:copy.deepcopy(spec))
 monkeypatch.setattr(p,'eligibility',lambda path:plan);published={'protocol':p.OWNER,'status':'completed'}
 def publish(*args,**kwargs):
  calls.append(('publish',kwargs))
  if mode['value']=='error':raise ValueError('publication corruption')
  if mode['value']=='nobenefit':
   (run/p.JOURNAL).write_text(json.dumps(journal));raise p.NoBenefit('measured non-saving')
  return published
 monkeypatch.setattr(p,'publish_completed_checkpoint',publish);monkeypatch.setattr(p,'validate_completed_archival',lambda path:published)
 import scripts.research_queue as native
 monkeypatch.setattr(native,'artifacts_for',lambda *a,**k:{'fixture':True})
 # The extracted donor namespace uses q.artifacts_for independently.
 monkeypatch.setattr(q,'artifacts_for',lambda *a,**k:{'fixture':True})
 cls=q.queue_class();queue=cls(tmp_path/'stock.json',state_path,False,activation_sha='a'*64);queue._guard=lambda:None
 return SimpleNamespace(p=p,queue=queue,job=entry,run=run,plan=plan,journal=journal,calls=calls,mode=mode,spec=spec,original=original,cls=cls,state_path=state_path)


def test_constructor_records_new_controller_history_without_relabelling_completed3a(setup):
 s=setup
 assert s.queue.state['jobs']==s.original['jobs']
 assert s.queue.state['numerical_execution']['controller_sha256']==q.sha(q.ROOT/q.SCRIPT)
 assert s.queue.state['controller_execution_history']==[s.original['numerical_execution']]
 assert s.queue.spec==s.spec and all('archive_completed_small' not in j for j in s.queue.spec['jobs'])
 assert s.queue.state['extra'][q.OWNER]['raw_authoritative_recipes']==['3a']


def test_only_four_future_sft_job_copies_archived(setup):
 s=setup;before=copy.deepcopy(s.job);record={};s.queue.compact_completed_job(s.job,record)
 assert s.job==before and s.queue.spec['jobs'][0]==before
 assert record['training_completed'] and record['checkpoint_archival']['protocol']==s.p.OWNER
 assert [v for k,v in s.calls if k=='raw_compact'][0]['archive_completed_small'] is True
 assert '--archive' not in s.job['argv']

@pytest.mark.parametrize('recipe,stage',[('3a','SFT'),('3c','A1'),('2a','A1')])
def test_raw3a_and_alignment_stages_preserve_stock_behavior(setup,tmp_path,recipe,stage):
 s=setup;job=make_job(tmp_path,recipe,stage);s.queue.spec['jobs']=[job];s.queue.compact_completed_job(job,{})
 assert not any(k=='publish' for k,v in s.calls)
 assert 'archive_completed_small' not in [v for k,v in s.calls if k=='raw_compact'][0]


def test_unknown_modified_job_refused_before_compaction(setup):
 s=setup;job=copy.deepcopy(s.job);job['argv'].append('--unexpected')
 with pytest.raises(ValueError,match='Unregistered'):s.queue.compact_completed_job(job,{})
 assert not any(k in ('publish','raw_compact') for k,v in s.calls)


def test_already_started_ordinary_refuses_new_archival(setup):
 with pytest.raises(ValueError,match='already-started'):setup.queue.compact_completed_job(setup.job,{'evaluation':{'status':'running'}})
 assert not any(k in ('publish','raw_compact') for k,v in setup.calls)


def test_partial_journal_retry_bypasses_old_mmap_compactor(setup):
 s=setup;(s.run/s.p.JOURNAL).write_text(json.dumps(s.journal));(s.run/'checkpoint_compaction.json').write_text('{"status":"completed"}')
 s.queue.compact_completed_job(s.job,{'training_completed':True})
 assert any(k=='publish' for k,v in s.calls) and not any(k=='raw_compact' for k,v in s.calls)


def test_publication_error_preserves_training_completed_and_stops(setup):
 s=setup;s.mode['value']='error';record={}
 with pytest.raises(ValueError,match='corruption'):s.queue.compact_completed_job(s.job,record)
 assert record['training_completed'] and record['status']=='training_completed_compaction_failed'
 assert not any(k=='evaluate' for k,v in s.calls)


def test_measured_no_benefit_strict_raw_receipt_is_reused_without_reencoding(setup):
 s=setup;s.mode['value']='nobenefit';record={};s.queue.compact_completed_job(s.job,record)
 assert record['checkpoint_archival_skip']['status']=='skipped_no_benefit_native_raw' and record['status']=='training_completed'
 count=sum(k=='publish' for k,v in s.calls);s.queue.compact_completed_job(s.job,record)
 assert sum(k=='publish' for k,v in s.calls)==count
 s.queue.evaluate_completed_job(s.job,record);assert any(k=='evaluate' for k,v in s.calls)

@pytest.mark.parametrize('mutation',['owner','bytes','mixed'])
def test_no_benefit_corrupt_or_mixed_roles_stop_without_evaluation(setup,mutation):
 s=setup;s.mode['value']='nobenefit';record={};s.queue.compact_completed_job(s.job,record)
 if mutation=='owner':
  s.journal['protocol']='foreign';(s.run/s.p.JOURNAL).write_text(json.dumps(s.journal))
 elif mutation=='bytes':Path(s.plan['roles']['best']['state']).write_bytes(b'PK\x03\x04changed')
 else:s.plan['roles']['last']['marker']['checkpoint_archival']={'foreign':True}
 with pytest.raises(ValueError):s.queue.evaluate_completed_job(s.job,record)
 assert not any(k=='evaluate' for k,v in s.calls)


def test_alternate_state_is_refused_before_new_lock_or_training(setup,tmp_path):
 with pytest.raises(ValueError,match='Same existing'):setup.cls(tmp_path/'stock.json',tmp_path/'other.state.json',False,activation_sha='a'*64)
 assert not (tmp_path/'other.state.json').exists()


def test_actual_default_cli_is_closed_without_gpu_or_new_state(tmp_path):
 result=subprocess.run([sys.executable,str(q.ROOT/q.SCRIPT)],capture_output=True,text=True,cwd=q.ROOT)
 assert result.returncode==2 and json.loads(result.stdout)['jobs_started'] is False
 assert not list(tmp_path.iterdir())


def test_each_stock_loop_boundary_rechecks_guard_and_retains_base_boundary(setup):
 s=setup;guard=Mock();s.queue._guard=guard;s.queue._stock_boundary(s.job)
 guard.assert_called_once();assert any(k=='stock_boundary' for k,v in s.calls)
 changed=copy.deepcopy(s.job);changed['argv'].append('--changed')
 with pytest.raises(ValueError):s.queue._stock_boundary(changed)
 assert sum(k=='stock_boundary' for k,v in s.calls)==1


def test_changed_existing_execution_profile_refuses_implicit_migration(setup,tmp_path):
 s=setup;s.original['extra'][q.OWNER]={'foreign_profile':True}
 with pytest.raises(ValueError,match='profile changed'):s.cls(tmp_path/'stock.json',s.state_path,False,activation_sha='a'*64)

@pytest.fixture
def cache_advice(tmp_path):
 jobs=[make_job(tmp_path/'prior','3a'),make_job(tmp_path/'current','3c')];plans={}
 for job in jobs:
  run=Path(job['argv'][-1]);run.mkdir(parents=True);(run/'completed.json').write_text('{"steps":170}')
  file=run/'.checkpoint_objects/object-complete/state.pt';file.parent.mkdir(parents=True);file.write_bytes(b'owned complete bytes')
  plans[job['id']]={'roles':{r:{'path':str(file),'stat':q.contract.stat(file),'marker':{'owner':'own'}} for r in ('best','last')},'metadata_sha256':{'owned':'a'}}
 queue=SimpleNamespace(spec={'jobs':jobs},state={'jobs':{j['id']:{'training_completed':True,'returncode':0,'status':'completed','pid':999} for j in jobs}})
 host=lambda:{'MemAvailable_bytes':100*2**30,'memory.max':200*2**30,'memory.current':100*2**30}
 return SimpleNamespace(jobs=jobs,plans=plans,queue=queue,host=host,lookup=lambda j:copy.deepcopy(plans[j['id']]),current=Path(jobs[1]['argv'][-1]))


def test_advice_covers_prior_raw3a_and_shared_current_roles_once_without_credit(cache_advice):
 s=cache_advice;events=[];before={r['path']:Path(r['path']).read_bytes() for p in s.plans.values() for r in p['roles'].values()}
 result=q.advise_completed(s.queue,s.current,lookup=s.lookup,advisor=lambda p,t:events.append(p) or {'readonly':True},gpu=lambda:{'processes':[]},host=s.host,process=lambda pid:None)
 assert result['unique_owned_files']==2 and len(events)==2 and result['reclaimable_cache_credit_assumed_bytes']==0
 assert result['actual_available_after_bytes']==100*2**30
 assert before=={p:Path(p).read_bytes() for p in before}

@pytest.mark.parametrize('reason',['trainer_alive','GPU_busy','mapped','foreign_source','post_mutation'])
def test_cache_advice_refuses_unsafe_lifecycle_and_own_maps(cache_advice,reason):
 import mmap
 s=cache_advice;events=[];mapping=None;handle=None
 gpu=lambda:{'processes':[{'pid':42}] if reason=='GPU_busy' else []}
 process=lambda pid:({'pid':pid} if reason=='trainer_alive' else None)
 source=Path(s.plans[s.jobs[1]['id']]['roles']['best']['path'])
 if reason=='mapped':handle=source.open('rb');mapping=mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ)
 def advisor(p,t):
  events.append(p)
  if reason=='post_mutation':p.write_bytes(p.read_bytes()+b'corrupt')
  return {'fixture':True}
 def lookup(job):
  result=s.lookup(job)
  if reason=='post_mutation':
   for r in result['roles'].values():r['stat']=q.contract.stat(r['path'])
  return result
 try:
  with pytest.raises(ValueError):q.advise_completed(s.queue,s.current,selected_source=(source if reason!='foreign_source' else source.parent/'foreign'),lookup=lookup,advisor=advisor,gpu=gpu,host=s.host,process=process)
  if reason!='post_mutation':assert not events
 finally:
  if mapping is not None:mapping.close();handle.close()


def test_advice_checks_actual_headroom_and_never_assumes_eviction_succeeded(cache_advice):
 s=cache_advice;low=lambda:{'MemAvailable_bytes':31*2**30,'memory.max':200*2**30,'memory.current':100*2**30}
 with pytest.raises(MemoryError,match='below32GiB'):q.advise_completed(s.queue,s.current,lookup=s.lookup,advisor=lambda *a:{'readonly':True},gpu=lambda:{'processes':[]},host=low,process=lambda pid:None)


def test_private_derive_hook_advises_after_journal_hash_before_original_memory_guard(setup,monkeypatch):
 # The wrapped private API is used, canonical publication sources stay untouched.
 s=setup;events=[]
 monkeypatch.setattr(q,'advise_completed',lambda queue,run,**kwargs:events.append(('advice',kwargs['selected_source'])) or {'readonly':True})
 # Wrapper closes over original private derive, so only use an injected early
 # exception in the immutable memory guard to observe ordering without encoding.
 from src.utils import research_lossless_delta as codec
 monkeypatch.setattr(codec,'memory_guard',lambda *a:events.append(('memory_guard',None)) or (_ for _ in ()).throw(MemoryError('fixture stop')))
 raw=Path(s.plan['roles']['best']['state'])
 import torch
 torch.save({'tiny':torch.tensor([1.])},raw)
 with pytest.raises(MemoryError,match='fixture stop'):s.p._derive_unlocked(raw,s.run/'stage',{'run':str(s.run)},s.run,None,1)
 assert [x[0] for x in events]==['advice','memory_guard']


@pytest.mark.parametrize('mutation',[None,'foreign_output','foreign_owner','bad_completion','state_symlink'])
def test_completed_owned_candidate_validation_is_strict(tmp_path,mutation):
 job=make_job(tmp_path);run=Path(job['argv'][-1]);target=run/'.checkpoint_objects/object-owned';target.mkdir(parents=True)
 (target/'state.pt').write_bytes(b'owned');(target/'.research_checkpoint.json').write_text(json.dumps({'owner':'own','step':170}))
 for role in ('best','last'):(run/role).symlink_to(Path('.checkpoint_objects')/target.name)
 config={'model':q.stock.stock.SMALL,'stage':'SFT','recipe':'3c','output':str(run)}
 done={'steps':170};manager={'owner':'own','mode':'max'}
 if mutation=='foreign_output':config['output']=str(tmp_path/'other')
 elif mutation=='foreign_owner':manager['owner']='foreign'
 elif mutation=='bad_completion':done['status']='failed'
 elif mutation=='state_symlink':
  (target/'state.pt').unlink();outside=tmp_path/'user';outside.write_bytes(b'unknown');(target/'state.pt').symlink_to(outside)
 for name,value in [('resolved_config.json',config),('completed.json',done),('.checkpoint_manager.json',manager)]:
  (run/name).write_text(json.dumps(value))
 if mutation:
  with pytest.raises(ValueError):q._owned_retained(job)
 else:assert set(q._owned_retained(job)['roles'])=={'best','last'}
