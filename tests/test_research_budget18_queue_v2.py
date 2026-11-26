"""Isolated future-resource dispatch/pruning; no real GPU/model/data mutation."""
import contextlib,copy,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import scripts.research_budget18_queue_v2 as v


def fake():
 q=object.__new__(v.Queue);q.spec={'jobs':v.old.jobs()[:7]};q.state={'jobs':{}};q.persist=lambda:None;return q


def test_A1_checkpoint_retained_until_shared_NF4_parent_can_prune(monkeypatch):
 import scripts.research_queue as base
 q=fake();jobs=q.spec['jobs'];nf,a1b,s1,bf,s2,a1c,s3=jobs;alive={str(v.old.directory(j)) for j in jobs};calls=[]
 for j in (nf,a1b,s1):q.state['jobs'][j['id']]={'status':'completed'}
 def retire(source,consumers):
  assert all(str(p) in alive for p in consumers);calls.append(source);alive.remove(str(source));return {'status':'pruned'}
 monkeypatch.setattr(base,'prune_checkpoint_run',retire)
 q.prune();assert calls==[] and str(v.old.directory(a1b)) in alive
 q.state['jobs'][a1c['id']]={'status':'completed'};q.prune()
 assert calls==[v.old.directory(nf),v.old.directory(a1b)] and q.state['jobs'][nf['id']]['checkpoint_pruned']
 assert str(v.old.directory(a1c)) in alive
 q.state['jobs'][s3['id']]={'status':'completed'};q.prune();assert calls[-1]==v.old.directory(a1c)


def test_BF16_parent_pruned_before_only_full_consumer_done_never(monkeypatch):
 import scripts.research_queue as base
 q=fake();jobs=q.spec['jobs'];bf,sft=jobs[3:5];q.state['jobs'][bf['id']]={'status':'completed'};calls=[]
 monkeypatch.setattr(base,'prune_checkpoint_run',lambda *a:calls.append(a) or {})
 q.prune();assert not calls
 q.state['jobs'][sft['id']]={'status':'completed'};q.prune();assert calls[0][0]==v.old.directory(bf)


@pytest.mark.parametrize('kind',['train','eval','validate'])
def test_current_three_manual_source_reserve120_and_argv_preserved(kind):
 job=v.old.jobs()[2];argv=['--run','old','--reserve-gb','120'];assert v.actual_argv(job,kind,argv)==argv


@pytest.mark.parametrize('kind',['train','eval','validate'])
def test_future_actual_args32_explicitprofile_and_original_intent_unchanged(kind):
 job=v.old.jobs()[3];before=copy.deepcopy(job);argv=list(job['argv']) if kind=='train' else ['--run','future','--reserve-gb','120'];actual=v.actual_argv(job,kind,argv)
 assert v.old.value(actual,'--reserve-gb')=='32' and v.old.value(actual,'--numerical-profile')=='stock_deterministic_small_v1';assert job==before


def test_private_scopes_do_not_patch_frozen_controller_or_global_Popen():
 import subprocess
 assert v.old.sha(v.ROOT/v.BASE)==v.BASE_SHA and v.old.__file__.endswith(v.BASE)
 assert v.old.subprocess.Popen is subprocess.Popen
 assert v.SCOPE['__file__'].endswith('research_budget18_queue_v2.py')
 assert any(isinstance(c,tuple) and 'resource_ready' in c and 'source_intent_argv' in c for c in v._execute.__code__.co_consts)


def test_all_manual_postprocessing_remains_original_reserve120_future32(monkeypatch):
 q=fake();calls=[]
 assert 120 in v._manual_postprocess.__code__.co_consts and 32 in v._postprocess.__code__.co_consts
 monkeypatch.setattr(v,'_manual_postprocess',lambda self,j,row:calls.append(('old120',j['id'])))
 monkeypatch.setattr(v,'_postprocess',lambda self,j,row:calls.append(('new32',j['id'])))
 q.postprocess(v.old.jobs()[2],{});q.postprocess(v.old.jobs()[4],{})
 assert [kind for kind,_ in calls]==['old120','new32']


def resource_fixture(tmp_path):
 paths={m.replace('.','/')+'.py':v.old.sha(v.ROOT/(m.replace('.','/')+'.py')) for m in v.MODULES.values()}
 cache=v.ROOT/'research_results/budget18_cache_actual_retirement_v1/completed.json';policy=v.ROOT/'configs/research/completed_r2_finalization_v1_ready.json'
 p=tmp_path/'ready.json';v.old.atomic(p,{'status':'passed','source_sha256':paths,'prerequisites':{'cache':{'path':str(cache),'sha256':v.old.sha(cache)},'policy':{'path':str(policy),'sha256':v.old.sha(policy)}}})
 return {'path':str(p),'sha256':v.old.sha(p)}


def test_resource_ready_code_cache17981_and_R2_prerequisites_bound(tmp_path):
 binding=resource_fixture(tmp_path);assert v.verify_resource(binding)['status']=='passed'
 p=Path(binding['path']);data=v.old.read(p);data['prerequisites']['cache']['sha256']='0'*64;v.old.atomic(p,data);binding['sha256']=v.old.sha(p)
 with pytest.raises(ValueError,match='prerequisite'):v.verify_resource(binding)


def test_missing_resource_ready_refuses_before_future_backend():
 with pytest.raises(ValueError,match='READY'):v.verify_resource(None)


def test_future_completed_requires_actualreserve32_and_exact_R2policy(tmp_path,monkeypatch):
 job=v.old.jobs()[4];monkeypatch.setattr(v.old,'directory',lambda j:tmp_path);v.old.atomic(tmp_path/'resolved_config.json',{'reserve_gb':120})
 with pytest.raises(ValueError,match='actual32'):v.validate_completed(job)
 v.old.atomic(tmp_path/'resolved_config.json',{'reserve_gb':32})
 with pytest.raises(ValueError,match='R2 lacks'):v.validate_completed(job)


def test_owned_model_only_recovery_prevents_any_new_optimizer_updates(monkeypatch):
 import src.utils.research_completed_r2_v1 as policy
 q=fake();job=v.old.jobs()[4];calls=[];monkeypatch.setattr(policy,'recover_if_published',lambda p:True);monkeypatch.setattr(v,'_adopt',lambda self,j,row:calls.append(row.copy()))
 row={};q.adopt_or_train(job,row);assert row['training_completed'] and row['model_only_publication_recovered_without_optimizer_updates'] and calls


def test_real_fresh_futureSmall_eval_dispatches_resourcev3_without_Torch(tmp_path):
 parent=v.old.proc(os.getpid());run=tmp_path/'run';run.mkdir();v.old.atomic(run/'resolved_config.json',{'recipe':'1c'});job=next(j for j in v.old.jobs() if j['backend']=='smallthinker4' and j['encoder']=='qwen25' and j['recipe']=='1c' and j['stage']=='SFT')
 req={'owner':v.old.OWNER,'kind':'eval','backend':'smallthinker4','argv':['--run',str(run),'--reserve-gb','32'],'parent':parent,'nonce':'fresh-unit-only','controller_source_sha256':v.old.sha(v.__file__),'data_release_sha256':v.old.RELEASE_SHA,'resource_ready':{'path':'unused-fixture','sha256':'0'*64},'job_id':job['id']};path=tmp_path/'request.json';v.old.atomic(path,req)
 code="import sys,types;sys.argv.append('--worker-request');import scripts.research_budget18_queue_v2 as v;v.verify_resource=lambda x:None;v.SCOPE['resources']=lambda **kw:None;v.OLD_SCOPE['resources']=lambda **kw:None;v.MODULES['smallthinker4']='_synthetic';sys.modules['_synthetic']=types.SimpleNamespace(eval_main=lambda argv:print('SYNTHETIC_RESOURCE32_EVAL',argv));v.worker(sys.argv[1],types.SimpleNamespace(request_sha256=sys.argv[2]));assert 'torch' not in sys.modules"
 result=subprocess.run([sys.executable,'-c',code,str(path),v.old.sha(path)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),capture_output=True,text=True,check=True,timeout=10)
 assert 'SYNTHETIC_RESOURCE32_EVAL' in result.stdout;ready=v.old.read(tmp_path/'worker_ready.json');assert ready['nonce']==req['nonce'] and ready['worker']['ppid']==os.getpid()


def advice_fixture(tmp_path,monkeypatch):
 q=fake();q.child=None;p=tmp_path/'owned-state.pt';p.write_bytes(b'actual tiny immutable source');meta=tmp_path/'marker.json';v.old.atomic(meta,{'owner':'synthetic'})
 @contextlib.contextmanager
 def candidates(queue):yield {p:v.physical(p)},{meta:v.old.sha(meta)}
 monkeypatch.setattr(v,'gpu_idle',lambda:None)
 return q,p,meta,candidates


def test_readonly_advice_descriptor_stat_host_remeasurement_and_no_credit(tmp_path,monkeypatch):
 q,p,meta,candidates=advice_fixture(tmp_path,monkeypatch);calls=[];before=v.physical(p);original=p.read_bytes()
 def advise(fd,offset,length,flag):
  assert offset==0 and length==0 and flag==os.POSIX_FADV_DONTNEED;assert os.read(fd,100)==original;calls.append('advice')
 def scan(ids):assert ids=={(before['st_dev'],before['st_ino'])};calls.append('scan');return {'checked':True}
 def guard(extra):assert extra==0;calls.append('actual_host_guard');return {'available_bytes':33*2**30,'reclaimable_cache_credit_bytes':0}
 receipt=v.advise_inactive(q,candidates=candidates,scan=scan,advisor=advise,guard=guard)
 assert calls==['scan','advice','scan','actual_host_guard'];assert p.read_bytes()==original and v.physical(p)==before
 assert receipt['assumed_cache_credit_bytes']==0 and not receipt['files_or_pointers_or_weights_modified']


def test_open_mapping_scan_refuses_before_any_cache_advice(tmp_path,monkeypatch):
 q,p,meta,candidates=advice_fixture(tmp_path,monkeypatch);calls=[]
 def scan(ids):raise RuntimeError('actual mmap/open found')
 with pytest.raises(RuntimeError,match='mmap'):v.advise_inactive(q,candidates=candidates,scan=scan,advisor=lambda *a:calls.append(a))
 assert not calls


def test_post_advice_memory_floor_is_not_assumed_reclaim_credit(tmp_path,monkeypatch):
 q,p,meta,candidates=advice_fixture(tmp_path,monkeypatch);calls=[]
 def guard(extra):raise MemoryError('actual unchanged32GiB floor')
 with pytest.raises(MemoryError,match='32GiB'):v.advise_inactive(q,candidates=candidates,scan=lambda ids:[],advisor=lambda *a:calls.append(a),guard=guard)
 assert len(calls)==1


def test_changed_metadata_during_advice_refuses_publication(tmp_path,monkeypatch):
 q,p,meta,candidates=advice_fixture(tmp_path,monkeypatch)
 def advice(*a):v.old.atomic(meta,{'owner':'changed'})
 with pytest.raises(ValueError,match='metadata changed'):v.advise_inactive(q,candidates=candidates,scan=lambda ids:[],advisor=advice)


def test_hf_advice_candidates_only_exact_three_revisions_and_sizes():
 q=fake();q.spec={'jobs':[]}
 with v.advice_candidates(q) as (paths,bindings):
  assert len(paths)==9 and sum(s['st_size'] for s in paths.values())==48433518312
  assert all(p.parents[1].name=='blobs' and p.is_file() for p in paths)
  assert len(bindings)==5 and all(v.old.sha(p)==h for p,h in bindings.items())


def test_owned_completed_checkpoint_shared_roles_dedup_and_foreign_owner_refusal(tmp_path,monkeypatch):
 q=fake();job=v.old.jobs()[2];q.spec={'jobs':[job]};q.state['jobs'][job['id']]={'training_completed':True}
 monkeypatch.setattr(v.old,'directory',lambda _:tmp_path);monkeypatch.setattr(v.old,'live_external',lambda _:None)
 v.old.atomic(tmp_path/'resolved_config.json',{'model':v.old.MODELS[job['backend']],'stage':job['stage'],'recipe':job['recipe']})
 v.old.atomic(tmp_path/'completed.json',{'steps':56,'records':3529});v.old.atomic(tmp_path/'.checkpoint_manager.json',{'owner':'tiny-owned'})
 (tmp_path/'.checkpoint.lock').write_bytes(b'')
 obj=tmp_path/'.checkpoint_objects/object-tiny';obj.mkdir(parents=True);state=obj/'state.pt';state.write_bytes(b'synthetic-not-a-Torch-model')
 v.old.atomic(obj/'.research_checkpoint.json',{'owner':'tiny-owned'});(tmp_path/'best').symlink_to(obj.relative_to(tmp_path));(tmp_path/'last').symlink_to(obj.relative_to(tmp_path))
 with v.advice_candidates(q) as (paths,bindings):assert len(paths)==10 and paths[state]==v.physical(state)
 v.old.atomic(obj/'.research_checkpoint.json',{'owner':'foreign'})
 with pytest.raises(ValueError,match='foreign'): 
  with v.advice_candidates(q):pass
