"""Proof-gated CPU routing, immutable provenance and readonly admission scope."""
import copy,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scripts import research_archival_stock_queue_v5 as q
PROOF_SHA='eed4f7b577d1335adad956a464849edf2202e1a89e34c661f7cd8e03ca05b807'
def job(path,recipe='3c'):return {'id':'job-'+recipe,'argv':['--output',str(path),'--stage','SFT','--recipe',recipe],'evaluation':{'output':str(path/'evaluation/heldout_v3')}}

def test_actual_ROOT_proof_raw64_census_dataset_and_source_bindings_without_weights(monkeypatch):
 original=Path.open
 def opened(p,*a,**k):
  assert p.name!='state.pt' and p.suffix!='.safetensors','Scientific tensor body opened'
  return original(p,*a,**k)
 monkeypatch.setattr(Path,'open',opened);result=q.proof_evidence(PROOF_SHA)
 assert result['records']==64 and result['tensor_census_entries']==8015 and result['single_decode_release_sha256']==q.SINGLE_READY_SHA

@pytest.mark.parametrize('field,value',[('status','running'),('runtime_census_exact',False),('records',63),('script_sha256','0'*64)])
def test_running_failed_partial_or_foreign_proofs_cannot_activate(tmp_path,monkeypatch,field,value):
 proof=q.read(q.PROOF);proof[field]=value;path=tmp_path/'report.json';path.write_text(json.dumps(proof));monkeypatch.setattr(q,'PROOF',path)
 with pytest.raises(ValueError):q.proof_evidence(q.sha(path))

def test_no_benefit_raw_and_old3a_not_routed_and_failed_zero_rows_preserved(tmp_path):
 archived=job(tmp_path/'arch');raw=job(tmp_path/'old','3a');state={'jobs':{archived['id']:{'checkpoint_archival':{'status':'completed'},'evaluation':{'status':'failed','output':str(tmp_path/'old-attempt'),'returncode':1}},raw['id']:{'status':'completed','evaluation':{'status':'completed','output':'original'}}}}
 before=copy.deepcopy(state['jobs'][raw['id']]);q.preserve_attempts(state,[archived,raw])
 assert q.routed(archived,state['jobs'][archived['id']]) and state['jobs'][raw['id']]==before
 assert state['jobs'][archived['id']]['evaluation_execution_history'][0]['output']==str(tmp_path/'old-attempt')
 assert not q.routed(archived,{'checkpoint_archival_skip':{'status':'skipped_no_benefit_native_raw'}})
 assert not q.routed(raw,{'checkpoint_archival':{'status':'completed'}})

@pytest.mark.parametrize('completed',[False,True])
def test_partial_or_completed_old_archive_rows_never_retagged(tmp_path,completed):
 j=job(tmp_path/'run');old=tmp_path/'old';old.mkdir();(old/'predictions.jsonl').write_text('preserve every byte\n')
 state={'jobs':{j['id']:{'checkpoint_archival':{'status':'completed'},'evaluation':{'status':'completed' if completed else 'failed','output':str(old)}}}}
 with pytest.raises(ValueError):q.preserve_attempts(state,[j])
 assert (old/'predictions.jsonl').read_text()=='preserve every byte\n'

def test_anchor_advice_exact_scope_no_body_reads_no_credit_and_maps_refused(tmp_path,monkeypatch):
 sizes={k:i+7 for i,k in enumerate(q.ANCHOR_BYTES)};monkeypatch.setattr(q,'ANCHOR_BYTES',sizes)
 snap=tmp_path/'a24932bb880c85c04c794283202a9194b1b42709';snap.mkdir()
 for name,count in sizes.items():(snap/name).write_bytes(b'x'*count)
 run=tmp_path/'run';j=job(run);queue=SimpleNamespace(spec={'jobs':[j]},state={'jobs':{j['id']:{'training_completed':True,'returncode':0,'pid':123}}},_routed=lambda j:True)
 context={'config':{'model':'Tiiny/SmallThinker-4BA0.6B-Instruct'},'archive_admission':{'transactions':{'best':{'reader_header':{'anchor_files':[{'snapshot':str(snap),'shard':n,'bytes':b} for n,b in sizes.items()]}}}}}
 calls=[];scan=Mock(return_value=[{'opened':[],'errors':[]}]);host=lambda:{'host_available_bytes':100*2**30,'cgroup_headroom_bytes':100*2**30}
 from src.utils import research_archive_gpu_parity as parity
 monkeypatch.setattr(parity,'conservative_host_headroom',lambda h:100*2**30)
 args=dict(inspect=lambda *a:context,assert_context=Mock(),scan=scan,advisor=lambda p,s:calls.append(p),gpu=lambda:{'processes':[]},process=lambda p:None,host=host)
 result=q.advise_anchors(queue,run,**args)
 assert set(calls)=={snap/n for n in sizes} and scan.call_count==2 and result['reclaimable_cache_credit_assumed_bytes']==0
 assert not result['bytes_pointers_or_weights_changed']
 args['scan']=Mock(side_effect=RuntimeError('foreign mapping'));calls.clear()
 with pytest.raises(RuntimeError):q.advise_anchors(queue,run,**args)
 assert not calls

def test_strict_exact_source_profile_artifact_and_metadata_corruption(tmp_path):
 run=tmp_path/'run';target=run/'.checkpoint_objects/object-owned';target.mkdir(parents=True);out=tmp_path/'output';out.mkdir()
 (run/'checkpoint_archival.json').write_text('{"immutable":true}');(run/'resolved_config.json').write_text('{"recipe":"3c"}');(target/'.research_checkpoint.json').write_text('{"step":100}')
 record={'record_id':'one','question':'question','image_hashes':['a'*64],'answers':['yes'],'scorer':'exact_match','task_group':'general_vqa','source':{'repo':'fixture'}}
 protocol={'single_decode_execution':q.single().execution_profile(),'numerical_profile':q.single().stock.definition(),
  'source_sha256':{**q.single().source_binding(),**{n:q.sha(q.ROOT/n) for n in q.NATIVE_SOURCES}},
  'single_decode_environment':{'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','TOKENIZERS_PARALLELISM':'false'},
  'selected_record_ids_sha256':__import__('hashlib').sha256(json.dumps(['one']).encode()).hexdigest()}
 loaded={'numerical_profile':q.single().stock.definition(),'stage':'SFT','recipe':'3c','training_step':100,'archived_evaluation':q.single().shared.execution_profile(),'archived_physical_sha256':'f'*64,'archived_publication_sha256':q.sha(run/'checkpoint_archival.json')}
 (out/'protocol.json').write_text(json.dumps(protocol));(out/'model_provenance.json').write_text(json.dumps(loaded));rows=[dict(record,generated_token_ids=[1,2],generated_tokens=2)];(out/'predictions.jsonl').write_text(json.dumps(rows[0])+'\n')
 identity={'output':out,'checkpoint':target/'state.pt','checkpoint_sha256':'f'*64,'records':[record]};q._strict_protocol(identity)
 changed=copy.deepcopy(protocol);changed['source_sha256'].pop(next(iter(changed['source_sha256'])));(out/'protocol.json').write_text(json.dumps(changed))
 with pytest.raises(ValueError,match='source/profile'):q._strict_protocol(identity)
 (out/'protocol.json').write_text(json.dumps(protocol));rows[0]['answers']=['corrupted'];(out/'predictions.jsonl').write_text(json.dumps(rows[0])+'\n')
 with pytest.raises(ValueError,match='metadata/reference'):q._strict_protocol(identity)

def test_private_full_stack_routes_only_new_eval_preserves_v3_tuple_and_super_closure():
 code=f"""
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts import research_archival_stock_queue_v5 as q
m=q.load_controller('{PROOF_SHA}');cls=m.queue_class()
import torch
assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
assert 'run' not in cls.__dict__ and 'compact_completed_job' not in cls.__dict__
g=next(c for c in cls.__mro__ if c.__name__=='GroundingResearchQueue')
assert g.evaluate_completed_job.__closure__[0].cell_contents is g
command=[sys.executable,'-u',str(q.ROOT/'scripts/eval_research_checkpoint.py'),'--run','/tmp/owned','--output','/tmp/owned/evaluation/'+q.single().NAMESPACE+'/none/best']
profile=cls.run.__globals__['_stock_command'](command)
assert profile[2]==str(q.NEW_SCRIPT)
proxy=cls.run.__globals__['subprocess'];assert isinstance(proxy,q._Spawn)
assert len(proxy.original.Popen.__func__.__globals__['GPU_SCRIPTS'])==3
print(json.dumps({{'CPU_only':True,'same_native_run':True,'fourth_GPU_route_explicit':True}}))
"""
 r=subprocess.run([sys.executable,'-c',code],cwd=q.ROOT,capture_output=True,text=True);assert r.returncode==0,r.stderr
 assert json.loads(r.stdout)['CPU_only']

def spawn_fixture(tmp_path,monkeypatch):
 from scripts import research_archival_stock_queue_v3 as old
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');events=[];j=job(tmp_path/'run')
 state={'jobs':{j['id']:{'checkpoint_archival':{'status':'completed'}}}}
 queue=SimpleNamespace(spec={'jobs':[j]},state=state,_guard=lambda:events.append('guard'),persist=lambda:events.append('persist'),_routed=lambda job:True)
 def popen(*a,**k):events.append('spawn');return SimpleNamespace(pid=42,kwargs=k)
 original=old._Subprocess(SimpleNamespace(Popen=popen))
 def retained(*a):events.append('retained');return {'actual_available_after_bytes':100*2**30}
 monkeypatch.setattr(q,'_live_output',lambda p:events.append('live_check'))
 monkeypatch.setattr(q,'advise_anchors',lambda *a:(events.append('anchors') or {'actual_available_after_bytes':110*2**30}))
 proxy=q._Spawn(original,{'queue':queue},retained);command=[sys.executable,'-u',str(q.NEW_SCRIPT),'--run',str(tmp_path/'run'),
  '--output',str(q.single().output_path(tmp_path/'run','none','best')),'--numerical-profile','stock_deterministic_small_v1','--allow-gpu']
 return proxy,queue,events,command,old

def test_fourth_child_only_after_retained_anchor_admission_and_original_tuple_unchanged(tmp_path,monkeypatch):
 proxy,queue,events,command,old=spawn_fixture(tmp_path,monkeypatch);before=old.GPU_SCRIPTS;child=proxy.Popen(command)
 assert child.kwargs['env']['CUDA_VISIBLE_DEVICES']=='0' and os.environ['CUDA_VISIBLE_DEVICES']==''
 assert events.index('retained')<events.index('anchors')<events.index('spawn') and old.GPU_SCRIPTS==before and len(before)==3
 assert queue.state['pre_evaluation_cache_advice'][0]['pinned_anchor_advice']['actual_available_after_bytes']==110*2**30

def test_failed_resource_or_foreign_output_never_spawns_child(tmp_path,monkeypatch):
 proxy,queue,events,command,_=spawn_fixture(tmp_path,monkeypatch)
 monkeypatch.setattr(q,'advise_anchors',Mock(side_effect=MemoryError('32GiB unchanged')))
 with pytest.raises(MemoryError):proxy.Popen(command)
 assert 'spawn' not in events
 command[command.index('--output')+1]=str(tmp_path/'foreign')
 with pytest.raises(ValueError,match='output differs'):proxy.Popen(command)
 assert 'spawn' not in events

def test_old_training_proxy_policy_still_delegated_without_eval_advice(tmp_path,monkeypatch):
 proxy,_,events,_,old=spawn_fixture(tmp_path,monkeypatch)
 command=[sys.executable,'-u',str(q.ROOT/'scripts/run_research_stock_deterministic.py'),'--numerical-profile','stock_deterministic_small_v1','--allow-gpu']
 child=proxy.Popen(command);assert child.kwargs['env']['CUDA_VISIBLE_DEVICES']=='0'
 assert 'retained' not in events and 'anchors' not in events
