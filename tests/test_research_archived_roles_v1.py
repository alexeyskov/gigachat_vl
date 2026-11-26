"""Fresh process and durable role merge invariants; no actual model/GPU loads."""
import copy,json,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scripts import validate_research_archived_roles_v1 as r


def context(tmp_path,shared=False):
 roles={};plan={}
 for role,step in [('best',100),('last',170)]:
  obj=str(tmp_path/'.checkpoint_objects'/('object-shared' if shared else 'object-'+role));marker={'owner':'manager','step':170 if shared else step,'metric':25.}
  roles[role]={'object':obj,'marker':marker};plan[role]={'state':obj+'/state.pt','marker':marker,'stat':{'st_size':123,'st_mtime_ns':456}}
 return {'run':str(tmp_path),'roles':roles,'config':{},'files':{'checkpoint_archival.json':{'sha256':'p'*64}},'archive_admission':{'publisher_plan':{'roles':plan},'transactions':{role:{'load_extra_bytes':1} for role in ('best','last')}}}
def report(entry,ctx):
 from src.utils import research_stock_determinism as stock
 role=entry['roles'][0];item=ctx['archive_admission']['publisher_plan']['roles'][role]
 receipt={'schema_version':1,'status':'passed','checkpoint_owner':'manager','checkpoint_path':item['state'],'checkpoint_labels':entry['roles'],'checkpoint_step':entry['marker']['step'],'checkpoint_size_bytes':123,'checkpoint_mtime_ns':456,'selection_metric_replayed':True,'official_benchmark_score':False,'restored_tensor_max_abs_diff':0.,'projector_frozen':True,'projector_unchanged_during_validation':True,'all_tensors_finite':True,'expected_keys_match':True,'validated_tensor_count':2,'group_macro_score_saved':25.,'group_macro_score_replayed':25.,'group_macro_score_abs_diff':0.,'group_macro_score_tolerance':.007,'selection_protocol':'internal_group_macro_v1','heldout_ce_replayed':False,'selection_manifest_sha256':'s'*64,'validation_records':116}
 return {'schema_version':1,'status':'passed','mode':'full_gpu_replay','run':ctx['run'],'checkpoints':[receipt],'validated_tensor_count':2,'numerical_profile':stock.definition()}
@pytest.fixture
def selection(monkeypatch):
 from scripts import research_grounding_queue as g
 from src.eval.research_selection import PROTOCOL_NAME
 monkeypatch.setattr(g,'checkpoint_selection_binding',lambda config:{'manifest_sha256':'s'*64,'records':116})
 return PROTOCOL_NAME

def test_unique_shared_roles_only_one_fresh_native_replay(tmp_path):
 assert [x['roles'] for x in r.unique_roles(context(tmp_path))]==[['best'],['last']]
 assert [x['roles'] for x in r.unique_roles(context(tmp_path,True))]==[['best','last']]

def test_strict_native_tensor_and_selection_receipt_then_both_unique_merge(tmp_path,selection):
 c=context(tmp_path);entries=r.unique_roles(c);reports=[report(e,c) for e in entries]
 for p in reports:p['checkpoints'][0]['selection_protocol']=selection
 assert r.merge_reports(c,entries,reports)['validated_tensor_count']==4
 with pytest.raises(ValueError,match='Both unique'):r.merge_reports(c,entries,reports[:1])
 for field,value in [('checkpoint_step',999),('restored_tensor_max_abs_diff',1e-8),('all_tensors_finite',False),('group_macro_score_replayed',26.),('selection_manifest_sha256','wrong')]:
  bad=copy.deepcopy(reports[0]);bad['checkpoints'][0][field]=value
  with pytest.raises(ValueError):r.validate_receipt(bad,entries[0],c)

def execute_fixture(tmp_path,monkeypatch,shared_roles=False):
 c=context(tmp_path,shared_roles);events=[];old=tmp_path/'checkpoint_roundtrip.json';old.write_text('{"status":"failed","preserved_passed_BEST":true}\n');f=tmp_path/'source.py';f.write_text('# exact fixture source\n');monkeypatch.setattr(r,'ROOT',tmp_path)
 h=SimpleNamespace(inspect=lambda *a:c,checkpoint_context=lambda *a:(events.append('fullphysical') or c),assert_context=lambda x:events.append('metadata'),implementation=lambda:SimpleNamespace(_shared=lambda p:__import__('contextlib').nullcontext()))
 monkeypatch.setattr(r,'shared',lambda:h);monkeypatch.setattr(r,'metadata',lambda run:c);monkeypatch.setattr(r,'readiness',lambda:{'sha256':'ready'});monkeypatch.setattr(r,'source_binding',lambda:{'source.py':r.sha(f)});monkeypatch.setattr(r,'assert_idle',lambda:events.append('idle'));monkeypatch.setattr(r,'advice',lambda run:(events.append('advice') or {'zero_credit':True}));monkeypatch.setattr(r,'validate_receipt',lambda p,e,c:p['checkpoints'][0]);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
 calls=[]
 def worker(request,directory,deadline):
  events.append('worker');calls.append(request);p=report(request['entry'],c)
  for name in ('request.json','worker_startup.json','child.json'):r.atomic(directory/name,{'fixture':True})
  r.atomic(directory/'report.json',p);return p
 monkeypatch.setattr(r,'run_worker',worker);return c,events,calls,old

def test_parent_fullphysical_then_advice_each_worker_canonical_only_after_both(tmp_path,monkeypatch):
 c,events,calls,old=execute_fixture(tmp_path,monkeypatch);result=r.execute(tmp_path)
 assert len(calls)==2 and [e['entry']['roles'] for e in calls]==[['best'],['last']]
 for i,e in enumerate(events):
  if e=='worker':assert 'fullphysical' in events[:i] and events[i-1]=='metadata' and 'advice' in events[:i]
 assert result['status']=='passed' and json.loads(old.read_bytes())==result
 attempts=list((tmp_path/r.NAMESPACE).glob('attempt-*'));assert len(attempts)==1
 assert (attempts[0]/'previous_checkpoint_roundtrip.json').read_text()=='{"status":"failed","preserved_passed_BEST":true}\n'
 assert json.loads((attempts[0]/'attempt.json').read_bytes())['status']=='passed'

def test_second_worker_failure_preserves_old_report_and_first_raw_evidence(tmp_path,monkeypatch):
 c,events,calls,old=execute_fixture(tmp_path,monkeypatch);before=old.read_bytes();native=r.run_worker
 def fail(request,directory,deadline):
  if request['entry']['roles']==['last']:raise MemoryError('unchanged32GiB floor')
  return native(request,directory,deadline)
 monkeypatch.setattr(r,'run_worker',fail)
 with pytest.raises(MemoryError):r.execute(tmp_path)
 assert old.read_bytes()==before
 attempt=next((tmp_path/r.NAMESPACE).glob('attempt-*'));assert (attempt/'object-00/report.json').exists() and json.loads((attempt/'attempt.json').read_bytes())['status']=='failed'

def test_shared_object_parent_executes_exactly_one_worker(tmp_path,monkeypatch):
 _,_,calls,_=execute_fixture(tmp_path,monkeypatch,True);r.execute(tmp_path);assert len(calls)==1 and calls[0]['entry']['roles']==['best','last']

def test_resource_refusal_never_starts_worker_or_replaces_canonical(tmp_path,monkeypatch):
 _,_,calls,old=execute_fixture(tmp_path,monkeypatch);before=old.read_bytes();monkeypatch.setattr(r,'advice',Mock(side_effect=MemoryError('no cache credit')))
 with pytest.raises(MemoryError):r.execute(tmp_path)
 assert not calls and old.read_bytes()==before

def test_real_CPU_child_registration_and_owned_cleanup(tmp_path):
 from scripts import probe_research_parallel_b1 as proc
 nonce='n';request_sha='h';code="""
import json,os,sys,time
from pathlib import Path
from scripts.probe_research_parallel_b1 import process_info,atomic_json
atomic_json(Path(sys.argv[1])/'worker_startup.json',{'owner':'fresh_process_archived_role_replay_v1','nonce':'n','request_sha256':'h','identity':process_info(os.getpid())})
time.sleep(10)
"""
 command=[sys.executable,'-c',code,str(tmp_path)];child=subprocess.Popen(command,cwd=r.ROOT)
 try:
  identity=r.wait_startup(child,command,tmp_path,nonce,request_sha,time.monotonic()+3);assert proc.same_process(identity,proc.process_info(child.pid))
 finally:r.stop(child)
 assert child.poll() is not None

def test_real_unregistered_child_timeout_cleanup(tmp_path):
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
 try:
  with pytest.raises(TimeoutError):r.wait_startup(child,child.args,tmp_path,'n','h',time.monotonic()+.15)
 finally:r.stop(child)
 assert child.poll() is not None

@pytest.mark.parametrize('refuse_constructor',[False,True])
def test_native_hooks_only_select_bound_object_and_redirect_own_report(tmp_path,monkeypatch,refuse_constructor):
 from src.utils import research_stock_determinism as stock
 c=context(tmp_path);entry=r.unique_roles(c)[0];events=[];native=SimpleNamespace();native.managed_checkpoints=lambda p:[{'path':Path(x['object'])/'state.pt','marker':x['marker'],'labels':x['roles']} for x in r.unique_roles(c)]
 native.atomic_json=lambda p,v:events.append(('write',p,v))
 def validate(run,device):
  rows=native.managed_checkpoints(run);assert len(rows)==1 and rows[0]['labels']==['best'];assert runner.ResearchModel() is model;p={'status':'passed'};native.atomic_json(Path(run)/'checkpoint_roundtrip.json',p);return p
 native.validate_run=validate;h=SimpleNamespace(assert_context=lambda c:None,implementation=lambda:SimpleNamespace(_shared=lambda p:__import__('contextlib').nullcontext(),CONSTRUCTOR_ALLOWANCE=16,METADATA_LIMIT=2))
 model=object();runner=SimpleNamespace(ResearchModel=lambda *a:(events.append(('constructor',)) or model))
 from src.utils import research_lossless_delta as codec
 def guard(n):
  events.append(('guard',n))
  if refuse_constructor:raise MemoryError('same constructor floor')
 monkeypatch.setattr(codec,'memory_guard',guard)
 monkeypatch.setattr(r,'shared',lambda:h);monkeypatch.setattr(stock,'activate',lambda:None);monkeypatch.setattr(stock,'assert_active',lambda:None);monkeypatch.setattr(stock,'runner',lambda:runner);monkeypatch.setattr(stock,'_clone',lambda path,dependencies:(events.append(('clone',path,dependencies)) or native))
 request={'owner':r.OWNER,'nonce':'n','request_sha256':'h','source_sha256':{},'entry':entry,'run':str(tmp_path),'context':c,'worker_report':str(tmp_path/'own.json')}
 if refuse_constructor:
  with pytest.raises(MemoryError):r.native_worker(request)
  assert [x[0] for x in events]==['clone','guard']
 else:
  r.native_worker(request);assert events[1:4]==[('guard',18),('constructor',),('guard',2)] and events[-1][1]==tmp_path/'own.json'
 assert events[0]==('clone','scripts/validate_research_checkpoint.py',{'scripts.run_research_stage':runner})
 assert not (tmp_path/'checkpoint_roundtrip.json').exists()

def test_original_native_restore_and_CE_math_tiny_CPU():
 import torch
 from torch import nn
 from scripts.validate_research_checkpoint import full_model_checks
 class Tiny(nn.Module):
  def __init__(self):super().__init__();self.layer=nn.Linear(2,3);self.projector=nn.Linear(2,2);self.projector.requires_grad_(False)
  def prepare(self,rows):return rows
  def forward_loss(self,rows):
   logits=self.layer(torch.tensor([[1.,2.],[3.,4.]]));loss=nn.functional.cross_entropy(logits,torch.tensor([1,2]),reduction='sum');return loss,2,2
 torch.manual_seed(42);model=Tiny();payload={'model':{k:v.clone() for k,v in model.state_dict().items()},'config':{'stage':'SFT'}};expected=model.forward_loss([0])[0].item()/2
 with torch.no_grad():model.layer.weight.zero_()
 result=full_model_checks(model,payload,[0],1,expected,0.,0.)
 assert result['validation_loss_abs_diff']==0 and result['restored_tensor_max_abs_diff']==0 and result['projector_unchanged_during_validation']

def test_foreign_GPU_and_PID_reuse_during_NVML_exit_never_accepted(monkeypatch):
 from scripts import probe_research_parallel_b1 as proc
 identity={'pid':123,'start_time':5,'ppid':456,'argv':['owned']};monkeypatch.setattr(proc,'gpu_snapshot',lambda:{'processes':[{'pid':999}]})
 with pytest.raises(RuntimeError,match='Foreign'):r.wait_gpu_exit(123,identity,time.monotonic()+1)
 monkeypatch.setattr(proc,'gpu_snapshot',lambda:{'processes':[{'pid':123}]});monkeypatch.setattr(proc,'process_info',lambda p:dict(identity,start_time=6))
 with pytest.raises(RuntimeError,match='reused'):r.wait_gpu_exit(123,identity,time.monotonic()+1)

def test_successful_private_worker_main_returns_zero_not_report_exitcode(monkeypatch):
 monkeypatch.setattr(r,'worker',lambda *a:{'status':'passed'})
 monkeypatch.delitem(sys.modules,'torch',raising=False)
 assert r.main(['--worker-request','owned.json','--request-sha256','hash'])==0
