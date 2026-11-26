import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scripts import probe_research_selection_replay as d


def fixture_rows():
 records=[];saved=[]
 for index,group in enumerate(('general_vqa','ocr_documents','charts_diagrams','knowledge_reasoning')):
  r={'record_id':str(index),'task_group':group,'scorer':'exact_match','answers':['yes'],'source':{'repo':'fixture','split':'train'},'question':'Q'+str(index),'image_paths':['image'+str(index)],'image_hashes':['a'*64]}
  g={'prediction':'yes','generated_token_ids':[1,2],'generated_tokens':2,'prompt_sha256':str(index)*64,'prefill_tokens':10,'visual_tokens':3,'logits_to_keep':1}
  row={k:copy.deepcopy(r[k]) for k in d.FIELDS if k in r};row.update(prediction='yes',score=1.,generation=g,validation_step=170)
  records.append(r);saved.append(row)
 return {'records':records},saved

def test_exact_comparison_no_mutation():
 selection,old=fixture_rows();snap=copy.deepcopy((selection,old));v=d.compare_rows(selection,old,copy.deepcopy(old))
 assert not v['mismatches'] and v['saved_metrics_recomputed']['group_macro_score']==100
 assert (selection,old)==snap

def test_token_difference_without_score_difference_still_reported():
 s,old=fixture_rows();new=copy.deepcopy(old);new[0]['generation']['generated_token_ids'][1]=3
 v=d.compare_rows(s,old,new)
 assert v['token_mismatch_records']==1 and v['score_mismatch_records']==0 and v['mismatches'][0]['question']=='Q0'

@pytest.mark.parametrize('key,value',[('record_id','wrong'),('answers',['other']),('source',{'repo':'corrupt'}),('task_group','charts_diagrams')])
def test_identity_reference_corruption_refused(key,value):
 s,old=fixture_rows();new=copy.deepcopy(old);new[0][key]=value
 with pytest.raises(ValueError,match='identity'):d.compare_rows(s,old,new)

@pytest.mark.parametrize('mutation',['count','score','prediction','id_type','prompt_missing','truncated'])
def test_corrupt_raw_refused(mutation):
 s,old=fixture_rows();new=copy.deepcopy(old)
 if mutation=='count':new[0]['generation']['generated_tokens']=99
 elif mutation=='score':new[0]['score']=0.
 elif mutation=='prediction':new[0]['generation']['prediction']='other'
 elif mutation=='id_type':new[0]['generation']['generated_token_ids'][0]=True
 elif mutation=='prompt_missing':new[0]['generation'].pop('prompt_sha256')
 else:new.pop()
 with pytest.raises(ValueError):d.compare_rows(s,old,new)

class Vector:
 def __init__(self,v):self.v=v
 def detach(self):return self
 def float(self):return self
 def cpu(self):return self
 def tolist(self):return self.v


def test_first_divergent_top_logits_no_historical_claim():
 output=SimpleNamespace(sequences=[Vector([1,3])],scores=[[Vector([0.,4.,3.,2.])],[Vector([0.,1.,2.,5.])]])
 value=d.first_divergent_logit([1,2],output)
 assert value['first_divergent_generated_index']==1 and value['saved_chosen_current_logit']==2.
 assert value['current_top10'][0]=={'token_id':3,'logit':5.} and value['current_minus_saved_chosen_margin']==3.
 assert not value['historical_logits_available']

def test_no_logit_change_evidence_for_equal_ids():
 assert d.first_divergent_logit([1],SimpleNamespace(sequences=[Vector([1])],scores=[[Vector([0.,1.])]])) is None

def test_current_early_end_has_no_fabricated_score_row():
 value=d.first_divergent_logit([1,2],SimpleNamespace(sequences=[Vector([1])],scores=[[Vector([0.,1.])]]))
 assert not value['current_score_row_available'] and value['saved_token_id']==2

@pytest.mark.parametrize('failure',[False,True])
def test_instance_generate_observer_returns_original_output_and_restores(failure):
 runtime=d.NativeRuntime();runtime.saved={'0':{'generation':{'generated_token_ids':[1]}}};runtime.logit_evidence=[];runtime.selection_census={}
 output=SimpleNamespace(sequences=[Vector([1])],scores=[[Vector([0.,1.])]])
 class Base:
  def generate(self,*a,**k):return output
 model=SimpleNamespace(base=Base());before=dict(model.base.__dict__)
 runtime.selection_census['test']={}
 def predict(model,row,**kw):
  assert model.base.generate() is output
  if failure:raise RuntimeError('own fixture')
  return {'prediction':'yes'}
 if failure:
  with pytest.raises(RuntimeError):runtime.observed_generate(model,{'record_id':'0'},predict,phase='test')
 else:assert runtime.observed_generate(model,{'record_id':'0'},predict,phase='test')=={'prediction':'yes'}
 assert model.base.__dict__==before

class Guard:
 def __init__(self):self.phase='fixture';self.samples=[]
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def check(self):pass
 def fast_check(self):pass


def test_execute_failure_preserves_raw_snapshot_and_never_overwrites_source(tmp_path,monkeypatch):
 # Exercises the actual orchestrator with a failing CPU fixture, no Torch load.
 monkeypatch.setattr(d,'assert_plan',lambda *a,**k:None)
 source=tmp_path/'source';source.mkdir();raw=source/'original.json';raw.write_text('{"retained":true}')
 fake_code=tmp_path/'fixture.py';fake_code.write_text('originalsource')
 monkeypatch.setattr(d,'ROOT',tmp_path)
 engine=Mock();engine.load.side_effect=RuntimeError('injected restore failure')
 plan={'run':str(source),'files':{'original':d.file_binding(raw)},'source_sha256':{'fixture.py':d.sha(fake_code)}}
 output=tmp_path/'diagnostic'
 with pytest.raises(RuntimeError,match='injected'):d.execute(plan,output,nonce='a'*64,runtime=engine,guard=Guard())
 assert (output/'original_evidence/original').read_bytes()==raw.read_bytes()
 assert json.loads((output/'report.json').read_text())['status']=='failed_diagnostic'
 assert raw.read_text()=='{"retained":true}' and (output/'source.zip').exists()
 with pytest.raises(ValueError,match='New separate'):d.execute(plan,output,nonce='a'*64,runtime=engine,guard=Guard())

@pytest.mark.parametrize('timeout,repeats',[(901,0),(119,0),(900,3),(True,1),(900,True)])
def test_execution_bounds_fail_before_any_load(tmp_path,timeout,repeats):
 with pytest.raises(ValueError,match='Bounded'):d.execute({'run':str(tmp_path/'source')},tmp_path/'out',nonce='a'*64,timeout=timeout,repeats=repeats,runtime=Mock(),guard=Guard())

def test_watchdog_released_never_signals(tmp_path,monkeypatch):
 from scripts import probe_research_parallel_b1 as proc
 p=tmp_path/'lease.json';identity={'pid':999,'start_time':1,'argv':['x','--nonce','a'*64]}
 d.write(p,{'owner':d.OWNER,'nonce':'a'*64,'deadline':100.,'process':identity,'status':'released'})
 monkeypatch.setattr(d.time,'monotonic',lambda:50.)
 kill=Mock();monkeypatch.setattr(d.os,'kill',kill)
 assert d.deadline_watchdog(p)==0;kill.assert_not_called()


def test_watchdog_reused_pid_never_signals(tmp_path,monkeypatch):
 from scripts import probe_research_parallel_b1 as proc
 p=tmp_path/'lease.json';identity={'pid':999,'start_time':1,'argv':['x','--nonce','a'*64]}
 d.write(p,{'owner':d.OWNER,'nonce':'a'*64,'deadline':100.,'process':identity,'status':'active'})
 monkeypatch.setattr(d.time,'monotonic',lambda:99.)
 monkeypatch.setattr(proc,'process_info',lambda p:dict(identity,start_time=2))
 kill=Mock();monkeypatch.setattr(d.os,'kill',kill)
 assert d.deadline_watchdog(p)==0;kill.assert_not_called()


def test_metadata_prepare_never_opens_checkpoint_body(monkeypatch):
 # Actual source metadata only; reject all state.pt opens during preflight.
 original=Path.open
 def guarded(path,*a,**k):
  if str(path).endswith('state.pt'):pytest.fail('Metadata must never open checkpoint tensor body')
  return original(path,*a,**k)
 monkeypatch.setattr(Path,'open',guarded)
 plan=d.prepare_plan()
 assert plan['records']==116 and plan['marker']['step']==170 and plan['checkpoint_sha256']==d.CHECKPOINT_SHA256


def test_jsonl_literal_unicode_line_separator_is_not_record_separator(tmp_path):
 p=tmp_path/'rows.jsonl';p.write_text(json.dumps({'question':'a\u2028b'},ensure_ascii=False)+'\n')
 assert d.rows(p)==[{'question':'a\u2028b'}]


def test_actual_deadline_watchdog_terminates_only_its_own_exact_parent(tmp_path):
 import subprocess,sys
 nonce='b'*64
 code="""
import os,sys,time,subprocess
from pathlib import Path
from scripts import probe_research_selection_replay as d
from scripts import probe_research_parallel_b1 as proc
path=Path(sys.argv[1]);identity=proc.process_info(os.getpid());nonce=sys.argv[3]
d.write(path,{'owner':d.OWNER,'nonce':nonce,'deadline':time.monotonic()+2,'process':identity,'status':'active'})
watcher=subprocess.Popen([sys.executable,str(Path(d.__file__).resolve()),'--deadline-watchdog',str(path)])
time.sleep(30)
"""
 child=subprocess.Popen([sys.executable,'-c',code,str(tmp_path/'lease.json'),'--nonce',nonce])
 try:
  assert child.wait(timeout=4)==-15
  assert d.read(tmp_path/'deadline_watchdog_ready.json')['watched']['pid']==child.pid
 finally:
  if child.poll() is None:child.terminate();child.wait(timeout=3)


def test_real_tiny_native_observers_preserve_generation_prompt_and_runtime():
 import torch
 from tests.test_research_inference import TinyResearchModel,record
 from src.model.research_moe import install_research_moe
 from src.eval.research_inference import generate_prediction
 from src.eval.research_selection import selection_generation_mode
 torch.manual_seed(31)
 model=TinyResearchModel();model.wrappers=install_research_moe(model.base,3,'b')
 original_flags=[(p.requires_grad) for p in model.parameters()]
 public={k:v for k,v in record().items() if k in ('record_id','question','image_paths','image_hashes')}
 with selection_generation_mode(model):baseline=generate_prediction(model,public,max_new_tokens=4)
 runtime=d.NativeRuntime();runtime.saved={public['record_id']:{'generation':baseline}};runtime.logit_evidence=[];runtime.selection_census={};runtime.prompts=[]
 callback=[];runtime.on_generated=lambda *a:callback.append(a)
 old_generate=dict(model.base.__dict__).get('generate');old_template=dict(model.tokenizer.__dict__).get('apply_chat_template')
 with selection_generation_mode(model):actual=runtime.observed_generate(model,public,generate_prediction,phase='real_tiny',max_new_tokens=4)
 assert actual==baseline and not runtime.logit_evidence and len(callback)==1
 assert runtime.prompts[0]['prompt_sha256']==baseline['prompt_sha256']
 assert len(runtime.prompts[0]['native_encode_calls'])==2 and all('GOLD' not in x['text'] for x in runtime.prompts[0]['native_encode_calls'])
 assert [p.requires_grad for p in model.parameters()]==original_flags and model.training
 assert model.base.__dict__.get('generate') is old_generate and model.tokenizer.__dict__.get('apply_chat_template') is old_template
 assert not runtime.selection_census['real_tiny']['modules']['model.']['training']
 assert any(v.get('rope_attributes') for v in runtime.selection_census['real_tiny']['modules'].values())
