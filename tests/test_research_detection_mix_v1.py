import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.dataset.research_detection_v1 import normalize_box,pixel_box,canonical_answer
from src.eval.research_detection_v1 import parse_prediction,match,score_record,aggregate
from src.utils import research_detection_training_v1 as training


def test_coordinate_clipping_roundtrip_and_integer_boundaries():
 assert normalize_box([0,0,640,480],640,480)==[0,0,1000,1000]
 assert normalize_box([64,48,128,96],640,480)==[100,100,300,300]
 assert pixel_box([100,100,300,300],640,480)==[64.,48.,192.,144.]
 assert normalize_box([-5,-10,30,20],640,480)==[0,0,39,21]
 with pytest.raises(ValueError):normalize_box([0,0,.001,.001],1000,1000)
 with pytest.raises(ValueError):normalize_box([0,0,float('nan'),20],640,480)


@pytest.mark.parametrize('text',['[{"label":"dog","bbox_2d":[false,0,20,20]}]',
 '[{"label":"dog","bbox_2d":[0.0,0,20,20]}]',
 '[{"label":"dog","bbox_2d":[0,0,1001,20]}]',
 '[{"label":"dog","bbox_2d":[20,0,20,20]}]',
 '[{"label":"cat","label":"dog","bbox_2d":[0,0,20,20]}]',
 'Here is the answer: []','```json\n[]\n```'])
def test_strict_format_rejects_noninteger_or_unbounded_outputs(text):
 assert parse_prediction(text)[1] is not None


def test_optimal_multibox_matching_avoids_greedy_trap_and_duplicate_credit():
 obj=lambda b:dict(label='dog',bbox_2d=b)
 targets=[obj([0,0,100,100]),obj([40,0,140,100])]
 predictions=[obj([20,0,120,100]),obj([0,0,90,100])]
 assert match(predictions,targets,.5)[0]==2
 assert match([targets[0],targets[0]],[targets[0]],.5)[0]==1


def test_empty_negative_requires_valid_json_and_is_not_detection_accuracy():
 record=dict(record_id='negative',detection=dict(objects=[],query_kind='negative'))
 valid=score_record(record,'[]');invalid=score_record(record,'No objects.')
 assert valid['negative_correct'] and not invalid['negative_correct']
 metrics=aggregate([valid,invalid])
 assert metrics['negative_query_accuracy']==50 and metrics['JSON_and_coordinate_valid_rate']==50
 assert metrics['primary_score']==0 and metrics['official_COCO_AP'] is False


class LossModel(nn.Module):
 def __init__(self,oom=False):super().__init__();self.w=nn.Parameter(torch.tensor(.25));self.calls=0;self.oom=oom
 def forward_loss(self,rows):
  self.calls+=1
  if self.oom and self.calls==2:raise torch.cuda.OutOfMemoryError('injected activation OOM')
  loss=sum((self.w-r[0]).square()*len(r[2]) for r in rows)
  return loss,sum(len(r[2]) for r in rows),0


def expected_gradient(model,rows,kinds):
 value=0
 for kind,weight in training.WEIGHTS.items():
  selected=[r for r,k in zip(rows,kinds) if k==kind]
  denominator=sum(len(r[2]) for r in selected)
  if denominator:value+=weight*sum((model.w-r[0]).square()*len(r[2]) for r in selected)/denominator
 return torch.autograd.grad(value,model.w)[0]


@pytest.mark.parametrize('oom',[False,True])
def test_three_stream_gradient_has_whole_group_denominators_and_oom_rollback(oom):
 model=LossModel(oom=oom);optimizer=torch.optim.SGD(model.parameters(),lr=1.)
 rows=[(1.,None,[1,2,3]),(2.,None,[1]),(-1.,None,[1,2]),(.5,None,[1,2,3,4])]
 kinds=['detection','multimodal_replay','text','multimodal_replay']
 expected=expected_gradient(model,rows,kinds)
 result=training.train_group(model,optimizer,rows,kinds,4)
 torch.testing.assert_close(model.w.grad,expected)
 assert model.w.item()==.25  # OOM retry never performs a partial optimizer update.
 assert result['supervised_tokens']==dict(detection=3,multimodal_replay=5,text=2)
 assert result['microbatch']==(2 if oom else 4)


def test_missing_stream_does_not_renormalize_other_weights():
 model=LossModel();optimizer=torch.optim.SGD(model.parameters(),lr=1.)
 rows=[(1.,None,[1]),(2.,None,[1,2])];kinds=['detection','multimodal_replay']
 expected=expected_gradient(model,rows,kinds)
 result=training.train_group(model,optimizer,rows,kinds,2)
 torch.testing.assert_close(model.w.grad,expected)
 assert result['stream_ce']['text']==0


def test_detection_release_has_real_boxes_ocr_replay_and_disjoint_partitions():
 release=training.validate_release();rows=training.load_rows(training.DATA/'sft_multimodal_train.jsonl')
 detection=[r for r in rows if r['task_group']=='object_detection']
 assert len(rows)==3000 and len(detection)==450
 assert sum(r['task_group']=='ocr_documents' for r in rows)==765
 assert len({o['label'] for r in detection for o in r['detection']['objects']})==80
 for row in detection:
  values,error=parse_prediction(row['answer']);assert error is None
  assert canonical_answer(row['detection']['objects'])==row['answer']
 splits=[{h for r in training.load_rows(training.DATA/name) for h in r['image_hashes']} for name in
  ('detection_train.jsonl','detection_validation.jsonl','detection_test.jsonl')]
 assert not splits[0]&splits[1] and not splits[0]&splits[2] and not splits[1]&splits[2]
 assert release['coordinate_range']==[0,1000]


def test_all18_methods_are_kept_and_private_additive_inference_is_bound():
 from scripts.research_detection_mix_queue_v1 import build_plan
 from src.model import research_additive_visual_v1 as additive
 plan=build_plan();assert len(plan['jobs'])==18
 assert {j['method'] for j in plan['jobs']}=={'R1','R2','R3'}
 assert all(j['branch']['blocks']==4 for j in plan['jobs'] if j['branch'])
 job=plan['jobs'][0];args=training.configure_args(job);module,inference,_=training.backend(args)
 assert inference.generate_prediction.__wrapped__.__globals__['set_modality_mask'] is additive.set_modality_mask
 assert not torch.cuda.is_initialized()


def test_generation_receives_no_detection_gold():
 record=dict(record_id='example',question='Find dogs',image_paths=['image.jpg'],image_hashes=['a'*64],
  answer='[]',detection=dict(objects=[],query_kind='negative'))
 seen=[]
 def generate(model,public,**kwargs):
  seen.append(public);assert set(public)=={'record_id','question','image_paths','image_hashes'}
  return dict(prediction='[]',generated_token_ids=[],prompt_sha256='hash')
 result=training.evaluate_detection(None,SimpleNamespace(generate_prediction=generate),
  SimpleNamespace(selection_generation_mode=lambda m:nullcontext()),[record])
 assert result['metrics']['negative_query_accuracy']==100 and len(seen)==1


def test_detection_status_overrides_completed_previous_matrix(tmp_path,monkeypatch):
 from scripts import monitor_research_additive_v1 as monitor
 monkeypatch.setattr(monitor,'ROOT',tmp_path)
 assert monitor.active_phase({'status':'completed'},{})['phase']=='completed'
 detection=dict(status='running',active_job='new_job',jobs={'new_job':dict(status='train')})
 run=tmp_path/'runs/research/detection_mix_v1/new_job';run.mkdir(parents=True)
 (run/'metrics.jsonl').write_text(json.dumps(dict(event='train',step=4,total_steps=56))+'\n')
 active=monitor.active_phase({'status':'completed'},{},detection)
 assert active['phase']=='detection_mix' and active['job']=='new_job' and active['progress']['step']==4
 detection['status']='completed'
 assert monitor.active_phase({}, {}, detection)['phase']=='detection_completed'


def test_scored_checkpoint_publication_reuses_same_physical_weights(tmp_path,monkeypatch):
 from src.utils.research_storage import BestLastCheckpointManager,DiskBudget
 manager=BestLastCheckpointManager(tmp_path,DiskBudget(tmp_path,0),mode='max')
 manager.save(lambda d:(d/'state.pt').write_bytes(b'actual trained weights'),step=56,metric=None,anticipated_bytes=22)
 pending=(tmp_path/'last/state.pt').resolve();inode=pending.stat().st_ino
 args=SimpleNamespace(output=str(tmp_path))
 monkeypatch.setattr(training,'load_rows',lambda p:[])
 monkeypatch.setattr(training,'evaluate_detection',lambda *a,**k:dict(metrics=dict(primary_score=12.),predictions=[]))
 monkeypatch.setattr(training,'source_cache_advice',lambda p:dict(status='test'))
 module=SimpleNamespace(prepare_group_macro_selection=lambda a:{})
 selection=SimpleNamespace(evaluate_internal_group_macro=lambda *a,**k:dict(metrics=dict(group_macro_score=24.),predictions=[]))
 training.finalize_evaluation(None,module,None,selection,args)
 assert (tmp_path/'best').resolve()==(tmp_path/'last').resolve()
 assert (tmp_path/'last/state.pt').stat().st_ino==inode
 assert not pending.exists()
 assert len(list((tmp_path/'.checkpoint_objects').iterdir()))==1
 assert json.loads((tmp_path/'completed.json').read_text())['detection_primary_score']==12.
