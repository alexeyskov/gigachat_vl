"""Strict JSON detection scoring; no claim of official COCO AP."""
from __future__ import annotations
import json,math
from src.dataset.research_detection_v1 import validate_box


def parse_prediction(text,max_boxes=64):
 try:
  def unique_object(pairs):
   value={}
   for key,item in pairs:
    if key in value:raise ValueError('Duplicate JSON key')
    value[key]=item
   return value
  values=json.loads(text.strip(),object_pairs_hook=unique_object)
  if not isinstance(values,list) or len(values)>max_boxes:raise ValueError('Bounded JSON array required')
  for value in values:
   if not isinstance(value,dict) or set(value)!={'bbox_2d','label'}:raise ValueError('Only bbox_2d/label keys required')
   validate_box(value['bbox_2d'])
   if not isinstance(value['label'],str) or not value['label'].strip():raise ValueError('Nonempty label required')
  return values,None
 except (ValueError,TypeError,KeyError) as error:return [],str(error)


def iou(a,b):
 dx=max(0,min(a[2],b[2])-max(a[0],b[0]));dy=max(0,min(a[3],b[3])-max(a[1],b[1]))
 intersection=dx*dy;union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
 return intersection/union if union>0 else 0.


def match(predictions,targets,threshold):
 # DP maximizes matched cardinality first, IoU second; target requests contain<=4 boxes.
 if len(targets)>8:raise ValueError('Bounded exact assignment requires<=8 targets')
 states={0:(0,0.)}
 for predicted in predictions:
  updated=dict(states)
  for mask,value in states.items():
   for index,target in enumerate(targets):
    if mask>>index&1 or predicted['label'].strip().casefold()!=target['label'].strip().casefold():continue
    overlap=iou(predicted['bbox_2d'],target['bbox_2d'])
    if overlap<threshold:continue
    key=mask|1<<index;candidate=(value[0]+1,value[1]+overlap)
    if candidate>updated.get(key,(-1,-1.)):updated[key]=candidate
  states=updated
 return max(states.values())


def score_record(record,prediction):
 target=record['detection']['objects'];values,error=parse_prediction(prediction)
 result=dict(record_id=record['record_id'],kind=record['detection']['query_kind'],prediction=prediction,
  format_valid=error is None,format_error=error,target_count=len(target),predicted_count=len(values))
 for threshold in (.5,.75):
  tp,total_iou=match(values,target,threshold) if error is None else (0,0.)
  fp=len(values)-tp;fn=len(target)-tp;denominator=2*tp+fp+fn
  result[f'iou{int(threshold*100)}']=dict(tp=tp,fp=fp,fn=fn,f1=(2*tp/denominator if denominator else float(error is None)),matched_iou_sum=total_iou)
 result['negative_correct']=bool(not target and error is None and not values)
 return result


def aggregate(rows):
 if not rows:raise ValueError('Nonempty prompted detection evaluation required')
 positives=[r for r in rows if r['target_count']];negatives=[r for r in rows if not r['target_count']]
 value=dict(records=len(rows),positive_queries=len(positives),negative_queries=len(negatives),
  JSON_and_coordinate_valid_rate=100*sum(r['format_valid'] for r in rows)/len(rows),
  negative_query_accuracy=100*sum(r['negative_correct'] for r in negatives)/len(negatives) if negatives else None,
  coordinate_system='xyxy_normalized_1000',official_COCO_AP=False)
 for threshold in (50,75):
  tp=sum(r[f'iou{threshold}']['tp'] for r in rows);fp=sum(r[f'iou{threshold}']['fp'] for r in rows);fn=sum(r[f'iou{threshold}']['fn'] for r in rows)
  value[f'micro_F1_IoU{threshold}']=100*2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
  value[f'positive_macro_F1_IoU{threshold}']=100*sum(r[f'iou{threshold}']['f1'] for r in positives)/len(positives) if positives else 0.
  value[f'precision_IoU{threshold}']=100*tp/(tp+fp) if tp+fp else 0.
  value[f'recall_IoU{threshold}']=100*tp/(tp+fn) if tp+fn else 0.
 value['primary_score']=value['positive_macro_F1_IoU50']
 return value
