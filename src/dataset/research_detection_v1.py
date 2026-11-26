"""Prompted COCO detection with explicit integer xyxy in inclusive0..1000."""
from __future__ import annotations
import hashlib,json,math

COORDINATE_SYSTEM='xyxy_normalized_1000'
FORMAT='Return only a JSON array of objects with keys "bbox_2d" and "label". Each bbox_2d is [x_min,y_min,x_max,y_max], using integer coordinates from 0 to 1000 relative to the full image. Return [] when no requested object is present.'


def normalize_box(box,width,height):
 if len(box)!=4 or width<=0 or height<=0 or any(not math.isfinite(float(v)) for v in box):raise ValueError('Finite COCO xywh and positive image size required')
 x,y,w,h=map(float,box)
 if w<=0 or h<=0:raise ValueError('Positive box area required')
 xy=(max(0.,min(float(width),x)),max(0.,min(float(height),y)),max(0.,min(float(width),x+w)),max(0.,min(float(height),y+h)))
 values=[int(math.floor(v/d*1000+.5)) for v,d in zip(xy,(width,height,width,height))]
 if values[0]>=values[2] or values[1]>=values[3]:raise ValueError('Box degenerates after clipping/quantization')
 return values


def pixel_box(box,width,height):
 validate_box(box)
 return [box[0]*width/1000,box[1]*height/1000,box[2]*width/1000,box[3]*height/1000]


def validate_box(box):
 if not isinstance(box,list) or len(box)!=4 or any(type(v) is not int or not 0<=v<=1000 for v in box):raise ValueError('Four integer coordinates in0..1000 required')
 if box[0]>=box[2] or box[1]>=box[3]:raise ValueError('Ordered positive-area xyxy required')
 return True


def canonical_answer(objects):
 values=sorted(objects,key=lambda o:(o['label'],*o['bbox_2d']))
 for value in values:
  validate_box(value['bbox_2d'])
  if not isinstance(value['label'],str) or not value['label']:raise ValueError('Object label required')
 return json.dumps([dict(bbox_2d=v['bbox_2d'],label=v['label']) for v in values],ensure_ascii=False,separators=(',',':'))


def prompt(kind,classes,*,relation=None,all_classes=None):
 if kind in ('class_all','negative'):instruction='Locate every '+classes[0]+' in the image.'
 elif kind=='spatial':instruction=f'Locate the {relation} {classes[0]} in the image.'
 elif kind=='all80':instruction='Detect every object from these categories: '+', '.join(all_classes)+'.'
 else:raise ValueError('Unknown prompted detection task')
 return instruction+' '+FORMAT


def rank(value,salt):return hashlib.sha256(f'42:{salt}:{value}'.encode()).hexdigest()
