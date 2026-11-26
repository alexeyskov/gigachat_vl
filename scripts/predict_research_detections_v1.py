#!/usr/bin/env python3
"""Prompt a trained continuation and return normalized and original-image boxes."""
import argparse,json,os,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from src.utils import research_detection_training_v1 as training
from src.dataset.research_detection_v1 import FORMAT,pixel_box
from src.dataset.vision_cache import image_content_hash
from src.eval.research_detection_v1 import parse_prediction

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--prompt',required=True);args=p.parse_args()
 config=json.loads((args.run/'resolved_config.json').read_text());model,module,inference,selection,_=training.construct(SimpleNamespace(**config),args.run/'best/state.pt',detection_weights=True)
 record=dict(record_id='user_image',question=args.prompt+' '+FORMAT,image_paths=[str(args.image.resolve(strict=True))],image_hashes=[image_content_hash(args.image)])
 with selection.selection_generation_mode(model):result=inference.generate_prediction(model,record,max_new_tokens=256)
 values,error=parse_prediction(result['prediction'])
 from PIL import Image,ImageOps
 with Image.open(args.image) as im:width,height=ImageOps.exif_transpose(im).size
 output=dict(format_valid=error is None,error=error,coordinate_system='xyxy_normalized_1000',image_size=[width,height],
  objects=[dict(**o,bbox_pixels_xyxy=pixel_box(o['bbox_2d'],width,height)) for o in values],raw_prediction=result['prediction'])
 print(json.dumps(output,ensure_ascii=False))
 if error:raise SystemExit(2)

if __name__=='__main__':main()
