#!/usr/bin/env python3
"""Separate derived-donor single-record native evaluation; no scorer or queue."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils import research_derived_vision_v1 as derived


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--checkpoint',choices=('best','last'),default='best')
    parser.add_argument('--export-root',type=Path,default=derived.DEFAULT_EXPORT_ROOT)
    parser.add_argument('--cache-dir',default='cache/vision')
    parser.add_argument('--reserve-gb',type=float,default=120.)
    parser.add_argument('--allow-gpu',action='store_true')
    parser.add_argument('--question')
    parser.add_argument('--image',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.reserve_gb<120:raise ValueError('Preserve120GiB disk reserve')
    config=derived.read(args.run/'resolved_config.json')
    proof=derived.metadata_preflight(config.get('vision'),export_root=args.export_root,
        preprocessing_profile=config.get('vision_preprocessing_profile'))
    expected=proof['config_contract']
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Run metadata lacks exact derived source/export/profile/cache/code binding')
    proof.update(checkpoint_payload_opened=False,checkpoint_payload_verified=False)
    if not args.allow_gpu:
        print(json.dumps(proof,ensure_ascii=False,indent=2));return 0
    if not args.question or not args.question.strip() or args.output is None:raise ValueError('Explicit question and new output required')
    if args.output.exists() or args.output.is_symlink():raise ValueError('Fresh single-record derived output required')
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('Prototype outputs stay outside the managed source run')
    model,provenance=derived.load_derived_checkpoint(args.run/args.checkpoint/'state.pt',
        export_root=args.export_root,cache_dir=args.cache_dir,reserve_gb=args.reserve_gb)
    from src.eval.research_inference import generate_prediction
    record={'record_id':'explicit_single_record','question':args.question,'image_paths':[str(args.image)] if args.image else []}
    prediction=generate_prediction(model,record,max_new_tokens=64)
    report={'schema_version':1,'owner':derived.OWNER,'scope':'opt_in_single_record_native_generation_no_benchmark_score',
        'record':record,'prediction':prediction,'model_provenance':provenance,
        'source_sha256':derived.code_binding(),'pruning_authorized':False,
        'paired_source_CUDA_parity_proved':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(report,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps({'status':'single_record_generated','output':str(args.output),'pruning_authorized':False}));return 0


if __name__=='__main__':raise SystemExit(main())
