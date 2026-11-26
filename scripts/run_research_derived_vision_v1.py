#!/usr/bin/env python3
"""Separate derived-donor training facade; metadata-only unless ROOT opts in."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils import research_derived_vision_v1 as derived


def value(argv,name,default=None):
    positions=[i for i,arg in enumerate(argv) if arg==name]
    if not positions:return default
    if len(positions)!=1 or positions[0]+1>=len(argv) or argv[positions[0]+1].startswith('--'):
        raise ValueError('Expected one explicit value for '+name)
    return argv[positions[0]+1]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--export-root',type=Path,default=derived.DEFAULT_EXPORT_ROOT)
    parser.add_argument('--allow-gpu',action='store_true')
    known,remaining=parser.parse_known_args(argv)
    repo=value(remaining,'--vision')
    profile=value(remaining,'--vision-preprocessing-profile')
    proof=derived.metadata_preflight(repo,export_root=known.export_root,preprocessing_profile=profile)
    if not known.allow_gpu:
        print(json.dumps(proof,ensure_ascii=False,indent=2));return 0
    private=derived._private_runner();args=private.parse_args(remaining)
    binding=proof['binding'];derived.configure_args(args,binding)
    private=derived.make_training_runner(args,binding)
    private.parse_args=lambda argv=None:args
    private.main()
    return 0


if __name__=='__main__':raise SystemExit(main())
