#!/usr/bin/env python3
"""CPU-only scoring of supplied, source-bound primary_text_en_v1 responses."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


metrics=module('primary_text_metrics','src/eval/research_text_metrics.py')


def save_immutable(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes()!=raw:raise ValueError('Existing scorer output differs; choose a new explicit namespace')
        return
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
        # A link provides no-replace commit, including competing scorers.
        try:os.link(temp,path)
        except FileExistsError:
            if path.read_bytes()!=raw:raise ValueError('Competing scorer output differs')
    finally:Path(temp).unlink(missing_ok=True)


def candidate_metadata_check(suite):
    assets=json.loads((ROOT/'datasets/research/text-eval-inputs/assets.json').read_text())
    verified={}
    for model,ids in metrics.TOKEN_IDS.items():
        item=next(a for a in assets['files'] if a['id']==model and a['filename']=='tokenizer.json')
        raw=Path(item['path']).read_bytes()
        if metrics.sha(raw)!=item['sha256'] or item['sha256']!=suite['census'][model]['tokenizer_assets_sha256']['tokenizer.json']:
            raise ValueError('Candidate tokenizer source asset differs')
        tokenizer=json.loads(raw)
        if tokenizer['model']['type']!='BPE' or any(tokenizer['model']['vocab'][c]!=v for c,v in ids.items()):
            raise ValueError('Pinned A/B/C/D token vocabulary differs')
        if any(r.get('candidate_continuation_token_lengths')!=dict.fromkeys(metrics.CHOICES,1)
               for r in suite['census'][model]['records'] if r['task']=='mmlu'):
            raise ValueError('Frozen census lacks actual stable one-token candidate boundaries')
        verified[model]={'tokenizer_sha256':item['sha256'],'candidate_token_ids':ids,
            'actual_native_prefix_stability_census_records':912}
    return verified


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',default=str(ROOT/'datasets/research/primary-text-en-v1'))
    parser.add_argument('--responses');parser.add_argument('--producer')
    parser.add_argument('--tasks',nargs='+',choices=['gsm8k','mmlu','ifeval'],default=['gsm8k','mmlu','ifeval'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args(argv)
    if len(args.tasks)!=len(set(args.tasks)):parser.error('Duplicate task scope')
    if not args.preflight_only and (not args.responses or not args.producer):
        parser.error('Supply actual --responses and --producer, or --preflight-only; no model generation is performed')
    output=Path(args.output)
    if output.resolve().is_relative_to(Path(args.bundle).resolve()):
        parser.error('Scoring output must be outside the immutable input bundle')
    try:
        suite=metrics.load_suite(args.bundle)
        boundary=candidate_metadata_check(suite)
        official=None
        if 'ifeval' in args.tasks:
            official=module('primary_ifeval_official','src/eval/research_ifeval_official.py').OfficialIFEval()
        source_files=['src/eval/research_text_metrics.py','src/eval/research_ifeval_official.py',
            'scripts/score_research_text_eval.py','scripts/setup_research_text_eval_env.py',
            'tests/test_research_text_metrics.py','tests/test_research_text_scoring_cli.py',
            'research_results/validate_official_ifeval_cpu.py']
        sources={p:metrics.sha((ROOT/p).read_bytes()) for p in source_files}
        protocol={'schema_version':1,'owner':metrics.OWNER,'suite_protocol_sha256':suite['protocol_sha256'],
            'manifest_sha256':suite['manifest_sha256'],'tasks':args.tasks,'candidate_boundary_provenance':boundary,
            'source_sha256':sources,'IFEval_official':official.provenance if official else None,
            'GSM_rule':'one final nonblank anchored #### finite Decimal line; comma removal; no last-number rescue',
            'MMLU_rule':'supplied conditional log likelihood; stable one-token A/B/C/D; no length normalization; A-first ties; equal subject macro',
            'model_generation_performed':False,'primary_completion_claim':False}
        if args.preflight_only:
            report={'schema_version':1,'status':'passed','scope':'scorer/dependency validation only; no text-model scores',
                'protocol':protocol,'ifeval_preflight':official.preflight(suite['records']) if official else None,
                'candidate_boundaries_verified_models':list(boundary),'model_generation_performed':False,
                'GPU_used':False,'torch_imported':'torch' in sys.modules}
            save_immutable(output/'scorer_preflight.json',(metrics.canonical(report)+'\n').encode())
            print(json.dumps({'status':'passed','scope':report['scope'],'output':str(output)}));return 0
        responses=metrics.read_jsonl(args.responses);producer=json.loads(Path(args.producer).read_text())
        scored,summary=metrics.score_rows(suite,responses,producer,tasks=args.tasks,ifeval=official)
        protocol.update({'raw_responses_sha256':metrics.sha(Path(args.responses).read_bytes()),
            'producer_file_sha256':metrics.sha(Path(args.producer).read_bytes()),
            'response_contract_sha256':metrics.object_sha(producer)})
        summary.update({'scoring_protocol_sha256':metrics.object_sha(protocol),
            'raw_responses_sha256':protocol['raw_responses_sha256'],'GPU_used':False})
        save_immutable(output/'scoring_protocol.json',(metrics.canonical(protocol)+'\n').encode())
        save_immutable(output/'scored_responses.jsonl',b''.join((metrics.canonical(r)+'\n').encode() for r in scored))
        save_immutable(output/'summary.json',(metrics.canonical(summary)+'\n').encode())
        print(json.dumps({'status':summary['status'],'records':len(scored),'output':str(output),
            'evidence_scope':producer['evidence_scope'],'primary_completion_claim':False}));return 0
    except (ImportError,LookupError,RuntimeError,ValueError,OSError,KeyError) as error:
        failure={'schema_version':1,'status':'failed_no_scores','error_type':type(error).__name__,
            'error':str(error),'instructions_skipped':0,'model_generation_performed':False,
            'primary_completion_claim':False}
        # Do not overwrite an earlier failure/proof when dependencies change.
        name='failure_'+metrics.object_sha(failure)[:16]+'.json'
        save_immutable(output/name,(metrics.canonical(failure)+'\n').encode())
        print(json.dumps(failure));return 1


if __name__=='__main__':raise SystemExit(main())
