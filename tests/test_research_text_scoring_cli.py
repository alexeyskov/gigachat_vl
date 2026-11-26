"""Actual CPU CLI tests using explicitly synthetic supplied responses."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('primary_text_cli_test_metrics',ROOT/'src/eval/research_text_metrics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def command(*args):
    return [sys.executable,str(ROOT/'scripts/score_research_text_eval.py'),*map(str,args)]


def test_cli_actual_mmlu_gsm_contract_immutable_resume_and_corruption(tmp_path):
    suite=m.load_suite(ROOT/'datasets/research/primary-text-en-v1');model='smallthinker4'
    tasks=['gsm8k','mmlu'];census={r['record_id']:r for r in suite['census'][model]['records']}
    producer={'schema_version':1,'owner':'research_primary_text_responses_v1',
        'suite_protocol_sha256':suite['protocol_sha256'],'manifest_sha256':suite['manifest_sha256'],
        'tasks':tasks,'model_id':model,'model_revision':suite['census'][model]['revision'],
        'stage':'baseline_before_vision','numerical_profile':{'weights':'nf4','compute':'bf16'},
        'checkpoint_identity':{'kind':'SYNTHETIC CPU UNIT FIXTURE; no model/checkpoint loaded'},
        'evidence_scope':'synthetic_scorer_validation','generation':suite['protocol']['evaluation']['generation']}
    raw=[]
    for record in suite['records']:
        if record['task'] not in tasks:continue
        c=census[record['record_id']]
        row={'record_id':record['record_id'],'task':record['task'],'source':record['source'],
            'input_text_sha256':m.sha(record['input_text']),'response_contract_sha256':m.object_sha(producer),
            'prompt_sha256':c['prompt_sha256'],'prompt_token_ids_sha256':c['prompt_token_ids_sha256'],
            'prompt_tokens':c['prompt_tokens']}
        if record['task']=='gsm8k':row['response']='#### 0' # Fixed, independent of released answers.
        else:
            row.update(candidate_log_likelihoods=dict.fromkeys(m.CHOICES,-1.),
                candidate_token_ids={x:[m.TOKEN_IDS[model][x]] for x in m.CHOICES},
                candidate_prefix_token_ids_sha256=dict.fromkeys(m.CHOICES,c['prompt_token_ids_sha256']),
                length_normalization=False)
        raw.append(row)
    p=tmp_path/'producer.json';p.write_text(json.dumps(producer))
    r=tmp_path/'responses.jsonl';r.write_text(''.join(m.canonical(row)+'\n' for row in raw))
    output=tmp_path/'scores'
    args=command('--responses',r,'--producer',p,'--tasks',*tasks,'--output',output)
    result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    summary=json.loads((output/'summary.json').read_text())
    assert summary['records']==1424 and summary['tasks']['mmlu']['subjects']==57
    assert summary['response_producer']['evidence_scope']=='synthetic_scorer_validation'
    assert summary['primary_completion_claim'] is False and summary['GPU_used'] is False
    assert summary['tasks']['mmlu']['subject_macro_accuracy']==summary['tasks']['mmlu']['all_record_accuracy']
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in output.iterdir()}
    assert subprocess.run(args,cwd=ROOT,capture_output=True).returncode==0
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in output.iterdir()}
    raw[0]['prompt_sha256']='0'*64;r.write_text(''.join(m.canonical(row)+'\n' for row in raw))
    result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==1
    assert all((output/name).read_bytes()==old[0] for name,old in before.items())
    failure=list(output.glob('failure_*.json'))
    assert len(failure)==1 and json.loads(failure[0].read_text())['status']=='failed_no_scores'


def test_cli_wrong_ifeval_environment_is_fatal_not_skipped(tmp_path):
    result=subprocess.run(command('--preflight-only','--output',tmp_path/'failure'),cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==1
    failures=list((tmp_path/'failure').glob('failure_*.json'))
    assert len(failures)==1
    report=json.loads(failures[0].read_text())
    assert report['status']=='failed_no_scores' and report['instructions_skipped']==0
    assert not (tmp_path/'failure/summary.json').exists()
