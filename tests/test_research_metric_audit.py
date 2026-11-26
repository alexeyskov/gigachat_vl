import ast
import copy
import json
from pathlib import Path
import re
import string

import numpy as np
import pytest

from src.eval.research_metric_audit import (
    ROOT,SEMANTIC_FIELDS,ROW_FIELDS,analysis_protocol,audit_cached_evaluation,
    gqa_lmms_exact_match,json_sha256,sha256_bytes,upstream_provenance,
)
from src.eval.research_metrics import aggregate_scores,score_prediction,vqa_accuracy


@pytest.fixture(scope='module')
def upstream_exact_match():
    upstream_provenance()
    path=ROOT/'third_party/research_metric_audit/lmms_eval/metrics.py'
    tree=ast.parse(path.read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='exact_match_hf_evaluate')
    namespace={'np':np,'re':re,'string':string}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['exact_match_hf_evaluate']


@pytest.mark.parametrize('prediction,gold',[
    ('No.','no'),('Yes.','yes'),('Girl.','girl'),('  YES.\n','yes'),
    ('a-b','ab'),('1.5','15'),("don't",'dont'),('“yes”','yes'),
    ('The answer is yes.','yes'),('yes because it is visible.','yes'),
    ('red blue','redblue'),('red  blue','red blue'),('the girl','girl'),
    ('two','2'),('no','yes'),('yes','yes '),('İ','i'),('Straße','STRASSE'),
    (None,''),('','no'),
])
def test_full_string_rule_matches_actual_pinned_upstream_function(upstream_exact_match,prediction,gold):
    upstream_prediction='' if prediction is None else prediction.strip()
    expected=upstream_exact_match(predictions=[upstream_prediction],references=[gold],
                                  ignore_case=True,ignore_punctuation=True)['exact_match']
    assert gqa_lmms_exact_match(prediction,[gold])==expected


def test_upstream_task_flags_pred_strip_callsite_and_license_are_pinned():
    import yaml
    class Loader(yaml.SafeLoader):pass
    Loader.add_constructor('!function',lambda loader,node:loader.construct_scalar(node))
    config=yaml.load((ROOT/'third_party/research_metric_audit/lmms_eval/gqa.yaml').read_text(),Loader=Loader)
    metric=config['metric_list'][0]
    assert metric['metric']=='exact_match' and metric['ignore_case'] and metric['ignore_punctuation']
    task=(ROOT/'third_party/research_metric_audit/lmms_eval/task.py').read_text()
    assert 'result = [res.strip() if res is not None else "" for res in results]' in task
    provenance=upstream_provenance()
    assert all(len(row['revision'])==40 for row in provenance['files'])
    assert 'Apache' in (ROOT/'third_party/research_metric_audit/lmms_eval/LICENSE').read_text()
    assert 'Copyright (c) 2014, Aishwarya Agrawal' in (ROOT/'third_party/research_metric_audit/vqa_official/LICENSE').read_text()


def test_vqa_legacy_conditional_both_side_normalization_stays_faithful():
    source=(ROOT/'third_party/research_metric_audit/vqa_official/vqaEval.py').read_text()
    lines=source.splitlines();start=next(i for i,line in enumerate(lines) if 'if len(set(gtAnswers)) > 1:' in line)
    condition_indent=len(lines[start])-len(lines[start].lstrip('\t'))
    for phrase in ['resAns = self.processPunctuation(resAns)','resAns = self.processDigitArticle(resAns)']:
        line=next(line for line in lines[start:start+9] if phrase in line)
        assert len(line)-len(line.lstrip('\t'))>condition_indent
    assert vqa_accuracy('Yes.',['yes']*10,protocol='vqav2_official_legacy')==0.
    assert vqa_accuracy('Yes.',['yes']*9+['no'],protocol='vqav2_official_legacy')==1.
    assert vqa_accuracy('Yes.',['yes']*10,protocol='textvqa_evalai')==1.


def canonical_records(count=4):
    groups=[('general_vqa','gqa','exact_match',['no'],'No.'),
            ('ocr_documents','docvqa','anls',['invoice'],'invoice'),
            ('charts_diagrams','chartqa','chart_relaxed',['10'],'10'),
            ('knowledge_reasoning','scienceqa','mcq_accuracy',['A'],'A.')]
    records=[];predictions={}
    for index in range(count):
        group,benchmark,scorer,answers,prediction=groups[index%4]
        identity=sha256_bytes(f'fixture-{index}'.encode())
        row={'record_id':identity,'question':'Fixed question without gold',
             'answers':answers,'scorer':scorer,'task_group':group,'benchmark':benchmark,
             'source':{'benchmark':benchmark,'split':'validation'},'image_paths':['fixture.image'],
             'image_hashes':[sha256_bytes(f'image-{index}'.encode())]}
        if scorer=='mcq_accuracy':row['choices']=['first option','second option']
        records.append(row);predictions[identity]=prediction
    return records,predictions


def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def write_cache(base,manifest,records,predictions,variant='none',checkpoint='c'*64):
    directory=base/variant;directory.mkdir(parents=True,exist_ok=True)
    original_bytes=manifest.read_bytes();manifest_sha=sha256_bytes(original_bytes)
    effective=copy.deepcopy(records)
    if variant!='none':
        for row in effective:
            row['image_hashes']=[sha256_bytes((variant+row['record_id']).encode())]
            row['image_ablation']={'variant':variant}
        write_jsonl(directory/'transformed_manifest.jsonl',effective)
    transformed=(directory/'transformed_manifest.jsonl').read_bytes() if variant!='none' else original_bytes
    ablation={'schema_version':1,'variant':variant,'seed':42,'source_manifest_sha256':manifest_sha,
              'transformed_manifest_sha256':sha256_bytes(transformed),'records':len(records)}
    protocol={'schema_version':1,'manifest_sha256':manifest_sha,'manifest_records':len(records),
              'checkpoint_sha256':checkpoint,'image_ablation':ablation,
              'selected_record_ids_sha256':sha256_bytes(json.dumps([row['record_id'] for row in records]).encode()),
              'source_sha256':{'src/eval/research_metrics.py':sha256_bytes((ROOT/'src/eval/research_metrics.py').read_bytes())}}
    protocol_sha=json_sha256(protocol)
    rows=[]
    for record in effective:
        row={key:record[key] for key in ROW_FIELDS if key in record}
        row.update(prediction=predictions[record['record_id']],score=score_prediction(record,predictions[record['record_id']]),
                   protocol_sha256=protocol_sha)
        rows.append(row)
    summary=aggregate_scores(rows)
    summary.update(complete_manifest=True,manifest_records=len(records),protocol_sha256=protocol_sha,
                   checkpoint_sha256=checkpoint,manifest_sha256=manifest_sha,image_ablation=ablation)
    (directory/'protocol.json').write_text(json.dumps(protocol))
    (directory/'summary.json').write_text(json.dumps(summary))
    write_jsonl(directory/'predictions.jsonl',rows)
    return directory


def fixture_cache(tmp_path,variant='none'):
    records,predictions=canonical_records();manifest=tmp_path/'data'/'manifest.jsonl'
    write_jsonl(manifest,records)
    directory=write_cache(tmp_path/'evaluation',manifest,records,predictions,variant)
    return directory,manifest,records,predictions


def test_cached_rescore_is_read_only_complete_and_gqa_only(tmp_path):
    directory,manifest,_,_=fixture_cache(tmp_path)
    before={str(path):path.read_bytes() for path in directory.iterdir() if path.is_file()}
    result=audit_cached_evaluation(directory,manifest,require_records=4)
    assert result['strict_v3_original']['group_macro_score']==75.
    assert result['supplemental_gqa_lmms']['group_macro_score']==100.
    assert result['gqa']['raw_literal_correct']==result['gqa']['strict_v3_correct']==0
    assert result['gqa']['lmms_case_punctuation_correct']==1
    assert result['gqa']['case_and_terminal_punctuation_only_recoveries']==1
    assert result['identity']['raw_predictions_sha256']==sha256_bytes((directory/'predictions.jsonl').read_bytes())
    assert all(row['strict_v3_score']==row['supplemental_score'] for row in result['rows'] if row['benchmark']!='gqa')
    assert before=={str(path):path.read_bytes() for path in directory.iterdir() if path.is_file()}


@pytest.mark.parametrize('kind',['score','gold','duplicate','mixed_protocol','summary','checkpoint','source','partial'])
def test_corrupt_cache_provenance_or_original_scores_are_rejected(tmp_path,kind):
    directory,manifest,_,_=fixture_cache(tmp_path)
    path=directory/'predictions.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    if kind=='score':rows[0]['score']=1.
    elif kind=='gold':rows[0]['answers']=['yes']
    elif kind=='duplicate':rows[-1]=rows[0]
    elif kind=='mixed_protocol':rows[0]['protocol_sha256']='0'*64
    elif kind=='partial':rows.pop()
    elif kind=='summary':
        p=directory/'summary.json';value=json.loads(p.read_text());value['group_macro_score']=99.;p.write_text(json.dumps(value))
    elif kind=='checkpoint':
        p=directory/'summary.json';value=json.loads(p.read_text());value['checkpoint_sha256']='d'*64;p.write_text(json.dumps(value))
    elif kind=='source':
        p=directory/'protocol.json';value=json.loads(p.read_text());value['source_sha256']['src/eval/research_metrics.py']='0'*64;p.write_text(json.dumps(value))
    write_jsonl(path,rows)
    with pytest.raises(ValueError):audit_cached_evaluation(directory,manifest,require_records=4)


def test_all_variants_and_different_checkpoints_share_one_analysis_protocol(tmp_path):
    records,predictions=canonical_records();manifest=tmp_path/'data'/'manifest.jsonl';write_jsonl(manifest,records)
    expected_protocol=analysis_protocol(sha256_bytes(manifest.read_bytes()))
    for variant in ('none','shuffle','blank','text_only'):
        directory=write_cache(tmp_path/'evaluation',manifest,records,predictions,variant)
        result=audit_cached_evaluation(directory,manifest,require_records=4)
        assert result['variant']==variant and result['supplemental_gqa_lmms']['group_macro_score']==100.
        assert analysis_protocol(result['identity']['manifest_sha256'])==expected_protocol
    directory=write_cache(tmp_path/'another_checkpoint',manifest,records,predictions,checkpoint='d'*64)
    result=audit_cached_evaluation(directory,manifest,require_records=4)
    assert result['identity']['checkpoint_sha256']=='d'*64
    assert analysis_protocol(result['identity']['manifest_sha256'])==expected_protocol


def test_transformed_gold_or_question_cannot_be_redefined_for_a_supplemental_score(tmp_path):
    directory,manifest,_,_=fixture_cache(tmp_path,'blank')
    transformed=[json.loads(line) for line in (directory/'transformed_manifest.jsonl').read_text().splitlines()]
    transformed[0]['question']='Changed question'
    write_jsonl(directory/'transformed_manifest.jsonl',transformed)
    p=directory/'protocol.json';protocol=json.loads(p.read_text())
    protocol['image_ablation']['transformed_manifest_sha256']=sha256_bytes((directory/'transformed_manifest.jsonl').read_bytes())
    p.write_text(json.dumps(protocol))
    summary=json.loads((directory/'summary.json').read_text());summary['protocol_sha256']=json_sha256(protocol)
    summary['image_ablation']=protocol['image_ablation'];(directory/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='semantic'):audit_cached_evaluation(directory,manifest,require_records=4)


def test_cli_discovers_complete_and_pending_caches_without_rewriting_parent_artifacts(tmp_path):
    from scripts.audit_research_metrics import main
    records,predictions=canonical_records(512);manifest=tmp_path/'data'/'manifest.jsonl';write_jsonl(manifest,records)
    runs=tmp_path/'runs';run=runs/'fixture__4q__SFT_pilot_scienceqa_context_v2_balanced_s42'
    run.mkdir(parents=True);(run/'resolved_config.json').write_text(json.dumps({'recipe':'4q','seed':42,'stage':'SFT'}))
    directory=write_cache(run/'evaluation',manifest,records,predictions)
    pending=run/'evaluation'/'blank';pending.mkdir();(pending/'protocol.json').write_text('{}')
    before={str(path):path.read_bytes() for path in directory.iterdir() if path.is_file()}
    output=tmp_path/'audit'
    result=main(['--runs-root',str(runs),'--manifest',str(manifest),'--output',str(output)])
    assert len(result['evaluations'])==1 and len(result['pending'])==1
    assert result['coverage']['4q']['none']==[str(directory)]
    assert result['gpu_used'] is False and result['generation_launched'] is False
    first_sha=result['analysis_protocol_sha256']
    assert main(['--runs-root',str(runs),'--manifest',str(manifest),'--output',str(output)])['analysis_protocol_sha256']==first_sha
    assert before=={str(path):path.read_bytes() for path in directory.iterdir() if path.is_file()}
    with pytest.raises(ValueError,match='separate'):
        main(['--evaluation',str(directory),'--manifest',str(manifest),'--output',str(directory/'audit')])
