import json

import pytest

from scripts import report_research_results_v2 as report


def final_rows():
    rows=[]
    for model in ('smallthinker4','lfm25','giga31'):
        for vision in ('siglip2','qwen25'):
            for method in ('R1','R2','R3'):
                rows.append(dict(id=f'{model}__{vision}__{method}',model=model,vision=vision,method=method,
                    before_internal116=dict(group_macro_score=20),after_internal116=dict(group_macro_score=25),
                    heldout_OD_test=dict(primary_score=0),native_proof_passed=True,status='completed'))
    return dict(rows=rows,completed=18,total=18)


def test_first_table_has_all18_final_rows_and_preserves_legitimate_zero_scores():
    text=report.render(final_rows(),None,{},None)
    first=text.split('## Новые эксперименты')[0]
    rows=[line for line in first.splitlines() if line.startswith(('| SmallThinker','| LFM2.5','| GigaChat3.1'))]
    assert len(rows)==18
    assert all('—' not in row and '| 0.00 | passed |' in row for row in rows)
    assert '**18/18**' in first
    assert 'Shuffle' not in first


def test_missing_verified_metric_is_rejected_instead_of_using_other_protocol():
    final=final_rows();final['rows'][0]['after_internal116']=None
    with pytest.raises(ValueError,match='missing published metrics'):
        report.render(final,None,{},None)


def test_new_pending_result_is_not_filled_from_completed_old_variants():
    row=dict(model='smallthinker4',method='R5',arm='small__R5',VQA116_SFT1=19,
             VQA116_SFT2=None,VQA116_DET=None,detection_test_F1_IoU50=None,
             native_proof_SFT1=False,native_proof_SFT2=False,native_proof_DET=False)
    text=report.render(final_rows(),dict(complete=0,rows=[row]),dict(status='running'),None)
    assert '| SmallThinker | R5 | в очереди | 19.00* | — | — | — |' in text


def test_main_archives_the_exact_old_report_and_keeps_it_on_refresh(tmp_path,monkeypatch):
    from scripts import report_research_detection_mix_v1 as detector
    from scripts import report_research_common_pool_v1 as common
    monkeypatch.setattr(report,'ROOT',tmp_path)
    monkeypatch.setattr(detector,'main',lambda:None)
    monkeypatch.setattr(common,'main',lambda:None)
    directory=tmp_path/'research_results/detection_mix_v1';directory.mkdir(parents=True)
    (directory/'current_metrics.json').write_text(json.dumps(final_rows()))
    old='# Результаты серии18\n\n512 legacy answers\n'.encode()
    (tmp_path/'RESEARCH_RESULTS.md').write_bytes(old)
    report.main()
    archive=tmp_path/'RESEARCH_RESULTS_LEGACY_512.md'
    assert archive.read_bytes()==old
    assert '**18/18**' in (tmp_path/'RESEARCH_RESULTS.md').read_text()
    report.main()
    assert archive.read_bytes()==old
