"""Source/phase/parent constraints before any warmup signals or GPU allocation."""
import copy,json
import pytest
import scripts.warm_research_budget18_qwen_v2 as w


def fixture(tmp_path):
    service={'statename':'RUNNING','pid':42};parent={'pid':42,'birth':7,'argv':['python','owned-controller']}
    request={'kind':'eval','argv':['--image-ablation','text_only'],'backend':'smallthinker4','job_id':w.TARGET,
        'controller_source_sha256':w.CONTROLLER_SHA,'data_release_sha256':w.RELEASE_SHA,'parent':copy.deepcopy(parent)}
    path=tmp_path/'request.json';path.write_text(json.dumps(request));proc={'request':str(path)}
    evaluation={'ppid':42,'argv':['python','worker','--request-sha256',w.q.sha(path)]}
    return [service,parent,{'active_job':w.TARGET},evaluation,request,proc]


def test_exact_target_accepts_without_signals(tmp_path):
    w.validate_target(*fixture(tmp_path))


@pytest.mark.parametrize('changed',['phase','birth','source','data','worker_parent','job','request_bytes'])
def test_inexact_target_refuses_before_signals(tmp_path,changed):
    args=fixture(tmp_path);service,parent,state,evaluation,request,proc=args
    if changed=='phase':request['argv']=['--image-ablation','blank']
    elif changed=='birth':request['parent']['birth']+=1
    elif changed=='source':request['controller_source_sha256']='0'*64
    elif changed=='data':request['data_release_sha256']='0'*64
    elif changed=='worker_parent':evaluation['ppid']+=1
    elif changed=='job':request['job_id']='different'
    else:
        from pathlib import Path
        p=Path(proc['request']);p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError):w.validate_target(*args)
