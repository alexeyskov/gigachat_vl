import copy
import json
import os
from pathlib import Path
import pytest
from scripts import handoff_research_budget18_v4 as h


@pytest.fixture
def owned(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'ROOT',tmp_path)
    source=tmp_path/'scripts/research_budget18_queue_v3.py'
    source.parent.mkdir();source.write_text('fixture controller')
    parent={'pid':999999,'birth':42,'argv':['python','controller']}
    path=tmp_path/'logs/research/budget18_workers/job/eval/request.json'
    path.parent.mkdir(parents=True)
    request={'parent':parent,'controller_source_sha256':h.base.sha(source),'kind':'eval'}
    path.write_text(json.dumps(request))
    child={'pid':os.getpid(),'ppid':parent['pid'],'birth':10,'state':'S',
           'argv':['python','research_budget18_queue_v3.py','--worker-request',str(path),
                   '--request-sha256',h.base.sha(path)]}
    monkeypatch.setattr(h.base,'proc',lambda pid:child if pid==os.getpid() else None)
    return parent,child,path


def test_only_exact_owned_worker_request_is_selected(owned):
    parent,child,_=owned
    result=h.worker_children(parent)
    assert len(result)==1 and result[0][0]==child


def test_pid_reuse_parent_birth_is_rejected_before_signals(owned):
    parent,_,_=owned;changed=copy.deepcopy(parent);changed['birth']+=1
    with pytest.raises(ValueError,match='exact old parent'):h.worker_children(changed)


def test_mutated_worker_request_is_rejected_before_signals(owned):
    parent,_,path=owned;path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='exact old parent'):h.worker_children(parent)


def test_foreign_child_is_ignored(owned,monkeypatch):
    parent,child,_=owned;child['ppid']+=1
    assert h.worker_children(parent)==[]
