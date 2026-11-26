from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from src.model.research_additive_visual_v1 import VisualBranchConfig
from src.utils import research_additive_runner_v1 as backend
from scripts import research_additive_queue_v1 as queue


def test_matrix_has_two_distinct_branches_for_each_qlora_cell_and_reuses_full_controls():
    plan=queue.build_plan()
    assert len(plan['jobs'])==48 and len(plan['controls'])==6
    finals=[j for j in plan['jobs'] if j['stage']=='SFT']
    assert len(finals)==24
    assert len({(j['backend'],j['vision'],j['mode'],j['learned_masks']) for j in finals})==24
    assert all(j['steps']==56 and j['records']==3529 for j in finals)
    assert all(j['branch']['blocks']==(2 if j['mode']=='fixed' else 8) for j in finals)
    assert all(j['branch']['top_k']==2 for j in finals)
    assert all(len(j['requires'])==1 for j in finals)
    assert len({j['id'] for j in plan['jobs']})==48


@pytest.mark.parametrize('model',list(backend.MODELS.values()))
@pytest.mark.parametrize('vision',list(backend.VISIONS.values()))
def test_private_composition_builds_without_cuda_or_mutating_frozen_sources(model,vision):
    from src.utils import research_stock_determinism as stock
    before=dict(stock.PINS)
    branch=asdict(VisualBranchConfig())
    runner,inference,selection=backend.runner(branch,model_repo=model,vision_repo=vision)
    assert runner.__name__!='scripts.run_research_stage'
    assert not torch.cuda.is_initialized()
    assert stock.PINS==before
    stock.definition()  # frozen source hashes must still pass
    fake=SimpleNamespace(model_ref={'repo':model},vision_ref={'repo':vision},cache=SimpleNamespace(fingerprint='cache'))
    with pytest.raises(ValueError,match='caption-aligned projector reuse'):
        runner.validate_initial_checkpoint({'model':{},'config':{'stage':'SFT'}},fake,SimpleNamespace(stage='A1'))


def test_main_preflights_all_48_real_jobs_without_gpu(capsys):
    for job in queue.build_plan()['jobs']:
        args=[v for v in job['argv'] if v!='--allow-gpu']+['--preflight']
        backend.main(args)
    lines=capsys.readouterr().out.splitlines()
    assert len(lines)==48
    assert all(json.loads(line)['status']=='CPU_preflight_only' for line in lines)
    assert not torch.cuda.is_initialized()
