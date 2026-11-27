from pathlib import Path
import pytest

from scripts.research_common_pool_queue_v1 import build_plan
from src.utils import research_common_pool_training_v1 as training


def test_six_independent_arms_have_four_bounded_stages_and_no_old_sft_sources():
    plan=build_plan()
    assert plan['configurations']==6 and len(plan['jobs'])==24
    assert len({job['arm'] for job in plan['jobs']})==6
    assert {job['method'] for job in plan['jobs']}=={'R4','R5'}
    assert all(job['vision']=='Qwen/Qwen2.5-VL-7B-Instruct' for job in plan['jobs'])
    for arm in {job['arm'] for job in plan['jobs']}:
        jobs=[job for job in plan['jobs'] if job['arm']==arm]
        assert [job['phase'] for job in jobs]==['A1','SFT1','SFT2','DET']
        assert 'projectors/' in jobs[0]['source_checkpoint']
        for previous,current in zip(jobs,jobs[1:]):
            assert current['source_checkpoint']==str(Path(previous['output'])/'last/state.pt')
        assert all(Path(job['output']).is_relative_to(training.BASE) for job in jobs)


def test_private_runner_preserves_shared_inference_binding_and_qlora_base_policy():
    from src.model.research_moe import set_modality_mask
    plan=build_plan()
    for job in (plan['jobs'][0],plan['jobs'][4],plan['jobs'][8],plan['jobs'][16]):
        args=training.configure_args(job)
        module,inference,selection=training.backend(args)
        assert args.recipe=='1b'
        assert args.common_method==job['method']
        assert args.batch_size==args.effective_batch_size==64
        assert inference.generate_prediction.__wrapped__.__globals__['set_modality_mask'] is set_modality_mask
    import torch
    assert not torch.cuda.is_initialized()


def test_monitor_prioritizes_new_common_pool_over_completed_old_matrix(tmp_path,monkeypatch):
    from scripts import monitor_research_additive_v1 as monitor
    monkeypatch.setattr(monitor,'ROOT',tmp_path)
    run=tmp_path/'runs/research/common_pool_v1/small__R4/SFT1';run.mkdir(parents=True)
    (run/'metrics.jsonl').write_text('{"event":"train","step":3,"total_steps":56}\n')
    common=dict(status='running',active_job='small__R4__SFT1',jobs={'small__R4__SFT1':dict(status='train')})
    active=monitor.active_phase(dict(status='completed'),{},dict(status='completed'),common)
    assert active['phase']=='common_pool' and active['progress']['step']==3


def test_retirement_refuses_old_download_checkpoint_paths():
    from scripts.research_common_pool_queue_v1 import retire_predecessor
    job=dict(build_plan()['jobs'][1])
    job['source_checkpoint']=str(training.ROOT/'runs/research/detection_mix_v1/old18/last/state.pt')
    with pytest.raises(ValueError,match='never old18/download'):
        retire_predecessor(job)
