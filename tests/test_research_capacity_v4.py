import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.model.research_additive_visual_v1 import AdditiveVisualBank,VisualBranchConfig
from src.utils import research_capacity_v4 as capacity
from scripts import research_capacity_queue_v4 as queue


def test_four_fixed_blocks_have_no_router_and_all_visual_queries_are_dispatched():
    torch.manual_seed(42)
    bank=AdditiveVisualBank(8,VisualBranchConfig(blocks=4,width=8,heads=2,learned_masks=True))
    nn.init.normal_(bank.up.weight,std=.1)
    x=torch.randn(2,6,8,requires_grad=True);visual=torch.zeros(2,6,dtype=torch.bool);visual[:,1:3]=True
    y=bank(x,visual,torch.ones_like(visual));y.square().sum().backward()
    assert bank.router is None and bank.last_dispatch['counts'][1].sum()==4*int(visual.sum())
    assert all(b.q.weight.grad is not None for b in bank.blocks)
    assert torch.isfinite(x.grad).all()
    changed=x.detach().clone();changed[:,4:]=100
    with torch.no_grad():
        torch.testing.assert_close(bank(x.detach(),visual,torch.ones_like(visual))[:,:4],
            bank(changed,visual,torch.ones_like(visual))[:,:4],rtol=0,atol=0)


def test_capacity_plan_limits_to_two_passes_and_keeps_matched_two_block_control():
    plan=queue.build_plan()
    assert len(plan['arms'])==12 and len(plan['controls'])==6
    assert len(plan['jobs'])==37 and plan['SFT_stages_total']==25
    assert {j['training_pass'] for j in plan['jobs']}=={1,2}
    assert all(j['mode']=='fixed' for j in plan['jobs'])
    controls=[j for j in plan['jobs'] if j['branch']['blocks']==2]
    assert len(controls)==1 and controls[0]['training_pass']==2
    assert plan['jobs'][0]['branch']['blocks']==4
    for arm in plan['arms']:
        stages=[j for j in plan['jobs'] if j['arm']==arm['id']]
        assert [(j['stage'],j['training_pass']) for j in stages]==[('A1',1),('SFT',1),('SFT',2)]
        assert all(j['branch']['blocks']==4 for j in stages)


def test_private_training_order_offset_and_lr_policy_leave_torch_global_unchanged():
    from src.utils import research_stock_determinism as stock
    stock.bootstrap_environment()
    raw=stock._clone('scripts/run_research_stage.py')
    capacity._offset_training_order(raw)
    assert 'data_epoch_offset' in raw.main.__code__.co_names
    one=capacity.execution(asdict(VisualBranchConfig(blocks=4)),1)
    two=capacity.execution(asdict(VisualBranchConfig(blocks=4)),2)
    assert one['lr_scale']==1 and two['lr_scale']==.5
    assert two['data_epoch_offset']==1 and two['maximum_SFT_passes']==2
    assert torch.optim.AdamW.__module__=='torch.optim.adamw'


def test_continuation_requires_verified_first_pass_and_exact_architecture(tmp_path):
    model=nn.Module();model.projector=nn.Linear(2,2)
    model.model_ref={'repo':'model','revision':'pin'};model.vision_ref={'repo':'vision','revision':'pin'}
    model.cache=SimpleNamespace(fingerprint='features')
    manifest=tmp_path/'train.jsonl';manifest.write_text('{}\n')
    run=tmp_path/'run';run.mkdir();checkpoint=run/'state.pt';checkpoint.write_bytes(b'owned-weight-source')
    (run/'resolved_config.json').write_text('{}')
    (run/'native_reload_proof.json').write_text(json.dumps(dict(status='passed',checkpoint_sha256=capacity.core.sha(checkpoint))))
    config=dict(stage='SFT',model='model',vision='vision',recipe='1b',seed=42,additive_execution={'branch':{'blocks':4}})
    args=SimpleNamespace(**config,training_pass=2,initial_checkpoint=str(checkpoint),manifest=str(manifest))
    payload=dict(config=config,model=model.state_dict(),step=56,seen=3529,training_completed=True,resumable_optimizer=False,
        model_ref=model.model_ref,vision_ref=model.vision_ref,cache_fingerprint='features',manifest_sha256=capacity.core.sha(manifest))
    inference=SimpleNamespace(_validate_learned_state=lambda m,s:True)
    assert capacity.validate_continuation(payload,model,args,inference)
    payload['config']=dict(config,training_pass=2)
    with pytest.raises(ValueError):capacity.validate_continuation(payload,model,args,inference)
    payload['config']=config
    payload['model']['projector.weight']=payload['model']['projector.weight'].half()
    with pytest.raises(ValueError,match='dtypes'):capacity.validate_continuation(payload,model,args,inference)


def test_every_capacity_job_cpu_preflights_frozen_data_and_stage_budget(capsys):
    for job in queue.build_plan()['jobs']:
        args=[x for x in job['argv'] if x!='--allow-gpu']+['--preflight']
        capacity.main(args)
    rows=capsys.readouterr().out.splitlines()
    assert len(rows)==37
    assert all(json.loads(x)['status']=='CPU_preflight_only' for x in rows)
    assert not torch.cuda.is_initialized()


def test_private_runner_applies_half_lr_to_real_adam_without_patching_global_torch():
    original=torch.optim.AdamW
    args=SimpleNamespace(model=capacity.core.MODELS['smallthinker4'],vision=capacity.core.VISIONS['siglip2'],
        training_pass=2,lr_scale=.5,data_epoch_offset=1)
    module=capacity.runner(args,asdict(VisualBranchConfig(blocks=4)))
    p=nn.Parameter(torch.ones(2))
    optimizer=module.torch.optim.AdamW([dict(params=[p],lr=1e-4)])
    assert optimizer.param_groups[0]['lr']==5e-5
    scheduler=module.torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    p.sum().backward();optimizer.step();scheduler.step()
    assert torch.isfinite(p).all() and torch.optim.AdamW is original


def test_heldout_ce_uses_teacher_reference_and_token_weighting():
    from contextlib import nullcontext
    prepared=[]
    class Model:
        def prepare(self,records):prepared.extend(records);return records
        def forward_loss(self,records):return torch.tensor(sum(r['count']*r['ce'] for r in records)),sum(r['count'] for r in records),0
    rows=[dict(image_paths=['heldout.png'],teacher_reference='gold',count=1,ce=2.),
          dict(image_paths=['heldout2.png'],teacher_reference='gold2',count=3,ce=4.)]
    result=capacity.multimodal_ce(Model(),rows,SimpleNamespace(selection_generation_mode=lambda _:nullcontext()),batch_size=1)
    assert result==dict(ce=3.5,supervised_tokens=4,records=2)
    assert [r['answer'] for r in prepared]==['gold','gold2']
    assert all('answer' not in r for r in rows)


def test_monitor_follows_declared_capacity_run_root(tmp_path):
    from scripts.monitor_research_additive_v1 import collect_jobs
    run=tmp_path/'capacity4';run.mkdir()
    (run/'resolved_config.json').write_text(json.dumps(dict(stage='SFT',training_pass=2,
        additive_execution=dict(branch=dict(blocks=4)))))
    jobs=collect_jobs(dict(jobs=[dict(output=str(run),arm='capacity_arm')]))
    assert len(jobs)==1 and jobs[0]['blocks']==4 and jobs[0]['training_pass']==2
