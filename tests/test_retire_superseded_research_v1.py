import json
from pathlib import Path

import pytest

from scripts import retire_superseded_research_v1 as cleanup
from src.utils.research_storage import BestLastCheckpointManager,DiskBudget


def create(root,name):
    run=root/'runs/research'/name;run.mkdir(parents=True)
    (run/'resolved_config.json').write_text(json.dumps({'output':str(run),'stage':'SFT'}))
    (run/'metrics.jsonl').write_text('original-metrics\n')
    manager=BestLastCheckpointManager(run,DiskBudget(run,0))
    manager.save(lambda d:(d/'state.pt').write_bytes(b'unneeded-weight-payload'),step=1,metric=1.,anticipated_bytes=100)
    return run


def plan(root,controls=()):
    value={'controls':[{'run':str(p)} for p in controls]}
    path=root/'configs/research/additive_visual_plan_v1.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value));return value


def test_protects_current_series_controls_and_required_projector_sources(tmp_path):
    old=create(tmp_path,'old_pilot')
    current=create(tmp_path,'additive_visual_v1')
    control=create(tmp_path,'smallthinker4__siglip2__4f__SFT__budget18_s42')
    source=create(tmp_path,'giga31__qwen25__1b__SFT__budget18_s42')
    result=cleanup.discover(tmp_path,plan(tmp_path,[control]))
    assert [r['run'] for r in result['runs']]==[str(old)]
    assert all((p/'last/state.pt').is_file() for p in (current,control,source))


def test_retirement_removes_only_owned_weights_and_preserves_metadata(tmp_path):
    run=create(tmp_path,'old_pilot');before=(run/'resolved_config.json').read_bytes()
    row=cleanup.inspect_run(run)
    result=cleanup.retire_one(row,tmp_path/'receipt.json',lambda _:[])
    assert result['logical_bytes_removed']>0
    assert not (run/'best').exists() and not (run/'last').exists()
    assert not list((run/'.checkpoint_objects').iterdir())
    assert (run/'resolved_config.json').read_bytes()==before
    assert (run/'metrics.jsonl').read_text()=='original-metrics\n'
    assert json.loads((run/'checkpoint_pruned.json').read_text())['status']=='pruned'


def test_live_mapping_blocks_retirement_without_removing_either_role(tmp_path):
    run=create(tmp_path,'old_pilot');row=cleanup.inspect_run(run)
    def busy(_):raise RuntimeError('live mapping')
    with pytest.raises(RuntimeError,match='live mapping'):cleanup.retire_one(row,tmp_path/'receipt',busy)
    assert (run/'best/state.pt').is_file() and (run/'last/state.pt').is_file()


def test_changed_checkpoint_is_not_deleted(tmp_path):
    run=create(tmp_path,'old_pilot');row=cleanup.inspect_run(run)
    (run/'last/state.pt').write_bytes(b'newer weight data')
    with pytest.raises(ValueError,match='changed'):cleanup.retire_one(row,tmp_path/'receipt',lambda _:[])
    assert (run/'last/state.pt').read_bytes()==b'newer weight data'


def test_foreign_owner_and_escaping_roles_are_rejected(tmp_path):
    run=create(tmp_path,'old_pilot')
    marker=run/'last/.research_checkpoint.json';data=json.loads(marker.read_text());data['owner']='foreign';marker.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Foreign'):cleanup.inspect_run(run)
    run2=create(tmp_path,'other_pilot');(run2/'last').unlink();(run2/'last').symlink_to(tmp_path)
    with pytest.raises(ValueError,match='escapes'):cleanup.inspect_run(run2)


def test_inode_shared_with_a_current_control_is_kept(tmp_path):
    old=create(tmp_path,'old_pilot');control=create(tmp_path,'current_control')
    (old/'last/state.pt').unlink();(old/'last/state.pt').hardlink_to(control/'last/state.pt')
    result=cleanup.discover(tmp_path,plan(tmp_path,[control]))
    assert result['runs']==[]
    assert result['skipped'][0]['reason']=='inode shared with a protected checkpoint'


def test_nested_symlink_to_live_weights_is_rejected(tmp_path):
    run=create(tmp_path,'old_pilot');(run/'last/unsafe').symlink_to(tmp_path)
    with pytest.raises(ValueError,match='Symlinks'):cleanup.inspect_run(run)


def test_automatic_source_retirement_waits_for_every_new_alignment():
    from scripts.watch_research_storage_retention_v1 import all_alignment_consumed
    p={'jobs':[dict(stage='A1',id=str(i)) for i in range(24)]}
    state={'jobs':{str(i):dict(status='completed') for i in range(23)}}
    assert not all_alignment_consumed(p,state)
    state['jobs']['23']={'status':'train'}
    assert not all_alignment_consumed(p,state)
    state['jobs']['23']={'status':'completed'}
    assert all_alignment_consumed(p,state)
