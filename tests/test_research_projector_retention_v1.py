import hashlib
import json
from pathlib import Path
import pytest
import torch

from src.utils.research_projector_retention_v1 import validate_projector_preservation,sha


def setup_source(root):
    run=root/'run';run.mkdir()
    config={'stage':'SFT','recipe':'1b'}
    (run/'resolved_config.json').write_text(json.dumps(config))
    model={'projector.0.weight':torch.randn(4,3),'projector.0.bias':torch.randn(4),
           'projector.2.weight':torch.randn(4,4),'projector.2.bias':torch.randn(4),
           'visual_experts.unchanged':torch.zeros(4)}
    path=run/'state.pt'
    torch.save(dict(model=model,config=config,step=56,seen=3529),path)
    (run/'best').symlink_to(run,target_is_directory=True);(run/'last').symlink_to(run,target_is_directory=True)
    initial=run/'initial.json'
    initial.write_text(json.dumps(dict(destination_config_sha256=sha(run/'resolved_config.json'),
        source_tensors={n:dict(sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for n,v in model.items()})))
    return run,initial,model


def test_unchanged_legacy_experts_do_not_block_verified_projector_reuse(tmp_path):
    run,initial,_=setup_source(tmp_path)
    report=validate_projector_preservation(run,initial,run/'report.json')
    assert report['status']=='passed'
    assert report['scope']=='caption_projector_preservation_only'
    assert report['common_tensor_groups']['projector']['changed_tensors']==0
    assert report['legacy_visual_expert_updates_claimed'] is False


def test_modified_projector_cannot_receive_a_passed_reuse_proof(tmp_path):
    run,initial,model=setup_source(tmp_path)
    model['projector.0.bias'][0]+=1
    torch.save(dict(model=model,config={'stage':'SFT','recipe':'1b'},step=56,seen=3529),run/'state.pt')
    with pytest.raises(ValueError,match='Caption projector changed'):
        validate_projector_preservation(run,initial,run/'report.json')
    assert not (run/'report.json').exists()
