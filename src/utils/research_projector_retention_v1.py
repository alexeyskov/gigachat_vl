"""Narrow proof for reusing P; never assert legacy visual experts were updated."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_projector_preservation(run,initial_path,output):
    import torch
    from src.utils.research_small_budget_v3 import require_complete_P,PKEYS
    from src.utils.research_storage import _atomic_json
    run,initial_path,output=Path(run),Path(initial_path),Path(output)
    config=json.loads((run/'resolved_config.json').read_text())
    initial=json.loads(initial_path.read_text())
    if config['stage']!='SFT' or config['recipe']!='1b':raise ValueError('Completed legacy R1 SFT source required')
    if sha(run/'resolved_config.json')!=initial['destination_config_sha256']:raise ValueError('Original SFT config changed')
    if (run/'best/state.pt').resolve(strict=True)!=(run/'last/state.pt').resolve(strict=True):
        raise ValueError('Projector proof requires identical final best/last source')
    checkpoint=(run/'last/state.pt').resolve(strict=True);before=checkpoint.stat()
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if saved['step']!=56 or saved['seen']!=3529 or saved['config']!=config:raise ValueError('Final56/3529 source required')
    state={name:saved['model'][name] for name in PKEYS}
    require_complete_P(state)
    for name,tensor in state.items():
        digest=hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if digest!=initial['source_tensors'][name]['sha256']:raise ValueError('Caption projector changed: '+name)
    report=dict(status='passed',scope='caption_projector_preservation_only',
        common_tensor_groups={'projector':dict(tensors=4,changed_tensors=0,elements=sum(t.numel() for t in state.values()))},
        legacy_visual_expert_training_validated=False,legacy_visual_expert_updates_claimed=False,
        previous_full_trainability_audit='No observed update in visual_experts; not a projector failure',
        source_snapshot_sha256=sha(initial_path),checkpoint=str(checkpoint),
        checkpoint_stat={k:getattr(before,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')},
        auditor_source_sha256=sha(__file__),scientific_weights_modified=False,created_unix=time.time())
    del state,saved
    after=checkpoint.stat()
    if any(getattr(before,k)!=getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns')):raise ValueError('Source changed during projector audit')
    _atomic_json(output,report);return report
