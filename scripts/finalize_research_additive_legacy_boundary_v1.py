#!/usr/bin/env python3
"""CPU-only lossless finalization of the legacy trainer at the handoff boundary.

The new queue does not run legacy compaction callbacks. Finish their storage
and projector audit explicitly, without claiming the deferred GPU replay passed
or taking the new queue's ownership of legacy state metadata.
"""
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.research_storage import _atomic_json
from src.utils.research_lossless_delta import memory_guard

def main():
    from scripts import research_budget18_queue_v1 as q
    import torch
    torch.set_num_threads(2)
    handoff=ROOT/'research_results/additive_visual_v1/handoff'
    output=ROOT/'research_results/additive_visual_v1/legacy_boundary_finalize.json'
    if output.exists() and json.loads(output.read_text()).get('status')=='passed_storage_and_projector_audit':return
    while True:
        if (handoff/'lease.json').exists():
            lease=q.read(handoff/'lease.json');worker=lease['training_worker']
            if not q.same(worker,q.proc(worker['pid'])) and (handoff/'completed.json').exists():break
        print(json.dumps(dict(event='awaiting_legacy_natural_completion_for_CPU_only_finalization')),flush=True)
        time.sleep(30)
    job=lease['current_job'];run=ROOT/'runs/research'/job
    if not (run/'completed.json').exists():raise ValueError('Legacy trainer ended without completed artifact')
    if job!='giga31__qwen25__1b__SFT__budget18_s42':raise ValueError('This boundary bridge is explicitly scoped to the observed Giga/Qwen R1')
    memory_guard(8*2**30)
    from scripts.compact_research_checkpoints import compact_completed_checkpoint
    compaction=compact_completed_checkpoint(run,reserve_gb=32)
    from scripts import audit_research_budget18_updates_v1 as audit
    directory=audit.OUT/job;initial=directory/'initial_tensors.json';report=directory/'report.json'
    if not initial.exists():raise ValueError('Original pre-SFT immutable tensor census required')
    config=json.loads((run/'resolved_config.json').read_text())
    if not report.exists():
        from src.utils.research_projector_retention_v1 import validate_projector_preservation
        audit.audit_locked(run,validate_projector_preservation,run,initial,report)
    if audit.read(report)['status']!='passed':raise ValueError('Legacy projector freeze audit failed')
    _atomic_json(output,dict(status='passed_storage_and_projector_audit',job=job,
        compaction=compaction,projector_audit_sha256=audit.sha(report),
        old_scientific_weights_changed=False,old_GPU_replay_passed_claimed=False,
        legacy_queue_state_modified=False,old_A1_weights_retired=False,
        next_use='verified frozen projector extraction for new Giga/Qwen additive experiments'))
    print(json.dumps(dict(event='legacy_boundary_storage_and_projector_ready',job=job)),flush=True)

if __name__=='__main__':main()
