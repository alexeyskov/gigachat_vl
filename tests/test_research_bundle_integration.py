"""Optional bundle profile across runner, validation, compaction and evaluation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from src.utils.research_checkpoint_compat import checkpoint_identity,load_research_payload
from src.utils.research_storage import BestLastCheckpointManager,DiskBudget


def test_runner_optional_bundle_resume_adam_and_completed_compaction(tmp_path,monkeypatch):
    import scripts.run_research_stage as runner
    from tests.test_research_runner import FakeStageModel
    from scripts.validate_research_checkpoint import validate_run
    from scripts.compact_research_checkpoints import compact_completed_checkpoint
    monkeypatch.chdir(tmp_path)
    train=tmp_path/'train.jsonl';validation=tmp_path/'validation.jsonl'
    train.write_text((json.dumps({'image_paths':[],'question':'q','answer':'a'})+'\n')*4)
    validation.write_text(json.dumps({'image_paths':[],'question':'q','answer':'a'})+'\n')
    monkeypatch.setattr(runner,'ResearchModel',FakeStageModel)
    monkeypatch.setattr('src.utils.research_provenance.write_run_provenance',lambda *a,**k:{'code_sha256':'test','source_archive':'fixture'})
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    parse=runner.parse_args
    def arguments(stage,out,extra=()):
        return parse(['--model','tiny','--vision','tinyvision','--manifest',str(train),'--validation',str(validation),
            '--stage',stage,'--recipe','4f','--output',str(out),'--checkpoint-format','bundle',
            '--batch-size','1','--effective-batch-size','1','--max-steps','1' if stage=='A0' else '2',
            '--eval-records','1','--eval-every','1','--reserve-gb','0',*extra])
    def execute(args):
        monkeypatch.setattr(runner,'parse_args',lambda:args)
        runner.main()
    aligned=tmp_path/'alignment'
    execute(arguments('A0',aligned))
    interrupted=tmp_path/'interrupted';initial=str(aligned/'last/state.pt')
    original_log=runner.log
    def crash(path,payload,**kwargs):
        if payload.get('event')=='train' and payload.get('step')==2:raise RuntimeError('simulated interrupted effective batch')
        return original_log(path,payload,**kwargs)
    monkeypatch.setattr(runner,'log',crash)
    with pytest.raises(RuntimeError,match='simulated'):
        execute(arguments('SFT',interrupted,['--initial-checkpoint',initial]))
    saved=load_research_payload(interrupted/'last')
    assert saved['step']==1 and saved['resumable_optimizer']
    monkeypatch.setattr(runner,'log',original_log)
    execute(arguments('SFT',interrupted,['--resume']))
    baseline=tmp_path/'baseline'
    execute(arguments('SFT',baseline,['--initial-checkpoint',initial]))
    resumed=load_research_payload(interrupted/'last');control=load_research_payload(baseline/'last')
    for key in control['model']:assert torch.equal(resumed['model'][key],control['model'][key])
    assert resumed['step']==control['step']==2
    for state_id,values in control['optimizer']['state'].items():
        for key,value in values.items():
            actual=resumed['optimizer']['state'][state_id][key]
            if torch.is_tensor(value):assert torch.equal(value,actual)
            else:assert value==actual
    receipt=validate_run(interrupted,cpu_structural_only=True)
    item=receipt['checkpoints'][0]
    assert item['checkpoint_bundle_identity']['sha256']==checkpoint_identity(interrupted/'last')['sha256']
    assert item['checkpoint_artifact_total_bytes']>item['checkpoint_size_bytes']
    before_model_sha=next(row['sha256'] for row in checkpoint_identity(interrupted/'last')['members'] if row['role']=='model')
    compact_completed_checkpoint(interrupted,reserve_gb=0)
    compacted=load_research_payload(interrupted/'last')
    assert compacted['training_completed'] and not compacted['resumable_optimizer'] and 'optimizer' not in compacted
    after_model_sha=next(row['sha256'] for row in checkpoint_identity(interrupted/'last')['members'] if row['role']=='model')
    assert before_model_sha==after_model_sha
    with pytest.raises(ValueError,match='no optimizer'):
        execute(arguments('SFT',interrupted,['--resume']))


def test_cli_bundle_protocol_hash_covers_model_components(tmp_path,monkeypatch):
    import scripts.eval_research_checkpoint as cli
    from tests.test_research_inference import cli_fixture
    argv=cli_fixture(tmp_path,monkeypatch)
    run=Path(argv[argv.index('--run')+1]);old=run/'best'
    (old/'state.pt').unlink();old.rmdir()
    payload={'step':1,'model':{'projector.weight':torch.ones(3,4)},
             'optimizer':{'state':{},'param_groups':[{'params':[0]}]}}
    manager=BestLastCheckpointManager(run,DiskBudget(run,0))
    manager.save_torch(payload,step=1,metric=.5,anticipated_bytes=1024**2,checkpoint_format='bundle')
    summary=cli.main(argv)
    identity=checkpoint_identity(run/'best')
    assert summary['checkpoint_sha256']==identity['sha256']
    assert summary['checkpoint_sha256']!=hashlib.sha256((run/'best/state.pt').read_bytes()).hexdigest()
    assert summary['checkpoint_bundle_identity']['total_bytes']==identity['total_bytes']
    assert cli.main(argv+['--resume'])==summary
    from scripts.research_queue import ResearchQueue
    manifest=Path(argv[argv.index('--manifest')+1]);destination=Path(argv[argv.index('--output')+1])
    verification={'checkpoint_sha256':identity['sha256'],'checkpoint_artifact_identity':identity,
                  'manifest_sha256':summary['manifest_sha256'],'output':destination,
                  'records':[json.loads(line) for line in manifest.read_text().splitlines()]}
    queue=ResearchQueue.__new__(ResearchQueue)
    job={'evaluation':{'max_new_tokens':64}}
    assert queue._validate_evaluation(job,verification)['status']=='completed'
    broken=dict(summary);broken.pop('checkpoint_bundle_identity')
    (destination/'summary.json').write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='every current bundle component'):
        queue._validate_evaluation(job,verification)
    (destination/'summary.json').write_text(json.dumps(summary))
    component=(run/'best').resolve()/'model.state.pt'
    with component.open('r+b') as handle:
        handle.seek(-8,2);value=handle.read(1);handle.seek(-1,1);handle.write(bytes([value[0]^1]))
    with pytest.raises(ValueError,match='SHA256'):cli.main(argv+['--resume'])


def test_matrix_bundle_evidence_requires_compound_roundtrip_binding(tmp_path):
    from scripts.research_matrix import checkpoint_evidence,roundtrip_passed
    from src.utils.research_checkpoint_bundle import MAGIC
    manager=BestLastCheckpointManager(tmp_path/'matrix',DiskBudget(tmp_path,0))
    manager.save_torch({'model':{'weight':torch.ones(4)},'step':1,
                        'optimizer':{'state':{},'param_groups':[{'params':[0]}]}},
                       checkpoint_format='bundle',step=1,metric=.5,anticipated_bytes=1024**2)
    evidence=checkpoint_evidence(manager.root)
    checkpoint=evidence['checkpoints']['last']
    assert evidence['unique_retained_bytes']==checkpoint['artifact_total_bytes']
    assert checkpoint['artifact_total_bytes']>checkpoint['bytes']
    receipt={'status':'passed','checkpoint_path':checkpoint['path'],'checkpoint_owner':checkpoint['owner'],
             'checkpoint_step':checkpoint['step'],'checkpoint_size_bytes':checkpoint['bytes'],
             'checkpoint_mtime_ns':checkpoint['mtime_ns'],'validated_tensor_count':1,
             'all_tensors_finite':True,'expected_keys_match':True}
    report=manager.root/'checkpoint_roundtrip.json'
    report.write_text(json.dumps(receipt))
    assert not roundtrip_passed(manager.root,checkpoint)
    receipt.update(checkpoint_sha256=checkpoint['sha256'],checkpoint_bundle_identity=checkpoint['bundle_identity'])
    report.write_text(json.dumps(receipt))
    assert roundtrip_passed(manager.root,checkpoint)
    # The report remains readable, but no broken bundle can count as retained.
    (manager.root/'last/state.pt').write_bytes(MAGIC+b'invalid manifest')
    broken=checkpoint_evidence(manager.root)
    assert not broken['retained'] and broken['errors']


def test_matrix_direct_script_can_import_bundle_helpers_outside_repository(tmp_path):
    run=tmp_path/'runs/research/tiny__pilot'
    manager=BestLastCheckpointManager(run,DiskBudget(tmp_path,0))
    manager.save_torch({'model':{'weight':torch.ones(4)},'step':1,
                        'optimizer':{'state':{},'param_groups':[{'params':[0]}]}},
                       checkpoint_format='bundle',step=1,metric=.5,anticipated_bytes=1024**2)
    (run/'resolved_config.json').write_text(json.dumps({'output':str(run),'stage':'A0'}))
    script=Path(__file__).resolve().parents[1]/'scripts/research_matrix.py'
    output=tmp_path/'report'
    result=subprocess.run([sys.executable,str(script),'--root',str(tmp_path),'--output-dir',str(output)],
                          cwd=tmp_path,env=dict(os.environ,PYTHONPATH=''),capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['required_primary']==396
    report=json.loads((output/'matrix.json').read_text())
    assert report['observed_stages'][0]['checkpoint_evidence']['retained']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual allocator-aware CUDA restore required')
@pytest.mark.parametrize('optim_bits',[8,32])
def test_bundle_preserves_streamed_paged_optimizer_resume(tmp_path,optim_bits):
    """Tiny real UVM states plus pending CPU gradients resume the same update."""
    from src.utils.research_optim import CPUGradientOffloadOptimizer
    torch.manual_seed(782)
    parameter=torch.nn.Parameter(torch.randn(100352,device='cuda')*.01)
    original=CPUGradientOffloadOptimizer([parameter],lr=.001,optim_bits=optim_bits)
    restored=None
    try:
        parameter.square().sum().backward();original.step()
        assert original.state[parameter]['state1'].is_paged
        (parameter.square().sum()*.25).backward()
        manager=BestLastCheckpointManager(tmp_path/'paged',DiskBudget(tmp_path,0))
        manager.save_torch({'model':{'weight':parameter.detach().cpu()},'optimizer':original.state_dict(),
                            'step':1,'rng_torch':torch.get_rng_state()},
                           checkpoint_format='bundle',step=1,metric=.5,anticipated_bytes=8*1024**2)
        saved=load_research_payload(manager.root/'last')
        other=torch.nn.Parameter(saved['model']['weight'].to('cuda'))
        restored=CPUGradientOffloadOptimizer([other],lr=.7,optim_bits=optim_bits)
        restored.load_state_dict(saved['optimizer'])
        assert restored.state[other]['state1'].is_paged
        assert restored.pending_parameter_count==1
        assert restored.param_groups[0]['lr']==.001
        (parameter.square().sum()*.75).backward();(other.square().sum()*.75).backward()
        original.clip_grad_norm_(.5);restored.clip_grad_norm_(.5)
        original.step();restored.step()
        torch.testing.assert_close(parameter,other,atol=1e-7,rtol=1e-6)
        assert original.optimizer_steps==restored.optimizer_steps==2
    finally:
        original.close()
        if restored is not None:restored.close()
