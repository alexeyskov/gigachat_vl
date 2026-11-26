"""Grounding lifecycle, strict provenance, corruption, retry and real tiny CLI."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import signal

import pytest
from PIL import Image

import scripts.research_queue as base
import scripts.research_grounding_queue as grounding
from src.dataset.vision_cache import image_content_hash
from src.eval.research_ablation import transform_image_ablation,write_transformed_manifest
from src.eval.research_metrics import aggregate_scores,score_prediction
from src.utils.research_checkpoint_compat import checkpoint_identity
from src.utils.research_storage import BestLastCheckpointManager,DiskBudget


def rowset(root,count=512):
    images=[]
    for i in range(4):
        image=root/f'image-{i}.png';Image.new('RGB',(9+i,7),(30*i,40,60)).save(image)
        images.append((str(image),image_content_hash(image)))
    groups=sorted(grounding.GROUPS)
    return [{'record_id':str(i),'question':'question red','answers':['red'],'scorer':'exact_match',
             'task_group':groups[i%4],'source':{'repo':'pinned-source','revision':'a'*40,'original_id':str(i)},
             'benchmark':'fixture','image_paths':[images[i%4][0]],'image_hashes':[images[i%4][1]]} for i in range(count)]


@pytest.fixture
def setup_queue(tmp_path,monkeypatch):
    monkeypatch.setattr(base,'ROOT',tmp_path);monkeypatch.setattr(grounding,'ROOT',tmp_path)
    manifest=tmp_path/'eval.jsonl';manifest.write_text(''.join(json.dumps(x)+'\n' for x in rowset(tmp_path)))
    selection=tmp_path/'internal_validation.jsonl'
    rows=rowset(tmp_path)
    for row in rows:
        row.update(answer='red',partition='internal_validation')
        row['source']['split']='train'
    selection.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    run=tmp_path/'sft'
    job={'id':'sft','argv':['--stage','SFT','--recipe','4q','--selection-metric','group_macro','--output',str(run)],
         'evaluation':{'manifest':str(manifest),'output':str(run/'evaluation/ordinary'),'checkpoint':'best','max_new_tokens':64}}
    spec=tmp_path/'queue.json';spec.write_text(json.dumps({'queue_id':'grounding-test','reserve_gb':0,'jobs':[job]}))
    calls=[]
    def process(command,**kwargs):
        calls.append(command)
        if Path(command[2]).name=='run_research_stage.py':
            manager=BestLastCheckpointManager(run,DiskBudget(run,0))
            manager.save(lambda d:(d/'state.pt').write_bytes(b'actual-test-artifact'),step=1,metric=.5,anticipated_bytes=100)
            (run/'completed.json').write_text(json.dumps({'steps':1}))
            (run/'resolved_config.json').write_text(json.dumps({'stage':'SFT','selection_metric':'group_macro','seed':42,
                'validation':str(selection),'selection_manifest_sha256':grounding.sha(selection)}))
        elif Path(command[2]).name=='validate_research_checkpoint.py':write_checkpoint_report(run)
        else:write_evaluation(command)
        return SimpleNamespace(pid=987654321,returncode=0,poll=lambda:0)
    monkeypatch.setattr(grounding.subprocess,'Popen',process)
    return SimpleNamespace(spec=spec,job=job,run=run,manifest=manifest,calls=calls,process=process)



def write_checkpoint_report(run):
    from scripts.validate_research_checkpoint import managed_checkpoints
    from src.eval.research_selection import PROTOCOL_NAME
    config=json.loads((run/'resolved_config.json').read_text());rows=[]
    selection=grounding.checkpoint_selection_binding(config)
    for entry in managed_checkpoints(run):
        path=entry['path'];identity=checkpoint_identity(path);stat=path.stat();metric=entry['marker']['metric']
        row={'schema_version':1,'status':'passed','checkpoint_owner':entry['owner'],'checkpoint_path':str(path),
             'checkpoint_labels':entry['labels'],'checkpoint_step':entry['marker']['step'],'checkpoint_size_bytes':stat.st_size,
             'checkpoint_mtime_ns':stat.st_mtime_ns,'all_tensors_finite':True,'expected_keys_match':True,'validated_tensor_count':1,
             'restored_tensor_max_abs_diff':0.,'projector_frozen':True,'projector_unchanged_during_validation':True,
             'selection_metric_replayed':True,'selection_protocol':PROTOCOL_NAME,'heldout_ce_replayed':False,
             'official_benchmark_score':False,'selection_manifest_sha256':selection['manifest_sha256'],
             'group_macro_score_saved':metric,'group_macro_score_replayed':metric,'group_macro_score_abs_diff':0.,
             'group_macro_score_tolerance':2e-3+2e-4*abs(metric),'validation_records':selection['records']}
        if identity['format']=='bundle_v1':row['checkpoint_bundle_identity']=identity
        rows.append(row)
    (run/'checkpoint_roundtrip.json').write_text(json.dumps({'schema_version':1,'status':'passed','mode':'full_gpu_replay',
                                                           'run':str(run),'checkpoints':rows,'validated_tensor_count':len(rows)}))


def write_evaluation(command):
    out=Path(base.argv_value(command,'--output'));out.mkdir(parents=True,exist_ok=True)
    original=Path(base.argv_value(command,'--manifest'));records=[json.loads(x) for x in original.read_text().splitlines()]
    variant=base.argv_value(command,'--image-ablation') or 'none'
    changed,descriptor=transform_image_ablation(records,variant=variant,seed=42,output_dir=out,
        source_manifest_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),reserve_bytes=0)
    if variant!='none':write_transformed_manifest(changed,out/'transformed_manifest.jsonl',reserve_bytes=0)
    identity=checkpoint_identity(Path(base.argv_value(command,'--run'))/'best/state.pt')
    protocol={'checkpoint_sha256':identity['sha256'],'manifest_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'manifest_records':len(records),'image_ablation':descriptor,
        'selected_record_ids_sha256':hashlib.sha256(json.dumps([r['record_id'] for r in changed]).encode()).hexdigest(),
        'generation':{'max_new_tokens':64,'do_sample':False,'use_cache':True,'num_beams':1},
        'source_sha256':{path:grounding.sha(grounding.CODE_ROOT/path) for path in grounding.REQUIRED_SOURCE_FILES}}
    if identity['format']=='bundle_v1':protocol['checkpoint_bundle_identity']=identity
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    rows=[]
    for record in changed:
        row={key:record[key] for key in grounding.ROW_FIELDS if key in record}
        row.update(prediction='red',score=score_prediction(record,'red'),generated_token_ids=[7],generated_tokens=1,protocol_sha256=digest,generation_seconds=.01)
        rows.append(row)
    summary=aggregate_scores(rows);summary.update(complete_manifest=True,manifest_records=len(rows),protocol_sha256=digest,
        checkpoint_sha256=identity['sha256'],manifest_sha256=protocol['manifest_sha256'],image_ablation=descriptor)
    if identity['format']=='bundle_v1':summary['checkpoint_bundle_identity']=identity
    (out/'protocol.json').write_text(json.dumps(protocol));(out/'summary.json').write_text(json.dumps(summary))
    (out/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))


def state(q):return json.loads(q.spec.with_suffix('.state.json').read_text())
def variant(command):return base.argv_value(command,'--image-ablation')


def test_runs_ordinary_then_three_variants_and_completed_restart_never_retrains(setup_queue):
    q=setup_queue
    assert grounding.GroundingResearchQueue(q.spec).run()==0
    assert [variant(c) for c in q.calls]==[None,None,None,'shuffle','blank','text_only']
    assert grounding.GroundingResearchQueue(q.spec).run()==0 and len(q.calls)==6
    result=state(q)['jobs']['sft']
    assert result['status']=='completed' and result['grounding_status']=='completed'
    assert set(result['grounding_evaluations'])==set(grounding.MODES)
    assert len({r['output'] for r in result['grounding_evaluations'].values()})==3
    assert all(r['records']==512 and r['seed']==42 for r in result['grounding_evaluations'].values())
    for command in q.calls[3:]:
        assert base.argv_value(command,'--manifest')==str(q.manifest)
        assert base.argv_value(command,'--checkpoint')=='best'
        assert base.argv_value(command,'--max-new-tokens')=='64'
        assert base.argv_value(command,'--ablation-seed')=='42'
    summary=json.loads(Path(result['grounding_summary']).read_text())
    assert set(summary['paired_image_group_bootstrap'])==set(grounding.MODES)
    assert all(r['paired_95_percentile_ci']==[0,0] for r in summary['paired_image_group_bootstrap'].values())


def test_none_guard_is_not_bypassed_by_ablation_outputs(setup_queue,monkeypatch):
    q=setup_queue
    def corrupt_none(command,**kwargs):
        p=q.process(command,**kwargs)
        if Path(command[2]).name=='eval_research_checkpoint.py':
            out=Path(base.argv_value(command,'--output'));summary=json.loads((out/'summary.json').read_text())
            summary['image_ablation']['variant']='shuffle';(out/'summary.json').write_text(json.dumps(summary))
        return p
    monkeypatch.setattr(grounding.subprocess,'Popen',corrupt_none)
    assert grounding.GroundingResearchQueue(q.spec).run()==1
    assert len(q.calls)==2 and state(q)['jobs']['sft']['status']=='training_completed_eval_failed'
    assert not state(q)['jobs']['sft'].get('grounding_evaluations')


def test_failed_grounding_needs_explicit_retry_and_never_retrains(setup_queue,monkeypatch):
    q=setup_queue
    def fail_blank(command,**kwargs):
        if variant(command)=='blank':
            q.calls.append(command);return SimpleNamespace(pid=987654321,returncode=9,poll=lambda:9)
        return q.process(command,**kwargs)
    monkeypatch.setattr(grounding.subprocess,'Popen',fail_blank)
    assert grounding.GroundingResearchQueue(q.spec).run()==1
    assert state(q)['jobs']['sft']['status']=='training_completed_grounding_failed'
    assert grounding.GroundingResearchQueue(q.spec).run()==1 and len(q.calls)==5
    monkeypatch.setattr(grounding.subprocess,'Popen',q.process)
    assert grounding.GroundingResearchQueue(q.spec,retry_failed=True).run()==0
    assert [variant(c) for c in q.calls[5:]]==['blank','text_only']
    assert sum(Path(c[2]).name=='run_research_stage.py' for c in q.calls)==1


def test_interrupted_partial_variant_resumes_only_remaining_evaluation(setup_queue,monkeypatch):
    q=setup_queue;controller=grounding.GroundingResearchQueue(q.spec)
    def interrupted(command,**kwargs):
        result=q.process(command,**kwargs)
        if variant(command)=='blank':
            out=Path(base.argv_value(command,'--output'))
            rows=(out/'predictions.jsonl').read_text().splitlines();(out/'predictions.jsonl').write_text('\n'.join(rows[:17])+'\n')
            (out/'summary.json').unlink();controller.interrupted_by=signal.SIGTERM
        return result
    monkeypatch.setattr(grounding.subprocess,'Popen',interrupted)
    assert controller.run()==128+signal.SIGTERM
    assert state(q)['jobs']['sft']['status']=='training_completed_grounding_interrupted'
    previous=len(q.calls);monkeypatch.setattr(grounding.subprocess,'Popen',q.process)
    assert grounding.GroundingResearchQueue(q.spec).run()==0
    assert [variant(c) for c in q.calls[previous:]]==['blank','text_only']
    assert '--resume' in q.calls[previous]


@pytest.mark.parametrize('field',['image_hashes','question','answers','source','scoring_protocol','score','protocol_sha256'])
def test_corrupt_raw_metadata_or_score_refused(setup_queue,field):
    q=setup_queue;controller=grounding.GroundingResearchQueue(q.spec)
    # Actual artifact generator runs through the same ordinary none validation.
    assert controller.run()==0
    controller=grounding.GroundingResearchQueue(q.spec)
    identity=controller._grounding_identity(q.job,'shuffle');path=identity['output']/'predictions.jsonl'
    rows=grounding.read_rows(path);rows[0][field]=.2 if field=='score' else ['wrong'] if field in ('image_hashes','answers') else 'wrong'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):controller._validate_grounding(q.job,identity,'shuffle')
    controller.lock_handle.close()


def test_manifest_protocol_checkpoint_corruption_and_fixed_modes_refused(setup_queue):
    q=setup_queue;controller=grounding.GroundingResearchQueue(q.spec);assert controller.run()==0
    controller=grounding.GroundingResearchQueue(q.spec);identity=controller._grounding_identity(q.job,'blank')
    target=identity['output']/'transformed_manifest.jsonl';original=target.read_bytes();target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='corrupted'):controller._validate_grounding(q.job,identity,'blank')
    target.write_bytes(original)
    protocol_path=identity['output']/'protocol.json';protocol=json.loads(protocol_path.read_text());protocol['image_ablation']['seed']=43
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='provenance'):controller._validate_grounding(q.job,identity,'blank')
    controller.lock_handle.close()
    spec=json.loads(q.spec.read_text());spec['jobs'][0]['evaluation']['checkpoint']='last';q.spec.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='spec changed'):grounding.GroundingResearchQueue(q.spec)


def test_heartbeat_rechecks_disk_and_signal_stops_owned_child(setup_queue,monkeypatch):
    q=setup_queue;controller=grounding.GroundingResearchQueue(q.spec);ticks=[0];checks=[]
    def monotonic():ticks[0]+=31;return ticks[0]
    monkeypatch.setattr(grounding.time,'monotonic',monotonic);monkeypatch.setattr(grounding.time,'sleep',lambda _:None)
    def delayed(command,**kwargs):
        result=q.process(command,**kwargs)
        if variant(command)=='shuffle':
            polls=iter([None,0]);result.poll=lambda:next(polls,0)
        return result
    monkeypatch.setattr(grounding.subprocess,'Popen',delayed)
    original_check=controller.budget.ensure_space
    def budget(n):checks.append(n);return original_check(n)
    monkeypatch.setattr(controller.budget,'ensure_space',budget)
    assert controller.run()==0 and 0 in checks
    killed=[];controller.process=SimpleNamespace(pid=7654321,poll=lambda:None)
    monkeypatch.setattr(grounding.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    controller.signal(signal.SIGINT,None)
    assert killed==[(7654321,signal.SIGTERM)] and controller.interrupted_by==signal.SIGINT


def test_full_bundle_identity_and_component_corruption_are_checked(setup_queue,monkeypatch):
    import os
    import torch
    from src.utils.research_checkpoint_bundle import save_checkpoint_bundle
    q=setup_queue
    def bundled(command,**kwargs):
        result=q.process(command,**kwargs)
        if Path(command[2]).name=='run_research_stage.py':
            payload=(q.run/'best/state.pt').resolve();payload.unlink()
            save_checkpoint_bundle({'model':{'weight':torch.arange(4.)},'step':1,'resumable_optimizer':False},payload.parent)
        return result
    monkeypatch.setattr(grounding.subprocess,'Popen',bundled)
    assert grounding.GroundingResearchQueue(q.spec).run()==0
    record=state(q)['jobs']['sft']
    assert all(row['checkpoint_artifact_identity']['format']=='bundle_v1' for row in record['grounding_evaluations'].values())
    component=(q.run/'best/state.pt').resolve().parent/'model.state.pt';before=component.stat()
    with component.open('r+b') as handle:
        handle.seek(-1,2);value=handle.read(1);handle.seek(-1,1);handle.write(bytes([value[0]^1]))
    os.utime(component,ns=(before.st_atime_ns,before.st_mtime_ns))
    previous=len(q.calls)
    assert grounding.GroundingResearchQueue(q.spec).run()==1
    assert len(q.calls)==previous


def test_live_owned_diagnostic_pid_prevents_duplicate_worker(setup_queue,monkeypatch):
    q=setup_queue;assert grounding.GroundingResearchQueue(q.spec).run()==0
    controller=grounding.GroundingResearchQueue(q.spec);record=state(q)['jobs']['sft']
    record['grounding_evaluations']['shuffle'].update(status='running',pid=123456789)
    target='/proc/123456789/cmdline';original_is_file=Path.is_file;original_read=Path.read_bytes
    monkeypatch.setattr(Path,'is_file',lambda p:True if str(p)==target else original_is_file(p))
    output=q.run/'evaluation/grounding_v1/shuffle'
    monkeypatch.setattr(Path,'read_bytes',lambda p:b'python\0eval_research_checkpoint.py\0--output\0'+str(output).encode() if str(p)==target else original_read(p))
    previous=len(q.calls)
    with pytest.raises(RuntimeError,match='still live'):controller._evaluate_variant(q.job,record,'shuffle')
    assert len(q.calls)==previous
    controller.lock_handle.close()


def test_signal_during_cpu_transform_launches_no_new_worker(setup_queue,monkeypatch):
    q=setup_queue;assert grounding.GroundingResearchQueue(q.spec).run()==0
    controller=grounding.GroundingResearchQueue(q.spec)
    original=controller._grounding_identity
    def signal_while_preparing(job,variant):
        identity=original(job,variant);controller.interrupted_by=signal.SIGTERM
        return identity
    monkeypatch.setattr(controller,'_grounding_identity',signal_while_preparing)
    previous=len(q.calls)
    assert controller.run()==128+signal.SIGTERM
    assert len(q.calls)==previous
    assert state(q)['jobs']['sft']['status']=='training_completed_grounding_interrupted'


def test_real_tiny_cpu_cli_resume_uses_only_pending_predictions(tmp_path,monkeypatch):
    import scripts.eval_research_checkpoint as cli
    from tests.test_research_inference import TinyResearchModel
    from src.eval.research_inference import freeze_for_generation
    import torch
    torch.set_num_threads(2)
    rows=rowset(tmp_path,4);manifest=tmp_path/'tiny.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    run=tmp_path/'tiny-run';(run/'best').mkdir(parents=True);(run/'best/state.pt').write_bytes(b'tiny-fixture')
    monkeypatch.setattr(cli,'assert_gpu_idle',lambda gpu:None)
    loads=[];initial=freeze_for_generation(TinyResearchModel());initial.args.max_length=128;initial.base.config.max_position_embeddings=128
    def load(*args,**kwargs):
        loads.append(1);return copy.deepcopy(initial),{'test_actual_native_CPU_model':True}
    monkeypatch.setattr(cli,'load_research_checkpoint',load)
    for mode in grounding.MODES:
        output=tmp_path/mode;argv=['--run',str(run),'--manifest',str(manifest),'--output',str(output),'--max-new-tokens','64','--image-ablation',mode,'--reserve-gb','0']
        # Preserve the actual full-selection protocol, then emulate interruption
        # after two complete fsynced prediction appends. No protocol is forged.
        first=cli.main(argv);assert first['complete_manifest']
        predictions=grounding.read_rows(output/'predictions.jsonl')[:2]
        (output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        (output/'summary.json').unlink()
        complete=cli.main(argv+['--resume']);assert complete['complete_manifest'] and complete['records']==4
        before=len(loads);assert cli.main(argv+['--resume'])==complete and len(loads)==before
    assert len(loads)==6


def test_checkpoint_gate_failure_requires_explicit_retry_without_retraining(setup_queue,monkeypatch):
    q=setup_queue
    def fail_validator(command,**kwargs):
        if Path(command[2]).name=='validate_research_checkpoint.py':
            q.calls.append(command);return SimpleNamespace(pid=987654321,returncode=4,poll=lambda:4)
        return q.process(command,**kwargs)
    monkeypatch.setattr(grounding.subprocess,'Popen',fail_validator)
    assert grounding.GroundingResearchQueue(q.spec).run()==1
    record=state(q)['jobs']['sft']
    assert record['status']=='training_completed_checkpoint_validation_failed'
    assert record['evaluation']['status']=='completed' and not record.get('grounding_evaluations')
    before=(q.run/'evaluation/ordinary/predictions.jsonl').read_bytes()
    assert grounding.GroundingResearchQueue(q.spec).run()==1 and len(q.calls)==3
    monkeypatch.setattr(grounding.subprocess,'Popen',q.process)
    assert grounding.GroundingResearchQueue(q.spec,retry_failed=True).run()==0
    assert [Path(c[2]).name for c in q.calls[3:]]==['validate_research_checkpoint.py']+['eval_research_checkpoint.py']*3
    assert state(q)['jobs']['sft']['checkpoint_validation']['mode']=='full_gpu_replay'
    assert (q.run/'evaluation/ordinary/predictions.jsonl').read_bytes()==before
    assert sum(Path(c[2]).name=='run_research_stage.py' for c in q.calls)==1


def test_interrupted_checkpoint_gate_resumes_validation_only(setup_queue,monkeypatch):
    q=setup_queue;controller=grounding.GroundingResearchQueue(q.spec)
    def interrupted(command,**kwargs):
        p=q.process(command,**kwargs)
        if Path(command[2]).name=='validate_research_checkpoint.py':
            (q.run/'checkpoint_roundtrip.json').unlink();controller.interrupted_by=signal.SIGTERM
            p.returncode=143
        return p
    monkeypatch.setattr(grounding.subprocess,'Popen',interrupted)
    assert controller.run()==128+signal.SIGTERM
    assert state(q)['jobs']['sft']['status']=='training_completed_checkpoint_validation_interrupted'
    previous=len(q.calls);monkeypatch.setattr(grounding.subprocess,'Popen',q.process)
    assert grounding.GroundingResearchQueue(q.spec).run()==0
    assert Path(q.calls[previous][2]).name=='validate_research_checkpoint.py'
    assert sum(Path(c[2]).name=='run_research_stage.py' for c in q.calls)==1


@pytest.mark.parametrize('field,value',[
    ('checkpoint_owner','foreign'),('checkpoint_step',99),('checkpoint_size_bytes',0),('checkpoint_mtime_ns',0),
    ('restored_tensor_max_abs_diff',1.),('projector_frozen',False),('projector_unchanged_during_validation',False),
    ('selection_metric_replayed',False),('selection_protocol','heldout_ce_smoke'),('validation_records',0),
    ('group_macro_score_saved',99.),('group_macro_score_tolerance',999.),('all_tensors_finite',False)])
def test_checkpoint_report_corrupt_replay_or_object_proof_refused(setup_queue,field,value):
    q=setup_queue
    q.process(['python','-u','run_research_stage.py',*q.job['argv']]);write_checkpoint_report(q.run)
    controller=grounding.GroundingResearchQueue(q.spec);identity=controller._checkpoint_validation_identity(q.job)
    assert controller._validate_checkpoint_report(identity)['mode']=='full_gpu_replay'
    report_path=q.run/'checkpoint_roundtrip.json';report=json.loads(report_path.read_text())
    report['checkpoints'][0][field]=value;report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):controller._validate_checkpoint_report(identity)
    controller.lock_handle.close()


def test_checkpoint_gate_rejects_cpu_mode_and_requires_both_retained_objects(setup_queue):
    q=setup_queue;q.process(['python','-u','run_research_stage.py',*q.job['argv']])
    manager=BestLastCheckpointManager(q.run,DiskBudget(q.run,0));manager.save(lambda d:(d/'state.pt').write_bytes(b'last-two'),step=2,metric=.6,anticipated_bytes=100)
    (q.run/'completed.json').write_text(json.dumps({'steps':2}));write_checkpoint_report(q.run)
    controller=grounding.GroundingResearchQueue(q.spec);identity=controller._checkpoint_validation_identity(q.job)
    assert len(identity['entries'])==2 and controller._validate_checkpoint_report(identity)['validated_tensor_count']==2
    path=q.run/'checkpoint_roundtrip.json';report=json.loads(path.read_text());report['mode']='cpu_native_full_reconstruction';path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='full_gpu_replay'):controller._validate_checkpoint_report(identity)
    report['mode']='full_gpu_replay';report['checkpoints']=report['checkpoints'][:1];path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='every current unique'):controller._validate_checkpoint_report(identity)
    controller.lock_handle.close()


def test_checkpoint_selection_binds_sampled_records_separately_from_source_file(setup_queue):
    q=setup_queue;q.process(['python','-u','run_research_stage.py',*q.job['argv']]);write_checkpoint_report(q.run)
    controller=grounding.GroundingResearchQueue(q.spec)
    try:
        identity=controller._checkpoint_validation_identity(q.job)
        source_sha=identity['config']['selection_manifest_sha256']
        selected_sha=identity['selection_binding']['manifest_sha256']
        assert source_sha!=selected_sha
        receipt=controller._validate_checkpoint_report(identity)
        assert receipt['selection_source_file_sha256']==source_sha
        assert receipt['selected_manifest_sha256']==selected_sha
        path=q.run/'checkpoint_roundtrip.json';original=json.loads(path.read_text())
        report=json.loads(json.dumps(original));report['checkpoints'][0]['selection_manifest_sha256']=source_sha
        path.write_text(json.dumps(report))
        with pytest.raises(ValueError):controller._validate_checkpoint_report(identity)
        report=json.loads(json.dumps(original));report['checkpoints'][0]['validation_records']-=1
        path.write_text(json.dumps(report))
        with pytest.raises(ValueError):controller._validate_checkpoint_report(identity)
    finally:controller.lock_handle.close()
