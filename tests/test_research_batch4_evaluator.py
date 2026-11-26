"""Isolated B4 policy, all-mode parity gates, atomic resume and native Tiny CPU."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import copy
import hashlib
import json

import pytest
from PIL import Image

import scripts.eval_research_checkpoint_batch4 as cli
from src.dataset.vision_cache import image_content_hash


def inputs(root):
    pictures=[]
    for i in range(4):
        path=root/f'image-{i}.png';Image.new('RGB',(9+i,7+i),(30*i,60,100)).save(path)
        pictures.append((str(path),image_content_hash(path)))
    records=[]
    for bench,group in cli.BENCHMARK_GROUPS.items():
        for i in range(64):
            path,sha=pictures[i%4];row={'record_id':f'{bench}-{i}','benchmark':bench,'task_group':group,
                'question':'question red '+('word '*(i%7)),'answers':['red'],'scorer':'exact_match',
                'source':{'repo':'fixture','revision':'a'*40,'split':'validation','original_id':str(i)},
                'image_paths':[path],'image_hashes':[sha]}
            if bench in ('vqav2','textvqa'):
                row.update(scorer='vqa_accuracy',answers=['red']*10,scoring_protocol='vqav2_official_legacy' if bench=='vqav2' else 'textvqa_evalai')
            elif bench=='docvqa':row['scorer']='anls'
            elif bench=='chartqa':row['scorer']='chart_relaxed'
            elif bench in ('ai2d','scienceqa','aokvqa'):
                row.update(scorer='mcq_accuracy',choices=['red','blue'],answers=['A'])
                row['question']+='\nA. red\nB. blue\nAnswer with the letter of the correct option only.'
            records.append(row)
    manifest=root/'eval.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    run=root/'run';(run/'best').mkdir(parents=True);(run/'best/state.pt').write_bytes(b'test-checkpoint-snapshot')
    return SimpleNamespace(run=run,manifest=manifest,records=records)


def prediction(row):
    assert set(row)=={'record_id','question','image_paths','image_hashes'}
    return {'prediction':'red','generated_token_ids':[7,2],'generated_tokens':2,
        'prompt_sha256':hashlib.sha256(row['question'].encode()).hexdigest(),
        'prefill_tokens':5,'visual_tokens':2 if row['image_paths'] else 0,'logits_to_keep':1}


@pytest.fixture
def fake(tmp_path,monkeypatch):
    f=inputs(tmp_path);f.loads=[];f.native=[];f.batch=[];f.fail_at=None;f.bad_gate=False
    monkeypatch.setattr(cli,'require_idle_gpu',lambda:None)
    def load(*args,**kwargs):
        f.loads.append(1);return SimpleNamespace(base=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2))),{'test_fixture':True}
    def native(model,row,**kwargs):f.native.append(row['record_id']);return prediction(row)
    def batch(model,rows,**kwargs):
        f.batch.append([r['record_id'] for r in rows])
        if f.fail_at==len(f.batch):raise InterruptedError('Simulated interrupted uncommitted batch')
        result=[dict(prediction(row),batch_size=4,padding_side='left',padded_prefill_tokens=5,batch_generation_schema_version=1) for row in rows]
        if f.bad_gate:result[0]['generated_token_ids']=[8,2]
        return result
    monkeypatch.setattr(cli,'load_research_checkpoint',load);monkeypatch.setattr(cli,'generate_prediction',native)
    monkeypatch.setattr(cli,'generate_batch_predictions',batch)
    f.argv=['--run',str(f.run),'--manifest',str(f.manifest),'--reserve-gb','0']
    return f


def test_fixed_production_chunks_select_all_eight_benchmarks(fake):
    indices=cli.guard_indices(fake.records)
    assert len(indices)==32 and Counter(fake.records[i]['benchmark'] for i in indices)==Counter({b:4 for b in cli.BENCHMARK_GROUPS})
    for offset in range(0,32,4):
        chunk=indices[offset:offset+4]
        assert chunk==list(range(chunk[0],chunk[0]+4)) and chunk[0]%4==0
    assert cli.PROFILE['batch_size']==4 and cli.PROFILE['max_new_tokens']==64 and cli.PROFILE['seed']==42


def test_all_four_modes_share_guard_and_completed_resume_loads_no_model(fake):
    for mode in cli.MODES:
        summary=cli.main(fake.argv+['--image-ablation',mode])
        assert summary['records']==512 and summary['complete_four_group_coverage']
        assert summary['profile']['name']=='batch4_v1' and summary['image_ablation']['variant']==mode
    assert len(fake.native)==128 and len(fake.loads)==4
    guard=json.loads((fake.run/'evaluation/batch4_v1/parity_guard.json').read_text())
    assert guard['status']=='passed' and set(guard['modes'])==set(cli.MODES)
    out=fake.run/'evaluation/batch4_v1/none'
    before={n:(out/n).read_bytes() for n in ['protocol.json','predictions.jsonl','summary.json']};counts=len(fake.batch),len(fake.loads)
    result=cli.main(fake.argv+['--resume'])
    assert result['records']==512 and counts==(len(fake.batch),len(fake.loads))
    assert all((out/n).read_bytes()==raw for n,raw in before.items())


def test_atomic_resume_keeps_original_four_row_membership(fake):
    fake.fail_at=35 #32 guard batches, then two committed scoring batches.
    with pytest.raises(InterruptedError):cli.main(fake.argv)
    output=fake.run/'evaluation/batch4_v1/none';files=sorted((output/'prediction_batches').glob('*.jsonl'))
    assert len(files)==2 and not (output/'summary.json').exists()
    before={p.name:p.read_bytes() for p in files};fake.fail_at=None;previous=len(fake.batch)
    result=cli.main(fake.argv+['--resume'])
    assert result['records']==512 and len(fake.native)==128
    assert fake.batch[previous]==[r['record_id'] for r in fake.records[8:12]]
    assert len(fake.batch)-previous==126
    assert all((output/'prediction_batches'/name).read_bytes()==raw for name,raw in before.items())


def test_failed_gate_never_scores_or_silently_retries(fake):
    fake.bad_gate=True
    with pytest.raises(ValueError,match='parity failed'):cli.main(fake.argv)
    base=fake.run/'evaluation/batch4_v1';assert json.loads((base/'parity_guard.json').read_text())['status']=='failed'
    assert not (base/'none/protocol.json').exists() and not (base/'none/prediction_batches').exists()
    before=len(fake.loads)
    with pytest.raises(ValueError,match='Failed parity guard'):cli.main(fake.argv+['--resume'])
    assert len(fake.loads)==before


@pytest.mark.parametrize('change',['checkpoint','profile','source'])
def test_guard_reuse_rejects_changed_bindings_before_model_load(fake,monkeypatch,change):
    cli.main(fake.argv);loads=len(fake.loads)
    if change=='checkpoint':(fake.run/'best/state.pt').write_bytes(b'changed-checkpoint')
    elif change=='profile':monkeypatch.setattr(cli,'PROFILE',dict(cli.PROFILE,batch_size=8))
    else:
        original=cli.source_identity();monkeypatch.setattr(cli,'source_identity',lambda:dict(original,unrelated='changed'))
    with pytest.raises(ValueError,match='binding differs'):cli.main(fake.argv+['--resume'])
    assert len(fake.loads)==loads


@pytest.mark.parametrize('complete',[False,True])
def test_corrupt_committed_batch_refused_without_regeneration(fake,complete):
    if complete:cli.main(fake.argv)
    else:
        fake.fail_at=35
        with pytest.raises(InterruptedError):cli.main(fake.argv)
        fake.fail_at=None
    path=fake.run/'evaluation/batch4_v1/none/prediction_batches/batch-0000.jsonl'
    rows=cli.read_rows(path);rows[0]['generated_token_ids']=[9,2]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows));loads=len(fake.loads)
    with pytest.raises(ValueError,match='checksum differs'):cli.main(fake.argv+['--resume'])
    assert len(fake.loads)==loads


def test_old_namespace_and_non_common_protocol_are_refused(fake):
    with pytest.raises(ValueError,match='separate'):cli.main(fake.argv+['--output',str(fake.run/'evaluation/heldout_v3')])
    with pytest.raises(ValueError,match='requires64'):cli.main(fake.argv+['--max-new-tokens','32'])
    with pytest.raises(ValueError,match='seed42'):cli.main(fake.argv+['--ablation-seed','43'])


def test_optional_full_native_comparison_preserves_original_cache(fake):
    cli.main(fake.argv);out=fake.run/'evaluation/batch4_v1/none';records=fake.records
    baseline=fake.run/'evaluation/native_fixture';baseline.mkdir()
    protocol={'checkpoint_sha256':json.loads((out/'protocol.json').read_text())['checkpoint_sha256'],
      'manifest_sha256':cli.sha(fake.manifest),'image_ablation':{'variant':'none'},
      'generation':{'max_new_tokens':64,'do_sample':False,'use_cache':True,'num_beams':1},
      'source_sha256':{p:cli.sha(cli.ROOT/p) for p in cli.FROZEN_SOURCES}}
    parent_sha=cli.protocol_digest(protocol);rows=[]
    for row in cli.read_rows(out/'predictions.jsonl'):
        row['protocol_sha256']=parent_sha;rows.append(row)
    (baseline/'protocol.json').write_text(json.dumps(protocol));(baseline/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary=cli.aggregate_scores(rows);summary.update(complete_manifest=True,protocol_sha256=parent_sha)
    (baseline/'summary.json').write_text(json.dumps(summary))
    before={p.name:p.read_bytes() for p in baseline.iterdir()}
    cli.main(fake.argv+['--resume','--compare-native-cache',str(baseline)])
    report=json.loads((out/'native_b1_comparison.json').read_text());assert report['exact_generated_token_ids'] and report['compared_records']==512
    assert all((baseline/name).read_bytes()==raw for name,raw in before.items())
    rows[0]['generated_token_ids']=[8,2];(baseline/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='comparison failed'):cli.main(fake.argv+['--resume','--compare-native-cache',str(baseline)])


def test_namespace_symlinks_cannot_alias_original_outputs(fake):
    old=fake.run/'evaluation/heldout_v3';old.mkdir(parents=True);(old/'predictions.jsonl').write_bytes(b'old scientific evidence')
    (fake.run/'evaluation/batch4_v1').symlink_to(old,target_is_directory=True)
    with pytest.raises(ValueError,match='symlinks'):cli.main(fake.argv)
    assert (old/'predictions.jsonl').read_bytes()==b'old scientific evidence' and not fake.loads


def test_actual_native_tiny_cpu_all_mode_guard_and_batch_journal(tmp_path,monkeypatch):
    import torch
    from tests.test_research_inference import TinyResearchModel
    from src.eval.research_inference import freeze_for_generation
    torch.set_num_threads(2);f=inputs(tmp_path);loads=[]
    monkeypatch.setattr(cli,'require_idle_gpu',lambda:None)
    def load(*args,**kwargs):
        loads.append(1);model=freeze_for_generation(TinyResearchModel(wrapped=True));model.args.max_length=256;model.base.config.max_position_embeddings=256
        with torch.no_grad():model.base.lm_head.weight.zero_()
        model.base.generation_config.eos_token_id=0
        return model,{'actual_one_layer_cpu_native_smallthinker':True}
    monkeypatch.setattr(cli,'load_research_checkpoint',load)
    args=['--run',str(f.run),'--manifest',str(f.manifest),'--reserve-gb','0']
    result=cli.main(args)
    assert result['records']==512 and len(result['journal_shards'])==128
    guard=json.loads((f.run/'evaluation/batch4_v1/parity_guard.json').read_text())
    assert all(r['native_b1_token_ids']==r['batch4_token_ids'] for r in guard['modes'].values())
    assert guard['eos_token_ids']==[0] and all(len(r['generated_token_ids'])==1 for r in cli.read_rows(f.run/'evaluation/batch4_v1/none/predictions.jsonl'))
    cli.main(args+['--resume']);assert len(loads)==1
