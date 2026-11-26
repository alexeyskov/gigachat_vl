"""Diagnostic captures preserve native generation; safety never signals peers."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import os

import pytest
import torch

import scripts.diagnose_research_batch4_logits as diag
from src.eval.research_inference import freeze_for_generation,generate_prediction
from src.eval.research_batch_inference import generate_batch_predictions
from tests.test_research_inference import TinyResearchModel


def records():
    return [{'record_id':str(i),'question':'question red '+('blue '*i),'image_paths':['fixture.png'],'image_hashes':['hash'],
             'answers':['GOLD_MUST_NOT_BE_USED']} for i in range(4)]


def test_real_tiny_cpu_first_step_capture_and_left_padding():
    torch.set_num_threads(2);torch.manual_seed(32)
    model=freeze_for_generation(TinyResearchModel(wrapped=True));rows=records()
    expected_native=[generate_prediction(model,diag.public(r),max_new_tokens=4)['generated_token_ids'] for r in rows]
    expected_batch=[x['generated_token_ids'] for x in generate_batch_predictions(model,[diag.public(r) for r in rows],max_new_tokens=4)]
    native=[diag.capture_generation(model,[r],batched=False,max_new_tokens=4) for r in rows]
    batch=diag.capture_generation(model,rows,batched=True,max_new_tokens=4)
    diag.require_reproduction([r['token_ids'][0] for r in native],batch['token_ids'],
                              {'expected_native_token_ids':expected_native,'expected_batch4_token_ids':expected_batch})
    assert batch['raw_logits'].shape==batch['output_scores'].shape==(4,32)
    assert batch['rows'][0]['left_padding_offset']>batch['rows'][-1]['left_padding_offset']==0
    assert all(r['first_real_position_id']==0 for r in batch['rows'])
    assert [r['rows'][0]['prompt_sha256'] for r in native]==[r['prompt_sha256'] for r in batch['rows']]
    assert all('GOLD' not in model.tokenizer.decode(x) for x in expected_native)
    result=diag.compare_logits(torch.cat([x['raw_logits'] for x in native]),batch['raw_logits'],model.tokenizer)
    assert len(result)==4 and all(len(r['native_top10'])==10 for r in result)


def test_exact_failed_production_batch_and_mismatch_offset():
    rows=[dict(records()[i%4],record_id=str(i)) for i in range(512)]
    selected=[str(i) for start in range(0,32,4) for i in range(start,start+4)]
    guard={'status':'failed','binding':{'record_ids':selected,'original_batch_start_indices':list(range(0,32,4))},
      'modes':{'none':{'status':'failed','mismatched_record_ids':['10'],'native_b1_token_ids':[[i] for i in range(32)],
                       'batch4_token_ids':[[i+1] for i in range(32)]}}}
    chunk,evidence=diag.failed_batch(guard,rows,'10')
    assert [r['record_id'] for r in chunk]==['8','9','10','11']
    assert evidence['original_batch_start']==8 and evidence['failed_row_offset']==2
    assert evidence['expected_native_token_ids']==[[8],[9],[10],[11]]
    changed=copy.deepcopy(guard);changed['binding']['record_ids'][8:12]=['11','10','9','8']
    with pytest.raises(ValueError,match='membership'):diag.failed_batch(changed,rows,'10')


def test_interpretation_refused_unless_full_token_sequences_reproduce():
    with pytest.raises(ValueError,match='no logit interpretation'):
        diag.require_reproduction([[1],[2]],[[1],[3]],{'expected_native_token_ids':[[1],[2]],'expected_batch4_token_ids':[[1],[4]]})


def test_logit_difference_top10_margins_and_changed_argmax():
    tokenizer=SimpleNamespace(decode=lambda ids,**kw:str(ids[0]))
    native=torch.tensor([[3.,2.,0.],[0.,2.,1.]])
    batch=torch.tensor([[2.,3.,0.],[0.,2.,1.]])
    result=diag.compare_logits(native,batch,tokenizer)
    assert result[0]['first_argmax_changed'] and not result[1]['first_argmax_changed']
    assert result[0]['max_abs_diff']==1 and abs(result[0]['rms_diff']-(2/3)**.5)<1e-10
    assert result[0]['native_top1_top2_margin']==result[0]['batch4_top1_top2_margin']==1
    assert result[0]['native_top10'][0]['token_id']==0 and result[0]['batch4_top10'][0]['token_id']==1


def allowed_runtime(tmp_path,*,completed=100,free_gib=40,own_gib=4):
    output=tmp_path/'native';output.mkdir()
    progress=output/'progress.json';progress.write_text(json.dumps({'event':'generated','selected_records':512,'manifest_records':512,'completed_records':completed}))
    argv=['python','-u',str(diag.ROOT/'scripts/eval_research_checkpoint.py'),'--run','run','--manifest','m','--output',str(output)]
    binding={'pid':123456,'start_time':998877,'argv':argv,'progress_path':str(progress)}
    path=tmp_path/'allow.json';path.write_text(json.dumps(binding))
    live={'pid':binding['pid'],'start_time':binding['start_time'],'argv':argv,'state':'S','cwd':str(diag.ROOT)}
    gpu={'free_bytes':free_gib*2**30,'total_bytes':96*2**30,'processes':{123456:4*2**30,os.getpid():own_gib*2**30}}
    guard=diag.RuntimeGuard(path,proc=lambda pid:live,gpu=lambda:gpu)
    return guard,live,gpu,progress,path


def test_exact_known_peer_and_safe_progress_allowed(tmp_path):
    guard,live,gpu,progress,path=allowed_runtime(tmp_path)
    assert guard.check()['free_bytes']==40*2**30
    assert guard.checks[-1]['peer_completed_records']==100
    live['start_time']+=1
    with pytest.raises(InterruptedError,match='same live process'):guard.check()


@pytest.mark.parametrize('fault',['progress','argv','free','memory','extra_gpu_pid','dead','binding_file'])
def test_known_eval_safety_failures_abort_own_work(tmp_path,fault):
    guard,live,gpu,progress,path=allowed_runtime(tmp_path)
    if fault=='progress':progress.write_text(json.dumps({'event':'generated','selected_records':512,'manifest_records':512,'completed_records':256}))
    elif fault=='argv':live['argv']=live['argv']+['--resume']
    elif fault=='free':gpu['free_bytes']=23*2**30
    elif fault=='memory':gpu['processes'][os.getpid()]=13*2**30
    elif fault=='extra_gpu_pid':gpu['processes'][555555]=1
    elif fault=='dead':live['state']='Z'
    else:path.write_text(path.read_text()+' ')
    with pytest.raises(InterruptedError):guard.check()


def test_exclusive_default_and_no_gpu_execution_without_flag(tmp_path):
    guard=diag.RuntimeGuard(gpu=lambda:{'free_bytes':50*2**30,'total_bytes':96*2**30,'processes':{123:1}})
    with pytest.raises(InterruptedError,match='Unexpected GPU task'):guard.check()
    with pytest.raises(RuntimeError,match='--allow-gpu'):
        diag.main(['--guard','missing','--manifest','missing','--output',str(tmp_path/'never.json')])
    with pytest.raises(ValueError):diag.RuntimeGuard(memory_cap_gib=13)
    with pytest.raises(ValueError):diag.RuntimeGuard(deadline_seconds=121)
