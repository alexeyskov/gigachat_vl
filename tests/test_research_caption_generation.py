"""CPU fake native owner, real files/images, delayed gold and recovery proofs."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path
import sys
import types

from PIL import Image
import pytest

from src.eval import research_caption_generation as c


def write(path,value):
    Path(path).write_bytes(c.canonical(value)+b'\n')


@pytest.fixture
def example(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'RECORDS',3)
    bundle=tmp_path/'bundle';bundle.mkdir();records=[];refs=[]
    captions=['a red cat sits on the wooden mat','a blue dog runs through the green park','a yellow flower stands beside the white wall']
    for index,color in enumerate([(120,0,0),(0,80,20),(100,40,0)]):
        path=tmp_path/f'image{index}.png';image=Image.new('RGB',(6,4),color);image.save(path)
        sha=hashlib.sha256(b'RGB:6:4:'+image.tobytes()).hexdigest();identity='caption-'+sha
        records.append({'record_id':identity,'image_paths':[str(path)],'image_hashes':[sha],'question':c.QUESTION})
        refs.append({'record_id':identity,'image_hash':sha,'references':[captions[index]],'reference_records':[
            {'source_record_id':f'ref{index}','reference':captions[index],'source':{'repo':'fixture/captions','revision':'a'*40}}]})
    (bundle/'generation.jsonl').write_bytes(b''.join(c.canonical(row)+b'\n' for row in records))
    (bundle/'references.jsonl').write_bytes(b''.join(c.canonical(row)+b'\n' for row in refs))
    train=tmp_path/'TRAIN_GOLD_DO_NOT_OPEN.jsonl';train.write_text('TRAIN_GOLD_CANARY')
    train_sha=c.file_sha(train)
    audit={'status':'passed','pool_images':3,'audited_unselected_pool':True,'all_exclusion_pixels_verified':True,
        'exclusions':[{'overlap_images':0},{'overlap_images':0}]};write(bundle/'audit.json',audit)
    prep=tmp_path/'preparation_source.py';prep.write_text('# pinned fixture preparation')
    protocol={'status':'complete','records':3,'reference_records':3,'generation_references_separate':True,
        'options':{'question':c.QUESTION,'seed':42,'public_fields':sorted(c.PUBLIC),'source_manifests':[{'path':str(train),'sha256':train_sha}]},
        'files_sha256':{name:c.file_sha(bundle/name) for name in ('generation.jsonl','references.jsonl','audit.json')},
        'preparation_source_sha256':{str(prep):c.file_sha(prep)}}
    write(bundle/'protocol.json',protocol);monkeypatch.setattr(c,'DATA_PROTOCOL_SHA256',c.file_sha(bundle/'protocol.json'))
    run=tmp_path/'run';run.mkdir();(run/'.checkpoint.lock').touch();objects=run/'.checkpoint_objects';objects.mkdir()
    target=objects/'object-owned';target.mkdir();source=target/'state.pt';source.write_bytes(b'CPU_FAKE_CHECKPOINT_NO_MODEL')
    config={'stage':'SFT','recipe':'4q','output':str(run),'model':'fixture/model','vision':'fixture/vision',
        'manifest':str(train),'manifest_sha256':train_sha,'text_manifest':None}
    write(run/'resolved_config.json',config);write(run/'completed.json',{'steps':170})
    write(run/'.checkpoint_manager.json',{'owner':'fixture_owner'})
    write(run/'checkpoint_compaction.json',{'status':'completed','training_completed':True,'resumable_optimizer':False,
        'run_dir':str(run),'checkpoints':[{'new_object':target.name}]})
    write(target/'.research_checkpoint.json',{'owner':'fixture_owner','step':170,'created_unix':0.,
        'training_completed':True,'resumable_optimizer':False,'optimizer_state_removed':True})
    for role in ('best','last'):(run/role).symlink_to(Path('.checkpoint_objects')/target.name,target_is_directory=True)
    def identity(path):
        path=Path(path).resolve();stat=path.stat()
        return {'format':'legacy_single_file','sha256':c.file_sha(path),'members':[{'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}]}
    monkeypatch.setattr(c,'_checkpoint_identity',identity);monkeypatch.setattr(c,'gpu_idle',lambda:None)
    assets=tmp_path/'assets.json';write(assets,{'scope':'CPU fake scorer fixture'})
    native_dir=tmp_path/'pinned_snapshot';native_dir.mkdir();native=native_dir/'config.json';write(native,{'fixture':'pinned native model metadata'})
    context=c.prepare_context(run,bundle,assets=assets)
    return types.SimpleNamespace(root=tmp_path,run=run,bundle=bundle,records=records,references=refs,source=source,
        train=train,context=context,assets=assets,native=native,captions=captions,output=tmp_path/'output')


class FakeRuntime:
    def __init__(self,example,*,fail_after=None,callback=None):self.example=example;self.calls=[];self.fail_after=fail_after;self.callback=callback;self.loads=0
    def load(self,*args):
        self.loads+=1
        return object(),{'recipe':'4q','stage':'SFT','training_step':170,'cache_fingerprint':'f'*64,
            'model_ref':{'repo':'fixture/model','revision':'a'*40},'vision_ref':{'repo':'fixture/vision','revision':'b'*40}}
    def describe(self,model,provenance):
        return {'loaded':provenance,'context_limit':4096,'master_parameter_dtypes':['torch.float32'],
            'pinned_snapshots':{kind:{'snapshot':str(self.example.native.parent),'files':{'config.json':c._file(self.example.native)}} for kind in ('model','vision')}}
    def generate(self,model,record):
        assert set(record)==c.PUBLIC and record['question']==c.QUESTION
        if self.fail_after is not None and len(self.calls)>=self.fail_after:raise InterruptedError('fake interrupted generation')
        self.calls.append(record['record_id'])
        if self.callback:self.callback()
        index=[row['record_id'] for row in self.example.records].index(record['record_id'])
        return {'prediction':self.example.captions[index],'generated_token_ids':[40+index,2],'generated_tokens':2,
            'prefix_token_ids':[1,4],'suffix_token_ids':[9],'visual_tokens':3,'prefill_tokens':6,
            'prompt_sha256':'c'*64,'logits_to_keep':1}


def fake_scores(references,predictions,**kwargs):
    assert set(references)==set(predictions)
    values={key:10. if predictions[key] in references[key] else 0. for key in predictions}
    return {'images':len(values),'per_image':values,'cider_raw':sum(values.values())/len(values),'cider_times100':100*sum(values.values())/len(values)}


def interval(scores,**kwargs):
    assert kwargs=={'replicates':1000,'seed':42}
    mean=sum(scores.values())/len(scores)
    return {'mean':mean,'image_95_percentile_ci':[mean,mean],'replicates':1000,'seed':42,'idf_policy':'fixed full corpus'}


def evaluate(example,**kwargs):
    return c.evaluate(example.context,example.output,scorer=fake_scores,bootstrap=interval,**kwargs)


def test_public_preflight_does_not_open_gold_or_import_torch(example,monkeypatch):
    torch_preexisting='torch' in sys.modules
    original=Path.open
    def opened(path,*args,**kwargs):
        assert path not in (example.bundle/'references.jsonl',example.train),'Gold read during preflight'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)
    result=c.prepare_context(example.run,example.bundle,assets=example.assets)
    assert len(result['records'])==3 and all(set(row)==c.PUBLIC for row in result['records'])
    assert ('torch' in sys.modules)==torch_preexisting


def test_clean_import_is_cpu_only_without_torch():
    result=subprocess.run([sys.executable,'-c',"import sys; import src.eval.research_caption_generation; assert 'torch' not in sys.modules"],
        cwd=c.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_gold_reads_only_after_all_durable_rows_and_full_generation_receipt(example,monkeypatch):
    original=Path.open;gold_reads=[]
    def opened(path,*args,**kwargs):
        assert path!=example.train,'Training source gold must never be opened'
        if path==example.bundle/'references.jsonl':
            complete=json.loads((example.output/'generation_complete.json').read_text())
            assert complete['records']==3 and len(list((example.output/'records').glob('*.json')))==3
            gold_reads.append(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened);runtime=FakeRuntime(example)
    summary=evaluate(example,runtime=runtime,allow_gpu=True)
    assert len(runtime.calls)==3 and gold_reads
    assert summary['complete_manifest'] and summary['no_vqa_score_computed'] and summary['caption_scores']['cider_raw']==10.
    rows=c._rows(example.output/'predictions.jsonl')
    for row in rows:
        assert row['generated_token_ids_sha256']==c.digest(row['generated_token_ids'])
        assert row['prompt_token_ids_sha256']==c.digest({key:row[key] for key in ('prefix_token_ids','suffix_token_ids','visual_tokens','prefill_tokens')})
        assert not any(key in row for key in ('references','answers','score'))


def test_interrupted_prefix_resumes_without_duplicate_generation_or_early_gold(example,monkeypatch):
    original=Path.open
    def opened(path,*args,**kwargs):
        if path==example.bundle/'references.jsonl' and not (example.output/'generation_complete.json').exists():pytest.fail('early gold read')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)
    first=FakeRuntime(example,fail_after=1)
    with pytest.raises(InterruptedError):evaluate(example,runtime=first,allow_gpu=True)
    assert len(c._rows(example.output/'predictions.jsonl'))==1 and not (example.output/'summary.json').exists()
    second=FakeRuntime(example);summary=evaluate(example,runtime=second,allow_gpu=True,resume=True)
    assert second.calls==[row['record_id'] for row in example.records[1:]] and summary['records']==3


def test_truncated_lastline_rebuilt_from_committed_rows_and_preserved(example):
    first=FakeRuntime(example,fail_after=2)
    with pytest.raises(InterruptedError):evaluate(example,runtime=first,allow_gpu=True)
    path=example.output/'predictions.jsonl';raw=path.read_bytes();truncated=raw[:-19];path.write_bytes(truncated)
    runtime=FakeRuntime(example);evaluate(example,runtime=runtime,allow_gpu=True,resume=True)
    assert runtime.calls==[example.records[2]['record_id']]
    backups=list((example.output/'recovery').glob('*.jsonl.fragment'))
    assert len(backups)==1 and backups[0].read_bytes()==truncated
    assert len(c._rows(path))==3


def test_divergent_raw_tail_refused_preserving_every_byte(example):
    with pytest.raises(InterruptedError):evaluate(example,runtime=FakeRuntime(example,fail_after=1),allow_gpu=True)
    path=example.output/'predictions.jsonl';raw=path.read_bytes()+b'FOREIGN';path.write_bytes(raw)
    with pytest.raises(ValueError,match='diverge'):evaluate(example,runtime=FakeRuntime(example),allow_gpu=True,resume=True)
    assert path.read_bytes()==raw


def test_uncommitted_atomic_temp_preserved_never_treated_as_prediction(example):
    with pytest.raises(InterruptedError):evaluate(example,runtime=FakeRuntime(example,fail_after=1),allow_gpu=True)
    temporary=example.output/'records'/('.000001.json.'+'a'*32+'.tmp');temporary.write_bytes(b'{"interrupted_before_replace":')
    runtime=FakeRuntime(example);summary=evaluate(example,runtime=runtime,allow_gpu=True,resume=True)
    assert runtime.calls==[row['record_id'] for row in example.records[1:]] and summary['records']==3
    backups=list((example.output/'recovery').glob('*.uncommitted'))
    assert len(backups)==1 and backups[0].read_bytes()==b'{"interrupted_before_replace":' and not temporary.exists()


def test_real_atomic_write_interrupted_before_replace_preserves_committed_prefix(example,monkeypatch):
    original_replace=c.os.replace;original_unlink=Path.unlink
    def replace(source,target):
        if Path(target).name=='000001.json':raise InterruptedError('simulated process loss after fsync before row publication')
        return original_replace(source,target)
    def unlink(path,*args,**kwargs):
        if path.name.startswith('.000001.json.') and path.name.endswith('.tmp'):return None
        return original_unlink(path,*args,**kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(c.os,'replace',replace);scoped.setattr(Path,'unlink',unlink)
        with pytest.raises(InterruptedError):evaluate(example,runtime=FakeRuntime(example),allow_gpu=True)
    assert len(list((example.output/'records').glob('[0-9]*.json')))==1
    staged=list((example.output/'records').glob('.*.tmp'));assert len(staged)==1
    staged_bytes=staged[0].read_bytes();assert not (example.output/'generation_complete.json').exists()
    runtime=FakeRuntime(example);evaluate(example,runtime=runtime,allow_gpu=True,resume=True)
    assert runtime.calls==[row['record_id'] for row in example.records[1:]]
    assert any(path.read_bytes()==staged_bytes for path in (example.output/'recovery').glob('*.uncommitted'))


def test_unknown_or_symlink_journal_staging_is_never_cleaned(example):
    with pytest.raises(InterruptedError):evaluate(example,runtime=FakeRuntime(example,fail_after=1),allow_gpu=True)
    path=example.output/'records/foreign.tmp';path.write_bytes(b'foreign')
    with pytest.raises(ValueError,match='Unexpected'):evaluate(example,runtime=FakeRuntime(example),allow_gpu=True,resume=True)
    assert path.read_bytes()==b'foreign'


@pytest.mark.parametrize('name',['predictions.jsonl','records','recovery'])
def test_symlink_output_never_reads_foreign_or_gold(example,name):
    with pytest.raises(InterruptedError):evaluate(example,runtime=FakeRuntime(example,fail_after=1),allow_gpu=True)
    path=example.output/name
    if name=='records':
        for child in path.iterdir():child.unlink()
        path.rmdir();path.symlink_to(example.bundle,target_is_directory=True)
    elif name=='predictions.jsonl':
        path.unlink();path.symlink_to(example.bundle/'references.jsonl')
    else:
        path.symlink_to(example.bundle,target_is_directory=True)
        raw=example.output/'predictions.jsonl';raw.write_bytes(raw.read_bytes()[:-10])
    with pytest.raises(ValueError,match='symlink'):evaluate(example,runtime=FakeRuntime(example),allow_gpu=True,resume=True)
    assert not (example.output/'generation_complete.json').exists()


@pytest.mark.parametrize('mutation',['source','checkpoint','image','model_asset'])
def test_mutation_during_generation_refuses_before_row_commit_or_gold(example,mutation):
    path={'checkpoint':example.source,'image':Path(example.records[0]['image_paths'][0]),'model_asset':example.native}.get(mutation)
    if mutation=='source':
        path=example.root/'critical_source.py';path.write_text('version1')
        example.context['static']['source_files']['fixture_source']=c._file(path)
    def mutate():path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):evaluate(example,runtime=FakeRuntime(example,callback=mutate),allow_gpu=True)
    assert not list((example.output/'records').glob('[0-9]*.json'))
    assert not (example.output/'generation_complete.json').exists()


def test_checkpoint_role_mutation_refuses_resume_before_model_loading(example):
    with pytest.raises(InterruptedError):evaluate(example,runtime=FakeRuntime(example,fail_after=1),allow_gpu=True)
    (example.run/'best').unlink();(example.run/'best').symlink_to('.checkpoint_objects/foreign')
    runtime=FakeRuntime(example)
    with pytest.raises(ValueError,match='role changed'):evaluate(example,runtime=runtime,allow_gpu=True,resume=True)
    assert runtime.loads==0


def test_source_protocol_or_environment_mismatch_never_retries_model(example,monkeypatch):
    with pytest.raises(InterruptedError):evaluate(example,runtime=FakeRuntime(example,fail_after=1),allow_gpu=True)
    changed=copy.deepcopy(example.context);changed['static']['generation']['max_new_tokens']=64
    runtime=FakeRuntime(example)
    with pytest.raises(ValueError,match='protocol differs'):c.evaluate(changed,example.output,resume=True,allow_gpu=True,runtime=runtime)
    assert runtime.loads==0
    monkeypatch.setenv('OMP_NUM_THREADS','599')
    with pytest.raises(ValueError,match='environment changed'):evaluate(example,runtime=runtime,allow_gpu=True,resume=True)


def test_complete_cpu_score_only_resume_never_loads_model(example):
    first=evaluate(example,runtime=FakeRuntime(example),allow_gpu=True)
    runtime=FakeRuntime(example)
    second=evaluate(example,runtime=runtime,resume=True,score_only=True)
    assert second==first and runtime.loads==0


def test_incomplete_score_only_does_not_read_gold_or_load_model(example):
    with pytest.raises(InterruptedError):evaluate(example,runtime=FakeRuntime(example,fail_after=1),allow_gpu=True)
    runtime=FakeRuntime(example)
    with pytest.raises(ValueError,match='Score-only'):evaluate(example,runtime=runtime,resume=True,score_only=True)
    assert runtime.loads==0 and not (example.output/'generation_complete.json').exists()


def test_changed_reference_file_refuses_even_without_reading_content(example):
    references=example.bundle/'references.jsonl';references.write_bytes(references.read_bytes()+b' ')
    with pytest.raises(ValueError,match='Deferred reference'):evaluate(example,runtime=FakeRuntime(example),allow_gpu=True)


def test_row_token_hash_corruption_refuses_resume(example):
    with pytest.raises(InterruptedError):evaluate(example,runtime=FakeRuntime(example,fail_after=1),allow_gpu=True)
    path=example.output/'records/000000.json';row=json.loads(path.read_text());row['generated_token_ids'][0]+=1;write(path,row)
    with pytest.raises(ValueError,match='token/prompt/protocol'):evaluate(example,runtime=FakeRuntime(example),allow_gpu=True,resume=True)


def test_native_observer_exact_builder_once_and_restored_on_error(monkeypatch,example):
    fake=types.ModuleType('src.eval.research_inference');calls=[]
    def builder(model,record,**kwargs):
        calls.append((record,kwargs));return {'prefix_token_ids':[1],'suffix_token_ids':[2],'visual_tokens':2,'prefill_tokens':4,'prompt_sha256':'d'*64}
    fake.build_generation_inputs=builder
    def generate(model,record,**kwargs):
        prepared=fake.build_generation_inputs(model,record,**kwargs)
        return {key:value for key,value in prepared.items() if key not in ('prefix_token_ids','suffix_token_ids')}
    fake.generate_prediction=generate
    monkeypatch.setitem(sys.modules,fake.__name__,fake)
    import src.eval
    monkeypatch.setattr(src.eval,'research_inference',fake,raising=False)
    result=c.NativeRuntime().generate(object(),example.records[0])
    assert result['prefix_token_ids']==[1] and len(calls)==1 and calls[0][1]=={'max_new_tokens':256}
    assert fake.build_generation_inputs is builder
    def fail(*args,**kwargs):raise ValueError('fake native fail')
    fake.generate_prediction=fail
    with pytest.raises(ValueError,match='fake native fail'):c.NativeRuntime().generate(object(),example.records[0])
    assert fake.build_generation_inputs is builder


def test_reference_fields_cannot_be_passed_to_native_builder(example):
    contaminated={**example.records[0],'references':['GOLD']}
    with pytest.raises(ValueError,match='four public'):c.public_record(contaminated)


def test_occupied_gpu_refuses_before_model_or_gold(example,monkeypatch):
    runtime=FakeRuntime(example)
    def occupied():raise RuntimeError('GPU occupied')
    monkeypatch.setattr(c,'gpu_idle',occupied)
    with pytest.raises(RuntimeError,match='occupied'):evaluate(example,runtime=runtime,allow_gpu=True)
    assert runtime.loads==0 and not (example.output/'generation_complete.json').exists()


def test_caption_context_bound_refuses_no_truncation(example):
    class Overflow(FakeRuntime):
        def generate(self,*args):
            row=super().generate(*args);row['visual_tokens']=4000;row['prefill_tokens']=4003;return row
    with pytest.raises(ValueError,match='context contract'):evaluate(example,runtime=Overflow(example),allow_gpu=True)
    assert not list((example.output/'records').glob('[0-9]*.json'))


def test_date_dependent_native_template_drift_refused_within_run(example):
    class ClockChanged(FakeRuntime):
        def generate(self,*args):
            generated=super().generate(*args)
            if len(self.calls)>1:generated['prefix_token_ids']=[7,8];generated['prompt_sha256']='e'*64
            return generated
    with pytest.raises(ValueError,match='template clock drifted'):evaluate(example,runtime=ClockChanged(example),allow_gpu=True)
    assert len(list((example.output/'records').glob('[0-9]*.json')))==1
    assert not (example.output/'generation_complete.json').exists()


def test_template_clock_drift_after_resume_never_mixes_prompts(example):
    with pytest.raises(InterruptedError):evaluate(example,runtime=FakeRuntime(example,fail_after=1),allow_gpu=True)
    class Tomorrow(FakeRuntime):
        def generate(self,*args):
            generated=super().generate(*args);generated['prompt_sha256']='e'*64;return generated
    with pytest.raises(ValueError,match='template clock drifted'):evaluate(example,runtime=Tomorrow(example),allow_gpu=True,resume=True)
    assert len(c._rows(example.output/'predictions.jsonl'))==1


def test_prompt_contract_corruption_refused_before_resume_generation(example):
    with pytest.raises(InterruptedError):evaluate(example,runtime=FakeRuntime(example,fail_after=1),allow_gpu=True)
    path=example.output/'prompt_contract.json';value=json.loads(path.read_text());value['prefix_token_ids']=[99];write(path,value)
    runtime=FakeRuntime(example)
    with pytest.raises(ValueError,match='initial template contract'):evaluate(example,runtime=runtime,allow_gpu=True,resume=True)
    assert runtime.loads==0


def test_actual_reference_corpus_missing_image_never_produces_score(example):
    # Simulates a preparation fixture signed with an incomplete reference file;
    # production has one immutable fixed372 data protocol SHA.
    path=example.bundle/'references.jsonl';path.write_bytes(b''.join(c.canonical(row)+b'\n' for row in example.references[:2]))
    example.context['data']['references_deferred']={'path':str(path),'stat':c.stat(path),'sha256':c.file_sha(path)}
    example.context['static']['references_deferred']=example.context['data']['references_deferred']
    with pytest.raises(ValueError,match='exactly cover'):evaluate(example,runtime=FakeRuntime(example),allow_gpu=True)
    assert (example.output/'generation_complete.json').is_file() and not (example.output/'summary.json').exists()


def test_real_independent_full_corpus_cider_and_ci(example):
    from src.eval.research_caption_metrics import score_caption_corpus,image_bootstrap_interval
    assets=c.ROOT/'configs/research/caption_metric_assets.json'
    example.context=c.prepare_context(example.run,example.bundle,assets=assets)
    result=c.evaluate(example.context,example.output,allow_gpu=True,runtime=FakeRuntime(example),
        scorer=score_caption_corpus,bootstrap=image_bootstrap_interval)
    refs={row['record_id']:row['references'] for row in example.references}
    predictions={row['record_id']:value for row,value in zip(example.records,example.captions)}
    independent=score_caption_corpus(refs,predictions)
    assert result['caption_scores']==independent
    assert result['image_bootstrap']==image_bootstrap_interval(independent['per_image'],replicates=1000,seed=42)
    assert result['caption_scores']['cider_raw']==pytest.approx(10.)
