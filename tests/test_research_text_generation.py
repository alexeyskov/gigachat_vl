"""CPU-only retention producer contracts; no actual backbone/GPU generation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('text_generation_test_owner',ROOT/'src/eval/research_text_generation.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def row(task):
    return {'record_id':g.digest(task),'task':task,'source':{'repo':'fixture','revision':'a'*40,'original_id':'key:1'},
        'input_text_sha256':g.digest('public question'),'prompt':'A public question','subject':'algebra' if task=='mmlu' else None,
        'max_new_tokens':{'gsm8k':1024,'mmlu':0,'ifeval':2048}[task]}


def census(record):
    return {'record_id':record['record_id'],'task':record['task'],'prompt_sha256':g.digest('native prompt'),
        'prompt_token_ids_sha256':g.digest([4,5]),'prompt_tokens':2,
        'candidate_continuation_token_lengths':dict.fromkeys('ABCD',1)}


@pytest.fixture
def setup(tmp_path,monkeypatch):
    records=[row(t) for t in g.TASKS];cs={r['record_id']:census(r) for r in records}
    bundle=tmp_path/'public';bundle.mkdir()
    files={'public_records.jsonl':b''.join((g.metrics.canonical(r)+'\n').encode() for r in records),
        'tokenizer_census.json':b'{}\n'}
    protocol={'parent_protocol_sha256':g.metrics.SUITE_PROTOCOL_SHA256,'parent_manifest_sha256':'e'*64,
        'parent_generation_policy':{'gsm8k':{'do_sample':False,'max_new_tokens':1024},
            'ifeval':{'do_sample':False,'max_new_tokens':2048}}}
    for name,raw in files.items():(bundle/name).write_bytes(raw)
    (bundle/'protocol.json').write_text(g.metrics.canonical(protocol)+'\n')
    context={'bundle':str(bundle),'protocol':protocol,'protocol_sha256':g.sha(bundle/'protocol.json'),
        'public_file_sha256':g.sha(bundle/'public_records.jsonl'),'census_file_sha256':g.sha(bundle/'tokenizer_census.json'),
        'model':{'model_id':'smallthinker4','repo':'fixture/model','revision':'f'*40},'records':records,'census':cs,'tasks':list(g.TASKS)}
    plan={'kind':'pinned_baseline','repo':'fixture/model','revision':'f'*40,'precision':'nf4','missing_weights':[]}
    monkeypatch.setattr(g,'source_files',lambda:{'fixture':'a'*64})
    monkeypatch.setattr(g.shutil,'disk_usage',lambda p:SimpleNamespace(free=10**18))
    return SimpleNamespace(context=context,plan=plan,output=tmp_path/'generated',records=records)


class FakeRuntime:
    def __init__(self,fail_after=None):self.calls=[];self.loads=0;self.fail_after=fail_after
    def load_baseline(self,plan):
        self.loads+=1
        return object(),{'model_ref':{'repo':plan['repo'],'revision':plan['revision']},
            'checkpoint_identity':{'format':'synthetic_fixture_not_model_evidence'}}
    def describe(self,model,loaded,plan):return {'loaded':loaded,'snapshot_assets':{},'fixture':'no model/GPU'}
    def _base(self,r,c):
        if self.fail_after is not None and len(self.calls)>=self.fail_after:raise InterruptedError('synthetic interruption')
        self.calls.append(r['record_id'])
        return {k:c[k] for k in ('prompt_sha256','prompt_token_ids_sha256','prompt_tokens')}
    def generate(self,model,r,c):return {**self._base(r,c),'response':'  raw whitespace\n','generated_token_ids':[9,2],'generated_tokens':2}
    def candidate_likelihoods(self,model,r,c,model_id):
        return {**self._base(r,c),'candidate_log_likelihoods':dict.fromkeys('ABCD',-1.),
            'candidate_token_ids':{x:[g.metrics.TOKEN_IDS[model_id][x]] for x in 'ABCD'},
            'candidate_prefix_token_ids_sha256':dict.fromkeys('ABCD',c['prompt_token_ids_sha256']),'length_normalization':False}


def test_public_model_fields_reject_gold_and_fake_images():
    r=row('ifeval');assert g.model_record(r)=={'question':r['prompt'],'image_paths':[],'image_hashes':[]}
    for field in ('reference','answer','kwargs','instruction_id_list'):
        with pytest.raises(ValueError):g.model_record(dict(r,**{field:'secret'}))
    with pytest.raises(ValueError):g.model_record(dict(r,prompt='[RESEARCH_IMAGE]'))


def test_public_projection_is_independent_of_all_gold_annotations():
    original=dict(row('ifeval'),input_text='Original public text',reference={'answer':'secret'},kwargs={'hidden':True})
    changed=deepcopy(original);changed.update(reference={'answer':'another'},kwargs={'private':'changed'},answer='do not expose')
    assert g.public_projection(original)==g.public_projection(changed)
    assert set(g.public_projection(original))==g.PUBLIC_FIELDS


def test_actual_public_bundle_has_no_gold_and_reads_no_original_gold(monkeypatch):
    original=Path.open
    def trapped(path,*args,**kwargs):
        assert not Path(path).resolve().is_relative_to(ROOT/'datasets/research/primary-text-en-v1')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',trapped)
    for model_id in g.metrics.TOKEN_IDS:
        context=g.public_context(ROOT/'datasets/research/primary-text-public-en-v1',model_id)
        assert len(context['records'])==1680
        assert all(set(r)==g.PUBLIC_FIELDS for r in context['records'])
        assert all(not set(r)&{'reference','kwargs','answer','instruction_id_list'} for r in context['records'])


def test_actual_public_derivative_idempotence_preserves_bytes_and_metadata():
    output=ROOT/'datasets/research/primary-text-public-en-v1'
    before={p.name:(p.read_bytes(),g.stat(p)) for p in output.iterdir()}
    result=g.prepare_public_bundle(ROOT/'datasets/research/primary-text-en-v1',output)
    assert result==g.read(output/'protocol.json')
    assert before=={p.name:(p.read_bytes(),g.stat(p)) for p in output.iterdir()}


def test_public_manifest_tampering_cannot_self_retag_its_checksum(tmp_path):
    import shutil
    output=tmp_path/'changed';shutil.copytree(ROOT/'datasets/research/primary-text-public-en-v1',output)
    rows=g.metrics.read_jsonl(output/'public_records.jsonl');rows[0]['source']['original_id']='foreign'
    (output/'public_records.jsonl').write_bytes(b''.join((g.metrics.canonical(r)+'\n').encode() for r in rows))
    protocol=g.read(output/'protocol.json');protocol['files_sha256']['public_records.jsonl']=g.sha(output/'public_records.jsonl')
    (output/'protocol.json').write_text(g.metrics.canonical(protocol)+'\n')
    with pytest.raises(ValueError,match='frozen released-row projection'):g.public_context(output,'smallthinker4')


def test_large_file_hash_is_streamed_without_read_bytes(tmp_path,monkeypatch):
    path=tmp_path/'weights';path.write_bytes(b'bounded chunk'*1000)
    expected=g.metrics.sha(path.read_bytes())
    monkeypatch.setattr(Path,'read_bytes',lambda self:pytest.fail('Weights must never be materialised by hashing'))
    assert g.sha(path)==expected


def test_actual_metadata_cli_does_not_import_torch_or_open_original_gold():
    import subprocess,sys
    script="""import importlib.util, pathlib, sys
root=pathlib.Path.cwd();original=pathlib.Path.open
def trapped(path,*a,**kw):
 assert not path.resolve().is_relative_to(root/'datasets/research/primary-text-en-v1')
 return original(path,*a,**kw)
pathlib.Path.open=trapped
spec=importlib.util.spec_from_file_location('text_cli_cpu',root/'scripts/eval_research_text_checkpoint.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.main(['--baseline-precision','full_fp32_master_bf16_autocast'])==0
assert 'torch' not in sys.modules
"""
    result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout)
    assert value['status']=='metadata_preflight_only_no_generation' and not value['GPU_used']
    assert value['plan']['numerical_profile']['weights']=='fp32'


def test_gpt_native_weight_inventory_excludes_alternate_original_export():
    context=g.public_context(ROOT/'datasets/research/primary-text-public-en-v1','gptoss20')
    plan=g.baseline_plan(context,'bf16')
    assert plan['required_weights'] and all('/' not in item['path'] for item in plan['required_weights'])
    assert 'original/model.safetensors' not in [item['path'] for item in plan['required_weights']]


def test_learned_actual_stage_plans_use_retained_roles_without_loading_tensors():
    context=g.public_context(ROOT/'datasets/research/primary-text-public-en-v1','smallthinker4')
    cases=[('smallthinker4__siglip2__A0_bf16__pilot_s42','A0','bf16'),
        ('smallthinker4__siglip2__1c__A1_pilot_scienceqa_disjoint_v1_s42','A1','nf4'),
        ('smallthinker4__siglip2__4q__SFT_pilot_scienceqa_context_v2_balanced_s42','SFT','nf4')]
    for name,stage,dtype in cases:
        plan=g.learned_plan(ROOT/'runs/research'/name,'best',context)
        assert plan['stage']==stage and plan['numerical_profile']['weights']==dtype
        assert plan['step']>0 and plan['checkpoint_stat']['st_size']>0
        assert g.read(plan['marker']['path'])['owner']==plan['owner']


def test_scoped_giga_native_dispatch_is_exact_and_restores_aliases(tmp_path,monkeypatch):
    import scripts.run_research_stage as runner
    checkpoint=tmp_path/'run/.checkpoint_objects/object/state.pt';checkpoint.parent.mkdir(parents=True)
    (tmp_path/'run/resolved_config.json').write_text(json.dumps({'model':'ai-sage/GigaChat3.1-10B-A1.8B-bf16'}))
    calls=[]
    class Original:
        @staticmethod
        def from_pretrained(path,*args,**kwargs):calls.append((Path(path),kwargs));return 'unchanged-delegate'
    for attr in ('AutoConfig','AutoTokenizer','AutoModelForCausalLM'):monkeypatch.setattr(runner,attr,Original)
    assets=g.read(ROOT/'datasets/research/text-eval-inputs/assets.json')
    selected=next(a for a in assets['files'] if a['id']=='giga31' and a['filename']=='config.json')
    target=Path(selected['path']).parent
    with g.known_giga_configuration(checkpoint) as receipt:
        native=runner.AutoConfig.from_pretrained(target)
        assert native.model_type=='deepseek_v3' and type(native.routed_scaling_factor) is float
        assert receipt['normalizations']==[dict(field='routed_scaling_factor',source_type='int',source_value=1,
            normalized_type='float',normalized_value=1.0,
            reason='Native strict DeepseekV3Config requires float; numeric value and routing scale are unchanged.')]
        runner.AutoTokenizer.from_pretrained(target)
        assert calls[-1][1]['config'] is native and calls[-1][1]['local_files_only'] is True
        runner.AutoModelForCausalLM.from_pretrained(target)
        assert calls[-1][1]['config'] is native
        assert runner.AutoConfig.from_pretrained(tmp_path/'foreign')=='unchanged-delegate'
        assert calls[-1][1]=={}
    assert all(getattr(runner,attr) is Original for attr in ('AutoConfig','AutoTokenizer','AutoModelForCausalLM'))


def test_fixed_native_clock_does_not_change_messages():
    class Tok:
        pad_token_id=2
        def apply_chat_template(self,messages,**kw):
            assert messages==[{'role':'user','content':'public'}]
            assert kw['date_string']==kw['current_date']=='2026-11-22'
            assert kw['strftime_now']('%Y-%m-%d')=='2026-11-22'
            return 'prompt'
    proxy=g.FixedClockTokenizer(Tok())
    assert proxy.pad_token_id==2
    assert proxy.apply_chat_template([{'role':'user','content':'public'}],tokenize=False)=='prompt'


@pytest.mark.parametrize('field',['prefix_token_ids','visual_tokens','prompt_sha256','prefill_tokens','suffix_token_ids'])
def test_native_prompt_census_mismatch_refuses(field):
    r=row('gsm8k');c=census(r)
    prepared={'prefix_token_ids':[],'visual_tokens':0,'prompt_sha256':c['prompt_sha256'],'prefill_tokens':2,'suffix_token_ids':[4,5]}
    prepared[field]=1
    with pytest.raises(ValueError):g.check_prepared(r,prepared,c)


def test_controller_synthetic_rows_never_claim_model_evidence(setup):
    x=setup;runtime=FakeRuntime()
    result=g.evaluate(x.context,x.plan,x.output,allow_gpu=True,runtime=runtime)
    assert result['status']=='generation_complete_unscored' and result['records']==3
    assert result['evidence_scope']=='synthetic_scorer_validation' and not result['primary_completion_claim']
    assert not result['gold_reference_files_opened'] and not result['scoring_performed']
    rows=g.metrics.read_jsonl(x.output/'responses.jsonl')
    assert [r['record_id'] for r in rows]==[r['record_id'] for r in x.records]
    assert rows[0]['response']=='  raw whitespace\n' and 'reference' not in rows[2]
    before={p.name:p.read_bytes() for p in (x.output/'records').iterdir()}
    again=FakeRuntime();assert g.evaluate(x.context,x.plan,x.output,allow_gpu=False,resume=True,runtime=again)==result
    assert again.loads==0 and again.calls==[]
    assert before=={p.name:p.read_bytes() for p in (x.output/'records').iterdir()}


def test_interrupted_prefix_resume_only_pending_no_gold(setup):
    x=setup;first=FakeRuntime(fail_after=1)
    with pytest.raises(InterruptedError):g.evaluate(x.context,x.plan,x.output,allow_gpu=True,runtime=first)
    saved=(x.output/'records/000000.json').read_bytes();resumed=FakeRuntime()
    result=g.evaluate(x.context,x.plan,x.output,resume=True,allow_gpu=True,runtime=resumed)
    assert result['records']==3 and resumed.calls==[r['record_id'] for r in x.records[1:]]
    assert (x.output/'records/000000.json').read_bytes()==saved


def test_commit_before_jsonl_append_recovers_without_regenerating(setup,monkeypatch):
    x=setup;original=g.immutable;armed=[True]
    def crash(path,raw):
        original(path,raw)
        if Path(path).name=='000000.json' and armed[0]:armed[0]=False;raise InterruptedError('after commit before append')
    monkeypatch.setattr(g,'immutable',crash)
    with pytest.raises(InterruptedError):g.evaluate(x.context,x.plan,x.output,allow_gpu=True,runtime=FakeRuntime())
    resumed=FakeRuntime();g.evaluate(x.context,x.plan,x.output,allow_gpu=True,resume=True,runtime=resumed)
    assert resumed.calls==[r['record_id'] for r in x.records[1:]]
    assert len(g.metrics.read_jsonl(x.output/'responses.jsonl'))==3


@pytest.mark.parametrize('bad',['row','producer','public_file','raw_tail'])
def test_corruption_refuses_before_reload_or_overwrite(setup,bad):
    x=setup
    with pytest.raises(InterruptedError):g.evaluate(x.context,x.plan,x.output,allow_gpu=True,runtime=FakeRuntime(fail_after=1))
    if bad=='row':
        p=x.output/'records/000000.json';v=json.loads(p.read_text());v['response']['source']={'foreign':True};v['response_sha256']=g.digest(v['response']);p.write_text(json.dumps(v))
    elif bad=='producer':(x.output/'producer.json').write_text('{}')
    elif bad=='public_file':(Path(x.context['bundle'])/'public_records.jsonl').write_text('changed')
    else:(x.output/'responses.jsonl').write_bytes(b'foreign tail')
    runtime=FakeRuntime()
    with pytest.raises(ValueError):g.evaluate(x.context,x.plan,x.output,allow_gpu=True,resume=True,runtime=runtime)
    assert runtime.loads==0 and runtime.calls==[]


def test_derived_producer_file_recovers_from_protocol_without_model(setup):
    x=setup;result=g.evaluate(x.context,x.plan,x.output,allow_gpu=True,runtime=FakeRuntime())
    before=(x.output/'producer.json').read_bytes();(x.output/'producer.json').unlink()
    runtime=FakeRuntime();assert g.evaluate(x.context,x.plan,x.output,resume=True,runtime=runtime)==result
    assert (x.output/'producer.json').read_bytes()==before and runtime.loads==0


def test_raw_truncated_projection_rebuilds_exact_journal(setup):
    x=setup;g.evaluate(x.context,x.plan,x.output,allow_gpu=True,runtime=FakeRuntime())
    path=x.output/'responses.jsonl';desired=path.read_bytes();path.write_bytes(desired[:-20])
    g.evaluate(x.context,x.plan,x.output,resume=True,runtime=FakeRuntime())
    assert path.read_bytes()==desired and list((x.output/'recovery').iterdir())


def test_owned_uncommitted_atomic_stage_preserved_before_pending_resume(setup):
    x=setup
    with pytest.raises(InterruptedError):g.evaluate(x.context,x.plan,x.output,allow_gpu=True,runtime=FakeRuntime(fail_after=1))
    stage=x.output/'records'/('.000001.json.'+'a'*32+'.tmp');raw=b'{"unfinished":'
    stage.write_bytes(raw);resumed=FakeRuntime()
    result=g.evaluate(x.context,x.plan,x.output,resume=True,allow_gpu=True,runtime=resumed)
    assert result['records']==3 and resumed.calls==[r['record_id'] for r in x.records[1:]]
    assert not stage.exists() and (x.output/'recovery'/(g.metrics.sha(raw)+'.uncommitted')).read_bytes()==raw
    receipts=list((x.output/'recovery').glob('*.recovery.json'));assert len(receipts)==1
    receipt=g.read(receipts[0]);assert receipt['owner']==g.OWNER and receipt['committed_prediction'] is False


def test_foreign_uncommitted_name_refuses_before_model_reload(setup):
    x=setup
    with pytest.raises(InterruptedError):g.evaluate(x.context,x.plan,x.output,allow_gpu=True,runtime=FakeRuntime(fail_after=1))
    stage=x.output/'records/.foreign.tmp';stage.write_bytes(b'foreign');runtime=FakeRuntime()
    with pytest.raises(ValueError,match='Foreign'):g.evaluate(x.context,x.plan,x.output,resume=True,allow_gpu=True,runtime=runtime)
    assert stage.read_bytes()==b'foreign' and runtime.loads==0


def test_public_metadata_and_jsonl_reader_never_follow_gold_symlink(setup):
    x=setup;outside=x.output.parent/'gold.json';outside.write_text('{"reference":"private"}')
    link=x.output.parent/'metadata.json';link.symlink_to(outside)
    with pytest.raises(ValueError):g.read(link)


def test_original_input_bundle_cannot_be_used_as_output(setup):
    x=setup
    with pytest.raises(ValueError):g.evaluate(x.context,x.plan,ROOT/'datasets/research/primary-text-en-v1',runtime=FakeRuntime())


def test_full_control_gpt_backend_gate_never_loads_model(setup):
    x=setup;plan=dict(x.plan,admission_blockers=['P10 closed'])
    runtime=FakeRuntime()
    with pytest.raises(RuntimeError,match='P10'):g.evaluate(x.context,plan,x.output,allow_gpu=True,runtime=runtime)
    assert runtime.loads==0


def test_full_initialization_matches_parameters_buffers_and_aliases_cpu():
    import torch
    from torch import nn
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.embed=nn.Embedding(8,4,dtype=torch.bfloat16);self.head=nn.Linear(4,8,bias=False,dtype=torch.bfloat16)
            self.head.weight=self.embed.weight;self.router=nn.Linear(4,2,dtype=torch.bfloat16);self.norm=nn.LayerNorm(4,dtype=torch.bfloat16)
            self.register_buffer('floating_mask',torch.ones(2,dtype=torch.bfloat16));self.register_buffer('integer_positions',torch.arange(2))
    model=Tiny();expected=deepcopy(model).requires_grad_(False).float().requires_grad_(True)
    for p in expected.parameters():
        if p.requires_grad and p.is_floating_point():p.data=p.data.float()
    report=g.full_initialization_control(model)
    assert model.embed.weight is model.head.weight
    assert all(p.dtype==torch.float32 for p in model.parameters())
    assert model.floating_mask.dtype==torch.float32 and model.integer_positions.dtype==torch.int64
    for (a,p),(b,q) in zip(model.named_parameters(),expected.named_parameters()):assert a==b;torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert report['before']['parameters']['embed.weight']['requires_grad'] is False
    assert report['after_promotion']['router_parameters'] and report['after_promotion']['norm_parameters']
    assert report['Module_float_includes_floating_buffers'] and not report['GPU_memory_fit_proven']
    assert not torch.cuda.is_initialized()


def tiny_native_model():
    import torch
    from torch import nn
    class Tokenizer:
        pad_token_id=2;padding_side='right'
        def apply_chat_template(self,messages,**kw):
            assert messages==[{'role':'user','content':'A public question'}]
            assert kw['date_string']==kw['current_date']=='2026-11-22'
            assert kw['strftime_now']('%Y-%m-%d')=='2026-11-22'
            return 'NATIVE'
        def encode(self,text,**kw):
            assert kw.get('add_special_tokens') is False
            if not text:return []
            if text=='NATIVE':return [4,5]
            if text.startswith('NATIVE') and text[6:] in 'ABCD':return [4,5,32+'ABCD'.index(text[6:])]
            raise AssertionError('Unexpected prompt/continuation')
        def decode(self,ids,**kw):
            assert ids==[9,2] and kw['skip_special_tokens'] is True
            return '  raw response with whitespace\n'
    class Base(nn.Module):
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(48,4);self.projection=nn.Linear(4,48,bias=False)
            self.config=SimpleNamespace(model_type='smallthinker',max_position_embeddings=4096)
            self.generation_config=SimpleNamespace(eos_token_id=2)
            with torch.no_grad():
                self.embedding.weight.copy_(torch.arange(48*4,dtype=torch.float32).reshape(48,4)/97)
                self.projection.weight.copy_(torch.arange(48*4,dtype=torch.float32).reshape(48,4).flip(0)/123)
        def get_input_embeddings(self):return self.embedding
        def forward(self,inputs_embeds,attention_mask=None,use_cache=False,return_dict=True,logits_to_keep=0):
            assert use_cache is False and return_dict is True
            values=self.projection(inputs_embeds)
            return SimpleNamespace(logits=values[:,-logits_to_keep:,:] if logits_to_keep else values)
        def generate(self,**kw):
            assert kw['do_sample'] is False and kw['num_beams']==1 and kw['use_cache'] is True
            assert kw['max_new_tokens']==2048 and kw['logits_to_keep']==1
            return SimpleNamespace(sequences=torch.tensor([[9,2]]),scores=[torch.zeros((1,48)),torch.zeros((1,48))])
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.base=Base();self.args=SimpleNamespace(max_length=4096)
            self.tokenizer=g.FixedClockTokenizer(Tokenizer())
    return Model().eval()


def tiny_census(r):
    c=census(r);c['prompt_sha256']=g.metrics.sha('NATIVE')
    return c


def test_native_one_forward_mmlu_likelihoods_match_teacher_forced_boundary_cpu():
    import torch
    model=tiny_native_model();r=row('mmlu');c=tiny_census(r)
    actual=g.NativeTextRuntime().candidate_likelihoods(model,r,c,'smallthinker4')
    for letter in 'ABCD':
        target=32+'ABCD'.index(letter)
        with torch.no_grad():
            # Score target at the last PREFIX position in a teacher-forced
            # prefix+target sequence, not the target position itself.
            embeddings=model.base.embedding(torch.tensor([[4,5,target]]))
            logits=model.base(inputs_embeds=embeddings,use_cache=False,return_dict=True).logits[0,1]
            expected=float(torch.log_softmax(logits.float(),-1)[target])
        assert actual['candidate_log_likelihoods'][letter]==pytest.approx(expected,abs=1e-7)
        assert actual['candidate_token_ids'][letter]==[target]
        assert actual['candidate_prefix_token_ids_sha256'][letter]==g.digest([4,5])
    assert actual['length_normalization'] is False and actual['log_probability_reduction_dtype']=='float32'
    assert not torch.cuda.is_initialized()


def test_native_generator_preserves_raw_decoding_without_strip_cpu():
    import torch
    model=tiny_native_model();r=row('ifeval');actual=g.NativeTextRuntime().generate(model,r,tiny_census(r))
    assert actual['response']=='  raw response with whitespace\n'
    assert actual['generated_token_ids']==[9,2] and actual['prompt_tokens']==2
    assert actual['prompt_sha256']==g.metrics.sha('NATIVE') and not torch.cuda.is_initialized()


def test_actual_native_dtype_and_dependency_report_on_tiny_cpu_model():
    import torch
    model=tiny_native_model();report=g.describe_model(model,{'snapshot_assets':{}},{'kind':'pinned_baseline'})
    assert report['actual_parameter_buffer_router_norm_dtypes']['parameters']
    dependencies=report['actual_native_class_and_generation_source_sha256']
    assert g.sha(__file__) in dependencies.values()
    assert any(name.endswith('transformers/generation/utils.py') for name in dependencies)
    assert report['fixed_template_clock']=='2026-11-22' and not torch.cuda.is_initialized()


def test_native_candidate_prefix_drift_fails_before_model_forward(monkeypatch):
    model=tiny_native_model();r=row('mmlu')
    monkeypatch.setattr(model.base,'forward',lambda **kw:pytest.fail('Bad prefix must fail before logits'))
    tokenizer=model.tokenizer._tokenizer;original=tokenizer.encode
    def altered(text,**kw):return [99,32] if text=='NATIVEA' else original(text,**kw)
    monkeypatch.setattr(tokenizer,'encode',altered)
    with pytest.raises(ValueError,match='continuation'):g.NativeTextRuntime().candidate_likelihoods(model,r,tiny_census(r),'smallthinker4')
