"""Private v2 metadata scope; frozen v1 implementation and gold policy intact."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from src.eval import research_caption_generation as legacy
from src.eval import research_caption_generation_v2 as v2
from tests.test_research_caption_generation import example,FakeRuntime,fake_scores,interval,write


def _bind_fixture(example,monkeypatch,*,records=3):
    protocol=json.loads((example.bundle/'protocol.json').read_text())
    protocol.update(protocol=v2.DATA_PROTOCOL,preparation_profile=v2.PROFILE,records=records,reference_records=records)
    protocol['preparation_source_sha256']={str(v2.ROOT/name):legacy.file_sha(v2.ROOT/name) for name in
        ('src/eval/research_caption_data_v2.py','scripts/prepare_research_caption_eval_v2.py')}
    write(example.bundle/'protocol.json',protocol)
    monkeypatch.setattr(v2,'DATA_PROTOCOL_SHA256',legacy.file_sha(example.bundle/'protocol.json'))
    monkeypatch.setattr(v2,'RECORDS',records)
    original=v2._private_owner
    def cpu_private():
        owner=original();owner.gpu_idle=lambda:None;return owner
    monkeypatch.setattr(v2,'_private_owner',cpu_private)
    example.context=v2.prepare_context(example.run,example.bundle,assets=example.assets)
    example.output=example.root/'caption_pilot_en_v2'
    return example


@pytest.fixture
def current(example,monkeypatch):return _bind_fixture(example,monkeypatch)


def _canonical_globals():
    return {name:getattr(legacy,name) for name in ('OWNER','DATA_PROTOCOL_SHA256','RECORDS','SOURCES')}


def test_closed_profile_import_and_cli_touch_no_data_models_or_globals(tmp_path,monkeypatch,capsys):
    before=_canonical_globals();monkeypatch.setattr(v2,'DATA_PROTOCOL_SHA256',None)
    assert v2.profile_status()['status'].startswith('closed')
    with pytest.raises(v2.ProfileNotReady):v2.prepare_context(tmp_path/'missing',tmp_path/'missing_bundle')
    from scripts import eval_research_caption_checkpoint_v2 as cli
    assert cli.main(['--run',str(tmp_path/'missing')])==2
    assert 'closed_waiting_ROOT' in capsys.readouterr().out and not list(tmp_path.iterdir())
    assert _canonical_globals()==before


def test_clean_import_has_no_torch_or_private_global_registration():
    result=subprocess.run([sys.executable,'-c',
        "import sys; import src.eval.research_caption_generation_v2; assert 'torch' not in sys.modules; assert 'src.eval._caption_v2_private_frozen_v1' not in sys.modules"],
        cwd=v2.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_private_clone_reuses_verbatim_function_bytecode_and_never_patches_v1(current):
    before=_canonical_globals();owner=v2._private_owner()
    assert owner.__dict__ is not legacy.__dict__ and owner.OWNER!=legacy.OWNER
    assert owner.RECORDS==3 and owner.DATA_PROTOCOL_SHA256==v2.DATA_PROTOCOL_SHA256
    for name in ('prepare_context','evaluate','_row','_references','committed_rows'):
        assert getattr(owner,name).__code__.co_code==getattr(legacy,name).__code__.co_code
        assert getattr(owner,name).__globals__ is owner.__dict__
    assert owner.NativeRuntime.generate.__code__.co_code==legacy.NativeRuntime.generate.__code__.co_code
    assert _canonical_globals()==before and owner.__name__ not in sys.modules


def test_static_protocol_records_new_wrapper_cli_and_frozen_source(current):
    static=current.context['static'];binding=static['version_wrapper']
    assert static['owner']==v2.OWNER and static['data_protocol_sha256']==v2.DATA_PROTOCOL_SHA256
    assert binding['reused_frozen_v1_sha256']==v2.V1_SOURCE_SHA256
    assert binding['preparation_profile']==v2.PROFILE and binding['data_protocol']==v2.DATA_PROTOCOL
    for name in (v2.V1_SOURCE,v2.SOURCE,v2.CLI):assert static['source_files'][name]['sha256']==legacy.file_sha(v2.ROOT/name)
    assert set(static['preparation_sources'])=={str(v2.ROOT/name) for name in
        ('src/eval/research_caption_data_v2.py','scripts/prepare_research_caption_eval_v2.py')}


def test_v2_success_resume_and_error_restore_all_canonical_v1_globals(current):
    before=_canonical_globals()
    runtime=FakeRuntime(current,fail_after=1)
    with pytest.raises(InterruptedError):v2.evaluate(current.context,current.output,allow_gpu=True,runtime=runtime,scorer=fake_scores,bootstrap=interval)
    assert _canonical_globals()==before
    next_runtime=FakeRuntime(current)
    result=v2.evaluate(current.context,current.output,resume=True,allow_gpu=True,runtime=next_runtime,scorer=fake_scores,bootstrap=interval)
    assert result['owner']==v2.OWNER and result['data_protocol_sha256']==v2.DATA_PROTOCOL_SHA256
    assert next_runtime.calls==[row['record_id'] for row in current.records[1:]] and _canonical_globals()==before
    assert json.loads((current.output/'protocol.json').read_text())['static']['version_wrapper']==v2._binding()


def test_v2_default_output_is_new_namespace_and_legacy_output_name_refused(current):
    result=v2.evaluate(current.context,allow_gpu=True,runtime=FakeRuntime(current),scorer=fake_scores,bootstrap=interval)
    assert result['records']==3 and (current.run/'evaluation/caption_pilot_en_v2/summary.json').is_file()
    assert not (current.run/'evaluation/caption_pilot_en_v1').exists()
    with pytest.raises(ValueError,match='legacy v1'):v2.evaluate(current.context,current.root/'caption_pilot_en_v1',score_only=True,resume=True)


def test_valid_v1_results_are_never_resumed_or_retagged_as_v2(example,monkeypatch):
    old_output=example.output
    legacy.evaluate(example.context,old_output,allow_gpu=True,runtime=FakeRuntime(example),scorer=fake_scores,bootstrap=interval)
    before={path.name:path.read_bytes() for path in old_output.iterdir() if path.is_file()}
    current=_bind_fixture(example,monkeypatch);runtime=FakeRuntime(current)
    with pytest.raises(ValueError,match='protocol differs'):
        v2.evaluate(current.context,old_output,resume=True,allow_gpu=True,runtime=runtime,scorer=fake_scores,bootstrap=interval)
    assert runtime.loads==0 and before=={path.name:path.read_bytes() for path in old_output.iterdir() if path.is_file()}


def test_old_context_and_v1_bundle_profile_refused_before_runtime(current):
    context=copy.deepcopy(current.context);context['static']['owner']=legacy.OWNER
    with pytest.raises(ValueError,match='different version'):v2.evaluate(context,current.output,allow_gpu=True,runtime=FakeRuntime(current))
    path=current.bundle/'protocol.json';value=json.loads(path.read_text());value['protocol']='research_caption_pilot_eval_v1';write(path,value)
    with pytest.raises(ValueError,match='pin differs'):v2.prepare_context(current.run,current.bundle,assets=current.assets)


def test_same_pin_cannot_rename_a_v1_preparation_profile_as_v2(current,monkeypatch):
    path=current.bundle/'protocol.json';value=json.loads(path.read_text());value['preparation_profile']='legacy';write(path,value)
    monkeypatch.setattr(v2,'DATA_PROTOCOL_SHA256',legacy.file_sha(path))
    with pytest.raises(ValueError,match='signal-safe v2'):v2.prepare_context(current.run,current.bundle,assets=current.assets)


def test_frozen_source_mutation_and_wrapper_binding_changes_refused(current,monkeypatch):
    monkeypatch.setattr(v2,'V1_SOURCE_SHA256','0'*64)
    with pytest.raises(ValueError,match='Frozen v1 caption'):v2.evaluate(current.context,current.output,score_only=True,resume=True)


def test_new_source_provenance_cannot_be_omitted_or_replaced(current):
    changed=copy.deepcopy(current.context);changed['static']['source_files'].pop(v2.SOURCE)
    with pytest.raises(ValueError,match='source binding'):v2.evaluate(changed,current.output,allow_gpu=True,runtime=FakeRuntime(current))


def test_cli_preflight_uses_v2_context_and_restores_signal_handlers(current,monkeypatch,capsys):
    import signal
    from scripts import eval_research_caption_checkpoint_v2 as cli
    before={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    monkeypatch.setattr(cli.caption,'prepare_context',lambda *args,**kwargs:current.context)
    assert cli.main(['--run',str(current.run)])==0
    value=json.loads(capsys.readouterr().out)
    assert value['owner']==v2.OWNER and value['data_protocol_sha256']==v2.DATA_PROTOCOL_SHA256 and value['gold_references_opened'] is False
    assert all(signal.getsignal(sig)==handler for sig,handler in before.items())


def test_whole372_gold_remains_inaccessible_at371_then_complete_score_only(example,monkeypatch):
    count=372;records=[];references=[]
    for index in range(count):
        path=example.root/f'whole{index}.png';image=Image.new('RGB',(2,1),(index//256,index%256,31));image.save(path)
        sha=hashlib.sha256(b'RGB:2:1:'+image.tobytes()).hexdigest();identity='caption-'+sha
        records.append({'record_id':identity,'image_paths':[str(path)],'image_hashes':[sha],'question':legacy.QUESTION})
        references.append({'record_id':identity,'image_hash':sha,'references':[f'actual reference {index}'],
            'reference_records':[{'source_record_id':f'whole_ref{index}','reference':f'actual reference {index}',
                'source':{'repo':'fixture/full372','revision':'a'*40}}]})
    example.records=records;example.references=references
    (example.bundle/'generation.jsonl').write_bytes(b''.join(legacy.canonical(row)+b'\n' for row in records))
    (example.bundle/'references.jsonl').write_bytes(b''.join(legacy.canonical(row)+b'\n' for row in references))
    audit=json.loads((example.bundle/'audit.json').read_text());audit['pool_images']=count;write(example.bundle/'audit.json',audit)
    protocol=json.loads((example.bundle/'protocol.json').read_text());protocol['files_sha256']={name:legacy.file_sha(example.bundle/name) for name in ('generation.jsonl','references.jsonl','audit.json')}
    write(example.bundle/'protocol.json',protocol);current=_bind_fixture(example,monkeypatch,records=count)
    owner=v2._private_owner();runtime=FakeRuntime(current);model,loaded=runtime.load(None,None,None)
    generation_protocol={'schema_version':1,'owner':v2.OWNER,'static':current.context['static'],'model_provenance':runtime.describe(model,loaded)}
    current.output.mkdir();(current.output/'records').mkdir();owner._atomic(current.output/'protocol.json',generation_protocol)
    generated={'prediction':'independent synthetic prediction','generated_token_ids':[9,2],'generated_tokens':2,
        'prefix_token_ids':[1,4],'suffix_token_ids':[5],'visual_tokens':3,'prefill_tokens':6,'prompt_sha256':'c'*64,'logits_to_keep':1}
    owner.check_prompt_contract(current.output,generation_protocol,generated)
    rows=[]
    for index,record in enumerate(records[:-1]):
        row=owner._row(record,index,generated,generation_protocol);owner._atomic(current.output/'records'/f'{index:06d}.json',row);rows.append(row)
    owner._atomic(current.output/'predictions.jsonl',b''.join(owner.canonical(row)+b'\n' for row in rows),raw=True)
    original=Path.open;reads=[]
    def opened(path,*args,**kwargs):
        if path==current.bundle/'references.jsonl':
            assert (current.output/'generation_complete.json').is_file()
            assert len(list((current.output/'records').glob('*.json')))==count
            reads.append(path)
        assert path!=current.train,'Original train gold must never enter this owner'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)
    with pytest.raises(ValueError,match='Score-only'):
        v2.evaluate(current.context,current.output,resume=True,score_only=True,scorer=fake_scores,bootstrap=interval)
    assert reads==[] and not (current.output/'generation_complete.json').exists()
    row=owner._row(records[-1],count-1,generated,generation_protocol);owner._atomic(current.output/'records'/f'{count-1:06d}.json',row);rows.append(row)
    owner.reconcile_predictions(current.output,rows,resume=True)
    result=v2.evaluate(current.context,current.output,resume=True,score_only=True,scorer=fake_scores,bootstrap=interval)
    assert result['records']==count and result['owner']==v2.OWNER and reads
