"""CPU synthetic publication tests. No actual trained activation is present."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch

from src.utils import research_checkpoint_archival as archival
from src.utils import research_trained_delta_probe as frozen
from src.utils import research_lossless_delta as codec
from src.utils.research_checkpoint_io import load_checkpoint
from test_research_trained_delta_probe import fixture, put


@pytest.fixture
def prepared(fixture, monkeypatch):
    monkeypatch.setattr(archival,'METADATA_LIMIT',512*2**10)
    snapshot=fixture['snapshot'];config=json.loads((snapshot/'config.json').read_text());config['hidden_size']=4
    put(snapshot/'config.json',config);monkeypatch.setattr(codec,'CONFIG_SHA256',frozen.file_sha(snapshot/'config.json'))
    payload=fixture['payload'];payload['model']={fixture['key']:payload['model'][fixture['key']],
        'projector.0.weight':torch.randn(4,3),'projector.0.bias':torch.randn(4),
        'projector.2.weight':torch.randn(4,4),'projector.2.bias':torch.randn(4)}
    torch.save(payload,fixture['source'])
    # Synthetic ZIP-only padding makes a tiny exact model a measurable storage
    # saving fixture; no trained checkpoint is padded or changed by this code.
    with zipfile.ZipFile(fixture['source'],'a') as archive:
        prefix=archive.namelist()[0].split('/')[0];archive.writestr(prefix+'/synthetic_test_padding',b'\0'*(2*2**20))
    put(fixture['run']/'trainable_parameters.json',{fixture['key']:24})
    evidence={'path':str(fixture['run']/'synthetic_activation.json'),'sha256':'a'*64,
        'code_sha256':archival.code_binding(),'synthetic_test_only':True}
    monkeypatch.setattr(archival,'activation_evidence',lambda path=None:evidence)
    return fixture


def publish(prepared,**kwargs):
    return archival.publish_completed_checkpoint(prepared['run'],snapshot=prepared['snapshot'],
        max_staging_bytes=192*2**20,**kwargs)


def roles(run):return {name:(run/name).resolve().name for name in ('best','last')}


def test_actual_default_gate_closed_no_torch_or_new_files(tmp_path):
    child=subprocess.run([sys.executable,'-c',
        'import sys; from src.utils import research_checkpoint_archival as a; '
        f"r=a.preflight({str(tmp_path/'absent')!r}); assert r['status']=='gate_closed'; "
        "assert 'torch' not in sys.modules; "
        f"\ntry:a.publish_completed_checkpoint({str(tmp_path/'absent')!r})\nexcept a.ArchivalGateClosed:pass\nelse:raise AssertionError()"],
        cwd=archival.ROOT,capture_output=True,text=True)
    assert child.returncode==0,child.stderr
    assert not list(tmp_path.iterdir())


def test_shared_best_last_exact_payload_marker_and_idempotence(prepared):
    source=prepared['source'];before=frozen.payload_signature(load_checkpoint(source));oldmarker=json.loads(prepared['marker'].read_text())
    result=publish(prepared);current=roles(prepared['run'])
    assert result['status']=='completed' and current['best']==current['last']
    assert len(result['transactions'])==1 and not source.exists()
    state=prepared['run']/'best/state.pt';restored=load_checkpoint(state,mmap=True)
    assert frozen.payload_signature(restored)==before
    marker=json.loads((state.parent/'.research_checkpoint.json').read_text())
    assert {k:v for k,v in marker.items() if k!='checkpoint_archival'}==oldmarker
    assert marker['checkpoint_archival']['payload_sha256']==before['sha256']
    assert restored['normal_same_object'] is restored['normal_backing']
    before_bytes=state.read_bytes();again=publish(prepared)
    assert again['roles']==current and len(again['transactions'])==1 and state.read_bytes()==before_bytes
    assert set(p.name for p in (prepared['run']/'.checkpoint_objects').iterdir())=={current['best']}
    assert archival.validate_completed_archival(prepared['run'])==again


def distinct(prepared):
    run=prepared['run'];target=run/'.checkpoint_objects/object-later';target.mkdir()
    payload=copy.deepcopy(prepared['payload']);payload['step']=171
    payload['model'][prepared['key']]=payload['model'][prepared['key']]+.125
    torch.save(payload,target/'state.pt')
    with zipfile.ZipFile(target/'state.pt','a') as archive:
        prefix=archive.namelist()[0].split('/')[0];archive.writestr(prefix+'/synthetic_test_padding',b'\0'*(2*2**20))
    marker=json.loads(prepared['marker'].read_text());marker.update(step=171,metric=21.5)
    put(target/'.research_checkpoint.json',marker)
    (run/'last').unlink();(run/'last').symlink_to(Path('.checkpoint_objects')/target.name,target_is_directory=True)
    put(run/'completed.json',{'steps':171,'records':10832})
    compact=json.loads((run/'checkpoint_compaction.json').read_text());compact['checkpoints'].append({'new_object':target.name})
    put(run/'checkpoint_compaction.json',compact)
    return target


def test_distinct_historical_best_never_advances_last_and_sequential_reclaims(prepared,monkeypatch):
    later=distinct(prepared);oldroles=roles(prepared['run']);observed=[];original=archival._derive_unlocked
    def derive(source,*args,**kwargs):
        observed.append((source.parent.name,roles(prepared['run']),prepared['source'].exists()))
        return original(source,*args,**kwargs)
    monkeypatch.setattr(archival,'_derive_unlocked',derive)
    result=publish(prepared)
    assert len(result['transactions'])==2
    assert observed[0][1]==oldroles
    assert observed[1][1]['last']==oldroles['last'] and observed[1][1]['best']!=oldroles['best']
    assert not observed[1][2] # raw BEST freed before staging distinct LAST
    best=load_checkpoint(prepared['run']/'best/state.pt');last=load_checkpoint(prepared['run']/'last/state.pt')
    assert best['step']==170 and last['step']==171
    assert json.loads((prepared['run']/'best/.research_checkpoint.json').read_text())['metric']==31.2
    assert json.loads((prepared['run']/'last/.research_checkpoint.json').read_text())['metric']==21.5
    assert not later.exists()


@pytest.mark.parametrize('failure',['after_first_pointer','after_stage_rename','after_derived_journal',
    'after_final_marker','after_first_source_unlink','after_source_marker_unlink'])
def test_crash_recovery_does_not_repeat_encoding_or_change_role_payload(prepared,monkeypatch,failure):
    count=[];original_derive=archival._derive_unlocked
    def derive(*args,**kwargs):count.append(1);return original_derive(*args,**kwargs)
    monkeypatch.setattr(archival,'_derive_unlocked',derive)
    injected=[False]
    if failure=='after_first_pointer':
        original=archival._set_role
        def fail(*args):
            original(*args)
            if not injected[0]:injected[0]=True;raise SystemExit('injected')
        monkeypatch.setattr(archival,'_set_role',fail)
    elif failure=='after_stage_rename':
        original=os.rename
        def fail(source,target):
            original(source,target)
            if Path(source).name.startswith('.stage-archive-') and not injected[0]:injected[0]=True;raise SystemExit('injected')
        monkeypatch.setattr(os,'rename',fail)
    elif failure in ('after_final_marker','after_derived_journal'):
        original=archival._atomic
        def fail(path,value):
            original(path,value)
            condition=(Path(path).name=='.research_checkpoint.json' and value.get('checkpoint_archival')) if failure=='after_final_marker' else value.get('phase')=='derived'
            if condition and not injected[0]:
                injected[0]=True;raise SystemExit('injected')
        monkeypatch.setattr(archival,'_atomic',fail)
    else:
        original=Path.unlink
        def fail(path,*args,**kwargs):
            original(path,*args,**kwargs)
            selected=prepared['source'] if failure=='after_first_source_unlink' else prepared['marker']
            if path==selected and not injected[0]:injected[0]=True;raise SystemExit('injected')
        monkeypatch.setattr(Path,'unlink',fail)
    with pytest.raises(SystemExit):publish(prepared)
    assert len(count)==1
    result=publish(prepared)
    assert result['status']=='completed' and len(count)==1
    assert frozen.payload_signature(load_checkpoint(prepared['run']/'best/state.pt'))==frozen.payload_signature(prepared['payload'])


def test_unknown_partial_stage_file_preserved_and_blocks_resume(prepared,monkeypatch):
    def fail(*args):raise RuntimeError('writer stopped')
    monkeypatch.setattr(archival,'_derive_unlocked',fail)
    with pytest.raises(RuntimeError):publish(prepared)
    journal=json.loads((prepared['run']/archival.JOURNAL).read_text());stage=prepared['run']/'.checkpoint_objects'/journal['stage']
    (stage/'user_file').write_bytes(b'keep me')
    with pytest.raises(ValueError,match='Unknown files'):publish(prepared)
    assert (stage/'user_file').read_bytes()==b'keep me' and prepared['source'].is_file()


def test_no_benefit_preserves_raw_roles_and_cleans_only_own_stage(prepared,monkeypatch):
    original=roles(prepared['run']);state_sha=frozen.file_sha(prepared['source'])
    def derive(source,stage,*args):
        (stage/'state.pt').write_bytes(source.read_bytes())
        return {'source_state_sha256':state_sha,'archive_sha256':state_sha,'archive_bytes':source.stat().st_size,
            'all_payload_fields_and_backing_bits_exact':True,'payload_sha256':'c'*64}
    monkeypatch.setattr(archival,'_derive_unlocked',derive)
    with pytest.raises(archival.NoBenefit):publish(prepared)
    assert roles(prepared['run'])==original and frozen.file_sha(prepared['source'])==state_sha
    assert set(p.name for p in (prepared['run']/'.checkpoint_objects').iterdir())=={'object-raw'}
    with pytest.raises(archival.NoBenefit):publish(prepared)


@pytest.mark.parametrize('name',['evaluation','checkpoint_roundtrip.json','broken_evaluation_symlink'])
def test_evaluation_already_begun_is_probe_only_no_transaction(prepared,name):
    path=prepared['run']/name
    if name=='evaluation':path.mkdir()
    elif name=='broken_evaluation_symlink':(prepared['run']/'evaluation').symlink_to('absent')
    else:put(path,{'status':'passed'})
    with pytest.raises(archival.ArchivalGateClosed,match='probe only'):publish(prepared)
    assert not (prepared['run']/archival.JOURNAL).exists() and prepared['source'].is_file()


def test_no_nested_shared_reader_lock(prepared,monkeypatch):
    from src.utils import research_checkpoint_compat as compatibility
    def never(*args,**kwargs):raise AssertionError('Shared reader lock under exclusive manager lock')
    monkeypatch.setattr(compatibility,'load_research_payload',never)
    assert publish(prepared)['status']=='completed'


def test_partial_writer_restarts_only_unpublished_owned_stage(prepared,monkeypatch):
    original=archival._derive_unlocked;count=[]
    def derive(source,stage,*args):
        count.append(1)
        if len(count)==1:
            (stage/'state.pt').write_bytes(b'partial own write');raise SystemExit('writer death')
        return original(source,stage,*args)
    monkeypatch.setattr(archival,'_derive_unlocked',derive)
    before=roles(prepared['run'])
    with pytest.raises(SystemExit):publish(prepared)
    assert roles(prepared['run'])==before and prepared['source'].is_file()
    assert publish(prepared)['status']=='completed' and len(count)==2


def test_verified_target_corruption_refuses_recovery_and_retains_source(prepared,monkeypatch):
    original=archival._atomic;injected=[]
    def stop(path,value):
        original(path,value)
        if value.get('phase')=='verified' and not injected:injected.append(1);raise SystemExit('before publish')
    monkeypatch.setattr(archival,'_atomic',stop)
    before=roles(prepared['run'])
    with pytest.raises(SystemExit):publish(prepared)
    journal=json.loads((prepared['run']/archival.JOURNAL).read_text())
    target=prepared['run']/'.checkpoint_objects'/journal['stage']/'state.pt'
    raw=bytearray(target.read_bytes());raw[-1]^=1;target.write_bytes(raw)
    with pytest.raises(ValueError,match='bytes'):publish(prepared)
    assert roles(prepared['run'])==before and prepared['source'].is_file()


@pytest.mark.parametrize('mutation',['bf16_projector','missing_llm','alias_original','optimizer','missing_novel_census'])
def test_model_or_whole_payload_eligibility_refuses_inexact_archives(prepared,mutation):
    payload=prepared['payload']
    if mutation=='bf16_projector':payload['model']['projector.0.weight']=payload['model']['projector.0.weight'].bfloat16()
    if mutation=='missing_llm':payload['model'].pop(prepared['key'])
    if mutation=='alias_original':payload['extra_alias']=payload['model'][prepared['key']]
    if mutation=='optimizer':payload['optimizer']={'state':{}}
    if mutation=='missing_novel_census':put(prepared['run']/'trainable_parameters.json',{})
    torch.save(payload,prepared['source'])
    before=roles(prepared['run'])
    with pytest.raises(ValueError):publish(prepared)
    assert roles(prepared['run'])==before


@pytest.mark.parametrize('kind',['low_disk','low_ram','source_corruption'])
def test_guard_or_source_change_never_publishes(prepared,monkeypatch,kind):
    before=roles(prepared['run'])
    if kind=='low_disk':
        from src.utils.research_storage import DiskBudget
        def no(*args):raise RuntimeError('disk reserve')
        monkeypatch.setattr(DiskBudget,'ensure_space',no)
    elif kind=='low_ram':
        def no(*args,**kwargs):raise MemoryError('memory reserve')
        monkeypatch.setattr(codec,'memory_guard',no)
    else:
        original=archival._derive_unlocked
        def no(source,*args,**kwargs):
            result=original(source,*args,**kwargs)
            with source.open('ab') as handle:handle.write(b'changed')
            return result
        monkeypatch.setattr(archival,'_derive_unlocked',no)
    with pytest.raises((RuntimeError,MemoryError,ValueError)):publish(prepared)
    assert roles(prepared['run'])==before and prepared['source'].is_file()


def test_archive_aware_real_queue_retry_never_retrains_or_legacy_mmaps(prepared,tmp_path,monkeypatch):
    from scripts import research_queue as base,compact_research_checkpoints as old_compactor
    from src.utils.research_checkpoint_archival_queue import ProspectiveSmallArchivalMixin
    class Queue(ProspectiveSmallArchivalMixin,base.ResearchQueue):pass
    def no_training(*args,**kwargs):raise AssertionError('Completed training must never restart')
    monkeypatch.setattr(base.subprocess,'Popen',no_training)
    spec=tmp_path/'queue.json';state=tmp_path/'queue.state.json'
    job={'id':'science','argv':['--stage','SFT','--output',str(prepared['run'])],
        'compact_completed':True,'archive_completed_small':True}
    put(spec,{'queue_id':'synthetic-archive-only','reserve_gb':120,'jobs':[job]})
    queue=Queue(spec,state,retry_failed=True)
    queue.state['jobs']['science']={'status':'training_completed_compaction_failed','training_completed':True,
        'compaction':json.loads((prepared['run']/'checkpoint_compaction.json').read_text())}
    queue.persist()
    original_publish=archival.publish_completed_checkpoint
    def tiny_publish(run,**kwargs):return original_publish(run,snapshot=prepared['snapshot'],max_staging_bytes=192*2**20,**kwargs)
    monkeypatch.setattr(archival,'publish_completed_checkpoint',tiny_publish)
    original=archival._set_role;injected=[]
    def fail_once(*args):
        original(*args)
        if not injected:injected.append(1);raise ValueError('crash after BEST published')
    monkeypatch.setattr(archival,'_set_role',fail_once)
    assert queue.run()==1
    failed=json.loads(state.read_text());record=failed['jobs']['science']
    assert record['training_completed'] and record['status']=='training_completed_compaction_failed'
    assert (prepared['run']/archival.JOURNAL).is_file()
    # Simulate a missing queue compaction field after crash. Disk lineage is
    # sufficient, but calling the legacy mmap compactor on archived BEST is not.
    record.pop('compaction');put(state,failed)
    def no_old_mmap(*args,**kwargs):raise AssertionError('Legacy compactor must never inspect an archive')
    monkeypatch.setattr(old_compactor,'compact_completed_checkpoint',no_old_mmap)
    assert Queue(spec,state,retry_failed=False).run()==1
    assert Queue(spec,state,retry_failed=True).run()==0
    completed=json.loads(state.read_text())['jobs']['science']
    assert completed['status']=='completed' and completed['checkpoint_archival']['status']=='completed'
    assert archival.validate_completed_archival(prepared['run'])['roles']==roles(prepared['run'])


def test_foreign_same_name_role_temporary_is_never_unlinked(prepared):
    destination='object-archive-'+'d'*32
    temporary=prepared['run']/('.best.archive.'+'d'*32+'.link')
    temporary.write_bytes(b'user file')
    before=roles(prepared['run'])
    with pytest.raises(ValueError,match='preserved'):archival._set_role(prepared['run'],'best',destination)
    assert temporary.read_bytes()==b'user file' and roles(prepared['run'])==before


def test_known_same_target_role_temporary_from_hard_kill_is_retryable(prepared):
    destination='object-archive-'+'d'*32
    temporary=prepared['run']/('.best.archive.'+'d'*32+'.link')
    temporary.symlink_to(Path('.checkpoint_objects')/destination,target_is_directory=True)
    archival._set_role(prepared['run'],'best',destination)
    assert not os.path.lexists(temporary) and os.readlink(prepared['run']/'best')==str(Path('.checkpoint_objects')/destination)


def test_source_marker_byte_change_before_publish_refused(prepared,monkeypatch):
    original=archival._derive_unlocked
    def mutate(*args,**kwargs):
        result=original(*args,**kwargs)
        prepared['marker'].write_bytes(prepared['marker'].read_bytes()+b' ')
        return result
    monkeypatch.setattr(archival,'_derive_unlocked',mutate)
    before=roles(prepared['run'])
    with pytest.raises(ValueError,match='source changed'):publish(prepared)
    assert roles(prepared['run'])==before and prepared['source'].exists()


@pytest.mark.parametrize('name',[archival.JOURNAL,archival.REPORT])
def test_broken_unknown_report_symlink_preserved(prepared,name):
    path=prepared['run']/name;path.symlink_to('missing-user-destination')
    before=roles(prepared['run'])
    with pytest.raises(ValueError,match='local bounded JSON'):publish(prepared)
    assert path.is_symlink() and os.readlink(path)=='missing-user-destination' and roles(prepared['run'])==before


def test_forged_archival_marker_without_transaction_is_never_completed(prepared):
    marker=json.loads(prepared['marker'].read_text());marker['checkpoint_archival']={'protocol':archival.OWNER}
    put(prepared['marker'],marker)
    with pytest.raises((OSError,ValueError)):publish(prepared)
    assert not (prepared['run']/archival.REPORT).exists() and prepared['source'].is_file()
