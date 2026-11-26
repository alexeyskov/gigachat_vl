"""CPU-native tensors/generation and synthetic lifecycle; no GPU proof."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from src.utils import research_archive_gpu_parity as p


def tiny():
    import torch
    from torch import nn
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.embed=nn.Embedding(8,4);self.head=nn.Linear(4,8,bias=False)
            self.head.weight=self.embed.weight;self.projector=nn.Linear(3,4)
            self.register_buffer('nonpersistent',torch.arange(2.),persistent=False)
        def forward(self,ids):return self.head(self.embed(ids))
    torch.manual_seed(42);return Model().requires_grad_(False).eval()


def test_import_default_closed_without_torch_or_output(tmp_path):
    command=[sys.executable,'-c',
        "import sys; from src.utils import research_archive_gpu_parity as p; "
        f"r=p.preflight({str(tmp_path/'missing.json')!r}); assert r['status']=='gate_closed'; "
        "assert 'torch' not in sys.modules; assert r['GPU_launched'] is False"]
    result=subprocess.run(command,cwd=p.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not list(tmp_path.iterdir())


def test_exact_fresh_cpu_model_all_tensor_buffer_alias_census():
    import torch
    a=tiny();b=tiny();report=p.tensor_census(a);p.assert_equal_census(report,p.tensor_census(b))
    assert 'model.nonpersistent' in report['tensors']
    assert ['model.embed.weight','model.head.weight'] in report['tensor_object_alias_groups']
    assert all(v['dtype']=='torch.float32' for v in report['tensors'].values())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mutation',['weight','buffer','alias','dtype','projector'])
def test_tensor_or_alias_or_frozen_projector_change_fails(mutation):
    import torch
    a=tiny();b=tiny()
    if mutation=='weight':b.embed.weight.data[0,0]+=1
    elif mutation=='buffer':b.nonpersistent[0]+=1
    elif mutation=='alias':b.head.weight=torch.nn.Parameter(b.head.weight.detach().clone(),requires_grad=False)
    elif mutation=='dtype':b=b.double()
    else:b.projector.bias.data[0]+=1
    with pytest.raises(ValueError):p.assert_equal_census(p.tensor_census(a),p.tensor_census(b))


def test_noncontiguous_or_trainable_models_refuse():
    import torch
    a=tiny().requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):p.tensor_census(a)
    a=tiny();a.register_buffer('bad_view',torch.ones(2,3).T)
    with pytest.raises(ValueError,match='contiguous'):p.tensor_census(a)


def generation(i=0):
    return {'record_id':str(i),'prediction':'native','generated_token_ids':[3,2],'generated_tokens':2,
        'prompt_sha256':'a'*64,'prefill_tokens':4,'visual_tokens':1,'logits_to_keep':1}


def test_native_generation_tokens_and_prompt_fields_all_exact():
    a=[generation()];p.compare_generations(a,copy.deepcopy(a))
    for key,bad in [('generated_token_ids',[4,2]),('prompt_sha256','b'*64),('prefill_tokens',5),('record_id','foreign')]:
        b=copy.deepcopy(a);b[0][key]=bad
        with pytest.raises(ValueError):p.compare_generations(a,b)


def test_real_tiny_native_forward_same_tokens_after_fresh_restore_cpu():
    import torch
    a=tiny();b=tiny();ids=torch.tensor([[1,2]])
    with torch.no_grad():left=a(ids).argmax(-1).tolist();right=b(ids).argmax(-1).tolist()
    assert left==right;p.assert_equal_census(p.tensor_census(a),p.tensor_census(b))
    assert not torch.cuda.is_initialized()


def test_frozen_native_Small_autoregressive_visual_generation_after_restore_cpu(tmp_path):
    import torch
    from tests.test_research_inference import TinyResearchModel,record
    from src.eval.research_inference import freeze_for_generation,generate_prediction
    torch.manual_seed(47);source=freeze_for_generation(TinyResearchModel())
    path=tmp_path/'tiny_source.pt';torch.save(source.state_dict(),path)
    archive=freeze_for_generation(TinyResearchModel());archive.load_state_dict(torch.load(path,weights_only=True))
    a=generate_prediction(source,record(),max_new_tokens=4)
    b=generate_prediction(archive,record(),max_new_tokens=4)
    p.compare_generations([dict(a,record_id='one')],[dict(b,record_id='one')])
    p.assert_equal_census(p.tensor_census(source),p.tensor_census(archive))
    assert a['generated_tokens']==4 and a['visual_tokens']==2 and not torch.cuda.is_initialized()


class FakeGuard:
    phase='test';samples=[]
    def check(self):return {}
    def __enter__(self):return self
    def __exit__(self,*args):pass


class FakeRuntime:
    def __init__(self,*,mutate=False,wrong_tokens=False):self.loads=[];self.live=None;self.mutate=mutate;self.wrong_tokens=wrong_tokens
    def load(self,path,*args):
        self.loads.append(str(path));self.live=tiny()
        return self.live,{'model_ref':{'repo':'synthetic native CPU fixture'},'stage':'SFT'}
    def generate(self,model,row):
        assert model is self.live
        result=generation();result.pop('record_id')
        if self.wrong_tokens and len(self.loads)==2:result['generated_token_ids']=[5,2]
        if self.mutate:model.projector.bias.data[0]+=1
        return result
    def selection(self,model,config,check):
        check();return {'metrics':{'group_macro_score':.5},'predictions':[{'record_id':str(i),'generation':generation(i)} for i in range(116)],'protocol':{'fixed':True}}


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    from src.utils import research_lossless_delta
    from src.eval import research_parallel_protocol
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(''.join(json.dumps({'record_id':str(i)})+'\n' for i in range(64)))
    plan={'run':str(tmp_path/'run'),'source':str(tmp_path/'source.pt'),'archive':str(tmp_path/'archive.pt'),
        'source_stat':{'st_size':1},'archive_stat':{'st_size':1},'decoded_model_bytes':1,'normal_storage_bytes':1,
        'manifest':{'path':str(manifest)},'reserve_gb':120,'role':'best',
        'gate':{'config':{},'selected':{'marker':{'metric':.5}}},'source_sha256':'a'*64,
        'archive_sha256':'b'*64,'source_payload_sha256':'c'*64,'restored_payload_sha256':'c'*64,
        'code_sha256':{},'helper_source_sha256':{}}
    metrics=tmp_path/'saved_metrics.json';protocol=tmp_path/'saved_protocol.json'
    metrics.write_text(json.dumps({'group_macro_score':.5,'validation_step':1}));protocol.write_text(json.dumps({'fixed':True,'validation_step':1}))
    plan['saved_selection_files']={'metrics.json':{'path':str(metrics)},'protocol.json':{'path':str(protocol)}}
    plan['gate']['selected']['marker']['step']=1
    monkeypatch.setattr(p,'assert_plan',lambda *a,**kw:None)
    monkeypatch.setattr(research_lossless_delta,'memory_guard',lambda *a,**kw:{'synthetic':True})
    monkeypatch.setattr(research_parallel_protocol,'select_guard_indices',lambda records:list(range(64)))
    return plan,tmp_path/'output'


def test_synthetic_lifecycle_sequential_and_never_ROOT_activation(fixture):
    plan,out=fixture;runtime=FakeRuntime()
    report=p.execute(plan,out,nonce='a'*64,runtime=runtime,guard=FakeGuard())
    assert runtime.loads==[plan['source'],plan['archive']]
    assert report['status']=='passed_synthetic_CPU_fixture' and report['producer']=='CPU_test_fixture'
    assert report['all_model_tensors_exact'] and report['generated_token_ids_exact']
    assert report['selection_replay_exact'] and report['selection_records']==116
    assert not report['historical_selection_raw_parity_claimed']
    assert report['publication_performed'] is False and report['full2_GPU_fit_claimed'] is False
    assert report['phases']['source']['math_contexts']==report['phases']['archive']['math_contexts']
    context=report['phases']['source']['math_contexts']
    assert context['internal116']['active']['matmul_allow_tf32'] is True
    assert context['internal116']['active']['torch_threads']==8
    assert context['internal116']['before']==context['internal116']['after']==context['native64']
    assert context['native64']['torch_threads']==4


@pytest.mark.parametrize('which',['tokens','parameters'])
def test_corruption_never_emits_passed_or_publishes_source(fixture,which):
    plan,out=fixture;runtime=FakeRuntime(mutate=which=='parameters',wrong_tokens=which=='tokens')
    with pytest.raises(ValueError):p.execute(plan,out,nonce='a'*64,runtime=runtime,guard=FakeGuard())
    report=p.read(out/'report.json');assert report['status']=='failed' and not report['publication_performed']


def test_existing_output_and_managed_source_namespace_refused(fixture):
    plan,out=fixture;out.mkdir()
    with pytest.raises(ValueError,match='New isolated'):p.execute(plan,out,nonce='a'*64,runtime=FakeRuntime(),guard=FakeGuard())
    with pytest.raises(ValueError,match='inside managed'):p.execute(plan,Path(plan['run'])/'inside',nonce='a'*64,runtime=FakeRuntime(),guard=FakeGuard())


def test_saved_selection_metric_and_protocol_strict_but_historic_raw_optional(fixture):
    plan,out=fixture;selection=FakeRuntime().selection(None,{},lambda:None)
    proof=p.check_saved_selection(plan,selection)
    assert proof['saved_metric_and_protocol_exact'] and not proof['historical_raw_rows_present']
    path=out.parent/'history.jsonl'
    historical=[dict(copy.deepcopy(r),validation_step=1) for r in selection['predictions']]
    path.write_text(''.join(json.dumps(r)+'\n' for r in historical))
    plan['saved_selection_files']['predictions.jsonl']={'path':str(path)}
    assert p.check_saved_selection(plan,selection)['historical_raw_row_parity_claimed']
    historical[0]['generation']['generated_token_ids']=[7,2]
    path.write_text(''.join(json.dumps(r)+'\n' for r in historical))
    proof=p.check_saved_selection(plan,selection)
    assert not proof['historical_raw_row_parity_claimed'] and proof['historical_raw_mismatched_record_ids']==['0']
    bad=copy.deepcopy(selection);bad['metrics']['group_macro_score']=.6
    with pytest.raises(ValueError,match='metric/protocol'):p.check_saved_selection(plan,bad)


def test_cli_without_allow_GPU_leaves_gate_closed(tmp_path):
    result=subprocess.run([sys.executable,str(p.ROOT/'scripts/probe_research_small_archive_gpu_parity.py'),
        '--cpu-report',str(tmp_path/'missing.json'),'--output',str(tmp_path/'never_created')],
        cwd=p.ROOT,capture_output=True,text=True)
    assert result.returncode==2 and json.loads(result.stdout)['status']=='gate_closed'
    assert not (tmp_path/'never_created').exists()


def test_exact_parent_nonce_deadline_and_GPU_exclusivity_before_cuda(monkeypatch,tmp_path):
    from scripts import probe_research_parallel_b1 as proc
    import os
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    parent={'pid':os.getppid(),'start_time':123,'command_sha256':'a'*64,'argv':['parent.py'],'state':'S'}
    own={'pid':os.getpid(),'start_time':456,'command_sha256':'b'*64,'argv':['helper.py'],'state':'R'}
    monkeypatch.setattr(proc,'process_info',lambda pid:parent if pid==parent['pid'] else own)
    monkeypatch.setattr(proc,'gpu_snapshot',lambda:{'processes':[],'free_bytes':90*p.GIB,'total_bytes':96*p.GIB,'gpu_uuid':'GPU-fixture'})
    armed=[];monkeypatch.setattr(proc,'arm_parent_death',lambda pid:armed.append(pid))
    monkeypatch.setattr(p.time,'monotonic',lambda:100.)
    expected={k:parent[k] for k in ('pid','start_time','command_sha256')}
    with pytest.raises(ValueError,match='coordinator parent'):
        p.ResourceGuard({'run':str(tmp_path),'reserve_gb':120},'a'*64,expected_parent=dict(expected,start_time=124),coordinator_deadline=300.)
    with pytest.raises(ValueError,match='deadline'):
        p.ResourceGuard({'run':str(tmp_path),'reserve_gb':120},'a'*64,expected_parent=expected,coordinator_deadline=99.)
    guard=p.ResourceGuard({'run':str(tmp_path),'reserve_gb':120},'a'*64,expected_parent=expected,coordinator_deadline=300.)
    assert guard.deadline==300. and armed==[parent['pid']]
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-fixture'
    monkeypatch.setattr(proc,'gpu_snapshot',lambda:{'processes':[{'pid':99999,'bytes':1}],'free_bytes':80*p.GIB,'total_bytes':96*p.GIB,'gpu_uuid':'GPU-fixture'})
    with pytest.raises(RuntimeError,match='Foreign GPU'):guard.check()


@pytest.mark.parametrize('precision',['highest','medium'])
@pytest.mark.parametrize('interrupted',[False,True])
def test_internal116_TF32_threads8_restores_exact_native_context_on_error(precision,interrupted):
    import torch
    original=p.math_context()
    try:
        torch.set_num_threads(2);torch.set_float32_matmul_precision(precision)
        native=p.math_context();receipt=None
        def work():
            nonlocal receipt
            with p.internal_selection_profile() as receipt:
                assert torch.backends.cuda.matmul.allow_tf32 is True
                assert torch.get_num_threads()==8
                if interrupted:raise InterruptedError('synthetic116 interruption')
        if interrupted:
            with pytest.raises(InterruptedError,match='synthetic116'):work()
        else:work()
        assert p.math_context()==native
        assert receipt['before']==receipt['after']==native
        assert receipt['active']['matmul_allow_tf32'] is True
        assert receipt['active']['torch_threads']==8
    finally:
        torch.set_num_threads(original['torch_threads'])
        torch.backends.cuda.matmul.allow_tf32=original['matmul_allow_tf32']
        torch.set_float32_matmul_precision(original['float32_matmul_precision'])


def test_execute_native64_unchanged_and_both_internal116_profiles_match(fixture):
    import torch
    original=p.math_context();observed={'native':[],'selection':[]}
    class ObservedRuntime(FakeRuntime):
        def generate(self,*args):
            observed['native'].append(p.math_context());return super().generate(*args)
        def selection(self,*args):
            observed['selection'].append(p.math_context());return super().selection(*args)
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        plan,out=fixture;report=p.execute(plan,out,nonce='a'*64,runtime=ObservedRuntime(),guard=FakeGuard())
        assert len(observed['native'])==128 and len(observed['selection'])==2
        assert all(x==observed['native'][0] for x in observed['native'])
        assert observed['native'][0]['matmul_allow_tf32'] is False
        assert observed['native'][0]['torch_threads']==4
        assert observed['selection'][0]==observed['selection'][1]
        assert observed['selection'][0]['matmul_allow_tf32'] is True
        assert observed['selection'][0]['torch_threads']==8
        assert report['runtime_environment']['native64_context']==observed['native'][0]
    finally:
        torch.set_num_threads(original['torch_threads'])
        torch.backends.cuda.matmul.allow_tf32=original['matmul_allow_tf32']
        torch.set_float32_matmul_precision(original['float32_matmul_precision'])


def test_sampled_host_floor_uses_charged_cgroup_and_no_cache_credit(monkeypatch,tmp_path):
    from scripts import probe_research_parallel_b1 as proc
    import os
    from types import SimpleNamespace
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    parent={'pid':os.getppid(),'start_time':123,'command_sha256':'a'*64,'argv':['parent.py'],'state':'S'}
    own={'pid':os.getpid(),'start_time':456,'command_sha256':'b'*64,'argv':['helper.py'],'state':'R'}
    monkeypatch.setattr(proc,'process_info',lambda pid:parent if pid==parent['pid'] else own)
    monkeypatch.setattr(proc,'gpu_snapshot',lambda:{'processes':[],'free_bytes':90*p.GIB,'total_bytes':96*p.GIB,'gpu_uuid':'GPU-fixture'})
    monkeypatch.setattr(proc,'arm_parent_death',lambda pid:None)
    monkeypatch.setattr(p.time,'monotonic',lambda:100.)
    monkeypatch.setattr(p.shutil,'disk_usage',lambda path:SimpleNamespace(free=200*p.GIB))
    expected={k:parent[k] for k in ('pid','start_time','command_sha256')}
    guard=p.ResourceGuard({'run':str(tmp_path),'reserve_gb':120},'a'*64,expected_parent=expected,coordinator_deadline=300.)
    host={'MemAvailable_bytes':150*p.GIB,'memory.max':171*p.GIB,'memory.current':139*p.GIB}
    monkeypatch.setattr(p,'host_snapshot',lambda:dict(host))
    guard.check();assert guard.samples[-1]['conservative_host_headroom_bytes']==32*p.GIB
    host['memory.current']+=1
    with pytest.raises(RuntimeError,match='host32GiB'):guard.check()
    assert guard.samples[-1]['conservative_host_headroom_bytes']==32*p.GIB-1
    host['memory.current']=100*p.GIB;host['MemAvailable_bytes']=31*p.GIB
    with pytest.raises(RuntimeError,match='host32GiB'):guard.check()


@pytest.fixture
def admitted_metadata(tmp_path,monkeypatch):
    from src.utils import research_trained_delta_probe as probe,research_trained_delta_probe_v2 as adapter
    from src.utils import research_checkpoint_archival as publication
    import scripts.validate_research_checkpoint as validator
    run=tmp_path/'run';run.mkdir();source=run/'state.pt';source.write_bytes(b'original trained source'*10)
    archive=tmp_path/'derived/archive.pt';archive.parent.mkdir();archive.write_bytes(b'compressed')
    manifest=tmp_path/'manifest.jsonl';manifest.write_text('{}\n')
    (run/'selection').mkdir()
    for suffix,content in [('metrics.json',{'group_macro_score':.5,'validation_step':1}),('protocol.json',{'validation_step':1})]:
        (run/'selection'/('best.'+suffix)).write_text(json.dumps(content))
    role={'source':str(source),'stat':p.stat(source),'marker':{'step':1,'metric':.5},'object':'owned_fixture'}
    gate={'run':str(run),'selected':role,'roles':{'best':role,'last':copy.deepcopy(role)},
        'config':{'vision':'google/siglip2-so400m-patch14-384','selection_metric':'group_macro','selection_max_new_tokens':64}}
    report={'protocol':probe.OWNER,'status':'verified_actual_trained_derived_copy_only',
        'all_payload_fields_and_backing_bits_exact':True,'scheduler_rng_config_compaction_metadata_preserved':True,
        'source_payload_sha256':'c'*64,'restored_payload_sha256':'c'*64,'decoded_model_bytes':16,
        'source_modified':False,'manager_roles_modified':False,'archive_deleted':False,
        'metadata_adapter':{'source_sha256':{'synthetic_adapter':'a'*64}},'code_sha256':{p.SOURCES[0]:p.sha(p.ROOT/p.SOURCES[0])},
        'run':str(run),'role':'best','source_roles_before_after':copy.deepcopy(gate['roles']),'source_stat':p.stat(source),
        'source_sha256':p.sha(source),'archive_sha256':p.sha(archive),'archive_path':str(archive),
        'archive_bytes':archive.stat().st_size,'normal_storage_bytes':16}
    path=tmp_path/'verified.json';path.write_text(json.dumps(report))
    monkeypatch.setattr(p,'verify_release',lambda:{})
    monkeypatch.setattr(probe,'BASE_ELEMENTS',4)
    monkeypatch.setattr(probe,'gate',lambda *a,**kw:copy.deepcopy(gate))
    monkeypatch.setattr(adapter,'source_binding',lambda:{'synthetic_adapter':'a'*64})
    monkeypatch.setattr(publication,'code_binding',lambda:{'test_fixture':'a'*64})
    monkeypatch.setattr(validator,'check_manifest_hashes',lambda config:None)
    monkeypatch.setattr(p,'MANIFEST',manifest);monkeypatch.setattr(p,'MANIFEST_SHA256',p.sha(manifest))
    return path,report,source,archive


def test_positive_metadata_gate_opens_without_source_tensor_bytes_or_GPU(admitted_metadata,monkeypatch):
    path,report,source,archive=admitted_metadata;original=Path.open
    def trap(candidate,*args,**kwargs):
        assert Path(candidate) not in (source,archive),'Metadata preflight must not read weight/archive payloads'
        return original(candidate,*args,**kwargs)
    monkeypatch.setattr(Path,'open',trap)
    plan=p.prepare_plan(path)
    assert plan['source']==str(source) and plan['archive']==str(archive)
    assert plan['actual_gpu_validated'] is False and plan['publication_performed'] is False


@pytest.mark.parametrize('mutation',['deleted','changed_source','payload_digest','adapter','codec','source_stat','archive_size','selection_step','roles'])
def test_invalid_actual_CPU_proof_metadata_keeps_GPU_gate_closed(admitted_metadata,mutation):
    path,report,source,archive=admitted_metadata
    if mutation=='deleted':report['archive_deleted']=True
    elif mutation=='changed_source':report['source_modified']=True
    elif mutation=='payload_digest':report['source_payload_sha256']=report['restored_payload_sha256']=None
    elif mutation=='adapter':report['metadata_adapter']['source_sha256']={}
    elif mutation=='codec':report['code_sha256']={p.SOURCES[0]:'0'*64}
    elif mutation=='source_stat':report['source_stat']['st_size']+=1
    elif mutation=='archive_size':report['archive_bytes']+=1
    elif mutation=='selection_step':
        saved=source.parent/'selection/best.metrics.json';saved.write_text(json.dumps({'group_macro_score':.5,'validation_step':2}))
    else:report['source_roles_before_after']['last']['object']='foreign_role'
    path.write_text(json.dumps(report));result=p.preflight(path)
    assert result['status']=='gate_closed' and result['GPU_launched'] is False
