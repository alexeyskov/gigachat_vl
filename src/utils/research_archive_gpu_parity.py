"""Isolated future actual4f source/archive GPU parity; import is CPU metadata only.

No managed pointer is changed. Actual execution is closed until a surviving
trained-source CPU proof exists and ROOT explicitly schedules an idle GPU.
"""
from contextlib import contextmanager
from datetime import datetime,timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
OWNER='actual_trained_small_archive_gpu_parity_v1'
READY=ROOT/'research_results/trained_small_archive_gpu_parity_ready_v2.json'
PUBLICATION_READY=ROOT/'research_results/completed_small_archival_publication_ready.json'
PUBLICATION_READY_SHA256='2afe3cd9f7ae60b10a23c96912fbf0016749b3beb09e01ffd035df02c3ffc22d'
MANIFEST=ROOT/'datasets/research/eval-heldout-v3/evaluation.jsonl'
MANIFEST_SHA256='9fcb6783de70f44a94505eefa4ee90eef5759319c1db59850bd351777f38f5f4'
GIB=2**30
CHUNK_BYTES=8*2**20
MAX_OUTPUT=64*2**20
SOURCES=('src/utils/research_archive_gpu_parity.py','scripts/probe_research_small_archive_gpu_parity.py',
    'tests/test_research_archive_gpu_parity.py','scripts/validate_research_checkpoint.py',
    'src/eval/research_inference.py','src/eval/research_selection.py','src/eval/research_metrics.py',
    'src/eval/research_parallel_protocol.py','scripts/probe_research_parallel_b1.py','scripts/run_research_stage.py',
    'src/model/smallthinker.py','src/model/gigachat_vl.py','src/model/research_moe.py',
    'src/model/research_vision.py','src/model/research_quantization.py','src/dataset/vision_cache.py',
    'src/model/visual_expert_gates.py','src/model/text_moe_backbones.py',
    'third_party/vqa_normalization/tables.json','tests/test_research_inference.py')


class GateClosed(ValueError):pass


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def stat(path):
    s=Path(path).stat();return {k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
def sha(path,check=None):
    before=stat(path);h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk:=handle.read(CHUNK_BYTES):
            if check is not None:check()
            h.update(chunk)
    if stat(path)!=before:raise ValueError('Archive parity input changed while hashing')
    return h.hexdigest()
def read(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size>16*2**20:raise GateClosed('Bounded local parity metadata required')
    return json.loads(p.read_bytes())
def rows(path):
    with Path(path).open('rb') as handle:return [json.loads(line) for line in handle if line.strip()]
def binding(path,*,expected=None):
    p=Path(path).resolve(strict=True);s=stat(p);h=sha(p)
    if expected is not None and h!=expected:raise GateClosed('Source-bound parity input digest differs')
    return {'path':str(p),'stat':s,'sha256':h}
def source_binding():return {p:sha(ROOT/p) for p in SOURCES}


def verify_release():
    if sha(PUBLICATION_READY)!=PUBLICATION_READY_SHA256:raise GateClosed('Reviewed publication proposal changed')
    publication=read(PUBLICATION_READY)
    if publication.get('status')!='tiny_cpu_tests_passed_actual_trained_gpu_gate_closed':raise GateClosed('Publication proposal is not reviewed')
    for path,h in publication['source_sha256'].items():
        if sha(ROOT/path)!=h:raise GateClosed('Publication/decoder source changed')
    ready=read(READY)
    if (ready.get('status')!='cpu_tests_passed_actual_gpu_gate_closed' or ready.get('actual_gpu_validated') is not False or
            ready.get('source_sha256')!=source_binding() or type(ready.get('tests_passed')) is not int or ready['tests_passed']<1):
        raise GateClosed('GPU parity helper CPU release is absent or changed')
    return {'helper_ready':binding(READY),'publication_ready':binding(PUBLICATION_READY)}


def prepare_plan(cpu_report,*,reserve_gb=120):
    """No Torch, source weights or CUDA: exact artifact bytes checked at admission."""
    if not math.isfinite(reserve_gb) or reserve_gb<120:raise GateClosed('Preserve120GiB reserve')
    release=verify_release()
    from src.utils import research_trained_delta_probe as probe,research_trained_delta_probe_v2 as adapter
    from src.utils import research_checkpoint_archival as publication
    report=read(cpu_report)
    if (report.get('protocol')!=probe.OWNER or report.get('status')!='verified_actual_trained_derived_copy_only' or
            report.get('all_payload_fields_and_backing_bits_exact') is not True or
            report.get('scheduler_rng_config_compaction_metadata_preserved') is not True or
            report.get('source_payload_sha256')!=report.get('restored_payload_sha256') or
            report.get('decoded_model_bytes')!=probe.BASE_ELEMENTS*4 or report.get('source_modified') is not False or
            report.get('manager_roles_modified') is not False or report.get('archive_deleted') is not False or
            report.get('metadata_adapter',{}).get('source_sha256')!=adapter.source_binding()):
        raise GateClosed('Actual trained4f exact surviving CPU archive proof required')
    for name in ('source_sha256','archive_sha256','source_payload_sha256','restored_payload_sha256'):
        h=report.get(name)
        if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
            raise GateClosed('Actual CPU proof lacks canonical raw-byte digests')
    if not report.get('code_sha256') or any(sha(ROOT/name)!=h for name,h in report['code_sha256'].items()):
        raise GateClosed('CPU probe/decoder source differs from its actual proof')
    gate=probe.gate(report['run'],report['role']);config=gate['config']
    if (config.get('vision')!='google/siglip2-so400m-patch14-384' or config.get('selection_metric')!='group_macro' or
            config.get('selection_max_new_tokens',64)!=64 or gate['roles']!=report['source_roles_before_after'] or
            gate['selected']['stat']!=report['source_stat']):raise GateClosed('Authoritative original4f roles/profile changed')
    archive=Path(report['archive_path'])
    if (archive.is_symlink() or not archive.is_file() or archive.parent==Path(gate['run']) or
            archive.resolve().is_relative_to(Path(gate['run'])) or archive.stat().st_size!=report['archive_bytes'] or
            not 0<report['archive_bytes']<report['source_stat']['st_size']):raise GateClosed('Surviving separate beneficial derived archive required')
    from scripts.validate_research_checkpoint import check_manifest_hashes
    check_manifest_hashes(config)
    if sha(MANIFEST)!=MANIFEST_SHA256:raise GateClosed('Fixed balanced512 manifest changed')
    source=Path(gate['selected']['source'])
    selected_files={}
    for name in ('metrics.json','protocol.json','predictions.jsonl'):
        path=Path(gate['run'])/'selection'/(report['role']+'.'+name)
        if path.exists():selected_files[name]=binding(path)
    if not {'metrics.json','protocol.json'}<=set(selected_files):
        raise GateClosed('Saved source selection metric and protocol artifacts required')
    saved_metrics=read(selected_files['metrics.json']['path']);saved_protocol=read(selected_files['protocol.json']['path'])
    if (saved_metrics.get('validation_step')!=gate['selected']['marker']['step'] or
            saved_protocol.get('validation_step')!=gate['selected']['marker']['step'] or
            saved_metrics.get('group_macro_score')!=gate['selected']['marker']['metric']):
        raise GateClosed('Original selection artifact role/step/metric differs')
    return {'schema_version':1,'owner':OWNER,'run':gate['run'],'role':report['role'],'gate':gate,
        'cpu_report':binding(cpu_report),'release':release,'code_sha256':publication.code_binding(),
        'helper_source_sha256':source_binding(),'source':str(source),'source_sha256':report['source_sha256'],
        'source_stat':report['source_stat'],'archive':str(archive.resolve()),'archive_sha256':report['archive_sha256'],
        'archive_stat':stat(archive),'source_payload_sha256':report['source_payload_sha256'],
        'restored_payload_sha256':report['restored_payload_sha256'],'manifest':binding(MANIFEST),
        'reserve_gb':reserve_gb,'decoded_model_bytes':report['decoded_model_bytes'],
        'normal_storage_bytes':report['normal_storage_bytes'],'source_authoritative':True,
        'saved_selection_files':selected_files,
        'publication_performed':False,'actual_gpu_validated':False}


def preflight(cpu_report,**kwargs):
    try:return {'status':'metadata_preflight_only_no_GPU','plan':prepare_plan(cpu_report,**kwargs),'GPU_launched':False}
    except (OSError,ValueError,KeyError,TypeError) as error:return {'status':'gate_closed','reason':str(error),'GPU_launched':False}


def assert_plan(plan,*,rehash=False,check=None):
    from src.utils import research_trained_delta_probe as probe,research_checkpoint_archival as publication
    if probe.gate(plan['run'],plan['role'])!=plan['gate'] or publication.code_binding()!=plan['code_sha256']:
        raise ValueError('Authoritative source roles/config/decoder changed')
    if source_binding()!=plan['helper_source_sha256']:raise ValueError('Parity helper/native source changed')
    for item in (plan['cpu_report'],plan['manifest'],*plan['release'].values(),*plan['saved_selection_files'].values()):
        if stat(item['path'])!=item['stat'] or sha(item['path'])!=item['sha256']:raise ValueError('Parity metadata/source inputs changed')
    for path,s,h in ((plan['source'],plan['source_stat'],plan['source_sha256']),
            (plan['archive'],plan['archive_stat'],plan['archive_sha256'])):
        if stat(path)!=s or (rehash and sha(path,check)!=h):raise ValueError('Source/derived checkpoint bytes changed')


def tensor_census(model,check=None):
    """All registered parameters+buffers, plus the plain extractor's frozen modules.

    Raw bytes are hashed in8MiB chunks; aliases compare by canonical name groups,
    never by allocator address across fresh models. Nonpersistent buffers count.
    """
    import torch
    roots=[('model',model)];extractor=getattr(model,'extractor',None)
    if extractor is not None:
        for key in ('vision_tower','vision_projector'):
            child=getattr(extractor,key,None)
            if isinstance(child,torch.nn.Module):roots.append(('extractor.'+key,child))
    tensors={};aliases={};seen={};storages={};total=0
    for prefix,module in roots:
        for kind,iterator in (('parameter',module.named_parameters(remove_duplicate=False)),('buffer',module.named_buffers(remove_duplicate=False))):
            for name,value in iterator:
                full=prefix+'.'+name
                if value.device.type=='meta' or not value.is_contiguous():raise ValueError('Actual contiguous materialized parity tensors required')
                if value.requires_grad:raise ValueError('All parity parameters must be frozen for native generation')
                aliases.setdefault(id(value),[]).append(full)
                storage=value.untyped_storage();sk=(str(value.device),storage.data_ptr(),storage.nbytes())
                storages.setdefault(sk,[]).append(full)
                if id(value) not in seen:
                    flat=value.detach().reshape(-1).view(torch.uint8);h=hashlib.sha256()
                    for chunk in flat.split(CHUNK_BYTES):
                        if check is not None:check()
                        h.update(chunk.cpu().numpy().tobytes())
                    seen[id(value)]=h.hexdigest();total+=flat.numel()
                tensors[full]={'kind':kind,'device':str(value.device),'dtype':str(value.dtype),'shape':list(value.shape),'stride':list(value.stride()),
                    'storage_offset':value.storage_offset(),'bytes':value.numel()*value.element_size(),'sha256':seen[id(value)]}
    if not tensors or not any('projector.' in n for n in tensors):raise ValueError('Complete native model and frozen projector census required')
    return {'tensors':tensors,'tensor_object_alias_groups':sorted(sorted(v) for v in aliases.values()),
        'storage_alias_groups':sorted([{'names':sorted(v),'storage_bytes':key[2]} for key,v in storages.items()],key=lambda x:x['names']),
        'unique_tensor_objects':len(seen),
        'unique_tensor_bytes_hashed':total,'comparison':'raw_SHA256_8MiB_chunks_exact_dtype_shape_stride_aliases_no_numeric_tolerance'}


def assert_equal_census(source,archive):
    if source!=archive:
        keys=set(source.get('tensors',{}))|set(archive.get('tensors',{}))
        changed=[k for k in sorted(keys) if source.get('tensors',{}).get(k)!=archive.get('tensors',{}).get(k)]
        raise ValueError('Source/archive model tensors or alias topology differ: '+str(changed[:8]))


def compare_generations(source,archive):
    from src.eval.research_parallel_protocol import GEN_FIELDS
    if len(source)!=len(archive):raise ValueError('Source/archive generation record counts differ')
    for a,b in zip(source,archive):
        if a.get('record_id')!=b.get('record_id') or any(a.get(k)!=b.get(k) for k in GEN_FIELDS):
            raise ValueError('Source/archive native tokens or prompt metadata differ')


def check_saved_selection(plan,selection):
    """Saved metric/protocol replay is distinct from optional historic raw rows."""
    files=plan['saved_selection_files']
    saved_metrics=read(files['metrics.json']['path']);saved_protocol=read(files['protocol.json']['path'])
    if ({k:v for k,v in saved_metrics.items() if k!='validation_step'}!=selection['metrics'] or
            {k:v for k,v in saved_protocol.items() if k!='validation_step'}!=selection['protocol']):
        raise ValueError('Original saved internal selection metric/protocol is not replayed exactly')
    historical=False;mismatches=[]
    if 'predictions.jsonl' in files:
        saved=rows(files['predictions.jsonl']['path'])
        if any(row.get('validation_step')!=plan['gate']['selected']['marker']['step'] for row in saved):
            raise ValueError('Historical selection raw rows belong to a different step')
        observed=[{k:v for k,v in row.items() if k!='validation_step'} for row in saved]
        if [r.get('record_id') for r in observed]!=[r.get('record_id') for r in selection['predictions']]:
            raise ValueError('Historical selection record scope/order differs')
        mismatches=[a['record_id'] for a,b in zip(observed,selection['predictions']) if a!=b]
        historical=True
    return {'saved_metric_and_protocol_exact':True,'historical_raw_rows_present':historical,
        'historical_raw_row_parity_claimed':historical and not mismatches,
        'historical_raw_mismatched_record_ids':mismatches,
        'historical_raw_comparison_scope':'observational; source-vs-archive raw equality and saved metric/protocol replay are mandatory'}


def host_snapshot():
    result={}
    for name in ('memory.current','memory.max'):
        p=Path('/sys/fs/cgroup')/name
        if p.is_file():
            text=p.read_text().strip();result[name]=None if text=='max' else int(text)
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,rest=line.split(':',1)
        if key in ('MemAvailable','MemTotal'):result[key+'_bytes']=int(rest.split()[0])*1024
    return result


def conservative_host_headroom(snapshot):
    """Observed uncharged headroom only; resident file pages receive no credit."""
    available=snapshot.get('MemAvailable_bytes')
    if type(available) is not int or available<0:raise RuntimeError('Actual host MemAvailable observation missing')
    maximum=snapshot.get('memory.max');current=snapshot.get('memory.current')
    if maximum is not None:
        if type(maximum) is not int or type(current) is not int or maximum<0 or current<0:
            raise RuntimeError('Actual bounded cgroup memory observation missing')
        available=min(available,max(0,maximum-current))
    return available


def math_context():
    import torch
    try:precision=torch.get_float32_matmul_precision()
    except RuntimeError as error:
        # Newer Torch reports no single aggregate precision after the frozen
        # legacy CUDA-only switch with a pre-existing CPU medium policy.
        if 'mix of the legacy and new APIs' not in str(error):raise
        precision='mixed_backend_policy_no_single_aggregate'
    return {'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision':precision,
        'torch_threads':torch.get_num_threads()}


@contextmanager
def internal_selection_profile():
    """Match frozen trainer/validator116 math; restore ordinary native math."""
    import torch
    before=math_context();receipt={'before':before}
    if before['float32_matmul_precision'] not in ('highest','high','medium'):
        raise RuntimeError('Unrestorable pre-existing mixed matmul policy refused')
    try:
        torch.backends.cuda.matmul.allow_tf32=True
        torch.set_num_threads(8)
        receipt['active']=math_context()
        yield receipt
    finally:
        torch.set_num_threads(before['torch_threads'])
        torch.backends.cuda.matmul.allow_tf32=before['matmul_allow_tf32']
        # Restore medium as medium, not merely its equivalent allow_tf32=True.
        torch.set_float32_matmul_precision(before['float32_matmul_precision'])
        receipt['after']=math_context()
        if receipt['after']!=before:raise RuntimeError('Internal116 math context was not restored exactly')


class ResourceGuard:
    """Exclusive own-process observations; no controller or peer is signalled."""
    def __init__(self,plan,nonce,timeout=1200,*,expected_parent=None,coordinator_deadline=None):
        from scripts import probe_research_parallel_b1 as proc
        if not isinstance(nonce,str) or len(nonce)!=64 or any(c not in '0123456789abcdef' for c in nonce):raise ValueError('Explicit64hex root scheduling nonce required')
        if type(timeout) is not int or not 120<=timeout<=1800:raise ValueError('Bounded120..1800s parity deadline required')
        self.proc=proc;self.plan=plan;self.nonce=nonce;self.parent=proc.process_info(os.getppid());self.own=proc.process_info(os.getpid())
        if self.parent is None or self.own is None:raise ValueError('Actual parent/PID birth/argv identity missing')
        if (not isinstance(expected_parent,dict) or set(expected_parent)!={'pid','start_time','command_sha256'} or
                any(self.parent.get(k)!=v for k,v in expected_parent.items())):
            raise ValueError('Actual immediate coordinator parent differs from ROOT scheduling identity')
        if (type(coordinator_deadline) not in (float,int) or not math.isfinite(coordinator_deadline) or
                not time.monotonic()<coordinator_deadline<=time.monotonic()+1800):
            raise ValueError('Explicit live bounded coordinator lease deadline required')
        self.coordinator_deadline=coordinator_deadline
        self.deadline=min(time.monotonic()+timeout,coordinator_deadline);self.samples=[];self.error=None;self.phase='admission'
        self.stop=threading.Event();self.thread=None;self.initial=proc.gpu_snapshot()
        if self.initial['processes'] or self.initial['free_bytes']<48*GIB:raise RuntimeError('Root must schedule an exclusive idle GPU with48GiB free')
        self.uuid=self.initial['gpu_uuid'];os.environ['CUDA_VISIBLE_DEVICES']=self.uuid
        proc.arm_parent_death(self.parent['pid'])
    def fast_check(self):
        if self.error is not None:raise RuntimeError('Parity resource observation failed: '+self.error)
        if time.monotonic()>=self.deadline:raise TimeoutError('Root-scheduled archive parity deadline expired')
    def check(self):
        self.fast_check()
        if not self.proc.same_process(self.parent,self.proc.process_info(self.parent['pid'])):raise RuntimeError('Bound parent birth/argv changed')
        gpu=self.proc.gpu_snapshot()
        if gpu['gpu_uuid']!=self.uuid or gpu['free_bytes']<24*GIB:raise RuntimeError('Parity GPU UUID/free-reserve changed')
        if any(p['pid']!=self.own['pid'] for p in gpu['processes']):raise RuntimeError('Foreign GPU process present; own parity aborts')
        if any(p['bytes']>48*GIB for p in gpu['processes']):raise RuntimeError('Parity own NVML48GiB ceiling exceeded')
        if shutil.disk_usage(self.plan['run']).free<self.plan['reserve_gb']*GIB+MAX_OUTPUT:raise OSError('Parity120GiB disk reserve reached')
        host=host_snapshot();headroom=conservative_host_headroom(host)
        if len(self.samples)<8192:self.samples.append({'monotonic':time.monotonic(),'phase':self.phase,
            'gpu':gpu,'host':host,'conservative_host_headroom_bytes':headroom})
        if headroom<32*GIB:raise RuntimeError('Parity sampled host32GiB headroom floor reached; no cache credit')
        return gpu
    def _watch(self):
        while not self.stop.wait(.5):
            try:self.check()
            except BaseException as error:self.error=type(error).__name__+': '+str(error);return
    def __enter__(self):
        self.check();self.thread=threading.Thread(target=self._watch,daemon=True);self.thread.start();return self
    def __exit__(self,*exc):self.stop.set();self.thread.join(timeout=6)


class NativeRuntime:
    def load(self,path,cache_dir,reserve_gb):
        from src.eval.research_inference import load_research_checkpoint
        return load_research_checkpoint(path,cache_dir=cache_dir,reserve_gb=reserve_gb)
    def generate(self,model,record):
        from src.eval.research_inference import generate_prediction
        from src.eval.research_parallel_protocol import public_record
        return generate_prediction(model,public_record(record),max_new_tokens=64)
    def selection(self,model,config,check):
        from scripts.validate_research_checkpoint import select_validation_records
        from scripts.run_research_stage import load_records
        from src.eval.research_selection import evaluate_internal_group_macro
        from src.eval.research_inference import generate_prediction
        selected=select_validation_records(config,load_records)
        if len(selected['records'])!=116:raise ValueError('Actual frozen internal selection must be116 records')
        def guarded(model,record,**kwargs):
            check();result=generate_prediction(model,record,**kwargs);check();return result
        return evaluate_internal_group_macro(model,selected,max_new_tokens=64,generator=guarded)


def write(path,value):
    p=Path(path);raw=canonical(value)+b'\n'
    if p.is_symlink() or len(raw)>16*2**20:raise ValueError('Bounded own parity metadata only')
    tmp=p.with_name('.'+p.name+'.tmp')
    try:
        with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:tmp.unlink(missing_ok=True)


def execute(plan,output,*,nonce,timeout=1200,cache_dir='cache/vision',runtime=None,guard=None,
            expected_parent=None,coordinator_deadline=None):
    """CLI cannot inject runtime/guard; injected unit cases never activate ROOT."""
    injected=runtime is not None or guard is not None
    if not injected:verify_release()
    output=Path(output)
    if output.exists() or output.is_symlink():raise ValueError('New isolated parity output required')
    if output.resolve().is_relative_to(Path(plan['run'])):raise ValueError('No parity output inside managed source run')
    if Path(plan['run']).is_relative_to(output.resolve()):raise ValueError('Parity output cannot contain managed source run')
    assert_plan(plan)
    engine=runtime or NativeRuntime();resource=guard or ResourceGuard(plan,nonce,timeout,
        expected_parent=expected_parent,coordinator_deadline=coordinator_deadline)
    output.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'protocol':OWNER,'status':'running','producer':'CPU_test_fixture' if injected else 'ROOT',
        'device':'cpu' if injected else 'cuda','run':plan['run'],'role':plan['role'],'source_authoritative':True,
        'publication_performed':False,'nonce':nonce,'admission':plan,'phases':{}}
    write(output/'report.json',report)
    try:
        with resource:
            from src.utils.research_lossless_delta import memory_guard
            unique_roles={r['source']:r for r in plan['gate'].get('roles',{}).values()}
            role_bytes=sum(r['stat']['st_size'] for r in unique_roles.values()) or plan['source_stat']['st_size']
            resource.phase='source_and_archive_admission_hashing';resource.check()
            report['memory_guard_before_models']=memory_guard(role_bytes+plan['archive_stat']['st_size']+plan['decoded_model_bytes']+12*GIB)
            fast=getattr(resource,'fast_check',resource.check)
            assert_plan(plan,rehash=True,check=fast);resource.check()
            role_hashes={path:{'stat':r['stat'],'sha256':sha(path,fast)} for path,r in unique_roles.items()}
            report['best_last_physical_sources_before']=role_hashes
            import torch
            torch.set_num_threads(4)
            versions={}
            for package in ('torch','transformers','peft','bitsandbytes','tokenizers','huggingface-hub'):
                try:versions[package]=importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:versions[package]=None
            report['runtime_environment']={'python':sys.version,'packages':versions,
                'torch_cuda_version':torch.version.cuda,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                'float32_matmul_precision':torch.get_float32_matmul_precision(),
                'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                'source_math':'unchanged frozen native B1 generation; internal116 scoped TF32=True and threads8 matching frozen trainer/validator',
                'native64_context':math_context(),
                'internal116_profile':{'matmul_allow_tf32':True,'torch_threads':8,'restores_native64_context':True}}
            if not injected:
                if torch.cuda.is_initialized():raise RuntimeError('Parity owns a fresh CUDA context')
                torch.cuda.set_per_process_memory_fraction(40*GIB/resource.initial['total_bytes'],device=0)
                report['runtime_environment'].update(GPU_name=torch.cuda.get_device_name(),
                    GPU_compute_capability=list(torch.cuda.get_device_capability()))
            from src.eval.research_parallel_protocol import select_guard_indices
            records=rows(plan['manifest']['path']);indices=select_guard_indices(records)
            for phase,path in (('source',plan['source']),('archive',plan['archive'])):
                resource.phase=phase+'_loading';resource.check();assert_plan(plan)
                if not injected:torch.cuda.reset_peak_memory_stats()
                model,provenance=engine.load(path,cache_dir,plan['reserve_gb']);resource.check()
                before=tensor_census(model,fast);generated=[]
                resource.phase=phase+'_native64'
                native_context=math_context()
                for index in indices:
                    resource.check();result=engine.generate(model,records[index]);resource.check()
                    generated.append(dict(result,record_id=records[index]['record_id']))
                    write(output/'progress.json',{'phase':resource.phase,'records':len(generated),'total_records':64})
                resource.phase=phase+'_internal116'
                with internal_selection_profile() as selection_context:
                    selection=engine.selection(model,plan['gate']['config'],resource.check)
                if math_context()!=native_context:raise RuntimeError('Ordinary native64 context changed after internal116')
                if selection['metrics']['group_macro_score']!=plan['gate']['selected']['marker']['metric']:
                    raise ValueError('Source saved internal selection metric is not replayed exactly')
                saved_selection=check_saved_selection(plan,selection)
                after=tensor_census(model,fast);assert_equal_census(before,after)
                phase_result={'provenance':provenance,'model_tensors':before,'native64':generated,'internal116':selection,
                    'math_contexts':{'native64':native_context,'internal116':selection_context},
                    'saved_selection_replay':saved_selection,
                    'projector_frozen':all(not p.requires_grad for p in model.projector.parameters()),
                    'all_parameter_and_buffer_bits_unchanged_after_forward':True}
                if not phase_result['projector_frozen']:raise ValueError('Aligned projector must remain frozen')
                if not injected:
                    phase_result.update(peak_pytorch_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_pytorch_reserved_bytes=torch.cuda.max_memory_reserved())
                report['phases'][phase]=phase_result;write(output/(phase+'.json'),phase_result)
                del model;gc.collect()
                if not injected:torch.cuda.empty_cache();torch.cuda.synchronize()
                resource.check()
            a,b=report['phases']['source'],report['phases']['archive']
            assert_equal_census(a['model_tensors'],b['model_tensors']);compare_generations(a['native64'],b['native64'])
            if (a['provenance']!=b['provenance'] or a['internal116']!=b['internal116'] or
                    a['math_contexts']!=b['math_contexts']):
                raise ValueError('Source/archive actual native provenance or internal116 token/score/protocol differs')
            assert_plan(plan,rehash=True,check=fast);resource.check()
            for path,old in role_hashes.items():
                if stat(path)!=old['stat'] or sha(path,fast)!=old['sha256']:raise ValueError('Original best/last physical bytes changed')
            samples=list(resource.samples)
            report.update(status='passed_synthetic_CPU_fixture' if injected else 'passed',
                source_state_sha256=plan['source_sha256'],archive_sha256=plan['archive_sha256'],
                source_payload_sha256=plan['source_payload_sha256'],restored_payload_sha256=plan['restored_payload_sha256'],
                all_model_tensors_exact=True,generated_token_ids_exact=True,generation_records=64,
                selection_replay_exact=True,selection_raw_source_archive_exact=True,selection_records=116,
                historical_selection_raw_parity_claimed=a['saved_selection_replay']['historical_raw_row_parity_claimed'],projector_frozen_unchanged=True,
                code_sha256=plan['code_sha256'],helper_source_sha256=plan['helper_source_sha256'],
                resources=samples,best_last_physical_sources_unchanged=True,
                sampled_peak_own_NVML_bytes=max((sum(p['bytes'] for p in x['gpu']['processes']) for x in samples),default=None),
                sampled_peak_cgroup_bytes=max((x['host']['memory.current'] for x in samples if x['host'].get('memory.current') is not None),default=None),
                sampled_min_GPU_free_bytes=min((x['gpu']['free_bytes'] for x in samples),default=None),
                resource_peak_scope='PyTorch allocator peaks are measured per phase; NVML/cgroup peaks are sampled, not instantaneous isolated peaks',
                full2_GPU_fit_claimed=False,finished_at=datetime.now(timezone.utc).isoformat())
            if not injected:report.update(parent=resource.parent,process=resource.own,
                expected_parent=expected_parent,coordinator_deadline_monotonic=resource.coordinator_deadline,
                deadline_monotonic=resource.deadline)
            write(output/'report.json',report)
            return report
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),resources=resource.samples)
        write(output/'report.json',report);raise
