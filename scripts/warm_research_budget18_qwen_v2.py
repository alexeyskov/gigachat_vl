#!/usr/bin/env python3
"""Bounded frozen-vision cache fill alongside the current text-only evaluation."""
from __future__ import annotations
import argparse,ctypes,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_budget18_queue_v1 as q
OUT=ROOT/'research_results/budget18_qwen_parallel_warmup_v2'
PROGRESS=ROOT/'logs/research/budget18_qwen_cache_progress_v2.json'
TARGET='smallthinker4__siglip2__1c__SFT__budget18_s42'
CONTROLLER_SHA='c75e29bbd4b8e54dc2dba90408c074dc60a4fd1c3408d92357e8a76b225109f5'
RELEASE_SHA='910448c7fd680ec949d6521e32716aee2832a99fc534d185f0a06b226c0ec1c0'


def worker(parent,seconds):
    if os.getppid()!=parent:raise RuntimeError('Vision worker parent differs')
    if ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise OSError('Cannot arm vision worker parent-death signal')
    if os.getppid()!=parent:raise RuntimeError('Vision coordinator disappeared')
    from src.utils import research_stock_determinism as stock
    stock.bootstrap_environment();stock.activate()
    import torch
    from src.utils.research_derived_vision_v1 import resolve_derived_vision,assert_binding,host_guard
    from src.model.research_vision import build_frozen_vision
    from src.dataset.vision_cache import VisionFeatureCache,image_content_hash
    binding=resolve_derived_vision('Qwen/Qwen2.5-VL-7B-Instruct',preprocessing_profile='qwen_native_bounded_729')
    assert_binding(binding,full=True,check=host_guard)
    from src.utils.research_lossless_delta import memory_guard
    memory_guard(4*2**30)
    if torch.cuda.mem_get_info()[0]<12*2**30:raise MemoryError('Vision warmup needs12GiB free GPU')
    encoder=build_frozen_vision(binding['path'],'cuda',torch.bfloat16,preprocessing_profile='qwen_native_bounded_729')
    if encoder.feature_dim!=3584 or any(p.requires_grad for p in encoder.vision_tower.parameters()):raise ValueError('Frozen Qwen-only tower required')
    cache=VisionFeatureCache(ROOT/'cache/vision/derived_vision_v1',binding['cache_fingerprint'],reserve_bytes=120*2**30)
    closure=ROOT/'research_results/budget18_cache_retirement_plan_v1/protected_image_closure.jsonl'
    if q.sha(closure)!='7acbb9b5e1a6022e5b7a09a0e02247a9754d8220a443570b9c1204eea68aeb2e':raise ValueError('Protected closure SHA changed')
    inputs=[json.loads(line) for line in closure.read_text().splitlines()]
    if len(inputs)!=7616 or len({r['image_hash'] for r in inputs})!=7616:raise ValueError('Exact protected RGB population required')
    population=len(inputs);inputs=[r for r in inputs if not cache.path_for_hash(r['image_hash']).exists()];scheduled=len(inputs)
    q.atomic(OUT/'scheduled.json',{'population':population,'missing_at_start':scheduled,'already_present_paths_not_rechecked':population-scheduled,'fingerprint':binding['cache_fingerprint'],'protected_closure_sha256':q.sha(closure)})
    stopped=False
    def stop(_n,_f):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    start=time.monotonic();done=0
    for row in inputs:
        if stopped or time.monotonic()-start>=seconds:break
        memory_guard(512*2**20)
        path=next((p for p in row['source_paths'] if Path(p).is_file()),None)
        if path is None or image_content_hash(path)!=row['image_hash']:raise ValueError('Protected image source pixels changed')
        output=cache.path_for_hash(row['image_hash']);existed=output.exists()
        cache.get_many([path],encoder.encode_paths,image_hashes=[row['image_hash']]);done+=1
        if not existed:
            # The native writer has fsynced/closed this worker's new output.
            # Evict only its clean file pages; retain exact features on disk.
            fd=os.open(output,os.O_RDONLY|os.O_NOFOLLOW)
            try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            finally:os.close(fd)
        if done%25==0:
            torch.cuda.synchronize()
            status={'owner':'budget18_frozen_Qwen_cache_fill_v2','pid':os.getpid(),'phase':'frozen_vision_cache_warmup',
                    'completed':done,'total':len(inputs),'population':population,'already_present_paths_not_rechecked':population-scheduled,'cache':dict(cache.stats),'elapsed_seconds':time.monotonic()-start,
                    'fingerprint':binding['cache_fingerprint'],'peak_vram_GiB':torch.cuda.max_memory_allocated()/2**30,
                    'LLM_or_MLP_constructed':False,'parameters_trained':0}
            q.atomic(PROGRESS,status);print(json.dumps(status),flush=True)
    torch.cuda.synchronize()
    q.atomic(OUT/'worker_completed.json',{'status':'completed_bounded_prefix','images_checked':done,'population':population,'missing_at_start':scheduled,'already_present_paths_not_rechecked':population-scheduled,
                'cache':dict(cache.stats),'fingerprint':binding['cache_fingerprint'],'elapsed_seconds':time.monotonic()-start,
                'parameters_trained':0,'worklist':'only absent paths in protected closure order','remaining_images_use_native_lazy_cache':done<scheduled})


def validate_target(service,parent,state,evaluation,request,proc):
    if (service['statename']!='RUNNING' or not parent or not evaluation or evaluation['ppid']!=parent['pid']
            or service['pid']!=parent['pid'] or request['kind']!='eval'
            or q.value(request['argv'],'--image-ablation')!='text_only'
            or request['backend']!='smallthinker4' or state['active_job']!=TARGET
            or request.get('job_id')!=TARGET or request.get('controller_source_sha256')!=CONTROLLER_SHA
            or request.get('data_release_sha256')!=RELEASE_SHA
            or q.sha(ROOT/'scripts/research_budget18_queue_v4.py')!=CONTROLLER_SHA):
        raise ValueError('Only exact source-bound active R3 text-only evaluation may overlap')
    if (q.sha(proc['request'])!=q.value(evaluation['argv'],'--request-sha256')
            or any(request['parent'].get(k)!=parent[k] for k in ('pid','birth','argv'))):
        raise ValueError('Text-only worker request/parent identity changed')


def wait_for_target():
    target=TARGET
    last=0
    while True:
        service=q.supervisor_info('research-budget18-v4')
        if service['statename']!='RUNNING':raise RuntimeError('Expected V4 queue not running; no vision worker launched')
        state=q.read(ROOT/'runs/research/budget18_queue_state_v1.json');row=state['jobs'].get(target,{})
        if row.get('grounding_text_only',{}).get('status')=='completed':
            print(json.dumps({'status':'skipped_target_already_completed','remaining_cache_filled_lazily':True}),flush=True);return False
        if state.get('active_job')==target and row.get('eval_process',{}).get('status')=='running':
            request=q.read(row['eval_process']['request'])
            if q.value(request['argv'],'--image-ablation')=='text_only':return True
        if time.monotonic()-last>=60:
            print(json.dumps({'event':'waiting_for_R3_text_only','target':target,'active_job':state.get('active_job'),'target_status':row.get('status')}),flush=True);last=time.monotonic()
        time.sleep(15)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-parent',type=int);p.add_argument('--seconds',type=int,default=600);p.add_argument('--execute',action='store_true');p.add_argument('--wait',action='store_true');args=p.parse_args()
    if not 1<=args.seconds<=600:p.error('Bounded warmup is1..600seconds')
    if args.worker_parent:
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        return worker(args.worker_parent,args.seconds)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.wait and args.execute and not wait_for_target():return
    service=q.supervisor_info('research-budget18-v4');parent=q.proc(service['pid'])
    state=q.read(ROOT/'runs/research/budget18_queue_state_v1.json');row=state['jobs'][state['active_job']]
    proc=row.get('eval_process',{});evaluation=q.proc(proc.get('pid',-1));request=q.read(proc['request'])
    validate_target(service,parent,state,evaluation,request,proc)
    if not args.execute:
        print(json.dumps({'status':'metadata_preflight_no_signals','parent':parent,'evaluation':evaluation}));return
    if OUT.exists():raise ValueError('Warmup receipt exists')
    OUT.mkdir();paused=False;child=None
    def stop(_n,_f):raise InterruptedError('Vision warmup observer interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        os.kill(parent['pid'],signal.SIGSTOP);paused=True
        if not q.same(evaluation,q.proc(evaluation['pid'])):raise RuntimeError('Evaluation boundary raced; no vision worker launched')
        q.atomic(OUT/'intent.json',{'controller':parent,'evaluation':evaluation,'max_seconds':args.seconds,
                 'only_CPU_owner_stopped':True,'LLM_worker_signals_sent':0,'training_updates':0,
                 'performance_note':'Text-only generation wall time overlaps useful frozen-Qwen cache computation'})
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-parent',str(os.getpid()),'--seconds',str(args.seconds)]
        with (OUT/'worker.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=q.proc(child.pid);start=time.monotonic();last=0
            while child.poll() is None:
                if time.monotonic()-start>args.seconds+60:
                    if q.same(identity,q.proc(child.pid)):os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        if q.same(identity,q.proc(child.pid)):os.killpg(child.pid,signal.SIGKILL)
                    break
                if time.monotonic()-last>=30:print(json.dumps({'event':'parallel_Qwen_cache_fill','pid':child.pid,'LLM_pid':evaluation['pid']}),flush=True);last=time.monotonic()
                time.sleep(1)
        code=child.wait(timeout=20)
        q.atomic(OUT/'completed.json',{'status':'completed' if code==0 else 'failed_vision_only','worker_returncode':code,
                 'LLM_worker_signals_sent':0,'controller_to_resume':parent,'worker_source_sha256':q.sha(__file__)})
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        if paused and q.same(parent,q.proc(parent['pid'])):os.kill(parent['pid'],signal.SIGCONT)


if __name__=='__main__':main()
