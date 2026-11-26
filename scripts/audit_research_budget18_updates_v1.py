#!/usr/bin/env python3
"""Read-only tensor evidence before consumed alignment checkpoints are retired."""
from __future__ import annotations
import contextlib,fcntl,gc,hashlib,json,os,signal,sys,time,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.research_storage import _atomic_json
OUT=ROOT/'research_results/budget18_parameter_updates_v1'
STATE=ROOT/'runs/research/budget18_queue_state_v1.json'


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(p):
    s=Path(p).stat()
    return {k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
def tensor_sha(t):
    import torch
    return hashlib.sha256(memoryview(t.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


@contextlib.contextmanager
def payload(run,path):
    import torch
    p=Path(path).resolve(strict=True);before=stat(p)
    value=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
    try:yield value,p,before
    finally:
        if stat(p)!=before:raise ValueError('Checkpoint identity changed during audit')


def audit_locked(run,callback,*args):
    # The queue holds shared manager locks through both open/maps scans and
    # FADV. Exclusive read-only audit locks serialize with that entire guard.
    # Keep the callback's tensor frames and cyclic mappings dead BEFORE unlock.
    with (run/'.checkpoint.lock').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:return callback(*args)
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:gc.collect()


def snapshot(run,config,output):
    source=Path(config['initial_checkpoint']);source_run=source.parent.parent
    with payload(source_run,source) as (p,physical,identity):
        hashes={k:{'sha256':tensor_sha(v),'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in p['model'].items()}
        result={'status':'captured','source_config_sha256':sha(source_run/'resolved_config.json'),
                'source_checkpoint':str(physical),'source_stat':identity,'source_step':p['step'],
                'source_tensors':hashes,'destination_config_sha256':sha(run/'resolved_config.json'),
                'created_unix':time.time()}
    _atomic_json(output,result)


def compare(run,config,source,output):
    import torch
    if sha(run/'resolved_config.json')!=source['destination_config_sha256']:raise ValueError('Destination config changed')
    with payload(run,run/'best/state.pt') as (p,physical,identity):
        if p['step']!=56 or p['seen']!=3529:raise ValueError('Only completed56/3529 SFT allowed')
        current=p['model'];groups={}
        for k,old in source['source_tensors'].items():
            if k not in current:raise ValueError('Alignment tensor absent in SFT')
            v=current[k]
            if list(v.shape)!=old['shape'] or str(v.dtype)!=old['dtype']:raise ValueError('Tensor shape/dtype changed')
            name='projector' if k.startswith('projector.') else next((n for n in ('visual_experts','router_adapter','modality_gate_logits','lora_') if n in k),'other')
            row=groups.setdefault(name,{'tensors':0,'changed_tensors':0,'elements':0})
            row['tensors']+=1;row['elements']+=v.numel();row['changed_tensors']+=int(tensor_sha(v)!=old['sha256'])
        if groups['projector']['changed_tensors']:raise ValueError('Frozen projector changed')
        for name in (('visual_experts','router_adapter','modality_gate_logits') if config['recipe']=='1c' else ('visual_experts','router_adapter') if config['recipe']=='1b' else ()):
            if not groups.get(name,{}).get('changed_tensors'):raise ValueError('No observed update in '+name)
        lora_b=[v for k,v in current.items() if 'lora_B' in k or 'lora_b' in k]
        lora_nonzero=sum(bool(torch.count_nonzero(v)) for v in lora_b)
        if config['recipe'] in ('1b','1c') and not lora_nonzero:raise ValueError('No nonzero LoRA-B tensor')
        base_samples=[]
        if config['recipe']=='4f':
            from safetensors import safe_open
            ref=p['model_ref'];hub=Path(os.environ.get('HF_HOME','/workspace/.hf_home'))/'hub'
            directory=hub/('models--'+ref['repo'].replace('/','--'))/'snapshots'/ref['revision']
            index=directory/'model.safetensors.index.json';mapping=read(index)['weight_map'] if index.exists() else None
            for k,v in current.items():
                original=k.removeprefix('llm.')
                # Hybrid LFM begins with two conv layers; inspect the first
                # available native attention matrices rather than layer0 only.
                if not (k.startswith('llm.') and '.layers.' in k and '.self_attn.' in k and v.ndim==2):continue
                file=directory/(mapping[original] if mapping else 'model.safetensors')
                with safe_open(file,framework='pt',device='cpu') as f:old=f.get_tensor(original).to(v.dtype)
                if old.shape!=v.shape:raise ValueError('Pinned source shape differs')
                changed=int(torch.count_nonzero(old!=v));base_samples.append({'name':k,'elements':v.numel(),'changed_elements':changed,'original_tensor_sha256_converted_to_saved_dtype':tensor_sha(old),'trained_tensor_sha256':tensor_sha(v),'pinned_source':{'path':str(file),'revision':ref['revision']}})
                if len(base_samples)==3:break
            if len(base_samples)!=3 or not all(x['changed_elements'] for x in base_samples):raise ValueError('Full-base sample updates not established')
        result={'status':'passed','created_unix':time.time(),'recipe':config['recipe'],
                'auditor_source_sha256':sha(Path(__file__)),
                'source_snapshot_sha256':sha(output.parent/'initial_tensors.json'),
                'checkpoint':str(physical),'checkpoint_stat':identity,'common_tensor_groups':groups,
                'nonzero_LoRA_B_tensors':lora_nonzero,'LoRA_B_tensors':len(lora_b),
                'full_base_update_samples':base_samples,
                'scope':'common A1/A0 to final SFT tensor hashes; R2 original-base evidence is three attention-weight samples; LoRA-B nonzero alone is not an A1-to-SFT difference',
                'frozen_NF4_base_content_comparison_claimed':False,'scientific_weights_modified':False}
    _atomic_json(output,result)


def cycle():
    state=read(STATE)
    ids=set(k for k,v in state['jobs'].items() if v.get('training_completed') and '__SFT__budget18_s42' in k)
    if state.get('active_job') and '__SFT__budget18_s42' in state['active_job']:ids.add(state['active_job'])
    for name in sorted(ids):
        run=ROOT/'runs/research'/name;directory=OUT/name;output=directory/'report.json'
        if output.exists():continue
        config=run/'resolved_config.json'
        if not config.exists():continue
        directory.mkdir(parents=True,exist_ok=True)
        try:
            config=read(config);initial=directory/'initial_tensors.json'
            if not initial.exists():
                source=Path(config['initial_checkpoint']).parent.parent
                audit_locked(source,snapshot,run,config,initial)
            # QLoRA compaction publishes a new physical best/last object.
            # The queue assigns this receipt only after that operation returns.
            if (run/'completed.json').exists() and state['jobs'][name].get('compaction'):
                audit_locked(run,compare,run,config,read(initial),output)
        except (FileNotFoundError,BlockingIOError) as e:
            # Never change the scientific queue or re-create a consumed source.
            _atomic_json(directory/'observation.json',{'status':'source_unavailable_or_busy','time':time.time(),'error':str(e)})
        except Exception as e:
            _atomic_json(directory/'error.json',{'status':'audit_failed','time':time.time(),'error':repr(e)})
            print(json.dumps({'run':name,'error':repr(e)}),flush=True)


def main():
    import torch
    torch.set_num_threads(2)
    running=True
    def stop(_s,_f):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while running:
        cycle()
        if '--watch' not in sys.argv:return
        for _ in range(10):
            if not running:break
            time.sleep(1)


if __name__=='__main__':main()
