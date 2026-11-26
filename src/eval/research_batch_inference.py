"""Isolated opt-in left-padded native greedy generation prototype.

Production inference/CLI/protocol are unchanged. Batch size can change numerical
rounding, so deployment requires trained-checkpoint token parity and a common
evaluation profile across all eleven recipes. Tiny fixtures imply no speedup.
"""
from __future__ import annotations

import inspect
import torch

from src.eval.research_inference import build_generation_inputs,_autocast
from src.model.research_moe import set_modality_mask

BATCH_GENERATION_SCHEMA_VERSION=1


@torch.no_grad()
def build_batch_generation_inputs(model,records,*,max_new_tokens=64):
    """Reuse the gold-independent single-row preparation, preserving row order."""
    if not isinstance(records,(list,tuple)) or not records:
        raise ValueError('A nonempty ordered list of records is required')
    prepared=[build_generation_inputs(model,row,max_new_tokens=max_new_tokens) for row in records]
    first=prepared[0]['inputs_embeds']
    if any(row['inputs_embeds'].device!=first.device or row['inputs_embeds'].dtype!=first.dtype or
           row['inputs_embeds'].shape[-1]!=first.shape[-1] for row in prepared):
        raise ValueError('All prepared rows must share model device, dtype and hidden size')
    width=max(row['prefill_tokens'] for row in prepared);count=len(prepared)
    embeddings=first.new_zeros((count,width,first.shape[-1]))
    attention=torch.zeros((count,width),dtype=torch.bool,device=first.device)
    modality=torch.zeros_like(attention)
    for index,row in enumerate(prepared):
        length=row['prefill_tokens'];offset=width-length
        embeddings[index,offset:]=row['inputs_embeds'][0]
        attention[index,offset:]=row['attention_mask'][0]
        modality[index,offset:]=row['modality_mask'][0]
    # Positions count real tokens independently in each row. Native generate
    # extends these IDs on cached steps, so padding never advances real RoPE.
    positions=attention.long().cumsum(-1)-1
    positions.masked_fill_(~attention,0)
    return {'inputs_embeds':embeddings,'attention_mask':attention,'modality_mask':modality,
            'position_ids':positions,'rows':prepared,'padded_prefill_tokens':width,
            'padding_side':'left','batch_generation_schema_version':BATCH_GENERATION_SCHEMA_VERSION}


def eos_token_ids(model):
    """Respect generation_config.eos_token_id=None, which explicitly disables EOS."""
    value=getattr(model.base.generation_config,'eos_token_id',None)
    if value is None:return set()
    if isinstance(value,torch.Tensor):value=value.detach().cpu().reshape(-1).tolist()
    if isinstance(value,int):value=[value]
    if not isinstance(value,(list,tuple)) or any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in value):
        raise ValueError('Invalid native generation EOS token IDs')
    return set(value)


def trim_generated_ids(ids,eos):
    """Keep the first EOS (matching native B1), discard only subsequent padding."""
    for index,token in enumerate(ids):
        if token in eos:return ids[:index+1]
    return ids


@torch.no_grad()
def generate_batch_predictions(model,records,*,max_new_tokens=64):
    """Generate an ordered batch with native KV caching and per-row EOS lengths."""
    if model.base.training:raise ValueError('Batched generation requires an evaluation-mode model')
    prepared=build_batch_generation_inputs(model,records,max_new_tokens=max_new_tokens)
    set_modality_mask(model.base,prepared['modality_mask'],prepared['attention_mask'])
    kwargs=dict(inputs_embeds=prepared['inputs_embeds'],attention_mask=prepared['attention_mask'],
        position_ids=prepared['position_ids'],do_sample=False,num_beams=1,
        max_new_tokens=max_new_tokens,use_cache=True,return_dict_in_generate=True,
        output_scores=True,pad_token_id=model.tokenizer.pad_token_id)
    if 'logits_to_keep' in inspect.signature(model.base.forward).parameters:
        kwargs['logits_to_keep']=1
    with _autocast(prepared['inputs_embeds'].device):
        generated=model.base.generate(**kwargs)
    count=len(generated.scores);batch_size=len(records)
    if (count>max_new_tokens or generated.sequences.ndim!=2 or generated.sequences.shape[0]!=batch_size or
            generated.sequences.shape[1]<count):
        raise ValueError('Unexpected native batch generation shape/length')
    if generated.scores:
        valid=[]
        for scores in generated.scores:
            if scores.ndim!=2 or scores.shape[0]!=batch_size:
                raise ValueError('Native batch score shape differs')
            valid.append(torch.isfinite(scores).any(-1).all() & ~torch.isnan(scores).any() & ~torch.isposinf(scores).any())
        if not torch.stack(valid).all():raise ValueError('Native batch generation produced invalid logits')
    generated_ids=generated.sequences[:,-count:].detach().cpu().tolist() if count else [[] for _ in records]
    eos=eos_token_ids(model);result=[]
    for row,ids in zip(prepared['rows'],generated_ids):
        ids=trim_generated_ids(ids,eos)
        result.append({'prediction':model.tokenizer.decode(ids,skip_special_tokens=True).strip(),
            'generated_token_ids':ids,'generated_tokens':len(ids),'prompt_sha256':row['prompt_sha256'],
            'prefill_tokens':row['prefill_tokens'],'visual_tokens':row['visual_tokens'],
            'logits_to_keep':kwargs.get('logits_to_keep'),'batch_size':batch_size,
            'padded_prefill_tokens':prepared['padded_prefill_tokens'],'padding_side':'left',
            'batch_generation_schema_version':BATCH_GENERATION_SCHEMA_VERSION})
    return result


def require_idle_gpu():
    """Benchmark only with no observed competing GPU process; never signal it."""
    import os
    import subprocess
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                          check=True,capture_output=True,text=True)
    pids=[]
    for row in result.stdout.splitlines():
        if not row.strip():continue
        if not row.strip().isdigit():raise RuntimeError('Unrecognized GPU process listing')
        pids.append(int(row.strip()))
    others=[pid for pid in pids if pid!=os.getpid()]
    if others:raise RuntimeError('Benchmark requires a coordinated idle GPU; existing processes were left running')
    return pids


def benchmark_main(argv=None):
    """Optional trained4q B1/4/8 probe. Nothing runs without --allow-gpu.

    Run only after the controller owner schedules an idle window. This command
    never edits/rescores production evaluations or claims a campaign ETA.
    """
    import argparse
    import hashlib
    import json
    import math
    from pathlib import Path
    import time
    from src.eval.research_inference import load_research_checkpoint,generate_prediction
    from src.utils.research_checkpoint_compat import checkpoint_identity,load_research_payload,same_checkpoint_identity
    from src.utils.research_storage import DiskBudget,_atomic_json
    parser=argparse.ArgumentParser(description=benchmark_main.__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--records',type=int,default=32)
    parser.add_argument('--batch-sizes',type=int,nargs='+',default=[1,4,8])
    parser.add_argument('--max-new-tokens',type=int,default=64)
    parser.add_argument('--minimum-free-gib',type=float,default=12)
    parser.add_argument('--reserve-gib',type=float,default=120);parser.add_argument('--cache-dir',default='cache/vision')
    parser.add_argument('--allow-gpu',action='store_true');args=parser.parse_args(argv)
    if not args.allow_gpu:raise RuntimeError('GPU probe disabled; coordinate an idle window before using --allow-gpu')
    if (args.records<1 or args.max_new_tokens<1 or not args.batch_sizes or args.batch_sizes[0]!=1 or
            len(set(args.batch_sizes))!=len(args.batch_sizes) or min(args.batch_sizes)<1 or
            any(args.records%size for size in args.batch_sizes) or
            not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib<=0):
        raise ValueError('Positive limits and distinct batch sizes beginning with the native B1 reference are required')
    require_idle_gpu()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('This guarded probe requires exactly one visible CUDA GPU')
    if torch.cuda.mem_get_info()[0]<args.minimum_free_gib*1024**3:
        raise RuntimeError('Insufficient free VRAM for the declared probe; no existing task stopped')
    output=Path(args.output)
    if output.exists():raise FileExistsError('Use a new standalone probe output; prior results are preserved')
    budget=DiskBudget(output.parent,args.reserve_gib);budget.ensure_space(16*1024**2)
    manifest=Path(args.manifest);raw=manifest.read_bytes()
    records=[json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(records)<args.records:raise ValueError('Manifest has fewer than the requested fixed records')
    selected=records[:args.records];ids=[row['record_id'] for row in selected]
    if len(set(ids))!=len(ids):raise ValueError('Selected record IDs are not unique')
    checkpoint=Path(args.checkpoint).resolve(strict=True)
    payload=load_research_payload(checkpoint,map_location='cpu',mmap=True)
    if (payload.get('config',{}).get('recipe')!='4q' or payload['config'].get('stage')!='SFT' or
            payload.get('training_completed') is not True):
        raise ValueError('This initial real-weight probe is restricted to a completed4q SFT checkpoint')
    del payload
    identity=checkpoint_identity(checkpoint)
    source_root=Path(__file__).resolve().parents[2]
    sources=('src/eval/research_batch_inference.py','src/eval/research_inference.py','src/model/research_moe.py','src/model/smallthinker.py')
    import importlib.metadata
    report={'schema_version':1,'status':'running','scope':'isolated token-parity/performance probe; no production evaluation change or campaign ETA',
        'batch_generation_schema_version':BATCH_GENERATION_SCHEMA_VERSION,'checkpoint_identity':identity,
        'manifest_sha256':hashlib.sha256(raw).hexdigest(),'record_ids':ids,'record_selection':'fixed manifest prefix',
        'max_new_tokens':args.max_new_tokens,'batch_sizes':args.batch_sizes,
        'source_sha256':{name:hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in sources},
        'environment':{'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),
                       'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},
        'profiles':[],'shared_gpu_timings_valid':False,'no_scientific_scores_computed':True}
    def event(name,**values):print(json.dumps(dict(event=name,time=time.time(),**values)),flush=True)
    try:
        event('loading_trained_4q_probe',checkpoint=str(checkpoint),records=len(selected))
        model,provenance=load_research_checkpoint(checkpoint,cache_dir=args.cache_dir,reserve_gb=args.reserve_gib)
        report['model_provenance']=provenance
        # Warm feature/projection I/O consistently before any timed profile.
        for row in selected:
            prepared=build_generation_inputs(model,row,max_new_tokens=args.max_new_tokens)
            del prepared
        reference=None
        for size in args.batch_sizes:
            require_idle_gpu();budget.ensure_space(1024**2)
            event('warming_batch_shape',batch_size=size)
            if size==1:generate_prediction(model,selected[0],max_new_tokens=args.max_new_tokens)
            else:generate_batch_predictions(model,selected[:size],max_new_tokens=args.max_new_tokens)
            torch.cuda.synchronize();require_idle_gpu();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter();predictions=[]
            for offset in range(0,len(selected),size):
                chunk=selected[offset:offset+size]
                if size==1:predictions.append(generate_prediction(model,chunk[0],max_new_tokens=args.max_new_tokens))
                else:predictions.extend(generate_batch_predictions(model,chunk,max_new_tokens=args.max_new_tokens))
                torch.cuda.synchronize()
                event('batch_probe_progress',batch_size=size,completed_records=len(predictions),total_records=len(selected))
            seconds=time.perf_counter()-started;require_idle_gpu()
            tokens=[row['generated_token_ids'] for row in predictions]
            if reference is None:reference=tokens
            parity=tokens==reference
            result={'batch_size':size,'implementation':'unchanged native B1' if size==1 else 'left-padded prototype',
                'records':len(selected),'seconds':seconds,'records_per_second':len(selected)/seconds,
                'generated_tokens':sum(map(len,tokens)),'generated_tokens_per_second':sum(map(len,tokens))/seconds,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                'exact_token_parity_with_native_b1':parity,'generated_token_ids':tokens}
            report['profiles'].append(result);_atomic_json(output,report)
            if not parity:raise ValueError('Trained checkpoint token parity failed; no profile deployment permitted')
        current=checkpoint_identity(checkpoint,verify_components=False)
        if not same_checkpoint_identity(identity,current):raise ValueError('Checkpoint changed during probe')
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=report['manifest_sha256']:
            raise ValueError('Probe manifest changed')
        report.update(status='passed',observed_gpu_process_checks_exclusive=True,
                      caveat='Measured fixed records only; no extrapolation to all recipes, models, encoders or campaign cost')
        _atomic_json(output,report);event('batch_probe_completed',output=str(output))
        return report
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        _atomic_json(output,report)
        raise


if __name__=='__main__':benchmark_main()

