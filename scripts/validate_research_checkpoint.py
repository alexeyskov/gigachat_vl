#!/usr/bin/env python3
"""Validate managed research checkpoints without confusing CPU checks with replay.

Full mode reconstructs the original ResearchModel on an idle GPU, restores every
saved tensor, and repeats the exact held-out CE or internal group-macro selection
used by the trainer. Internal teacher-reference macro is not an official score.
CPU structural mode never emits ``status=passed`` and cannot authorize pruning.
Only trusted locally generated state.pt payloads should be supplied.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def atomic_json(path,payload):
    from src.utils.research_storage import _atomic_json
    _atomic_json(Path(path),payload)


def select_validation_records(config,load_records):
    if config.get('selection_metric','ce')=='group_macro':
        from scripts.run_research_stage import prepare_group_macro_selection
        return prepare_group_macro_selection(config,loader=load_records)
    import random
    records=load_records(config['validation'])
    if config['stage']=='SFT' and config.get('text_manifest') and config.get('text_validation'):
        records+=load_records(config['text_validation'])
    if len(records)>config['eval_records']:
        records=random.Random(config['seed']).sample(records,config['eval_records'])
    if not records: raise ValueError('Validation selection is empty')
    return records


def assert_idle_gpu():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    pids=[int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    busy=[pid for pid in pids if pid!=os.getpid()]
    if busy:
        raise RuntimeError(f'GPU is occupied by process(es) {busy}; full validation requires an idle GPU')


def managed_checkpoints(run):
    root=Path(run).resolve()
    owner=json.loads((root/'.checkpoint_manager.json').read_text())['owner']
    completed=json.loads((root/'completed.json').read_text())
    if completed.get('steps',0)<1: raise ValueError('Run does not have a successful completion marker')
    unique={}
    for label in ['best','last']:
        pointer=root/label
        if not pointer.is_symlink(): raise ValueError(f'{label} must be a managed checkpoint symlink')
        state=(pointer/'state.pt').resolve()
        if state.parent.parent!=root/'.checkpoint_objects': raise ValueError(f'{label} points outside checkpoint objects')
        marker=json.loads((state.parent/'.research_checkpoint.json').read_text())
        if marker.get('owner')!=owner: raise ValueError('Checkpoint ownership mismatch')
        if marker.get('step',0)>completed['steps']: raise ValueError('Checkpoint is newer than completed run')
        if label=='last' and marker['step']!=completed['steps']: raise ValueError('Last checkpoint is not final completed step')
        entry=unique.setdefault(str(state),{'path':state,'marker':marker,'labels':[],'owner':owner})
        entry['labels'].append(label)
    return list(unique.values())


def check_manifest_hashes(config):
    mappings={'manifest':'manifest_sha256','validation':'validation_sha256',
              'text_manifest':'text_manifest_sha256','text_validation':'text_validation_sha256'}
    for file_key,hash_key in mappings.items():
        if config.get(file_key) and config.get(hash_key):
            actual=hashlib.sha256(Path(config[file_key]).read_bytes()).hexdigest()
            if actual!=config[hash_key]: raise ValueError(f'{file_key} changed since training')
    if config.get('selection_metric','ce')=='group_macro':
        expected=config.get('selection_manifest_sha256')
        if not expected:raise ValueError('Group macro selection lacks its immutable data SHA')
        actual=hashlib.sha256(Path(config.get('selection_manifest') or config['validation']).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('selection_manifest changed since training')


def finite_model_tensors(state):
    import torch
    count=0
    for name,tensor in state.items():
        if not isinstance(tensor,torch.Tensor): raise ValueError(f'Model entry is not a tensor: {name}')
        if tensor.numel()==0: raise ValueError(f'Model entry is empty: {name}')
        # Bound temporary allocations for large packed expert/embedding matrices.
        if tensor.is_floating_point() or tensor.is_complex():
            for chunk in tensor.reshape(-1).split(4*1024**2):
                if not torch.isfinite(chunk).all(): raise ValueError(f'Non-finite checkpoint tensor: {name}')
        count+=1
    if not count: raise ValueError('Checkpoint contains no model tensors')
    return count


def structural_checks(payload,config,trainable):
    if not isinstance(payload,dict) or not isinstance(payload.get('model'),dict):
        raise ValueError('Missing model tensor payload')
    from scripts.run_research_stage import validate_resume_config
    try:
        validate_resume_config(payload.get('config',{}),config)
    except ValueError as error:
        raise ValueError('Saved checkpoint config differs from resolved_config.json') from error
    saved=payload['model']
    expected=set(trainable)|{name for name in saved if name.startswith('projector.')}
    if set(saved)!=expected:
        raise ValueError(f'Expected saved keys mismatch: missing={sorted(expected-set(saved))}, unexpected={sorted(set(saved)-expected)}')
    if not any(name.startswith('projector.') for name in saved):
        raise ValueError('Checkpoint lacks aligned projector')
    for name,numel in trainable.items():
        if saved[name].numel()!=numel: raise ValueError(f'Trainable tensor size mismatch: {name}')
    count=finite_model_tensors(saved)
    return {'validated_tensor_count':count,'all_tensors_finite':True,'expected_keys_match':True,
            'saved_parameter_elements':sum(t.numel() for t in saved.values())}


def _tensor_equal_after_restore(actual,expected,name):
    import torch
    if actual.shape!=expected.shape:
        raise ValueError(f'Restored tensor shape mismatch: {name}')
    # Copies are bounded; do not duplicate a full 4B state_dict on CPU/GPU.
    a=actual.detach().reshape(-1)
    b=expected.reshape(-1)
    maximum=0.
    for offset in range(0,a.numel(),4*1024**2):
        x=a[offset:offset+4*1024**2].to('cpu')
        y=b[offset:offset+4*1024**2].to(device='cpu')
        promoted=torch.promote_types(x.dtype,y.dtype)
        x=x.to(promoted);y=y.to(promoted)
        if not torch.equal(x,y):
            maximum=max(maximum,(x.float()-y.float()).abs().max().item())
            raise ValueError(f'Restored tensor differs from saved value: {name}, max_abs_diff={maximum}')
    return maximum


def full_model_checks(model,payload,records,batch_size,expected_loss,atol,rtol):
    import torch
    available=model.state_dict()
    for name,tensor in payload['model'].items():
        if name not in available: raise ValueError(f'Saved key absent from reconstructed model: {name}')
        if available[name].shape!=tensor.shape: raise ValueError(f'Reconstructed tensor shape mismatch: {name}')
    restored=model.load_state_dict(payload['model'],strict=False)
    if restored.unexpected_keys: raise ValueError(f'Unexpected saved model keys: {restored.unexpected_keys}')
    actual=model.state_dict()
    maximum=0.
    for name,tensor in payload['model'].items():
        maximum=max(maximum,_tensor_equal_after_restore(actual[name],tensor,name))
    frozen_projector=payload['config']['stage']!='A0'
    if frozen_projector and any(p.requires_grad for p in model.projector.parameters()):
        raise ValueError('Projector is unexpectedly trainable after alignment')
    projector_before={name:value.detach().cpu().clone() for name,value in model.projector.state_dict().items()}
    if payload['config'].get('selection_metric','ce')=='group_macro':
        from src.eval.research_selection import evaluate_internal_group_macro,PROTOCOL_NAME
        if payload.get('selection_metric')!=PROTOCOL_NAME:
            raise ValueError('Checkpoint does not identify the internal teacher-reference macro protocol')
        saved_protocol=payload.get('selection_protocol') or {}
        if saved_protocol.get('selection_manifest_sha256')!=records['protocol']['selection_manifest_sha256']:
            raise ValueError('Reconstructed internal selection differs from saved selection manifest')
        result=evaluate_internal_group_macro(model,records,max_new_tokens=payload['config'].get('selection_max_new_tokens',64))
        measured=result['metrics']['group_macro_score']
        if not math.isfinite(measured) or not math.isfinite(expected_loss):
            raise ValueError('Nonfinite internal group macro selection value')
        tolerance=atol+rtol*abs(expected_loss)
        difference=abs(measured-expected_loss)
        if difference>tolerance:
            raise ValueError(f'Checkpoint group macro replay mismatch: measured={measured}, saved={expected_loss}, diff={difference}, tolerance={tolerance}')
        for name,tensor in model.projector.state_dict().items():
            _tensor_equal_after_restore(tensor,projector_before[name],f'projector.{name}')
        return {'selection_metric_replayed':True,'selection_protocol':PROTOCOL_NAME,'official_benchmark_score':False,
                'group_macro_score_replayed':measured,'group_macro_score_saved':expected_loss,
                'group_macro_score_abs_diff':difference,'group_macro_score_tolerance':tolerance,
                'validation_records':len(records['records']),'heldout_ce_replayed':False,
                'restored_tensor_max_abs_diff':maximum,'projector_frozen':frozen_projector,
                'projector_unchanged_during_validation':True,
                'selection_manifest_sha256':records['protocol']['selection_manifest_sha256']}
    model.eval()
    total=0.;tokens=0
    with torch.no_grad():
        for offset in range(0,len(records),batch_size):
            rows=model.prepare(records[offset:offset+batch_size])
            loss,count,_=model.forward_loss(rows)
            total+=loss.item();tokens+=count
    if tokens<=0: raise ValueError('Validation yielded no supervised tokens')
    measured=total/tokens
    if not math.isfinite(measured): raise ValueError('Restored validation loss is non-finite')
    if not math.isfinite(expected_loss): raise ValueError('Saved validation loss is non-finite')
    tolerance=atol+rtol*abs(expected_loss)
    difference=abs(measured-expected_loss)
    if difference>tolerance:
        raise ValueError(f'Checkpoint CE replay mismatch: measured={measured}, saved={expected_loss}, diff={difference}, tolerance={tolerance}')
    for name,tensor in model.projector.state_dict().items():
        _tensor_equal_after_restore(tensor,projector_before[name],f'projector.{name}')
    return {'selection_metric_replayed':True,'selection_protocol':'heldout_ce_smoke_proxy_not_official_benchmark',
            'heldout_ce_replayed':True,'validation_loss_replayed':measured,'validation_loss_saved':expected_loss,
            'validation_loss_abs_diff':difference,'validation_loss_tolerance':tolerance,
            'validation_records':len(records),'validation_supervised_tokens':tokens,
            'restored_tensor_max_abs_diff':maximum,'projector_frozen':frozen_projector,
            'projector_unchanged_during_validation':True}



def cpu_native_full_checks(payload,config):
    """Reconstruct a complete full-trained SmallThinker and exercise native forward.

    This proves restoration/architecture and finite multimodal-style execution;
    it is explicitly not a replay of scientific held-out task metrics.
    """
    import torch
    from torch import nn
    from transformers import AutoConfig,AutoModelForCausalLM
    from src.model.smallthinker import register_smallthinker
    from scripts.run_research_stage import resolve_model
    if config['recipe'] not in ['2a','2b','2c','3a','3b','3c','4f']:
        raise ValueError('CPU native-full validation requires a full-training recipe')
    if config['stage']!='SFT' and not (config['stage']=='A1' and config['recipe']=='4f'):
        raise ValueError('CPU native-full validation requires every original language parameter in checkpoint')
    register_smallthinker()
    path,_=resolve_model(config['model'])
    cfg=AutoConfig.from_pretrained(path,trust_remote_code=False)
    if cfg.model_type!='smallthinker':
        raise NotImplementedError('CPU native-full replay currently validates SmallThinker only')
    torch.set_num_threads(8)
    # Avoid initializing and copying 4B parameters before loading the complete
    # checkpoint. assign=True keeps mmap pages available without optimizer reads.
    with torch.device('meta'):
        model=AutoModelForCausalLM.from_config(cfg,attn_implementation='sdpa')
        if config['recipe'][0] in '23':
            from src.model.research_moe import install_research_moe
            install_research_moe(model,int(config['recipe'][0]),config['recipe'][1],added_experts=2)
    language={name.removeprefix('llm.'):tensor for name,tensor in payload['model'].items() if name.startswith('llm.')}
    if not language: raise ValueError('Full checkpoint contains no original LLM weights')
    available=model.state_dict()
    unexpected=set(language)-set(available)
    if unexpected: raise ValueError(f'Checkpoint contains unexpected language keys: {sorted(unexpected)[:10]}')
    for name,tensor in language.items():
        if tensor.shape!=available[name].shape: raise ValueError(f'Native reconstruction shape mismatch: {name}')
    restored=model.load_state_dict(language,strict=False,assign=True)
    if restored.unexpected_keys: raise ValueError(f'Unexpected language keys: {restored.unexpected_keys}')
    model.tie_weights()  # restore aliases omitted by named_parameters deduplication
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    model.model.rotary_emb=LlamaRotaryEmbedding(config=cfg,device='cpu')
    # Every parameter, including routers/visual experts, must be materialized.
    missing=[name for name,param in model.named_parameters() if param.is_meta]
    if missing: raise ValueError(f'Checkpoint lacks full language parameters: {missing[:10]}')
    if any(buffer.is_meta for buffer in model.buffers()):
        raise ValueError('Native reconstruction still has unmaterialized buffers')
    current=model.state_dict()
    for name,tensor in language.items():
        _tensor_equal_after_restore(current[name],tensor,'llm.'+name)
    projector_state={name.removeprefix('projector.'):tensor for name,tensor in payload['model'].items() if name.startswith('projector.')}
    if set(projector_state)!={'0.weight','0.bias','2.weight','2.bias'}:
        raise ValueError('CPU native replay expects the documented two-linear GELU MLP projector')
    projector=nn.Sequential(nn.Linear(projector_state['0.weight'].shape[1],projector_state['0.weight'].shape[0]),
                            nn.GELU(),nn.Linear(projector_state['2.weight'].shape[1],projector_state['2.weight'].shape[0]))
    projector.load_state_dict(projector_state,strict=True,assign=True)
    projector.requires_grad_(False).eval();model.requires_grad_(False).eval()
    if config['recipe'][0] in '23':
        from src.model.research_moe import set_modality_mask
        set_modality_mask(model,torch.tensor([[True,True,False,False]]),torch.ones(1,4,dtype=torch.bool))
    ids=torch.tensor([[min(7,cfg.vocab_size-1),min(17,cfg.vocab_size-1)]])
    with torch.no_grad():
        visual=projector(torch.full((2,projector_state['0.weight'].shape[1]),0.125,dtype=projector_state['0.weight'].dtype))
        embeddings=torch.cat([visual.to(model.get_input_embeddings().weight.dtype).unsqueeze(0),model.get_input_embeddings()(ids)],dim=1)
        first=model(inputs_embeds=embeddings,attention_mask=torch.ones(1,4,dtype=torch.long),use_cache=False,return_dict=True).logits
        if first.shape!=(1,4,cfg.vocab_size) or not torch.isfinite(first).all():
            raise ValueError('CPU reconstructed model returned invalid/non-finite native logits')
        # Repeat restoration before a second deterministic native forward.
        model.load_state_dict(language,strict=False,assign=True);model.tie_weights()
        second=model(inputs_embeds=embeddings,attention_mask=torch.ones(1,4,dtype=torch.long),use_cache=False,return_dict=True).logits
        difference=(first.float()-second.float()).abs().max().item()
        if not torch.allclose(first,second,atol=1e-5,rtol=1e-5):
            raise ValueError(f'CPU native forward changed after repeated restoration: {difference}')
    for name,tensor in projector.state_dict().items():
        _tensor_equal_after_restore(tensor,projector_state[name],'projector.'+name)
    result={'validation_mode':'cpu_native_full_reconstruction','native_forward_tokens':4,
            'native_visual_tokens':2,'native_text_tokens':2,'native_logits_shape':list(first.shape),
            'native_logits_abs_max':first.float().abs().max().item(),
            'native_logits_roundtrip_max_abs_diff':difference,
            'projector_frozen':True,'projector_unchanged_during_validation':True,
            'heldout_ce_replayed':False,'language_saved_tensor_count':len(language),
            'unmaterialized_parameters':0}
    del model,projector,first,second
    gc.collect()
    return result

def validate_run(run,*,cpu_structural_only=False,device='cuda',native_full_only=False,atol=2e-3,rtol=2e-4):
    import torch
    torch.set_num_threads(8)
    root=Path(run).resolve()
    report_path=root/'checkpoint_roundtrip.json'
    report={'schema_version':1,'status':'running','run':str(root),'started_unix':time.time(),
            'mode':'cpu_structural_only' if cpu_structural_only else 'cpu_native_full_reconstruction' if device=='cpu' else 'full_gpu_replay','checkpoints':[]}
    try:
        config=json.loads((root/'resolved_config.json').read_text())
        trainable=json.loads((root/'trainable_parameters.json').read_text())
        check_manifest_hashes(config)
        checkpoints=managed_checkpoints(root)
        if not cpu_structural_only and device!='cpu': assert_idle_gpu()
        if device=='cpu' and not cpu_structural_only and not native_full_only:
            raise ValueError('CPU full mode requires explicit --native-full-only; heldout CE replay uses GPU')
        for entry in checkpoints:
            path=entry['path'];stat=path.stat()
            from src.utils.research_checkpoint_compat import load_research_payload,checkpoint_kind,checkpoint_identity,same_checkpoint_identity
            bundle_identity=checkpoint_identity(path) if checkpoint_kind(path)=='bundle_v1' else None
            payload=load_research_payload(path,map_location='cpu',mmap=True)
            checks=structural_checks(payload,config,trainable)
            if payload.get('step')!=entry['marker']['step']: raise ValueError('Payload and marker step mismatch')
            receipt={'schema_version':1,'status':'structural_only' if cpu_structural_only else 'passed',
                     'checkpoint_owner':entry['owner'],'checkpoint_path':str(path),'checkpoint_labels':entry['labels'],
                     'checkpoint_step':entry['marker']['step'],'checkpoint_size_bytes':stat.st_size,
                     'checkpoint_mtime_ns':stat.st_mtime_ns,**checks}
            if bundle_identity is not None:
                receipt.update(checkpoint_bundle_identity=bundle_identity,checkpoint_sha256=bundle_identity['sha256'],checkpoint_artifact_total_bytes=bundle_identity['total_bytes'])
            receipt['selection_metric_replayed']=False
            receipt['official_benchmark_score']=False
            if not cpu_structural_only and device=='cpu':
                receipt.update(cpu_native_full_checks(payload,config))
            elif not cpu_structural_only:
                from scripts.run_research_stage import ResearchModel,load_records,validate_model_binding
                from transformers import set_seed
                set_seed(config['seed']);torch.set_num_threads(8)
                torch.backends.cuda.matmul.allow_tf32=True
                # Pre-message-schema checkpoints used question-only prompts.
                # Replay their measured CE exactly, rather than applying a later
                # prompt fix retroactively to historical selection values.
                args=argparse.Namespace(**dict(config,prompt_schema_version=config.get('prompt_schema_version',1)))
                model=ResearchModel(args)
                validate_model_binding(payload,model)
                records=select_validation_records(config,load_records)
                expected=entry['marker'].get('metric')
                if expected is None: raise ValueError('Managed checkpoint lacks a measured selection metric')
                receipt.update(full_model_checks(model,payload,records,config['batch_size'],expected,atol,rtol))
                del model
                gc.collect();torch.cuda.empty_cache()
            if path.stat().st_size!=stat.st_size or path.stat().st_mtime_ns!=stat.st_mtime_ns:
                raise RuntimeError('Checkpoint changed during validation')
            if bundle_identity is not None and not same_checkpoint_identity(bundle_identity,checkpoint_identity(path,verify_components=False)):
                raise RuntimeError('Bundle components changed during validation')
            report['checkpoints'].append(receipt)
            del payload
            gc.collect()
        report['status']='structural_only' if cpu_structural_only else 'passed'
        report['finished_unix']=time.time()
        report['validated_tensor_count']=sum(r['validated_tensor_count'] for r in report['checkpoints'])
        atomic_json(report_path,report)
        return report
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),finished_unix=time.time())
        atomic_json(report_path,report)
        raise


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--cpu-structural-only',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--native-full-only',action='store_true',help='CPU: reconstruct full SmallThinker and replay deterministic native multimodal-style forward, not heldout CE')
    parser.add_argument('--atol',type=float,default=2e-3)
    parser.add_argument('--rtol',type=float,default=2e-4)
    args=parser.parse_args()
    if not all(math.isfinite(v) and v>=0 for v in [args.atol,args.rtol]): parser.error('Tolerances must be finite and nonnegative')
    print(json.dumps(validate_run(args.run,cpu_structural_only=args.cpu_structural_only,device=args.device,native_full_only=args.native_full_only,atol=args.atol,rtol=args.rtol),indent=2))

if __name__=='__main__': main()
