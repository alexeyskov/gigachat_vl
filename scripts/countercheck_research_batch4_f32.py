#!/usr/bin/env python3
"""Isolated F32 language/MLP countercheck of an existing failed B4 batch.

Original NF4 parameter bytes and all quantization-state metadata stay unchanged.
Only ordinary floating language/projector tensors and runtime4-bit compute_dtype
are changed in this separately loaded process. No vision encoding or gate edit.
"""
from __future__ import annotations
from collections import Counter
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from scripts.diagnose_research_batch4_logits import RuntimeGuard,compare_logits,failed_batch,public,sha,digest,read_rows
from src.eval.research_inference import load_research_checkpoint,IMAGE_MARKER
from src.eval.research_batch_inference import eos_token_ids,trim_generated_ids
from src.model.research_moe import set_modality_mask
from src.utils.research_checkpoint_compat import checkpoint_identity,load_research_payload,same_checkpoint_identity
from src.utils.research_storage import DiskBudget,_atomic_json


class CountercheckBlocked(RuntimeError):pass


def language_roots(model):return [('llm',model.llm),('projector',model.projector)]
def quantized_parameter(parameter):
    return parameter.__class__.__name__ in ('Params4bit','Int8Params') or getattr(parameter,'quant_state',None) is not None


def tensor_digest(tensor):
    tensor=tensor.detach();flat=tensor.reshape(-1);h=hashlib.sha256()
    for offset in range(0,flat.numel(),2**20):
        chunk=flat[offset:offset+2**20].contiguous().view(torch.uint8).cpu()
        h.update(chunk.numpy().tobytes())
    return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'device':str(tensor.device),'sha256':h.hexdigest()}


def quant_metadata(value,tensor_ids,seen=None):
    seen={} if seen is None else seen
    if isinstance(value,torch.Tensor):tensor_ids.add(id(value));return {'tensor':tensor_digest(value)}
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if isinstance(value,(torch.dtype,torch.device)):return str(value)
    if isinstance(value,(list,tuple,torch.Size)):return [quant_metadata(x,tensor_ids,seen) for x in value]
    if isinstance(value,dict):return {str(k):quant_metadata(v,tensor_ids,seen) for k,v in sorted(value.items(),key=lambda p:str(p[0]))}
    if isinstance(value,nn.Module):return {'module_reference_id':id(value),'class':type(value).__qualname__}
    if id(value) in seen:return {'reference':seen[id(value)]}
    if hasattr(value,'__dict__'):
        seen[id(value)]=len(seen)
        return {'class':type(value).__qualname__,'attributes':quant_metadata(vars(value),tensor_ids,seen)}
    raise CountercheckBlocked('Unsupported quantization metadata type: '+str(type(value)))


def quantization_snapshot(model):
    rows=[];tensor_ids=set();seen=set()
    for prefix,root in language_roots(model):
        for name,parameter in root.named_parameters():
            if id(parameter) in seen or not quantized_parameter(parameter):continue
            seen.add(id(parameter));tensor_ids.add(id(parameter))
            rows.append({'name':prefix+'.'+name,'parameter_class':type(parameter).__qualname__,
                'packed_tensor':tensor_digest(parameter),
                'metadata':quant_metadata(vars(parameter),tensor_ids)})
    return {'parameters':rows,'sha256':digest(rows),'quantized_parameter_count':len(rows)},tensor_ids


def promote_ordinary_f32(model,quant_tensor_ids):
    changes=[];seen_parameters=set();converted_buffers={};compute_changes=[];ordinary_bytes=0;added_bytes=0
    for prefix,root in language_roots(model):
        for module_name,module in root.named_modules():
            path=prefix+('.'+module_name if module_name else '')
            for name,parameter in module._parameters.items():
                if parameter is None or id(parameter) in seen_parameters:continue
                seen_parameters.add(id(parameter))
                if quantized_parameter(parameter) or id(parameter) in quant_tensor_ids:continue
                if parameter.is_floating_point():
                    if parameter.dtype not in (torch.float16,torch.bfloat16,torch.float32):raise CountercheckBlocked('Unsupported ordinary parameter dtype')
                    ordinary_bytes+=parameter.numel()*4
                    if parameter.dtype!=torch.float32:
                        old=str(parameter.dtype);added_bytes+=parameter.numel()*(4-parameter.element_size())
                        parameter.data=parameter.data.to(dtype=torch.float32)
                        changes.append({'name':path+'.'+name,'kind':'parameter','before':old,'after':'torch.float32','elements':parameter.numel()})
            for name,buffer in list(module._buffers.items()):
                if buffer is None or id(buffer) in quant_tensor_ids or not buffer.is_floating_point():continue
                if buffer.dtype not in (torch.float16,torch.bfloat16,torch.float32):raise CountercheckBlocked('Unsupported ordinary buffer dtype')
                if buffer.dtype!=torch.float32:
                    old_id=id(buffer);old=str(buffer.dtype)
                    if old_id not in converted_buffers:
                        added_bytes+=buffer.numel()*(4-buffer.element_size());converted_buffers[old_id]=buffer.to(dtype=torch.float32)
                    module._buffers[name]=converted_buffers[old_id]
                    changes.append({'name':path+'.'+name,'kind':'buffer','before':old,'after':'torch.float32','elements':buffer.numel()})
            if hasattr(module,'compute_dtype'):
                before=getattr(module,'compute_dtype')
                # bnb Linear4bit and custom wrappers must explicitly request F32.
                if any(quantized_parameter(p) for p in module._parameters.values() if p is not None) or '4bit' in type(module).__name__.lower() or 'nf4' in type(module).__name__.lower():
                    module.compute_dtype=torch.float32
                    previous_flag=getattr(module,'compute_type_is_set',None)
                    if hasattr(module,'compute_type_is_set'):module.compute_type_is_set=True
                    compute_changes.append({'module':path,'class':type(module).__qualname__,'before':str(before),'after':'torch.float32',
                                            'compute_type_is_set_before':previous_flag,'compute_type_is_set_after':getattr(module,'compute_type_is_set',None)})
    model.requires_grad_(False).eval()
    return {'ordinary_tensor_changes':changes,'ordinary_tensor_change_count':len(changes),
            'runtime_nf4_compute_changes':compute_changes,'ordinary_f32_parameter_bytes':ordinary_bytes,'additional_tensor_bytes':added_bytes,
            'quant_metadata_or_constructor_specs_modified':False,'vision_modules_promoted':False}


def ordinary_floating_dtypes(model,quant_ids):
    dtypes=Counter();seen=set()
    for _,root in language_roots(model):
        for value in list(root.parameters())+list(root.buffers()):
            if id(value) in seen or id(value) in quant_ids or quantized_parameter(value):continue
            seen.add(id(value))
            if value.is_floating_point():dtypes[str(value.dtype)]+=value.numel()
    return dict(dtypes)


def f32_inputs(model,record,max_new_tokens=64):
    paths=record.get('image_paths',[])
    if len(paths)>1 or IMAGE_MARKER in record['question']:raise ValueError('Unsupported image count/marker collision')
    content=(IMAGE_MARKER+'\n' if paths else '')+record['question']
    prompt=model.tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    before,after=prompt.split(IMAGE_MARKER,1) if paths else ('',prompt)
    prefix=model.tokenizer.encode(before,add_special_tokens=False);suffix=model.tokenizer.encode(after,add_special_tokens=False)
    try:
        features=model.cache.get_many(paths,lambda _:(_ for _ in ()).throw(CountercheckBlocked('Vision encoding is forbidden')),
            image_hashes=record.get('image_hashes',[]),read_only=True) if paths else []
    except (FileNotFoundError,KeyError) as error:
        raise CountercheckBlocked('Required vision features are absent; no encoder fallback permitted') from error
    embedding=model.base.get_input_embeddings();device=embedding.weight.device
    if embedding.weight.dtype!=torch.float32:raise CountercheckBlocked('Embedding is not genuinely F32')
    with torch.autocast(device_type=device.type,enabled=False):
        chunks=[embedding(torch.tensor(prefix,device=device,dtype=torch.long))]
        for feature in features:chunks.append(model.projector(feature.to(device=device,dtype=torch.float32)))
        chunks.append(embedding(torch.tensor(suffix,device=device,dtype=torch.long)));inputs=torch.cat(chunks).unsqueeze(0)
    if inputs.dtype!=torch.float32:raise CountercheckBlocked('Prepared language/MLP inputs are not F32')
    visual=sum(x.shape[0] for x in features);total=inputs.shape[1]
    if total+max_new_tokens>min(model.args.max_length,model.base.config.max_position_embeddings):raise CountercheckBlocked('Countercheck exceeds unchanged context')
    attention=torch.ones((1,total),device=device,dtype=torch.bool)
    modality=torch.tensor([[False]*len(prefix)+[True]*visual+[False]*len(suffix)],device=device,dtype=torch.bool)
    return {'inputs_embeds':inputs,'attention_mask':attention,'modality_mask':modality,'prefill_tokens':total,'visual_tokens':visual,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}


@torch.no_grad()
def capture_f32(model,records,*,batched,max_new_tokens=64):
    prepared=[f32_inputs(model,public(row),max_new_tokens) for row in records]
    if not batched and len(records)!=1:raise ValueError('Native reference is B1')
    width=max(x['prefill_tokens'] for x in prepared);device=prepared[0]['inputs_embeds'].device
    if batched:
        embeddings=prepared[0]['inputs_embeds'].new_zeros((len(records),width,prepared[0]['inputs_embeds'].shape[-1]))
        attention=torch.zeros((len(records),width),dtype=torch.bool,device=device);modality=torch.zeros_like(attention)
        for i,row in enumerate(prepared):
            offset=width-row['prefill_tokens'];embeddings[i,offset:]=row['inputs_embeds'][0]
            attention[i,offset:]=row['attention_mask'][0];modality[i,offset:]=row['modality_mask'][0]
        positions=attention.long().cumsum(-1)-1;positions.masked_fill_(~attention,0)
        kwargs={'inputs_embeds':embeddings,'attention_mask':attention,'position_ids':positions}
    else:
        attention=prepared[0]['attention_mask'];modality=prepared[0]['modality_mask'];positions=None
        kwargs={'inputs_embeds':prepared[0]['inputs_embeds'],'attention_mask':attention}
    set_modality_mask(model.base,modality,attention)
    kwargs.update(do_sample=False,num_beams=1,max_new_tokens=max_new_tokens,use_cache=True,return_dict_in_generate=True,
                  output_scores=True,output_logits=True,pad_token_id=model.tokenizer.pad_token_id)
    if 'logits_to_keep' in inspect.signature(model.base.forward).parameters:kwargs['logits_to_keep']=1
    with torch.autocast(device_type=device.type,enabled=False):generated=model.base.generate(**kwargs)
    count=len(generated.scores)
    if not count or not getattr(generated,'logits',None):raise CountercheckBlocked('First-step tensors unavailable')
    ids=generated.sequences[:,-count:].detach().cpu().tolist()
    if batched:ids=[trim_generated_ids(x,eos_token_ids(model)) for x in ids]
    return {'token_ids':ids,'predictions':[model.tokenizer.decode(x,skip_special_tokens=True).strip() for x in ids],
        'raw_logits':generated.logits[0].detach().cpu().clone(),'output_scores':generated.scores[0].detach().cpu().clone(),
        'rows':[{'record_id':record['record_id'],'prompt_sha256':row['prompt_sha256'],'prefill_tokens':row['prefill_tokens'],
                 'visual_tokens':row['visual_tokens'],'left_padding_offset':width-row['prefill_tokens'] if batched else 0,
                 'first_real_position_id':int(positions[i,width-row['prefill_tokens']]) if batched else 0}
                for i,(record,row) in enumerate(zip(records,prepared))]}


class DtypeAudit:
    def __init__(self,model):
        self.rows={};self.handles=[];seen=set()
        for prefix,root in language_roots(model):
            for name,module in root.named_modules():
                if id(module) in seen or list(module.children()):continue
                seen.add(id(module));path=prefix+('.'+name if name else '')
                self.handles.append(module.register_forward_pre_hook(lambda mod,args,path=path:self.observe(path,'inputs',args)))
                self.handles.append(module.register_forward_hook(lambda mod,args,out,path=path:self.observe(path,'outputs',out)))
    def observe(self,path,field,value):
        row=self.rows.setdefault(path,{'inputs':set(),'outputs':set(),'autocast_observed':False})
        def visit(item):
            if isinstance(item,torch.Tensor) and item.is_floating_point():
                row[field].add(str(item.dtype));row['autocast_observed']|=torch.is_autocast_enabled(item.device.type)
            elif isinstance(item,(tuple,list)):
                for x in item:visit(x)
            elif isinstance(item,dict):
                for x in item.values():visit(x)
        visit(value)
    def finish(self):
        for handle in self.handles:handle.remove()
        result={path:{'inputs':sorted(row['inputs']),'outputs':sorted(row['outputs']),'autocast_observed':row['autocast_observed']} for path,row in self.rows.items()}
        self.result=result
        if any(row['autocast_observed'] or any(x!='torch.float32' for x in row['inputs']+row['outputs']) for row in result.values()):
            raise CountercheckBlocked('Observed language/MLP leaf computation is not F32 with autocast disabled')
        return result


def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--bf16-report',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--allow-gpu',action='store_true');p.add_argument('--allow-known-eval');p.add_argument('--memory-cap-gib',type=float,default=12)
    p.add_argument('--deadline-seconds',type=float,default=120);p.add_argument('--reserve-gib',type=float,default=120);p.add_argument('--cache-dir',default='cache/vision');args=p.parse_args(argv)
    if not args.allow_gpu:raise RuntimeError('F32 countercheck requires explicit root--allow-gpu')
    output=Path(args.output).resolve();archive=output.with_suffix('.first_step_logits.pt')
    if output.exists() or archive.exists():raise FileExistsError('Use fresh countercheck artifacts')
    bf16_path=Path(args.bf16_report).resolve();bf16=json.loads(bf16_path.read_text())
    if bf16.get('status')!='passed_diagnostic_reproduction' or bf16.get('full_guard_token_sequences_reproduced') is not True:raise ValueError('Reproduced BF16 reference required')
    guard_path=Path(bf16['guard_path']);guard=json.loads(guard_path.read_text());manifest=Path(args.manifest)
    if sha(guard_path)!=bf16['guard_sha256'] or sha(manifest)!=bf16['manifest_sha256']:raise ValueError('BF16 source evidence changed')
    records,evidence=failed_batch(guard,read_rows(manifest),guard['modes']['none']['mismatched_record_ids'][0])
    if [r['record_id'] for r in records]!=bf16['record_ids']:raise ValueError('Exact four-row membership differs')
    for name,expected in bf16['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen BF16 source changed')
    checkpoint=Path(bf16['checkpoint_identity']['primary_path']);identity=checkpoint_identity(checkpoint)
    if not same_checkpoint_identity(identity,bf16['checkpoint_identity']):raise ValueError('Checkpoint differs')
    reference_archive=Path(bf16['raw_first_step_archive'])
    if sha(reference_archive)!=bf16['raw_first_step_archive_sha256']:raise ValueError('BF16 logits archive changed')
    baseline=torch.load(reference_archive,map_location='cpu',weights_only=True)
    if (baseline.get('record_ids')!=bf16['record_ids'] or baseline.get('checkpoint_sha256')!=identity['sha256'] or
        baseline.get('guard_sha256')!=bf16['guard_sha256']):raise ValueError('BF16 archive identity/order differs')
    from scripts.eval_research_checkpoint_batch4 import package_identity
    environment=package_identity()
    if environment!=bf16['environment']:raise CountercheckBlocked('Package/default precision environment differs from BF16 reference')
    safety=RuntimeGuard(args.allow_known_eval,memory_cap_gib=args.memory_cap_gib,deadline_seconds=args.deadline_seconds)
    safety.arm_allocator_cap();DiskBudget(output.parent,args.reserve_gib).ensure_space(64*2**20)
    report={'schema_version':1,'status':'running','scope':'isolated precision countercheck, not a new evaluation profile or parity-gate pass',
        'bf16_report':str(bf16_path),'bf16_report_sha256':sha(bf16_path),'checkpoint_identity':identity,'guard_sha256':sha(guard_path),'manifest_sha256':sha(manifest),
        'record_ids':bf16['record_ids'],'source_sha256':dict(bf16['source_sha256'],**{'scripts/countercheck_research_batch4_f32.py':sha(__file__)}),
        'environment':environment,'runtime_limits':{'own_memory_cap_bytes':safety.cap,'pytorch_allocator_cap_bytes':int(safety.cap*2/3),
            'minimum_global_free_vram_bytes':24*2**30,'gpu_work_deadline_seconds':args.deadline_seconds,'known_peer_progress_below':256},
        'vision_features_remain_cached_bf16':True,'vision_encoder_forward_calls':0,'cache_identity_changed':False,'gates_or_profiles_changed':False,
        'timing_or_speed_claims':False,'shared_known_eval_binding':safety.peer,'shared_known_eval_binding_sha256':safety.binding_sha,
        'limits':'Requested and observed F32 ordinary tensor math/disabled autocast; opaque4-bit kernel internals are not independently guaranteed by dtype observations.'}
    _atomic_json(output,report)
    try:
        safety.check();model,provenance=load_research_checkpoint(checkpoint,cache_dir=args.cache_dir,reserve_gb=args.reserve_gib);report['model_provenance']=provenance
        if provenance.get('recipe')!='4q' or provenance.get('stage')!='SFT':raise CountercheckBlocked('Only completed4q SFT is supported')
        if model.cache.fingerprint!=bf16['model_provenance']['cache_fingerprint']:raise CountercheckBlocked('Cached vision identity differs')
        before,quant_ids=quantization_snapshot(model)
        if not before['quantized_parameter_count']:raise CountercheckBlocked('No original4-bit parameters found')
        report['quantization_before']=before;safety.check()
        promotion=promote_ordinary_f32(model,quant_ids);report['promotion']=promotion;safety.check()
        if not promotion['runtime_nf4_compute_changes']:raise CountercheckBlocked('No supported explicit4-bit compute_dtype override found')
        promoted,_=quantization_snapshot(model)
        if promoted!=before:raise CountercheckBlocked('Quantized bytes or metadata changed during promotion')
        if set(ordinary_floating_dtypes(model,quant_ids))!={'torch.float32'}:raise CountercheckBlocked('Ordinary tensors are not uniformly F32')
        audit=DtypeAudit(model);native=[]
        for row in records:
            safety.check();native.append(capture_f32(model,[row],batched=False));safety.check()
        safety.check();batch=capture_f32(model,records,batched=True);safety.check()
        try:report['observed_dtype_audit']=audit.finish()
        except CountercheckBlocked:
            report['observed_dtype_audit']=audit.result;raise
        after,_=quantization_snapshot(model)
        if after!=before:raise CountercheckBlocked('Quantized bytes or metadata changed during inference')
        report['all_quantized_bytes_and_metadata_unchanged']=True
        for i,row in enumerate(batch['rows']):
            if row!=bf16['batch4_prompt_rows'][i] or native[i]['rows'][0]!=bf16['native_prompt_rows'][i]:raise CountercheckBlocked('Prompt/position/visual-token membership changed')
        raw=torch.cat([x['raw_logits'] for x in native]);scores=torch.cat([x['output_scores'] for x in native])
        report['f32_raw_comparison']=compare_logits(raw,batch['raw_logits'],model.tokenizer)
        report['f32_processed_scores_comparison']=compare_logits(scores,batch['output_scores'],model.tokenizer)
        report['f32_native_predictions']=[x['predictions'][0] for x in native];report['f32_batch4_predictions']=batch['predictions']
        native_ids=[x['token_ids'][0] for x in native];report['f32_native_token_ids']=native_ids;report['f32_batch4_token_ids']=batch['token_ids']
        report['f32_native_batch4_exact_token_parity']=native_ids==batch['token_ids']
        report['precision_sensitive_rows']=[records[i]['record_id'] for i in range(4) if native_ids[i]!=bf16['captured_native_token_ids'][i] or batch['token_ids'][i]!=bf16['captured_batch4_token_ids'][i]]
        report['bf16_vs_f32_batch_sensitivity']=[{'record_id':records[i]['record_id'],'bf16_raw_rms':bf16['raw_logits_comparison'][i]['rms_diff'],
            'f32_raw_rms':report['f32_raw_comparison'][i]['rms_diff'],'bf16_first_argmax_changed':bf16['raw_logits_comparison'][i]['first_argmax_changed'],
            'f32_first_argmax_changed':report['f32_raw_comparison'][i]['first_argmax_changed']} for i in range(4)]
        report['f32_vs_bf16_native_raw']=compare_logits(baseline['native_raw_logits'],raw,model.tokenizer)
        report['f32_vs_bf16_batch4_raw']=compare_logits(baseline['batch4_raw_logits'],batch['raw_logits'],model.tokenizer)
        archive.parent.mkdir(parents=True,exist_ok=True);fd,temp=tempfile.mkstemp(prefix='.f32-logits-',dir=archive.parent)
        try:
            with os.fdopen(fd,'wb') as h:
                torch.save({'record_ids':bf16['record_ids'],'native_raw_logits':raw,'batch4_raw_logits':batch['raw_logits'],
                            'native_output_scores':scores,'batch4_output_scores':batch['output_scores'],'checkpoint_sha256':identity['sha256']},h)
                h.flush();os.fsync(h.fileno())
            os.replace(temp,archive)
        finally:
            if os.path.exists(temp):os.unlink(temp)
        if sha(bf16_path)!=report['bf16_report_sha256'] or any(sha(ROOT/name)!=expected for name,expected in report['source_sha256'].items()) or not same_checkpoint_identity(identity,checkpoint_identity(checkpoint)):
            raise ValueError('Frozen countercheck inputs changed')
        report.update(status='passed_isolated_f32_countercheck',raw_first_step_archive=str(archive),raw_first_step_archive_sha256=sha(archive),
            safety_checks=safety.checks,peak_own_allocated_bytes=torch.cuda.max_memory_allocated(),peak_own_reserved_bytes=torch.cuda.max_memory_reserved())
        _atomic_json(output,report);return report
    except BaseException as error:
        blocked=isinstance(error,(CountercheckBlocked,torch.cuda.OutOfMemoryError,NotImplementedError)) or (isinstance(error,RuntimeError) and 'promotion' in report)
        report.update(status='blocked' if blocked else 'aborted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'failed',
                      error_type=type(error).__name__,error=str(error),safety_checks=safety.checks,automatic_fallback=False)
        _atomic_json(output,report);raise

if __name__=='__main__':
    def stopped(signum,_frame):raise InterruptedError('Own F32 countercheck interrupted by signal '+str(signum))
    signal.signal(signal.SIGINT,stopped);signal.signal(signal.SIGTERM,stopped);main()
