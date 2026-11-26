#!/usr/bin/env python3
"""Validate one actual pinned LFM checkpoint on CPU, with GPU access hidden.

Tests frozen-backbone input gradients and native cached decoding. This is not a
GPU benchmark, an optimizer/full-weight training test, or a multimodal score.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def json_ready(value):
    if isinstance(value,dict): return {str(k):json_ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)): return [json_ready(v) for v in value]
    return value


def emit(event,**values):
    print(json.dumps({'time':time.time(),'event':event,**values},ensure_ascii=False),flush=True)


def require_gpu_hidden(torch):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ['', '-1']:
        raise RuntimeError('CPU checkpoint validation requires CUDA_VISIBLE_DEVICES empty or -1 before importing torch')
    if torch.cuda.is_available() or torch.cuda.device_count()!=0 or torch.cuda.is_initialized():
        raise RuntimeError('CPU checkpoint validation may not allocate or expose a GPU')


def conv_state_snapshot(cache,config):
    import torch
    states={}
    for index,layer_type in enumerate(config.layer_types):
        if layer_type!='conv': continue
        layer=cache.layers[index]
        values=getattr(layer,'conv_states',None)
        if not isinstance(values,(tuple,list,dict)) or len(values)!=1 or not torch.is_tensor(values[0]):
            raise ValueError(f'Convolution layer {index} has no native convolution cache tensor')
        state=values[0]
        if state.device.type!='cpu' or state.shape[-1]!=config.conv_L_cache or not torch.isfinite(state).all():
            raise ValueError(f'Convolution cache layer {index} has invalid shape/device/values')
        states[index]=state.detach().clone()
    if len(states)!=config.layer_types.count('conv'):
        raise ValueError('Native cache lost one or more convolution states')
    return states


def check_actual_checkpoint(repo='LiquidAI/LFM2.5-8B-A1B',output='research_results/lfm_cpu_checkpoint_validation.json'):
    started=time.monotonic()
    report={'schema_version':1,'status':'running','repo':repo,'scope':'actual_pinned_checkpoint_cpu_compatibility_only',
            'gpu_training_validated':False,'full_weight_training_validated':False,'optimizer_created':False}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    from src.utils.research_storage import _atomic_json
    try:
        import torch
        from torch.nn import functional as F
        from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
        import transformers
        from huggingface_hub import snapshot_download
        require_gpu_hidden(torch)
        torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
        inventory=json.loads((ROOT/'research_results/model_inventory.json').read_text())
        row=next(item for item in inventory if item['repo']==repo)
        if repo!='LiquidAI/LFM2.5-8B-A1B': raise ValueError('This actual compatibility test is scoped to the LFM checkpoint')
        snapshot=snapshot_download(repo,revision=row['sha'],local_files_only=True,
            allow_patterns=['*.json','*.safetensors','*.jinja','*.model','vocab.json','merges.txt'])
        config=AutoConfig.from_pretrained(snapshot,trust_remote_code=False,local_files_only=True)
        if config.model_type!='lfm2_moe': raise ValueError('Pinned checkpoint does not describe native LFM2 MoE')
        tokenizer=AutoTokenizer.from_pretrained(snapshot,trust_remote_code=False,local_files_only=True)
        template=tokenizer.chat_template
        if not isinstance(template,str) or not template.strip(): raise ValueError('Checkpoint has no usable official chat template')
        original_template=(Path(snapshot)/'chat_template.jinja').read_text()
        if template.strip()!=original_template.strip(): raise ValueError('Tokenizer chat template differs from pinned original file')
        rendered=tokenizer.apply_chat_template([{'role':'user','content':'Describe the image.'}],tokenize=False,add_generation_prompt=True)
        if 'Describe the image.' not in rendered or 'assistant' not in rendered or 'user' not in rendered:
            raise ValueError('Official chat template did not preserve expected role/content/generation prompt')
        report.update(revision=row['sha'],snapshot=str(Path(snapshot).resolve()),model_type=config.model_type,
            tokenizer_class=type(tokenizer).__name__,chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
            official_chat_template_matches=True,rendered_chat_prompt=rendered,
            software={'torch':torch.__version__,'transformers':transformers.__version__},cpu_threads=torch.get_num_threads(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        emit('loading_actual_checkpoint',repo=repo,revision=row['sha'],device='cpu',dtype='bfloat16')
        model,loading=AutoModelForCausalLM.from_pretrained(snapshot,trust_remote_code=False,local_files_only=True,
            dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='sdpa',output_loading_info=True)
        failures={name:loading.get(name,[]) for name in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'] if loading.get(name)}
        if failures: raise ValueError(f'Actual checkpoint loading contains unresolved keys/errors: {failures}')
        model.requires_grad_(False).eval()
        if any(parameter.device.type!='cpu' for parameter in model.parameters()): raise ValueError('A model parameter was loaded off CPU')
        require_gpu_hidden(torch)
        count=sum(parameter.numel() for parameter in model.parameters())
        layer_details=[]
        for index,layer in enumerate(model.model.layers):
            ff=layer.feed_forward
            expert=hasattr(ff,'experts')
            entry={'layer':index,'token_mixer':config.layer_types[index],'feed_forward_class':type(ff).__name__,'routed':expert}
            if expert:
                entry.update(experts=config.num_experts,top_k=config.num_experts_per_tok,
                    gate_up_shape=list(ff.experts.gate_up_proj.shape),down_shape=list(ff.experts.down_proj.shape),
                    has_expert_bias=hasattr(ff,'expert_bias'))
            layer_details.append(entry)
        report.update(model_class=type(model).__name__,parameter_count=count,parameter_storage_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            loading_info=json_ready(loading),all_parameters_frozen=True,all_parameters_cpu=True,
            architecture={'hidden_size':config.hidden_size,'num_hidden_layers':config.num_hidden_layers,'vocab_size':config.vocab_size,
                'num_dense_layers':config.num_dense_layers,'num_routed_layers':sum(item['routed'] for item in layer_details),
                'experts_per_routed_layer':config.num_experts,'top_k':config.num_experts_per_tok,
                'moe_intermediate_size':config.moe_intermediate_size,'conv_layers':config.layer_types.count('conv'),
                'attention_layers':config.layer_types.count('full_attention'),'conv_cache_length':config.conv_L_cache,
                'tie_word_embeddings':config.tie_word_embeddings,'norm_topk_prob':config.norm_topk_prob,
                'routed_scaling_factor':config.routed_scaling_factor,'layers':layer_details})
        emit('actual_checkpoint_loaded',parameters=count,seconds=time.monotonic()-started)
        ids=tokenizer.encode('Explain why the sky appears blue during daytime.',add_special_tokens=False)
        if len(ids)<5: raise ValueError('Native tokenizer did not yield enough tokens for fixed CPU probe')
        prompt=torch.tensor([ids[:4]],dtype=torch.long)
        targets=torch.tensor(ids[1:5],dtype=torch.long)
        inputs=model.get_input_embeddings()(prompt).detach().clone().requires_grad_(True)
        hidden=model.model(inputs_embeds=inputs,attention_mask=torch.ones_like(prompt),use_cache=False,return_dict=True).last_hidden_state
        logits=model.lm_head(hidden.reshape(-1,config.hidden_size))
        loss=F.cross_entropy(logits.float(),targets)
        if not torch.isfinite(loss): raise ValueError('Frozen actual checkpoint CE is non-finite')
        loss.backward()
        gradient=inputs.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.float().abs().max().item()==0:
            raise ValueError('Actual frozen checkpoint did not provide finite nonzero input gradients')
        if any(parameter.grad is not None for parameter in model.parameters()): raise ValueError('CPU probe allocated full-weight gradients')
        report['input_gradient_check']={'input_shape':list(inputs.shape),'supervised_head_positions':4,'ce_loss':loss.item(),
            'input_gradient_norm':gradient.float().norm().item(),'input_gradient_abs_max':gradient.float().abs().max().item(),
            'input_gradient_per_position_norm':gradient.float().norm(dim=-1).tolist(),'all_finite':True,'all_weight_gradients_none':True}
        emit('input_gradient_passed',**report['input_gradient_check'])
        del hidden,logits,inputs,gradient,loss
        with torch.no_grad():
            first=model(input_ids=prompt,attention_mask=torch.ones_like(prompt),use_cache=True,logits_to_keep=1,return_dict=True)
            cache=first.past_key_values
            before=conv_state_snapshot(cache,config)
            token1=first.logits[:,-1].argmax(-1,keepdim=True)
            next_output=model(input_ids=token1,attention_mask=torch.ones(1,5,dtype=torch.long),past_key_values=cache,
                              use_cache=True,logits_to_keep=1,return_dict=True)
            after=conv_state_snapshot(next_output.past_key_values,config)
            if any(before[i].shape!=after[i].shape for i in before): raise ValueError('Convolution state shape changed during cached decode')
            changed=sum(not torch.equal(before[i],after[i]) for i in before)
            if changed==0: raise ValueError('Convolution cache states did not update during native cached decode')
            token2=next_output.logits[:,-1].argmax(-1,keepdim=True)
            cached_tokens=torch.cat([token1,token2],dim=1)
            current=prompt.clone();uncached=[]
            for _ in range(2):
                prediction=model(input_ids=current,attention_mask=torch.ones_like(current),use_cache=False,logits_to_keep=1,return_dict=True)
                token=prediction.logits[:,-1].argmax(-1,keepdim=True);uncached.append(token);current=torch.cat([current,token],dim=1)
            uncached_tokens=torch.cat(uncached,dim=1)
            if not torch.equal(cached_tokens,uncached_tokens):
                raise ValueError(f'Native cached/noncached greedy tokens differ: {cached_tokens.tolist()} vs {uncached_tokens.tolist()}')
            generated=model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),max_new_tokens=2,do_sample=False,
                use_cache=True,eos_token_id=None,pad_token_id=tokenizer.pad_token_id or config.pad_token_id)
            generated_tokens=generated[:,prompt.shape[1]:]
            if generated_tokens.shape[1]!=2 or not torch.equal(generated_tokens,cached_tokens):
                raise ValueError('Native generate disagrees with manual cached greedy decode')
        report['cached_generation_check']={'cache_class':type(cache).__name__,'prompt_token_ids':prompt.tolist(),
            'cached_new_token_ids':cached_tokens.tolist(),'uncached_new_token_ids':uncached_tokens.tolist(),
            'native_generate_new_token_ids':generated_tokens.tolist(),'cached_and_uncached_match':True,'native_generate_matches':True,
            'conv_state_count':len(before),'conv_states_changed_after_one_decode':changed,
            'conv_state_shapes':{str(i):list(tensor.shape) for i,tensor in after.items()},'conv_state_preserved_and_updated':True}
        require_gpu_hidden(torch)
        report.update(status='passed',elapsed_seconds=time.monotonic()-started,gpu_initialized=False,
                      limits='CPU bf16 actual-checkpoint compatibility and frozen-input-gradient probe only; CUDA kernels, optimizer, full-weight training and VLM metrics remain unvalidated here')
        _atomic_json(output,json_ready(report))
        emit('actual_checkpoint_cpu_validation_passed',output=str(output),elapsed_seconds=report['elapsed_seconds'])
        return report
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        _atomic_json(output,json_ready(report))
        emit('actual_checkpoint_cpu_validation_failed',error_type=type(error).__name__,error=str(error))
        raise


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--repo',default='LiquidAI/LFM2.5-8B-A1B')
    parser.add_argument('--output',default='research_results/lfm_cpu_checkpoint_validation.json')
    args=parser.parse_args();check_actual_checkpoint(args.repo,args.output)

if __name__=='__main__':main()
