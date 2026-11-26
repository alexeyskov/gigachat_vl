"""Inference-only adapters for source-bound research checkpoints and interleaved images."""
from __future__ import annotations
import copy,hashlib,inspect,json,re
from contextlib import ExitStack
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MARKER='[BENCHMARK_IMAGE_'


def prepare_interleaved(model,message,max_new_tokens):
    import torch
    pieces=[];paths=[]
    for item in message:
        if item['type']=='text':
            if MARKER in item['value']:raise ValueError('Reserved image marker collision')
            pieces.append(item['value'])
        elif item['type']=='image':
            pieces.append(f'{MARKER}{len(paths)}]\n');paths.append(item['value'])
        else:raise ValueError('Only image/text benchmark inputs are supported')
    prompt=model.tokenizer.apply_chat_template([{'role':'user','content':''.join(pieces)}],
        tokenize=False,add_generation_prompt=True)
    parts=re.split(r'(\[BENCHMARK_IMAGE_\d+\])',prompt)
    tokens=[part for part in parts if re.fullmatch(r'\[BENCHMARK_IMAGE_\d+\]',part)]
    expected=[f'{MARKER}{i}]' for i in range(len(paths))]
    if tokens!=expected:raise ValueError('Chat template changed image placement/order')
    features=model.cache.get_many(paths,model.encode) if paths else []
    if len(features)!=len(paths):raise ValueError('Image/cache count differs')
    embedding=model.base.get_input_embeddings();device=embedding.weight.device
    chunks=[];modality=[];image_index=0;counts=[]
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        for part in parts:
            if re.fullmatch(r'\[BENCHMARK_IMAGE_\d+\]',part):
                feature=features[image_index];image_index+=1
                if feature.ndim!=2 or not feature.shape[0] or not torch.isfinite(feature).all():
                    raise ValueError('Invalid cached visual features')
                value=model.projector(feature.to(device=device,dtype=next(model.projector.parameters()).dtype))
                chunks.append(value);modality.extend([True]*len(value));counts.append(len(value))
            elif part:
                ids=model.tokenizer.encode(part,add_special_tokens=False)
                chunks.append(embedding(torch.tensor(ids,device=device,dtype=torch.long)))
                modality.extend([False]*len(ids))
    inputs=torch.cat(chunks,dim=0).unsqueeze(0);length=inputs.shape[1]
    maximum=min(model.args.max_length,int(getattr(model.base.config,'max_position_embeddings',model.args.max_length)))
    if length+max_new_tokens>maximum:raise ValueError('Benchmark context exceeded; no images/tokens silently dropped')
    return dict(inputs_embeds=inputs,attention_mask=torch.ones((1,length),device=device,dtype=torch.bool),
                modality_mask=torch.tensor([modality],device=device,dtype=torch.bool),
                prefill_tokens=length,visual_token_counts=counts,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


class ResearchCheckpointVLM:
    def __init__(self,run,*,checkpoint='best',max_new_tokens=64,max_context=4096,
                 cache_dir=None,reserve_gb=32):
        from src.utils import research_stock_determinism as stock
        stock.bootstrap_environment();stock.activate()
        run=Path(run).resolve(strict=True);config=json.loads((run/'resolved_config.json').read_bytes())
        path=run/checkpoint/'state.pt'
        if not path.is_file():raise ValueError('Retained best/last checkpoint required')
        model_name=config['model']
        cache_dir=cache_dir or config.get('cache_dir','cache/vision')
        with ExitStack() as stack:
            if config['vision']=='Qwen/Qwen2.5-VL-7B-Instruct' and model_name in (
                'LiquidAI/LFM2.5-8B-A1B','ai-sage/GigaChat3.1-10B-A1.8B-bf16'):
                from src.utils.research_qwen_payload_bridge_v1 import patched_private_validation
                stack.enter_context(patched_private_validation())
            if 'additive_execution' in config:
                from src.utils.research_additive_runner_v1 import load_checkpoint
                model,_,_,_,payload=load_checkpoint(path)
                provenance=dict(additive_execution=config['additive_execution'],model_ref=payload['model_ref'],
                    vision_ref=payload['vision_ref'],training_step=payload['step'])
                del payload
            elif model_name==stock.SMALL:
                from src.utils import research_small_budget_v3 as small
                small.validate_execution(config)
                if config['vision']==small.QWEN:
                    from src.utils import research_small_router_stability_v1 as corrected
                    corrected.validate_execution(config);stack.enter_context(corrected.patched_runner())
                stack.enter_context(small.patched_stock_runner(small.RELEASE))
                if config['vision']==small.QWEN:
                    from src.utils.research_resource_vision_v3 import composed
                    model,provenance=composed().load_checkpoint(path,cache_dir=cache_dir,reserve_gb=reserve_gb)
                else:model,provenance=stock.inference().load_research_checkpoint(path,cache_dir=cache_dir,reserve_gb=reserve_gb)
            elif model_name=='LiquidAI/LFM2.5-8B-A1B':
                from src.utils.research_lfm_budget_v3 import inference
                model,provenance=inference().load_research_checkpoint(path,cache_dir=cache_dir,reserve_gb=reserve_gb)
            elif model_name=='ai-sage/GigaChat3.1-10B-A1.8B-bf16':
                from src.utils.research_giga_budget_v3 import implementation
                model,provenance=implementation().inference().load_research_checkpoint(path,cache_dir=cache_dir,reserve_gb=reserve_gb)
            else:raise ValueError('Unsupported research checkpoint model')
        # The saved training configuration and checkpoint remain unchanged.
        model.args=copy.copy(model.args);model.args.max_length=max_context
        if model_name=='ai-sage/GigaChat3.1-10B-A1.8B-bf16':
            from src.eval.research_giga_eot_v1 import install_generation_stop
            install_generation_stop(model)
        self.model=model;self.max_new_tokens=max_new_tokens
        self.loading=dict(backend='source_bound_research_checkpoint',run=str(run),checkpoint=str(path.resolve()),
            max_context=max_context,max_new_tokens=max_new_tokens,provenance=provenance,
            training_context=config['max_length'],interleaved_images=True)

    def generate(self,message):
        import torch
        if hasattr(self.model.args,'additive_execution'):
            from src.model.research_additive_visual_v1 import set_modality_mask
        else:
            from src.model.research_moe import set_modality_mask
        prepared=prepare_interleaved(self.model,message,self.max_new_tokens)
        set_modality_mask(self.model.base,prepared['modality_mask'],prepared['attention_mask'])
        kwargs={k:prepared[k] for k in ('inputs_embeds','attention_mask')}
        kwargs.update(do_sample=False,num_beams=1,max_new_tokens=self.max_new_tokens,use_cache=True,
            return_dict_in_generate=True,output_scores=True,pad_token_id=self.model.tokenizer.pad_token_id)
        if 'logits_to_keep' in inspect.signature(self.model.base.forward).parameters:kwargs['logits_to_keep']=1
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.model.base.generate(**kwargs)
        for scores in output.scores:
            if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any(-1).all():
                raise FloatingPointError('Invalid benchmark generation logits')
        count=len(output.scores);ids=output.sequences[0,-count:].cpu().tolist() if count else []
        return dict(prediction=self.model.tokenizer.decode(ids,skip_special_tokens=True).strip(),
            generated_tokens=count,generated_token_ids=ids,
            **{k:prepared[k] for k in ('prefill_tokens','visual_token_counts','prompt_sha256')})
