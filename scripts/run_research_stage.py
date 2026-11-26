#!/usr/bin/env python3
"""Finite-manifest research trainer. Uses frozen content-addressed vision features.

No network publication. Checkpoints are local best/last resumable payloads.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageOps
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from huggingface_hub import snapshot_download, hf_hub_download
from src.model.smallthinker import register_smallthinker
from src.model.gigachat_vl import _without_transformers_allocator_warmup
from scripts.extract_vision_features import VisionFeatureExtractor
from src.utils.research_storage import DiskBudget, BestLastCheckpointManager, trainable_state_dict
from src.dataset.vision_cache import VisionFeatureCache, encoder_fingerprint

SELECTION_DEFAULTS = {'selection_metric':'ce','selection_manifest':None,
                      'selection_per_group':32,'selection_max_new_tokens':64,
                      'selection_manifest_sha256':None}


def log(path, payload, console=True):
    payload = dict(time=time.time(), **payload)
    with Path(path).open('a') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')
    if console:print(json.dumps(payload, ensure_ascii=False), flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Tiiny/SmallThinker-4BA0.6B-Instruct')
    p.add_argument('--vision', default='google/siglip2-so400m-patch14-384')
    p.add_argument('--vision-preprocessing-profile',default=None)
    p.add_argument('--manifest', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--text-manifest')
    p.add_argument('--text-validation')
    p.add_argument('--text-fraction',type=float,default=.15)
    p.add_argument('--text-loss-weight',type=float,default=None,
                   help='SFT only: fixed text objective weight with per-stream global effective-batch token denominators')
    p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=['A0','A1','SFT'], required=True)
    p.add_argument('--recipe', choices=['1a','1b','1c','2a','2b','2c','3a','3b','3c','4q','4f'], default='4f')
    p.add_argument('--alignment-checkpoint')
    p.add_argument('--initial-checkpoint', help='Restore all learned tensors from previous stage')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--checkpoint-format',choices=['legacy','bundle'],default='legacy',help='Optional split model/optimizer bundle; default keeps legacy artifacts')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--optimizer-backend',choices=['torch_adamw32','bnb_paged_adamw32','cpu_grad_bnb_paged_adamw32','cpu_grad_bnb_paged_adamw8'],default='torch_adamw32')
    p.add_argument('--max-records', type=int)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--effective-batch-size', type=int, default=64)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--eval-records', type=int, default=64)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--selection-metric',choices=['ce','group_macro'],default='ce')
    p.add_argument('--selection-manifest',help='Group macro: train-derived heldout JSONL; defaults to --validation')
    p.add_argument('--selection-per-group',type=int,default=32)
    p.add_argument('--selection-max-new-tokens',type=int,default=64)
    p.add_argument('--reserve-gb', type=float, default=120)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cache-dir', default='cache/vision')
    return p.parse_args(argv)


def validate_args(args):
    if not hasattr(args,'text_loss_weight'):args.text_loss_weight=None
    if args.text_loss_weight is not None:
        if not math.isfinite(args.text_loss_weight) or not 0<=args.text_loss_weight<=1:
            raise ValueError('text_loss_weight must be finite and between zero and one')
        if getattr(args,'stage',None)!='SFT':raise ValueError('text_loss_weight is SFT only')
    if not hasattr(args,'checkpoint_format'):args.checkpoint_format='legacy'
    if args.checkpoint_format not in ('legacy','bundle'):raise ValueError('Unknown checkpoint format')
    if not hasattr(args,'prompt_schema_version'):args.prompt_schema_version=2
    if args.prompt_schema_version not in (1,2):raise ValueError('Unsupported prompt schema version')
    if not hasattr(args,'vision_preprocessing_profile'):args.vision_preprocessing_profile=None
    if args.vision_preprocessing_profile and 'siglip' in getattr(args,'vision','').lower():
        raise ValueError('SigLIP uses its native fixed preprocessing; this profile flag is for donor backends')
    for name,value in SELECTION_DEFAULTS.items():
        if name!='selection_manifest_sha256' and not hasattr(args,name):setattr(args,name,value)
    if args.selection_metric not in ('ce','group_macro'):raise ValueError('Unknown selection metric')
    for name in ('batch_size','effective_batch_size','max_length','epochs','eval_records','eval_every'):
        if getattr(args,name)<=0:raise ValueError(f'{name} must be positive')
    for name in ('max_records','max_steps'):
        if getattr(args,name) is not None and getattr(args,name)<=0:raise ValueError(f'{name} must be positive')
    if args.effective_batch_size<args.batch_size or args.effective_batch_size%args.batch_size:
        raise ValueError('effective_batch_size must be a positive multiple of batch_size')
    if not 0<=args.text_fraction<1:raise ValueError('text_fraction must be in [0,1)')
    if args.initial_checkpoint and args.alignment_checkpoint:
        raise ValueError('Choose exactly one initialization checkpoint')
    if args.selection_metric=='group_macro' and getattr(args,'stage',None)!='SFT':
        raise ValueError('group_macro checkpoint selection is SFT only')
    if args.selection_metric=='ce' and args.selection_manifest:
        raise ValueError('--selection-manifest requires group_macro selection')
    for name in ('selection_per_group','selection_max_new_tokens'):
        if isinstance(getattr(args,name),bool) or getattr(args,name)<=0:raise ValueError(f'{name} must be positive')


def validate_resume_config(saved,current):
    permitted={'resume','output','eval_every','reserve_gb','cache_dir','initial_checkpoint','alignment_checkpoint'}
    defaults={'text_loss_weight':None,'checkpoint_format':'legacy','optimizer_backend':'torch_adamw32','vision_preprocessing_profile':None,'prompt_schema_version':1,**SELECTION_DEFAULTS}
    differences={k:(saved.get(k,defaults.get(k)),current.get(k,defaults.get(k))) for k in set(saved)|set(current)
                 if k not in permitted and saved.get(k,defaults.get(k))!=current.get(k,defaults.get(k))}
    if differences:raise ValueError(f'Incompatible resume config: {differences}')


def validate_model_binding(payload,model):
    """Bind a delta checkpoint to its exact frozen language/vision bases."""
    for field,ref in [('model_ref',model.model_ref),('vision_ref',model.vision_ref)]:
        if payload.get(field) is not None and payload[field]!=ref:
            raise ValueError(f'Initialization {field} differs from current pinned checkpoint')
    fingerprint=payload.get('cache_fingerprint')
    if fingerprint is not None and fingerprint!=model.cache.fingerprint:
        raise ValueError('Initialization vision feature fingerprint differs')


def validate_initial_checkpoint(payload,model,args):
    """Prevent transferring a projector/router across incompatible experiments."""
    validate_model_binding(payload,model)
    previous=payload.get('config',{})
    old_quant=str(previous.get('recipe','')).startswith('1') or previous.get('recipe')=='4q'
    new_quant=args.recipe.startswith('1') or args.recipe=='4q'
    if 'recipe' in previous and old_quant!=new_quant:
        raise ValueError('Initialization quantization profile differs from current experiment')
    old_stage=previous.get('stage')
    if args.stage=='A0':raise ValueError('A0 must start with a freshly initialized alignment MLP')
    if old_stage is not None:
        allowed={'A1':{'A0'},'SFT':{'A0','A1','SFT'}}
        if old_stage not in allowed.get(args.stage,set()):
            raise ValueError(f'Invalid initialization stage {old_stage} -> {args.stage}')
        if old_stage!='A0' and previous.get('recipe')!=args.recipe:
            raise ValueError('Initialization recipe differs from current architecture')
    return True


def load_records(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def deduplicate_selection_records(records):
    """Remove identical weighted-mixture repetitions, never conflicting labels."""
    unique={}
    for record in records:
        record_id=record.get('record_id')
        if not isinstance(record_id,str) or not record_id:raise ValueError('Selection needs stable record_id values')
        if record_id in unique:
            if unique[record_id]!=record:raise ValueError(f'Conflicting duplicate selection record_id: {record_id}')
        else:unique[record_id]=record
    return list(unique.values())


def prepare_group_macro_selection(config,loader=None,base_records=None):
    """Shared by training and checkpoint replay; no eval-records subsampling."""
    from src.eval.research_selection import build_internal_validation
    loader=loader or load_records
    values=config if isinstance(config,dict) else vars(config)
    if values.get('stage')!='SFT' or values.get('selection_metric','ce')!='group_macro':
        raise ValueError('Internal group macro selection requires an SFT group_macro configuration')
    explicit=values.get('selection_manifest')
    path=explicit or values['validation']
    records=list(base_records) if base_records is not None and not explicit else loader(path)
    if values.get('text_validation'):
        text=loader(values['text_validation'])
        if any(row.get('image_paths') for row in text):raise ValueError('Text validation must contain text-only records')
        records+=text
    unique=deduplicate_selection_records(records)
    return build_internal_validation(unique,per_group=values.get('selection_per_group',32),seed=values['seed'])


def evaluate_text_only_ce(model,records,batch_size):
    from src.eval.research_selection import selection_generation_mode
    if not records:return None
    if any(row.get('image_paths') for row in records):raise ValueError('Text-only CE received an image-bearing record')
    total=0.;tokens=0
    with selection_generation_mode(model):
        for offset in range(0,len(records),batch_size):
            loss,count,_=model.forward_loss(model.prepare(records[offset:offset+batch_size]))
            total+=loss.item();tokens+=count
    if tokens<=0:raise ValueError('Text-only validation has no supervised tokens')
    value=total/tokens
    if not math.isfinite(value):raise FloatingPointError('Nonfinite text-only validation CE')
    return {'text_only_heldout_ce':value,'text_only_records':len(records),'text_only_supervised_tokens':tokens}


def write_selection_artifacts(output,result,step,is_best):
    """At most best/last prediction sets, rather than one set per validation."""
    from src.utils.research_storage import _atomic_json
    directory=Path(output)/'selection';directory.mkdir(parents=True,exist_ok=True)
    for label in (['last','best'] if is_best else ['last']):
        destination=directory/f'{label}.predictions.jsonl'
        temporary=directory/f'.{label}.predictions.tmp'
        try:
            with temporary.open('w') as handle:
                for row in result['predictions']:
                    handle.write(json.dumps(dict(row,validation_step=step),ensure_ascii=False)+'\n')
                handle.flush();os.fsync(handle.fileno())
            os.replace(temporary,destination)
        finally:temporary.unlink(missing_ok=True)
        _atomic_json(directory/f'{label}.protocol.json',dict(result['protocol'],validation_step=step))
        _atomic_json(directory/f'{label}.metrics.json',dict(result['metrics'],validation_step=step))


def training_prompt_messages(record,num_images,prompt_schema_version=2):
    """Preserve public system/history instructions and exclude the current gold.

    Unified rows contain a conversation prefix ending in the current user turn;
    ``answer`` is appended separately only when constructing supervised targets.
    """
    question=record['question']
    if not isinstance(question,str) or not question.strip():raise ValueError('Question must be nonempty text')
    if prompt_schema_version not in (1,2):raise ValueError('Unsupported prompt schema version')
    supplied=record.get('messages') if prompt_schema_version==2 else None
    if supplied is None or supplied==[]:
        messages=[{'role':'user','content':question}]
    else:
        if not isinstance(supplied,list):raise ValueError('record.messages must be a conversation prefix list')
        messages=[]
        for message in supplied:
            if (not isinstance(message,dict) or message.get('role') not in ('system','developer','user','assistant','tool') or
                    not isinstance(message.get('content'),str)):
                raise ValueError('record.messages must contain valid text role/content messages')
            messages.append(dict(message))
        if messages[-1]['role']!='user':raise ValueError('Conversation prefix must end at the target user turn, before the gold assistant answer')
        if messages[-1]['content'].strip()!=question.strip():raise ValueError('Final user message does not align with record.question')
        messages[-1]['content']=question
    marker='[RESEARCH_IMAGE]'
    if any(marker in message['content'] for message in messages):raise ValueError('Question/messages collide with the reserved image insertion marker')
    if num_images:
        messages[-1]['content']=marker+'\n'+messages[-1]['content']
    return messages


def resolve_model(repo):
    if Path(repo).is_dir():
        return str(Path(repo).resolve()), {'repo':repo,'revision':'local'}
    inventory = json.loads((ROOT/'research_results/model_inventory.json').read_text())
    row = next(x for x in inventory if x['repo']==repo)
    # A vision-only donor may intentionally contain just its indexed vision shard.
    path = str(Path(hf_hub_download(repo,'config.json',revision=row['sha'],local_files_only=True)).parent)
    return path, {'repo':repo,'revision':row['sha']}


class ResearchModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        register_smallthinker()
        self.model_path,self.model_ref=resolve_model(args.model)
        self.vision_path,self.vision_ref=resolve_model(args.vision)
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_path,trust_remote_code=False)
        self.tokenizer.pad_token=self.tokenizer.eos_token
        cfg=AutoConfig.from_pretrained(self.model_path,trust_remote_code=False)
        quantized=args.recipe.startswith('1') or args.recipe=='4q'
        if quantized and cfg.model_type not in ('smallthinker','lfm2_moe','deepseek_v3','gpt_oss'):
            raise NotImplementedError('Packed expert QLoRA backend must be validated before use.')
        kwargs=dict(dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False)
        if quantized:
            kwargs.update(device_map={'':0},quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,llm_int8_skip_modules=['primary_router','lm_head']+(['gate','router'] if cfg.model_type!='smallthinker' else [])))
        with _without_transformers_allocator_warmup():
            if cfg.model_type=='gpt_oss':
                from src.model.research_loading import load_gpt_oss_for_research
                self.llm,self.model_loading_report=load_gpt_oss_for_research(self.model_path,
                    revision=self.model_ref['revision'],stage=args.stage,recipe=args.recipe,device='cuda:0')
            else:
                self.llm=AutoModelForCausalLM.from_pretrained(self.model_path,**kwargs)
        if not quantized: self.llm.to('cuda')
        self.llm.requires_grad_(False)
        if quantized and args.stage!='A0' and cfg.model_type!='gpt_oss':
            from peft import LoraConfig,get_peft_model
            targets=['q_proj','k_proj','v_proj','o_proj','up','gate','down'] if cfg.model_type=='smallthinker' else [
                n for n,m in self.llm.named_modules() if isinstance(m,nn.Linear) and n!='lm_head' and not n.endswith(('.gate','.router')) and '.experts.' not in n]
            self.llm=get_peft_model(self.llm,LoraConfig(task_type='CAUSAL_LM',r=16,
                lora_alpha=32,lora_dropout=.05,bias='none',target_modules=targets))
        self.base=self.llm.get_base_model() if hasattr(self.llm,'get_base_model') else self.llm
        if quantized and cfg.model_type=='gpt_oss':
            from src.model.research_quantization import set_packed_lora_trainable
            set_packed_lora_trainable(self.base,args.stage=='SFT' or (args.stage=='A1' and args.recipe=='4q'))
        if quantized and cfg.model_type in ('lfm2_moe','deepseek_v3'):
            from src.model.research_quantization import quantize_packed_experts,set_packed_lora_trainable
            self.packed_quantization_report=quantize_packed_experts(self.base,rank=0 if args.stage=='A0' else 16)
            set_packed_lora_trainable(self.base,args.stage=='SFT' or (args.stage=='A1' and args.recipe=='4q'))
        if 'siglip' in args.vision.lower():
            self.extractor=VisionFeatureExtractor(self.vision_path,torch.device('cuda'),torch.bfloat16)
            vision_cfg=AutoConfig.from_pretrained(self.vision_path)
            vcfg=getattr(vision_cfg,'vision_config',vision_cfg)
            dim=vcfg.hidden_size
        else:
            from src.model.research_vision import build_frozen_vision
            profile=getattr(args,'vision_preprocessing_profile',None)
            options={'preprocessing_profile':profile} if profile is not None else {}
            self.extractor=build_frozen_vision(self.vision_path,torch.device('cuda'),torch.bfloat16,**options)
            dim=self.extractor.feature_dim
            vision_cfg=AutoConfig.from_pretrained(self.vision_path)
            vcfg=getattr(vision_cfg,'vision_config',vision_cfg)
        self.extractor.vision_tower.requires_grad_(False).eval()
        if self.extractor.vision_projector is not None:
            self.extractor.vision_projector.requires_grad_(False).eval()
        self.projector=nn.Sequential(nn.Linear(dim,cfg.hidden_size),nn.GELU(),
                                    nn.Linear(cfg.hidden_size,cfg.hidden_size)).to('cuda',dtype=torch.float32)
        self.wrappers=[]
        if args.stage!='A0':
            self.projector.requires_grad_(False)
            if not quantized:
                # Keep trainable base weights and Adam moments in fp32; autocast
                # supplies bf16 GEMMs without losing small full-training updates.
                self.llm.float().requires_grad_(True)
            if args.recipe[0] in '123':
                from src.model.research_moe import install_research_moe,configure_research_trainability
                if args.stage=='A1': self.llm.requires_grad_(False)
                self.wrappers=install_research_moe(self.base,int(args.recipe[0]),args.recipe[1],added_experts=2)
                configure_research_trainability(self.base,stage=args.stage)
        for parameter in self.parameters():
            if parameter.requires_grad and parameter.is_floating_point():parameter.data=parameter.data.float()
        self.llm.config.use_cache=False
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        processor_config = (self.extractor.fingerprint_config if hasattr(self.extractor,'fingerprint_config') else self.extractor.processor.to_dict() if hasattr(self.extractor.processor,'to_dict') else vcfg.to_dict())
        fingerprint=dict(**self.vision_ref,backend=self.extractor.backend,
            processor=processor_config,feature_point='native_patch_sequence_before_our_mlp',
            compute_dtype='bf16',storage_dtype='bf16',schema_version=1)
        self.cache=VisionFeatureCache(args.cache_dir,encoder_fingerprint(fingerprint),reserve_bytes=int(args.reserve_gb*2**30),ram_max_bytes=2*2**30)

    def encode(self,paths):
        try:
            return self._encode_once(paths)
        except torch.cuda.OutOfMemoryError:
            if len(paths)<=1:raise
            print(json.dumps({'event':'vision_oom_retry','images':len(paths),'next_batch':len(paths)//2}),flush=True)
        torch.cuda.empty_cache()
        middle=len(paths)//2
        return self.encode(paths[:middle])+self.encode(paths[middle:])

    def _encode_once(self,paths):
        if hasattr(self.extractor,'encode_paths'):return self.extractor.encode_paths(paths)
        images=[]
        for path in paths:
            with Image.open(path) as im: images.append(ImageOps.exif_transpose(im).convert('RGB'))
        tensors,_=self.extractor.encode_batch(images)
        return [x.detach().cpu().clone() for x in tensors]

    def tokenize(self,record):
        n=len(record.get('image_paths',[]))
        if n>1: raise NotImplementedError('Initial smoke profile supports a single image per record.')
        # Split the user message at a dedicated marker without growing base vocabulary.
        marker='[RESEARCH_IMAGE]'
        # Old checkpoints deliberately retain their historical question-only
        # tokenizer for exact replay; fresh main() jobs record schema version 2.
        messages=training_prompt_messages(record,n,getattr(self.args,'prompt_schema_version',2))
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        if getattr(self.base.config,'model_type',None)=='gpt_oss':
            prompt+='<|channel|>final<|message|>'
        full=self.tokenizer.apply_chat_template(messages+[{'role':'assistant','content':record['answer']}],tokenize=False,add_generation_prompt=False)
        if not full.startswith(prompt): raise ValueError('Chat answer prefix mismatch')
        before,after=(prompt.split(marker,1) if n else ('',prompt))
        pre=self.tokenizer.encode(before,add_special_tokens=False)
        post=self.tokenizer.encode(after,add_special_tokens=False)
        answer=self.tokenizer.encode(full[len(prompt):],add_special_tokens=False)
        if not answer: raise ValueError('Empty answer')
        return pre,post,answer

    def prepare(self,records):
        paths=[p for record in records for p in record.get('image_paths',[])]
        hashes=[h for record in records for h in record.get('image_hashes',[])]
        cache_kwargs={'image_hashes':hashes} if len(hashes)==len(paths) else {}
        all_features=self.cache.get_many(paths,self.encode,**cache_kwargs) if paths else []
        rows=[];cursor=0
        for record in records:
            pre,post,answer=self.tokenize(record)
            count=len(record.get('image_paths',[]))
            features=all_features[cursor:cursor+count];cursor+=count
            n=sum(x.shape[0] for x in features)
            available=self.args.max_length-n-len(pre)-len(post)
            if available<1:
                raise ValueError(f'Expanded prompt too long: {len(pre)+len(post)+n}; max={self.args.max_length}')
            answer=answer[:available]
            rows.append((pre,post,answer,features))
        return rows

    def forward_loss(self,rows):
        return self._forward_loss(rows,components=False)

    def forward_loss_components(self,rows):
        return self._forward_loss(rows,components=True)

    def _forward_loss(self,rows,components=False):
        embed=self.llm.get_input_embeddings()
        embs=[]; labels=[]; masks=[]
        for pre,post,answer,features in rows:
            pre_ids=torch.tensor(pre,device='cuda',dtype=torch.long)
            tail_ids=torch.tensor(post+answer,device='cuda',dtype=torch.long)
            chunks=[embed(pre_ids)]
            for feature in features:
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_available()):
                    chunks.append(self.projector(feature.to('cuda',dtype=next(self.projector.parameters()).dtype)))
            chunks.append(embed(tail_ids))
            x=torch.cat(chunks)
            embs.append(x)
            prompt_len=len(pre)+len(post)+sum(f.shape[0] for f in features)
            labels.append(torch.tensor([-100]*prompt_len+answer,device='cuda'))
            masks.append(torch.tensor([False]*len(pre)+[True]*sum(f.shape[0] for f in features)+[False]*(len(post)+len(answer)),device='cuda'))
        size=max(x.shape[0] for x in embs)
        xs=torch.stack([F.pad(x,(0,0,0,size-x.shape[0])) for x in embs])
        ys=torch.stack([F.pad(y,(0,size-y.shape[0]),value=-100) for y in labels])
        am=torch.stack([torch.arange(size,device='cuda')<x.shape[0] for x in embs])
        mm=torch.stack([F.pad(m,(0,size-m.shape[0])) for m in masks])
        if self.wrappers:
            from src.model.research_moe import set_modality_mask
            set_modality_mask(self.base,mm,am)
        # Only supervised positions need a 152K-wide vocabulary projection.
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_available()):
            hidden=self.base.model(inputs_embeds=xs,attention_mask=am,use_cache=False,return_dict=True).last_hidden_state
            target=ys[:,1:];valid=target.ne(-100)
            logits=self.base.lm_head(hidden[:,:-1][valid])
        if not components:
            loss=F.cross_entropy(logits.float(),target[valid],reduction='sum')
            return loss,int(valid.sum()),int(am.sum())
        per_token=F.cross_entropy(logits.float(),target[valid],reduction='none')
        multimodal_samples=torch.tensor([bool(row[3]) for row in rows],device=target.device,dtype=torch.bool)
        multimodal_targets=multimodal_samples[:,None].expand_as(target)[valid]
        return {'multimodal_loss_sum':per_token[multimodal_targets].sum(),
                'text_only_loss_sum':per_token[~multimodal_targets].sum(),
                'multimodal_supervised_tokens':int(multimodal_targets.sum()),
                'text_only_supervised_tokens':int((~multimodal_targets).sum()),
                'sequence_tokens':int(am.sum())}


def train_accumulation_group(model,optimizer,prepared,parameters,microbatch_size,text_loss_weight=None):
    """Retry activation OOMs without changing effective batch or CE denominator."""
    rows=[row for batch in prepared for row in batch]
    denominator=sum(len(row[2]) for row in rows)
    text_only_supervised_tokens=sum(len(row[2]) for row in rows if not row[3])
    text_only_records=sum(not row[3] for row in rows)
    multimodal_supervised_tokens=denominator-text_only_supervised_tokens
    if text_loss_weight is not None and (not math.isfinite(text_loss_weight) or not 0<=text_loss_weight<=1):
        raise ValueError('Invalid fixed text loss weight')
    if not denominator:raise ValueError('No supervised targets in effective batch')
    rng_python=random.getstate();rng_numpy=np.random.get_state();rng_torch=torch.get_rng_state()
    rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    while True:
        optimizer.zero_grad(set_to_none=True)
        losses=0.;actual_tokens=0;sequence_tokens=0;loss=None
        multimodal_losses=0.;text_losses=0.;objective=0.
        try:
            for offset in range(0,len(rows),microbatch_size):
                batch=rows[offset:offset+microbatch_size]
                if text_loss_weight is None:
                    loss,n,seq=model.forward_loss(batch)
                    (loss/denominator).backward()
                    losses+=loss.item();actual_tokens+=n;sequence_tokens+=seq
                else:
                    parts=model.forward_loss_components(batch)
                    expected_text=sum(len(row[2]) for row in batch if not row[3])
                    expected_mm=sum(len(row[2]) for row in batch if row[3])
                    if parts['text_only_supervised_tokens']!=expected_text or parts['multimodal_supervised_tokens']!=expected_mm:
                        raise ValueError('Forward stream target counts differ from global effective-batch denominators')
                    loss=(parts['multimodal_loss_sum']*((1-text_loss_weight)/multimodal_supervised_tokens)
                          if multimodal_supervised_tokens else parts['multimodal_loss_sum']*0)
                    loss=loss+(parts['text_only_loss_sum']*(text_loss_weight/text_only_supervised_tokens)
                               if text_only_supervised_tokens else parts['text_only_loss_sum']*0)
                    loss.backward()
                    mm_value=parts['multimodal_loss_sum'].item();text_value=parts['text_only_loss_sum'].item()
                    multimodal_losses+=mm_value;text_losses+=text_value
                    losses+=mm_value+text_value;actual_tokens+=expected_text+expected_mm
                    sequence_tokens+=parts['sequence_tokens'];objective+=loss.item()
            return {'loss_sum':losses,'supervised_tokens':actual_tokens,
                    'sequence_tokens':sequence_tokens,'microbatch_size':microbatch_size,
                    'text_only_supervised_tokens':text_only_supervised_tokens,
                    'multimodal_supervised_tokens':multimodal_supervised_tokens,
                    'text_only_records':text_only_records,'multimodal_records':len(rows)-text_only_records,
                    'text_only_supervised_fraction':text_only_supervised_tokens/denominator,
                    'objective_value':losses/denominator if text_loss_weight is None else objective,
                    'objective_profile':'joint_supervised_token_ce' if text_loss_weight is None else 'fixed_stream_weights_global_token_denominators',
                    'text_loss_weight':text_loss_weight,
                    'missing_loss_streams':(['multimodal'] if not multimodal_supervised_tokens else [])+(['text_only'] if not text_only_supervised_tokens else []),
                    'missing_stream_policy':'not_applicable' if text_loss_weight is None else 'zero_contribution_no_weight_renormalization',
                    **({'multimodal_ce':multimodal_losses/multimodal_supervised_tokens if multimodal_supervised_tokens else None,
                        'text_only_ce':text_losses/text_only_supervised_tokens if text_only_supervised_tokens else None}
                       if text_loss_weight is not None else {})}
        except torch.cuda.OutOfMemoryError:
            # An optimizer step has not occurred; partial gradients are disposable.
            optimizer.zero_grad(set_to_none=True)
            parts=None
            random.setstate(rng_python);np.random.set_state(rng_numpy);torch.set_rng_state(rng_torch)
            if rng_cuda is not None:torch.cuda.set_rng_state_all(rng_cuda)
            if microbatch_size<=1:raise
            microbatch_size=max(1,microbatch_size//2)
            loss=None
            print(json.dumps({'event':'activation_oom_retry','microbatch_size':microbatch_size,
                              'effective_records':len(rows)}),flush=True)
        # Release traceback-held activations before returning memory to the pool.
        torch.cuda.empty_cache()


def main():
    args=parse_args();validate_args(args);set_seed(args.seed);torch.set_num_threads(8)
    Path('logs/research').mkdir(parents=True,exist_ok=True)
    if args.stage!='A0' and not (args.alignment_checkpoint or args.initial_checkpoint or args.resume):
        raise ValueError('Non-A0 stage requires aligned projector/checkpoint')
    torch.backends.cuda.matmul.allow_tf32=True
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    budget=DiskBudget(out,reserve_gb=args.reserve_gb)
    budget.ensure_space(2*2**30)
    train=load_records(args.manifest)
    val=load_records(args.validation)
    selection=(prepare_group_macro_selection(args,base_records=val)
               if args.selection_metric=='group_macro' else None)
    selection_text=[] if selection is None else list(selection['text_records'])
    if len(selection_text)>args.eval_records:
        selection_text=random.Random(args.seed).sample(selection_text,args.eval_records)
    if args.stage=='SFT' and args.text_manifest:
        text=load_records(args.text_manifest)
        wanted=round(len(train)*args.text_fraction/(1-args.text_fraction))
        if len(text)<wanted:raise ValueError(f'Text replay requires {wanted}, only {len(text)} records available')
        random.Random(args.seed).shuffle(text);train+=text[:wanted]
        if args.text_validation:val+=load_records(args.text_validation)
    # A bounded smoke subset must still cover all task groups, not source order.
    if args.max_records and args.max_records<len(train):
        selected=random.Random(args.seed).sample(range(len(train)),args.max_records)
        train=[train[i] for i in selected]
    if len(val)>args.eval_records:
        val=random.Random(args.seed).sample(val,args.eval_records)
    if not train or not val:raise ValueError('Empty training or validation manifest')
    manifest_hash=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    config=dict(vars(args),runner_schema_version=3,manifest_sha256=manifest_hash,validation_sha256=hashlib.sha256(Path(args.validation).read_bytes()).hexdigest(),
        text_manifest_sha256=hashlib.sha256(Path(args.text_manifest).read_bytes()).hexdigest() if args.text_manifest else None,
        text_validation_sha256=hashlib.sha256(Path(args.text_validation).read_bytes()).hexdigest() if args.text_validation else None,
        selection_manifest_sha256=hashlib.sha256(Path(args.selection_manifest or args.validation).read_bytes()).hexdigest() if selection is not None else None)
    # Resume is an invocation flag, not a distinct experiment identity.
    config['resume']=False
    config_path=out/'resolved_config.json'
    if config_path.exists():
        existing_config=json.loads(config_path.read_text())
        if args.resume:
            args.prompt_schema_version=existing_config.get('prompt_schema_version',1)
            config['prompt_schema_version']=args.prompt_schema_version
            if existing_config.get('manifest_sha256')!=manifest_hash:raise ValueError('Resume manifest changed')
            validate_resume_config(existing_config,config)
            for key in ('initial_checkpoint','alignment_checkpoint'):
                config[key]=existing_config.get(key)
        elif (out/'last'/'state.pt').exists() or (out/'completed.json').exists():
            raise ValueError('Output already contains a checkpoint; use resume or a new output directory')
        elif existing_config!=config:
            raise ValueError('Output contains a different experiment configuration')
    log(out/'metrics.jsonl',{'event':'loading_model','run_id':out.name,**budget.report()})
    model=ResearchModel(args)
    if hasattr(model,'model_loading_report'):
        (out/'model_loading_report.json').write_text(json.dumps(model.model_loading_report,indent=2))
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    (out/'trainable_parameters.json').write_text(json.dumps({n:p.numel() for n,p in trainable},indent=2))
    if not trainable:raise ValueError('No trainable parameters')
    groups={}
    for name,param in trainable:
        lr=(1e-3 if args.stage=='A0' or 'gate_logits' in name else
            1e-4 if (args.recipe.startswith('2') and '.primary_router.' in name) or any(t in name for t in ['visual_experts','router_adapter','router_new','lora_','router_mass']) else 1e-5)
        wd=0. if param.ndim<2 or 'gate_logits' in name or name.endswith('.bias') or 'norm' in name else .01
        groups.setdefault((lr,wd),[]).append(param)
    optimizer_backend=getattr(args,'optimizer_backend','torch_adamw32')
    parameter_groups=[{'params':p,'lr':lr,'weight_decay':wd} for (lr,wd),p in groups.items()]
    if optimizer_backend=='torch_adamw32':
        optimizer=torch.optim.AdamW(parameter_groups)
    else:
        from src.utils.research_optim import make_full_training_optimizer
        optimizer=make_full_training_optimizer(parameter_groups,backend=optimizer_backend)
    log(out/'metrics.jsonl',{'event':'optimizer_profile','backend':optimizer_backend,
        'master_weight_dtype':'float32','optimizer_state_bits':8 if optimizer_backend.endswith('adamw8') else 32,
        'cpu_gradient_offload':optimizer_backend.startswith('cpu_grad_')})
    accum=max(1,math.ceil(args.effective_batch_size/args.batch_size))
    steps_per_epoch=math.ceil(math.ceil(len(train)/args.batch_size)/accum)
    total_steps=min(args.max_steps or steps_per_epoch*args.epochs,steps_per_epoch*args.epochs)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/max(1,int(total_steps*.03)),1)*.5*(1+math.cos(math.pi*min(s,total_steps)/total_steps)))
    from src.utils.research_checkpoint_compat import load_research_payload
    manager=BestLastCheckpointManager(out,disk_budget=budget,mode='max' if selection is not None else 'min')
    start_step=0;seen=0;start=time.monotonic();history_elapsed=0;microbatch_size=args.batch_size
    if args.initial_checkpoint and not args.resume:
        initial=load_research_payload(args.initial_checkpoint,map_location='cpu',mmap=True)
        validate_initial_checkpoint(initial,model,args)
        restored=model.load_state_dict(initial['model'],strict=False)
        if restored.unexpected_keys:raise ValueError(restored.unexpected_keys)
        del initial
    if args.alignment_checkpoint and not args.resume:
        initial=load_research_payload(args.alignment_checkpoint,map_location='cpu',mmap=True)
        validate_initial_checkpoint(initial,model,args)
        model.projector.load_state_dict({k.removeprefix('projector.'):v for k,v in initial['model'].items() if k.startswith('projector.')})
        del initial
    if args.resume and not (out/'last'/'state.pt').exists():
        raise FileNotFoundError('Resume requested but last/state.pt missing')
    if args.resume and (out/'last'/'state.pt').exists():
        saved=load_research_payload(out/'last'/'state.pt',map_location='cpu',mmap=True)
        if saved['manifest_sha256']!=manifest_hash:raise ValueError('Resume manifest changed')
        validate_resume_config(saved['config'],config)
        validate_model_binding(saved,model)
        if saved.get('resumable_optimizer') is False:
            raise ValueError('Completed checkpoint has no optimizer; initialize a new stage explicitly')
        result=model.load_state_dict(saved['model'],strict=False)
        if result.unexpected_keys:raise ValueError(result.unexpected_keys)
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start_step=saved['step'];seen=saved['seen'];history_elapsed=saved.get('elapsed_seconds',0)
        microbatch_size=saved.get('active_microbatch_size',args.batch_size)
        if isinstance(microbatch_size,bool) or not isinstance(microbatch_size,int) or not 1<=microbatch_size<=args.batch_size:
            raise ValueError('Checkpoint active microbatch size is incompatible with configured batch size')
        random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
        del saved
    from src.utils.research_storage import _atomic_json
    _atomic_json(config_path,config)
    from src.utils.research_provenance import write_run_provenance
    provenance=write_run_provenance(ROOT,out,config,reserve_gb=args.reserve_gb,
        model_ref=model.model_ref,vision_ref=model.vision_ref,cache_fingerprint=model.cache.fingerprint)
    log(out/'metrics.jsonl',{'event':'source_snapshot','code_sha256':provenance['code_sha256'],
                           'source_archive':provenance['source_archive']})
    objective_profile={'name':'joint_supervised_token_ce' if args.text_loss_weight is None else 'fixed_stream_weights_global_token_denominators',
        'text_loss_weight':args.text_loss_weight,'multimodal_loss_weight':None if args.text_loss_weight is None else 1-args.text_loss_weight,
        'missing_stream_policy':'not_applicable' if args.text_loss_weight is None else 'zero_contribution_no_weight_renormalization'}
    log(out/'metrics.jsonl',dict(event='loss_objective',**objective_profile))
    log(out/'metrics.jsonl',{'event':'ready','trainable':sum(p.numel() for _,p in trainable),'train_records':len(train),'total_steps':total_steps,'batch_size':args.batch_size})

    def evaluate():
        if selection is not None:
            from src.eval.research_selection import evaluate_internal_group_macro
            last_progress=[0.]
            def report_selection_progress(row):
                public={key:value for key,value in row.items() if key not in ('prediction','score')}
                public['step']=step
                _atomic_json(out/'selection_progress.json',public)
                if time.monotonic()-last_progress[0]>=30 or row['completed_records']==row['total_records']:
                    log(out/'metrics.jsonl',public)
                    last_progress[0]=time.monotonic()
            result=evaluate_internal_group_macro(model,selection,max_new_tokens=args.selection_max_new_tokens,
                progress=report_selection_progress)
            text_metrics=evaluate_text_only_ce(model,selection_text,min(args.batch_size,microbatch_size))
            return result['metrics']['group_macro_score'],result,text_metrics
        model.eval();total=0.;tokens=0
        with torch.no_grad():
            for offset in range(0,len(val),min(args.batch_size,microbatch_size)):
                batch=model.prepare(val[offset:offset+min(args.batch_size,microbatch_size)])
                loss,count,_=model.forward_loss(batch);total+=loss.item();tokens+=count
        model.train();model.extractor.vision_tower.eval()
        return total/max(1,tokens),None,None

    def save(step,metric,selection_result=None):
        # Include frozen projector for independent stage restore, no frozen LLM copy.
        state=trainable_state_dict(model)
        state.update({'projector.'+k:v.detach().cpu() for k,v in model.projector.state_dict().items()})
        anticipated=sum(v.numel()*v.element_size() for v in state.values())*4+128*2**20
        payload={'model':state,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'step':step,'seen':seen,'manifest_sha256':manifest_hash,'config':config,
            'model_ref':model.model_ref,'vision_ref':model.vision_ref,
            'selection_metric':selection_result['protocol']['name'] if selection_result is not None else 'heldout_ce_smoke_proxy_not_official_benchmark',
            'selection_value':metric,
            'selection_protocol':selection_result['protocol'] if selection_result is not None else None,
            'selection_statistics':selection_result['metrics'] if selection_result is not None else None,
            'loss_objective':objective_profile,'active_microbatch_size':microbatch_size,
            'cache_fingerprint':model.cache.fingerprint,
            'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
            'rng_cuda':torch.cuda.get_rng_state_all(),'elapsed_seconds':history_elapsed+time.monotonic()-start}
        report=manager.save_torch(payload,step=step,metric=metric,anticipated_bytes=anticipated,checkpoint_format=args.checkpoint_format,
            metadata={'selection_metric':payload['selection_metric'],'selection_direction':'max' if selection_result is not None else 'min',
                      'official_benchmark_score':False})
        if selection_result is not None:
            write_selection_artifacts(out,selection_result,step,is_best=report['best']['step']==step)

    step=start_step
    try:
        for epoch in range(args.epochs):
            order=list(range(len(train)));random.Random(args.seed+epoch).shuffle(order)
            chunks=[order[i:i+args.batch_size] for i in range(0,len(order),args.batch_size)]
            for group_start in range(0,len(chunks),accum):
                logical_step=epoch*steps_per_epoch+group_start//accum
                if logical_step<start_step:continue
                if step>=total_steps:break
                budget.ensure_space(64*2**20)
                model.train();model.extractor.vision_tower.eval();optimizer.zero_grad(set_to_none=True)
                prepared=[model.prepare([train[i] for i in idx]) for idx in chunks[group_start:group_start+accum]]
                ntokens=sum(len(row[2]) for batch in prepared for row in batch)
                t0=time.monotonic()
                result=train_accumulation_group(model,optimizer,prepared,[p for _,p in trainable],microbatch_size,text_loss_weight=args.text_loss_weight)
                losses=result['loss_sum'];actual_tokens=result['supervised_tokens'];sequence_tokens=result['sequence_tokens']
                microbatch_size=result['microbatch_size']
                if hasattr(optimizer,'clip_grad_norm_'):
                    optimizer.clip_grad_norm_(1.)
                else:
                    torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();torch.cuda.synchronize()
                step+=1;count=sum(len(batch) for batch in prepared);seen+=count
                elapsed=history_elapsed+time.monotonic()-start
                progress={'event':'train','run_id':out.name,'stage':args.stage,'step':step,'total_steps':total_steps,
                    'epoch':epoch,'microbatch_size':microbatch_size,'completed_records':seen,'total_records':min(len(train)*args.epochs,total_steps*args.effective_batch_size),
                    'elapsed_seconds':elapsed,'step_seconds':time.monotonic()-t0,'loss':result['objective_value'],
                    'joint_token_ce':losses/max(1,actual_tokens),
                    **{key:result[key] for key in ('objective_profile','text_loss_weight','missing_stream_policy','missing_loss_streams')},
                    **{key:result[key] for key in ('multimodal_ce','text_only_ce') if key in result},
                    'supervised_tokens':actual_tokens,'sequence_tokens':sequence_tokens,
                    **{key:result[key] for key in ('text_only_supervised_tokens','multimodal_supervised_tokens','text_only_records','multimodal_records','text_only_supervised_fraction')},
                    'cache':dict(model.cache.stats),'peak_vram_gb':torch.cuda.max_memory_allocated()/2**30,'free_disk_gb':budget.report().get('free_gib')}
                log(out/'metrics.jsonl',progress)
                from src.utils.research_storage import _atomic_json
                _atomic_json(Path('logs/research/progress.json'),progress)
                if not math.isfinite(progress['loss']):raise FloatingPointError('Nonfinite loss')
                if step%args.eval_every==0 or step==total_steps:
                    metric,selection_result,text_metrics=evaluate()
                    if selection_result is None:
                        log(out/'metrics.jsonl',{'event':'validation','step':step,'loss':metric})
                    else:
                        log(out/'metrics.jsonl',{'event':'validation','step':step,'group_macro_score':metric,
                            'selection_metric':selection_result['protocol']['name'],'official_benchmark_score':False,
                            'selection_manifest_sha256':selection_result['protocol']['selection_manifest_sha256'],
                            'groups':selection_result['metrics']['groups'],'records':selection_result['metrics']['records']})
                        if text_metrics is not None:log(out/'metrics.jsonl',dict(event='text_validation',step=step,**text_metrics))
                    if model.wrappers:
                        log(out/'routing.jsonl',{'step':step,'layers':[w.diagnostics() for w in model.wrappers]},console=False)
                    save(step,metric,selection_result)
            if step>=total_steps:break
        if step!=total_steps:raise RuntimeError(f'Incomplete: {step}/{total_steps} steps')
        (out/'completed.json').write_text(json.dumps({'steps':step,'records':seen,'elapsed_seconds':history_elapsed+time.monotonic()-start}))
    except Exception as exc:
        log(out/'metrics.jsonl',{'event':'failed','step':step,'error_type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        if callable(getattr(optimizer,'close',None)):optimizer.close()

if __name__=='__main__':main()
