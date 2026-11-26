"""Opt-in text-only retention producer; importing it loads no model or Torch.

Generation consumes a separate immutable public derivative, never the original
gold-bearing benchmark records. Scoring remains a separate CPU-only command.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[2]
OWNER='research_primary_text_native_b1_v1'
PUBLIC_OWNER='research_primary_text_public_inputs_v1'
PUBLIC_FIELDS=frozenset(('record_id','task','source','input_text_sha256','prompt','subject','max_new_tokens'))
TASKS=('gsm8k','mmlu','ifeval')
STAGES=('baseline_before_vision','A0','A1','SFT')
CONTEXT=4096
DATE='2026-11-22'
CENSUS_SHA256='ce3e23738fddfa78afc44dfaef10abcbd03538df2104b3b658d14a6c4b67acf6'
PUBLIC_RECORDS_SHA256='31f72c1823644f449cd3b8b9f6cb2d36c0b9e662ee26d3e18971f0fbf461cf8c'
FULL_INIT_CONTROL='full_fp32_master_bf16_autocast'


def _pure(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


metrics=_pure('text_producer_scoring_contract','src/eval/research_text_metrics.py')


def read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>32*2**20:
        raise ValueError('Expected bounded regular local JSON metadata')
    return json.loads(path.read_text())
def sha(path):
    # Native snapshot shards can be several GiB. Hash by bounded chunks rather
    # than materialising a second checkpoint-sized CPU allocation.
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk:=handle.read(8*2**20):value.update(chunk)
    return value.hexdigest()
def digest(value):return metrics.object_sha(value)
def stat(path):
    value=Path(path).stat()
    return {key:getattr(value,key) for key in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}


def atomic(path,value):
    path=Path(path);raw=(metrics.canonical(value)+'\n').encode()
    stage=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with stage.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
        os.replace(stage,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:stage.unlink(missing_ok=True)


def immutable(path,raw):
    path=Path(path)
    if path.exists():
        if path.is_symlink() or path.read_bytes()!=raw:raise ValueError('Immutable text producer artifact differs')
        return
    stage=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with stage.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
        try:os.link(stage,path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes()!=raw:raise ValueError('Competing text producer artifact differs')
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:stage.unlink(missing_ok=True)


def public_record(row):
    if set(row)!=PUBLIC_FIELDS or row['task'] not in TASKS or not isinstance(row['prompt'],str) or not row['prompt'].strip():
        raise ValueError('Expected the exact gold-free text input allowlist')
    if '[RESEARCH_IMAGE]' in row['prompt']:raise ValueError('Text input contains reserved image marker')
    if not isinstance(row['record_id'],str) or not metrics.hex64(row['record_id']):raise ValueError('Released record ID missing')
    if type(row['max_new_tokens']) is not int:raise ValueError('Text task budget must be an integer')
    return json.loads(metrics.canonical(row))


def public_projection(record):
    return public_record({'record_id':record['record_id'],'task':record['task'],'source':record['source'],
        'input_text_sha256':metrics.sha(record['input_text']),'prompt':record['prompt'],
        'subject':record['subject'],'max_new_tokens':record['max_new_tokens']})


def prepare_public_bundle(parent,output,*,reserve_gb=120):
    """Explicit CPU preparation only; the later model process does not call it."""
    if not math.isfinite(reserve_gb) or reserve_gb<120:raise ValueError('Preserve120GiB reserve')
    suite=metrics.load_suite(parent)
    rows=[public_projection(r) for r in suite['records']]
    files={'public_records.jsonl':b''.join((metrics.canonical(r)+'\n').encode() for r in rows),
        'tokenizer_census.json':(Path(parent)/'tokenizer_census.json').read_bytes()}
    if Path(output).exists():
        # A runtime revision must not re-tag an existing derivative as newly
        # prepared. Verify its explicit historical preparer and exact current
        # projection, then return the original immutable protocol unchanged.
        existing=public_context(output,'smallthinker4')['protocol']
        if (existing['parent_manifest_sha256']!=suite['manifest_sha256'] or
                existing['parent_generation_policy']!=suite['protocol']['evaluation']['generation'] or
                any((Path(output)/name).read_bytes()!=raw for name,raw in files.items())):
            raise ValueError('Existing public derivative differs from its pinned gold-source projection')
        return existing
    # Preparation and runtime share this new module. Keep the exact preparation
    # source as an explicit immutable artifact, so later runtime revisions do
    # not erase the evidence that produced an already immutable public bundle.
    source=Path(__file__).read_bytes();source_sha=metrics.sha(source)
    archived=ROOT/'research_results/text_public_preparation_sources'/source_sha/'research_text_generation.py'
    archived.parent.mkdir(parents=True,exist_ok=True);immutable(archived,source)
    protocol={'schema_version':1,'owner':PUBLIC_OWNER,'parent_protocol_sha256':suite['protocol_sha256'],
        'parent_manifest_sha256':suite['manifest_sha256'],'records':1680,'counts':metrics.TASK_COUNTS,
        'public_fields':sorted(PUBLIC_FIELDS),'gold_fields_in_model_process':False,
        'parent_generation_policy':suite['protocol']['evaluation']['generation'],
        'context':CONTEXT,'fixed_date':DATE,'MMLU':'exact conditional LL of A/B/C/D; summed, no length normalization',
        'preparation_source_sha256':{'src/eval/research_text_generation.py':source_sha},
        'preparation_archive':{'path':str(archived),'sha256':source_sha},
        'preparation_dependencies_sha256':{p:sha(ROOT/p) for p in ('src/eval/research_text_data.py','src/eval/research_text_metrics.py')}}
    helper=_pure('text_public_bundle_writer','src/eval/research_text_data.py')
    return helper.write_immutable_bundle(output,files,protocol,reserve_bytes=math.ceil(reserve_gb*2**30))


def public_context(bundle,model_id,*,tasks=TASKS):
    """Read only public derivative/census; no original suite/reference access."""
    bundle=Path(bundle).resolve(strict=True);protocol=read(bundle/'protocol.json')
    if (protocol.get('schema_version')!=1 or protocol.get('owner')!=PUBLIC_OWNER or
            protocol.get('parent_protocol_sha256')!=metrics.SUITE_PROTOCOL_SHA256 or
            protocol.get('records')!=1680 or protocol.get('public_fields')!=sorted(PUBLIC_FIELDS) or
            protocol.get('fixed_date')!=DATE or protocol.get('context')!=CONTEXT):
        raise ValueError('Public text derivative differs from fixed primary suite')
    archive=protocol.get('preparation_archive',{})
    target=Path(archive.get('path',''))
    if (not isinstance(archive.get('path'),str) or target.is_symlink() or not target.is_file() or
            not target.resolve().is_relative_to(ROOT/'research_results/text_public_preparation_sources') or
            target.name!='research_text_generation.py' or target.stat().st_size>2*2**20 or
            not metrics.hex64(archive.get('sha256')) or sha(target)!=archive['sha256'] or
            protocol.get('preparation_source_sha256',{}).get('src/eval/research_text_generation.py')!=archive['sha256']):
        raise ValueError('Explicit archived public preparation source differs')
    if set(protocol.get('preparation_dependencies_sha256',{}))!={'src/eval/research_text_data.py','src/eval/research_text_metrics.py'}:
        raise ValueError('Public preparation dependency map is incomplete')
    for name,expected in protocol.get('preparation_dependencies_sha256',{}).items():
        if (Path(name).is_absolute() or not (ROOT/name).resolve().is_relative_to(ROOT) or sha(ROOT/name)!=expected):
            raise ValueError('Frozen public preparation dependency differs')
    if set(p.name for p in bundle.iterdir())!=set(protocol['files_sha256'])|{'protocol.json'}:
        raise ValueError('Unexpected public derivative members')
    if (set(protocol['files_sha256'])!={'public_records.jsonl','tokenizer_census.json'} or
            protocol['files_sha256']['public_records.jsonl']!=PUBLIC_RECORDS_SHA256):
        raise ValueError('Public derivative differs from the frozen released-row projection')
    for name,expected in protocol['files_sha256'].items():
        if Path(name).name!=name or (bundle/name).is_symlink() or sha(bundle/name)!=expected:
            raise ValueError('Public input/census bytes changed')
    if sha(bundle/'tokenizer_census.json')!=CENSUS_SHA256:
        raise ValueError('Public derivative must preserve the original trusted tokenizer census bytes')
    all_rows=[public_record(r) for r in metrics.read_jsonl(bundle/'public_records.jsonl')]
    from collections import Counter
    if Counter(r['task'] for r in all_rows)!=Counter(metrics.TASK_COUNTS) or len({r['record_id'] for r in all_rows})!=1680:
        raise ValueError('Incomplete public1680-row derivative')
    if not tasks or len(set(tasks))!=len(tasks) or set(tasks)-set(TASKS):raise ValueError('Explicit unique task scope required')
    census=read(bundle/'tokenizer_census.json')
    selected=[m for m in census['models'] if m['model_id']==model_id]
    if len(selected)!=1 or model_id not in metrics.TOKEN_IDS:raise ValueError('Unknown pinned backbone tokenizer')
    model=selected[0]
    if [r['record_id'] for r in model['records']]!=[r['record_id'] for r in all_rows]:
        raise ValueError('Public row/census order differs')
    by_id={r['record_id']:r for r in model['records']}
    for row in all_rows:
        if (row['task']!=by_id[row['record_id']]['task'] or
                row['max_new_tokens']!={'gsm8k':1024,'mmlu':0,'ifeval':2048}[row['task']]):
            raise ValueError('Public task identity or preregistered generation budget differs')
    rows=[r for r in all_rows if r['task'] in tasks]
    if any(by_id[r['record_id']]['required_context_tokens']>CONTEXT for r in rows):raise ValueError('Fixed text context exceeded')
    return {'bundle':str(bundle),'protocol':protocol,'protocol_sha256':sha(bundle/'protocol.json'),
        'public_file_sha256':sha(bundle/'public_records.jsonl'),'census_file_sha256':sha(bundle/'tokenizer_census.json'),
        'model':model,'records':rows,'census':by_id,'tasks':list(tasks)}


class FixedClockTokenizer:
    """Forward every tokenizer operation; fix only native template clock inputs."""
    def __init__(self,tokenizer):self._tokenizer=tokenizer
    def __getattr__(self,name):return getattr(self._tokenizer,name)
    def apply_chat_template(self,*args,**kwargs):
        from datetime import datetime,timezone
        clock=datetime(2026,11,22,tzinfo=timezone.utc)
        kwargs.update(date_string=DATE,current_date=DATE,strftime_now=lambda fmt:clock.strftime(fmt))
        return self._tokenizer.apply_chat_template(*args,**kwargs)


@contextmanager
def known_giga_configuration(checkpoint):
    """Process-local recorded int1→float1 dispatch, no frozen-source edits."""
    run=Path(checkpoint).resolve().parents[2]
    config=read(run/'resolved_config.json')
    if config.get('model')!='ai-sage/GigaChat3.1-10B-A1.8B-bf16':
        yield None;return
    import scripts.run_research_stage as runner
    from unittest.mock import patch
    from src.model.research_config import load_research_native_config
    assets=read(ROOT/'datasets/research/text-eval-inputs/assets.json')
    item=next(a for a in assets['files'] if a['id']=='giga31' and a['filename']=='config.json')
    if sha(item['path'])!=item['sha256']:raise ValueError('Giga native config source changed')
    native,receipt=load_research_native_config(item['path']);target=Path(item['path']).parent.resolve()
    original_config=runner.AutoConfig;original_tokenizer=runner.AutoTokenizer;original_model=runner.AutoModelForCausalLM
    def selected(path):return Path(path).resolve()==target
    class Config:
        @staticmethod
        def from_pretrained(path,*args,**kwargs):
            return native if selected(path) else original_config.from_pretrained(path,*args,**kwargs)
    class Tokenizer:
        @staticmethod
        def from_pretrained(path,*args,**kwargs):
            if selected(path):kwargs.update(config=native,local_files_only=True)
            return original_tokenizer.from_pretrained(path,*args,**kwargs)
    class Model:
        @staticmethod
        def from_pretrained(path,*args,**kwargs):
            if selected(path):kwargs.update(config=native,local_files_only=True)
            return original_model.from_pretrained(path,*args,**kwargs)
    with patch.object(runner,'AutoConfig',Config),patch.object(runner,'AutoTokenizer',Tokenizer),patch.object(runner,'AutoModelForCausalLM',Model):
        yield receipt


def model_record(row):
    row=public_record(row)
    return {'question':row['prompt'],'image_paths':[],'image_hashes':[]}


def check_prepared(row,prepared,census):
    if (prepared.get('prefix_token_ids')!=[] or prepared.get('visual_tokens')!=0 or
            prepared.get('prompt_sha256')!=census['prompt_sha256'] or
            prepared.get('prefill_tokens')!=census['prompt_tokens'] or
            digest(prepared.get('suffix_token_ids'))!=census['prompt_token_ids_sha256']):
        raise ValueError('Actual native text prompt/token boundary differs from fixed census')
    reserve=1 if row['task']=='mmlu' else row['max_new_tokens']
    if prepared['prefill_tokens']+reserve>CONTEXT:raise ValueError('No text prompt truncation or budget fallback is allowed')


def baseline_plan(context,precision):
    if precision not in ('bf16','nf4',FULL_INIT_CONTROL):raise ValueError('Select a named baseline/initialization control explicitly')
    model=context['model'];inventory_path=ROOT/'research_results/model_inventory.json'
    inventory=read(inventory_path)
    matches=[r for r in inventory if r['repo']==model['repo'] and r['sha']==model['revision']]
    if len(matches)!=1:raise ValueError('Baseline inventory does not match pinned suite backbone')
    receipt=read(ROOT/'datasets/research/text-eval-inputs/assets.json')
    asset=next(a for a in receipt['files'] if a['id']==model['model_id'] and a['filename']=='config.json')
    snapshot=Path(asset['path']).parent
    if sha(asset['path'])!=model['tokenizer_assets_sha256']['config.json']:
        raise ValueError('Pinned baseline configuration source differs')
    # Use the native root index, excluding alternate exports such as GPT's
    # original/model.safetensors. Never request duplicate complete formats.
    index=snapshot/'model.safetensors.index.json'
    required_paths=set(json.loads(index.read_bytes())['weight_map'].values()) if index.is_file() else None
    if required_paths is not None and any(Path(name).is_absolute() or '..' in Path(name).parts for name in required_paths):
        raise ValueError('Native model weight index escaped its snapshot')
    required=[f for f in matches[0]['files'] if f['path'].endswith('.safetensors') and
        (f['path'] in required_paths if required_paths is not None else len(Path(f['path']).parts)==1)]
    if required_paths is not None and {f['path'] for f in required}!=required_paths:
        raise ValueError('Native weight index differs from pinned inventory')
    if not required:raise ValueError('Pinned baseline weight file inventory is empty')
    missing=[f['path'] for f in required if not (snapshot/f['path']).is_file()]
    changed=[f['path'] for f in required if (snapshot/f['path']).is_file() and (snapshot/f['path']).stat().st_size!=f['bytes']]
    if changed:raise ValueError('Local baseline weight sizes differ from pinned inventory')
    return {'kind':'pinned_baseline','model_id':model['model_id'],'repo':model['repo'],'revision':model['revision'],
        'precision':precision,'snapshot':str(snapshot),'weight_inventory_metadata_sha256':digest(matches[0]),
        'inventory_file_sha256':sha(inventory_path),'required_weights':required,'missing_weights':missing,
        'weight_downloads_allowed':False,'config_sha256':asset['sha256'],
        'numerical_profile':{'weights':'fp32' if precision==FULL_INIT_CONTROL else precision,
            'compute':'native bf16 autocast','initialization_control':precision},
        'GPU_memory_fit_proven':False,
        'admission_blockers':['GPT full-backend P10 remains closed; no FP32-fit assumption']
            if model['model_id']=='gptoss20' and precision==FULL_INIT_CONTROL else []}


def snapshot_identity(snapshot):
    snapshot=Path(snapshot).resolve(strict=True);result={}
    for path in sorted(snapshot.rglob('*')):
        if not path.is_file() or path.suffix not in ('.json','.safetensors','.model','.jinja','.txt'):continue
        before=stat(path);value=sha(path)
        if stat(path)!=before:raise ValueError('Pinned source snapshot changed while hashing')
        target=path.resolve()
        if path.suffix=='.safetensors' and target.parent.name=='blobs' and metrics.hex64(target.name) and value!=target.name:
            raise ValueError('HF cached weight bytes differ from their pinned content-addressed blob')
        result[str(path.relative_to(snapshot))]={'path':str(path.resolve()),'stat':before,'sha256':value}
    if not result:raise ValueError('Pinned source snapshot is empty')
    return result


def dtype_report(model):
    def describe(value):return {'dtype':str(value.dtype),'shape':list(value.shape),'numel':value.numel()}
    parameters={name:dict(describe(value),requires_grad=value.requires_grad) for name,value in model.named_parameters()}
    buffers={name:describe(value) for name,value in model.named_buffers()}
    router=lambda name:'.router.' in '.'+name or '.primary_router.' in '.'+name or '.mlp.gate.' in '.'+name
    return {'parameters':parameters,'buffers':buffers,
        'router_parameters':{k:v for k,v in parameters.items() if router(k)},
        'norm_parameters':{k:v for k,v in parameters.items() if 'norm' in k.lower()}}


def full_initialization_control(language):
    """Mirror current frozen full-stage casts, including floating buffers."""
    language.requires_grad_(False)  # The frozen runner first freezes its loaded base.
    before=dtype_report(language)
    # Frozen ResearchModel.__init__: stage!=A0 and not quantized, before E+K
    # installation. Module.float is intentional: the source casts buffers too.
    language.float().requires_grad_(True)
    for parameter in language.parameters():
        if parameter.requires_grad and parameter.is_floating_point():parameter.data=parameter.data.float()
    return {'profile':FULL_INIT_CONTROL,'source_policy_sha256':sha(ROOT/'scripts/run_research_stage.py'),
        'operations':['native BF16 source initialization','llm.float().requires_grad_(True)',
            'trainable floating parameter.data.float()','freeze for inference; no optimizer/visual expansion'],
        'before':before,'after_promotion':dtype_report(language),'Module_float_includes_floating_buffers':True,
        'GPU_memory_fit_proven':False}


class NativeTextRuntime:
    def load_baseline(self,plan):
        if plan.get('admission_blockers'):raise RuntimeError('; '.join(plan['admission_blockers']))
        if plan['missing_weights']:
            raise FileNotFoundError('Pinned baseline weights are not local; producer never downloads them')
        assets=snapshot_identity(plan['snapshot'])
        import torch
        from torch import nn
        from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
        from src.model.research_config import load_research_native_config
        from src.model.research_quantization import quantize_packed_experts,set_packed_lora_trainable
        from scripts.run_research_stage import _without_transformers_allocator_warmup
        from src.eval.research_inference import freeze_for_generation
        cfg,config_receipt=load_research_native_config(Path(plan['snapshot'])/'config.json')
        tokenizer=AutoTokenizer.from_pretrained(plan['snapshot'],config=cfg,local_files_only=True,trust_remote_code=False)
        tokenizer.pad_token=tokenizer.eos_token
        quantized=plan['precision']=='nf4'
        kwargs={'config':cfg,'dtype':torch.bfloat16,'attn_implementation':'sdpa',
            'trust_remote_code':False,'local_files_only':True}
        if quantized:
            kwargs.update(device_map={'':0},quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,llm_int8_skip_modules=['primary_router','lm_head']+
                (['gate','router'] if cfg.model_type!='smallthinker' else [])))
        report={}
        with _without_transformers_allocator_warmup():
            if cfg.model_type=='gpt_oss':
                from src.model.research_loading import load_gpt_oss_for_research
                language,report=load_gpt_oss_for_research(plan['snapshot'],revision=plan['revision'],
                    stage='A0',recipe='4q' if quantized else '4f',device='cuda:0')
            else:
                language,loading_info=AutoModelForCausalLM.from_pretrained(plan['snapshot'],output_loading_info=True,**kwargs)
                if any(loading_info.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
                    raise ValueError('Pinned native baseline checkpoint has missing/unexpected/mismatched weights')
        if not quantized:language.to('cuda')
        initialization=full_initialization_control(language) if plan['precision']==FULL_INIT_CONTROL else None
        language.requires_grad_(False)
        packed=None
        if quantized and cfg.model_type in ('lfm2_moe','deepseek_v3'):
            packed=quantize_packed_experts(language,rank=0);set_packed_lora_trainable(language,False)
        class Baseline(nn.Module):
            def __init__(self):
                super().__init__();self.llm=language;self.args=SimpleNamespace(max_length=CONTEXT)
                self.tokenizer=FixedClockTokenizer(tokenizer)
            @property
            def base(self):return self.llm
        model=freeze_for_generation(Baseline())
        weights={name:info for name,info in assets.items() if name.endswith('.safetensors')}
        identity={'format':'pinned_baseline','repo':plan['repo'],'revision':plan['revision'],
            'weight_inventory_sha256':digest({'repo':plan['repo'],'revision':plan['revision'],'files':weights})}
        self.loading_diagnostics={key:report[key] for key in ('memory_before','memory_after') if key in report}
        stable_report={key:value for key,value in report.items() if key not in ('memory_before','memory_after')}
        provenance={'stage':'baseline_before_vision','model_ref':{'repo':plan['repo'],'revision':plan['revision']},
            'config_receipt':config_receipt,'snapshot_assets':assets,'checkpoint_identity':identity,
            'precision':plan['precision'],'quantization_loading_report':stable_report,'packed_quantization_report':packed,
            'initialization_control':initialization,
            'baseline_policy':'native unwrapped base with explicit full-stage dtype control' if initialization else
                'same frozen native A0 language-base initialization, no projector/vision/LoRA/expert expansion'}
        return model,provenance

    def load_learned(self,checkpoint,cache_dir,reserve_gb):
        from src.eval.research_inference import load_research_checkpoint
        with known_giga_configuration(checkpoint) as compatibility:
            model,provenance=load_research_checkpoint(checkpoint,cache_dir=cache_dir,reserve_gb=reserve_gb)
        provenance={**provenance,'producer_configuration_compatibility':compatibility}
        if int(getattr(model.args,'max_length',0))<CONTEXT:raise ValueError('Learned checkpoint context cannot implement common4096 protocol')
        model.tokenizer=FixedClockTokenizer(model.tokenizer)
        return model,provenance

    def generate(self,model,row,census):
        from unittest.mock import patch
        from src.eval import research_inference as native
        actual_record=model_record(row);original=native.build_generation_inputs;captured=[]
        def observe(actual_model,record,**kwargs):
            if actual_model is not model or record!=actual_record or kwargs.get('max_new_tokens')!=row['max_new_tokens']:
                raise ValueError('Native text generator changed public input/budget')
            prepared=original(actual_model,record,**kwargs);check_prepared(row,prepared,census)
            captured.append(prepared);return prepared
        with patch.object(native,'build_generation_inputs',observe):
            generated=native.generate_prediction(model,actual_record,max_new_tokens=row['max_new_tokens'])
        if len(captured)!=1:raise ValueError('Native text builder must run exactly once')
        # Native visual evaluation strips text; retain the raw decoding here for
        # official IFEval formatting rules, without changing generated tokens.
        raw=model.tokenizer.decode(generated['generated_token_ids'],skip_special_tokens=True)
        return {'response':raw,'generated_token_ids':generated['generated_token_ids'],
            'generated_tokens':generated['generated_tokens'],'stop_policy':'native EOS/pad, greedy B1',
            'prompt_sha256':captured[0]['prompt_sha256'],
            'prompt_token_ids_sha256':digest(captured[0]['suffix_token_ids']),
            'prompt_tokens':captured[0]['prefill_tokens']}

    def candidate_likelihoods(self,model,row,census,model_id):
        import torch
        from src.eval import research_inference as native
        from src.model.research_moe import set_modality_mask
        with torch.no_grad():
            prepared=native.build_generation_inputs(model,model_record(row),max_new_tokens=1)
            check_prepared(row,prepared,census)
            prefix=prepared['suffix_token_ids'];candidate_ids={}
            for letter in metrics.CHOICES:
                full=model.tokenizer.encode(prepared['prompt']+letter,add_special_tokens=False)
                if full[:len(prefix)]!=prefix or full[len(prefix):]!=[metrics.TOKEN_IDS[model_id][letter]]:
                    raise ValueError('Actual native candidate continuation/prefix differs; no relabeling fallback')
                candidate_ids[letter]=full[len(prefix):]
            set_modality_mask(model.base,prepared['modality_mask'],prepared['attention_mask'])
            kwargs={'inputs_embeds':prepared['inputs_embeds'],'attention_mask':prepared['attention_mask'],
                'use_cache':False,'return_dict':True}
            if 'logits_to_keep' in inspect.signature(model.base.forward).parameters:kwargs['logits_to_keep']=1
            with native._autocast(prepared['inputs_embeds'].device):output=model.base(**kwargs)
            logits=output.logits[0,-1,:]
            if not torch.isfinite(logits).all():raise ValueError('Native MMLU logits must be finite')
            logprobs=torch.log_softmax(logits.float(),dim=-1)
            supplied={c:float(logprobs[ids[0]].cpu()) for c,ids in candidate_ids.items()}
        return {'candidate_log_likelihoods':supplied,'candidate_token_ids':candidate_ids,
            'candidate_prefix_token_ids_sha256':dict.fromkeys(metrics.CHOICES,digest(prefix)),
            'length_normalization':False,'log_probability_reduction_dtype':'float32',
            'prompt_sha256':prepared['prompt_sha256'],'prompt_token_ids_sha256':digest(prefix),
            'prompt_tokens':prepared['prefill_tokens']}


def learned_plan(run,role,context):
    run=Path(run).resolve(strict=True)
    if role not in ('best','last'):raise ValueError('Retained best/last checkpoint required')
    config=read(run/'resolved_config.json');completed=read(run/'completed.json');manager=read(run/'.checkpoint_manager.json')
    if config.get('stage') not in ('A0','A1','SFT') or type(completed.get('steps')) is not int or completed['steps']<=0:
        raise ValueError('Text retention requires a completed actual A0/A1/SFT checkpoint')
    declared=Path(config.get('output',''));declared=(declared if declared.is_absolute() else ROOT/declared).resolve()
    if declared!=run or completed.get('status') in ('failed','incomplete','interrupted'):
        raise ValueError('Checkpoint output/completion identity differs')
    if config.get('model')!=context['model']['repo']:raise ValueError('Checkpoint backbone differs from fixed tokenizer suite')
    pointer=run/role
    if not pointer.is_symlink():raise ValueError('Managed checkpoint role symlink required')
    target=pointer.resolve(strict=True)
    if target.parent!=run/'.checkpoint_objects':raise ValueError('Checkpoint role escaped owned object store')
    marker=read(target/'.research_checkpoint.json')
    if (not manager.get('owner') or marker.get('owner')!=manager.get('owner') or
            type(marker.get('step')) is not int or not 0<marker['step']<=completed['steps'] or
            (role=='last' and marker['step']!=completed['steps'])):
        raise ValueError('Managed checkpoint owner/step differs')
    # Raw A0/A1 checkpoints predate optimizer compaction and intentionally lack
    # its training_completed marker. Establish completion from the immutable
    # run receipt plus the same owner's actual terminal last snapshot instead.
    terminal_pointer=run/'last';terminal_target=terminal_pointer.resolve(strict=True)
    terminal_marker=read(terminal_target/'.research_checkpoint.json')
    if (not terminal_pointer.is_symlink() or terminal_target.parent!=run/'.checkpoint_objects' or
            terminal_marker.get('owner')!=manager['owner'] or terminal_marker.get('step')!=completed['steps']):
        raise ValueError('Completed run has no matching terminal last checkpoint')
    checkpoint=target/'state.pt'
    if checkpoint.is_symlink() or not checkpoint.is_file():raise ValueError('Local retained checkpoint missing')
    return {'kind':'learned','run':str(run),'checkpoint':str(checkpoint),'role':role,'pointer':os.readlink(pointer),
        'owner':manager['owner'],'step':marker['step'],'stage':config['stage'],'recipe':config['recipe'],
        'numerical_profile':{'weights':'nf4' if config['recipe'].startswith('1') or config['recipe']=='4q' else
            ('bf16' if config['stage']=='A0' else 'fp32'),'compute':'native bf16 autocast; preserved checkpoint parameter dtypes'},
        'metadata':{name:{'sha256':sha(run/name),'stat':stat(run/name)} for name in
            ('resolved_config.json','completed.json','.checkpoint_manager.json')},
        'marker':{'path':str(target/'.research_checkpoint.json'),'sha256':sha(target/'.research_checkpoint.json'),
            'stat':stat(target/'.research_checkpoint.json')},'checkpoint_stat':stat(checkpoint),
        'terminal_last':{'pointer':os.readlink(terminal_pointer),'object':str(terminal_target),
            'marker_sha256':sha(terminal_target/'.research_checkpoint.json'),
            'marker_stat':stat(terminal_target/'.research_checkpoint.json')}}


def source_files():
    names=('src/eval/research_text_generation.py','scripts/eval_research_text_checkpoint.py',
        'src/eval/research_text_metrics.py','src/eval/research_inference.py','scripts/run_research_stage.py',
        'src/model/research_config.py','src/model/research_loading.py','src/model/research_quantization.py',
        'src/model/research_moe.py','src/model/smallthinker.py','src/model/research_vision.py',
        'src/model/gigachat_vl.py','src/model/visual_expert_gates.py','src/model/text_moe_backbones.py',
        'src/utils/research_storage.py','src/utils/research_checkpoint_compat.py','src/utils/research_checkpoint_bundle.py',
        'src/utils/research_checkpoint_io.py','src/dataset/vision_cache.py','scripts/extract_vision_features.py',
        'tests/test_research_text_generation.py')
    return {name:sha(ROOT/name) for name in names}


def environment():
    versions={}
    for package in ('torch','transformers','peft','bitsandbytes','tokenizers','huggingface-hub'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    return {'python':sys.version,'packages':versions,'fixed_clock':DATE,
        'environment':{name:os.environ.get(name) for name in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG',
            'OMP_NUM_THREADS','MKL_NUM_THREADS','TOKENIZERS_PARALLELISM','PYTORCH_CUDA_ALLOC_CONF')}}


def static_context(context,plan):
    return {'schema_version':1,'owner':OWNER,'public_protocol_sha256':context['protocol_sha256'],
        'public_file_sha256':context['public_file_sha256'],'census_file_sha256':context['census_file_sha256'],
        'record_ids_sha256':digest([r['record_id'] for r in context['records']]),'tasks':context['tasks'],
        'model_id':context['model']['model_id'],'plan':plan,'sources':source_files(),'environment':environment(),
        'context':CONTEXT,'fixed_date':DATE,'generation':context['protocol']['parent_generation_policy'],
        'MMLU':'native all-text forward; FP32 log_softmax of last prompt-position logits; exact one-token A/B/C/D',
        'gold_policy':'model process reads public derivative/census only; scoring is a separate command'}


def assert_static(context,static):
    root=Path(context['bundle'])
    if (sha(root/'protocol.json')!=static['public_protocol_sha256'] or
            sha(root/'public_records.jsonl')!=static['public_file_sha256'] or
            sha(root/'tokenizer_census.json')!=static['census_file_sha256']):
        raise ValueError('Public input/census changed during text generation')
    if environment()!=static['environment'] or source_files()!=static['sources']:
        raise ValueError('Text producer source/runtime environment changed')
    plan=static['plan']
    if plan['kind']=='learned':
        run=Path(plan['run']);pointer=run/plan['role']
        if not pointer.is_symlink() or os.readlink(pointer)!=plan['pointer'] or str(pointer.resolve()/'state.pt')!=plan['checkpoint']:
            raise ValueError('Retained model role changed during generation')
        if stat(plan['checkpoint'])!=plan['checkpoint_stat']:raise ValueError('Managed checkpoint bytes/stat changed')
        for name,value in plan['metadata'].items():
            if stat(run/name)!=value['stat'] or sha(run/name)!=value['sha256']:raise ValueError('Managed model metadata changed')
        marker=plan['marker']
        if stat(marker['path'])!=marker['stat'] or sha(marker['path'])!=marker['sha256']:
            raise ValueError('Managed checkpoint ownership marker changed')
        terminal=plan['terminal_last'];last=run/'last';terminal_marker=Path(terminal['object'])/'.research_checkpoint.json'
        if (not last.is_symlink() or os.readlink(last)!=terminal['pointer'] or str(last.resolve())!=terminal['object'] or
                stat(terminal_marker)!=terminal['marker_stat'] or sha(terminal_marker)!=terminal['marker_sha256']):
            raise ValueError('Completed-run terminal checkpoint binding changed')


def common_response(row,census,producer):
    return {'record_id':row['record_id'],'task':row['task'],'source':row['source'],
        'input_text_sha256':row['input_text_sha256'],'response_contract_sha256':digest(producer),
        'prompt_sha256':census['prompt_sha256'],'prompt_token_ids_sha256':census['prompt_token_ids_sha256'],
        'prompt_tokens':census['prompt_tokens']}


def validate_generated(row,census,generated,producer):
    if any(generated.get(k)!=census[k] for k in ('prompt_sha256','prompt_token_ids_sha256','prompt_tokens')):
        raise ValueError('Actual generated response prompt boundary differs from trusted census')
    if set(generated)&{'reference','gold','answer','answers','kwargs','instruction_id_list'}:
        raise ValueError('Gold fields appeared in model output envelope')
    if set(generated)&{'record_id','task','source','input_text_sha256','response_contract_sha256'}:
        raise ValueError('Native output may not override source/record/producer ownership')
    if row['task']=='mmlu':
        # This validates only supplied likelihoods/boundaries and chooses no
        # label: the real benchmark answer remains in the separate scorer.
        ll=generated.get('candidate_log_likelihoods')
        if (not isinstance(ll,dict) or set(ll)!=set(metrics.CHOICES) or
                any(type(v) not in (float,int) or not math.isfinite(v) or v>0 for v in ll.values()) or
                generated.get('candidate_token_ids')!={c:[metrics.TOKEN_IDS[producer['model_id']][c]] for c in metrics.CHOICES} or
                generated.get('candidate_prefix_token_ids_sha256')!=dict.fromkeys(metrics.CHOICES,census['prompt_token_ids_sha256']) or
                generated.get('length_normalization') is not False):raise ValueError('Native MMLU continuation likelihood contract differs')
    else:
        ids=generated.get('generated_token_ids')
        if (not isinstance(generated.get('response'),str) or not isinstance(ids,list) or
                any(type(v) is not int or v<0 for v in ids) or generated.get('generated_tokens')!=len(ids) or
                len(ids)>row['max_new_tokens']):raise ValueError('Native raw generation token/budget contract differs')
    return {**common_response(row,census,producer),**generated}


def read_committed(output,context,producer,*,resume=False):
    directory=Path(output)/'records';directory.mkdir(exist_ok=True)
    if directory.is_symlink():raise ValueError('Text journal directory may not be a symlink')
    expected={f'{i:06d}.json' for i in range(len(context['records']))}
    for path in directory.iterdir():
        match=re.fullmatch(r'\.(\d{6})\.json\.([0-9a-f]{32})\.tmp',path.name)
        if match and not path.is_symlink() and path.is_file():
            if not resume or int(match[1])>=len(context['records']) or path.stat().st_size>32*2**20:
                raise ValueError('Uncommitted text staging requires bounded owned explicit resume')
            # A SIGKILL can leave an incomplete atomic-write staging file. It
            # is never a response: preserve raw evidence before removing only
            # that narrowly named file from the durable journal directory.
            before=stat(path);raw=path.read_bytes()
            if stat(path)!=before:raise ValueError('Uncommitted text staging changed during recovery')
            recovery=Path(output)/'recovery';recovery.mkdir(exist_ok=True)
            if recovery.is_symlink():raise ValueError('Text recovery path may not be a symlink')
            identity=metrics.sha(raw)
            immutable(recovery/(identity+'.uncommitted'),raw)
            receipt={'owner':OWNER,'response_contract_sha256':digest(producer),'source_name':path.name,
                'source_stat':before,'sha256':identity,'bytes':len(raw),'committed_prediction':False}
            immutable(recovery/(digest(receipt)+'.recovery.json'),(metrics.canonical(receipt)+'\n').encode())
            path.unlink();fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            continue
        if path.is_symlink() or path.name not in expected or not path.is_file():raise ValueError('Foreign text journal entry')
    result=[]
    for index,row in enumerate(context['records']):
        path=directory/f'{index:06d}.json'
        if not path.exists():
            if any((directory/f'{later:06d}.json').exists() for later in range(index+1,len(context['records']))):
                raise ValueError('Text journal is not a contiguous selected-record prefix')
            break
        envelope=read(path);raw=envelope.get('response')
        if (envelope.get('owner')!=OWNER or envelope.get('index')!=index or envelope.get('record_id')!=row['record_id'] or
                envelope.get('response_contract_sha256')!=digest(producer) or not isinstance(raw,dict) or
                envelope.get('response_sha256')!=digest(raw)):
            raise ValueError('Committed text response ownership/checksum differs')
        generated={k:v for k,v in raw.items() if k not in ('record_id','task','source','input_text_sha256','response_contract_sha256')}
        if validate_generated(row,context['census'][row['record_id']],generated,producer)!=raw:
            raise ValueError('Committed text response native metadata differs')
        result.append(raw)
    return result


def reconcile(output,rows,*,resume):
    path=Path(output)/'responses.jsonl';desired=b''.join((metrics.canonical(r)+'\n').encode() for r in rows)
    if path.is_symlink():raise ValueError('Raw response file may not be a symlink')
    if not path.exists():
        if rows and not resume:raise ValueError('Committed prefix requires explicit resume')
        if desired:immutable(path,desired)
        return
    old=path.read_bytes()
    if old==desired:return
    if not resume or not desired.startswith(old):raise ValueError('Raw responses diverge from durable journal; no repair')
    recovery=Path(output)/'recovery';recovery.mkdir(exist_ok=True)
    if recovery.is_symlink():raise ValueError('Raw recovery path may not be a symlink')
    immutable(recovery/(metrics.sha(old)+'.partial.jsonl'),old)
    stage=path.with_name('.responses.'+uuid.uuid4().hex+'.tmp')
    try:
        with stage.open('xb') as handle:handle.write(desired);handle.flush();os.fsync(handle.fileno())
        os.replace(stage,path)
    finally:stage.unlink(missing_ok=True)


def gpu_idle():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=15)
    if result.stdout.strip():raise RuntimeError('Text production requires an exclusive root-scheduled idle GPU')


def describe_model(model,loaded,plan):
    import torch
    from transformers.generation.utils import GenerationMixin
    refs=loaded.get('model_ref',{})
    native_sources={}
    for item in (type(model.base),GenerationMixin):
        source=inspect.getsourcefile(item)
        if source:native_sources[str(Path(source).resolve())]=sha(source)
    result={'loaded':loaded,'parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),
        'actual_parameter_buffer_router_norm_dtypes':dtype_report(model.base),
        'max_context':min(CONTEXT,int(getattr(model.base.config,'max_position_embeddings',CONTEXT))),
        'native_model_type':model.base.config.model_type,
        'eos_token_id':model.base.generation_config.eos_token_id,'pad_token_id':model.tokenizer.pad_token_id,
        'padding_side':model.tokenizer.padding_side,'fixed_template_clock':DATE,
        'actual_native_class_and_generation_source_sha256':native_sources,
        'cuda_version':torch.version.cuda,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
    if plan['kind']=='learned':
        result['snapshot_assets']={'model':snapshot_identity(model.model_path),
            'vision':snapshot_identity(model.vision_path)}
    else:result['snapshot_assets']={'model':loaded['snapshot_assets']}
    return result


def assert_runtime(runtime):
    for assets in runtime['snapshot_assets'].values():
        for item in assets.values():
            if stat(item['path'])!=item['stat']:raise ValueError('Actual loaded pretrained snapshot changed during text evaluation')


def evaluate(context,plan,output,*,resume=False,allow_gpu=False,cache_dir='cache/vision',reserve_gb=120,runtime=None,progress=None):
    if not math.isfinite(reserve_gb) or reserve_gb<120:raise ValueError('Preserve120GiB disk reserve')
    if plan.get('admission_blockers'):raise RuntimeError('; '.join(plan['admission_blockers']))
    output=Path(output)
    if output.is_symlink():raise ValueError('Text output may not be a symlink')
    output=output.resolve()
    if (output.is_relative_to(Path(context['bundle'])) or
            output.is_relative_to(ROOT/'datasets/research/primary-text-en-v1')):
        raise ValueError('Text output cannot alter input/gold bundles')
    if plan['kind']=='learned' and (output==Path(plan['run']) or output.is_relative_to(Path(plan['run'])/'.checkpoint_objects')):
        raise ValueError('Text output cannot alter managed checkpoint objects')
    parent=output
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<reserve_gb*2**30+64*2**20:raise OSError('Insufficient text journal disk reserve')
    if output.is_dir() and not (output/'generation_protocol.json').is_file() and any(
            p.name!='.text_generation.lock' for p in output.iterdir()):
        raise ValueError('Unowned partial text output has no protocol; no files are changed')
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'.text_generation.lock'
    if lock.is_symlink():raise ValueError('Text producer lock may not be a symlink')
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        static=static_context(context,plan);assert_static(context,static)
        state_path=output/'generation_protocol.json'
        previous=read(state_path) if state_path.exists() else None
        if previous is not None and not resume:raise ValueError('Existing text generation requires explicit --resume')
        if previous is not None and (previous.get('owner')!=OWNER or previous.get('static')!=static):
            raise ValueError('Resume model/source/input/environment profile changed')
        if previous is None and any(p.name!='.text_generation.lock' for p in output.iterdir()):
            raise ValueError('Unowned partial text output has no immutable generation protocol')
        producer=previous.get('producer') if previous is not None else None
        if previous is not None:
            if digest(producer)!=previous['producer_sha256']:raise ValueError('Text response producer contract changed')
            immutable(output/'producer.json',(metrics.canonical(producer)+'\n').encode())
            assert_runtime(previous['runtime'])
        rows=read_committed(output,context,producer,resume=resume) if previous else []
        if previous is not None:reconcile(output,rows,resume=resume)
        if len(rows)<len(context['records']):
            if not allow_gpu:raise RuntimeError('Pending text generation requires explicit root-scheduled --allow-gpu')
            if runtime is None:gpu_idle()
            engine=runtime or NativeTextRuntime()
            if plan['kind']=='pinned_baseline':model,loaded=engine.load_baseline(plan)
            else:model,loaded=engine.load_learned(plan['checkpoint'],cache_dir,reserve_gb)
            # Tests may inject a clearly synthetic runtime; CLI cannot inject
            # one and never writes synthetic rows as actual model evidence.
            injected=runtime is not None
            actual_runtime=engine.describe(model,loaded,plan) if injected else describe_model(model,loaded,plan)
            ref=loaded.get('model_ref',{})
            if ref.get('repo')!=context['model']['repo'] or ref.get('revision')!=context['model']['revision']:
                raise ValueError('Loaded actual model source differs from pinned tokenizer census')
            if plan['kind']=='pinned_baseline':identity=loaded['checkpoint_identity'];stage='baseline_before_vision'
            else:
                from src.utils.research_checkpoint_compat import checkpoint_identity
                identity=checkpoint_identity(plan['checkpoint']);stage=plan['stage']
                if loaded.get('stage')!=stage or loaded.get('training_step')!=plan['step'] or loaded.get('recipe')!=plan['recipe']:
                    raise ValueError('Loaded learned stage/step/recipe differs from managed checkpoint metadata')
            actual_producer={'schema_version':1,'owner':'research_primary_text_responses_v1',
                'suite_protocol_sha256':context['protocol']['parent_protocol_sha256'],
                'manifest_sha256':context['protocol']['parent_manifest_sha256'],'tasks':context['tasks'],
                'model_id':context['model']['model_id'],'model_revision':context['model']['revision'],'stage':stage,
                'numerical_profile':plan.get('numerical_profile',{'weights':plan.get('precision'),'compute':'native bf16 autocast'}),
                'checkpoint_identity':identity,'checkpoint_owner':plan.get('owner'),'training_step':plan.get('step'),
                'evidence_scope':'synthetic_scorer_validation' if injected else 'model_responses',
                'generation':context['protocol']['parent_generation_policy'],'native_runtime':actual_runtime,
                'producer_source_sha256':static['sources'],'public_input_protocol_sha256':context['protocol_sha256']}
            if not injected:
                metrics.validate_checkpoint_descriptor(identity,stage,context['model'],owner=plan.get('owner'),step=plan.get('step'))
            protocol={'schema_version':1,'owner':OWNER,'static':static,'runtime':actual_runtime,
                'producer_sha256':digest(actual_producer),'producer':actual_producer}
            if previous and (protocol!=previous or actual_producer!=producer):raise ValueError('Reloaded native model/runtime/precision differs during resume')
            if not previous:
                producer=actual_producer
                immutable(state_path,(metrics.canonical(protocol)+'\n').encode())
                immutable(output/'producer.json',(metrics.canonical(producer)+'\n').encode())
            (output/'records').mkdir(exist_ok=True);reconcile(output,rows,resume=resume)
            for index in range(len(rows),len(context['records'])):
                assert_static(context,static);assert_runtime(actual_runtime)
                if shutil.disk_usage(output).free<reserve_gb*2**30+64*2**20:raise OSError('Text generation reached disk reserve')
                row=context['records'][index];census=context['census'][row['record_id']]
                generated=engine.candidate_likelihoods(model,row,census,context['model']['model_id']) if row['task']=='mmlu' else engine.generate(model,row,census)
                raw=validate_generated(row,census,generated,producer);assert_static(context,static);assert_runtime(actual_runtime)
                envelope={'owner':OWNER,'index':index,'record_id':row['record_id'],
                    'response_contract_sha256':digest(producer),'response_sha256':digest(raw),'response':raw}
                immutable(output/'records'/f'{index:06d}.json',(metrics.canonical(envelope)+'\n').encode())
                with (output/'responses.jsonl').open('ab') as sink:sink.write((metrics.canonical(raw)+'\n').encode());sink.flush();os.fsync(sink.fileno())
                rows.append(raw)
                event={'status':'generating','task':row['task'],'completed_records':len(rows),'records':len(context['records'])}
                atomic(output/'progress.json',event)
                if progress:progress(event)
            del model
        else:reconcile(output,rows,resume=resume)
        rows=read_committed(output,context,producer,resume=resume)
        if len(rows)!=len(context['records']):raise ValueError('Incomplete text generation has no completion receipt')
        assert_static(context,static)
        summary={'schema_version':1,'owner':OWNER,'status':'generation_complete_unscored','records':len(rows),
            'response_contract_sha256':digest(producer),'responses_sha256':sha(output/'responses.jsonl'),
            'producer_sha256':sha(output/'producer.json'),'public_protocol_sha256':context['protocol_sha256'],
            'evidence_scope':producer['evidence_scope'],'gold_reference_files_opened':False,
            'scoring_performed':False,'primary_completion_claim':False}
        immutable(output/'generation_complete.json',(metrics.canonical(summary)+'\n').encode())
        return summary
