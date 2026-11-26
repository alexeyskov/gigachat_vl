"""Pinned native Giga DSV3 config normalization and ordinary PEFT scope only."""
from __future__ import annotations
import copy,hashlib,json,math
from pathlib import Path
MODEL='ai-sage/GigaChat3.1-10B-A1.8B-bf16';REVISION='189fff27a1dee68473960c3d5bca53e0e07a3191'
CONFIG_SHA='6a6b8260f08791c4968f70934903e9aa892a53ee3f2e2e05b61a68ea1ff33503'
INDEX_SHA='1f08c4954c2e8319ad6abe35f9874cd28c4e74bb8cc589984e31bf9f0267c1fa'

def normalize_config(values):
 result=copy.deepcopy(values)
 if result.get('model_type')!='deepseek_v3':raise ValueError('Expected native Giga DeepseekV3 config')
 value=result.get('routed_scaling_factor')
 if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Positive native routed scaling factor required')
 result['routed_scaling_factor']=float(value)
 return result

def pinned_config(path):
 from transformers import AutoConfig
 path=Path(path);raw=(path/'config.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=CONFIG_SHA or path.resolve().name!=REVISION:raise ValueError('Exact pinned Giga source config required')
 values=normalize_config(json.loads(raw));values.pop('model_type')
 return AutoConfig.for_model('deepseek_v3',**values)

def tokenizer_proxy(native=None):
 class Tokenizer:
  @staticmethod
  def from_pretrained(path,*args,**kwargs):
   from transformers import AutoTokenizer
   config=pinned_config(path)
   if 'config' in kwargs and kwargs['config'].to_dict()!=config.to_dict():raise ValueError('Giga tokenizer caller config differs')
   kwargs.update(config=config,local_files_only=True,trust_remote_code=False)
   return (native or AutoTokenizer).from_pretrained(path,*args,**kwargs)
 return Tokenizer

class ScopedPEFT:
 def __getattr__(self,name):
  import peft
  return getattr(peft,name)
 def get_peft_model(self,model,config,*args,**kwargs):
  from src.model.research_giga_peft import get_giga_ordinary_peft_model,resolve_giga_ordinary_targets
  if args or kwargs or config.r!=16 or config.lora_alpha!=32 or config.lora_dropout!=.05 or config.bias!='none' or config.target_parameters or set(config.target_modules)!=set(resolve_giga_ordinary_targets(model)):raise ValueError('Exact native ordinary Giga rank16/alpha32/dropout.05 scope required')
  result,receipt=get_giga_ordinary_peft_model(model,target_modules=list(config.target_modules));return result

def install_loader(module):
 """Only the private runner is changed; native packed conversion remains native."""
 if module.__name__=='scripts.run_research_stage':raise ValueError('Canonical Giga loader patch forbidden')
 NativeConfig=module.AutoConfig;NativeModel=module.AutoModelForCausalLM
 class Config:
  @staticmethod
  def from_pretrained(path,*args,**kwargs):
   values=json.loads((Path(path)/'config.json').read_bytes())
   if values.get('model_type')=='deepseek_v3':return pinned_config(path)
   return NativeConfig.from_pretrained(path,*args,**kwargs)
 class Model:
  @staticmethod
  def from_pretrained(path,*args,**kwargs):
   config=pinned_config(path)
   if 'config' in kwargs and kwargs['config'].to_dict()!=config.to_dict():raise ValueError('Giga caller config differs from pinned normalized source')
   kwargs.update(config=config,local_files_only=True,trust_remote_code=False,experts_implementation='eager',output_loading_info=True)
   index_raw=(Path(path)/'model.safetensors.index.json').read_bytes()
   if hashlib.sha256(index_raw).hexdigest()!=INDEX_SHA:raise ValueError('Pinned Giga source index changed')
   source_keys=json.loads(index_raw)['weight_map'];unused={name for name in source_keys if name.startswith('model.layers.26.')}
   if len(source_keys)!=5323 or len(unused)!=210:raise ValueError('Pinned Giga native/MTP source schema differs')
   model,info=NativeModel.from_pretrained(path,*args,**kwargs)
   unexpected=info.get('unexpected_keys',[])
   if (info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs') or
       any(name not in unused for name in unexpected)):
    raise ValueError('Native Giga LM core source coverage differs: '+str(info))
   if getattr(model.config,'model_type',None)!='deepseek_v3' or type(model.config.routed_scaling_factor) is not float:raise ValueError('Native materialized Giga config normalization failed')
   model._giga_budget_loading={'source_config_sha256':CONFIG_SHA,'source_index_sha256':INDEX_SHA,'source_index_keys':len(source_keys),'unused_MTP_index_keys':len(unused),'source_revision':REVISION,'routed_scaling_factor_source_type':'int','routed_scaling_factor_runtime_type':'float','routed_scaling_factor_value':config.routed_scaling_factor,'source_file_rewritten':False,'experts_implementation':'eager','packed_quantization':'native research_quantization after ordinary scopedPEFT; no genericPEFT target_parameters','native_loading_info':info,'unused_original_MTP_layer':'model.layers.26; native causalLM has26decoder layers and no auxiliary MTP forward','LM_core_missing_or_mismatched_keys':False}
   return model
 module.AutoConfig=Config;module.AutoModelForCausalLM=Model
 if hasattr(module,'AutoTokenizer'):module.AutoTokenizer=tokenizer_proxy(module.AutoTokenizer)
 return module
