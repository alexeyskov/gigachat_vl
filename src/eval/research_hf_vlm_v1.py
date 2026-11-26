"""Native HF image/text generation, with optional runtime NF4 quantization."""
from pathlib import Path
import hashlib,time

NATIVE_UNQUANTIZED_PATTERNS=[r'(^|.*\.)(visual|vision_tower|vision_model|multi_modal_projector)(\.|$)','lm_head']


class NativeHFVLM:
    def __init__(self,path,*,quantization='nf4',max_visual_tokens=729,max_new_tokens=64,
                 device='cuda:0',local_files_only=True,max_context=8192):
        import torch
        import bitsandbytes as bnb
        from transformers import AutoProcessor,AutoModelForImageTextToText,BitsAndBytesConfig
        self.torch=torch;self.device=device;self.max_new_tokens=max_new_tokens
        self.max_visual_tokens=max_visual_tokens;self.max_context=max_context
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=local_files_only,
            trust_remote_code=False,min_pixels=4*28*28,max_pixels=max_visual_tokens*28*28)
        kwargs=dict(dtype=torch.bfloat16,device_map={'':device},attn_implementation='sdpa',
                    local_files_only=local_files_only,trust_remote_code=False)
        if quantization=='nf4':
            kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=list(NATIVE_UNQUANTIZED_PATTERNS))
        elif quantization!='none':raise ValueError('Supported quantization: nf4 or none')
        from src.model.gigachat_vl import _without_transformers_allocator_warmup
        with _without_transformers_allocator_warmup():
            self.model=AutoModelForImageTextToText.from_pretrained(path,**kwargs).eval().requires_grad_(False)
        self.quantized_linear_modules=[n for n,m in self.model.named_modules() if isinstance(m,bnb.nn.Linear4bit)]
        if quantization=='nf4' and not self.quantized_linear_modules:raise ValueError('No actual INT4 modules loaded')
        self.loading={'quantization':quantization,'quantized_linear_modules':len(self.quantized_linear_modules),
                      'unquantized_module_patterns':NATIVE_UNQUANTIZED_PATTERNS,
                      'vision_quantized':any(any(p in n for p in ('visual','vision_tower','vision_model')) for n in self.quantized_linear_modules),
                      'max_visual_tokens_per_image':max_visual_tokens,'max_new_tokens':max_new_tokens,
                      'model_class':type(self.model).__name__,'processor_class':type(self.processor).__name__}
        if self.loading['vision_quantized']:raise ValueError('Native vision encoder must preserve BF16')
        self.loading['parameter_dtypes']=sorted({str(p.dtype) for p in self.model.parameters()})
        self.loading['model_type']=self.model.config.model_type
        vision_dtypes=sorted({str(p.dtype) for n,p in self.model.named_parameters()
            if any(part in n.split('.') for part in ('visual','vision_tower','vision_model'))})
        self.loading['vision_parameter_dtypes']=vision_dtypes
        if vision_dtypes and vision_dtypes!=['torch.bfloat16']:raise ValueError('Native vision parameters must retain BF16')

    def generate(self,message):
        from PIL import Image,ImageOps
        torch=self.torch;content=[];images=[]
        for item in message:
            if item['type']=='image':
                with Image.open(item['value']) as im:image=ImageOps.exif_transpose(im).convert('RGB')
                images.append(image);content.append({'type':'image','image':image})
            elif item['type']=='text':content.append({'type':'text','text':item['value']})
            else:raise ValueError('Image/text messages only')
        prompt=self.processor.apply_chat_template([{'role':'user','content':content}],
            tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[prompt],images=images or None,return_tensors='pt')
        if inputs['input_ids'].shape[1]+self.max_new_tokens>self.max_context:
            raise ValueError('Prompt plus generation exceeds declared context; no silent truncation')
        counts=[]
        if 'image_grid_thw' in inputs:
            merge=int(getattr(self.processor.image_processor,'merge_size',2))
            counts=inputs['image_grid_thw'].prod(-1)//merge**2
            if int(counts.max())>self.max_visual_tokens:raise ValueError('Native image processor exceeded visual-token bound')
            counts=counts.tolist()
        inputs=inputs.to(self.device);start=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.model.generate(**inputs,do_sample=False,num_beams=1,
                max_new_tokens=self.max_new_tokens,use_cache=True,return_dict_in_generate=True,
                output_scores=True,pad_token_id=self.processor.tokenizer.pad_token_id)
        for scores in output.scores:
            if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any():
                raise FloatingPointError('Invalid native generation logits')
        ids=output.sequences[0,inputs['input_ids'].shape[1]:].detach().cpu().tolist()
        text=self.processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
        return {'prediction':text,'generated_token_ids':ids,'generated_tokens':len(ids),
                'generation_seconds':time.monotonic()-start,'prefill_tokens':inputs['input_ids'].shape[1],
                'visual_token_counts':counts,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
