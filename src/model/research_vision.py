"""Strict frozen vision-only adapters for the research training/cache pipeline.

Feature extraction reuses the existing extractor's backend contracts. Qwen and
Gemma load only vision-prefixed tensors; no fallback loads a donor language LM.
All calls are image-only: Gemma receives a fixed image placeholder, never a user
question. Required donor weight files must already exist locally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import hashlib
import json
import re

import torch
from PIL import Image, ImageOps
from safetensors import safe_open

from scripts.extract_vision_features import VisionFeatureExtractor
from src.model.gigachat_vl import _maybe_tensor_to


FEATURE_POINTS = {
    'siglip2': 'last_hidden_patch_sequence_before_our_mlp',
    'siglip': 'last_hidden_patch_sequence_before_our_mlp',
    'qwen2_5_vl': 'native_spatial_merger_output_before_our_mlp',
    'gemma4': 'frozen_donor_multimodal_embedder_output_before_our_mlp',
}



QWEN_PREPROCESSING_PROFILES = {
    'qwen_native_bounded_2048': 2048,
    'qwen_native_bounded_1024': 1024,
    'qwen_native_bounded_729': 729,
}


def apply_preprocessing_profile(extractor,profile_name):
    """Apply an explicit in-memory native area cap before fingerprint creation.

    None leaves the donor processor unchanged. A token cap preserves Qwen's
    native smart-resize/aspect ratio; it does not pool, crop, or force every image
    to have an equal number of tokens. The729 profile is only an upper-budget
    comparison with SigLIP2's729-token grid, not equal geometry/compute/storage.
    """
    if profile_name is None:return None
    if extractor.backend!='qwen2_5_vl':
        raise ValueError('Qwen preprocessing profiles may only be applied to the Qwen2.5 vision backend')
    if profile_name not in QWEN_PREPROCESSING_PROFILES:
        raise ValueError(f'Unknown preprocessing profile {profile_name}')
    processor=extractor.processor.image_processor
    patch=int(getattr(processor,'patch_size',14));merge=int(getattr(processor,'merge_size',2))
    if patch!=14 or merge!=2:raise ValueError('Declared Qwen2.5 research profiles require patch14/merge2')
    grid_factor=patch*merge;budget=QWEN_PREPROCESSING_PROFILES[profile_name]
    native=processor.to_dict() if hasattr(processor,'to_dict') else {}
    size=dict(getattr(processor,'size',{}) or {})
    minimum=getattr(processor,'min_pixels',None) or native.get('min_pixels') or size.get('shortest_edge')
    original_max=getattr(processor,'max_pixels',None) or native.get('max_pixels') or size.get('longest_edge')
    if minimum is None or original_max is None:raise ValueError('Cannot establish actual Qwen processor pixel limits')
    maximum=budget*grid_factor**2
    if minimum>maximum:raise ValueError('Native minimum image area exceeds selected visual-token budget')
    # Support classic and newer size-dictionary image processor APIs. This only
    # mutates the instance; pinned preprocessor files and active profiles stay intact.
    if hasattr(processor,'max_pixels'):processor.max_pixels=maximum
    if 'longest_edge' in size:
        size['longest_edge']=maximum;processor.size=size
    if not hasattr(processor,'max_pixels') and 'longest_edge' not in size:
        raise ValueError('Unsupported Qwen processor pixel-limit override API')
    contract={'name':profile_name,'version':1,'max_visual_tokens':budget,
              'max_pixels':maximum,'min_pixels':int(minimum),'native_max_pixels':int(original_max),
              'patch_size':patch,'merge_size':merge,'grid_factor':grid_factor,
              'max_context_tokens':4096,'reserved_text_tokens':4096-budget,
              'resize':'native_smart_resize_preserve_aspect_ratio',
              'constant_visual_token_count':False,'post_encoder_pooling':False}
    if budget==729:
        contract['comparison_limitation']='Matches only SigLIP2 maximum729-token budget; Qwen token counts remain variable, aspect ratios and feature width differ'
    extractor.research_preprocessing_profile=contract
    return contract

def _local_model_dir(model_path):
    path=Path(model_path)
    if path.is_dir():
        return path
    from transformers.utils.hub import cached_file
    config_path=cached_file(model_path,'config.json')
    return Path(config_path).parent



def required_vision_weight_files(model_dir, backend):
    """List indexed donor shards needed by the strict loader, without reading weights."""
    path=Path(model_dir)
    index=path/'model.safetensors.index.json'
    if not index.exists():
        return ['model.safetensors']
    prefixes={'qwen2_5_vl':['model.visual.','visual.'],
              'gemma4':['model.vision_tower.','model.embed_vision.'],
              'siglip':['vision_model.'],'siglip2':['vision_model.']}[backend]
    weight_map=json.loads(index.read_text())['weight_map']
    files=sorted({file for key,file in weight_map.items() if any(key.startswith(p) for p in prefixes)})
    if not files:
        raise KeyError(f'No vision tensors for backend {backend} in donor index')
    return files


def _load_prefix_states(model_dir: Path, prefixes: Sequence[str]):
    """Read only matching tensors from selected index shards or one safetensor."""
    index=model_dir/'model.safetensors.index.json'
    if index.exists():
        weight_map=json.loads(index.read_text())['weight_map']
        matching={key:file for key,file in weight_map.items() if any(key.startswith(p) for p in prefixes)}
        if not matching:
            raise KeyError(f'No donor vision tensors match {prefixes}')
        files=sorted(set(matching.values()))
    elif (model_dir/'model.safetensors').exists():
        files=['model.safetensors']
    else:
        raise FileNotFoundError(f'No local donor safetensors/index under {model_dir}')
    missing=[file for file in files if not (model_dir/file).exists()]
    if missing:
        raise FileNotFoundError(f'Missing required vision weight shards under {model_dir}: {missing}; no full donor fallback is allowed')
    states={p:{} for p in prefixes}
    for file in files:
        with safe_open(str(model_dir/file),framework='pt',device='cpu') as reader:
            for key in reader.keys():
                for prefix in prefixes:
                    if key.startswith(prefix):
                        states[prefix][key[len(prefix):]]=reader.get_tensor(key)
                        break
    return states,files


def _select_nonempty(states, prefixes):
    matching=[states[p] for p in prefixes if states[p]]
    if len(matching)!=1:
        raise ValueError(f'Expected exactly one vision prefix from {prefixes}, found {len(matching)}')
    return matching[0]


class _StrictVisionFeatureExtractor(VisionFeatureExtractor):
    def _load_modules(self):
        if self.backend=='qwen2_5_vl':
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
            model_dir=_local_model_dir(self.vision_model)
            prefixes=['model.visual.','visual.']
            states,self.loaded_weight_files=_load_prefix_states(model_dir,prefixes)
            self.vision_tower=Qwen2_5_VisionTransformerPretrainedModel(self.config.vision_config)
            self.vision_tower.load_state_dict(_select_nonempty(states,prefixes),strict=True)
            self.vision_tower.to(device=self.device,dtype=self.dtype).requires_grad_(False).eval()
            return
        if self.backend=='gemma4':
            from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel,Gemma4MultimodalEmbedder
            model_dir=_local_model_dir(self.vision_model)
            prefixes=['model.vision_tower.','model.embed_vision.']
            states,self.loaded_weight_files=_load_prefix_states(model_dir,prefixes)
            if any(not states[p] for p in prefixes):
                raise KeyError('Gemma4 donor requires both vision_tower and embed_vision tensor prefixes')
            self.vision_tower=Gemma4VisionModel(self.config.vision_config)
            self.vision_projector=Gemma4MultimodalEmbedder(self.config.vision_config,self.config.text_config)
            self.vision_tower.load_state_dict(states[prefixes[0]],strict=True)
            self.vision_projector.load_state_dict(states[prefixes[1]],strict=True)
            self.vision_tower.to(device=self.device,dtype=self.dtype).requires_grad_(False).eval()
            self.vision_projector.to(device=self.device,dtype=self.dtype).requires_grad_(False).eval()
            return
        if self.backend not in {'siglip','siglip2'}:
            raise ValueError(f'Encoder {self.backend} is outside the research donor matrix')
        super()._load_modules()  # SigLIP vision-only class; no language tower.

    def _encode_siglip_batch(self,images):
        # Keep integer spatial_shapes/masks; only floating-point pixels change dtype.
        try:
            batch=self.processor(images=images,return_tensors='pt',padding=True)
        except TypeError:
            batch=self.processor(images=images,return_tensors='pt')
        prepared={k:_maybe_tensor_to(v,self.device,self.dtype) for k,v in batch.items()}
        kwargs={'pixel_values':prepared['pixel_values']}
        if self.backend=='siglip2':
            kwargs.update(pixel_attention_mask=prepared['pixel_attention_mask'],spatial_shapes=prepared['spatial_shapes'])
        hidden=self.vision_tower(**kwargs).last_hidden_state
        if hidden.ndim!=3 or hidden.shape[0]!=len(images):
            raise ValueError(f'Invalid SigLIP patch states: {tuple(hidden.shape)}')
        features=[]
        for i in range(len(images)):
            feature=hidden[i]
            mask=prepared.get('pixel_attention_mask')
            if mask is not None:
                if mask.ndim!=2 or mask.shape[1]!=feature.shape[0]:
                    raise ValueError('SigLIP pixel mask does not match patch-token dimension')
                feature=feature[mask[i].bool()]
            features.append(feature)
        metadata={'valid_token_count':[torch.tensor(len(x)) for x in features]}
        if 'spatial_shapes' in prepared:
            metadata['spatial_shapes']=[v.detach().cpu() for v in prepared['spatial_shapes']]
        return features,metadata

    def _encode_gemma_batch(self,images):
        image_token=getattr(self.processor,'image_token',None) or getattr(getattr(self.processor,'tokenizer',None),'image_token',None)
        if not image_token:
            raise ValueError('Gemma4 processor has no image placeholder token')
        # Fixed image-only placeholder is required to determine native soft-token
        # counts. User prompts/answers are deliberately absent from this API.
        batch=self.processor(images=[[image] for image in images],text=[image_token]*len(images),return_tensors='pt',padding=True)
        prepared={k:_maybe_tensor_to(v,self.device,self.dtype) for k,v in batch.items()}
        positions=prepared['image_position_ids']
        try:
            output=self.vision_tower(pixel_values=prepared['pixel_values'],pixel_position_ids=positions,return_dict=True)
        except TypeError as error:
            if 'pixel_position_ids' not in str(error):
                raise
            output=self.vision_tower(pixel_values=prepared['pixel_values'],image_position_ids=positions,return_dict=True)
        hidden=self.vision_projector(output.last_hidden_state)
        token_id=self.processor.tokenizer.convert_tokens_to_ids(image_token)
        counts=(batch['input_ids']==token_id).sum(dim=1).tolist()
        counts=[int(n) for n in counts]
        if any(n<=0 for n in counts):
            raise ValueError('Gemma4 processor produced no image soft-token positions')
        if hidden.ndim==2:
            if sum(counts)!=hidden.shape[0]:
                raise ValueError('Gemma4 token counts do not match flattened embedder output')
            features=list(hidden.split(counts,dim=0))
        elif hidden.ndim==3:
            if hidden.shape[0]!=len(images) or any(n>hidden.shape[1] for n in counts):
                raise ValueError('Gemma4 batched embedder output does not match image counts')
            features=[hidden[i,:n] for i,n in enumerate(counts)]
        elif hidden.ndim==4 and hidden.shape[1]==1:
            if hidden.shape[0]!=len(images) or any(n>hidden.shape[2] for n in counts):
                raise ValueError('Gemma4 image-axis output does not match image counts')
            features=[hidden[i,0,:n] for i,n in enumerate(counts)]
        else:
            raise ValueError(f'Invalid Gemma4 donor embedder shape: {tuple(hidden.shape)}')
        metadata={'image_token_count':[torch.tensor(n) for n in counts],
                  'image_position_ids':[positions[i].detach().cpu() for i in range(len(images))]}
        return features,metadata


def _jsonable(value):
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value,dict): return {str(k):_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_jsonable(v) for v in value]
    if isinstance(value,(str,int,float,bool)) or value is None: return value
    return str(value)


def _sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while chunk:=f.read(16*1024**2): h.update(chunk)
    return h.hexdigest()


class FrozenResearchVision:
    def __init__(self,extractor,*,model_path):
        self.extractor=extractor
        self.backend=extractor.backend
        self.config=extractor.config
        self.vision_tower=extractor.vision_tower
        self.vision_projector=extractor.vision_projector
        self.processor=extractor.processor
        for module in [self.vision_tower,self.vision_projector]:
            if module is not None: module.requires_grad_(False).eval()
        vision_config=getattr(self.config,'vision_config',self.config)
        if self.backend=='qwen2_5_vl':
            self.feature_dim=getattr(vision_config,'out_hidden_size',None)
        elif self.backend=='gemma4':
            self.feature_dim=self.config.text_config.hidden_size
        elif self.backend in {'siglip','siglip2'}:
            self.feature_dim=vision_config.hidden_size
        else:
            raise ValueError(f'Unsupported research vision backend {self.backend}')
        if not isinstance(self.feature_dim,int) or self.feature_dim<=0:
            raise ValueError(f'Invalid donor feature dimension {self.feature_dim}')
        self.preprocessing_profile=getattr(extractor,'research_preprocessing_profile',None)
        self.fingerprint_config=self._fingerprint(model_path)

    def _fingerprint(self,model_path):
        path=Path(model_path)
        revision=getattr(self.config,'_commit_hash',None)
        if path.is_dir() and re.fullmatch(r'[0-9a-f]{40}',path.name): revision=path.name
        metadata_path=path/'.cache/huggingface/download/config.json.metadata'
        if not revision and metadata_path.exists(): revision=metadata_path.read_text().splitlines()[0]
        config=self.config.to_dict() if hasattr(self.config,'to_dict') else vars(self.config)
        processor_config={}
        image_processor=getattr(self.processor,'image_processor',self.processor)
        if hasattr(image_processor,'to_dict'): processor_config=image_processor.to_dict()
        # Local unpinned directories must include actual weight checksums, rather
        # than sharing a cache solely because they have the same model basename.
        weights={}
        if not revision and path.is_dir():
            names=getattr(self.extractor,'loaded_weight_files',None)
            files=[path/n for n in names] if names else sorted(path.glob('*.safetensors'))
            weights={p.name:_sha256_file(p) for p in files}
        result={'encoder_revision':revision,'encoder_config':config,
                          'local_weight_checksums':weights,'backend':self.backend,
                          'processor_config':processor_config,'feature_dim':self.feature_dim,
                          'feature_point':FEATURE_POINTS[self.backend],
                          'compute_dtype':str(self.extractor.dtype),
                          'extraction_code_sha256':_sha256_file(Path(__file__)),
                          'image_decode_policy':'PIL_EXIF_transpose_RGB',
                          'fixed_image_placeholder_only':self.backend=='gemma4'}
        if self.preprocessing_profile is not None:
            result['preprocessing_profile']=self.preprocessing_profile
        return _jsonable(result)

    @torch.no_grad()
    def encode_paths(self,paths:Sequence[str]):
        if not paths: return []
        images=[]; sizes=[]
        for path in paths:
            with Image.open(path) as image:
                rgb=ImageOps.exif_transpose(image).convert('RGB')
                images.append(rgb)
                sizes.append(list(rgb.size))
        try:
            for module in [self.vision_tower,self.vision_projector]:
                if module is not None: module.eval()
            features,metadata=self.extractor.encode_batch(images)
        finally:
            for image in images: image.close()
        if len(features)!=len(paths):
            raise ValueError('Vision output count does not match image-path order')
        if any(len(values)!=len(paths) for values in metadata.values()):
            raise ValueError('Vision metadata count does not match images')
        result=[]
        for i,feature in enumerate(features):
            if feature.ndim!=2 or feature.shape[1]!=self.feature_dim or feature.shape[0]<=0:
                raise ValueError(f'Expected nonempty [tokens,{self.feature_dim}] features, got {tuple(feature.shape)}')
            if not torch.isfinite(feature).all(): raise ValueError('Donor returned non-finite vision features')
            if self.preprocessing_profile is not None and feature.shape[0]>self.preprocessing_profile['max_visual_tokens']:
                raise ValueError(f"Native Qwen output exceeds declared visual-token budget: {feature.shape[0]}>{self.preprocessing_profile['max_visual_tokens']}")
            row_metadata={key:_jsonable(value[i]) for key,value in metadata.items()}
            row_metadata.update(original_size=sizes[i],valid_token_count=feature.shape[0],backend=self.backend,feature_point=FEATURE_POINTS[self.backend])
            result.append({'features':feature.detach(),'metadata':row_metadata})
        return result


def build_frozen_vision(model_path,device,dtype,*,preprocessing_profile=None):
    """Build a strict vision-only frozen encoder from a prepared donor snapshot.

    This function never downloads large donor weight shards or falls back to
    the full multimodal LM. Prepare the required local shards before calling.
    """
    extractor=_StrictVisionFeatureExtractor(str(model_path),torch.device(device),dtype)
    apply_preprocessing_profile(extractor,preprocessing_profile)
    return FrozenResearchVision(extractor,model_path=str(model_path))
