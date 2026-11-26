from pathlib import Path
from types import SimpleNamespace
import sys
import json

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model.research_vision import FrozenResearchVision,_StrictVisionFeatureExtractor,_load_prefix_states
from src.dataset.vision_cache import VisionFeatureCache,encoder_fingerprint


class FakeConfig(SimpleNamespace):
    def to_dict(self):
        return {'model_type':self.model_type,'vision_hidden':self.vision_config.hidden_size}


class FakeExtractor:
    def __init__(self,backend,feature_dim):
        self.backend=backend
        self.dtype=torch.float32
        self.config=FakeConfig(model_type=backend,vision_config=SimpleNamespace(hidden_size=7,out_hidden_size=feature_dim),text_config=SimpleNamespace(hidden_size=feature_dim))
        self.processor=SimpleNamespace(to_dict=lambda:{'size':8})
        self.vision_tower=torch.nn.Linear(3,3)
        self.vision_projector=torch.nn.Linear(3,3) if backend=='gemma4' else None
        self.calls=0
    def encode_batch(self,images):
        self.calls+=1
        dim=self.config.vision_config.out_hidden_size
        features=[]
        for image in images:
            value=float(image.getpixel((0,0))[0])
            features.append(torch.full((2,dim),value))
        return features,{'grid':[torch.tensor([1,2,3]) for _ in images]}


@pytest.mark.parametrize('backend,expected',[('siglip2',7),('qwen2_5_vl',11),('gemma4',13)])
def test_feature_dimensions_freeze_and_image_order(tmp_path,backend,expected):
    extractor=FakeExtractor(backend,expected)
    wrapper=FrozenResearchVision(extractor,model_path='mock/revision')
    assert wrapper.feature_dim==expected
    assert all(not p.requires_grad for p in extractor.vision_tower.parameters())
    paths=[]
    for i,color in enumerate([(20,0,0),(40,0,0)]):
        p=tmp_path/f'{i}.png';Image.new('RGB',(8,9),color).save(p);paths.append(str(p))
    output=wrapper.encode_paths(paths)
    assert output[0]['features'][0,0]==20 and output[1]['features'][0,0]==40
    assert output[0]['metadata']['grid']==[1,2,3]
    assert output[0]['metadata']['original_size']==[8,9]
    assert output[1]['metadata']['valid_token_count']==2
    assert 'feature_point' in wrapper.fingerprint_config
    cache=VisionFeatureCache(tmp_path/'cache',encoder_fingerprint(wrapper.fingerprint_config),reserve_bytes=0)
    cached=cache.get_many(paths,wrapper.encode_paths)
    assert torch.equal(cached[0],output[0]['features'])
    calls=extractor.calls
    reordered=cache.get_many(paths[::-1],wrapper.encode_paths,read_only=True)
    assert extractor.calls==calls and torch.equal(reordered[0],output[1]['features'])


def test_only_required_vision_shards_are_read(tmp_path):
    save_file({'visual.a':torch.ones(2,3),'language.unrelated':torch.zeros(1)},tmp_path/'vision.safetensors')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'visual.a':'vision.safetensors','language.other':'NOT_DOWNLOADED_TEXT.safetensors'}}))
    states,files=_load_prefix_states(tmp_path,['model.visual.','visual.'])
    assert files==['vision.safetensors']
    assert list(states['visual.'])==['a']
    assert states['model.visual.']=={}


def test_missing_vision_shard_fails_without_full_donor_fallback(tmp_path):
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'visual.a':'missing.safetensors'}}))
    extractor=object.__new__(_StrictVisionFeatureExtractor)
    extractor.backend='qwen2_5_vl';extractor.vision_model=str(tmp_path)
    extractor.config=SimpleNamespace(vision_config=None)
    with pytest.raises(FileNotFoundError,match='no full donor fallback'):
        extractor._load_modules()


def test_gemma_fixed_image_only_processor_and_token_counts():
    seen={}
    class Processor:
        image_token='<image>'
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token:99)
        def __call__(self,**kwargs):
            seen.update(kwargs)
            return {'input_ids':torch.tensor([[99,99],[99,0]]),
                    'pixel_values':torch.zeros(2,2,3),
                    'image_position_ids':torch.tensor([[[0,0],[0,1]],[[0,0],[-1,-1]]])}
    class Tower(torch.nn.Module):
        def forward(self,pixel_values,pixel_position_ids,return_dict):
            return SimpleNamespace(last_hidden_state=torch.ones(2,2,4))
    extractor=object.__new__(_StrictVisionFeatureExtractor)
    extractor.processor=Processor();extractor.device=torch.device('cpu');extractor.dtype=torch.float32
    extractor.vision_tower=Tower();extractor.vision_projector=torch.nn.Linear(4,6)
    images=[Image.new('RGB',(4,4)),Image.new('RGB',(4,4))]
    features,metadata=extractor._encode_gemma_batch(images)
    assert seen['text']==['<image>','<image>']
    assert [f.shape for f in features]==[torch.Size([2,6]),torch.Size([1,6])]
    assert len(metadata['image_position_ids'])==2


def test_invalid_feature_dimension_is_not_silently_projected(tmp_path):
    extractor=FakeExtractor('qwen2_5_vl',11)
    wrapper=FrozenResearchVision(extractor,model_path='mock/revision')
    extractor.encode_batch=lambda images:([torch.ones(2,7)],{})
    path=tmp_path/'image.png';Image.new('RGB',(3,3)).save(path)
    with pytest.raises(ValueError,match='tokens,11'):
        wrapper.encode_paths([str(path)])
