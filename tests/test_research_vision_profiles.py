from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model.research_vision import apply_preprocessing_profile,QWEN_PREPROCESSING_PROFILES,FrozenResearchVision


class PixelProcessor:
    patch_size=14
    merge_size=2
    min_pixels=3136
    max_pixels=12845056
    def to_dict(self):return {'min_pixels':self.min_pixels,'max_pixels':self.max_pixels,'patch_size':14,'merge_size':2}


def test_explicit_profile_limits_are_mutated_only_in_memory():
    processor=PixelProcessor();extractor=SimpleNamespace(backend='qwen2_5_vl',processor=SimpleNamespace(image_processor=processor))
    before=processor.to_dict()
    assert apply_preprocessing_profile(extractor,None) is None and processor.to_dict()==before
    contract=apply_preprocessing_profile(extractor,'qwen_native_bounded_2048')
    assert processor.max_pixels==2048*28*28 and processor.min_pixels==3136
    assert contract['reserved_text_tokens']==2048 and contract['constant_visual_token_count'] is False
    assert extractor.research_preprocessing_profile==contract


def test_qwen_profiles_cannot_silently_modify_siglip():
    extractor=SimpleNamespace(backend='siglip2',processor=SimpleNamespace(image_processor=PixelProcessor()))
    with pytest.raises(ValueError,match='only be applied'):
        apply_preprocessing_profile(extractor,'qwen_native_bounded_2048')


@pytest.mark.parametrize('profile',list(QWEN_PREPROCESSING_PROFILES))
def test_actual_cached_qwen_processor_square_and_elongated_limits(profile):
    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor
    from PIL import Image
    try:path=hf_hub_download('Qwen/Qwen2.5-VL-7B-Instruct','config.json',revision='cc594898137f460bfe9f0759e9844b3ce807cfb5',local_files_only=True)
    except Exception:pytest.skip('Pinned actual Qwen processor metadata is not cached; no network download in tests')
    processor=AutoProcessor.from_pretrained(Path(path).parent,local_files_only=True,trust_remote_code=False)
    extractor=SimpleNamespace(backend='qwen2_5_vl',processor=processor)
    contract=apply_preprocessing_profile(extractor,profile)
    counts=[]
    for size in [(2048,2048),(4096,512),(8192,128),(1792,896)]:
        image=Image.new('RGB',size,(30,80,120))
        batch=processor.image_processor(images=[image],return_tensors='pt')
        grid=batch['image_grid_thw'][0].tolist();tokens=grid[0]*grid[1]*grid[2]//4
        assert tokens<=QWEN_PREPROCESSING_PROFILES[profile]
        assert tokens*28*28<=contract['max_pixels']
        counts.append(tokens)
        del batch
    if profile=='qwen_native_bounded_2048':assert counts[-1]==2048


def test_explicit_profile_has_separate_fingerprint_and_default_has_no_profile_key():
    import torch
    from src.dataset.vision_cache import encoder_fingerprint
    def extractor():
        config=SimpleNamespace(vision_config=SimpleNamespace(hidden_size=1280,out_hidden_size=2),to_dict=lambda:{'model_type':'qwen2_5_vl'})
        return SimpleNamespace(backend='qwen2_5_vl',config=config,processor=SimpleNamespace(image_processor=PixelProcessor()),
            vision_tower=torch.nn.Linear(1,1),vision_projector=None,dtype=torch.float32)
    native=FrozenResearchVision(extractor(),model_path='mock')
    changed=extractor();apply_preprocessing_profile(changed,'qwen_native_bounded_2048')
    bounded=FrozenResearchVision(changed,model_path='mock')
    assert 'preprocessing_profile' not in native.fingerprint_config
    assert bounded.fingerprint_config['preprocessing_profile']['max_visual_tokens']==2048
    assert encoder_fingerprint(native.fingerprint_config)!=encoder_fingerprint(bounded.fingerprint_config)
