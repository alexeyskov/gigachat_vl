import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.download_research_model import native_assets,vision_scope_assets


def test_vision_download_uses_index_shards_and_excludes_text_only_weights(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'qwen2_5_vl'}))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'visual.blocks.0.weight':'model-1.safetensors','model.layers.0.weight':'model-2.safetensors'}}))
    row={'files':[{'path':name,'bytes':100} for name in ['config.json','model.safetensors.index.json','tokenizer.json','model-1.safetensors','model-2.safetensors','original/model.safetensors']]}
    selected,required,excluded=vision_scope_assets(row,tmp_path)
    assert required==['model-1.safetensors'] and excluded==['model-2.safetensors']
    assert {f['path'] for f in selected}=={'config.json','model.safetensors.index.json','tokenizer.json','model-1.safetensors'}
    assert all('/' not in f['path'] for f in native_assets(row))


def test_vision_download_rejects_text_only_repository(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'lfm2_moe'}))
    with pytest.raises(ValueError,match='non-vision'):
        vision_scope_assets({'files':[{'path':'model.safetensors','bytes':10}]},tmp_path)


def test_vision_index_cannot_bypass_pinned_inventory(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'qwen2_5_vl'}))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'visual.weight':'unlisted.safetensors'}}))
    with pytest.raises(ValueError,match='absent from pinned inventory'):
        vision_scope_assets({'files':[{'path':'config.json','bytes':10}]},tmp_path)
