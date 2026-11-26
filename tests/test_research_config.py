import copy
import hashlib
import json

import pytest

from src.model.research_config import load_research_native_config,normalize_research_config_dict


def test_only_deepseek_integer_one_is_normalized_without_source_mutation(tmp_path):
    raw={'model_type':'deepseek_v3','routed_scaling_factor':1,
         'hidden_size':8,'num_hidden_layers':1,'num_attention_heads':2,
         'transformers_version':'4.57.3'}
    original=copy.deepcopy(raw)
    normalized,changes=normalize_research_config_dict(raw)
    assert raw==original and type(raw['routed_scaling_factor']) is int
    assert type(normalized['routed_scaling_factor']) is float
    assert normalized['routed_scaling_factor']==raw['routed_scaling_factor']
    assert [row['field'] for row in changes]==['routed_scaling_factor']
    path=tmp_path/'config.json';path.write_text(json.dumps(raw))
    source=path.read_bytes();config,receipt=load_research_native_config(path)
    assert config.routed_scaling_factor==1.0
    assert path.read_bytes()==source
    assert receipt['source_sha256']==hashlib.sha256(source).hexdigest()
    assert receipt['declared_transformers_version']=='4.57.3'
    assert receipt['installed_transformers_version']
    assert receipt['normalized_native_validation_passed']
    assert receipt['original_native_validation_error']['type']=='StrictDataclassFieldValidationError'


@pytest.mark.parametrize('kind,value',[('deepseek_v3',2),('deepseek_v3',True),('deepseek_v3','1'),('gpt_oss',1),('deepseek_v3',1.0)])
def test_no_broad_or_boolean_coercion(kind,value):
    raw={'model_type':kind,'routed_scaling_factor':value}
    normalized,changes=normalize_research_config_dict(raw)
    assert normalized==raw and type(normalized['routed_scaling_factor']) is type(value)
    assert changes==[]


def test_other_native_field_errors_are_not_ignored(tmp_path):
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'model_type':'deepseek_v3','routed_scaling_factor':1,'hidden_size':'invalid'}))
    with pytest.raises(Exception):load_research_native_config(path)
