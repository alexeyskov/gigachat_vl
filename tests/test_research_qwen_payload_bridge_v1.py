import copy
from types import SimpleNamespace
import pytest
from src.utils.research_qwen_payload_bridge_v1 import validation_view,MODELS
from src.utils.research_completed_r2_v1 import definition


def sample():
    contract={'derived_vision_schema':'derived','vision_export_ref':{'sha':'exact'},'vision_profile_ref':{'name':'729'},
              'vision_cache_contract':{'source':'exact'},'vision_cache_fingerprint':'fingerprint','derived_vision_code_sha256':{'code':'pin'}}
    model=next(iter(MODELS));policy=definition()
    payload=dict(config=dict(contract,model=model,vision='Qwen/Qwen2.5-VL-7B-Instruct',recipe='4f',stage='SFT',seed=42,
        epochs=1,effective_batch_size=64,completed_r2_execution_policy=policy),model={'untouched':object()},step=56,seen=3529,
        training_completed=True,resumable_optimizer=False,checkpoint_finalization={'owner':policy['owner']},
        model_ref={'repo':model,'revision':MODELS[model]})
    return payload,SimpleNamespace(config_contract=lambda _:contract),contract


def test_metadata_only_view_preserves_tensor_identity_and_saved_payload():
    payload,derived,contract=sample();before=set(payload)
    view,bridged=validation_view(payload,{},derived)
    assert bridged and set(payload)==before and view['model'] is payload['model']
    assert all(view[k]==v for k,v in contract.items())


@pytest.mark.parametrize('change',['config','recipe','incomplete','revision','optimizer'])
def test_incompatible_payloads_fail_closed(change):
    payload,derived,contract=sample()
    if change=='config':payload['config']['vision_cache_fingerprint']='wrong'
    if change=='recipe':payload['config']['recipe']='1b'
    if change=='incomplete':payload['step']=55
    if change=='revision':payload['model_ref']['revision']='0'*40
    if change=='optimizer':payload['resumable_optimizer']=True
    with pytest.raises(ValueError):validation_view(payload,{},derived)


def test_partial_metadata_is_never_repaired_or_overwritten():
    payload,derived,contract=sample();payload['derived_vision_schema']='wrong'
    view,bridged=validation_view(payload,{},derived)
    assert not bridged and view is payload and view['derived_vision_schema']=='wrong'
