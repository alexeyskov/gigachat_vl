from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts.run_research_stage import validate_initial_checkpoint,validate_model_binding,validate_resume_config,validate_args


def model():
    return SimpleNamespace(model_ref={'repo':'llm','revision':'123'},vision_ref={'repo':'vision','revision':'456'},cache=SimpleNamespace(fingerprint='features-v1'))


def checkpoint():
    m=model()
    return {'model_ref':m.model_ref,'vision_ref':m.vision_ref,'cache_fingerprint':m.cache.fingerprint,'config':{'stage':'A0','recipe':'4f'}}


def test_checkpoint_binding_checks_frozen_bases_and_feature_version():
    for key in ['model_ref','vision_ref','cache_fingerprint']:
        payload=deepcopy(checkpoint());payload[key]={'repo':'other'} if key.endswith('_ref') else 'features-v2'
        with pytest.raises(ValueError):validate_model_binding(payload,model())
    validate_model_binding(checkpoint(),model())


def test_shared_a0_allowed_but_wrong_precision_and_recipe_rejected():
    validate_initial_checkpoint(checkpoint(),model(),SimpleNamespace(stage='A1',recipe='3c'))
    with pytest.raises(ValueError,match='quantization'):
        validate_initial_checkpoint(checkpoint(),model(),SimpleNamespace(stage='A1',recipe='1c'))
    payload=checkpoint();payload['config']={'stage':'A1','recipe':'3b'}
    with pytest.raises(ValueError,match='recipe'):
        validate_initial_checkpoint(payload,model(),SimpleNamespace(stage='SFT',recipe='3c'))


def test_resume_legacy_torch_optimizer_default_only():
    validate_resume_config({'model':'same','resume':False},{'model':'same','resume':True,'optimizer_backend':'torch_adamw32'})
    with pytest.raises(ValueError):validate_resume_config({'model':'same'},{'model':'same','optimizer_backend':'cpu_grad_bnb_paged_adamw8'})
    with pytest.raises(ValueError):validate_resume_config({'eval_records':64},{'eval_records':16})


def test_effective_batch_cannot_silently_round_up():
    args=SimpleNamespace(batch_size=3,effective_batch_size=64,max_length=100,epochs=1,eval_records=1,eval_every=1,max_records=None,max_steps=None,text_fraction=.15,initial_checkpoint=None,alignment_checkpoint=None)
    with pytest.raises(ValueError,match='multiple'):validate_args(args)
