"""Separate execution contract for additive visual branches, never legacy resume."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'datasets/research/budget18-en-v1'
RELEASE_SHA = '910448c7fd680ec949d6521e32716aee2832a99fc534d185f0a06b226c0ec1c0'
MODELS = {'smallthinker4':'Tiiny/SmallThinker-4BA0.6B-Instruct',
          'lfm25':'LiquidAI/LFM2.5-8B-A1B', 'giga31':'ai-sage/GigaChat3.1-10B-A1.8B-bf16'}
VISIONS = {'siglip2':'google/siglip2-so400m-patch14-384', 'qwen25':'Qwen/Qwen2.5-VL-7B-Instruct'}
PKEYS = ('projector.0.weight','projector.0.bias','projector.2.weight','projector.2.bias')
SOURCES = ('src/model/research_additive_visual_v1.py', 'src/utils/research_additive_runner_v1.py',
           'scripts/run_research_additive_v1.py', 'src/eval/research_giga_eot_v1.py',
           'src/model/research_giga_loading_v2.py', 'src/model/research_giga_loading_v1.py',
           'src/model/research_giga_peft.py', 'src/utils/research_resource_vision_v3.py',
           'src/utils/research_small_router_stability_v1.py',
           'research_results/model_inventory.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*2**20), b''):h.update(chunk)
    return h.hexdigest()


def definition(config):
    from src.utils import research_stock_determinism as stock
    from src.utils.research_resource_vision_v3 import derived
    return {'name':'additive_visual_transformer_v1', 'branch':dict(config),
            'source_sha256':{name:sha(ROOT/name) for name in SOURCES},
            'native_source_contract':stock.definition(), 'derived_source_contract':derived.code_binding(),
            'seed':42, 'base':'frozen NF4 + LoRA r16 alpha32 dropout0.05',
            'native_routing':'original module, unchanged expert set, native k and native weights',
            'small_normalized_sigmoid':'native ordinary rows; equivalent log-sigmoid fallback only for a zero denominator',
            'extra_routing':'separate top-k over new blocks only; fixed mode all blocks',
            'activation':'valid visual positions and subsequent text; text-only request bypass',
            'checkpoint_policy':'one completed model-only object shared by best/last, no optimizer resume',
            'caption_A1_records':256, 'SFT_records':3529, 'disk_floor_GiB':32}


def _tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def extract_projector(backend, vision, destination):
    """Reuse proven frozen A0 projector, without relabelling an SFT as an A0."""
    import fcntl
    import torch
    from src.utils.research_storage import DiskBudget
    source = ROOT/f'runs/research/{backend}__{vision}__1b__SFT__budget18_s42'
    audit = ROOT/f'research_results/budget18_parameter_updates_v1/{source.name}'
    report = json.loads((audit/'report.json').read_text())
    initial = json.loads((audit/'initial_tensors.json').read_text())
    config = json.loads((source/'resolved_config.json').read_text())
    if (report.get('status') != 'passed' or report['common_tensor_groups']['projector']['changed_tensors'] != 0
        or config['model'] != MODELS[backend] or config['vision'] != VISIONS[vision]
        or config['stage'] != 'SFT' or config['recipe'] != '1b'):
        raise ValueError('Proven frozen alignment projector required')
    with (source/'.checkpoint.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        checkpoint = (source/'last/state.pt').resolve(strict=True)
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        if saved['step'] != 56 or saved['seen'] != 3529 or saved['config'] != config:
            raise ValueError('Completed frozen source checkpoint differs')
        state = {name:saved['model'][name].clone() for name in PKEYS}
        for name,tensor in state.items():
            if _tensor_sha(tensor) != initial['source_tensors'][name]['sha256']:
                raise ValueError('Projector differs from immutable pre-SFT tensor census')
        origin = {'kind':'alignment_projector_reuse_v1', 'source_run':str(source),
                  'source_checkpoint':str(checkpoint), 'source_checkpoint_sha256':sha(checkpoint),
                  'source_config_sha256':sha(source/'resolved_config.json'),
                  'audit_sha256':sha(audit/'report.json'), 'initial_census_sha256':sha(audit/'initial_tensors.json'),
                  'projector_sha256':{k:_tensor_sha(v) for k,v in state.items()},
                  'scope':'P unchanged across A1/SFT; reuse caption-aligned P only; not a new A0 training result'}
        payload = {'kind':'alignment_projector_reuse_v1','model':state,'origin':origin,
                   'model_ref':saved['model_ref'],'vision_ref':saved['vision_ref'],
                   'cache_fingerprint':saved['cache_fingerprint'], 'data_release_sha256':RELEASE_SHA}
        del saved
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    DiskBudget(destination.parent,32).ensure_space(sum(t.numel()*t.element_size() for t in state.values())+2**20)
    if destination.exists():
        existing = torch.load(destination, map_location='cpu', weights_only=False)
        if existing['origin'] != origin:raise ValueError('Existing reused projector provenance differs')
        for name in PKEYS:
            if not torch.equal(existing['model'][name], state[name]):raise ValueError('Existing projector changed')
    else:
        temporary=destination.with_suffix('.tmp')
        with temporary.open('xb') as f:
            torch.save(payload,f);f.flush()
            import os
            os.fsync(f.fileno())
        temporary.rename(destination)
    return origin


def runner(branch_config, *, model_repo, vision_repo):
    from src.utils import research_stock_determinism as stock
    stock.bootstrap_environment()
    from src.model import research_additive_visual_v1 as additive
    from src.model.research_additive_visual_v1 import VisualBranchConfig
    branch = VisualBranchConfig(**branch_config)
    expected = definition(asdict(branch))
    proxy = types.ModuleType('_additive_runner_proxy')
    inference = stock._clone('src/eval/research_inference.py',
        {'src.model.research_moe':additive, 'scripts.run_research_stage':proxy})
    selection = stock._clone('src/eval/research_selection.py', {'src.eval.research_inference':inference})
    def install(llm, family, policy, added_experts=2):
        if family != 1 or policy != ('c' if branch.learned_masks else 'b'):
            raise ValueError('Additive R1/R3 QLoRA recipe mismatch')
        return additive.install_additive_visual(llm, branch)
    def configure(llm, stage):
        for wrapper in additive.iter_additive(llm):
            wrapper.visual_experts.requires_grad_(stage in ('A1','SFT'))
            wrapper.visual_experts.warmup_force_new = stage == 'A1'
    shim = types.SimpleNamespace(install_research_moe=install,
        configure_research_trainability=configure, set_modality_mask=additive.set_modality_mask)
    dependencies = {'src.model.research_moe':shim, 'src.eval.research_selection':selection}
    if model_repo == MODELS['giga31']:
        from src.model.research_giga_loading_v2 import ScopedPEFT, install_loader
        dependencies['peft'] = ScopedPEFT()
    module = stock._clone('scripts/run_research_stage.py', dependencies)
    if model_repo == MODELS['smallthinker4']:
        from src.utils.research_small_router_stability_v1 import private_smallthinker
        module.register_smallthinker = private_smallthinker().register_smallthinker
    if model_repo == MODELS['giga31']:install_loader(module)
    if vision_repo == VISIONS['qwen25']:
        from src.utils.research_resource_vision_v3 import derived
        binding = derived.resolve_derived_vision(vision_repo, preprocessing_profile='qwen_native_bounded_729')
        module = derived.install_private_runner_hooks(module,binding,model_repo=model_repo)
    Native = module.ResearchModel
    class Model(Native):
        def __init__(self,args):
            if args.additive_execution != expected or args.model != model_repo or args.vision != vision_repo:
                raise ValueError('Changed additive architecture/source/model contract')
            super().__init__(args)
            from src.utils.research_small_budget_v3 import fixed_clock
            self.tokenizer = fixed_clock(self.tokenizer)
            # Each native generation call owns and finally clears extra caches.
            native_generate = self.base.generate
            def generate(*a,**k):
                if k.get('use_cache') is False or k.get('num_beams',1) != 1:
                    raise ValueError('Additive generation requires native cache and greedy batch-stable decoding')
                with additive.generation(self.base):return native_generate(*a,**k)
            self.base.generate = generate
            if model_repo == MODELS['giga31']:
                from src.eval.research_giga_eot_v1 import install_generation_stop
                install_generation_stop(self)
    module.ResearchModel = Model
    native_state = module.trainable_state_dict
    def state(model):
        value = native_state(model)
        # A1 also records initial LoRA factors, so SFT does not rely on random
        # constructor consumption order to recreate their frozen values.
        value.update({n:p.detach().cpu() for n,p in model.named_parameters() if 'lora_' in n})
        return value
    module.trainable_state_dict = state
    def initial(payload,model,args):
        if args.stage == 'A1':
            if payload.get('kind') != 'alignment_projector_reuse_v1' or set(payload['model']) != set(PKEYS):
                raise ValueError('A1 accepts only explicitly proven caption-aligned projector reuse')
            if payload.get('data_release_sha256') != RELEASE_SHA:raise ValueError('Projector data release differs')
            for name,tensor in payload['model'].items():
                if _tensor_sha(tensor) != payload['origin']['projector_sha256'][name]:raise ValueError('Reused projector changed')
        else:
            old = payload['config']
            if (old.get('additive_execution') != expected or old.get('stage') != 'A1'
                or old.get('recipe') != args.recipe or payload.get('step') != 4 or payload.get('seen') != 256):
                raise ValueError('SFT requires same-architecture completed additive A1')
            required = {n for n,p in model.named_parameters() if p.requires_grad} | set(PKEYS)
            if required-set(payload['model']):raise ValueError('A1 missing learned branch or initial LoRA factors')
        # Binding a projector reuse checks original refs/cache, rather than
        # pretending it contains the derived full-checkpoint metadata header.
        for field,actual in [('model_ref',model.model_ref),('vision_ref',model.vision_ref),('cache_fingerprint',model.cache.fingerprint)]:
            if payload.get(field) != actual:raise ValueError('Initialization base/vision/cache differs: '+field)
        expected_state = model.state_dict()
        for name,tensor in payload['model'].items():
            target = expected_state.get(name)
            if target is None or target.shape != tensor.shape or target.dtype != tensor.dtype:
                raise ValueError('Initialization tensor shape/dtype differs: '+name)
            if not tensor.is_floating_point() or not __import__('torch').isfinite(tensor).all():
                raise ValueError('Initialization tensors must be finite floating point')
    module.validate_initial_checkpoint = initial
    # Model-only final checkpoints avoid retaining Adam moments for 24 cells.
    NativeManager = module.BestLastCheckpointManager
    class Manager(NativeManager):
        def save_torch(self,payload,*a,**kw):
            import copy
            from src.utils.research_budget_storage_v1 import payload_storage_budget
            if payload['step'] != {'A1':4,'SFT':56}[payload['config']['stage']]:
                raise ValueError('This series retains completed final checkpoints only')
            payload = dict(payload, optimizer=None, resumable_optimizer=False, training_completed=True)
            kw['anticipated_bytes'] = payload_storage_budget(payload)['anticipated_bytes']
            return super().save_torch(payload,*a,**kw)
    module.BestLastCheckpointManager = Manager
    proxy.__dict__.update(module.__dict__)
    return module, inference, selection


def load_checkpoint(checkpoint):
    import torch
    from types import SimpleNamespace
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    args=SimpleNamespace(**saved['config'])
    branch=args.additive_execution['branch']
    if args.additive_execution!=definition(branch):raise ValueError('Saved additive source/architecture contract differs')
    module,inference,selection=runner(branch,model_repo=args.model,vision_repo=args.vision)
    model=module.ResearchModel(args)
    for field,actual in [('model_ref',model.model_ref),('vision_ref',model.vision_ref),('cache_fingerprint',model.cache.fingerprint)]:
        if saved.get(field)!=actual:raise ValueError('Reload pinned source differs: '+field)
    inference._validate_learned_state(model,saved['model'])
    restored=model.load_state_dict(saved['model'],strict=False)
    if restored.unexpected_keys:raise ValueError('Unknown additive learned state')
    actual=model.state_dict()
    for name,tensor in saved['model'].items():
        if not torch.equal(actual[name].detach().cpu(),tensor):raise ValueError('Learned tensor reload differs: '+name)
    inference.freeze_for_generation(model)
    return model,module,inference,selection,saved


def main(argv=None):
    extras=argparse.ArgumentParser(add_help=False)
    extras.add_argument('--visual-branch',choices=['fixed','routed'],required=True)
    extras.add_argument('--learned-masks',action='store_true')
    extras.add_argument('--preflight',action='store_true')
    extras.add_argument('--allow-gpu',action='store_true')
    options,remaining=extras.parse_known_args(argv)
    from src.utils import research_stock_determinism as stock
    stock.bootstrap_environment()
    from src.model.research_additive_visual_v1 import VisualBranchConfig
    from dataclasses import asdict
    branch=VisualBranchConfig(mode=options.visual_branch,blocks=2 if options.visual_branch=='fixed' else 8,
                              learned_masks=options.learned_masks)
    # Parse with the pinned native parser before constructing an encoder/LLM.
    raw=stock._clone('scripts/run_research_stage.py')
    args=raw.parse_args(remaining);raw.validate_args(args)
    if args.model not in MODELS.values() or args.vision not in VISIONS.values():raise ValueError('Model outside requested matrix')
    if args.stage not in ('A1','SFT') or args.recipe != ('1c' if options.learned_masks else '1b'):
        raise ValueError('Use caption projector reuse -> A1 -> SFT with R1/R3 QLoRA')
    fixed={'seed':42,'epochs':1,'effective_batch_size':64,'max_length':4096,'eval_every':10000,
           'max_records':None,'max_steps':None,'reserve_gb':32.,'optimizer_backend':'torch_adamw32'}
    for key,value in fixed.items():
        if getattr(args,key) != value:raise ValueError('Reduced additive budget mismatch: '+key)
    if args.resume:raise ValueError('Completed model-only checkpoints cannot resume an optimizer; rerun incomplete stages explicitly')
    if not args.initial_checkpoint or args.alignment_checkpoint:raise ValueError('Explicit initial checkpoint required')
    proof=json.loads((DATA/'release.json').read_text())
    if sha(DATA/'release.json') != RELEASE_SHA or proof['status'] != 'passed':raise ValueError('Frozen data release differs')
    name='alignment_a1_train.jsonl' if args.stage=='A1' else 'sft_multimodal_train.jsonl'
    required={'manifest':name,'validation':'alignment_validation.jsonl' if args.stage=='A1' else 'sft_validation.jsonl'}
    if args.stage=='SFT':required.update(text_manifest='text_train.jsonl')
    if args.text_validation is not None:raise ValueError('Text heldout records are already in the frozen SFT validation manifest')
    inputs={}
    for flag,name in required.items():
        p=Path(getattr(args,flag)).resolve(strict=True)
        if p != DATA/name or sha(p) != proof['files'][name]['sha256']:raise ValueError('Frozen data input differs: '+flag)
        inputs[flag]=dict(proof['files'][name])
    if args.stage=='SFT' and (args.text_fraction != .15 or args.text_loss_weight != .15 or args.selection_metric != 'group_macro'):
        raise ValueError('SFT must retain stream-weighted85/15 objective and internal116 selection')
    args.additive_execution=definition(asdict(branch));args.additive_inputs=inputs
    if args.vision == VISIONS['qwen25']:
        from src.utils.research_resource_vision_v3 import derived
        binding=derived.resolve_derived_vision(args.vision,preprocessing_profile=args.vision_preprocessing_profile)
        derived.configure_args(args,binding)
    if options.preflight:
        print(json.dumps({'status':'CPU_preflight_only','config':vars(args)},sort_keys=True));return
    if not options.allow_gpu:raise ValueError('Actual training needs --allow-gpu')
    stock.activate()
    module,_,_=runner(asdict(branch),model_repo=args.model,vision_repo=args.vision)
    module.parse_args=lambda:args
    return module.main()
