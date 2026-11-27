"""Separate, bounded R4/R5 study: caption warmup -> SFT1 -> SFT2 -> DET."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import types

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'datasets/research/budget18-en-v1'
DET = ROOT/'datasets/research/detection-v1'
BASE = ROOT/'runs/research/common_pool_v1'
SOURCES = ('src/model/research_common_pool_v1.py', 'src/utils/research_common_pool_training_v1.py',
           'scripts/run_research_common_pool_v1.py', 'scripts/research_common_pool_queue_v1.py')
PHASES = ('A1', 'SFT1', 'SFT2', 'DET')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(8*2**20): digest.update(chunk)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def definition():
    from src.utils import research_stock_determinism as stock
    from src.utils import research_additive_runner_v1 as core
    from src.utils import research_detection_training_v1 as detection
    return dict(name='common_pool_eight_experts_v1', seed=42, methods=['R4', 'R5'],
        added_experts_per_moe=8, new_expert='native-shape FFN, fully trained FP32 masters',
        top_k='native k unchanged, shared E+8 candidates for every valid token',
        masks='no modality gates or modality eligibility restrictions; padding excluded',
        R4='fresh full trainable FP32 Linear(H,E+8) router, no inherited routing bias',
        R5='frozen native router and native correction bias; rank16 alpha32 LoRA residual over E+8 logits; new base logits zero',
        LLM='frozen NF4 base, rank16 alpha32 dropout0.05 LoRA; frozen during A1',
        vision='frozen Qwen2.5-VL-7B donor vision+merger, native bounded729 BF16 cached features',
        projector='reuse proven caption-aligned MLP for each LLM; frozen thereafter',
        phases=dict(A1=dict(records=256, steps=4, objective='caption supervised-token CE'),
                    SFT1=dict(records=3529, steps=56, objective='0.85MM+0.15text per-stream token CE'),
                    SFT2=dict(records=3529, steps=56, objective='same records, epoch1 shuffle, half LR'),
                    DET=dict(records=3529, steps=56, objective='0.15OD+0.70MM+0.15text per-stream token CE')),
        caption_exploration='A1 only: rotate one guaranteed new candidate per token, use unchanged scores for routing weights; no forcing at SFT/evaluation',
        auxiliary_router_loss=None, sampling='deterministic native top-k, no stochastic routing',
        optimizer='AdamW32 foreach=False; fresh at each phase, no optimizer resume',
        checkpoint_policy='one model-only best/last per phase; consumed phase weights retired after independent successor proof',
        reserve_disk_GiB=32, source_sha256={name:sha(ROOT/name) for name in SOURCES},
        legacy_execution=stock.definition(), additive_dependencies={p:sha(ROOT/p) for p in core.SOURCES},
        detection_execution=detection.definition(), legacy_release_sha256=sha(OLD/'release.json'))


def configure_args(job):
    from src.utils import research_stock_determinism as stock
    from src.utils import research_additive_runner_v1 as core
    stock.bootstrap_environment()
    phase = job['phase']
    if phase not in PHASES or job['method'] not in ('R4', 'R5'):
        raise ValueError('Invalid common-pool phase/method')
    if job['model'] not in core.MODELS.values() or job['vision'] != core.VISIONS['qwen25']:
        raise ValueError('Only the three requested LLMs and Qwen donor are supported')
    raw = stock._clone('scripts/run_research_stage.py')
    manifest = OLD/'alignment_a1_train.jsonl' if phase == 'A1' else (DET if phase == 'DET' else OLD)/'sft_multimodal_train.jsonl'
    flags = ['--model',job['model'],'--vision',job['vision'],'--stage','A1' if phase == 'A1' else 'SFT',
        '--recipe','1b','--manifest',str(manifest),'--validation',str(OLD/('alignment_validation.jsonl' if phase == 'A1' else 'sft_validation.jsonl')),
        '--initial-checkpoint',job['source_checkpoint'],'--output',job['output'],'--seed','42','--epochs','1',
        '--batch-size','64','--effective-batch-size','64','--max-length','4096','--eval-every','10000',
        '--reserve-gb','32','--vision-preprocessing-profile','qwen_native_bounded_729']
    if phase != 'A1':
        flags += ['--text-manifest',str((DET if phase == 'DET' else OLD)/'text_train.jsonl'),
                  '--text-fraction','.15','--text-loss-weight','.15','--selection-metric','group_macro']
    args = raw.parse_args(flags); raw.validate_args(args)
    args.common_pool_execution = definition()
    args.common_method, args.common_phase = job['method'], phase
    args.prompt_schema_version = 2
    from src.utils.research_resource_vision_v3 import derived
    binding = derived.resolve_derived_vision(args.vision, preprocessing_profile='qwen_native_bounded_729')
    derived.configure_args(args, binding)
    return args


def backend(args):
    from src.utils import research_stock_determinism as stock
    from src.utils import research_additive_runner_v1 as core
    from src.model import research_common_pool_v1 as common
    from src.model.research_moe import set_modality_mask
    proxy = types.ModuleType('_common_pool_runner_proxy')
    inference = stock._clone('src/eval/research_inference.py', {'scripts.run_research_stage':proxy})
    selection = stock._clone('src/eval/research_selection.py', {'src.eval.research_inference':inference})
    def install(llm, family, policy, added_experts=2):
        if family != 1 or policy != 'b': raise ValueError('NF4 base recipe must be1b')
        return common.install(llm, args.common_method)
    shim = types.SimpleNamespace(install_research_moe=install, configure_research_trainability=common.configure,
                                 set_modality_mask=set_modality_mask)
    dependencies = {'src.model.research_moe':shim, 'src.eval.research_selection':selection}
    if args.model == core.MODELS['giga31']:
        from src.model.research_giga_loading_v2 import ScopedPEFT, install_loader
        dependencies['peft'] = ScopedPEFT()
    module = stock._clone('scripts/run_research_stage.py', dependencies)
    if args.model == core.MODELS['giga31']: install_loader(module)
    if args.model == core.MODELS['smallthinker4']:
        from src.utils.research_small_router_stability_v1 import private_smallthinker
        module.register_smallthinker = private_smallthinker().register_smallthinker
    from src.utils.research_resource_vision_v3 import derived
    binding = derived.resolve_derived_vision(args.vision, preprocessing_profile='qwen_native_bounded_729')
    module = derived.install_private_runner_hooks(module, binding, model_repo=args.model)
    Native = module.ResearchModel
    class Model(Native):
        def __init__(self, current):
            if current.common_pool_execution != definition(): raise ValueError('R4/R5 source binding changed')
            super().__init__(current)
            # Save frozen initial packed LoRA factors in the same exact dtype
            # used at SFT, avoiding any lossy stage-boundary conversion.
            for name, parameter in self.named_parameters():
                if 'lora_' in name and parameter.is_floating_point(): parameter.data = parameter.data.float()
            from src.utils.research_small_budget_v3 import fixed_clock
            self.tokenizer = fixed_clock(self.tokenizer)
            if args.model == core.MODELS['giga31']:
                from src.eval.research_giga_eot_v1 import install_generation_stop
                install_generation_stop(self)
            audit_trainability(self)
    module.ResearchModel = Model
    proxy.__dict__.update(module.__dict__)
    return module, inference, selection


def audit_trainability(model):
    method, phase = model.args.common_method, model.args.common_phase
    groups = dict(new_experts=0, router=0, llm_lora=0)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        if '.visual_experts.' in name: key='new_experts'
        elif '.router_adapter.' in name and method == 'R5': key='router'
        elif '.primary_router.' in name and method == 'R4': key='router'
        elif 'lora_' in name and phase != 'A1': key='llm_lora'
        else: raise ValueError('Unexpected trainable base/projector tensor: '+name)
        groups[key] += parameter.numel()
        if parameter.dtype != __import__('torch').float32:
            raise ValueError('All learned masters must be FP32')
    if not groups['new_experts'] or not groups['router']: raise ValueError('Missing expert/router parameters')
    if phase != 'A1' and not groups['llm_lora']: raise ValueError('SFT needs language QLoRA')
    return groups


def construct(args, checkpoint, *, recovery=False):
    import torch
    from src.utils import research_stock_determinism as stock
    from src.utils import research_additive_runner_v1 as core
    from src.utils.research_capacity_v4 import validate_cached_projector
    from src.utils.research_detection_training_v1 import source_cache_advice
    stock.activate()
    module, inference, selection = backend(args)
    module.set_seed(42)
    model = module.ResearchModel(args)
    checkpoint = Path(checkpoint).resolve(strict=True)
    source_sha = sha(checkpoint)
    saved = (validate_cached_projector(checkpoint) if args.common_phase == 'A1' and not recovery
             else torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False))
    if args.common_phase == 'A1' and not recovery:
        if set(saved['model']) != set(core.PKEYS): raise ValueError('Only aligned P may initialize fresh R4/R5')
    else:
        config = saved['config']
        expected_phase = args.common_phase if recovery else PHASES[PHASES.index(args.common_phase)-1]
        if (config.get('common_phase') != expected_phase or config.get('common_method') != args.common_method
                or config.get('model') != args.model or config.get('common_pool_execution') != definition()
                or saved['step'] != (4 if expected_phase == 'A1' else 56)
                or saved['seen'] != (256 if expected_phase == 'A1' else 3529)):
            raise ValueError('Same-architecture completed predecessor required')
        required = {n for n,p in model.named_parameters() if p.requires_grad or 'lora_' in n} | set(core.PKEYS)
        if required-set(saved['model']): raise ValueError('Missing learned initialization tensors')
        if not recovery:
            source_run = checkpoint.parents[2]
            proof = read(source_run/'native_reload_proof.json')
            if proof['status'] != 'passed' or proof['checkpoint_sha256'] != source_sha:
                raise ValueError('Independent predecessor proof required')
    for name, actual in (('model_ref',model.model_ref),('vision_ref',model.vision_ref),('cache_fingerprint',model.cache.fingerprint)):
        if saved.get(name) != actual: raise ValueError('Pinned initialization binding differs: '+name)
    target = model.state_dict()
    for name, value in saved['model'].items():
        if name not in target or value.shape != target[name].shape or value.dtype != target[name].dtype:
            raise ValueError('Initialization shape/dtype differs: '+name)
    result = model.load_state_dict(saved['model'], strict=False)
    if result.unexpected_keys: raise ValueError('Unexpected initialization tensors')
    for name, value in saved['model'].items():
        for actual, expected in zip(target[name].reshape(-1).split(2**20),value.reshape(-1).split(2**20),strict=True):
            if not torch.equal(actual.detach().cpu(), expected): raise ValueError('Initialization bytes differ: '+name)
    info = dict(source_sha256=source_sha, learned_tensors_exactly_loaded=len(saved['model']),
                model_ref=model.model_ref, vision_ref=model.vision_ref, cache_fingerprint=model.cache.fingerprint)
    del saved,target,value,actual,expected
    gc.collect()
    source_cache_advice(checkpoint)
    for path in Path(model.model_path).glob('*.safetensors'): source_cache_advice(path)
    return model, module, inference, selection, info


def training_records(phase):
    if phase == 'A1': return rows(OLD/'alignment_a1_train.jsonl')
    base = DET if phase == 'DET' else OLD
    return rows(base/'sft_multimodal_train.jsonl')+rows(base/'text_train.jsonl')


def save_weights(model, args, seen, steps, info):
    import torch
    from src.utils.research_storage import BestLastCheckpointManager, DiskBudget, trainable_state_dict, _atomic_json
    from src.utils.research_budget_storage_v1 import payload_storage_budget
    from src.utils.research_lossless_delta import memory_guard
    run = Path(args.output)
    anticipated = sum(p.numel()*p.element_size() for name,p in model.named_parameters()
                      if p.requires_grad or 'lora_' in name or name.startswith('projector.'))
    memory_guard(anticipated+2*2**30)
    state = trainable_state_dict(model)
    state.update({name:p.detach().cpu().clone() for name,p in model.named_parameters() if 'lora_' in name or name.startswith('projector.')})
    payload = dict(model=state, config=vars(args), step=steps, seen=seen, optimizer=None,
                   resumable_optimizer=False, training_completed=True, initialization=info,
                   model_ref=model.model_ref, vision_ref=model.vision_ref, cache_fingerprint=model.cache.fingerprint,
                   checkpoint_kind='common_pool_model_only_v1')
    manager = BestLastCheckpointManager(run, DiskBudget(run,32), mode='max')
    manager.save_torch(payload,step=steps,metric=None,anticipated_bytes=payload_storage_budget(payload)['anticipated_bytes'])
    _atomic_json(run/'training_completed.json',dict(phase=args.common_phase,steps=steps,records=seen,
                                                  checkpoint_sha256=sha(run/'last/state.pt')))


def evaluate(model, module, inference, selection, args):
    import torch
    from src.utils.research_storage import _atomic_json,BestLastCheckpointManager,DiskBudget
    from src.utils import research_detection_training_v1 as detection
    run = Path(args.output)
    def progress(value): _atomic_json(run/'evaluation_progress.json',value)
    if args.common_phase == 'A1':
        records = rows(OLD/'alignment_validation.jsonl')[:8]
        predictions = []
        with selection.selection_generation_mode(model):
            for record in records:
                public = {k:record[k] for k in ('record_id','question','image_paths','image_hashes')}
                value = inference.generate_prediction(model,public,max_new_tokens=64)
                predictions.append(dict(record_id=record['record_id'],prediction=value['prediction'],generation=value))
        detection.store_evaluation(run,'caption_probe',dict(metrics=dict(records=8,scope='caption generation replay probe, not CIDEr'),predictions=predictions))
        score = 0.
    else:
        selected = module.prepare_group_macro_selection(args)
        if len(selected['records']) != 116: raise ValueError('Exactly116 internal VQA records required')
        result = selection.evaluate_internal_group_macro(model,selected,max_new_tokens=64,progress=progress)
        detection.store_evaluation(run,'after_internal116',result)
        score = result['metrics']['group_macro_score']
        if args.common_phase == 'DET':
            result = detection.evaluate_detection(model,inference,selection,rows(DET/'detection_validation.jsonl'),progress=progress)
            detection.store_evaluation(run,'after_detection_validation',result)
    manager = BestLastCheckpointManager(run,DiskBudget(run,32),mode='max')
    source = (run/'last/state.pt').resolve(strict=True)
    done = read(run/'training_completed.json')
    manager.save(lambda directory:os.link(source,directory/'state.pt'),step=done['steps'],metric=score,anticipated_bytes=0,
                 metadata=dict(evaluation='passed',phase=args.common_phase))
    _atomic_json(run/'completed.json',dict(**done,internal116_score=score if args.common_phase!='A1' else None))


def train(job, *, evaluate_existing=False, probe=False):
    import torch
    from src.utils import research_detection_training_v1 as detection
    from src.utils.research_storage import DiskBudget,_atomic_json
    detection.validate_release()
    args = configure_args(job); run=Path(args.output); run.mkdir(parents=True,exist_ok=True)
    config = run/'resolved_config.json'
    if config.exists() and read(config) != vars(args): raise ValueError('Existing R4/R5 configuration differs')
    _atomic_json(config,vars(args))
    if (run/'last/state.pt').exists() and not evaluate_existing: raise ValueError('Use evaluation-only recovery for saved final weights')
    model,module,inference,selection,info = construct(args,run/'last/state.pt' if evaluate_existing else args.initial_checkpoint,recovery=evaluate_existing)
    _atomic_json(run/'initialization.json' if not evaluate_existing else run/'recovery_initialization.json',info)
    if evaluate_existing: return evaluate(model,module,inference,selection,args)
    groups = audit_trainability(model)
    _atomic_json(run/'trainability_audit.json',dict(status='passed',parameter_groups=groups,
        phase=args.common_phase,method=args.common_method,trainable_base_parameters=0,
        trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}))
    params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    anticipated = 4*(sum(groups.values())+sum(p.numel() for p in model.projector.parameters()))+256*2**20
    DiskBudget(run,32).ensure_space(anticipated)
    lr = 1e-4 if args.common_phase in ('A1','SFT1') else 5e-5
    optimizer = torch.optim.AdamW([dict(params=[p for _,p in params],lr=lr,weight_decay=.01)],foreach=False)
    records = training_records(args.common_phase)
    count = 256 if args.common_phase == 'A1' else 3529
    total = 4 if args.common_phase == 'A1' else 56
    if len(records) != count: raise ValueError('Fixed training budget differs')
    random.Random(43 if args.common_phase=='SFT2' else 42).shuffle(records)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/max(1,int(total*.03)),1)*.5*(1+math.cos(math.pi*min(s,total)/total)))
    micro,seen = args.batch_size,0
    start=time.monotonic()
    def log(value): module.log(run/'metrics.jsonl',value)
    log(dict(event='ready',phase=args.common_phase,method=args.common_method,total_steps=total,records=count,
             trainable_groups=groups,encoder_cache_fingerprint=model.cache.fingerprint,**DiskBudget(run,32).report()))
    for step,offset in enumerate(range(0,len(records),64),1):
        DiskBudget(run,32).ensure_space(anticipated)
        batch=records[offset:offset+64]
        model.train();model.extractor.vision_tower.eval()
        prepared=model.prepare(batch); t0=time.monotonic()
        if args.common_phase == 'DET':
            kinds=['detection' if row.get('task_group')=='object_detection' else 'multimodal_replay' if row.get('image_paths') else 'text' for row in batch]
            result=detection.train_group(model,optimizer,prepared,kinds,micro)
            objective=result['loss'];micro=result['microbatch']
        else:
            result=module.train_accumulation_group(model,optimizer,[prepared],[p for _,p in params],micro,
                text_loss_weight=None if args.common_phase=='A1' else .15)
            objective=result['objective_value'];micro=result['microbatch_size']
        if not math.isfinite(objective): raise FloatingPointError('Nonfinite common-pool objective')
        norm=torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step();torch.cuda.synchronize()
        seen+=len(batch)
        log(dict(event='train',step=step,total_steps=total,records_seen=seen,loss=objective,
                 grad_norm=float(norm),microbatch=micro,step_seconds=time.monotonic()-t0,
                 elapsed_seconds=time.monotonic()-start,peak_vram_GiB=torch.cuda.max_memory_allocated()/2**30))
        if step==1 or step%8==0 or step==total:
            _atomic_json(run/'routing_diagnostics.json',dict(scope='last training forward/recompute only, not epoch aggregate',
                phase=args.common_phase,step=step,layers=[w.diagnostics() for w in model.wrappers]))
        if probe: return dict(status='actual_GPU_forward_backward_optimizer_passed',loss=objective,groups=groups,step_seconds=time.monotonic()-t0)
    if seen!=count or step!=total: raise ValueError('Incomplete common-pool phase')
    optimizer.state.clear();model.zero_grad(set_to_none=True)
    del optimizer,scheduler,params,prepared
    gc.collect();torch.cuda.empty_cache()
    save_weights(model,args,seen,step,info)
    log(dict(event='trained_weights_saved_before_evaluation',step=step,records=seen))
    return evaluate(model,module,inference,selection,args)


def validate(job):
    from src.utils import research_detection_training_v1 as detection
    from src.utils.research_storage import _atomic_json
    args=configure_args(job);run=Path(args.output)
    if read(run/'resolved_config.json') != vars(args): raise ValueError('Reload configuration differs')
    model,module,inference,selection,info=construct(args,run/'last/state.pt',recovery=True)
    name='caption_probe' if args.common_phase=='A1' else 'after_internal116'
    old={r['record_id']:r for r in rows(run/'evaluation'/(name+'.predictions.jsonl'))}
    records=(rows(OLD/'alignment_validation.jsonl')[:8] if args.common_phase=='A1'
             else module.prepare_group_macro_selection(args)['records'][::14][:8])
    mismatches=[]
    with selection.selection_generation_mode(model):
        for record in records:
            public={k:record[k] for k in ('record_id','question','image_paths','image_hashes')}
            actual=inference.generate_prediction(model,public,max_new_tokens=64)
            expected=old[record['record_id']]
            for key in ('prediction','generated_token_ids','prompt_sha256'):
                value=expected['prediction'] if key=='prediction' else expected['generation'][key]
                if actual[key] != value: mismatches.append(dict(id=record['record_id'],field=key))
    od_replayed=0
    if args.common_phase=='DET':
        old={r['record_id']:r for r in rows(run/'evaluation/after_detection_validation.predictions.jsonl')}
        source=rows(DET/'detection_validation.jsonl')
        subset=[r for kind in ('class_all','spatial','all80','negative') for r in [s for s in source if s['detection']['query_kind']==kind][:2]]
        replay=detection.evaluate_detection(model,inference,selection,subset)
        od_replayed=len(subset)
        for actual in replay['predictions']:
            expected=old[actual['record_id']]
            if actual['prediction']!=expected['prediction'] or actual['generation']['generated_token_ids']!=expected['generation']['generated_token_ids']:
                mismatches.append(dict(id=actual['record_id'],field='detection_generation'))
        test=detection.evaluate_detection(model,inference,selection,rows(DET/'detection_test.jsonl'),
             progress=lambda value:_atomic_json(run/'test_progress.json',value))
        detection.store_evaluation(run,'heldout_detection_test',test)
    report=dict(status='passed' if not mismatches else 'failed',checkpoint_sha256=info['source_sha256'],
                learned_tensors_exactly_restored=info['learned_tensors_exactly_loaded'],replayed_records=len(records),
                detection_validation_replayed=od_replayed,heldout_detection_test_records=64 if args.common_phase=='DET' else 0,
                mismatches=mismatches,execution=definition())
    _atomic_json(run/'native_reload_proof.json',report)
    if mismatches: raise ValueError('Independent common-pool replay differs')
    return report
