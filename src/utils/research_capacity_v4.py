"""Fixed4 capacity study and bounded one-pass SFT continuations.

Uses the frozen model/feature/loss implementation. Separate run IDs and an
execution receipt distinguish block count, data pass and fresh optimizer.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

from src.utils import research_additive_runner_v1 as core
ROOT=core.ROOT
SOURCES=('src/utils/research_capacity_v4.py','scripts/run_research_capacity_v4.py',
         'src/utils/research_additive_init_dtype_v1.py')


def execution(branch,training_pass):
    return dict(name='fixed_capacity_training_v4',branch=dict(branch),training_pass=training_pass,
        maximum_SFT_passes=2,optimizer='fresh AdamW per pass; completed predecessors have no optimizer state',
        lr_scale=1.0 if training_pass==1 else .5,data_epoch_offset=training_pass-1,
        source_sha256={p:core.sha(ROOT/p) for p in SOURCES},data_release_sha256=core.RELEASE_SHA)


def validate_cached_projector(path):
    import torch
    from src.utils.research_small_budget_v3 import require_complete_P
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('kind')!='alignment_projector_reuse_v1' or saved.get('data_release_sha256')!=core.RELEASE_SHA:
        raise ValueError('Explicit retained caption-projector artifact required')
    require_complete_P(saved['model'])
    if set(saved['model'])!=set(core.PKEYS):raise ValueError('Projector artifact must contain exactly four P tensors')
    for name,tensor in saved['model'].items():
        if core._tensor_sha(tensor)!=saved['origin']['projector_sha256'][name]:raise ValueError('Retained projector changed')
    return saved


def validate_continuation(payload,model,args,inference):
    old=payload.get('config',{})
    if (args.stage!='SFT' or args.training_pass!=2 or old.get('stage')!='SFT'
        or old.get('training_pass',1)!=1 or payload.get('step')!=56 or payload.get('seen')!=3529
        or old.get('model')!=args.model or old.get('vision')!=args.vision or old.get('recipe')!=args.recipe
        or old.get('seed')!=args.seed or old.get('additive_execution')!=args.additive_execution
        or payload.get('manifest_sha256')!=core.sha(args.manifest)
        or payload.get('training_completed') is not True or payload.get('resumable_optimizer') is not False):
        raise ValueError('Continuation requires a same-architecture completed first SFT pass')
    checkpoint=Path(args.initial_checkpoint).resolve(strict=True)
    runs=[p for p in checkpoint.parents if (p/'resolved_config.json').is_file()]
    if not runs:raise ValueError('Owned continuation source run required')
    run=runs[0];proof=json.loads((run/'native_reload_proof.json').read_text())
    if proof.get('status')!='passed' or proof.get('checkpoint_sha256')!=core.sha(checkpoint):
        raise ValueError('Continuation requires independent GPU-verified source weights')
    for field,value in [('model_ref',model.model_ref),('vision_ref',model.vision_ref),('cache_fingerprint',model.cache.fingerprint)]:
        if payload.get(field)!=value:raise ValueError('Continuation source binding differs: '+field)
    inference._validate_learned_state(model,payload['model'])
    expected=model.state_dict()
    if set(payload['model'])-set(expected):raise ValueError('Unknown continuation tensors')
    if any(t.dtype!=expected[n].dtype for n,t in payload['model'].items()):raise ValueError('Continuation requires exact saved/target dtypes')
    return True


def _offset_training_order(module):
    tree=ast.parse(textwrap.dedent(inspect.getsource(module.main)))
    class Offset(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='random' and node.func.attr=='Random' and len(node.args)==1
                and ast.unparse(node.args[0])=='args.seed + epoch'):
                self.count+=1
                node.args[0]=ast.BinOp(left=node.args[0],op=ast.Add(),right=ast.Attribute(value=ast.Name('args',ast.Load()),attr='data_epoch_offset',ctx=ast.Load()))
            return self.generic_visit(node)
    change=Offset();tree=change.visit(tree)
    if change.count!=1:raise ValueError('Exact native training-order seed hook required')
    exec(compile(ast.fix_missing_locations(tree),str(ROOT/'scripts/run_research_stage.py'),'exec'),module.__dict__)


def multimodal_ce(model,records,selection,batch_size=4):
    """Teacher-forced heldout CE, separate from the generation metric."""
    import torch
    rows=[dict(r,answer=r['teacher_reference']) for r in records]
    if any(not r.get('image_paths') for r in rows):raise ValueError('Multimodal CE needs image-bearing heldout records')
    loss=0.;tokens=0
    with selection.selection_generation_mode(model),torch.no_grad():
        for i in range(0,len(rows),batch_size):
            value,count,_=model.forward_loss(model.prepare(rows[i:i+batch_size]))
            loss+=float(value);tokens+=count
    if not tokens:raise ValueError('Empty multimodal heldout labels')
    return dict(ce=loss/tokens,supervised_tokens=tokens,records=len(rows))


def runner(args,branch):
    from src.utils.research_additive_init_dtype_v1 import patched_runner
    # This factory adds only the exact packed A1 BF16->SFT FP32 read bridge.
    with patched_runner():module,inference,selection=core.runner(branch,model_repo=args.model,vision_repo=args.vision)
    if args.training_pass==2:
        module.validate_initial_checkpoint=lambda payload,model,current:validate_continuation(payload,model,current,inference)
    import torch
    class Optim:
        def __getattr__(self,name):return getattr(torch.optim,name)
        def AdamW(self,groups,*a,**k):
            scaled=[dict(g,lr=g['lr']*args.lr_scale) for g in groups]
            return torch.optim.AdamW(scaled,*a,**k)
    class Torch:
        optim=Optim()
        def __getattr__(self,name):return getattr(torch,name)
    module.torch=Torch()  # private namespace only; the global Torch module stays intact
    _offset_training_order(module)
    native_eval=selection.evaluate_internal_group_macro
    def evaluate(model,records_or_selection,**kwargs):
        result=native_eval(model,records_or_selection,**kwargs)
        records=records_or_selection['records']
        ce=multimodal_ce(model,records,selection)
        from src.utils.research_storage import _atomic_json
        _atomic_json(Path(args.output)/'multimodal_validation_ce.json',dict(**ce,training_pass=args.training_pass))
        module.log(Path(args.output)/'metrics.jsonl',dict(event='multimodal_validation_ce',**ce,training_pass=args.training_pass))
        return result
    selection.evaluate_internal_group_macro=evaluate
    return module


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--visual-blocks',type=int,choices=[2,4],required=True)
    parser.add_argument('--training-pass',type=int,choices=[1,2],default=1)
    parser.add_argument('--learned-masks',action='store_true')
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--allow-gpu',action='store_true')
    opts,remaining=parser.parse_known_args(argv)
    from src.utils import research_stock_determinism as stock
    stock.bootstrap_environment()
    from src.model.research_additive_visual_v1 import VisualBranchConfig
    raw=stock._clone('scripts/run_research_stage.py');args=raw.parse_args(remaining);raw.validate_args(args)
    if args.model not in core.MODELS.values() or args.vision not in core.VISIONS.values():raise ValueError('Model/encoder outside active matrix')
    if args.stage not in ('A1','SFT') or args.recipe!=('1c' if opts.learned_masks else '1b'):raise ValueError('Fixed QLoRA R1/R3 only')
    if args.stage=='A1' and opts.training_pass!=1:raise ValueError('Only one fresh A1 for each architecture')
    for key,value in dict(seed=42,epochs=1,effective_batch_size=64,max_length=4096,eval_every=10000,
                          max_records=None,max_steps=None,reserve_gb=32.,optimizer_backend='torch_adamw32').items():
        if getattr(args,key)!=value:raise ValueError('Bounded capacity budget differs: '+key)
    if args.resume or args.alignment_checkpoint or not args.initial_checkpoint:raise ValueError('Explicit weights initialization, no optimizer resume')
    proof=json.loads((core.DATA/'release.json').read_text())
    if core.sha(core.DATA/'release.json')!=core.RELEASE_SHA or proof['status']!='passed':raise ValueError('Frozen data release changed')
    required={'manifest':'alignment_a1_train.jsonl' if args.stage=='A1' else 'sft_multimodal_train.jsonl',
              'validation':'alignment_validation.jsonl' if args.stage=='A1' else 'sft_validation.jsonl'}
    if args.stage=='SFT':required['text_manifest']='text_train.jsonl'
    for flag,name in required.items():
        path=Path(getattr(args,flag)).resolve(strict=True)
        if path!=core.DATA/name or core.sha(path)!=proof['files'][name]['sha256']:raise ValueError('Frozen data input differs')
    if args.stage=='SFT' and (args.text_fraction!=.15 or args.text_loss_weight!=.15 or args.selection_metric!='group_macro'):
        raise ValueError('Same85/15 objective and internal116 required')
    branch=asdict(VisualBranchConfig(mode='fixed',blocks=opts.visual_blocks,learned_masks=opts.learned_masks))
    args.additive_execution=core.definition(branch)
    args.training_pass=opts.training_pass;args.data_epoch_offset=opts.training_pass-1
    args.lr_scale=1. if opts.training_pass==1 else .5
    args.capacity_execution=execution(branch,opts.training_pass)
    if args.vision==core.VISIONS['qwen25']:
        from src.utils.research_resource_vision_v3 import derived
        binding=derived.resolve_derived_vision(args.vision,preprocessing_profile=args.vision_preprocessing_profile)
        derived.configure_args(args,binding)
    if args.stage=='A1':validate_cached_projector(args.initial_checkpoint)
    if opts.preflight:
        print(json.dumps(dict(status='CPU_preflight_only',config=vars(args)),sort_keys=True));return vars(args)
    if not opts.allow_gpu:raise ValueError('Actual capacity training requires --allow-gpu')
    stock.activate();module=runner(args,branch);module.parse_args=lambda:args
    return module.main()
