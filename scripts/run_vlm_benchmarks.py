#!/usr/bin/env python3
"""Run paper-selected VLM benchmarks using pinned VLMEvalKit prompts/scorers."""
from __future__ import annotations
import argparse,importlib.metadata,json,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.vlm_benchmark_runtime_v1 import (CATALOG,DataBudget,atomic,catalog,check_harness,choose,run_one,sha)


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list',action='store_true');p.add_argument('--check-harness',action='store_true')
    p.add_argument('--suite',choices=['core','ocr','reasoning','hallucination','all'],default='core')
    p.add_argument('--datasets',nargs='+');p.add_argument('--execute',action='store_true')
    source=p.add_mutually_exclusive_group();source.add_argument('--run',type=Path);source.add_argument('--hf-model')
    p.add_argument('--revision');p.add_argument('--checkpoint',choices=['best','last'],default='best')
    p.add_argument('--quantization',choices=['nf4','none'],default='nf4')
    p.add_argument('--allow-model-download',action='store_true');p.add_argument('--allow-data-download',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'research_results/standard_benchmarks_v1')
    p.add_argument('--data-root',type=Path,default=ROOT/'datasets/benchmarks/VLMEvalKit')
    p.add_argument('--data-budget-gib',type=float,default=12);p.add_argument('--reserve-gib',type=float,default=32)
    p.add_argument('--mode',choices=['infer','score','all'],default='all')
    p.add_argument('--limit',type=int);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-new-tokens',type=int,default=64);p.add_argument('--max-context',type=int,default=4096)
    p.add_argument('--max-visual-tokens',type=int,default=729)
    p.add_argument('--allow-judge',action='store_true');p.add_argument('--judge-model');p.add_argument('--judge-workers',type=int,default=2)
    args=p.parse_args(argv)
    if min(args.max_new_tokens,args.max_context,args.max_visual_tokens,args.judge_workers)<=0:p.error('Token budgets and judge workers must be positive')
    if args.allow_judge!=bool(args.judge_model):p.error('Use --allow-judge together with an explicit --judge-model')
    if args.reserve_gib<32:p.error('Keep at least32GiB disk reserve')
    return args


def resolve_hf(args):
    path=Path(args.hf_model)
    if path.is_dir():return str(path.resolve()),dict(kind='local_hf',path=str(path.resolve()),
        files={p.name:sha(p) for p in sorted(path.glob('*')) if p.is_file() and p.suffix in ('.json','.safetensors','.model','.jinja','.txt')})
    if not re.fullmatch(r'[0-9a-f]{40,64}',args.revision or ''):raise ValueError('Remote HF models require an immutable commit --revision')
    from huggingface_hub import HfApi,snapshot_download
    if args.allow_model_download:
        from src.utils.research_storage import DiskBudget
        info=HfApi().model_info(args.hf_model,revision=args.revision,files_metadata=True)
        total=sum(f.size or 0 for f in info.siblings)
        if not total:raise ValueError('Cannot preflight model download bytes')
        DiskBudget(ROOT,args.reserve_gib).ensure_space(total+256*2**20)
    path=snapshot_download(args.hf_model,revision=args.revision,local_files_only=not args.allow_model_download,
        allow_patterns=['*.json','*.safetensors','*.model','*.jinja','*.txt'])
    return path,dict(kind='pinned_hf',repo=args.hf_model,revision=args.revision)


def main(argv=None):
    args=arguments(argv);entries=choose(args.suite,args.datasets)
    if args.list:
        for entry in catalog()['datasets']:
            print(f"{entry['name']:25} {entry['domain']:34} judge={'required' if entry['judge_required'] else 'optional'}")
        return
    if args.check_harness:
        os.environ['CUDA_VISIBLE_DEVICES']='';check_harness()
        print(json.dumps(dict(status='passed_CPU_registry_check',datasets=len(catalog()['datasets']),harness=catalog()['harness'])))
        return
    if not(args.run or args.hf_model):raise ValueError('Provide --run or --hf-model (or --list / --check-harness)')
    if args.limit is not None and any(e.get('paired') for e in entries):raise ValueError('--limit cannot split paired/grouped benchmarks')
    if args.mode!='infer' and any(e['judge_required'] for e in entries) and not args.allow_judge:
        raise ValueError('This suite includes judge-required tasks: use --mode infer or explicit --allow-judge --judge-model')
    plan=dict(datasets=[e['name'] for e in entries],source=str(args.run or args.hf_model),
              mode=args.mode,limit=args.limit,seed=args.seed,max_new_tokens=args.max_new_tokens,
              max_context=args.max_context,max_visual_tokens_per_image=args.max_visual_tokens,
              output=str(args.output),harness=catalog()['harness'],external_judge=bool(args.judge_model),
              disk_reserve_GiB=args.reserve_gib,data_budget_GiB=args.data_budget_gib)
    if not args.execute:print(json.dumps(dict(status='plan_only',**plan),indent=2));return
    if args.mode=='score':os.environ['CUDA_VISIBLE_DEVICES']=''
    else:
        # Standard suites never contend with a live trainer. The separately
        # bounded native116 comparison has its own supervised admission lease.
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if gpu.stdout.strip():raise RuntimeError('GPU occupied; active training preserved. Schedule standard suites after training.')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['LMUData']=str(args.data_root.resolve())
    budget=DataBudget(args.data_root,maximum_gib=args.data_budget_gib,reserve_gib=args.reserve_gib);budget.ensure()
    build_dataset=check_harness()
    import vlmeval.dataset.image_base as image_base
    def download(url,filename=None):
        if not args.allow_data_download:raise ValueError('Missing benchmark data: use --allow-data-download within the declared disk budget')
        return budget.download(url,filename)
    image_base.download_file=download
    adapter=None
    saved_protocol=None
    if args.mode=='score':
        saved_protocol=json.loads((args.output/entries[0]['name']/'protocol.json').read_bytes())
        source=saved_protocol['model_source']
        if args.run:
            if source.get('kind')!='research' or source.get('run')!=str(args.run.resolve()) or source.get('checkpoint')!=args.checkpoint:
                raise ValueError('Scoring source differs from the recorded research run')
        elif args.hf_model not in (source.get('repo'),source.get('path')):
            raise ValueError('Scoring source differs from recorded HF source')
    elif args.run:
        run=args.run.resolve(strict=True);checkpoint=run/args.checkpoint/'state.pt'
        source=dict(kind='research',run=str(run),checkpoint=args.checkpoint,
            checkpoint_sha256=sha(checkpoint),config_sha256=sha(run/'resolved_config.json'))
        if args.mode!='score':
            from src.eval.research_benchmark_model_v1 import ResearchCheckpointVLM
            adapter=ResearchCheckpointVLM(run,checkpoint=args.checkpoint,max_new_tokens=args.max_new_tokens,
                max_context=args.max_context,reserve_gb=args.reserve_gib)
    else:
        path,source=resolve_hf(args)
        if args.mode!='score':
            import torch
            from transformers import set_seed
            set_seed(args.seed);torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
            from src.eval.research_hf_vlm_v1 import NativeHFVLM
            adapter=NativeHFVLM(path,quantization=args.quantization,max_new_tokens=args.max_new_tokens,
                max_visual_tokens=args.max_visual_tokens,max_context=args.max_context)
    packages={}
    for name in ('torch','transformers','peft','bitsandbytes','Pillow','numpy','pandas','openai'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    protocol=saved_protocol or dict(schema_version=1,model_source=source,harness=catalog()['harness'],seed=args.seed,
        generation=dict(max_new_tokens=args.max_new_tokens,do_sample=False,num_beams=1,max_context=args.max_context),
        quantization=args.quantization if args.hf_model else 'saved_research_recipe',
        visual_tokens_per_image=args.max_visual_tokens,packages=packages,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),CATALOG,
            ROOT/'src/eval/vlm_benchmark_runtime_v1.py',ROOT/'src/eval/research_benchmark_model_v1.py',ROOT/'src/eval/research_hf_vlm_v1.py',ROOT/'src/eval/research_giga_eot_v1.py']})
    training_hashes=None
    if args.run:
        manifests=[ROOT/'datasets/research/budget18-en-v1'/n for n in
            ('alignment_train.jsonl','alignment_a1_train.jsonl','sft_multimodal_train.jsonl')]
        reference={str(p.relative_to(ROOT)):sha(p) for p in manifests}
        if saved_protocol and saved_protocol.get('local_training_image_reference')!=reference:
            raise ValueError('Recorded local training image reference differs')
        protocol['local_training_image_reference']=reference
        training_hashes={h for p in manifests for line in p.read_text().splitlines() for h in json.loads(line).get('image_hashes',[])}
    args.output.mkdir(parents=True,exist_ok=True);atomic(args.output/'plan.json',plan)
    if adapter:atomic(args.output/'model_loading.json',adapter.loading)
    for entry in entries:
        budget.ensure(64*2**20);dataset=build_dataset(entry['name'])
        if dataset is None:raise ValueError('Dataset construction failed: '+entry['name'])
        actual_protocol=json.loads((args.output/entry['name']/'protocol.json').read_bytes()) if args.mode=='score' else protocol
        if actual_protocol['model_source']!=source:raise ValueError('Mixed model sources in the scoring output')
        result=run_one(dataset,entry,adapter,args.output/entry['name'],actual_protocol,
            limit=args.limit,seed=args.seed,mode=args.mode,judge_model=args.judge_model,
            allow_judge=args.allow_judge,judge_workers=args.judge_workers,budget=budget,training_hashes=training_hashes)
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
