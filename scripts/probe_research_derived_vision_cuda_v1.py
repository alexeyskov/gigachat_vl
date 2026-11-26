#!/usr/bin/env python3
"""ROOT-only opt-in technical donor parity + one synthetic A0 MLP update.

Default metadata mode imports no Torch or model weights. No original/cache file
is deleted. This is not a caption benchmark, primary run or pruning authority.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils import research_derived_vision_v1 as derived
from src.utils import research_stock_determinism as stock
stock.bootstrap_environment()
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
OWNER='technical_stock_derived_vision_cuda_v1'
SCRIPT='scripts/probe_research_derived_vision_cuda_v1.py'
TEST='tests/test_research_derived_vision_cuda_probe_v1.py'
READY='configs/research/derived_vision_v1_ready.json'
READY_SHA='f91d3bea9b10bc160d48b3abef37cdf74378a3a15198349c1a6890611e043e8d'
FIXTURES=('research_results/vision_probes/qwen25/landscape.png','research_results/vision_probes/qwen25/portrait.png')
FIXTURE_RGB=('a3c18d16f9713f78007058d98ff19f9a5b8a9b3af4ede04d322949c05a17d3b3','562070b445c60d60a0601478d59f9a9c0838d26f0ef8dd55a52fc85d0c65b71c')
GIB=2**30


def file_sha(path,check=None):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block:=handle.read(8*2**20):
            if check:check()
            h.update(block)
    return h.hexdigest()


def state_stat(path):
    s=Path(path).stat();return {k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}


def write(path,value):
    from src.utils.research_storage import _atomic_json
    _atomic_json(Path(path),value)


def processes():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    result={}
    for line in raw.splitlines():
        if line.strip():
            pid,mib=line.split(',');result[int(pid)]=int(mib)*2**20
    return result


def plan():
    if file_sha(ROOT/READY)!=READY_SHA:raise ValueError('Reviewed derived prototype READY changed')
    from huggingface_hub import hf_hub_download
    from src.dataset.vision_cache import image_content_hash
    paths=[ROOT/p for p in FIXTURES]
    if tuple(image_content_hash(p) for p in paths)!=FIXTURE_RGB:raise ValueError('Fixed synthetic RGB fixtures changed')
    rows=[]
    for repo in derived.DONORS:
        spec=derived.DONORS[repo]
        profile='qwen_native_bounded_2048' if spec['backend']=='qwen2_5_vl' else None
        binding=derived.resolve_derived_vision(repo,preprocessing_profile=profile)
        original=Path(hf_hub_download(repo,'config.json',revision=spec['revision'],local_files_only=True)).parent
        source=derived.read(Path(binding['path'])/'derivation.json')['source']
        files={}
        for name,item in source['source_files'].items():
            p=original/name
            if not p.is_file() or p.stat().st_size!=item['bytes']:raise ValueError('Required cached source missing/size changed')
            if not name.endswith('.safetensors') and file_sha(p)!=item['sha256']:raise ValueError('Original source asset differs')
            files[name]={'path':str(p),'resolved_path':str(p.resolve()),'stat':state_stat(p),'sha256':item['sha256']}
        rows.append({'repo':repo,'binding':binding,'original_snapshot':str(original),'original_files':files})
    return {'schema_version':1,'owner':OWNER,'status':'metadata_preflight_only','gpu_used':False,
        'fixture_paths':[str(p) for p in paths],'fixture_rgb_hashes':list(FIXTURE_RGB),'fixtures_are_synthetic':True,
        'paired_records':4,'targets':['This is a synthetic landscape test image.','This is a synthetic portrait test image.'],
        'question':'Describe the image briefly.','generation_max_new_tokens':8,'model':stock.SMALL,
        'numerical_profile':stock.definition(),'derived_source_sha256':derived.code_binding(),
        'probe_source_sha256':file_sha(ROOT/SCRIPT),'derived_READY_sha256':READY_SHA,'donors':rows,
        'initial_host_headroom_required_bytes':72*GIB,'runtime_host_floor_bytes':32*GIB,
        'torch_allocator_cap_bytes':40*GIB,'NVML_process_cap_bytes':48*GIB,'staging_cap_bytes':GIB,
        'disk_reserve_bytes':120*GIB,'optimizer_updates_per_donor':1,'pruning_authorized':False,
        'runtime_adopted':False,'scientific_caption_score_claimed':False,
        'limits':['Two small RGB fixtures only; no full-corpus/context-bound claim.','No original HF/cache files deleted or replaced.','Fresh original donor full SHAs and GPU-required vision bits are checked in GPU execution, not metadata mode.']}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def composed_runner(binding):
    private=stock.runner()
    return derived.install_private_runner_hooks(private,binding,model_repo=stock.SMALL)


def reopen_disk_cache(cache,cache_type):
    return cache_type(cache.root.parent,cache.fingerprint,reserve_bytes=120*GIB,ram_max_bytes=0)


def compare_generations(left,right):
    if len(left)!=2 or len(right)!=2 or left!=right:raise ValueError('Both fixed downstream generation/logit records must match exactly')


def run(args):
    if args.output is None or args.output.exists():raise ValueError('GPU probe needs a NEW --output directory')
    if processes():raise RuntimeError('GPU occupied; ROOT must schedule an idle window')
    p=plan();available=derived.host_guard()
    if available<p['initial_host_headroom_required_bytes']:raise MemoryError('Actual host/cgroup headroom below72GiB; no cache credit')
    if shutil.disk_usage(ROOT).free<p['disk_reserve_bytes']+p['staging_cap_bytes']:raise OSError('120GiB reserve plus1GiB staging required')
    import torch
    from src.model.research_vision import build_frozen_vision
    from src.dataset.vision_cache import VisionFeatureCache,encoder_fingerprint
    from src.eval.research_inference import generate_prediction,build_generation_inputs,freeze_for_generation,_validate_learned_state
    from src.model.research_moe import set_modality_mask
    from src.utils.research_checkpoint_compat import load_research_payload
    from transformers import set_seed
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible CUDA GPU required')
    stock.activate();torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(40*GIB/torch.cuda.get_device_properties(0).total_memory)
    args.output.mkdir(parents=True);started=time.monotonic()
    report=dict(p,status='running',producer='ROOT',gpu_used=True,started_unix=time.time(),pid=os.getpid(),donor_results=[])
    report['device']={'name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0))}
    sources={**p['derived_source_sha256'],**p['numerical_profile']['source_sha256'],SCRIPT:p['probe_source_sha256'],TEST:file_sha(ROOT/TEST)}
    with zipfile.ZipFile(args.output/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,h in sorted(sources.items()):
            raw=(ROOT/name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Proof source changed before archive')
            archive.writestr(name,raw)
    report['source_archive']={'sha256':file_sha(args.output/'source.zip'),'source_sha256':sources}
    write(args.output/'plan.json',p);write(args.output/'report.json',report)
    def guard():
        head=derived.host_guard();current=processes()
        if set(current)-{os.getpid()}:raise RuntimeError('Foreign CUDA process appeared')
        if current.get(os.getpid(),0)>48*GIB or torch.cuda.max_memory_reserved()>40*GIB:raise MemoryError('Technical probe GPU budget exceeded')
        if sum(q.stat().st_size for q in args.output.rglob('*') if q.is_file())>GIB:raise OSError('Probe staging cap exceeded')
        if shutil.disk_usage(ROOT).free<120*GIB:raise OSError('Disk reserve crossed')
        if file_sha(ROOT/SCRIPT)!=p['probe_source_sha256'] or derived.code_binding()!=p['derived_source_sha256']:raise ValueError('Executing proof source changed')
        stock.assert_active()
        return {'host_actual_headroom':head,'NVML_process_bytes':current.get(os.getpid(),0),'torch_peak_allocated':torch.cuda.max_memory_allocated(),'torch_peak_reserved':torch.cuda.max_memory_reserved()}
    def tensor_hash(t):
        h=hashlib.sha256();flat=t.detach().contiguous().reshape(-1);chunk=max(1,8*2**20//t.element_size())
        for start in range(0,flat.numel(),chunk):
            h.update(flat[start:start+chunk].view(torch.uint8).cpu().numpy().tobytes())
        return h.hexdigest()
    def module_state(vision):
        result={}
        for prefix,module in [('vision_tower',vision.vision_tower),('vision_projector',vision.vision_projector)]:
            if module is None:continue
            if any(t.requires_grad or t.device.type!='cuda' for t in module.parameters()):raise ValueError('Vision modules must remain frozen onCUDA')
            if any('ForCausalLM' in type(m).__name__ or 'language_model' in name for name,m in module.named_modules()):raise ValueError('Donor language tower was instantiated')
            for name,t in module.state_dict().items():
                result[prefix+'.'+name]={'shape':list(t.shape),'dtype':str(t.dtype),'sha256':tensor_hash(t)}
        return result
    def projector_state(model):return {n:tensor_hash(t) for n,t in model.projector.state_dict().items()}
    def signatures(model):return {n:(id(t),t.data_ptr(),t._version,str(t.dtype),str(t.device),t.requires_grad) for n,t in model.llm.named_parameters()}
    def adam_state(optimizer):
        state=optimizer.state_dict()
        return {'param_groups':state['param_groups'],'state':{str(i):{n:({'shape':list(t.shape),'dtype':str(t.dtype),'sha256':tensor_hash(t)} if torch.is_tensor(t) else t) for n,t in values.items()} for i,values in state['state'].items()}}
    def factory(donor,donor_dir):
        set_seed(42);stock.activate();binding=donor['binding']
        derived.assert_binding(binding,full=True,check=derived.host_guard)
        private=composed_runner(binding)
        old=sys.argv
        try:
            sys.argv=['technical-probe','--stage','A0','--recipe','4f','--model',stock.SMALL,'--vision',donor['repo'],
                '--manifest','technical_synthetic_only','--validation','technical_synthetic_only','--output',str(donor_dir),
                '--cache-dir',str(donor_dir/'cache'),'--max-length','4096','--reserve-gb','120']
            if binding['profile_ref']['name']!='native':sys.argv+=['--vision-preprocessing-profile',binding['profile_ref']['name']]
            config=private.parse_args()
        finally:sys.argv=old
        config.numerical_profile=stock.definition();config.allow_legacy_a0_initialization=False
        composition={'owner':OWNER,'probe_source_sha256':p['probe_source_sha256'],'derived_source_sha256':p['derived_source_sha256'],'stock_profile':p['numerical_profile']}
        config.execution_composition=composition
        derived.configure_args(config,binding)
        model=private.ResearchModel(config);stock.assert_active()
        if model.cache.fingerprint!=binding['cache_fingerprint'] or model.vision_ref!=binding['upstream_ref']:raise ValueError('Composed factory changed derived binding')
        if model.base.get_input_embeddings().weight.dtype!=torch.bfloat16 or any(t.requires_grad for t in model.llm.parameters()):raise ValueError('Technical A0 requires native BF16 frozen language base')
        return model,config,composition
    def downstream(model,cache,records):
        old=model.cache;model.cache=cache;result=[]
        try:
            freeze_for_generation(model);stock.assert_active()
            for record in records:
                public={k:record[k] for k in ('record_id','question','image_paths','image_hashes')}
                with torch.no_grad():
                    prepared=build_generation_inputs(model,public,max_new_tokens=8)
                    set_modality_mask(model.base,prepared['modality_mask'],prepared['attention_mask'])
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model.base(inputs_embeds=prepared['inputs_embeds'],attention_mask=prepared['attention_mask'],
                            logits_to_keep=1,use_cache=False,return_dict=True).logits
                    result.append({'record_id':record['record_id'],'prefill_logits_sha256':tensor_hash(logits),'prefill_logits_shape':list(logits.shape),
                        'generation':generate_prediction(model,public,max_new_tokens=8)})
                    del prepared,logits
                guard()
        finally:model.cache=old
        return result
    try:
        for donor in p['donors']:
            guard();binding=donor['binding'];directory=args.output/binding['export_ref']['donor'];directory.mkdir()
            for name,item in donor['original_files'].items():
                if state_stat(item['path'])!=item['stat'] or file_sha(item['path'],derived.host_guard)!=item['sha256']:raise ValueError('Fresh original source file digest differs')
            original=build_frozen_vision(donor['original_snapshot'],torch.device('cuda'),torch.bfloat16,
                preprocessing_profile=None if binding['profile_ref']['name']=='native' else binding['profile_ref']['name'])
            model,config,composition=factory(donor,directory)
            original_state=module_state(original);derived_state=module_state(model.extractor)
            if original_state!=derived_state:raise ValueError('Native original/derived vision state keys/dtypes/raw bits differ')
            captured={};encoded_calls={}
            original_fp=encoder_fingerprint({'profile':OWNER,'original_vision_fingerprint':original.fingerprint_config,'source_ref':binding['upstream_ref']})
            cache_orig=VisionFeatureCache(directory/'cache-original',original_fp,reserve_bytes=120*GIB,ram_max_bytes=0)
            caches={'original':cache_orig,'derived':model.cache}
            for label,vision in [('original',original),('derived',model.extractor)]:
                captured[label]=[];encoded_calls[label]=0
                def encode(paths,vision=vision,label=label):
                    encoded_calls[label]+=1;rows=vision.encode_paths(paths);captured[label].extend(rows);return rows
                caches[label].get_many(p['fixture_paths'],encode,image_hashes=p['fixture_rgb_hashes'])
                caches[label]=reopen_disk_cache(caches[label],VisionFeatureCache)
                restored=caches[label].get_many(p['fixture_paths'][::-1],lambda _:(_ for _ in ()).throw(AssertionError('Unexpected warm encoder call')),
                    image_hashes=p['fixture_rgb_hashes'][::-1],read_only=True)
                if any(not torch.equal(r['features'].cpu(),restored[1-i]) for i,r in enumerate(captured[label])):raise ValueError('Disk cache roundtrip differs')
            if original_fp==binding['cache_fingerprint']:raise ValueError('Original andderived cache namespaces must be distinct')
            for left,right in zip(captured['original'],captured['derived']):
                if left['metadata']!=right['metadata'] or not torch.equal(left['features'],right['features']):raise ValueError('Native BF16 paired features/metadata differ')
            model.cache=caches['derived']
            model.train();model.extractor.vision_tower.eval()
            if model.extractor.vision_projector is not None:model.extractor.vision_projector.eval()
            records=[{'record_id':OWNER+'_'+str(i),'question':p['question'],'answer':p['targets'][i],
                'image_paths':[path],'image_hashes':[p['fixture_rgb_hashes'][i]]} for i,path in enumerate(p['fixture_paths'])]
            if {n for n,t in model.named_parameters() if t.requires_grad}!={'projector.'+n for n,_ in model.projector.named_parameters()}:raise ValueError('Only A0 MLP may be trainable')
            language_before=signatures(model);p_before=projector_state(model);branch={}
            for label in ('original','derived'):
                model.cache=caches[label];model.zero_grad(set_to_none=True);prepared=model.prepare(records);loss,count,_=model.forward_loss(prepared)
                if count<=0 or not torch.isfinite(loss):raise ValueError('Invalid technical caption loss')
                (loss/count).backward()
                gradients={n:t.grad for n,t in model.projector.named_parameters()}
                if any(t is None or not torch.isfinite(t).all() for t in gradients.values()) or not any(torch.count_nonzero(t).item() for t in gradients.values()):raise ValueError('MLP gradients must be finite/nonzero')
                if any(t.grad is not None for t in model.llm.parameters()):raise ValueError('Frozen LLM received gradients')
                branch[label]={'loss_sum':float(loss.detach()),'supervised_tokens':count,'MLP_gradient_sha256':{n:tensor_hash(t) for n,t in gradients.items()}}
                del prepared,loss,gradients;guard()
            if branch['original']!=branch['derived']:raise ValueError('Original/derived technical A0 losses orallMLPgradients differ')
            optimizer=torch.optim.AdamW([{'params':[t for n,t in model.projector.named_parameters() if t.ndim>=2],'lr':1e-3,'weight_decay':.01},
                {'params':[t for n,t in model.projector.named_parameters() if t.ndim<2],'lr':1e-3,'weight_decay':0.}])
            norm=torch.nn.utils.clip_grad_norm_(model.projector.parameters(),1.0)
            if not torch.isfinite(norm) or norm<=0:raise ValueError('Invalid A0 gradient norm')
            optimizer.step();model.zero_grad(set_to_none=True)
            if signatures(model)!=language_before or projector_state(model)==p_before:raise ValueError('MLP update absent orfrozen LLM identity/version changed')
            model.cache=caches['derived'];p_after=projector_state(model);adam_before=adam_state(optimizer)
            payload={'owner':OWNER,'technical_only':True,'step':1,'training_updates':1,'config':vars(config),
                'model':{'projector.'+n:t.detach().cpu() for n,t in model.projector.state_dict().items()},'optimizer':optimizer.state_dict(),
                'model_ref':model.model_ref,'vision_ref':model.vision_ref,'cache_fingerprint':model.cache.fingerprint,'execution_composition':composition}
            payload=derived.bind_checkpoint_payload(payload,binding);checkpoint=directory/'technical_A0_P_only.pt';torch.save(payload,checkpoint)
            if checkpoint.stat().st_size>100*2**20:raise OSError('Technical P+Adam delta exceeded100MiB bound')
            before_generations=downstream(model,caches['original'],records);after_generations=downstream(model,caches['derived'],records);compare_generations(before_generations,after_generations)
            if module_state(original)!=original_state or module_state(model.extractor)!=derived_state:raise ValueError('Vision state changed during update/generation')
            features=[{'shape':list(x['features'].shape),'dtype':str(x['features'].dtype),'sha256':tensor_hash(x['features']),'metadata':x['metadata']} for x in captured['derived']]
            del optimizer,payload,model,original,captured,vision,encode,left,right;gc.collect();torch.cuda.empty_cache();guard()
            saved=load_research_payload(checkpoint,map_location='cpu',mmap=True);derived.validate_payload_binding(saved,binding);stock.validate_config(saved['config'])
            restored,_,restored_composition=factory(donor,directory)
            if restored_composition!=saved['execution_composition'] or vars(restored.args)!=saved['config']:raise ValueError('Composed reload config/source differs')
            _validate_learned_state(restored,saved['model']);loaded=restored.load_state_dict(saved['model'],strict=False)
            if loaded.unexpected_keys or projector_state(restored)!=p_after:raise ValueError('Actual P-only native reload differs')
            restored_optimizer=torch.optim.AdamW([{'params':[t for n,t in restored.projector.named_parameters() if t.ndim>=2],'lr':1e-3,'weight_decay':.01},
                {'params':[t for n,t in restored.projector.named_parameters() if t.ndim<2],'lr':1e-3,'weight_decay':0.}])
            restored_optimizer.load_state_dict(saved['optimizer'])
            if adam_state(restored_optimizer)!=adam_before:raise ValueError('Actual Adam restore metadata/rawbits differ')
            restored.encode=lambda _:(_ for _ in ()).throw(AssertionError('Unexpected encoder during cached reload'))
            reloaded=downstream(restored,restored.cache,records);compare_generations(after_generations,reloaded)
            if any(state['exp_avg'].dtype!=torch.float32 or state['exp_avg_sq'].dtype!=torch.float32 or not torch.isfinite(state['exp_avg']).all() or not torch.isfinite(state['exp_avg_sq']).all() for state in saved['optimizer']['state'].values()):raise ValueError('Invalid saved FP32 Adam state')
            if not any(torch.count_nonzero(state['exp_avg']).item() for state in saved['optimizer']['state'].values()):raise ValueError('Adam first moment remainszero')
            entry={'repo':donor['repo'],'status':'passed_technical_only','features':features,
                'vision_state':derived_state,'vision_original_state':original_state,'vision_original_derived_bits_exact':True,
                'feature_and_metadata_bits_exact':True,'encoded_batches':encoded_calls,'cache_fingerprints':{k:v.fingerprint for k,v in caches.items()},
                'source_and_derived_caption_loss_and_gradients_exact':branch,'one_MLP_optimizer_update':True,'preclip_gradient_norm':float(norm),
                'P_before_sha256':p_before,'P_after_sha256':p_after,'LLM_parameter_identity_version_unchanged_during_update':True,
                'checkpoint':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,'checkpoint_sha256':file_sha(checkpoint),
                'actual_P_only_composed_reload_exact':True,'actual_Adam_restore_exact_without_second_update':True,
                'saved_Adam_moments_FP32_finite_nonzero':True,'downstream_original':before_generations,
                'downstream_derived':after_generations,'downstream':reloaded,'memory':guard()}
            report['donor_results'].append(entry);write(args.output/'report.json',report)
            print(json.dumps({'repo':donor['repo'],'status':entry['status'],'pairs':2,'checkpoint_bytes':entry['checkpoint_bytes']}),flush=True)
            del saved,restored,restored_optimizer,caches,entry;gc.collect();torch.cuda.empty_cache()
        for donor in p['donors']:
            derived.assert_binding(donor['binding'])
            if any(state_stat(v['path'])!=v['stat'] for v in donor['original_files'].values()):raise ValueError('Original donor identity changed during proof')
        report.update(status='passed_technical_four_pairs_no_pruning_authority',completed_pairs=4,elapsed_seconds=time.monotonic()-started,
            final_memory=guard(),pruning_authorized=False,runtime_adopted=False,primary_complete=False,scientific_scores_generated=0)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        raise
    finally:write(args.output/'report.json',report)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--allow-gpu',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.allow_gpu:print(json.dumps(plan(),indent=2));return 0
    return run(args)


if __name__=='__main__':raise SystemExit(main())
