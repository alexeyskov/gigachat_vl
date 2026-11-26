#!/usr/bin/env python3
"""Journaled original Qwen INT4 baseline on the unchanged internal116 selection."""
import argparse,ctypes,hashlib,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'research_results/qwen25_native_int4_internal116_v1'


def load_journal(path,records,digest):
    from src.eval.research_metrics import score_prediction
    if not path.exists():return []
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):raise ValueError('Incomplete journal tail; preserve and repair explicitly')
    rows=[json.loads(x) for x in raw.splitlines()]
    if len(rows)>len(records):raise ValueError('Too many baseline records')
    for row,record in zip(rows,records):
        if (row.get('protocol_sha256')!=digest or any(row.get(k)!=v for k,v in record.items())
                or row.get('score')!=score_prediction(record,row['prediction'])):
            raise ValueError('Baseline journal order, inputs, protocol or score changed')
    return rows


def summarize(rows,protocol,loading,journal):
    from src.eval.research_metrics import aggregate_scores
    from src.utils.research_storage import _atomic_json
    result=aggregate_scores(rows,require_four_groups=True)
    result.update(status='completed',records=116,
        protocol_sha256=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest(),
        official_benchmark_score=False,selection_protocol=protocol['selection_protocol'],
        model_loading=loading,predictions_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
        scope='same train-derived internal116 single-teacher-reference protocol as research models; not official benchmark accuracy')
    _atomic_json(OUT/'summary.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=int,required=True)
    p.add_argument('--gpu-fraction',type=float,default=.12);args=p.parse_args()
    if os.getppid()!=args.parent:raise ValueError('Baseline owner differs')
    if ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError('Cannot arm parent-death signal')
    if os.getppid()!=args.parent:raise ValueError('Baseline owner disappeared')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from transformers import set_seed
    from src.eval.research_hf_vlm_v1 import NativeHFVLM
    from src.eval.research_selection import build_internal_validation
    from scripts.run_research_stage import deduplicate_selection_records
    from src.eval.research_metrics import aggregate_scores,score_prediction
    from src.utils.research_storage import _atomic_json
    set_seed(42);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=True;torch.cuda.set_per_process_memory_fraction(args.gpu_fraction,0)
    data=ROOT/'datasets/research/budget18-en-v1'
    validation=[json.loads(x) for x in (data/'sft_validation.jsonl').read_text().splitlines()]
    selection=build_internal_validation(deduplicate_selection_records(validation),per_group=32,seed=42)
    records=[json.loads(x) for x in (data/'internal_selection_116.jsonl').read_text().splitlines()]
    if records!=selection['records'] or len(records)!=116:raise ValueError('Exact frozen internal116 required')
    ready=json.loads((OUT/'source_ready.json').read_bytes())
    manifest=json.loads((OUT/'source_content_hashes.json').read_bytes())
    if manifest['repo']!=ready['repo'] or manifest['revision']!=ready['revision']:raise ValueError('Source content revision differs')
    for item in manifest['files']:
        source=Path(item['path']);hasher=hashlib.sha256()
        with source.open('rb') as handle:
            for chunk in iter(lambda:handle.read(8*2**20),b''):hasher.update(chunk)
        if source.stat().st_size!=item['bytes'] or hasher.hexdigest()!=item['sha256']:raise ValueError('Native source file content differs')
        if not item['borrowed_shared_file']:
            fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW)
            try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            finally:os.close(fd)
    protocol={'model':ready['repo'],'revision':ready['revision'],'quantization':'bnb_nf4_double_quant',
        'vision_dtype':'bfloat16','LLM_compute_dtype':'bfloat16','seed':42,
        'deterministic_algorithms':True,'matmul_allow_tf32':True,'torch_threads':2,'cublas_workspace_config':':4096:8',
        'max_visual_tokens_per_image':729,'generation':{'max_new_tokens':64,'do_sample':False,'num_beams':1},
        'selection_protocol':selection['protocol'],'prompt':'native user image plus original question, no answer mining',
        'source_content_manifest_sha256':hashlib.sha256((OUT/'source_content_hashes.json').read_bytes()).hexdigest(),
        'code_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'src/eval/research_hf_vlm_v1.py']}}
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_bytes())!=protocol:raise ValueError('Baseline resume protocol changed')
    _atomic_json(path,protocol);journal=OUT/'predictions.jsonl'
    rows=load_journal(journal,records,digest)
    if len(rows)==116:return summarize(rows,protocol,json.loads((OUT/'model_loaded.json').read_bytes()),journal)
    stopped=False
    def stop(_n,_f):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    model=NativeHFVLM(ready['source'],quantization='nf4',max_visual_tokens=729,max_new_tokens=64,max_context=4096)
    if model.model.config.model_type!='qwen2_5_vl':raise ValueError('Original native Qwen2.5-VL required')
    _atomic_json(OUT/'model_loaded.json',dict(model.loading,worker_pid=os.getpid(),created_unix=time.time()))
    done={r['record_id'] for r in rows}
    for record in records:
        if stopped:break
        if record['record_id'] in done:continue
        if len(record['image_paths'])!=1:raise ValueError('Internal baseline expects exact one-image rows')
        from src.dataset.vision_cache import image_content_hash
        if record.get('image_hashes') and image_content_hash(record['image_paths'][0])!=record['image_hashes'][0]:
            raise ValueError('Frozen selection image pixels differ')
        prediction=model.generate([{'type':'image','value':record['image_paths'][0]},
                                   {'type':'text','value':record['question']}])
        row=dict(record,**prediction,protocol_sha256=digest)
        row['score']=score_prediction(record,row['prediction'])
        with journal.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        rows.append(row);_atomic_json(OUT/'progress.json',{'completed_records':len(rows),'total_records':116,
            'peak_VRAM_GiB':torch.cuda.max_memory_allocated()/2**30,'time':time.time()})
        print(json.dumps({'event':'baseline_record','completed':len(rows),'total':116,'score':row['score']}),flush=True)
    if len(rows)==116:
        summarize(rows,protocol,model.loading,journal)


if __name__=='__main__':main()
