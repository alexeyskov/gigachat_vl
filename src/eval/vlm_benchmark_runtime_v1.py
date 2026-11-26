"""Pinned VLMEvalKit dataset prompts/scorers, with resumable local inference.

Imports model libraries only when executing; list/plan stay CPU/stdlib only.
No credentials, answer columns or scorer output are passed to model adapters.
"""
from __future__ import annotations
import hashlib,importlib.metadata,json,math,os,random,shutil,ssl,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CATALOG=ROOT/'configs/eval/vlm_benchmarks_v1.json'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*2**20),b''):digest.update(chunk)
    return digest.hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def catalog():return json.loads(CATALOG.read_bytes())


def choose(suite=None,names=None):
    spec=catalog();lookup={row['name']:row for row in spec['datasets']}
    names=names or (list(lookup) if suite=='all' else spec['suites'].get(suite or 'core',[]))
    if not names or len(set(names))!=len(names):raise ValueError('Nonempty unique benchmark names required')
    if set(names)-set(lookup):raise ValueError('Unknown benchmarks: '+str(sorted(set(names)-set(lookup))))
    return [lookup[name] for name in names]


def check_harness():
    expected=catalog()['harness'];path=ROOT/expected['path']
    result=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True,check=True)
    if result.stdout.strip()!=expected['revision']:raise ValueError('Pinned VLMEvalKit commit differs')
    dirty=subprocess.run(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True)
    if dirty.stdout.strip():raise ValueError('VLMEvalKit tracked source has local modifications')
    sys.path.insert(0,str(path))
    # The upstream top-level initializer temporarily changes this global. Keep
    # normal verified HTTPS in this process, including when import fails.
    context=ssl._create_default_https_context
    try:
        from vlmeval.dataset import build_dataset,SUPPORTED_DATASETS
    finally:ssl._create_default_https_context=context
    missing={r['name'] for r in catalog()['datasets']}-set(SUPPORTED_DATASETS)
    if missing:raise ValueError('Benchmarks absent from pinned registry: '+str(sorted(missing)))
    return build_dataset


class DataBudget:
    def __init__(self,root,*,maximum_gib=12,reserve_gib=32):
        self.root=Path(root).resolve();self.maximum=int(maximum_gib*2**30);self.reserve=int(reserve_gib*2**30)
        if self.maximum<=0 or self.reserve<32*2**30:raise ValueError('Positive data budget and at least32GiB reserve required')
        self.root.mkdir(parents=True,exist_ok=True)

    def used(self):return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def ensure(self,extra=0):
        used=self.used();free=shutil.disk_usage(self.root).free
        if used+extra>self.maximum or free-extra<self.reserve:raise OSError('Benchmark data budget/disk reserve exceeded')
        return dict(data_bytes=used,free_bytes=free,reserve_bytes=self.reserve,maximum_data_bytes=self.maximum)

    def download(self,url,filename=None):
        import requests
        if not filename:raise ValueError('Explicit bounded download destination required')
        path=Path(filename).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Dataset download outside declared data root')
        temporary=path.with_name(path.name+'.partial');path.parent.mkdir(parents=True,exist_ok=True)
        try:
            with requests.get(url,stream=True,timeout=(20,120),headers={'Accept-Encoding':'identity'}) as response:
                response.raise_for_status();expected=int(response.headers.get('Content-Length',0));self.ensure(3*expected)
                count=0
                with temporary.open('wb') as f:
                    for chunk in response.iter_content(4*2**20):
                        self.ensure(len(chunk));f.write(chunk);f.flush();count+=len(chunk)
                    os.fsync(f.fileno())
                if expected and count!=expected:raise IOError('Incomplete dataset download')
                # Upstream localizes large base64 TSVs into image files and a
                # second metadata table. Reserve their conservative expansion
                # before returning control to the unchanged official loader.
                self.ensure(2*count)
            os.replace(temporary,path);return str(path)
        except BaseException:
            temporary.unlink(missing_ok=True);raise


def selected_rows(dataset,limit,seed,entry):
    if dataset.data['index'].astype(str).duplicated().any():raise ValueError('Dataset indices are not unique')
    if limit is not None:
        if limit<1:raise ValueError('Limit must be positive')
        if entry.get('paired'):raise ValueError('Paired/grouped benchmark must run complete to preserve its metric')
    indices=list(range(len(dataset.data)))
    if limit is not None and limit<len(indices):indices=sorted(random.Random(seed).sample(indices,limit))
    return indices


def journal_rows(path,indices,source_indices,protocol_sha):
    if not path.exists():return []
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):raise ValueError('Partial prediction journal; preserve and repair explicitly')
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)>len(indices):raise ValueError('Journal has extra rows')
    for row,ordinal in zip(rows,indices):
        if (row['ordinal']!=ordinal or row['index']!=str(source_indices[ordinal]) or row['protocol_sha256']!=protocol_sha
                or not isinstance(row['prediction'],str)):
            raise ValueError('Journal input order/source/protocol mismatch')
    return rows


def evaluate_official(dataset,prediction_file,entry,*,judge_model=None,allow_judge=False,judge_workers=2):
    if entry['judge_required'] and not(allow_judge and judge_model):
        raise ValueError('This official scorer requires explicit --allow-judge and --judge-model')
    if judge_model and not allow_judge:raise ValueError('External judge calls need --allow-judge')
    # Required-judge tasks never fall back to a homemade exact-match metric.
    # Upstream MCQ/YorN may fall back; reject that as an evaluation failure.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('error',message='.*will use exact matching for evaluation.*')
        return dataset.evaluate(str(prediction_file),model=judge_model or 'exact_matching',nproc=judge_workers)


def json_metrics(value):
    if hasattr(value,'to_json'):return json.loads(value.to_json(orient='records'))
    if isinstance(value,dict):return {str(k):json_metrics(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_metrics(v) for v in value]
    if hasattr(value,'item'):return json_metrics(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def run_one(dataset,entry,adapter,output,protocol,*,limit=None,seed=42,mode='all',
            judge_model=None,allow_judge=False,judge_workers=2,budget=None,training_hashes=None):
    from vlmeval.smp import dump
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if mode=='score':
        indices=protocol['selected_ordinals']
        if (not indices or indices!=sorted(set(indices)) or indices[0]<0 or indices[-1]>=len(dataset.data)
                or [str(dataset.data.iloc[i]['index']) for i in indices]!=protocol['selected_indices']):
            raise ValueError('Recorded scoring sample differs from dataset')
    else:indices=selected_rows(dataset,limit,seed,entry)
    protocol=dict(protocol,dataset=entry,dataset_source_sha256=sha(dataset.data_path),
                  selected_ordinals=indices,selected_indices=[str(dataset.data.iloc[i]['index']) for i in indices],
                  total_source_records=len(dataset.data),diagnostic_subset=len(indices)<len(dataset.data))
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_bytes())!=protocol:
        raise ValueError('Resume protocol changed; choose a new output directory')
    atomic(output/'protocol.json',protocol)
    journal=output/'predictions.jsonl';rows=journal_rows(journal,indices,list(dataset.data['index']),digest)
    if mode!='score':
        for ordinal in indices[len(rows):]:
            if budget:budget.ensure(64*2**20)
            # Dataset-owned official prompts. Adapter sees this message only.
            message=dataset.build_prompt(dataset.data.iloc[ordinal])
            if any(m['type'] not in ('image','text') for m in message):raise ValueError('Image benchmark adapter received other media')
            if budget:budget.ensure()
            if training_hashes is not None:
                from src.dataset.vision_cache import image_content_hash
                image_hashes=[image_content_hash(m['value']) for m in message if m['type']=='image']
            else:image_hashes=None
            result=adapter.generate(message)
            if not isinstance(result.get('prediction'),str):raise ValueError('Generator must return text')
            row=dict(result,ordinal=ordinal,index=str(dataset.data.iloc[ordinal]['index']),protocol_sha256=digest)
            if image_hashes is not None:
                row['local_training_image_overlap']=sorted(set(image_hashes)&training_hashes)
            with journal.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            rows.append(row);atomic(output/'progress.json',dict(completed=len(rows),total=len(indices),updated_unix=time.time()))
            print(json.dumps(dict(event='benchmark_prediction',dataset=entry['name'],completed=len(rows),total=len(indices))),flush=True)
    if len(rows)!=len(indices):raise ValueError('Scoring requires complete predictions for the declared sample')
    table=dataset.data.iloc[indices].copy();table['prediction']=[r['prediction'] for r in rows]
    # Keep official metadata/options/references in the scoring artifact only.
    for column in ('image','image_path'):
        if column in table:table=table.drop(columns=[column])
    prediction_file=output/'predictions.xlsx';dump(table,str(prediction_file))
    result=dict(status='inference_completed',dataset=entry['name'],records=len(rows),
                total_source_records=len(dataset.data),diagnostic_subset=protocol['diagnostic_subset'],
                protocol_sha256=digest,predictions_sha256=sha(journal),
                official_full_score=False,metric_description=entry['metric'])
    if training_hashes is not None:
        result['local_training_image_overlap']=dict(records=sum(bool(r.get('local_training_image_overlap')) for r in rows),
            unique_images=len({h for r in rows for h in r.get('local_training_image_overlap',[])}),
            scope='exact decoded RGB matches against released alignment/A1/SFT training images; no samples excluded')
    result['synthetic_fixture']=bool(protocol.get('test_fixture'))
    atomic(output/'summary.json',result)
    if mode!='infer':
        evaluation=dict(judge_model=judge_model or 'exact_matching',external_judge=bool(judge_model),
                        judge_workers=judge_workers,harness=catalog()['harness'],
                        scorer_wrapper_sha256=sha(Path(__file__)))
        evaluation_path=output/'evaluation_protocol.json'
        if evaluation_path.exists() and json.loads(evaluation_path.read_bytes())!=evaluation:
            raise ValueError('Judge/scoring protocol changed; use a new output directory')
        atomic(evaluation_path,evaluation)
        result['metrics']=json_metrics(evaluate_official(dataset,prediction_file,entry,judge_model=judge_model,
            allow_judge=allow_judge,judge_workers=judge_workers))
        result.update(status='completed',official_full_score=not(protocol['diagnostic_subset'] or protocol.get('test_fixture')),evaluation=evaluation)
    atomic(output/'summary.json',result);return result
