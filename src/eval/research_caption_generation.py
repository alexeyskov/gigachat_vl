"""Caption-only native B1 owner; gold is opened only after all rows are durable.

Importing this module loads neither Torch nor a model. Production execution
reuses the frozen research inference implementation and never calls VQA scores.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
DATA_PROTOCOL_SHA256='5ec185231b73beaadbc25fe4bc1cee26ecb4e51beac652e22843d2aab688ca79'
RECORDS=372
MAX_NEW_TOKENS=256
QUESTION='Describe this image in detail.'
PUBLIC=frozenset(('record_id','image_paths','image_hashes','question'))
OWNER='research_caption_native_b1_v1'
HEX=re.compile(r'^[0-9a-f]{64}$')
SOURCES=('src/eval/research_caption_generation.py','scripts/eval_research_caption_checkpoint.py',
    'src/eval/research_caption_metrics.py','src/eval/research_inference.py','scripts/run_research_stage.py',
    'src/model/research_moe.py','src/model/smallthinker.py','src/model/research_quantization.py',
    'src/model/research_vision.py','src/model/gigachat_vl.py','src/dataset/vision_cache.py',
    'scripts/extract_vision_features.py','src/utils/research_checkpoint_compat.py',
    'src/utils/research_checkpoint_bundle.py','src/utils/research_checkpoint_io.py')


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def stat(path):
    value=Path(path).stat()
    return {key:getattr(value,key) for key in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
def file_sha(path):
    before=stat(path);result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*2**20),b''):result.update(chunk)
    if stat(path)!=before:raise ValueError('File changed while hashing: '+str(path))
    return result.hexdigest()
def read_json(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>32*2**20:raise ValueError('Expected bounded local JSON file')
    return json.loads(path.read_bytes())
def _rows(path):
    result=[]
    with Path(path).open('rb') as handle:
        while raw:=handle.readline(2**20+1):
            if len(raw)>2**20 or not raw.endswith(b'\n') or not raw.strip():raise ValueError('Malformed/truncated manifest row')
            value=json.loads(raw)
            if not isinstance(value,dict):raise ValueError('Expected JSON row object')
            result.append(value)
    return result
def _atomic(path,value,*,raw=False):
    path=Path(path);payload=value if raw else canonical(value)+b'\n'
    if len(payload)>32*2**20:raise ValueError('Caption artifact exceeds32MiB bound')
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as handle:handle.write(payload);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)
def _file(path,expected=None):
    path=Path(path).resolve(strict=True);value={'path':str(path),'stat':stat(path),'sha256':file_sha(path)}
    if expected is not None and value['sha256']!=expected:raise ValueError('Pinned artifact SHA differs: '+str(path))
    return value


def public_record(record):
    if set(record)!=PUBLIC:raise ValueError('Caption model input must contain exactly four public fields')
    if (not isinstance(record['record_id'],str) or record['question']!=QUESTION or
        not isinstance(record['image_paths'],list) or len(record['image_paths'])!=1 or
        not isinstance(record['image_hashes'],list) or len(record['image_hashes'])!=1 or
        not isinstance(record['image_hashes'][0],str) or not HEX.fullmatch(record['image_hashes'][0]) or
        record['record_id']!='caption-'+record['image_hashes'][0]):raise ValueError('Invalid fixed caption input')
    path=record['image_paths'][0]
    if not isinstance(path,str) or not Path(path).is_absolute() or Path(path).resolve(strict=True)!=Path(path):raise ValueError('Caption images must use their audited canonical absolute paths')
    return {key:list(value) if isinstance(value,list) else value for key,value in record.items()}


def _image(path,expected):
    from PIL import Image,ImageOps
    path=Path(path).resolve(strict=True);before=stat(path)
    with Image.open(path) as raw:
        with ImageOps.exif_transpose(raw) as oriented:
            with oriented.convert('RGB') as rgb:
                value=hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode())
                step=max(1,(8*2**20)//max(1,rgb.width*3))
                for top in range(0,rgb.height,step):
                    with rgb.crop((0,top,rgb.width,min(rgb.height,top+step))) as strip:value.update(strip.tobytes())
    if stat(path)!=before or value.hexdigest()!=expected:raise ValueError('Caption RGB pixels/stat changed')
    return {'path':str(path),'stat':before,'canonical_rgb_sha256':expected}


def data_context(bundle):
    """No reference or original train/validation JSONL is opened here."""
    bundle=Path(bundle).resolve(strict=True)
    if set(path.name for path in bundle.iterdir())!={'generation.jsonl','references.jsonl','protocol.json','audit.json'}:
        raise ValueError('Incomplete/unexpected caption data bundle')
    protocol_file=_file(bundle/'protocol.json',DATA_PROTOCOL_SHA256);protocol=read_json(bundle/'protocol.json')
    if (protocol.get('status')!='complete' or protocol.get('records')!=RECORDS or
        protocol.get('generation_references_separate') is not True or protocol.get('options',{}).get('question')!=QUESTION or
        protocol['options'].get('seed')!=42 or protocol['options'].get('public_fields')!=sorted(PUBLIC)):
        raise ValueError('Caption bundle protocol is not the fixed372 pilot profile')
    inputs=_file(bundle/'generation.jsonl',protocol['files_sha256']['generation.jsonl'])
    audit_file=_file(bundle/'audit.json',protocol['files_sha256']['audit.json']);audit=read_json(bundle/'audit.json')
    if (audit.get('status')!='passed' or audit.get('pool_images')!=RECORDS or
        audit.get('audited_unselected_pool') is not True or audit.get('all_exclusion_pixels_verified') is not True or
        len(audit.get('exclusions',[]))<2 or any(row.get('overlap_images')!=0 for row in audit['exclusions'])):
        raise ValueError('Full-pool/train-exclusion pixel audit is absent or failed')
    records=[public_record(row) for row in _rows(bundle/'generation.jsonl')]
    if len(records)!=RECORDS or len({row['record_id'] for row in records})!=RECORDS:raise ValueError('Caption manifest is incomplete/duplicate')
    images=[_image(row['image_paths'][0],row['image_hashes'][0]) for row in records]
    references=bundle/'references.jsonl'
    if references.is_symlink() or not references.is_file():raise ValueError('Local separate caption references required')
    preparations={path:_file(path,sha) for path,sha in protocol.get('preparation_source_sha256',{}).items()}
    return {'bundle':str(bundle),'records':records,'protocol':protocol,'protocol_file':protocol_file,
        'generation_file':inputs,'audit_file':audit_file,'images':images,'preparation_sources':preparations,
        'references_deferred':{'path':str(references),'stat':stat(references),'sha256':protocol['files_sha256']['references.jsonl']},
        'gold_read_policy':'references and original source manifests are not opened before all predictions are durable'}


def _checkpoint_identity(path):
    from src.utils.research_checkpoint_compat import checkpoint_identity
    return checkpoint_identity(path)


def checkpoint_context(run,role):
    run=Path(run).resolve(strict=True)
    if role not in ('best','last'):raise ValueError('Retained best/last required')
    lock=run/'.checkpoint.lock'
    if not lock.is_file() or lock.is_symlink():raise ValueError('Existing manager lock required')
    with lock.open('rb') as handle:
        fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
        manager=read_json(run/'.checkpoint_manager.json');config=read_json(run/'resolved_config.json');completion=read_json(run/'completed.json')
        compaction=read_json(run/'checkpoint_compaction.json')
        output=Path(config.get('output',''));output=(output if output.is_absolute() else ROOT/output).resolve()
        if (output!=run or config.get('stage')!='SFT' or type(completion.get('steps')) is not int or completion['steps']<1 or
            completion.get('status') in ('failed','incomplete','interrupted')):raise ValueError('Completed managed SFT run required')
        if (not isinstance(manager.get('owner'),str) or not manager['owner'] or compaction.get('status')!='completed' or
            compaction.get('training_completed') is not True or compaction.get('resumable_optimizer') is not False or
            Path(compaction.get('run_dir','')).resolve()!=run):raise ValueError('Completed owned optimizer compaction required')
        roles={}
        for name in ('best','last'):
            pointer=run/name
            if not pointer.is_symlink():raise ValueError('Owned best/last symlinks required')
            target=pointer.resolve(strict=True)
            if target.parent!=run/'.checkpoint_objects' or not target.name.startswith('object-'):raise ValueError('Checkpoint role escaped manager')
            marker=read_json(target/'.research_checkpoint.json')
            if (marker.get('owner')!=manager.get('owner') or marker.get('training_completed') is not True or
                marker.get('resumable_optimizer') is not False or marker.get('optimizer_state_removed') is not True or
                marker.get('created_unix',math.inf)>(run/'completed.json').stat().st_mtime+1):
                raise ValueError('Only completed compacted owned SFT checkpoints accepted')
            if type(marker.get('step')) is not int or not 1<=marker['step']<=completion['steps']:raise ValueError('Checkpoint step is outside completed training')
            if not any(item.get('new_object')==target.name for item in compaction.get('checkpoints',[])):raise ValueError('Owned checkpoint is absent from completed compaction receipt')
            if name=='last' and marker.get('step')!=completion['steps']:raise ValueError('LAST differs from completion')
            roles[name]={'pointer':os.readlink(pointer),'object':str(target),'marker':marker,'marker_file':_file(target/'.research_checkpoint.json')}
        checkpoint=Path(roles[role]['object'])/'state.pt'
        if checkpoint.is_symlink() or not checkpoint.is_file():raise ValueError('Local checkpoint payload required')
        identity=_checkpoint_identity(checkpoint)
        files={name:_file(run/name) for name in ('.checkpoint_manager.json','resolved_config.json','completed.json','checkpoint_compaction.json')}
        for name in ('provenance.json','provenance_latest.json'):
            if (run/name).is_file():files[name]=_file(run/name)
    return {'run':str(run),'role':role,'roles':roles,'checkpoint_path':str(checkpoint),'artifact':identity,'config':config,'files':files,
        'member_stats':{item['path']:stat(item['path']) for item in identity['members']}}


def environment_context():
    packages={}
    for name in ('torch','transformers','peft','bitsandbytes','tokenizers','huggingface-hub','numpy','Pillow'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    allowed=('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','TOKENIZERS_PARALLELISM','PYTORCH_CUDA_ALLOC_CONF')
    return {'python':platform.python_version(),'packages':packages,'allowlisted_environment':{key:os.environ.get(key) for key in allowed}}


def prepare_context(run,bundle,*,checkpoint='best',cache_dir='cache/vision',assets=None):
    data=data_context(bundle);model=checkpoint_context(run,checkpoint)
    declared={item['path']:item['sha256'] for item in data['protocol']['options']['source_manifests']}
    train=Path(model['config'].get('manifest',''));train=(train if train.is_absolute() else ROOT/train).resolve()
    if declared.get(str(train))!=model['config'].get('manifest_sha256'):raise ValueError('SFT train manifest is outside pinned caption exclusion audit')
    if model['config'].get('text_manifest'):raise ValueError('Separate text manifest needs an explicit matching exclusion review')
    asset_path=Path(assets) if assets else ROOT/'configs/research/caption_metric_assets.json'
    scorer_assets=_file(asset_path);source_files={name:_file(ROOT/name) for name in SOURCES}
    static={'schema_version':1,'owner':OWNER,'data_protocol_sha256':data['protocol_file']['sha256'],
        'generation_manifest_sha256':data['generation_file']['sha256'],'data_audit_sha256':data['audit_file']['sha256'],
        'records':RECORDS,'record_order_sha256':digest([row['record_id'] for row in data['records']]),
        'image_census':data['images'],'references_deferred':data['references_deferred'],
        'checkpoint':model,'source_files':source_files,'preparation_sources':data['preparation_sources'],
        'environment':environment_context(),'scorer_assets':scorer_assets,'cache_dir':str(Path(cache_dir).resolve()),
        'generation':{'implementation':'frozen research_inference native B1','batch_size':1,'do_sample':False,'num_beams':1,
            'use_cache':True,'max_new_tokens':MAX_NEW_TOKENS,'context_policy':'native builder enforces full prefill+256 <= min training/model context; no truncation',
            'prompt':'native chat user: fixed caption instruction with image marker; no references',
            'prompt_ids':'actual prefix/suffix IDs observed from frozen builder in this process, without changing tensors or generation arguments',
            'template_clock':'first actual prefix/suffix IDs and exact prompt SHA fixed in immutable prompt_contract.json; every row/resume must match, including date-dependent templates',
            'autocast':'frozen native cuda_bf16 preserving checkpoint parameter dtypes'},
        'scoring':{'metric':'pinned COCO-caption CIDEr only; no VQA scorer','idf':'full372 actual reference-image corpus fixed',
            'bootstrap':{'unit':'image','replicates':1000,'seed':42},'references_opened_only_after_all_predictions':True},
        'scope':'internal pilot captioning holdout; separate caption protocol, not primary VQA evaluation'}
    return {'static':static,'data':data,'records':data['records']}


def assert_context(context,*,rehash=False):
    data=context['data'];static=context['static'];model=static['checkpoint']
    files=[data['protocol_file'],data['generation_file'],data['audit_file'],static['scorer_assets'],
        *static['source_files'].values(),*static['preparation_sources'].values(),*model['files'].values(),
        *(role['marker_file'] for role in model['roles'].values())]
    for item in files:
        if stat(item['path'])!=item['stat'] or (rehash and file_sha(item['path'])!=item['sha256']):raise ValueError('Caption source/provenance file changed: '+item['path'])
    for image in data['images']:
        if stat(image['path'])!=image['stat']:raise ValueError('Caption image changed after RGB census')
    deferred=data['references_deferred']
    if stat(deferred['path'])!=deferred['stat']:raise ValueError('Deferred reference file changed; no gold read attempted')
    for role,value in model['roles'].items():
        pointer=Path(model['run'])/role
        if not pointer.is_symlink() or os.readlink(pointer)!=value['pointer'] or str(pointer.resolve())!=value['object']:
            raise ValueError('Checkpoint best/last role changed')
    for member in model['artifact']['members']:
        if stat(member['path'])!=model['member_stats'][member['path']]:raise ValueError('Checkpoint member changed')
    if rehash and _checkpoint_identity(model['checkpoint_path'])!=model['artifact']:raise ValueError('Checkpoint content identity changed')
    if environment_context()!=static['environment']:raise ValueError('Caption execution environment changed')


class NativeRuntime:
    def load(self,checkpoint,cache_dir,reserve_gb):
        from src.eval.research_inference import load_research_checkpoint
        return load_research_checkpoint(checkpoint,cache_dir=cache_dir,reserve_gb=reserve_gb)
    def describe(self,model,provenance):
        import torch
        maximum=int(getattr(model.args,'max_length',4096));native=int(getattr(model.base.config,'max_position_embeddings',maximum))
        snapshots={}
        for kind,path in (('model',model.model_path),('vision',model.vision_path)):
            path=Path(path).resolve(strict=True)
            snapshots[kind]={'snapshot':str(path),'files':{str(file.relative_to(path)):_file(file) for file in native_asset_files(path)}}
        device=torch.cuda.get_device_properties(0)
        return {'loaded':provenance,'context_limit':min(maximum,native),'master_parameter_dtypes':sorted({str(value.dtype) for value in model.parameters()}),
            'model_type':getattr(model.base.config,'model_type',None),'pinned_snapshots':snapshots,
            'runtime':{'cuda_version':torch.version.cuda,'gpu_name':device.name,'gpu_total_bytes':device.total_memory,
                'gpu_capability':[device.major,device.minor],'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                'cudnn_tf32':torch.backends.cudnn.allow_tf32,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}}
    def generate(self,model,record):
        from unittest.mock import patch
        from src.eval import research_inference as native
        record=public_record(record);original=native.build_generation_inputs;captured=[]
        def observe(actual_model,actual_record,**kwargs):
            if actual_model is not model or actual_record!=record or kwargs.get('max_new_tokens')!=MAX_NEW_TOKENS:raise ValueError('Native caption builder input changed')
            prepared=original(actual_model,actual_record,**kwargs)
            captured.append({key:prepared[key] for key in ('prefix_token_ids','suffix_token_ids','prompt_sha256','visual_tokens','prefill_tokens')})
            return prepared
        with patch.object(native,'build_generation_inputs',observe):generated=native.generate_prediction(model,record,max_new_tokens=MAX_NEW_TOKENS)
        if len(captured)!=1 or captured[0]['prompt_sha256']!=generated['prompt_sha256']:raise ValueError('Actual native prompt-token observation differs')
        return {**generated,**captured[0]}


def gpu_idle():
    value=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
    if value.stdout.strip():raise RuntimeError('GPU occupied; caption worker must run in an exclusive root-scheduled slot')


def _model_provenance(value,context):
    loaded=value.get('loaded',{});config=context['static']['checkpoint']['config'];marker=context['static']['checkpoint']['roles'][context['static']['checkpoint']['role']]['marker']
    if (loaded.get('recipe')!=config['recipe'] or loaded.get('stage')!='SFT' or loaded.get('training_step')!=marker['step'] or
        type(value.get('context_limit')) is not int or value['context_limit']<=MAX_NEW_TOKENS):raise ValueError('Loaded model generation metadata differs')
    for key,field in (('model_ref','model'),('vision_ref','vision')):
        ref=loaded.get(key,{})
        if ref.get('repo')!=config[field] or not re.fullmatch('[0-9a-f]{40,64}',str(ref.get('revision',''))):raise ValueError('Loaded model/vision must be pinned')
    if not HEX.fullmatch(str(loaded.get('cache_fingerprint',''))):raise ValueError('Actual cache fingerprint missing')
    if set(value.get('pinned_snapshots',{}))!={'model','vision'} or any(not item.get('files') for item in value['pinned_snapshots'].values()):raise ValueError('Actual pinned snapshot-file bindings missing')


def native_asset_files(path):
    return [file for file in sorted(Path(path).rglob('*')) if file.is_file() and
        (file.suffix in ('.json','.safetensors','.model','.jinja','.txt') or file.name=='merges.txt')]


def assert_model_files(protocol,*,rehash=False):
    for snapshot in protocol['model_provenance']['pinned_snapshots'].values():
        if set(snapshot['files'])!={str(file.relative_to(snapshot['snapshot'])) for file in native_asset_files(snapshot['snapshot'])}:
            raise ValueError('Pinned model/vision snapshot file census changed')
        for name,item in snapshot['files'].items():
            if str((Path(snapshot['snapshot'])/name).resolve(strict=True))!=item['path']:raise ValueError('Pinned snapshot asset target changed')
            if stat(item['path'])!=item['stat'] or (rehash and file_sha(item['path'])!=item['sha256']):raise ValueError('Pinned model/vision snapshot file changed')


def _row(record,index,generated,protocol):
    ids=generated.get('generated_token_ids');prefix=generated.get('prefix_token_ids');suffix=generated.get('suffix_token_ids')
    if any(not isinstance(value,list) or any(type(token) is not int or token<0 for token in value) for value in (ids,prefix,suffix)):
        raise ValueError('Actual generated/prompt token ID lists required')
    if (not isinstance(generated.get('prediction'),str) or type(generated.get('generated_tokens')) is not int or generated.get('generated_tokens')!=len(ids) or len(ids)>MAX_NEW_TOKENS or
        type(generated.get('visual_tokens')) is not int or generated['visual_tokens']<1 or
        type(generated.get('prefill_tokens')) is not int or generated.get('prefill_tokens')!=len(prefix)+len(suffix)+generated['visual_tokens'] or
        generated['prefill_tokens']+MAX_NEW_TOKENS>protocol['model_provenance']['context_limit'] or
        not HEX.fullmatch(str(generated.get('prompt_sha256','')))):raise ValueError('Caption output/token/context contract differs')
    tokens={'prefix_token_ids':prefix,'suffix_token_ids':suffix,'visual_tokens':generated['visual_tokens'],'prefill_tokens':generated['prefill_tokens']}
    row={'record_id':record['record_id'],'index':index,'public_record_sha256':digest(record),'image_hash':record['image_hashes'][0],
        'prediction':generated['prediction'],'generated_token_ids':ids,'generated_tokens':len(ids),'generated_token_ids_sha256':digest(ids),
        **tokens,'prompt_token_ids_sha256':digest(tokens),'prompt_sha256':generated['prompt_sha256'],
        'logits_to_keep':generated.get('logits_to_keep'),'protocol_sha256':digest(protocol),'model_provenance_sha256':digest(protocol['model_provenance']),
        'prompt_contract_sha256':digest(prompt_contract(generated,protocol))}
    return {**row,'row_sha256':digest(row)}


def prompt_contract(generated,protocol):
    return {'prefix_token_ids':generated['prefix_token_ids'],'suffix_token_ids':generated['suffix_token_ids'],
        'prompt_sha256':generated['prompt_sha256'],'protocol_sha256':digest(protocol)}


def check_prompt_contract(output,protocol,generated=None):
    path=Path(output)/'prompt_contract.json'
    if generated is not None:
        actual=prompt_contract(generated,protocol)
        if path.exists() and read_json(path)!=actual:raise ValueError('Native caption text prompt/template clock drifted from initial contract')
        if not path.exists():_atomic(path,actual)
        return actual
    if not path.is_file():raise ValueError('Committed captions lack their initial prompt-token contract')
    saved=read_json(path)
    if saved.get('protocol_sha256')!=digest(protocol):raise ValueError('Prompt-token contract belongs to a different protocol')
    return saved


def committed_rows(output,records,protocol,*,resume=False):
    directory=Path(output)/'records'
    if directory.is_symlink():raise ValueError('Caption journal directory cannot be a symlink')
    directory.mkdir(exist_ok=True)
    paths=list(directory.iterdir());expected={f'{index:06d}.json' for index in range(len(records))}
    for path in paths:
        match=re.fullmatch(r'\.(\d{6})\.json\.([0-9a-f]{32})\.tmp',path.name)
        if match:
            if (not resume or protocol.get('owner')!=OWNER or int(match[1])>=len(records) or
                path.is_symlink() or not path.is_file() or path.stat().st_size>2**20):
                raise ValueError('Uncommitted caption staging requires explicit owned resume')
            before=stat(path);raw=path.read_bytes()
            if stat(path)!=before:raise ValueError('Uncommitted caption staging changed while preserving')
            sha=hashlib.sha256(raw).hexdigest();recovery=Path(output)/'recovery'
            if recovery.is_symlink():raise ValueError('Caption recovery directory cannot be a symlink')
            recovery.mkdir(exist_ok=True)
            name=path.name+'.'+sha+'.uncommitted';saved=recovery/name
            if saved.is_symlink() or (saved.exists() and saved.read_bytes()!=raw):raise ValueError('Uncommitted staging recovery differs')
            if not saved.exists():_atomic(saved,raw,raw=True)
            receipt=recovery/(name+'.json')
            value={'source_name':path.name,'sha256':sha,'bytes':len(raw),'committed_prediction':False,'protocol_sha256':digest(protocol)}
            if receipt.exists() and read_json(receipt)!=value:raise ValueError('Uncommitted staging receipt differs')
            if not receipt.exists():_atomic(receipt,value)
            if stat(path)!=before:raise ValueError('Uncommitted staging changed before removal')
            path.unlink()
            fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
    paths=list(directory.iterdir())
    if any(path.is_symlink() or not path.is_file() or path.name not in expected for path in paths):raise ValueError('Unexpected/non-owned caption journal entry')
    rows=[]
    for index,record in enumerate(records):
        path=directory/f'{index:06d}.json'
        if not path.exists():
            if any((directory/f'{later:06d}.json').exists() for later in range(index+1,len(records))):raise ValueError('Caption journal is not a contiguous manifest prefix')
            break
        row=read_json(path)
        generated={key:row.get(key) for key in ('prediction','generated_token_ids','generated_tokens','prefix_token_ids','suffix_token_ids','visual_tokens','prefill_tokens','prompt_sha256','logits_to_keep')}
        if row!=_row(record,index,generated,protocol):raise ValueError('Caption row token/prompt/protocol hash differs')
        if prompt_contract(generated,protocol)!=check_prompt_contract(output,protocol):raise ValueError('Committed caption prompt differs from initial template contract')
        rows.append(row)
    return rows


def reconcile_predictions(output,rows,*,resume):
    output=Path(output);path=output/'predictions.jsonl';desired=b''.join(canonical(row)+b'\n' for row in rows)
    if path.is_symlink():raise ValueError('Caption raw predictions cannot be a symlink')
    if not path.exists():
        if rows and not resume:raise ValueError('Durable rows require explicit resume')
        if rows:_atomic(path,desired,raw=True)
        return
    raw=path.read_bytes()
    if len(raw)>32*2**20:raise ValueError('Caption raw predictions bound exceeded')
    if raw==desired:return
    if not resume:raise ValueError('Existing predictions require explicit resume')
    # Appending occurs only AFTER an atomic row commit. Thus an incomplete tail
    # must be the exact byte prefix of the already durable journal projection.
    if not desired.startswith(raw):raise ValueError('Raw predictions diverge from durable caption journal; no repair')
    recovery=output/'recovery'
    if recovery.is_symlink():raise ValueError('Caption recovery directory cannot be a symlink')
    recovery.mkdir(exist_ok=True);sha=hashlib.sha256(raw).hexdigest()
    backup=recovery/(sha+'.jsonl.fragment')
    if backup.is_symlink() or (backup.exists() and backup.read_bytes()!=raw):raise ValueError('Recovery fragment changed')
    if not backup.exists():_atomic(backup,raw,raw=True)
    _atomic(path,desired,raw=True)
    _atomic(recovery/(sha+'.json'),{'old_sha256':sha,'new_sha256':hashlib.sha256(desired).hexdigest(),
        'reason':'rebuild exact durable atomic-row projection; truncated tail/missing suffix preserved','journal_rows':len(rows)})


def _references(context,rows):
    if len(rows)!=RECORDS:raise ValueError('Gold inaccessible until all372 predictions are durable')
    deferred=context['data']['references_deferred']
    if stat(deferred['path'])!=deferred['stat'] or file_sha(deferred['path'])!=deferred['sha256']:raise ValueError('Separate references changed')
    references=_rows(deferred['path']);records=context['records'];lookup={row['record_id']:row for row in references}
    if len(references)!=RECORDS or len(lookup)!=RECORDS or set(lookup)!={row['record_id'] for row in records}:raise ValueError('Reference corpus must exactly cover full372 images')
    result={};seen=set()
    for record in records:
        row=lookup[record['record_id']];actual=row.get('reference_records');refs=row.get('references')
        if (row.get('image_hash')!=record['image_hashes'][0] or not isinstance(actual,list) or not actual or
            not isinstance(refs,list) or refs!=[item.get('reference') for item in actual] or any(not isinstance(value,str) or not value.strip() for value in refs)):
            raise ValueError('Only all actual grouped reference captions are accepted')
        for item in actual:
            source=item.get('source',{});identity=item.get('source_record_id')
            if not isinstance(identity,str) or not identity or identity in seen or not source.get('repo') or not re.fullmatch('[0-9a-f]{40}',str(source.get('revision',''))):raise ValueError('Reference source identity/revision differs')
            seen.add(identity)
        result[record['record_id']]=refs
    if len(seen)!=context['data']['protocol']['reference_records']:raise ValueError('Actual reference count differs from preparation')
    return result


def evaluate(context,output,*,resume=False,allow_gpu=False,score_only=False,runtime=None,reserve_gb=120,progress=None,scorer=None,bootstrap=None):
    """Root schedules production; optional runtime/scorers support CPU unit tests."""
    if not math.isfinite(reserve_gb) or reserve_gb<120:raise ValueError('Preserve120GiB disk reserve')
    output=Path(output).resolve();run=Path(context['static']['checkpoint']['run'])
    bundle=Path(context['data']['bundle'])
    if output==run or output.is_relative_to(run/'.checkpoint_objects') or output==bundle or output.is_relative_to(bundle):raise ValueError('Caption output cannot replace checkpoint/data objects')
    parent=output
    while not parent.exists():parent=parent.parent
    if shutil.disk_usage(parent).free<reserve_gb*2**30+32*2**20:raise OSError('Caption disk reserve reached before staging')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'.caption.lock').is_symlink():raise ValueError('Caption output lock cannot be a symlink')
    with (output/'.caption.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert_context(context,rehash=True)
        protocol_path=output/'protocol.json';previous=read_json(protocol_path) if protocol_path.exists() else None
        if previous is not None and not resume:raise ValueError('Caption results require explicit --resume')
        if previous is not None and previous.get('static')!=context['static']:raise ValueError('Resume checkpoint/data/source/environment protocol differs')
        if previous is not None:
            if previous.get('owner')!=OWNER or previous.get('schema_version')!=1:raise ValueError('Foreign caption protocol')
            _model_provenance(previous.get('model_provenance',{}),context);assert_model_files(previous,rehash=True)
            if (output/'prompt_contract.json').exists():check_prompt_contract(output,previous)
        if previous is None and any(path.name!='.caption.lock' for path in output.iterdir()):raise ValueError('Partial/unowned output lacks protocol; preserve and refuse')
        protocol=previous;rows=committed_rows(output,context['records'],protocol,resume=resume) if protocol else []
        if score_only and len(rows)!=RECORDS:raise ValueError('Score-only requires complete durable predictions; gold remains unopened')
        if len(rows)<RECORDS:
            if not allow_gpu:raise RuntimeError('GPU generation requires explicit root-scheduled --allow-gpu')
            gpu_idle();runtime=runtime or NativeRuntime()
            model,loaded=runtime.load(context['static']['checkpoint']['checkpoint_path'],context['static']['cache_dir'],reserve_gb)
            provenance=runtime.describe(model,loaded);_model_provenance(provenance,context)
            if previous and provenance!=previous.get('model_provenance'):raise ValueError('Resumed loaded model/cache/context provenance differs')
            protocol=previous or {'schema_version':1,'owner':OWNER,'static':context['static'],'model_provenance':provenance}
            if previous is None:_atomic(protocol_path,protocol)
            (output/'records').mkdir(exist_ok=True)
            reconcile_predictions(output,rows,resume=resume)
            for index in range(len(rows),RECORDS):
                if shutil.disk_usage(output).free<reserve_gb*2**30+32*2**20:raise OSError('Caption disk reserve reached')
                assert_context(context);assert_model_files(protocol)
                record=public_record(context['records'][index]);generated=runtime.generate(model,record)
                row=_row(record,index,generated,protocol);assert_context(context);assert_model_files(protocol)
                check_prompt_contract(output,protocol,generated)
                _atomic(output/'records'/f'{index:06d}.json',row)
                with (output/'predictions.jsonl').open('ab') as handle:handle.write(canonical(row)+b'\n');handle.flush();os.fsync(handle.fileno())
                rows.append(row)
                event={'event':'generated','completed_records':len(rows),'records':RECORDS,'record_id':row['record_id']}
                _atomic(output/'progress.json',event)
                if progress:progress(event)
            del model
        else:reconcile_predictions(output,rows,resume=resume)
        rows=committed_rows(output,context['records'],protocol,resume=resume)
        if len(rows)!=RECORDS:raise ValueError('Incomplete caption journal has no score')
        assert_context(context,rehash=True)
        assert_model_files(protocol,rehash=True)
        raw_sha=file_sha(output/'predictions.jsonl')
        _atomic(output/'generation_complete.json',{'owner':OWNER,'records':RECORDS,'protocol_sha256':digest(protocol),'predictions_sha256':raw_sha,
            'checkpoint_sha256':context['static']['checkpoint']['artifact']['sha256'],'gold_opened_for_generation':False})
        references=_references(context,rows)  # FIRST reference-file content read.
        if scorer is None or bootstrap is None:
            from src.eval.research_caption_metrics import score_caption_corpus,image_bootstrap_interval
            scorer=scorer or score_caption_corpus;bootstrap=bootstrap or image_bootstrap_interval
        predictions={row['record_id']:row['prediction'] for row in rows}
        scores=scorer(references,predictions,config_path=context['static']['scorer_assets']['path'])
        if (scores.get('images')!=RECORDS or set(scores.get('per_image',{}))!=set(predictions) or
            any(type(value) not in (float,int) or not math.isfinite(value) for value in scores['per_image'].values())):raise ValueError('Caption scorer did not cover the complete image corpus')
        interval=bootstrap(scores['per_image'],replicates=1000,seed=42)
        assert_context(context,rehash=True)
        summary={'schema_version':1,'owner':OWNER,'complete_manifest':True,'records':RECORDS,'protocol_sha256':digest(protocol),
            'checkpoint_sha256':context['static']['checkpoint']['artifact']['sha256'],'checkpoint_roles':context['static']['checkpoint']['roles'],
            'data_protocol_sha256':DATA_PROTOCOL_SHA256,'generation_manifest_sha256':context['static']['generation_manifest_sha256'],
            'references_sha256':context['data']['references_deferred']['sha256'],'raw_predictions_sha256':raw_sha,
            'prompt_contract_sha256':digest(check_prompt_contract(output,protocol)),
            'caption_scores':scores,'image_bootstrap':interval,'cider_times100_image_95_percentile_ci':[100*value for value in interval['image_95_percentile_ci']],
            'max_new_tokens':MAX_NEW_TOKENS,'no_vqa_score_computed':True,'gold_used_only_after_complete_generation':True,
            'scope':context['static']['scope']}
        if (output/'summary.json').exists() and read_json(output/'summary.json')!=summary:raise ValueError('Existing caption summary differs; no overwrite')
        if not (output/'summary.json').exists():_atomic(output/'summary.json',summary)
        _atomic(output/'progress.json',{'event':'completed','records':RECORDS,'summary_sha256':file_sha(output/'summary.json')})
        return summary
