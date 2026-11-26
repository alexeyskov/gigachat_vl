#!/usr/bin/env python3
"""Resume audited pilot SFT after fixed caption warmups and weighted-loss tests.

This one-time local handoff replaces only the named research Supervisor service.
It never changes a running trainer, dataset bytes, old configs or old checkpoints.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.research_storage import DiskBudget,_atomic_json
from scripts.research_queue import validate_spec,argv_value,artifacts_for

SPEC='configs/research/smallthinker_pilot_scienceqa_context_v2_balanced_queue.json'
WARMUPS='configs/research/smallthinker_caption_warmup_precompute_queue.json'
MIXTURE='datasets/research/pilot-en-scienceqa-context-v2-mixture'
READY='configs/research/weighted_sft_ready.json'
PREFLIGHT='research_results/context_v2_token_preflight.json'
RECEIPT='research_results/sft_balance_handoff.json'
CHECKS=('weighted_largebatch_gradient_equivalence','weighted_oom_retry_equivalence',
        'weighted_legacy_default_resume','weighted_missing_stream','schema2_system_prompt')


def emit(event,**fields):
    print(json.dumps(dict(event=event,time=time.time(),**fields)),flush=True)


def validate_sft_spec(spec):
    validate_spec(spec)
    sft=[j for j in spec['jobs'] if argv_value(j['argv'],'--stage')=='SFT']
    recipes={argv_value(j['argv'],'--recipe') for j in sft}
    expected={'1a','1b','1c','2a','2b','2c','3a','3b','3c','4q','4f'}
    if len(sft)!=11 or recipes!=expected:raise ValueError('Expected all eleven SFT branches')
    by_id={job['id']:job for job in spec['jobs']}
    for job in sft:
        argv=job['argv']
        dependencies=job.get('requires',[])
        predecessor=by_id.get(dependencies[0],{}) if len(dependencies)==1 else {}
        parent_argv=predecessor.get('argv',[])
        if (argv_value(argv,'--text-loss-weight')!='0.15' or
                argv_value(argv,'--selection-metric')!='group_macro' or
                argv_value(argv,'--selection-per-group')!='32' or
                argv_value(argv,'--selection-max-new-tokens')!='64' or
                argv_value(argv,'--max-length') not in (None,'4096') or '--resume' in argv or
                argv_value(argv,'--output')!='runs/research/'+job['id'] or
                argv_value(parent_argv,'--stage')!='A1' or
                argv_value(parent_argv,'--recipe')!=argv_value(argv,'--recipe') or
                argv_value(argv,'--initial-checkpoint')!=str(argv_value(parent_argv,'--output'))+'/best/state.pt' or
                argv_value(argv,'--manifest')!=MIXTURE+'/sft_train.jsonl' or
                argv_value(argv,'--validation')!=MIXTURE+'/sft_validation.jsonl' or
                'scienceqa_context_v2_balanced' not in job['id']):
            raise ValueError('SFT branch does not use the identical audited weighted-loss profile')
    return spec


def current_tokenizer_binding():
    from scripts.filter_research_context import make_native_tokenizer
    _,binding=make_native_tokenizer()
    return binding


def validate_code_proof(root,proof,preflight):
    sources=proof.get('source_sha256',{})
    if (not isinstance(sources,dict) or 'scripts/run_research_stage.py' not in sources or
            any(proof.get('checks',{}).get(name) is not True for name in CHECKS) or
            not proof.get('test_command') or not isinstance(proof.get('tests_passed'),int) or proof['tests_passed']<=0):
        raise ValueError('Weighted-loss readiness lacks tested-source/regression evidence')
    for name,expected in sources.items():
        path=root/name
        if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Weighted-loss source changed after tests: '+name)
    actual=current_tokenizer_binding()
    for key in ('model_ref','prompt_schema_version','tokenizer_asset_sha256','tokenize_source_sha256'):
        if preflight.get(key)!=actual.get(key):raise ValueError('Tokenizer/preflight binding changed: '+key)


def validate_warmup_state(root,state):
    spec=validate_spec(json.loads((root/WARMUPS).read_text()))
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if state.get('spec_sha256')!=digest:raise ValueError('Warmup state belongs to a different specification')
    for job in spec['jobs']:
        if state.get('jobs',{}).get(job['id'],{}).get('status')!='completed':
            raise ValueError('Warmup job is not completed: '+job['id'])
        output=Path(argv_value(job['argv'],'--output'))
        output=(root/output).resolve() if not output.is_absolute() else output.resolve()
        completion=json.loads((output/'completed.json').read_text())
        owner=json.loads((output/'.checkpoint_manager.json').read_text())['owner']
        for label in ('best','last'):
            pointer=output/label
            if not pointer.is_symlink():raise ValueError('Warmup lacks an owned checkpoint pointer')
            target=pointer.resolve()
            if target.parent!=(output/'.checkpoint_objects').resolve():raise ValueError('Warmup checkpoint escapes its object store')
            marker=json.loads((target/'.research_checkpoint.json').read_text())
            if marker.get('owner')!=owner or not (target/'state.pt').is_file():raise ValueError('Warmup checkpoint ownership/payload mismatch')
            if label=='last' and marker.get('step')!=completion.get('steps'):raise ValueError('Warmup completion/checkpoint step differs')


def readiness(root,*,recovering_started=False):
    root=Path(root)
    required=[SPEC,READY,PREFLIGHT,MIXTURE+'/mix_audit.json',WARMUPS,
              str(Path(WARMUPS).with_suffix('.state.json'))]
    missing=[name for name in required if not (root/name).is_file()]
    if missing:return None,missing
    proof=json.loads((root/READY).read_text())
    preflight=json.loads((root/PREFLIGHT).read_text())
    if proof.get('status')!='passed' or preflight.get('status')!='passed':
        raise ValueError('Weighted-loss or context preflight has not passed')
    if (preflight.get('prompt_schema_version')!=2 or preflight.get('max_length')!=4096 or
            preflight.get('fixed_visual_tokens_per_image')!=729):
        raise ValueError('Context proof uses a different tokenizer/context profile')
    validate_code_proof(root,proof,preflight)
    warmup=json.loads((root/Path(WARMUPS).with_suffix('.state.json')).read_text())
    if not recovering_started:
        if warmup.get('status')=='failed':raise ValueError('Caption warmup failed; inspect its logs')
        if warmup.get('status')!='completed':return None,['caption_warmups_not_completed']
        validate_warmup_state(root,warmup)
    spec=validate_sft_spec(json.loads((root/SPEC).read_text()))
    mixture=json.loads((root/MIXTURE/'mix_audit.json').read_text())
    hashes={}
    for split in ('train','validation'):
        path=root/MIXTURE/f'sft_{split}.jsonl'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=mixture['splits'][split]['sha256']:
            raise ValueError('Mixture changed since its audit')
        token_stats=preflight.get('reports',{}).get(path.name,{})
        if (token_stats.get('manifest_sha256')!=digest or
                token_stats.get('expanded_prompt_overflow')!=[] or
                token_stats.get('answer_truncated_records')!=0):
            raise ValueError('Context preflight is stale or reports truncated/overflow records')
        hashes[split]=digest
    return {'queue_spec_sha256':hashlib.sha256((root/SPEC).read_bytes()).hexdigest(),
            'mixture_sha256':hashes,'code_readiness_sha256':hashlib.sha256((root/READY).read_bytes()).hexdigest(),
            'context_preflight_sha256':hashlib.sha256((root/PREFLIGHT).read_bytes()).hexdigest(),
            'spec':spec},[]


def service_status(name):
    result=subprocess.run(['supervisorctl','status',name],capture_output=True,text=True,check=False)
    return result.stdout.strip()


def wait_for_running(name,timeout=30):
    deadline=time.monotonic()+timeout
    while True:
        status=service_status(name)
        if 'RUNNING' in status:return status
        if 'STARTING' not in status or time.monotonic()>=deadline:
            raise RuntimeError('Research service did not become RUNNING: '+status)
        time.sleep(.5)


def replacement_is_running(status,configuration):
    command='command=/venv/main/bin/python -u scripts/research_queue.py --spec '+SPEC
    if command not in configuration.splitlines():return False
    match=re.search(r'\bpid\s+(\d+)\b',status)
    if not match:return False
    try:argv=Path('/proc')/match[1]/'cmdline';arguments=argv.read_bytes().split(b'\0')
    except OSError:return False
    return b'scripts/research_queue.py' in arguments and SPEC.encode() in arguments


def main():
    receipt=ROOT/RECEIPT
    previous=json.loads(receipt.read_text()) if receipt.exists() else {}
    if previous.get('status')=='completed':return
    while True:
        DiskBudget(ROOT,120).ensure_space(2*1024**3)
        status=service_status('research-pilot')
        target=Path('/etc/supervisor/conf.d/research-pilot.conf')
        recovering=(previous.get('status')=='starting' and 'RUNNING' in status and
                    replacement_is_running(status,target.read_text()))
        ready,pending=readiness(ROOT,recovering_started=recovering)
        if ready:
            if recovering:
                for key in ('queue_spec_sha256','mixture_sha256','code_readiness_sha256','context_preflight_sha256'):
                    if previous.get(key)!=ready[key]:raise ValueError('Started handoff identity changed: '+key)
                _atomic_json(receipt,dict(previous,status='completed',recovered_after_start=True))
                emit('sft_balance_handoff_recovered',receipt=str(receipt))
                return
            if 'RUNNING' in service_status('research-caption-warmup'):
                pending=['caption_controller_still_exiting']
            else:
                # No CUDA task may be displaced by a background handoff.
                gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                                   capture_output=True,text=True,check=True)
                if gpu.stdout.strip():pending=['cuda_process_still_active']
            if not pending:break
        emit('sft_balance_handoff_waiting',pending=pending)
        time.sleep(15)
    status=service_status('research-pilot')
    target=Path('/etc/supervisor/conf.d/research-pilot.conf')
    original=target.read_text()
    ready_fields={key:value for key,value in ready.items() if key!='spec'}
    if 'RUNNING' in status:
        if (previous.get('status')=='starting' and previous.get('queue_spec_sha256')==ready['queue_spec_sha256']
                and replacement_is_running(status,original)):
            _atomic_json(receipt,dict(previous,status='completed',recovered_after_start=True))
            emit('sft_balance_handoff_recovered',receipt=str(receipt))
            return
        raise RuntimeError('Source pilot unexpectedly running; refusing concurrent launch')
    # These checkpoints finished before the SFT recipe change and remain valid.
    for job in json.loads((ROOT/WARMUPS).read_text())['jobs']:
        artifacts_for(job)
    backup=ROOT/'research_results/sft_balance_source_supervisor.conf'
    if not backup.exists():backup.write_text(original)
    command='command=/venv/main/bin/python -u scripts/research_queue.py --spec '+SPEC
    lines=original.splitlines()
    if sum(line.startswith('command=') for line in lines)!=1:raise ValueError('Unexpected Supervisor command layout')
    _atomic_json(receipt,dict(status='starting',time=time.time(),
                             old_incomplete_sft_retained_as_diagnostic=True,**ready_fields))
    target.write_text('\n'.join(command if line.startswith('command=') else line for line in lines)+'\n')
    emit('sft_balance_handoff_starting',queue=SPEC)
    subprocess.run(['supervisorctl','reread'],check=True)
    subprocess.run(['supervisorctl','update','research-pilot'],check=True)
    status=service_status('research-pilot')
    if 'STOPPED' in status or 'EXITED' in status:
        subprocess.run(['supervisorctl','start','research-pilot'],check=True)
        status=service_status('research-pilot')
    if 'RUNNING' not in status:status=wait_for_running('research-pilot')
    _atomic_json(receipt,dict(status='completed',time=time.time(),reason='fixed4kcontext_and_explicit_multimodal_text_loss_weights',
                             old_incomplete_sft_retained_as_diagnostic=True,**ready_fields))
    emit('sft_balance_handoff_completed',receipt=str(receipt))


if __name__=='__main__':main()
