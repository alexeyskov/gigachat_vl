"""Explicit v2 data binding around a byte-pinned, privately scoped v1 owner.

Generation, token capture, journaling, gold ordering and scoring execute the
unmodified v1 implementation. Only metadata constants live in a private module
dictionary; canonical v1 globals and its existing protocols are never changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from types import ModuleType

ROOT=Path(__file__).resolve().parents[2]
OWNER='research_caption_native_b1_v2'
V1_SOURCE='src/eval/research_caption_generation.py'
V1_SOURCE_SHA256='754c261266ab3544d81560abd90447d6abc2ad6b74b987321d007815407abefa'
SOURCE='src/eval/research_caption_generation_v2.py'
CLI='scripts/eval_research_caption_checkpoint_v2.py'
PROFILE='caption_pilot_en_v2_signal_safe'
DATA_PROTOCOL='research_caption_pilot_eval_v2'
DATA_PROTOCOL_SHA256='af605a57cdfa89deb99a98aa071f02bc4ddd3975fb196ff16f083287bbdb1aa6'
RECORDS=372
DEFAULT_BUNDLE=ROOT/'datasets/research/caption-eval-pilot-en-v2'
DEFAULT_OUTPUT='caption_pilot_en_v2'


class ProfileNotReady(ValueError):
    pass


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def profile_status():
    pinned=isinstance(DATA_PROTOCOL_SHA256,str) and re.fullmatch('[0-9a-f]{64}',DATA_PROTOCOL_SHA256) is not None
    return {'status':'pinned_cpu_proposal' if pinned else 'closed_waiting_ROOT_real_v2_bundle_pin',
        'data_protocol_sha256':DATA_PROTOCOL_SHA256,'profile':PROFILE,'records':RECORDS,
        'actual_GPU_generation_validated':False,'owner':OWNER}


def _private_owner():
    if profile_status()['status'].startswith('closed'):raise ProfileNotReady('ROOT actual full-v2 bundle pin is not published; no gold/model/GPU work permitted')
    path=ROOT/V1_SOURCE;raw=path.read_bytes()
    if _sha(raw)!=V1_SOURCE_SHA256:raise ValueError('Frozen v1 caption implementation SHA differs')
    # exec is restricted to the known local source digest. It creates distinct
    # function-global dictionaries, not a patch of src.eval.research_caption_generation.
    module=ModuleType('src.eval._caption_v2_private_frozen_v1')
    module.__file__=str(path);module.__package__='src.eval'
    exec(compile(raw,str(path),'exec'),module.__dict__)
    if path.read_bytes()!=raw:raise ValueError('Frozen v1 caption source changed during private loading')
    module.DATA_PROTOCOL_SHA256=DATA_PROTOCOL_SHA256;module.RECORDS=RECORDS;module.OWNER=OWNER
    module.SOURCES=tuple(dict.fromkeys((*module.SOURCES,SOURCE,CLI)))
    return module


def _binding():
    return {'owner':OWNER,'data_protocol':DATA_PROTOCOL,'preparation_profile':PROFILE,
        'data_protocol_sha256':DATA_PROTOCOL_SHA256,'reused_frozen_v1_source':V1_SOURCE,
        'reused_frozen_v1_sha256':V1_SOURCE_SHA256,'version_wrapper_source':SOURCE,'version_wrapper_cli':CLI,
        'metadata_scope':'private module dictionary; canonical v1 globals never patched',
        'native_generation_math':'unchanged frozen v1 owner +native inference; greedy B1 max_new_tokens256',
        'default_output_namespace':DEFAULT_OUTPUT}


def _verify_bundle(owner,bundle):
    bundle=Path(bundle).resolve(strict=True);path=bundle/'protocol.json'
    value=owner.read_json(path)
    if owner.file_sha(path)!=DATA_PROTOCOL_SHA256:raise ValueError('Explicit v2 bundle pin differs; no v1 retagging')
    if (value.get('protocol')!=DATA_PROTOCOL or value.get('preparation_profile')!=PROFILE or
        value.get('status')!='complete' or value.get('records')!=RECORDS):raise ValueError('Only the actual pinned signal-safe v2 preparation profile is accepted')
    preparations=value.get('preparation_source_sha256',{})
    for name in ('src/eval/research_caption_data_v2.py','scripts/prepare_research_caption_eval_v2.py'):
        path=ROOT/name
        if preparations.get(str(path))!=owner.file_sha(path):raise ValueError('V2 bundle is not bound to its explicit isolated preparation sources')


def prepare_context(run,bundle=DEFAULT_BUNDLE,*,checkpoint='best',cache_dir='cache/vision',assets=None):
    owner=_private_owner();_verify_bundle(owner,bundle)
    context=owner.prepare_context(run,bundle,checkpoint=checkpoint,cache_dir=cache_dir,assets=assets)
    context['static']['version_wrapper']=_binding()
    _verify_context(owner,context)
    return context


def _verify_context(owner,context):
    static=context['static']
    if (static.get('owner')!=OWNER or static.get('data_protocol_sha256')!=DATA_PROTOCOL_SHA256 or
        static.get('version_wrapper')!=_binding()):raise ValueError('Caption context belongs to a different version/data profile')
    for name in (V1_SOURCE,SOURCE,CLI):
        item=static.get('source_files',{}).get(name,{})
        if item.get('sha256')!=owner.file_sha(ROOT/name):raise ValueError('Version wrapper/native source binding differs')
    if static['source_files'][V1_SOURCE]['sha256']!=V1_SOURCE_SHA256:raise ValueError('Reused v1 code is not the frozen implementation')


def evaluate(context,output=None,**kwargs):
    owner=_private_owner();_verify_context(owner,context);_verify_bundle(owner,context['data']['bundle'])
    output=Path(output) if output is not None else Path(context['static']['checkpoint']['run'])/'evaluation'/DEFAULT_OUTPUT
    if output.name=='caption_pilot_en_v1':raise ValueError('V2 cannot publish into the legacy v1 output namespace')
    return owner.evaluate(context,output,**kwargs)


def protocol_digest(value):
    return _sha(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
