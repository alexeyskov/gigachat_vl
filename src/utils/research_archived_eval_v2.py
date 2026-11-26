"""Phased admission around the sealed single-decode archive loader.

Decoder scratch and constructor scratch have disjoint lifetimes. This profile
charges archive file pages plus the full frozen decoder bound before decoding;
the original actual-headroom constructor checks run after torch.load returns.
All 32 GiB floors, physical proofs and tensor checks remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType

ROOT=Path(__file__).resolve().parents[2]
OWNER='owned_small_archived_evaluation_phased_v2'
PARENT='src/utils/research_archived_eval_v1.py'
PARENT_SHA='c79265a3047c8aa4f4f293dddeee892cc069f55060261f8c500fdd590f9204dd'
PARENT_READY=ROOT/'configs/research/archived_eval_v1_ready.json'
PARENT_READY_SHA='5258d5b454d77c87c349ad2bb3819143d65abaff701669b2f271e534434c2738'
OWN=('src/utils/research_archived_eval_v2.py','tests/test_research_archived_eval_v2.py')
READY=ROOT/'configs/research/archived_eval_v2_ready.json'
_MODULE=None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_parent():
    if sha(ROOT/PARENT)!=PARENT_SHA or sha(PARENT_READY)!=PARENT_READY_SHA:
        raise ValueError('Sealed single-decode parent release changed')
    release=json.loads(PARENT_READY.read_bytes())
    if any(sha(ROOT/name)!=expected for name,expected in release['source_sha256'].items()):
        raise ValueError('Sealed archive parent dependency changed')


def implementation():
    global _MODULE
    verify_parent()
    if _MODULE is not None:return _MODULE
    module=ModuleType('_private_phased_archived_eval_v2')
    module.__file__=str(ROOT/PARENT);module.__package__='src.utils'
    exec(compile((ROOT/PARENT).read_bytes(),module.__file__,'exec'),module.__dict__)
    module.OWNER=OWNER;module.READY=READY;module.OWN=(*module.OWN,*OWN)
    original_sources=module.source_binding
    def sources():
        verify_parent()
        return {**original_sources(),str(PARENT_READY.relative_to(ROOT)):PARENT_READY_SHA}
    module.source_binding=sources
    original_profile=module.execution_profile
    def profile():
        return dict(original_profile(),parent_CPU_release_sha256=PARENT_READY_SHA,
            lifetime_admission='decode_then_signature_then_constructor',
            constructor_allowance_charged_with_decoder_scratch=False,
            initial_charge='full_frozen_decoder_bound_plus_archive_file_pages',
            post_decode_charge='actual_remaining_headroom_plus_original_constructor_allowance',
            allocator_or_clean_file_cache_release_credit_bytes=0)
    module.execution_profile=profile
    original_metadata=module._metadata_unlocked
    def metadata(*args,**kwargs):
        value=original_metadata(*args,**kwargs)
        for transaction in value['transactions'].values():
            simultaneous=(transaction['decoder_extra_bytes']+transaction['archive_file_bytes']+
                          module.CONSTRUCTOR_ALLOWANCE)
            if transaction['load_extra_bytes']!=simultaneous:
                raise ValueError('Pinned parent lifetime charge structure changed')
            transaction['load_extra_bytes']=transaction['decoder_extra_bytes']+transaction['archive_file_bytes']
            transaction['constructor_phase_extra_bytes']=module.CONSTRUCTOR_ALLOWANCE+module.METADATA_LIMIT
            transaction['constructor_checked_after_complete_torch_load']=True
        return value
    module._metadata_unlocked=metadata
    _MODULE=module
    return module


def source_binding():return implementation().source_binding()
def execution_profile():return implementation().execution_profile()
def inspect(*args,**kwargs):return implementation().inspect(*args,**kwargs)
def checkpoint_context(*args,**kwargs):return implementation().checkpoint_context(*args,**kwargs)
def assert_context(*args,**kwargs):return implementation().assert_context(*args,**kwargs)
def load_payload(*args,**kwargs):return implementation().load_payload(*args,**kwargs)
def inference(*args,**kwargs):return implementation().inference(*args,**kwargs)
def verify_release():return implementation().verify_release()
