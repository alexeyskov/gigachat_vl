"""Narrow, recorded compatibility for pinned research configuration metadata.

This helper does not download weights or alter source JSON. It is intentionally
not wired into the training/inference loaders pending their own review.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import transformers
from transformers import AutoConfig


def normalize_research_config_dict(raw):
    """Only repair DeepSeek's documented integer-one/float-one type mismatch.

    No general numeric coercion is performed. Other values and other strict
    configuration errors continue through the native configuration validator.
    """
    normalized=copy.deepcopy(raw)
    changes=[]
    value=raw.get('routed_scaling_factor')
    if raw.get('model_type')=='deepseek_v3' and type(value) is int and value==1:
        normalized['routed_scaling_factor']=1.0
        changes.append({'field':'routed_scaling_factor','source_type':'int','source_value':1,
                        'normalized_type':'float','normalized_value':1.0,
                        'reason':'Native strict DeepseekV3Config requires float; numeric value and routing scale are unchanged.'})
    return normalized,changes


def load_research_native_config(config_path):
    """Load local source bytes with native validation and a normalization receipt."""
    path=Path(config_path)
    source=path.read_bytes()
    raw=json.loads(source)
    normalized,changes=normalize_research_config_dict(raw)
    if raw.get('model_type')=='smallthinker':
        from src.model.smallthinker import register_smallthinker
        register_smallthinker()
    def construct(values):
        values=copy.deepcopy(values)
        return AutoConfig.for_model(values.pop('model_type'),**values)
    original_error=None
    if changes:
        try:
            construct(raw)
        except Exception as error:
            # Only the expected field-level native validation error is handled.
            expected="Field 'routed_scaling_factor' expected float, got int (value: 1)"
            if type(error).__name__!='StrictDataclassFieldValidationError' or str(error.__cause__)!=expected:
                raise
            original_error={'type':type(error).__name__,'message':str(error)}
    config=construct(normalized)
    return config,{'source_path':str(path),'source_sha256':hashlib.sha256(source).hexdigest(),
                   'model_type':raw.get('model_type'),
                   'declared_transformers_version':raw.get('transformers_version'),
                   'installed_transformers_version':transformers.__version__,
                   'normalizations':changes,'original_native_validation_error':original_error,
                   'normalized_native_validation_passed':True}
