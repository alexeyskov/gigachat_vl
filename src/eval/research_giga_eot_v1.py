"""Explicit native Giga chat turn termination; no model or routing changes."""
from __future__ import annotations
import hashlib
from pathlib import Path
MODEL='ai-sage/GigaChat3.1-10B-A1.8B-bf16'
REVISION='189fff27a1dee68473960c3d5bca53e0e07a3191'
EOT_TEXT='<|message_sep|>\n\n'
EOT_ID=128001


def definition():
    return dict(owner='native_giga_assistant_turn_stop_v1',model=MODEL,revision=REVISION,
        assistant_turn_end_id=EOT_ID,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_selection_protocol_preserved=True)


def validate_tokenizer(tokenizer):
    ids=tokenizer.encode(EOT_TEXT,add_special_tokens=False)
    if ids!=[EOT_ID]:raise ValueError('Pinned Giga native turn marker differs')
    full=tokenizer.apply_chat_template([{'role':'user','content':'Question?'},
        {'role':'assistant','content':'Answer.'}],tokenize=False,add_generation_prompt=False)
    if not full.endswith(EOT_TEXT):raise ValueError('Native assistant target does not terminate with the expected marker')
    return EOT_ID


def terminated_prefix(ids,tokenizer):
    stop=validate_tokenizer(tokenizer)
    end=ids.index(stop) if stop in ids else len(ids)
    prefix=ids[:end]
    return dict(prediction=tokenizer.decode(prefix,skip_special_tokens=True).strip(),
        generated_token_ids=prefix,native_turn_end_found=end<len(ids),discarded_tokens_after_native_turn_end=max(0,len(ids)-end-1))


def install_generation_stop(model):
    if model.model_ref!={'repo':MODEL,'revision':REVISION}:raise ValueError('Pinned native Giga model required')
    validate_tokenizer(model.tokenizer);base=model.base
    if getattr(base,'_research_giga_eot_v1',False):return
    native=base.generate
    def generate(*args,**kwargs):
        previous=kwargs.get('eos_token_id',base.generation_config.eos_token_id)
        values=[] if previous is None else list(previous) if isinstance(previous,(list,tuple)) else [previous]
        kwargs['eos_token_id']=list(dict.fromkeys([*values,EOT_ID]))
        return native(*args,**kwargs)
    base.generate=generate;base._research_giga_eot_v1=True
