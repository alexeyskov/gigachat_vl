"""Correct native Giga turn stopping for external evaluation only.

Training and checkpoint replay keep the historical selection contract. The
corrected external generation has its own explicit source-bound protocol field.
"""
import contextlib
from src.utils import research_giga_payload_bridge_v1 as base
from src.eval.research_giga_eot_v1 import install_generation_stop,definition as stop_definition

def definition():return base.definition()
def implementation():return base.implementation()
def train_main(argv=None):return base.train_main(argv)
def validate_main(argv=None):return base.validate_main(argv)


@contextlib.contextmanager
def corrected_evaluation_clones():
    from src.utils import research_stock_determinism as stock
    native=stock._clone
    def clone(relative,*args,**kwargs):
        module=native(relative,*args,**kwargs)
        if relative=='src/eval/research_inference.py':
            load=module.load_research_checkpoint
            def checkpoint(*a,**k):
                model,provenance=load(*a,**k);install_generation_stop(model)
                return model,dict(provenance,giga_native_turn_stop=stop_definition())
            module.load_research_checkpoint=checkpoint
        if relative=='scripts/eval_research_checkpoint.py':
            main=module.main
            def evaluate(*a,**k):
                binder=module._bind_stock_protocol
                module._bind_stock_protocol=lambda value:dict(binder(value),giga_native_turn_stop=stop_definition())
                try:return main(*a,**k)
                finally:module._bind_stock_protocol=binder
            module.main=evaluate
        return module
    stock._clone=clone
    try:yield
    finally:stock._clone=native


def eval_main(argv=None):
    with corrected_evaluation_clones():return base.eval_main(argv)
