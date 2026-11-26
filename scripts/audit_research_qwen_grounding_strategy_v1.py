#!/usr/bin/env python3
"""CPU-only primary-source and local token-supervision audit for grounding."""
from __future__ import annotations

import ast
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'research_results/qwen_detection_strategy_v1'


def source_checks():
    from src.utils.research_detection_training_v1 import sha

    refs = json.loads((OUT / 'references.json').read_text())
    files = {}
    for item in refs['sources']:
        if item['status'] != 'downloaded_reference_only':
            continue
        path = OUT / 'sources' / item['file']
        if sha(path) != item['sha256']:
            raise ValueError('Primary reference snapshot changed')
        files[item['file']] = path.read_text()
    train = ast.parse(files['qwen3_vl_train.py'])
    standard_trainer = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                           and n.func.id == 'Trainer' for n in ast.walk(train))
    model = ast.parse(files['qwen3_5_modeling.py'])
    model_loss_calls = sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                           and n.func.attr == 'loss_function' for n in ast.walk(model))
    loss_tree = ast.parse(files['transformers_loss.py'])
    mappings = next(n.value for n in loss_tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'LOSS_MAPPING' for t in n.targets))
    loss_mapping = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(mappings.keys, mappings.values)}
    ce = next(n for n in loss_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fixed_cross_entropy')
    explicit_ce = any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'nn.functional.cross_entropy'
                      for n in ast.walk(ce))
    if not (standard_trainer and model_loss_calls and explicit_ce
            and loss_mapping['ForConditionalGeneration'] == 'ForCausalLMLoss'):
        raise ValueError('Primary exported SFT implementation differs from inspected CE route')
    return dict(reference_sha256=sha(OUT / 'references.json'), revisions=refs['github_revisions'],
                qwen3_VL_standard_Trainer=standard_trainer,
                qwen3_5_model_loss_function_calls=model_loss_calls,
                exported_conditional_generation_loss=loss_mapping['ForConditionalGeneration'],
                terminal_loss_is_cross_entropy=explicit_ce,
                scope='Public/exported SFT path; not proof of undisclosed production pretraining or RL objectives')


def main():
    import torch
    from scripts.research_detection_mix_queue_v1 import build_plan
    from src.eval.research_grounding_loss_audit_v1 import token_fields, summarize
    from src.utils import research_detection_training_v1 as training
    from src.utils.research_small_budget_v3 import fixed_clock
    from src.utils.research_storage import _atomic_json

    checked = source_checks()
    training.validate_release()
    jobs = build_plan()['jobs']
    records = training.load_rows(training.DATA / 'detection_train.jsonl')
    reports = []
    for key in ('smallthinker4', 'lfm25', 'giga31'):
        job = next(j for j in jobs if j['model_key'] == key)
        args = training.configure_args(job)
        module, _, _ = training.backend(args)
        snapshot, _ = module.resolve_model(args.model)
        if key == 'giga31':
            from src.model.research_giga_loading_v2 import tokenizer_proxy
            tokenizer = tokenizer_proxy().from_pretrained(snapshot, local_files_only=True)
        else:
            module.register_smallthinker()
            tokenizer = module.AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        tokenizer = fixed_clock(tokenizer)
        fake = SimpleNamespace(tokenizer=tokenizer, args=args,
                               base=SimpleNamespace(config=SimpleNamespace(model_type=key)))
        rows = []
        for record in records:
            pre, post, ids = module.ResearchModel.tokenize(fake, record)
            messages = module.training_prompt_messages(record, len(record['image_paths']), args.prompt_schema_version)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages + [dict(role='assistant', content=record['answer'])],
                                                 tokenize=False, add_generation_prompt=False)
            if not full.startswith(prompt):
                raise ValueError('Native prompt and answer do not align')
            suffix = full[len(prompt):]
            # FixedClockTokenizer forwards attributes, while Python special
            # method lookup bypasses __getattr__ for the tokenizer(...) syntax.
            encoded = tokenizer.__call__(suffix, add_special_tokens=False, return_offsets_mapping=True)
            if encoded['input_ids'] != ids:
                raise ValueError('Audited supervision differs from native training targets')
            if len(pre) + len(post) + 729 + len(ids) > args.max_length:
                raise ValueError('Coordinate targets would be truncated')
            row = token_fields(record['answer'], suffix, encoded['offset_mapping'])
            rows.append(dict(row, query_kind=record['detection']['query_kind']))
        reports.append(dict(model=key, **summarize(rows),
                            by_query_kind={kind: summarize([r for r in rows if r['query_kind'] == kind])
                                           for kind in sorted({r['query_kind'] for r in rows})}))
        print(json.dumps({k: reports[-1][k] for k in ('model', 'records', 'coordinate_token_fraction',
                                                     'all_coordinate_digits_covered')}), flush=True)
    if torch.cuda.is_initialized():
        raise ValueError('Read-only audit unexpectedly initialized CUDA')
    report = dict(status='passed', updated_UTC=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                  primary_source_checks=checked, training_token_audit=reports,
                  audit_source_sha256={name: training.sha(ROOT / name) for name in (
                      'scripts/audit_research_qwen_grounding_strategy_v1.py',
                      'src/eval/research_grounding_loss_audit_v1.py',
                      'tests/test_research_grounding_loss_audit_v1.py')},
                  current_release_sha256=training.sha(training.DATA / 'release.json'),
                  current_training_definition=training.definition(), CUDA_initialized=False,
                  decision='Keep existing assistant-only token CE; no evidenced Qwen-specific IoU/GIoU bbox loss to substitute',
                  square_root_reweighting='Mentioned in Qwen3-VL report, exact coefficients/denominator unpublished; do not invent an exact reproduction',
                  current_GPU_worker_stopped=False, training_numerics_changed=False)
    _atomic_json(OUT / 'report.json', report)
    print('Primary-source CE route and all1350 native detection targets audited without GPU use')


if __name__ == '__main__':
    main()
