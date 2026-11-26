#!/usr/bin/env python3
"""Prepare pinned primary_text_en_v1 offline, without model weights or CUDA.

Input assets are individually source-verified by the bounded metadata/tokenizer
download receipt. Reruns validate the existing immutable bundle byte for byte.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

# Set these before Transformers imports: this program never needs Torch.
os.environ['USE_TORCH'] = '0'
os.environ['USE_TF'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# src.eval.__init__ has optional heavy registrations; load the pure module only.
import importlib.util
_spec = importlib.util.spec_from_file_location('research_text_data', ROOT/'src/eval/research_text_data.py')
data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data)

MODELS = {
    'smallthinker4': ('Tiiny/SmallThinker-4BA0.6B-Instruct', 'a24932bb880c85c04c794283202a9194b1b42709'),
    'lfm25': ('LiquidAI/LFM2.5-8B-A1B', 'dd83f5cb02aa2938b66bba9a9998d38d0517c1f3'),
    'giga31': ('ai-sage/GigaChat3.1-10B-A1.8B-bf16', '189fff27a1dee68473960c3d5bca53e0e07a3191'),
    'gptoss20': ('openai/gpt-oss-20b', '6cee5e81ee83917806bbde320786a8fb61efebee'),
}
DATA_FILES = {'gsm8k': 'main/test-00000-of-00001.parquet',
              'mmlu': 'all/test-00000-of-00001.parquet', 'ifeval': 'ifeval_input_data.jsonl'}
TOKENIZER_FILES = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'chat_template.jinja', 'vocab.json', 'merges.txt',
    'tokenizer.model', 'added_tokens.json'}


def load_json(path): return json.loads(Path(path).read_text())
def sha_file(path): return data.digest(Path(path).read_bytes())


def verify_assets(path):
    receipt = load_json(path)
    if receipt.get('status') != 'passed' or receipt.get('schema_version') != 1:
        raise ValueError('Asset download has not completed successfully')
    cap = 250 * 2**20
    files = receipt.get('files', [])
    new_bytes = sum(a['bytes'] for a in files if not a['cached_before'])
    if (receipt.get('new_content_bytes') != new_bytes or receipt.get('cap_bytes') != cap or
            new_bytes + receipt.get('metadata_allowance_bytes', cap) > cap):
        raise ValueError('Download accounting exceeds/differs from authorized 250 MiB cap')
    seen = set()
    for asset in files:
        key = asset['repo_type'], asset['repo'], asset['revision'], asset['filename']
        if key in seen: raise ValueError('Duplicate source asset')
        seen.add(key)
        if asset['kind'] == 'tokenizer':
            if (asset['id'] not in MODELS or (asset['repo'], asset['revision']) != MODELS[asset['id']] or
                    asset['repo_type'] != 'model' or not (asset['filename'] in TOKENIZER_FILES or
                    (asset['filename'].startswith('chat_templates/') and asset['filename'].endswith('.jinja')))):
                raise ValueError('Unapproved tokenizer/model-weight asset')
        elif asset['kind'] == 'dataset':
            tasks = [t for t in data.TASKS if (asset['repo'], asset['revision']) == data.REPOS[t][:2]]
            if len(tasks) != 1 or asset['repo_type'] != 'dataset' or asset['filename'] != DATA_FILES[tasks[0]]:
                raise ValueError('Unexpected dataset asset')
        else: raise ValueError('Unexpected asset kind')
        raw = Path(asset['path']).read_bytes()
        sha = data.digest(raw)
        if len(raw) != asset['bytes'] or sha != asset['sha256']:
            raise ValueError('Local source asset bytes differ from receipt')
        declared = asset.get('declared_sha256')
        if declared and sha != declared: raise ValueError('Asset differs from HF LFS SHA256')
        if not declared:
            blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            if blob != asset.get('git_blob_sha1'): raise ValueError('Asset differs from pinned Git blob')
    return receipt


def prepare(args):
    receipt = verify_assets(args.assets)
    metadata = load_json(args.mmlu_metadata)
    if metadata.get('sha') != data.REPOS['mmlu'][1]: raise ValueError('MMLU metadata revision differs')
    subjects = sorted(info['config_name'] for info in metadata['cardData']['dataset_info']
                      if info['config_name'] not in {'all', 'auxiliary_train'})
    if len(set(subjects)) != 57: raise ValueError('MMLU official metadata must declare57 subjects')
    from pyarrow import parquet
    from transformers import AutoTokenizer, PreTrainedConfig
    pools, sources = {}, {}
    for task in data.TASKS:
        candidates = [a for a in receipt['files'] if a['kind'] == 'dataset' and a['repo'] == data.REPOS[task][0]]
        if len(candidates) != 1: raise ValueError('Exactly one source asset per task is required')
        asset = candidates[0]; sources[task] = asset
        if task == 'ifeval': rows = [json.loads(line) for line in Path(asset['path']).read_text().splitlines() if line]
        else: rows = parquet.read_table(asset['path']).to_pylist()
        pools[task] = data.build_pool(task, rows, asset, subjects)
    records = data.select_records(pools, subjects)
    exclusions = data.exclusion_rows(pools)
    if len({r['record_id'] for r in exclusions}) != sum(data.COUNTS.values()):
        raise ValueError('Released source IDs are not unique')
    censuses = []
    for model_id, (repo, revision) in MODELS.items():
        assets = [a for a in receipt['files'] if a['kind'] == 'tokenizer' and a['id'] == model_id]
        configs = [a for a in assets if a['filename'] == 'config.json']
        if len(configs) != 1 or not {'tokenizer.json', 'tokenizer_config.json'} <= {a['filename'] for a in assets}:
            raise ValueError('Pinned tokenizer assets incomplete')
        config = load_json(configs[0]['path'])
        native_limit = config.get('max_position_embeddings')
        if type(native_limit) is not int or native_limit <= 0: raise ValueError('Native context capacity missing')
        snapshot = Path(configs[0]['path']).parent
        tokenizer_kwargs = {}
        if model_id == 'giga31':
            # The released config has routed_scaling_factor=1 (int), rejected by
            # installed strict DeepSeek config validation. Tokenizer dispatch
            # needs model_type only; preserve source bytes and avoid creating a
            # language-model config. Native model-loading normalization is not
            # part of this tokenizer-only census.
            generic = PreTrainedConfig()
            generic.model_type = config['model_type']
            tokenizer_kwargs['config'] = generic
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
            trust_remote_code=False, **tokenizer_kwargs)
        census = data.context_census(records, tokenizer, model_id, native_limit)
        census.update({'repo': repo, 'revision': revision, 'tokenizer_class': type(tokenizer).__name__,
            'chat_template_sha256': data.value_sha(tokenizer.chat_template),
            'tokenizer_assets_sha256': {a['filename']: a['sha256'] for a in assets},
            'config_dispatch': 'metadata-only generic config model_type; source unchanged' if model_id=='giga31' else 'native AutoTokenizer',
            'prompt_rendering': {'native_chat_template': True, 'add_generation_prompt': True,
                'fixed_date': '2026-11-22', 'date_kwargs': ['date_string', 'current_date', 'strftime_now'],
                'gpt_final_channel_suffix': model_id == 'gptoss20', 'truncation': False}})
        censuses.append(census)
        print(json.dumps({'event': 'tokenizer_census', 'model': model_id,
            'max_prompt_tokens': census['max_prompt_tokens'],
            'max_required_context_tokens': census['max_required_context_tokens']}), flush=True)
    context = data.choose_context(censuses)
    # Transformers 5.15 imports Torch during tokenizer dispatcher import even
    # with USE_TORCH=0. Record that honestly; no CUDA initialization is allowed.
    torch = sys.modules.get('torch')
    cuda_initialized = bool(torch is not None and torch.cuda.is_initialized())
    if cuda_initialized: raise RuntimeError('Tokenizer-only preparation initialized CUDA')
    source_code = ['src/eval/research_text_data.py', 'scripts/prepare_research_text_eval.py',
                   'tests/test_research_text_data.py']
    code_hashes = {p: sha_file(ROOT/p) for p in source_code}
    audit = {'schema_version': 1, 'status': 'data_and_context_ready_scorers_and_primary_contamination_pending',
        'selected_counts': dict(Counter(r['task'] for r in records)), 'selected_total': len(records),
        'released_test_exclusion_counts': dict(Counter(r['task'] for r in exclusions)),
        'released_test_exclusion_total': len(exclusions), 'mmlu_subject_counts': dict(Counter(
            r['subject'] for r in records if r['task'] == 'mmlu')), 'all_mmlu_subjects_16': True,
        'source_schemas_checked_all_rows': True, 'source_SHA_and_HF_content_identifiers_verified': True,
        'selection_uses_references': False, 'input_gold_filter': 'public_model_input explicit allowlist',
        'primary_training_manifests_exist_and_screened': False, 'pretraining_contamination_cleared': False,
        'download': {'new_content_bytes': receipt['new_content_bytes'],
            'metadata_allowance_bytes': receipt['metadata_allowance_bytes'], 'cap_bytes': receipt['cap_bytes'],
            'receipt_sha256': sha_file(args.assets), 'weights_downloaded': False},
        'GPU_used': False, 'torch_imported': torch is not None, 'cuda_initialized':cuda_initialized,
        'CUDA_visible_devices':'empty; GPU unavailable to preparation', 'context': context}
    protocol = {'schema_version': 1, 'name': data.NAME, 'seed': data.SEED,
        'status': audit['status'], 'source_sha256': code_hashes,
        'source_assets': {t: {k: a[k] for k in ('repo','revision','filename','bytes','sha256','git_blob_sha1','declared_sha256')}
            for t, a in sources.items()}, 'mmlu_metadata_sha256': sha_file(args.mmlu_metadata),
        'selection': {'algorithm': 'ascending SHA256(name|42|repo@revision|config|split|original_id), tie original_id',
            'references_excluded': True, 'caps': data.CAPS, 'mmlu_per_subject': 16,
            'final_order': 'task GSM8K,MMLU,IFEval; original source file row order within task',
            'few_shot_examples': 0, 'task_choice_preceded_first_text_scores': True},
        'model_input': {'allowlist': ['messages.role=user', 'messages.content=prompt', 'image_paths=[]','image_hashes=[]'],
            'gold_references': 'evaluator only; never pass full row as template context',
            'native_chat_template': True, 'fixed_clock': '2026-11-22 UTC', 'gpt_channel': 'final'},
        'context': context,
        'evaluation': {'generation': {'gsm8k': {'max_new_tokens':1024,'do_sample':False},
            'ifeval': {'max_new_tokens':2048,'do_sample':False}},
            'mmlu': {'mode':'zero_shot_conditional_logprob_of_A_B_C_D', 'length_normalization':False,
                'choice_order':['A','B','C','D'],'tie_break':'first in A,B,C,D','prefix_stability_checked':True},
            'scorers_implemented':False, 'expected_scorers': {
                'gsm8k':'numeric_exact_match_final_anchored_hashes_line',
                'mmlu':'subject_macro_accuracy_and_all_record_accuracy',
                'ifeval':'official_strict_and_loose_prompt_and_instruction_accuracy'},
            'aggregation':'report tasks separately; no cross-task composite',
            'baseline_precision':'separate BF16 and NF4; never compare NF4 SFT only with BF16 baseline',
            'checkpoints':'baseline_before_vision,A0,A1_when_present,SFT; same row IDs/prompt policy'},
        'training_exclusion': {'all_released_rows_protected': True, 'IFEval_train_split_is_heldout':True,
            'index':'released_test_exclusion.jsonl','required_future_checks':['exact raw input/question SHA256',
                'NFKC+casefold+whitespace input/question SHA256','near text/long substring overlap audit'],
            'primary_train_screen_complete':False,'pretraining_contamination_cleared':False},
        'software':{p:importlib.metadata.version(p) for p in ('transformers','tokenizers','pyarrow','jinja2')},
        'proof': {'model_weights_loaded':False,'new_weight_download_bytes':0,
            'torch_imported':torch is not None,'cuda_initialized':cuda_initialized,
            'CUDA_VISIBLE_DEVICES':'','GPU_used':False}}
    files = {'records.jsonl': data.jsonl(records),
        **{task+'.jsonl':data.jsonl([r for r in records if r['task']==task]) for task in data.TASKS},
        'released_test_exclusion.jsonl':data.jsonl(exclusions),
        'tokenizer_census.json':(data.canonical({'schema_version':1,'models':censuses,'context':context})+'\n').encode(),
        'preparation_audit.json':(data.canonical(audit)+'\n').encode()}
    # No replacing a competing writer: process-wide advisory lock surrounds check+rename.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent/('.'+output.name+'.prepare.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = data.write_immutable_bundle(output, files, protocol)
    print(json.dumps({'status':audit['status'],'output':str(output.resolve()), 'selected':len(records),
        'exclusion_rows':len(exclusions),'context':context,'protocol_sha256':sha_file(output/'protocol.json')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', default='datasets/research/text-eval-inputs/assets.json')
    parser.add_argument('--mmlu-metadata', default='research_results/primary_readiness_sources/hf_cais__mmlu.json')
    parser.add_argument('--output', default='datasets/research/primary-text-en-v1')
    prepare(parser.parse_args())


if __name__ == '__main__': main()
