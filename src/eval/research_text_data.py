"""Immutable English retention-suite rows and tokenizer-only context census.

No model generation, training, external judging or contamination-clearance claim.
Selections depend only on pinned source IDs and input text, never references.
"""
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

NAME = 'primary_text_en_v1'
SEED = 42
FIXED_DATE = datetime(2026, 11, 22, tzinfo=timezone.utc)
TASKS = ('gsm8k', 'mmlu', 'ifeval')
COUNTS = {'gsm8k': 1319, 'mmlu': 14042, 'ifeval': 541}
CAPS = {'gsm8k': 512, 'mmlu': 912, 'ifeval': 256}
GENERATION = {'gsm8k': 1024, 'mmlu': 0, 'ifeval': 2048}
REPOS = {'gsm8k': ('openai/gsm8k', '740312add88f781978c0658806c59bc2815b9866', 'main', 'test'),
    'mmlu': ('cais/mmlu', 'c30699e8356da336a370243923dbaf21066bb9fe', 'all', 'test'),
    'ifeval': ('google/IFEval', '966cd89545d6b6acfd7638bc708b98261ca58e84', 'default', 'train')}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(raw): return hashlib.sha256(raw if isinstance(raw, bytes) else raw.encode()).hexdigest()
def value_sha(value): return digest(canonical(value))
def normalise(text): return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
def jsonl(rows): return ''.join(canonical(row) + '\n' for row in rows).encode()


def input_text(task, row):
    if task == 'ifeval': return row['prompt']
    question = row['question']
    return question + ('\n' + '\n'.join(f'{chr(65+i)}. {v}' for i, v in enumerate(row['choices'])) if task == 'mmlu' else '')


def source_id(task, row, index, filename):
    if task == 'ifeval': return 'key:' + str(row['key'])
    return f'{filename}#{index}:' + digest(input_text(task, row))


def selection_key(source):
    text = f"{NAME}|42|{source['repo']}@{source['revision']}|{source['config']}|{source['split']}|{source['original_id']}"
    return digest(text), source['original_id']


def validate_raw_rows(task, rows, subjects):
    if task not in TASKS or len(rows) != COUNTS[task]: raise ValueError('Released source row count differs')
    seen = set()
    for row in rows:
        if task == 'mmlu' and (not isinstance(row.get('question'), str) or
                not isinstance(row.get('choices'), list) or len(row['choices']) != 4 or
                any(not isinstance(v, str) for v in row['choices'])):
            raise ValueError('MMLU question/options schema differs')
        text = input_text(task, row)
        if not isinstance(text, str) or not text.strip(): raise ValueError('Empty released question/prompt')
        if task == 'gsm8k':
            if not isinstance(row.get('answer'), str): raise ValueError('GSM8K solution missing')
            numeric_reference(row['answer'])
        elif task == 'mmlu':
            if (row.get('subject') not in subjects or not isinstance(row.get('choices'), list) or
                    len(row['choices']) != 4 or any(not isinstance(v, str) for v in row['choices']) or
                    type(row.get('answer')) is not int or not 0 <= row['answer'] < 4):
                raise ValueError('MMLU subject/options/answer schema differs')
        else:
            if (type(row.get('key')) is not int or row['key'] in seen or
                    not isinstance(row.get('instruction_id_list'), list) or not row['instruction_id_list'] or
                    not isinstance(row.get('kwargs'), list) or len(row['kwargs']) != len(row['instruction_id_list']) or
                    any(not isinstance(x, str) or not x for x in row['instruction_id_list']) or
                    any(not isinstance(x, dict) for x in row['kwargs'])):
                raise ValueError('IFEval key/instruction/kwargs schema differs')
            seen.add(row['key'])
    if task == 'mmlu' and set(row['subject'] for row in rows) != set(subjects):
        raise ValueError('MMLU does not cover exactly the57 declared subjects')


def numeric_reference(answer):
    matches = re.findall(r'^####\s*([^\n]+)\s*$', answer, re.M)
    if len(matches) != 1: raise ValueError('GSM8K gold final answer is ambiguous/missing')
    try: value = Decimal(matches[0].strip().replace(',', ''))
    except InvalidOperation as error: raise ValueError('GSM8K gold numeric answer malformed') from error
    if not value.is_finite(): raise ValueError('GSM8K gold answer is not finite')
    return str(value.normalize())


def build_pool(task, rows, asset, subjects):
    validate_raw_rows(task, rows, subjects)
    repo, revision, config, split = REPOS[task]
    if asset['repo'] != repo or asset['revision'] != revision:
        raise ValueError('Source repository/revision differs from preregistered source')
    result = []
    for index, raw in enumerate(rows):
        original = source_id(task, raw, index, asset['filename'])
        source = {'repo': repo, 'revision': revision, 'config': config, 'split': split,
            'filename': asset['filename'], 'row_index': index, 'original_id': original,
            'source_file_sha256': asset['sha256']}
        text = input_text(task, raw)
        if task == 'gsm8k':
            prompt = raw['question'] + '\nSolve the problem. You may show your reasoning. Finish with one final line exactly: #### <number>.'
            reference = {'numeric_answer': numeric_reference(raw['answer']), 'reference_solution': raw['answer']}
        elif task == 'mmlu':
            prompt = 'The following is a multiple choice question about ' + raw['subject'].replace('_', ' ') + '.\n' + text + '\nAnswer with the correct letter only.'
            reference = {'answer_index': raw['answer'], 'answer_letter': chr(65 + raw['answer'])}
        else:
            prompt = raw['prompt']; reference = {'instruction_id_list': raw['instruction_id_list'], 'kwargs': raw['kwargs']}
        result.append({'record_id': digest(f'{repo}@{revision}|{config}|{split}|{original}'),
            'task': task, 'family': {'gsm8k': 'reasoning', 'mmlu': 'knowledge', 'ifeval': 'instruction'}[task],
            'subject': raw.get('subject'), 'source': source, 'question': raw.get('question', raw.get('prompt')),
            'input_text': text, 'prompt': prompt, 'image_paths': [], 'image_hashes': [],
            'reference': reference, 'max_new_tokens': GENERATION[task]})
    return result


def select_records(pools, subjects):
    selected = []
    for task in TASKS:
        pool = pools[task]
        if task == 'mmlu':
            chosen = set()
            for subject in sorted(subjects):
                candidates = [r for r in pool if r['subject'] == subject]
                if len(candidates) < 16: raise ValueError('Insufficient MMLU subject rows')
                chosen.update(r['record_id'] for r in sorted(candidates, key=lambda r: selection_key(r['source']))[:16])
        else:
            chosen = {r['record_id'] for r in sorted(pool, key=lambda r: selection_key(r['source']))[:CAPS[task]]}
        selected.extend(r for r in pool if r['record_id'] in chosen)
    if Counter(r['task'] for r in selected) != Counter(CAPS): raise ValueError('Selected suite counts differ')
    return selected


def exclusion_rows(pools):
    result = []
    for task in TASKS:
        for row in pools[task]:
            question = row['question']; full = row['input_text']
            result.append({'record_id': row['record_id'], 'task': task, 'source': row['source'],
                'heldout_only_despite_nominal_train_split': task == 'ifeval',
                'question_text': question, 'input_text': full,
                'raw_question_sha256': digest(question), 'raw_input_sha256': digest(full),
                'normalised_question_sha256': digest(normalise(question)),
                'normalised_input_sha256': digest(normalise(full)),
                'normalisation': 'NFKC+casefold+whitespace_collapse',
                'training_overlap_screen_performed': False})
    return result


def public_model_input(record):
    return {'messages': [{'role': 'user', 'content': record['prompt']}], 'image_paths': [], 'image_hashes': []}


def render_prompt(tokenizer, record, model_id):
    kwargs = {'date_string': '2026-11-22', 'current_date': '2026-11-22',
              'strftime_now': lambda fmt: FIXED_DATE.strftime(fmt)}
    prompt = tokenizer.apply_chat_template(public_model_input(record)['messages'], tokenize=False,
                                            add_generation_prompt=True, **kwargs)
    if model_id == 'gptoss20':
        # Same direct final-channel convention as the frozen research inference.
        prompt += '<|channel|>final<|message|>'
    if '[RESEARCH_IMAGE]' in prompt: raise ValueError('Text-only prompt contains a fake image marker')
    return prompt


def context_census(records, tokenizer, model_id, native_limit):
    measured = []
    candidate_lengths = {letter: len(tokenizer.encode(letter, add_special_tokens=False)) for letter in 'ABCD'}
    if min(candidate_lengths.values()) < 1: raise ValueError('MMLU answer candidate tokenization is empty')
    for row in records:
        prompt = render_prompt(tokenizer, row, model_id)
        ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
        continuations = {letter: tokenizer.encode(prompt + letter, add_special_tokens=False, truncation=False)
                         for letter in 'ABCD'} if row['task'] == 'mmlu' else {}
        if any(full[:len(ids)] != ids for full in continuations.values()):
            raise ValueError('MMLU candidate changes prompt token prefix; scorer protocol needs explicit revision')
        continuation_lengths = {letter: len(full) - len(ids) for letter, full in continuations.items()}
        reserve = max(continuation_lengths.values()) if continuations else row['max_new_tokens']
        if reserve < 1: raise ValueError('Candidate/generation reserve must be positive')
        measured.append({'record_id': row['record_id'], 'task': row['task'], 'prompt_tokens': len(ids),
            'prompt_sha256': digest(prompt), 'prompt_token_ids_sha256': value_sha(ids),
            'generation_or_candidate_reserve': reserve, 'required_context_tokens': len(ids) + reserve,
            **({'candidate_continuation_token_lengths': continuation_lengths} if continuations else {})})
    return {'model_id': model_id, 'native_context_limit': native_limit,
        'MMLU_candidate_token_lengths': candidate_lengths, 'records': measured,
        'max_prompt_tokens': max(r['prompt_tokens'] for r in measured),
        'max_required_context_tokens': max(r['required_context_tokens'] for r in measured),
        'per_task': {task: {'records': sum(r['task'] == task for r in measured),
            'max_prompt_tokens': max(r['prompt_tokens'] for r in measured if r['task'] == task),
            'max_required_context_tokens': max(r['required_context_tokens'] for r in measured if r['task'] == task)}
            for task in TASKS}}


def choose_context(censuses, proposed=4096):
    if len(censuses) != 4: raise ValueError('Exactly four pinned backbone tokenizer censuses are required')
    required = max(c['max_required_context_tokens'] for c in censuses)
    chosen = proposed
    while chosen < required: chosen *= 2
    if chosen > min(c['native_context_limit'] for c in censuses):
        raise ValueError('Common context requirement exceeds a declared native capacity; no silent row drop')
    return {'proposed_context_limit': proposed, 'chosen_context_limit': chosen,
        'max_required_context_tokens': required, 'changed_from_proposal': chosen != proposed,
        'basis': 'actual tokenizer lengths before any model scores; budgets/rows unchanged',
        'all_selected_records_fit': True, 'GPU_memory_fit_proven': False}


def write_immutable_bundle(output, files, protocol, *, reserve_bytes=120 * 2**30):
    output = Path(output)
    if 'protocol.json' in files or any(Path(name).name != name for name in files):
        raise ValueError('Invalid/reserved bundle member name')
    if output.is_symlink(): raise ValueError('Output may not be a symlink')
    hashes = {name: digest(raw) for name, raw in files.items()}
    payload = dict(protocol, files_sha256=hashes)
    expected = dict(files, **{'protocol.json': (canonical(payload) + '\n').encode()})
    if output.exists():
        if not output.is_dir() or {p.name for p in output.iterdir()} != set(expected):
            raise ValueError('Existing immutable bundle membership differs')
        if any((output/name).is_symlink() or (output/name).read_bytes() != raw for name, raw in expected.items()):
            raise ValueError('Existing immutable bundle bytes/protocol differ')
        return payload
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free - sum(len(v) for v in expected.values()) < reserve_bytes:
        raise RuntimeError('Insufficient disk above the declared reserve')
    stage = Path(tempfile.mkdtemp(prefix='.'+output.name+'.', dir=output.parent))
    try:
        for name, raw in expected.items():
            with (stage/name).open('xb') as handle: handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        if output.exists(): raise ValueError('Immutable output appeared during write')
        stage.rename(output)
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return payload
