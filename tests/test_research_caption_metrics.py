"""Real pinned PTB/JRE integration and independent original-scorer parity."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np
import pytest

from src.eval import research_caption_metrics as metrics


@pytest.fixture(scope='module')
def config():
    # Missing assets fail explicitly; a fallback tokenizer cannot certify this.
    return metrics.assets()


def original_scorer(config):
    path=metrics.ROOT/'third_party/research_caption/original_cider_scorer_py2.py'
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==config['sources'][path.name]['sha256']
    source=raw.decode().replace('xrange(','range(').replace('.iteritems()','.items()')
    module=types.ModuleType('original_coco_cider_py3_syntax_only')
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module.CiderScorer(n=4,sigma=6.)


def upstream_tokenize(captions,config,tmp_path,monkeypatch):
    path=metrics.ROOT/'third_party/research_caption/ptbtokenizer_reference.py'
    assert metrics.file_sha(path)==config['sources'][path.name]['sha256']
    spec=importlib.util.spec_from_file_location('upstream_ptb_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.__file__=str(tmp_path/'ptbtokenizer_reference.py')
    module.STANFORD_CORENLP_3_4_1_JAR=config['ptb_jar']['path']
    monkeypatch.setenv('PATH',str(Path(config['java']['path']).parent)+os.pathsep+os.environ['PATH'])
    normalized=[x.replace('\r\n','\n').replace('\r','\n').replace('\n',' ') for x in captions]
    values={str(i):[{'caption':text}] for i,text in enumerate(normalized)}
    result=module.PTBTokenizer().tokenize(values)
    return [result[str(i)][0] for i in range(len(values))]


def test_real_stanford_tokenizer_matches_pinned_wrapper(config,tmp_path,monkeypatch):
    captions=["A child's dog (small) can't jump.",'CAFÉ — bright colors!',
              'A cat\r\nsits\ron the\nred mat.','He said "hello"; then left...','', 'Blue sky.']
    assert metrics.ptb_tokenize(captions)==upstream_tokenize(captions,config,tmp_path,monkeypatch)


def test_empty_final_predictions_preserve_all_lines(config):
    assert metrics.ptb_tokenize(['','',''])==['','','']
    assert metrics.ptb_tokenize(['A dog.','',''])==['a dog','','']


def test_exact_original_cider_numerics_with_fixed_multireference_corpus(config):
    rng=random.Random(401);words=['cat','dog','red','blue','sits','jumps','under','over','chair','table','grass','park']
    refs={str(i):[' '.join(rng.choices(words,k=length)) for length in (5,8,12)] for i in range(12)}
    predictions={str(i):' '.join(rng.choices(words,k=(i%9)+1)) if i else '' for i in range(12)}
    actual=metrics.score_caption_corpus(refs,predictions)
    tokenized=metrics.ptb_tokenize([text for k in sorted(refs) for text in refs[k]]+[predictions[k] for k in sorted(refs)])
    scorer=original_scorer(config);count=3*len(refs)
    for i,k in enumerate(sorted(refs)):scorer+=(tokenized[count+i],tokenized[3*i:3*i+3])
    mean,scores=scorer.compute_score()
    assert actual['cider_raw']==float(mean)
    assert list(actual['per_image'].values())==[float(x) for x in scores]
    assert actual['single_reference_only'] is False


def test_perfect_zero_and_no_accumulating_document_frequency(config):
    refs={'cat':['A cat sits on a bright red mat.'],'dog':['A dog plays with a big blue ball.']}
    exact={key:value[0] for key,value in refs.items()}
    first=metrics.score_caption_corpus(refs,exact);second=metrics.score_caption_corpus(refs,exact)
    assert first==second
    assert first['cider_raw']==pytest.approx(10.,abs=1e-12)
    assert first['cider_times100']==pytest.approx(1000.,abs=1e-10)
    empty=metrics.score_caption_corpus(refs,{key:'' for key in refs})
    assert empty['cider_raw']==0.
    assert empty['references_sha256']==first['references_sha256']
    changed=metrics.score_caption_corpus(refs,dict(exact,dog='A completely unrelated airplane.'))
    assert changed['per_image']['cat']==first['per_image']['cat']
    assert changed['tokenized_references_sha256']==first['tokenized_references_sha256']


@pytest.mark.parametrize('refs,predictions',[
    ({'one':['A cat.']},{'one':'A cat.'}),
    ({'one':['A cat.'],'two':['A dog.']},{'one':'A cat.'}),
    ({'one':'A cat.','two':['A dog.']},{'one':'A cat.','two':'A dog.'}),
    ({'one':[''],'two':['A dog.']},{'one':'','two':''}),
    ({'one':['A cat.'],'two':['A dog.']},{'one':None,'two':'A dog.'}),
])
def test_invalid_or_incomplete_reference_corpus_refused(refs,predictions):
    with pytest.raises(ValueError):metrics.score_caption_corpus(refs,predictions)


def test_changed_scorer_fails_before_any_tokenizer_fallback(config,tmp_path,monkeypatch):
    altered=dict(config);path=tmp_path/'cider.py';path.write_text('changed')
    altered['scorer_path']=str(path);cfg=tmp_path/'assets.json';cfg.write_text(json.dumps(altered))
    monkeypatch.setattr(metrics.subprocess,'run',lambda *a,**k:pytest.fail('No tokenizer may run after source mutation'))
    with pytest.raises(ValueError,match='scorer changed'):metrics.ptb_tokenize(['A dog'],config_path=cfg)


def test_bootstrap_preserves_fixed_corpus_scores_and_seed():
    values={'a':0.,'b':2.,'c':4.,'d':8.};first=metrics.image_bootstrap_interval(values)
    assert first==metrics.image_bootstrap_interval(values)
    assert first['mean']==3.5
    assert first['image_95_percentile_ci'][0]<=3.5<=first['image_95_percentile_ci'][1]
    assert 'IDF fixed' in first['idf_policy']
    with pytest.raises(ValueError):metrics.image_bootstrap_interval(values,replicates=20)


def test_import_is_cpu_only_and_does_not_load_training_stack():
    result=subprocess.run([sys.executable,'-c',
        "import sys; import src.eval.research_caption_metrics; assert 'torch' not in sys.modules"],
        cwd=metrics.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
