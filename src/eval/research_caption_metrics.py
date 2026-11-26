"""Pinned COCO-caption CIDEr and Stanford PTB, independent of pilot metrics.

Document frequencies always use the complete supplied reference-image corpus.
Bootstrapping resamples its image scores with those frequencies held fixed.
No downloads, fallback tokenizer, model loading or CUDA operations occur here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT/'configs/research/caption_metric_assets.json'
PUNCTUATION = frozenset(["''", "'", '``', '`', '-LRB-', '-RRB-', '-LCB-', '-RCB-',
                         '.', '?', '!', ',', ':', '-', '--', '...', ';'])


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):result.update(data)
    return result.hexdigest()


def assets(config_path=ASSETS):
    path=Path(config_path);config=json.loads(path.read_text())
    if config.get('schema_version')!=1 or config.get('algorithm',{}).get('ngrams')!=4 or config['algorithm'].get('sigma')!=6.:
        raise ValueError('Expected the pinned four-gram COCO-caption CIDEr protocol')
    for key,digest_key in [('scorer_path','scorer_sha256')]:
        if file_sha(config[key])!=config[digest_key]:raise ValueError('Pinned scorer changed')
    for item,path_key,hash_key in [(config['ptb_jar'],'path','sha256'),
            (config['java'],'path','sha256'),(config['java'],'libjvm_path','libjvm_sha256')]:
        if file_sha(item[path_key])!=item[hash_key]:raise ValueError('Pinned PTB/JRE asset changed')
    return config


def ptb_tokenize(captions, *, config_path=ASSETS):
    if not isinstance(captions,list) or any(not isinstance(x,str) for x in captions):
        raise ValueError('Caption list must contain strings')
    if len(captions)>100000 or sum(len(x.encode('utf8')) for x in captions)>20*2**20:
        raise ValueError('Bounded caption tokenizer input exceeded')
    if not captions:return []
    config=assets(config_path)
    # Universal newlines are one input sentence; gold/prediction text otherwise
    # stays intact. The final delimiter preserves an empty final prediction.
    normalized=[x.replace('\r\n','\n').replace('\r','\n').replace('\n',' ') for x in captions]
    with tempfile.TemporaryDirectory(prefix='research-caption-ptb-') as directory:
        path=Path(directory)/'captions.txt';path.write_text('\n'.join(normalized)+'\n',encoding='utf8')
        command=[config['java']['path'],'-Xms16m','-Xmx256m','-XX:+UseSerialGC',
            '-Dfile.encoding=UTF-8','-cp',config['ptb_jar']['path'],
            'edu.stanford.nlp.process.PTBTokenizer','-preserveLines','-lowerCase',str(path)]
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120,check=True)
    lines=result.stdout.decode('utf8').split('\n')
    if len(lines)==len(captions)+1 and lines[-1]=='':lines.pop()
    if len(lines)!=len(captions):raise ValueError('PTB output lost or added a caption line')
    return [' '.join(word for word in line.rstrip().split(' ') if word not in PUNCTUATION) for line in lines]


def _scorer(config):
    spec=importlib.util.spec_from_file_location('research_pinned_coco_cider',config['scorer_path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.CiderScorer(n=4,sigma=6.)


def score_caption_corpus(references,predictions,*,config_path=ASSETS):
    """One prediction and all actual references per unique image, native0..10."""
    if (not isinstance(references,dict) or len(references)<2 or
            any(not isinstance(k,str) or not k for k in references) or
            not isinstance(predictions,dict) or set(predictions)!=set(references)):
        raise ValueError('At least two reference-image IDs and exactly matching predictions are required')
    ids=sorted(references)
    if any(not isinstance(references[k],list) or not references[k] or
            any(not isinstance(x,str) or not x.strip() for x in references[k]) for k in ids):
        raise ValueError('Every image requires at least one actual nonempty reference')
    if any(not isinstance(predictions[k],str) for k in ids):raise ValueError('Predictions must be strings, including empty outputs')
    config=assets(config_path);flat=[];positions={}
    for key in ids:
        start=len(flat);flat.extend(references[key]);positions[key]=(start,len(flat))
    nrefs=len(flat);flat.extend(predictions[key] for key in ids)
    tokenized=ptb_tokenize(flat,config_path=config_path)
    if not any(tokenized[:nrefs]):raise ValueError('Reference corpus has no PTB content tokens')
    scorer=_scorer(config)
    tokenized_refs={};tokenized_predictions={}
    for index,key in enumerate(ids):
        start,end=positions[key];tokenized_refs[key]=tokenized[start:end]
        tokenized_predictions[key]=tokenized[nrefs+index]
        scorer+=(tokenized_predictions[key],tokenized_refs[key])
    mean,scores=scorer.compute_score();values=[float(x) for x in scores]
    if not math.isfinite(float(mean)) or any(not math.isfinite(x) or not -1e-10<=x<=10+1e-10 for x in values):
        raise FloatingPointError('CIDEr returned an invalid image score')
    return {'schema_version':1,'metric':'COCO-caption CIDEr','images':len(ids),
        'references_per_image':{key:len(references[key]) for key in ids},
        'single_reference_only':all(len(references[key])==1 for key in ids),
        'cider_raw':float(mean),'raw_scale':'0-10','cider_times100':100*float(mean),
        'times100_scale':'0-1000; explicit reporting conversion, not VQA accuracy',
        'per_image':dict(zip(ids,values)),'references_sha256':digest(references),
        'predictions_sha256':digest(predictions),'tokenized_references_sha256':digest(tokenized_refs),
        'tokenized_predictions_sha256':digest(tokenized_predictions),'assets_sha256':file_sha(config_path),
        'scorer_sha256':config['scorer_sha256'],'tokenizer_jar_sha256':config['ptb_jar']['sha256'],
        'numpy_version':__import__('numpy').__version__,
        'idf_policy':'fixed full supplied reference-image corpus; no prediction-derived IDF',
        'tokenizer':'Stanford PTB3.4.1 preserveLines lowerCase; original COCO punctuation filter',
        'input_newlines':'CRLF/CR/LF converted to spaces before PTB; final empty outputs retained',
        'checkpoint_binding':'not established by this scorer; generation owner must bind actual predictions'}


def image_bootstrap_interval(scores,*,replicates=1000,seed=42):
    if (not isinstance(scores,dict) or len(scores)<2 or type(replicates) is not int or replicates<1000 or
            type(seed) is not int or any(type(x) not in (int,float) or not math.isfinite(x) for x in scores.values())):
        raise ValueError('Finite per-image scores and at least1000 bootstrap replicates are required')
    values=[scores[key] for key in sorted(scores)];rng=random.Random(seed)
    draws=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(replicates))
    return {'mean':sum(values)/len(values),
        'image_95_percentile_ci':[draws[math.floor(.025*(replicates-1))],draws[math.ceil(.975*(replicates-1))]],
        'replicates':replicates,'seed':seed,'unit':'unique canonical image',
        'idf_policy':'original full-corpus IDF fixed; per-image scores resampled',
        'cross_seed_inference':False}
