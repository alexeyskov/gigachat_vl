import importlib.util
from pathlib import Path
import sys
import json
import io

import pytest
import torch
from PIL import Image

# The research data/cache modules are intentionally usable without loading model code.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dataset.vision_cache import VisionFeatureCache, encoder_fingerprint, image_content_hash
from src.dataset.research_data import prepare_from_specs, partition_for_group


def image_file(tmp_path, name, color):
    path=tmp_path/name
    Image.new('RGB',(7,9),color=color).save(path)
    return path


def test_cache_only_unique_misses_and_reuse_across_order(tmp_path):
    a=image_file(tmp_path,'a.png','red'); b=image_file(tmp_path,'b.png','blue')
    calls=[]
    def encode(paths):
        calls.append(list(paths))
        with torch.inference_mode():
            return [torch.full((3,4),float(i+1)) for i,p in enumerate(paths)]
    cache=VisionFeatureCache(tmp_path/'cache',encoder_fingerprint({'encoder':'x','revision':'sha'}),reserve_bytes=0,ram_max_bytes=0)
    result=cache.get_many([str(a),str(a)],encode)
    assert len(calls)==1 and len(calls[0])==1
    assert torch.equal(result[0],result[1])
    layer=torch.nn.Linear(4,2)
    layer(result[0]).sum().backward()  # inference tensor must not leak into autograd
    cache.get_many([str(a),str(b)],encode)
    assert len(calls)==2 and calls[-1]==[str(b)]
    cache.get_many([str(b),str(a)],encode,read_only=True)
    assert len(calls)==2
    assert cache.stats['encoded_images']==2


def test_cache_fingerprint_and_corruption_fail(tmp_path):
    p=image_file(tmp_path,'a.png','red')
    f1=encoder_fingerprint({'encoder':'x','revision':'one'})
    f2=encoder_fingerprint({'encoder':'x','revision':'two'})
    assert f1!=f2
    c=VisionFeatureCache(tmp_path/'cache',f1,reserve_bytes=0,ram_max_bytes=0)
    c.get_many([str(p)],lambda paths:[torch.ones(2,3)])
    path=c.path_for_hash(image_content_hash(p))
    payload=torch.load(path,weights_only=True); payload['features'][0,0]=99
    torch.save(payload,path)
    with pytest.raises(ValueError,match='checksum'):
        c.get_many([str(p)],lambda _:pytest.fail('must not reencode corrupt data silently'))
    c2=VisionFeatureCache(tmp_path/'cache',f2,reserve_bytes=0)
    with pytest.raises(FileNotFoundError,match='read_only'):
        c2.get_many([str(p)],lambda _:[],read_only=True)


def test_manifest_provenance_dedup_and_immutable_split(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    bio=io.BytesIO(); Image.new('RGB',(7,9),'red').save(bio,format='PNG')
    rows=[{'image':{'bytes':bio.getvalue(),'path':None},'caption':'A red square.'}]*2
    raw=tmp_path/'train.parquet';pq.write_table(pa.Table.from_pylist(rows),raw)
    specs=[{'kind':'caption','repo':'test/caption','revision':'abc','config':'main','files':[str(raw)],'limit':10}]
    audit=prepare_from_specs(specs,tmp_path/'out',reserve_bytes=0)
    assert audit['alignment_train_records']+audit['alignment_validation_records']==1
    assert audit['filter_counts']['duplicate_record']==1
    records=[]
    for partition in ['train','validation']:
        records += [json.loads(l) for l in (tmp_path/'out'/f'alignment_{partition}.jsonl').read_text().splitlines()]
    r=records[0]
    assert r['source']['revision']=='abc' and r['source']['split']=='train'
    assert image_content_hash(r['image_paths'][0])==r['image_hashes'][0]
    assert r['partition']==partition_for_group(r['image_hashes'][0])
    assert prepare_from_specs(specs,tmp_path/'out',reserve_bytes=0)==audit
    with pytest.raises(FileExistsError):
        prepare_from_specs([{**specs[0],'limit':1}],tmp_path/'out',reserve_bytes=0)


def test_external_eval_audit_refuses_empty_and_detects_repackaged_image(tmp_path):
    from src.dataset.research_data import audit_eval_overlap
    train=tmp_path/'train.jsonl'; evaluate=tmp_path/'eval.jsonl'
    train.write_text(json.dumps({'record_id':'train','image_hashes':['same-image'],'source':{'repo':'repacked/train'}})+'\n')
    evaluate.write_text(json.dumps({'record_id':'eval','image_hashes':['same-image'],'source':{'repo':'official/eval'}})+'\n')
    with pytest.raises(ValueError,match='evaluation manifests'):
        audit_eval_overlap([train],[])
    audit=audit_eval_overlap([train],[evaluate])
    assert audit['overlap_records']==1
    assert audit['overlaps'][0]['image_hashes']==['same-image']


def test_disk_read_during_eval_does_not_poison_training_ram_cache(tmp_path):
    p=image_file(tmp_path,'a.png','red')
    fp=encoder_fingerprint({'encoder':'x','revision':'a'})
    writer=VisionFeatureCache(tmp_path/'cache',fp,reserve_bytes=0)
    writer.get_many([str(p)],lambda paths:[torch.ones(3,4)])
    reader=VisionFeatureCache(tmp_path/'cache',fp,reserve_bytes=0)
    with torch.inference_mode():
        reader.get_many([str(p)],lambda _:pytest.fail('cached'),read_only=True)
    x=reader.get_many([str(p)],lambda _:pytest.fail('cached'))[0]
    torch.nn.Linear(4,2)(x).sum().backward()
