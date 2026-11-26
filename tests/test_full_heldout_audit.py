from pathlib import Path
import io
import json
import sys

import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.eval.full_heldout_audit import RawImageSignatureCache,full_heldout_audit
from src.dataset.vision_cache import image_content_hash
from src.dataset.research_data import image_perceptual_hash


def raw_png(color='red'):
    buffer=io.BytesIO();Image.new('RGB',(9,8),color).save(buffer,format='PNG');return buffer.getvalue()


def test_raw_signature_matches_canonical_policy_and_decodes_duplicates_once(tmp_path):
    raw=raw_png();path=tmp_path/'image.png';path.write_bytes(raw)
    cache=RawImageSignatureCache(tmp_path/'cache.jsonl')
    signature=cache.signature({'bytes':raw})
    assert signature['content_hash']==image_content_hash(path)
    assert signature['phash']==image_perceptual_hash(path)
    assert cache.signature({'bytes':raw})==signature
    assert cache.decodes==1 and cache.hits==1
    cache.close()
    reopened=RawImageSignatureCache(tmp_path/'cache.jsonl')
    assert reopened.signature({'bytes':raw})==signature and reopened.decodes==0
    reopened.close()


def test_signature_cache_repairs_interrupted_tail_without_hiding_middle_corruption(tmp_path):
    path=tmp_path/'cache.jsonl';cache=RawImageSignatureCache(path)
    cache.signature({'bytes':raw_png()});cache.close()
    with path.open('a') as handle:handle.write('{broken_partial')
    resumed=RawImageSignatureCache(path);resumed.signature({'bytes':raw_png('blue')});resumed.close()
    assert len(RawImageSignatureCache(path).values)==2
    path.write_text('{broken}\n'+path.read_text())
    with pytest.raises(json.JSONDecodeError):RawImageSignatureCache(path)


def test_full_split_row_mismatch_reports_partial_not_complete(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import src.eval.full_heldout_audit as module
    monkeypatch.chdir(tmp_path);(tmp_path/'research_results').mkdir()
    root=tmp_path/'datasets';path=root/'eval-raw/lmms-lab--VQAv2/data/validation-0.parquet';path.parent.mkdir(parents=True)
    row={'image':{'bytes':raw_png(),'path':'original.jpg'},'question':'Where is it?',
         'answers':[{'answer':'here'}]*10,'question_id':1,'image_id':1}
    pq.write_table(pa.Table.from_pylist([row]),path)
    source={'vqav2':{'repo':'lmms-lab/VQAv2','revision':'pinned','splits':{'validation':[{'path':'data/validation-0.parquet','size':path.stat().st_size}]}}}
    monkeypatch.setattr(module,'selected_source_files',lambda root:source)
    monkeypatch.setattr(module,'training_image_index',lambda *args:({'images':{}},{}))
    monkeypatch.setattr(module,'EXPECTED_ROWS',{'vqav2':{'validation':2}})
    output=tmp_path/'audit'
    with pytest.raises(ValueError,match='coverage mismatch'):
        full_heldout_audit(root,output,['unused'],reserve_bytes=0)
    report=json.loads((output/'audit.json').read_text())
    assert report['status']=='partial' and report['coverage_complete'] is False
    assert not list(output.glob('images/*'))
