"""Run with BENCHMARK_HARNESS_TESTS=1 in the isolated benchmark environment."""
import json,os
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif(os.environ.get('BENCHMARK_HARNESS_TESTS')!='1',reason='Isolated optional official harness')


def test_actual_official_mcq_and_generation_free_scoring_resume(tmp_path):
    import pandas as pd
    from PIL import Image
    from src.eval.vlm_benchmark_runtime_v1 import check_harness,choose,run_one
    check_harness()
    from vlmeval.dataset.image_mcq import ImageMCQDataset
    image=tmp_path/'image.png';Image.new('RGB',(2,2),(255,0,0)).save(image)
    dataset=ImageMCQDataset.__new__(ImageMCQDataset);dataset.dataset_name='MMStar';dataset.meta_only=True
    dataset.data=pd.DataFrame([dict(index=i,question='Select the matching color.',image_path=str(image),
        A='red',B='blue',answer='A') for i in range(4)])
    dataset.data_path=str(tmp_path/'source.tsv');dataset.data.to_csv(dataset.data_path,sep='\t',index=False)
    class Fixture:
        def __init__(self):self.calls=0
        def generate(self,message):
            assert all(set(m)=={'type','value'} for m in message)
            assert [m['type'] for m in message]==['image','text']
            prediction='A' if self.calls%2==0 else 'B';self.calls+=1
            return {'prediction':prediction}
    fixture=Fixture();out=tmp_path/'run';protocol={'test_fixture':True,'no_real_model':True}
    result=run_one(dataset,choose(names=['MMStar'])[0],fixture,out,protocol)
    assert result['metrics']==[{'split':'none','Overall':.5}]
    assert fixture.calls==4 and not result['official_full_score']
    saved=json.loads((out/'protocol.json').read_bytes())
    scored=run_one(dataset,choose(names=['MMStar'])[0],None,out,saved,mode='score')
    assert scored['metrics']==result['metrics']
    partial=tmp_path/'partial';sample=Fixture()
    run_one(dataset,choose(names=['MMStar'])[0],sample,partial,protocol,mode='infer',limit=2)
    saved=json.loads((partial/'protocol.json').read_bytes())
    scored=run_one(dataset,choose(names=['MMStar'])[0],None,partial,saved,mode='score')
    assert scored['records']==2 and scored['diagnostic_subset'] and not scored['official_full_score']


def test_original_ocrbench_scale_is_preserved(tmp_path):
    import pandas as pd
    from src.eval.vlm_benchmark_runtime_v1 import check_harness,choose,evaluate_official
    check_harness()
    from vlmeval.dataset.image_vqa import OCRBench
    from vlmeval.smp import dump
    file=tmp_path/'ocr.xlsx'
    dump(pd.DataFrame([dict(index=0,answer="['cat']",prediction='cat',category='Regular Text Recognition'),
        dict(index=1,answer="['dog']",prediction='bird',category='Regular Text Recognition')]),str(file))
    result=evaluate_official(OCRBench,file,choose(names=['OCRBench'])[0])
    assert result['Final Score']==1 and result['Final Score Norm']==.1


def test_document_anls_and_chart_numeric_tolerance(tmp_path):
    import pandas as pd
    from src.eval.vlm_benchmark_runtime_v1 import check_harness,choose,evaluate_official
    check_harness()
    from vlmeval.dataset.image_vqa import ImageVQADataset
    from vlmeval.smp import dump
    for name,reference,prediction,expected in [('DocVQA_VAL','hello','hallo',80.),('ChartQA_TEST','100','104',100.)]:
        dataset=ImageVQADataset.__new__(ImageVQADataset);dataset.dataset_name=name
        file=tmp_path/(name+'.xlsx');dump(pd.DataFrame([dict(index=0,answer=repr([reference]),prediction=prediction)]),str(file))
        result=evaluate_official(dataset,file,choose(names=[name])[0])
        assert abs(float(result['Overall'].iloc[0])-expected)<1e-6
