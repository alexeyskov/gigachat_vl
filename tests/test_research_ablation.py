"""Image interventions preserve held-out labels and never retain shuffled images."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from src.dataset.vision_cache import image_content_hash
from src.eval.research_ablation import transform_image_ablation,write_transformed_manifest
from src.eval.research_inference import build_generation_inputs,freeze_for_generation
import scripts.eval_research_checkpoint as cli


def image_records(tmp_path):
    rows=[]
    colors=[(255,0,0),(0,255,0),(255,0,0),(0,0,255)]
    groups=['general_vqa','ocr_documents','charts_diagrams','knowledge_reasoning']
    for index,(color,group) in enumerate(zip(colors,groups)):
        path=tmp_path/f'image-{index}.png'
        Image.new('RGB',(9,7),color).save(path)
        rows.append({'record_id':str(index),'question':'question red','answer':'red','answers':['red'],
                     'choices':['red','blue'],'scorer':'exact_match','task_group':group,
                     'source':{'repo':'original-source','original_image_id':str(index)},
                     'image_paths':[str(path)],'image_hashes':[image_content_hash(path)],
                     'image':str(path),'images':[str(path)],'image_phashes':['original-phash']})
    return rows


def test_shuffle_deranges_unique_image_groups_including_repeated_questions(tmp_path):
    records=image_records(tmp_path);before=copy.deepcopy(records)
    first,protocol=transform_image_ablation(records,variant='shuffle',seed=42)
    second,again=transform_image_ablation(records,variant='shuffle',seed=42)
    assert first==second and protocol==again
    assert records==before
    assert first[0]['image_hashes']==first[2]['image_hashes']
    for original,changed in zip(records,first):
        assert changed['image_hashes']!=original['image_hashes']
        assert image_content_hash(changed['image_paths'][0])==changed['image_hashes'][0]
        for field in ['record_id','question','answer','answers','choices','scorer','task_group','source']:
            assert changed[field]==original[field]
        assert changed['image_ablation']['original_image_hashes']==original['image_hashes']
        assert changed['image_ablation']['actual_image_hashes']==changed['image_hashes']
        assert changed['image_ablation']['original_image_phashes']==original['image_phashes']


def test_group_mapping_is_gold_and_input_order_independent(tmp_path):
    rows=image_records(tmp_path)
    original,_=transform_image_ablation(rows,variant='shuffle',seed=99)
    changed=copy.deepcopy(rows)
    for row in changed:row['answers']=['arbitrary different gold'];row['answer']='other';row['choices']=['x','y']
    changed.reverse()
    actual,_=transform_image_ablation(changed,variant='shuffle',seed=99)
    expected={r['record_id']:r['image_hashes'] for r in original}
    assert {r['record_id']:r['image_hashes'] for r in actual}==expected


def test_no_gold_getitem_or_get_is_used_to_choose_images(tmp_path):
    class Guard(dict):
        def __getitem__(self,key):
            if key in {'answer','answers','label','choices'}:raise AssertionError('Gold accessed')
            return super().__getitem__(key)
        def get(self,key,*args):
            if key in {'answer','answers','label','choices'}:raise AssertionError('Gold accessed')
            return super().get(key,*args)
    rows=[Guard(row) for row in image_records(tmp_path)]
    changed,_=transform_image_ablation(rows,variant='shuffle',seed=42)
    assert all(row['image_ablation']['variant']=='shuffle' for row in changed)


def test_blank_images_are_black_rgb_with_original_dimensions_and_canonical_hash(tmp_path):
    records=image_records(tmp_path)
    special=tmp_path/'different-dimensions.png';Image.new('RGBA',(13,11),(123,45,67,89)).save(special)
    records[1]['image_paths']=[str(special)];records[1]['image_hashes']=[image_content_hash(special)]
    changed,protocol=transform_image_ablation(records,variant='blank',output_dir=tmp_path/'out',reserve_bytes=0)
    for original,actual in zip(records,changed):
        original_size=Image.open(original['image_paths'][0]).size
        with Image.open(actual['image_paths'][0]) as black:
            assert black.mode=='RGB' and black.size==original_size
            assert black.getextrema()==((0,0),(0,0),(0,0))
            assert image_content_hash(black)==actual['image_hashes'][0]
        assert actual['image_ablation']['original_image_hashes']==original['image_hashes']
    assert len(list((tmp_path/'out/ablation_images').glob('*.png')))==2
    assert not list((tmp_path/'out/ablation_images').glob('*.tmp'))
    again,_=transform_image_ablation(records,variant='blank',output_dir=tmp_path/'out',reserve_bytes=0)
    assert again==changed
    manifest=tmp_path/'out/transformed_manifest.jsonl'
    assert write_transformed_manifest(changed,manifest,reserve_bytes=0)==protocol['transformed_manifest_sha256']
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==protocol['transformed_manifest_sha256']


def test_blank_disk_reserve_checked_and_mismatched_source_hash_refused(tmp_path,monkeypatch):
    rows=image_records(tmp_path)
    import src.eval.research_ablation as ablation
    monkeypatch.setattr(ablation,'ensure_disk_space',lambda *a:(_ for _ in ()).throw(OSError('reserved disk')))
    with pytest.raises(OSError,match='reserved'):transform_image_ablation(rows,variant='blank',output_dir=tmp_path/'out')
    assert not (tmp_path/'out/ablation_images').exists()
    rows[0]['image_hashes']=['wrong']
    with pytest.raises(ValueError,match='hash differs'):transform_image_ablation(rows,variant='shuffle')


def test_single_unique_shuffle_fails_and_none_is_exact_identity_without_files(tmp_path):
    records=image_records(tmp_path)
    with pytest.raises(ValueError,match='at least two'):transform_image_ablation([records[0],records[2]],variant='shuffle')
    fake=[{'record_id':'one','question':'q','answers':['a'],'image_paths':['missing.png'],'image_hashes':['opaque']}]
    transformed,protocol=transform_image_ablation(fake,variant='none',source_manifest_sha256='a'*64)
    assert transformed==fake
    assert protocol['transformed_manifest_sha256']==protocol['source_manifest_sha256']=='a'*64
    assert not (tmp_path/'ablation_images').exists()


def test_text_only_has_no_visual_features_or_legacy_image_aliases(tmp_path):
    from tests.test_research_inference import TinyResearchModel
    records=image_records(tmp_path);before=copy.deepcopy(records)
    changed,_=transform_image_ablation(records,variant='text_only')
    assert records==before
    for row in changed:
        assert row['image_paths']==[] and row['image_hashes']==[] and row['image'] is None and row['images']==[]
    model=freeze_for_generation(TinyResearchModel())
    prepared=build_generation_inputs(model,changed[0],max_new_tokens=4)
    assert prepared['visual_tokens']==0 and not prepared['modality_mask'].any()
    assert model.cache.calls==[]


def test_cli_ablation_protocol_resume_and_distinct_output_requirement(tmp_path,monkeypatch):
    from tests.test_research_inference import cli_fixture
    argv=cli_fixture(tmp_path,monkeypatch)
    manifest=Path(argv[argv.index('--manifest')+1]);records=image_records(tmp_path)
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in records))
    generated=[]
    def prediction(_model,row,**kwargs):
        generated.append(row)
        return {'prediction':'red','generated_token_ids':[7],'generated_tokens':1}
    monkeypatch.setattr(cli,'generate_prediction',prediction)
    result=cli.main(argv+['--image-ablation','shuffle'])
    assert result['group_macro_score']==100
    protocol=json.loads((tmp_path/'output/protocol.json').read_text())
    descriptor=protocol['image_ablation']
    assert descriptor['variant']=='shuffle' and descriptor['seed']==42
    assert descriptor['source_manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert descriptor['transformed_manifest_sha256']==hashlib.sha256((tmp_path/'output/transformed_manifest.jsonl').read_bytes()).hexdigest()
    assert descriptor['transformation_source_sha256']
    for old,new in zip(records,generated):assert old['image_hashes']!=new['image_hashes'] and old['record_id']==new['record_id']
    assert cli.main(argv+['--image-ablation','shuffle','--resume'])==result
    with pytest.raises(ValueError,match='variant/seed differs'):cli.main(argv+['--resume','--image-ablation','text_only'])
    with pytest.raises(ValueError,match='variant/seed differs'):cli.main(argv+['--resume','--image-ablation','shuffle','--ablation-seed','43'])


def test_cli_none_defaults_keep_existing_resume_macro_behavior(tmp_path,monkeypatch):
    from tests.test_research_inference import cli_fixture
    argv=cli_fixture(tmp_path,monkeypatch)
    result=cli.main(argv)
    assert result['complete_manifest'] and result['group_macro_score']==100
    assert result['image_ablation']['variant']=='none'
    assert not (tmp_path/'output/transformed_manifest.jsonl').exists()
    assert cli.main(argv+['--resume'])==result
