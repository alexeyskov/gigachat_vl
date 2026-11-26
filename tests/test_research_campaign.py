import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts import research_campaign as campaign


def test_clean_sft_revision_reuses_unchanged_caption_args_and_enables_eval(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(campaign,'ROOT',tmp_path)
    Path('configs/research').mkdir(parents=True)
    Path('configs/research/smallthinker_smoke_queue.state.json').write_text(json.dumps({'status':'completed'}))
    clean=Path('datasets/research/clean');clean.mkdir(parents=True);(clean/'audit.json').write_text('{}')
    mixture=Path('datasets/research/clean-mixture');mixture.mkdir();(mixture/'sft_train.jsonl').write_text('{}\n')
    args=SimpleNamespace(pilot_data=str(clean),seed=42,alignment_data='datasets/research/original',queue_tag='scienceqa_disjoint_v1')
    with patch('argparse.ArgumentParser.parse_args',return_value=args),patch.object(campaign.subprocess,'run') as run:
        campaign.main()
    spec=json.loads(Path('configs/research/smallthinker_pilot_scienceqa_disjoint_v1_queue.json').read_text())
    assert len(spec['jobs'])==24
    for job in spec['jobs']:
        argv=job['argv'];stage=argv[argv.index('--stage')+1];manifest=argv[argv.index('--manifest')+1]
        if stage in ('A0','A1'):
            assert manifest=='datasets/research/original/alignment_train.jsonl'
        else:
            assert manifest=='datasets/research/clean-mixture/sft_train.jsonl'
            assert job['evaluation']['manifest']=='datasets/research/eval-heldout-v3/evaluation.jsonl'
            assert job['evaluation']['checkpoint']=='best'
        assert argv[argv.index('--effective-batch-size')+1]=='64'
    assert run.call_count==2
    provenance=json.loads(Path('configs/research/smallthinker_pilot_scienceqa_disjoint_v1_queue.provenance.json').read_text())
    assert Path(provenance['source_archive']).is_file()
    assert provenance['data_profile']=='datasets/research/clean'
