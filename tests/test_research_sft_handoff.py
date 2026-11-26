"""Automatic SFT launch requires matching data bytes and the complete recipe set."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import handoff_research_sft_balance as handoff


class TestSFTHandoff(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        actual=Path(__file__).resolve().parents[1]/handoff.SPEC
        self.spec=json.loads(actual.read_text())
        self.binding={'model_ref':{'repo':'test','revision':'a'*40},'prompt_schema_version':2,
                      'tokenizer_asset_sha256':{'tokenizer.json':'hash'},'tokenize_source_sha256':'code-hash'}
        self.tokenizer_patch=patch.object(handoff,'current_tokenizer_binding',return_value=self.binding)
        self.tokenizer_patch.start()
        self.warmup_patch=patch.object(handoff,'validate_warmup_state')
        self.warmup_patch.start()

    def tearDown(self):
        self.warmup_patch.stop();self.tokenizer_patch.stop();self.temporary.cleanup()

    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def fixture(self):
        self.write(handoff.SPEC,self.spec)
        self.write(handoff.WARMUPS,{'jobs':[]})
        source=self.root/'scripts/run_research_stage.py';source.parent.mkdir(parents=True,exist_ok=True)
        source.write_text('validated source')
        self.write(handoff.READY,{'status':'passed','source_sha256':{'scripts/run_research_stage.py':hashlib.sha256(source.read_bytes()).hexdigest()},
            'checks':dict.fromkeys(handoff.CHECKS,True),'test_command':'pytest focused','tests_passed':5})
        self.write(Path(handoff.WARMUPS).with_suffix('.state.json'),{'status':'completed'})
        reports={};audit={'splits':{}}
        for split in ('train','validation'):
            path=self.root/handoff.MIXTURE/f'sft_{split}.jsonl'
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{"record_id":"one"}\n')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            audit['splits'][split]={'sha256':digest}
            reports[path.name]={'manifest_sha256':digest,'expanded_prompt_overflow':[],
                                'answer_truncated_records':0}
        self.write(handoff.MIXTURE+'/mix_audit.json',audit)
        self.write(handoff.PREFLIGHT,{'status':'passed','max_length':4096,
                                    'fixed_visual_tokens_per_image':729,'reports':reports,**self.binding})

    def test_waits_for_readiness_and_precomputed_warmups(self):
        ready,missing=handoff.readiness(self.root)
        self.assertIsNone(ready);self.assertIn(handoff.READY,missing)
        self.fixture()
        self.write(Path(handoff.WARMUPS).with_suffix('.state.json'),{'status':'running'})
        ready,missing=handoff.readiness(self.root)
        self.assertIsNone(ready);self.assertIn('caption_warmups_not_completed',missing)
        self.write(Path(handoff.WARMUPS).with_suffix('.state.json'),{'status':'completed'})
        ready,missing=handoff.readiness(self.root)
        self.assertIsNotNone(ready);self.assertEqual(missing,[])

    def test_stale_data_or_preflight_never_launch(self):
        self.fixture()
        path=self.root/handoff.MIXTURE/'sft_train.jsonl';path.write_text('changed\n')
        with self.assertRaisesRegex(ValueError,'Mixture changed'):
            handoff.readiness(self.root)
        self.fixture()
        proof=json.loads((self.root/handoff.PREFLIGHT).read_text())
        proof['reports']['sft_train.jsonl']['manifest_sha256']='stale'
        self.write(handoff.PREFLIGHT,proof)
        with self.assertRaisesRegex(ValueError,'preflight is stale'):
            handoff.readiness(self.root)

    def test_every_recipe_must_use_identical_objective(self):
        self.assertEqual(handoff.validate_sft_spec(self.spec),self.spec)
        bad=copy.deepcopy(self.spec)
        job=next(j for j in bad['jobs'] if '--text-loss-weight' in j['argv'])
        job['argv'][job['argv'].index('--text-loss-weight')+1]='0.9'
        with self.assertRaisesRegex(ValueError,'identical audited'):
            handoff.validate_sft_spec(bad)
        bad=copy.deepcopy(self.spec)
        bad['jobs']=[j for j in bad['jobs'] if '__1a__SFT_' not in j['id']]
        with self.assertRaisesRegex(ValueError,'eleven'):
            handoff.validate_sft_spec(bad)

    def test_readiness_requires_tested_source_and_exact_tokenizer(self):
        self.fixture()
        (self.root/'scripts/run_research_stage.py').write_text('untested change')
        with self.assertRaisesRegex(ValueError,'source changed'):
            handoff.readiness(self.root)
        self.fixture()
        self.binding['tokenize_source_sha256']='changed-tokenizer'
        with self.assertRaisesRegex(ValueError,'Tokenizer/preflight binding'):
            handoff.readiness(self.root)

    def test_warmup_must_match_its_specification(self):
        self.warmup_patch.stop()
        spec={'jobs':[{'id':'warmup','argv':['--stage','A1','--output',str(self.root/'run')]}]}
        self.write(handoff.WARMUPS,spec)
        with self.assertRaisesRegex(ValueError,'different specification'):
            handoff.validate_warmup_state(self.root,{'spec_sha256':'stale'})
        digest=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError,'not completed'):
            handoff.validate_warmup_state(self.root,{'spec_sha256':digest,'jobs':{}})

    def test_supervisor_starting_is_waited_without_restarting_training(self):
        with patch.object(handoff,'service_status',side_effect=['research-pilot STARTING','research-pilot RUNNING pid 123']), \
             patch.object(handoff.time,'sleep') as sleep:
            self.assertIn('RUNNING',handoff.wait_for_running('research-pilot'))
            sleep.assert_called_once_with(.5)
