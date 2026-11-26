"""Runner correctness tests: causal loss, multimodal masks and stage persistence.

The real runner methods execute on tiny CPU models. Only hardcoded CUDA device
placement and unrelated checkpoint loading are replaced; no pretrained weights
or GPU memory are required.
"""
import argparse
import copy
import contextlib
import importlib.util
import json
import os
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location("run_research_stage", Path(__file__).parents[1] / "scripts/run_research_stage.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@contextlib.contextmanager
def cpu_placements():
    tensor, arange, tensor_to = torch.tensor, torch.arange, torch.Tensor.to
    def make_tensor(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return tensor(*args, **kwargs)
    def make_arange(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return arange(*args, **kwargs)
    def move(self, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)) and str(args[0]).startswith("cuda"):
            args = ("cpu", *args[1:])
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs["device"] = "cpu"
        return tensor_to(self, *args, **kwargs)
    with patch.object(torch, "tensor", make_tensor), patch.object(torch, "arange", make_arange), patch.object(torch.Tensor, "to", move):
        yield


class TinyBody(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)
    def forward(self, inputs_embeds, **kwargs):
        self.last_hidden = self.linear(inputs_embeds.cumsum(dim=1))
        return SimpleNamespace(last_hidden_state=self.last_hidden)


class TinyLLM(nn.Module):
    def __init__(self, width=8, vocab=17):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.model = TinyBody(width)
        self.lm_head = nn.Linear(width, vocab)
    def get_input_embeddings(self):
        return self.embedding


def tiny_research_model():
    model = runner.ResearchModel.__new__(runner.ResearchModel)
    nn.Module.__init__(model)
    model.args = SimpleNamespace(max_length=24)
    model.llm = TinyLLM().to(dtype=torch.bfloat16)
    model.base = model.llm
    model.projector = nn.Linear(4, 8).to(dtype=torch.bfloat16)
    model.wrappers = []
    return model


class TestResearchRunnerLoss(unittest.TestCase):
    def test_sparse_supervised_head_equals_full_causal_ce_with_padding(self):
        torch.manual_seed(7)
        model = tiny_research_model()
        feature = torch.randn(2, 4)
        rows = [([1], [2], [3, 4], [feature]), ([], [2, 3], [4], [])]
        with cpu_placements():
            sparse_loss, targets, sequence_tokens = model.forward_loss(rows)
        hidden = model.base.model.last_hidden
        logits = model.base.lm_head(hidden).float()
        labels = torch.tensor([[-100, -100, -100, -100, 3, 4], [-100, -100, 4, -100, -100, -100]])
        dense_loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        self.assertEqual(targets, 3)
        self.assertEqual(sequence_tokens, 9)
        torch.testing.assert_close(sparse_loss, dense_loss, rtol=1e-6, atol=1e-6)
        sparse_grad = torch.autograd.grad(sparse_loss, model.projector.weight, retain_graph=True)[0]
        dense_grad = torch.autograd.grad(dense_loss, model.projector.weight)[0]
        torch.testing.assert_close(sparse_grad, dense_grad, rtol=2e-2, atol=2e-3)
        self.assertGreater(sparse_grad.abs().sum().item(), 0)

    def test_frozen_llm_still_transmits_gradient_to_projector_and_masks(self):
        model = tiny_research_model()
        model.llm.requires_grad_(False)
        model.wrappers = [object()]
        rows = [([1], [2], [3], [torch.randn(2, 4)]), ([], [2], [4], [])]
        with patch("src.model.research_moe.set_modality_mask") as setter, cpu_placements():
            loss, _, _ = model.forward_loss(rows)
            loss.backward()
        self.assertIsNotNone(model.projector.weight.grad)
        self.assertGreater(model.projector.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.llm.parameters()))
        _, modality, attention = setter.call_args.args
        self.assertEqual(modality.tolist(), [[False, True, True, False, False], [False, False, False, False, False]])
        self.assertEqual(attention.tolist(), [[True, True, True, True, True], [True, True, False, False, False]])

    def test_stream_components_match_dense_targets_with_one_model_forward(self):
        model=tiny_research_model()
        rows=[([1],[2],[3,4],[torch.randn(2,4)]),([],[2,3],[4],[])]
        with cpu_placements(),patch.object(model.base.model,'forward',wraps=model.base.model.forward) as forward:
            parts=model.forward_loss_components(rows)
        self.assertEqual(forward.call_count,1)
        logits=model.base.lm_head(model.base.model.last_hidden).float()
        labels=torch.tensor([[-100,-100,-100,-100,3,4],[-100,-100,4,-100,-100,-100]])
        losses=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),
                              ignore_index=-100,reduction='none').reshape(2,-1)
        torch.testing.assert_close(parts['multimodal_loss_sum'],losses[0].sum(),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(parts['text_only_loss_sum'],losses[1].sum(),atol=1e-6,rtol=1e-6)
        self.assertEqual(parts['multimodal_supervised_tokens'],2)
        self.assertEqual(parts['text_only_supervised_tokens'],1)

    def test_prepare_batches_cache_lookups_and_enforces_expanded_context(self):
        model = tiny_research_model()
        model.tokenize = lambda record: (record.get("pre", [1]), [2], [3, 4])
        calls = []
        class Cache:
            def get_many(self, paths, encoder):
                calls.append(list(paths))
                return [torch.ones(2, 4) for _ in paths]
        model.cache = Cache()
        model.encode = lambda _paths: None
        records = [{"image_paths": ["one.png"]}, {"image_paths": []}, {"image_paths": ["one.png"]}]
        rows = model.prepare(records)
        self.assertEqual(calls, [["one.png", "one.png"]])
        self.assertEqual([len(row[3]) for row in rows], [1, 0, 1])
        model.args.max_length = 4
        with self.assertRaisesRegex(ValueError, "Expanded prompt too long"):
            model.prepare([{"image_paths": ["one.png"]}])


class TestResearchPromptMessages(unittest.TestCase):
    def model(self,version=2):
        model=runner.ResearchModel.__new__(runner.ResearchModel)
        nn.Module.__init__(model)
        model.args=SimpleNamespace(prompt_schema_version=version)
        model.base=SimpleNamespace(config=SimpleNamespace(model_type='smallthinker'))
        class Tokenizer:
            def __init__(self):self.calls=[]
            def apply_chat_template(self,messages,tokenize,add_generation_prompt):
                self.calls.append(copy.deepcopy(messages))
                text=''.join('<'+m['role']+'>'+m['content']+'</'+m['role']+'>' for m in messages)
                return text+('<assistant>' if add_generation_prompt else '')
            def encode(self,text,add_special_tokens):return [ord(c) for c in text]
        model.tokenizer=Tokenizer()
        return model

    def test_system_instruction_retained_and_gold_only_in_full_target(self):
        model=self.model()
        row={'question':'the email body','answer':'GOLD summary','image_paths':[],
             'messages':[{'role':'system','content':'Summarize the supplied email in one sentence.'},
                         {'role':'user','content':'the email body'}]}
        original=copy.deepcopy(row)
        before,prompt,answer=model.tokenize(row)
        self.assertEqual(before,[])
        prompt=''.join(chr(c) for c in prompt)
        self.assertIn('Summarize the supplied email',prompt)
        self.assertNotIn('GOLD',prompt)
        self.assertNotIn('[RESEARCH_IMAGE]',prompt)
        self.assertIn('GOLD summary',''.join(chr(c) for c in answer))
        self.assertEqual(model.tokenizer.calls[0][-1]['role'],'user')
        self.assertEqual(model.tokenizer.calls[1][-1],{'role':'assistant','content':'GOLD summary'})
        changed=dict(row,answer='ANOTHER target')
        self.assertEqual(model.tokenize(changed)[:2],model.tokenize(row)[:2])
        self.assertEqual(row,original)

    def test_history_preserved_and_image_marker_added_only_to_final_user(self):
        row={'question':'current question','answer':'target','image_paths':['image'],
             'messages':[{'role':'system','content':'instruction'},{'role':'user','content':'prior question'},
                         {'role':'assistant','content':'prior public answer'},{'role':'user','content':'current question'}]}
        model=self.model();model.tokenize(row)
        messages=model.tokenizer.calls[0]
        self.assertEqual(messages[:3],row['messages'][:3])
        self.assertEqual(messages[-1]['content'],'[RESEARCH_IMAGE]\ncurrent question')
        self.assertEqual(sum('[RESEARCH_IMAGE]' in m['content'] for m in messages),1)

    def test_invalid_prefix_or_question_alignment_fails_explicitly(self):
        model=self.model()
        for messages in ([{'role':'user','content':'different'}],
                         [{'role':'assistant','content':'target already included'}],
                         [{'role':'user','content':None}]):
            with self.subTest(messages=messages),self.assertRaises(ValueError):
                model.tokenize({'question':'q','answer':'a','image_paths':[],'messages':messages})

    def test_question_fallback_unchanged_and_legacy_version_replays_old_prompts(self):
        row={'question':'question','answer':'target','image_paths':[]}
        model=self.model()
        self.assertEqual(model.tokenize(row),model.tokenize(dict(row,messages=[{'role':'user','content':'question'}])))
        with_system=dict(row,messages=[{'role':'system','content':'summarize'},{'role':'user','content':'question'}])
        legacy=self.model(version=1)
        self.assertEqual(legacy.tokenize(row),legacy.tokenize(with_system))
        self.assertNotEqual(model.tokenize(with_system)[:2],legacy.tokenize(with_system)[:2])


class FakeStageModel(nn.Module):
    """Tiny CPU substitute for loading a full checkpoint; main() remains real."""
    instances = []
    def __init__(self, args):
        super().__init__()
        self.args=args
        self.llm = nn.Linear(1, 1, bias=False)
        self.projector = nn.Linear(1, 1, bias=False)
        self.llm.requires_grad_(args.stage != "A0")
        self.projector.requires_grad_(args.stage == "A0")
        self.extractor = SimpleNamespace(vision_tower=nn.Identity())
        self.cache = SimpleNamespace(stats={"hits": 0, "misses": 0}, fingerprint="test-fingerprint")
        self.wrappers = []
        self.model_ref = {"repo": "tiny", "revision": "test"}
        self.vision_ref = {"repo": "tiny-vision", "revision": "test"}
        self.initial_llm_weight = self.llm.weight.detach().clone()
        self.instances.append(self)
    def prepare(self, records):
        return [([], [1], [1], [torch.zeros(1,1)] if row.get('image_paths') else []) for row in records]
    def forward_loss(self, rows):
        output = self.llm(torch.ones(len(rows), 1)) + self.projector(torch.ones(len(rows), 1))
        loss = ((output - 2) ** 2).sum()
        return loss, len(rows), len(rows) * 2
    def forward_loss_components(self,rows):
        self.components_calls=getattr(self,'components_calls',0)+1
        output=self.llm(torch.ones(len(rows),1))+self.projector(torch.ones(len(rows),1))
        per_row=(output-2).square().flatten()
        mask=torch.tensor([bool(row[3]) for row in rows])
        return {'multimodal_loss_sum':per_row[mask].sum(),'text_only_loss_sum':per_row[~mask].sum(),
                'multimodal_supervised_tokens':int(mask.sum()),'text_only_supervised_tokens':int((~mask).sum()),
                'sequence_tokens':len(rows)*2}


class TestResearchRunnerStages(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cwd = Path.cwd()
        os.chdir(self.root)
        self.train = self.root / "train.jsonl"
        self.val = self.root / "val.jsonl"
        for path in (self.train, self.val):
            path.write_text(json.dumps({"image_paths": [], "question": "q", "answer": "a"}) + "\n")
        FakeStageModel.instances = []
    def tearDown(self):
        os.chdir(self.cwd)
        self.temp.cleanup()
    def arguments(self, stage, name, **kwargs):
        args = dict(model="tiny", vision="tinyvision", manifest=str(self.train), validation=str(self.val),
                    output=str(self.root / name), stage=stage, recipe="4f", alignment_checkpoint=None,
                    initial_checkpoint=None, resume=False, epochs=1, max_records=None, max_steps=1,
                    text_manifest=None, text_validation=None, text_fraction=0.15,
                    batch_size=1, effective_batch_size=1, max_length=32, eval_records=1, eval_every=1,
                    reserve_gb=0, seed=42, cache_dir=str(self.root / "cache"))
        args.update(kwargs)
        return argparse.Namespace(**args)
    def run_stage(self, args,model_class=FakeStageModel):
        with patch.object(runner, "parse_args", return_value=args), patch.object(runner, "ResearchModel", model_class), \
             patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "get_rng_state_all", return_value=[]), \
             patch('src.utils.research_provenance.write_run_provenance',return_value={'code_sha256':'test','source_archive':'test.zip'}), \
             patch.object(torch.cuda, "max_memory_allocated", return_value=0):
            runner.main()
        path = Path(args.output) / "last/state.pt"
        return torch.load(path, map_location="cpu", weights_only=False)

    def test_a0_delta_and_projector_survive_a1_sft_round_trip(self):
        a0 = self.run_stage(self.arguments("A0", "a0"))
        self.assertEqual(set(a0["model"]), {"projector.weight"})
        aligned = a0["model"]["projector.weight"].clone()
        a1 = self.run_stage(self.arguments("A1", "a1", alignment_checkpoint=str(self.root / "a0/last/state.pt")))
        self.assertIn("llm.weight", a1["model"])
        torch.testing.assert_close(a1["model"]["projector.weight"], aligned)
        sft = self.run_stage(self.arguments("SFT", "sft", initial_checkpoint=str(self.root / "a1/last/state.pt")))
        torch.testing.assert_close(sft["model"]["projector.weight"], aligned)
        self.assertFalse(torch.equal(sft["model"]["llm.weight"], a1["model"]["llm.weight"]))
        self.assertEqual(sft["step"], 1)
        self.assertIn("optimizer", sft)
        self.assertIn("rng_python", sft)
        self.assertEqual(len(list((self.root / "sft/.checkpoint_objects").iterdir())), 1)

    def test_missing_alignment_rejected_before_loading_model(self):
        args = self.arguments("SFT", "invalid")
        with patch.object(runner, "parse_args", return_value=args), patch.object(runner, "ResearchModel") as loader:
            with self.assertRaisesRegex(ValueError, "requires aligned"):
                runner.main()
            loader.assert_not_called()

    def test_changed_manifest_rejected_on_resume(self):
        args = self.arguments("A0", "resume")
        self.run_stage(args)
        self.train.write_text(self.train.read_text() + self.train.read_text())
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.run_stage(self.arguments("A0", "resume", resume=True))

    def test_legacy_completed_resume_adopts_question_only_schema_without_new_option_drift(self):
        args=self.arguments('A0','legacy-resume')
        payload=self.run_stage(args)
        old=dict(payload['config'])
        for key in (*runner.SELECTION_DEFAULTS,'prompt_schema_version','vision_preprocessing_profile'):
            old.pop(key,None)
        payload['config']=old
        torch.save(payload,self.root/'legacy-resume/last/state.pt')
        (self.root/'legacy-resume/resolved_config.json').write_text(json.dumps(old))
        self.run_stage(self.arguments('A0','legacy-resume',resume=True))
        resolved=json.loads((self.root/'legacy-resume/resolved_config.json').read_text())
        self.assertEqual(resolved['prompt_schema_version'],1)
        self.assertEqual(resolved['selection_metric'],'ce')
        self.assertIsNone(resolved['vision_preprocessing_profile'])
        self.assertEqual(FakeStageModel.instances[-1].args.prompt_schema_version,1)

    def test_text_replay_manifest_hash_is_part_of_resume_contract(self):
        self.run_stage(self.arguments("A0", "a0-text"))
        self.train.write_text(self.train.read_text() * 8)
        text = self.root / "text.jsonl"
        text.write_text(json.dumps({"image_paths": [], "question": "text", "answer": "answer"}) + "\n")
        args = self.arguments("SFT", "sft-text", alignment_checkpoint=str(self.root / "a0-text/last/state.pt"),
                              text_manifest=str(text))
        saved = self.run_stage(args)
        self.assertIsNotNone(saved["config"]["text_manifest_sha256"])
        text.write_text(text.read_text() * 2)
        with self.assertRaisesRegex(ValueError, "Incompatible resume"):
            self.run_stage(self.arguments("SFT", "sft-text", resume=True, text_manifest=str(text)))

    def test_weighted_sft_objective_is_saved_logged_and_immutable(self):
        self.run_stage(self.arguments('A0','a0-weighted'))
        rows=[{'image_paths':['image'],'question':'q','answer':'a'},
              {'image_paths':[],'question':'q','answer':'a'}]
        self.train.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        args=self.arguments('SFT','weighted',alignment_checkpoint=str(self.root/'a0-weighted/last/state.pt'),
                            text_loss_weight=.15,batch_size=2,effective_batch_size=2)
        saved=self.run_stage(args)
        self.assertEqual(saved['loss_objective']['text_loss_weight'],.15)
        self.assertEqual(saved['loss_objective']['missing_stream_policy'],'zero_contribution_no_weight_renormalization')
        self.assertEqual(saved['active_microbatch_size'],2)
        self.assertEqual(FakeStageModel.instances[-1].components_calls,1)
        progress=[json.loads(line) for line in (self.root/'weighted/metrics.jsonl').read_text().splitlines()]
        update=next(row for row in progress if row['event']=='train')
        self.assertEqual(update['text_only_records'],1)
        self.assertEqual(update['multimodal_records'],1)
        self.assertIn('multimodal_ce',update);self.assertIn('text_only_ce',update)
        self.assertEqual(update['objective_profile'],'fixed_stream_weights_global_token_denominators')
        with self.assertRaisesRegex(ValueError,'Incompatible resume'):
            self.run_stage(self.arguments('SFT','weighted',resume=True,text_loss_weight=.2,batch_size=2,effective_batch_size=2))

    def test_resume_keeps_reduced_microbatch_after_interrupted_logical_step(self):
        class RetryingStage(FakeStageModel):
            stop_early=True
            def __init__(self,args):
                super().__init__(args);self.calls=[];self.successes=0
            def forward_loss(self,rows):
                if self.training:
                    self.calls.append(len(rows))
                    if len(rows)>1:raise torch.cuda.OutOfMemoryError('controlled batch OOM')
                    self.successes+=1
                    if self.stop_early and self.successes>2:raise RuntimeError('controlled interruption')
                return super().forward_loss(rows)
        self.train.write_text(self.train.read_text()*4)
        args=self.arguments('A0','micro-resume',batch_size=2,effective_batch_size=2,max_steps=2)
        with self.assertRaisesRegex(RuntimeError,'controlled interruption'):
            self.run_stage(args,model_class=RetryingStage)
        last=torch.load(self.root/'micro-resume/last/state.pt',map_location='cpu',weights_only=False)
        self.assertEqual(last['active_microbatch_size'],1)
        self.assertEqual(last['step'],1)
        RetryingStage.stop_early=False
        resumed=self.run_stage(self.arguments('A0','micro-resume',resume=True,batch_size=2,
            effective_batch_size=2,max_steps=2),model_class=RetryingStage)
        self.assertEqual(resumed['step'],2)
        self.assertEqual(resumed['active_microbatch_size'],1)
        self.assertEqual(FakeStageModel.instances[-1].calls,[1,1])

    def internal_rows(self):
        groups=['general_vqa','ocr_documents','charts_diagrams','knowledge_reasoning']
        return [{'record_id':group,'image_paths':[group+'.png'],'image_hashes':[group],
                 'question':'question','answer':'yes','task_group':group,'partition':'validation',
                 'source':{'repo':'training-source','split':'train'}} for group in groups]

    def test_group_macro_selects_max_from_full_validation_and_bounds_prediction_files(self):
        self.run_stage(self.arguments('A0','a0-group'))
        self.train.write_text(self.train.read_text()*2)
        rows=self.internal_rows()
        self.val.write_text(''.join(json.dumps(row)+'\n' for row in rows+rows))
        text=self.root/'text-val.jsonl'
        text.write_text(json.dumps({'record_id':'text','image_paths':[],'question':'text',
            'answer':'teacher','partition':'validation','source':{'split':'train'},'task_group':'text_only'})+'\n')
        calls=[]
        def generate(_model,record,**_):
            self.assertNotIn('answer',record);self.assertNotIn('answers',record)
            calls.append(record['record_id'])
            return {'prediction':'yes' if len(calls)<=4 else 'wrong'}
        args=self.arguments('SFT','sft-group',alignment_checkpoint=str(self.root/'a0-group/last/state.pt'),
                            selection_metric='group_macro',selection_per_group=1,selection_max_new_tokens=2,
                            text_validation=str(text),max_steps=2,eval_records=1)
        with patch('src.eval.research_selection.generate_prediction',side_effect=generate):
            saved=self.run_stage(args)
        self.assertEqual(len(calls),8)
        self.assertEqual(saved['selection_value'],0.)
        self.assertEqual(saved['selection_metric'],'internal_teacher_reference_group_macro')
        self.assertEqual(saved['selection_statistics']['records'],4)
        self.assertFalse(saved['selection_statistics']['official_benchmark_score'])
        self.assertIsNotNone(saved['config']['selection_manifest_sha256'])
        best=torch.load(self.root/'sft-group/best/state.pt',map_location='cpu',weights_only=False)
        self.assertEqual(best['step'],1);self.assertEqual(best['selection_value'],100.)
        manager=json.loads((self.root/'sft-group/.checkpoint_manager.json').read_text())
        self.assertEqual(manager['mode'],'max')
        files=list((self.root/'sft-group/selection').iterdir())
        self.assertEqual(len(files),6)
        for label,expected_step in [('last',2),('best',1)]:
            predictions=[json.loads(line) for line in (self.root/f'sft-group/selection/{label}.predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(predictions),4)
            self.assertTrue(all(row['validation_step']==expected_step for row in predictions))
        metrics=[json.loads(line) for line in (self.root/'sft-group/metrics.jsonl').read_text().splitlines()]
        validation=[row for row in metrics if row['event']=='validation']
        self.assertTrue(all('group_macro_score' in row and 'loss' not in row for row in validation))
        self.assertTrue(any(row['event']=='text_validation' and 'text_only_heldout_ce' in row for row in metrics))
        self.assertFalse(any(row.get('event')=='validation' and 'text_only_heldout_ce' in row for row in metrics))

    def test_group_selection_rejects_official_sources_before_loading(self):
        rows=self.internal_rows();rows[0]['source']['split']='validation'
        self.val.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        args=self.arguments('SFT','invalid-group',alignment_checkpoint='unused-alignment.pt',
                            selection_metric='group_macro')
        with patch.object(runner,'parse_args',return_value=args),patch.object(runner,'ResearchModel') as loader:
            with self.assertRaisesRegex(ValueError,'source.split must be train'):
                runner.main()
            loader.assert_not_called()


class TestResearchRunnerConfiguration(unittest.TestCase):
    def arguments(self):
        return argparse.Namespace(batch_size=2, effective_batch_size=8, max_length=32,
                                  epochs=1, eval_records=2, eval_every=10, max_records=None,
                                  max_steps=5, initial_checkpoint=None, alignment_checkpoint=None, text_fraction=0.15)

    def test_invalid_budgets_rejected(self):
        for name in ("batch_size", "effective_batch_size", "max_length", "epochs", "eval_records", "eval_every", "max_records", "max_steps"):
            with self.subTest(name=name):
                args = self.arguments()
                setattr(args, name, 0)
                with self.assertRaisesRegex(ValueError, "positive"):
                    runner.validate_args(args)

    def test_ambiguous_initialization_rejected(self):
        args = self.arguments()
        args.initial_checkpoint = "a1/state.pt"
        args.alignment_checkpoint = "a0/state.pt"
        with self.assertRaisesRegex(ValueError, "one initialization"):
            runner.validate_args(args)

    def test_resume_cannot_change_recipe_model_or_effective_dataset(self):
        original = dict(model="model-one", vision="vision-one", stage="SFT", recipe="3c",
                        max_records=200, seed=42, max_length=4096, batch_size=4,
                        effective_batch_size=64, epochs=1)
        for name, changed in (("model", "model-two"), ("vision", "vision-two"), ("recipe", "3b"),
                              ("stage", "A0"), ("max_records", 100), ("seed", 123),
                              ("max_length", 2048), ("batch_size", 8), ("epochs", 2)):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "Incompatible resume"):
                    runner.validate_resume_config(original, dict(original, **{name: changed}))
        runner.validate_resume_config(dict(original, resume=False, reserve_gb=120),
                                      dict(original, resume=True, reserve_gb=140))

    def test_legacy_resume_defaults_remain_ce_and_all_selection_options_are_immutable(self):
        legacy={'stage':'A0','recipe':'4q','eval_records':64}
        modern=dict(legacy,optimizer_backend='torch_adamw32',**runner.SELECTION_DEFAULTS)
        runner.validate_resume_config(legacy,modern)
        for key,value in [('selection_metric','group_macro'),('selection_manifest','another.jsonl'),
                          ('selection_manifest_sha256','changed'),('selection_per_group',64),
                          ('selection_max_new_tokens',32),('eval_records',32)]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Incompatible resume'):
                runner.validate_resume_config(modern,dict(modern,**{key:value}))

    def test_group_macro_is_sft_only_and_new_flags_parse(self):
        parsed=runner.parse_args(['--manifest','train','--validation','val','--output','out','--stage','SFT',
                                  '--selection-metric','group_macro','--selection-per-group','8',
                                  '--selection-max-new-tokens','16'])
        self.assertEqual(parsed.selection_per_group,8)
        self.assertEqual(parsed.selection_max_new_tokens,16)
        runner.validate_args(parsed)
        parsed.stage='A0'
        with self.assertRaisesRegex(ValueError,'SFT only'):
            runner.validate_args(parsed)

    def test_selection_dedup_only_accepts_identical_weighted_repetitions(self):
        records=[{'record_id':'same','answer':'yes'},{'record_id':'same','answer':'yes'}]
        self.assertEqual(len(runner.deduplicate_selection_records(records)),1)
        records[1]['answer']='no'
        with self.assertRaisesRegex(ValueError,'Conflicting duplicate'):
            runner.deduplicate_selection_records(records)

    def test_text_weight_is_optional_sft_only_and_legacy_default_is_none(self):
        runner.validate_resume_config({'stage':'A0'},{'stage':'A0','text_loss_weight':None})
        args=self.arguments();args.stage='A0';args.text_loss_weight=.15
        with self.assertRaisesRegex(ValueError,'SFT only'):runner.validate_args(args)
        for value in (-.1,1.1,float('nan'),float('inf')):
            args=self.arguments();args.stage='SFT';args.text_loss_weight=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'finite and between'):
                runner.validate_args(args)


class OOMRetryModel(nn.Module):
    """Deterministic CPU loss with controlled forward/backward OOM failures."""
    def __init__(self, max_rows=None, fail_once_on_second=None, ordinary_error=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.75))
        self.max_rows = max_rows
        self.fail_once_on_second = fail_once_on_second
        self.ordinary_error = ordinary_error
        self.failure_spent = False
        self.calls = []
        self.partial_gradient_seen = None

    def forward_loss(self, rows):
        self.calls.append(len(rows))
        if self.ordinary_error:
            raise RuntimeError("ordinary non-OOM error")
        if self.max_rows is not None and len(rows) > self.max_rows:
            raise torch.cuda.OutOfMemoryError("simulated batch size OOM")
        trigger = len(self.calls) == 2 and not self.failure_spent
        if trigger:
            self.partial_gradient_seen = None if self.weight.grad is None else float(self.weight.grad)
        if trigger and self.fail_once_on_second == "forward":
            self.failure_spent = True
            raise torch.cuda.OutOfMemoryError("simulated OOM after one successful microbatch")
        loss = sum(((self.weight * (row[0][0] + 1) - row[2][0]) ** 2) * len(row[2]) for row in rows)
        if trigger and self.fail_once_on_second == "backward":
            def fail_backward(gradient):
                self.failure_spent = True
                raise torch.cuda.OutOfMemoryError("simulated backward OOM after partial gradients")
            loss.register_hook(fail_backward)
        tokens = sum(len(row[2]) for row in rows)
        sequence = sum(len(row[0]) + len(row[1]) + len(row[2]) for row in rows)
        return loss, tokens, sequence


class TestResearchRunnerOOMRetry(unittest.TestCase):
    def rows(self):
        # Unequal answer lengths detect incorrect per-microbatch averaging.
        return [([i], [1, 2], [2] * (i + 1), []) for i in range(6)]

    def expected(self):
        model = OOMRetryModel()
        rows = self.rows()
        loss, tokens, sequence = model.forward_loss(rows)
        (loss / tokens).backward()
        return float(loss.detach()), tokens, sequence, model.weight.grad.detach().clone()

    def run_group(self, model, size):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        original = model.weight.detach().clone()
        rows = self.rows()
        prepared = [rows[:3], rows[3:]]
        with patch.object(optimizer, "step", wraps=optimizer.step) as step, patch.object(torch.cuda, "empty_cache"):
            result = runner.train_accumulation_group(model, optimizer, prepared, list(model.parameters()), microbatch_size=size)
            step.assert_not_called()
        torch.testing.assert_close(model.weight.detach(), original, rtol=0, atol=0)
        return result

    def assert_expected(self, model, result):
        loss, tokens, sequence, gradient = self.expected()
        self.assertAlmostEqual(float(result["loss_sum"]), loss, places=5)
        self.assertEqual(result["supervised_tokens"], tokens)
        self.assertEqual(result["sequence_tokens"], sequence)
        torch.testing.assert_close(model.weight.grad, gradient, rtol=1e-6, atol=1e-6)

    def test_forward_batch_size_oom_halves_and_retries_whole_effective_batch(self):
        model = OOMRetryModel(max_rows=2)
        result = self.run_group(model, 4)
        self.assertEqual(result["microbatch_size"], 2)
        self.assertIn(4, model.calls)
        self.assertGreaterEqual(model.calls.count(2), 3)
        self.assert_expected(model, result)

    def test_forward_oom_after_partial_accumulation_discards_prior_gradients(self):
        model = OOMRetryModel(fail_once_on_second="forward")
        result = self.run_group(model, 2)
        self.assertEqual(result["microbatch_size"], 1)
        self.assertTrue(model.failure_spent)
        self.assertIsNotNone(model.partial_gradient_seen)
        self.assertNotEqual(model.partial_gradient_seen, 0)
        self.assertEqual(model.calls[-6:], [1] * 6)
        self.assert_expected(model, result)

    def test_backward_oom_after_partial_accumulation_discards_prior_gradients(self):
        model = OOMRetryModel(fail_once_on_second="backward")
        result = self.run_group(model, 2)
        self.assertEqual(result["microbatch_size"], 1)
        self.assertTrue(model.failure_spent)
        self.assertIsNotNone(model.partial_gradient_seen)
        self.assertNotEqual(model.partial_gradient_seen, 0)
        self.assert_expected(model, result)

    def test_non_oom_exceptions_propagate_without_retry(self):
        model = OOMRetryModel(ordinary_error=True)
        with self.assertRaisesRegex(RuntimeError, "ordinary non-OOM"):
            self.run_group(model, 4)
        self.assertEqual(len(model.calls), 1)

    def test_microbatch_one_oom_propagates_without_endless_retry(self):
        model = OOMRetryModel(max_rows=0)
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            self.run_group(model, 1)
        self.assertEqual(len(model.calls), 1)

    def test_unrecoverable_oom_discards_even_partial_gradients(self):
        model = OOMRetryModel(fail_once_on_second="forward")
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            self.run_group(model, 1)
        self.assertTrue(model.failure_spent)
        self.assertIsNotNone(model.partial_gradient_seen)
        self.assertIsNone(model.weight.grad)

    def test_empty_supervision_rejected_before_forward(self):
        model = OOMRetryModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, "supervised"):
            runner.train_accumulation_group(model, optimizer, [[([1], [2], [], [])]], list(model.parameters()), microbatch_size=1)
        self.assertEqual(model.calls, [])

    def test_logged_text_token_fraction_differs_from_record_fraction(self):
        model=OOMRetryModel()
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        rows=[([i],[1],[2]*length,[] if i<2 else [torch.zeros(1,4)])
              for i,length in enumerate([1,3,5,7,9,11])]
        with patch.object(torch.cuda,'empty_cache'):
            result=runner.train_accumulation_group(model,optimizer,[rows[:3],rows[3:]],list(model.parameters()),microbatch_size=2)
        self.assertEqual(result['text_only_records'],2)
        self.assertEqual(result['multimodal_records'],4)
        self.assertEqual(result['text_only_supervised_tokens'],4)
        self.assertEqual(result['multimodal_supervised_tokens'],32)
        self.assertAlmostEqual(result['text_only_supervised_fraction'],4/36)
        self.assertNotEqual(result['text_only_supervised_fraction'],2/6)

    def test_successful_group_preserves_requested_microbatch_size(self):
        model = OOMRetryModel()
        result = self.run_group(model, 2)
        self.assertEqual(result["microbatch_size"], 2)
        self.assert_expected(model, result)


class WeightedLossModel(nn.Module):
    def __init__(self,max_rows=None,fail_call=None,randomized=False):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(.75))
        self.calls=[];self.max_rows=max_rows;self.fail_call=fail_call
        self.randomized=randomized;self.failure_spent=False
    def forward_loss_components(self,rows):
        import numpy as np
        self.calls.append(len(rows))
        noise=torch.rand(len(rows))+random.random()+np.random.rand() if self.randomized else torch.ones(len(rows))
        trigger=len(self.calls)==2 and not self.failure_spent
        if self.max_rows is not None and len(rows)>self.max_rows:
            raise torch.cuda.OutOfMemoryError('weighted oversized microbatch')
        if trigger and self.fail_call=='forward':
            self.failure_spent=True
            raise torch.cuda.OutOfMemoryError('weighted forward partial OOM')
        mm=self.weight*0;text=self.weight*0;nmm=ntext=seq=0
        for index,row in enumerate(rows):
            n=len(row[2]);term=(self.weight*(row[0][0]+1)*noise[index]-row[2][0]).square()*n
            if row[3]:mm=mm+term;nmm+=n
            else:text=text+term;ntext+=n
            seq+=len(row[0])+len(row[1])+n
        if trigger and self.fail_call=='backward':
            def fail(_gradient):
                self.failure_spent=True
                raise torch.cuda.OutOfMemoryError('weighted backward partial OOM')
            mm.register_hook(fail)
        return {'multimodal_loss_sum':mm,'text_only_loss_sum':text,
                'multimodal_supervised_tokens':nmm,'text_only_supervised_tokens':ntext,'sequence_tokens':seq}


class TestWeightedEffectiveBatch(unittest.TestCase):
    def rows(self):
        return [([index],[1],[2]*length,[torch.zeros(1,4)] if index%2==0 else [])
                for index,length in enumerate([1,9,2,12,3,8])]
    def reference(self,rows,weight=.15):
        model=WeightedLossModel()
        parts=model.forward_loss_components(rows)
        mm=parts['multimodal_supervised_tokens'];text=parts['text_only_supervised_tokens']
        objective=(parts['multimodal_loss_sum']*((1-weight)/mm) if mm else model.weight*0)
        objective=objective+(parts['text_only_loss_sum']*(weight/text) if text else model.weight*0)
        objective.backward()
        return objective.item(),model.weight.grad.detach().clone()
    def run_group(self,model,rows,size):
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        with patch.object(torch.cuda,'empty_cache'),patch.object(optimizer,'step',wraps=optimizer.step) as step:
            result=runner.train_accumulation_group(model,optimizer,[rows[:2],rows[2:]],list(model.parameters()),
                                                  microbatch_size=size,text_loss_weight=.15)
            step.assert_not_called()
        return result

    def test_weighted_accumulation_matches_explicit_large_batch_with_unequal_lengths(self):
        rows=self.rows();value,gradient=self.reference(rows)
        for size in (1,2,4,6):
            with self.subTest(size=size):
                model=WeightedLossModel();result=self.run_group(model,rows,size)
                torch.testing.assert_close(model.weight.grad,gradient,atol=1e-6,rtol=1e-6)
                self.assertAlmostEqual(result['objective_value'],value,places=5)
                self.assertEqual(result['multimodal_supervised_tokens'],6)
                self.assertEqual(result['text_only_supervised_tokens'],29)
                self.assertAlmostEqual(result['text_only_supervised_fraction'],29/35)
                self.assertNotAlmostEqual(result['objective_value'],result['loss_sum']/35,places=3)

    def test_missing_stream_keeps_fixed_coefficient_without_renormalization(self):
        for keep_mm,coefficient in ((True,.85),(False,.15)):
            rows=[row for row in self.rows() if bool(row[3])==keep_mm]
            model=WeightedLossModel();result=self.run_group(model,rows,1)
            value,gradient=self.reference(rows)
            self.assertAlmostEqual(result['objective_value'],coefficient*result['loss_sum']/result['supervised_tokens'],places=5)
            self.assertAlmostEqual(result['objective_value'],value,places=5)
            torch.testing.assert_close(model.weight.grad,gradient,atol=1e-6,rtol=1e-6)
            self.assertEqual(result['missing_loss_streams'],['text_only'] if keep_mm else ['multimodal'])

    def test_forward_and_backward_oom_retry_preserve_whole_effective_denominators(self):
        rows=self.rows();value,gradient=self.reference(rows)
        for mode in ('forward','backward'):
            model=WeightedLossModel(fail_call=mode)
            result=self.run_group(model,rows,2)
            self.assertEqual(result['microbatch_size'],1)
            self.assertTrue(model.failure_spent)
            self.assertAlmostEqual(result['objective_value'],value,places=5)
            torch.testing.assert_close(model.weight.grad,gradient,atol=1e-6,rtol=1e-6)

    def test_oom_retry_restores_all_rng_streams_before_smaller_profile(self):
        import numpy as np
        rows=self.rows()
        def seed():torch.manual_seed(51);random.seed(52);np.random.seed(53)
        seed();baseline=WeightedLossModel(randomized=True)
        expected=self.run_group(baseline,rows,1)
        expected_torch=torch.get_rng_state();expected_python=random.getstate();expected_numpy=np.random.get_state()
        seed();retry=WeightedLossModel(fail_call='forward',randomized=True)
        actual=self.run_group(retry,rows,2)
        torch.testing.assert_close(retry.weight.grad,baseline.weight.grad,atol=0,rtol=0)
        self.assertEqual(actual['objective_value'],expected['objective_value'])
        self.assertTrue(torch.equal(torch.get_rng_state(),expected_torch))
        self.assertEqual(random.getstate(),expected_python)
        after=np.random.get_state()
        self.assertEqual(after[0],expected_numpy[0]);self.assertTrue(np.array_equal(after[1],expected_numpy[1]))
        self.assertEqual(after[2:],expected_numpy[2:])


if __name__ == "__main__":
    unittest.main()
