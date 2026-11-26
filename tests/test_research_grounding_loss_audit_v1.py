import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.eval.research_grounding_loss_audit_v1 import answer_character_fields, token_fields


def test_coordinate_spans_do_not_capture_digits_in_labels():
    answer = '[{"bbox_2d":[0,12,999,1000],"label":"item42"}]'
    fields = answer_character_fields(answer)
    assert sum(k == 'coordinate' for k in fields) == 10
    assert fields[answer.index('42')] == 'label'
    suffix = answer + '<eos>'
    result = token_fields(answer, suffix, [(i, i + 1) for i in range(len(suffix))])
    assert result['coordinate_overlap_tokens'] == 10
    assert result['fields']['terminator'] == 5


def test_partial_or_malformed_coordinate_supervision_is_rejected():
    answer = '[{"bbox_2d":[0,0,1000,1000],"label":"cat"}]'
    with pytest.raises(ValueError, match='lost'):
        token_fields(answer, answer, [(0, answer.index('1000'))])
    with pytest.raises(ValueError):
        answer_character_fields('[{"bbox_2d":[0,0,0,0],"label":"cat"}]')
    assert answer_character_fields('[]') == ['format', 'format']


class NativeMathOnCPU(ast.NodeTransformer):
    def visit_Constant(self, node):
        # Test-only device substitution. The live trainer and its source stay frozen.
        if node.value == 'cuda':
            return ast.copy_location(ast.Constant('cpu'), node)
        return node


def test_frozen_training_loss_and_gradients_match_masked_autoregressive_CE(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    path = Path(__file__).resolve().parents[1] / 'scripts/run_research_stage.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ResearchModel')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_forward_loss')
    private = ast.Module(body=[NativeMathOnCPU().visit(copy.deepcopy(function))], type_ignores=[])
    scope = dict(torch=torch, F=F)
    exec(compile(ast.fix_missing_locations(private), str(path), 'exec'), scope)

    class CausalBody(nn.Module):
        def forward(self, inputs_embeds, attention_mask, **kwargs):
            return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1))

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.model = CausalBody()
            self.lm_head = nn.Linear(8, 16, bias=False)

        def get_input_embeddings(self):
            return self.embed

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.llm = Language()
            self.base = self.llm
            self.projector = nn.Linear(4, 8)
            self.wrappers = []

    torch.manual_seed(42)
    model = Model()
    reference = copy.deepcopy(model)
    rows = [([1, 2], [3], [4, 5, 6], [torch.randn(2, 4)]),
            ([], [2, 1], [7, 6], [])]
    loss, count, _ = scope['_forward_loss'](model, rows)
    expected = 0
    for pre, post, answer, features in rows:
        embed = reference.llm.embed
        chunks = [embed(torch.tensor(pre, dtype=torch.long))]
        chunks += [reference.projector(f) for f in features]
        chunks += [embed(torch.tensor(post + answer))]
        hidden = torch.cat(chunks).cumsum(dim=0)
        start = len(pre) + len(post) + sum(f.shape[0] for f in features) - 1
        logits = reference.llm.lm_head(hidden[start:start + len(answer)])
        expected = expected + F.cross_entropy(logits, torch.tensor(answer), reduction='sum')
    assert count == 5
    torch.testing.assert_close(loss, expected)
    loss.backward()
    expected.backward()
    for (name, actual), (other, target) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert name == other
        torch.testing.assert_close(actual.grad, target.grad)
