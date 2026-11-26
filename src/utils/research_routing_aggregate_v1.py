"""Opt-in transactional Small routing observations; canonical math is immutable.

GPU counters describe every valid token of successful optimizer windows. No
token activations are copied to CPU. Exports are explicit synchronization
boundaries. Fixed-reference/causal diagnostics remain separate required work.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[2]
OWNER = 'small_routing_aggregate_v1'
PINS = {
    'src/model/research_moe.py': 'f8cbfd6a8fdb7044d25d031f0a028055cb114a7ebd71a9b4f873dcd752674c6c',
    'src/model/smallthinker.py': '7d5aaa551b98e7958f1a69930c797f7906d7c770084b156ba36cc5d577cbdd26',
    'scripts/run_research_stage.py': 'f7fd22d1c94fb337af2be60d3324415b2b2c8cd3d62e9799f2da4c05ace87cfc',
}
OWN = ('src/utils/research_routing_aggregate_v1.py', 'tests/test_research_routing_aggregate_v1.py')
SCALARS = ('tokens', 'unfilled_topk_slots', 'unassigned_tokens', 'forced_new_tokens', 'bad_masks', 'bad_scores',
           'full_entropy_sum', 'eligible_entropy_sum', 'full_entropy_tokens', 'eligible_entropy_tokens', 'alpha_sum', 'alpha_sq_sum',
           'conditional_old_distribution_tokens', 'zero_old_mass_tokens', 'zero_full_mass_tokens', 'zero_eligible_mass_tokens')
EXPERTS = ('dispatch_counts', 'selected_weight_sum', 'pre_mask_mass_sum', 'eligible_mass_sum',
           'conditional_old_current_mass_sum', 'output_l2_sum', 'output_l2_sq_sum',
           'weighted_output_l2_sum', 'weighted_output_l2_sq_sum', 'norm_dispatch_counts', 'bad_norms')
INTEGER = {'tokens', 'unfilled_topk_slots', 'unassigned_tokens', 'forced_new_tokens', 'bad_masks', 'bad_scores',
           'full_entropy_tokens', 'eligible_entropy_tokens', 'dispatch_counts', 'norm_dispatch_counts', 'bad_norms'}
INTEGER |= {'conditional_old_distribution_tokens', 'zero_old_mass_tokens', 'zero_full_mass_tokens', 'zero_eligible_mass_tokens'}


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def state_signature(value):
    """Bind bounded diagnostic tensors and every plain metadata field."""
    import torch
    def walk(item):
        if torch.is_tensor(item):
            raw = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return {'tensor': True, 'dtype': str(item.dtype), 'shape': list(item.shape), 'sha256': hashlib.sha256(raw).hexdigest()}
        if isinstance(item, dict): return {key: walk(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)): return [walk(child) for child in item]
        return item
    return digest(walk({key: child for key, child in value.items() if key != 'state_sha256'}))


def source_binding():
    result = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (*PINS, *OWN)}
    if any(result[name] != expected for name, expected in PINS.items()): raise ValueError('Frozen Small routing source changed')
    return result


def profile(task_groups, *, gate_stride=10):
    groups = tuple(task_groups)
    if not groups or len(groups) > 32 or len(set(groups)) != len(groups) or any(not isinstance(x, str) or not x for x in groups):
        raise ValueError('Explicit unique raw task-group vocabulary required')
    if type(gate_stride) is not int or gate_stride < 1: raise ValueError('Positive gate snapshot stride required')
    return {'owner': OWNER, 'schema_version': 1, 'source_sha256': source_binding(), 'task_groups': list(groups),
        'gate_snapshot_stride': gate_stride, 'origin': 'fixed input image/text origin; padding excluded; teacher answers are text',
        'coverage': 'all successful optimizer-window microbatches; failed attempts and backward recomputation excluded',
        'counter_dtype': 'int64 counts and float64 sums; FP32 per-token probability/norm calculations',
        'pre_topk_probability_precision': 'original native activation precision, then FP32 diagnostic normalization/reductions; selected weights keep their actual represented values',
        'gate_snapshot_timing': 'post-update gates; scalar snapshot describes the just-completed pre-update forward window',
        'field_applicability': 'R3 alpha/conditional-current-old fields only for family3; learned gates only for policyc; absent fields are not measured zeros',
        'R3_hard_policy_a_alpha': 'observed score only; group competition is not identifiable after hard eligibility',
        'expert_output_norm': 'effective FFN output including modality input/output gates; separate actual weighted contribution norm',
        'shared_experts': 'structurally absent in verified Small paths', 'capacity_overflow': 'no capacity/drop mechanism in verified Small paths',
        'fixed_original_reference_router_scores_collected': False, 'causal_contextualized_state_influence_measured': False,
        'actual_GPU_parity_and_overhead_validated': False, 'P06_complete': False}


class TaggedRow(tuple):
    """The native four tuple values stay unchanged; only CPU provenance is added."""
    def __new__(cls, values, record):
        value = super().__new__(cls, values)
        value.routing_record_id = record['record_id']; value.routing_task_group = record['task_group']
        if not isinstance(value.routing_record_id, str) or not value.routing_record_id or not isinstance(value.routing_task_group, str):
            raise ValueError('Routing observations require raw record_id/task_group')
        return value


def _notify(owner, method, *args):
    collector = getattr(owner, '_routing_aggregate_observer', None)
    if collector is not None and collector.active:
        getattr(collector, method)(owner, *args)


def _function(relative, class_name, name):
    tree = ast.parse((ROOT / relative).read_bytes())
    body = tree.body if class_name is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


def _instrument_forward(module, *, native):
    node = _function('src/model/smallthinker.py' if native else 'src/model/research_moe.py',
        'SmallThinkerMoeBlock' if native else 'ResearchSmallThinkerMoE', 'forward')
    class Observe(ast.NodeTransformer):
        routes = norms = split = 0
        def visit_Assign(self, item):
            names = [n.id for n in item.targets if isinstance(n, ast.Name)]
            if native and names == ['delta'] and isinstance(item.value, ast.BinOp) and isinstance(item.value.left, ast.Call):
                self.split += 1
                before = ast.Assign([ast.Name('_routing_raw_delta', ast.Store())], item.value.left)
                item.value.left = ast.Name('_routing_raw_delta', ast.Load()); return [before, item]
            if names == ['weights'] and isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Attribute) and item.value.func.attr == 'to':
                self.routes += 1
                call = '_routing_notify(self, "native_route", logits, indices, weights)' if native else '_routing_notify(self, "route", scores, indices, weights, slot_valid, visual, valid, _routing_forced, alpha)'
                return [item, *ast.parse(call).body]
            if not native and names == ['need_new']:
                return [item, ast.Assign([ast.Name('_routing_forced', ast.Store())], ast.Name('need_new', ast.Load()))]
            return self.generic_visit(item)
        def visit_Expr(self, item):
            call = item.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'output' and call.func.attr == 'index_add_':
                self.norms += 1
                contribution = ast.Assign([ast.Name('_routing_contribution', ast.Store())], call.args[2])
                call.args[2] = ast.Name('_routing_contribution', ast.Load())
                raw = '_routing_raw_delta' if native else 'delta'
                observe = ast.parse(f'_routing_notify(self, "expert_norm", expert_idx, token_idx, {raw}, _routing_contribution)').body
                release = ast.parse('del _routing_contribution' + (', _routing_raw_delta' if native else '')).body
                return [contribution, *observe, item, *release]
            return self.generic_visit(item)
    edit = Observe(); node = edit.visit(node)
    if not native: node.body.insert(0, ast.parse('_routing_forced = None').body[0])
    if (edit.routes, edit.norms, edit.split) != (1, 1, 1 if native else 0): raise ValueError('Pinned Small forward observation sites changed')
    namespace = dict(module.__dict__, _routing_notify=_notify)
    namespace['smallthinker_grouped_dispatch'] = _instrument_grouped()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), module.__file__, 'exec'), namespace)
    return namespace['forward']


def _instrument_grouped():
    from src.model import smallthinker
    node = _function('src/model/smallthinker.py', None, 'smallthinker_grouped_dispatch')
    class Observe(ast.NodeTransformer):
        changed = 0
        def visit_Expr(self, item):
            call = item.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'index_add_' and isinstance(call.func.value, ast.Name) and call.func.value.id == 'output':
                self.changed += 1
                before = ast.Assign([ast.Name('_routing_contribution', ast.Store())], call.args[2])
                call.args[2] = ast.Name('_routing_contribution', ast.Load())
                return [before, *ast.parse('_routing_notify(owner, "grouped_norm", delta, _routing_contribution, token, counts, len(experts))').body,
                    item, *ast.parse('del _routing_contribution').body]
            return self.generic_visit(item)
    edit = Observe(); node = edit.visit(node)
    if edit.changed != 1: raise ValueError('Pinned grouped contribution site changed')
    namespace = dict(smallthinker.__dict__, _routing_notify=_notify)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), smallthinker.__file__, 'exec'), namespace)
    return namespace['smallthinker_grouped_dispatch']


class RoutingAggregate:
    def __init__(self, model, task_groups, *, gate_stride=10):
        import torch
        from src.model import research_moe, smallthinker
        task_groups = tuple(task_groups)
        self.profile = profile(task_groups, gate_stride=gate_stride); self.groups = {value: i for i, value in enumerate(task_groups)}
        self.model = model; self.active = False; self.phase = 'idle'; self.step = 0; self.seen = 0; self.tokens_seen = 0; self.coverage_sha256 = digest([])
        self.layers = {}; self.restores = []; self.snapshots = []; self.discarded_attempts = 0
        candidates = []
        for name, block in model.named_modules():
            if isinstance(block, (smallthinker.SmallThinkerMoeBlock, research_moe.ResearchSmallThinkerMoE)) and type(block) not in (smallthinker.SmallThinkerMoeBlock, research_moe.ResearchSmallThinkerMoE):
                raise ValueError('Routing collector refuses unsupported Small subclasses')
            native = type(block) is smallthinker.SmallThinkerMoeBlock
            if not native and type(block) is not research_moe.ResearchSmallThinkerMoE: continue
            if any(key in block.__dict__ for key in ('forward', '_router_logits', '_grouped_original_dispatch', '_routing_aggregate_observer')):
                raise ValueError('Routing collector refuses an already overridden Small path')
            candidates.append((name, block, native))
        if not candidates: raise ValueError('No exact supported native/expanded Small routing layers')
        grouped = _instrument_grouped()
        for name, block, native in candidates:
            old = len(block.experts); added = 0 if native else block.added_count
            device = next(block.parameters()).device
            info = {'name': name, 'block': block, 'native': native, 'old': old, 'added': added, 'experts': old + added,
                'family': 4 if native else block.family, 'policy': 'b' if native else block.policy,
                'softmax': bool(block.config.moe_primary_router_apply_softmax), 'top_k': int(block.config.moe_num_active_primary_experts), 'device': device}
            total = old + added
            def buffers():
                return {key: torch.zeros((len(self.groups), 2) + ((total,) if key in EXPERTS else ()),
                    dtype=torch.int64 if key in INTEGER else torch.float64, device=device) for key in (*SCALARS, *EXPERTS)}
            info.update(stage=buffers(), cumulative=buffers()); self.layers[name] = info
            self._set(block, '_routing_aggregate_observer', self)
            self._set(block, 'forward', types.MethodType(_instrument_forward(smallthinker if native else research_moe, native=native), block))
            if not native:
                def dispatch(owner, states, visual, weights, order, counts, k):
                    return grouped(owner, owner.experts, states, weights, order, counts, visual, owner.modality_gate_logits)
                self._set(block, '_grouped_original_dispatch', types.MethodType(dispatch, block))
                if block.family == 3:
                    original = block._router_logits
                    def logits(owner, inputs, original=original):
                        result = original(inputs)
                        _notify(owner, 'old_logits', result)
                        return result
                    self._set(block, '_router_logits', types.MethodType(logits, block))
        self.by_id = {id(value['block']): value for value in self.layers.values()}

    def _set(self, block, name, value):
        self.restores.append((block, name, name in block.__dict__, block.__dict__.get(name)))
        setattr(block, name, value)

    def close(self):
        self.active = False
        for block, name, existed, value in reversed(self.restores):
            if existed: setattr(block, name, value)
            else: delattr(block, name)
        self.restores.clear()

    def begin_window(self, prepared):
        if self.phase != 'idle': raise ValueError('Previous routing window was not committed/aborted')
        rows = [row for batch in prepared for row in batch]
        if not rows or any(not isinstance(row, TaggedRow) or row.routing_task_group not in self.groups for row in rows):
            raise ValueError('Every training row needs declared raw task-group provenance')
        self.window_rows = rows; self.phase = 'window'; self.attempt_records = []; self.attempt_tokens = 0

    def begin_attempt(self):
        if self.phase != 'window' or self.active: raise ValueError('Invalid routing attempt boundary')
        for info in self.layers.values():
            for tensor in info['stage'].values(): tensor.zero_()
        self.attempt_records = []; self.attempt_tokens = 0; self.phase = 'attempt'

    def abort_attempt(self):
        if self.phase == 'idle': return
        self.active = False
        if self.phase in ('attempt', 'ready'): self.discarded_attempts += 1
        self.phase = 'window'
        for info in self.layers.values():
            for tensor in info['stage'].values(): tensor.zero_()
        self.attempt_records = []; self.attempt_tokens = 0

    def abort_window(self):
        if self.phase != 'idle': self.abort_attempt()
        self.phase = 'idle'; self.window_rows = []

    @contextmanager
    def forward_scope(self, rows):
        if self.phase != 'attempt' or not self.model.training:
            yield; return
        if self.active: raise ValueError('Nested routing forward scope')
        if any(not isinstance(row, TaggedRow) or row.routing_task_group not in self.groups for row in rows):
            raise ValueError('Untagged/mixed routing microbatch')
        self.scope_rows = rows; self.scope_cache = {}; self.visited = set(); self.active = True
        try:
            yield
            if self.visited != set(self.layers): raise ValueError('Routing forward did not observe every layer exactly once')
            self.attempt_records.extend({'record_id': row.routing_record_id, 'task_group': row.routing_task_group,
                'tokens': len(row[0]) + len(row[1]) + len(row[2]) + sum(x.shape[0] for x in row[3])} for row in rows)
            self.attempt_tokens += sum(item['tokens'] for item in self.attempt_records[-len(rows):])
        finally:
            self.active = False; self.scope_rows = None; self.scope_cache = {}

    def finish_attempt(self):
        if self.phase != 'attempt' or self.active or len(self.attempt_records) != len(self.window_rows):
            raise ValueError('Incomplete successful routing forward/backward window')
        if [item['record_id'] for item in self.attempt_records] != [row.routing_record_id for row in self.window_rows]:
            raise ValueError('Routing microbatch order differs from logical optimizer window')
        self.phase = 'ready'

    def commit(self, step, seen):
        if self.phase != 'ready' or type(step) is not int or step != self.step + 1 or seen != self.seen + len(self.window_rows):
            raise ValueError('Routing commits require the successful native optimizer/scheduler/sync step boundary')
        for info in self.layers.values():
            for key in info['stage']: info['cumulative'][key].add_(info['stage'][key])
        self.step = step; self.seen = seen; self.tokens_seen += self.attempt_tokens
        self.coverage_sha256 = digest([self.coverage_sha256, step, self.attempt_records])
        if step == 1 or step % self.profile['gate_snapshot_stride'] == 0:
            self.snapshots.append({'step': step, 'gates': {name: value['block'].modality_gate_logits.detach().sigmoid().clone()
                for name, value in self.layers.items() if not value['native'] and value['block'].modality_gate_logits is not None},
                'committed_window_scalars': {name: {key: value['stage'][key].clone() for key in
                    ('tokens', 'alpha_sum', 'alpha_sq_sum', 'full_entropy_sum', 'eligible_entropy_sum', 'unfilled_topk_slots')}
                    for name, value in self.layers.items()}})
        self.phase = 'idle'; self.window_rows = []; self.attempt_records = []

    def _labels(self, info):
        import torch
        device = info['device']; key = str(device)
        if key not in self.scope_cache:
            lengths = [len(r[0]) + len(r[1]) + len(r[2]) + sum(x.shape[0] for x in r[3]) for r in self.scope_rows]
            width = max(lengths); positions = torch.arange(width, device=device)[None, :]
            length = torch.tensor(lengths, device=device)[:, None]
            start = torch.tensor([len(r[0]) for r in self.scope_rows], device=device)[:, None]
            images = torch.tensor([sum(x.shape[0] for x in r[3]) for r in self.scope_rows], device=device)[:, None]
            groups = torch.tensor([self.groups[r.routing_task_group] for r in self.scope_rows], device=device)[:, None].expand(-1, width).reshape(-1)
            valid = (positions < length).reshape(-1)
            visual = ((positions >= start) & (positions < start + images)).reshape(-1) & valid
            self.scope_cache[key] = groups, visual, valid
        return self.scope_cache[key]

    def _scalar(self, info, key, values, valid, visual):
        import torch
        groups, _, _ = self._labels(info); index = groups * 2 + visual.long()
        target = info['stage'][key].reshape(-1)
        source = torch.where(valid, values, 0).to(target.dtype)
        target.scatter_add_(0, index, source)

    def _expert(self, info, key, values, indices, valid, visual, groups=None):
        import torch
        if groups is None: groups, _, _ = self._labels(info)
        total = info['experts']; index = (groups * 2 + visual.long())[:, None] * total + indices
        target = info['stage'][key].reshape(-1)
        target.scatter_add_(0, index.reshape(-1), torch.where(valid, values, 0).to(target.dtype).reshape(-1))

    def old_logits(self, owner, logits):
        import torch
        info = self.by_id[id(owner)]
        with torch.no_grad():
            groups, visual, valid = self._labels(info)
            scores = logits.detach().float().softmax(-1) if info['softmax'] else logits.detach().float().sigmoid()
            denominator = scores.sum(-1, keepdim=True)
            probability = scores / denominator.clamp_min(torch.finfo(torch.float32).tiny)
            usable = valid & torch.isfinite(scores).all(-1) & denominator[:, 0].gt(0)
            self._scalar(info, 'conditional_old_distribution_tokens', torch.ones_like(valid).long(), usable, visual)
            self._scalar(info, 'zero_old_mass_tokens', denominator[:, 0].le(0).long(), valid, visual)
            indices = torch.arange(info['old'], device=logits.device)[None, :].expand(len(groups), -1)
            self._expert(info, 'conditional_old_current_mass_sum', probability, indices, usable[:, None], visual)

    def native_route(self, owner, logits, indices, weights):
        import torch
        info = self.by_id[id(owner)]; _, visual, valid = self._labels(info)
        with torch.no_grad():
            scores = logits.detach().softmax(-1, dtype=torch.float32) if info['softmax'] else logits.detach().sigmoid().float()
        self.route(owner, scores, indices, weights, valid[:, None].expand_as(indices), visual, valid, None, None)

    def route(self, owner, scores, indices, weights, slot_valid, visual, valid, forced, alpha):
        import torch
        info = self.by_id[id(owner)]
        if info['name'] in self.visited: raise ValueError('Duplicate routing layer in original microbatch forward')
        self.visited.add(info['name'])
        with torch.no_grad():
            groups, expected_visual, expected_valid = self._labels(info)
            if len(groups) != len(indices): raise ValueError('Routing token shape differs from expanded/padded rows')
            visual = visual.detach(); valid = valid.detach(); slots = slot_valid.detach(); weights = weights.detach().float(); scores = scores.detach().float()
            ones = torch.ones_like(valid, dtype=torch.int64)
            self._scalar(info, 'tokens', ones, valid, visual)
            self._scalar(info, 'bad_masks', ((visual != expected_visual) | (valid != expected_valid)).long(), expected_valid | valid, visual)
            self._scalar(info, 'unfilled_topk_slots', (~slots).sum(-1), valid, visual)
            self._scalar(info, 'unassigned_tokens', (~slots.any(-1)).long(), valid, visual)
            if forced is not None: self._scalar(info, 'forced_new_tokens', forced.detach().long(), valid, visual)
            self._expert(info, 'dispatch_counts', torch.ones_like(indices), indices, slots, visual)
            self._expert(info, 'selected_weight_sum', weights, indices, slots, visual)
            eligible = torch.ones_like(scores, dtype=torch.bool)
            if info['policy'] == 'a':
                eligible[:, :info['old']] = ~visual[:, None]; eligible[:, info['old']:] = visual[:, None]
            eligible &= valid[:, None]
            finite = torch.isfinite(scores).all(-1)
            self._scalar(info, 'bad_scores', (~finite).long(), valid, visual)
            tiny = torch.finfo(torch.float32).tiny
            full_sum = scores.sum(-1, keepdim=True); p = scores / full_sum.clamp_min(tiny)
            allowed = scores * eligible; allowed_sum = allowed.sum(-1, keepdim=True); q = allowed / allowed_sum.clamp_min(tiny)
            self._scalar(info, 'zero_full_mass_tokens', full_sum[:, 0].le(0).long(), valid & finite, visual)
            self._scalar(info, 'zero_eligible_mass_tokens', allowed_sum[:, 0].le(0).long(), valid & finite, visual)
            expert_ids = torch.arange(info['experts'], device=scores.device)[None, :].expand_as(scores)
            self._expert(info, 'pre_mask_mass_sum', p, expert_ids, (valid & finite)[:, None], visual)
            self._expert(info, 'eligible_mass_sum', q, expert_ids, (valid & finite)[:, None], visual)
            for prefix, probability, denominator in (('full', p, full_sum), ('eligible', q, allowed_sum)):
                entropy = -(probability * probability.clamp_min(tiny).log()).sum(-1)
                usable = valid & finite & denominator[:, 0].gt(0)
                self._scalar(info, prefix + '_entropy_sum', entropy, usable, visual)
                self._scalar(info, prefix + '_entropy_tokens', ones, usable, visual)
            if info['family'] == 3:
                if alpha is None: raise ValueError('R3 actual group-mass alpha was not observed')
                observed_alpha = alpha.detach().reshape(-1)
                self._scalar(info, 'alpha_sum', observed_alpha, valid, visual)
                self._scalar(info, 'alpha_sq_sum', observed_alpha.square(), valid, visual)

    def expert_norm(self, owner, expert, token, effective, contribution):
        import torch
        info = self.by_id[id(owner)]
        ids = torch.full_like(token, expert)
        self._norm(info, ids, token, effective, contribution)

    def grouped_norm(self, owner, effective, contribution, token, counts, expert_count):
        import torch
        info = self.by_id[id(owner)]
        ids = torch.repeat_interleave(torch.arange(expert_count, device=token.device),
            torch.tensor(counts[:expert_count], device=token.device), output_size=len(token))
        self._norm(info, ids, token, effective, contribution)

    def _norm(self, info, ids, token, effective, contribution):
        import torch
        with torch.no_grad(), torch.autocast(device_type=effective.device.type, enabled=False):
            groups, visual, valid = self._labels(info)
            original = torch.linalg.vector_norm(effective.detach(), dim=-1, dtype=torch.float32)
            weighted = torch.linalg.vector_norm(contribution.detach(), dim=-1, dtype=torch.float32)
            finite = torch.isfinite(original) & torch.isfinite(weighted)
            visual = visual[token]; keep = valid[token]; groups = groups[token]; indices = ids[:, None]
            for key, values in (('norm_dispatch_counts', torch.ones_like(ids)), ('bad_norms', (~finite).long()),
                                ('output_l2_sum', original), ('output_l2_sq_sum', original.square()),
                                ('weighted_output_l2_sum', weighted), ('weighted_output_l2_sq_sum', weighted.square())):
                self._expert(info, key, values[:, None], indices, keep[:, None], visual, groups=groups)

    def export_state(self, *, step, seen):
        import torch
        if self.phase != 'idle' or step != self.step or seen != self.seen: raise ValueError('Aggregate checkpoint must match a committed native checkpoint boundary')
        layers = {}
        for name, info in self.layers.items():
            counters = {key: value.detach().cpu().clone() for key, value in info['cumulative'].items()}
            if any(not torch.isfinite(value).all() for value in counters.values()): raise ValueError('Nonfinite aggregate observations')
            if any(torch.count_nonzero(counters[key]) for key in ('bad_masks', 'bad_scores', 'bad_norms')):
                raise ValueError('Invalid mask/score/norm observations prevent certified aggregate export')
            if not torch.equal(counters['dispatch_counts'], counters['norm_dispatch_counts']): raise ValueError('Expert norm/dispatch coverage differs')
            if int(counters['tokens'].sum()) != self.tokens_seen: raise ValueError('Layer routing coverage differs from all committed valid tokens')
            layers[name] = {'metadata': {key: info[key] for key in ('native', 'old', 'added', 'experts', 'family', 'policy', 'softmax', 'top_k')}, 'counters': counters}
        state = {'owner': OWNER, 'profile': self.profile, 'step': self.step, 'seen': self.seen, 'observed_valid_tokens': self.tokens_seen,
            'coverage_sha256': self.coverage_sha256, 'discarded_attempts': self.discarded_attempts, 'layers': layers,
            'gate_snapshots': [{'step': item['step'], 'gates': {key: value.detach().cpu().clone() for key, value in item['gates'].items()},
                'committed_window_scalars': {name: {key: value.detach().cpu().clone() for key, value in values.items()}
                    for name, values in item['committed_window_scalars'].items()}} for item in self.snapshots],
            'shared_expert_contribution': {'present': False, 'structural_contribution': 0},
            'capacity_overflow_and_dropped_tokens': {'mechanism_present': False, 'structural_count': 0},
            'fixed_original_reference_router_scores_collected': False, 'causal_contextualized_state_influence_measured': False,
            'P06_complete': False}
        state['state_sha256'] = state_signature(state)
        return state

    def restore_state(self, state, *, step, seen):
        import torch
        if self.phase != 'idle' or self.step or self.seen or state.get('owner') != OWNER or state.get('profile') != self.profile or state.get('step') != step or state.get('seen') != seen:
            raise ValueError('Missing/mixed routing aggregate resume checkpoint binding')
        if state.get('state_sha256') != state_signature(state): raise ValueError('Routing aggregate state checksum differs')
        if type(state.get('observed_valid_tokens')) is not int or state['observed_valid_tokens'] < 0:
            raise ValueError('Routing aggregate valid-token coverage is missing')
        if set(state.get('layers', {})) != set(self.layers): raise ValueError('Routing aggregate layer census differs')
        pending = []
        for name, info in self.layers.items():
            saved = state['layers'][name]
            expected = {key: info[key] for key in ('native', 'old', 'added', 'experts', 'family', 'policy', 'softmax', 'top_k')}
            if saved.get('metadata') != expected or set(saved.get('counters', {})) != set(info['cumulative']): raise ValueError('Routing aggregate architecture/counter schema differs')
            for key, target in info['cumulative'].items():
                value = saved['counters'][key]
                if not torch.is_tensor(value) or value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all() or (key in INTEGER and (value < 0).any()): raise ValueError('Routing aggregate saved counter shape/dtype/value differs')
                pending.append((target, value))
            if int(saved['counters']['tokens'].sum()) != state['observed_valid_tokens'] or not torch.equal(saved['counters']['dispatch_counts'], saved['counters']['norm_dispatch_counts']):
                raise ValueError('Routing aggregate saved coverage differs')
        snapshots = [{'step': item['step'], 'gates': {name: value.to(self.layers[name]['device']).clone() for name, value in item['gates'].items()},
            'committed_window_scalars': {name: {key: value.to(self.layers[name]['device']).clone() for key, value in values.items()}
                for name, values in item['committed_window_scalars'].items()}} for item in state['gate_snapshots']]
        for target, value in pending: target.copy_(value.to(target.device))
        self.step = step; self.seen = seen; self.tokens_seen = state['observed_valid_tokens']; self.coverage_sha256 = state['coverage_sha256']; self.discarded_attempts = state['discarded_attempts']
        self.snapshots = snapshots
