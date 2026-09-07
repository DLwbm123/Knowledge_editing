"""Selective-write primitives. No router, benchmark or safety claims."""
import math
import random
from collections import defaultdict
from statistics import mean

import torch
from torch import nn

from methods.medtrace.core import AsymmetricCPExpert

CONDITIONS = ("W0_TASK_ONLY", "W1_KL_0.1", "W1_KL_1", "W1_KL_10", "W2_GROUP_CONSTRAINED")
RELATIONS = {"H": "same_question_different_image_conflicting_source_answer", "U": "broad_unrelated_source_qa"}


class LowRankExpert(nn.Module):
    def __init__(self, cp, seed, rank=16):
        super().__init__()
        if rank < cp.rank:
            raise ValueError("lossless transfer requires rank >= CP rank")
        self.d_in, self.d_out, self.rank = cp.d_in, cp.d_out, rank
        self.epsilon = cp.epsilon
        rng = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.randn(rank, cp.d_in, generator=rng, dtype=torch.float32)
        a /= a.norm(dim=1, keepdim=True)
        self.A = nn.Parameter(a.to(cp.rho.device))
        self.B = nn.Parameter(torch.zeros(cp.d_out, rank, device=cp.rho.device))
        with torch.no_grad():
            self.A[:cp.rank].copy_(cp.input_basis().T)
            self.B[:, :cp.rank].copy_(cp.output_basis() * cp.rho * (cp.beta / math.sqrt(cp.rank)))

    def residual(self, activation):
        a = activation.to(self.A.dtype)
        a = a / (a.square().mean(dim=-1, keepdim=True).sqrt() + self.epsilon)
        return (a @ self.A.T) @ self.B.T

    @torch.no_grad()
    def normalize_factors_(self, **_):
        norms = self.A.norm(dim=1).clamp_min(self.epsilon)
        self.A.div_(norms[:, None])
        self.B.mul_(norms[None, :])


def optimizer_for(expert, backbone):
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("backbone must be frozen before optimizer construction")
    if isinstance(expert, AsymmetricCPExpert):
        inputs, outputs = [expert.u_in, expert.v_in], [expert.u_out, expert.v_out, expert.rho]
    else:
        inputs, outputs = [expert.A], [expert.B]
    if {id(p) for p in backbone.parameters()} & {id(p) for p in inputs + outputs}:
        raise ValueError("expert and backbone parameters overlap")
    if any(p.dtype != torch.float32 for p in inputs + outputs):
        raise ValueError("FP32 expert master parameters required")
    return torch.optim.Adam([{"params": inputs, "lr": 1e-4}, {"params": outputs, "lr": 1e-3}],
                            betas=(.9, .999), eps=1e-8, weight_decay=0)


def predictor_mask(labels, attention=None):
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("only batch-size-one teacher trajectories are supported")
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :-1] = labels[:, 1:] != -100
    if attention is not None:
        mask[:, :-1] &= attention[:, 1:].bool() & attention[:, :-1].bool()
    if not mask.any():
        raise ValueError("no answer predictors")
    return mask


def full_vocab_kl(student_logits, teacher_logp, chunk=16):
    """FP32 Base||student, equal positions, full vocab; teacher may live on CPU."""
    if student_logits.ndim != 2 or student_logits.shape != teacher_logp.shape or not len(student_logits):
        raise ValueError("teacher/student predictor or vocabulary mismatch")
    if teacher_logp.requires_grad:
        raise ValueError("teacher must be detached")
    total = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, len(student_logits), chunk):
        q = teacher_logp[start:start+chunk].to(student_logits.device, dtype=torch.float32)
        p = student_logits[start:start+chunk].float().log_softmax(-1)
        total = total + (q.exp() * (q - p)).sum()
    result = total / len(student_logits)
    if not torch.isfinite(result):
        raise FloatingPointError("nonfinite full-vocabulary KL")
    return result


def fit_groups(rows, group):
    """A training caller cannot silently turn calibration/evaluation into fit."""
    grouped, seen = defaultdict(list), set()
    for row in rows:
        if row["role"] != "fit" or row["label"] != "negative" or row["fact_relation"] != RELATIONS[group]:
            raise ValueError("unapproved role/relation in negative fit pool")
        key = row["eqkey"]
        if key not in seen:
            grouped[row["source_group"]].append(row)
            seen.add(key)
    if not grouped:
        raise ValueError(f"unsupported fit stratum {group}")
    return {k: sorted(v, key=lambda r: r["eqkey"]) for k, v in sorted(grouped.items())}


def balanced_schedule(groups, steps, seed):
    rng = random.Random(seed)
    keys = list(groups)
    order, row_order, cursors = [], {}, defaultdict(int)
    for key, rows in groups.items():
        row_order[key] = list(range(len(rows)))
        rng.shuffle(row_order[key])
    while len(order) < steps:
        cycle = keys.copy()
        rng.shuffle(cycle)
        for key in cycle:
            cursor = cursors[key]
            indices = row_order[key]
            if cursor and cursor % len(indices) == 0:
                rng.shuffle(indices)
            order.append(groups[key][indices[cursor % len(indices)]])
            cursors[key] += 1
    return order[:steps]


def group_mean(values, rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["source_group"]][row["eqkey"]] = values[row["logical_id"]]
    if not groups:
        raise ValueError("empty metric stratum")
    return mean(mean(v.values()) for v in groups.values())


class Protection:
    """Detached fit-only primal-dual state, initialized before any optimizer step."""
    def __init__(self, initial, condition):
        if condition not in CONDITIONS or set(initial) != {"H", "U"}:
            raise ValueError("unknown condition or incomplete fit groups")
        if any(not math.isfinite(k) or k < -1e-5 for k in initial.values()):
            raise FloatingPointError("invalid initial KL")
        self.condition = condition
        self.scale = {g: max(k, .001) for g, k in initial.items()}
        self.epsilon = {g: max(.5*k, .0001) for g, k in initial.items()}
        self.ema = {g: k/self.scale[g] for g, k in initial.items()}
        self.dual = dict(H=.5, U=.5)
        self.saturation = dict(H=0, U=0)

    def coefficient(self, group):
        if self.condition == CONDITIONS[0]:
            return 0.
        weight = self.dual[group] if self.condition == CONDITIONS[-1] else float(self.condition.split("_")[-1])/2
        return weight/self.scale[group]

    def update(self, observed, *, role="fit"):
        if role != "fit" or set(observed) != {"H", "U"}:
            raise ValueError("dual state accepts complete fit observations only")
        for g, value in observed.items():
            value = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
            if not math.isfinite(value):
                raise FloatingPointError("nonfinite dual observation")
            self.ema[g] = .9*self.ema[g] + .1*value/self.scale[g]
            if self.condition == CONDITIONS[-1]:
                self.dual[g] = min(20., max(0., self.dual[g]+.05*(self.ema[g]-self.epsilon[g]/self.scale[g])))
                self.saturation[g] += self.dual[g] == 20.


def select_global_lambda(calibration, expected_edits=7):
    """One selection per parameterization; never per edit or from evaluation."""
    if len(calibration) != expected_edits or len({r["edit"] for r in calibration}) != expected_edits:
        raise ValueError("incomplete calibration cohort")
    if any(r["role"] != "calibration" for r in calibration):
        raise ValueError("evaluation cannot select lambda")
    eligible = []
    macro = lambda c, k: mean(r[c][k] for r in calibration)
    for lam in (.1, 1., 10.):
        c = f"W1_KL_{lam:g}"
        pos = macro(c, "positive_semantic")
        if (pos >= macro("A2", "positive_semantic")-.05 and pos >= macro(CONDITIONS[0], "positive_semantic")-.05
                and macro(c, "U_kl") <= macro(CONDITIONS[0], "U_kl")):
            eligible.append((macro(c, "H_kl"), lam))
    return {"lambda": min(eligible)[1] if eligible else 1., "qualified": bool(eligible),
            "eligible": [lam for _, lam in sorted(eligible)], "role": "calibration"}
