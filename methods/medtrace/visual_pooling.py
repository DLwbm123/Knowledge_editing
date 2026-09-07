"""Fixed pre-CP pooling ablation with a frozen question head."""
import torch
import torch.nn.functional as F

from .frozen_verifier import LinearApplicabilityVerifier, _group_mean, l2_normalize, verifier_features

GROUPS = ("G0", "G1", "G2")


def pooled_feature(expert, prompt, visual, kind):
    if prompt.ndim != 1 or visual.ndim != 2 or not len(visual):
        raise ValueError("pooling requires one prompt and nonempty visual tokens")
    if kind == "G0":
        return verifier_features(expert, prompt, visual).pre_cp_visual
    uq = expert.normalize_activation(prompt)
    ui = expert.normalize_activation(visual)
    if kind == "G1":
        return l2_normalize(ui.mean(0))
    if kind == "G2":
        weights = F.softmax((l2_normalize(ui) @ l2_normalize(uq)) / 0.1, dim=0)
        return l2_normalize((weights[:, None] * ui).sum(0))
    raise ValueError(f"unknown pooling: {kind}")


def fit_image_head(question_state, features, rows):
    """Rows must be the original sorted matched fit rows, never cal/eval."""
    if not rows or any(row["role"] != "fit" for row in rows):
        raise ValueError("fit-only supervision required")
    if [row["logical_id"] for row in rows] != sorted(row["logical_id"] for row in rows):
        raise ValueError("frozen pair enumeration requires sorted logical IDs")
    labels = torch.tensor([r["label"] == "positive" for r in rows], device=features.device)
    if len(features) != len(rows) or not labels.any() or not (~labels).any():
        raise ValueError("aligned two-class fit supervision required")
    indices = {r["logical_id"]: i for i, r in enumerate(rows)}
    pairs = [(indices[r["positive_logical_id"]], i) for i, r in enumerate(rows) if r["label"] == "negative"]
    groups = [[r["source_group"] for r, keep in zip(rows, mask.tolist(), strict=True) if keep] for mask in (labels, ~labels)]
    verifier = LinearApplicabilityVerifier(features.shape[-1]).to(features.device)
    verifier.question.load_state_dict(question_state)
    verifier.question.requires_grad_(False)
    optimizer = torch.optim.Adam(verifier.image.parameters(), lr=0.01, weight_decay=0)
    curve = []
    for step in range(1, 801):
        optimizer.zero_grad(set_to_none=True)
        logits = verifier.image(features).squeeze(-1)
        bce = 0.5 * (_group_mean(F.softplus(-logits[labels]), groups[0]) + _group_mean(F.softplus(logits[~labels]), groups[1]))
        margin = torch.stack([F.softplus(1 - (logits[p] - logits[n])) for p, n in pairs]).mean()
        l2 = 0.001 * verifier.image.weight.square().sum()
        loss = bce + 0.5 * margin + l2
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(verifier.image.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError("nonfinite image-head fit")
        optimizer.step()
        if step in {1, 80, 320, 800}:
            curve.append({"step": step, "loss": loss.item(), "bce": bce.item(), "pair_margin": margin.item(), "l2": l2.item(), "gradient_norm": norm.item()})
    if any(not torch.equal(question_state[k].to(v.device), v) for k, v in verifier.question.state_dict().items()):
        raise RuntimeError("changed frozen question head")
    return verifier, curve
