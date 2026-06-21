import copy
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _expand_query(query: torch.Tensor, batch_size: int) -> torch.Tensor:
    return query.expand(batch_size, -1, -1)


def _require_nonempty_mask(mask: torch.Tensor, name: str) -> None:
    if mask is None:
        raise RuntimeError(f"LiveEdit requires `{name}` but it was not provided.")
    if mask.dim() != 2:
        raise RuntimeError(f"LiveEdit `{name}` must be [batch, seq], got {tuple(mask.shape)}.")
    empty = mask.bool().sum(dim=1) == 0
    if empty.any():
        raise RuntimeError(f"LiveEdit `{name}` has no selected tokens for batch rows {empty.nonzero().flatten().tolist()}.")


def _get_mask(batch: Dict[str, Any], hidden: torch.Tensor, key: str) -> torch.Tensor:
    if key not in batch or batch[key] is None:
        raise RuntimeError(f"LiveEdit requires full-sequence masks; missing ['{key}'].")
    mask = batch[key].to(hidden.device).bool()
    if mask.shape != hidden.shape[:2]:
        raise RuntimeError(
            f"LiveEdit `{key}` shape {tuple(mask.shape)} does not match hidden sequence {tuple(hidden.shape[:2])}."
        )
    return mask


def get_liveedit_routing_masks(batch: Dict[str, Any], hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = ("vision_mask", "prompt_mask", "attention_mask")
    missing = [key for key in required if key not in batch or batch[key] is None]
    if missing:
        raise RuntimeError(f"LiveEdit requires full-sequence masks; missing {missing}.")
    vision_mask = _get_mask(batch, hidden, "vision_mask")
    prompt_mask = _get_mask(batch, hidden, "prompt_mask")
    attention_mask = _get_mask(batch, hidden, "attention_mask")
    _require_nonempty_mask(prompt_mask, "prompt_mask")
    if batch.get("image") is not None:
        _require_nonempty_mask(vision_mask, "vision_mask")
    if (vision_mask & prompt_mask).any():
        raise RuntimeError("LiveEdit region masks must be disjoint.")
    if ((vision_mask | prompt_mask) & ~attention_mask).any():
        raise RuntimeError("LiveEdit region masks include padding positions.")
    return vision_mask, prompt_mask, attention_mask


def get_liveedit_masks(batch: Dict[str, Any], hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vision_mask, prompt_mask, attention_mask = get_liveedit_routing_masks(batch, hidden)
    answer_mask = _get_mask(batch, hidden, "answer_mask")
    _require_nonempty_mask(answer_mask, "answer_mask")
    if (vision_mask & answer_mask).any() or (prompt_mask & answer_mask).any():
        raise RuntimeError("LiveEdit region masks must be disjoint.")
    if (answer_mask & ~attention_mask).any():
        raise RuntimeError("LiveEdit region masks include padding positions.")
    return vision_mask, prompt_mask, answer_mask, attention_mask


class CrossAttentionReadout(nn.Module):
    """Multi-head cross-attention readout used by official LiveEdit modules."""

    def __init__(
        self,
        query_dim: int,
        token_dim: int,
        module_dim: int,
        output_dim: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if module_dim % num_heads != 0:
            raise ValueError("module_dim must be divisible by num_heads.")
        if output_dim % num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.qk_head_dim = module_dim // num_heads
        self.v_head_dim = output_dim // num_heads
        self.scale = self.qk_head_dim ** -0.5
        self.q_proj = nn.Linear(query_dim, module_dim, bias=bias)
        self.k_proj = nn.Linear(token_dim, module_dim, bias=bias)
        self.v_proj = nn.Linear(token_dim, output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()

    def forward(self, query: torch.Tensor, tokens: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if query.dim() != 3 or tokens.dim() != 3:
            raise RuntimeError("CrossAttentionReadout expects query and tokens shaped [batch, seq, dim].")
        if query.shape[0] == 1 and tokens.shape[0] != 1:
            query = _expand_query(query, tokens.shape[0])
        if query.shape[0] != tokens.shape[0]:
            raise RuntimeError("CrossAttentionReadout batch sizes do not match.")
        if token_mask is not None:
            token_mask = token_mask.to(tokens.device).bool()
            _require_nonempty_mask(token_mask, "token_mask")

        batch_size, query_len, _ = query.shape
        token_len = tokens.shape[1]
        q = self.q_proj(query).reshape(batch_size, query_len, self.num_heads, self.qk_head_dim)
        k = self.k_proj(tokens).reshape(batch_size, token_len, self.num_heads, self.qk_head_dim)
        v = self.v_proj(tokens).reshape(batch_size, token_len, self.num_heads, self.v_head_dim)
        scores = torch.einsum("blhd,bmhd->blmh", q, k).float() * self.scale
        if token_mask is not None:
            scores = scores.masked_fill(~token_mask[:, None, :, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=2).to(v.dtype)
        out = torch.einsum("blmh,bmhd->blhd", weights, v)
        return out.reshape(batch_size, query_len, -1)


class LowRankGenerator(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int,
        module_dim: int,
        num_heads: int,
        lora_scale: float,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, rank, module_dim))
        self.readout = CrossAttentionReadout(module_dim, hidden_size, module_dim, hidden_size, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.scale = 1.0 / (float(lora_scale) * math.sqrt(rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.query)
        self.readout.reset_parameters()
        self.layer_norm.reset_parameters()

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.layer_norm(hidden)
        return self.readout(self.query, hidden, mask) * self.scale


class ExpertGenerator(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int,
        module_dim: int,
        num_heads: int,
        lora_scale: float,
    ) -> None:
        super().__init__()
        self.u_generator = LowRankGenerator(hidden_size, rank, module_dim, num_heads, lora_scale)
        self.v_generator = LowRankGenerator(hidden_size, rank, module_dim, num_heads, lora_scale)

    def reset_parameters(self) -> None:
        self.u_generator.reset_parameters()
        self.v_generator.reset_parameters()

    def forward(self, hidden: torch.Tensor, edit_signal_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.u_generator(hidden, edit_signal_mask), self.v_generator(hidden, edit_signal_mask)


class LiveEditFeatureExtractor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        feature_k: int,
        module_dim: int,
        num_heads: int,
        sentinel_tokens: Optional[int] = None,
        with_sentinel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_k = feature_k
        self.module_dim = module_dim
        self.prompt_norm_for_vision = nn.LayerNorm(hidden_size)
        self.vision_norm = nn.LayerNorm(hidden_size)
        self.prompt_norm = nn.LayerNorm(hidden_size)
        self.phi_query = nn.Parameter(torch.empty(1, feature_k, module_dim))
        self.psi_query = nn.Parameter(torch.empty(1, feature_k, module_dim))
        self.prompt_to_phi = CrossAttentionReadout(module_dim, hidden_size, module_dim, module_dim, num_heads)
        self.vision_to_phi = CrossAttentionReadout(module_dim, hidden_size, module_dim, module_dim, num_heads)
        self.prompt_to_psi = CrossAttentionReadout(module_dim, hidden_size, module_dim, module_dim, num_heads)
        if with_sentinel:
            sentinel_tokens = sentinel_tokens or 32
            self.vision_sentinel = nn.Parameter(torch.empty(1, sentinel_tokens, hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.prompt_norm_for_vision.reset_parameters()
        self.vision_norm.reset_parameters()
        self.prompt_norm.reset_parameters()
        nn.init.kaiming_normal_(self.phi_query)
        nn.init.kaiming_normal_(self.psi_query)
        self.prompt_to_phi.reset_parameters()
        self.vision_to_phi.reset_parameters()
        self.prompt_to_psi.reset_parameters()
        if hasattr(self, "vision_sentinel"):
            nn.init.kaiming_normal_(self.vision_sentinel)

    def extract_vision(self, hidden: torch.Tensor, vision_mask: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
        _require_nonempty_mask(vision_mask, "vision_mask")
        _require_nonempty_mask(prompt_mask, "prompt_mask")
        prompt_hidden = self.prompt_norm_for_vision(hidden)
        vision_hidden = self.vision_norm(hidden)
        prompt_readout = self.prompt_to_phi(self.phi_query, prompt_hidden, prompt_mask)
        return self.vision_to_phi(prompt_readout, vision_hidden, vision_mask)

    def extract_prompt(self, hidden: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
        _require_nonempty_mask(prompt_mask, "prompt_mask")
        prompt_hidden = self.prompt_norm(hidden)
        return self.prompt_to_psi(self.psi_query, prompt_hidden, prompt_mask)

    def extract(self, hidden: torch.Tensor, vision_mask: torch.Tensor, prompt_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.extract_vision(hidden, vision_mask, prompt_mask), self.extract_prompt(hidden, prompt_mask)

    def extract_sentinel(self, hidden: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "vision_sentinel"):
            raise RuntimeError("This LiveEditFeatureExtractor was created without a vision sentinel.")
        _require_nonempty_mask(prompt_mask, "prompt_mask")
        batch_size = hidden.shape[0]
        prompt_hidden = self.prompt_norm_for_vision(hidden)
        prompt_readout = self.prompt_to_phi(self.phi_query, prompt_hidden, prompt_mask)
        sentinel = self.vision_sentinel.expand(batch_size, -1, -1).to(hidden.device, hidden.dtype)
        sentinel_mask = torch.ones(sentinel.shape[:2], dtype=torch.bool, device=hidden.device)
        return self.vision_to_phi(prompt_readout, sentinel, sentinel_mask)


class ExpertRepository(nn.Module):
    def __init__(
        self,
        rank: int,
        hidden_size: int,
        feature_k: int,
        module_dim: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.hidden_size = int(hidden_size)
        self.feature_k = int(feature_k)
        self.module_dim = int(module_dim)
        self.metadata: List[Dict[str, Any]] = []
        self.register_buffer("u", torch.empty(0, rank, hidden_size, device=device, dtype=dtype))
        self.register_buffer("v", torch.empty(0, rank, hidden_size, device=device, dtype=dtype))
        self.register_buffer("phi", torch.empty(0, feature_k, module_dim, device=device, dtype=dtype))
        self.register_buffer("psi", torch.empty(0, feature_k, module_dim, device=device, dtype=dtype))

    def __len__(self) -> int:
        return int(self.u.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def clear(self) -> None:
        self.u = self.u[:0]
        self.v = self.v[:0]
        self.phi = self.phi[:0]
        self.psi = self.psi[:0]
        self.metadata = []

    def _validate(self, u: torch.Tensor, v: torch.Tensor, phi: torch.Tensor, psi: torch.Tensor) -> None:
        expected = {
            "u": (self.rank, self.hidden_size),
            "v": (self.rank, self.hidden_size),
            "phi": (self.feature_k, self.module_dim),
            "psi": (self.feature_k, self.module_dim),
        }
        actual = {"u": u.shape[1:], "v": v.shape[1:], "phi": phi.shape[1:], "psi": psi.shape[1:]}
        for key in expected:
            if tuple(actual[key]) != expected[key]:
                raise RuntimeError(f"LiveEdit repository expected {key} shape (*, {expected[key]}), got {tuple(actual[key])}.")
        counts = {u.shape[0], v.shape[0], phi.shape[0], psi.shape[0]}
        if len(counts) != 1:
            raise RuntimeError("LiveEdit repository tensors must have the same batch count.")

    def append(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        phi: torch.Tensor,
        psi: torch.Tensor,
        metadata: Optional[Iterable[Dict[str, Any]]] = None,
        detach: bool = True,
    ) -> None:
        if u.dim() == 2:
            u, v, phi, psi = u.unsqueeze(0), v.unsqueeze(0), phi.unsqueeze(0), psi.unsqueeze(0)
        self._validate(u, v, phi, psi)
        if detach:
            u, v, phi, psi = u.detach(), v.detach(), phi.detach(), psi.detach()
        u = u.to(device=self.u.device, dtype=self.u.dtype)
        v = v.to(device=self.v.device, dtype=self.v.dtype)
        phi = phi.to(device=self.phi.device, dtype=self.phi.dtype)
        psi = psi.to(device=self.psi.device, dtype=self.psi.dtype)
        self.u = torch.cat([self.u, u], dim=0)
        self.v = torch.cat([self.v, v], dim=0)
        self.phi = torch.cat([self.phi, phi], dim=0)
        self.psi = torch.cat([self.psi, psi], dim=0)
        if metadata is None:
            self.metadata.extend({} for _ in range(u.shape[0]))
        else:
            metadata = list(metadata)
            if len(metadata) != u.shape[0]:
                raise RuntimeError("LiveEdit metadata count must match appended expert count.")
            self.metadata.extend(copy.deepcopy(metadata))

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "hidden_size": self.hidden_size,
            "feature_k": self.feature_k,
            "module_dim": self.module_dim,
            "metadata": copy.deepcopy(self.metadata),
        }

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        if state is None:
            return
        self.metadata = copy.deepcopy(state.get("metadata", []))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        for name in ("u", "v", "phi", "psi"):
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "ExpertRepository":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LiveEdit expert repository not found: {path}")
        try:
            payload = torch.load(path, map_location=map_location or "cpu")
        except Exception as exc:
            raise RuntimeError(f"LiveEdit expert repository could not be loaded from {path}: {exc}") from exc
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        if not isinstance(state, dict):
            raise RuntimeError(f"LiveEdit expert repository at {path} is invalid: expected a state dict.")
        required = ("u", "v", "phi", "psi")
        missing = [key for key in required if key not in state]
        if missing:
            raise RuntimeError(f"LiveEdit expert repository at {path} is invalid: missing tensors {missing}.")
        extra = state.get("_extra_state", {})
        repo = cls(
            rank=extra.get("rank", state["u"].shape[1]),
            hidden_size=extra.get("hidden_size", state["u"].shape[2]),
            feature_k=extra.get("feature_k", state["phi"].shape[1]),
            module_dim=extra.get("module_dim", state["phi"].shape[2]),
            dtype=state["u"].dtype,
        )
        repo.load_state_dict(state)
        return repo


def liveedit_similarity(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    mode: str = "inner_product",
    scale_dim: Optional[int] = None,
) -> torch.Tensor:
    if lhs.dim() != 3 or rhs.dim() != 3:
        raise RuntimeError("LiveEdit similarity expects [batch, feature_k, dim] tensors.")
    if lhs.shape[1:] != rhs.shape[1:]:
        raise RuntimeError(f"LiveEdit similarity shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}.")
    if mode == "cosine":
        lhs_flat = F.normalize(lhs.flatten(1).float(), dim=-1)
        rhs_flat = F.normalize(rhs.flatten(1).float(), dim=-1)
        return lhs_flat @ rhs_flat.t()
    if mode != "inner_product":
        raise ValueError(f"Unsupported LiveEdit similarity mode: {mode}")
    scale_dim = scale_dim or lhs.shape[-1]
    return torch.einsum("bkd,nkd->bnk", lhs.float(), rhs.float()).mean(dim=-1) / math.sqrt(scale_dim)


def hard_route(
    phi_bar: torch.Tensor,
    phi_hat: torch.Tensor,
    phi_theta: torch.Tensor,
    similarity: str = "inner_product",
    hard_topk: Optional[int] = None,
    force_topk_when_empty: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if phi_hat.shape[0] == 0:
        empty = torch.zeros(phi_bar.shape[0], 0, dtype=torch.bool, device=phi_bar.device)
        threshold = torch.zeros(phi_bar.shape[0], 1, dtype=phi_bar.dtype, device=phi_bar.device)
        sim = torch.zeros_like(empty, dtype=phi_bar.dtype)
        return empty, sim, threshold
    sim = liveedit_similarity(phi_bar, phi_hat, similarity, phi_bar.shape[-1]).to(phi_bar.dtype)
    threshold = liveedit_similarity(phi_bar, phi_theta, similarity, phi_bar.shape[-1]).diag().unsqueeze(1).to(phi_bar.dtype)
    selected = sim > threshold
    if hard_topk is not None and hard_topk > 0 and sim.shape[1] > hard_topk:
        topk = torch.topk(sim, k=hard_topk, dim=1).indices
        topk_mask = torch.zeros_like(selected)
        topk_mask.scatter_(1, topk, True)
        selected = selected & topk_mask
    if force_topk_when_empty and selected.shape[1] > 0:
        empty_rows = selected.sum(dim=1) == 0
        if empty_rows.any():
            best = sim[empty_rows].argmax(dim=1, keepdim=True)
            selected[empty_rows] = False
            selected[empty_rows].scatter_(1, best, True)
    return selected, sim, threshold


def soft_routing_weights(
    psi_bar: torch.Tensor,
    psi_hat: torch.Tensor,
    selected: torch.Tensor,
    similarity: str = "inner_product",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if psi_hat.shape[0] == 0:
        weights = torch.zeros(psi_bar.shape[0], 0, dtype=psi_bar.dtype, device=psi_bar.device)
        sims = torch.zeros_like(weights)
        return weights, sims
    sims = liveedit_similarity(psi_bar, psi_hat, similarity, psi_bar.shape[-1]).to(psi_bar.dtype)
    selected = selected.to(sims.device).bool()
    masked = sims.masked_fill(~selected, torch.finfo(sims.dtype).min)
    rel = torch.softmax(masked.float(), dim=1).to(sims.dtype)
    rel = torch.where(selected, rel, torch.zeros_like(rel))
    abs_weight = torch.sigmoid(sims)
    return abs_weight * rel, sims


def low_rank_residual(hidden: torch.Tensor, u: torch.Tensor, v: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if u.shape[0] == 0 or weights.shape[1] == 0:
        return torch.zeros_like(hidden)
    hidden_dtype = hidden.dtype
    activations = torch.relu(torch.einsum("btd,nrd->btnr", hidden.float(), u.float()))
    residual = torch.einsum("btnr,nrd,bn->btd", activations, v.float(), weights.float())
    return residual.to(hidden_dtype)


def apply_liveedit_residual_to_output(output: Any, residual: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return output + residual
    if isinstance(output, tuple):
        values = list(output)
        values[0] = values[0] + residual
        return tuple(values)
    if isinstance(output, list):
        values = list(output)
        values[0] = values[0] + residual
        return values
    if isinstance(output, dict):
        values = output.copy()
        key = "last_hidden_state" if "last_hidden_state" in values else next(iter(values))
        values[key] = values[key] + residual
        return values
    if hasattr(output, "last_hidden_state"):
        new_hidden = output.last_hidden_state + residual
        if is_dataclass(output):
            try:
                return replace(output, last_hidden_state=new_hidden)
            except TypeError:
                pass
        output.last_hidden_state = new_hidden
        return output
    raise RuntimeError(f"Unsupported LiveEdit hook output type: {type(output)}")


def first_hidden_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        key = "last_hidden_state" if "last_hidden_state" in output else next(iter(output))
        return output[key]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise RuntimeError(f"Unsupported LiveEdit hook output type: {type(output)}")


def find_module_by_path(model: nn.Module, module_path: str) -> nn.Module:
    current: Any = model
    for part in module_path.split("."):
        try:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"LiveEdit path `{module_path}` could not resolve component `{part}`.") from exc
    if not isinstance(current, nn.Module):
        raise ValueError(f"LiveEdit path `{module_path}` did not resolve to a module.")
    return current


@dataclass
class LiveEditContext:
    batch: Optional[Dict[str, Any]] = None
    u: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    phi: Optional[torch.Tensor] = None
    psi: Optional[torch.Tensor] = None
    capture_only: bool = False
    force_all_experts: bool = False
    enabled: bool = True


def info_nce_loss(alpha: torch.Tensor, beta_positive: torch.Tensor, beta_candidates: torch.Tensor, similarity: str) -> torch.Tensor:
    logits = liveedit_similarity(alpha, beta_candidates, similarity, alpha.shape[-1])
    positive_logits = liveedit_similarity(alpha, beta_positive, similarity, alpha.shape[-1]).diag()
    denom = torch.logsumexp(logits, dim=1)
    return -(positive_logits - denom).mean()


def liveedit_routing_losses(
    phi_hat_g: torch.Tensor,
    psi_hat_g: torch.Tensor,
    phi_bar_g: torch.Tensor,
    psi_bar_g: torch.Tensor,
    phi_bar_l: torch.Tensor,
    psi_hat_l: torch.Tensor,
    phi_theta_g: torch.Tensor,
    phi_theta_l: torch.Tensor,
    similarity: str = "inner_product",
) -> Dict[str, torch.Tensor]:
    eps = 1.0e-8
    batch_size = phi_hat_g.shape[0]
    scale_dim = phi_hat_g.shape[-1]
    hard_g_edit_logits = liveedit_similarity(phi_bar_g, phi_hat_g, similarity, scale_dim)
    hard_g_sentinel = liveedit_similarity(phi_bar_g, phi_theta_g, similarity, scale_dim).diag().unsqueeze(1)
    hard_g_logits = torch.cat([hard_g_edit_logits, hard_g_sentinel], dim=1)
    hard_g_target = torch.arange(batch_size, device=phi_hat_g.device)
    hard_g = F.cross_entropy(hard_g_logits, hard_g_target)

    hard_l_edit_logits = liveedit_similarity(phi_bar_l, phi_hat_g, similarity, scale_dim)
    hard_l_sentinel = liveedit_similarity(phi_bar_l, phi_theta_l, similarity, scale_dim).diag().unsqueeze(1)
    hard_l_logits = torch.cat([hard_l_edit_logits, hard_l_sentinel], dim=1)
    hard_l_target = torch.full((batch_size,), batch_size, dtype=torch.long, device=phi_hat_g.device)
    hard_l = F.cross_entropy(hard_l_logits, hard_l_target)
    hr = hard_g + hard_l

    prompt_candidates = torch.cat([psi_hat_g, psi_hat_l], dim=0)
    prompt_logits = liveedit_similarity(psi_bar_g, prompt_candidates, similarity, psi_hat_g.shape[-1])
    pos = prompt_logits[torch.arange(batch_size, device=prompt_logits.device), torch.arange(batch_size, device=prompt_logits.device)]
    neg_logits = prompt_logits.clone()
    neg_logits[torch.arange(batch_size, device=prompt_logits.device), torch.arange(batch_size, device=prompt_logits.device)] = torch.finfo(prompt_logits.dtype).min
    if neg_logits.shape[1] <= 1:
        raise RuntimeError("LiveEdit sr1 requires at least one negative prompt feature.")
    neg = neg_logits.argmax(dim=1)
    neg_values = prompt_logits[torch.arange(batch_size, device=prompt_logits.device), neg]
    sr1 = -(torch.log(torch.sigmoid(pos) + eps) + torch.log(1.0 - torch.sigmoid(neg_values) + eps)).mean()
    sr2 = F.cross_entropy(prompt_logits, hard_g_target)
    return {
        "hr": hr,
        "sr1": sr1,
        "sr2": sr2,
        "hard_g": hard_g,
        "hard_l": hard_l,
    }


@contextmanager
def temporary_context(obj: Any, context: LiveEditContext):
    old_context = getattr(obj, "_liveedit_context", None)
    obj._liveedit_context = context
    try:
        yield
    finally:
        obj._liveedit_context = old_context
