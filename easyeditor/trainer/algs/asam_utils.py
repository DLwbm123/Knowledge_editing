from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)
_POOL_MASK_WARNING_EMITTED = False
_HOOK_FALLBACK_WARNING_EMITTED = False
_MASK_TRIM_WARNING_EMITTED = False


@dataclass
class AsamAlignmentResult:
    loss: torch.Tensor
    info: Dict[str, float]
    deltas: List[torch.Tensor]


def normalized_alg_name(config) -> str:
    return str(getattr(config, "alg", "")).upper().replace("-", "_")


def asam_enabled(config) -> bool:
    default_enabled = normalized_alg_name(config) in {"ASAM_FT", "ASAM_MEND"}
    return bool(getattr(config, "asam_enabled", default_enabled))


def asam_use_lar(config) -> bool:
    return bool(getattr(config, "asam_use_lar", True))


def asam_epsilon(config) -> float:
    return float(getattr(config, "asam_epsilon", 1.0e-3))


def asam_num_variants(config) -> int:
    return int(getattr(config, "asam_num_variants", 4))


def asam_lar_step_size(config) -> float:
    return float(getattr(config, "asam_lar_step_size", asam_epsilon(config)))


def asam_beta(config) -> float:
    return float(getattr(config, "asam_beta", 10.0))


def asam_tau(config) -> float:
    return float(getattr(config, "asam_tau", 4.0))


def asam_use_dataset_variants(config) -> bool:
    return bool(getattr(config, "asam_use_dataset_variants", False))


def asam_variant_ce_weight(config) -> float:
    return float(getattr(config, "asam_variant_ce_weight", 0.0))


def asam_pooling(config) -> str:
    return str(getattr(config, "asam_pooling", "prompt_or_image_only"))


def asam_debug_grad_check(config) -> bool:
    return bool(getattr(config, "asam_debug_grad_check", False))


def asam_debug_require_all_inner_grads(config) -> bool:
    return bool(getattr(config, "asam_debug_require_all_inner_grads", False))


def asam_capture_module(config) -> Optional[str]:
    value = getattr(config, "asam_capture_module", None)
    return str(value) if value not in (None, "", "null") else None


def asam_alignment_params(config) -> List[str]:
    params = getattr(config, "asam_alignment_params", None)
    if params:
        return list(params)
    return list(getattr(config, "inner_params", []) or [])


def _base_model(editable_model):
    return editable_model.model if hasattr(editable_model, "model") else editable_model


def _model_output_field(output, name: str):
    if output is None:
        return None
    if isinstance(output, dict):
        return output.get(name)
    return getattr(output, name, None)


def _output_logits(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    logits = _model_output_field(output, "logits")
    if torch.is_tensor(logits):
        return logits
    raise RuntimeError("ASAM forward expected a tensor or an output object/dict containing `.logits`.")


def _first_tensor(value) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    if hasattr(value, "last_hidden_state") and torch.is_tensor(value.last_hidden_state):
        return value.last_hidden_state
    if hasattr(value, "hidden_states") and value.hidden_states is not None:
        tensor = _first_tensor(value.hidden_states)
        if tensor is not None:
            return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def get_module_by_path(model, module_path: str) -> torch.nn.Module:
    module = model
    if not module_path:
        return module
    for comp in module_path.split("."):
        if hasattr(module, comp):
            module = getattr(module, comp)
        elif comp.isdigit() and hasattr(module, "__getitem__"):
            module = module[int(comp)]
        else:
            raise RuntimeError(f"Could not resolve ASAM capture module component `{comp}` in `{module_path}`")
    if not isinstance(module, torch.nn.Module):
        raise RuntimeError(f"ASAM capture path `{module_path}` did not resolve to a torch.nn.Module")
    return module


def _parent_module_path(param_name: str) -> str:
    parts = param_name.split(".")
    if len(parts) <= 1:
        return ""
    return ".".join(parts[:-1])


def _common_prefix(paths: Sequence[str]) -> str:
    split_paths = [path.split(".") for path in paths if path]
    if not split_paths:
        return ""
    prefix: List[str] = []
    for comps in zip(*split_paths):
        if len(set(comps)) != 1:
            break
        prefix.append(comps[0])
    return ".".join(prefix)


def _numeric_block_path(param_name: str) -> Optional[Tuple[str, int, str]]:
    parts = param_name.split(".")
    best: Optional[Tuple[str, int, str]] = None
    # Exclude the final parameter name; numeric module indices before it can
    # identify transformer blocks such as decoder.layers.31.
    for idx, part in enumerate(parts[:-1]):
        if part.isdigit() and idx > 0:
            collection = ".".join(parts[:idx])
            block_index = int(part)
            block_path = ".".join(parts[: idx + 1])
            best = (collection, block_index, block_path)
    return best


def infer_asam_capture_module_path(model, config) -> str:
    explicit = asam_capture_module(config)
    if explicit:
        get_module_by_path(model, explicit)
        return explicit

    inner_params = list(getattr(config, "inner_params", []) or [])
    if not inner_params:
        raise ValueError("ASAM cannot infer a capture module because config.inner_params is empty.")

    numeric_blocks = [_numeric_block_path(param) for param in inner_params]
    if all(item is not None for item in numeric_blocks):
        collections = {item[0] for item in numeric_blocks if item is not None}
        if len(collections) == 1:
            selected = max((item for item in numeric_blocks if item is not None), key=lambda item: item[1])
            get_module_by_path(model, selected[2])
            return selected[2]

    parent_paths = [_parent_module_path(param) for param in inner_params]
    common = _common_prefix(parent_paths)
    if common:
        try:
            get_module_by_path(model, common)
            return common
        except RuntimeError:
            pass

    raise ValueError(
        "ASAM could not safely infer a capture module covering all edited parameters. "
        "Set `asam_capture_module`, e.g. `opt_model.model.decoder.layers.31`."
    )


def find_asam_capture_module(model, config) -> Tuple[torch.nn.Module, str]:
    module_path = infer_asam_capture_module_path(model, config)
    return get_module_by_path(model, module_path), module_path


def register_asam_output_hook(module, storage: List[torch.Tensor]):
    """Capture module output first; only fall back to input if output has no tensor."""

    def hook(_, inputs, output):
        global _HOOK_FALLBACK_WARNING_EMITTED
        tensor = _first_tensor(output)
        if tensor is None:
            tensor = _first_tensor(inputs)
            if tensor is not None and not _HOOK_FALLBACK_WARNING_EMITTED:
                LOG.warning("ASAM hook fell back to module input because output had no tensor.")
                _HOOK_FALLBACK_WARNING_EMITTED = True
        if tensor is not None:
            storage.append(tensor)

    return module.register_forward_hook(hook)


@contextlib.contextmanager
def asam_latent_context(model, delta: Optional[torch.Tensor] = None, capture: bool = False):
    had_delta = hasattr(model, "_asam_latent_delta")
    old_delta = getattr(model, "_asam_latent_delta", None)
    had_capture = hasattr(model, "_asam_capture_latent")
    old_capture = getattr(model, "_asam_capture_latent", None)
    try:
        setattr(model, "_asam_latent_delta", delta)
        setattr(model, "_asam_capture_latent", capture)
        if hasattr(model, "_asam_last_latent"):
            delattr(model, "_asam_last_latent")
        if hasattr(model, "_asam_last_perturb_mask"):
            delattr(model, "_asam_last_perturb_mask")
        yield
    finally:
        if had_delta:
            setattr(model, "_asam_latent_delta", old_delta)
        elif hasattr(model, "_asam_latent_delta"):
            delattr(model, "_asam_latent_delta")
        if had_capture:
            setattr(model, "_asam_capture_latent", old_capture)
        elif hasattr(model, "_asam_capture_latent"):
            delattr(model, "_asam_capture_latent")


def maybe_apply_asam_delta(
    model,
    latent: torch.Tensor,
    perturb_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Called by supported model wrappers after joint vision-language embeddings are built."""
    capture = getattr(model, "_asam_capture_latent", False)
    delta = getattr(model, "_asam_latent_delta", None)
    active = capture or delta is not None

    if active and perturb_mask is None:
        raise NotImplementedError(
            "ASAM LAR requires a perturb_mask aligned to joint input embeddings. "
            "The mask must allow image/prompt positions and forbid answer/padding positions."
        )
    if perturb_mask is not None:
        if perturb_mask.shape != latent.shape[:2]:
            raise ValueError(
                f"ASAM perturb_mask shape {tuple(perturb_mask.shape)} does not match latent sequence {tuple(latent.shape[:2])}"
            )
        perturb_mask = perturb_mask.to(device=latent.device, dtype=torch.bool)

    if capture:
        model._asam_last_latent = latent
        model._asam_last_perturb_mask = perturb_mask
    if delta is None:
        return latent
    if delta.shape != latent.shape:
        raise ValueError(f"ASAM latent delta shape {tuple(delta.shape)} does not match latent {tuple(latent.shape)}")
    masked_delta = delta.to(device=latent.device, dtype=latent.dtype) * perturb_mask.unsqueeze(-1).to(latent.dtype)
    return latent + masked_delta


def get_last_asam_latent_and_mask(model) -> Tuple[torch.Tensor, torch.Tensor]:
    latent = getattr(model, "_asam_last_latent", None)
    perturb_mask = getattr(model, "_asam_last_perturb_mask", None)
    if latent is None or perturb_mask is None:
        raise NotImplementedError(
            "asam_use_lar=true requires this backbone to expose both joint embeddings and a perturb_mask "
            "through `maybe_apply_asam_delta`. Supported local wrappers: Blip2OPT and MiniGPT4."
        )
    return latent, perturb_mask


def _labels_from_output_or_batch(output, batch: dict) -> Optional[torch.Tensor]:
    labels = _model_output_field(output, "labels")
    if torch.is_tensor(labels):
        return labels
    labels = batch.get("labels") if isinstance(batch, dict) else None
    return labels if torch.is_tensor(labels) else None


def _attention_from_output_or_batch(output, batch: dict) -> Optional[torch.Tensor]:
    attention = _model_output_field(output, "attention_mask")
    if torch.is_tensor(attention):
        return attention
    attention = batch.get("attention_mask") if isinstance(batch, dict) else None
    return attention if torch.is_tensor(attention) else None


def _perturb_mask_from_output_or_batch(output, batch: dict) -> Optional[torch.Tensor]:
    mask = _model_output_field(output, "asam_perturb_mask")
    if torch.is_tensor(mask):
        return mask
    mask = batch.get("asam_perturb_mask") if isinstance(batch, dict) else None
    return mask if torch.is_tensor(mask) else None


def _align_mask_to_length(
    mask: torch.Tensor,
    seq_len: int,
    *,
    mask_name: str,
    allow_trim: bool = False,
) -> Optional[torch.Tensor]:
    global _MASK_TRIM_WARNING_EMITTED
    if mask.dim() != 2:
        return None
    if mask.shape[1] == seq_len:
        return mask
    if allow_trim and mask.shape[1] > seq_len:
        if not _MASK_TRIM_WARNING_EMITTED:
            LOG.warning(
                "ASAM pooling aligned %s by taking the leftmost %d positions from a mask of length %d.",
                mask_name,
                seq_len,
                mask.shape[1],
            )
            _MASK_TRIM_WARNING_EMITTED = True
        return mask[:, :seq_len]
    return None


def pool_representation(
    rep: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    perturb_mask: Optional[torch.Tensor] = None,
    pooling: str = "prompt_or_image_only",
    require_reliable_mask: bool = False,
    allow_attention_fallback: bool = False,
) -> torch.Tensor:
    """Pool hidden states to one [D] vector per batch example."""
    global _POOL_MASK_WARNING_EMITTED

    rep = rep.to(torch.float32)
    if rep.dim() == 1:
        return rep.unsqueeze(0)
    if rep.dim() == 2:
        return rep
    if rep.dim() > 3:
        rep = rep.reshape(rep.shape[0], -1, rep.shape[-1])

    batch, seq_len, _ = rep.shape
    mask: Optional[torch.Tensor] = None
    rejection_reasons: List[str] = []

    if pooling == "prompt_or_image_only" and perturb_mask is not None:
        aligned_perturb = _align_mask_to_length(perturb_mask.to(rep.device), seq_len, mask_name="asam_perturb_mask")
        if aligned_perturb is not None:
            mask = aligned_perturb.bool()
            if not mask.any(dim=1).all():
                rejection_reasons.append("asam_perturb_mask has no allowed prompt/image token for at least one example")
                mask = None
        else:
            rejection_reasons.append(
                f"asam_perturb_mask shape {tuple(perturb_mask.shape)} is not aligned to captured sequence length {seq_len}"
            )

    if mask is None and pooling == "prompt_or_image_only" and labels is not None:
        aligned_labels = _align_mask_to_length(labels.to(rep.device), seq_len, mask_name="labels")
        aligned_attention = (
            _align_mask_to_length(attention_mask.to(rep.device), seq_len, mask_name="attention_mask")
            if attention_mask is not None
            else None
        )
        if aligned_labels is not None and aligned_attention is not None:
            mask = aligned_labels.eq(-100)
            mask = mask & aligned_attention.bool()
            if not mask.any(dim=1).all():
                rejection_reasons.append("labels/attention_mask produced no allowed prompt/image token for at least one example")
                mask = None
        else:
            if aligned_labels is None:
                rejection_reasons.append(
                    f"labels shape {tuple(labels.shape)} is not aligned to captured sequence length {seq_len}"
                )
            if attention_mask is None:
                rejection_reasons.append("attention_mask is missing, so labels cannot reliably exclude padding")
            elif aligned_attention is None:
                rejection_reasons.append(
                    f"attention_mask shape {tuple(attention_mask.shape)} is not aligned to captured sequence length {seq_len}"
                )

    if mask is None and attention_mask is not None and (allow_attention_fallback or not require_reliable_mask):
        aligned_attention = _align_mask_to_length(attention_mask.to(rep.device), seq_len, mask_name="attention_mask")
        if aligned_attention is not None:
            mask = aligned_attention.bool()
            if pooling == "prompt_or_image_only" and not _POOL_MASK_WARNING_EMITTED:
                LOG.warning("ASAM pooling fell back to attention_mask; target answer tokens may be included.")
                _POOL_MASK_WARNING_EMITTED = True

    if mask is None:
        if require_reliable_mask and pooling == "prompt_or_image_only":
            reason = "; ".join(rejection_reasons) if rejection_reasons else "no full-sequence mask was provided"
            raise RuntimeError(
                "ASAM RCSL pooling requires a reliable full-sequence prompt/image mask when asam_use_lar=true. "
                "Expected output.asam_perturb_mask aligned to captured hidden states, or aligned full-sequence "
                f"labels plus attention_mask. Refusing to average all sequence positions. Reason: {reason}."
            )
        if pooling == "prompt_or_image_only" and not _POOL_MASK_WARNING_EMITTED:
            LOG.warning("ASAM pooling has no reliable prompt/image mask; averaging all sequence positions.")
            _POOL_MASK_WARNING_EMITTED = True
        return rep.mean(dim=1)

    weights = mask.to(rep.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (rep * weights).sum(dim=1) / denom


def _mask_delta(delta: torch.Tensor, perturb_mask: torch.Tensor) -> torch.Tensor:
    return delta * perturb_mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)


def _project_l2_masked(delta: torch.Tensor, epsilon: float, perturb_mask: torch.Tensor) -> torch.Tensor:
    delta = _mask_delta(delta, perturb_mask)
    flat = delta.reshape(delta.shape[0], -1)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(float(epsilon) / norm, max=1.0)
    return (flat * scale).reshape_as(delta)


def _normalize_like_delta(grad: torch.Tensor, perturb_mask: torch.Tensor) -> torch.Tensor:
    grad = _mask_delta(grad, perturb_mask)
    flat = grad.reshape(grad.shape[0], -1)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return (flat / norm).reshape_as(grad)


def masked_delta_norm(delta: torch.Tensor, perturb_mask: torch.Tensor) -> torch.Tensor:
    delta = _mask_delta(delta, perturb_mask)
    return delta.reshape(delta.shape[0], -1).norm(p=2, dim=1)


def _forward_logits(editable_model, batch: dict, delta: Optional[torch.Tensor] = None, capture_latent: bool = False):
    model = _base_model(editable_model)
    with asam_latent_context(model, delta=delta, capture=capture_latent):
        if hasattr(editable_model, "forward_raw"):
            output = editable_model.forward_raw(batch)
        elif hasattr(editable_model, "asam_forward_raw"):
            output = editable_model.asam_forward_raw(batch)
        elif hasattr(editable_model, "model"):
            output = model(batch)
        else:
            output = editable_model(batch)
    logits = _output_logits(output)
    return output, logits


def capture_joint_latent(editable_model, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    model = _base_model(editable_model)
    _forward_logits(editable_model, batch, delta=None, capture_latent=True)
    latent, perturb_mask = get_last_asam_latent_and_mask(model)
    if latent.dim() != 3:
        raise RuntimeError(f"ASAM expected joint latent shape [B, L, D], got {tuple(latent.shape)}")
    if perturb_mask.shape != latent.shape[:2]:
        raise RuntimeError(
            f"ASAM perturb mask shape {tuple(perturb_mask.shape)} does not match latent shape {tuple(latent.shape[:2])}"
        )
    if not perturb_mask.bool().any(dim=1).all():
        raise RuntimeError("ASAM perturb_mask must allow at least one image/prompt position per example.")
    latent = latent.detach()
    perturb_mask = perturb_mask.detach().to(device=latent.device, dtype=torch.bool)
    if hasattr(model, "_asam_last_latent"):
        delattr(model, "_asam_last_latent")
    if hasattr(model, "_asam_last_perturb_mask"):
        delattr(model, "_asam_last_perturb_mask")
    return latent, perturb_mask


def generate_lar_deltas(editable_model, config, batch: dict) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    if not asam_use_lar(config):
        return [], {"asam/num_lar_variants": 0.0}

    latent, perturb_mask = capture_joint_latent(editable_model, batch)
    epsilon = asam_epsilon(config)
    step_size = asam_lar_step_size(config)
    num_variants = asam_num_variants(config)
    deltas: List[torch.Tensor] = []

    for _ in range(num_variants):
        delta = torch.randn_like(latent)
        delta = _project_l2_masked(delta, epsilon, perturb_mask)
        delta = delta.detach().requires_grad_(True)
        _, logits = _forward_logits(editable_model, batch, delta=delta, capture_latent=False)
        loss = editable_model.edit_loss_fn(config, logits, batch["labels"])["nll"]
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False, only_inputs=True)[0]
        with torch.no_grad():
            updated = delta + step_size * _normalize_like_delta(grad, perturb_mask)
            updated = _project_l2_masked(updated, epsilon, perturb_mask)
            updated = _mask_delta(updated, perturb_mask)
        deltas.append(updated.detach())

    norms = torch.stack([masked_delta_norm(d, perturb_mask) for d in deltas], dim=0)
    return deltas, {
        "asam/lar_delta_norm_mean": norms.mean().item() if norms.numel() else 0.0,
        "asam/lar_delta_norm_max": norms.max().item() if norms.numel() else 0.0,
        "asam/num_lar_variants": float(len(deltas)),
    }


def capture_asam_representations(
    editable_model,
    config,
    batch: dict,
    deltas: Sequence[Optional[torch.Tensor]],
) -> List[torch.Tensor]:
    model = _base_model(editable_model)
    module, module_name = find_asam_capture_module(model, config)
    captured: List[torch.Tensor] = []
    reps: List[torch.Tensor] = []

    handle = register_asam_output_hook(module, captured)
    try:
        for delta in deltas:
            captured.clear()
            output, _ = _forward_logits(editable_model, batch, delta=delta, capture_latent=False)
            if not captured:
                LOG.debug("ASAM hook on %s captured no tensor for one variant.", module_name)
                continue
            reps.append(
                pool_representation(
                    captured[-1],
                    labels=_labels_from_output_or_batch(output, batch),
                    attention_mask=_attention_from_output_or_batch(output, batch),
                    perturb_mask=_perturb_mask_from_output_or_batch(output, batch),
                    pooling=asam_pooling(config),
                    require_reliable_mask=asam_use_lar(config),
                    allow_attention_fallback=bool(getattr(config, "asam_allow_attention_pooling_fallback", False)),
                )
            )
    finally:
        handle.remove()
    return reps


def rank_constrained_subspace_loss(
    representations: Sequence[torch.Tensor],
    tau: float,
    return_info: bool = False,
):
    if len(representations) < 2:
        if representations:
            loss = representations[0].sum() * 0.0
        else:
            loss = torch.tensor(0.0)
        return (loss, {}) if return_info else loss

    anchor = representations[0].detach()
    variants = list(representations[1:])
    batch = anchor.shape[0]
    losses: List[torch.Tensor] = []
    sigma1_values: List[torch.Tensor] = []
    rest_values: List[torch.Tensor] = []

    for batch_idx in range(batch):
        rows = [anchor[batch_idx], *[variant[batch_idx] for variant in variants]]
        h_s = torch.stack(rows, dim=0)
        h_s = F.normalize(h_s, p=2, dim=-1, eps=1e-8)
        singular_values = torch.linalg.svdvals(h_s)
        logits = singular_values / max(float(tau), 1.0e-6)
        target = torch.zeros(1, dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        sigma1_values.append(singular_values[0].detach())
        if singular_values.numel() > 1:
            rest_values.append(singular_values[1:].mean().detach())

    loss = torch.stack(losses).mean()
    info = {
        "asam/sigma1": torch.stack(sigma1_values).mean().item(),
        "asam/sigma_rest_mean": torch.stack(rest_values).mean().item() if rest_values else 0.0,
    }
    return (loss, info) if return_info else loss


def gradient_nonzero_fraction(loss: torch.Tensor, parameters: Iterable[torch.Tensor]) -> float:
    all_params = [p for p in parameters if p is not None]
    params = [p for p in all_params if p.requires_grad]
    if not params:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    grad_by_id = {id(param): grad for param, grad in zip(params, grads)}
    total = 0
    nonzero = 0
    for param in all_params:
        total += param.numel()
        grad = grad_by_id.get(id(param))
        if grad is None:
            continue
        nonzero += grad.ne(0).sum().item()
    return float(nonzero / total) if total else 0.0


def gradient_diagnostics_for_params(
    loss: torch.Tensor,
    parameters: Iterable[torch.Tensor],
    prefix: str,
) -> Dict[str, float]:
    all_params = [p for p in parameters if p is not None]
    params = [p for p in all_params if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True) if params else []
    grad_by_id = {id(param): grad for param, grad in zip(params, grads)}
    total = 0
    nonzero = 0
    unused = 0
    for param in all_params:
        total += param.numel()
        grad = grad_by_id.get(id(param))
        if grad is None:
            unused += 1
            continue
        nonzero += grad.ne(0).sum().item()
        if grad.norm().item() == 0.0:
            unused += 1
    return {
        f"{prefix}_grad_nonzero_fraction": float(nonzero / total) if total else 0.0,
        f"{prefix}_grad_num_unused_params": float(unused),
        f"{prefix}_grad_total_params_checked": float(total),
    }


def _sanitize_metric_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_./-]+", "_", name).replace(".", "_")


def gradient_diagnostics_for_named_params(
    loss: torch.Tensor,
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
    param_names: Sequence[str],
    require_all: bool = False,
) -> Dict[str, float]:
    param_dict = dict(named_parameters)
    selected: List[Tuple[str, torch.Tensor]] = []
    missing = []
    frozen: List[Tuple[str, torch.Tensor]] = []
    for name in param_names:
        param = param_dict.get(name)
        if param is None:
            missing.append(name)
        elif not param.requires_grad:
            frozen.append((name, param))
        else:
            selected.append((name, param))

    params = [param for _, param in selected]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True) if params else []
    info: Dict[str, float] = {}
    total = 0
    nonzero = 0
    unused = len(missing) + len(frozen)
    failures = list(missing)

    for name, param in frozen:
        metric_name = f"asam/grad_norm/{_sanitize_metric_name(name)}"
        total += param.numel()
        info[metric_name] = 0.0
        failures.append(name)

    for (name, param), grad in zip(selected, grads):
        metric_name = f"asam/grad_norm/{_sanitize_metric_name(name)}"
        total += param.numel()
        if grad is None:
            unused += 1
            info[metric_name] = 0.0
            failures.append(name)
            continue
        grad_norm = grad.norm().item()
        info[metric_name] = grad_norm
        nonzero += grad.ne(0).sum().item()
        if grad_norm == 0.0:
            failures.append(name)

    info["asam/grad_nonzero_fraction_all_inner_params"] = float(nonzero / total) if total else 0.0
    info["asam/grad_num_unused_inner_params"] = float(unused)
    info["asam/grad_total_params_checked"] = float(total)
    if require_all and failures:
        raise RuntimeError(
            "ASAM alignment produced missing or zero gradients for intended parameters: "
            + ", ".join(failures)
        )
    return info


def asam_alignment_loss(editable_model, config, batch: dict, return_info: bool = False):
    if not asam_use_lar(config):
        raise NotImplementedError(
            "ASAM LAR is disabled. Dataset rephrases are not a replacement for LAR; "
            "set asam_use_lar=true for ASAM_FT/ASAM_MEND."
        )

    deltas, info = generate_lar_deltas(editable_model, config, batch)
    reps = capture_asam_representations(editable_model, config, batch, [None, *deltas])
    if len(reps) < 2:
        raise RuntimeError("ASAM needs original plus at least one LAR variant representation.")

    loss, rcsl_info = rank_constrained_subspace_loss(reps, asam_tau(config), return_info=True)
    info.update(rcsl_info)
    if asam_debug_grad_check(config):
        info.update(
            gradient_diagnostics_for_named_params(
                loss,
                _base_model(editable_model).named_parameters(),
                asam_alignment_params(config),
                require_all=asam_debug_require_all_inner_grads(config),
            )
        )

    result = AsamAlignmentResult(loss=loss, info=info, deltas=list(deltas))
    return result if return_info else loss
