#!/usr/bin/env python3
import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run a tiny DSCA Stage 1 real-backbone smoke.")
    parser.add_argument("--model", default="blip2", choices=["blip2", "llava-med"])
    parser.add_argument("--hparams")
    parser.add_argument("--num_edit_samples", type=int, default=4)
    parser.add_argument("--num_replay_samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--min_samples", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-root", default="datasets/MedMKEB/images")
    parser.add_argument("--base-signature-limit", type=int, default=32)
    parser.add_argument("--log_dir", default=os.environ.get("LOG_DIR", "outputs/dsca_server_validation/manual"))
    return parser.parse_args()


def load_config(path, args):
    from easyeditor.trainer.training_hparams.dsca_multimodal_training_hparams import (
        DSCAMultimodalTrainingHparams,
    )

    if path is None:
        path = (
            "hparams/TRAINING/DSCA/llava_med_stage1_smoke.yaml"
            if args.model == "llava-med"
            else "hparams/TRAINING/DSCA/blip2_stage1_smoke.yaml"
        )
    config = DSCAMultimodalTrainingHparams.from_hparams(path)
    config.device = args.device
    config.dsca_rank = args.rank
    config.dsca_min_samples = args.min_samples
    config.dsca_refine_interval = 1
    config.dsca_route_temperature = 0.07
    config.dsca_lambda_align = 0.5
    config.dsca_lambda_distill = 1.0
    config.dsca_lambda_sparse = 1.0e-2
    config.batch_size = 1
    config.accumulate_bs = 1
    config.dsca_debug = True
    return config


def first_images(image_root, count):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [path for path in sorted(Path(image_root).rglob("*")) if path.is_file() and path.suffix.lower() in suffixes]
    if not images:
        raise RuntimeError(f"No images found under {image_root}")
    return [str(images[idx % len(images)]) for idx in range(count)]


def make_batch(model, idx, device, prefix, image_paths=None):
    colors = [" red", " blue", " green", " yellow"]
    target = colors[idx % len(colors)]
    prompt = f"Question: What color is synthetic object {prefix}-{idx}? Answer:"
    text = prompt + target
    if hasattr(model, "opt_tokenizer"):
        tokenizer = model.opt_tokenizer
        prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        labels = tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        generator = torch.Generator(device="cpu").manual_seed(9000 + idx + (100 if prefix == "replay" else 0))
        image = torch.randn(1, 3, 364, 364, generator=generator).to(device)
        return {
            "image": image,
            "text_input": [text],
            "labels": labels,
            "prompts_len": [prompt_len],
        }
    if hasattr(model, "llava_tokenizer"):
        tokenizer = model.llava_tokenizer
        prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        labels = tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if not image_paths:
            raise RuntimeError("LLaVA-Med smoke requires image paths.")
        return {
            "image_path": [image_paths[idx % len(image_paths)]],
            "prompt": [prompt],
            "target": [target],
            "text_input": [text],
            "labels": labels,
            "prompts_len": [prompt_len],
        }
    raise RuntimeError(f"Unsupported smoke model wrapper: {type(model)}")


def base_signature(model, limit=0):
    signature = []
    with torch.no_grad():
        for idx, (name, param) in enumerate(model.named_parameters()):
            if limit and idx >= limit:
                break
            data = param.detach().float()
            signature.append((name, float(data.sum().cpu()), float(data.norm().cpu())))
    return signature


def signature_delta_norm(before, after):
    total = 0.0
    for (name_a, sum_a, norm_a), (name_b, sum_b, norm_b) in zip(before, after):
        if name_a != name_b:
            return float("inf")
        total += (sum_a - sum_b) ** 2 + (norm_a - norm_b) ** 2
    return math.sqrt(total)


def active_dsam_grad_norms(repository):
    norms = []
    for idx, dsam in enumerate(repository.dsams):
        if idx >= len(repository.active) or not bool(repository.active[idx].item()):
            continue
        grads = []
        for param in dsam.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().float().norm())
        if grads:
            norms.append(float(torch.stack(grads).mean().cpu()))
    return norms


def has_duplicate_optimizer_params(optimizer):
    seen = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            ident = id(param)
            if ident in seen:
                return True
            seen.add(ident)
    return False


def repository_round_trip_ok(repository):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "dsca_repo.pt"
        repository.save(str(path))
        loaded = type(repository).load(str(path))
    return len(loaded) == len(repository) and loaded.num_active() == repository.num_active()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    torch.manual_seed(1234)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    from easyeditor.trainer.algs.dsca import DSCA
    from easyeditor.trainer.models import get_model

    config = load_config(args.hparams, args)
    model = get_model(config).to(args.device).eval()
    alg = DSCA(model, config, lambda: None).to(args.device)
    optimizer = torch.optim.Adam(alg.outer_parameters(), lr=config.lr)
    before_sig = base_signature(alg.model, args.base_signature_limit)
    image_paths = first_images(args.image_root, max(args.num_edit_samples, args.num_replay_samples)) if args.model == "llava-med" else None

    rows = []
    finite_losses = True
    for step in range(1, args.steps + 1):
        edit = make_batch(alg.model, step % args.num_edit_samples, args.device, "edit", image_paths=image_paths)
        replay = make_batch(alg.model, step % args.num_replay_samples, args.device, "replay", image_paths=image_paths)
        batch = {"edit_inner": edit, "loc_image": replay, "loc": replay}

        optimizer.zero_grad(set_to_none=True)
        _, _, _, _, info = alg.edit_step(batch, training=True, optimizer=optimizer)
        grad_norms = active_dsam_grad_norms(alg.repository)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        after_sig = base_signature(alg.model, args.base_signature_limit)
        delta_norm = signature_delta_norm(before_sig, after_sig)
        row = {
            "step": step,
            "loss/dsca_total": info.get("loss/dsca_total", 0.0),
            "loss/dsca_task": info.get("loss/dsca_task", 0.0),
            "loss/dsca_align": info.get("loss/dsca_align", 0.0),
            "loss/dsca_cdistill": info.get("loss/dsca_cdistill", 0.0),
            "loss/dsca_sparse": info.get("loss/dsca_sparse", 0.0),
            "dsca/num_clusters": info.get("dsca/num_clusters", 0.0),
            "dsca/num_active_dsams": info.get("dsca/num_active_dsams", 0.0),
            "dsca/num_candidates_mean": info.get("dsca/num_candidates_mean", 0.0),
            "dsca/route_weight_mean": info.get("dsca/route_weight_mean", 0.0),
            "dsca/route_weight_max": info.get("dsca/route_weight_max", 0.0),
            "dsca/residual_norm_mean": info.get("dsca/residual_norm_mean", 0.0),
            "dsca/mean_subspace_overlap": info.get("dsca/mean_subspace_overlap", 0.0),
            "dsca/new_clusters_created": info.get("dsca/new_clusters_created", 0.0),
            "dsca/new_dsams_activated": info.get("dsca/new_dsams_activated", 0.0),
            "base_vlm_params_changed": delta_norm != 0.0,
            "base_param_delta_norm": delta_norm,
            "R_k_requires_grad_any": any(dsam.R.requires_grad for dsam in alg.repository.dsams),
            "active_dsam_grad_norm_mean": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
            "optimizer_duplicate_dsam_param_groups": has_duplicate_optimizer_params(optimizer),
            "repository_round_trip_ok": repository_round_trip_ok(alg.repository),
        }
        rows.append(row)
        finite_losses = finite_losses and all(
            math.isfinite(float(row[name]))
            for name in (
                "loss/dsca_total",
                "loss/dsca_task",
                "loss/dsca_align",
                "loss/dsca_cdistill",
                "loss/dsca_sparse",
            )
        )
        print(row, flush=True)

    csv_name = "llava_med_dsca_3step_smoke_metrics.csv" if args.model == "llava-med" else "blip2_dsca_3step_smoke_metrics.csv"
    csv_path = Path(args.log_dir) / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote_metrics={csv_path}")

    final = rows[-1]
    failures = []
    if not finite_losses:
        failures.append("losses are not finite")
    if float(final["dsca/num_clusters"]) <= 0:
        failures.append("num_clusters did not become positive")
    if float(final["dsca/num_active_dsams"]) <= 0:
        failures.append("num_active_dsams did not become positive")
    if final["base_vlm_params_changed"]:
        failures.append("base VLM parameters changed")
    if float(final["base_param_delta_norm"]) != 0.0:
        failures.append("base_param_delta_norm is nonzero")
    if final["R_k_requires_grad_any"]:
        failures.append("R_k requires grad")
    if float(final["active_dsam_grad_norm_mean"]) <= 0.0:
        failures.append("active DSAM parameters did not receive gradients")
    if final["optimizer_duplicate_dsam_param_groups"]:
        failures.append("duplicate optimizer param groups detected")
    if not final["repository_round_trip_ok"]:
        failures.append("repository save/load failed")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
