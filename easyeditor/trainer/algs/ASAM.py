import logging

import torch
from tqdm import trange

from .FT import FT
from .MEND import MEND
from .asam_utils import (
    asam_alignment_loss,
    asam_beta,
    asam_enabled,
    asam_use_dataset_variants,
    asam_variant_ce_weight,
)

LOG = logging.getLogger(__name__)


class ASAM_FT(FT):
    """FT with ASAM LAR variants and RCSL alignment."""

    def _forward_logits(self, batch):
        outputs = self(batch)
        return outputs if isinstance(outputs, torch.Tensor) else outputs.logits

    def _edit_loss(self, batch):
        outputs = self._forward_logits(batch)
        return self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]

    def edit(self, batch, condition=None, detach_history=False, return_factors=False):
        if self.save_weight is not None:
            self.model.load_state_dict(self.save_weight, strict=False)
        self.model.train()

        if self.config.inner_params[0] in ["Qformer", "mm_projector", "vision_model"]:
            weights = {
                n: p
                for n, p in self.model.named_parameters()
                if n.find(self.config.inner_params[0]) != -1
            }
        else:
            names = set([n for n, p in self.model.named_parameters()])
            pset = set(self.config.inner_params)
            for p in pset:
                assert p in names, f"inner param {p} not in model"

            weights = {
                n: p
                for n, p in self.model.named_parameters()
                if n in pset
            }

        self.save_weight = {k: v.detach().clone() for k, v in weights.items()}

        opt = torch.optim.AdamW(
            [v for _, v in weights.items()],
            lr=self.config.edit_lr,
            weight_decay=float(getattr(self.config, "weight_decay", 0.0)),
        )
        for name, w in self.model.named_parameters():
            w.requires_grad = name in weights

        variant_weight = asam_variant_ce_weight(self.config)
        pbar = trange(self.config.num_steps, ncols=120)
        last_info = {}
        for _ in pbar:
            opt.zero_grad()
            primary_loss = self._edit_loss(batch)
            variant_loss = primary_loss.new_tensor(0.0)
            if asam_use_dataset_variants(self.config) and variant_weight and isinstance(condition, dict):
                dataset_variants = [
                    condition[key]
                    for key in ("edit_outer", "edit_outer_image")
                    if isinstance(condition.get(key), dict)
                ]
                if dataset_variants:
                    variant_losses = [self._edit_loss(variant) for variant in dataset_variants]
                    variant_loss = torch.stack(variant_losses).mean()

            align_loss = primary_loss.new_tensor(0.0)
            align_info = {}
            if asam_enabled(self.config):
                align_result = asam_alignment_loss(self, self.config, batch, return_info=True)
                align_loss = align_result.loss
                align_info = align_result.info

            loss = primary_loss + variant_weight * variant_loss + asam_beta(self.config) * align_loss
            loss.backward()
            opt.step()
            last_info = {
                "loss": loss.detach().item(),
                "primary": primary_loss.detach().item(),
                "variant": variant_loss.detach().item(),
                "asam_align": align_loss.detach().item(),
            }
            last_info.update(align_info)
            pbar.set_postfix(last_info)

        return (
            ASAM_FT(
                self.model,
                self.config,
                self.model_constructor,
            ),
            {
                "asam/inner_primary_loss": last_info.get("primary"),
                "asam/inner_variant_loss": last_info.get("variant"),
                "asam/inner_align_loss": last_info.get("asam_align"),
                "asam/lar_delta_norm_mean": last_info.get("asam/lar_delta_norm_mean"),
                "asam/lar_delta_norm_max": last_info.get("asam/lar_delta_norm_max"),
                "asam/num_lar_variants": last_info.get("asam/num_lar_variants"),
                "asam/sigma1": last_info.get("asam/sigma1"),
                "asam/sigma_rest_mean": last_info.get("asam/sigma_rest_mean"),
                "asam/grad_nonzero_fraction_all_inner_params": last_info.get("asam/grad_nonzero_fraction_all_inner_params"),
                "asam/grad_num_unused_inner_params": last_info.get("asam/grad_num_unused_inner_params"),
            },
        )


class ASAM_MEND(MEND):
    """MEND trained with ASAM alignment in `MultimodalTrainer`.

    The edit rule remains MEND-compatible so checkpoints can be applied through
    the existing multimodal MEND executor. The ASAM objective is added in the
    trainer after constructing the edited functional model.
    """

    pass
