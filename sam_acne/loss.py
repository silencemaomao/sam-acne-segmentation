from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class PartialLabelCrossEntropyDiceLoss(nn.Module):
    """Softmax loss for mutually-exclusive classes with set-valued targets."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: float | list[float] = 1.0,
        smooth: float = 1.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.ce_weight, self.dice_weight = float(ce_weight), float(dice_weight)
        self.class_weights, self.smooth = class_weights, float(smooth)

    def forward(
        self, logits: torch.Tensor, allowed_classes: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if logits.shape != allowed_classes.shape:
            raise ValueError(
                f"logits shape {tuple(logits.shape)} != allowed_classes shape "
                f"{tuple(allowed_classes.shape)}"
            )
        allowed = allowed_classes.bool()
        allowed_count = allowed.sum(dim=1)
        if torch.any(allowed_count == 0):
            raise ValueError("allowed_classes contains an empty target set.")

        log_probabilities = F.log_softmax(logits, dim=1)
        allowed_log_probability = torch.logsumexp(
            log_probabilities.masked_fill(~allowed, -torch.inf), dim=1
        )
        informative = allowed_count < logits.shape[1]
        if torch.any(informative):
            per_pixel = -allowed_log_probability
            exact = allowed_count == 1
            weights = torch.as_tensor(
                self.class_weights, device=logits.device, dtype=logits.dtype
            )
            if weights.ndim == 0:
                weights = weights.repeat(logits.shape[1])
            if weights.numel() != logits.shape[1]:
                raise ValueError(
                    f"class_weights has {weights.numel()} entries; expected {logits.shape[1]}."
                )
            exact_target = allowed.to(torch.int64).argmax(dim=1)
            pixel_weights = torch.ones_like(per_pixel)
            pixel_weights[exact] = weights[exact_target[exact]]
            pixel = (per_pixel[informative] * pixel_weights[informative]).sum()
            pixel = pixel / pixel_weights[informative].sum().clamp_min(1.0)
        else:
            pixel = logits.sum() * 0.0

        exact = allowed_count == 1
        probabilities = torch.softmax(logits, dim=1)[:, 1:]
        exact_target = allowed.to(torch.int64).argmax(dim=1)
        target = F.one_hot(
            exact_target.clamp_max(logits.shape[1] - 1), num_classes=logits.shape[1]
        ).permute(0, 3, 1, 2)[:, 1:].to(logits.dtype)
        valid = exact[:, None].to(logits.dtype)
        reduce_dims = (0, 2, 3)
        intersection = (probabilities * target * valid).sum(dim=reduce_dims)
        denominator = ((probabilities + target) * valid).sum(dim=reduce_dims)
        dice_per_class = 1.0 - (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        has_positive = (target * valid).sum(dim=reduce_dims) > 0
        dice = (
            dice_per_class[has_positive].mean()
            if torch.any(has_positive)
            else logits.sum() * 0.0
        )
        total = self.ce_weight * pixel + self.dice_weight * dice
        return {"total": total, "pixel": pixel.detach(), "dice": dice.detach()}


def build_loss(config: dict[str, Any]) -> nn.Module:
    values = dict(config["loss"])
    loss_type = values.pop("type")
    if loss_type != "partial_label_ce_dice":
        raise ValueError(
            f"Unknown loss.type: {loss_type}. Mutually-exclusive partial labels require "
            "'partial_label_ce_dice'."
        )
    class_names = ["background", *list(config["data"]["classes"])]
    class_weights = values.get("class_weights", 1.0)
    if isinstance(class_weights, dict):
        missing = [name for name in class_names[1:] if name not in class_weights]
        if missing:
            raise ValueError(f"loss.class_weights is missing classes: {missing}")
        values["class_weights"] = [
            float(class_weights.get("background", 1.0)),
            *[float(class_weights[name]) for name in class_names[1:]],
        ]
    elif isinstance(class_weights, list) and len(class_weights) != len(class_names):
        raise ValueError(
            "loss.class_weights list must contain background followed by every data class; "
            f"got {len(class_weights)} entries, expected {len(class_names)}."
        )
    return PartialLabelCrossEntropyDiceLoss(**values)
