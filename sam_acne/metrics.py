from __future__ import annotations

import torch


def confusion_counts(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> torch.Tensor:
    prediction = torch.sigmoid(logits.squeeze(1)) >= threshold
    valid, positive, negative = target >= 0, target == 1, target == 0
    return torch.stack(
        [
            (prediction & positive & valid).sum(),
            (prediction & negative & valid).sum(),
            ((~prediction) & positive & valid).sum(),
            ((~prediction) & negative & valid).sum(),
        ]
    ).double()


def metrics_from_counts(counts: torch.Tensor) -> dict[str, float]:
    tp, fp, fn, tn = [float(value) for value in counts.tolist()]
    eps = 1e-12
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "dice": dice,
        "f1": dice,
        "iou": tp / (tp + fp + fn + eps),
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
    }
