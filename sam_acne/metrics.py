from __future__ import annotations

import torch


def confusion_counts(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} != target shape {tuple(target.shape)}")
    prediction = torch.sigmoid(logits) >= threshold
    valid, positive, negative = target >= 0, target == 1, target == 0
    reduce_dims = (0, 2, 3)
    return torch.stack(
        [
            (prediction & positive & valid).sum(dim=reduce_dims),
            (prediction & negative & valid).sum(dim=reduce_dims),
            ((~prediction) & positive & valid).sum(dim=reduce_dims),
            ((~prediction) & negative & valid).sum(dim=reduce_dims),
        ],
        dim=1,
    ).double()


def _binary_metrics(counts: torch.Tensor) -> dict[str, float]:
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


def metrics_from_counts(counts: torch.Tensor, class_names: list[str]) -> dict[str, float]:
    if counts.shape != (len(class_names), 4):
        raise ValueError(
            f"Expected counts shape {(len(class_names), 4)}, received {tuple(counts.shape)}"
        )
    per_class = [_binary_metrics(counts[index]) for index in range(len(class_names))]
    result: dict[str, float] = {}
    for class_name, values in zip(class_names, per_class, strict=True):
        for metric_name, value in values.items():
            result[f"class/{class_name}/{metric_name}"] = value
    for metric_name in ("precision", "recall", "dice", "f1", "iou", "accuracy"):
        result[f"macro_{metric_name}"] = sum(values[metric_name] for values in per_class) / len(per_class)
    micro = _binary_metrics(counts.sum(dim=0))
    for metric_name, value in micro.items():
        result[f"micro_{metric_name}"] = value
    return result
