from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .config import resolve_path


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]):
    values = config["optimizer"]
    if values.get("type", "adamw") != "adamw":
        raise ValueError("Only optimizer.type=adamw is supported.")
    learning_rate = float(values["learning_rate"])
    encoder_multiplier = float(values.get("encoder_lr_multiplier", 1.0))
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder if name.startswith("encoder.") else decoder).append(parameter)
    groups = []
    if encoder:
        groups.append({"params": encoder, "lr": learning_rate * encoder_multiplier})
    if decoder:
        groups.append({"params": decoder, "lr": learning_rate})
    return torch.optim.AdamW(
        groups, lr=learning_rate, weight_decay=float(values.get("weight_decay", 0.01))
    )


def build_scheduler(optimizer, config: dict[str, Any]):
    values = config.get("scheduler", {"type": "constant"})
    kind, warmup = values.get("type", "constant"), int(values.get("warmup_steps", 0))
    max_steps = int(config["train"]["max_steps"])

    def scale(step: int) -> float:
        if warmup and step < warmup:
            return max(step, 1) / warmup
        if kind == "constant":
            return 1.0
        if kind != "cosine":
            raise ValueError(f"Unknown scheduler.type: {kind}")
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def output_path(config: dict[str, Any]) -> Path:
    return resolve_path(config, config["train"]["output_dir"])


def save_exported_model(accelerator, model, folder: Path) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        folder.mkdir(parents=True, exist_ok=True)
        accelerator.save(accelerator.unwrap_model(model).state_dict(), folder / "model.pt")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
