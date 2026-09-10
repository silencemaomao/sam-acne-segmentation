from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from sam_acne.config import load_config, resolve_path
from sam_acne.data import build_dataloader
from sam_acne.engine import evaluate
from sam_acne.loss import build_loss
from sam_acne.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Test automatic SAM acne segmentation.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def save_predictions(
    model, dataloader, accelerator, output_dir: Path, class_names: list[str]
) -> None:
    model.eval()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    for batch in dataloader:
        probabilities = torch.sigmoid(model(batch["image"])).float().cpu()
        for index, sample_probabilities in enumerate(probabilities):
            height, width = [int(value) for value in batch["original_size"][index].tolist()]
            heatmaps = torch.nn.functional.interpolate(
                sample_probabilities[None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            for class_index, class_name in enumerate(class_names):
                destination = output_dir / class_name / f"{batch['name'][index]}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                array = np.uint8(torch.clamp(heatmaps[class_index] * 255, 0, 255).numpy())
                Image.fromarray(array).save(destination)


def main() -> None:
    config = load_config(parse_args().config)
    acc_config = config["accelerator"]
    accelerator = Accelerator(
        mixed_precision=str(acc_config.get("mixed_precision", "no")),
        cpu=bool(acc_config.get("cpu", False)),
    )
    model = build_model(config)
    checkpoint = resolve_path(config, config["test"]["checkpoint"])
    checkpoint = checkpoint / "model.pt" if checkpoint.is_dir() else checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    dataloader = build_dataloader(config, "test")
    criterion = build_loss(config).to(accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    class_names = list(config["data"]["classes"])
    metrics = evaluate(
        model, dataloader, criterion, accelerator,
        float(config.get("evaluation", {}).get("threshold", 0.5)),
        class_names,
    )
    accelerator.print(json.dumps(metrics, indent=2))
    metrics_file = resolve_path(config, config["test"]["metrics_file"])
    if accelerator.is_main_process:
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if config["test"].get("save_heatmaps", True):
        save_predictions(
            model, dataloader, accelerator,
            resolve_path(config, config["test"]["output_dir"]), class_names,
        )


if __name__ == "__main__":
    main()
