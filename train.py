from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from sam_acne.config import load_config, resolve_path
from sam_acne.data import build_dataloader
from sam_acne.engine import evaluate
from sam_acne.loss import build_loss
from sam_acne.model import build_model
from sam_acne.utils import append_jsonl, build_optimizer, build_scheduler, output_path, save_exported_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train automatic SAM acne segmentation.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    set_seed(int(config.get("seed", 42)))
    train_config, acc_config = config["train"], config["accelerator"]
    run_dir = output_path(config)
    log_with = acc_config.get("log_with")
    accelerator = Accelerator(
        mixed_precision=str(acc_config.get("mixed_precision", "no")),
        gradient_accumulation_steps=int(acc_config.get("gradient_accumulation_steps", 1)),
        cpu=bool(acc_config.get("cpu", False)),
        log_with=log_with,
        project_config=ProjectConfiguration(project_dir=run_dir, logging_dir=run_dir / "logs"),
    )
    if log_with:
        accelerator.init_trackers("sam-acne", config=config)
    train_loader, val_loader = build_dataloader(config, "train"), build_dataloader(config, "val")
    model, criterion = build_model(config), build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    criterion = criterion.to(accelerator.device)
    run_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    greater = bool(train_config.get("greater_is_better", True))
    best_value = -math.inf if greater else math.inf
    resume = train_config.get("resume_from")
    if resume:
        resume_path = resolve_path(config, resume)
        accelerator.load_state(resume_path)
        state_file = resume_path / "trainer_state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            global_step, best_value = int(state["global_step"]), float(state["best_value"])

    max_steps = int(train_config["max_steps"])
    val_every, save_every = int(train_config["val_every_steps"]), int(train_config["save_every_steps"])
    log_every = int(train_config.get("log_every_steps", 20))
    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))
    metric_name = str(train_config.get("metric_for_best", "dice"))
    optimizer.zero_grad()

    for epoch in range(int(train_config.get("num_epochs", 100))):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                losses = criterion(model(batch["image"]), batch["target"])
                accelerator.backward(losses["total"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), float(train_config.get("max_grad_norm", 1.0))
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            if global_step == 1 or global_step % log_every == 0:
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": float(losses["total"].detach().item()),
                    "pixel_loss": float(losses["pixel"].item()),
                    "dice_loss": float(losses["dice"].item()),
                    "learning_rate": float(scheduler.get_last_lr()[-1]),
                }
                accelerator.print(record)
                if accelerator.is_main_process:
                    append_jsonl(run_dir / "metrics.jsonl", record)
                if log_with:
                    accelerator.log(record, step=global_step)

            if global_step % val_every == 0 or global_step == max_steps:
                metrics = evaluate(model, val_loader, criterion, accelerator, threshold)
                record = {"step": global_step, **{f"val_{k}": v for k, v in metrics.items()}}
                accelerator.print(record)
                if accelerator.is_main_process:
                    append_jsonl(run_dir / "metrics.jsonl", record)
                value = metrics[metric_name]
                improved = value > best_value if greater else value < best_value
                if improved:
                    best_value = value
                    save_exported_model(accelerator, model, run_dir / "best")
                    if accelerator.is_main_process:
                        (run_dir / "best" / "metrics.json").write_text(
                            json.dumps(record, indent=2), encoding="utf-8"
                        )

            if global_step % save_every == 0 or global_step == max_steps:
                checkpoint = run_dir / f"checkpoint-step-{global_step:06d}"
                accelerator.save_state(checkpoint)
                save_exported_model(accelerator, model, checkpoint)
                if accelerator.is_main_process:
                    (checkpoint / "trainer_state.json").write_text(
                        json.dumps({"global_step": global_step, "best_value": best_value}, indent=2),
                        encoding="utf-8",
                    )
            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break
    accelerator.print(f"Training finished at step {global_step}; best={best_value:.6f}")
    if log_with:
        accelerator.end_training()


if __name__ == "__main__":
    main()
