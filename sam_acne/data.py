from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import resolve_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_VALUES = {0, 128, 255}
SAM_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
SAM_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def _key(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix().lower()


def _index(root: Path, kind: str) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"{kind} directory does not exist: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = _key(path, root)
            if key in result:
                raise ValueError(f"Duplicate {kind} key {key}: {result[key]} and {path}")
            result[key] = path
    if not result:
        raise FileNotFoundError(f"No supported {kind} files under {root}")
    return result


class AcneHeatmapDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        input_dir: str | Path,
        label_dir: str | Path,
        image_size: tuple[int, int],
        horizontal_flip: float = 0.0,
        strict_labels: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.label_dir = Path(label_dir)
        self.image_size = tuple(int(value) for value in image_size)
        self.horizontal_flip = float(horizontal_flip)
        self.strict_labels = bool(strict_labels)
        images, labels = _index(self.input_dir, "image"), _index(self.label_dir, "label")
        missing, extra = sorted(set(images) - set(labels)), sorted(set(labels) - set(images))
        if missing:
            raise FileNotFoundError(f"Missing labels for {len(missing)} images, e.g. {missing[:3]}")
        if extra:
            raise ValueError(f"Labels without images: {extra[:3]}")
        self.samples = [(key, images[key], labels[key]) for key in sorted(images)]

    def __len__(self) -> int:
        return len(self.samples)

    def _label(self, path: Path) -> tuple[Image.Image, np.ndarray]:
        image = Image.open(path).convert("L")
        array = np.asarray(image, dtype=np.uint8)
        unique = set(np.unique(array).tolist())
        if self.strict_labels and not unique.issubset(LABEL_VALUES):
            raise ValueError(
                f"{path} contains invalid values {sorted(unique - LABEL_VALUES)[:20]}; "
                "expected only 0, 128, 255. Use lossless PNG and nearest resize."
            )
        return image, array

    def has_positive(self, index: int) -> bool:
        return bool(np.any(self._label(self.samples[index][2])[1] == 255))

    def __getitem__(self, index: int) -> dict[str, Any]:
        name, image_path, label_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        label, _ = self._label(label_path)
        height, width = self.image_size
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        label = label.resize((width, height), Image.Resampling.NEAREST)
        if self.horizontal_flip and random.random() < self.horizontal_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - SAM_MEAN) / SAM_STD
        label_array = np.asarray(label, dtype=np.uint8).copy()
        target = torch.full((height, width), -1.0, dtype=torch.float32)
        target[torch.from_numpy(label_array == 0)] = 0.0
        target[torch.from_numpy(label_array == 255)] = 1.0
        return {
            "image": image_tensor,
            "target": target,
            "name": name,
            "original_size": torch.tensor([original_height, original_width]),
        }


def build_dataset(config: dict[str, Any], split: str) -> AcneHeatmapDataset:
    data = config["data"]
    split_config = data[split]
    return AcneHeatmapDataset(
        resolve_path(config, split_config["input_dir"]),
        resolve_path(config, split_config["label_dir"]),
        tuple(data["image_size"]),
        split_config.get("horizontal_flip", 0.0) if split == "train" else 0.0,
        data.get("strict_labels", True),
    )


def build_dataloader(config: dict[str, Any], split: str) -> DataLoader:
    dataset = build_dataset(config, split)
    loader = config["dataloader"]
    is_train = split == "train"
    sampler, shuffle = None, is_train
    if is_train and loader.get("type") == "weighted":
        positive_weight = float(loader.get("positive_sample_weight", 2.0))
        weights = [positive_weight if dataset.has_positive(i) else 1.0 for i in range(len(dataset))]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    workers = int(loader.get("num_workers", 4))
    batch_size = int(
        loader.get("train_batch_size", 1) if is_train else loader.get("eval_batch_size", 1)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(loader.get("pin_memory", True)),
        persistent_workers=bool(loader.get("persistent_workers", True)) and workers > 0,
        drop_last=is_train and len(dataset) > batch_size,
    )
