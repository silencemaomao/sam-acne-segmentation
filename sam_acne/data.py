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


def _index(root: Path, kind: str, allow_empty: bool = False) -> dict[str, Path]:
    if not root.is_dir():
        if allow_empty:
            return {}
        raise FileNotFoundError(f"{kind} directory does not exist: {root}")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = _key(path, root)
            if key in result:
                raise ValueError(f"Duplicate {kind} key {key}: {result[key]} and {path}")
            result[key] = path
    if not result and not allow_empty:
        raise FileNotFoundError(f"No supported {kind} files under {root}")
    return result


class AcneHeatmapDataset(Dataset[dict[str, Any]]):
    """Images with possibly partial, mutually-exclusive 0/128/255 class masks."""

    def __init__(
        self,
        input_dir: str | Path,
        label_dir: str | Path,
        classes: list[str],
        image_size: tuple[int, int],
        horizontal_flip: float = 0.0,
        strict_labels: bool = True,
        unlabeled_image_policy: str = "error",
    ) -> None:
        self.input_dir, self.label_dir = Path(input_dir), Path(label_dir)
        self.classes = [str(name) for name in classes]
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("data.classes must contain unique, non-empty class names.")
        if unlabeled_image_policy not in {"error", "skip"}:
            raise ValueError("data.unlabeled_image_policy must be 'error' or 'skip'.")
        self.image_size = tuple(int(value) for value in image_size)
        self.horizontal_flip = float(horizontal_flip)
        self.strict_labels = bool(strict_labels)

        images = _index(self.input_dir, "image")
        labels_by_class: dict[str, dict[str, Path]] = {}
        for class_name in self.classes:
            class_labels = _index(
                self.label_dir / class_name,
                f"label for class {class_name}",
                allow_empty=True,
            )
            extra = sorted(set(class_labels) - set(images))
            if extra:
                raise ValueError(f"Class {class_name!r} has masks without images: {extra[:3]}")
            labels_by_class[class_name] = class_labels

        samples, unlabeled = [], []
        for key in sorted(images):
            paths = {
                class_name: labels_by_class[class_name].get(key)
                for class_name in self.classes
            }
            if not any(path is not None for path in paths.values()):
                unlabeled.append(key)
                if unlabeled_image_policy == "skip":
                    continue
            samples.append((key, images[key], paths))
        if unlabeled and unlabeled_image_policy == "error":
            raise FileNotFoundError(
                f"{len(unlabeled)} images have no class mask at all, e.g. {unlabeled[:3]}. "
                "Add at least one mask or set data.unlabeled_image_policy: skip."
            )
        if not samples:
            raise FileNotFoundError("No labeled samples remain after applying the label policy.")
        self.samples = samples

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
        paths = self.samples[index][2]
        return any(
            path is not None and np.any(self._label(path)[1] == 255)
            for path in paths.values()
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        name, image_path, label_paths = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        labels = {
            class_name: self._label(path)[0] if path is not None else None
            for class_name, path in label_paths.items()
        }
        height, width = self.image_size
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        labels = {
            class_name: label.resize((width, height), Image.Resampling.NEAREST)
            if label is not None
            else None
            for class_name, label in labels.items()
        }
        if self.horizontal_flip > 0 and random.random() < self.horizontal_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            labels = {
                class_name: label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if label is not None
                else None
                for class_name, label in labels.items()
            }

        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - SAM_MEAN) / SAM_STD

        # Channel 0 is background. Present masks intersect a set of possible labels.
        allowed = torch.ones((len(self.classes) + 1, height, width), dtype=torch.bool)
        annotation_present = torch.zeros(len(self.classes), dtype=torch.bool)
        for class_offset, class_name in enumerate(self.classes, start=1):
            label = labels[class_name]
            if label is None:
                continue
            annotation_present[class_offset - 1] = True
            values = torch.from_numpy(np.asarray(label, dtype=np.uint8).copy())
            negative, uncertain, positive = values == 0, values == 128, values == 255
            allowed[class_offset, negative] = False
            for output_index in range(1, len(self.classes) + 1):
                if output_index != class_offset:
                    allowed[output_index, uncertain] = False
            for output_index in range(len(self.classes) + 1):
                if output_index != class_offset:
                    allowed[output_index, positive] = False

        contradictory = ~allowed.any(dim=0)
        if torch.any(contradictory):
            y, x = torch.nonzero(contradictory, as_tuple=False)[0].tolist()
            raise ValueError(
                f"Contradictory mutually-exclusive masks for sample {name!r} at resized "
                f"pixel (x={x}, y={y}). Check overlapping 255/128 regions across classes."
            )
        return {
            "image": image_tensor,
            "allowed_classes": allowed,
            "annotation_present": annotation_present,
            "name": name,
            "original_size": torch.tensor([original_height, original_width], dtype=torch.int64),
        }


def build_dataset(config: dict[str, Any], split: str) -> AcneHeatmapDataset:
    data = config["data"]
    if split not in data:
        raise ValueError(f"YAML has no data.{split} section.")
    split_config = data[split]
    return AcneHeatmapDataset(
        resolve_path(config, split_config["input_dir"]),
        resolve_path(config, split_config["label_dir"]),
        list(data["classes"]),
        tuple(data["image_size"]),
        split_config.get("horizontal_flip", 0.0) if split == "train" else 0.0,
        data.get("strict_labels", True),
        data.get("unlabeled_image_policy", "error"),
    )


def build_dataloader(config: dict[str, Any], split: str) -> DataLoader:
    dataset = build_dataset(config, split)
    loader = config["dataloader"]
    is_train = split == "train"
    loader_type = loader.get("type", "standard")
    sampler, shuffle = None, is_train
    if is_train and loader_type == "weighted":
        positive_weight = float(loader.get("positive_sample_weight", 2.0))
        weights = [positive_weight if dataset.has_positive(i) else 1.0 for i in range(len(dataset))]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    elif loader_type not in {"standard", "weighted"}:
        raise ValueError(f"Unknown dataloader.type: {loader_type}")
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
