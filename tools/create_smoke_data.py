from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CLASSES = ("acne", "pigmentation", "scar")
CENTERS = ((16, 20), (32, 38), (48, 20))
COLORS = ((205, 65, 65), (135, 95, 70), (190, 125, 115))


def sample(image_path: Path, label_root: Path, seed: int, index: int) -> None:
    rng = np.random.default_rng(seed)
    pixels = np.full((64, 64, 3), [188, 142, 122], dtype=np.int16)
    pixels += rng.normal(0, 6, pixels.shape).astype(np.int16)
    image = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))
    image_draw = ImageDraw.Draw(image)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    # Every fourth image is fully annotated; the rest contain exactly one class mask.
    annotated = range(len(CLASSES)) if index % 4 == 0 else (index % len(CLASSES),)
    for class_index in annotated:
        class_name = CLASSES[class_index]
        label = Image.new("L", (64, 64), 0)
        label_draw = ImageDraw.Draw(label)
        x0 = class_index * 3
        label_draw.rectangle((x0, 0, x0 + 1, 63), fill=128)
        positive = index % 4 != 0 or (index + class_index) % 2 == 0
        if positive:
            x, y = CENTERS[class_index]
            radius = 4
            image_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=COLORS[class_index],
            )
            label_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        label_path = label_root / class_name / f"sample_{index:03d}.png"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label.save(label_path)
    image.save(image_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("smoke_data_partial"))
    args = parser.parse_args()
    for split, count, offset in (("train", 8, 0), ("val", 4, 100), ("test", 4, 200)):
        for index in range(count):
            sample(
                args.output_dir / split / "images" / f"sample_{index:03d}.jpg",
                args.output_dir / split / "labels",
                offset + index,
                index,
            )
    print(f"Created partial-label smoke data under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
